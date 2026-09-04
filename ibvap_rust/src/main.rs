use serde::{Deserialize, Serialize};
use slint::Model;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::thread;

mod database;
mod python_connector;
mod streaming;

mod web_server;

slint::include_modules!();

// ============================================================
// Camera Representation shared with Python
// ============================================================

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct DiscoveredCamera {
    #[serde(default)]
    pub id: String,

    /// Human-visible default name from discovery.
    /// On first INSERT this becomes the camera's label.
    /// On UPDATE this field is IGNORED — the operator's custom name is preserved.
    #[serde(default)]
    pub name: String,

    #[serde(default)]
    pub ip: String,

    #[serde(default)]
    pub rtsp: String,

    #[serde(default)]
    pub onvif_uid: String,

    #[serde(default)]
    pub is_restricted: bool,
}

// ============================================================
// Helpers
// ============================================================

fn generate_id(_index: usize) -> String {
    format!("manual-{}", uuid::Uuid::new_v4())
}

fn to_slint_camera(camera: &DiscoveredCamera, index: usize) -> Camera {
    let id = if camera.id.is_empty() {
        generate_id(index)
    } else {
        camera.id.clone()
    };

    // `camera.name` already contains the operator-assigned label as returned
    // from the database (get_cameras reads `name`, which rename_camera updates).
    let display_name = if camera.name.is_empty() {
        format!("Camera {}", index + 1)
    } else {
        camera.name.clone()
    };

    let empty_buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(1, 1);

    Camera {
        id: id.into(),
        name: display_name.clone().into(),
        tag: display_name.into(), // tag mirrors name; both updated by rename_camera
        ip: camera.ip.clone().into(),
        is_online: true,
        has_onvif: true,
        is_restricted: camera.is_restricted,
        last_seen: "Online".into(),
        live_frame: slint::Image::from_rgba8(empty_buffer),
    }
}

fn sync_ui_cameras_from_db(ui: &AppWindow, db: &rusqlite::Connection) {
    if let Ok(stored) = database::get_cameras(db) {
        let slint_cameras = stored
            .iter()
            .enumerate()
            .map(|(index, camera)| to_slint_camera(camera, index))
            .collect::<Vec<_>>();

        ui.set_cameras(Rc::new(slint::VecModel::from(slint_cameras)).into());
    }
}

// ============================================================
// Main Application Entry
// ============================================================

fn main() -> Result<(), slint::PlatformError> {
    // --------------------------------------------------------
    // Initialize Python runtime
    // --------------------------------------------------------
    pyo3::prepare_freethreaded_python();

    // --------------------------------------------------------
    // Slint application & SQLite database setup
    // --------------------------------------------------------
    let ui = AppWindow::new()?;
    let db_conn = database::open().expect("Failed to open cameras.db");
    println!("[INFO] Connected to local SQLite database: cameras.db");
    let db = Arc::new(Mutex::new(db_conn));

    // Populate cameras in UI from database on startup
    {
        let conn = db.lock().unwrap();
        sync_ui_cameras_from_db(&ui, &conn);
        println!("[INFO] Synced UI cameras from database.");
    }
    
    // Determine local IP for web server display
    let local_ip = local_ip_address::local_ip().map(|ip| ip.to_string()).unwrap_or_else(|_| "localhost".to_string());
    ui.set_web_server_url(format!("http://{}:3000", local_ip).into());

    // --------------------------------------------------------
    // Multi-camera concurrent streaming runtime (Tokio)
    // --------------------------------------------------------
    let rt = tokio::runtime::Runtime::new().expect("Failed to start Tokio runtime");
    let rt_handle = rt.handle().clone();

    let (frame_tx, frame_rx) =
        tokio::sync::mpsc::channel::<streaming::FrameUpdate>(6);

    let selected_camera: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let stream_registry = streaming::StreamRegistry::default();
    
    let shared_alerts = Arc::new(Mutex::new(Vec::new()));
    let latest_frames: Arc<Mutex<HashMap<String, Vec<u8>>>> = Arc::new(Mutex::new(HashMap::new()));

    let (tx_ws, _) = tokio::sync::broadcast::channel(100);

    // Spawn Web Server
    let web_state = web_server::AppState {
        alerts: shared_alerts.clone(),
        db_pool: db.clone(),
        latest_frames: latest_frames.clone(),
        ws_sender: tx_ws.clone(),
    };
    rt.spawn(async move {
        println!("[INFO] Starting Web Server Tokio task.");
        web_server::run(web_state).await;
    });

    println!("[INFO] Starting UI Aggregator Tokio task.");
    rt.spawn(streaming::run_aggregator(
        frame_rx,
        ui.as_weak(),
        selected_camera.clone(),
        shared_alerts,
        latest_frames.clone(),
        db.clone(),
        tx_ws.clone(),
    ));

    // Auto-start streams for existing database cameras
    {
        let conn = db.lock().unwrap();
        if let Ok(existing) = database::get_cameras(&conn) {
            for cam in &existing {
                if !cam.rtsp.is_empty() {
                    streaming::start_camera_stream(
                        &rt_handle,
                        stream_registry.clone(),
                        cam.id.clone(),
                        cam.rtsp.clone(),
                        frame_tx.clone(),
                    );
                }
            }
            if let Some(first) = existing.first() {
                println!("[INFO] Automatically selecting first camera: {}", first.id);
                *selected_camera.lock().unwrap() = first.id.clone();
                ui.set_selected_camera_id(first.id.clone().into());
            }
        }
        
        // Load ONVIF credentials from DB and populate UI
        if let Some(user) = database::get_setting(&conn, "onvif_username") {
            ui.set_default_user(user.into());
        }
        if let Some(pass) = database::get_setting(&conn, "onvif_password") {
            ui.set_default_pass(pass.into());
        }

        // Load AI Reference image credentials from DB
        let ai_tag = database::get_setting(&conn, "ai_ref_tag");
        let ai_path = database::get_setting(&conn, "ai_ref_path");

        if let (Some(tag), Some(path)) = (ai_tag.as_ref(), ai_path.as_ref()) {
            ui.set_ai_ref_tag(tag.into());
            ui.set_ai_ref_path(path.into());

            // Auto-register in the background
            let path_clone = path.clone();
            let tag_clone = tag.clone();
            thread::spawn(move || {
                match python_connector::register_reference_face(&path_clone, &tag_clone) {
                    Ok(_) => println!("[INFO] Auto-registered AI Reference Face: {}", tag_clone),
                    Err(e) => println!("[ERROR] Failed to auto-register AI Reference Face: {}", e),
                }
            });
        }
    }

    // ========================================================
    // AUTHENTICATION: LOGIN
    // ========================================================
    let ui_weak = ui.as_weak();
    let db_login = db.clone();

    ui.on_login(move |username, password| {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };

        let user_str = username.trim();
        let pass_str = password.trim();
        println!("[INFO] [UI Event] Login attempt for user: {}", user_str);

        if user_str.is_empty() || pass_str.is_empty() {
            ui.set_login_error("Please enter username and password.".into());
            return;
        }

        let Ok(conn) = db_login.lock() else {
            ui.set_login_error("Database lock error.".into());
            return;
        };

        match database::authenticate(&conn, user_str, pass_str) {
            Ok(Some(auth_user)) => {
                ui.set_login_error("".into());
                ui.set_auth_user(auth_user.username.clone().into());
                ui.set_auth_role(auth_user.role.clone().into());
                ui.set_is_authenticated(true);

                // Add audit entry in notifications
                let mut notifs: Vec<Notification> = ui.get_notifications().iter().collect();
                notifs.insert(
                    0,
                    Notification {
                        time: chrono::Local::now().format("%H:%M:%S").to_string().into(),
                        message: format!("User '{}' authenticated [{}]", auth_user.username, auth_user.role).into(),
                        kind: NotifKind::Info,
                        camera_id: "".into(),
                        media_path: "".into(),
                    },
                );
                ui.set_notifications(Rc::new(slint::VecModel::from(notifs)).into());

                ui.set_toast_message(format!("Welcome, {}. Surveillance console active.", auth_user.username).into());
                ui.set_toast_kind(NotifKind::Info);
                ui.set_toast_visible(true);
            }
            Ok(None) => {
                ui.set_login_error("Invalid credentials. Verify username and password.".into());
            }
            Err(e) => {
                ui.set_login_error(format!("Authentication error: {}", e).into());
            }
        }
    });

    // ========================================================
    // AUTHENTICATION: LOGOUT
    // ========================================================
    let ui_weak = ui.as_weak();

    ui.on_logout(move || {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };

        ui.set_is_authenticated(false);
        ui.set_auth_user("".into());
        ui.set_auth_role("".into());
        ui.set_login_error("".into());

        ui.set_toast_message("Security terminal locked.".into());
        ui.set_toast_kind(NotifKind::Info);
        ui.set_toast_visible(true);
    });

    // ========================================================
    // SEARCH CAMERAS (ONVIF DISCOVERY & RTSP RESOLUTION)
    // ========================================================
    let ui_weak = ui.as_weak();
    let rt_handle_discover = rt_handle.clone();
    let frame_tx_discover = frame_tx.clone();
    let stream_registry_discover = stream_registry.clone();
    let db_discover = db.clone();

    ui.on_search_cameras(move || {
        let ui_weak = ui_weak.clone();
        let rt_handle_discover = rt_handle_discover.clone();
        let frame_tx_discover = frame_tx_discover.clone();
        let stream_registry_discover = stream_registry_discover.clone();
        let db_discover = db_discover.clone();

        let (username, password) = if let Some(ui) = ui_weak.upgrade() {
            ui.set_is_scanning(true);
            ui.set_toast_message("Searching network for ONVIF cameras...".into());
            ui.set_toast_kind(NotifKind::Info);
            ui.set_toast_visible(true);
            (ui.get_default_user().to_string(), ui.get_default_pass().to_string())
        } else {
            ("cam".to_string(), "12345678".to_string())
        };

        let user = if username.is_empty() { "cam".to_string() } else { username };
        let pass = if password.is_empty() { "12345678".to_string() } else { password };

        thread::spawn(move || {
            let result = python_connector::discover_cameras(&user, &pass, 3);

            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = ui_weak.upgrade() else {
                    return;
                };

                ui.set_is_scanning(false);

                match result {
                    Ok(cameras) => {
                        let Ok(conn) = db_discover.lock() else {
                            return;
                        };

                        for camera in &cameras {
                            let _ = database::upsert_camera(&conn, camera);
                        }

                        // Reload persistent state into UI
                        sync_ui_cameras_from_db(&ui, &conn);

                        // Start concurrent streams for newly found cameras
                        for camera in &cameras {
                            if !camera.rtsp.is_empty() {
                                streaming::start_camera_stream(
                                    &rt_handle_discover,
                                    stream_registry_discover.clone(),
                                    camera.id.clone(),
                                    camera.rtsp.clone(),
                                    frame_tx_discover.clone(),
                                );
                            }
                        }

                        let msg = if cameras.is_empty() {
                            "Discovery complete. No new ONVIF devices found.".to_string()
                        } else {
                            format!("Discovery complete. {} camera(s) detected.", cameras.len())
                        };

                        // Add to notification log
                        let mut notifs: Vec<Notification> = ui.get_notifications().iter().collect();
                        notifs.insert(
                            0,
                            Notification {
                                time: chrono::Local::now().format("%H:%M:%S").to_string().into(),
                                message: msg.clone().into(),
                                kind: NotifKind::Info,
                                camera_id: "".into(),
                                media_path: "".into(),
                            },
                        );
                        ui.set_notifications(Rc::new(slint::VecModel::from(notifs)).into());

                        ui.set_toast_message(msg.into());
                        ui.set_toast_kind(NotifKind::Info);
                        ui.set_toast_visible(true);
                    }
                    Err(error) => {
                        eprintln!("Discovery error: {}", error);
                        ui.set_toast_message("Camera discovery encountered an issue.".into());
                        ui.set_toast_kind(NotifKind::Alert);
                        ui.set_toast_visible(true);
                    }
                }
            });
        });
    });

    // ========================================================
    // SYNC CLOUD
    // ========================================================
    let ui_weak = ui.as_weak();
    let db_sync = db.clone();

    ui.on_sync_cloud(move || {
        let ui_weak = ui_weak.clone();
        let db_sync = db_sync.clone();

        if let Some(ui) = ui_weak.upgrade() {
            ui.set_is_syncing(true);
        }

        thread::spawn(move || {
            let payload = {
                let Ok(conn) = db_sync.lock() else { return; };
                let cams = database::get_cameras(&conn).unwrap_or_default();
                serde_json::to_string(&cams).unwrap_or_else(|_| "[]".into())
            };

            let sync_res = python_connector::sync_cloud(&payload);

            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = ui_weak.upgrade() else {
                    return;
                };

                ui.set_is_syncing(false);

                match sync_res {
                    Ok(resp) => {
                        let mut notifs: Vec<Notification> = ui.get_notifications().iter().collect();
                        
                        if let Ok(conn) = db_sync.lock() {
                            let _ = database::mark_events_synced(&conn);
                            let _ = database::cleanup_old_events(&conn);
                        }

                        notifs.insert(
                            0,
                            Notification {
                                time: chrono::Local::now().format("%H:%M:%S").to_string().into(),
                                message: resp.message.clone().into(),
                                kind: NotifKind::Info,
                                camera_id: "".into(),
                                media_path: "".into(),
                            },
                        );
                        ui.set_notifications(Rc::new(slint::VecModel::from(notifs)).into());

                        ui.set_toast_message(resp.message.into());
                        ui.set_toast_kind(NotifKind::Info);
                        ui.set_toast_visible(true);
                    }
                    Err(e) => {
                        ui.set_toast_message(format!("Sync failed: {}", e).into());
                        ui.set_toast_kind(NotifKind::Alert);
                        ui.set_toast_visible(true);
                    }
                }
            });
        });
    });

    // ========================================================
    // CHECK UPDATES
    // ========================================================
    let ui_weak = ui.as_weak();

    ui.on_check_updates(move || {
        let ui_weak = ui_weak.clone();

        thread::spawn(move || {
            let update_res = python_connector::check_updates();

            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = ui_weak.upgrade() else {
                    return;
                };

                match update_res {
                    Ok(info) => {
                        ui.set_update_available(info.update_available);

                        let mut notifs: Vec<Notification> = ui.get_notifications().iter().collect();
                        notifs.insert(
                            0,
                            Notification {
                                time: chrono::Local::now().format("%H:%M:%S").to_string().into(),
                                message: format!("{}: v{} ({})", info.title, info.latest_version, info.details).into(),
                                kind: NotifKind::Update,
                                camera_id: "".into(),
                                media_path: "".into(),
                            },
                        );
                        ui.set_notifications(Rc::new(slint::VecModel::from(notifs)).into());

                        ui.set_toast_message(
                            format!("Update Available: v{} - {}", info.latest_version, info.title).into()
                        );
                        ui.set_toast_kind(NotifKind::Update);
                        ui.set_toast_visible(true);
                    }
                    Err(e) => {
                        ui.set_toast_message(format!("Update check failed: {}", e).into());
                        ui.set_toast_kind(NotifKind::Alert);
                        ui.set_toast_visible(true);
                    }
                }
            });
        });
    });

    // ========================================================
    // REMOVE CAMERA
    // ========================================================
    let ui_weak = ui.as_weak();
    let db_remove = db.clone();
    let stream_registry_remove = stream_registry.clone();
    let selected_camera_remove = selected_camera.clone();

    ui.on_remove_camera(move |id| {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };

        // Stop the background stream
        stream_registry_remove.stop(&id);

        // Delete from database
        if let Ok(conn) = db_remove.lock() {
            let _ = database::delete_camera(&conn, &id);
            sync_ui_cameras_from_db(&ui, &conn);
        }

        // If selected camera was removed, reset live view
        let is_curr = {
            let curr = selected_camera_remove.lock().unwrap();
            *curr == id.as_str()
        };

        if is_curr {
            *selected_camera_remove.lock().unwrap() = String::new();
            ui.set_selected_camera_id("".into());
            ui.set_stream_active(false);
        }

        ui.set_toast_message("Camera removed.".into());
        ui.set_toast_kind(NotifKind::Info);
        ui.set_toast_visible(true);
    });

    // ========================================================
    // RENAME CAMERA
    // ========================================================
    let ui_weak = ui.as_weak();
    let db_rename = db.clone();

    ui.on_rename_camera(move |id, new_tag| {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };

        if let Ok(conn) = db_rename.lock() {
            let _ = database::rename_camera(&conn, &id, &new_tag);
            sync_ui_cameras_from_db(&ui, &conn);
        }

        ui.set_toast_message("Camera renamed.".into());
        ui.set_toast_kind(NotifKind::Info);
        ui.set_toast_visible(true);
    });

    // ========================================================
    // TOGGLE CAMERA RESTRICTED MODE
    // ========================================================
    let ui_weak_toggle = ui.as_weak();
    let db_toggle = db.clone();
    ui.on_toggle_restricted_mode(move |cam_id, is_restricted| {
        let cam_id_str = cam_id.to_string();
        if let Ok(conn) = db_toggle.lock() {
            let _ = database::set_camera_restricted_mode(&conn, &cam_id_str, is_restricted);
        }
        
        if let Some(ui) = ui_weak_toggle.upgrade() {
            let model = ui.get_cameras();
            for i in 0..model.row_count() {
                if let Some(mut cam) = model.row_data(i) {
                    if cam.id == cam_id_str {
                        cam.is_restricted = is_restricted;
                        model.set_row_data(i, cam);
                        break;
                    }
                }
            }
        }
    });

    // ========================================================
    // MANUALLY ADD CAMERA
    // ========================================================
    let ui_weak = ui.as_weak();
    let db_add = db.clone();
    let rt_handle_add = rt_handle.clone();
    let frame_tx_add = frame_tx.clone();
    let stream_registry_add = stream_registry.clone();
    let selected_camera_add = selected_camera.clone();

    ui.on_add_camera_manual(move |ip_or_url| {
        let ui_weak = ui_weak.clone();
        let db_add = db_add.clone();
        let rt_handle_add = rt_handle_add.clone();
        let frame_tx_add = frame_tx_add.clone();
        let stream_registry_add = stream_registry_add.clone();
        let selected_camera_add = selected_camera_add.clone();

        let (user, pass) = if let Some(ui) = ui_weak.upgrade() {
            (ui.get_default_user().to_string(), ui.get_default_pass().to_string())
        } else {
            ("cam".to_string(), "12345678".to_string())
        };

        let raw_target = ip_or_url.to_string();

        thread::spawn(move || {
            let resolved = python_connector::resolve_manual_camera(&raw_target, &user, &pass);

            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = ui_weak.upgrade() else {
                    return;
                };

                match resolved {
                    Ok(Some(cam)) => {
                        if let Ok(conn) = db_add.lock() {
                            let _ = database::upsert_camera(&conn, &cam);
                            sync_ui_cameras_from_db(&ui, &conn);
                        }

                        // Start stream & auto-select
                        streaming::start_camera_stream(
                            &rt_handle_add,
                            stream_registry_add.clone(),
                            cam.id.clone(),
                            cam.rtsp.clone(),
                            frame_tx_add.clone(),
                        );

                        *selected_camera_add.lock().unwrap() = cam.id.clone();
                        ui.set_selected_camera_id(cam.id.clone().into());
                        ui.set_stream_active(false);

                        ui.set_toast_message(format!("Camera '{}' added.", cam.name).into());
                        ui.set_toast_kind(NotifKind::Info);
                        ui.set_toast_visible(true);
                    }
                    Ok(None) | Err(_) => {
                        // Fallback manual entry
                        let id = generate_id(1);
                        let fallback_rtsp = format!("rtsp://{}:{}@{}:8554/live", user, pass, raw_target);
                        let fallback_cam = DiscoveredCamera {
                            id: id.clone(),
                            name: format!("Manual {}", raw_target),
                            ip: raw_target.clone(),
                            rtsp: fallback_rtsp.clone(),
                            onvif_uid: String::new(),
                            is_restricted: false,
                        };

                        if let Ok(conn) = db_add.lock() {
                            let _ = database::upsert_camera(&conn, &fallback_cam);
                            sync_ui_cameras_from_db(&ui, &conn);
                        }

                        streaming::start_camera_stream(
                            &rt_handle_add,
                            stream_registry_add.clone(),
                            id.clone(),
                            fallback_rtsp,
                            frame_tx_add.clone(),
                        );

                        *selected_camera_add.lock().unwrap() = id.clone();
                        ui.set_selected_camera_id(id.into());

                        ui.set_toast_message("Camera added (manual fallback).".into());
                        ui.set_toast_kind(NotifKind::Info);
                        ui.set_toast_visible(true);
                    }
                }
            });
        });
    });

    // ========================================================
    // SELECT CAMERA
    // ========================================================
    let ui_weak = ui.as_weak();
    let rt_handle_select = rt_handle.clone();
    let frame_tx_select = frame_tx.clone();
    let stream_registry_select = stream_registry.clone();
    let selected_camera_select = selected_camera.clone();
    let db_select = db.clone();

    let ui_weak_snapshot = ui.as_weak();
    ui.on_load_snapshot(move |path| {
        let Some(ui) = ui_weak_snapshot.upgrade() else { return; };
        if let Ok(img) = image::open(path.as_str()) {
            let rgba = img.to_rgba8();
            let buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
                rgba.as_raw(),
                rgba.width(),
                rgba.height(),
            );
            ui.set_snapshot_image(slint::Image::from_rgba8(buffer));
        }
    });

    ui.on_select_camera(move |camera_id| {
        let Some(ui) = ui_weak.upgrade() else {
            return;
        };
        println!("[INFO] [UI Event] Selected camera changed to: {}", camera_id);

        *selected_camera_select.lock().unwrap() = camera_id.to_string();
        ui.set_selected_camera_id(camera_id.clone());
        ui.set_stream_active(false);

        // Ensure this camera's stream is actively producing frames
        let Ok(conn) = db_select.lock() else {
            return;
        };
        let Ok(cameras) = database::get_cameras(&conn) else {
            return;
        };

        if let Some(camera) = cameras.iter().find(|c| c.id == camera_id.as_str()) {
            if !camera.rtsp.is_empty() {
                streaming::start_camera_stream(
                    &rt_handle_select,
                    stream_registry_select.clone(),
                    camera.id.clone(),
                    camera.rtsp.clone(),
                    frame_tx_select.clone(),
                );
            }
        }
    });

    let ui_weak_ai_sel = ui.as_weak();
    ui.on_select_ai_reference(move || {
        let Some(ui) = ui_weak_ai_sel.upgrade() else { return; };
        
        // Spawn a thread since rfd blocks
        let thread_ui_weak = ui_weak_ai_sel.clone();
        thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Images", &["png", "jpg", "jpeg", "webp"])
                .pick_file() {
                    
                let path_str = path.to_string_lossy().to_string();
                let _ = slint::invoke_from_event_loop(move || {
                    if let Some(ui) = thread_ui_weak.upgrade() {
                        ui.set_ai_ref_path(path_str.into());
                    }
                });
            }
        });
    });

    let ui_weak_ai_reg = ui.as_weak();
    let db_ai_reg = db.clone();
    ui.on_register_ai_reference(move || {
        let Some(ui) = ui_weak_ai_reg.upgrade() else { return; };
        
        let tag = ui.get_ai_ref_tag().to_string();
        let path = ui.get_ai_ref_path().to_string();
        
        if tag.is_empty() || path.is_empty() {
            ui.set_toast_message("Error: Tag and Path required for AI Reference.".into());
            ui.set_toast_kind(NotifKind::Alert);
            ui.set_toast_visible(true);
            return;
        }

        // Save to DB
        if let Ok(conn) = db_ai_reg.lock() {
            let _ = database::set_setting(&conn, "ai_ref_tag", &tag);
            let _ = database::set_setting(&conn, "ai_ref_path", &path);
        }

        ui.set_toast_message(format!("Registering identity '{}'...", tag).into());
        ui.set_toast_kind(NotifKind::Info);
        ui.set_toast_visible(true);

        let thread_ui_weak = ui_weak_ai_reg.clone();
        thread::spawn(move || {
            let res = python_connector::register_reference_face(&path, &tag);
            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = thread_ui_weak.upgrade() else { return; };
                match res {
                    Ok(_) => {
                        ui.set_toast_message(format!("Identity '{}' successfully registered to AI.", tag).into());
                        ui.set_toast_kind(NotifKind::Info);
                        ui.set_toast_visible(true);
                    }
                    Err(e) => {
                        ui.set_toast_message(format!("Failed to register identity: {}", e).into());
                        ui.set_toast_kind(NotifKind::Alert);
                        ui.set_toast_visible(true);
                    }
                }
            });
        });
    });

    // ========================================================
    // DISMISS TOAST
    // ========================================================
    let ui_weak = ui.as_weak();
    ui.on_dismiss_toast(move || {
        if let Some(ui) = ui_weak.upgrade() {
            ui.set_toast_visible(false);
        }
    });

    // ========================================================
    // START APPLICATION
    // ========================================================
    println!("Starting IBVAP Edge Command Center...");
    let result = ui.run();
    std::process::exit(0);
}
