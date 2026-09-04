use serde::{Deserialize, Serialize};
use slint::Model;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
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

    #[serde(default)]
    pub name: String,

    #[serde(default)]
    pub ip: String,

    #[serde(default)]
    pub rtsp: String,
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

    let name = if camera.name.is_empty() {
        format!("Camera {}", index + 1)
    } else {
        camera.name.clone()
    };

    Camera {
        id: id.into(),
        name: name.clone().into(),
        tag: name.into(),
        ip: camera.ip.clone().into(),
        is_online: true,
        has_onvif: true,
        last_seen: "Online".into(),
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
    let db = Arc::new(Mutex::new(db_conn));

    // Populate cameras in UI from database on startup
    {
        let conn = db.lock().unwrap();
        sync_ui_cameras_from_db(&ui, &conn);
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
        tokio::sync::mpsc::unbounded_channel::<streaming::FrameUpdate>();

    let selected_camera: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let stream_registry = streaming::StreamRegistry::default();
    
    let shared_alerts = Arc::new(Mutex::new(Vec::new()));

    // Spawn Web Server
    let web_state = web_server::AppState {
        alerts: shared_alerts.clone(),
        db_pool: db.clone(),
    };
    rt.spawn(async move {
        web_server::run(web_state).await;
    });

    rt.spawn(streaming::run_aggregator(
        frame_rx,
        ui.as_weak(),
        selected_camera.clone(),
        shared_alerts,
        db.clone(),
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
                *selected_camera.lock().unwrap() = first.id.clone();
                ui.set_selected_camera_id(first.id.clone().into());
            }
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
    ui.run()
}
