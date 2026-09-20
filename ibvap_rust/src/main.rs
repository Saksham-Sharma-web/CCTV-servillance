#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

mod database;
mod python_connector;
mod streaming;
mod web_server;
mod tauri_cmds;

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
    #[serde(default)]
    pub onvif_uid: String,
    #[serde(default)]
    pub is_restricted: bool,
    #[serde(default)]
    pub rtsp_user: Option<String>,
    #[serde(default)]
    pub rtsp_pass: Option<String>,
}

impl DiscoveredCamera {
    pub fn get_active_rtsp(&self) -> String {
        let custom_user = self.rtsp_user.clone().unwrap_or_default();
        let custom_pass = self.rtsp_pass.clone().unwrap_or_default();

        if !custom_user.is_empty() && !custom_pass.is_empty() && self.rtsp.starts_with("rtsp://") {
            let without_scheme = &self.rtsp[7..];
            let host_path = if let Some(idx) = without_scheme.find('@') {
                &without_scheme[idx + 1..]
            } else {
                without_scheme
            };
            format!("rtsp://{}:{}@{}", custom_user, custom_pass, host_path)
        } else {
            self.rtsp.clone()
        }
    }
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------
fn main() {
    pyo3::prepare_freethreaded_python();

    let db_conn = database::open().expect("Failed to open cameras.db");
    println!("[INFO] Connected to local SQLite database: cameras.db");
    let db = Arc::new(Mutex::new(db_conn));

    let local_ip = local_ip_address::local_ip().map(|ip| ip.to_string()).unwrap_or_else(|_| "localhost".to_string());
    println!("[INFO] Web Server IP: {}", local_ip);

    let rt = tokio::runtime::Runtime::new().expect("Failed to start Tokio runtime");
    let rt_handle = rt.handle().clone();

    let (frame_tx, frame_rx) = tokio::sync::mpsc::channel::<streaming::FrameUpdate>(6);
    let selected_camera: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));
    let camera_liveness: Arc<Mutex<HashMap<String, std::time::Instant>>> = Arc::new(Mutex::new(HashMap::new()));
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

    // Spawn Aggregator (Slint stripped out, just tracks latest frames for Web MJPEG)
    rt.spawn(streaming::run_aggregator(
        frame_rx,
        selected_camera.clone(),
        shared_alerts,
        latest_frames.clone(),
        db.clone(),
        tx_ws.clone(),
        camera_liveness.clone(),
    ));

    // Liveness checker
    let liveness_clone = camera_liveness.clone();
    rt.spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            interval.tick().await;
            let now = std::time::Instant::now();
            let mut offline_cams = Vec::new();
            {
                let mut liveness = liveness_clone.lock().unwrap();
                for (cam_id, last_seen) in liveness.iter() {
                    if now.duration_since(*last_seen).as_secs() > 3 {
                        offline_cams.push(cam_id.clone());
                    }
                }
                for id in &offline_cams {
                    liveness.remove(id);
                }
            }
            if !offline_cams.is_empty() {
                println!("[WARN] Cameras offline: {:?}", offline_cams);
            }
        }
    });

    // Python process supervisor
    let db_clone = db.clone();
    let frame_tx_clone = frame_tx.clone();
    let stream_registry_clone = stream_registry.clone();
    let rt_handle_clone = rt_handle.clone();
    
    rt.spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            if let Ok(conn) = db_clone.lock() {
                if let Ok(cams) = database::get_cameras(&conn) {
                    for cam in cams {
                        if !stream_registry_clone.is_running(&cam.id) {
                            println!("[INFO] Auto-restarting stream for {}", cam.id);
                            streaming::start_camera_stream(
                                &rt_handle_clone,
                                stream_registry_clone.clone(),
                                cam.id.clone(),
                                cam.get_active_rtsp(),
                                frame_tx_clone.clone(),
                            );
                        }
                    }
                }
            }
        }
    });

    // Initial Stream Boot
    if let Ok(conn) = db.lock() {
        if let Ok(cams) = database::get_cameras(&conn) {
            for cam in cams {
                if !cam.rtsp.is_empty() {
                    streaming::start_camera_stream(
                        &rt_handle,
                        stream_registry.clone(),
                        cam.id.clone(),
                        cam.get_active_rtsp(),
                        frame_tx.clone(),
                    );
                }
            }
        }
    }

    println!("Starting IBVAP Tauri Command Center...");
    tauri::Builder::default()
        .manage(db) // Pass db connection to Tauri commands
        .invoke_handler(tauri::generate_handler![
            tauri_cmds::login,
            tauri_cmds::get_cameras,
            tauri_cmds::get_events,
            tauri_cmds::reset_password,
            tauri_cmds::factory_reset,
            tauri_cmds::add_camera,
            tauri_cmds::remove_camera,
            tauri_cmds::rename_camera,
            tauri_cmds::get_onvif_settings,
            tauri_cmds::save_onvif_settings,
            tauri_cmds::select_ai_reference,
            tauri_cmds::register_ai_reference,
            tauri_cmds::get_stream_token,
            tauri_cmds::discover_cameras
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
