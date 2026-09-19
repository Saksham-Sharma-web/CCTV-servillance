import re

# PATCH streaming.rs
with open("src/streaming.rs", "r") as f:
    streaming = f.read()

streaming = streaming.replace(
    "pub async fn run_aggregator(\n    mut rx: Receiver<FrameUpdate>,\n    ui_weak: slint::Weak<AppWindow>,\n    selected_camera: Arc<Mutex<String>>,\n    shared_alerts: Arc<Mutex<Vec<Notification>>>,\n    latest_frames: Arc<Mutex<HashMap<String, Vec<u8>>>>,\n    db_conn: Arc<Mutex<rusqlite::Connection>>,\n    ws_sender: tokio::sync::broadcast::Sender<Notification>,\n)",
    "pub async fn run_aggregator(\n    mut rx: Receiver<FrameUpdate>,\n    ui_weak: slint::Weak<AppWindow>,\n    selected_camera: Arc<Mutex<String>>,\n    shared_alerts: Arc<Mutex<Vec<Notification>>>,\n    latest_frames: Arc<Mutex<HashMap<String, Vec<u8>>>>,\n    db_conn: Arc<Mutex<rusqlite::Connection>>,\n    ws_sender: tokio::sync::broadcast::Sender<Notification>,\n    camera_liveness: Arc<Mutex<HashMap<String, std::time::Instant>>>,\n)"
)

streaming = streaming.replace(
    "        // ── Push latest JPEG to shared map for Web Server MJPEG stream ────────",
    "        if let Ok(mut liveness) = camera_liveness.lock() {\n            liveness.insert(update.camera_id.clone(), std::time::Instant::now());\n        }\n\n        // ── Push latest JPEG to shared map for Web Server MJPEG stream ────────"
)

with open("src/streaming.rs", "w") as f:
    f.write(streaming)

# PATCH main.rs
with open("src/main.rs", "r") as f:
    main_rs = f.read()

# 1. Init camera_liveness
main_rs = main_rs.replace(
    "let selected_camera: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));",
    "let selected_camera: Arc<Mutex<String>> = Arc::new(Mutex::new(String::new()));\n    let camera_liveness: Arc<Mutex<HashMap<String, std::time::Instant>>> = Arc::new(Mutex::new(HashMap::new()));"
)

# 2. Pass to run_aggregator
main_rs = main_rs.replace(
    "        db.clone(),\n        tx_ws.clone(),\n    ));",
    "        db.clone(),\n        tx_ws.clone(),\n        camera_liveness.clone(),\n    ));"
)

# 3. Add the liveness background task
liveness_task = """
    let ui_weak_liveness = ui.as_weak();
    let liveness_clone = camera_liveness.clone();
    rt.spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(2));
        loop {
            interval.tick().await;
            
            let mut statuses = Vec::new();
            let now = std::time::Instant::now();
            if let Ok(liveness) = liveness_clone.lock() {
                for (id, time) in liveness.iter() {
                    let online = now.duration_since(*time).as_secs() < 5;
                    statuses.push((id.clone(), online));
                }
            }

            let ui_weak = ui_weak_liveness.clone();
            let _ = slint::invoke_from_event_loop(move || {
                let Some(ui) = ui_weak.upgrade() else { return };
                let model = ui.get_cameras();
                for i in 0..model.row_count() {
                    if let Some(mut cam) = model.row_data(i) {
                        let cam_id = cam.id.to_string();
                        let mut updated = false;
                        
                        if let Some((_, online)) = statuses.iter().find(|(id, _)| id == &cam_id) {
                            if cam.is_online != *online {
                                cam.is_online = *online;
                                if !*online {
                                    cam.last_seen = "Offline".into();
                                } else {
                                    cam.last_seen = "Online".into();
                                }
                                updated = true;
                            }
                        } else {
                            if cam.is_online {
                                cam.is_online = false;
                                cam.last_seen = "Connecting...".into();
                                updated = true;
                            }
                        }
                        
                        if updated {
                            model.set_row_data(i, cam);
                        }
                    }
                }
            });
        }
    });
"""

main_rs = main_rs.replace(
    "    // Auto-start streams for existing database cameras",
    liveness_task + "\n    // Auto-start streams for existing database cameras"
)

with open("src/main.rs", "w") as f:
    f.write(main_rs)
