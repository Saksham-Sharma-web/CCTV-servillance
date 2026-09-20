use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{Receiver, Sender};
use crate::database;
use crate::web_server::{Notification, NotifKind};
use pyo3::types::{PyAnyMethods, PyModuleMethods, PyDictMethods, PyListMethods};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct AiEvent {
    pub event_type: String,
    pub confidence: f32,
    pub metadata: String,
}

pub struct FrameUpdate {
    pub camera_id: String,
    pub jpeg: Vec<u8>,
    pub w: u32,
    pub h: u32,
    pub events: Vec<AiEvent>,
    pub t0_capture: f64,
    pub t1_encode_start: f64,
    pub t2_encode_end: f64,
    pub t3_pyo3: f64,
}

#[derive(Clone, Default)]
pub struct StreamRegistry {
    stop_flags: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl StreamRegistry {
    pub fn is_running(&self, id: &str) -> bool {
        self.stop_flags.lock().unwrap().contains_key(id)
    }

    fn register(&self, id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));
        self.stop_flags.lock().unwrap().insert(id.to_owned(), flag.clone());
        flag
    }

    pub fn stop(&self, id: &str) {
        if let Some(flag) = self.stop_flags.lock().unwrap().remove(id) {
            flag.store(true, Ordering::Relaxed);
        }
    }
}

pub fn start_camera_stream(
    rt_handle: &tokio::runtime::Handle,
    registry: StreamRegistry,
    camera_id: String,
    rtsp_url: String,
    frame_tx: Sender<FrameUpdate>,
) {
    if registry.is_running(&camera_id) {
        return;
    }
    let stop_flag = registry.register(&camera_id);
    let cam_id_clone = camera_id.clone();
    
    rt_handle.spawn_blocking(move || {
        let stream_obj: pyo3::PyResult<pyo3::Py<pyo3::PyAny>> = pyo3::Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let cwd = std::env::current_dir().unwrap_or_default();
            sys.getattr("path")?.call_method1("insert", (0, cwd.to_string_lossy().to_string()))?;
            let module = pyo3::types::PyModule::import(py, "live_streaming")?;
            let class  = module.getattr("LiveCameraStream")?;
            let stream = class.call1((cam_id_clone.clone(), rtsp_url.clone()))?;
            Ok(stream.into())
        });

        let stream = match stream_obj {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[ERROR] [stream] Failed to initialize camera '{}': {}", cam_id_clone, e);
                registry.stop(&cam_id_clone);
                return;
            }
        };

        while !stop_flag.load(Ordering::Relaxed) {
            let tx_status = pyo3::Python::with_gil(|py| -> pyo3::PyResult<bool> {
                let result = stream.call_method0(py, "next_frame")?;
                if result.is_none(py) {
                    py.allow_threads(|| std::thread::sleep(std::time::Duration::from_millis(5)));
                    return Ok(true);
                }

                let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
                let bytes_obj = tuple.get_item(0)?;
                let jpeg: Vec<u8> = bytes_obj.extract()?;
                let w: u32 = tuple.get_item(1)?.extract()?;
                let h: u32 = tuple.get_item(2)?.extract()?;
                let events_str: String = tuple.get_item(3)?.extract()?;
                let t0_capture: f64 = tuple.get_item(4)?.extract()?;
                let t1_encode_start: f64 = tuple.get_item(5)?.extract()?;
                let t2_encode_end: f64 = tuple.get_item(6)?.extract()?;
                let t3_pyo3 = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64();

                let mut ai_events = Vec::new();
                if let Ok(events_json) = serde_json::from_str::<Vec<serde_json::Value>>(&events_str) {
                    for ev in events_json {
                        ai_events.push(AiEvent {
                            event_type: ev["type"].as_str().unwrap_or("").to_string(),
                            confidence: ev["confidence"].as_f64().unwrap_or(0.0) as f32,
                            metadata: String::new(),
                        });
                    }
                }

                let latency_ms = (t3_pyo3 - t0_capture) * 1000.0;
                if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open("latency_metrics.csv") {
                    let dt = chrono::Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();
                    use std::io::Write;
                    let _ = writeln!(file, "{},{},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.2}",
                        dt, cam_id_clone, t0_capture, t1_encode_start, t2_encode_end, t2_encode_end, t2_encode_end, t3_pyo3, latency_ms);
                }

                let update = FrameUpdate {
                    camera_id: cam_id_clone.clone(),
                    jpeg,
                    w,
                    h,
                    events: ai_events,
                    t0_capture,
                    t1_encode_start,
                    t2_encode_end,
                    t3_pyo3,
                };

                Ok(frame_tx.try_send(update).is_ok())
            });

            if let Err(e) = tx_status {
                eprintln!("[ERROR] stream failed: {}", e);
                break;
            }
        }
        registry.stop(&cam_id_clone);
    });
}

const UI_HZ: std::time::Duration = std::time::Duration::from_millis(50);

pub async fn run_aggregator(
    mut rx: Receiver<FrameUpdate>,
    _selected_camera: Arc<Mutex<String>>,
    shared_alerts: Arc<Mutex<Vec<Notification>>>,
    latest_frames: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    db_conn: Arc<Mutex<rusqlite::Connection>>,
    ws_sender: tokio::sync::broadcast::Sender<Notification>,
    camera_liveness: Arc<Mutex<HashMap<String, std::time::Instant>>>,
) {
    while let Some(update) = rx.recv().await {
        let has_events = !update.events.is_empty();

        if let Ok(mut liveness) = camera_liveness.lock() {
            liveness.insert(update.camera_id.clone(), std::time::Instant::now());
        }

        // ── Push latest JPEG to shared map for Web Server MJPEG stream ────────
        if let Ok(mut map) = latest_frames.lock() {
            map.insert(update.camera_id.clone(), update.jpeg.clone());
        }

        // ── DB + file I/O on Tokio thread ────────────────
        if has_events {
            let mut camera_name = update.camera_id.clone();
            let mut is_restricted = false;

            if let Ok(conn) = db_conn.lock() {
                camera_name = database::get_camera_name(&conn, &update.camera_id);
                is_restricted = conn.query_row(
                    "SELECT is_restricted FROM cameras WHERE id = ?1",
                    rusqlite::params![update.camera_id],
                    |row| row.get::<_, i32>(0).map(|v| v != 0)
                ).unwrap_or(false);
            }

            let mut new_notifs: Vec<Notification> = Vec::new();
            for event in &update.events {
                let is_info_event = if is_restricted {
                    event.event_type.contains("FACE_MATCHED") || 
                    event.event_type.contains("WATCHLIST_VEHICLE")
                } else {
                    event.event_type.contains("PERSON_DETECTED") ||
                    event.event_type.contains("FACE_MATCHED") ||
                    event.event_type.contains("UNKNOWN_PERSON") ||
                    event.event_type.contains("VEHICLE_DETECTED") ||
                    event.event_type.contains("PLATE_DETECTED")
                };

                let kind = if is_info_event { NotifKind::Info } else { NotifKind::Alert };

                let ts = chrono::Local::now();
                let event_id   = format!("evt_{}_{}", update.camera_id, ts.timestamp_millis());
                let media_path = format!("events/{}.jpg", event_id);
                println!(
                    "[INFO] Alert: camera='{}' type='{}' conf={:.0}%",
                    camera_name, event.event_type, event.confidence * 100.0
                );

                let notif = Notification {
                    time:       ts.format("%H:%M:%S").to_string().into(),
                    message:    format!(
                        "{} on {} [{:.0}% confidence]",
                        event.event_type.replace('_', " "),
                        camera_name,
                        event.confidence * 100.0
                    ).into(),
                    kind,
                    camera_id:  update.camera_id.clone().into(),
                    media_path: media_path.clone().into(),
                };

                let _ = std::fs::create_dir_all("events");
                let _ = std::fs::write(&media_path, &update.jpeg);

                if let Ok(conn) = db_conn.lock() {
                    let _ = database::insert_event(
                        &conn,
                        &event_id,
                        &update.camera_id,
                        &camera_name,
                        &event.event_type,
                        event.confidence as f64,
                        &notif.time.to_string(),
                        &media_path,
                    );
                }

                let _ = ws_sender.send(notif.clone());
                new_notifs.push(notif);
            }

            if let Ok(mut shared) = shared_alerts.lock() {
                for n in new_notifs.iter().rev() {
                    shared.insert(0, n.clone());
                }
                shared.truncate(100);
            }
        }
    }
}
