use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::Deserialize;
use slint::Model;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::{database, AppWindow, NotifKind, Notification};

// ================================================================
// One AI event, as produced by AnalyticsEvent.to_dict() in Python
// ================================================================

#[derive(Debug, Deserialize, Default)]
pub struct AiEvent {
    #[serde(default)]
    pub event_type: String,
    #[serde(default)]
    pub confidence: f64,
}

// One AI-annotated frame + whatever events fired while producing it.
pub struct FrameUpdate {
    pub camera_id: String,
    pub rgba: Vec<u8>,
    pub rgb: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub events: Vec<AiEvent>,
}

// ================================================================
// Tracks which cameras already have a running Tokio stream task.
// Selecting a camera twice never spins up a second capture loop.
// ================================================================

#[derive(Clone, Default)]
pub struct StreamRegistry {
    stop_flags: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl StreamRegistry {
    pub fn is_running(&self, camera_id: &str) -> bool {
        self.stop_flags
            .lock()
            .unwrap()
            .contains_key(camera_id)
    }

    fn register(&self, camera_id: &str) -> Arc<AtomicBool> {
        let flag = Arc::new(AtomicBool::new(false));

        self.stop_flags
            .lock()
            .unwrap()
            .insert(camera_id.to_string(), flag.clone());

        flag
    }

    pub fn stop(&self, camera_id: &str) {
        if let Some(flag) = self.stop_flags.lock().unwrap().remove(camera_id) {
            flag.store(true, Ordering::Relaxed);
        }
    }
}

// ================================================================
// Spawns one Tokio task per camera.
// Capture + AI inference is blocking Python/OpenCV work,
// so it runs on Tokio's blocking-thread pool (spawn_blocking).
// ================================================================

pub fn start_camera_stream(
    rt: &tokio::runtime::Handle,
    registry: StreamRegistry,
    camera_id: String,
    rtsp_url: String,
    tx: UnboundedSender<FrameUpdate>,
) {
    if rtsp_url.is_empty() || registry.is_running(&camera_id) {
        return;
    }

    let stop_flag = registry.register(&camera_id);

    rt.spawn_blocking(move || {
        let outcome: PyResult<()> = Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let cwd = std::env::current_dir().unwrap_or_default();

            sys.getattr("path")?.call_method1(
                "insert",
                (0, cwd.to_string_lossy().to_string()),
            )?;

            let module = PyModule::import(py, "live_streaming")?;
            let class = module.getattr("LiveCameraStream")?;
            let stream = class.call1((camera_id.clone(), rtsp_url.clone()))?;

            while !stop_flag.load(Ordering::Relaxed) {
                let result = stream.call_method0("next_frame")?;

                if result.is_none() {
                    // Queue empty — release GIL and yield to Python capture thread
                    py.allow_threads(|| {
                        std::thread::sleep(std::time::Duration::from_millis(10));
                    });
                    continue;
                }

                let (bgr, width, height, events_json): (Vec<u8>, u32, u32, String) =
                    result.extract()?;

                let rgba = bgr_to_rgba(&bgr, width, height);
                let rgb = bgr_to_rgb(&bgr, width, height);
                let events: Vec<AiEvent> =
                    serde_json::from_str(&events_json).unwrap_or_default();

                if tx
                    .send(FrameUpdate {
                        camera_id: camera_id.clone(),
                        rgba,
                        rgb,
                        width,
                        height,
                        events,
                    })
                    .is_err()
                {
                    break; // UI side is gone, stop pulling frames
                }
                // No artificial sleep — the Python side drives the rate.
                // The live queue blocks until a new frame arrives, so we
                // naturally run at (at most) camera FPS.
            }

            let _ = stream.call_method0("release");
            Ok(())
        });

        if let Err(e) = outcome {
            eprintln!("[stream] Camera '{}' stopped: {}", camera_id, e);
        }
    });
}

fn bgr_to_rgba(bgr: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let mut rgba = Vec::with_capacity(pixel_count * 4);

    for chunk in bgr.chunks_exact(3).take(pixel_count) {
        rgba.push(chunk[2]); // R
        rgba.push(chunk[1]); // G
        rgba.push(chunk[0]); // B
        rgba.push(255);       // A
    }

    rgba
}

fn bgr_to_rgb(bgr: &[u8], width: u32, height: u32) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    let mut rgb = Vec::with_capacity(pixel_count * 3);

    for chunk in bgr.chunks_exact(3).take(pixel_count) {
        rgb.push(chunk[2]); // R
        rgb.push(chunk[1]); // G
        rgb.push(chunk[0]); // B
    }

    rgb
}

// ================================================================
// Single consumer task: drains frames from every camera producer
// and applies them to the UI.
//
// AI events are persisted for ALL cameras regardless of which one
// is on screen. The video frame is only painted when it belongs
// to the selected camera.
// ================================================================

pub async fn run_aggregator(
    mut rx: UnboundedReceiver<FrameUpdate>,
    ui_weak: slint::Weak<AppWindow>,
    selected_camera: Arc<Mutex<String>>,
    shared_alerts: Arc<Mutex<Vec<Notification>>>,
    db_conn: Arc<Mutex<rusqlite::Connection>>,
) {
    while let Some(update) = rx.recv().await {
        let ui_weak = ui_weak.clone();
        let selected_camera = selected_camera.clone();
        let shared_alerts = shared_alerts.clone();
        let db_conn = db_conn.clone();

        let _ = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            if !update.events.is_empty() {
                let mut notifications: Vec<Notification> =
                    ui.get_notifications().iter().collect();

                let mut new_notifs = Vec::new();

                // Resolve the human-readable camera name once per batch
                let camera_name = if let Ok(conn) = db_conn.lock() {
                    database::get_camera_name(&conn, &update.camera_id)
                } else {
                    update.camera_id.clone()
                };

                for event in &update.events {
                    let kind = if event.event_type.contains("BLACKLIST")
                        || event.event_type.contains("FENCE")
                        || event.event_type.contains("SUSPICIOUS")
                        || event.event_type.contains("UNATTENDED")
                        || event.event_type.contains("INTRUSION")
                    {
                        NotifKind::Alert
                    } else {
                        NotifKind::Info
                    };

                    let event_id = format!(
                        "evt_{}_{}",
                        update.camera_id,
                        chrono::Local::now().timestamp_millis()
                    );
                    let media_path = format!("events/{}.jpg", event_id);

                    let notif = Notification {
                        time: chrono::Local::now()
                            .format("%H:%M:%S")
                            .to_string()
                            .into(),
                        message: format!(
                            "{} on {} [{:.0}% confidence]",
                            event.event_type.replace('_', " "),
                            camera_name,
                            event.confidence * 100.0
                        )
                        .into(),
                        kind,
                        camera_id: update.camera_id.clone().into(),
                        media_path: media_path.clone().into(),
                    };

                    // Save snapshot from raw frame
                    if let Some(img) = image::RgbImage::from_raw(
                        update.width,
                        update.height,
                        update.rgb.clone(),
                    ) {
                        let _ = std::fs::create_dir_all("events");
                        if let Err(e) = img.save(&media_path) {
                            eprintln!("Failed to save snapshot: {}", e);
                        }
                    }

                    // Persist event to SQLite with camera_name captured now
                    if let Ok(conn) = db_conn.lock() {
                        let _ = database::insert_event(
                            &conn,
                            &event_id,
                            &update.camera_id,
                            &camera_name,
                            &event.event_type,
                            event.confidence,
                            &notif.time.to_string(),
                            &media_path,
                        );
                    }

                    new_notifs.push(notif.clone());
                    notifications.insert(0, notif);
                }

                // Push to in-memory shared list for the web server fallback
                if let Ok(mut shared) = shared_alerts.lock() {
                    for n in new_notifs.into_iter().rev() {
                        shared.insert(0, n);
                    }
                    shared.truncate(100);
                }

                notifications.truncate(50);

                ui.set_notifications(
                    Rc::new(slint::VecModel::from(notifications)).into(),
                );

                ui.set_unread_alert_count(
                    ui.get_unread_alert_count() + update.events.len() as i32,
                );

                ui.set_toast_message(
                    format!(
                        "{} AI event(s) on {}",
                        update.events.len(),
                        camera_name
                    )
                    .into(),
                );
                ui.set_toast_kind(NotifKind::Alert);
                ui.set_toast_visible(true);
            }

            let current_selected = selected_camera.lock().unwrap().clone();
            let is_selected = current_selected == update.camera_id;

            if is_selected {
                let mut buffer = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(
                    update.width,
                    update.height,
                );

                buffer.make_mut_bytes().copy_from_slice(&update.rgba);

                ui.set_live_frame(slint::Image::from_rgba8(buffer));
                ui.set_stream_active(true);
            }
        });
    }
}
