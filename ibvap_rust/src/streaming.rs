use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::Deserialize;
use slint::Model;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

use crate::{AppWindow, NotifKind, Notification};

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
    pub width: u32,
    pub height: u32,
    pub events: Vec<AiEvent>,
}

// ================================================================
// Tracks which cameras already have a running Tokio stream task, so
// selecting a camera twice (or re-discovering the same camera) never
// spins up a second capture loop for it.
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
// Spawns one Tokio task per camera. Capture + AI inference is
// blocking Python/OpenCV work, so it runs on Tokio's blocking-thread
// pool (`spawn_blocking`) — that's what lets several cameras run
// concurrently without one slow RTSP source starving the others or
// the Slint UI thread.
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
                    break;
                }

                let (bgr, width, height, events_json): (Vec<u8>, u32, u32, String) =
                    result.extract()?;

                let rgba = bgr_to_rgba(&bgr, width, height);
                let events: Vec<AiEvent> =
                    serde_json::from_str(&events_json).unwrap_or_default();

                if tx
                    .send(FrameUpdate {
                        camera_id: camera_id.clone(),
                        rgba,
                        width,
                        height,
                        events,
                    })
                    .is_err()
                {
                    break; // UI side is gone, stop pulling frames
                }

                // Cap AI throughput per camera — running full YOLO +
                // OCR at native FPS across several concurrent streams
                // isn't necessary and just burns CPU.
                py.allow_threads(|| {
                    std::thread::sleep(std::time::Duration::from_millis(120));
                });
            }

            let _ = stream.call_method0("release");
            Ok(())
        });

        if let Err(e) = outcome {
            eprintln!("Camera stream '{}' stopped: {}", camera_id, e);
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
        rgba.push(255); // A
    }

    rgba
}

// ================================================================
// Single consumer task: drains frames from every camera's producer
// task and applies them to the UI. AI events are logged for ALL
// cameras regardless of which one is on screen; the video frame
// itself is only painted when it belongs to the selected camera.
// ================================================================

pub async fn run_aggregator(
    mut rx: UnboundedReceiver<FrameUpdate>,
    ui_weak: slint::Weak<AppWindow>,
    selected_camera: Arc<Mutex<String>>,
    shared_alerts: Arc<Mutex<Vec<Notification>>>,
) {
    while let Some(update) = rx.recv().await {
        let ui_weak = ui_weak.clone();
        let selected_camera = selected_camera.clone();
        let shared_alerts = shared_alerts.clone();

        let _ = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak.upgrade() else {
                return;
            };

            if !update.events.is_empty() {
                let mut notifications: Vec<Notification> =
                    ui.get_notifications().iter().collect();

                let mut new_notifs = Vec::new();

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

                    let notif = Notification {
                        time: chrono::Local::now()
                            .format("%H:%M:%S")
                            .to_string()
                            .into(),
                        message: format!(
                            "{} on {} [{:.0}% confidence]",
                            event.event_type.replace('_', " "),
                            update.camera_id,
                            event.confidence * 100.0
                        )
                        .into(),
                        kind,
                        camera_id: update.camera_id.clone().into(),
                    };
                    
                    new_notifs.push(notif.clone());
                    notifications.insert(0, notif);
                }
                
                // Add to shared alerts for web server
                if let Ok(mut shared) = shared_alerts.lock() {
                    for n in new_notifs.into_iter().rev() {
                        shared.insert(0, n);
                    }
                    shared.truncate(100); // keep last 100
                }

                notifications.truncate(50);

                ui.set_notifications(
                    Rc::new(slint::VecModel::from(notifications)).into(),
                );

                ui.set_unread_alert_count(ui.get_unread_alert_count() + update.events.len() as i32);

                ui.set_toast_message(
                    format!(
                        "{} AI event(s) on {}",
                        update.events.len(),
                        update.camera_id
                    )
                    .into(),
                );
                ui.set_toast_kind(NotifKind::Alert);
                ui.set_toast_visible(true);
            }

            let is_selected = *selected_camera.lock().unwrap() == update.camera_id;

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
