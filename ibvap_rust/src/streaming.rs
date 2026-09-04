use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::Deserialize;
use slint::Model;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{database, AppWindow, NotifKind, Notification};

// ================================================================
// AI event — from Python AnalyticsEvent.to_dict()
// ================================================================

#[derive(Debug, Deserialize, Default, Clone)]
pub struct AiEvent {
    #[serde(default)]
    pub event_type: String,
    #[serde(default)]
    pub confidence: f64,
}

// ================================================================
// FrameUpdate — carries raw JPEG bytes (Vec<u8> is Send).
// SharedPixelBuffer / Image are NOT created here; they are created
// on the UI thread inside invoke_from_event_loop.
// ================================================================

pub struct FrameUpdate {
    pub camera_id: String,
    pub jpeg: Vec<u8>,
    pub events: Vec<AiEvent>,
}

// ================================================================
// StreamRegistry — one stop-flag per camera
// ================================================================

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

// ================================================================
// Per-camera streaming task.
// Uses a BOUNDED sender (capacity 6).  When the channel is full
// (UI can't keep up), try_send() returns Err and the frame is
// dropped on the Rust side — natural backpressure, zero queue
// buildup, zero UI blocking.
// ================================================================

pub fn start_camera_stream(
    rt: &tokio::runtime::Handle,
    registry: StreamRegistry,
    camera_id: String,
    rtsp_url: String,
    tx: Sender<FrameUpdate>,
) {
    println!("[INFO] Attempting to start stream for camera '{}' at {}", camera_id, rtsp_url);

    if rtsp_url.is_empty() || registry.is_running(&camera_id) {
        println!("[WARN] Stream '{}' already running or no RTSP URL — skipped.", camera_id);
        return;
    }

    let stop_flag = registry.register(&camera_id);

    rt.spawn_blocking(move || {
        // 1. Acquire GIL once to setup the stream object
        let stream_obj: PyResult<Py<PyAny>> = Python::with_gil(|py| {
            let sys = py.import("sys")?;
            let cwd = std::env::current_dir().unwrap_or_default();
            sys.getattr("path")?.call_method1(
                "insert",
                (0, cwd.to_string_lossy().to_string()),
            )?;

            let module = PyModule::import(py, "live_streaming")?;
            let class  = module.getattr("LiveCameraStream")?;
            let stream = class.call1((camera_id.clone(), rtsp_url.clone()))?;
            Ok(stream.into())
        });

        let stream = match stream_obj {
            Ok(s) => s,
            Err(e) => {
                eprintln!("[ERROR] [stream] Failed to initialize camera '{}': {}", camera_id, e);
                return;
            }
        };

        // 2. Loop OUTSIDE the GIL closure.
        while !stop_flag.load(Ordering::Relaxed) {
            // 3. Acquire GIL per-frame. All Python objects are freed at the end of this closure!
            let tx_status = Python::with_gil(|py| -> PyResult<bool> {
                let result = stream.call_method0(py, "next_frame")?;

                if result.is_none(py) {
                    // No new frame yet — yield the GIL so Python threads run
                    py.allow_threads(|| {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    });
                    return Ok(true); // continue loop
                }

                let (jpeg, _w, _h, events_json): (Vec<u8>, u32, u32, String) =
                    result.extract(py)?;

                let events: Vec<AiEvent> =
                    serde_json::from_str(&events_json).unwrap_or_default();

                let update = FrameUpdate { camera_id: camera_id.clone(), jpeg, events };

                // try_send — if channel is full, drop this frame.
                // The UI is catching up; we'll get the next frame instead.
                match tx.try_send(update) {
                    Ok(_) => Ok(true),
                    Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => {
                        // Frame dropped — UI is busy, live path unaffected
                        Ok(true)
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                        Ok(false) // UI is gone, break loop
                    }
                }
            });

            match tx_status {
                Ok(true) => continue,
                Ok(false) => break,
                Err(e) => {
                    eprintln!("[ERROR] [stream] Camera '{}' loop error: {}", camera_id, e);
                    break;
                }
            }
        }

        // Cleanup
        let _ = Python::with_gil(|py| {
            stream.call_method0(py, "release")
        });

        println!("[INFO]  [stream] Camera '{}' exited cleanly.", camera_id);
    });
}

// ================================================================
// Decode JPEG → Slint SharedPixelBuffer.
// This is CPU intensive (~20-30ms) so it MUST run on the Tokio
// background thread, not the UI thread.
// slint::SharedPixelBuffer is Send, so we can pass it across.
// ================================================================

fn jpeg_to_pixel_buffer(jpeg: &[u8]) -> Option<slint::SharedPixelBuffer<slint::Rgba8Pixel>> {
    let img  = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg).ok()?;
    let rgba = img.into_rgba8();
    let w = rgba.width();
    let h = rgba.height();
    let mut buf = slint::SharedPixelBuffer::<slint::Rgba8Pixel>::new(w, h);
    buf.make_mut_bytes().copy_from_slice(rgba.as_raw());
    Some(buf)
}

// ================================================================
// Aggregator — one Tokio task, drains the bounded channel.
//
// Rate-limits UI updates to UI_HZ per camera so the Slint event
// loop never gets flooded with more closures than it can paint.
// Events (AI alerts) bypass the rate limit and are always delivered.
// ================================================================

const UI_HZ: std::time::Duration = std::time::Duration::from_millis(50); // 20 fps cap

pub async fn run_aggregator(
    mut rx: Receiver<FrameUpdate>,
    ui_weak: slint::Weak<AppWindow>,
    selected_camera: Arc<Mutex<String>>,
    shared_alerts: Arc<Mutex<Vec<Notification>>>,
    latest_frames: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    db_conn: Arc<Mutex<rusqlite::Connection>>,
    ws_sender: tokio::sync::broadcast::Sender<Notification>,
) {
    // Per-camera last-UI-update timestamp for rate limiting
    let mut last_ui: HashMap<String, std::time::Instant> = HashMap::new();

    while let Some(update) = rx.recv().await {
        let has_events   = !update.events.is_empty();
        let events_count = update.events.len();

        if has_events {
            println!(
                "[DEBUG] Event frame: camera='{}' events={} jpeg={}B",
                update.camera_id, events_count, update.jpeg.len()
            );
        }

        // ── Push latest JPEG to shared map for Web Server MJPEG stream ────────
        if let Ok(mut map) = latest_frames.lock() {
            map.insert(update.camera_id.clone(), update.jpeg.clone());
        }

        // ── Rate-limit UI frame updates (but never skip events) ───────────────
        let now = std::time::Instant::now();
        let ui_due = last_ui
            .get(&update.camera_id)
            .map(|t| now.duration_since(*t) >= UI_HZ)
            .unwrap_or(true);

        if !ui_due && !has_events {
            // Too soon for a UI repaint and no events — drop this frame
            continue;
        }

        if ui_due {
            last_ui.insert(update.camera_id.clone(), now);
        }

        // ── Decode JPEG to Pixel Buffer (Tokio background thread) ─────────────
        // Doing this here unblocks the UI thread completely.
        let Some(pixel_buf) = jpeg_to_pixel_buffer(&update.jpeg) else {
            continue;
        };

        // ── DB + file I/O on Tokio thread (never on UI thread) ────────────────
        let mut new_notifs: Vec<Notification> = Vec::new();
        let mut camera_name = update.camera_id.clone();
        let mut is_restricted = false;

        if has_events {
            if let Ok(conn) = db_conn.lock() {
                camera_name = database::get_camera_name(&conn, &update.camera_id);
                is_restricted = conn.query_row(
                    "SELECT is_restricted FROM cameras WHERE id = ?1",
                    rusqlite::params![update.camera_id],
                    |row| row.get::<_, i32>(0).map(|v| v != 0)
                ).unwrap_or(false);
            }

            for event in &update.events {
                let kind = if event.event_type.contains("BLACKLIST")
                    || event.event_type.contains("FENCE")
                    || event.event_type.contains("SUSPICIOUS")
                    || event.event_type.contains("UNATTENDED")
                    || event.event_type.contains("INTRUSION")
                    || event.event_type.contains("LOITERING")
                {
                    NotifKind::Alert
                } else if is_restricted {
                    // Restricted mode: alert on UNKNOWN_PERSON or PERSON_DETECTED, but NOT FACE_MATCHED
                    if event.event_type.contains("UNKNOWN_PERSON") || event.event_type.contains("PERSON_DETECTED") {
                        NotifKind::Alert
                    } else {
                        NotifKind::Info
                    }
                } else {
                    // Public mode: alert on any person/face detection
                    if event.event_type.contains("PERSON_DETECTED") || event.event_type.contains("FACE_MATCHED") || event.event_type.contains("UNKNOWN_PERSON") {
                        NotifKind::Alert
                    } else {
                        NotifKind::Info
                    }
                };

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

                // Write snapshot (JPEG bytes → disk, zero re-encode)
                let _ = std::fs::create_dir_all("events");
                if let Err(e) = std::fs::write(&media_path, &update.jpeg) {
                    eprintln!("[ERROR] Snapshot write failed '{}': {}", media_path, e);
                }

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

                // Push to WebSocket clients
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

        // ── UI update: minimal closure, fast exit on failure ──────────────────
        // Move the pre-decoded pixel_buf into the UI closure.
        let ui_weak2         = ui_weak.clone();
        let cam_id           = update.camera_id.clone();
        let selected_camera2 = selected_camera.clone();

        let _ = slint::invoke_from_event_loop(move || {
            let Some(ui) = ui_weak2.upgrade() else { return };

            // Create Slint Image from the already-decoded buffer.
            // This is virtually instantaneous (~1µs) so the UI stays fully responsive.
            let frame_img = slint::Image::from_rgba8(pixel_buf);

            // Notifications (only when events present)
            if has_events {
                let mut notifications: Vec<Notification> =
                    ui.get_notifications().iter().collect();
                for n in new_notifs.into_iter().rev() {
                    notifications.insert(0, n);
                }
                notifications.truncate(50);
                ui.set_notifications(Rc::new(slint::VecModel::from(notifications)).into());
                ui.set_unread_alert_count(ui.get_unread_alert_count() + events_count as i32);
                ui.set_toast_message(
                    format!("{} event(s) on {}", events_count, camera_name).into(),
                );
                ui.set_toast_kind(NotifKind::Alert);
                ui.set_toast_visible(true);
            }

            // Update grid thumbnail for this camera
            let model = ui.get_cameras();
            for i in 0..model.row_count() {
                if let Some(mut cam) = model.row_data(i) {
                    if cam.id == cam_id {
                        cam.live_frame = frame_img.clone();
                        model.set_row_data(i, cam);
                        break;
                    }
                }
            }

            // Update fullscreen view for selected camera
            let selected = selected_camera2.lock().unwrap().clone();
            if selected == cam_id {
                ui.set_live_frame(frame_img);
                ui.set_stream_active(true);
            }
        });
    }
}
