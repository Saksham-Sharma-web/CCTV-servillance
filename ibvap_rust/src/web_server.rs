use axum::{
    extract::{State, Path, Query},
    http::{header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE}, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use jsonwebtoken::{decode, DecodingKey, Validation, Algorithm};
use axum_server::tls_rustls::RustlsConfig;

use crate::{database, Notification};

#[derive(Clone)]
pub struct AppState {
    #[allow(dead_code)] // Retained for future real-time push without DB query
    pub alerts: Arc<Mutex<Vec<Notification>>>,
    pub db_pool: Arc<Mutex<rusqlite::Connection>>,
    pub latest_frames: Arc<Mutex<HashMap<String, Vec<u8>>>>,
}

pub async fn run(state: AppState) {
    let app = Router::new()
        .route("/", get(dashboard_html))
        .route("/api/stream/:camera_id", get(stream_camera))
        .route("/api/cameras", get(get_cameras))
        .route("/api/events", get(get_events))
        .route("/api/events/{id}", get(get_event_by_id))
        .route("/api/snapshots/{event_id}", get(get_snapshot))
        .with_state(state);

    let subject_alt_names = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "0.0.0.0".to_string(),
    ];
    let cert = rcgen::generate_simple_self_signed(subject_alt_names).unwrap();
    let tls_config = RustlsConfig::from_der(
        vec![cert.cert.der().to_vec()],
        cert.signing_key.serialize_der(),
    )
    .await
    .unwrap();

    println!("Web server listening on https://0.0.0.0:3000");
    axum_server::bind_rustls(
        "0.0.0.0:3000".parse::<std::net::SocketAddr>().unwrap(),
        tls_config,
    )
    .serve(app.into_make_service())
    .await
    .unwrap();
}

// ──────────────────────────────────────────────────────────────────────────
// Auth helper — Basic authentication via SQLite users table
// ──────────────────────────────────────────────────────────────────────────

fn verify_basic_auth(headers: &HeaderMap, db_pool: &Arc<Mutex<rusqlite::Connection>>) -> bool {
    let Some(auth_val) = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()) else {
        return false;
    };
    if !auth_val.starts_with("Basic ") {
        return false;
    }
    use base64::{engine::general_purpose, Engine as _};
    let Ok(decoded) = general_purpose::STANDARD.decode(&auth_val[6..]) else {
        return false;
    };
    let Ok(creds) = String::from_utf8(decoded) else {
        return false;
    };
    let parts: Vec<&str> = creds.splitn(2, ':').collect();
    if parts.len() != 2 {
        return false;
    }
    let Ok(conn) = db_pool.lock() else {
        return false;
    };
    matches!(database::authenticate(&conn, parts[0], parts[1]), Ok(Some(_)))
}

// ──────────────────────────────────────────────────────────────────────────
// GET /api/cameras
// ──────────────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct CameraResponse {
    id: String,
    name: String,
    ip: String,
    rtsp: String,
    onvif_uid: String,
}

async fn get_cameras(headers: HeaderMap, State(state): State<AppState>) -> Response {
    if !verify_basic_auth(&headers, &state.db_pool) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }
    let Ok(conn) = state.db_pool.lock() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "DB lock error").into_response();
    };
    let cams = database::get_cameras(&conn).unwrap_or_default();
    let resp: Vec<CameraResponse> = cams
        .iter()
        .map(|c| CameraResponse {
            id: c.id.clone(),
            name: c.name.clone(),
            ip: c.ip.clone(),
            rtsp: c.rtsp.clone(),
            onvif_uid: c.onvif_uid.clone(),
        })
        .collect();
    Json(resp).into_response()
}

// ──────────────────────────────────────────────────────────────────────────
// GET /api/events
// ──────────────────────────────────────────────────────────────────────────

async fn get_events(headers: HeaderMap, State(state): State<AppState>) -> Response {
    if !verify_basic_auth(&headers, &state.db_pool) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }
    let Ok(conn) = state.db_pool.lock() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "DB lock error").into_response();
    };
    let events = database::get_events(&conn, 100).unwrap_or_default();
    Json(events).into_response()
}

// ──────────────────────────────────────────────────────────────────────────
// GET /api/events/:id
// ──────────────────────────────────────────────────────────────────────────

async fn get_event_by_id(
    Path(id): Path<String>,
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Response {
    if !verify_basic_auth(&headers, &state.db_pool) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }
    let Ok(conn) = state.db_pool.lock() else {
        return (StatusCode::INTERNAL_SERVER_ERROR, "DB lock error").into_response();
    };
    match database::get_event_by_id(&conn, &id) {
        Some(ev) => Json(ev).into_response(),
        None => (StatusCode::NOT_FOUND, "Event not found").into_response(),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// GET /api/snapshots/:event_id
// Validates the event exists, then streams the JPEG bytes from disk.
// ──────────────────────────────────────────────────────────────────────────

async fn get_snapshot(
    Path(event_id): Path<String>,
    headers: HeaderMap,
    State(state): State<AppState>,
) -> Response {
    if !verify_basic_auth(&headers, &state.db_pool) {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }

    // Validate the event exists to prevent path traversal
    let media_path = {
        let Ok(conn) = state.db_pool.lock() else {
            return (StatusCode::INTERNAL_SERVER_ERROR, "DB lock error").into_response();
        };
        match database::get_event_by_id(&conn, &event_id) {
            Some(ev) => ev.media_path,
            None => return (StatusCode::NOT_FOUND, "Event not found").into_response(),
        }
    };

    match tokio::fs::read(&media_path).await {
        Ok(bytes) => (
            StatusCode::OK,
            [(CONTENT_TYPE, "image/jpeg")],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Snapshot file not found").into_response(),
    }
}

// ──────────────────────────────────────────────────────────────────────────
// GET / — Interactive operator dashboard
// ──────────────────────────────────────────────────────────────────────────

async fn dashboard_html() -> Html<&'static str> {
    Html(r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>IBVAP Operator Dashboard</title>
  <style>
    :root{--bg:#1e1e2e;--surface:#313244;--surface2:#45475a;--blue:#89b4fa;--green:#a6e3a1;--red:#f38ba8;--yellow:#f9e2af;--text:#cdd6f4;--subtext:#a6adc8;--muted:#6c7086}
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);min-height:100vh}
    #app{display:none;flex-direction:column;height:100vh}
    header{background:var(--surface);padding:14px 24px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--surface2)}
    header h1{font-size:18px;font-weight:800;color:var(--blue);letter-spacing:1px}
    header span{font-size:12px;color:var(--subtext)}
    .btn{padding:8px 16px;border:none;border-radius:6px;cursor:pointer;font-weight:700;font-size:13px;transition:.15s}
    .btn-primary{background:var(--blue);color:var(--bg)}
    .btn-primary:hover{filter:brightness(1.15)}
    .btn-danger{background:#3d1a1a;color:var(--red);border:1px solid var(--red)}
    main{display:flex;flex:1;overflow:hidden}
    aside{width:300px;background:var(--surface);border-right:1px solid var(--surface2);display:flex;flex-direction:column;overflow:hidden}
    aside h2{font-size:13px;font-weight:700;color:var(--subtext);padding:14px 16px;letter-spacing:.5px;border-bottom:1px solid var(--surface2)}
    #camera-list{flex:1;overflow-y:auto;padding:8px}
    .cam-card{padding:10px 12px;border-radius:8px;margin-bottom:6px;background:var(--surface2);cursor:pointer;transition:.15s;border:2px solid transparent}
    .cam-card:hover,.cam-card.active{border-color:var(--blue);background:#3d4166}
    .cam-name{font-weight:700;font-size:13px}
    .cam-ip{font-size:11px;color:var(--subtext);margin-top:3px}
    section{flex:1;display:flex;flex-direction:column;overflow:hidden}
    #events-panel{flex:1;overflow-y:auto;padding:16px}
    .event-card{background:var(--surface);border-radius:8px;margin-bottom:12px;overflow:hidden;border-left:4px solid var(--red)}
    .event-card.info{border-left-color:var(--blue)}
    .event-header{padding:10px 14px;display:flex;justify-content:space-between;align-items:center}
    .event-type{font-weight:800;font-size:13px;color:var(--red)}
    .event-card.info .event-type{color:var(--blue)}
    .event-meta{font-size:11px;color:var(--subtext)}
    .event-cam{font-size:12px;font-weight:600;color:var(--green)}
    .event-img{width:100%;max-height:240px;object-fit:cover;display:none;cursor:pointer}
    .event-img.loaded{display:block}
    /* Login */
    #login-screen{display:flex;align-items:center;justify-content:center;height:100vh;background:var(--bg)}
    .login-box{background:var(--surface);border-radius:16px;padding:40px;width:420px;border:1px solid var(--surface2)}
    .login-box h1{color:var(--blue);font-size:22px;margin-bottom:4px;text-align:center}
    .login-box p{color:var(--subtext);font-size:12px;text-align:center;margin-bottom:28px}
    label{display:block;font-size:12px;font-weight:600;color:var(--subtext);margin-bottom:4px;letter-spacing:.5px}
    input{width:100%;padding:10px 12px;background:var(--bg);border:1px solid var(--surface2);border-radius:6px;color:var(--text);font-size:14px;margin-bottom:16px}
    input:focus{outline:none;border-color:var(--blue)}
    #login-error{color:var(--red);font-size:12px;margin-bottom:12px;text-align:center;min-height:16px}
    .conf-bar{height:4px;background:var(--surface2);border-radius:2px;margin:6px 14px 10px}
    .conf-fill{height:100%;border-radius:2px;background:var(--red);transition:width .3s}
    ::-webkit-scrollbar{width:6px} ::-webkit-scrollbar-track{background:var(--bg)} ::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:3px}
  </style>
</head>
<body>

<div id="login-screen">
  <div class="login-box">
    <h1>🛡 IBVAP</h1>
    <p>OPERATOR DASHBOARD · RESTRICTED ACCESS</p>
    <label>USERNAME</label>
    <input id="u" type="text" autocomplete="username" placeholder="admin">
    <label>PASSWORD</label>
    <input id="p" type="password" autocomplete="current-password" placeholder="••••••••">
    <div id="login-error"></div>
    <button class="btn btn-primary" style="width:100%" onclick="doLogin()">🔓 Authenticate</button>
  </div>
</div>

<div id="app">
  <header>
    <h1>IBVAP EDGE COMMAND CENTER</h1>
    <span id="header-info">Loading…</span>
    <button class="btn btn-danger" onclick="doLogout()">🔒 Logout</button>
  </header>
  <main>
    <aside>
      <h2>REGISTERED CAMERAS</h2>
      <div id="camera-list"></div>
    </aside>
    <section>
      <div id="events-panel"></div>
    </section>
  </main>
</div>

<script>
  let auth = localStorage.getItem('ibvap_auth') || '';
  let pollInterval = null;

  if (auth) startup();

  async function doLogin() {
    const u = document.getElementById('u').value;
    const p = document.getElementById('p').value;
    auth = 'Basic ' + btoa(u + ':' + p);
    const res = await fetch('/api/events', { headers: { Authorization: auth } });
    if (res.ok) {
      localStorage.setItem('ibvap_auth', auth);
      document.getElementById('login-error').textContent = '';
      startup();
    } else {
      auth = '';
      document.getElementById('login-error').textContent = 'Invalid credentials';
    }
  }

  function doLogout() {
    auth = '';
    localStorage.removeItem('ibvap_auth');
    if (pollInterval) clearInterval(pollInterval);
    document.getElementById('app').style.display = 'none';
    document.getElementById('login-screen').style.display = 'flex';
  }

  function startup() {
    document.getElementById('login-screen').style.display = 'none';
    document.getElementById('app').style.display = 'flex';
    loadCameras();
    loadEvents();
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(loadEvents, 3000);
  }

  let activeCam = null;

  async function loadCameras() {
    const res = await fetch('/api/cameras', { headers: { Authorization: auth } });
    if (!res.ok) return;
    const cams = await res.json();
    const list = document.getElementById('camera-list');
    list.innerHTML = '';
    document.getElementById('header-info').textContent = cams.length + ' camera(s) registered';
    cams.forEach(c => {
      const d = document.createElement('div');
      d.className = 'cam-card' + (c.id === activeCam ? ' active' : '');
      d.innerHTML = `<div class="cam-name">${esc(c.name)}</div><div class="cam-ip">${esc(c.ip)} &bull; ${esc(c.id)}</div>`;
      d.onclick = () => { activeCam = c.id; loadCameras(); loadEvents(); };
      list.appendChild(d);
    });
  }

  async function loadEvents() {
    const res = await fetch('/api/events', { headers: { Authorization: auth } });
    if (!res.ok) { if (res.status === 401) doLogout(); return; }
    const events = await res.json();
    const panel = document.getElementById('events-panel');
    panel.innerHTML = '';

    const filtered = activeCam ? events.filter(e => e.camera_id === activeCam) : events;

    if (filtered.length === 0) {
      panel.innerHTML = '<p style="color:var(--muted);text-align:center;margin-top:60px">No events recorded yet.</p>';
      return;
    }

    filtered.forEach(ev => {
      const isAlert = /FENCE|INTRUSION|BLACKLIST|SUSPICIOUS|UNATTENDED/i.test(ev.event_type);
      const pct = Math.round(ev.confidence * 100);
      const d = document.createElement('div');
      d.className = 'event-card' + (isAlert ? '' : ' info');
      d.innerHTML = `
        <div class="event-header">
          <div>
            <div class="event-type">${esc(ev.event_type.replace(/_/g,' '))}</div>
            <div class="event-cam">${esc(ev.camera_name || ev.camera_id)}</div>
          </div>
          <div class="event-meta">${esc(ev.timestamp)}</div>
        </div>
        <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
        <img class="event-img" data-src="/api/snapshots/${esc(ev.id)}" alt="snapshot" loading="lazy"
             onerror="this.style.display='none'"
             onclick="window.open('/api/snapshots/${esc(ev.id)}','_blank')">
      `;
      panel.appendChild(d);

      // Lazy-load snapshot image
      const img = d.querySelector('.event-img');
      img.src = img.dataset.src + '?auth=' + encodeURIComponent(auth);
      img.onload = () => img.classList.add('loaded');
    });
  }

  function esc(s) {
    return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }

  document.getElementById('u').addEventListener('keydown', e => e.key==='Enter' && doLogin());
  document.getElementById('p').addEventListener('keydown', e => e.key==='Enter' && doLogin());
</script>
</body>
</html>"#)
}

#[derive(Deserialize)]
pub struct StreamQuery {
    pub token: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct Claims {
    sub: String,
    exp: usize,
    type_: String,
}

const STREAM_SECRET: &[u8] = b"v8!x@9Pq2L#mZ5$k*RyT^7&wF4(cD1%h";

async fn stream_camera(
    Path(camera_id): Path<String>,
    Query(query): Query<StreamQuery>,
    State(state): State<AppState>,
) -> impl IntoResponse {
    let validation = Validation::new(Algorithm::HS256);
    match decode::<Claims>(
        &query.token,
        &DecodingKey::from_secret(STREAM_SECRET),
        &validation,
    ) {
        Ok(_) => {},
        Err(_) => return (StatusCode::FORBIDDEN, "Invalid token").into_response(),
    };

    let stream = async_stream::stream! {
        let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(33));
        loop {
            interval.tick().await;
            
            let jpeg_bytes = {
                if let Ok(map) = state.latest_frames.lock() {
                    map.get(&camera_id).cloned()
                } else {
                    None
                }
            };

            if let Some(bytes) = jpeg_bytes {
                yield Ok::<_, axum::Error>(
                    format!(
                        "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                        bytes.len()
                    )
                    .into_bytes(),
                );
                yield Ok::<_, axum::Error>(bytes);
                yield Ok::<_, axum::Error>("\r\n".into());
            } else {
                yield Ok::<_, axum::Error>(vec![]);
            }
        }
    };

    Response::builder()
        .header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        .body(axum::body::Body::from_stream(stream))
        .unwrap()
}
