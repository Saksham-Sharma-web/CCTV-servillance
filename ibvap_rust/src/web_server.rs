use axum::{
    extract::State,
    http::{header::{HeaderMap, AUTHORIZATION}, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use serde::Serialize;
use std::sync::{Arc, Mutex};
use axum_server::tls_rustls::RustlsConfig;

use crate::{database, Notification, NotifKind};

#[derive(Clone)]
pub struct AppState {
    pub alerts: Arc<Mutex<Vec<Notification>>>,
    pub db_pool: Arc<Mutex<rusqlite::Connection>>,
}

pub async fn run(state: AppState) {
    let app = Router::new()
        .route("/", get(dashboard_html))
        .route("/api/alerts", get(get_alerts))
        .with_state(state);

    let subject_alt_names = vec!["localhost".to_string(), "127.0.0.1".to_string(), "0.0.0.0".to_string()];
    let cert = rcgen::generate_simple_self_signed(subject_alt_names).unwrap();
    let tls_config = RustlsConfig::from_der(
        vec![cert.cert.der().to_vec()],
        cert.signing_key.serialize_der(),
    ).await.unwrap();

    println!("Web server listening on https://0.0.0.0:3000");
    axum_server::bind_rustls("0.0.0.0:3000".parse::<std::net::SocketAddr>().unwrap(), tls_config)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn dashboard_html() -> Html<&'static str> {
    Html(
        r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>IBVAP Analytics Dashboard</title>
    <style>
        body { font-family: 'Inter', sans-serif; background: #1e1e2e; color: #cdd6f4; margin: 0; padding: 20px; }
        .container { max-width: 800px; margin: auto; }
        .card { background: #313244; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        h1, h2 { color: #89b4fa; }
        .alert { background: #582125; color: #f38ba8; padding: 10px; margin: 5px 0; border-radius: 5px; }
        .info { background: #45475a; color: #a6adc8; padding: 10px; margin: 5px 0; border-radius: 5px; }
        input { width: 100%; padding: 10px; margin: 5px 0 15px; border-radius: 4px; border: 1px solid #45475a; background: #1e1e2e; color: #cdd6f4; box-sizing: border-box; }
        button { background: #89b4fa; color: #1e1e2e; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; width: 100%; }
        button:hover { background: #b4befe; }
        #alerts-container { max-height: 400px; overflow-y: auto; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>IBVAP Analytics</h1>
        
        <div class="card" id="login-card">
            <h2>Operator Login</h2>
            <form id="login-form">
                <label>Username</label>
                <input type="text" id="username" required>
                <label>Password</label>
                <input type="password" id="password" required>
                <button type="submit">Login</button>
            </form>
            <p id="login-msg"></p>
        </div>

        <div class="card" id="alerts-card" style="display: none;">
            <h2>Live Alerts</h2>
            <button id="logout-btn" style="margin-bottom: 10px;">Logout</button>
            <div id="alerts-container"></div>
        </div>
    </div>

    <script>
        let authHeader = localStorage.getItem('authHeader') || '';

        if (authHeader) {
            checkAuth();
        }

        document.getElementById('login-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const u = document.getElementById('username').value;
            const p = document.getElementById('password').value;
            authHeader = 'Basic ' + btoa(u + ':' + p);
            checkAuth();
        });

        document.getElementById('logout-btn').addEventListener('click', () => {
            authHeader = '';
            localStorage.removeItem('authHeader');
            location.reload();
        });

        async function checkAuth() {
            try {
                const res = await fetch('/api/alerts', { headers: { 'Authorization': authHeader } });
                if (res.ok) {
                    localStorage.setItem('authHeader', authHeader);
                    document.getElementById('login-card').style.display = 'none';
                    document.getElementById('alerts-card').style.display = 'block';
                    document.getElementById('alerts-container').style.display = 'block';
                    fetchAlerts();
                    setInterval(fetchAlerts, 2000);
                } else {
                    authHeader = '';
                    localStorage.removeItem('authHeader');
                    document.getElementById('login-msg').textContent = 'Invalid credentials';
                    document.getElementById('login-msg').style.color = '#f38ba8';
                    document.getElementById('login-card').style.display = 'block';
                    document.getElementById('alerts-card').style.display = 'none';
                }
            } catch (err) {
                document.getElementById('login-msg').textContent = 'Connection error';
            }
        }

        async function fetchAlerts() {
            if (!authHeader) return;
            try {
                const res = await fetch('/api/alerts', { headers: { 'Authorization': authHeader } });
                if (!res.ok) {
                    if (res.status === 401) {
                        authHeader = '';
                        localStorage.removeItem('authHeader');
                        location.reload();
                    }
                    return;
                }
                const data = await res.json();
                const container = document.getElementById('alerts-container');
                container.innerHTML = '';
                
                if (data.length === 0) {
                    container.innerHTML = '<p>No recent alerts.</p>';
                } else {
                    data.forEach(alert => {
                        const div = document.createElement('div');
                        div.className = alert.kind === 'alert' ? 'alert' : 'info';
                        div.textContent = `[${alert.time}] ${alert.message}`;
                        container.appendChild(div);
                    });
                }
            } catch (err) {
                console.error('Error fetching alerts:', err);
            }
        }
    </script>
</body>
</html>
        "#,
    )
}

#[derive(Serialize)]
struct AlertResponse {
    time: String,
    message: String,
    kind: String,
}

async fn get_alerts(headers: HeaderMap, State(state): State<AppState>) -> Response {
    let auth = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok());
    
    let mut authenticated = false;
    if let Some(auth_val) = auth {
        if auth_val.starts_with("Basic ") {
            let encoded = &auth_val[6..];
            use base64::{Engine as _, engine::general_purpose};
            if let Ok(decoded) = general_purpose::STANDARD.decode(encoded) {
                if let Ok(credentials) = String::from_utf8(decoded) {
                    let parts: Vec<&str> = credentials.splitn(2, ':').collect();
                    if parts.len() == 2 {
                        let conn = state.db_pool.lock().unwrap();
                        if let Ok(Some(_)) = database::authenticate(&conn, parts[0], parts[1]) {
                            authenticated = true;
                        }
                    }
                }
            }
        }
    }

    if !authenticated {
        return (StatusCode::UNAUTHORIZED, "Unauthorized").into_response();
    }

    let alerts = state.alerts.lock().unwrap();
    let response: Vec<AlertResponse> = alerts
        .iter()
        .map(|a| AlertResponse {
            time: a.time.to_string(),
            message: a.message.to_string(),
            kind: match a.kind {
                NotifKind::Alert => "alert".to_string(),
                NotifKind::Info => "info".to_string(),
                NotifKind::Update => "info".to_string(),
            },
        })
        .collect();
    Json(response).into_response()
}
