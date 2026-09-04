use axum::{
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

use crate::{database, Notification, NotifKind};

#[derive(Clone)]
pub struct AppState {
    pub alerts: Arc<Mutex<Vec<Notification>>>,
    pub db_pool: Arc<Mutex<rusqlite::Connection>>,
}

#[derive(Deserialize)]
pub struct CreateUserPayload {
    pub username: String,
    pub password_hash: String, // Actually just password, will rename for clarity
    pub role: String,
}

pub async fn run(state: AppState) {
    let app = Router::new()
        .route("/", get(dashboard_html))
        .route("/api/alerts", get(get_alerts))
        .route("/api/users", post(create_user))
        .with_state(state);

    let listener = TcpListener::bind("0.0.0.0:3000").await.unwrap();
    println!("Web server listening on 0.0.0.0:3000");
    axum::serve(listener, app).await.unwrap();
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
        input, select { width: 100%; padding: 10px; margin: 5px 0 15px; border-radius: 4px; border: 1px solid #45475a; background: #1e1e2e; color: #cdd6f4; }
        button { background: #89b4fa; color: #1e1e2e; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        button:hover { background: #b4befe; }
        #alerts-container { max-height: 400px; overflow-y: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>IBVAP Analytics Dashboard</h1>
        
        <div class="card">
            <h2>Live Alerts</h2>
            <div id="alerts-container">Loading...</div>
        </div>

        <div class="card">
            <h2>Add New User (App Access)</h2>
            <form id="add-user-form">
                <label>Username</label>
                <input type="text" id="username" required>
                
                <label>Password</label>
                <input type="password" id="password" required>
                
                <label>Role</label>
                <select id="role">
                    <option value="OPERATOR">Operator</option>
                    <option value="SUPERVISOR">Supervisor</option>
                </select>
                
                <button type="submit">Add User</button>
            </form>
            <p id="user-msg"></p>
        </div>
    </div>

    <script>
        async function fetchAlerts() {
            try {
                const res = await fetch('/api/alerts');
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

        setInterval(fetchAlerts, 2000);
        fetchAlerts();

        document.getElementById('add-user-form').addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('username').value;
            const password_hash = document.getElementById('password').value; // Sending plain, hashed on server
            const role = document.getElementById('role').value;
            
            const msgEl = document.getElementById('user-msg');
            msgEl.textContent = 'Submitting...';
            
            try {
                const res = await fetch('/api/users', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password_hash, role })
                });
                
                if (res.ok) {
                    msgEl.textContent = 'User added successfully!';
                    msgEl.style.color = '#a6e3a1';
                    e.target.reset();
                } else {
                    const err = await res.text();
                    msgEl.textContent = 'Error: ' + err;
                    msgEl.style.color = '#f38ba8';
                }
            } catch (err) {
                msgEl.textContent = 'Error: ' + err.message;
                msgEl.style.color = '#f38ba8';
            }
        });
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

async fn get_alerts(State(state): State<AppState>) -> impl IntoResponse {
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
    Json(response)
}

async fn create_user(
    State(state): State<AppState>,
    Json(payload): Json<CreateUserPayload>,
) -> impl IntoResponse {
    let conn = state.db_pool.lock().unwrap();
    match database::register_user(&conn, &payload.username, &payload.password_hash, &payload.role) {
        Ok(_) => (StatusCode::CREATED, "User registered successfully").into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Database error: {}", e),
        )
            .into_response(),
    }
}
