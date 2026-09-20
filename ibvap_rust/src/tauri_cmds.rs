use tauri::State;
use std::sync::{Arc, Mutex};
use rusqlite::Connection;
use serde::Serialize;
use crate::database;
use crate::python_connector;

// ─── Tauri-facing Camera DTO ──────────────────────────────────────────────────
#[derive(Serialize, Clone)]
pub struct CameraDto {
    pub id: String,
    pub name: String,
    pub ip: String,
    pub rtsp: String,
    pub onvif_uid: String,
    pub is_restricted: bool,
}

// ─── Tauri-facing Event DTO ───────────────────────────────────────────────────
#[derive(Serialize, Clone)]
pub struct EventDto {
    pub id: String,
    pub camera_id: String,
    pub camera_name: String,
    pub event_type: String,
    pub confidence: f64,
    pub timestamp: String,
    pub media_path: String,
}

// ─── COMMANDS ─────────────────────────────────────────────────────────────────

#[tauri::command]
pub fn login(
    db: State<'_, Arc<Mutex<Connection>>>,
    username: String,
    pass: String,
) -> Result<bool, String> {
    let conn = db.lock().unwrap();
    match database::authenticate(&conn, &username, &pass) {
        Ok(Some(_)) => Ok(true),
        Ok(None)    => Ok(false),
        Err(e)      => Err(e.to_string()),
    }
}

#[tauri::command]
pub fn get_cameras(
    db: State<'_, Arc<Mutex<Connection>>>,
) -> Result<Vec<CameraDto>, String> {
    let conn = db.lock().unwrap();
    let cams = database::get_cameras(&conn).map_err(|e| e.to_string())?;
    Ok(cams.into_iter().map(|c| CameraDto {
        id:            c.id,
        name:          c.name,
        ip:            c.ip,
        rtsp:          c.rtsp,
        onvif_uid:     c.onvif_uid,
        is_restricted: c.is_restricted,
    }).collect())
}

#[tauri::command]
pub fn get_events(
    db: State<'_, Arc<Mutex<Connection>>>,
    limit: Option<i64>,
) -> Result<Vec<EventDto>, String> {
    let conn = db.lock().unwrap();
    let evs = database::get_events(&conn, limit.unwrap_or(100))
        .map_err(|e| e.to_string())?;
    Ok(evs.into_iter().map(|e| EventDto {
        id:          e.id,
        camera_id:   e.camera_id,
        camera_name: e.camera_name,
        event_type:  e.event_type,
        confidence:  e.confidence,
        timestamp:   e.timestamp,
        media_path:  e.media_path,
    }).collect())
}

#[tauri::command]
pub fn reset_password(
    db: State<'_, Arc<Mutex<Connection>>>,
    username: String,
    current_pass: String,
    new_pass: String,
) -> Result<(), String> {
    let conn = db.lock().unwrap();
    // Authenticate first
    match database::authenticate(&conn, &username, &current_pass) {
        Ok(Some(_)) => {},
        Ok(None)    => return Err("Current password is incorrect".to_string()),
        Err(e)      => return Err(e.to_string()),
    }
    // Find user_id
    let user_id: i64 = conn.query_row(
        "SELECT id FROM users WHERE username = ?1",
        rusqlite::params![username],
        |row| row.get(0),
    ).map_err(|e| e.to_string())?;
    database::change_password(&conn, user_id, &new_pass).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn factory_reset(
    db: State<'_, Arc<Mutex<Connection>>>,
) -> Result<(), String> {
    let conn = db.lock().unwrap();
    conn.execute("DELETE FROM cameras", []).map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM events", []).map_err(|e| e.to_string())?;
    conn.execute("DELETE FROM settings", []).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn add_camera(
    db: State<'_, Arc<Mutex<Connection>>>,
    rtsp: String,
    name: String,
    user: String,
    pass: String,
) -> Result<String, String> {
    let conn = db.lock().unwrap();
    let id = format!("manual-{}", uuid::Uuid::new_v4());
    let cam = crate::DiscoveredCamera {
        id:            id.clone(),
        name:          if name.is_empty() { "Camera".to_string() } else { name },
        ip:            String::new(),
        rtsp:          rtsp,
        onvif_uid:     String::new(),
        is_restricted: false,
        rtsp_user:     if user.is_empty() { None } else { Some(user.clone()) },
        rtsp_pass:     if pass.is_empty() { None } else { Some(pass.clone()) },
    };
    database::upsert_camera(&conn, &cam).map_err(|e| e.to_string())?;
    // Persist credentials
    if !user.is_empty() {
        conn.execute(
            "UPDATE cameras SET rtsp_user = ?1, rtsp_pass = ?2 WHERE id = ?3",
            rusqlite::params![user, pass, id],
        ).map_err(|e| e.to_string())?;
    }
    Ok(id)
}

#[tauri::command]
pub fn remove_camera(
    db: State<'_, Arc<Mutex<Connection>>>,
    id: String,
) -> Result<(), String> {
    let conn = db.lock().unwrap();
    database::delete_camera(&conn, &id).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn rename_camera(
    db: State<'_, Arc<Mutex<Connection>>>,
    id: String,
    new_name: String,
) -> Result<(), String> {
    let conn = db.lock().unwrap();
    database::rename_camera(&conn, &id, &new_name).map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_onvif_settings(
    db: State<'_, Arc<Mutex<Connection>>>,
) -> Result<(Option<String>, Option<String>), String> {
    let conn = db.lock().unwrap();
    let username = database::get_setting(&conn, "onvif_username");
    let password = database::get_setting(&conn, "onvif_password");
    Ok((username, password))
}

#[tauri::command]
pub fn save_onvif_settings(
    db: State<'_, Arc<Mutex<Connection>>>,
    username: String,
    password: String,
) -> Result<(), String> {
    let conn = db.lock().unwrap();
    let _ = database::set_setting(&conn, "onvif_username", &username);
    let _ = database::set_setting(&conn, "onvif_password", &password);
    Ok(())
}

#[tauri::command]
pub fn select_ai_reference() -> Result<String, String> {
    if let Some(path) = rfd::FileDialog::new()
        .add_filter("Images", &["png", "jpg", "jpeg", "webp", "bmp"])
        .pick_file()
    {
        Ok(path.to_string_lossy().to_string())
    } else {
        Err("No file selected".to_string())
    }
}

#[tauri::command]
pub fn register_ai_reference(
    db: State<'_, Arc<Mutex<Connection>>>,
    tag: String,
    path: String,
) -> Result<(), String> {
    let conn = db.lock().unwrap();
    let _ = database::set_setting(&conn, "ai_ref_tag", &tag);
    let _ = database::set_setting(&conn, "ai_ref_path", &path);
    python_connector::register_reference_face(&path, &tag).map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn get_stream_token() -> Result<String, String> {
    // Generate a short-lived JWT for the MJPEG stream endpoint
    use jsonwebtoken::{encode, EncodingKey, Header, Algorithm};
    use serde::Serialize;

    #[derive(Serialize)]
    struct Claims {
        sub: String,
        exp: usize,
        type_: String,
    }

    const SECRET: &[u8] = b"v8!x@9Pq2L#mZ5$k*RyT^7&wF4(cD1%h";

    let exp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as usize + 3600; // 1 hour

    let claims = Claims {
        sub: "desktop-client".to_string(),
        exp,
        type_: "stream".to_string(),
    };

    encode(
        &Header::new(Algorithm::HS256),
        &claims,
        &EncodingKey::from_secret(SECRET),
    ).map_err(|e| e.to_string())
}

#[tauri::command]
pub async fn discover_cameras(
    db: State<'_, Arc<Mutex<Connection>>>,
) -> Result<String, String> {
    // 1) Read credentials
    let (user, pass) = {
        let conn = db.lock().unwrap();
        let u = database::get_setting(&conn, "onvif_username").unwrap_or_else(|| "admin".to_string());
        let p = database::get_setting(&conn, "onvif_password").unwrap_or_else(|| "".to_string());
        (u, p)
    };

    // 2) Run python discovery in a blocking thread so it doesn't freeze the async runtime or UI
    let found = tokio::task::spawn_blocking(move || {
        python_connector::discover_cameras(&user, &pass, 5)
    })
    .await
    .map_err(|e| e.to_string())? // JoinError
    .map_err(|e| e.to_string())?; // Inner string error

    // 3) Upsert to DB
    let mut added = 0;
    {
        let conn = db.lock().unwrap();
        for cam in found {
            if database::upsert_camera(&conn, &cam).is_ok() {
                added += 1;
            }
        }
    }

    Ok(format!("Discovered and updated {} cameras", added))
}
