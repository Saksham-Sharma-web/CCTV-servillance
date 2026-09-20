use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::DiscoveredCamera;

#[derive(Debug, Clone)]
pub struct AuthUser {
    pub username: String,
    pub role: String,
}

pub fn database_path() -> PathBuf {
    let mut path = std::env::current_dir().expect("Could not get current directory");
    path.push("cameras.db");
    path
}

pub fn hash_password(password: &str, salt: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(password.as_bytes());
    hasher.update(salt.as_bytes());
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

pub fn open() -> Result<Connection, rusqlite::Error> {
    let conn = Connection::open(database_path())?;

    // Enable WAL mode for better concurrent read/write performance
    conn.execute_batch("PRAGMA journal_mode=WAL;")?;

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS cameras (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            tag         TEXT NOT NULL,
            ip          TEXT NOT NULL,
            rtsp        TEXT,
            is_online   INTEGER NOT NULL DEFAULT 0,
            last_seen   TEXT,
            has_onvif   INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS events (
            id          TEXT PRIMARY KEY,
            camera_id   TEXT NOT NULL,
            event_type  TEXT NOT NULL,
            confidence  REAL NOT NULL,
            timestamp   TEXT NOT NULL,
            media_path  TEXT NOT NULL,
            synced      INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_cameras_ip
        ON cameras(ip);

        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'ADMIN',
            created_at    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS settings (
            key           TEXT PRIMARY KEY,
            value         TEXT NOT NULL
        );
        ",
    )?;

    // ─── Safe incremental schema migrations ───────────────────────────────────
    // Each ALTER TABLE is wrapped in an ignore — SQLite returns an error if the
    // column already exists, which is fine; we just skip it.

    // onvif_uid: stable hardware identifier that survives IP/DHCP changes
    let _ = conn.execute(
        "ALTER TABLE cameras ADD COLUMN onvif_uid TEXT",
        [],
    );

    // camera_name snapshot on every event row (historical human label)
    let _ = conn.execute(
        "ALTER TABLE events ADD COLUMN camera_name TEXT NOT NULL DEFAULT ''",
        [],
    );

    // camera restricted mode
    let _ = conn.execute(
        "ALTER TABLE cameras ADD COLUMN is_restricted INTEGER NOT NULL DEFAULT 0",
        [],
    );

    // per-camera RTSP credentials
    let _ = conn.execute(
        "ALTER TABLE cameras ADD COLUMN rtsp_user TEXT",
        [],
    );
    let _ = conn.execute(
        "ALTER TABLE cameras ADD COLUMN rtsp_pass TEXT",
        [],
    );

    // Person-centric event correlation columns on events table
    let _ = conn.execute("ALTER TABLE events ADD COLUMN person_id TEXT", []);
    let _ = conn.execute("ALTER TABLE events ADD COLUMN session_id TEXT", []);
    let _ = conn.execute("ALTER TABLE events ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'", []);
    let _ = conn.execute("ALTER TABLE events ADD COLUMN duration_seconds REAL NOT NULL DEFAULT 0.0", []);
    let _ = conn.execute("ALTER TABLE events ADD COLUMN last_seen TEXT NOT NULL DEFAULT ''", []);
    let _ = conn.execute("ALTER TABLE events ADD COLUMN event_metadata TEXT", []);
    let _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_events_person_cam ON events(person_id, camera_id, status)", []);
    let _ = conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera_type ON events(camera_id, event_type, status)", []);

    // Seed default administrative users if database is fresh
    init_default_users(&conn)?;

    Ok(conn)
}

fn init_default_users(conn: &Connection) -> Result<(), rusqlite::Error> {
    let count: i64 = conn.query_row("SELECT COUNT(*) FROM users", [], |r| r.get(0))?;
    if count == 0 {
        let now = chrono::Local::now().to_rfc3339();

        // Default administrator: admin / admin
        let salt_admin = "ibvap-salt-admin-2026";
        let hash_admin = hash_password("admin", salt_admin);
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, created_at)
             VALUES (?1, ?2, ?3, 'SUPERVISOR', ?4)",
            params!["admin", hash_admin, salt_admin, now],
        )?;

        // Default operator: operator / operator
        let salt_op = "ibvap-salt-operator-2026";
        let hash_op = hash_password("operator", salt_op);
        conn.execute(
            "INSERT INTO users (username, password_hash, salt, role, created_at)
             VALUES (?1, ?2, ?3, 'OPERATOR', ?4)",
            params!["operator", hash_op, salt_op, now],
        )?;
    }
    Ok(())
}

pub fn authenticate(
    conn: &Connection,
    username: &str,
    password: &str,
) -> Result<Option<AuthUser>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT username, password_hash, salt, role FROM users WHERE username = ?1",
    )?;
    let mut rows = stmt.query(params![username])?;

    if let Some(row) = rows.next()? {
        let db_user: String = row.get(0)?;
        let db_hash: String = row.get(1)?;
        let salt: String = row.get(2)?;
        let role: String = row.get(3)?;

        let computed = hash_password(password, &salt);
        if computed == db_hash {
            return Ok(Some(AuthUser {
                username: db_user,
                role,
            }));
        }
    }
    Ok(None)
}

pub fn change_password(conn: &Connection, user_id: i64, new_password: &str) -> Result<(), rusqlite::Error> {
    let salt = format!("ibvap-salt-{}", chrono::Local::now().timestamp_nanos_opt().unwrap_or(0));
    let hash = hash_password(new_password, &salt);
    conn.execute(
        "UPDATE users SET password_hash = ?1, salt = ?2 WHERE id = ?3",
        params![hash, salt, user_id],
    )?;
    Ok(())
}

pub fn get_setting(conn: &Connection, key: &str) -> Option<String> {
    conn.query_row(
        "SELECT value FROM settings WHERE key = ?1",
        params![key],
        |row| row.get(0),
    ).ok()
}

pub fn set_setting(conn: &Connection, key: &str, value: &str) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO settings (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

#[allow(dead_code)]
pub fn register_user(
    conn: &Connection,
    username: &str,
    password: &str,
    role: &str,
) -> Result<(), rusqlite::Error> {
    let salt = uuid::Uuid::new_v4().to_string();
    let hash = hash_password(password, &salt);
    let now = chrono::Local::now().to_rfc3339();

    conn.execute(
        "INSERT INTO users (username, password_hash, salt, role, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(username) DO UPDATE SET
         password_hash = excluded.password_hash,
         salt = excluded.salt,
         role = excluded.role",
        params![username, hash, salt, role, now],
    )?;
    Ok(())
}

// ------------------------------------------------------------
// Get all cameras stored locally
// Returns the user-assigned `name` (which survives rediscovery)
// ------------------------------------------------------------

pub fn get_cameras(
    conn: &Connection,
) -> Result<Vec<DiscoveredCamera>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "
        SELECT id, name, ip, COALESCE(rtsp, ''), COALESCE(onvif_uid, ''), is_restricted, COALESCE(rtsp_user, ''), COALESCE(rtsp_pass, '')
        FROM cameras
        ORDER BY created_at
        ",
    )?;

    let rows = stmt.query_map([], |row| {
        Ok(DiscoveredCamera {
            id: row.get(0)?,
            name: row.get(1)?,
            ip: row.get(2)?,
            rtsp: row.get(3)?,
            onvif_uid: row.get(4)?,
            is_restricted: row.get::<_, i32>(5)? != 0,
            rtsp_user: row.get(6)?,
            rtsp_pass: row.get(7)?,
        })
    })?;

    rows.collect()
}

pub fn set_camera_restricted_mode(
    conn: &Connection,
    camera_id: &str,
    is_restricted: bool,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE cameras SET is_restricted = ?1 WHERE id = ?2",
        params![if is_restricted { 1 } else { 0 }, camera_id],
    )?;
    Ok(())
}

// ------------------------------------------------------------
// Upsert a discovered camera.
//
// Key design decisions:
//  1. A stable `id` is derived from onvif_uid when available,
//     otherwise from the IP (DHCP-unstable but better than nothing).
//  2. On conflict (same id), we update hardware metadata ONLY —
//     the operator's human `name` is NEVER touched by discovery.
// ------------------------------------------------------------

pub fn upsert_camera(
    conn: &Connection,
    camera: &DiscoveredCamera,
) -> Result<(), rusqlite::Error> {
    let now = chrono::Local::now().to_rfc3339();

    // Derive the stable primary key
    let stable_id = derive_stable_id(camera);

    conn.execute(
        "
        INSERT INTO cameras
            (id, name, tag, ip, rtsp, is_online,
             last_seen, has_onvif, created_at, updated_at, onvif_uid, rtsp_user, rtsp_pass)

        VALUES
            (?1, ?2, ?2, ?3, ?4, 1,
             ?5, 1, ?5, ?5, ?6, ?7, ?8)

        ON CONFLICT(id) DO UPDATE SET
            -- Hardware/network fields are always refreshed
            ip       = excluded.ip,
            rtsp     = excluded.rtsp,
            is_online = 1,
            last_seen = excluded.last_seen,
            updated_at = excluded.updated_at,
            onvif_uid  = excluded.onvif_uid
            -- NOTE: `name` and `tag` are intentionally NOT in this list.
            -- The operator's custom label must survive rediscovery.
        ",
        params![
            stable_id,
            camera.name,  // only used on INSERT (first discovery)
            camera.ip,
            camera.rtsp,
            now,
            camera.onvif_uid,
            camera.rtsp_user,
            camera.rtsp_pass
        ],
    )?;

    Ok(())
}

/// Build a stable camera id that does NOT change when the IP changes.
/// Priority:  onvif_uid  >  ip-based fallback
pub fn derive_stable_id(camera: &DiscoveredCamera) -> String {
    if !camera.onvif_uid.is_empty() {
        // Strip urn:uuid: prefix if present; keep the UUID portion only
        let uid = camera.onvif_uid.trim_start_matches("urn:uuid:");
        format!("onvif-{}", uid)
    } else if !camera.id.is_empty() && !camera.id.starts_with("192.") {
        // The Python side already computed a reasonable id
        camera.id.clone()
    } else {
        // Fallback: use IP (unstable under DHCP but better than random UUIDs)
        format!("ip-{}", camera.ip.replace('.', "-"))
    }
}

// ------------------------------------------------------------
// Delete exactly ONE camera
// ------------------------------------------------------------

pub fn delete_camera(
    conn: &Connection,
    id: &str,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "DELETE FROM cameras WHERE id = ?1",
        params![id],
    )?;

    Ok(())
}

// ------------------------------------------------------------
// Rename ONE camera — updates the human-visible `name` AND `tag`
// ------------------------------------------------------------

pub fn rename_camera(
    conn: &Connection,
    id: &str,
    new_name: &str,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "
        UPDATE cameras
        SET name = ?1,
            tag  = ?1,
            updated_at = ?2
        WHERE id = ?3
        ",
        params![
            new_name,
            chrono::Local::now().to_rfc3339(),
            id
        ],
    )?;

    Ok(())
}

// ------------------------------------------------------------
// Look up a camera's current human name by its id
// ------------------------------------------------------------
pub fn get_camera_name(conn: &Connection, camera_id: &str) -> String {
    conn.query_row(
        "SELECT name FROM cameras WHERE id = ?1",
        params![camera_id],
        |row| row.get::<_, String>(0),
    )
    .unwrap_or_else(|_| camera_id.to_string())
}

// ------------------------------------------------------------
// Events Management
// ------------------------------------------------------------

pub fn insert_event_full(
    conn: &Connection,
    id: &str,
    camera_id: &str,
    camera_name: &str,
    event_type: &str,
    confidence: f64,
    timestamp: &str,
    media_path: &str,
    person_id: Option<&str>,
    session_id: Option<&str>,
    status: &str,
    duration_seconds: f64,
    metadata_json: Option<&str>,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "INSERT INTO events (id, camera_id, camera_name, event_type, confidence, timestamp, media_path, person_id, session_id, status, duration_seconds, last_seen, event_metadata)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?6, ?12)",
        params![
            id, camera_id, camera_name, event_type, confidence, timestamp, media_path,
            person_id, session_id, status, duration_seconds, metadata_json
        ],
    )?;
    Ok(())
}

#[allow(dead_code)]
pub fn insert_event(
    conn: &Connection,
    id: &str,
    camera_id: &str,
    camera_name: &str,
    event_type: &str,
    confidence: f64,
    timestamp: &str,
    media_path: &str,
) -> Result<(), rusqlite::Error> {
    insert_event_full(
        conn, id, camera_id, camera_name, event_type, confidence, timestamp, media_path,
        None, None, "ACTIVE", 0.0, None
    )
}

pub fn update_event(
    conn: &Connection,
    id: &str,
    confidence: f64,
    timestamp: &str,
    duration_seconds: f64,
    status: &str,
    metadata_json: &str,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE events SET
            confidence = MAX(confidence, ?2),
            last_seen = ?3,
            duration_seconds = ?4,
            status = ?5,
            event_metadata = ?6
         WHERE id = ?1",
        params![id, confidence, timestamp, duration_seconds, status, metadata_json],
    )?;
    Ok(())
}

pub fn get_events(conn: &Connection, limit: i64) -> Result<Vec<EventRecord>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, camera_id, COALESCE(camera_name,''), event_type, confidence, timestamp, media_path,
                person_id, session_id, COALESCE(status, 'ACTIVE'), COALESCE(duration_seconds, 0.0), COALESCE(last_seen, timestamp)
         FROM events
         ORDER BY timestamp DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit], |row| {
        Ok(EventRecord {
            id: row.get(0)?,
            camera_id: row.get(1)?,
            camera_name: row.get(2)?,
            event_type: row.get(3)?,
            confidence: row.get(4)?,
            timestamp: row.get(5)?,
            media_path: row.get(6)?,
            person_id: row.get(7)?,
            session_id: row.get(8)?,
            status: row.get(9)?,
            duration_seconds: row.get(10)?,
            last_seen: row.get(11)?,
        })
    })?;
    rows.collect()
}

pub fn get_event_by_id(conn: &Connection, event_id: &str) -> Option<EventRecord> {
    conn.query_row(
        "SELECT id, camera_id, COALESCE(camera_name,''), event_type, confidence, timestamp, media_path,
                person_id, session_id, COALESCE(status, 'ACTIVE'), COALESCE(duration_seconds, 0.0), COALESCE(last_seen, timestamp)
         FROM events WHERE id = ?1",
        params![event_id],
        |row| {
            Ok(EventRecord {
                id: row.get(0)?,
                camera_id: row.get(1)?,
                camera_name: row.get(2)?,
                event_type: row.get(3)?,
                confidence: row.get(4)?,
                timestamp: row.get(5)?,
                media_path: row.get(6)?,
                person_id: row.get(7)?,
                session_id: row.get(8)?,
                status: row.get(9)?,
                duration_seconds: row.get(10)?,
                last_seen: row.get(11)?,
            })
        },
    )
    .ok()
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct EventRecord {
    pub id: String,
    pub camera_id: String,
    pub camera_name: String,
    pub event_type: String,
    pub confidence: f64,
    pub timestamp: String,
    pub media_path: String,
    pub person_id: Option<String>,
    pub session_id: Option<String>,
    pub status: String,
    pub duration_seconds: f64,
    pub last_seen: String,
}

pub fn mark_events_synced(conn: &Connection) -> Result<(), rusqlite::Error> {
    conn.execute("UPDATE events SET synced = 1", [])?;
    Ok(())
}

pub fn cleanup_old_events(conn: &Connection) -> Result<(), rusqlite::Error> {
    // Find media paths of events to delete (beyond the most recent 500)
    let mut stmt = conn.prepare("SELECT media_path FROM events ORDER BY timestamp DESC LIMIT -1 OFFSET 500")?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    
    // Delete local snapshot files
    for path_result in rows {
        if let Ok(path) = path_result {
            let _ = std::fs::remove_file(path);
        }
    }

    // Delete records from database
    conn.execute(
        "DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY timestamp DESC LIMIT 500)",
        [],
    )?;

    Ok(())
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn in_memory_db() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "
            CREATE TABLE IF NOT EXISTS cameras (
                id          TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                tag         TEXT NOT NULL,
                ip          TEXT NOT NULL,
                rtsp        TEXT,
                is_online   INTEGER NOT NULL DEFAULT 0,
                last_seen   TEXT,
                has_onvif   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                onvif_uid   TEXT,
                is_restricted INTEGER NOT NULL DEFAULT 0,
                rtsp_user   TEXT,
                rtsp_pass   TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id          TEXT PRIMARY KEY,
                camera_id   TEXT NOT NULL,
                camera_name TEXT NOT NULL DEFAULT '',
                event_type  TEXT NOT NULL,
                confidence  REAL NOT NULL,
                timestamp   TEXT NOT NULL,
                media_path  TEXT NOT NULL,
                synced      INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt          TEXT NOT NULL,
                role          TEXT NOT NULL DEFAULT 'ADMIN',
                created_at    TEXT NOT NULL
            );
            ",
        ).unwrap();
        init_default_users(&conn).unwrap();
        conn
    }

    #[test]
    fn test_default_users_authentication() {
        let conn = in_memory_db();

        let admin_auth = authenticate(&conn, "admin", "admin").unwrap();
        assert!(admin_auth.is_some());
        let admin = admin_auth.unwrap();
        assert_eq!(admin.username, "admin");
        assert_eq!(admin.role, "SUPERVISOR");

        let op_auth = authenticate(&conn, "operator", "operator").unwrap();
        assert!(op_auth.is_some());
        let op = op_auth.unwrap();
        assert_eq!(op.username, "operator");
        assert_eq!(op.role, "OPERATOR");

        let bad_pass = authenticate(&conn, "admin", "wrongpassword").unwrap();
        assert!(bad_pass.is_none());

        let no_user = authenticate(&conn, "unknown", "admin").unwrap();
        assert!(no_user.is_none());
    }

    #[test]
    fn test_custom_user_registration() {
        let conn = in_memory_db();
        register_user(&conn, "analyst", "securepass123", "ANALYST").unwrap();

        let auth = authenticate(&conn, "analyst", "securepass123").unwrap();
        assert!(auth.is_some());
        let user = auth.unwrap();
        assert_eq!(user.username, "analyst");
        assert_eq!(user.role, "ANALYST");
    }

    #[test]
    fn test_camera_crud() {
        let conn = in_memory_db();

        let cam = DiscoveredCamera {
            id: "cam-01".into(),
            name: "Front Gate".into(),
            ip: "192.168.0.105".into(),
            rtsp: "rtsp://cam:12345678@192.168.0.105:8554/live".into(),
            onvif_uid: "uuid-abc123".into(),
            is_restricted: false,
            rtsp_user: None,
            rtsp_pass: None,
        };

        upsert_camera(&conn, &cam).unwrap();

        let list = get_cameras(&conn).unwrap();
        assert_eq!(list.len(), 1);
        // id should be derived from onvif_uid
        assert_eq!(list[0].id, "onvif-uuid-abc123");
        assert_eq!(list[0].name, "Front Gate");

        // Rename
        rename_camera(&conn, "onvif-uuid-abc123", "North Gate").unwrap();
        let renamed = get_cameras(&conn).unwrap();
        assert_eq!(renamed[0].name, "North Gate");

        // Rediscovery must NOT overwrite user-assigned name
        let same_cam_new_ip = DiscoveredCamera {
            id: "cam-01".into(),
            name: "Camera 192.168.0.200".into(), // discovery would give this default
            ip: "192.168.0.200".into(),
            rtsp: "rtsp://cam:12345678@192.168.0.200:8554/live".into(),
            onvif_uid: "uuid-abc123".into(),
            is_restricted: false,
            rtsp_user: None,
            rtsp_pass: None,
        };
        upsert_camera(&conn, &same_cam_new_ip).unwrap();
        let after_rediscovery = get_cameras(&conn).unwrap();
        assert_eq!(after_rediscovery[0].name, "North Gate", "name must survive rediscovery");
        assert_eq!(after_rediscovery[0].ip, "192.168.0.200", "ip must be updated");

        // Delete
        delete_camera(&conn, "onvif-uuid-abc123").unwrap();
        let empty = get_cameras(&conn).unwrap();
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_event_with_camera_name() {
        let conn = in_memory_db();
        insert_event(
            &conn,
            "evt-001",
            "onvif-uuid-abc123",
            "North Gate",
            "FENCE_INTRUSION",
            0.91,
            "19:42:15",
            "events/evt-001.jpg",
        ).unwrap();

        let events = get_events(&conn, 10).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].camera_name, "North Gate");
        assert_eq!(events[0].event_type, "FENCE_INTRUSION");

        let ev = get_event_by_id(&conn, "evt-001").unwrap();
        assert_eq!(ev.camera_name, "North Gate");
    }

    #[test]
    fn test_extract_unknown_id() {
        assert_eq!(
            super::extract_unknown_id("PERSON_REIDENTIFIED (UNK-P-6C207648)"),
            Some("UNK-P-6C207648".to_string())
        );
        assert_eq!(
            super::extract_unknown_id("UNK-P-1234ABCD: Camera transition"),
            Some("UNK-P-1234ABCD".to_string())
        );
        assert_eq!(
            super::extract_unknown_id("Just a regular person event"),
            None
        );
    }
}

pub fn update_camera_credentials(
    conn: &Connection,
    camera_id: &str,
    user: &str,
    pass: &str,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "UPDATE cameras SET rtsp_user = ?1, rtsp_pass = ?2 WHERE id = ?3",
        params![user, pass, camera_id],
    )?;
    Ok(())
}

// ============================================================
// Person Journey / Trajectory Querying
// ============================================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrajectorySighting {
    pub sighting_id: String,
    pub unknown_id: String,
    pub camera_id: String,
    pub camera_name: String,
    pub track_id: i64,
    pub timestamp_iso: String,
    pub similarity: f64,
    pub face_quality: f64,
    pub snapshot_path: String,
    pub time_gap_str: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct PersonTrajectory {
    pub unknown_id: String,
    pub first_camera_id: String,
    pub last_camera_id: String,
    pub camera_sequence: Vec<String>,
    pub created_at_iso: String,
    pub updated_at_iso: String,
    pub sightings: Vec<TrajectorySighting>,
}

pub fn extract_unknown_id(text: &str) -> Option<String> {
    if let Some(pos) = text.find("UNK-P-") {
        let remainder = &text[pos..];
        let end = remainder
            .find(|c: char| !c.is_ascii_alphanumeric() && c != '-')
            .unwrap_or(remainder.len());
        let id = &remainder[..end];
        if id.len() >= 8 {
            return Some(id.to_string());
        }
    }
    // Also match P001, P002, etc. (P followed by digits)
    for word in text.split(|c: char| !c.is_ascii_alphanumeric()) {
        if word.starts_with('P') && word.len() >= 4 && word[1..].chars().all(|c| c.is_ascii_digit()) {
            return Some(word.to_string());
        }
    }
    None
}

fn open_unknown_persons_db() -> Option<Connection> {
    let candidate_paths = [
        std::path::PathBuf::from(r"C:\CCTV-servillance\data\unknown_persons.db"),
        std::path::PathBuf::from("../data/unknown_persons.db"),
        std::path::PathBuf::from("data/unknown_persons.db"),
    ];

    for p in &candidate_paths {
        if p.exists() {
            if let Ok(conn) = Connection::open(p) {
                return Some(conn);
            }
        }
    }
    None
}

pub fn get_person_trajectory(input_id: &str) -> Option<PersonTrajectory> {
    let unk_conn = open_unknown_persons_db()?;
    let trimmed = input_id.trim();

    let target_id = if trimmed.is_empty() || trimmed == "latest" {
        unk_conn.query_row(
            "SELECT unknown_id FROM unknown_persons ORDER BY last_seen_timestamp DESC LIMIT 1",
            [],
            |r| r.get::<_, String>(0),
        ).ok()?
    } else if let Some(extracted) = extract_unknown_id(trimmed) {
        extracted
    } else if trimmed.starts_with("evt_") {
        // Look up the event in cameras.db
        let mut resolved = None;
        let mut event_info = None;
        if let Ok(c_conn) = Connection::open("cameras.db") {
            if let Ok((ev_type, cam_id, cam_name, ts, media_path, db_pid)) = c_conn.query_row(
                "SELECT event_type, camera_id, camera_name, timestamp, media_path, person_id FROM events WHERE id = ?1",
                params![trimmed],
                |r| Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, Option<String>>(5)?,
                )),
            ) {
                resolved = db_pid.filter(|s| !s.is_empty()).or_else(|| extract_unknown_id(&ev_type));
                event_info = Some((cam_id, cam_name, ts, media_path));
            }
        }

        if let Some(uid) = resolved {
            uid
        } else if let Some((cam_id, cam_name, ts, media_path)) = event_info {
            let display_name = if cam_name.is_empty() { cam_id.clone() } else { cam_name.clone() };
            return Some(PersonTrajectory {
                unknown_id: format!("Sighting ({})", display_name),
                first_camera_id: cam_id.clone(),
                last_camera_id: cam_id.clone(),
                camera_sequence: vec![display_name.clone()],
                created_at_iso: ts.clone(),
                updated_at_iso: ts.clone(),
                sightings: vec![TrajectorySighting {
                    sighting_id: trimmed.to_string(),
                    unknown_id: trimmed.to_string(),
                    camera_id: cam_id.clone(),
                    camera_name: display_name,
                    track_id: 1,
                    timestamp_iso: ts,
                    similarity: 1.0,
                    face_quality: 1.0,
                    snapshot_path: media_path,
                    time_gap_str: "Event Sighting".to_string(),
                }],
            });
        } else {
            return None;
        }
    } else {
        trimmed.to_string()
    };

    let mut stmt = unk_conn.prepare(
        "SELECT unknown_id, first_camera_id, last_camera_id, camera_sequence, created_at_iso, updated_at_iso
         FROM unknown_persons WHERE unknown_id = ?1"
    ).ok()?;

    let (uid, first_cam, last_cam, cam_seq_json, created_at, updated_at) = stmt.query_row(
        params![target_id],
        |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
            ))
        }
    ).ok()?;

    let camera_sequence: Vec<String> = serde_json::from_str(&cam_seq_json).unwrap_or_default();

    let mut sight_stmt = unk_conn.prepare(
        "SELECT sighting_id, unknown_id, camera_id, track_id, timestamp, timestamp_iso, similarity, face_quality, COALESCE(snapshot_path, '')
         FROM unknown_person_sightings WHERE unknown_id = ?1 ORDER BY timestamp ASC"
    ).ok()?;

    let mut sightings = Vec::new();
    let mut prev_ts: Option<f64> = None;

    let rows = sight_stmt.query_map(params![target_id], |r| {
        let sighting_id: String = r.get(0)?;
        let unknown_id: String = r.get(1)?;
        let camera_id: String = r.get(2)?;
        let track_id: i64 = r.get(3)?;
        let ts: f64 = r.get(4)?;
        let timestamp_iso: String = r.get(5)?;
        let similarity: f64 = r.get(6)?;
        let face_quality: f64 = r.get(7)?;
        let mut snapshot_path: String = r.get(8)?;

        if snapshot_path.is_empty() {
            let candidate = format!("events/{}.jpg", sighting_id);
            if std::path::Path::new(&candidate).exists() {
                snapshot_path = candidate;
            }
        }

        Ok((sighting_id, unknown_id, camera_id, track_id, ts, timestamp_iso, similarity, face_quality, snapshot_path))
    }).ok()?;

    let cam_names = {
        let mut map = HashMap::new();
        if let Ok(c_conn) = Connection::open("cameras.db") {
            if let Ok(mut c_stmt) = c_conn.prepare("SELECT id, name FROM cameras") {
                if let Ok(c_rows) = c_stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))) {
                    for cr in c_rows.flatten() {
                        map.insert(cr.0, cr.1);
                    }
                }
            }
        }
        map
    };

    for row in rows.flatten() {
        let (sighting_id, unknown_id, camera_id, track_id, ts, timestamp_iso, similarity, face_quality, mut snapshot_path) = row;
        
        let time_gap_str = match prev_ts {
            None => "Initial Sighting".to_string(),
            Some(p_ts) => {
                let diff = (ts - p_ts).max(0.0);
                if diff < 60.0 {
                    format!("+{:.*}s", 1, diff)
                } else {
                    let mins = (diff / 60.0).floor() as u64;
                    let secs = (diff % 60.0) as u64;
                    format!("+{}m {}s", mins, secs)
                }
            }
        };

        prev_ts = Some(ts);

        // If snapshot_path is still empty, look for any event snapshot for this camera
        if snapshot_path.is_empty() {
            if let Ok(c_conn) = Connection::open("cameras.db") {
                let candidate: Option<String> = c_conn.query_row(
                    "SELECT media_path FROM events WHERE camera_id = ?1 ORDER BY rowid DESC LIMIT 1",
                    params![&camera_id],
                    |r| r.get(0),
                ).ok();
                if let Some(p) = candidate {
                    if std::path::Path::new(&p).exists() {
                        snapshot_path = p;
                    }
                }
            }
        }

        let camera_name = cam_names.get(&camera_id).cloned().unwrap_or_else(|| camera_id.clone());

        sightings.push(TrajectorySighting {
            sighting_id,
            unknown_id,
            camera_id,
            camera_name,
            track_id,
            timestamp_iso,
            similarity,
            face_quality,
            snapshot_path,
            time_gap_str,
        });
    }

    Some(PersonTrajectory {
        unknown_id: uid,
        first_camera_id: first_cam,
        last_camera_id: last_cam,
        camera_sequence,
        created_at_iso: created_at,
        updated_at_iso: updated_at,
        sightings,
    })
}

