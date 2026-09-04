use rusqlite::{params, Connection};
use sha2::{Digest, Sha256};
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
        SELECT id, name, ip, COALESCE(rtsp, ''), COALESCE(onvif_uid, '')
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
        })
    })?;

    rows.collect()
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
             last_seen, has_onvif, created_at, updated_at, onvif_uid)

        VALUES
            (?1, ?2, ?2, ?3, ?4, 1,
             ?5, 1, ?5, ?5, ?6)

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
            camera.onvif_uid
        ],
    )?;

    Ok(())
}

/// Build a stable camera id that does NOT change when the IP changes.
/// Priority:  onvif_uid  >  ip-based fallback
fn derive_stable_id(camera: &DiscoveredCamera) -> String {
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
    conn.execute(
        "INSERT INTO events (id, camera_id, camera_name, event_type, confidence, timestamp, media_path)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![id, camera_id, camera_name, event_type, confidence, timestamp, media_path],
    )?;
    Ok(())
}

pub fn get_events(conn: &Connection, limit: i64) -> Result<Vec<EventRecord>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "SELECT id, camera_id, COALESCE(camera_name,''), event_type, confidence, timestamp, media_path
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
        })
    })?;
    rows.collect()
}

pub fn get_event_by_id(conn: &Connection, event_id: &str) -> Option<EventRecord> {
    conn.query_row(
        "SELECT id, camera_id, COALESCE(camera_name,''), event_type, confidence, timestamp, media_path
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
                onvif_uid   TEXT
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
}
