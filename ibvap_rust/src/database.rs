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
        ",
    )?;

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
// ------------------------------------------------------------

pub fn get_cameras(
    conn: &Connection,
) -> Result<Vec<DiscoveredCamera>, rusqlite::Error> {
    let mut stmt = conn.prepare(
        "
        SELECT id, name, ip, COALESCE(rtsp, '')
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
        })
    })?;

    rows.collect()
}

// ------------------------------------------------------------
// Upsert discovered camera
// ------------------------------------------------------------

pub fn upsert_camera(
    conn: &Connection,
    camera: &DiscoveredCamera,
) -> Result<(), rusqlite::Error> {
    let now = chrono::Local::now().to_rfc3339();

    conn.execute(
        "
        INSERT INTO cameras
            (id, name, tag, ip, rtsp, is_online,
             last_seen, has_onvif, created_at, updated_at)

        VALUES
            (?1, ?2, ?2, ?3, ?4, 1,
             ?5, 1, ?5, ?5)

        ON CONFLICT(id) DO UPDATE SET

            name = excluded.name,
            ip = excluded.ip,
            rtsp = excluded.rtsp,
            tag = excluded.tag,
            is_online = 1,
            last_seen = excluded.last_seen,
            updated_at = excluded.updated_at
        ",
        params![
            camera.id,
            camera.name,
            camera.ip,
            camera.rtsp,
            now
        ],
    )?;

    Ok(())
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
// Rename ONE camera
// ------------------------------------------------------------

pub fn rename_camera(
    conn: &Connection,
    id: &str,
    tag: &str,
) -> Result<(), rusqlite::Error> {
    conn.execute(
        "
        UPDATE cameras
        SET tag = ?1,
            updated_at = ?2
        WHERE id = ?3
        ",
        params![
            tag,
            chrono::Local::now().to_rfc3339(),
            id
        ],
    )?;

    Ok(())
}

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
                updated_at  TEXT NOT NULL
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

        // Test admin login
        let admin_auth = authenticate(&conn, "admin", "admin").unwrap();
        assert!(admin_auth.is_some());
        let admin = admin_auth.unwrap();
        assert_eq!(admin.username, "admin");
        assert_eq!(admin.role, "SUPERVISOR");

        // Test operator login
        let op_auth = authenticate(&conn, "operator", "operator").unwrap();
        assert!(op_auth.is_some());
        let op = op_auth.unwrap();
        assert_eq!(op.username, "operator");
        assert_eq!(op.role, "OPERATOR");

        // Test wrong password
        let bad_pass = authenticate(&conn, "admin", "wrongpassword").unwrap();
        assert!(bad_pass.is_none());

        // Test non-existent user
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
        };

        upsert_camera(&conn, &cam).unwrap();

        let list = get_cameras(&conn).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, "cam-01");
        assert_eq!(list[0].name, "Front Gate");

        // Rename
        rename_camera(&conn, "cam-01", "Updated Gate").unwrap();

        // Delete
        delete_camera(&conn, "cam-01").unwrap();
        let empty = get_cameras(&conn).unwrap();
        assert_eq!(empty.len(), 0);
    }
}


