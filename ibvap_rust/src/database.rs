use rusqlite::{params, Connection};
use std::path::PathBuf;

use crate::DiscoveredCamera;



pub fn database_path() -> PathBuf {

    let mut path = std::env::current_dir()
        .expect("Could not get current directory");

    path.push("cameras.db");

    path
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
        "
    )?;

    Ok(conn)
}


// ------------------------------------------------------------
// Get all cameras stored locally
// ------------------------------------------------------------

pub fn get_cameras(
    conn: &Connection
) -> Result<Vec<DiscoveredCamera>, rusqlite::Error> {

    let mut stmt = conn.prepare(
        "
        SELECT id, name, ip, COALESCE(rtsp, '')
        FROM cameras
        ORDER BY created_at
        "
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

    let now = chrono::Local::now()
        .to_rfc3339();

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
