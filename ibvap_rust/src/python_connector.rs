use pyo3::prelude::*;
use pyo3::types::PyModule;
use serde::{Deserialize, Serialize};

use crate::DiscoveredCamera;

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct UpdateResponse {
    pub current_version: String,
    pub latest_version: String,
    pub update_available: bool,
    pub title: String,
    pub details: String,
    pub timestamp: i64,
    pub status: String,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct SyncResponse {
    pub status: String,
    pub node_id: String,
    pub synced_at: i64,
    pub message: String,
    pub echo_count: usize,
}

pub fn discover_cameras(
    username: &str,
    password: &str,
    timeout: u32,
) -> Result<Vec<DiscoveredCamera>, String> {
    Python::with_gil(|py| {
        let sys = py.import("sys").map_err(|e| e.to_string())?;
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;

        sys.getattr("path")
            .map_err(|e| e.to_string())?
            .call_method1("insert", (0, cwd.to_string_lossy().to_string()))
            .map_err(|e| e.to_string())?;

        let stream = PyModule::import(py, "stream")
            .map_err(|e| format!("Failed to import stream.py:\n{}", e))?;

        let asyncio = py.import("asyncio").map_err(|e| e.to_string())?;

        let main_fn = stream.getattr("main").map_err(|e| e.to_string())?;

        let coroutine = main_fn
            .call1((username, password, timeout))
            .map_err(|e| e.to_string())?;

        let result = asyncio
            .call_method1("run", (coroutine,))
            .map_err(|e| format!("Python stream.main() failed:\n{}", e))?;

        let json = py.import("json").map_err(|e| e.to_string())?;

        let json_string: String = json
            .getattr("dumps")
            .map_err(|e| e.to_string())?
            .call1((result,))
            .map_err(|e| e.to_string())?
            .extract()
            .map_err(|e| e.to_string())?;

        serde_json::from_str::<Vec<DiscoveredCamera>>(&json_string).map_err(|e| {
            format!(
                "Python returned invalid camera JSON:\n{}\n\nError: {}",
                json_string, e
            )
        })
    })
}

pub fn resolve_manual_camera(
    ip_or_url: &str,
    username: &str,
    password: &str,
) -> Result<Option<DiscoveredCamera>, String> {
    Python::with_gil(|py| {
        let sys = py.import("sys").map_err(|e| e.to_string())?;
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;

        sys.getattr("path")
            .map_err(|e| e.to_string())?
            .call_method1("insert", (0, cwd.to_string_lossy().to_string()))
            .map_err(|e| e.to_string())?;

        let stream = PyModule::import(py, "stream")
            .map_err(|e| format!("Failed to import stream.py:\n{}", e))?;

        let resolve_fn = stream
            .getattr("resolve_manual_camera")
            .map_err(|e| e.to_string())?;

        let json_string: String = resolve_fn
            .call1((ip_or_url, username, password))
            .map_err(|e| e.to_string())?
            .extract()
            .map_err(|e| e.to_string())?;

        if json_string.trim().is_empty() || json_string.trim() == "{}" {
            return Ok(None);
        }

        let cam = serde_json::from_str::<DiscoveredCamera>(&json_string).map_err(|e| {
            format!(
                "Failed to parse manual camera JSON:\n{}\nError: {}",
                json_string, e
            )
        })?;

        Ok(Some(cam))
    })
}

pub fn check_updates() -> Result<UpdateResponse, String> {
    Python::with_gil(|py| {
        let sys = py.import("sys").map_err(|e| e.to_string())?;
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;

        sys.getattr("path")
            .map_err(|e| e.to_string())?
            .call_method1("insert", (0, cwd.to_string_lossy().to_string()))
            .map_err(|e| e.to_string())?;

        let stream = PyModule::import(py, "stream")
            .map_err(|e| format!("Failed to import stream.py:\n{}", e))?;

        let check_fn = stream
            .getattr("check_updates")
            .map_err(|e| e.to_string())?;

        let json_string: String = check_fn
            .call0()
            .map_err(|e| e.to_string())?
            .extract()
            .map_err(|e| e.to_string())?;

        serde_json::from_str::<UpdateResponse>(&json_string).map_err(|e| {
            format!(
                "Failed to parse update JSON:\n{}\nError: {}",
                json_string, e
            )
        })
    })
}

pub fn sync_cloud(payload_json: &str) -> Result<SyncResponse, String> {
    Python::with_gil(|py| {
        let sys = py.import("sys").map_err(|e| e.to_string())?;
        let cwd = std::env::current_dir().map_err(|e| e.to_string())?;

        sys.getattr("path")
            .map_err(|e| e.to_string())?
            .call_method1("insert", (0, cwd.to_string_lossy().to_string()))
            .map_err(|e| e.to_string())?;

        let stream = PyModule::import(py, "stream")
            .map_err(|e| format!("Failed to import stream.py:\n{}", e))?;

        let sync_fn = stream
            .getattr("sync_cloud")
            .map_err(|e| e.to_string())?;

        let json_string: String = sync_fn
            .call1((payload_json,))
            .map_err(|e| e.to_string())?
            .extract()
            .map_err(|e| e.to_string())?;

        serde_json::from_str::<SyncResponse>(&json_string).map_err(|e| {
            format!(
                "Failed to parse sync JSON:\n{}\nError: {}",
                json_string, e
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_check_updates() {
        pyo3::prepare_freethreaded_python();
        let res = check_updates();
        assert!(res.is_ok(), "check_updates failed: {:?}", res.err());
        let info = res.unwrap();
        assert_eq!(info.latest_version, "1.2.4");
        assert!(info.update_available);
    }

    #[test]
    fn test_python_sync_cloud() {
        pyo3::prepare_freethreaded_python();
        let res = sync_cloud(r#"{"cameras":[{"id":"cam-1"}]}"#);
        assert!(res.is_ok(), "sync_cloud failed: {:?}", res.err());
        let sync = res.unwrap();
        assert_eq!(sync.status, "SUCCESS");
        assert_eq!(sync.echo_count, 1);
    }

    #[test]
    fn test_python_resolve_manual_camera_webcam() {
        pyo3::prepare_freethreaded_python();
        let res = resolve_manual_camera("0", "admin", "admin");
        assert!(res.is_ok(), "resolve_manual_camera failed: {:?}", res.err());
        let cam = res.unwrap();
        assert!(cam.is_some());
        let c = cam.unwrap();
        assert_eq!(c.ip, "127.0.0.1");
        assert_eq!(c.rtsp, "0");
    }

    #[test]
    fn test_python_discover_cameras() {
        pyo3::prepare_freethreaded_python();
        let res = discover_cameras("cam", "12345678", 1);
        assert!(res.is_ok(), "discover_cameras failed: {:?}", res.err());
        let cams = res.unwrap();
        println!("Discovered cams count: {}", cams.len());
    }
}

