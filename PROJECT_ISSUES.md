# Project Issues Report (CCTV Surveillance & IBVAP)

Based on a scan of the project codebase, logs, and system state, the following issues have been identified:

## 1. Camera Connection Instability (RTSP Drop & Infinite Retry Loop)
- **Issue**: The application successfully connects to the camera at `192.168.1.7` via RTSP initially but drops the connection almost immediately (`Frame read failed`). It then enters an aggressive loop of retrying RTSP and falling back to HTTP MJPEG, which both return `Connection Refused`. 
- **Reason / Root Cause**: The camera is either overloaded by the connection, enforcing strict stream limits, or the network is unstable. In `ibvap_rust/live_streaming.py`, the fallback logic has a fixed `_RECONNECT_DELAY = 3.0`. When the RTSP stream drops, it repeatedly hammers the camera with connection attempts instead of using an exponential backoff.

## 2. Missing Reference Image for Face Recognition
- **Issue**: The face recognition initialization fails silently because the reference image is missing.
- **Reason / Root Cause**: In `main.py`, the variable `REF_FACE_IMAGE` is hardcoded to point to `"test.png"` (line 19: `os.path.join(os.path.dirname(__file__), "test.png")`). However, this file does not exist in the root directory (there is only `test_car.png`). The `register_reference_face` function handles this gracefully by logging a warning, but the core functionality (matching "Authorized User") is completely broken as a result.

## 3. Python Environment Not Managed Correctly (PEP 668)
- **Issue**: The system is running on a modern Linux distribution that enforces PEP 668 (`externally-managed-environment`). Trying to install Python dependencies globally via `pip` fails.
- **Reason / Root Cause**: The project heavily relies on Python (`stream.py`, `live_streaming.py`, `main.py`) alongside Rust, but there is no mechanism (like a `venv` setup in `start.sh` or `build.rs`) ensuring that `pip install -r requirements.txt` is run within an isolated virtual environment. Developers will face environment errors when setting up the project.

## 4. Hardcoded Fallbacks and Credentials
- **Issue**: If the ONVIF discovery fails or connection fails, the system defaults to unsafe hardcoded fallbacks.
- **Reason / Root Cause**: 
  - `main.py` defaults to `uri=0` (Local Webcam) if no ONVIF cameras are discovered. If this is deployed on a headless server without a USB webcam, OpenCV will fail to capture frames.
  - Hardcoded credentials (`cam:12345678`) are scattered across `stream.py` and `live_streaming.py` as default arguments. While `main.py` tries to load them from environment variables, the default values are embedded deep in the fallback logic.

## 5. Potential Git Synchronization Issues
- **Issue**: The recent git operations showed an error: `Your local changes to the following files would be overwritten by checkout: README.md`. 
- **Reason / Root Cause**: There are active, uncommitted changes that had to be stashed (`WIP on tauri-ui: b806887 Broken Commit, Tauri switch`). This indicates a fragmented branch state where local modifications might not be correctly synced with the `Saksham` branch, leading to testing on a potentially broken or incomplete codebase.
