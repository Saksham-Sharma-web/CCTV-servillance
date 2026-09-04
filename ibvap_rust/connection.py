import asyncio
import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
# pyrefly: ignore [missing-import]
import cv2
from onvif import ONVIFCamera


async def connect_camera(device, username="cam", passwd="12345678"):
    """Connects to an ONVIF device and returns the ONVIFCamera instance."""
    xaddrs = device.get("xAddrs", "")
    if not xaddrs:
        return None

    try:
        parts = xaddrs.split("/")
        if len(parts) > 2:
            s = parts[2]
        else:
            s = xaddrs
        host, _, port_str = s.partition(":")
        port = int(port_str) if port_str.isdigit() else 80
    except Exception as e:
        print(f"[connection] Error parsing xAddrs {xaddrs}: {e}")
        return None

    cam = None
    try:
        cam = ONVIFCamera(
            host=host,
            port=port,
            user=username,
            passwd=passwd,
            encrypt=True
        )
        await cam._devicemgmt_with_time()
        await cam.update_xaddrs()
        print(f"[connection] ONVIF connection to {host}:{port} successful")
        return cam
    except Exception as e:
        print(f"[connection] ONVIF connection failed for {host}:{port}: {e}")
        if cam is not None:
            try:
                await cam.close()
            except Exception:
                pass
        return None


async def get_rtsp_url(cam, username="cam", passwd="12345678"):
    """Queries media service profiles and returns an authenticated RTSP URL."""
    try:
        media = await cam.create_media_service()
        profiles = await media.GetProfiles()
        if not profiles:
            return None

        profile = profiles[0]
        req = media.create_type("GetStreamUri")
        req.ProfileToken = profile.token
        req.StreamSetup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"}
        }

        response = await media.GetStreamUri(req)
        uri = response.Uri

        # Inject credentials if not present
        if "@" not in uri and username:
            uri = uri.replace("rtsp://", f"rtsp://{username}:{passwd}@")
        return uri
    except Exception as e:
        print(f"[connection] Failed to get RTSP URL: {e}")
        return None


def test_rtsp_stream(rtsp_url: str, timeout_sec: int = 3) -> bool:
    """Verifies that an RTSP URL or camera index can be opened and a frame can be captured."""
    if not rtsp_url:
        return False
    try:
        if rtsp_url.isdigit():
            cap = cv2.VideoCapture(int(rtsp_url))
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            cap.release()
            return False

        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except Exception as e:
        print(f"[connection] test_rtsp_stream error for {rtsp_url}: {e}")
        return False


def resolve_manual_camera(raw_input: str, username: str = "cam", passwd: str = "12345678") -> dict:
    """
    Takes an input IP, host, or RTSP URL and returns a resolved camera dict:
    { "id": ..., "name": ..., "ip": ..., "rtsp": ... }
    """
    raw_input = raw_input.strip()
    if not raw_input:
        return None

    # Case 1: Local webcam
    if raw_input.isdigit() or raw_input == "webcam":
        cam_idx = "0" if not raw_input.isdigit() else raw_input
        return {
            "id": f"webcam-{cam_idx}",
            "name": f"Local Device #{cam_idx}",
            "ip": "127.0.0.1",
            "rtsp": cam_idx
        }

    # Case 2: Full RTSP URL supplied
    if raw_input.startswith("rtsp://"):
        url = raw_input
        if "@" not in url and username:
            url = url.replace("rtsp://", f"rtsp://{username}:{passwd}@")
        host = url.split("@")[-1].split(":")[0].split("/")[0]
        return {
            "id": f"rtsp-{host}",
            "name": f"Manual RTSP ({host})",
            "ip": host,
            "rtsp": url
        }

    # Case 3: Raw IP or IP:port (e.g. 192.168.0.105 or 192.168.0.105:8554)
    clean_ip = raw_input
    # Check if a port was specified
    if ":" in clean_ip:
        host, _, port = clean_ip.partition(":")
    else:
        host = clean_ip
        port = "8554"

    # Try common RTSP candidates
    candidates = [
        f"rtsp://{username}:{passwd}@{host}:{port}/live",
        f"rtsp://{username}:{passwd}@{host}:8554/live",
        f"rtsp://{username}:{passwd}@{host}:554/live",
        f"rtsp://{username}:{passwd}@{host}:554/h264",
        f"rtsp://{username}:{passwd}@{host}:554/ch0",
    ]

    for candidate in candidates:
        if test_rtsp_stream(candidate):
            return {
                "id": f"manual-{host}",
                "name": f"Camera {host}",
                "ip": host,
                "rtsp": candidate
            }

    # If test did not immediately succeed, default to standard RTSP URL
    fallback_url = f"rtsp://{username}:{passwd}@{host}:8554/live"
    return {
        "id": f"manual-{host}",
        "name": f"Camera {host}",
        "ip": host,
        "rtsp": fallback_url
    }

