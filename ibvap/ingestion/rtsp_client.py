"""
RTSP Streaming Client and Manager.
Handles authenticated RTSP URL resolution and low-latency VideoCapture streaming.
"""

from typing import Optional, Dict, Any, Tuple
import cv2
import logging
from .connection import connect
from .buffer import FrameRingBuffer

logger = logging.getLogger("ibvap.ingestion.rtsp")


async def resolve_rtsp_url(cam, username: str = "cam", passwd: str = "12345678") -> str:
    """
    Queries ONVIF Media service to obtain the primary profile RTSP stream URI.
    """
    media = await cam.create_media_service()
    profiles = await media.GetProfiles()
    if not profiles:
        raise RuntimeError("No media profiles returned by camera")
    main_profile = profiles[0].token

    req = media.create_type('GetStreamUri')
    req.ProfileToken = main_profile
    req.StreamSetup = {
        'Stream': 'RTP-Unicast',
        'Transport': {'Protocol': 'RTSP'}
    }

    stream_response = await media.GetStreamUri(req)
    raw_rtsp_url = stream_response.Uri

    if "rtsp://" in raw_rtsp_url:
        return raw_rtsp_url.replace("rtsp://", f"rtsp://{username}:{passwd}@")
    return raw_rtsp_url


class RTSPStreamClient:
    """
    Configures low-latency OpenCV VideoCapture for RTSP video streams with single-frame buffering.
    """

    def __init__(self, rtsp_url: str, camera_id: str = "camera-01", tcp_transport: bool = True):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.tcp_transport = tcp_transport
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Opens the RTSP stream with low-latency parameters."""
        # Enforce TCP transport via environment or backend flags where supported
        self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            logger.error(f"Failed to open RTSP stream for {self.camera_id}: {self.rtsp_url}")
            return False

        # Single frame buffer depth to prevent latency accumulation
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return True

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Reads a single frame from the stream."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        return self._cap.read()

    def release(self):
        """Closes and releases the video capture stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
