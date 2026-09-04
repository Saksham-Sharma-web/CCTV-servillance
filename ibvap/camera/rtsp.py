import cv2
import time
import threading
import queue
import logging
from typing import Optional

from ibvap.camera.models import CameraConfig, FramePacket, StreamHealth

logger = logging.getLogger(__name__)

class RTSPSource:
    """
    Manages an RTSP stream using a background thread and a bounded queue
    to ensure the AI pipeline only processes the latest available frames.
    """
    def __init__(self, config: CameraConfig):
        self.config = config
        self.queue: queue.Queue[FramePacket] = queue.Queue(maxsize=2)
        self.health = StreamHealth()
        
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        self._frame_index = 0
        self._cap = None

    def start(self):
        """Starts the background frame ingestion thread."""
        if self._thread is not None and self._thread.is_alive():
            return
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"RTSP-{self.config.id}")
        self._thread.start()

    def stop(self):
        """Stops the ingestion thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
            
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_latest(self, timeout: float = 0.5) -> Optional[FramePacket]:
        """
        Retrieves the latest frame from the queue.
        Blocks for up to `timeout` seconds if empty.
        """
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self):
        """The core ingestion loop."""
        import os
        # Set ffmpeg to prefer TCP for stability over erratic networks
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        
        uri = self.config.uri
        if not uri:
            if self.config.source_type.value == "usb":
                uri = 0 # Default USB device
            else:
                self.health.error_message = "No URI provided for stream"
                self.health.is_active = False
                return

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                logger.info(f"[{self.config.id}] Connecting to stream: {uri}")
                self._cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG if isinstance(uri, str) else cv2.CAP_ANY)
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                if not self._cap.isOpened():
                    self.health.is_active = False
                    self.health.reconnect_count += 1
                    logger.error(f"[{self.config.id}] Failed to open stream. Retrying in 5 seconds...")
                    time.sleep(5)
                    continue
                else:
                    self.health.is_active = True
                    logger.info(f"[{self.config.id}] Stream connected.")

            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning(f"[{self.config.id}] Stream dropped or EOF. Reconnecting...")
                self.health.is_active = False
                self.health.dropped_frames += 1
                self._cap.release()
                self._cap = None
                time.sleep(2)
                continue

            # We successfully got a frame
            self.health.is_active = True
            self.health.last_frame_timestamp = time.time()
            self._frame_index += 1
            
            packet = FramePacket(
                camera_id=self.config.id,
                timestamp=self.health.last_frame_timestamp,
                frame=frame,
                frame_index=self._frame_index
            )

            # If queue is full, discard the oldest frame to maintain latest-frame semantics
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.health.dropped_frames += 1
                except queue.Empty:
                    pass
            
            try:
                self.queue.put_nowait(packet)
            except queue.Full:
                pass
