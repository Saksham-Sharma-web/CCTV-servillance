"""
Bridges a single RTSP camera into the IBVAP AI pipeline for the Rust/Slint
UI. Each `LiveCameraStream` owns one cv2.VideoCapture + one IBVAPPipeline
instance (trackers are per-camera, so track/identity state from one
camera never bleeds into another). Rust drives it one frame at a time
over PyO3 from its own OS thread, so capture + inference for N cameras
can run concurrently without blocking the Slint event loop.
"""

import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibvap.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import VirtualBoundary, ZoneType

class LiveCameraStream:

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.boundary_initialized = False

        os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
        os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

        if rtsp_url.isdigit():
            self.cap = cv2.VideoCapture(int(rtsp_url))
        else:
            self.cap = cv2.VideoCapture(rtsp_url, cv2.CAP_ANY)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Do not raise an exception if it fails to open, just let `next_frame()` handle it gracefully

        # Redis/DB integrations stay off here — the Rust side already
        # owns persistence (cameras.db) and the on-screen event log.
        self.pipeline = IBVAPPipeline(
            config=IBVAPConfig(redis_enabled=False, db_enabled=False)
        )

    def next_frame(self):
        """
        Reads one frame and runs it through the full AI pipeline
        (detection -> tracking -> face/ANPR -> behavioral analytics).

        Returns (annotated_bgr_bytes, width, height, events_json), or
        None once the stream ends / drops so Rust can stop the loop.
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None

        actual_h, actual_w = frame.shape[:2]

        # Setup spatial boundary (center tripwire) once frame size is known
        if not self.boundary_initialized:
            mid_x = actual_w // 2
            self.pipeline.add_boundary(
                VirtualBoundary(
                    id=f"fence-{self.camera_id}",
                    name=f"Perimeter Line ({self.camera_id})",
                    zone_type=ZoneType.LINE,
                    coordinates=[(mid_x, 0), (mid_x, actual_h)],
                    target_classes=["person", "car", "motorcycle"]
                )
            )
            self.boundary_initialized = True

        result = self.pipeline.process_frame(
            frame, camera_id=self.camera_id, timestamp=time.time()
        )
        annotated = np.ascontiguousarray(
            self.pipeline.draw_debug(frame, result)
        )
        h, w = annotated.shape[:2]

        events = [e.to_dict() for e in result.events]

        return (bytes(annotated), w, h, json.dumps(events))

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

