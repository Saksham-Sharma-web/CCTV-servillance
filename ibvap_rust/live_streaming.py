"""
LiveCameraStream — dual-path architecture.

PATH 1 (live):  RTSP decode → bounded queue[1] → Rust/Slint UI
PATH 2 (AI):    bounded queue → every-N-frame AI worker → events queue → Rust

The live path has ZERO AI overhead.  The UI always receives the latest raw
frame; if the AI is still crunching YOLO it just skips frames instead of
building a latency backlog.

One instance per camera.  Rust creates and drives it via PyO3 from a
dedicated OS thread (spawn_blocking), so N cameras can run truly in parallel.
"""

import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

os.environ.setdefault("OPENCV_LOG_LEVEL", "FATAL")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibvap.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import VirtualBoundary, ZoneType


class LiveCameraStream:

    # How many raw frames to skip between AI inference runs.
    # At ~25 FPS a value of 5 gives ~5 AI FPS — plenty for analytics.
    AI_FRAME_SKIP = 5

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self._stop = threading.Event()
        self._boundary_ready = False

        # ── live path: single-element "latest frame" slot ────────────────────
        # maxsize=1 means the producer always overwrites stale frames;
        # the consumer (Rust) always gets the newest available image.
        self._live_q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=2)

        # ── AI path: events accumulated between Rust poll calls ──────────────
        self._event_q: "queue.Queue[list]" = queue.Queue()

        # ── AI pipeline (one per camera, thread-safe within a single thread) ─
        self._pipeline = IBVAPPipeline(
            config=IBVAPConfig(redis_enabled=False, db_enabled=False)
        )

        # ── start background threads ─────────────────────────────────────────
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name=f"capture-{camera_id}", daemon=True
        )
        self._ai_thread = threading.Thread(
            target=self._ai_loop, name=f"ai-{camera_id}", daemon=True
        )
        self._capture_thread.start()
        self._ai_thread.start()

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD 1: Raw RTSP capture — feeds the live queue as fast as possible
    # ─────────────────────────────────────────────────────────────────────────

    def _open_cap(self):
        """Open (or re-open) cv2.VideoCapture. Never throws."""
        try:
            if self.rtsp_url.isdigit():
                cap = cv2.VideoCapture(int(self.rtsp_url))
            else:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap if cap.isOpened() else None
        except Exception:
            return None

    def _capture_loop(self):
        cap = None
        reconnect_delay = 5.0  # seconds between reconnect attempts

        while not self._stop.is_set():
            if cap is None or not cap.isOpened():
                if cap is not None:
                    cap.release()
                cap = self._open_cap()
                if cap is None:
                    time.sleep(reconnect_delay)
                    continue

            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(reconnect_delay)
                continue

            # Drop old frame, keep newest — implements "latest frame" semantics
            if self._live_q.full():
                try:
                    self._live_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._live_q.put_nowait(frame)
            except queue.Full:
                pass

        if cap is not None:
            cap.release()

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD 2: AI inference — processes every Nth frame independently
    # ─────────────────────────────────────────────────────────────────────────

    def _ai_loop(self):
        skip_count = 0

        while not self._stop.is_set():
            try:
                frame = self._live_q.get(timeout=0.5)
            except queue.Empty:
                continue

            skip_count += 1
            if skip_count < self.AI_FRAME_SKIP:
                continue
            skip_count = 0

            h, w = frame.shape[:2]

            # Set up the perimeter tripwire once we know the frame size
            if not self._boundary_ready:
                mid_x = w // 2
                try:
                    self._pipeline.add_boundary(
                        VirtualBoundary(
                            id=f"fence-{self.camera_id}",
                            name=f"Perimeter Line ({self.camera_id})",
                            zone_type=ZoneType.LINE,
                            coordinates=[(mid_x, 0), (mid_x, h)],
                            target_classes=["person", "car", "motorcycle"],
                        )
                    )
                    self._boundary_ready = True
                except Exception:
                    pass

            try:
                result = self._pipeline.process_frame(
                    frame, camera_id=self.camera_id, timestamp=time.time()
                )
                if result and result.events:
                    events = [e.to_dict() for e in result.events]
                    self._event_q.put(events)
            except Exception as exc:
                # Never crash the AI thread — log and continue
                print(f"[live_streaming] AI error on {self.camera_id}: {exc}")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API: called by Rust over PyO3 from spawn_blocking thread
    # ─────────────────────────────────────────────────────────────────────────

    def next_frame(self):
        """
        Returns (raw_bgr_bytes, width, height, events_json_str) or None.

        • `raw_bgr_bytes` is the unmodified camera frame — zero AI overhead
          on the live path.
        • `events_json_str` contains any AI events that completed
          asynchronously since the previous call.
        """
        if self._stop.is_set():
            return None

        # Do not block and hold the GIL here. If empty, return None immediately.
        try:
            frame = self._live_q.get_nowait()
        except queue.Empty:
            return None

        if frame is None:
            return None

        h, w = frame.shape[:2]

        # Drain all AI events accumulated since last call
        all_events: list = []
        while not self._event_q.empty():
            try:
                all_events.extend(self._event_q.get_nowait())
            except queue.Empty:
                break

        # Rust expects contiguous bytes
        bgr_bytes = np.ascontiguousarray(frame).tobytes()
        return (bgr_bytes, w, h, json.dumps(all_events))

    def release(self):
        """Signal all background threads to stop."""
        self._stop.set()
