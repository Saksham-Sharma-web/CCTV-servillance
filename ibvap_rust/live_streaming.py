"""
LiveCameraStream — 3-thread architecture.

THREAD 1 ─ Network/Connection Manager
  Owns the cv2.VideoCapture object entirely.
  Reads raw decoded BGR frames from the RTSP stream (H.264 or whatever
  the camera sends — no codec forcing, no MJPEG option).
  Reconnects silently on any failure.
  Writes the latest frame into a shared atomic slot so consumers
  always get the newest frame without ever blocking the reader.

THREAD 2 ─ Frame Encoder / Display Queue
  Reads from the raw-frame slot (latest only).
  JPEG-encodes the frame for the live display path.
  Maintains a small smoothing ring buffer (max 2 frames) to absorb
  momentary capture jitter without building latency.
  Also submits raw frames to the global AI worker at 10 fps.

THREAD 3 ─ AI Worker (global singleton, shared across ALL cameras)
  Owned by _GlobalAIWorker.  Serialises all YOLO inference so only
  one model call runs at a time — prevents OOM and segfaults.

KEY INVARIANTS
- Only H.264 TCP RTSP is used; the codec is chosen by the camera.
- The live display path never waits for AI.
- The AI never causes camera disconnections.
- A dead camera retries every 3 s without blocking other cameras.
"""

import datetime
import json
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np


# ── Suppress OpenCV / FFMPEG noise ─────────────────────────────────────────
os.environ["OPENCV_LOG_LEVEL"]            = "FATAL"
os.environ["OPENCV_FFMPEG_LOGLEVEL"]      = "-8"
# TCP transport + 5-second socket timeout.  NO codec forcing — let the
# camera negotiate whatever codec it supports (H.264, H.265, etc.)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

_LIVE_JPEG_QUALITY = 60          # q60 → ~15-30 KB per frame at 720p
_JPEG_PARAMS       = [cv2.IMWRITE_JPEG_QUALITY, _LIVE_JPEG_QUALITY]
_AI_INTERVAL       = 0.10        # submit one frame to AI every 100 ms
_RECONNECT_DELAY   = 3.0         # seconds between reconnect attempts
_DISPLAY_FPS_CAP   = 30          # max frames pushed to the display queue
_DISPLAY_INTERVAL  = 1.0 / _DISPLAY_FPS_CAP

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ibvap.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import VirtualBoundary, ZoneType


def _log(camera_id: str, msg: str):
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] [PYTHON] [{camera_id}] {msg}", flush=True)


# ════════════════════════════════════════════════════════════════════════════
# THREAD 3 — GLOBAL AI WORKER (singleton)
# ════════════════════════════════════════════════════════════════════════════

class _GlobalAIWorker:
    """
    Single background thread that owns the IBVAPPipeline.
    All cameras share this one inference slot — zero concurrent YOLO calls.
    """
    _instance: "_GlobalAIWorker | None" = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls) -> "_GlobalAIWorker":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        _log("AI-WORKER", "Initialising IBVAPPipeline (shared global)…")
        self._pipeline = IBVAPPipeline(
            config=IBVAPConfig(redis_enabled=False, db_enabled=False)
        )
        _log("AI-WORKER", "Pipeline ready.")

        # Bounded work queue: when full, submit() drops silently so the live
        # path is never slowed down waiting for AI.
        self._work_q: "queue.Queue[tuple[str, np.ndarray] | None]" = queue.Queue(maxsize=4)

        # Results per camera: {camera_id: [event_dict, …]}
        self._results: "dict[str, list]" = {}
        self._results_lock = threading.Lock()

        # Track which cameras have had their boundary set up
        self._boundaries: "set[str]" = set()
        self._boundaries_lock = threading.Lock()

        self._thread = threading.Thread(
            target=self._loop, name="ai-global-worker", daemon=True
        )
        self._thread.start()
        _log("AI-WORKER", "Worker thread started.")

    # ── Public API ────────────────────────────────────────────────────────

    def submit(self, camera_id: str, frame: np.ndarray):
        """Non-blocking frame submission. Drops on full queue."""
        try:
            self._work_q.put_nowait((camera_id, frame))
        except queue.Full:
            pass

    def drain_events(self, camera_id: str) -> list:
        """Pop and return all events accumulated for camera_id."""
        with self._results_lock:
            return self._results.pop(camera_id, [])

    # ── Private ───────────────────────────────────────────────────────────

    def _ensure_boundary(self, camera_id: str, w: int, h: int):
        with self._boundaries_lock:
            if camera_id in self._boundaries:
                return
        mid_x = w // 2
        try:
            self._pipeline.add_boundary(
                VirtualBoundary(
                    id=f"fence-{camera_id}",
                    name=f"Perimeter Line ({camera_id})",
                    zone_type=ZoneType.LINE,
                    coordinates=[(mid_x, 0), (mid_x, h)],
                    target_classes=["person", "car", "motorcycle"],
                )
            )
            with self._boundaries_lock:
                self._boundaries.add(camera_id)
            _log("AI-WORKER", f"Boundary set for {camera_id} at x={mid_x}")
        except Exception as exc:
            _log("AI-WORKER", f"Boundary setup failed for {camera_id}: {exc}")

    def _loop(self):
        while True:
            item = self._work_q.get()
            if item is None:
                break  # shutdown signal
            camera_id, frame = item
            h, w = frame.shape[:2]
            self._ensure_boundary(camera_id, w, h)
            try:
                result = self._pipeline.process_frame(
                    frame, camera_id=camera_id, timestamp=time.time()
                )
                if result and result.events:
                    events = []
                    import os
                    os.makedirs("events", exist_ok=True)
                    
                    for e in result.events:
                        edict = e.to_dict()
                        
                        # Generate annotated snapshot
                        ann_frame = frame.copy()
                        
                        # Prepare text
                        dt_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ev_type_str = edict.get("event_type", "UNKNOWN")
                        conf = edict.get("confidence", 1.0) * 100
                        metadata = edict.get("metadata", {})
                        
                        # Extract basic reason from metadata or event type
                        reason = metadata.get("reason", "")
                        identity = metadata.get("name", metadata.get("plate_number", ""))
                        
                        lines = [
                            f"CAMERA: {camera_id}",
                            f"EVENT: {ev_type_str}",
                            f"TIME: {dt_str}",
                            f"CONFIDENCE: {conf:.1f}%"
                        ]
                        if identity:
                            lines.append(f"IDENTITY: {identity}")
                        if reason:
                            lines.append(f"REASON: {reason}")
                            
                        # Draw semi-transparent background box
                        h, w = ann_frame.shape[:2]
                        box_w = 400
                        box_h = 30 * len(lines) + 20
                        
                        overlay = ann_frame.copy()
                        cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.6, ann_frame, 0.4, 0, ann_frame)
                        
                        # Draw text
                        y = 40
                        for line in lines:
                            cv2.putText(ann_frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            y += 30
                            
                        # Optional: Draw bounding box if available
                        bbox = metadata.get("plate_bbox") or metadata.get("bbox")
                        if bbox and len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                            cv2.rectangle(ann_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                        snap_path = f"events/{edict['event_id']}.jpg"
                        cv2.imwrite(snap_path, ann_frame)
                        edict["snapshot_path"] = snap_path
                        
                        events.append(edict)
                        
                    _log("AI-WORKER", f"{camera_id}: {len(events)} event(s)")
                    with self._results_lock:
                        self._results.setdefault(camera_id, []).extend(events)
            except Exception as exc:
                _log("AI-WORKER", f"Inference error on {camera_id}: {exc}")


# ════════════════════════════════════════════════════════════════════════════
# PER-CAMERA STREAM  ──  3-thread design
# ════════════════════════════════════════════════════════════════════════════

class LiveCameraStream:
    """One instance per camera, three daemon threads."""

    def __init__(self, camera_id: str, rtsp_url: str):
        self.camera_id = camera_id
        self.rtsp_url  = rtsp_url
        self._stop     = threading.Event()

        # ── THREAD 1 → THREAD 2 handoff: latest raw BGR frame (atomic slot) ──
        # Lock + plain slot is faster than queue.Queue(maxsize=1) for this case
        self._raw_lock  = threading.Lock()
        self._raw_frame: "np.ndarray | None" = None

        # ── THREAD 2 → Rust handoff: ring buffer of up to 2 JPEG frames ──────
        # A ring of 2 absorbs single-frame jitter while keeping latency ≤ 2×33ms
        self._display_q: "queue.Queue[tuple[bytes,int,int]]" = queue.Queue(maxsize=2)

        # ── AI worker (global singleton) ─────────────────────────────────────
        self._ai            = _GlobalAIWorker.get()
        self._last_ai_time  = 0.0

        # ── Start threads ─────────────────────────────────────────────────────
        _log(camera_id, f"Initialising — RTSP: {rtsp_url}")

        self._net_thread = threading.Thread(
            target=self._network_thread,
            name=f"net-{camera_id}",
            daemon=True,
        )
        self._enc_thread = threading.Thread(
            target=self._encoder_thread,
            name=f"enc-{camera_id}",
            daemon=True,
        )
        self._net_thread.start()
        self._enc_thread.start()
        _log(camera_id, "Network + Encoder threads started.")

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD 1: Network / Connection Manager
    # Owns cv2.VideoCapture — nothing else touches it.
    # Writes the newest decoded frame into _raw_slot.
    # ─────────────────────────────────────────────────────────────────────────

    def _open_capture(self) -> "cv2.VideoCapture | None":
        _log(self.camera_id, f"Connecting to RTSP: {self.rtsp_url}")
        try:
            if self.rtsp_url.isdigit():
                cap = cv2.VideoCapture(int(self.rtsp_url))
            else:
                # CAP_FFMPEG with TCP transport set via env var above.
                # Let the camera choose its codec — no forced MJPEG.
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            # Internal FFMPEG buffer of 1 frame so we always decode newest
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                cap.release()
                _log(self.camera_id, "RTSP connection refused or timed out")
                return None

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            _log(self.camera_id, f"Connected — {w}x{h} @ {fps:.1f} fps")
            return cap
        except Exception as exc:
            _log(self.camera_id, f"Exception opening capture: {exc}")
            return None

    def _network_thread(self):
        """
        Thread 1: RTSP capture.
        Reads frames as fast as the camera delivers them.
        Writes the latest frame atomically into _raw_frame (lock + swap).
        On any failure, releases the capture and sleeps before retry.
        """
        cap = None

        while not self._stop.is_set():
            # (Re)connect
            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = self._open_capture()
                if cap is None:
                    time.sleep(_RECONNECT_DELAY)
                    continue

            # Decode one frame (blocking until camera delivers it)
            ok, frame = cap.read()

            if not ok or frame is None:
                _log(self.camera_id, "Frame read failed — will reconnect")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                time.sleep(_RECONNECT_DELAY)
                continue

            # Atomic slot write: encoder thread always gets latest frame
            with self._raw_lock:
                self._raw_frame = frame   # reference swap — O(1)

        # Cleanup
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        _log(self.camera_id, "Network thread exited.")

    # ─────────────────────────────────────────────────────────────────────────
    # THREAD 2: Frame Encoder / Display Queue
    # Reads the latest raw frame, JPEG-encodes it, drops into the display
    # ring buffer, and submits raw frames to the AI worker.
    # ─────────────────────────────────────────────────────────────────────────

    def _encoder_thread(self):
        """
        Thread 2: encode + queue.
        Caps display output at _DISPLAY_FPS_CAP.
        Submits raw BGR frames to AI at _AI_INTERVAL.
        Absorbs jitter via the bounded display_q ring buffer.
        """
        last_enc_time = 0.0

        while not self._stop.is_set():
            now = time.monotonic()
            elapsed = now - last_enc_time

            # Enforce display FPS cap
            if elapsed < _DISPLAY_INTERVAL:
                time.sleep(_DISPLAY_INTERVAL - elapsed)
                continue

            # Grab latest raw frame
            with self._raw_lock:
                frame = self._raw_frame
                self._raw_frame = None  # consume — avoid re-encoding same frame

            if frame is None:
                # No new frame yet from network thread; wait a bit
                time.sleep(0.005)
                continue

            last_enc_time = time.monotonic()
            h, w = frame.shape[:2]

            # ── JPEG encode for live display ─────────────────────────────────
            try:
                if frame is None or frame.size == 0:
                    continue
                ok_enc, jpg_buf = cv2.imencode(".jpg", frame, _JPEG_PARAMS)
                if not ok_enc:
                    continue
            except Exception as e:
                import logging
                logging.getLogger("live_streaming").warning(f"Failed to encode frame: {e}")
                continue

            jpg_bytes = jpg_buf.tobytes()

            # Push to display ring (drop oldest if full to keep latency low)
            if self._display_q.full():
                try:
                    self._display_q.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._display_q.put_nowait((jpg_bytes, w, h))
            except queue.Full:
                pass

            # ── Throttled AI submission ───────────────────────────────────────
            now = time.monotonic()
            if now - self._last_ai_time >= _AI_INTERVAL:
                self._ai.submit(self.camera_id, frame.copy())
                self._last_ai_time = now

        _log(self.camera_id, "Encoder thread exited.")

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC API — called from Rust via PyO3
    # ─────────────────────────────────────────────────────────────────────────

    def next_frame(self):
        """
        Returns (jpeg_bytes, width, height, events_json) or None.

        Called by the Rust spawn_blocking loop at the rate Rust drains it.
        The display ring buffer decouples Rust's polling rate from the
        camera's frame rate, smoothing any network jitter.
        """
        if self._stop.is_set():
            return None

        # Non-blocking get — Rust sleeps 5 ms and retries when None
        try:
            jpg_bytes, w, h = self._display_q.get_nowait()
        except queue.Empty:
            return None

        events = self._ai.drain_events(self.camera_id)
        return (jpg_bytes, w, h, json.dumps(events))

    def release(self):
        """Signal all threads to stop."""
        _log(self.camera_id, "Release called — stopping all threads")
        self._stop.set()
