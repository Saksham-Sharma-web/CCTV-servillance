"""
IBVAP Performance Profiler — Thread-safe rolling statistics singleton.

Collected automatically by pipeline.py, object_detector.py, and live_streaming.py.
Accessible at runtime via:
    from ibvap.core.profiler import Profiler
    stats = Profiler.get().snapshot()

The Rust web server reads Python stats via the same call at /api/perf.
"""

import math
import threading
import time
from collections import deque
from typing import Any, Dict, Optional


_WINDOW = 200  # samples kept per metric


class _RollingStats:
    """Lock-free rolling window over the last N float samples."""

    __slots__ = ("_dq", "_lock")

    def __init__(self, window: int = _WINDOW):
        self._dq: deque = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, value_ms: float) -> None:
        with self._lock:
            self._dq.append(value_ms)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            data = list(self._dq)
        n = len(data)
        if n == 0:
            return {"n": 0, "mean_ms": 0.0, "median_ms": 0.0,
                    "p95_ms": 0.0, "p99_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
        s = sorted(data)
        mean = sum(s) / n
        variance = sum((x - mean) ** 2 for x in s) / n
        return {
            "n": n,
            "mean_ms":   round(mean, 2),
            "median_ms": round(s[n // 2], 2),
            "p95_ms":    round(s[min(n - 1, int(n * 0.95))], 2),
            "p99_ms":    round(s[min(n - 1, int(n * 0.99))], 2),
            "min_ms":    round(s[0], 2),
            "max_ms":    round(s[-1], 2),
            "stdev_ms":  round(math.sqrt(variance), 2),
        }


class _FpsCounter:
    """Counts events per second using a rolling 5-second window."""

    __slots__ = ("_timestamps", "_lock", "_window_s")

    def __init__(self, window_s: float = 5.0):
        self._timestamps: deque = deque()
        self._lock = threading.Lock()
        self._window_s = window_s

    def tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._timestamps.append(now)
            # prune old entries
            cutoff = now - self._window_s
            while self._timestamps and self._timestamps[0] < cutoff:
                self._timestamps.popleft()

    def fps(self) -> float:
        now = time.monotonic()
        with self._lock:
            cutoff = now - self._window_s
            count = sum(1 for t in self._timestamps if t >= cutoff)
        return round(count / self._window_s, 2)


class _AtomicCounter:
    __slots__ = ("_v", "_lock")

    def __init__(self):
        self._v = 0
        self._lock = threading.Lock()

    def inc(self, by: int = 1) -> None:
        with self._lock:
            self._v += by

    def value(self) -> int:
        with self._lock:
            return self._v


class Profiler:
    """
    Process-wide singleton collecting latency for every pipeline stage.
    Access via Profiler.get().
    """

    _instance: Optional["Profiler"] = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls) -> "Profiler":
        if cls._instance is None:
            with cls._init_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        # ── Python live_streaming.py stage timings ─────────────────────────
        # Thread 1: RTSP decode time (cap.read wall time)
        self.rtsp_decode_ms      = _RollingStats()
        # Thread 2: JPEG encode time
        self.jpeg_encode_ms      = _RollingStats()
        # Thread 2: inter-frame interval (time between successive encoded frames)
        self.interframe_ms       = _RollingStats()
        # Display ring queue wait (time from put to get in next_frame)
        self.display_queue_ms    = _RollingStats()
        # AI submit rate (frames actually submitted vs dropped)
        self.ai_submit_fps       = _FpsCounter()
        self.ai_dropped_frames   = _AtomicCounter()
        self.display_fps         = _FpsCounter()

        # ── Python AI pipeline step timings ──────────────────────────────────
        # Step 1: YOLO object detection
        self.yolo_detect_ms      = _RollingStats()
        # Step 2: Multi-object tracking (Kalman + IoU)
        self.tracking_ms         = _RollingStats()
        # Step 3: Face detection (YuNet)
        self.face_detect_ms      = _RollingStats()
        # Step 3: Face verification (FaceNet/ArcFace)
        self.face_verify_ms      = _RollingStats()
        # Step 4: ANPR plate detection
        self.plate_detect_ms     = _RollingStats()
        # Step 4: OCR
        self.ocr_ms              = _RollingStats()
        # Step 5: Behavioral analytics (all three combined)
        self.analytics_ms        = _RollingStats()
        # Step 6: Event engine dedup
        self.event_engine_ms     = _RollingStats()
        # Full pipeline (sum of all steps per frame)
        self.pipeline_total_ms   = _RollingStats()
        # AI frames actually processed
        self.ai_processed_fps    = _FpsCounter()
        self.ai_total_frames     = _AtomicCounter()
        self.ai_event_count      = _AtomicCounter()

        # ── Counters ──────────────────────────────────────────────────────────
        self.face_check_count    = _AtomicCounter()  # how many face checks ran
        self.face_skip_quality   = _AtomicCounter()  # quality gate rejects
        self.ocr_check_count     = _AtomicCounter()
        self.ocr_skip_quality    = _AtomicCounter()

        # ── Start timestamp ───────────────────────────────────────────────────
        self.started_at = time.time()

    # ── Convenience context manager ────────────────────────────────────────────
    class _Timer:
        __slots__ = ("_stat", "_t0")

        def __init__(self, stat: _RollingStats):
            self._stat = stat
            self._t0 = 0.0

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *_):
            self._stat.record((time.perf_counter() - self._t0) * 1000.0)

    def timer(self, stat: _RollingStats) -> "_Timer":
        return self._Timer(stat)

    # ── Public snapshot ────────────────────────────────────────────────────────
    def snapshot(self) -> Dict[str, Any]:
        uptime = round(time.time() - self.started_at, 1)
        return {
            "uptime_s": uptime,
            "streaming": {
                "rtsp_decode_ms":    self.rtsp_decode_ms.snapshot(),
                "jpeg_encode_ms":    self.jpeg_encode_ms.snapshot(),
                "interframe_ms":     self.interframe_ms.snapshot(),
                "display_queue_ms":  self.display_queue_ms.snapshot(),
                "display_fps":       self.display_fps.fps(),
                "ai_submit_fps":     self.ai_submit_fps.fps(),
                "ai_dropped_frames": self.ai_dropped_frames.value(),
            },
            "ai_pipeline": {
                "yolo_detect_ms":    self.yolo_detect_ms.snapshot(),
                "tracking_ms":       self.tracking_ms.snapshot(),
                "face_detect_ms":    self.face_detect_ms.snapshot(),
                "face_verify_ms":    self.face_verify_ms.snapshot(),
                "plate_detect_ms":   self.plate_detect_ms.snapshot(),
                "ocr_ms":            self.ocr_ms.snapshot(),
                "analytics_ms":      self.analytics_ms.snapshot(),
                "event_engine_ms":   self.event_engine_ms.snapshot(),
                "pipeline_total_ms": self.pipeline_total_ms.snapshot(),
                "ai_processed_fps":  self.ai_processed_fps.fps(),
                "ai_total_frames":   self.ai_total_frames.value(),
                "ai_event_count":    self.ai_event_count.value(),
            },
            "quality_gates": {
                "face_checks":       self.face_check_count.value(),
                "face_quality_skip": self.face_skip_quality.value(),
                "ocr_checks":        self.ocr_check_count.value(),
                "ocr_quality_skip":  self.ocr_skip_quality.value(),
            },
        }
