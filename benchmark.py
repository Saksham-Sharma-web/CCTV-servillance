"""
IBVAP — Practical Benchmark & Profiler
=======================================
Run this script to measure REAL latency and resource usage across every
stage of the IBVAP processing pipeline.

Usage
-----
  # Benchmark AI pipeline only (no real cameras needed):
  python benchmark.py --mode ai

  # Benchmark a single real camera stream:
  python benchmark.py --mode stream --rtsp "rtsp://192.168.1.100:554/stream"

  # Benchmark multiple cameras (5 cameras from one RTSP source):
  python benchmark.py --mode multi --rtsp "rtsp://192.168.1.100:554/stream" --cameras 5

  # Simulate high camera count (no real cameras, uses synthetic frames):
  python benchmark.py --mode simulate --cameras 100

  # Full benchmark suite (runs all modes sequentially):
  python benchmark.py --mode full

  # Run for longer to get steady-state numbers:
  python benchmark.py --mode ai --duration 60

Output
------
  - Live rolling stats printed to terminal every 2 seconds
  - Final report saved to: benchmark_report_<timestamp>.json + benchmark_report_<timestamp>.md

Requirements
------------
  pip install psutil numpy opencv-python  (already in requirements.txt)
"""

import argparse
import datetime
import json
import os
import platform
import queue
import statistics
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import psutil

# ─── Try to import optional heavy deps, fail gracefully ───────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Add project root so we can import ibvap
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "ibvap_rust"))


# ══════════════════════════════════════════════════════════════════════════════
# ANSI colours for terminal output
# ══════════════════════════════════════════════════════════════════════════════

class C:
    RED    = "\033[91m"
    YEL    = "\033[93m"
    GRN    = "\033[92m"
    BLU    = "\033[94m"
    CYN    = "\033[96m"
    MAG    = "\033[95m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"

def _header(text: str):
    width = 72
    print(f"\n{C.BOLD}{C.BLU}{'═' * width}{C.RESET}")
    print(f"{C.BOLD}{C.BLU}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.BLU}{'═' * width}{C.RESET}")

def _section(text: str):
    print(f"\n{C.CYN}{C.BOLD}▶ {text}{C.RESET}")

def _ok(text: str):
    print(f"  {C.GRN}✓{C.RESET}  {text}")

def _warn(text: str):
    print(f"  {C.YEL}⚠{C.RESET}  {text}")

def _err(text: str):
    print(f"  {C.RED}✗{C.RESET}  {text}")

def _stat(label: str, value: str, unit: str = ""):
    print(f"  {C.DIM}{label:<40}{C.RESET} {C.BOLD}{value}{C.RESET} {C.DIM}{unit}{C.RESET}")


# ══════════════════════════════════════════════════════════════════════════════
# System info collector
# ══════════════════════════════════════════════════════════════════════════════

def collect_system_info() -> dict:
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_model": platform.processor() or "unknown",
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_freq_mhz": round(cpu_freq.max, 0) if cpu_freq else "unknown",
        "ram_total_gb": round(mem.total / 1024**3, 2),
        "ram_available_gb": round(mem.available / 1024**3, 2),
        "cuda_available": CUDA_AVAILABLE,
        "torch_version": torch.__version__ if TORCH_AVAILABLE else "not installed",
        "gpu_name": "unknown",
        "gpu_vram_mb": 0,
        "opencv_available": CV2_AVAILABLE,
        "opencv_version": cv2.__version__ if CV2_AVAILABLE else "not installed",
    }
    if CUDA_AVAILABLE:
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_vram_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2)
        except Exception:
            pass
    return info


# ══════════════════════════════════════════════════════════════════════════════
# Resource monitor (background thread)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ResourceSnapshot:
    ts: float
    cpu_pct: float
    ram_mb: float
    ram_pct: float
    gpu_pct: Optional[float] = None
    gpu_mem_mb: Optional[float] = None


class ResourceMonitor:
    def __init__(self, interval_s: float = 0.5):
        self._interval = interval_s
        self._samples: List[ResourceSnapshot] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="res-monitor")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def _loop(self):
        while not self._stop.is_set():
            try:
                cpu = self._proc.cpu_percent(interval=None)
                mem = self._proc.memory_info().rss / 1024**2
                mem_pct = self._proc.memory_percent()
                gpu_pct = gpu_mem = None
                if CUDA_AVAILABLE:
                    try:
                        gpu_mem = torch.cuda.memory_allocated(0) / 1024**2
                    except Exception:
                        pass
                snap = ResourceSnapshot(
                    ts=time.time(), cpu_pct=cpu, ram_mb=mem,
                    ram_pct=mem_pct, gpu_pct=gpu_pct, gpu_mem_mb=gpu_mem
                )
                with self._lock:
                    self._samples.append(snap)
            except Exception:
                pass
            time.sleep(self._interval)

    def summary(self) -> dict:
        with self._lock:
            snaps = list(self._samples)
        if not snaps:
            return {}
        cpus = [s.cpu_pct for s in snaps]
        rams = [s.ram_mb for s in snaps]
        return {
            "cpu_avg_pct": round(statistics.mean(cpus), 1),
            "cpu_max_pct": round(max(cpus), 1),
            "ram_avg_mb": round(statistics.mean(rams), 1),
            "ram_max_mb": round(max(rams), 1),
            "samples": len(snaps),
        }

    def current(self) -> Optional[ResourceSnapshot]:
        with self._lock:
            return self._samples[-1] if self._samples else None


# ══════════════════════════════════════════════════════════════════════════════
# Latency tracker — rolling window statistics
# ══════════════════════════════════════════════════════════════════════════════

class LatencyTracker:
    def __init__(self, window: int = 120):
        self._window = window
        self._samples: List[float] = []
        self._lock = threading.Lock()

    def record(self, ms: float):
        with self._lock:
            self._samples.append(ms)
            if len(self._samples) > self._window:
                self._samples.pop(0)

    def stats(self) -> dict:
        with self._lock:
            data = list(self._samples)
        if len(data) < 2:
            return {"n": len(data), "mean": 0, "median": 0, "p95": 0, "p99": 0, "min": 0, "max": 0}
        data_sorted = sorted(data)
        n = len(data_sorted)
        return {
            "n": n,
            "mean":   round(statistics.mean(data_sorted), 2),
            "median": round(statistics.median(data_sorted), 2),
            "stdev":  round(statistics.stdev(data_sorted), 2) if n > 1 else 0,
            "p95":    round(data_sorted[int(n * 0.95)], 2),
            "p99":    round(data_sorted[int(n * 0.99)], 2),
            "min":    round(data_sorted[0], 2),
            "max":    round(data_sorted[-1], 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Synthetic frame generator (for camera-less benchmarks)
# ══════════════════════════════════════════════════════════════════════════════

def make_synthetic_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """Generate a random BGR frame to simulate a real camera frame."""
    frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    # Add timestamp overlay so each frame is unique
    ts_text = f"BENCH {time.time():.3f}"
    cv2.putText(frame, ts_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    return frame


def frame_to_jpeg(frame: np.ndarray, quality: int = 60) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return bytes(buf) if ok else b""


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 1: AI Pipeline latency (no streaming, direct process_frame calls)
# ══════════════════════════════════════════════════════════════════════════════

def bench_ai_pipeline(duration_s: int = 30, width: int = 1280, height: int = 720) -> dict:
    _section(f"AI Pipeline Benchmark — {duration_s}s duration, {width}x{height} synthetic frames")

    results = {
        "benchmark": "ai_pipeline",
        "duration_s": duration_s,
        "frame_size": f"{width}x{height}",
        "stages": {}
    }

    # Try to import the IBVAP pipeline
    try:
        from ibvap.pipeline import IBVAPPipeline
        from ibvap.core.config import IBVAPConfig
        _ok("IBVAPPipeline imported successfully")
    except Exception as e:
        _err(f"Could not import IBVAPPipeline: {e}")
        results["error"] = str(e)
        return results

    # Init pipeline
    t0 = time.perf_counter()
    try:
        pipeline = IBVAPPipeline(config=IBVAPConfig(redis_enabled=False, db_enabled=False))
        init_ms = (time.perf_counter() - t0) * 1000
        _ok(f"Pipeline initialized in {init_ms:.0f} ms")
        results["pipeline_init_ms"] = round(init_ms, 1)
    except Exception as e:
        _err(f"Pipeline init failed: {e}")
        results["error"] = str(e)
        return results

    # Stage trackers
    t_detect   = LatencyTracker()
    t_total    = LatencyTracker()
    t_jpeg_enc = LatencyTracker()
    t_jpeg_dec = LatencyTracker()

    frame_count = 0
    event_count = 0
    error_count = 0
    rmon = ResourceMonitor()
    rmon.start()

    deadline = time.time() + duration_s
    _section("Running... (Ctrl+C to stop early)")

    last_print = time.time()

    while time.time() < deadline:
        # ── Generate synthetic frame ───────────────────────────────────────────
        raw_frame = make_synthetic_frame(width, height)

        # ── Stage A: JPEG encode (like live_streaming Thread 2) ───────────────
        t_enc_start = time.perf_counter()
        jpeg_bytes = frame_to_jpeg(raw_frame, quality=60)
        t_jpeg_enc.record((time.perf_counter() - t_enc_start) * 1000)

        # ── Stage B: JPEG decode (like Rust aggregator jpeg_to_pixel_buffer) ──
        t_dec_start = time.perf_counter()
        decoded = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        t_jpeg_dec.record((time.perf_counter() - t_dec_start) * 1000)

        # ── Stage C: Full AI pipeline (the real thing) ─────────────────────────
        t_ai_start = time.perf_counter()
        try:
            result = pipeline.process_frame(decoded, camera_id="bench-cam", timestamp=time.time())
            ai_ms = (time.perf_counter() - t_ai_start) * 1000
            t_total.record(ai_ms)
            event_count += len(result.events)
            frame_count += 1
        except Exception as ex:
            ai_ms = (time.perf_counter() - t_ai_start) * 1000
            error_count += 1
            if error_count <= 3:
                _warn(f"AI frame error: {ex}")

        # ── Print rolling stats every 2s ──────────────────────────────────────
        now = time.time()
        if now - last_print >= 2.0:
            res = rmon.current()
            ai_s = t_total.stats()
            enc_s = t_jpeg_enc.stats()
            dec_s = t_jpeg_dec.stats()
            elapsed = now - (deadline - duration_s)
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(
                f"  [{elapsed:5.1f}s] "
                f"frames={frame_count:4d}  fps={fps:4.1f}  "
                f"AI: mean={ai_s['mean']:6.1f}ms p95={ai_s['p95']:6.1f}ms  "
                f"JPEG-enc={enc_s['mean']:5.1f}ms  JPEG-dec={dec_s['mean']:5.1f}ms  "
                f"CPU={res.cpu_pct:.0f}%  RAM={res.ram_mb:.0f}MB"
                if res else ""
            )
            last_print = now

    rmon.stop()
    elapsed_total = duration_s

    results["frames_processed"] = frame_count
    results["events_detected"] = event_count
    results["errors"] = error_count
    results["effective_fps"] = round(frame_count / elapsed_total, 2)
    results["stages"]["jpeg_encode_ms"]  = t_jpeg_enc.stats()
    results["stages"]["jpeg_decode_ms"]  = t_jpeg_dec.stats()
    results["stages"]["ai_pipeline_ms"]  = t_total.stats()
    results["resources"] = rmon.summary()

    # Print final summary
    _section("AI Pipeline Results")
    _stat("Frames processed", str(frame_count))
    _stat("Effective AI throughput", f"{results['effective_fps']}", "fps")
    _stat("Events detected", str(event_count))

    ai_s = results["stages"]["ai_pipeline_ms"]
    _stat("AI latency — mean",   f"{ai_s['mean']}",   "ms")
    _stat("AI latency — median", f"{ai_s['median']}", "ms")
    _stat("AI latency — p95",    f"{ai_s['p95']}",    "ms")
    _stat("AI latency — p99",    f"{ai_s['p99']}",    "ms")
    _stat("AI latency — max",    f"{ai_s['max']}",    "ms")

    enc_s = results["stages"]["jpeg_encode_ms"]
    dec_s = results["stages"]["jpeg_decode_ms"]
    _stat("JPEG encode (live display path)", f"{enc_s['mean']}", "ms mean")
    _stat("JPEG decode (aggregator cost)",   f"{dec_s['mean']}", "ms mean")

    r = results["resources"]
    _stat("CPU average", f"{r['cpu_avg_pct']}", "%")
    _stat("CPU peak",    f"{r['cpu_max_pct']}", "%")
    _stat("RAM average", f"{r['ram_avg_mb']}", "MB")
    _stat("RAM peak",    f"{r['ram_max_mb']}", "MB")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 2: Single real camera stream
# ══════════════════════════════════════════════════════════════════════════════

def bench_single_stream(rtsp_url: str, duration_s: int = 30) -> dict:
    _section(f"Single Camera Stream Benchmark — {duration_s}s — {rtsp_url}")

    if not CV2_AVAILABLE:
        _err("OpenCV not available")
        return {"error": "opencv not available"}

    results = {
        "benchmark": "single_stream",
        "rtsp_url": rtsp_url,
        "duration_s": duration_s,
        "stages": {}
    }

    t_connect   = LatencyTracker()
    t_read      = LatencyTracker()
    t_jpeg_enc  = LatencyTracker()
    t_jpeg_dec  = LatencyTracker()
    t_e2e       = LatencyTracker()  # end-to-end: read → JPEG → decode

    frame_count = 0
    drop_count  = 0

    # Try to import live_streaming
    stream_obj = None
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "ibvap_rust"))
        from live_streaming import LiveCameraStream
        use_live_stream = True
        _ok("Using LiveCameraStream (3-thread architecture)")
    except ImportError:
        use_live_stream = False
        _warn("LiveCameraStream not importable — using direct cv2.VideoCapture")

    rmon = ResourceMonitor()
    rmon.start()
    deadline = time.time() + duration_s
    last_print = time.time()

    if use_live_stream:
        # ── Use the real LiveCameraStream class ────────────────────────────────
        try:
            stream = LiveCameraStream("bench-cam-0", rtsp_url)
        except Exception as e:
            _err(f"Could not start LiveCameraStream: {e}")
            rmon.stop()
            results["error"] = str(e)
            return results

        time.sleep(3)  # Let threads connect

        prev_ts = time.perf_counter()
        while time.time() < deadline:
            t_read_start = time.perf_counter()
            frame_data = stream.next_frame()

            if frame_data is None:
                drop_count += 1
                time.sleep(0.005)
                continue

            jpg_bytes, w, h, events_json = frame_data
            now_ts = time.perf_counter()

            t_read.record((now_ts - t_read_start) * 1000)

            # JPEG decode (simulates Rust aggregator)
            t_dec_s = time.perf_counter()
            decoded = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            t_jpeg_dec.record((time.perf_counter() - t_dec_s) * 1000)

            inter_frame_ms = (now_ts - prev_ts) * 1000
            t_e2e.record(inter_frame_ms)
            prev_ts = now_ts
            frame_count += 1

            if time.time() - last_print >= 2.0:
                res = rmon.current()
                e2e_s = t_e2e.stats()
                fps = frame_count / max(1, (time.time() - (deadline - duration_s)))
                print(
                    f"  [{time.time() - (deadline - duration_s):5.1f}s] "
                    f"frames={frame_count:4d}  fps={fps:4.1f}  "
                    f"inter-frame={e2e_s['mean']:6.1f}ms  "
                    f"JPEG-dec={t_jpeg_dec.stats()['mean']:5.1f}ms  "
                    f"drops={drop_count}  "
                    f"{'CPU=' + str(round(res.cpu_pct)) + '%  RAM=' + str(round(res.ram_mb)) + 'MB' if res else ''}"
                )
                last_print = time.time()

        stream.release()

    else:
        # ── Fallback: direct cv2.VideoCapture ─────────────────────────────────
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"
        t_conn_start = time.perf_counter()
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            _err(f"Could not open RTSP stream: {rtsp_url}")
            rmon.stop()
            results["error"] = "RTSP connection failed"
            return results

        conn_ms = (time.perf_counter() - t_conn_start) * 1000
        _ok(f"Connected in {conn_ms:.0f} ms  |  {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {cap.get(cv2.CAP_PROP_FPS):.1f} fps")
        results["rtsp_connect_ms"] = round(conn_ms, 1)
        results["resolution"] = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
        results["camera_fps"] = cap.get(cv2.CAP_PROP_FPS)

        prev_ts = time.perf_counter()
        while time.time() < deadline:
            t_read_start = time.perf_counter()
            ok, frame = cap.read()
            read_ms = (time.perf_counter() - t_read_start) * 1000

            if not ok or frame is None:
                drop_count += 1
                continue

            t_read.record(read_ms)

            # JPEG encode (like Thread 2)
            t_enc_s = time.perf_counter()
            jpg_bytes = frame_to_jpeg(frame)
            t_jpeg_enc.record((time.perf_counter() - t_enc_s) * 1000)

            # JPEG decode (like Rust aggregator)
            t_dec_s = time.perf_counter()
            cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
            t_jpeg_dec.record((time.perf_counter() - t_dec_s) * 1000)

            now_ts = time.perf_counter()
            inter_ms = (now_ts - prev_ts) * 1000
            t_e2e.record(inter_ms)
            prev_ts = now_ts
            frame_count += 1

            if time.time() - last_print >= 2.0:
                res = rmon.current()
                e2e_s = t_e2e.stats()
                fps = frame_count / max(1, (time.time() - (deadline - duration_s)))
                print(
                    f"  [{time.time() - (deadline - duration_s):5.1f}s] "
                    f"frames={frame_count:4d}  fps={fps:4.1f}  "
                    f"inter-frame={e2e_s['mean']:6.1f}ms  "
                    f"drops={drop_count}  "
                    f"{'CPU=' + str(round(res.cpu_pct)) + '%  RAM=' + str(round(res.ram_mb)) + 'MB' if res else ''}"
                )
                last_print = time.time()

        cap.release()

    rmon.stop()

    elapsed = duration_s
    results["frames_received"] = frame_count
    results["frames_dropped"] = drop_count
    results["effective_fps"] = round(frame_count / elapsed, 2)
    results["drop_rate_pct"] = round(drop_count / max(1, frame_count + drop_count) * 100, 2)
    results["stages"]["inter_frame_ms"]  = t_e2e.stats()
    results["stages"]["jpeg_decode_ms"]  = t_jpeg_dec.stats()
    if not use_live_stream:
        results["stages"]["frame_read_ms"]   = t_read.stats()
        results["stages"]["jpeg_encode_ms"]  = t_jpeg_enc.stats()
    results["resources"] = rmon.summary()

    _section("Single Stream Results")
    _stat("Frames received", str(frame_count))
    _stat("Frames dropped", str(drop_count))
    _stat("Effective FPS", f"{results['effective_fps']}", "fps")
    _stat("Drop rate", f"{results['drop_rate_pct']}", "%")
    e2e = results["stages"]["inter_frame_ms"]
    _stat("Inter-frame interval — mean",   f"{e2e['mean']}",   "ms")
    _stat("Inter-frame interval — p95",    f"{e2e['p95']}",    "ms")
    dec = results["stages"]["jpeg_decode_ms"]
    _stat("JPEG decode (aggregator cost)", f"{dec['mean']}", "ms mean")
    r = results["resources"]
    _stat("CPU avg/peak", f"{r['cpu_avg_pct']}% / {r['cpu_max_pct']}%")
    _stat("RAM avg/peak", f"{r['ram_avg_mb']} / {r['ram_max_mb']} MB")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 3: Multi-camera simulation (synthetic frames)
# ══════════════════════════════════════════════════════════════════════════════

def bench_multi_camera_simulate(n_cameras: int, duration_s: int = 30,
                                width: int = 1280, height: int = 720) -> dict:
    _section(f"Multi-Camera Simulation — {n_cameras} cameras, {duration_s}s, {width}x{height}")

    results = {
        "benchmark": "multi_camera_simulate",
        "n_cameras": n_cameras,
        "duration_s": duration_s,
        "frame_size": f"{width}x{height}",
        "per_camera": {},
        "aggregate": {}
    }

    if not CV2_AVAILABLE:
        _err("OpenCV not available")
        return {"error": "opencv not available"}

    # Per-camera counters
    per_cam_frames  = {i: 0 for i in range(n_cameras)}
    per_cam_drops   = {i: 0 for i in range(n_cameras)}
    per_cam_lat     = {i: LatencyTracker() for i in range(n_cameras)}
    per_cam_enc     = {i: LatencyTracker() for i in range(n_cameras)}

    # Simulate the AI queue (one global queue, maxsize=4)
    ai_queue: queue.Queue = queue.Queue(maxsize=4)
    ai_latencies = LatencyTracker()
    ai_frames_processed = [0]
    ai_frames_dropped   = [0]
    stop_event = threading.Event()

    # AI worker thread (mirrors _GlobalAIWorker._loop)
    def ai_worker_thread():
        try:
            from ibvap.pipeline import IBVAPPipeline
            from ibvap.core.config import IBVAPConfig
            pipeline = IBVAPPipeline(config=IBVAPConfig(redis_enabled=False, db_enabled=False))
            _ok("AI worker: IBVAPPipeline ready")
        except Exception as e:
            _warn(f"AI worker: pipeline unavailable ({e}), timing inference only with sleep")
            pipeline = None

        while not stop_event.is_set():
            try:
                cam_id, frame, submit_ts = ai_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            t_start = time.perf_counter()
            try:
                if pipeline:
                    pipeline.process_frame(frame, camera_id=cam_id, timestamp=time.time())
                else:
                    time.sleep(0.090)  # simulate ~90ms YOLO on CPU
            except Exception:
                pass
            ai_ms = (time.perf_counter() - t_start) * 1000

            # Latency from submit to completion
            total_ai_latency = (time.perf_counter() - submit_ts) * 1000
            ai_latencies.record(total_ai_latency)
            ai_frames_processed[0] += 1

    ai_thread = threading.Thread(target=ai_worker_thread, daemon=True, name="ai-worker")
    ai_thread.start()

    # Per-camera encoder threads (mirrors Thread 2: _encoder_thread)
    cam_threads = []

    def camera_thread(cam_idx: int):
        AI_INTERVAL = 0.10  # match live_streaming.py
        DISPLAY_INTERVAL = 1.0 / 30.0  # 30 fps cap
        last_enc = 0.0
        last_ai  = 0.0

        while not stop_event.is_set():
            now = time.monotonic()
            if now - last_enc < DISPLAY_INTERVAL:
                time.sleep(DISPLAY_INTERVAL - (now - last_enc))
                continue
            last_enc = time.monotonic()

            raw = make_synthetic_frame(width, height)

            # JPEG encode
            t_enc = time.perf_counter()
            jpg = frame_to_jpeg(raw, quality=60)
            enc_ms = (time.perf_counter() - t_enc) * 1000
            per_cam_enc[cam_idx].record(enc_ms)

            # Simulate display ring buffer push — measure latency to next display
            t_push = time.perf_counter()

            # JPEG decode (simulates Rust aggregator)
            decoded = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            lat_ms = (time.perf_counter() - t_push) * 1000
            per_cam_lat[cam_idx].record(lat_ms)
            per_cam_frames[cam_idx] += 1

            # AI submission (throttled to AI_INTERVAL)
            now2 = time.monotonic()
            if now2 - last_ai >= AI_INTERVAL:
                try:
                    ai_queue.put_nowait((f"bench-cam-{cam_idx}", raw.copy(), time.perf_counter()))
                    last_ai = now2
                except queue.Full:
                    ai_frames_dropped[0] += 1

    for i in range(n_cameras):
        t = threading.Thread(target=camera_thread, args=(i,), daemon=True,
                             name=f"cam-{i}")
        cam_threads.append(t)

    rmon = ResourceMonitor()
    rmon.start()
    deadline = time.time() + duration_s
    last_print = time.time()

    for t in cam_threads:
        t.start()

    _section("Running multi-camera simulation...")
    while time.time() < deadline:
        time.sleep(0.5)
        if time.time() - last_print >= 2.0:
            res = rmon.current()
            total_frames = sum(per_cam_frames.values())
            elapsed = time.time() - (deadline - duration_s)
            fps_total = total_frames / max(1, elapsed)
            ai_s = ai_latencies.stats()
            print(
                f"  [{elapsed:5.1f}s] "
                f"total_frames={total_frames:6d}  total_fps={fps_total:6.1f}  "
                f"ai_processed={ai_frames_processed[0]}  ai_dropped={ai_frames_dropped[0]}  "
                f"ai_lat_mean={ai_s['mean']:6.1f}ms  "
                f"{'CPU=' + str(round(res.cpu_pct)) + '%  RAM=' + str(round(res.ram_mb)) + 'MB' if res else ''}"
            )
            last_print = time.time()

    stop_event.set()
    for t in cam_threads:
        t.join(timeout=3)
    rmon.stop()

    # Collect results
    total_frames = sum(per_cam_frames.values())
    elapsed = duration_s
    fps_per_cam = {
        str(i): round(per_cam_frames[i] / elapsed, 2) for i in range(n_cameras)
    }

    results["aggregate"]["total_frames"] = total_frames
    results["aggregate"]["total_fps"] = round(total_frames / elapsed, 2)
    results["aggregate"]["ai_frames_processed"] = ai_frames_processed[0]
    results["aggregate"]["ai_frames_dropped"] = ai_frames_dropped[0]
    results["aggregate"]["ai_drop_rate_pct"] = round(
        ai_frames_dropped[0] / max(1, ai_frames_processed[0] + ai_frames_dropped[0]) * 100, 2
    )
    results["aggregate"]["ai_latency_ms"] = ai_latencies.stats()

    # Average across cameras
    all_enc = []
    all_lat = []
    for i in range(n_cameras):
        s = per_cam_enc[i].stats()
        if s["n"] > 0:
            all_enc.append(s["mean"])
        s2 = per_cam_lat[i].stats()
        if s2["n"] > 0:
            all_lat.append(s2["mean"])

    results["aggregate"]["jpeg_encode_avg_ms"] = round(statistics.mean(all_enc), 2) if all_enc else 0
    results["aggregate"]["jpeg_decode_avg_ms"] = round(statistics.mean(all_lat), 2) if all_lat else 0
    results["resources"] = rmon.summary()

    _section(f"Multi-Camera ({n_cameras}x) Results")
    _stat("Total frames produced", str(total_frames))
    _stat("Total throughput", f"{results['aggregate']['total_fps']}", "fps")
    _stat("Per-camera average FPS",
          f"{round(results['aggregate']['total_fps'] / max(1, n_cameras), 2)}")
    _stat("AI frames processed", str(ai_frames_processed[0]))
    _stat("AI frames dropped", str(ai_frames_dropped[0]))
    _stat("AI drop rate", f"{results['aggregate']['ai_drop_rate_pct']}", "%")
    ai_s = results["aggregate"]["ai_latency_ms"]
    if ai_s["n"] > 0:
        _stat("AI event latency — mean", f"{ai_s['mean']}", "ms (submit → complete)")
        _stat("AI event latency — p95",  f"{ai_s['p95']}", "ms")
    _stat("JPEG encode avg", f"{results['aggregate']['jpeg_encode_avg_ms']}", "ms")
    _stat("JPEG decode avg", f"{results['aggregate']['jpeg_decode_avg_ms']}", "ms")
    r = results["resources"]
    _stat("CPU avg/peak", f"{r['cpu_avg_pct']}% / {r['cpu_max_pct']}%")
    _stat("RAM avg/peak", f"{r['ram_avg_mb']} / {r['ram_max_mb']} MB")

    # Scalability projection
    _section("Scalability Projection (based on observed resource usage)")
    cpu_per_cam = r["cpu_avg_pct"] / max(1, n_cameras)
    ram_per_cam = r["ram_avg_mb"] / max(1, n_cameras)
    total_logical = psutil.cpu_count(logical=True) or 8
    total_ram = psutil.virtual_memory().total / 1024**2

    safe_cpu_cams = int((total_logical * 100 * 0.80) / max(0.1, cpu_per_cam))
    safe_ram_cams = int(total_ram * 0.80 / max(0.1, ram_per_cam))
    safe_max = min(safe_cpu_cams, safe_ram_cams)

    _stat(f"CPU per camera (avg)", f"{cpu_per_cam:.2f}", "% of total")
    _stat(f"RAM per camera (avg)", f"{ram_per_cam:.2f}", "MB")
    _stat(f"Projected max cameras (CPU-limited, 80%)", str(safe_cpu_cams))
    _stat(f"Projected max cameras (RAM-limited, 80%)", str(safe_ram_cams))
    _stat(f"Projected safe maximum", f"{C.BOLD}{safe_max}{C.RESET}", "cameras")

    results["scalability"] = {
        "cpu_pct_per_camera": round(cpu_per_cam, 2),
        "ram_mb_per_camera": round(ram_per_cam, 2),
        "projected_max_cpu": safe_cpu_cams,
        "projected_max_ram": safe_ram_cams,
        "projected_safe_max": safe_max,
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARK 4: Multi-camera real streams (multiple caps from one RTSP)
# ══════════════════════════════════════════════════════════════════════════════

def bench_multi_stream(rtsp_url: str, n_cameras: int, duration_s: int = 30) -> dict:
    _section(f"Multi-Stream Benchmark — {n_cameras} streams from {rtsp_url}, {duration_s}s")

    if not CV2_AVAILABLE:
        _err("OpenCV not available")
        return {"error": "opencv not available"}

    results = {
        "benchmark": "multi_stream",
        "rtsp_url": rtsp_url,
        "n_cameras": n_cameras,
        "duration_s": duration_s,
    }

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000"

    per_cam_frames  = {i: 0 for i in range(n_cameras)}
    per_cam_drops   = {i: 0 for i in range(n_cameras)}
    per_cam_lat     = {i: LatencyTracker() for i in range(n_cameras)}
    stop_event = threading.Event()
    connect_ms_list = []

    def stream_thread(idx: int):
        t_conn = time.perf_counter()
        try:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            _err(f"  Cam {idx} connect failed: {e}")
            return

        if not cap.isOpened():
            per_cam_drops[idx] += 1
            return

        conn_ms = (time.perf_counter() - t_conn) * 1000
        connect_ms_list.append(conn_ms)

        prev_ts = time.perf_counter()
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                per_cam_drops[idx] += 1
                continue
            now_ts = time.perf_counter()
            per_cam_lat[idx].record((now_ts - prev_ts) * 1000)
            prev_ts = now_ts
            per_cam_frames[idx] += 1

        cap.release()

    threads = []
    _ok(f"Connecting {n_cameras} capture handles...")
    for i in range(n_cameras):
        t = threading.Thread(target=stream_thread, args=(i,), daemon=True,
                             name=f"stream-{i}")
        threads.append(t)
        t.start()
        time.sleep(0.1)  # stagger connections slightly

    rmon = ResourceMonitor()
    rmon.start()
    deadline = time.time() + duration_s
    last_print = time.time()

    while time.time() < deadline:
        time.sleep(0.5)
        if time.time() - last_print >= 2.0:
            res = rmon.current()
            total = sum(per_cam_frames.values())
            elapsed = time.time() - (deadline - duration_s)
            fps = total / max(1, elapsed)
            print(
                f"  [{elapsed:5.1f}s] frames={total:6d}  total_fps={fps:6.1f}  "
                f"{'CPU=' + str(round(res.cpu_pct)) + '%  RAM=' + str(round(res.ram_mb)) + 'MB' if res else ''}"
            )
            last_print = time.time()

    stop_event.set()
    for t in threads:
        t.join(timeout=5)
    rmon.stop()

    elapsed = duration_s
    total_frames = sum(per_cam_frames.values())
    total_drops = sum(per_cam_drops.values())

    results["total_frames"] = total_frames
    results["total_drops"] = total_drops
    results["total_fps"] = round(total_frames / elapsed, 2)
    results["per_camera_avg_fps"] = round(total_frames / max(1, n_cameras) / elapsed, 2)
    results["rtsp_connect_avg_ms"] = round(statistics.mean(connect_ms_list), 1) if connect_ms_list else 0

    all_lats = []
    for i in range(n_cameras):
        s = per_cam_lat[i].stats()
        if s["n"] > 0:
            all_lats.append(s["mean"])

    results["inter_frame_avg_ms"] = round(statistics.mean(all_lats), 2) if all_lats else 0
    results["resources"] = rmon.summary()

    _section(f"Multi-Stream ({n_cameras}x) Results")
    _stat("Total frames", str(total_frames))
    _stat("Total FPS", f"{results['total_fps']}")
    _stat("Per-camera avg FPS", f"{results['per_camera_avg_fps']}")
    _stat("RTSP connect avg", f"{results['rtsp_connect_avg_ms']}", "ms")
    _stat("Inter-frame avg", f"{results['inter_frame_avg_ms']}", "ms")
    r = results["resources"]
    _stat("CPU avg/peak", f"{r['cpu_avg_pct']}% / {r['cpu_max_pct']}%")
    _stat("RAM avg/peak", f"{r['ram_avg_mb']} / {r['ram_max_mb']} MB")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Report writer
# ══════════════════════════════════════════════════════════════════════════════

def write_report(all_results: dict, sys_info: dict, output_dir: str = "."):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # JSON report
    json_path = os.path.join(output_dir, f"benchmark_report_{ts}.json")
    with open(json_path, "w") as f:
        json.dump({"system": sys_info, "results": all_results, "timestamp": ts}, f, indent=2)
    _ok(f"JSON report saved: {json_path}")

    # Markdown report
    md_path = os.path.join(output_dir, f"benchmark_report_{ts}.md")
    lines = [
        f"# IBVAP Benchmark Report",
        f"",
        f"**Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## System Information",
        f"",
        f"| Property | Value |",
        f"|---|---|",
    ]
    for k, v in sys_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    for bench_name, result in all_results.items():
        lines.append(f"## {bench_name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(result, indent=2))
        lines.append("```")
        lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    _ok(f"Markdown report saved: {md_path}")

    return json_path, md_path


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="IBVAP Practical Benchmark & Profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark.py --mode ai
  python benchmark.py --mode stream --rtsp rtsp://192.168.1.100:554/stream
  python benchmark.py --mode multi --rtsp rtsp://192.168.1.100:554/stream --cameras 5
  python benchmark.py --mode simulate --cameras 50
  python benchmark.py --mode simulate --cameras 100 --duration 60
  python benchmark.py --mode full --rtsp rtsp://192.168.1.100:554/stream
        """
    )
    parser.add_argument("--mode", choices=["ai", "stream", "multi", "simulate", "full"],
                        default="ai", help="Benchmark mode (default: ai)")
    parser.add_argument("--rtsp",      default="",    help="RTSP URL for real camera tests")
    parser.add_argument("--cameras",   type=int, default=5,  help="Number of cameras (default: 5)")
    parser.add_argument("--duration",  type=int, default=30, help="Duration in seconds (default: 30)")
    parser.add_argument("--width",     type=int, default=1280, help="Synthetic frame width")
    parser.add_argument("--height",    type=int, default=720,  help="Synthetic frame height")
    parser.add_argument("--output",    default=".",  help="Output directory for reports")
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    _header("IBVAP Practical Benchmark & Profiler")

    sys_info = collect_system_info()

    _section("System Information")
    for k, v in sys_info.items():
        _stat(k, str(v))

    if CUDA_AVAILABLE:
        _ok(f"CUDA GPU detected: {sys_info['gpu_name']} ({sys_info['gpu_vram_mb']} MB VRAM)")
    else:
        _warn("No CUDA GPU detected — AI will run on CPU (significantly slower)")

    if not CV2_AVAILABLE:
        _warn("OpenCV (cv2) not found — some benchmarks will be skipped")

    all_results = {}

    try:
        # ── Run requested benchmarks ──────────────────────────────────────────
        if args.mode in ("ai", "full"):
            all_results["ai_pipeline"] = bench_ai_pipeline(
                duration_s=args.duration,
                width=args.width,
                height=args.height
            )

        if args.mode in ("stream", "full") and args.rtsp:
            all_results["single_stream"] = bench_single_stream(
                rtsp_url=args.rtsp,
                duration_s=args.duration
            )
        elif args.mode == "stream" and not args.rtsp:
            _err("--rtsp URL required for stream mode. Using simulate instead.")
            all_results["single_stream_simulated"] = bench_multi_camera_simulate(
                n_cameras=1, duration_s=args.duration,
                width=args.width, height=args.height
            )

        if args.mode == "multi":
            if args.rtsp:
                all_results["multi_stream"] = bench_multi_stream(
                    rtsp_url=args.rtsp,
                    n_cameras=args.cameras,
                    duration_s=args.duration
                )
            else:
                _warn("No --rtsp provided, falling back to simulation for multi mode")
                all_results["multi_simulate"] = bench_multi_camera_simulate(
                    n_cameras=args.cameras,
                    duration_s=args.duration,
                    width=args.width,
                    height=args.height
                )

        if args.mode in ("simulate", "full"):
            # Run a few camera counts to build a scaling curve
            cam_counts = [1, 5, args.cameras] if args.mode == "full" else [args.cameras]
            # De-dup
            cam_counts = sorted(set(cam_counts))
            for n in cam_counts:
                key = f"simulate_{n}cam"
                all_results[key] = bench_multi_camera_simulate(
                    n_cameras=n,
                    duration_s=max(15, args.duration // len(cam_counts)),
                    width=args.width,
                    height=args.height
                )

        if args.mode == "full" and args.rtsp:
            all_results["multi_stream_5"] = bench_multi_stream(
                rtsp_url=args.rtsp,
                n_cameras=5,
                duration_s=args.duration
            )

    except KeyboardInterrupt:
        _warn("\nBenchmark interrupted by user — saving partial results...")

    # ── Write report ──────────────────────────────────────────────────────────
    _header("Writing Reports")
    os.makedirs(args.output, exist_ok=True)
    json_path, md_path = write_report(all_results, sys_info, args.output)

    _header("Benchmark Complete")
    print(f"\n  {C.BOLD}JSON report:{C.RESET} {json_path}")
    print(f"  {C.BOLD}MD report: {C.RESET} {md_path}")
    print()


if __name__ == "__main__":
    main()
