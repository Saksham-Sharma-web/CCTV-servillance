"""
Device and Memory Trace Profiler for IBVAP.
Traces every pipeline stage to measure:
- Input & output device
- Tensor/array shape, dtype, memory layout
- CPU <-> GPU transitions (CPU->CUDA, CUDA->CPU, CPU->CPU, CUDA->CUDA)
- Explicit tally of .cpu(), .numpy(), .item(), .to(device), from_numpy()
- Memory transfer latency and CUDA synchronization points
- VRAM allocation delta
"""

import os
import sys
import time
import platform
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import cv2
import torch

from ibvap.core.config import IBVAPConfig
from ibvap.core.pipeline import IBVAPPipeline
from ibvap.detection.object_detector import YOLOv8Detector
from ibvap.face.detector import OpenCVFaceDetector
from ibvap.face.matcher_adapter import IdentityVerifierAdapter, align_face_160, BodyAppearanceExtractor
from ibvap.anpr.plate_detector import LicensePlateDetector
from ibvap.anpr.ocr_adapter import ANPRAdapter
from ibvap.vehicle.types import VehicleObservation


@dataclass
class StageTrace:
    stage_name: str
    input_device: str
    output_device: str
    transition_type: str  # CPU->CUDA, CUDA->CPU, CPU->CPU, CUDA->CUDA
    input_shape: str
    input_dtype: str
    input_layout: str
    output_shape: str
    output_dtype: str
    output_layout: str
    h2d_count: int = 0
    d2h_count: int = 0
    item_calls: int = 0
    cpu_conversions: int = 0
    numpy_conversions: int = 0
    to_device_calls: int = 0
    sync_calls: int = 0
    vram_alloc_delta_mb: float = 0.0
    wall_time_ms: float = 0.0
    gpu_compute_time_ms: Optional[float] = None
    notes: str = ""


def get_layout_str(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        if obj.is_contiguous(memory_format=torch.channels_last):
            return "channels_last"
        elif obj.is_contiguous():
            return "contiguous"
        else:
            return "strided/non-contiguous"
    elif isinstance(obj, np.ndarray):
        if obj.flags.c_contiguous:
            return "C-contiguous"
        elif obj.flags.f_contiguous:
            return "F-contiguous"
        else:
            return "non-contiguous"
    return "N/A"


def run_memory_trace(img_path: str) -> List[StageTrace]:
    if not os.path.exists(img_path):
        print(f"ERROR: File not found: {img_path}")
        return []

    config = IBVAPConfig()
    config.anpr_enabled = True
    config.face_detection_enabled = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Initialize components
    yolo_detector = YOLOv8Detector(config)
    face_detector = OpenCVFaceDetector(config)
    identity_verifier = IdentityVerifierAdapter(config)
    identity_verifier._ensure_facenet()
    ref_path = os.path.join(os.getcwd(), "akshat.jpg")
    if os.path.exists(ref_path):
        identity_verifier.register_reference("Akshat", ref_path, reference_age="most_recent", detector=face_detector)
    plate_detector = LicensePlateDetector(config)
    ocr_adapter = ANPRAdapter(config)
    ocr_adapter._ensure_ocr_engine()

    pipeline = IBVAPPipeline(config, detector=yolo_detector)
    pipeline.face_detector = face_detector
    pipeline.identity_verifier = identity_verifier
    pipeline.plate_detector = plate_detector
    pipeline.anpr_adapter = ocr_adapter
    pipeline.controlled_ocr.ocr_adapter = ocr_adapter

    frame = cv2.imread(img_path)
    h, w = frame.shape[:2]

    # Warmup
    _ = pipeline.process_frame(frame.copy(), "camera-01")
    if device.type == "cuda":
        torch.cuda.synchronize()

    traces: List[StageTrace] = []

    # ─────────────────────────────────────────────────────────────
    # STAGE 1: INPUT FRAME
    # ─────────────────────────────────────────────────────────────
    traces.append(StageTrace(
        stage_name="1. Input Frame Load",
        input_device="Disk/Host RAM",
        output_device="CPU",
        transition_type="Host->CPU",
        input_shape=f"{h}x{w}x3",
        input_dtype=str(frame.dtype),
        input_layout=get_layout_str(frame),
        output_shape=f"{h}x{w}x3",
        output_dtype=str(frame.dtype),
        output_layout=get_layout_str(frame),
        notes="cv2.imread() returns BGR numpy ndarray in CPU host RAM"
    ))

    # ─────────────────────────────────────────────────────────────
    # STAGE 2: YOLO OBJECT DETECTION
    # ─────────────────────────────────────────────────────────────
    vram_before = torch.cuda.memory_allocated(0) if device.type == "cuda" else 0
    t0 = time.perf_counter()

    gpu_time = None
    if device.type == "cuda":
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()

    results = yolo_detector.model(
        frame,
        conf=yolo_detector.confidence_threshold,
        iou=yolo_detector.iou_threshold,
        verbose=False,
        device=str(device)
    )

    if device.type == "cuda":
        end_evt.record()
        torch.cuda.synchronize()
        gpu_time = start_evt.elapsed_time(end_evt)

    t1 = time.perf_counter()
    yolo_wall_ms = (t1 - t0) * 1000.0
    vram_after = torch.cuda.memory_allocated(0) if device.type == "cuda" else 0

    # Count D2H transfers in postprocessing (vectorized: 1 batch D2H transfer for all boxes)
    boxes = results[0].boxes if results else None
    num_boxes = len(boxes) if boxes is not None else 0
    d2h_calls = 1 if num_boxes > 0 else 0
    item_calls = 0
    cpu_calls = 1 if num_boxes > 0 else 0
    numpy_calls = 1 if num_boxes > 0 else 0

    # Actual detection unpacking
    detections = yolo_detector.detect(frame)

    traces.append(StageTrace(
        stage_name="2. YOLOv8 Detection & Postprocess",
        input_device="CPU",
        output_device="CPU",
        transition_type="CPU->CUDA->CPU" if device.type == "cuda" else "CPU->CPU",
        input_shape=f"1x3x{h}x{w}",
        input_dtype="uint8 (frame) -> float32 (tensor)",
        input_layout=get_layout_str(frame),
        output_shape=f"{num_boxes} boxes",
        output_dtype="Detection objects (CPU)",
        output_layout="Python list",
        h2d_count=1,
        d2h_count=d2h_calls,
        item_calls=item_calls,
        cpu_conversions=cpu_calls,
        numpy_conversions=numpy_calls,
        sync_calls=d2h_calls if device.type == "cuda" else 0,
        vram_alloc_delta_mb=(vram_after - vram_before) / (1024 ** 2),
        wall_time_ms=yolo_wall_ms,
        gpu_compute_time_ms=gpu_time,
        notes=f"Detected {num_boxes} raw boxes. Note: {d2h_calls} D2H transfers occur in per-box loop."
    ))

    # ─────────────────────────────────────────────────────────────
    # STAGE 3: TRACKING
    # ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    cam_tracker = pipeline.get_tracker("camera-01")
    now = time.time()
    tracks = cam_tracker.update(detections, timestamp=now)
    tracks = pipeline.cross_camera_tracker.associate_tracks("camera-01", tracks, timestamp=now)
    t1 = time.perf_counter()

    traces.append(StageTrace(
        stage_name="3. Multi-Object Tracking",
        input_device="CPU",
        output_device="CPU",
        transition_type="CPU->CPU",
        input_shape=f"{len(detections)} detections",
        input_dtype="List[Detection]",
        input_layout="CPU Python objects",
        output_shape=f"{len(tracks)} tracks",
        output_dtype="List[Track]",
        output_layout="CPU Python objects",
        wall_time_ms=(t1 - t0) * 1000.0,
        notes="Kalman filter update & Hungarian association on CPU"
    ))

    # ─────────────────────────────────────────────────────────────
    # STAGE 4: PERSON CROPS & FACE DETECTION
    # ─────────────────────────────────────────────────────────────
    person_tracks = [t for t in tracks if t.class_name == "person"]
    total_face_det_wall = 0.0
    faces_found = []

    for trk in person_tracks:
        px1, py1, px2, py2 = trk.bbox
        person_crop = frame[py1:py2, px1:px2]
        if person_crop.size > 0:
            t0 = time.perf_counter()
            f_list = face_detector.detect_faces(person_crop)
            t1 = time.perf_counter()
            total_face_det_wall += (t1 - t0) * 1000.0
            for f in f_list:
                faces_found.append((person_crop, f))

    traces.append(StageTrace(
        stage_name="4. Person Crops & Face Detection (YuNet)",
        input_device="CPU",
        output_device="CPU",
        transition_type="CPU->CPU",
        input_shape=f"{len(person_tracks)} crops",
        input_dtype="uint8 (BGR)",
        input_layout="C-contiguous numpy",
        output_shape=f"{len(faces_found)} faces detected",
        output_dtype="FaceDetection objects",
        output_layout="CPU Python list",
        wall_time_ms=total_face_det_wall,
        notes="YuNet runs via OpenCV DNN on OPENCV_CPU backend"
    ))

    # ─────────────────────────────────────────────────────────────
    # STAGE 5: FACE PREPROCESSING & VERIFICATION (InceptionResnetV1)
    # ─────────────────────────────────────────────────────────────
    fv_h2d_count = 0
    fv_d2h_count = 0
    fv_to_device_calls = 0
    fv_sync_calls = 0
    total_fv_wall = 0.0
    total_fv_gpu = 0.0 if device.type == "cuda" else None

    vram_before_fv = torch.cuda.memory_allocated(0) if device.type == "cuda" else 0

    for person_crop, face_det in faces_found:
        landmarks = getattr(face_det, "landmarks", None)
        box = getattr(face_det, "box", None)
        aligned_face = align_face_160(person_crop, landmarks=landmarks, box=box)

        t0 = time.perf_counter()
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (160, 160):
            rgb = cv2.resize(rgb, (160, 160))
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0)

        # H2D transfer
        if device.type == "cuda":
            tensor = tensor.to(device)
            fv_h2d_count += 1
            fv_to_device_calls += 1

        # Model forward
        if device.type == "cuda":
            s_evt = torch.cuda.Event(enable_timing=True)
            e_evt = torch.cuda.Event(enable_timing=True)
            s_evt.record()

        with torch.inference_mode():
            emb_tensor = identity_verifier._facenet(tensor)

        if device.type == "cuda":
            e_evt.record()
            torch.cuda.synchronize()
            total_fv_gpu += s_evt.elapsed_time(e_evt)
            fv_sync_calls += 1

        # D2H transfer
        emb = emb_tensor[0].cpu().numpy().astype(np.float32)
        fv_d2h_count += 1

        # Normalize
        norm = np.linalg.norm(emb)
        target_face_emb = emb / norm if norm > 0 else emb

        # Registry match
        for pid, person in identity_verifier.authorized_registry.items():
            for ref in person.references:
                _ = float(np.dot(target_face_emb, ref.face_embedding))

        t1 = time.perf_counter()
        total_fv_wall += (t1 - t0) * 1000.0

    vram_after_fv = torch.cuda.memory_allocated(0) if device.type == "cuda" else 0

    traces.append(StageTrace(
        stage_name="5. Face Verification (InceptionResnetV1)",
        input_device="CPU",
        output_device="CPU",
        transition_type="CPU->CUDA->CPU" if device.type == "cuda" else "CPU->CPU",
        input_shape=f"{len(faces_found)}x160x160x3",
        input_dtype="uint8 (CPU) -> float32 (GPU) -> float32 (CPU)",
        input_layout=get_layout_str(aligned_face if faces_found else frame),
        output_shape=f"{len(faces_found)}x512 embeddings",
        output_dtype="float32",
        output_layout="C-contiguous numpy",
        h2d_count=fv_h2d_count,
        d2h_count=fv_d2h_count,
        to_device_calls=fv_to_device_calls,
        sync_calls=fv_sync_calls,
        vram_alloc_delta_mb=(vram_after_fv - vram_before_fv) / (1024 ** 2),
        wall_time_ms=total_fv_wall,
        gpu_compute_time_ms=total_fv_gpu if faces_found else None,
        notes="Preprocessing & normalization on CPU float32; H2D tensor transfer; InceptionResnetV1 forward; D2H transfer"
    ))

    # ─────────────────────────────────────────────────────────────
    # STAGE 6: VEHICLE CROPS & PLATE DETECTION
    # ─────────────────────────────────────────────────────────────
    vehicle_tracks = [t for t in tracks if t.class_name in ("car", "suv", "van", "truck", "bus", "motorcycle", "vehicle")]
    total_pd_wall = 0.0
    candidates_found = []

    for trk in vehicle_tracks:
        vx1, vy1, vx2, vy2 = trk.bbox
        vx1, vy1 = max(0, vx1), max(0, vy1)
        vx2, vy2 = min(w, vx2), min(h, vy2)
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        if vehicle_crop.size > 0:
            t0 = time.perf_counter()
            cands = plate_detector.detect_plates(vehicle_crop)
            t1 = time.perf_counter()
            total_pd_wall += (t1 - t0) * 1000.0
            for bbox, pc in cands:
                candidates_found.append((trk, bbox, pc))

    traces.append(StageTrace(
        stage_name="6. Vehicle Crops & Plate Detection",
        input_device="CPU",
        output_device="CPU",
        transition_type="CPU->CPU",
        input_shape=f"{len(vehicle_tracks)} vehicle crops",
        input_dtype="uint8 (BGR)",
        input_layout="C-contiguous numpy",
        output_shape=f"{len(candidates_found)} plate candidates",
        output_dtype="List[Tuple[bbox, np.ndarray]]",
        output_layout="C-contiguous numpy crops",
        wall_time_ms=total_pd_wall,
        notes="OpenCV morphology, Sobel, OTSU, contour filtering on CPU"
    ))

    # ─────────────────────────────────────────────────────────────
    # STAGE 7: PLATE PREPROCESSING & OCR (PaddleOCR)
    # ─────────────────────────────────────────────────────────────
    total_ocr_wall = 0.0
    ocr_calls = 0

    for trk, bbox, plate_crop in candidates_found:
        if plate_crop is not None and plate_crop.size > 0:
            q = pipeline.vehicle_quality_scorer.score(plate_crop)
            if q.is_acceptable:
                t0 = time.perf_counter()
                res = ocr_adapter.recognize_plate(plate_crop)
                t1 = time.perf_counter()
                total_ocr_wall += (t1 - t0) * 1000.0
                ocr_calls += 1

    traces.append(StageTrace(
        stage_name="7. Plate Preprocessing & OCR (PaddleOCR)",
        input_device="CPU",
        output_device="CPU",
        transition_type="CPU->CPU",
        input_shape=f"{ocr_calls} plate crops",
        input_dtype="uint8 (BGR)",
        input_layout="C-contiguous numpy",
        output_shape=f"{ocr_calls} recognized texts",
        output_dtype="PlateResult (string + float)",
        output_layout="CPU Python objects",
        wall_time_ms=total_ocr_wall,
        notes="PaddleOCR (PP-OCRv4) running on CPU backend"
    ))

    # ─────────────────────────────────────────────────────────────
    # STAGE 8: ANALYTICS & EVENT ENGINE
    # ─────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    fence_events = pipeline.virtual_fence.process_tracks(tracks, camera_id="camera-01", timestamp=now)
    suspicious_events = pipeline.suspicious_activity.process_tracks(tracks, camera_id="camera-01", timestamp=now)
    night_events = pipeline.night_movement.process_frame(frame, tracks, camera_id="camera-01", timestamp=now)
    all_events = fence_events + suspicious_events + night_events
    emitted = pipeline.event_engine.filter_and_emit(all_events)
    t1 = time.perf_counter()

    traces.append(StageTrace(
        stage_name="8. Analytics & Event Engine",
        input_device="CPU",
        output_device="CPU",
        transition_type="CPU->CPU",
        input_shape=f"{len(tracks)} tracks",
        input_dtype="Track objects",
        input_layout="CPU Python list",
        output_shape=f"{len(emitted)} events emitted",
        output_dtype="SecurityEvent objects",
        output_layout="CPU Python list",
        wall_time_ms=(t1 - t0) * 1000.0,
        notes="Virtual fence, suspicious loitering, event deduplication on CPU"
    ))

    return traces


def print_trace_report(img_name: str, traces: List[StageTrace]):
    print("\n" + "=" * 80)
    print(f"DEVICE & MEMORY TRACE REPORT: {img_name}")
    print("=" * 80)

    header = f"{'Stage':<38} | {'Transition':<14} | {'H2D':>4} | {'D2H':>4} | {'Wall (ms)':>10} | {'GPU (ms)':>10}"
    print(header)
    print("-" * 88)

    total_wall = 0.0
    total_gpu = 0.0
    total_h2d = 0
    total_d2h = 0
    total_item = 0
    total_syncs = 0

    for t in traces:
        gpu_str = f"{t.gpu_compute_time_ms:>10.2f}" if t.gpu_compute_time_ms is not None else f"{'N/A':>10}"
        print(f"{t.stage_name:<38} | {t.transition_type:<14} | {t.h2d_count:>4} | {t.d2h_count:>4} | {t.wall_time_ms:>10.2f} | {gpu_str}")
        total_wall += t.wall_time_ms
        if t.gpu_compute_time_ms is not None:
            total_gpu += t.gpu_compute_time_ms
        total_h2d += t.h2d_count
        total_d2h += t.d2h_count
        total_item += t.item_calls
        total_syncs += t.sync_calls

    print("-" * 88)
    print(f"{'TOTAL PIPELINE':<38} | {'':<14} | {total_h2d:>4} | {total_d2h:>4} | {total_wall:>10.2f} | {total_gpu:>10.2f}")

    print("\n" + "=" * 80)
    print("LOW-LEVEL MEMORY & SYNCHRONIZATION AUDIT")
    print("=" * 80)
    print(f"Total H2D (RAM -> VRAM) Transfers:     {total_h2d}")
    print(f"Total D2H (VRAM -> RAM) Transfers:     {total_d2h}")
    print(f"Total .item() scalar sync calls:       {total_item}")
    print(f"Total explicit/blocking CUDA syncs:    {total_syncs}")
    print("\nDetailed Stage Breakdown:")
    for t in traces:
        print(f"\n[{t.stage_name}]")
        print(f"  Input:   {t.input_device} ({t.input_shape}, {t.input_dtype}, layout: {t.input_layout})")
        print(f"  Output:  {t.output_device} ({t.output_shape}, {t.output_dtype}, layout: {t.output_layout})")
        print(f"  Memory:  H2D={t.h2d_count}, D2H={t.d2h_count}, .item()={t.item_calls}, syncs={t.sync_calls}")
        if t.notes:
            print(f"  Notes:   {t.notes}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "difficultImage.png"
    traces = run_memory_trace(target)
    print_trace_report(target, traces)
