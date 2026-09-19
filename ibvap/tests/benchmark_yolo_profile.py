"""
YOLO and Face Detection Internal Profiler & Micro-Benchmark for IBVAP.
Instruments every sub-stage of YOLOv8 and YuNet with high-precision timers,
CUDA synchronization, device checks, and statistical reporting.
"""

import os
import sys
import time
import numpy as np
import cv2
import torch

from ibvap.core.config import IBVAPConfig
from ibvap.detection.object_detector import YOLOv8Detector
from ibvap.face.detector import OpenCVFaceDetector

def profile_yolo_and_face():
    print("=" * 70)
    print("PHASE 2: HARDWARE & CUDA VERIFICATION")
    print("=" * 70)
    cuda_available = torch.cuda.is_available()
    print(f"PyTorch Version:         {torch.__version__}")
    print(f"CUDA Available:          {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        vram_res = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"GPU Name:                {gpu_name}")
        print(f"Initial VRAM Allocated:  {vram_alloc:.2f} MB")
        print(f"Initial VRAM Reserved:   {vram_res:.2f} MB")
    else:
        gpu_name = "N/A (CPU only)"
        print(f"GPU Name:                {gpu_name}")

    # Load configuration and detector
    config = IBVAPConfig()
    detector = YOLOv8Detector(config)

    print("\n--- Model Device & Dtype Inspection ---")
    print(f"detector.device:         {detector.device}")
    if detector.model is not None and hasattr(detector.model, "model"):
        param = next(detector.model.model.parameters(), None)
        if param is not None:
            print(f"Model Parameter Device:  {param.device}")
            print(f"Model Parameter Dtype:   {param.dtype}")
        else:
            print("Model parameters: None found")
    else:
        print("detector.model is None or has no model attribute")

    # Load or generate test image
    test_img_path = os.path.join(os.getcwd(), "test_car.png")
    if os.path.exists(test_img_path):
        frame = cv2.imread(test_img_path)
        print(f"\nLoaded test image from:  {test_img_path}")
    else:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Draw some shapes to simulate objects
        cv2.rectangle(frame, (100, 100), (300, 500), (0, 255, 0), -1)
        cv2.rectangle(frame, (400, 200), (800, 600), (0, 0, 255), -1)
        print("\nGenerated synthetic test frame (720x1280)")

    h, w = frame.shape[:2]
    print(f"Source Frame Resolution: {w}x{h}")

    # =========================================================================
    # PHASE 1 & 3: INTERNAL INSTRUMENTATION OF YOLO STAGES
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: YOLO INTERNAL STAGE INSTRUMENTATION")
    print("=" * 70)

    # Instrumenting the components of detector.detect(frame)
    # 1. Preprocessing (letterbox resize, BGR->RGB, HWC->CHW, /255, tensor)
    # 2. CPU -> GPU transfer
    # 3. Model forward pass
    # 4. GPU synchronization
    # 5. NMS / postprocessing
    # 6. GPU -> CPU transfer
    # 7. Result conversion (box unpacking, string lookup, Detection objects)

    from ultralytics.data.augment import LetterBox
    from ultralytics.engine.results import Results

    model = detector.model
    device = detector.device

    # Warmup runs
    print("Warming up YOLO (10 iterations)...")
    for _ in range(10):
        _ = detector.detect(frame)
    if cuda_available:
        torch.cuda.synchronize()

    # Detailed instrumented breakdown over 100 iterations
    N_RUNS = 100
    times_preprocess = []
    times_transfer_gpu = []
    times_forward = []
    times_sync = []
    times_nms = []
    times_transfer_cpu = []
    times_result_conv = []
    times_total_yolo = []

    # Also test the standard detector.detect(frame) directly
    times_detect_direct = []

    letterbox = LetterBox(new_shape=(640, 640), auto=True)

    print(f"Running {N_RUNS} warm iterations with sub-stage timing...")
    for i in range(N_RUNS):
        t_direct_start = time.perf_counter()
        _ = detector.detect(frame)
        if cuda_available:
            torch.cuda.synchronize()
        times_detect_direct.append((time.perf_counter() - t_direct_start) * 1000.0)

        # Detailed breakdown run:
        t0 = time.perf_counter()
        # Preprocessing
        img = letterbox(image=frame)
        img = img.transpose((2, 0, 1))[::-1]  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img)
        t1 = time.perf_counter()

        # Tensor creation & GPU transfer
        tensor_img = torch.from_numpy(img).to(device)
        tensor_img = tensor_img.float() / 255.0
        if tensor_img.ndimension() == 3:
            tensor_img = tensor_img.unsqueeze(0)
        t2 = time.perf_counter()

        # Model forward pass
        if cuda_available:
            torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        with torch.no_grad():
            preds = model.model(tensor_img)
        t3 = time.perf_counter()

        # GPU synchronization
        if cuda_available:
            torch.cuda.synchronize()
        t4 = time.perf_counter()

        # NMS / postprocessing
        from ultralytics.utils import ops
        preds_nms = ops.non_max_suppression(
            preds,
            conf_thres=detector.confidence_threshold,
            iou_thres=detector.iou_threshold,
            classes=None,
            agnostic=False,
            max_det=300
        )
        t5 = time.perf_counter()

        # GPU -> CPU transfer & Result conversion
        # Case 1: The current implementation (.item() and .cpu().numpy() in loop)
        t_res_start = time.perf_counter()
        dets = []
        for pred in preds_nms:
            if pred is not None and len(pred):
                # Current unoptimized loop:
                for box in pred:
                    cls_id = int(box[5].item())
                    conf = float(box[4].item())
                    xyxy = box[:4].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    dets.append((cls_id, conf, (x1, y1, x2, y2)))
        t6 = time.perf_counter()

        times_preprocess.append((t1 - t0) * 1000.0)
        times_transfer_gpu.append((t2 - t1) * 1000.0)
        times_forward.append((t3 - t_fwd_start) * 1000.0)
        times_sync.append((t4 - t3) * 1000.0)
        times_nms.append((t5 - t4) * 1000.0)
        times_result_conv.append((t6 - t_res_start) * 1000.0)
        times_total_yolo.append((t6 - t0) * 1000.0)

    def stats(arr):
        a = np.array(arr)
        return {
            "min": float(np.min(a)),
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": float(np.max(a)),
            "stdev": float(np.std(a)),
        }

    s_direct = stats(times_detect_direct)
    s_pre = stats(times_preprocess)
    s_gpu_xfer = stats(times_transfer_gpu)
    s_fwd = stats(times_forward)
    s_sync = stats(times_sync)
    s_nms = stats(times_nms)
    s_conv = stats(times_result_conv)
    s_tot = stats(times_total_yolo)

    print("\n--- YOLO Direct detect() Stats (100 runs) ---")
    print(f"Mean:   {s_direct['mean']:.2f} ms")
    print(f"Median: {s_direct['median']:.2f} ms")
    print(f"P95:    {s_direct['p95']:.2f} ms")
    print(f"P99:    {s_direct['p99']:.2f} ms")
    print(f"Max:    {s_direct['max']:.2f} ms")
    print(f"Min:    {s_direct['min']:.2f} ms")
    print(f"Stdev:  {s_direct['stdev']:.2f} ms")

    print("\n--- YOLO Instrumented Stage Breakdown (Mean / P95 / Max) ---")
    print(f"1. Preprocessing:        {s_pre['mean']:.2f} ms  | P95: {s_pre['p95']:.2f} ms  | Max: {s_pre['max']:.2f} ms")
    print(f"2. Tensor/GPU Transfer:  {s_gpu_xfer['mean']:.2f} ms  | P95: {s_gpu_xfer['p95']:.2f} ms  | Max: {s_gpu_xfer['max']:.2f} ms")
    print(f"3. Model Forward:        {s_fwd['mean']:.2f} ms  | P95: {s_fwd['p95']:.2f} ms  | Max: {s_fwd['max']:.2f} ms")
    print(f"4. GPU Sync:             {s_sync['mean']:.2f} ms  | P95: {s_sync['p95']:.2f} ms  | Max: {s_sync['max']:.2f} ms")
    print(f"5. NMS / Postprocessing: {s_nms['mean']:.2f} ms  | P95: {s_nms['p95']:.2f} ms  | Max: {s_nms['max']:.2f} ms")
    print(f"6. Result Conversion:    {s_conv['mean']:.2f} ms  | P95: {s_conv['p95']:.2f} ms  | Max: {s_conv['max']:.2f} ms")
    print(f"TOTAL Instrumented:      {s_tot['mean']:.2f} ms  | P95: {s_tot['p95']:.2f} ms  | Max: {s_tot['max']:.2f} ms")

    # =========================================================================
    # PHASE 7: FACE DETECTION (YuNet) PROFILING
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 7: FACE DETECTION (YuNet) PROFILING")
    print("=" * 70)

    face_detector = OpenCVFaceDetector(config)
    print(f"Active Detector Type:    {face_detector.active_detector_type}")

    # Test crops: normal (300x400) and large (800x1000)
    crop_normal = frame[:min(400, h), :min(300, w)]
    crop_large = frame[:min(1000, h), :min(800, w)]

    # Warmup
    for _ in range(5):
        _ = face_detector.detect_faces(crop_normal)

    times_face_normal = []
    times_face_large = []
    times_set_size = []
    times_yunet_detect = []

    for _ in range(50):
        # Measure setInputSize vs detect
        ch, cw = crop_normal.shape[:2]
        t_start = time.perf_counter()
        _ = face_detector.detect_faces(crop_normal)
        times_face_normal.append((time.perf_counter() - t_start) * 1000.0)

        # Breakdown of setInputSize vs detect inside YuNet
        if face_detector.yunet is not None:
            target_max = 640.0
            max_d = max(ch, cw)
            scale = target_max / float(max_d) if max_d > target_max else 1.0
            nw, nh = int(round(cw * scale)), int(round(ch * scale))
            resized = cv2.resize(crop_normal, (nw, nh), interpolation=cv2.INTER_AREA) if scale < 1.0 else crop_normal

            t_sis = time.perf_counter()
            face_detector.yunet.setInputSize((nw, nh))
            t_sis_end = time.perf_counter()

            _, raw = face_detector.yunet.detect(resized)
            t_det_end = time.perf_counter()

            times_set_size.append((t_sis_end - t_sis) * 1000.0)
            times_yunet_detect.append((t_det_end - t_sis_end) * 1000.0)

    s_fn = stats(times_face_normal)
    print("\n--- Face Detection Normal Crop (50 runs) ---")
    print(f"Mean:   {s_fn['mean']:.2f} ms")
    print(f"Median: {s_fn['median']:.2f} ms")
    print(f"P95:    {s_fn['p95']:.2f} ms")
    print(f"P99:    {s_fn['p99']:.2f} ms")
    print(f"Max:    {s_fn['max']:.2f} ms")

    if times_set_size and times_yunet_detect:
        s_ss = stats(times_set_size)
        s_yd = stats(times_yunet_detect)
        print(f"  setInputSize() Mean:   {s_ss['mean']:.3f} ms | Max: {s_ss['max']:.3f} ms")
        print(f"  yunet.detect() Mean:   {s_yd['mean']:.2f} ms | Max: {s_yd['max']:.2f} ms")

    if cuda_available:
        final_vram_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        final_vram_res = torch.cuda.memory_reserved(0) / (1024 ** 2)
        print(f"\nFinal VRAM Allocated:    {final_vram_alloc:.2f} MB")
        print(f"Final VRAM Reserved:     {final_vram_res:.2f} MB")

if __name__ == "__main__":
    profile_yolo_and_face()
