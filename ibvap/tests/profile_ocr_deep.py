"""
Deep OCR Profiler & Diagnosis for IBVAP.
Investigates the exact breakdown of OCR latency:
- Initialization vs Warm inference
- Number of candidates, variants, and OCR forward passes
- Device (CPU vs CUDA)
- Micro-benchmark (1 crop 1 pass vs 3 variants vs 3 candidates)
- Preprocessing vs Inference vs Postprocessing vs Consensus
"""

import os
import sys
import time
import numpy as np
import cv2

from ibvap.core.config import IBVAPConfig
from ibvap.anpr.plate_detector import LicensePlateDetector
from ibvap.anpr.ocr_adapter import ANPRAdapter
from ibvap.vehicle.quality import PlateQualityScorer
from ibvap.vehicle.selector import BestObservationSelector
from ibvap.vehicle.consensus import ControlledOCRRunner, PlateConsensusEngine
from ibvap.vehicle.types import VehicleObservation, VehicleTrackState

def diagnose_ocr():
    print("=" * 70)
    print("OCR PERFORMANCE INVESTIGATION & PROFILING")
    print("=" * 70)

    config = IBVAPConfig()

    # 1. Check Paddle / PaddleOCR environment and device
    print("\n--- 1. PADDLE / PADDLEOCR ENVIRONMENT & DEVICE ---")
    try:
        import paddle
        print(f"Paddle Version:          {paddle.__version__}")
        print(f"Paddle Compiled with CUDA: {paddle.is_compiled_with_cuda()}")
        print(f"Paddle Device:           {paddle.device.get_device()}")
    except Exception as e:
        print(f"Paddle import check:     {e}")

    try:
        import paddleocr
        print(f"PaddleOCR Version:       {paddleocr.__version__}")
    except Exception as e:
        print(f"PaddleOCR import check:  {e}")

    try:
        import paddlex
        print(f"PaddleX Version:         {getattr(paddlex, '__version__', 'unknown')}")
    except Exception as e:
        print(f"PaddleX import check:    {e}")

    # 2. Check Cold Model Initialization Time
    print("\n--- 2. COLD INITIALIZATION TIMING ---")
    adapter = ANPRAdapter(config)
    t_init_start = time.perf_counter()
    adapter._ensure_ocr_engine()
    t_init_end = time.perf_counter()
    init_ms = (t_init_end - t_init_start) * 1000.0
    print(f"ANPRAdapter._ensure_ocr_engine() time: {init_ms:.2f} ms ({init_ms/1000.0:.2f} s)")
    print(f"Reader object type:      {type(adapter.reader)}")
    if hasattr(adapter.reader, "predict"):
        print("Reader interface:        PaddleX (predict)")
    elif hasattr(adapter.reader, "ocr"):
        print("Reader interface:        PaddleOCR (ocr)")
    else:
        print("Reader interface:        Unknown / None")

    # 3. Load Test Image & Extract Plate Crop
    test_img_path = os.path.join(os.getcwd(), "test_car.png")
    if os.path.exists(test_img_path):
        frame = cv2.imread(test_img_path)
    else:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
    # Detect vehicle plate candidates using existing LicensePlateDetector
    plate_detector = LicensePlateDetector(config)
    t_pd_start = time.perf_counter()
    candidates = plate_detector.detect_plates(frame)
    t_pd_end = time.perf_counter()
    pd_ms = (t_pd_end - t_pd_start) * 1000.0
    print(f"\nLicensePlateDetector found {len(candidates)} candidates in {pd_ms:.2f} ms")

    if not candidates:
        # Create a synthetic plate crop for testing
        print("Creating synthetic plate crop (120x40 with text)")
        plate_crop = np.ones((40, 120, 3), dtype=np.uint8) * 255
        cv2.putText(plate_crop, "DL01AB1234", (5, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        candidates = [(((10, 10, 130, 50)), plate_crop)]

    top_candidate_crop = candidates[0][1]
    print(f"Top plate crop shape: {top_candidate_crop.shape}")

    # 4. Break down Preprocessing & Variants
    print("\n--- 3. IMAGE PREPROCESSING & VARIANTS BREAKDOWN ---")
    t_var0 = time.perf_counter()
    # Variant 1: Natural padded
    h_c, w_c = top_candidate_crop.shape[:2]
    target_h = 48 if h_c < 48 else (64 if h_c > 96 else h_c)
    scale = float(target_h) / float(max(1, h_c))
    target_w = max(96, int(w_c * scale))
    resized = cv2.resize(top_candidate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    var1 = cv2.copyMakeBorder(resized, 8, 8, 12, 12, cv2.BORDER_REPLICATE)
    t_var1 = time.perf_counter()

    # Variant 2: GaussianBlur + unsharp
    blurred = cv2.GaussianBlur(var1, (0, 0), 1.5)
    var2 = cv2.addWeighted(var1, 1.4, blurred, -0.4, 0)
    t_var2 = time.perf_counter()

    # Variant 3: Bilateral + CLAHE
    gray = cv2.cvtColor(var1, cv2.COLOR_BGR2GRAY)
    denoised = cv2.bilateralFilter(gray, 7, 50, 50)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    var3 = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    t_var3 = time.perf_counter()

    print(f"Variant 1 (Padded BGR) creation:       {(t_var1 - t_var0)*1000.0:.2f} ms")
    print(f"Variant 2 (Unsharp Mask) creation:     {(t_var2 - t_var1)*1000.0:.2f} ms")
    print(f"Variant 3 (Bilateral+CLAHE) creation:  {(t_var3 - t_var2)*1000.0:.2f} ms")
    print(f"Total preprocessing for 1 candidate:   {(t_var3 - t_var0)*1000.0:.2f} ms")

    # 5. Diagnostic Comparison: 1 candidate/1 variant vs 1 candidate/3 variants vs 3 candidates/3 variants
    print("\n--- 4. INFERENCE TIMING DIAGNOSTIC TABLE ---")
    # A. 1 candidate, 1 variant
    t_a0 = time.perf_counter()
    if hasattr(adapter.reader, "predict"):
        res_a = list(adapter.reader.predict([var1]))
    elif hasattr(adapter.reader, "ocr"):
        res_a = adapter.reader.ocr(var1, det=False, rec=True)
    t_a1 = time.perf_counter()
    time_1c_1v = (t_a1 - t_a0) * 1000.0
    print(f"Configuration 1: (1 candidate / 1 variant)   = {time_1c_1v:.2f} ms")

    # B. 1 candidate, 3 variants
    t_b0 = time.perf_counter()
    if hasattr(adapter.reader, "predict"):
        res_b = list(adapter.reader.predict([var1, var2, var3]))
    elif hasattr(adapter.reader, "ocr"):
        res_b = [adapter.reader.ocr(v, det=False, rec=True) for v in [var1, var2, var3]]
    t_b1 = time.perf_counter()
    time_1c_3v = (t_b1 - t_b0) * 1000.0
    print(f"Configuration 2: (1 candidate / 3 variants)  = {time_1c_3v:.2f} ms")

    # C. 3 candidates, 1 variant each
    cands_3 = [c[1] for c in candidates[:3]]
    while len(cands_3) < 3:
        cands_3.append(top_candidate_crop)
    var1_list = [cv2.copyMakeBorder(cv2.resize(c, (120, 48)), 8, 8, 12, 12, cv2.BORDER_REPLICATE) for c in cands_3]
    t_c0 = time.perf_counter()
    if hasattr(adapter.reader, "predict"):
        res_c = list(adapter.reader.predict(var1_list))
    elif hasattr(adapter.reader, "ocr"):
        res_c = [adapter.reader.ocr(v, det=False, rec=True) for v in var1_list]
    t_c1 = time.perf_counter()
    time_3c_1v = (t_c1 - t_c0) * 1000.0
    print(f"Configuration 3: (3 candidates / 1 variant)  = {time_3c_1v:.2f} ms")

    # D. 3 candidates, 3 variants each (9 images total!)
    all_9_vars = []
    for c in cands_3:
        all_9_vars.extend(adapter.preprocess_plate_crop(c))
    t_d0 = time.perf_counter()
    if hasattr(adapter.reader, "predict"):
        res_d = list(adapter.reader.predict(all_9_vars))
    elif hasattr(adapter.reader, "ocr"):
        res_d = [adapter.reader.ocr(v, det=False, rec=True) for v in all_9_vars]
    t_d1 = time.perf_counter()
    time_3c_3v = (t_d1 - t_d0) * 1000.0
    print(f"Configuration 4: (3 candidates / 3 variants) = {time_3c_3v:.2f} ms")

    # 6. Micro-Benchmark: 20 Warm Iterations of 1 candidate / 1 variant
    print("\n--- 5. PURE OCR MICRO-BENCHMARK (20 Warm Runs of 1 candidate / 1 variant) ---")
    times_micro = []
    for _ in range(20):
        t0 = time.perf_counter()
        if hasattr(adapter.reader, "predict"):
            _ = list(adapter.reader.predict([var1]))
        elif hasattr(adapter.reader, "ocr"):
            _ = adapter.reader.ocr(var1, det=False, rec=True)
        times_micro.append((time.perf_counter() - t0) * 1000.0)

    a_m = np.array(times_micro)
    print(f"Min:    {np.min(a_m):.2f} ms")
    print(f"Mean:   {np.mean(a_m):.2f} ms")
    print(f"Median: {np.median(a_m):.2f} ms")
    print(f"P95:    {np.percentile(a_m, 95):.2f} ms")
    print(f"P99:    {np.percentile(a_m, 99):.2f} ms")
    print(f"Max:    {np.max(a_m):.2f} ms")

    # 7. Measure Complete ControlledOCRRunner & Consensus
    print("\n--- 6. COMPLETE CONTROLLED OCR & CONSENSUS BREAKDOWN ---")
    controlled_ocr = ControlledOCRRunner(ocr_adapter=adapter, max_ocr_attempts_per_track=3)
    consensus_engine = PlateConsensusEngine()

    quality_scorer = PlateQualityScorer()
    observations = []
    for i, (b, c) in enumerate(candidates[:3]):
        q = quality_scorer.score(c)
        obs = VehicleObservation(
            track_id=1,
            frame_index=1,
            timestamp=time.time(),
            plate_bbox=(0, 0, 10, 10),
            plate_crop=c,
            quality=q
        )
        observations.append(obs)

    track_state = VehicleTrackState(
        track_id=1,
        camera_id="camera-01",
        vehicle_class="car",
        status=None,
        first_seen=time.time(),
        last_seen=time.time(),
        observations=observations
    )

    t_ocr_run_start = time.perf_counter()
    ocr_results = controlled_ocr.run_ocr(observations, track_state=track_state)
    t_ocr_run_end = time.perf_counter()
    ocr_run_ms = (t_ocr_run_end - t_ocr_run_start) * 1000.0

    t_cons_start = time.perf_counter()
    consensus = consensus_engine.evaluate(ocr_results)
    t_cons_end = time.perf_counter()
    cons_ms = (t_cons_end - t_cons_start) * 1000.0

    print(f"ControlledOCRRunner.run_ocr() time: {ocr_run_ms:.2f} ms")
    print(f"PlateConsensusEngine.evaluate() time: {cons_ms:.2f} ms")
    print(f"Consensus plate: '{consensus.plate_number}', conf: {consensus.confidence}, confirmed: {consensus.is_confirmed}")

if __name__ == "__main__":
    diagnose_ocr()
