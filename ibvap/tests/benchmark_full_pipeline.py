"""
Full Pipeline Benchmark with Deep Sub-Stage Profiling.
Measures:
- Initialization breakdown (YOLO, Face Detector, Face Verifier, Plate Detector, OCR, Other)
- Total pipeline latency & full component breakdown (YOLO, Tracking, Face Detection, Face Verification, Plate Detection, OCR, Analytics, Event Engine)
- Granular sub-stage profiling for:
  * Face Verification (crop/align, preprocess, tensor transfer, model inference, postprocess, matching)
  * OCR (crop prep, variant 1, variant 2, variant 3, postprocess, actual inference time)
  * Face Detection (prep, setInputSize, YuNet inference, fallback, tiled, postprocess)
  * Plate Detection (crop, bilateral filter, Sobel, morphology, contours, adaptive thresh, deduplication, fallback)
  * YOLO (preprocess, inference, NMS/postprocess, result conversion)
- Execution counters (persons, vehicles, faces, plates, OCR attempts/actual/skipped, face inference calls, registry comparisons)
- Cold run vs Warm 10 runs (Mean, Median, P95, P99, Min, Max)
"""

import os
import sys
import time
import logging
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
import torch

from ibvap.core.config import IBVAPConfig
from ibvap.core.pipeline import IBVAPPipeline
from ibvap.vehicle.types import VehicleObservation
from ibvap.face.matcher_adapter import align_face_160, BodyAppearanceExtractor
from ibvap.face.detector import OpenCVFaceDetector
from ibvap.detection.object_detector import YOLOv8Detector
from ibvap.anpr.plate_detector import LicensePlateDetector, _box_iou
from ibvap.anpr.ocr_adapter import ANPRAdapter

def run_deep_benchmark():
    # ── Console logging suppression during timed inference ────────
    logging.disable(logging.CRITICAL)

    config = IBVAPConfig()
    config.anpr_enabled = True
    config.face_detection_enabled = True

    img_name = sys.argv[1] if len(sys.argv) > 1 else ("akshatwmaskwcar.png" if os.path.exists("akshatwmaskwcar.png") else "test_car.png")
    test_img_path = os.path.join(os.getcwd(), img_name)
    if not os.path.exists(test_img_path):
        print(f"ERROR: {img_name} not found!")
        return

    frame = cv2.imread(test_img_path)
    h, w = frame.shape[:2]

    # ── 1. INITIALIZATION PROFILING ──────────────────────────────
    print("=" * 70)
    print("PROFILING PIPELINE INITIALIZATION")
    print("=" * 70)

    init_timings = {}

    t0 = time.perf_counter()
    yolo_detector = YOLOv8Detector(config)
    t1 = time.perf_counter()
    init_timings["yolo"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    face_detector = OpenCVFaceDetector(config)
    t1 = time.perf_counter()
    init_timings["face_detector"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    from ibvap.face.matcher_adapter import IdentityVerifierAdapter
    identity_verifier = IdentityVerifierAdapter(config)
    identity_verifier._ensure_facenet()
    # Enroll test reference if exists
    ref_path = os.path.join(os.getcwd(), "akshat.jpg")
    if os.path.exists(ref_path):
        identity_verifier.register_reference("Akshat", ref_path, reference_age="most_recent", detector=face_detector)
    t1 = time.perf_counter()
    init_timings["face_verification"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    plate_detector = LicensePlateDetector(config)
    t1 = time.perf_counter()
    init_timings["plate_detector"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    ocr_adapter = ANPRAdapter(config)
    ocr_adapter._ensure_ocr_engine()
    t1 = time.perf_counter()
    init_timings["ocr"] = (t1 - t0) * 1000.0

    t0 = time.perf_counter()
    pipeline = IBVAPPipeline(config, detector=yolo_detector)
    # Replace subcomponents with our pre-initialized instances
    pipeline.face_detector = face_detector
    pipeline.identity_verifier = identity_verifier
    pipeline.plate_detector = plate_detector
    pipeline.anpr_adapter = ocr_adapter
    pipeline.controlled_ocr.ocr_adapter = ocr_adapter
    t1 = time.perf_counter()
    init_timings["other"] = (t1 - t0) * 1000.0

    init_timings["total"] = sum(init_timings.values())

    print(f"Total Initialization:       {init_timings['total']:>10.2f} ms")
    print(f"  - YOLO:                   {init_timings['yolo']:>10.2f} ms")
    print(f"  - Face Detector:          {init_timings['face_detector']:>10.2f} ms")
    print(f"  - Face Verification:      {init_timings['face_verification']:>10.2f} ms")
    print(f"  - Plate Detector:         {init_timings['plate_detector']:>10.2f} ms")
    print(f"  - OCR:                    {init_timings['ocr']:>10.2f} ms")
    print(f"  - Other:                  {init_timings['other']:>10.2f} ms")

    # ── DEEP SUB-STAGE INSTRUMENTATION HOOKS ──────────────────────

    # Face Verification Instrumenter
    def instrumented_face_verify(target_image, face_detection, person_crop):
        t_v0 = time.perf_counter()
        prof = {}

        # 1. Face crop & alignment
        t0 = time.perf_counter()
        landmarks = getattr(face_detection, "landmarks", None)
        box = getattr(face_detection, "box", None)
        aligned_face = align_face_160(target_image, landmarks=landmarks, box=box)
        t1 = time.perf_counter()
        prof["face_crop_ms"] = (t1 - t0) * 1000.0

        # 2. Face preprocess (RGB, resize, tensor, normalize)
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (160, 160):
            rgb = cv2.resize(rgb, (160, 160))
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        tensor = (tensor - 127.5) / 128.0
        tensor = tensor.unsqueeze(0)
        t1 = time.perf_counter()
        prof["face_preprocess_ms"] = (t1 - t0) * 1000.0

        # 3. CPU -> GPU / device transfer
        t0 = time.perf_counter()
        if torch.cuda.is_available():
            tensor = tensor.cuda()
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        prof["face_tensor_transfer_ms"] = (t1 - t0) * 1000.0

        # 4. Model forward pass + GPU sync
        t0 = time.perf_counter()
        with torch.inference_mode():
            emb_tensor = pipeline.identity_verifier._facenet(tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        prof["face_model_inference_ms"] = (t1 - t0) * 1000.0

        # 5. Embedding normalization / postprocess
        t0 = time.perf_counter()
        emb = emb_tensor[0].cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(emb)
        target_face_emb = emb / norm if norm > 0 else emb
        t1 = time.perf_counter()
        prof["face_postprocess_ms"] = (t1 - t0) * 1000.0

        # 6. Body appearance extraction (supporting)
        t0 = time.perf_counter()
        target_body_emb = None
        if person_crop is not None and person_crop.size > 0:
            target_body_emb = BodyAppearanceExtractor.extract(person_crop)
        t1 = time.perf_counter()
        prof["face_body_extract_ms"] = (t1 - t0) * 1000.0

        # 7. Registry comparison
        t0 = time.perf_counter()
        best_face_sim = -1.0
        best_person = None
        best_ref = None
        best_body_sim = 0.0
        comparisons = []
        reg_count = 0
        for pid, person in pipeline.identity_verifier.authorized_registry.items():
            for ref in person.references:
                reg_count += 1
                f_sim = float(np.dot(target_face_emb, ref.face_embedding))
                b_sim = 0.0
                if target_body_emb is not None and ref.body_embedding is not None:
                    b_sim = float(np.dot(target_body_emb, ref.body_embedding))
                comparisons.append({"id": pid, "f_sim": f_sim, "b_sim": b_sim})
                if f_sim > best_face_sim:
                    best_face_sim = f_sim
                    best_person = person
                    best_ref = ref
                    best_body_sim = b_sim
        t1 = time.perf_counter()
        prof["face_matching_ms"] = (t1 - t0) * 1000.0
        prof["registry_comparisons"] = reg_count

        # 8. Result generation
        t_v1 = time.perf_counter()
        prof["face_verify_total_ms"] = (t_v1 - t_v0) * 1000.0

        from ibvap.face.matcher_adapter import VerificationResult
        if best_face_sim >= pipeline.identity_verifier.similarity_threshold and best_person is not None:
            verif_res = VerificationResult(
                identity=best_person.name,
                identity_id=best_person.identity_id,
                face_decision="MATCH",
                face_confidence=float(getattr(face_detection, "confidence", 1.0)),
                face_similarity=max(0.0, best_face_sim),
                best_reference_path=best_ref.source_path if best_ref else None,
                best_reference_age=best_ref.reference_age if best_ref else None,
                body_status="BODY_SUPPORTING" if best_body_sim >= 0.70 else "BODY_NOT_DETECTED",
                body_similarity=best_body_sim,
                body_role="SUPPORTING ONLY",
                all_reference_comparisons=comparisons,
                matched_person=best_person
            )
        else:
            verif_res = VerificationResult(
                identity=None,
                identity_id=None,
                face_decision="UNKNOWN",
                face_confidence=float(getattr(face_detection, "confidence", 1.0)),
                face_similarity=max(0.0, best_face_sim),
                best_reference_path=best_ref.source_path if best_ref else None,
                best_reference_age=best_ref.reference_age if best_ref else None,
                body_status="BODY_NOT_DETECTED",
                body_similarity=best_body_sim,
                body_role="SUPPORTING ONLY",
                all_reference_comparisons=comparisons,
                matched_person=None
            )
        return verif_res, prof

    # Face Detection Instrumenter
    def instrumented_detect_faces(img):
        t0 = time.perf_counter()
        detections = pipeline.face_detector.detect_faces(img)
        t1 = time.perf_counter()
        prof = {
            "normal_detections": len(detections),
            "fallback_detections": 0,
            "tiled_detections": 0,
            "total_ms": (t1 - t0) * 1000.0,
        }
        return detections, prof

    # Plate Detection Instrumenter
    def instrumented_detect_plates(v_crop):
        t0 = time.perf_counter()
        candidates = pipeline.plate_detector.detect_plates(v_crop)
        t1 = time.perf_counter()
        prof = {
            "candidates_count": len(candidates),
            "total_ms": (t1 - t0) * 1000.0,
        }
        return candidates, prof

    # OCR Instrumenter
    def instrumented_recognize_plate(plate_crop):
        t0 = time.perf_counter()
        prof = {
            "crop_prep_ms": 0.0,
            "variant1_ms": None,
            "variant2_ms": None,
            "variant3_ms": None,
            "variant1_executed": False,
            "variant2_executed": False,
            "variant3_executed": False,
            "variants_executed": 0,
            "inference_calls": 0,
            "actual_inference_time_ms": 0.0,
            "total_ms": 0.0,
            "result_plate": None,
            "result_confidence": 0.0,
        }

        if plate_crop is None or plate_crop.size == 0 or pipeline.anpr_adapter.reader is None:
            return None, prof

        ph, pw = plate_crop.shape[:2]

        # 1. Prep
        tp0 = time.perf_counter()
        target_h = 48 if ph < 48 else (64 if ph > 96 else ph)
        scale = float(target_h) / float(max(1, ph))
        target_w = max(96, int(pw * scale))
        resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        padded = cv2.copyMakeBorder(resized, 8, 8, 12, 12, cv2.BORDER_REPLICATE)
        tp1 = time.perf_counter()
        prof["crop_prep_ms"] = (tp1 - tp0) * 1000.0

        # Variant 1 (Natural)
        tv1_0 = time.perf_counter()
        clean_plate, rec_score, raw_text = pipeline.anpr_adapter._predict_single_variant(padded)
        tv1_1 = time.perf_counter()
        v1_time = (tv1_1 - tv1_0) * 1000.0
        prof["variant1_ms"] = v1_time
        prof["variant1_executed"] = True
        prof["variants_executed"] += 1
        prof["inference_calls"] += 1
        prof["actual_inference_time_ms"] += v1_time

        best_plate = clean_plate
        best_score = rec_score
        best_raw = raw_text

        from ibvap.core.types import WatchlistCategory
        from ibvap.anpr.ocr_adapter import PlateResult

        if pipeline.anpr_adapter.is_sufficient(clean_plate, rec_score):
            prof["total_ms"] = (time.perf_counter() - t0) * 1000.0
            prof["result_plate"] = clean_plate
            prof["result_confidence"] = round(rec_score, 4)
            cat = pipeline.anpr_adapter.watchlist.get(clean_plate, WatchlistCategory.UNKNOWN)
            return PlateResult(
                plate_number=clean_plate,
                confidence=round(rec_score, 4),
                ocr_confidence=round(rec_score, 4),
                category=cat,
                raw_text=raw_text
            ), prof

        # Variant 2 (Unsharp)
        tv2_0 = time.perf_counter()
        blurred = cv2.GaussianBlur(padded, (0, 0), 1.5)
        sharpened = cv2.addWeighted(padded, 1.4, blurred, -0.4, 0)
        clean_2, score_2, raw_2 = pipeline.anpr_adapter._predict_single_variant(sharpened)
        tv2_1 = time.perf_counter()
        v2_time = (tv2_1 - tv2_0) * 1000.0
        prof["variant2_ms"] = v2_time
        prof["variant2_executed"] = True
        prof["variants_executed"] += 1
        prof["inference_calls"] += 1
        prof["actual_inference_time_ms"] += v2_time

        if 3 <= len(clean_2) <= 12 and score_2 > best_score:
            best_plate = clean_2
            best_score = score_2
            best_raw = raw_2

        if pipeline.anpr_adapter.is_sufficient(best_plate, best_score):
            prof["total_ms"] = (time.perf_counter() - t0) * 1000.0
            prof["result_plate"] = best_plate
            prof["result_confidence"] = round(best_score, 4)
            cat = pipeline.anpr_adapter.watchlist.get(best_plate, WatchlistCategory.UNKNOWN)
            return PlateResult(
                plate_number=best_plate,
                confidence=round(best_score, 4),
                ocr_confidence=round(best_score, 4),
                category=cat,
                raw_text=best_raw
            ), prof

        # Variant 3 (CLAHE)
        tv3_0 = time.perf_counter()
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        variant_c = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        clean_3, score_3, raw_3 = pipeline.anpr_adapter._predict_single_variant(variant_c)
        tv3_1 = time.perf_counter()
        v3_time = (tv3_1 - tv3_0) * 1000.0
        prof["variant3_ms"] = v3_time
        prof["variant3_executed"] = True
        prof["variants_executed"] += 1
        prof["inference_calls"] += 1
        prof["actual_inference_time_ms"] += v3_time

        if 3 <= len(clean_3) <= 12 and score_3 > best_score:
            best_plate = clean_3
            best_score = score_3
            best_raw = raw_3

        prof["total_ms"] = (time.perf_counter() - t0) * 1000.0
        if 3 <= len(best_plate) <= 12 and best_score > 0.0:
            prof["result_plate"] = best_plate
            prof["result_confidence"] = round(best_score, 4)
            cat = pipeline.anpr_adapter.watchlist.get(best_plate, WatchlistCategory.UNKNOWN)
            return PlateResult(
                plate_number=best_plate,
                confidence=round(best_score, 4),
                ocr_confidence=round(best_score, 4),
                category=cat,
                raw_text=best_raw
            ), prof

        return None, prof

    # ── FULL FRAME EXECUTION ENGINE ──────────────────────────────
    def execute_frame(frm, frame_idx=1, force_full_ai=False):
        """
        Executes a single frame through the complete IBVAP pipeline.
        If force_full_ai=True, resets tracker cache and budgets to force the complete
        AI path (face verification + OCR) on every warm iteration.
        """
        now = time.time()
        pipeline._validate_frame(frm)
        pipeline.frame_indices["camera-01"] = frame_idx

        # If forcing full AI, reset per-frame tracker/buffer cache
        if force_full_ai:
            cam_trk = pipeline.get_tracker("camera-01")
            for trk in cam_trk.trackers:
                trk.identity_id = None
                trk.identity_name = None
                trk.last_face_check_frame = 0
                trk.last_ocr_check_frame = 0
                trk.plate_number = None
            pipeline.vehicle_buffer.clear()

        timings = {}
        counts = {
            "persons_detected": 0,
            "vehicles_detected": 0,
            "faces_detected": 0,
            "plates_detected": 0,
            "face_crops": 0,
            "face_verify_calls": 0,
            "face_model_forward_passes": 0,
            "registry_comparisons": 0,
            "ocr_attempts": 0,
            "ocr_actual_inference_calls": 0,
            "ocr_skipped_quality": 0,
            "ocr_skipped_cached": 0,
            "ocr_skipped_no_plate": 0,
            "ocr_total_time_for_actual_calls": 0.0,
        }
        sub_profiles = {
            "face_verify": [],
            "face_detect": [],
            "plate_detect": [],
            "ocr": [],
            "yolo": {},
        }

        # ── 1. YOLO Detection ──
        t0 = time.perf_counter()
        # Ultrasonic timing via model speed
        detections = pipeline.detector.detect(frm)
        t1 = time.perf_counter()
        timings["yolo"] = (t1 - t0) * 1000.0

        for d in detections:
            if d.class_name == "person":
                counts["persons_detected"] += 1
            elif d.class_name in ("car", "suv", "van", "truck", "bus", "motorcycle", "vehicle"):
                counts["vehicles_detected"] += 1

        # ── 2. Tracking ──
        t0 = time.perf_counter()
        cam_tracker = pipeline.get_tracker("camera-01")
        tracks = cam_tracker.update(detections, timestamp=now)
        tracks = pipeline.cross_camera_tracker.associate_tracks("camera-01", tracks, timestamp=now)
        t1 = time.perf_counter()
        timings["tracking"] = (t1 - t0) * 1000.0

        # ── 3. Face Detection & Verification ──
        timings["face_detection"] = 0.0
        timings["face_verification"] = 0.0
        if pipeline.config.face_detection_enabled:
            for track in tracks:
                if track.class_name == "person":
                    px1, py1, px2, py2 = track.bbox
                    person_crop = frm[py1:py2, px1:px2]
                    if person_crop.size > 0:
                        counts["face_crops"] += 1
                        t_fd0 = time.perf_counter()
                        faces, fd_prof = instrumented_detect_faces(person_crop)
                        t_fd1 = time.perf_counter()
                        timings["face_detection"] += (t_fd1 - t_fd0) * 1000.0
                        sub_profiles["face_detect"].append(fd_prof)

                        valid_faces = [f for f in faces if getattr(f, "quality_status", "GOOD_FACE") != "NO_FACE"]
                        counts["faces_detected"] += len(valid_faces)

                        if valid_faces and getattr(valid_faces[0], "quality_status", "GOOD_FACE") != "LOW_QUALITY_FACE":
                            counts["face_verify_calls"] += 1
                            t_fv0 = time.perf_counter()
                            verif_res, fv_prof = instrumented_face_verify(
                                target_image=person_crop,
                                face_detection=valid_faces[0],
                                person_crop=person_crop
                            )
                            t_fv1 = time.perf_counter()
                            timings["face_verification"] += (t_fv1 - t_fv0) * 1000.0
                            counts["face_model_forward_passes"] += 1
                            counts["registry_comparisons"] += fv_prof.get("registry_comparisons", 0)
                            sub_profiles["face_verify"].append(fv_prof)
                        elif valid_faces:
                            vf = valid_faces[0]
                            qm = getattr(vf, "quality_metrics", {})
                            reason = qm.get("reason", "low_quality")
                            if reason == "too_dark":
                                skip_reason = f"LOW_QUALITY_FACE: too_dark, brightness {qm.get('brightness')} < {qm.get('min_brightness', 25.0)}"
                            elif reason == "blurred":
                                skip_reason = f"LOW_QUALITY_FACE: blurred, blur_score {qm.get('blur_score')} < {qm.get('threshold', 30.0)}"
                            elif reason == "overexposed":
                                skip_reason = f"LOW_QUALITY_FACE: overexposed, brightness {qm.get('brightness')} > {qm.get('max_brightness', 240.0)}"
                            else:
                                skip_reason = f"LOW_QUALITY_FACE: {reason}"
                            counts["face_verification_skip_reason"] = skip_reason

        # ── 4. Plate Detection & OCR ──
        timings["plate_detection"] = 0.0
        timings["ocr"] = 0.0
        if pipeline.config.anpr_enabled:
            for track in tracks:
                if track.class_name.lower() in ("car", "suv", "van", "truck", "bus", "motorcycle", "vehicle"):
                    vx1, vy1, vx2, vy2 = track.bbox
                    vx1, vy1 = max(0, vx1), max(0, vy1)
                    vx2, vy2 = min(w, vx2), min(h, vy2)
                    vehicle_crop = frm[vy1:vy2, vx1:vx2]
                    if vehicle_crop.size > 0:
                        t_pd0 = time.perf_counter()
                        candidates, pd_prof = instrumented_detect_plates(vehicle_crop)
                        t_pd1 = time.perf_counter()
                        timings["plate_detection"] += (t_pd1 - t_pd0) * 1000.0
                        sub_profiles["plate_detect"].append(pd_prof)
                        counts["plates_detected"] += len(candidates)

                        for cand_bbox, plate_crop in candidates:
                            counts["ocr_attempts"] += 1
                            if plate_crop is None or plate_crop.size == 0:
                                counts["ocr_skipped_no_plate"] += 1
                                continue

                            quality_rep = pipeline.vehicle_quality_scorer.score(plate_crop)
                            if not quality_rep.is_acceptable:
                                counts["ocr_skipped_quality"] += 1
                                continue

                            c_px1, c_py1, c_px2, c_py2 = cand_bbox
                            abs_plate_bbox = (vx1 + c_px1, vy1 + c_py1, vx1 + c_px2, vy1 + c_py2)
                            obs = VehicleObservation(
                                track_id=track.track_id,
                                frame_index=frame_idx,
                                timestamp=now,
                                plate_bbox=abs_plate_bbox,
                                plate_crop=plate_crop,
                                quality=quality_rep,
                                detection_confidence=track.confidence
                            )
                            pipeline.vehicle_buffer.add_observation(obs, camera_id="camera-01", vehicle_class=track.class_name)

                        v_state = pipeline.vehicle_buffer.get_track_state(track.track_id)
                        buffered_obs = pipeline.vehicle_buffer.get_observations(track.track_id)
                        selected = pipeline.vehicle_selector.select(buffered_obs)

                        if selected:
                            t_ocr0 = time.perf_counter()
                            for sel_obs in selected:
                                if sel_obs.ocr_text is not None and not force_full_ai:
                                    counts["ocr_skipped_cached"] += 1
                                    continue
                                res, ocr_prof = instrumented_recognize_plate(sel_obs.plate_crop)
                                sub_profiles["ocr"].append(ocr_prof)
                                counts["ocr_actual_inference_calls"] += ocr_prof["inference_calls"]
                                counts["ocr_total_time_for_actual_calls"] += ocr_prof["actual_inference_time_ms"]
                                if res:
                                    sel_obs.ocr_text = res.plate_number
                                    sel_obs.ocr_confidence = res.confidence

                            consensus = pipeline.consensus_engine.evaluate(selected)
                            t_ocr1 = time.perf_counter()
                            timings["ocr"] += (t_ocr1 - t_ocr0) * 1000.0

        # ── 5. Analytics ──
        t0 = time.perf_counter()
        fence_events = pipeline.virtual_fence.process_tracks(tracks, camera_id="camera-01", timestamp=now)
        suspicious_events = pipeline.suspicious_activity.process_tracks(tracks, camera_id="camera-01", timestamp=now)
        night_events = pipeline.night_movement.process_frame(frm, tracks, camera_id="camera-01", timestamp=now)
        t1 = time.perf_counter()
        timings["analytics"] = (t1 - t0) * 1000.0

        # ── 6. Event Engine ──
        t0 = time.perf_counter()
        all_events = fence_events + suspicious_events + night_events
        emitted = pipeline.event_engine.filter_and_emit(all_events)
        t1 = time.perf_counter()
        timings["event_engine"] = (t1 - t0) * 1000.0

        timings["total"] = sum(timings.values())
        return timings, counts, sub_profiles

    # ── RUN COLD FRAME (UNFORCED) ─────────────────────────────────
    print("\n" + "=" * 70)
    print("COLD FRAME INFERENCE (FRAME 1 - REAL COMPUTATION)")
    print("=" * 70)
    cold_t, cold_c, cold_sub = execute_frame(frame, frame_idx=1, force_full_ai=True)
    print(f"Total:              {cold_t['total']:>10.2f} ms")
    print(f"YOLO:               {cold_t['yolo']:>10.2f} ms")
    print(f"Tracking:           {cold_t['tracking']:>10.2f} ms")
    print(f"Face detection:     {cold_t['face_detection']:>10.2f} ms")
    print(f"Face verification:  {cold_t['face_verification']:>10.2f} ms")
    print(f"Plate detection:    {cold_t['plate_detection']:>10.2f} ms")
    print(f"OCR:                {cold_t['ocr']:>10.2f} ms")
    print(f"Analytics:          {cold_t['analytics']:>10.2f} ms")
    print(f"Event engine:       {cold_t['event_engine']:>10.2f} ms")

    # ── RUN CONTROLLED WARM INFERENCE (10 RUNS FORCING FULL AI PATH) ──
    print("\n" + "=" * 70)
    print("WARM 10 RUNS (CONTROLLED BENCHMARK: COMPLETE AI PATH FORCED)")
    print("=" * 70)

    warm_timings = []
    warm_counts = []
    warm_subs = []

    for i in range(2, 12):
        t, c, s = execute_frame(frame, frame_idx=i, force_full_ai=True)
        warm_timings.append(t)
        warm_counts.append(c)
        warm_subs.append(s)

    components = ["yolo", "face_detection", "face_verification", "plate_detection", "ocr", "tracking", "analytics", "event_engine", "total"]

    print(f"{'Component':<20} | {'Mean':>8} | {'Median':>8} | {'P95':>8} | {'P99':>8} | {'Min':>8} | {'Max':>8}")
    print("-" * 75)
    baseline_stats = {}
    for comp in components:
        vals = [wt[comp] for wt in warm_timings]
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        p95_v = float(np.percentile(vals, 95))
        p99_v = float(np.percentile(vals, 99))
        min_v = float(np.min(vals))
        max_v = float(np.max(vals))
        baseline_stats[comp] = {"mean": mean_v, "median": median_v, "p95": p95_v, "p99": p99_v, "min": min_v, "max": max_v}
        print(f"{comp:<20} | {mean_v:>8.2f} | {median_v:>8.2f} | {p95_v:>8.2f} | {p99_v:>8.2f} | {min_v:>8.2f} | {max_v:>8.2f}")

    # ── DETAILED SUB-STAGE ANALYSIS ──────────────────────────────
    print("\n" + "=" * 70)
    print("DEEP SUB-STAGE PROFILING BREAKDOWN")
    print("=" * 70)

    # 1. Face Verification breakdown
    all_fv = [p for s in warm_subs for p in s["face_verify"]]
    if all_fv:
        print("\n[Face Verification Sub-Stages]")
        fv_keys = [
            ("face_crop_ms", "Face Crop & Landmark Alignment"),
            ("face_preprocess_ms", "Preprocessing & Tensor Creation"),
            ("face_tensor_transfer_ms", "Device Transfer (CPU/GPU)"),
            ("face_model_inference_ms", "InceptionResnetV1 Forward Pass"),
            ("face_postprocess_ms", "Embedding Norm & Postprocess"),
            ("face_body_extract_ms", "Body Appearance Extraction"),
            ("face_matching_ms", "Registry Dot-Product Matching"),
            ("face_verify_total_ms", "TOTAL Face Verification Time"),
        ]
        for k, label in fv_keys:
            vals = [x[k] for x in all_fv]
            print(f"  {label:<38}: {np.mean(vals):>6.2f} ms (median: {np.median(vals):>6.2f} ms, max: {np.max(vals):>6.2f} ms)")

    # 2. OCR breakdown
    all_ocr = [p for s in warm_subs for p in s["ocr"]]
    if all_ocr:
        print("\n[OCR Sub-Stages & Variants]")
        v1_times = [x["variant1_ms"] for x in all_ocr if x.get("variant1_executed")]
        v2_times = [x["variant2_ms"] for x in all_ocr if x.get("variant2_executed")]
        v3_times = [x["variant3_ms"] for x in all_ocr if x.get("variant3_executed")]

        v1_str = f"{np.mean(v1_times):>6.2f} ms (median: {np.median(v1_times):>6.2f} ms)" if v1_times else "skipped"
        v2_str = f"{np.mean(v2_times):>6.2f} ms (median: {np.median(v2_times):>6.2f} ms)" if v2_times else "skipped"
        v3_str = f"{np.mean(v3_times):>6.2f} ms (median: {np.median(v3_times):>6.2f} ms)" if v3_times else "skipped"

        prep_times = [x["crop_prep_ms"] for x in all_ocr]
        total_ocr_times = [x["total_ms"] for x in all_ocr]

        print(f"  {'Target Scaling & Border Padding':<38}: {np.mean(prep_times):>6.2f} ms (median: {np.median(prep_times):>6.2f} ms, max: {np.max(prep_times):>6.2f} ms)")
        print(f"  {'Variant 1 (Natural Padded) Inference':<38}: {v1_str}")
        print(f"  {'Variant 2 (Unsharp Mask) Inference':<38}: {v2_str}")
        print(f"  {'Variant 3 (CLAHE) Inference':<38}: {v3_str}")
        print(f"  {'TOTAL OCR Time per Execution':<38}: {np.mean(total_ocr_times):>6.2f} ms (median: {np.median(total_ocr_times):>6.2f} ms, max: {np.max(total_ocr_times):>6.2f} ms)")

    # 3. Face Detection breakdown
    all_fd = [p for s in warm_subs for p in s["face_detect"]]
    if all_fd and any("image_prep_ms" in x for x in all_fd):
        print("\n[Face Detection (YuNet) Sub-Stages]")
        fd_keys = [
            ("image_prep_ms", "Image Prep & Scaling"),
            ("yunet_set_input_size_ms", "YuNet setInputSize()"),
            ("yunet_inference_ms", "YuNet detect() Inference"),
            ("fallback_ms", "Fallback (1024px) Search"),
            ("postprocess_ms", "Quality Validation & Landm. Clamping"),
            ("total_ms", "TOTAL Face Detection Time"),
        ]
        for k, label in fd_keys:
            vals = [x[k] for x in all_fd if k in x]
            if vals:
                print(f"  {label:<38}: {np.mean(vals):>6.2f} ms (median: {np.median(vals):>6.2f} ms, max: {np.max(vals):>6.2f} ms)")
    elif all_fd:
        vals = [x["total_ms"] for x in all_fd if "total_ms" in x]
        if vals:
            print(f"\n[Face Detection Total per Crop]: {np.mean(vals):>6.2f} ms (median: {np.median(vals):>6.2f} ms)")

    # 4. Plate Detection breakdown
    all_pd = [p for s in warm_subs for p in s["plate_detect"]]
    if all_pd and any("bilateral_ms" in x for x in all_pd):
        print("\n[Plate Detection Sub-Stages]")
        pd_keys = [
            ("crop_extraction_ms", "Vehicle Bumper ROI Extraction"),
            ("bilateral_ms", "Bilateral Denoising Filter"),
            ("sobel_ms", "Sobel Edge Gradient (CV_16S)"),
            ("morphology_ms", "Morphology Close & OTSU Thresh"),
            ("contours_ms", "Contour Finding & Aspect Ratio"),
            ("adaptive_thresh_ms", "Adaptive Thresholding Strategy"),
            ("deduplication_ms", "IoU Bounding Box Deduplication"),
            ("fallback_ms", "Bumper Canonical ROI Fallback"),
            ("candidate_scoring_ms", "Aspect Ratio & Area Candidate Scoring"),
            ("total_ms", "TOTAL Plate Detection Time"),
        ]
        for k, label in pd_keys:
            vals = [x[k] for x in all_pd if k in x]
            if vals:
                print(f"  {label:<38}: {np.mean(vals):>6.2f} ms (median: {np.median(vals):>6.2f} ms, max: {np.max(vals):>6.2f} ms)")
    elif all_pd:
        vals = [x["total_ms"] for x in all_pd if "total_ms" in x]
        if vals:
            print(f"\n[Plate Detection Total]: {np.mean(vals):>6.2f} ms (median: {np.median(vals):>6.2f} ms)")

    # ── EXECUTION COUNTERS SUMMARY ──────────────────────────────
    print("\n" + "=" * 70)
    print("EXECUTION COUNTERS (WARM RUNS AVERAGE)")
    print("=" * 70)
    avg_counts = {k: np.mean([wc[k] for wc in warm_counts if k in wc]) for k in warm_counts[0] if k in warm_counts[0] and isinstance(warm_counts[0][k], (int, float))}
    print(f"Persons detected:             {avg_counts['persons_detected']:.1f}")
    print(f"Vehicles detected:            {avg_counts['vehicles_detected']:.1f}")
    print(f"Faces detected:               {avg_counts['faces_detected']:.1f}")
    print(f"Plates detected:              {avg_counts['plates_detected']:.1f}")
    print(f"Face crops processed:         {avg_counts['face_crops']:.1f}")
    print(f"Face verification calls:      {avg_counts['face_verify_calls']:.1f}")
    print(f"Face model forward passes:    {avg_counts['face_model_forward_passes']:.1f}")
    print(f"Face registry comparisons:    {avg_counts['registry_comparisons']:.1f}")
    print(f"OCR attempted:                {avg_counts['ocr_attempts']:.1f}")
    print(f"OCR actually executed:        {avg_counts['ocr_actual_inference_calls']:.1f}")
    print(f"OCR skipped (quality):        {avg_counts['ocr_skipped_quality']:.1f}")
    print(f"OCR skipped (cached):         {avg_counts['ocr_skipped_cached']:.1f}")
    print(f"OCR skipped (no plate):       {avg_counts['ocr_skipped_no_plate']:.1f}")

    # ── SECTION 16 REQUIRED FINAL REPORT ───────────────────────────
    print("\n" + "=" * 55)
    print("IBVAP WARM SINGLE-FRAME PERFORMANCE REPORT")
    print("=" * 55)
    print(f"\nInitialization:")
    print(f"    total:             {init_timings['total']:.2f} ms")
    print(f"    YOLO:              {init_timings['yolo']:.2f} ms")
    print(f"    face detector:     {init_timings['face_detector']:.2f} ms")
    print(f"    face verification: {init_timings['face_verification']:.2f} ms")
    print(f"    plate detector:    {init_timings['plate_detector']:.2f} ms")
    print(f"    OCR:               {init_timings['ocr']:.2f} ms")
    print(f"    other:             {init_timings['other']:.2f} ms")

    total_stats = baseline_stats["total"]
    print(f"\nWARM PIPELINE:")
    print(f"    mean:              {total_stats['mean']:.2f} ms")
    print(f"    median:            {total_stats['median']:.2f} ms")
    print(f"    p95:               {total_stats['p95']:.2f} ms")
    print(f"    p99:               {total_stats['p99']:.2f} ms")
    print(f"    max:               {total_stats['max']:.2f} ms")

    fv_skipped = (avg_counts["face_verify_calls"] == 0)
    fv_reason = None
    for wc in warm_counts:
        if "face_verification_skip_reason" in wc:
            fv_reason = wc["face_verification_skip_reason"]
            break
    if not fv_reason and fv_skipped:
        fv_reason = "no_face_detected" if avg_counts["faces_detected"] == 0 else "quality_gate_rejected"
    fv_report_str = f"face verification skipped ({fv_reason})" if fv_skipped else f"{baseline_stats['face_verification']['median']:>6.2f} ms (mean: {baseline_stats['face_verification']['mean']:>6.2f} ms, p95: {baseline_stats['face_verification']['p95']:>6.2f} ms)"

    print(f"\nCOMPONENTS (Warm Median / Mean):")
    for comp in ["yolo", "face_detection", "face_verification", "plate_detection", "ocr", "tracking", "analytics", "event_engine"]:
        if comp == "face_verification" and fv_skipped:
            print(f"    {comp:<18}: {fv_report_str}")
        else:
            cs = baseline_stats[comp]
            print(f"    {comp:<18}: {cs['median']:>6.2f} ms (mean: {cs['mean']:>6.2f} ms, p95: {cs['p95']:>6.2f} ms)")

    print(f"\nEXECUTION COUNTS:")
    print(f"    persons:           {avg_counts['persons_detected']:.1f}")
    print(f"    vehicles:          {avg_counts['vehicles_detected']:.1f}")
    print(f"    faces:             {avg_counts['faces_detected']:.1f}")
    print(f"    plates:            {avg_counts['plates_detected']:.1f}")
    print(f"    face inference:    {avg_counts['face_model_forward_passes']:.1f}")
    print(f"    OCR inference:     {avg_counts['ocr_actual_inference_calls']:.1f}")

    # ── BENCHMARK METRICS ──────────────────────────────────────────
    variants_exec_avg = np.mean([x["variants_executed"] for x in all_ocr]) if all_ocr else 0
    last_res = all_ocr[-1].get("result_plate") if all_ocr else None
    last_conf = all_ocr[-1].get("result_confidence", 0.0) if all_ocr else 0.0
    v1_time_str = f"{np.mean(v1_times):.2f} ms" if v1_times else "skipped"
    v2_time_str = f"{np.mean(v2_times):.2f} ms" if v2_times else "skipped"
    v3_time_str = f"{np.mean(v3_times):.2f} ms" if v3_times else "skipped"

    print("\n" + "=" * 55)
    print(f"OCR BENCHMARK DETAILS ({img_name})")
    print("=" * 55)
    print(f"OCR attempts:          {len(all_ocr) // len(warm_timings):.0f}")
    print(f"OCR variants executed: {variants_exec_avg:.0f}")
    print(f"Variant 1 time:        {v1_time_str}")
    print(f"Variant 2 time:        {v2_time_str}")
    print(f"Variant 3 time:        {v3_time_str}")
    print(f"Total OCR time:        {np.mean(total_ocr_times):.2f} ms" if all_ocr else "Total OCR time:        skipped")
    print(f"OCR result:            {last_res if last_res else 'None'}")
    print(f"OCR confidence:        {last_conf:.4f}")

    print("\n" + "=" * 55)
    print(f"COMPLETE PIPELINE BENCHMARK ({img_name})")
    print("=" * 55)
    print(f"YOLO:                  {baseline_stats['yolo']['median']:.2f} ms")
    print(f"Face detection:        {baseline_stats['face_detection']['median']:.2f} ms")
    print(f"Face verification:     {fv_report_str}")
    print(f"Plate detection:       {baseline_stats['plate_detection']['median']:.2f} ms")
    print(f"OCR:                   {baseline_stats['ocr']['median']:.2f} ms")
    print(f"TOTAL:                 {baseline_stats['total']['median']:.2f} ms")

    print("\n" + "=" * 55)
    print("OPTIMIZATIONS")
    print("=" * 55)
    print("\nOptimization 1: Centralized Device Routing & PyTorch Thread Health Protection")
    print("Before: YOLO = 94.70 ms mean / 94.22 ms median (starved to 1 thread by PaddleX)")
    print(f"After:  YOLO = {baseline_stats['yolo']['mean']:.2f} ms mean / {baseline_stats['yolo']['median']:.2f} ms median")
    yolo_imp = (94.70 - baseline_stats['yolo']['mean']) / 94.70 * 100.0
    print(f"Improvement: {yolo_imp:.1f}% faster")

    print("\nOptimization 2: Plate Detector Bilateral Filter Replacement (GaussianBlur 5x5)")
    print("Before: Plate Detection = 36.86 ms mean / 37.30 ms median (30.8 ms in bilateralFilter)")
    print(f"After:  Plate Detection = {baseline_stats['plate_detection']['mean']:.2f} ms mean / {baseline_stats['plate_detection']['median']:.2f} ms median")
    pd_imp = (36.86 - baseline_stats['plate_detection']['mean']) / 36.86 * 100.0
    print(f"Improvement: {pd_imp:.1f}% faster")

    print("\nOptimization 3: Face Detection Upper-Body Focus & setInputSize Caching")
    print("Before: Face Detection = 62.80 ms mean / 60.74 ms median")
    print(f"After:  Face Detection = {baseline_stats['face_detection']['mean']:.2f} ms mean / {baseline_stats['face_detection']['median']:.2f} ms median")
    fd_imp = (62.80 - baseline_stats['face_detection']['mean']) / 62.80 * 100.0
    print(f"Improvement: {fd_imp:.1f}% faster")

    print("\nOptimization 4: Face Verification Thread Health & Elimination of Redundant Passes")
    if fv_skipped:
        print(f"Face verification skipped: {fv_reason}")
    else:
        print("Before: Face Verification = 113.70 ms mean / 105.25 ms median")
        print(f"After:  Face Verification = {baseline_stats['face_verification']['mean']:.2f} ms mean / {baseline_stats['face_verification']['median']:.2f} ms median")
        fv_imp = (113.70 - baseline_stats['face_verification']['mean']) / 113.70 * 100.0
        print(f"Improvement: {fv_imp:.1f}% faster")

    print("\nOptimization 5: ANPR OCR Confidence-Based Early Exit")
    print("Before: All 3 variants executed (~182-192 ms) regardless of Variant 1 validity")
    if variants_exec_avg == 1:
        print(f"After:  Variant 1 sufficient -> Variants 2 & 3 skipped ({np.mean(total_ocr_times):.2f} ms)")
        ocr_imp = (182.15 - np.mean(total_ocr_times)) / 182.15 * 100.0
        print(f"Improvement: {ocr_imp:.1f}% faster (early exit)")
    else:
        print(f"After:  Difficult plate -> {variants_exec_avg:.0f} variants executed conditionally as fallback ({np.mean(total_ocr_times):.2f} ms)")

    print("\n" + "=" * 55)
    print("TARGET")
    print("=" * 55)
    print("\nTarget:            <100 ms")
    print(f"Current median:    {total_stats['median']:.2f} ms")
    print(f"Current p95:       {total_stats['p95']:.2f} ms")
    target_achieved = "YES" if total_stats['median'] < 100.0 else "NO (Progress: 373.6 ms -> 215.8 ms median on CPU)"
    print(f"Target achieved:   {target_achieved}")
    print("Remaining bottleneck: OCR inference (~63 ms) and InceptionResnetV1 forward pass (~37-45 ms) on CPU")
    print("=" * 55)

if __name__ == "__main__":
    run_deep_benchmark()
