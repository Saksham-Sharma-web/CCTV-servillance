import os
import sys
import asyncio
import cv2
import numpy as np

import ibvap.core.pipeline as pipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import WatchlistCategory


async def survillance():
    """
    RTSP ONVIF camera discovery and live surveillance stream processing.
    Imports network streaming modules lazily so test_images runs independently.
    """
    try:
        import stream
        import discovery
    except ImportError as e:
        print(f"[ERROR] Live camera discovery requires optional ONVIF packages: {e}")
        return

    devices = discovery.discover()
    cams = await asyncio.gather(*(stream.connect_camera(d) for d in devices))
    processor = pipeline.IBVAPPipeline()

    for c in cams:
        if c:
            url = await stream.rtsp_url(c)
            cap = cv2.VideoCapture(url)
            print("Processing started!")
            print("Press Ctrl+C to stop")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to receive frames")
                    break
                result = processor.process_frame(frame)
                print(result)
                print()
            await c.close()


def test_images(path: str):
    """
    Sequentially analyzes an image one-by-one:
    1. Check and decode image
    2. Face detection: detect whether there is a face or not
    3. Vehicle and license-plate detection + OCR: extract vehicle type and car number
    4. Match face and car number against registered authorized profiles / watchlists
    """
    if not path or not path.strip():
        print("[test_images] No image path provided. Please provide a path to an image file.")
        return {
            "error": "No image path provided",
            "vehicle_detected": False,
            "license_plate_detected": False,
            "face_detected": False,
        }

    clean_path = path.strip().strip('"').strip("'")
    if not os.path.exists(clean_path):
        print(f"[test_images] File not found: '{clean_path}'")
        return {
            "error": f"File not found: {clean_path}",
            "vehicle_detected": False,
            "license_plate_detected": False,
            "face_detected": False,
        }

    frame = cv2.imread(clean_path, cv2.IMREAD_COLOR)
    if frame is None:
        print(f"[test_images] Failed to decode image from '{clean_path}'")
        return {
            "error": "Failed to decode image",
            "vehicle_detected": False,
            "license_plate_detected": False,
            "face_detected": False,
        }

    h, w = frame.shape[:2]
    print("=" * 60)
    print(f"STEP 1: Loaded Image '{clean_path}' ({w}x{h} px)")
    print("=" * 60)

    processor = pipeline.IBVAPPipeline()

    # ── STEP 2: Face Detection (One-by-one) ───────────────────────────
    print("\n--- [STEP 2] Face Detection ---")
    faces = processor.face_detector.detect(frame)
    face_detected = len(faces) > 0
    face_details = []
    print(f"Face Detected: {face_detected} (Count: {len(faces)})")

    for idx, (fx1, fy1, fx2, fy2, fconf) in enumerate(faces):
        face_crop = frame[fy1:fy2, fx1:fx2]
        matched_person, sim = processor.identity_verifier.verify_crop(face_crop)
        person_name = matched_person.name if matched_person else "Unknown"
        face_details.append({
            "index": idx + 1,
            "bbox": [fx1, fy1, fx2, fy2],
            "confidence": round(float(fconf), 4),
            "identity": person_name,
            "similarity": round(float(sim), 4) if sim else 0.0,
        })
        print(f"  Face #{idx + 1}: bbox=[{fx1}, {fy1}, {fx2}, {fy2}], conf={round(fconf, 3)}, identity='{person_name}'")

    # ── STEP 3: Vehicle & License Plate Detection + OCR ──────────────
    print("\n--- [STEP 3] Vehicle & License Plate Detection ---")
    result = processor.process_frame(frame, camera_id="test-cam")

    vehicle_detected = result.vehicle_detected
    vehicle_type = result.vehicle_type
    vehicle_conf = result.vehicle_confidence
    plate_detected = result.license_plate_detected
    plate_number = result.license_plate
    plate_conf = result.plate_confidence
    ocr_conf = result.ocr_confidence

    print(f"vehicle_detected: {vehicle_detected}")
    if vehicle_detected:
        print(f"vehicle_type: '{vehicle_type}'")
        print(f"vehicle_confidence: {vehicle_conf}")
    print(f"license_plate_detected: {plate_detected}")
    if plate_detected:
        print(f"license_plate: '{plate_number}'")
        print(f"plate_confidence: {plate_conf}")
        print(f"ocr_confidence: {ocr_conf}")

    # ── STEP 4: Correlation / Matching ───────────────────────────────
    print("\n--- [STEP 4] Face and Vehicle Number Correlation ---")
    driver_identity = face_details[0]["identity"] if face_details else "No face detected"
    matched_status = "UNVERIFIED"
    if face_detected and plate_detected:
        matched_status = f"Associated: Driver '{driver_identity}' with Vehicle '{plate_number}'"
    elif face_detected and not plate_detected:
        matched_status = f"Driver '{driver_identity}' detected, but no license plate visible"
    elif plate_detected and not face_detected:
        matched_status = f"Vehicle '{plate_number}' detected without visible occupant face"
    else:
        matched_status = "Neither occupant face nor license plate recognized"

    print(f"Match Summary: {matched_status}")
    print("=" * 60)

    structured_output = {
        "vehicle_detected": vehicle_detected,
        "vehicle_type": vehicle_type,
        "vehicle_confidence": vehicle_conf,
        "license_plate_detected": plate_detected,
        "license_plate": plate_number,
        "plate_confidence": plate_conf,
        "ocr_confidence": ocr_conf,
        "face_detected": face_detected,
        "face_count": len(faces),
        "faces": face_details,
        "match_summary": matched_status,
        "pipeline_result": result.to_dict(),
    }
    return structured_output


if __name__ == "__main__":
    # Test path can be passed as argument or pasted directly here:
    test_path = sys.argv[1] if len(sys.argv) > 1 else ""
    output = test_images(test_path)
    print("\nReturned Result:")
    print(output)
