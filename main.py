import os
import sys
import asyncio
import cv2
import numpy as np

import stream
import discovery
import ibvap.core.pipeline as pipeline
from ibvap.core.types import VirtualBoundary, ZoneType, WatchlistCategory
from ibvap.core.config import IBVAPConfig

REF_FACE_IMAGE = r"C:\Users\amitm\OneDrive\Pictures\Camera Roll\WIN_20260904_13_43_05_Pro.jpg"


def register_reference_face(processor, ref_path):
    """Registers an authorized face into the pipeline from a reference photo."""
    if not os.path.exists(ref_path):
        print(f"Warning: Reference image not found at: {ref_path}")
        return False

    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"Failed to read image at: {ref_path}")
        return False

    # Extract person head/face crop for clean embedding
    dets = processor.detector.detect(ref_img)
    if dets:
        x1, y1, x2, y2 = dets[0].bbox
        p_crop = ref_img[y1:y2, x1:x2]
        ph, pw = p_crop.shape[:2]
        face_crop = p_crop[0:int(ph * 0.45), 0:pw]
    else:
        face_crop = ref_img

    success = processor.register_authorized_person(
        identity_id="USER-01",
        name="Amit",
        face_bgr_image=face_crop,
        role="AUTHORIZED"
    )
    if success:
        print(f"[+] Successfully registered face for 'Amit' (USER-01) from: {ref_path}")
    return success


async def survillance():
    """
    RTSP ONVIF camera discovery and live surveillance stream processing
    with local webcam fallback and visual overlay.
    """
    processor = pipeline.IBVAPPipeline()

    # 1. Register Reference Face into Biometric Database
    register_reference_face(processor, REF_FACE_IMAGE)

    # 2. Discover available cameras on network
    print("[*] Scanning network for CCTV / Phone cameras...")
    devices = discovery.discover(timeout=3)
    cams = []
    if devices:
        cams = await asyncio.gather(*(stream.connect_camera(d) for d in devices))

    # Determine video source (Discovered RTSP or Local Webcam fallback)
    cap = None
    stream_url = None

    for c in cams:
        if c:
            try:
                stream_url = await stream.rtsp_url(c)
                print(f"[+] Connected to Discovered Camera Stream: {stream_url}")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                break
            except Exception as e:
                print(f"Failed to start stream on camera: {e}")

    if cap is None or not cap.isOpened():
        print("[!] No RTSP camera found. Falling back to local webcam (Device 0)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open any camera source.")
        return

    print("\n" + "=" * 65)
    print("  🚀 LIVE AI SURVEILLANCE & BIOMETRIC TRACKING ACTIVE")
    print("  - Face Recognition: Registered for 'Amit'")
    print("  - Bounding Boxes: Green for Person, Cyan for Vehicle")
    print("  - Press 'q' or 'ESC' in the video window to stop")
    print("=" * 65 + "\n")

    PROCESS_EVERY_N_FRAMES = 5  # Responsive AI processing interval
    frame_counter = 0
    last_result = None
    boundary_initialized = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to receive frames or stream interrupted.")
                break

            actual_h, actual_w = frame.shape[:2]

            # Setup spatial boundaries once frame size is known
            if not boundary_initialized:
                mid_x = actual_w // 2
                processor.add_boundary(
                    VirtualBoundary(
                        id="fence-center",
                        name="Center Tripwire Line",
                        zone_type=ZoneType.LINE,
                        coordinates=[(mid_x, 0), (mid_x, actual_h)],
                        target_classes=["person", "car", "motorcycle"]
                    )
                )
                boundary_initialized = True

            frame_counter += 1

            # Run AI pipeline periodically (every N frames)
            if frame_counter % PROCESS_EVERY_N_FRAMES == 0 or last_result is None:
                last_result = processor.process_frame(frame)

                # Print face match events in console
                if last_result and last_result.events:
                    for ev in last_result.events:
                        if ev.event_type.value == "FACE_MATCHED":
                            print(f"✨ [MATCH CONFIRMED] {ev.metadata.get('name', 'Known Person')} (Sim: {ev.confidence:.2f})")
                        elif "INTRUSION" in ev.event_type.value or "LOITERING" in ev.event_type.value:
                            print(f"🚨 [THREAT DETECTED] {ev.event_type.value} | Track #{ev.track_id}")

            # Render live visual AI overlay on video
            if last_result is not None:
                annotated = processor.draw_debug(frame, last_result)
            else:
                annotated = frame

            cv2.imshow("IBVAP CCTV Live Surveillance & Biometrics (Press 'q' to quit)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


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
    # If image path argument is provided, run single image analysis
    if len(sys.argv) > 1 and sys.argv[1].strip():
        output = test_images(sys.argv[1])
        print("\nReturned Result:")
        print(output)
    else:
        # Check if default test image exists, otherwise run live surveillance
        default_test_image = r"C:\ibvap\akshat.jpeg"
        if os.path.exists(default_test_image):
            print(f"Running test on default image: {default_test_image}")
            output = test_images(default_test_image)
            print("\nReturned Result:")
            print(output)
        else:
            asyncio.run(survillance())
