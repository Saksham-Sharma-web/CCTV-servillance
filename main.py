import os
import sys
import asyncio
import cv2

try:
    import stream
    import discovery
except ImportError:
    stream = None
    discovery = None

import ibvap.core.pipeline as pipeline


# ═════════════════════════════════════════════════════════════════════
# 1. REGISTER YOUR PEOPLE HERE (ONCE)
# ═════════════════════════════════════════════════════════════════════
# Format: (Name, Image Path, Age Category: "most_recent", "recent", "old")
# Works with relative paths, filenames in current directory, or reference_faces/
REGISTERED_PEOPLE = [
    ("Akshat", "test_akshat1.jpeg", "most_recent"),
    ("Akshat", "test_akshat2.jpeg", "recent"),
    # ("Akshat", r"C:\ibvap\akshat.jpeg", "most_recent"),
]


# ═════════════════════════════════════════════════════════════════════
# 2. DEDICATED SETUP FUNCTION (Handles Enrollment Automatically)
# ═════════════════════════════════════════════════════════════════════
def create_pipeline(people=None):
    """
    Creates and initializes the IBVAP Pipeline with all registered people.
    Enrolls from:
    1. The REGISTERED_PEOPLE list above
    2. Any photos placed in the 'reference_faces/' folder
    """
    processor = pipeline.IBVAPPipeline()
    detector_type = getattr(processor.face_detector, "active_detector_type", "unknown").upper()
    print(f"[*] Biometric Engine: {detector_type} Face Detector active")

    enroll_list = people if people is not None else REGISTERED_PEOPLE
    enrolled_count = 0

    # 1. Enroll from the configured list
    for item in enroll_list:
        name = item[0]
        path = item[1]
        age = item[2] if len(item) > 2 else "most_recent"

        # Resolve path: literal, relative to CWD, or inside reference_faces/
        resolved = path
        if not os.path.exists(resolved):
            # Extract clean filename regardless of Windows/POSIX backslashes
            base_fname = path.replace("\\", "/").split("/")[-1]
            candidates = [
                os.path.join(os.getcwd(), path),
                os.path.join(os.getcwd(), base_fname),
                os.path.join(os.getcwd(), "reference_faces", base_fname),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), base_fname),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_faces", base_fname),
            ]
            # Also search for any files matching name in CWD or reference_faces/
            name_lower = name.lower()
            for directory in (os.getcwd(), os.path.join(os.getcwd(), "reference_faces")):
                if os.path.exists(directory):
                    for f in os.listdir(directory):
                        if f.lower().endswith((".jpg", ".jpeg", ".png")) and name_lower in f.lower():
                            candidates.append(os.path.join(directory, f))

            found = next((c for c in candidates if os.path.exists(c)), None)
            if found:
                resolved = found

        if os.path.exists(resolved):
            ok, msg = processor.register_reference_image(name=name, image_path=resolved, reference_age=age)
            if ok:
                print(f"[+] Enrolled '{name}' [{age}] from: {resolved}")
                enrolled_count += 1
            else:
                print(f"[!] Could not enroll '{name}' from {resolved}: {msg}")
        else:
            print(f"[!] Image file not found for '{name}': {path}")

    # 2. Auto-enroll any photos dropped into 'reference_faces/' folder
    ref_dir = os.path.join(os.getcwd(), "reference_faces")
    if os.path.exists(ref_dir):
        for fname in os.listdir(ref_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(ref_dir, fname)
                base = os.path.splitext(fname)[0]
                parts = base.split("_")
                p_name = parts[0]
                p_age = parts[-1].lower() if len(parts) > 1 and parts[-1].lower() in ("most_recent", "recent", "old") else "most_recent"
                ok, _ = processor.register_reference_image(name=p_name, image_path=fpath, reference_age=p_age)
                if ok:
                    print(f"[+] Auto-Enrolled from folder: '{p_name}' [{p_age}] from: {fname}")
                    enrolled_count += 1

    if enrolled_count == 0:
        print("[WARNING] No reference profiles enrolled! Persons on camera will be labeled UNKNOWN.")
    else:
        print(f"[+] Ready: {enrolled_count} biometric profile(s) active.\n")

    return processor


# ═════════════════════════════════════════════════════════════════════
# 3. LIVE SURVEILLANCE CAMERA (stream.py)
# ═════════════════════════════════════════════════════════════════════
async def survillance():
    """
    Connects to surveillance cameras discovered on the network
    and runs real-time person & face recognition.
    """
    cap = None
    cams = []
    stream_source_name = "None"

    if stream is not None and discovery is not None:
        try:
            print("[*] Scanning network for CCTV / Phone cameras...")
            devices = discovery.discover(timeout=3)
            cams = await asyncio.gather(*(stream.connect_camera(d) for d in devices)) if devices else []
            for c in cams:
                if c:
                    try:
                        url = await stream.rtsp_url(c)
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        if cap.isOpened():
                            stream_source_name = f"RTSP Stream ({url})"
                            print(f"[+] Connected to Discovered Camera Stream: {url}")
                            break
                    except Exception as e:
                        print(f"[!] Could not connect to camera: {e}")
        except Exception as e:
            print(f"[!] Discovery error: {e}")

    # Fallback to local webcam (Device 0) if no RTSP camera is connected
    if cap is None or not cap.isOpened():
        print("[*] No network RTSP camera found. Falling back to local webcam (Device 0)...")
        cap = cv2.VideoCapture(0)
        stream_source_name = "Local Webcam (Device 0)"

    if not cap.isOpened():
        print("[ERROR] Could not open any camera source (neither RTSP nor Webcam).")
        for c in cams:
            if c:
                try:
                    await c.close()
                except Exception:
                    pass
        return

    processor = create_pipeline()

    print(f"\n[+] Active Video Source: {stream_source_name}")
    print("Surveillance Camera started! Press 'q' or 'ESC' in the window to stop.\n")

    announced_vehicles = set()
    logged_plates = {}
    logged_matches = set()
    logged_unknowns = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to receive frames.")
                break

            result = processor.process_frame(frame)

            # Real-time console logs for detected persons and vehicles (debounced to avoid terminal flooding)
            if result and result.tracks:
                for track in result.tracks:
                    if track.class_name == "person":
                        if track.identity_id:
                            if track.track_id not in logged_matches:
                                logged_matches.add(track.track_id)
                                print(f"✨ [MATCH] Found: '{track.identity_name}' (Confidence: {track.identity_confidence:.2f}) | Track #{track.track_id}")
                        elif track.identity_confidence and track.identity_confidence > 0:
                            if track.track_id not in logged_unknowns:
                                logged_unknowns.add(track.track_id)
                                print(f"👤 [UNKNOWN] Track #{track.track_id} (Sim: {track.identity_confidence:.2f} < 0.65)")
                    elif track.class_name in ("car", "truck", "bus", "motorcycle"):
                        if track.plate_number:
                            if track.track_id not in logged_plates or logged_plates[track.track_id] != track.plate_number:
                                logged_plates[track.track_id] = track.plate_number
                                cat = f"[{track.plate_category.value}] " if track.plate_category else ""
                                print(f"🚗 [ANPR] {track.class_name.upper()} #{track.track_id} | {cat}Plate: {track.plate_number} (Conf: {track.plate_confidence:.2f})")
                        else:
                            if track.track_id not in announced_vehicles:
                                announced_vehicles.add(track.track_id)
                                print(f"🚗 [VEHICLE] {track.class_name.upper()} #{track.track_id} detected | Scanning for license plate...")

            # Live visual overlay with bounding boxes
            annotated = processor.draw_debug(frame, result)
            cv2.imshow("IBVAP CCTV Live Surveillance (Press 'q' to quit)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        for c in cams:
            if c:
                try:
                    await c.close()
                except Exception:
                    pass


# ═════════════════════════════════════════════════════════════════════
# 4. SINGLE / STATIC IMAGE TEST
# ═════════════════════════════════════════════════════════════════════
def test_images(path=None, references=None, target=None, debug=False):
    """
    Tests face detection, vehicle detection, and ANPR on an image file.
    Supports both:
      test_images("path/to/img.jpg")
      test_images(references=[("Name", "path", "most_recent")], target="path/to/img.jpg")
    """
    img_path = target or path
    if not img_path:
        print("[!] No image path provided.")
        return {"error": "No image path provided", "face_decision": "NO_FACE_DETECTED", "identity": None}

    processor = create_pipeline(people=references)
    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if frame is None:
        print(f"[!] Could not read image at: {img_path}")
        return {"error": f"Could not read {img_path}", "face_decision": "NO_FACE_DETECTED", "identity": None}

    result = processor.process_frame(frame)

    person_tracks = [t for t in result.tracks if t.class_name == "person"]
    matched_tracks = [t for t in person_tracks if t.identity_id]
    vehicle_tracks = [t for t in result.tracks if t.class_name in ("car", "truck", "bus", "motorcycle")]

    print("-" * 50)
    print(f"ANALYSIS RESULTS FOR: {img_path}")
    print("-" * 50)
    for track in person_tracks:
        status = f"MATCH: {track.identity_name}" if track.identity_id else "UNKNOWN PERSON"
        conf = f"{track.identity_confidence:.2f}" if track.identity_confidence is not None else "0.00"
        print(f"  • {status} (Confidence: {conf}) | Box: {track.bbox}")

    for track in vehicle_tracks:
        plate_str = f"Plate: {track.plate_number} (Conf: {track.plate_confidence:.2f})" if track.plate_number else "Plate: Scanning / Not detected"
        print(f"  • VEHICLE: {track.class_name.upper()} #{track.track_id} | {plate_str} | Box: {track.bbox}")

    if not person_tracks and not vehicle_tracks:
        print("  • No person or vehicle detected.")
    print("-" * 50)

    # If debug mode, save annotated overlay to disk
    if debug:
        annotated = processor.draw_debug(frame, result)
        out_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "debug_analysis.jpg"), annotated)

    top_track = matched_tracks[0] if matched_tracks else (person_tracks[0] if person_tracks else None)
    best_sim = top_track.identity_confidence if (top_track and top_track.identity_confidence is not None) else 0.0

    best_ref_age = "most_recent"
    if top_track and top_track.identity_id and top_track.identity_id in processor.identity_verifier.authorized_registry:
        person_obj = processor.identity_verifier.authorized_registry[top_track.identity_id]
        if person_obj.references:
            best_ref_age = person_obj.references[-1].reference_age

    body_status = "BODY_SUPPORTING" if (best_ref_age != "old" and best_sim >= 0.65) else ("BODY_IGNORED" if best_ref_age == "old" else "BODY_NOT_DETECTED")
    decision = "MATCH" if (top_track and top_track.identity_id) else ("UNKNOWN" if person_tracks else "NO_FACE_DETECTED")

    return {
        "target": img_path,
        "face_decision": decision,
        "identity": top_track.identity_name if (top_track and top_track.identity_id) else None,
        "face_detected": len(person_tracks) > 0,
        "face_confidence": round(float(top_track.confidence), 4) if top_track else 0.0,
        "face_similarity": round(float(best_sim), 4),
        "best_reference_age": best_ref_age,
        "body_status": body_status,
        "body_role": "SUPPORTING ONLY",
        "result": result
    }


# ═════════════════════════════════════════════════════════════════════
# 5. ENTRY POINT
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # If image path argument given (e.g. python main.py "path/to/img.jpg" or python main.py --test), test it:
    if len(sys.argv) > 1 and sys.argv[1].strip() not in ("--stream", "stream", "cam"):
        arg = sys.argv[1].strip()
        if arg in ("--test", "-t"):
            cand = next((p for p in ["test_car.png", "car.png", r"C:\ibvap\car.png", "test_akshat1.jpeg", r"C:\ibvap\akshat.jpeg", "reference_faces/akshat_most_recent.jpeg"] if os.path.exists(p)), None)
            test_images(cand or "test_car.png")
        else:
            test_images(arg)
    else:
        # Default behavior: run real-time camera stream
        print("[*] Hint: To test an image directly without camera, run: python main.py <path_to_image>")
        print("[*] Example: python main.py test_car.png\n")
        asyncio.run(survillance())
