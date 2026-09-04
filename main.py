import os
import sys
import asyncio
from typing import List, Tuple, Optional, Dict, Any
import cv2
import numpy as np

# Optional stream/discovery dependencies for CCTV surveillance
try:
    import stream
    import discovery
except ImportError:
    stream = None
    discovery = None

import ibvap.core.pipeline as pipeline
from ibvap.core.types import VirtualBoundary, ZoneType, WatchlistCategory
from ibvap.core.config import IBVAPConfig
from ibvap.face.detector import OpenCVFaceDetector
from ibvap.face.matcher_adapter import IdentityVerifierAdapter, BodyAppearanceExtractor, align_face_160

# Default reference photos to enroll into the live surveillance pipeline
# Format: (Person Name, File Path, Reference Age: "most_recent", "recent", "old")
DEFAULT_STREAM_REFERENCES = [
    ("Akshat", r"C:\ibvap\akshat.jpeg", "most_recent"),
]


def enroll_surveillance_references(processor: pipeline.IBVAPPipeline) -> int:
    """
    Enrolls reference photos into the live surveillance pipeline.
    Checks:
    1. DEFAULT_STREAM_REFERENCES list
    2. 'reference_faces/' directory for any dropped photos
    """
    enrolled_count = 0

    # 1. Enroll configured list
    for name, path, age in DEFAULT_STREAM_REFERENCES:
        if os.path.exists(path):
            ok, msg = processor.register_reference_image(
                name=name,
                image_path=path,
                reference_age=age
            )
            if ok:
                print(f"[+] Biometric Profile Enrolled: '{name}' [{age}] from: {path}")
                enrolled_count += 1
            else:
                print(f"[!] Could not enroll '{name}' from {path}: {msg}")

    # 2. Check reference_faces directory
    ref_dir = os.path.join(os.getcwd(), "reference_faces")
    if os.path.exists(ref_dir):
        for fname in os.listdir(ref_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(ref_dir, fname)
                base = os.path.splitext(fname)[0]
                parts = base.split("_")
                p_name = parts[0]
                p_age = "most_recent"
                if len(parts) > 1 and parts[-1].lower() in ("old", "recent", "most_recent"):
                    p_age = parts[-1].lower()
                ok, msg = processor.register_reference_image(
                    name=p_name,
                    image_path=fpath,
                    reference_age=p_age
                )
                if ok:
                    print(f"[+] Biometric Profile Enrolled from folder: '{p_name}' [{p_age}] from: {fname}")
                    enrolled_count += 1

    if enrolled_count == 0:
        print("[WARNING] No reference profiles enrolled! All persons seen on camera will be labeled UNKNOWN.")
        print("          Drop photos into the 'reference_faces/' folder or configure DEFAULT_STREAM_REFERENCES.")
    else:
        print(f"[+] Total Enrolled Profiles: {enrolled_count}")

    return enrolled_count


async def survillance():
    """
    RTSP ONVIF camera discovery and live surveillance stream processing
    with local webcam fallback and visual overlay.
    """
    if stream is None or discovery is None:
        print("[!] Network discovery/stream modules not available. Exiting surveillance loop.")
        return

    processor = pipeline.IBVAPPipeline()

    # 1. Register Reference Faces into Biometric Database
    enroll_surveillance_references(processor)

    # 2. Discover available cameras on network
    print("[*] Scanning network for CCTV / Phone cameras...")
    devices = discovery.discover(timeout=3)
    cams = []
    if devices:
        cams = await asyncio.gather(*(stream.connect_camera(d) for d in devices))

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
    print("  LIVE AI SURVEILLANCE & BIOMETRIC TRACKING ACTIVE")
    print("  - Face Recognition: Registered for 'Amit'")
    print("  - Bounding Boxes: Green for Person, Cyan for Vehicle")
    print("  - Press 'q' or 'ESC' in the video window to stop")
    print("=" * 65 + "\n")

    PROCESS_EVERY_N_FRAMES = 5
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

            if frame_counter % PROCESS_EVERY_N_FRAMES == 0 or last_result is None:
                last_result = processor.process_frame(frame)

                # Real-time console logs for persons detected in camera frames
                if last_result and last_result.tracks:
                    for track in last_result.tracks:
                        if track.class_name == "person":
                            if track.identity_id:
                                print(f"✨ [CAMERA IDENTIFIED] '{track.identity_name}' (Confidence: {track.identity_confidence:.2f}) | Track #{track.track_id}")
                            elif track.identity_confidence and track.identity_confidence > 0:
                                print(f"👤 [CAMERA UNKNOWN] Track #{track.track_id} (Similarity: {track.identity_confidence:.2f} < 0.65 threshold)")

                if last_result and last_result.events:
                    for ev in last_result.events:
                        if ev.event_type.value == "FACE_MATCHED":
                            print(f"  [MATCH CONFIRMED] {ev.metadata.get('name', 'Known Person')} (Sim: {ev.confidence:.2f})")
                        elif "INTRUSION" in ev.event_type.value or "LOITERING" in ev.event_type.value:
                            print(f"  [THREAT DETECTED] {ev.event_type.value} | Track #{ev.track_id}")

            annotated = processor.draw_debug(frame, last_result) if last_result else frame
            cv2.imshow("IBVAP CCTV Live Surveillance & Biometrics (Press 'q' to quit)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def test_images(
    *args,
    references: Optional[List[Tuple[str, str, str]]] = None,
    target: Optional[str] = None,
    debug: bool = False,
    path: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    High-accuracy face analysis and supporting body appearance verification.
    Supports multiple reference images per identity with reference-age categories
    ('most_recent', 'recent', 'old') and backward compatibility with single-image paths.

    Example:
        test_images(
            references=[
                ("amit", r"C:\\ibvap\\amit1.jpeg", "most_recent"),
                ("amit", r"C:\\ibvap\\amit2.jpeg", "recent"),
                ("amit", r"C:\\ibvap\\amit3.jpeg", "old"),
                ("rahul", r"C:\\ibvap\\rahul.jpeg", "most_recent"),
            ],
            target=r"C:\\ibvap\\target.jpeg",
            debug=True
        )
    """
    # ── Argument Normalization for Backward Compatibility ─────────────
    if len(args) > 0:
        if isinstance(args[0], list):
            references = args[0]
            if len(args) > 1 and isinstance(args[1], str):
                target = args[1]
        elif isinstance(args[0], str):
            target = args[0]

    target_path = target or path
    if target_path:
        target_path = target_path.strip().strip('"').strip("'")

    # If no target specified
    if not target_path:
        print("[test_images] No target image path provided.")
        return {
            "error": "No target image path provided",
            "face_decision": "NO_FACE_DETECTED",
            "identity": None,
            "face_confidence": 0.0,
            "face_similarity": 0.0,
        }

    if not os.path.exists(target_path):
        print(f"[test_images] Target file not found: '{target_path}'")
        return {
            "error": f"File not found: {target_path}",
            "face_decision": "NO_FACE_DETECTED",
            "identity": None,
            "face_confidence": 0.0,
            "face_similarity": 0.0,
        }

    target_img = cv2.imread(target_path)
    if target_img is None or target_img.size == 0:
        print(f"[test_images] Failed to decode target image from: '{target_path}'")
        return {
            "error": f"Failed to decode image from {target_path}",
            "face_decision": "NO_FACE_DETECTED",
            "identity": None,
        }

    th, tw = target_img.shape[:2]

    # Initialize components
    config = IBVAPConfig()
    detector = OpenCVFaceDetector(config)
    verifier = IdentityVerifierAdapter(config)

    # ── 1. Process References First (Strict Biometric Invariant) ──────
    enrolled_references_info = []
    if references:
        for ref_item in references:
            if len(ref_item) == 3:
                r_name, r_path, r_age = ref_item
            elif len(ref_item) == 2:
                r_name, r_path = ref_item
                r_age = "most_recent"
            else:
                continue

            r_path_clean = r_path.strip().strip('"').strip("'")
            ok, status_msg = verifier.register_reference(
                name=r_name,
                image_path=r_path_clean,
                reference_age=r_age,
                detector=detector
            )
            enrolled_references_info.append({
                "name": r_name,
                "path": r_path_clean,
                "reference_age": r_age,
                "enrolled": ok,
                "status": status_msg
            })

    # ── 2. Target Face Detection & Person Body Detection ──────────────
    raw_faces = detector.detect_faces(target_img)
    valid_faces = [f for f in raw_faces if f.quality_status != "NO_FACE"]

    # Person body detection for supporting appearance signal
    person_crop = None
    yolo_detector = pipeline.IBVAPPipeline(config).detector
    person_dets = yolo_detector.detect(target_img)
    if person_dets:
        # Largest person bounding box
        p_box = max(person_dets, key=lambda d: (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1])).bbox
        person_crop = target_img[p_box[1]:p_box[3], p_box[0]:p_box[2]]
    else:
        person_crop = target_img

    # ── 3. Decision Logic & Invariant Enforcement ─────────────────────
    if len(valid_faces) == 0:
        target_face = None
        face_status = "NOT DETECTED"
        final_decision = "NO_FACE_DETECTED"
        verif_result = verifier.verify(target_img, face_detection=None, person_crop=person_crop)
    elif len(valid_faces) == 1:
        target_face = valid_faces[0]
        face_status = "DETECTED"
        if target_face.quality_status == "LOW_QUALITY_FACE":
            final_decision = "INSUFFICIENT_FACE_QUALITY"
            verif_result = verifier.verify(target_img, face_detection=target_face, person_crop=person_crop)
        else:
            verif_result = verifier.verify(target_img, face_detection=target_face, person_crop=person_crop)
            final_decision = verif_result.face_decision
    else:
        # Multiple faces detected
        face_status = f"DETECTED ({len(valid_faces)} FACES)"
        target_face = max(valid_faces, key=lambda f: f.confidence)
        verif_result = verifier.verify(target_img, face_detection=target_face, person_crop=person_crop)
        final_decision = verif_result.face_decision if verif_result.face_decision == "MATCH" else "MULTIPLE_FACES_DETECTED"

    # ── 4. Structured Console Output Formatting ───────────────────────
    print("\n" + "=" * 60)
    print("IBVAP FACE ANALYSIS")
    print("=" * 60)
    print(f"\nTarget:\n{target_path}")
    print(f"Target Dimensions: {tw}x{th} px")
    print("\n" + "-" * 60)
    print(f"{final_decision}")
    print("-" * 60)

    chosen_identity = verif_result.identity if final_decision == "MATCH" else "UNKNOWN"
    print(f"\nIdentity: {chosen_identity}")

    if verif_result.best_reference_path:
        print(f"\nReference:\n{verif_result.best_reference_path}")
        print(f"\nReference Age:\n{verif_result.best_reference_age}")

    print(f"\nFace:\n{face_status}")
    if target_face:
        print(f"\nFace Confidence:\n{target_face.confidence:.2f}")
    if verif_result.face_similarity > 0:
        print(f"\nFace Similarity:\n{verif_result.face_similarity:.2f}")

    print(f"\nBody:\n{'DETECTED' if verif_result.body_status != 'BODY_NOT_DETECTED' else 'NOT DETECTED'}")
    if verif_result.body_similarity > 0:
        print(f"\nBody Similarity:\n{verif_result.body_similarity:.2f}")
    print(f"\nBody Role:\n{verif_result.body_role}")

    print(f"\nFinal Face Decision:\n{final_decision}")

    if final_decision == "NO_FACE_DETECTED":
        print("\nIMPORTANT:\nBody evidence was NOT used to create a face identity.")

    # ── 5. Multiple Reference Breakdown for Enrolled Persons ──────────
    if verif_result.all_reference_comparisons:
        print("\n" + "-" * 60)
        print("INDIVIDUAL REFERENCE COMPARISONS:")
        print("-" * 60)
        by_person: Dict[str, List[Dict[str, Any]]] = {}
        for comp in verif_result.all_reference_comparisons:
            by_person.setdefault(comp["name"], []).append(comp)

        for person_name, comp_list in by_person.items():
            print(f"\nPerson: {person_name}")
            for idx, c in enumerate(comp_list, 1):
                body_sim_str = f"{c['body_similarity']:.2f}" if c["reference_age"] != "old" else "ignored"
                print(f"  Reference {idx}:")
                print(f"    Path: {c['reference_path']}")
                print(f"    Age:  {c['reference_age']}")
                print(f"    Face similarity: {c['face_similarity']:.2f}")
                print(f"    Body similarity: {body_sim_str}")

            best_p_face = max(c["face_similarity"] for c in comp_list)
            face_evidence = "STRONG" if best_p_face >= 0.75 else ("MODERATE" if best_p_face >= 0.65 else "WEAK")
            body_evidence = "SUPPORTING" if any(c["body_similarity"] >= 0.70 for c in comp_list if c["reference_age"] != "old") else "NONE"
            print(f"  --> Best face similarity: {best_p_face:.2f}")
            print(f"  --> Face evidence: {face_evidence}")
            print(f"  --> Body evidence: {body_evidence}")

    print("=" * 60 + "\n")

    # ── 6. Debug Visualization Mode ───────────────────────────────────
    debug_image_path = None
    if debug:
        dbg_canvas = target_img.copy()
        # Draw face detections
        for f in raw_faces:
            bx1, by1, bx2, by2 = f.box
            color = (0, 255, 0) if f.quality_status == "GOOD_FACE" else (0, 165, 255)
            cv2.rectangle(dbg_canvas, (bx1, by1), (bx2, by2), color, 2)
            label = f"{f.detector}: {f.confidence:.2f} ({f.quality_status})"
            cv2.putText(dbg_canvas, label, (bx1, max(15, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw 5 landmarks
            if f.landmarks is not None:
                for lx, ly in f.landmarks:
                    cv2.circle(dbg_canvas, (int(lx), int(ly)), 4, (0, 255, 255), -1)

        # Draw person crop boundary if detected
        if person_dets:
            px1, py1, px2, py2 = person_dets[0].bbox
            cv2.rectangle(dbg_canvas, (px1, py1), (px2, py2), (255, 100, 0), 2)
            cv2.putText(dbg_canvas, "Person Body Context", (px1, max(15, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)

        out_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(out_dir, exist_ok=True)
        debug_image_path = os.path.join(out_dir, "debug_analysis.jpg")
        cv2.imwrite(debug_image_path, dbg_canvas)
        print(f"[DEBUG] Diagnostic overlay saved to: {debug_image_path}")

    # Return structured dictionary
    result_dict = {
        "target": target_path,
        "face_decision": final_decision,
        "identity": chosen_identity,
        "identity_id": verif_result.identity_id,
        "face_detected": len(valid_faces) > 0,
        "face_count": len(valid_faces),
        "face_confidence": round(float(target_face.confidence), 4) if target_face else 0.0,
        "face_similarity": round(float(verif_result.face_similarity), 4),
        "best_reference_path": verif_result.best_reference_path,
        "best_reference_age": verif_result.best_reference_age,
        "body_status": verif_result.body_status,
        "body_similarity": round(float(verif_result.body_similarity), 4),
        "body_role": verif_result.body_role,
        "references_enrolled": enrolled_references_info,
        "reference_comparisons": verif_result.all_reference_comparisons,
        "debug_image_path": debug_image_path,
    }
    return result_dict


if __name__ == "__main__":
    # If user explicitly provides an image path or --test flag, run static test
    if len(sys.argv) > 1 and sys.argv[1].strip() not in ("--stream", "stream", "cam"):
        target_arg = sys.argv[2] if sys.argv[1] == "--test" and len(sys.argv) > 2 else sys.argv[1]
        default_ref = r"C:\ibvap\akshat.jpeg"
        if os.path.exists(default_ref) and os.path.exists(target_arg):
            output = test_images(
                references=[("Akshat", default_ref, "most_recent")],
                target=target_arg,
                debug=True
            )
        else:
            output = test_images(target=target_arg, debug=True)
    else:
        # DEFAULT MODE: Run live camera surveillance stream!
        print("[*] Starting Live CCTV Surveillance Stream via Camera / stream.py...")
        asyncio.run(survillance())
