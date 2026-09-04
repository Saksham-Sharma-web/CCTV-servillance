import asyncio
import os
import cv2
import stream
import discovery
import ibvap.core.pipeline as pipeline
from ibvap.core.types import VirtualBoundary, ZoneType

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


if __name__ == "__main__":
    asyncio.run(survillance())
