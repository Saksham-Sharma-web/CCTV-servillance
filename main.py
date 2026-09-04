import asyncio
import os
import sys
import cv2
import logging

# Ensure root paths are in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from ibvap.pipeline import IBVAPPipeline
from ibvap.core.types import VirtualBoundary, ZoneType
from ibvap.camera.manager import CameraManager
from ibvap.camera.models import CameraConfig, SourceType
from ibvap.camera import discovery, onvif

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("IBVAP")

REF_FACE_IMAGE = os.path.join(os.path.dirname(__file__), "test.png")

def register_reference_face(processor, ref_path):
    """Registers an authorized face into the pipeline from a reference photo."""
    if not os.path.exists(ref_path):
        logger.warning(f"Reference image not found at: {ref_path}")
        return False
    
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        logger.error(f"Failed to read image at: {ref_path}")
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
        name="Authorized User",
        face_bgr_image=face_crop,
        role="AUTHORIZED"
    )
    if success:
        logger.info(f"[+] Successfully registered face for 'Authorized User' (USER-01) from: {ref_path}")
    return success

class IBVAPApplication:
    """Thin entrypoint orchestrator for IBVAP."""
    def __init__(self):
        self.camera_manager = CameraManager()
        self.pipelines = {}  # Camera ID -> IBVAPPipeline
        self.running = False

    async def run(self):
        self.running = True
        logger.info("Starting IBVAP Camera Manager...")
        
        # 1. Discover available cameras on network
        logger.info("Scanning network for ONVIF cameras...")
        devices = discovery.discover(timeout=3)
        
        # 2. Add cameras to manager
        username = os.environ.get("CAMERA_USER", "cam")
        password = os.environ.get("CAMERA_PASS", "12345678")
        
        if devices:
            for idx, device in enumerate(devices):
                rtsp_uri = await onvif.connect_and_get_rtsp(device, username, password)
                if rtsp_uri:
                    cam_id = f"cam_{idx+1}"
                    config = CameraConfig(
                        id=cam_id,
                        name=f"Camera {idx+1}",
                        location="auto-discovered",
                        source_type=SourceType.ONVIF,
                        uri=rtsp_uri
                    )
                    self.camera_manager.add_camera(config)
        else:
            logger.warning("No ONVIF cameras found. Falling back to local USB camera.")
            config = CameraConfig(
                id="usb_1",
                name="Local Webcam",
                location="Local",
                source_type=SourceType.USB,
                uri=0
            )
            self.camera_manager.add_camera(config)

        # 3. Start camera streams and initialize pipelines
        for session in self.camera_manager.list_cameras():
            cam_id = session.config.id
            self.camera_manager.start_camera(cam_id)
            
            pipeline = IBVAPPipeline()
            register_reference_face(pipeline, REF_FACE_IMAGE)
            self.pipelines[cam_id] = pipeline

        logger.info("\n" + "=" * 65)
        logger.info("  🚀 LIVE AI SURVEILLANCE & BIOMETRIC TRACKING ACTIVE")
        logger.info("  - Face Recognition: Registered for 'Authorized User'")
        logger.info("  - Bounding Boxes: Green for Person, Cyan for Vehicle")
        logger.info("  - Press 'q' or 'ESC' in the video window to stop")
        logger.info("=" * 65 + "\n")

        # 4. Processing Loop
        try:
            self._processing_loop()
        finally:
            self.shutdown()

    def _processing_loop(self):
        PROCESS_EVERY_N_FRAMES = 5
        frame_counters = {cid: 0 for cid in self.pipelines.keys()}
        last_results = {cid: None for cid in self.pipelines.keys()}
        boundaries_initialized = {cid: False for cid in self.pipelines.keys()}

        while self.running:
            active_windows = 0
            
            for cam_id, pipeline in self.pipelines.items():
                source = self.camera_manager.get_source(cam_id)
                if not source:
                    continue
                    
                # Get latest frame from bounded queue (blocking with short timeout)
                packet = source.read_latest(timeout=0.1)
                if not packet:
                    continue
                    
                frame = packet.frame
                actual_h, actual_w = frame.shape[:2]
                active_windows += 1

                if not boundaries_initialized[cam_id]:
                    mid_x = actual_w // 2
                    pipeline.add_boundary(
                        VirtualBoundary(
                            id=f"fence-center-{cam_id}",
                            name="Center Tripwire Line",
                            zone_type=ZoneType.LINE,
                            coordinates=[(mid_x, 0), (mid_x, actual_h)],
                            target_classes=["person", "car", "motorcycle"]
                        )
                    )
                    boundaries_initialized[cam_id] = True

                frame_counters[cam_id] += 1

                # Run AI pipeline periodically (every N frames)
                if frame_counters[cam_id] % PROCESS_EVERY_N_FRAMES == 0 or last_results[cam_id] is None:
                    res = pipeline.process_frame(frame)
                    last_results[cam_id] = res

                    # Print events
                    if res and res.events:
                        for ev in res.events:
                            if ev.event_type.value == "FACE_MATCHED":
                                logger.info(f"✨ [MATCH CONFIRMED - {cam_id}] {ev.metadata.get('name', 'Known Person')} (Sim: {ev.confidence:.2f})")
                            elif "INTRUSION" in ev.event_type.value or "LOITERING" in ev.event_type.value:
                                logger.warning(f"🚨 [THREAT DETECTED - {cam_id}] {ev.event_type.value} | Track #{ev.track_id}")

                # Render live visual AI overlay on video
                res = last_results[cam_id]
                if res is not None:
                    annotated = pipeline.draw_debug(frame, res)
                else:
                    annotated = frame

                # Draw health metrics overlay
                health = source.health
                status_text = f"Frames Dropped: {health.dropped_frames} | Reconnects: {health.reconnect_count}"
                cv2.putText(annotated, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow(f"IBVAP CCTV - {cam_id}", annotated)

            if active_windows == 0:
                # No cameras are emitting frames right now, prevent tight spin loop
                cv2.waitKey(100)
                continue
                
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                self.running = False
                break

    def shutdown(self):
        logger.info("Shutting down IBVAP Application...")
        self.running = False
        self.camera_manager.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = IBVAPApplication()
    asyncio.run(app.run())
