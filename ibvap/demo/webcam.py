"""
IBVAP Webcam & Video Stream Live Demo Runner.
Demonstrates the source-agnostic AI analytics pipeline using a laptop webcam or video file.
"""

import argparse
import time
import logging
import cv2
import numpy as np

from ..core.pipeline import IBVAPPipeline
from ..core.config import IBVAPConfig
from ..core.types import VirtualBoundary, ZoneType, WatchlistCategory

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ibvap.demo")


def run_demo(source: str = "0", width: int = 1280, height: int = 720):
    # Parse source as int if it's a camera index
    cam_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(cam_source)

    if not cap.isOpened():
        logger.error(f"Failed to open video source: {source}")
        return

    # Request resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Video stream active: {actual_w}x{actual_h}")

    # Initialize IBVAP Pipeline
    config = IBVAPConfig(
        detection_confidence=0.40,
        loitering_duration_seconds=5.0,  # Fast 5s loitering trigger for live demo
        fence_cooldown_seconds=3.0,
        redis_enabled=False,  # Can enable if Redis is running locally
        db_enabled=False
    )
    pipeline = IBVAPPipeline(config=config)

    # 1. Add sample virtual fence across middle of frame
    mid_x = actual_w // 2
    pipeline.add_boundary(
        VirtualBoundary(
            id="fence-01",
            name="Restricted Border Line",
            zone_type=ZoneType.LINE,
            coordinates=[(mid_x, 0), (mid_x, actual_h)],
            target_classes=["person", "car", "motorcycle"]
        )
    )

    # 2. Add sample restricted polygon zone in right half
    pipeline.add_boundary(
        VirtualBoundary(
            id="zone-02",
            name="Perimeter Zone",
            zone_type=ZoneType.POLYGON,
            coordinates=[
                (int(actual_w * 0.65), int(actual_h * 0.2)),
                (int(actual_w * 0.95), int(actual_h * 0.2)),
                (int(actual_w * 0.95), int(actual_h * 0.8)),
                (int(actual_w * 0.65), int(actual_h * 0.8)),
            ],
            target_classes=["person"]
        )
    )

    # 3. Add sample ANPR watchlist
    pipeline.add_watchlist_vehicle("DL01AB1234", WatchlistCategory.BLACKLIST)
    pipeline.add_watchlist_vehicle("HR26DK8392", WatchlistCategory.WHITELIST)

    logger.info("Starting real-time processing loop. Press 'q' in the display window to exit.")

    fps_history = []
    prev_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("End of video stream or cannot read frame.")
                break

            now = time.time()
            fps = 1.0 / max(1e-5, now - prev_time)
            prev_time = now
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # ── Process Frame with IBVAP ──────────────────────────────
            result = pipeline.process_frame(
                frame=frame,
                camera_id="webcam-demo",
                timestamp=now
            )

            # ── Render Visual Debug Overlay ───────────────────────────
            annotated = pipeline.draw_debug(frame, result)

            # Draw live FPS counter
            cv2.putText(
                annotated,
                f"IBVAP LIVE | FPS: {avg_fps:.1f} | Tracks: {len(result.tracks)}",
                (20, actual_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("IBVAP Surveillance Platform (Press 'q' to quit)", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Webcam demo stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IBVAP live webcam demo.")
    parser.add_argument("--source", type=str, default="0", help="Video device index or file path (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
    args = parser.parse_args()

    run_demo(source=args.source, width=args.width, height=args.height)
