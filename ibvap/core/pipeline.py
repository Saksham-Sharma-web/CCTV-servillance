"""
IBVAP Master Orchestration Pipeline.
Accepts raw uncompressed BGR numpy.ndarray frames.
Executes Detection -> Tracking -> Selective Biometrics -> Selective ANPR -> Behavioral Analytics -> Event Engine -> Integration.
Completely decoupled from RTSP/codecs/streaming sources.
"""

from typing import List, Dict, Optional, Tuple, Any
import time
import logging
import cv2
import numpy as np

from .types import (
    Detection,
    Track,
    AnalyticsEvent,
    PipelineResult,
    EventType,
    VirtualBoundary,
    WatchlistCategory,
)
from .config import IBVAPConfig, default_config
from ..detection.base import BaseObjectDetector
from ..detection.object_detector import YOLOv8Detector
from ..tracking.tracker import PersistentTracker
from ..face.detector import OpenCVFaceDetector
from ..face.matcher_adapter import IdentityVerifierAdapter, AuthorizedPerson
from ..anpr.plate_detector import LicensePlateDetector
from ..anpr.ocr_adapter import ANPRAdapter
from ..analytics.virtual_fence import VirtualFenceAnalytics
from ..analytics.suspicious_activity import SuspiciousActivityAnalytics
from ..analytics.night_movement import NightMovementAnalytics
from ..events.event_engine import EventEngine
from ..integration.redis_publisher import RedisAlertPublisher
from ..integration.db_logger import DatabaseEventLogger
from ..integration.storage import SnapshotStorage
from ..visualization.debug_renderer import DebugRenderer

logger = logging.getLogger("ibvap.pipeline")


class IBVAPPipeline:
    """
    Intelligent Border Video Analytics Platform Pipeline.
    Thread-safe, source-agnostic frame analytics pipeline.
    """

    def __init__(
        self,
        config: Optional[IBVAPConfig] = None,
        detector: Optional[BaseObjectDetector] = None,
        yunet_model_path: Optional[str] = None,
    ):
        self.config = config or default_config
        self.frame_indices: Dict[str, int] = {}

        # 1. Object Detector (Pluggable abstraction)
        self.detector: BaseObjectDetector = detector or YOLOv8Detector(self.config)

        # 2. Multi-Object Trackers (Isolated per camera stream)
        self.trackers: Dict[str, PersistentTracker] = {}

        # 3. Face Detection & Biometric Verification
        self.face_detector = OpenCVFaceDetector(self.config, yunet_model_path=yunet_model_path)
        self.identity_verifier = IdentityVerifierAdapter(self.config)

        # 4. ANPR (Plate Detection + PaddleOCR)
        self.plate_detector = LicensePlateDetector(self.config)
        self.anpr_adapter = ANPRAdapter(self.config)

        # 5. Behavioral Analytics
        self.virtual_fence = VirtualFenceAnalytics(self.config)
        self.suspicious_activity = SuspiciousActivityAnalytics(self.config)
        self.night_movement = NightMovementAnalytics(self.config)

        # 6. Event Engine & Integrations
        self.event_engine = EventEngine(self.config)
        self.redis_publisher = RedisAlertPublisher(self.config)
        self.db_logger = DatabaseEventLogger(self.config)
        self.storage = SnapshotStorage(self.config)

        # 7. Visualization
        self.renderer = DebugRenderer()

    def get_tracker(self, camera_id: str = "camera-01") -> PersistentTracker:
        """
        Retrieves or instantiates an isolated PersistentTracker for a camera stream.
        Guarantees camera-01 track_ids never collide with camera-02.
        """
        if camera_id not in self.trackers:
            self.trackers[camera_id] = PersistentTracker(self.config)
        return self.trackers[camera_id]

    @property
    def tracker(self) -> PersistentTracker:
        """Convenience accessor returning default camera tracker for backward compatibility."""
        return self.get_tracker("camera-01")

    # ── Input Contract Validation ────────────────────────────────
    def _validate_frame(self, frame: np.ndarray) -> None:
        """
        Lightweight validation of the incoming OpenCV BGR frame.
        Guarantees frame is a valid 3-channel (BGR) np.uint8 array.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None. Expected a numpy.ndarray.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Expected frame to be a numpy.ndarray, got {type(frame).__name__}.")
        if frame.ndim != 3:
            raise ValueError(
                f"Invalid frame dimensions: {frame.ndim}D with shape {frame.shape}. "
                f"Expected a 3D numpy.ndarray of shape (height, width, 3)."
            )
        if frame.shape[2] != 3:
            raise ValueError(
                f"Invalid channel count: {frame.shape[2]}. Expected exactly 3 channels (BGR)."
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"Invalid frame dtype: {frame.dtype}. Expected np.uint8."
            )
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(
                f"Invalid frame shape: {frame.shape}. Frame height and width must be > 0."
            )

    # ── Configuration & Boundary Management ─────────────────────
    def add_boundary(self, boundary: VirtualBoundary):
        """Adds a virtual line or polygon boundary."""
        self.virtual_fence.add_boundary(boundary)

    def remove_boundary(self, boundary_id: str):
        """Removes a virtual boundary."""
        self.virtual_fence.remove_boundary(boundary_id)

    def register_authorized_person(
        self,
        identity_id: str,
        name: str,
        face_bgr_image: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        role: str = "EMPLOYEE"
    ) -> bool:
        """Registers a known authorized person into the biometric database."""
        return self.identity_verifier.register_person(
            identity_id=identity_id,
            name=name,
            face_bgr_image=face_bgr_image,
            embedding=embedding,
            role=role
        )

    def add_watchlist_vehicle(self, plate_number: str, category: WatchlistCategory):
        """Registers a license plate into the ANPR watchlist."""
        self.anpr_adapter.add_watchlist_entry(plate_number, category)

    # ── Primary Frame Processing Method ──────────────────────────
    def process_frame(
        self,
        frame: np.ndarray,
        camera_id: str = "camera-01",
        timestamp: Optional[float] = None
    ) -> PipelineResult:
        """
        Main entry point for processing a single OpenCV video frame.

        Args:
            frame: numpy.ndarray, shape (height, width, 3), dtype uint8, color format BGR.
            camera_id: Identifier for camera source (maintains independent tracker state).
            timestamp: Frame capture timestamp in epoch seconds (defaults to time.time()).

        Returns:
            PipelineResult containing detections, persistent tracks, and debounced events.
        """
        self._validate_frame(frame)

        self.frame_indices[camera_id] = self.frame_indices.get(camera_id, 0) + 1
        frame_index = self.frame_indices[camera_id]
        now = timestamp if timestamp is not None else time.time()
        h, w = frame.shape[:2]
        candidate_events: List[AnalyticsEvent] = []

        # ── Step 1: Object Detection ──────────────────────────────────
        detections = self.detector.detect(frame)

        # ── Step 2: Multi-Object Tracking (Camera-Isolated) ───────────
        cam_tracker = self.get_tracker(camera_id)
        tracks = cam_tracker.update(detections, timestamp=now)

        # ── Step 3: Selective Face Detection & Verification ───────────
        if self.config.face_detection_enabled:
            for track in tracks:
                if track.class_name == "person":
                    # Run face check on first appearance (frame 0) or periodically if unverified
                    need_face_check = (
                        track.identity_id is None
                        and (
                            track.last_face_check_frame == 0
                            or (frame_index - track.last_face_check_frame) >= self.config.face_verification_interval_frames
                        )
                    )

                    if need_face_check:
                        cam_tracker.mark_face_checked(track.track_id, frame_index)
                        px1, py1, px2, py2 = track.bbox
                        person_crop = frame[py1:py2, px1:px2]

                        if person_crop.size > 0:
                            faces = self.face_detector.detect(person_crop)
                            if faces:
                                # Top face in crop
                                fx1, fy1, fx2, fy2, fconf = faces[0]
                                face_crop = person_crop[fy1:fy2, fx1:fx2]
                            else:
                                # Robust fallback: upper 45% of detected person box contains head & face
                                ph, pw = person_crop.shape[:2]
                                face_crop = person_crop[0:max(10, int(ph * 0.45)), 0:pw]

                            if face_crop.size > 0:
                                matched_person, sim = self.identity_verifier.verify_crop(face_crop)
                                track.identity_confidence = sim
                                if matched_person is not None:
                                    track.identity_id = matched_person.identity_id
                                    track.identity_name = matched_person.name
                                    # Bind identity to track in tracker session
                                    cam_tracker.update_track_identity(
                                        track_id=track.track_id,
                                        identity_id=matched_person.identity_id,
                                        identity_name=matched_person.name,
                                        confidence=sim
                                    )
                                    candidate_events.append(
                                        AnalyticsEvent(
                                            camera_id=camera_id,
                                            timestamp=now,
                                            event_type=EventType.FACE_MATCHED,
                                            track_id=track.track_id,
                                            identity_id=matched_person.identity_id,
                                            confidence=sim,
                                            metadata={
                                                "name": matched_person.name,
                                                "role": matched_person.role,
                                                "similarity": round(sim, 4),
                                                "track_id": track.track_id,
                                            }
                                        )
                                    )
                                else:
                                    track.identity_id = None
                                    track.identity_name = "UNKNOWN PERSON"

        # ── Step 4: Selective ANPR (License Plate OCR) ────────────────
        if self.config.anpr_enabled:
            for track in tracks:
                if track.class_name in ("car", "motorcycle", "bus", "truck"):
                    need_ocr_check = (
                        track.plate_number is None
                        and (
                            track.last_ocr_check_frame == 0
                            or (frame_index - track.last_ocr_check_frame) >= self.config.anpr_ocr_interval_frames
                        )
                    )

                    if need_ocr_check:
                        cam_tracker.mark_ocr_checked(track.track_id, frame_index)
                        vx1, vy1, vx2, vy2 = track.bbox
                        vehicle_crop = frame[vy1:vy2, vx1:vx2]

                        if vehicle_crop.size > 0:
                            candidates = self.plate_detector.detect_plates(vehicle_crop)
                            for _, plate_crop in candidates:
                                plate_res = self.anpr_adapter.recognize_plate(plate_crop)
                                if plate_res:
                                    cam_tracker.update_track_plate(
                                        track_id=track.track_id,
                                        plate_number=plate_res.plate_number,
                                        category=plate_res.category,
                                        confidence=plate_res.confidence
                                    )

                                    candidate_events.append(
                                        AnalyticsEvent(
                                            camera_id=camera_id,
                                            timestamp=now,
                                            event_type=EventType.PLATE_DETECTED,
                                            track_id=track.track_id,
                                            confidence=plate_res.confidence,
                                            metadata={
                                                "plate_number": plate_res.plate_number,
                                                "category": plate_res.category.value,
                                                "vehicle_class": track.class_name,
                                            }
                                        )
                                    )

                                    # High-priority watchlist alerts
                                    if plate_res.category == WatchlistCategory.BLACKLIST:
                                        candidate_events.append(
                                            AnalyticsEvent(
                                                camera_id=camera_id,
                                                timestamp=now,
                                                event_type=EventType.BLACKLISTED_VEHICLE,
                                                track_id=track.track_id,
                                                confidence=plate_res.confidence,
                                                metadata={
                                                    "plate_number": plate_res.plate_number,
                                                    "vehicle_class": track.class_name,
                                                    "warning": "Vehicle is on the security BLACKLIST!",
                                                }
                                            )
                                        )
                                    break

        # ── Step 5: Behavioral Analytics ──────────────────────────────
        # Virtual Fence Intrusion
        fence_events = self.virtual_fence.process_tracks(tracks, camera_id=camera_id, timestamp=now)
        candidate_events.extend(fence_events)

        # Suspicious Activity (Loitering, Sudden Speed, Unattended Luggage)
        suspicious_events = self.suspicious_activity.process_tracks(tracks, camera_id=camera_id, timestamp=now)
        candidate_events.extend(suspicious_events)

        # Night Movement
        night_events = self.night_movement.process_frame(frame, tracks, camera_id=camera_id, timestamp=now)
        candidate_events.extend(night_events)

        # ── Step 6: Event Deduplication & Debouncing ───────────────────
        emitted_events = self.event_engine.filter_and_emit(candidate_events)

        # ── Step 7: Integrations (Storage, DB, Redis) ──────────────────
        if emitted_events:
            for ev in emitted_events:
                # Save snapshot crop if track is present, else full frame
                if ev.track_id:
                    matched_trks = [t for t in tracks if t.track_id == ev.track_id]
                    if matched_trks:
                        bx1, by1, bx2, by2 = matched_trks[0].bbox
                        crop = frame[by1:by2, bx1:bx2]
                        self.storage.save_snapshot(ev, crop if crop.size > 0 else frame)
                    else:
                        self.storage.save_snapshot(ev, frame)
                else:
                    self.storage.save_snapshot(ev, frame)

            # Publish to Redis pub/sub for Node.js Socket.IO server
            self.redis_publisher.publish_events(emitted_events)
            # Log to PostgreSQL
            self.db_logger.log_events(emitted_events)

        return PipelineResult(
            frame_shape=(h, w),
            timestamp=now,
            detections=detections,
            tracks=tracks,
            events=emitted_events,
            camera_id=camera_id,
            success=True,
            metadata={"frame_index": frame_index, "camera_id": camera_id}
        )

    # ── Visual Debug Overlay ───────────────────────────────────────
    def draw_debug(self, frame: np.ndarray, result: PipelineResult) -> np.ndarray:
        """
        Draws debug overlay (bounding boxes, track IDs, identities, plates, zones, alerts).
        """
        return self.renderer.render(
            frame=frame,
            result=result,
            boundaries=self.virtual_fence.boundaries
        )
