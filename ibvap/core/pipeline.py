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
    VEHICLE_CLASSES,
)
from .config import IBVAPConfig, default_config
from ..detection.base import BaseObjectDetector
from ..detection.object_detector import YOLOv8Detector
from ..tracking.tracker import PersistentTracker
from ..face.detector import OpenCVFaceDetector
from ..face.matcher_adapter import IdentityVerifierAdapter, AuthorizedPerson
from ..anpr.plate_detector import LicensePlateDetector
from ..anpr.ocr_adapter import ANPRAdapter
from .sampler import FrameSampler
from ..vehicle import (
    PlateQualityScorer,
    VehicleTrackBuffer,
    BestObservationSelector,
    ControlledOCRRunner,
    PlateConsensusEngine,
    VehicleObservation,
    VehicleStatus,
)
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

        # 4. ANPR (Plate Detection + Track-Centric Pipeline)
        self.plate_detector = LicensePlateDetector(self.config)
        self.anpr_adapter = ANPRAdapter(self.config)

        # Track-Centric ANPR Subsystem & Frame Sampling (Phases 1–8)
        self.sampler = FrameSampler(
            target_fps=self.config.analysis_fps,
            source_fps=self.config.camera_fps,
            enabled=self.config.frame_sampling_enabled,
        )
        self.vehicle_quality_scorer = PlateQualityScorer(
            min_acceptable_score=self.config.vehicle_min_quality_threshold
        )
        self.vehicle_buffer = VehicleTrackBuffer(
            max_observations_per_track=self.config.vehicle_max_observations_per_track,
            stale_track_timeout_seconds=self.config.vehicle_stale_track_timeout_seconds,
        )
        self.vehicle_selector = BestObservationSelector(
            max_k=self.config.vehicle_selector_max_k,
            min_quality_threshold=self.config.vehicle_min_quality_threshold,
        )
        self.controlled_ocr = ControlledOCRRunner(
            ocr_adapter=self.anpr_adapter,
            max_ocr_attempts_per_track=self.config.vehicle_max_ocr_attempts_per_track,
        )
        self.consensus_engine = PlateConsensusEngine(
            min_consensus_observations=self.config.vehicle_min_consensus_observations,
            min_agreement_ratio=self.config.vehicle_min_agreement_ratio,
            min_confidence_threshold=self.config.vehicle_min_confidence_threshold,
            single_observation_confidence_threshold=self.config.vehicle_single_obs_threshold,
        )

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

    def register_reference_image(
        self,
        name: str,
        image_path: str,
        reference_age: str = "most_recent",
        identity_id: Optional[str] = None,
        role: str = "AUTHORIZED"
    ) -> Tuple[bool, str]:
        """Registers an authorized person from a reference photograph with full face validation."""
        return self.identity_verifier.register_reference(
            name=name,
            image_path=image_path,
            reference_age=reference_age,
            identity_id=identity_id,
            role=role,
            detector=self.face_detector
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
        # Phase 7: Frame Sampling Gate (Disabled by default for test compatibility)
        if self.config.frame_sampling_enabled and not self.sampler.should_process(now, frame_index):
            return PipelineResult(
                frame_shape=(h, w),
                timestamp=now,
                detections=[],
                tracks=[],
                events=[],
                camera_id=camera_id,
                success=True,
                metadata={"frame_index": frame_index, "camera_id": camera_id, "sampled_out": True}
            )

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
                            faces = self.face_detector.detect_faces(person_crop)
                            valid_faces = [f for f in faces if getattr(f, "quality_status", "GOOD_FACE") != "NO_FACE"]

                            if not valid_faces:
                                # INVARIANT: NO VALID FACE -> NO FACE EMBEDDING -> NO IDENTITY
                                track.identity_id = None
                                track.identity_name = "UNKNOWN PERSON"
                                track.identity_confidence = 0.0
                            else:
                                top_face = valid_faces[0]
                                if getattr(top_face, "quality_status", "GOOD_FACE") == "LOW_QUALITY_FACE":
                                    track.identity_id = None
                                    track.identity_name = "UNKNOWN PERSON"
                                    track.identity_confidence = 0.0
                                else:
                                    verif_res = self.identity_verifier.verify(
                                        target_image=person_crop,
                                        face_detection=top_face,
                                        person_crop=person_crop
                                    )
                                    sim = verif_res.face_similarity
                                    track.identity_confidence = sim
                                    if verif_res.face_decision == "MATCH" and verif_res.matched_person is not None:
                                        track.identity_id = verif_res.identity_id
                                        track.identity_name = verif_res.identity
                                        # Bind identity to track in tracker session
                                        cam_tracker.update_track_identity(
                                            track_id=track.track_id,
                                            identity_id=verif_res.identity_id,
                                            identity_name=verif_res.identity,
                                            confidence=sim
                                        )
                                        candidate_events.append(
                                            AnalyticsEvent(
                                                camera_id=camera_id,
                                                timestamp=now,
                                                event_type=EventType.FACE_MATCHED,
                                                track_id=track.track_id,
                                                identity_id=verif_res.identity_id,
                                                confidence=sim,
                                                metadata={
                                                    "name": verif_res.identity,
                                                    "role": verif_res.matched_person.role,
                                                    "similarity": round(sim, 4),
                                                    "track_id": track.track_id,
                                                    "body_status": verif_res.body_status,
                                                    "body_similarity": round(verif_res.body_similarity, 4),
                                                    "body_role": verif_res.body_role,
                                                }
                                            )
                                        )
                                    else:
                                        track.identity_id = None
                                        track.identity_name = "UNKNOWN PERSON"

        # ── Step 4: Selective Track-Centric ANPR (Phases 1–8) ─────────
        if self.config.anpr_enabled:
            # Clean up stale tracks from vehicle observation buffer
            self.vehicle_buffer.cleanup_stale_tracks(current_time=now)

            for track in tracks:
                if track.class_name.lower() in VEHICLE_CLASSES:
                    vx1, vy1, vx2, vy2 = track.bbox
                    vx1, vy1 = max(0, vx1), max(0, vy1)
                    vx2, vy2 = min(w, vx2), min(h, vy2)
                    vehicle_crop = frame[vy1:vy2, vx1:vx2]

                    if vehicle_crop.size > 0:
                        # 1. Candidate plate detection
                        candidates = self.plate_detector.detect_plates(vehicle_crop)

                        # 2. Quality Scoring & Ingestion into Bounded Buffer
                        for cand_bbox, plate_crop in candidates:
                            if plate_crop is None or plate_crop.size == 0:
                                continue

                            quality_rep = self.vehicle_quality_scorer.score(plate_crop)
                            if not quality_rep.is_acceptable:
                                continue

                            c_px1, c_py1, c_px2, c_py2 = cand_bbox
                            abs_plate_bbox = (vx1 + c_px1, vy1 + c_py1, vx1 + c_px2, vy1 + c_py2)

                            obs = VehicleObservation(
                                track_id=track.track_id,
                                frame_index=frame_index,
                                timestamp=now,
                                plate_bbox=abs_plate_bbox,
                                plate_crop=plate_crop,
                                quality=quality_rep,
                                detection_confidence=track.confidence,
                            )
                            self.vehicle_buffer.add_observation(
                                obs, camera_id=camera_id, vehicle_class=track.class_name
                            )

                    # 3. Controlled OCR & Temporal Consensus
                    v_state = self.vehicle_buffer.get_track_state(track.track_id)
                    buffered_obs = self.vehicle_buffer.get_observations(track.track_id)

                    need_ocr_check = (
                        track.plate_number is None
                        and buffered_obs
                        and (v_state is None or v_state.ocr_attempts < self.config.vehicle_max_ocr_attempts_per_track)
                        and (
                            track.last_ocr_check_frame == 0
                            or (frame_index - track.last_ocr_check_frame) >= self.config.anpr_ocr_interval_frames
                            or len(buffered_obs) == 1
                        )
                    )

                    if need_ocr_check:
                        cam_tracker.mark_ocr_checked(track.track_id, frame_index)
                        selected = self.vehicle_selector.select(buffered_obs)

                        if selected:
                            ocr_results = self.controlled_ocr.run_ocr(selected, track_state=v_state)
                            consensus = self.consensus_engine.evaluate(ocr_results)

                            if v_state:
                                v_state.consensus = consensus
                                v_state.status = consensus.status

                            if consensus.is_confirmed and consensus.plate_number:
                                best_obs = self.vehicle_buffer.get_best_observation(track.track_id)
                                abs_plate_bbox = best_obs.plate_bbox if best_obs else None

                                cam_tracker.update_track_plate(
                                    track_id=track.track_id,
                                    plate_number=consensus.plate_number,
                                    category=consensus.category,
                                    confidence=consensus.confidence,
                                    ocr_confidence=consensus.confidence,
                                    plate_bbox=abs_plate_bbox,
                                )

                                track.plate_number = consensus.plate_number
                                track.plate_category = consensus.category
                                track.plate_confidence = consensus.confidence
                                track.ocr_confidence = consensus.confidence
                                track.plate_bbox = abs_plate_bbox

                                candidate_events.append(
                                    AnalyticsEvent(
                                        camera_id=camera_id,
                                        timestamp=now,
                                        event_type=EventType.PLATE_DETECTED,
                                        track_id=track.track_id,
                                        confidence=consensus.confidence,
                                        metadata={
                                            "plate_number": consensus.plate_number,
                                            "category": (
                                                consensus.category.value
                                                if hasattr(consensus.category, "value")
                                                else str(consensus.category)
                                            ),
                                            "vehicle_class": track.class_name,
                                            "ocr_confidence": consensus.confidence,
                                            "agreement_ratio": consensus.agreement_ratio,
                                            "observation_count": consensus.observation_count,
                                            "plate_bbox": list(abs_plate_bbox) if abs_plate_bbox else None,
                                        }
                                    )
                                )

                                if consensus.category == WatchlistCategory.BLACKLIST:
                                    candidate_events.append(
                                        AnalyticsEvent(
                                            camera_id=camera_id,
                                            timestamp=now,
                                            event_type=EventType.BLACKLISTED_VEHICLE,
                                            track_id=track.track_id,
                                            confidence=consensus.confidence,
                                            metadata={
                                                "plate_number": consensus.plate_number,
                                                "vehicle_class": track.class_name,
                                                "warning": "Vehicle is on the security BLACKLIST!",
                                            }
                                        )
                                    )

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
