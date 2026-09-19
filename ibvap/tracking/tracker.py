"""
Persistent Multi-Object Tracker (SORT / ByteTrack variant).
Maintains visual continuity IDs across consecutive frames.
CRITICAL INVARIANT: track_id is purely visual continuity, NOT identity.
"""

from typing import List, Optional, Tuple
import time
import numpy as np

from ..core.types import Detection, Track
from ..core.config import IBVAPConfig, default_config
from .kalman import KalmanBoxTracker
from .matching import associate_detections_to_trackers


class PersistentTracker:
    """
    Manages active Kalman-filtered tracks across consecutive frames.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.max_age = self.config.tracking_max_lost_frames
        self.min_hits = self.config.tracking_min_hits
        self.iou_threshold = self.config.tracking_iou_threshold
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: List[Detection], timestamp: Optional[float] = None) -> List[Track]:
        """
        Updates the tracker with detections from the current frame.

        Args:
            detections: List of Detection instances in current frame.
            timestamp: Optional frame timestamp (defaults to current time).

        Returns:
            List of active Track objects for the current frame.
        """
        self.frame_count += 1
        now = timestamp or time.time()

        # 1. Predict new locations of existing trackers
        trks = []
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            # If prediction is invalid / NaN, mark for deletion
            if np.any(np.isnan(pos)):
                to_del.append(t)
            else:
                trks.append(pos)

        for t in reversed(to_del):
            self.trackers.pop(t)

        # 2. Extract bounding boxes from current detections
        dets = [d.bbox for d in detections]

        # 3. Associate detections to existing trackers using Hungarian matching
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, iou_threshold=self.iou_threshold
        )

        # 4. Update matched trackers with assigned detections
        for m in matched:
            d_idx = m[0]
            t_idx = m[1]
            detection = detections[d_idx]
            self.trackers[t_idx].update(detection.bbox, detection.confidence)
            self.trackers[t_idx].class_name = detection.class_name

        # 5. Create new trackers for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            new_trk = KalmanBoxTracker(det.bbox, det.class_name, det.confidence)
            self.trackers.append(new_trk)

        # 6. Build active Track objects and cull dead trackers
        active_tracks: List[Track] = []
        surviving_trackers: List[KalmanBoxTracker] = []

        for trk in self.trackers:
            # Check if tracker should be deleted
            if trk.time_since_update > self.max_age:
                continue

            surviving_trackers.append(trk)

            # Output track if it was updated in this frame and meets min_hits threshold
            if trk.time_since_update == 0 and (trk.hits >= self.min_hits or self.frame_count <= self.min_hits):
                state_box = trk.get_state()
                x1, y1, x2, y2 = state_box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                track_obj = Track(
                    track_id=trk.id,
                    bbox=state_box,
                    class_name=trk.class_name,
                    confidence=trk.confidence,
                    center=(cx, cy),
                    velocity=trk.velocity,
                    age=trk.age,
                    hits=trk.hits,
                    frames_since_update=trk.time_since_update,
                    last_seen=now,
                    history=list(trk.centroid_history),
                    identity_id=trk.identity_id,
                    identity_confidence=trk.identity_confidence,
                    identity_name=trk.identity_name,
                    last_face_check_frame=trk.last_face_check_frame,
                    plate_number=trk.plate_number,
                    plate_category=trk.plate_category,
                    plate_confidence=trk.plate_confidence,
                    ocr_confidence=getattr(trk, "ocr_confidence", None),
                    plate_bbox=getattr(trk, "plate_bbox", None),
                    last_ocr_check_frame=trk.last_ocr_check_frame,
                    stationary_since=trk.stationary_since,
                )
                active_tracks.append(track_obj)

        self.trackers = surviving_trackers
        return active_tracks

    def update_track_identity(self, track_id: int, identity_id: str, identity_name: str, confidence: float):
        """
        Attaches confirmed biometric identity to an active track.
        """
        for trk in self.trackers:
            if trk.id == track_id:
                trk.identity_id = identity_id
                trk.identity_name = identity_name
                trk.identity_confidence = confidence
                break

    def update_track_plate(
        self,
        track_id: int,
        plate_number: str,
        category,
        confidence: float,
        ocr_confidence: Optional[float] = None,
        plate_bbox: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Attaches verified license plate to an active track.
        """
        for trk in self.trackers:
            if trk.id == track_id:
                trk.plate_number = plate_number
                trk.plate_category = category
                trk.plate_confidence = confidence
                trk.ocr_confidence = ocr_confidence or confidence
                trk.plate_bbox = plate_bbox
                break

    def mark_face_checked(self, track_id: int, frame_idx: int):
        for trk in self.trackers:
            if trk.id == track_id:
                trk.last_face_check_frame = frame_idx
                break

    def mark_ocr_checked(self, track_id: int, frame_idx: int):
        for trk in self.trackers:
            if trk.id == track_id:
                trk.last_ocr_check_frame = frame_idx
                break

    def reset(self):
        self.trackers.clear()
        self.frame_count = 0
        KalmanBoxTracker.count = 0
