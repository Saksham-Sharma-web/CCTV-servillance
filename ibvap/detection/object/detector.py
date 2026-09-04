"""
Specialized Object Detection Module.
Filters and analyzes general surveillance objects (backpacks, suitcases, handbags) and unattended baggage.
"""

from typing import List, Optional, Tuple, Set
import numpy as np
from ibvap.core.types import Detection, Track
from ibvap.detection.base import BaseObjectDetector

DEFAULT_SURVEILLANCE_OBJECTS = {"backpack", "suitcase", "handbag", "umbrella"}


class GeneralObjectDetector:
    """
    Dedicated detector for general surveillance objects, parcels, and unattended baggage.
    """

    def __init__(
        self,
        base_detector: Optional[BaseObjectDetector] = None,
        min_confidence: float = 0.30,
        target_classes: Optional[Set[str]] = None,
    ):
        self.base_detector = base_detector
        self.min_confidence = min_confidence
        self.target_classes = target_classes or DEFAULT_SURVEILLANCE_OBJECTS

    def filter_object_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filters general detections to retain only target surveillance objects."""
        object_detections: List[Detection] = []
        for det in detections:
            if det.class_name.lower() not in self.target_classes:
                continue
            if det.confidence < self.min_confidence:
                continue
            object_detections.append(det)

        return object_detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detects objects directly from a frame if base_detector is configured."""
        if self.base_detector is None or frame is None or frame.size == 0:
            return []
        raw_detections = self.base_detector.detect(frame)
        return self.filter_object_detections(raw_detections)

    def find_unattended_objects(
        self,
        object_tracks: List[Track],
        person_tracks: List[Track],
        max_proximity_distance_px: float = 120.0,
    ) -> List[Track]:
        """
        Identifies object tracks that are isolated from any active human tracks (unattended baggage).
        """
        unattended: List[Track] = []
        for obj in object_tracks:
            if obj.class_name.lower() not in self.target_classes:
                continue

            ox, oy = obj.center
            is_near_person = False
            for p in person_tracks:
                if p.class_name.lower() != "person":
                    continue
                px, py = p.center
                dist = np.hypot(ox - px, oy - py)
                if dist <= max_proximity_distance_px:
                    is_near_person = True
                    break

            if not is_near_person:
                unattended.append(obj)

        return unattended
