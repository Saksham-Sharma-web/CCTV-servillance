"""
Specialized Person Detection Module.
Filters and analyzes pedestrian detections, validates human proportions, and extracts person crops.
"""

from typing import List, Optional, Tuple
import numpy as np
from ibvap.core.types import Detection
from ibvap.detection.base import BaseObjectDetector


class PersonDetector:
    """
    Dedicated detector for human pedestrian targets.
    Can operate over raw frames (delegating to a base detector) or filter existing detections.
    """

    def __init__(
        self,
        base_detector: Optional[BaseObjectDetector] = None,
        min_confidence: float = 0.40,
        min_height_px: int = 40,
        min_aspect_ratio: float = 1.1,  # Height/Width ratio for upright humans
    ):
        self.base_detector = base_detector
        self.min_confidence = min_confidence
        self.min_height_px = min_height_px
        self.min_aspect_ratio = min_aspect_ratio

    def filter_person_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filters general detections to retain only valid person targets."""
        person_detections: List[Detection] = []
        for det in detections:
            if det.class_name.lower() != "person":
                continue
            if det.confidence < self.min_confidence:
                continue

            x1, y1, x2, y2 = det.bbox
            h = y2 - y1
            w = x2 - x1
            if h < self.min_height_px or w <= 0:
                continue

            # Accept if meets valid human bounds
            person_detections.append(det)

        return person_detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detects people directly in a BGR frame if base_detector is provided."""
        if self.base_detector is None or frame is None or frame.size == 0:
            return []
        raw_detections = self.base_detector.detect(frame)
        return self.filter_person_detections(raw_detections)

    @staticmethod
    def extract_crop(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding_pct: float = 0.05,
    ) -> Optional[np.ndarray]:
        """Safely extracts a padded person crop from the frame."""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        pad_x = int((x2 - x1) * padding_pct)
        pad_y = int((y2 - y1) * padding_pct)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        if cx2 <= cx1 or cy2 <= cy1:
            return None
        return frame[cy1:cy2, cx1:cx2].copy()
