"""
Specialized Vehicle Detection Module.
Filters and analyzes vehicles (cars, trucks, buses, motorcycles, bicycles) for surveillance and ANPR readiness.
"""

from typing import List, Optional, Tuple, Set
import numpy as np
from ibvap.core.types import Detection
from ibvap.detection.base import BaseObjectDetector

DEFAULT_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class VehicleDetector:
    """
    Dedicated detector for motorized and wheeled vehicle targets.
    """

    def __init__(
        self,
        base_detector: Optional[BaseObjectDetector] = None,
        min_confidence: float = 0.35,
        min_width_px: int = 50,
        min_height_px: int = 40,
        vehicle_classes: Optional[Set[str]] = None,
    ):
        self.base_detector = base_detector
        self.min_confidence = min_confidence
        self.min_width_px = min_width_px
        self.min_height_px = min_height_px
        self.vehicle_classes = vehicle_classes or DEFAULT_VEHICLE_CLASSES

    def filter_vehicle_detections(self, detections: List[Detection]) -> List[Detection]:
        """Filters general detections to retain only valid vehicle targets."""
        vehicle_detections: List[Detection] = []
        for det in detections:
            if det.class_name.lower() not in self.vehicle_classes:
                continue
            if det.confidence < self.min_confidence:
                continue

            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            if w < self.min_width_px or h < self.min_height_px:
                continue

            vehicle_detections.append(det)

        return vehicle_detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detects vehicles directly from a frame if base_detector is configured."""
        if self.base_detector is None or frame is None or frame.size == 0:
            return []
        raw_detections = self.base_detector.detect(frame)
        return self.filter_vehicle_detections(raw_detections)

    @staticmethod
    def extract_crop(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding_pct: float = 0.05,
    ) -> Optional[np.ndarray]:
        """Safely extracts a padded vehicle crop from the frame."""
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
