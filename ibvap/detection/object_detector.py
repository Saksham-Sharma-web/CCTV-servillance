"""
Object Detector Implementations.
Provides YOLOv8Detector (via Ultralytics) and MockDetector (for deterministic testing).
"""

from typing import List, Optional, Set
import logging
import numpy as np
import torch

from .base import BaseObjectDetector
from ..core.types import Detection
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.detection")

# Ensure PyTorch weights_only doesn't break YOLO model unpickling
try:
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass


class YOLOv8Detector(BaseObjectDetector):
    """
    Production YOLOv8 Object Detector.
    Filters specifically for surveillance classes: person, vehicles, luggage/backpacks.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, model_weights: Optional[str] = None):
        import os
        self.config = config or default_config

        # Check candidate locations for weights
        candidate_weights = [
            model_weights,
            self.config.detector_model_path,
            os.path.join(self.config.models_dir, self.config.detector_model_name),
            os.path.join(os.getcwd(), self.config.detector_model_name),
            self.config.detector_model_name
        ]
        weights_path = self.config.detector_model_name
        for cand in candidate_weights:
            if cand and os.path.exists(cand):
                weights_path = cand
                break

        self.confidence_threshold = self.config.detection_confidence
        self.iou_threshold = self.config.detection_iou_threshold
        self.target_classes: Set[str] = {c.lower() for c in self.config.target_classes}

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from '{weights_path}' onto device '{self.device}'...")
            self.model = YOLO(weights_path)
            # Warm up model if possible
            dummy_frame = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model(dummy_frame, verbose=False, device=self.device)
            logger.info(f"YOLO detector initialized successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO detector: {e}")
            logger.warning("Object detection will run in degraded mode (no detections).")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None or frame is None or frame.size == 0:
            return []

        try:
            h, w = frame.shape[:2]
            logger.debug(f"[Detection] Input frame dimensions: {w}x{h}")

            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )

            detections: List[Detection] = []
            if not results:
                return detections

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    raw_cls_name = r.names.get(cls_id, f"class_{cls_id}").lower()

                    # Class name synonym normalization
                    cls_name = raw_cls_name
                    if raw_cls_name in ("automobile", "sedan", "coupe", "suv", "van"):
                        cls_name = "car"

                    # Filter only requested surveillance classes
                    if self.target_classes and cls_name not in self.target_classes and raw_cls_name not in self.target_classes:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                    # Clip to frame boundary
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(x1 + 1, min(w, x2))
                    y2 = max(y1 + 1, min(h, y2))

                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            class_id=cls_id,
                            class_name=cls_name,
                            confidence=conf
                        )
                    )

            vehicle_dets = [d for d in detections if d.class_name in ("car", "suv", "van", "truck", "bus", "motorcycle", "vehicle")]
            if vehicle_dets:
                logger.info(
                    f"[Detection] Found {len(vehicle_dets)} vehicles in frame: "
                    f"{[(d.class_name, d.bbox, round(d.confidence, 3)) for d in vehicle_dets]}"
                )

            return detections
        except Exception as e:
            logger.error(f"Error during YOLO frame detection: {e}")
            return []


class MockDetector(BaseObjectDetector):
    """
    Mock detector for deterministic testing without neural network weights.
    Returns pre-configured detections or generates programmatic movements.
    """

    def __init__(self, preset_detections: Optional[List[Detection]] = None):
        self.preset_detections = preset_detections or []

    def set_detections(self, detections: List[Detection]) -> None:
        self.preset_detections = detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return list(self.preset_detections)
