from .base import BaseObjectDetector
from .object_detector import YOLOv8Detector, MockDetector
from .person.detector import PersonDetector
from .vehicle.detector import VehicleDetector
from .object.detector import GeneralObjectDetector

__all__ = [
    "BaseObjectDetector",
    "YOLOv8Detector",
    "MockDetector",
    "PersonDetector",
    "VehicleDetector",
    "GeneralObjectDetector",
]
