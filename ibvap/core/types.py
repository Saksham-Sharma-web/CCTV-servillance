"""
IBVAP Data Contracts & Core Types.
Standardizes data models across all detection, tracking, analytics, and event stages.
All bounding boxes use [x1, y1, x2, y2] format (top-left, bottom-right).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import time
import uuid
import numpy as np


class EventType(str, Enum):
    PERSON_DETECTED = "PERSON_DETECTED"
    VEHICLE_DETECTED = "VEHICLE_DETECTED"
    FACE_MATCHED = "FACE_MATCHED"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    PLATE_DETECTED = "PLATE_DETECTED"
    BLACKLISTED_VEHICLE = "BLACKLISTED_VEHICLE"
    WATCHLIST_VEHICLE = "WATCHLIST_VEHICLE"
    FENCE_INTRUSION = "FENCE_INTRUSION"
    LOITERING = "LOITERING"
    SUSPICIOUS_MOVEMENT = "SUSPICIOUS_MOVEMENT"
    UNATTENDED_OBJECT = "UNATTENDED_OBJECT"
    NIGHT_MOVEMENT = "NIGHT_MOVEMENT"


class WatchlistCategory(str, Enum):
    WHITELIST = "WHITELIST"
    BLACKLIST = "BLACKLIST"
    WATCHLIST = "WATCHLIST"
    FREQUENTLY_OBSERVED = "FREQUENTLY_OBSERVED"
    UNKNOWN = "UNKNOWN"


class ZoneType(str, Enum):
    LINE = "LINE"
    POLYGON = "POLYGON"


@dataclass
class VirtualBoundary:
    id: str
    name: str
    zone_type: ZoneType
    coordinates: List[Tuple[int, int]]  # 2 points for LINE, >= 3 points for POLYGON
    target_classes: List[str] = field(default_factory=lambda: ["person", "car", "motorcycle", "truck"])


@dataclass
class Detection:
    """
    Single object detection candidate produced by an ObjectDetector.
    bbox format: (x1, y1, x2, y2)
    """
    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
        }


VEHICLE_CLASSES = {"car", "suv", "van", "truck", "bus", "motorcycle", "vehicle"}


@dataclass
class Track:
    """
    Persistent track maintained across consecutive video frames by the Tracker.
    CRITICAL: track_id is purely visual continuity, NOT identity.
    identity_id is populated independently by biometric face verification.
    """
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    center: Tuple[int, int]
    velocity: Tuple[float, float] = (0.0, 0.0)  # (vx, vy) in px/frame
    age: int = 1  # Total lifetime frames
    hits: int = 1  # Number of detection matches
    frames_since_update: int = 0
    last_seen: float = field(default_factory=time.time)
    history: List[Tuple[int, int]] = field(default_factory=list)  # Centroid history trail
    
    # Biometric verification result (Separated from track_id)
    identity_id: Optional[str] = None
    identity_confidence: Optional[float] = None
    identity_name: Optional[str] = None
    last_face_check_frame: int = 0

    # ANPR result
    plate_number: Optional[str] = None
    plate_category: Optional[WatchlistCategory] = None
    plate_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
    last_ocr_check_frame: int = 0

    # Behavioral state
    stationary_since: Optional[float] = None
    first_detected_in_zone: Dict[str, float] = field(default_factory=dict)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "bbox": list(self.bbox),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "center": list(self.center),
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "identity_confidence": round(float(self.identity_confidence), 4) if self.identity_confidence is not None else None,
            "plate_number": self.plate_number,
            "plate_category": self.plate_category.value if self.plate_category is not None else None,
            "plate_confidence": round(float(self.plate_confidence), 4) if self.plate_confidence is not None else None,
            "ocr_confidence": round(float(self.ocr_confidence), 4) if self.ocr_confidence is not None else None,
            "plate_bbox": list(self.plate_bbox) if self.plate_bbox is not None else None,
        }


@dataclass
class AnalyticsEvent:
    """
    Standardized event produced by the IBVAP Event Engine.
    """
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str = "camera-01"
    timestamp: float = field(default_factory=time.time)
    event_type: EventType = EventType.PERSON_DETECTED
    track_id: Optional[int] = None
    identity_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    snapshot_path: Optional[str] = None
    snapshot_crop: Optional[np.ndarray] = None  # Temporary BGR image crop for persistence

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            "track_id": self.track_id,
            "identity_id": self.identity_id,
            "confidence": round(float(self.confidence), 4),
            "metadata": self.metadata,
            "snapshot_path": self.snapshot_path,
        }


@dataclass
class PipelineResult:
    """
    Composite result returned by IBVAPPipeline.process_frame().
    """
    frame_shape: Tuple[int, int]  # (height, width)
    timestamp: float
    detections: List[Detection]
    tracks: List[Track]
    events: List[AnalyticsEvent]
    camera_id: str = "camera-01"
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_vehicle(self) -> Optional[Track]:
        for t in self.tracks:
            if t.class_name.lower() in VEHICLE_CLASSES:
                return t
        return None

    @property
    def vehicle_detected(self) -> bool:
        return self.primary_vehicle is not None or any(d.class_name.lower() in VEHICLE_CLASSES for d in self.detections)

    @property
    def vehicle_type(self) -> Optional[str]:
        pv = self.primary_vehicle
        if pv:
            return pv.class_name
        for d in self.detections:
            if d.class_name.lower() in VEHICLE_CLASSES:
                return d.class_name
        return None

    @property
    def vehicle_confidence(self) -> Optional[float]:
        pv = self.primary_vehicle
        if pv:
            return round(float(pv.confidence), 4)
        for d in self.detections:
            if d.class_name.lower() in VEHICLE_CLASSES:
                return round(float(d.confidence), 4)
        return None

    @property
    def license_plate_detected(self) -> bool:
        return any(t.plate_number is not None for t in self.tracks)

    @property
    def license_plate(self) -> Optional[str]:
        for t in self.tracks:
            if t.plate_number is not None:
                return t.plate_number
        return None

    @property
    def plate_confidence(self) -> Optional[float]:
        for t in self.tracks:
            if t.plate_number is not None and t.plate_confidence is not None:
                return round(float(t.plate_confidence), 4)
        return None

    @property
    def ocr_confidence(self) -> Optional[float]:
        for t in self.tracks:
            if t.plate_number is not None and t.ocr_confidence is not None:
                return round(float(t.ocr_confidence), 4)
        return None

    @property
    def face_detected(self) -> bool:
        return any(t.identity_id is not None or t.class_name == "person" for t in self.tracks) or any(d.class_name == "person" for d in self.detections)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "frame_shape": list(self.frame_shape),
            "vehicle_detected": self.vehicle_detected,
            "vehicle_type": self.vehicle_type,
            "vehicle_confidence": self.vehicle_confidence,
            "license_plate_detected": self.license_plate_detected,
            "license_plate": self.license_plate,
            "plate_confidence": self.plate_confidence,
            "ocr_confidence": self.ocr_confidence,
            "vehicle_analysis": {
                "vehicle_detected": self.vehicle_detected,
                "vehicle_type": self.vehicle_type,
                "vehicle_confidence": self.vehicle_confidence,
                "license_plate_detected": self.license_plate_detected,
                "license_plate": self.license_plate,
                "plate_confidence": self.plate_confidence,
                "ocr_confidence": self.ocr_confidence,
            },
            "detections": [d.to_dict() for d in self.detections],
            "tracks": [t.to_dict() for t in self.tracks],
            "events": [e.to_dict() for e in self.events],
            "metadata": self.metadata,
        }


