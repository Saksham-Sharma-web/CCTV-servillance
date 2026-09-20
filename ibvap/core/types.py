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
    # Core Entities
    PERSON_DETECTED = "PERSON_DETECTED"
    VEHICLE_DETECTED = "VEHICLE_DETECTED"
    FACE_MATCHED = "FACE_MATCHED"
    UNKNOWN_PERSON = "UNKNOWN_PERSON"
    PERSON_REIDENTIFIED = "PERSON_REIDENTIFIED"
    UNKNOWN_PERSON_SIGHTING = "UNKNOWN_PERSON_SIGHTING"
    PLATE_DETECTED = "PLATE_DETECTED"
    
    # Watchlists
    BLACKLISTED_VEHICLE = "BLACKLISTED_VEHICLE"
    WATCHLIST_VEHICLE = "WATCHLIST_VEHICLE"
    
    # Intrusions & Boundaries
    FENCE_INTRUSION = "FENCE_INTRUSION"
    REGION_INTRUSION = "REGION_INTRUSION"
    BORDER_CROSSING = "BORDER_CROSSING"
    LINE_CROSSING = "LINE_CROSSING"
    TRESPASSING = "TRESPASSING"
    UNAUTHORIZED_VEHICLE = "UNAUTHORIZED_VEHICLE"
    VEHICLE_RESTRICTED_ZONE = "VEHICLE_RESTRICTED_ZONE"
    
    # Movement & Trajectory
    DIRECTION_VIOLATION = "DIRECTION_VIOLATION"
    WRONG_DIRECTION = "WRONG_DIRECTION"
    REPEATED_MOVEMENT = "REPEATED_MOVEMENT"
    PATH_DEVIATION = "PATH_DEVIATION"
    SUDDEN_DIRECTION_CHANGE = "SUDDEN_DIRECTION_CHANGE"
    CIRCLING = "CIRCLING"
    REPEATED_ENTRY_EXIT = "REPEATED_ENTRY_EXIT"
    
    # Speed & Posture
    RUNNING = "RUNNING"
    FALLING = "FALLING"
    CROUCHING = "CROUCHING"
    SUDDEN_STOPPING = "SUDDEN_STOPPING"
    SUSPICIOUS_MOVEMENT = "SUSPICIOUS_MOVEMENT"
    
    # Stationary & Loitering
    LOITERING = "LOITERING"
    VEHICLE_LOITERING = "VEHICLE_LOITERING"
    
    # Group & Interaction
    GROUP_FORMATION = "GROUP_FORMATION"
    FOLLOWING = "FOLLOWING"
    
    # Objects
    UNATTENDED_OBJECT = "UNATTENDED_OBJECT"
    ABANDONED_OBJECT = "ABANDONED_OBJECT"
    OBJECT_REMOVED = "OBJECT_REMOVED"
    OBJECT_RESTRICTED_ZONE = "OBJECT_RESTRICTED_ZONE"
    UNUSUAL_OBJECT_PLACEMENT = "UNUSUAL_OBJECT_PLACEMENT"
    
    # Environment
    NIGHT_MOVEMENT = "NIGHT_MOVEMENT"
    ROUTE_DEVIATION = "ROUTE_DEVIATION"
    CROWD_GATHERING = "CROWD_GATHERING"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    CHECKPOINT_VIOLATION = "CHECKPOINT_VIOLATION"
    MASKED_PERSON = "MASKED_PERSON"



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
    camera_id: str = "camera-01"


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

    # Mask & Concealment state (Independent from identity)
    is_masked: Optional[bool] = None
    mask_confidence: Optional[float] = None
    concealment_type: Optional[str] = None  # "MASKED", "UNMASKED", "UNKNOWN", "NO_FACE"
    consecutive_masked_frames: int = 0

    # Behavioral state
    stationary_since: Optional[float] = None
    first_detected_in_zone: Dict[str, float] = field(default_factory=dict)

    # Cross-Camera Tracking
    global_track_id: Optional[str] = None

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "global_track_id": self.global_track_id,
            "bbox": list(self.bbox),
            "class_name": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "center": list(self.center),
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "identity_confidence": round(float(self.identity_confidence), 4) if self.identity_confidence is not None else None,
            "is_masked": self.is_masked,
            "mask_confidence": round(float(self.mask_confidence), 4) if self.mask_confidence is not None else None,
            "concealment_type": self.concealment_type,
            "consecutive_masked_frames": self.consecutive_masked_frames,
            "plate_number": self.plate_number,
            "plate_category": self.plate_category.value if self.plate_category is not None else None,
            "plate_confidence": round(float(self.plate_confidence), 4) if self.plate_confidence is not None else None,
            "ocr_confidence": round(float(self.ocr_confidence), 4) if self.ocr_confidence is not None else None,
            "plate_bbox": list(self.plate_bbox) if self.plate_bbox is not None else None,
        }


@dataclass
class PresenceSession:
    """
    A continuous period during which a person is present on a specific camera.
    """
    session_id: str
    person_id: str
    camera_id: str
    first_seen: float
    last_seen: float
    duration_seconds: float = 0.0
    status: str = "ACTIVE"  # "ACTIVE", "CLOSED"
    tracker_ids: List[int] = field(default_factory=list)
    event_id: Optional[str] = None
    created_at_iso: str = ""
    updated_at_iso: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "person_id": self.person_id,
            "camera_id": self.camera_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": round(float(self.duration_seconds), 2),
            "status": self.status,
            "tracker_ids": list(self.tracker_ids),
            "event_id": self.event_id,
            "created_at_iso": self.created_at_iso,
            "updated_at_iso": self.updated_at_iso,
            "metadata": self.metadata,
        }


@dataclass
class TrajectorySegment:
    """
    An ordered camera presence segment in a person's movement trajectory.
    """
    camera_id: str
    entry_time: float
    exit_time: float
    duration_seconds: float
    tracker_ids: List[int] = field(default_factory=list)
    session_id: Optional[str] = None
    entry_time_iso: str = ""
    exit_time_iso: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "duration_seconds": round(float(self.duration_seconds), 2),
            "tracker_ids": list(self.tracker_ids),
            "session_id": self.session_id,
            "entry_time_iso": self.entry_time_iso,
            "exit_time_iso": self.exit_time_iso,
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
    global_track_id: Optional[str] = None
    identity_id: Optional[str] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    snapshot_path: Optional[str] = None
    snapshot_crop: Optional[np.ndarray] = None  # Temporary BGR image crop for persistence

    # Person-Centric Correlation & Event Lifecycle extensions
    person_id: Optional[str] = None
    session_id: Optional[str] = None
    event_status: str = "ACTIVE"  # "ACTIVE", "RESOLVED", "CLOSED"
    first_seen: Optional[float] = None
    last_seen: Optional[float] = None
    duration_seconds: float = 0.0
    is_update: bool = False  # True if this is an update to an existing event, False if newly created

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else str(self.event_type),
            "track_id": self.track_id,
            "global_track_id": self.global_track_id,
            "identity_id": self.identity_id,
            "confidence": round(float(self.confidence), 4),
            "metadata": self.metadata,
            "snapshot_path": self.snapshot_path,
            "person_id": self.person_id or self.identity_id,
            "session_id": self.session_id,
            "event_status": self.event_status,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_seconds": round(float(self.duration_seconds), 2),
            "is_update": self.is_update,
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


@dataclass
class UnknownPersonSighting:
    """A single observation/sighting of an unknown person on a camera."""
    sighting_id: str
    unknown_id: str
    camera_id: str
    track_id: int
    timestamp: float
    timestamp_iso: str
    bbox: Tuple[int, int, int, int]
    similarity: float = 0.0
    face_quality: float = 1.0
    snapshot_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sighting_id": self.sighting_id,
            "unknown_id": self.unknown_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "bbox": list(self.bbox),
            "similarity": round(float(self.similarity), 4),
            "face_quality": round(float(self.face_quality), 4),
            "snapshot_path": self.snapshot_path,
            "metadata": self.metadata,
        }


@dataclass
class UnknownPersonRecord:
    """Master record for an unknown individual tracked across multiple cameras."""
    unknown_id: str
    first_seen_timestamp: float
    last_seen_timestamp: float
    first_camera_id: str
    last_camera_id: str
    prototype_embedding: np.ndarray  # 512-D L2-normalized
    representative_embeddings: List[np.ndarray] = field(default_factory=list)  # Bounded (up to 5)
    sightings: List[UnknownPersonSighting] = field(default_factory=list)
    camera_sequence: List[str] = field(default_factory=list)
    created_at_iso: str = ""
    updated_at_iso: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unknown_id": self.unknown_id,
            "first_seen_timestamp": self.first_seen_timestamp,
            "last_seen_timestamp": self.last_seen_timestamp,
            "first_camera_id": self.first_camera_id,
            "last_camera_id": self.last_camera_id,
            "camera_sequence": list(self.camera_sequence),
            "sightings_count": len(self.sightings),
            "representative_embeddings_count": len(self.representative_embeddings),
            "created_at_iso": self.created_at_iso,
            "updated_at_iso": self.updated_at_iso,
            "metadata": self.metadata,
        }


