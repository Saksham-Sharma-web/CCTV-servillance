"""
IBVAP Vehicle ANPR Data Contracts & Core Types.
Standardizes data models for track-centric vehicle surveillance, license plate
quality assessment, observation buffering, and multi-frame temporal consensus.

Strict Architectural Invariant:
Vehicle evidence (plate, vehicle class, bounding box) is strictly decoupled from
human evidence (face biometric identity). Vehicle tracks never carry human identities.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import time
import numpy as np

from ..core.types import WatchlistCategory


class VehicleStatus(str, Enum):
    """
    Explicit lifecycle states for vehicle tracking, plate detection, and ANPR consensus.
    Guarantees transparent failure states without false identity or fabricated plate claims.
    """
    NO_VEHICLE = "NO_VEHICLE"
    VEHICLE_TENTATIVE = "VEHICLE_TENTATIVE"           # Detected, hits < min_hits (e.g. < 3)
    VEHICLE_TRACKED = "VEHICLE_TRACKED"               # Confirmed track active
    PLATE_NOT_LOCATED = "PLATE_NOT_LOCATED"           # Vehicle crop contains no detectable plate contour
    PLATE_TOO_SMALL = "PLATE_TOO_SMALL"               # Plate contour below minimum resolution threshold
    PLATE_QUALITY_INSUFFICIENT = "PLATE_QUALITY_INSUFFICIENT"  # Quality score below operational threshold
    OBSERVATION_COLLECTED = "OBSERVATION_COLLECTED"   # Valid plate crop buffered
    OCR_RECOGNITION = "OCR_RECOGNITION"               # Dispatched to OCR engine
    OCR_CONFIDENCE_LOW = "OCR_CONFIDENCE_LOW"         # OCR ran, but score below confidence threshold
    MULTI_FRAME_CONFLICT = "MULTI_FRAME_CONFLICT"     # Observations disagree across frames without consensus
    PLATE_CONFIRMED = "PLATE_CONFIRMED"               # Consensus reached with required temporal confidence
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"   # Track ended or expired without readable plate; no plate claimed
    TRACK_EXPIRED = "TRACK_EXPIRED"                   # Vehicle has departed from camera field of view


@dataclass
class PlateQualityReport:
    """
    Data contract for candidate license plate image quality assessment.
    Holds deterministic, lightweight quality metrics without implementing algorithms in Phase 1.
    All scores are normalized in range [0.0, 100.0].
    """
    overall_score: float = 0.0
    sharpness_score: float = 0.0       # Higher = sharper stroke edges (Laplacian variance)
    resolution_score: float = 0.0      # Adequacy of width & height relative to optimal OCR input
    aspect_ratio_score: float = 0.0    # Closeness to standard plate ratio (~3.1 to 3.4)
    contrast_score: float = 0.0        # RMS dynamic range across plate characters
    luminance_score: float = 0.0       # Penalty for underexposure (<30) or headlight glare (>230)
    is_acceptable: bool = False        # True if overall_score >= operational threshold
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": round(float(self.overall_score), 2),
            "sharpness_score": round(float(self.sharpness_score), 2),
            "resolution_score": round(float(self.resolution_score), 2),
            "aspect_ratio_score": round(float(self.aspect_ratio_score), 2),
            "contrast_score": round(float(self.contrast_score), 2),
            "luminance_score": round(float(self.luminance_score), 2),
            "is_acceptable": self.is_acceptable,
            "details": self.details,
        }


@dataclass
class VehicleObservation:
    """
    Single temporal observation of a candidate license plate associated with a vehicle track.

    Memory Safety Invariant:
    Full camera source frames are NEVER stored. Only the bounded candidate plate crop
    (or crop metadata) is retained.
    """
    track_id: int
    frame_index: int
    timestamp: float
    plate_bbox: Tuple[int, int, int, int]             # (px1, py1, px2, py2) relative to vehicle crop
    global_plate_bbox: Optional[Tuple[int, int, int, int]] = None  # (gx1, gy1, gx2, gy2) relative to full frame
    plate_crop: Optional[np.ndarray] = None           # Cropped BGR plate image (never full frame)
    detection_confidence: float = 1.0
    quality: Optional[PlateQualityReport] = None
    ocr_text: Optional[str] = None                    # Raw text from OCR if executed on this observation
    ocr_confidence: Optional[float] = None            # Confidence of OCR execution on this observation
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def crop_shape(self) -> Optional[Tuple[int, ...]]:
        return self.plate_crop.shape if self.plate_crop is not None else None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes observation to JSON-safe dictionary.
        Does NOT serialize raw numpy pixel array into dictionary.
        """
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "plate_bbox": list(self.plate_bbox),
            "global_plate_bbox": list(self.global_plate_bbox) if self.global_plate_bbox else None,
            "crop_dimensions": [int(self.plate_crop.shape[1]), int(self.plate_crop.shape[0])] if self.plate_crop is not None else None,
            "detection_confidence": round(float(self.detection_confidence), 4),
            "quality": self.quality.to_dict() if self.quality else None,
            "ocr_text": self.ocr_text,
            "ocr_confidence": round(float(self.ocr_confidence), 4) if self.ocr_confidence is not None else None,
            "metadata": self.metadata,
        }


@dataclass
class ConsensusResult:
    """
    Data representation for temporal multi-frame OCR consensus.
    Reconciles candidate readings across multiple observations of a vehicle track.
    """
    plate_number: Optional[str] = None                # Final confirmed plate string (None if unconfirmed)
    confidence: float = 0.0                           # Aggregate consensus confidence [0.0, 1.0]
    observation_count: int = 0                        # Number of observations contributing to consensus
    agreement_ratio: float = 0.0                      # Ratio of matching observations to total evaluated
    candidate_strings: List[str] = field(default_factory=list)  # Candidate strings from evaluated frames
    status: VehicleStatus = VehicleStatus.INSUFFICIENT_EVIDENCE
    is_confirmed: bool = False
    category: WatchlistCategory = WatchlistCategory.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plate_number": self.plate_number,
            "confidence": round(float(self.confidence), 4),
            "observation_count": self.observation_count,
            "agreement_ratio": round(float(self.agreement_ratio), 4),
            "candidate_strings": list(self.candidate_strings),
            "status": self.status.value if isinstance(self.status, VehicleStatus) else str(self.status),
            "is_confirmed": self.is_confirmed,
            "category": self.category.value if isinstance(self.category, WatchlistCategory) else str(self.category),
            "metadata": self.metadata,
        }


@dataclass
class VehicleTrackState:
    """
    ANPR-specific state associated with an active vehicle track.
    Decoupled from spatial tracking mechanics (managed by PersistentTracker).
    Accumulates bounded plate observations, tracks ANPR lifecycle status,
    and maintains temporal consensus evidence.
    """
    track_id: int
    camera_id: str = "camera-01"
    vehicle_class: str = "vehicle"
    status: VehicleStatus = VehicleStatus.VEHICLE_TENTATIVE
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    observations: List[VehicleObservation] = field(default_factory=list)
    best_observation: Optional[VehicleObservation] = None
    consensus: Optional[ConsensusResult] = None
    total_frames_tracked: int = 1
    ocr_attempts: int = 0                             # Counter for expensive OCR executions
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_confirmed_plate(self) -> bool:
        return self.consensus is not None and self.consensus.is_confirmed and bool(self.consensus.plate_number)

    @property
    def confirmed_plate_number(self) -> Optional[str]:
        if self.has_confirmed_plate and self.consensus:
            return self.consensus.plate_number
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "camera_id": self.camera_id,
            "vehicle_class": self.vehicle_class,
            "status": self.status.value if isinstance(self.status, VehicleStatus) else str(self.status),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observation_count": len(self.observations),
            "observations": [obs.to_dict() for obs in self.observations],
            "best_observation": self.best_observation.to_dict() if self.best_observation else None,
            "consensus": self.consensus.to_dict() if self.consensus else None,
            "total_frames_tracked": self.total_frames_tracked,
            "ocr_attempts": self.ocr_attempts,
            "has_confirmed_plate": self.has_confirmed_plate,
            "confirmed_plate_number": self.confirmed_plate_number,
            "metadata": self.metadata,
        }
