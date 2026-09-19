"""
IBVAP Vehicle & ANPR Subsystem.
Modular, track-centric vehicle surveillance, license plate detection, and recognition.
"""

from .types import (
    VehicleStatus,
    PlateQualityReport,
    VehicleObservation,
    ConsensusResult,
    VehicleTrackState,
)
from .quality import PlateQualityScorer
from .buffer import VehicleTrackBuffer
from .selector import BestObservationSelector
from .consensus import ControlledOCRRunner, PlateConsensusEngine
from .plate_detection.detector import PlateDetector
from .ocr.engine import PlateOCREngine

__all__ = [
    "VehicleStatus",
    "PlateQualityReport",
    "VehicleObservation",
    "ConsensusResult",
    "VehicleTrackState",
    "PlateQualityScorer",
    "VehicleTrackBuffer",
    "BestObservationSelector",
    "ControlledOCRRunner",
    "PlateConsensusEngine",
    "PlateDetector",
    "PlateOCREngine",
]
