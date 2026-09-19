"""
IBVAP Vehicle & ANPR Subsystem.
Modular, track-centric vehicle surveillance and license plate recognition.
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
]


