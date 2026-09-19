"""
License Plate OCR Subsystem.
"""

from .engine import PlateOCREngine, OCRAdapter, ControlledOCRRunner, PlateConsensusEngine

__all__ = [
    "PlateOCREngine",
    "OCRAdapter",
    "ControlledOCRRunner",
    "PlateConsensusEngine",
]
