"""
License Plate OCR Subsystem.
Text extraction using OCR engines (EasyOCR, PaddleOCR, Tesseract fallback), regex cleaning, and consensus voting.
"""

from ibvap.anpr.ocr_adapter import ANPRAdapter
from ibvap.vehicle.consensus import ControlledOCRRunner, PlateConsensusEngine

# Provide aliases for OCRAdapter and PlateOCREngine
OCRAdapter = ANPRAdapter
PlateOCREngine = ANPRAdapter

__all__ = [
    "PlateOCREngine",
    "OCRAdapter",
    "ANPRAdapter",
    "ControlledOCRRunner",
    "PlateConsensusEngine",
]
