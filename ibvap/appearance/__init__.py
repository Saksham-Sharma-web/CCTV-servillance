"""
IBVAP Appearance Subsystem.
Specialized attribute and visual appearance analysis.
"""

from .masked_person.detector import MaskedPersonDetector, MaskDetectionResult
from ibvap.face.matcher_adapter import BodyAppearanceExtractor

__all__ = [
    "MaskedPersonDetector",
    "MaskDetectionResult",
    "BodyAppearanceExtractor",
]
