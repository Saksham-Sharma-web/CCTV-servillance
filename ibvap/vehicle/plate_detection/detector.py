"""
License Plate Detection Subsystem.
Candidate bounding box localization, rectangular aspect-ratio filters, edge gradients, and plate crop extraction.
"""

from ibvap.anpr.plate_detector import LicensePlateDetector

# Provide PlateDetector as an alias for LicensePlateDetector
PlateDetector = LicensePlateDetector

__all__ = ["PlateDetector", "LicensePlateDetector"]
