"""
Intrusion Detection Subsystem.
Virtual fence tripwire lines and restricted polygon boundary intrusion analytics.
"""

from typing import List, Dict, Tuple, Optional
import time

from ibvap.core.types import Track, AnalyticsEvent, EventType
from ibvap.core.camera_config import CameraConfig, RegionType, LineDirection
from ibvap.core.config import IBVAPConfig, default_config
from ibvap.analytics.virtual_fence import (
    VirtualFenceAnalytics,
    point_in_polygon,
    lines_intersect,
    determine_line_crossing_direction,
)

# IntrusionDetector aliases VirtualFenceAnalytics
IntrusionDetector = VirtualFenceAnalytics

__all__ = [
    "IntrusionDetector",
    "VirtualFenceAnalytics",
    "point_in_polygon",
    "lines_intersect",
    "determine_line_crossing_direction",
]
