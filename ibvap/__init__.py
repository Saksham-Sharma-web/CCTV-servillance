"""
IBVAP - Intelligent Border Video Analytics Platform
Modular, source-agnostic computer vision and behavioral analytics engine.
"""

from .core.pipeline import IBVAPPipeline
from .core.config import IBVAPConfig
from .core.types import (
    Detection,
    Track,
    AnalyticsEvent,
    PipelineResult,
    EventType,
    VirtualBoundary,
    ZoneType,
    WatchlistCategory,
)
from .core.camera_config import (
    CameraConfig,
    CameraManager,
    Region,
    Border,
    VirtualLine,
    LineDirection,
    RegionType,
    CameraEventRule,
    DetectionRule,
)
from .tracking.cross_camera import CrossCameraTracker, CrossCameraEntity

__all__ = [
    "IBVAPPipeline",
    "IBVAPConfig",
    "Detection",
    "Track",
    "AnalyticsEvent",
    "PipelineResult",
    "EventType",
    "VirtualBoundary",
    "ZoneType",
    "WatchlistCategory",
    "CameraConfig",
    "CameraManager",
    "Region",
    "Border",
    "VirtualLine",
    "LineDirection",
    "RegionType",
    "CameraEventRule",
    "DetectionRule",
    "CrossCameraTracker",
    "CrossCameraEntity",
]
