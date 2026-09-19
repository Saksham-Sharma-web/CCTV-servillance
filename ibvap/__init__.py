"""
IBVAP - Intelligent Border Video Analytics Platform
Modular, source-agnostic computer vision and behavioral analytics engine.

Subsystem Hierarchy:
- ingestion: Stream acquisition, ONVIF discovery, RTSP clients, and ring buffering
- detection: Specialized detectors for persons, vehicles, and general objects
- tracking: Dedicated pedestrian tracking, vehicle kinematics, and cross-camera Re-ID
- face: Face detection (YuNet/Haar), 5-point alignment, and 512D biometric verification
- vehicle: License plate detection (ANPR), OCR, and multi-frame consensus
- behavior: Perimeter intrusion, loitering, route deviation, checkpoints, and crowd detection
- appearance: Masked face concealment detection and body appearance profiling
- events: Sliding-window deduplication, alert manager dispatch, and forensic evidence packaging
- output: Live surveillance dashboards, HUD overlays, and REST/WebSocket API router
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

# Subsystem Modules
from . import ingestion
from . import detection
from . import tracking
from . import face
from . import vehicle
from . import behavior
from . import appearance
from . import events
from . import output

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
    # Subsystems
    "ingestion",
    "detection",
    "tracking",
    "face",
    "vehicle",
    "behavior",
    "appearance",
    "events",
    "output",
]
