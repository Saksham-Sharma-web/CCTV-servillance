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
]
