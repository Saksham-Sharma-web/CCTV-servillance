"""
IBVAP Behavior Subsystem.
Modular behavioral analytics covering intrusion, loitering, route deviation, checkpoints, and crowd gatherings.
"""

from .intrusion.detector import (
    IntrusionDetector,
    VirtualFenceAnalytics,
    point_in_polygon,
    lines_intersect,
    determine_line_crossing_direction,
)
from .loitering.detector import LoiteringDetector
from .route_deviation.detector import RouteDeviationDetector, PermittedRoute
from .checkpoint.monitor import CheckpointMonitor, CheckpointGate, PassageRecord
from .crowd.detector import CrowdDetector, CrowdCluster

__all__ = [
    "IntrusionDetector",
    "VirtualFenceAnalytics",
    "point_in_polygon",
    "lines_intersect",
    "determine_line_crossing_direction",
    "LoiteringDetector",
    "RouteDeviationDetector",
    "PermittedRoute",
    "CheckpointMonitor",
    "CheckpointGate",
    "PassageRecord",
    "CrowdDetector",
    "CrowdCluster",
]
