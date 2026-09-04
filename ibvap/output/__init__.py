"""
IBVAP Output Subsystem.
Visual dashboards, HUD overlays, telemetry streaming, and REST/WebSocket APIs.
"""

from .dashboard.renderer import SurveillanceDashboard, DebugRenderer
from .api.router import IBVAPApiRouter

__all__ = [
    "SurveillanceDashboard",
    "DebugRenderer",
    "IBVAPApiRouter",
]
