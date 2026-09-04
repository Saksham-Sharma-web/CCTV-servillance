"""
Surveillance Dashboard HUD & Overlay Subsystem.
Draws dynamic bounding boxes, spatial boundary lines, colored trajectory trails,
threat alerts, and live operational telemetry onto camera video frames.
"""

from ibvap.visualization.debug_renderer import DebugRenderer

# Provide SurveillanceDashboard as the primary class
SurveillanceDashboard = DebugRenderer

__all__ = ["SurveillanceDashboard", "DebugRenderer"]
