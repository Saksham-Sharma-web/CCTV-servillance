"""
Re-Identification Subsystem.
Provides cross-camera tracking, appearance-based handover, and identity continuity across camera network.
"""

from typing import Optional
from ibvap.tracking.cross_camera import CrossCameraTracker, CrossCameraEntity
from ibvap.core.config import IBVAPConfig


class CrossCameraReID(CrossCameraTracker):
    """
    Subsystem for multi-camera Re-Identification and entity continuity.
    Inherits from and extends CrossCameraTracker.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, association_timeout_seconds: float = 300.0):
        super().__init__(config=config, association_timeout_seconds=association_timeout_seconds)


__all__ = ["CrossCameraReID", "CrossCameraTracker", "CrossCameraEntity"]
