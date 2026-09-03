"""
Snapshot and Media Storage Manager.
Saves BGR event snapshot crops to disk or cloud storage.
"""

from typing import Optional
import os
import time
import logging
import cv2
import numpy as np

from ..core.types import AnalyticsEvent
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.integration.storage")


class SnapshotStorage:
    """
    Manages saving event snapshots and thumbnail crops.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.enabled = self.config.storage_enabled
        self.storage_dir = self.config.storage_dir

        if self.enabled:
            os.makedirs(self.storage_dir, exist_ok=True)

    def save_snapshot(self, event: AnalyticsEvent, frame_or_crop: np.ndarray) -> Optional[str]:
        """
        Saves image snapshot to storage directory and updates event.snapshot_path.
        """
        if not self.enabled or frame_or_crop is None or frame_or_crop.size == 0:
            return None

        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
            event_type_str = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            filename = f"{event.camera_id}_{event_type_str}_{timestamp_str}_{event.event_id[:8]}.jpg"
            file_path = os.path.join(self.storage_dir, filename)

            cv2.imwrite(file_path, frame_or_crop)
            event.snapshot_path = file_path
            return file_path
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None
