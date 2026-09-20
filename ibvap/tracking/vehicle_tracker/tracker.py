"""
Specialized Vehicle Tracking Subsystem.
Tracks vehicles, estimates road kinematics, heading directions, and speed vectors.
"""

from typing import List, Optional, Tuple, Dict, Set
import math
import numpy as np
from ibvap.core.types import Detection, Track
from ibvap.core.config import IBVAPConfig
from ibvap.tracking.tracker import PersistentTracker

DEFAULT_VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class VehicleTracker:
    """
    Dedicated tracker for vehicle kinematics and trajectories.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, vehicle_classes: Optional[Set[str]] = None):
        self.config = config
        self.vehicle_classes = vehicle_classes or DEFAULT_VEHICLE_CLASSES
        self._inner_tracker = PersistentTracker(config=config)
        self._headings: Dict[int, float] = {}  # degrees [0, 360)
        self._speeds_px: Dict[int, float] = {}  # px/frame

    def update(self, detections: List[Detection], timestamp: Optional[float] = None) -> List[Track]:
        """
        Updates vehicle tracks filtering exclusively for vehicle classes.
        """
        veh_dets = [d for d in detections if d.class_name.lower() in self.vehicle_classes]
        tracks = self._inner_tracker.update(veh_dets, timestamp=timestamp)

        active_ids = set()
        for trk in tracks:
            active_ids.add(trk.track_id)
            vx, vy = trk.velocity
            speed = math.hypot(vx, vy)
            self._speeds_px[trk.track_id] = speed

            if speed > 1.0:
                angle_rad = math.atan2(vy, vx)
                angle_deg = (math.degrees(angle_rad) + 360) % 360
                self._headings[trk.track_id] = angle_deg

        # Cleanup expired tracks
        for tid in list(self._headings.keys()):
            if tid not in active_ids:
                del self._headings[tid]
                self._speeds_px.pop(tid, None)

        return tracks

    def get_heading(self, track_id: int) -> Optional[float]:
        """Returns heading in degrees [0, 360) where 0 is right, 90 is down."""
        return self._headings.get(track_id)

    def get_speed_px(self, track_id: int) -> float:
        """Returns instantaneous speed in pixels per frame."""
        return self._speeds_px.get(track_id, 0.0)
