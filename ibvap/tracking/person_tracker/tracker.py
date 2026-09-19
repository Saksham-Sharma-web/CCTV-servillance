"""
Specialized Pedestrian Tracking Subsystem.
Tracks human targets, calculates foot anchor points, ground trajectories, and gait velocities.
"""

from typing import List, Optional, Tuple, Dict
import numpy as np
from ibvap.core.types import Detection, Track
from ibvap.core.config import IBVAPConfig
from ibvap.tracking.tracker import PersistentTracker


class PersonTracker:
    """
    Dedicated tracker for pedestrian targets.
    Maintains ground plane foot anchor points and walking dynamics.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config
        self._inner_tracker = PersistentTracker(config=config)
        self._ground_anchors: Dict[int, List[Tuple[int, int]]] = {}

    def update(self, detections: List[Detection], timestamp: Optional[float] = None) -> List[Track]:
        """
        Updates pedestrian tracks filtering exclusively for person detections.
        """
        person_dets = [d for d in detections if d.class_name.lower() == "person"]
        tracks = self._inner_tracker.update(person_dets, timestamp=timestamp)

        # Update ground anchor points (bottom-center of bounding box)
        active_ids = set()
        for trk in tracks:
            active_ids.add(trk.track_id)
            x1, y1, x2, y2 = trk.bbox
            foot_anchor = ((x1 + x2) // 2, y2)
            if trk.track_id not in self._ground_anchors:
                self._ground_anchors[trk.track_id] = []
            self._ground_anchors[trk.track_id].append(foot_anchor)
            if len(self._ground_anchors[trk.track_id]) > 60:
                self._ground_anchors[trk.track_id].pop(0)

        # Cleanup expired tracks
        for tid in list(self._ground_anchors.keys()):
            if tid not in active_ids:
                del self._ground_anchors[tid]

        return tracks

    def get_ground_anchor(self, track_id: int) -> Optional[Tuple[int, int]]:
        """Returns the latest ground anchor (foot location) for a given person track."""
        history = self._ground_anchors.get(track_id)
        return history[-1] if history else None

    def get_ground_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        """Returns full ground trajectory history for a given track."""
        return list(self._ground_anchors.get(track_id, []))
