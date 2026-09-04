"""
Virtual Fence & Boundary Intrusion Analytics.
Supports virtual line-crossing and polygon restricted zones.
Detects outside -> inside transitions and strictly prevents repeated duplicate alerts.
"""

from typing import List, Dict, Tuple, Optional
import time
import numpy as np

from ..core.types import Track, VirtualBoundary, ZoneType, AnalyticsEvent, EventType
from ..core.config import IBVAPConfig, default_config


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """
    Ray-casting algorithm to test if a 2D point is inside a polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def lines_intersect(p1: Tuple[int, int], p2: Tuple[int, int], q1: Tuple[int, int], q2: Tuple[int, int]) -> bool:
    """
    Checks if 2D line segment (p1, p2) intersects segment (q1, q2).
    """
    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))


class VirtualFenceAnalytics:
    """
    Evaluates tracks against virtual line and polygon boundaries.
    Generates alerts strictly on crossing transitions with state debouncing.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.cooldown = self.config.fence_cooldown_seconds
        self.boundaries: Dict[str, VirtualBoundary] = {}
        
        # State: (camera_id, track_id, zone_id) -> bool (is_currently_inside)
        self.track_inside_state: Dict[Tuple[str, int, str], bool] = {}
        # Cooldown state: (camera_id, track_id, zone_id) -> float (timestamp of last alert)
        self.last_alert_times: Dict[Tuple[str, int, str], float] = {}

    def add_boundary(self, boundary: VirtualBoundary):
        self.boundaries[boundary.id] = boundary

    def remove_boundary(self, boundary_id: str):
        if boundary_id in self.boundaries:
            del self.boundaries[boundary_id]

    def process_tracks(self, tracks: List[Track], camera_id: str = "camera-01", timestamp: Optional[float] = None) -> List[AnalyticsEvent]:
        now = timestamp if timestamp is not None else time.time()
        events: List[AnalyticsEvent] = []

        active_track_ids = {t.track_id for t in tracks}

        # Cleanup state for expired tracks for this camera
        for (cam, tid, zid) in list(self.track_inside_state.keys()):
            if cam == camera_id and tid not in active_track_ids:
                del self.track_inside_state[(cam, tid, zid)]
                if (cam, tid, zid) in self.last_alert_times:
                    del self.last_alert_times[(cam, tid, zid)]

        for track in tracks:
            # Need history to establish motion trajectory
            curr_pos = track.center
            prev_pos = track.history[-2] if len(track.history) >= 2 else curr_pos

            for zone_id, boundary in self.boundaries.items():
                if boundary.target_classes and track.class_name not in boundary.target_classes:
                    continue

                state_key = (camera_id, track.track_id, zone_id)
                was_inside = self.track_inside_state.get(state_key, False)
                now_inside = False
                crossed_line = False

                if boundary.zone_type == ZoneType.POLYGON:
                    now_inside = point_in_polygon(curr_pos, boundary.coordinates)
                    transition = (not was_inside) and now_inside
                    self.track_inside_state[state_key] = now_inside

                    if transition:
                        last_alert = self.last_alert_times.get(state_key, -1e9)
                        if (now - last_alert) >= self.cooldown:
                            self.last_alert_times[state_key] = now
                            events.append(
                                AnalyticsEvent(
                                    camera_id=camera_id,
                                    timestamp=now,
                                    event_type=EventType.FENCE_INTRUSION,
                                    track_id=track.track_id,
                                    identity_id=track.identity_id,
                                    confidence=track.confidence,
                                    metadata={
                                        "zone_id": zone_id,
                                        "zone_name": boundary.name,
                                        "zone_type": "POLYGON",
                                        "object_class": track.class_name,
                                        "transition": "OUTSIDE_TO_INSIDE",
                                        "position": curr_pos,
                                    }
                                )
                            )

                elif boundary.zone_type == ZoneType.LINE and len(boundary.coordinates) >= 2:
                    q1, q2 = boundary.coordinates[0], boundary.coordinates[1]
                    crossed_line = lines_intersect(prev_pos, curr_pos, q1, q2)

                    if crossed_line:
                        last_alert = self.last_alert_times.get(state_key, -1e9)
                        if (now - last_alert) >= self.cooldown:
                            self.last_alert_times[state_key] = now
                            events.append(
                                AnalyticsEvent(
                                    camera_id=camera_id,
                                    timestamp=now,
                                    event_type=EventType.FENCE_INTRUSION,
                                    track_id=track.track_id,
                                    identity_id=track.identity_id,
                                    confidence=track.confidence,
                                    metadata={
                                        "zone_id": zone_id,
                                        "zone_name": boundary.name,
                                        "zone_type": "LINE",
                                        "object_class": track.class_name,
                                        "transition": "LINE_CROSSED",
                                        "position": curr_pos,
                                    }
                                )
                            )

        return events
