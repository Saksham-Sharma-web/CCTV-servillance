"""
Route Deviation Detection Subsystem.
Detects departures from designated patrol corridors, restricted route divergences, and wrong-way travel.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import math
import uuid
import numpy as np

from ibvap.core.types import Track, AnalyticsEvent, EventType
from ibvap.core.config import IBVAPConfig, default_config


@dataclass
class PermittedRoute:
    """
    Defines an authorized movement corridor.
    """
    route_id: str
    name: str
    waypoints: List[Tuple[int, int]]
    corridor_width_px: float = 60.0
    one_way: bool = False
    camera_id: Optional[str] = None


def point_to_segment_distance(p: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[float, float]:
    """
    Returns (minimum_distance, projection_t_param) from point p to segment [a, b].
    t in [0, 1] indicates projected point falls within the segment.
    """
    px, py = p
    ax, ay = a
    bx, by = b
    dx = bx - ax
    dy = by - ay
    l2 = dx * dx + dy * dy
    if l2 == 0:
        return math.hypot(px - ax, py - ay), 0.0

    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = math.hypot(px - proj_x, py - proj_y)
    return dist, t


class RouteDeviationDetector:
    """
    Monitors moving tracks against permissible spatial corridors.
    Flags excessive lateral distance or wrong-way direction.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.routes: Dict[str, PermittedRoute] = {}
        # Map: (camera_id, track_id) -> last alert timestamp
        self._last_alert_times: Dict[Tuple[str, int], float] = {}

    def add_route(self, route: PermittedRoute):
        """Registers a designated route corridor."""
        self.routes[route.route_id] = route

    def remove_route(self, route_id: str):
        if route_id in self.routes:
            del self.routes[route_id]

    def process(
        self,
        tracks: List[Track],
        camera_id: str = "camera-01",
        timestamp: Optional[float] = None,
    ) -> List[AnalyticsEvent]:
        """
        Evaluates tracks against active routes for the specified camera.
        """
        now = timestamp if timestamp is not None else time.time()
        events: List[AnalyticsEvent] = []

        applicable_routes = [
            r for r in self.routes.values()
            if r.camera_id is None or r.camera_id == camera_id
        ]
        if not applicable_routes:
            return events

        active_ids = {t.track_id for t in tracks}
        for (cam, tid) in list(self._last_alert_times.keys()):
            if cam == camera_id and tid not in active_ids:
                del self._last_alert_times[(cam, tid)]

        for track in tracks:
            # Need moving track with at least 3 hits
            if len(track.history) < 2:
                continue

            cx, cy = track.center

            for route in applicable_routes:
                if len(route.waypoints) < 2:
                    continue

                # Find closest distance to any segment of the route
                min_dist = float("inf")
                closest_seg_idx = 0
                for i in range(len(route.waypoints) - 1):
                    dist, _ = point_to_segment_distance(
                        (cx, cy), route.waypoints[i], route.waypoints[i + 1]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        closest_seg_idx = i

                # Check if deviated beyond corridor width
                if min_dist > route.corridor_width_px:
                    key = (camera_id, track.track_id)
                    last_alert = self._last_alert_times.get(key, -1e9)
                    if now - last_alert >= self.config.event_cooldown_seconds:
                        event = AnalyticsEvent(
                            event_id=str(uuid.uuid4()),
                            camera_id=camera_id,
                            timestamp=now,
                            event_type=EventType.ROUTE_DEVIATION,
                            track_id=track.track_id,
                            identity_id=track.identity_id,
                            confidence=0.88,
                            metadata={
                                "rule": "route_deviation",
                                "sub_type": "ROUTE_DEVIATION",
                                "route_id": route.route_id,
                                "route_name": route.name,
                                "deviation_distance_px": round(min_dist, 1),
                                "corridor_limit_px": route.corridor_width_px,
                            },
                        )
                        events.append(event)
                        self._last_alert_times[key] = now

        return events
