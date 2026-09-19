"""
Virtual Fence & Boundary Intrusion Analytics.
Supports admin-controlled:
- Polygon restricted regions (outside -> inside transitions)
- Border boundaries (polyline crossings)
- Directional virtual lines (ENTRY, EXIT, BIDIRECTIONAL, etc.)
Strictly camera-isolated: Configuration from Camera A never leaks into Camera B.
If a camera has no regions/lines configured, processing is skipped.
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import time
import math
import numpy as np

from ..core.types import Track, VirtualBoundary, ZoneType, AnalyticsEvent, EventType
from ..core.camera_config import Region, Border, VirtualLine, LineDirection, RegionType, CameraConfig
from ..core.config import IBVAPConfig, default_config


def point_in_polygon(point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
    """
    Ray-casting algorithm to test if a 2D point is inside a polygon.
    """
    if not polygon or len(polygon) < 3:
        return False
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


def determine_line_crossing_direction(
    prev_pos: Tuple[int, int],
    curr_pos: Tuple[int, int],
    line_start: Tuple[int, int],
    line_end: Tuple[int, int]
) -> LineDirection:
    """
    Determines the directional crossing orientation of a track moving across a directed line.
    Uses 2D cross product with respect to directed line segment (line_start -> line_end).
    - Cross product > 0: point is to the LEFT of the directed line.
    - Cross product < 0: point is to the RIGHT of the directed line.
    - Transition Left -> Right is defined as ENTRY / LEFT_TO_RIGHT.
    - Transition Right -> Left is defined as EXIT / RIGHT_TO_LEFT.
    """
    ax, ay = line_start
    bx, by = line_end
    dx, dy = bx - ax, by - ay

    # 2D cross product: (B - A) x (P - A)
    def side_of_line(p: Tuple[int, int]) -> float:
        px, py = p
        return dx * (py - ay) - dy * (px - ax)

    side_prev = side_of_line(prev_pos)
    side_curr = side_of_line(curr_pos)

    if side_prev > 0 and side_curr <= 0:
        return LineDirection.ENTRY
    elif side_prev < 0 and side_curr >= 0:
        return LineDirection.EXIT
    elif side_prev >= 0 and side_curr < 0:
        return LineDirection.ENTRY
    elif side_prev <= 0 and side_curr > 0:
        return LineDirection.EXIT
    else:
        return LineDirection.BIDIRECTIONAL


class VirtualFenceAnalytics:
    """
    Admin-controlled spatial boundary analytics.
    Evaluates tracks strictly on a per-camera basis against:
    1. Admin-defined polygonal regions (Restricted zones)
    2. Admin-defined borders (Boundary perimeters)
    3. Admin-defined directional virtual lines (Entry / Exit / Tripwires)
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.cooldown = self.config.fence_cooldown_seconds

        # Camera-isolated boundaries: camera_id -> {boundary_id: boundary_object}
        self.camera_boundaries: Dict[str, Dict[str, Any]] = {}

        # State: (camera_id, track_id, zone_id) -> bool (is_currently_inside)
        self.track_inside_state: Dict[Tuple[str, int, str], bool] = {}
        # Cooldown state: (camera_id, track_id, zone_id) -> float (timestamp of last alert)
        self.last_alert_times: Dict[Tuple[str, int, str], float] = {}

    # ── Backward Compatibility Property ─────────────────────────────
    @property
    def boundaries(self) -> Dict[str, VirtualBoundary]:
        """Convenience property returning default camera boundaries for legacy callers."""
        return self.camera_boundaries.get("camera-01", {})

    # ── Configuration Registration Methods ──────────────────────────
    def add_boundary(self, boundary: VirtualBoundary, camera_id: Optional[str] = None):
        """Adds a legacy VirtualBoundary bound to a specific camera."""
        cid = camera_id or getattr(boundary, "camera_id", "camera-01")
        if cid not in self.camera_boundaries:
            self.camera_boundaries[cid] = {}
        boundary.camera_id = cid
        self.camera_boundaries[cid][boundary.id] = boundary

    def add_region(self, region: Region):
        """Adds an admin-defined Region to its camera configuration."""
        cid = region.camera_id
        if cid not in self.camera_boundaries:
            self.camera_boundaries[cid] = {}
        self.camera_boundaries[cid][region.region_id] = region

    def add_border(self, border: Border):
        """Adds an admin-defined Border to its camera configuration."""
        cid = border.camera_id
        if cid not in self.camera_boundaries:
            self.camera_boundaries[cid] = {}
        self.camera_boundaries[cid][border.border_id] = border

    def add_virtual_line(self, line: VirtualLine):
        """Adds an admin-defined directional VirtualLine to its camera configuration."""
        cid = line.camera_id
        if cid not in self.camera_boundaries:
            self.camera_boundaries[cid] = {}
        self.camera_boundaries[cid][line.line_id] = line

    def remove_boundary(self, boundary_id: str, camera_id: Optional[str] = None):
        """Removes a boundary from a specific camera, or across all cameras if not specified."""
        if camera_id:
            if camera_id in self.camera_boundaries and boundary_id in self.camera_boundaries[camera_id]:
                del self.camera_boundaries[camera_id][boundary_id]
        else:
            for cid in list(self.camera_boundaries.keys()):
                if boundary_id in self.camera_boundaries[cid]:
                    del self.camera_boundaries[cid][boundary_id]

    def clear_camera(self, camera_id: str):
        """Removes all boundaries for a given camera."""
        if camera_id in self.camera_boundaries:
            del self.camera_boundaries[camera_id]

    # ── Main Tracking Evaluation ───────────────────────────────────
    def process_tracks(
        self,
        tracks: List[Track],
        camera_id: str = "camera-01",
        timestamp: Optional[float] = None,
        camera_config: Optional[CameraConfig] = None
    ) -> List[AnalyticsEvent]:
        """
        Evaluates tracks against ONLY the specified camera's configuration.
        If camera has no regions, borders, or lines configured, returns [] immediately.
        Never defaults or copies regions across cameras.
        """
        now = timestamp if timestamp is not None else time.time()
        events: List[AnalyticsEvent] = []

        # 1. Resolve spatial boundaries exclusively for this camera
        configured_items: Dict[str, Any] = {}

        if camera_config is not None:
            # Single source of truth: Admin CameraConfig
            for r_id, r in camera_config.regions.items():
                configured_items[f"region:{r_id}"] = r
            for b_id, b in camera_config.borders.items():
                configured_items[f"border:{b_id}"] = b
            for l_id, l in camera_config.virtual_lines.items():
                configured_items[f"line:{l_id}"] = l

        # Merge any boundaries registered directly on virtual_fence for this camera
        direct_items = self.camera_boundaries.get(camera_id, {})
        for item_id, item in direct_items.items():
            if f"region:{item_id}" not in configured_items and f"line:{item_id}" not in configured_items:
                configured_items[item_id] = item

        # RULE: If camera has NO configured boundaries, skip stage completely!
        if not configured_items:
            return []

        active_track_ids = {t.track_id for t in tracks}

        # Cleanup state for expired tracks for this camera
        for (cam, tid, zid) in list(self.track_inside_state.keys()):
            if cam == camera_id and tid not in active_track_ids:
                del self.track_inside_state[(cam, tid, zid)]
                if (cam, tid, zid) in self.last_alert_times:
                    del self.last_alert_times[(cam, tid, zid)]

        for track in tracks:
            curr_pos = track.center
            prev_pos = track.history[-2] if len(track.history) >= 2 else curr_pos

            for item_id, item in configured_items.items():
                target_classes = getattr(item, "target_classes", None)
                if target_classes and track.class_name not in target_classes:
                    continue

                state_key = (camera_id, track.track_id, item_id)
                was_inside = self.track_inside_state.get(state_key, False)

                # ── Branch A: Polygon Region (or Polygon VirtualBoundary) ──
                if isinstance(item, Region) or (isinstance(item, VirtualBoundary) and item.zone_type == ZoneType.POLYGON):
                    polygon_coords = item.polygon if isinstance(item, Region) else item.coordinates
                    now_inside = point_in_polygon(curr_pos, polygon_coords)
                    transition = (not was_inside) and now_inside
                    self.track_inside_state[state_key] = now_inside

                    if transition:
                        last_alert = self.last_alert_times.get(state_key, -1e9)
                        rule_cooldown = self.cooldown
                        if camera_config:
                            for rule in camera_config.event_rules.values():
                                if rule.region_id == getattr(item, "region_id", getattr(item, "id", None)):
                                    if rule.cooldown_seconds is not None:
                                        rule_cooldown = rule.cooldown_seconds
                                    break

                        if (now - last_alert) >= rule_cooldown:
                            self.last_alert_times[state_key] = now
                            reg_id = getattr(item, "region_id", getattr(item, "id", item_id))
                            reg_name = getattr(item, "name", reg_id)
                            reg_type = getattr(item, "region_type", "RESTRICTED")
                            reg_type_str = reg_type.value if hasattr(reg_type, "value") else str(reg_type)

                            is_legacy = isinstance(item, VirtualBoundary) or getattr(item, "metadata", {}).get("is_legacy_boundary", False)
                            ev_type = EventType.FENCE_INTRUSION if is_legacy else EventType.REGION_INTRUSION
                            events.append(
                                AnalyticsEvent(
                                    camera_id=camera_id,
                                    timestamp=now,
                                    event_type=ev_type,
                                    track_id=track.track_id,
                                    global_track_id=track.global_track_id,
                                    identity_id=track.identity_id,
                                    confidence=track.confidence,
                                    metadata={
                                        "region_id": reg_id,
                                        "zone_id": reg_id,
                                        "zone_name": reg_name,
                                        "region_type": reg_type_str,
                                        "object_class": track.class_name,
                                        "transition": "OUTSIDE_TO_INSIDE",
                                        "position": curr_pos,
                                        "camera_id": camera_id,
                                    }
                                )
                            )

                # ── Branch B: Directional Virtual Line ─────────────────────
                elif isinstance(item, VirtualLine) or (isinstance(item, VirtualBoundary) and item.zone_type == ZoneType.LINE):
                    is_legacy = isinstance(item, VirtualBoundary) or getattr(item, "metadata", {}).get("is_legacy_boundary", False)
                    if isinstance(item, VirtualLine):
                        q1, q2 = item.coordinates[0], item.coordinates[1]
                        allowed_dir = item.direction
                        line_id = item.line_id
                        line_name = item.name
                    else:
                        q1, q2 = item.coordinates[0], item.coordinates[1]
                        allowed_dir = LineDirection.BIDIRECTIONAL
                        line_id = item.id
                        line_name = item.name

                    crossed_line = lines_intersect(prev_pos, curr_pos, q1, q2)

                    if crossed_line:
                        crossing_dir = determine_line_crossing_direction(prev_pos, curr_pos, q1, q2)

                        # Check direction compatibility
                        dir_match = False
                        if allowed_dir in (LineDirection.BIDIRECTIONAL, LineDirection.ENTRY, LineDirection.LEFT_TO_RIGHT):
                            if allowed_dir == LineDirection.BIDIRECTIONAL or crossing_dir in (LineDirection.ENTRY, LineDirection.LEFT_TO_RIGHT, LineDirection.BIDIRECTIONAL):
                                dir_match = True
                        elif allowed_dir in (LineDirection.EXIT, LineDirection.RIGHT_TO_LEFT):
                            if crossing_dir in (LineDirection.EXIT, LineDirection.RIGHT_TO_LEFT, LineDirection.BIDIRECTIONAL):
                                dir_match = True

                        last_alert = self.last_alert_times.get(state_key, -1e9)
                        if (now - last_alert) >= self.cooldown:
                            self.last_alert_times[state_key] = now

                            if dir_match:
                                ev_type = EventType.FENCE_INTRUSION if is_legacy else EventType.LINE_CROSSING
                                events.append(
                                    AnalyticsEvent(
                                        camera_id=camera_id,
                                        timestamp=now,
                                        event_type=ev_type,
                                        track_id=track.track_id,
                                        global_track_id=track.global_track_id,
                                        identity_id=track.identity_id,
                                        confidence=track.confidence,
                                        metadata={
                                            "line_id": line_id,
                                            "zone_id": line_id,
                                            "zone_name": line_name,
                                            "zone_type": "LINE",
                                            "direction": crossing_dir.value,
                                            "configured_direction": allowed_dir.value,
                                            "object_class": track.class_name,
                                            "transition": "LINE_CROSSED",
                                            "position": curr_pos,
                                            "camera_id": camera_id,
                                        }
                                    )
                                )
                            else:
                                # Reverse direction attempt
                                events.append(
                                    AnalyticsEvent(
                                        camera_id=camera_id,
                                        timestamp=now,
                                        event_type=EventType.DIRECTION_VIOLATION,
                                        track_id=track.track_id,
                                        global_track_id=track.global_track_id,
                                        identity_id=track.identity_id,
                                        confidence=track.confidence,
                                        metadata={
                                            "line_id": line_id,
                                            "zone_id": line_id,
                                            "zone_name": line_name,
                                            "direction": crossing_dir.value,
                                            "configured_direction": allowed_dir.value,
                                            "object_class": track.class_name,
                                            "transition": "WRONG_WAY_CROSSING",
                                            "position": curr_pos,
                                            "camera_id": camera_id,
                                        }
                                    )
                                )

                # ── Branch C: Border Polyline Crossing ────────────────────
                elif isinstance(item, Border):
                    crossed = False
                    coords = item.coordinates
                    for i in range(len(coords) - 1):
                        seg1, seg2 = coords[i], coords[i + 1]
                        if lines_intersect(prev_pos, curr_pos, seg1, seg2):
                            crossed = True
                            break

                    if crossed:
                        last_alert = self.last_alert_times.get(state_key, -1e9)
                        if (now - last_alert) >= self.cooldown:
                            self.last_alert_times[state_key] = now
                            events.append(
                                AnalyticsEvent(
                                    camera_id=camera_id,
                                    timestamp=now,
                                    event_type=EventType.BORDER_CROSSING,
                                    track_id=track.track_id,
                                    global_track_id=track.global_track_id,
                                    identity_id=track.identity_id,
                                    confidence=track.confidence,
                                    metadata={
                                        "border_id": item.border_id,
                                        "zone_id": item.border_id,
                                        "zone_name": item.name,
                                        "object_class": track.class_name,
                                        "transition": "BORDER_BREACH",
                                        "position": curr_pos,
                                        "camera_id": camera_id,
                                    }
                                )
                            )

        return events
