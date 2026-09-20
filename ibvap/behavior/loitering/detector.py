"""
Loitering Detection Subsystem.
Monitors stationary dwell duration and spatial drift within a localized radius.
"""

from typing import List, Dict, Tuple, Optional
import time
import uuid

from ibvap.core.types import Track, AnalyticsEvent, EventType
from ibvap.core.config import IBVAPConfig, default_config


class LoiteringDetector:
    """
    Dedicated behavioral detector for prolonged stationary presence (loitering).
    """

    def __init__(
        self,
        config: Optional[IBVAPConfig] = None,
        duration_threshold_seconds: Optional[float] = None,
        radius_px: Optional[float] = None,
    ):
        self.config = config or default_config
        self.duration_threshold = (
            duration_threshold_seconds
            if duration_threshold_seconds is not None
            else self.config.loitering_duration_seconds
        )
        self.radius_px = (
            radius_px
            if radius_px is not None
            else self.config.loitering_distance_radius_px
        )
        # Map: (camera_id, track_id) -> (anchor_center, first_seen_time, stationary_start_time, last_alert_time)
        self._states: Dict[Tuple[str, int], Dict[str, float]] = {}

    def process(
        self,
        tracks: List[Track],
        camera_id: str = "camera-01",
        timestamp: Optional[float] = None,
    ) -> List[AnalyticsEvent]:
        """
        Evaluates active tracks for loitering behavior.
        """
        now = timestamp if timestamp is not None else time.time()
        events: List[AnalyticsEvent] = []

        active_ids = {t.track_id for t in tracks}
        # Cleanup expired tracks
        for (cam, tid) in list(self._states.keys()):
            if cam == camera_id and tid not in active_ids:
                del self._states[(cam, tid)]

        for track in tracks:
            # Loitering typically targets pedestrians and suspicious vehicles
            key = (camera_id, track.track_id)
            cx, cy = track.center

            if key not in self._states:
                self._states[key] = {
                    "anchor_x": float(cx),
                    "anchor_y": float(cy),
                    "start_time": now,
                    "last_alert": -1e9,
                }
                continue

            state = self._states[key]
            dx = cx - state["anchor_x"]
            dy = cy - state["anchor_y"]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > self.radius_px:
                # Reset anchor to current location if drifted beyond radius
                state["anchor_x"] = float(cx)
                state["anchor_y"] = float(cy)
                state["start_time"] = now
            else:
                stationary_time = now - state["start_time"]
                if stationary_time >= self.duration_threshold:
                    if now - state["last_alert"] >= self.config.event_cooldown_seconds:
                        event = AnalyticsEvent(
                            event_id=str(uuid.uuid4()),
                            camera_id=camera_id,
                            timestamp=now,
                            event_type=EventType.LOITERING,
                            track_id=track.track_id,
                            identity_id=track.identity_id,
                            confidence=0.85,
                            metadata={
                                "rule": "loitering_duration",
                                "stationary_duration": round(stationary_time, 2),
                                "drift_distance_px": round(dist, 1),
                                "threshold_seconds": self.duration_threshold,
                                "anchor": (int(state["anchor_x"]), int(state["anchor_y"])),
                            },
                        )
                        events.append(event)
                        state["last_alert"] = now

        return events
