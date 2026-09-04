"""
Suspicious Activity Analytics.
Rule-based behavioral layer with deterministic metrics:
- Loitering (duration exceeding threshold in local radius)
- Sudden acceleration / Erratic movement (velocity surge)
- Unattended stationary objects (backpack/luggage with no person in proximity)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import math
import numpy as np

from ..core.types import Track, AnalyticsEvent, EventType
from ..core.config import IBVAPConfig, default_config


@dataclass
class TrackActivityState:
    anchor_position: Tuple[int, int]
    first_seen_time: float
    stationary_start_time: Optional[float] = None
    last_loitering_alert_time: float = -1e9
    last_speed_alert_time: float = -1e9
    last_unattended_alert_time: float = -1e9


class SuspiciousActivityAnalytics:
    """
    Evaluates tracks against measurable behavioral rules.
    Always includes deterministic diagnostic rationale in metadata.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.loitering_duration = self.config.loitering_duration_seconds
        self.loitering_radius = self.config.loitering_distance_radius_px
        self.speed_threshold = self.config.sudden_acceleration_threshold_px
        self.unattended_duration = self.config.unattended_object_duration_seconds
        self.unattended_radius = self.config.unattended_object_proximity_px

        self.track_states: Dict[Tuple[str, int], TrackActivityState] = {}

    def process_tracks(self, tracks: List[Track], camera_id: str = "camera-01", timestamp: Optional[float] = None) -> List[AnalyticsEvent]:
        now = timestamp if timestamp is not None else time.time()
        events: List[AnalyticsEvent] = []

        active_track_ids = {t.track_id for t in tracks}
        # Cull expired track states for this camera
        for (cam, tid) in list(self.track_states.keys()):
            if cam == camera_id and tid not in active_track_ids:
                del self.track_states[(cam, tid)]

        # Separate persons, vehicles, and objects
        persons = [t for t in tracks if t.class_name == "person"]
        objects = [t for t in tracks if t.class_name in ("backpack", "handbag", "suitcase")]

        for track in tracks:
            tid = track.track_id
            curr_pos = track.center
            state_key = (camera_id, tid)

            if state_key not in self.track_states:
                self.track_states[state_key] = TrackActivityState(
                    anchor_position=curr_pos,
                    first_seen_time=now,
                    stationary_start_time=now
                )

            state = self.track_states[state_key]

            # ── 1. Loitering Detection (Persons) ───────────────────────────
            if track.class_name == "person":
                dist_from_anchor = math.hypot(
                    curr_pos[0] - state.anchor_position[0],
                    curr_pos[1] - state.anchor_position[1]
                )

                if dist_from_anchor <= self.loitering_radius:
                    # Still in vicinity of anchor
                    time_in_vicinity = now - state.first_seen_time
                    if time_in_vicinity >= self.loitering_duration:
                        # Check cooldown (alert once every loitering_duration)
                        if (now - state.last_loitering_alert_time) >= self.loitering_duration:
                            state.last_loitering_alert_time = now
                            events.append(
                                AnalyticsEvent(
                                    camera_id=camera_id,
                                    timestamp=now,
                                    event_type=EventType.LOITERING,
                                    track_id=track.track_id,
                                    identity_id=track.identity_id,
                                    confidence=track.confidence,
                                    metadata={
                                        "reason": "Track remained within radius longer than threshold",
                                        "duration_seconds": round(time_in_vicinity, 1),
                                        "threshold_seconds": self.loitering_duration,
                                        "distance_from_anchor_px": round(dist_from_anchor, 1),
                                        "anchor_radius_px": self.loitering_radius,
                                        "position": curr_pos,
                                    }
                                )
                            )
                else:
                    # Reset anchor position and timer
                    state.anchor_position = curr_pos
                    state.first_seen_time = now

            # ── 2. Sudden Acceleration / Erratic Movement ──────────────────
            vx, vy = track.velocity
            speed = math.hypot(vx, vy)
            if speed >= self.speed_threshold:
                if (now - state.last_speed_alert_time) >= 5.0:
                    state.last_speed_alert_time = now
                    events.append(
                        AnalyticsEvent(
                            camera_id=camera_id,
                            timestamp=now,
                            event_type=EventType.SUSPICIOUS_MOVEMENT,
                            track_id=track.track_id,
                            identity_id=track.identity_id,
                            confidence=track.confidence,
                            metadata={
                                "reason": "Sudden acceleration / abnormal velocity jump",
                                "speed_px_per_frame": round(speed, 2),
                                "speed_threshold": self.speed_threshold,
                                "object_class": track.class_name,
                                "position": curr_pos,
                            }
                        )
                    )

        # ── 3. Unattended Object Detection ─────────────────────────────────
        for obj in objects:
            obj_state = self.track_states.get((camera_id, obj.track_id))
            if not obj_state:
                continue

            obj_vx, obj_vy = obj.velocity
            obj_speed = math.hypot(obj_vx, obj_vy)

            # Object must be stationary
            if obj_speed < 5.0:
                if obj_state.stationary_start_time is None:
                    obj_state.stationary_start_time = now

                stationary_duration = now - obj_state.stationary_start_time

                # Check proximity to nearest person
                min_dist_to_person = float("inf")
                for p in persons:
                    d = math.hypot(obj.center[0] - p.center[0], obj.center[1] - p.center[1])
                    if d < min_dist_to_person:
                        min_dist_to_person = d

                # If no person is nearby and object has been stationary beyond threshold
                if min_dist_to_person > self.unattended_radius and stationary_duration >= self.unattended_duration:
                    if (now - obj_state.last_unattended_alert_time) >= 10.0:
                        obj_state.last_unattended_alert_time = now
                        events.append(
                            AnalyticsEvent(
                                camera_id=camera_id,
                                timestamp=now,
                                event_type=EventType.UNATTENDED_OBJECT,
                                track_id=obj.track_id,
                                confidence=obj.confidence,
                                metadata={
                                    "reason": "Stationary object left without person in proximity",
                                    "object_class": obj.class_name,
                                    "stationary_duration_seconds": round(stationary_duration, 1),
                                    "threshold_seconds": self.unattended_duration,
                                    "nearest_person_distance_px": round(min_dist_to_person, 1) if min_dist_to_person != float("inf") else -1,
                                    "proximity_threshold_px": self.unattended_radius,
                                    "position": obj.center,
                                }
                            )
                        )
            else:
                obj_state.stationary_start_time = now

        return events
