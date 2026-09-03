"""
Night-Time Movement Analytics.
Calculates mean frame illumination / luminance dynamically.
Generates NIGHT_MOVEMENT events when motion or tracked entities appear under low-light conditions.
"""

from typing import List, Optional
import time
import cv2
import numpy as np

from ..core.types import Track, AnalyticsEvent, EventType
from ..core.config import IBVAPConfig, default_config


class NightMovementAnalytics:
    """
    Evaluates frame illumination and tracked motion to detect night-time activity.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.brightness_threshold = self.config.night_brightness_threshold
        self.cooldown = self.config.night_movement_cooldown_seconds
        self.last_alert_times: Dict[str, float] = {}

    def compute_luminance(self, frame: np.ndarray) -> float:
        """
        Calculates average brightness across frame in range [0.0, 255.0].
        Uses sub-sampling for real-time efficiency.
        """
        if frame is None or frame.size == 0:
            return 128.0

        # Subsample frame for performance (step 4x4)
        sample = frame[::4, ::4]
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def process_frame(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        camera_id: str = "camera-01",
        timestamp: Optional[float] = None
    ) -> List[AnalyticsEvent]:
        now = timestamp or time.time()
        luminance = self.compute_luminance(frame)
        is_night = luminance < self.brightness_threshold

        events: List[AnalyticsEvent] = []

        if not is_night:
            return events

        # Filter tracks with significant motion or confirmed presence
        moving_tracks = [
            t for t in tracks
            if t.class_name in ("person", "car", "motorcycle", "truck") and t.hits >= 2
        ]

        last_alert = self.last_alert_times.get(camera_id, 0.0)
        if moving_tracks and (now - last_alert) >= self.cooldown:
            self.last_alert_times[camera_id] = now
            primary_track = moving_tracks[0]

            events.append(
                AnalyticsEvent(
                    camera_id=camera_id,
                    timestamp=now,
                    event_type=EventType.NIGHT_MOVEMENT,
                    track_id=primary_track.track_id,
                    identity_id=primary_track.identity_id,
                    confidence=primary_track.confidence,
                    metadata={
                        "reason": "Movement detected in low-light / night conditions",
                        "frame_luminance": round(luminance, 2),
                        "luminance_threshold": self.brightness_threshold,
                        "moving_objects_count": len(moving_tracks),
                        "primary_class": primary_track.class_name,
                        "position": primary_track.center,
                    }
                )
            )

        return events
