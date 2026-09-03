"""
Unit tests for Suspicious Activity Analytics.
Verifies Loitering, Sudden Movement, and Unattended Object detection.
"""

import pytest
from ibvap.core.types import Track, EventType
from ibvap.core.config import IBVAPConfig
from ibvap.analytics.suspicious_activity import SuspiciousActivityAnalytics


def test_loitering_duration_threshold():
    config = IBVAPConfig(loitering_duration_seconds=10.0, loitering_distance_radius_px=50.0)
    sa = SuspiciousActivityAnalytics(config=config)

    # Frame at t=0.0: Person stands at (200, 200)
    track_t0 = Track(
        track_id=10,
        bbox=(180, 150, 220, 250),
        class_name="person",
        confidence=0.9,
        center=(200, 200)
    )
    events_t0 = sa.process_tracks([track_t0], timestamp=0.0)
    assert len(events_t0) == 0, "No loitering at start."

    # Frame at t=5.0: Still at (205, 202) -> 5s elapsed (< 10s)
    track_t5 = Track(
        track_id=10,
        bbox=(185, 152, 225, 252),
        class_name="person",
        confidence=0.9,
        center=(205, 202)
    )
    events_t5 = sa.process_tracks([track_t5], timestamp=5.0)
    assert len(events_t5) == 0, "Duration below threshold must NOT trigger LOITERING."

    # Frame at t=11.0: Still at (208, 204) -> 11s elapsed (> 10s)
    track_t11 = Track(
        track_id=10,
        bbox=(188, 154, 228, 254),
        class_name="person",
        confidence=0.9,
        center=(208, 204)
    )
    events_t11 = sa.process_tracks([track_t11], timestamp=11.0)
    assert len(events_t11) == 1, "Duration exceeding threshold must trigger LOITERING."
    assert events_t11[0].event_type == EventType.LOITERING
    assert events_t11[0].track_id == 10
    assert events_t11[0].metadata["duration_seconds"] >= 10.0
    assert "threshold_seconds" in events_t11[0].metadata


def test_sudden_acceleration():
    config = IBVAPConfig(sudden_acceleration_threshold_px=80.0)
    sa = SuspiciousActivityAnalytics(config=config)

    # Normal speed
    track_normal = Track(
        track_id=12,
        bbox=(100, 100, 150, 200),
        class_name="person",
        confidence=0.9,
        center=(125, 150),
        velocity=(10.0, 5.0)
    )
    events_normal = sa.process_tracks([track_normal], timestamp=10.0)
    assert len(events_normal) == 0

    # High speed jump (velocity = (80.0, 60.0) -> speed = 100px/frame)
    track_speeding = Track(
        track_id=12,
        bbox=(180, 160, 230, 260),
        class_name="person",
        confidence=0.9,
        center=(205, 210),
        velocity=(80.0, 60.0)
    )
    events_speeding = sa.process_tracks([track_speeding], timestamp=11.0)
    assert len(events_speeding) == 1
    assert events_speeding[0].event_type == EventType.SUSPICIOUS_MOVEMENT
    assert events_speeding[0].metadata["speed_px_per_frame"] == 100.0


def test_unattended_stationary_object():
    config = IBVAPConfig(unattended_object_duration_seconds=10.0, unattended_object_proximity_px=100.0)
    sa = SuspiciousActivityAnalytics(config=config)

    # Stationary backpack at (300, 300)
    backpack = Track(
        track_id=99,
        bbox=(280, 280, 320, 320),
        class_name="backpack",
        confidence=0.85,
        center=(300, 300),
        velocity=(0.0, 0.0)
    )

    # Person nearby at (320, 310) -> distance < 100px
    person_near = Track(
        track_id=1,
        bbox=(310, 250, 350, 370),
        class_name="person",
        confidence=0.9,
        center=(330, 310)
    )

    # t=0: person is near
    events_0 = sa.process_tracks([backpack, person_near], timestamp=0.0)
    assert len(events_0) == 0

    # Person walks far away: at (800, 800) -> distance ~ 707px (> 100px)
    person_far = Track(
        track_id=1,
        bbox=(780, 750, 820, 850),
        class_name="person",
        confidence=0.9,
        center=(800, 800)
    )

    # t=5: unattended for 5s (< 10s)
    events_5 = sa.process_tracks([backpack, person_far], timestamp=5.0)
    assert len(events_5) == 0

    # t=12: unattended for 12s (> 10s)
    events_12 = sa.process_tracks([backpack, person_far], timestamp=12.0)
    assert len(events_12) == 1
    assert events_12[0].event_type == EventType.UNATTENDED_OBJECT
    assert events_12[0].track_id == 99
    assert events_12[0].metadata["stationary_duration_seconds"] >= 10.0
