"""
Unit tests for Event Engine.
Verifies event deduplication, cooldown debouncing, and multi-track dispatching.
"""

import pytest
from ibvap.core.types import AnalyticsEvent, EventType
from ibvap.core.config import IBVAPConfig
from ibvap.events.event_engine import EventEngine


def test_event_deduplication_flooding_suppression():
    config = IBVAPConfig(event_deduplication_window_seconds=3.0)
    engine = EventEngine(config=config)

    # 10 consecutive intrusion events within 1 second for the same track
    raw_events = [
        AnalyticsEvent(
            camera_id="cam-01",
            timestamp=100.0 + (i * 0.1),
            event_type=EventType.FENCE_INTRUSION,
            track_id=17,
            metadata={"zone_id": "zone-A"}
        )
        for i in range(10)
    ]

    emitted = engine.filter_and_emit(raw_events)
    assert len(emitted) == 1, "10 consecutive intrusion alerts in 1s must be debounced to exactly 1 emitted alert."
    assert emitted[0].track_id == 17


def test_event_re_emission_after_cooldown():
    config = IBVAPConfig(event_deduplication_window_seconds=3.0)
    engine = EventEngine(config=config)

    # First event at t=10.0
    ev1 = [
        AnalyticsEvent(
            camera_id="cam-01",
            timestamp=10.0,
            event_type=EventType.FENCE_INTRUSION,
            track_id=17,
            metadata={"zone_id": "zone-A"}
        )
    ]
    assert len(engine.filter_and_emit(ev1)) == 1

    # Second event at t=11.5 (< 3.0s window) -> should be suppressed
    ev2 = [
        AnalyticsEvent(
            camera_id="cam-01",
            timestamp=11.5,
            event_type=EventType.FENCE_INTRUSION,
            track_id=17,
            metadata={"zone_id": "zone-A"}
        )
    ]
    assert len(engine.filter_and_emit(ev2)) == 0, "Event within cooldown window must be suppressed."

    # Third event at t=14.0 (> 3.0s window) -> should be emitted
    ev3 = [
        AnalyticsEvent(
            camera_id="cam-01",
            timestamp=14.0,
            event_type=EventType.FENCE_INTRUSION,
            track_id=17,
            metadata={"zone_id": "zone-A"}
        )
    ]
    assert len(engine.filter_and_emit(ev3)) == 1, "Event after cooldown window expires must be emitted."


def test_different_tracks_not_suppressed():
    config = IBVAPConfig(event_deduplication_window_seconds=3.0)
    engine = EventEngine(config=config)

    # Event for Track 1 and Track 2 at same timestamp
    events = [
        AnalyticsEvent(
            camera_id="cam-01",
            timestamp=50.0,
            event_type=EventType.FENCE_INTRUSION,
            track_id=1,
            metadata={"zone_id": "zone-A"}
        ),
        AnalyticsEvent(
            camera_id="cam-01",
            timestamp=50.0,
            event_type=EventType.FENCE_INTRUSION,
            track_id=2,
            metadata={"zone_id": "zone-A"}
        )
    ]

    emitted = engine.filter_and_emit(events)
    assert len(emitted) == 2, "Events for distinct tracks must both be emitted."
