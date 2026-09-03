"""
Unit tests for Virtual Fence Analytics.
Verifies polygon intrusion, line crossing, and suppression of duplicate alerts.
"""

import pytest
from ibvap.core.types import Track, VirtualBoundary, ZoneType, EventType
from ibvap.core.config import IBVAPConfig
from ibvap.analytics.virtual_fence import VirtualFenceAnalytics


def test_polygon_intrusion_and_debouncing():
    config = IBVAPConfig(fence_cooldown_seconds=10.0)
    vf = VirtualFenceAnalytics(config=config)

    # Define restricted square polygon zone: (100, 100) to (300, 300)
    zone = VirtualBoundary(
        id="zone-A",
        name="Restricted Area",
        zone_type=ZoneType.POLYGON,
        coordinates=[(100, 100), (300, 100), (300, 300), (100, 300)],
        target_classes=["person"]
    )
    vf.add_boundary(zone)

    # Step 1: Outside -> Outside
    track_outside = Track(
        track_id=1,
        bbox=(20, 20, 50, 70),
        class_name="person",
        confidence=0.95,
        center=(35, 45),
        history=[(30, 40), (35, 45)]
    )
    events1 = vf.process_tracks([track_outside], timestamp=100.0)
    assert len(events1) == 0, "Outside -> outside must produce no events."

    # Step 2: Outside -> Inside (Intrusion transition!)
    track_inside = Track(
        track_id=1,
        bbox=(180, 180, 220, 240),
        class_name="person",
        confidence=0.95,
        center=(200, 210),
        history=[(35, 45), (200, 210)]
    )
    events2 = vf.process_tracks([track_inside], timestamp=101.0)
    assert len(events2) == 1, "Outside -> inside transition must trigger FENCE_INTRUSION."
    assert events2[0].event_type == EventType.FENCE_INTRUSION
    assert events2[0].track_id == 1
    assert events2[0].metadata["zone_id"] == "zone-A"

    # Step 3: Inside -> Inside (Consecutive frames inside zone)
    track_still_inside = Track(
        track_id=1,
        bbox=(185, 185, 225, 245),
        class_name="person",
        confidence=0.95,
        center=(205, 215),
        history=[(200, 210), (205, 215)]
    )
    events3 = vf.process_tracks([track_still_inside], timestamp=102.0)
    assert len(events3) == 0, "Remaining inside zone must NOT emit duplicate alerts."


def test_line_crossing_intrusion():
    config = IBVAPConfig(fence_cooldown_seconds=10.0)
    vf = VirtualFenceAnalytics(config=config)

    # Line fence vertically at x = 500, from y=0 to y=1000
    line_fence = VirtualBoundary(
        id="line-fence-01",
        name="North Border",
        zone_type=ZoneType.LINE,
        coordinates=[(500, 0), (500, 1000)],
        target_classes=["car"]
    )
    vf.add_boundary(line_fence)

    # Car moves from x=450 to x=550 (crosses line x=500)
    car_crossing = Track(
        track_id=42,
        bbox=(520, 300, 580, 400),
        class_name="car",
        confidence=0.91,
        center=(550, 350),
        history=[(450, 350), (550, 350)]
    )
    events = vf.process_tracks([car_crossing], timestamp=200.0)
    assert len(events) == 1
    assert events[0].event_type == EventType.FENCE_INTRUSION
    assert events[0].metadata["zone_type"] == "LINE"
