"""
Unit tests for new behavior detectors:
- RouteDeviationDetector
- CheckpointMonitor
- CrowdDetector
"""

import pytest
import time
from ibvap.core.types import Track, EventType
from ibvap.core.camera_config import LineDirection
from ibvap.behavior.route_deviation import RouteDeviationDetector, PermittedRoute
from ibvap.behavior.checkpoint import CheckpointMonitor, CheckpointGate
from ibvap.behavior.crowd import CrowdDetector


def make_track(track_id: int, center: tuple, history: list, class_name: str = "person") -> Track:
    x, y = center
    return Track(
        track_id=track_id,
        bbox=(x - 20, y - 40, x + 20, y + 40),
        class_name=class_name,
        confidence=0.90,
        center=center,
        velocity=(1.0, 0.0),
        age=10,
        hits=5,
        history=history,
    )


# ── Route Deviation Tests ───────────────────────────────────────────
def test_route_deviation_normal_and_divergent():
    detector = RouteDeviationDetector()
    route = PermittedRoute(
        route_id="r1",
        name="Main Hallway Corridor",
        waypoints=[(100, 100), (500, 100)],
        corridor_width_px=50.0,
        camera_id="cam1",
    )
    detector.add_route(route)

    # Track 1 is within corridor (y = 110, distance is 10px <= 50px)
    t1 = make_track(1, (250, 110), [(240, 110), (250, 110)])
    events = detector.process([t1], camera_id="cam1", timestamp=100.0)
    assert len(events) == 0

    # Track 2 is far away (y = 200, distance is 100px > 50px)
    t2 = make_track(2, (250, 200), [(240, 200), (250, 200)])
    events = detector.process([t2], camera_id="cam1", timestamp=101.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.track_id == 2
    assert ev.metadata["sub_type"] == "ROUTE_DEVIATION"
    assert ev.metadata["deviation_distance_px"] >= 50.0


# ── Checkpoint Monitor Tests ─────────────────────────────────────────
def test_checkpoint_passage_and_violation():
    monitor = CheckpointMonitor()
    # Vertical gate line from (300, 0) to (300, 600)
    # Allows ONLY LEFT_TO_RIGHT crossing
    gate = CheckpointGate(
        gate_id="g1",
        name="Turnstile North",
        line_start=(300, 0),
        line_end=(300, 600),
        allowed_direction=LineDirection.LEFT_TO_RIGHT,
        camera_id="cam1",
    )
    monitor.add_gate(gate)

    # 1. Valid crossing: from x=280 to x=320 (Left to Right)
    t_valid = make_track(1, (320, 200), [(280, 200), (320, 200)])
    events = monitor.process([t_valid], camera_id="cam1", timestamp=200.0)
    assert len(events) == 1
    assert events[0].metadata["is_violation"] is False
    assert monitor.get_stats("g1")["inbound"] == 1
    assert monitor.get_stats("g1")["violations"] == 0

    # 2. Invalid reverse crossing: from x=320 to x=280 (Right to Left)
    t_invalid = make_track(2, (280, 250), [(320, 250), (280, 250)])
    events_rev = monitor.process([t_invalid], camera_id="cam1", timestamp=201.0)
    assert len(events_rev) == 1
    assert events_rev[0].metadata["is_violation"] is True
    assert monitor.get_stats("g1")["outbound"] == 1
    assert monitor.get_stats("g1")["violations"] == 1


# ── Crowd Gathering Tests ────────────────────────────────────────────
def test_crowd_clustering_and_detection():
    detector = CrowdDetector(crowd_threshold=3, cluster_radius_px=100.0)

    # 4 people standing close to each other near (200, 200)
    t1 = make_track(1, (190, 200), [(190, 200)])
    t2 = make_track(2, (200, 210), [(200, 210)])
    t3 = make_track(3, (210, 195), [(210, 195)])
    t4 = make_track(4, (205, 205), [(205, 205)])

    # 1 person far away at (800, 800)
    t5 = make_track(5, (800, 800), [(800, 800)])

    events = detector.process([t1, t2, t3, t4, t5], camera_id="cam1", timestamp=300.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.metadata["sub_type"] == "CROWD_GATHERING"
    assert ev.metadata["headcount"] == 4
    assert 5 not in ev.metadata["involved_tracks"]
    assert set(ev.metadata["involved_tracks"]) == {1, 2, 3, 4}
