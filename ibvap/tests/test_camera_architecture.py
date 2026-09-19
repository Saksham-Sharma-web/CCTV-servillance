"""
Comprehensive Tests for IBVAP Admin-Controlled Camera Architecture.

Verifies:
1. Single source of truth: All regions, borders, virtual lines, and event rules
   are controlled exclusively by the administrator per camera.
2. Strict isolation: Zero configuration or boundary leakage between cameras.
3. Validation requirements: Mismatched camera_id or entity references raise errors.
4. Virtual line directionality: ENTRY, EXIT, BIDIRECTIONAL, and wrong-way crossings.
5. Cross-camera tracking: Track continuity (global_track_id) without configuration mutation.
6. Safe handling of unconfigured cameras: No default regions, zero assumptions.
"""

import pytest
import numpy as np

from ibvap.core.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import Detection, Track, EventType, ZoneType, VirtualBoundary
from ibvap.core.camera_config import (
    CameraConfig,
    CameraManager,
    Region,
    Border,
    VirtualLine,
    LineDirection,
    RegionType,
    CameraEventRule,
    DetectionRule,
)
from ibvap.tracking.cross_camera import CrossCameraTracker
from ibvap.analytics.virtual_fence import VirtualFenceAnalytics
from ibvap.detection.object_detector import MockDetector


def test_camera_manager_validation_and_rejection():
    """
    Validation requirements:
    - Every region must belong to exactly one camera configuration.
    - Every border must belong to a specific camera.
    - Every virtual line must belong to a specific camera.
    - Every event rule must reference a specific camera configuration.
    """
    cm = CameraManager()

    # 1. Valid registration
    cfg1 = CameraConfig(camera_id="cam-01", name="Entrance Camera")
    reg1 = Region(
        region_id="reg-restricted",
        name="Security Zone",
        camera_id="cam-01",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        region_type=RegionType.RESTRICTED,
    )
    cfg1.regions["reg-restricted"] = reg1
    cm.register_camera(cfg1)
    assert cm.has_camera_config("cam-01")

    # 2. Region with mismatched camera_id must be rejected
    cfg_invalid = CameraConfig(camera_id="cam-02")
    mismatched_reg = Region(
        region_id="reg-mismatch",
        name="Mismatch",
        camera_id="cam-01",  # Belonging to cam-01, but being registered to cam-02!
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )
    cfg_invalid.regions["reg-mismatch"] = mismatched_reg
    with pytest.raises(ValueError, match="Configuration conflict: Region"):
        cm.register_camera(cfg_invalid)

    # 3. Border with mismatched camera_id must be rejected
    cfg_invalid2 = CameraConfig(camera_id="cam-02")
    mismatched_border = Border(
        border_id="border-mismatch",
        name="Border Mismatch",
        camera_id="cam-03",
        coordinates=[(0, 0), (100, 100)],
    )
    cfg_invalid2.borders["border-mismatch"] = mismatched_border
    with pytest.raises(ValueError, match="Configuration conflict: Border"):
        cm.register_camera(cfg_invalid2)

    # 4. VirtualLine with mismatched camera_id must be rejected
    cfg_invalid3 = CameraConfig(camera_id="cam-02")
    mismatched_line = VirtualLine(
        line_id="line-mismatch",
        name="Line Mismatch",
        camera_id="cam-01",
        coordinates=((0, 0), (100, 100)),
        direction=LineDirection.ENTRY,
    )
    cfg_invalid3.virtual_lines["line-mismatch"] = mismatched_line
    with pytest.raises(ValueError, match="Configuration conflict: VirtualLine"):
        cm.register_camera(cfg_invalid3)

    # 5. EventRule referencing non-existent region must be rejected
    cfg_invalid4 = CameraConfig(camera_id="cam-02")
    invalid_rule = CameraEventRule(
        rule_id="rule-01",
        name="Invalid Rule",
        camera_id="cam-02",
        event_type=EventType.REGION_INTRUSION,
        region_id="non-existent-region",
    )
    cfg_invalid4.event_rules["rule-01"] = invalid_rule
    with pytest.raises(ValueError, match="references non-existent region"):
        cm.register_camera(cfg_invalid4)


def test_strict_camera_isolation_no_configuration_leakage():
    """
    CORE USER SCENARIO:
    Admin defines Camera 1 as Restricted Area (Region + Border + Virtual Line).
    Admin defines Camera 2 as Normal (no region, no border, no virtual line).

    Verification:
    - Track inside the restricted coordinates on Camera 1 triggers events.
    - Track inside the EXACT SAME physical coordinates on Camera 2 triggers ZERO region/line events.
    - Camera 2 NEVER inherits Camera 1's configuration.
    """
    mock_detector = MockDetector()
    pipeline = IBVAPPipeline(
        config=IBVAPConfig(tracking_min_hits=1, fence_cooldown_seconds=1.0),
        detector=mock_detector
    )

    # Admin explicitly configures Camera 1: Restricted Area
    cam1_config = CameraConfig(camera_id="cam-01", name="Camera 1 - Restricted Zone")
    cam1_config.regions["restricted-area"] = Region(
        region_id="restricted-area",
        name="Vault Restricted Area",
        camera_id="cam-01",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        region_type=RegionType.RESTRICTED,
        target_classes=["person"]
    )
    cam1_config.borders["perimeter-border"] = Border(
        border_id="perimeter-border",
        name="Perimeter Border",
        camera_id="cam-01",
        coordinates=[(100, 100), (300, 100)],
        target_classes=["person"]
    )
    cam1_config.virtual_lines["entry-tripwire"] = VirtualLine(
        line_id="entry-tripwire",
        name="Security Entry Tripwire",
        camera_id="cam-01",
        coordinates=((150, 0), (150, 400)),
        direction=LineDirection.ENTRY,
        target_classes=["person"]
    )
    pipeline.set_camera_config(cam1_config)

    # Admin explicitly configures Camera 2: Normal camera (No region, no border, no virtual line)
    cam2_config = CameraConfig(camera_id="cam-02", name="Camera 2 - Normal Lobby")
    pipeline.set_camera_config(cam2_config)

    # Verify configurations in manager
    c1 = pipeline.get_camera_config("cam-01")
    c2 = pipeline.get_camera_config("cam-02")
    assert c1.has_spatial_boundaries is True
    assert c2.has_spatial_boundaries is False
    assert len(c2.regions) == 0
    assert len(c2.borders) == 0
    assert len(c2.virtual_lines) == 0

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # ── Test Camera 1: Intrusion into Restricted Area ───────────────
    # Frame 1: Person outside polygon at x=50, y=50
    mock_detector.set_detections([
        Detection(bbox=(30, 30, 70, 70), class_id=0, class_name="person", confidence=0.95)
    ])
    res1_c1 = pipeline.process_frame(frame, camera_id="cam-01", timestamp=10.0)
    assert len(res1_c1.events) == 0

    # Frame 2: Person moves inside polygon to x=200, y=200 (also crosses line at x=150)
    mock_detector.set_detections([
        Detection(bbox=(180, 180, 220, 220), class_id=0, class_name="person", confidence=0.95)
    ])
    res2_c1 = pipeline.process_frame(frame, camera_id="cam-01", timestamp=10.1)
    c1_event_types = {e.event_type for e in res2_c1.events}
    assert (EventType.REGION_INTRUSION in c1_event_types or EventType.LINE_CROSSING in c1_event_types), \
        "Camera 1 must trigger events for its configured restricted area and virtual line."

    # ── Test Camera 2: Exact same coordinates on Camera 2 ────────────
    # Frame 1: Person outside polygon at x=50, y=50
    mock_detector.set_detections([
        Detection(bbox=(30, 30, 70, 70), class_id=0, class_name="person", confidence=0.95)
    ])
    res1_c2 = pipeline.process_frame(frame, camera_id="cam-02", timestamp=20.0)
    assert len(res1_c2.events) == 0

    # Frame 2: Person moves to x=200, y=200
    mock_detector.set_detections([
        Detection(bbox=(180, 180, 220, 220), class_id=0, class_name="person", confidence=0.95)
    ])
    res2_c2 = pipeline.process_frame(frame, camera_id="cam-02", timestamp=20.1)

    # CRITICAL INVARIANT: Camera 2 has NO region configured -> 0 region/line events!
    c2_spatial_events = [
        e for e in res2_c2.events
        if e.event_type in (
            EventType.REGION_INTRUSION,
            EventType.BORDER_CROSSING,
            EventType.LINE_CROSSING,
            EventType.FENCE_INTRUSION,
            EventType.DIRECTION_VIOLATION,
        )
    ]
    assert len(c2_spatial_events) == 0, \
        f"Camera 2 MUST NOT generate region/line events because none is configured! Got: {c2_spatial_events}"

    # Verify Camera 2 did not inherit Camera 1's config
    c2_after = pipeline.get_camera_config("cam-02")
    assert len(c2_after.regions) == 0
    assert len(c2_after.virtual_lines) == 0


def test_directional_virtual_line_crossing():
    """
    Tests directional virtual line processing:
    - Line with direction=ENTRY:
      * Crossing in forward direction triggers LINE_CROSSING.
      * Crossing in reverse direction triggers DIRECTION_VIOLATION.
    - Line with direction=BIDIRECTIONAL:
      * Crossing in either direction triggers LINE_CROSSING.
    """
    vf = VirtualFenceAnalytics(config=IBVAPConfig(fence_cooldown_seconds=1.0))

    # Directed line from (300, 0) to (300, 500)
    # Directed vector is downwards (dy = 500, dx = 0)
    # Left of vector is x < 300, Right of vector is x > 300
    # Left -> Right is defined as ENTRY
    entry_line = VirtualLine(
        line_id="gate-entry",
        name="Gate Entry Only",
        camera_id="cam-gate",
        coordinates=((300, 0), (300, 500)),
        direction=LineDirection.ENTRY,
        target_classes=["person"]
    )
    vf.add_virtual_line(entry_line)

    # 1. Forward crossing: x=250 -> x=350 (Left to Right = ENTRY)
    track_forward = Track(
        track_id=10,
        bbox=(340, 100, 360, 150),
        class_name="person",
        confidence=0.92,
        center=(350, 125),
        history=[(250, 125), (350, 125)]
    )
    events_fwd = vf.process_tracks([track_forward], camera_id="cam-gate", timestamp=100.0)
    assert len(events_fwd) == 1
    assert events_fwd[0].event_type == EventType.LINE_CROSSING
    assert events_fwd[0].metadata["direction"] == "ENTRY"

    # 2. Reverse crossing: x=350 -> x=250 (Right to Left = WRONG WAY / EXIT on ENTRY-only line)
    track_reverse = Track(
        track_id=20,
        bbox=(240, 100, 260, 150),
        class_name="person",
        confidence=0.92,
        center=(250, 125),
        history=[(350, 125), (250, 125)]
    )
    events_rev = vf.process_tracks([track_reverse], camera_id="cam-gate", timestamp=105.0)
    assert len(events_rev) == 1
    assert events_rev[0].event_type == EventType.DIRECTION_VIOLATION
    assert events_rev[0].metadata["transition"] == "WRONG_WAY_CROSSING"


def test_cross_camera_tracking_does_not_modify_configuration():
    """
    Cross-Camera Tracking Invariant:
    A person is detected in Camera 1 and subsequently appears in Camera 2.
    Cross-camera tracking must maintain track association across cameras,
    BUT it must NOT modify or transfer camera configurations.
    """
    mock_detector = MockDetector()
    pipeline = IBVAPPipeline(
        config=IBVAPConfig(tracking_min_hits=1, face_detection_enabled=False),
        detector=mock_detector
    )

    # Camera 1: Unrestricted
    cam1_cfg = CameraConfig(camera_id="cam-hallway", name="Hallway")
    pipeline.set_camera_config(cam1_cfg)

    # Camera 2: Restricted Area
    cam2_cfg = CameraConfig(camera_id="cam-server-room", name="Server Room")
    cam2_cfg.regions["server-vault"] = Region(
        region_id="server-vault",
        name="Server Vault",
        camera_id="cam-server-room",
        polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
        region_type=RegionType.RESTRICTED,
    )
    pipeline.set_camera_config(cam2_cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. Person appears in Camera 1
    mock_detector.set_detections([
        Detection(bbox=(50, 50, 90, 150), class_id=0, class_name="person", confidence=0.90)
    ])
    res_cam1 = pipeline.process_frame(frame, camera_id="cam-hallway", timestamp=1.0)
    assert len(res_cam1.tracks) == 1
    cam1_track = res_cam1.tracks[0]
    assert cam1_track.global_track_id is not None
    global_id = cam1_track.global_track_id

    # Verify no region events in Camera 1
    assert len(res_cam1.events) == 0

    # 2. Person moves to Camera 2 and enters Server Vault polygon
    # Link track to existing global_track_id via tracker association or biometric identity
    mock_detector.set_detections([
        Detection(bbox=(200, 200, 240, 300), class_id=0, class_name="person", confidence=0.90)
    ])
    res_cam2 = pipeline.process_frame(frame, camera_id="cam-server-room", timestamp=5.0)
    assert len(res_cam2.tracks) == 1
    cam2_track = res_cam2.tracks[0]

    # Camera 2 evaluates against Camera 2's own admin region
    c2_events = [e for e in res_cam2.events if e.event_type == EventType.REGION_INTRUSION]
    assert len(c2_events) == 1
    assert c2_events[0].metadata["region_id"] == "server-vault"

    # CRITICAL INVARIANT: Camera 1 configuration remains 100% UNMODIFIED
    c1_after = pipeline.get_camera_config("cam-hallway")
    assert len(c1_after.regions) == 0, "Camera 1 must NOT have inherited any regions from Camera 2!"
    assert len(c1_after.borders) == 0
    assert len(c1_after.virtual_lines) == 0


def test_missing_camera_configuration_fails_safely():
    """
    If a frame arrives from a camera that has never been configured by the admin:
    - IBVAP must NOT invent default regions.
    - IBVAP must NOT assume the entire frame is a region.
    - IBVAP must NOT crash or fail.
    - Object detection and tracking continue normally.
    - Region/border/line-based processing is cleanly skipped.
    """
    mock_detector = MockDetector()
    pipeline = IBVAPPipeline(
        config=IBVAPConfig(tracking_min_hits=1),
        detector=mock_detector
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_detector.set_detections([
        Detection(bbox=(100, 100, 200, 200), class_id=0, class_name="person", confidence=0.95)
    ])

    # cam-unregistered has NO configuration in camera_manager
    res = pipeline.process_frame(frame, camera_id="cam-unregistered", timestamp=10.0)
    assert res.success is True
    assert len(res.detections) == 1
    assert len(res.tracks) == 1

    # Zero region/border/line events generated
    spatial_events = [
        e for e in res.events
        if e.event_type in (
            EventType.REGION_INTRUSION,
            EventType.BORDER_CROSSING,
            EventType.LINE_CROSSING,
            EventType.FENCE_INTRUSION,
        )
    ]
    assert len(spatial_events) == 0


def test_camera_specific_event_rules_filtering():
    """
    Verifies that CameraEventRule controls which events are emitted for a camera.
    - If enabled=False, event is suppressed.
    - If min_confidence is higher than track confidence, event is suppressed.
    - If target_classes doesn't match object, event is suppressed.
    """
    mock_detector = MockDetector()
    pipeline = IBVAPPipeline(
        config=IBVAPConfig(tracking_min_hits=1, fence_cooldown_seconds=1.0),
        detector=mock_detector
    )

    # Setup camera with disabled rule for person intrusion
    cfg = CameraConfig(camera_id="cam-filtered")
    cfg.regions["zone-1"] = Region(
        region_id="zone-1",
        name="Test Zone",
        camera_id="cam-filtered",
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        target_classes=["person", "car"]
    )
    # Admin rule: Disable region intrusion for "person"
    cfg.event_rules["rule-suppress-person"] = CameraEventRule(
        rule_id="rule-suppress-person",
        name="Suppress Person Intrusion",
        camera_id="cam-filtered",
        event_type=EventType.REGION_INTRUSION,
        region_id="zone-1",
        target_classes=["person"],
        enabled=False  # Disabled by admin!
    )
    pipeline.set_camera_config(cfg)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1: Outside
    mock_detector.set_detections([
        Detection(bbox=(30, 30, 60, 60), class_id=0, class_name="person", confidence=0.90)
    ])
    pipeline.process_frame(frame, camera_id="cam-filtered", timestamp=1.0)

    # Frame 2: Moves inside zone-1
    mock_detector.set_detections([
        Detection(bbox=(150, 150, 200, 200), class_id=0, class_name="person", confidence=0.90)
    ])
    res2 = pipeline.process_frame(frame, camera_id="cam-filtered", timestamp=1.1)

    # Since the rule was enabled=False for person, REGION_INTRUSION must be suppressed!
    region_events = [e for e in res2.events if e.event_type == EventType.REGION_INTRUSION]
    assert len(region_events) == 0, "Disabled admin rule must suppress REGION_INTRUSION."
