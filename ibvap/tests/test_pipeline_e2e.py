"""
End-to-End Pipeline Integration Test.
Verifies the complete IBVAP frame processing lifecycle using synthetic frames and MockDetector.
"""

import numpy as np
import pytest

from ibvap.core.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import Detection, VirtualBoundary, ZoneType, WatchlistCategory, EventType
from ibvap.detection.object_detector import MockDetector


def test_pipeline_end_to_end_frame_processing():
    mock_detector = MockDetector()
    config = IBVAPConfig(
        tracking_min_hits=1,
        fence_cooldown_seconds=1.0,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    # Setup virtual fence line at x = 300 for cam-test
    pipeline.add_boundary(
        VirtualBoundary(
            id="fence-01",
            name="Test Line",
            zone_type=ZoneType.LINE,
            coordinates=[(300, 0), (300, 500)],
            target_classes=["person"],
            camera_id="cam-test"
        ),
        camera_id="cam-test"
    )

    # Frame 1: Synthetic black BGR frame with person at center x=280 (left of fence line at x=300)
    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_detector.set_detections([
        Detection(bbox=(250, 100, 310, 250), class_id=0, class_name="person", confidence=0.92)
    ])

    res1 = pipeline.process_frame(frame1, camera_id="cam-test", timestamp=10.0)
    assert len(res1.detections) == 1
    assert len(res1.tracks) == 1
    initial_track_id = res1.tracks[0].track_id
    assert len(res1.events) == 0, "Person has not crossed fence yet."

    # Frame 2: Person moves across line to center x=310 (right of fence line at x=300, IoU ~0.33)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_detector.set_detections([
        Detection(bbox=(280, 100, 340, 250), class_id=0, class_name="person", confidence=0.92)
    ])

    res2 = pipeline.process_frame(frame2, camera_id="cam-test", timestamp=10.1)
    assert len(res2.tracks) == 1
    assert res2.tracks[0].track_id == initial_track_id, "Track ID must remain continuous across frames."
    event_types = {e.event_type for e in res2.events}
    assert EventType.FENCE_INTRUSION in event_types, "Crossing the line must trigger FENCE_INTRUSION."
    assert EventType.NIGHT_MOVEMENT in event_types, "Moving in pitch-black frame must trigger NIGHT_MOVEMENT."
    assert res2.events[0].track_id == initial_track_id

    # Test visual debug renderer
    annotated = pipeline.draw_debug(frame2, res2)
    assert annotated is not None
    assert annotated.shape == frame2.shape
    assert annotated.dtype == np.uint8


def test_frame_contract_hd_resolution():
    """
    Explicitly tests input frame contract with 720p HD frame:
    shape: (720, 1280, 3), dtype: uint8, format: BGR
    """
    mock_detector = MockDetector()
    config = IBVAPConfig(
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)
    hd_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    result = pipeline.process_frame(
        frame=hd_frame,
        camera_id="cam-hd-01",
        timestamp=100.0
    )

    assert result.frame_shape == (720, 1280)
    assert isinstance(result.detections, list)
    assert isinstance(result.tracks, list)
    assert isinstance(result.events, list)
    assert result.timestamp == 100.0
    assert result.camera_id == "cam-hd-01"
    assert result.success is True

    # Test dictionary export
    res_dict = result.to_dict()
    assert isinstance(res_dict, dict)
    assert res_dict["camera_id"] == "cam-hd-01"
    assert res_dict["success"] is True
    assert res_dict["frame_shape"] == [720, 1280]


def test_frame_contract_validation_rejects_invalid_inputs():
    """
    Lightweight validation at the public API boundary must reject invalid inputs:
    - None
    - Wrong dimensionality (2D)
    - Wrong channel count (4 channels)
    - Wrong dtype (float32)
    - Empty shape (0, 0, 3)
    """
    test_cfg = IBVAPConfig(redis_enabled=False, db_enabled=False, storage_enabled=False)
    pipeline = IBVAPPipeline(config=test_cfg, detector=MockDetector())

    # 1. Reject None
    with pytest.raises(ValueError, match="Input frame cannot be None"):
        pipeline.process_frame(None)

    # 2. Reject 2D (height, width)
    with pytest.raises(ValueError, match="Invalid frame dimensions"):
        pipeline.process_frame(np.zeros((480, 640), dtype=np.uint8))

    # 3. Reject 4 channels (RGBA)
    with pytest.raises(ValueError, match="Invalid channel count"):
        pipeline.process_frame(np.zeros((480, 640, 4), dtype=np.uint8))

    # 4. Reject float32 dtype
    with pytest.raises(ValueError, match="Invalid frame dtype"):
        pipeline.process_frame(np.zeros((480, 640, 3), dtype=np.float32))

    # 5. Reject empty shape
    with pytest.raises(ValueError, match="Invalid frame shape"):
        pipeline.process_frame(np.zeros((0, 0, 3), dtype=np.uint8))


def test_frame_immutability_caller_frame_not_modified():
    """
    Guarantees the caller's frame is never mutated by process_frame() or draw_debug().
    """
    test_cfg = IBVAPConfig(redis_enabled=False, db_enabled=False, storage_enabled=False)
    mock_detector = MockDetector()
    pipeline = IBVAPPipeline(config=test_cfg, detector=mock_detector)

    # Create random BGR frame
    np.random.seed(42)
    original_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    caller_frame = original_frame.copy()

    mock_detector.set_detections([
        Detection(bbox=(100, 100, 200, 200), class_id=0, class_name="person", confidence=0.9)
    ])

    result = pipeline.process_frame(caller_frame, camera_id="cam-immutability")
    assert np.array_equal(caller_frame, original_frame), "process_frame must not mutate caller frame!"

    annotated = pipeline.draw_debug(caller_frame, result)
    assert np.array_equal(caller_frame, original_frame), "draw_debug must not mutate caller frame!"
    assert not np.array_equal(annotated, original_frame), "annotated frame should contain overlays"


def test_camera_id_tracking_state_isolation():
    """
    Verifies that tracking state is isolated per camera stream.
    Objects in camera-01 and camera-02 must maintain their own tracker states
    and not interfere with or overwrite each other.
    """
    mock_detector = MockDetector()
    config = IBVAPConfig(
        tracking_min_hits=1,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Frame 1 on camera-A: person at (100, 100, 150, 150)
    mock_detector.set_detections([
        Detection(bbox=(100, 100, 150, 150), class_id=0, class_name="person", confidence=0.9)
    ])
    res_a1 = pipeline.process_frame(frame, camera_id="camera-A", timestamp=1.0)
    assert len(res_a1.tracks) == 1
    track_id_a = res_a1.tracks[0].track_id

    # Frame 1 on camera-B: different person at (400, 400, 450, 450)
    mock_detector.set_detections([
        Detection(bbox=(400, 400, 450, 450), class_id=0, class_name="person", confidence=0.9)
    ])
    res_b1 = pipeline.process_frame(frame, camera_id="camera-B", timestamp=1.0)
    assert len(res_b1.tracks) == 1
    track_id_b = res_b1.tracks[0].track_id

    # Both cameras have distinct tracker objects in the pipeline
    tracker_a = pipeline.get_tracker("camera-A")
    tracker_b = pipeline.get_tracker("camera-B")
    assert tracker_a is not tracker_b, "Each camera must have an isolated tracker instance!"

    # Frame 2 on camera-A: person moved slightly to (110, 100, 160, 150)
    mock_detector.set_detections([
        Detection(bbox=(110, 100, 160, 150), class_id=0, class_name="person", confidence=0.9)
    ])
    res_a2 = pipeline.process_frame(frame, camera_id="camera-A", timestamp=1.1)
    assert res_a2.tracks[0].track_id == track_id_a, "Camera-A track continuity must be preserved!"

    # Frame 2 on camera-B: person moved slightly to (410, 400, 460, 450)
    mock_detector.set_detections([
        Detection(bbox=(410, 400, 460, 450), class_id=0, class_name="person", confidence=0.9)
    ])
    res_b2 = pipeline.process_frame(frame, camera_id="camera-B", timestamp=1.1)
    assert res_b2.tracks[0].track_id == track_id_b, "Camera-B track continuity must be preserved!"


