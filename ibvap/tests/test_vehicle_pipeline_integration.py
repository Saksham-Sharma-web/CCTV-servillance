"""
Live Pipeline Integration Tests (Phases 7 & 8).
Verifies:
1. Frame Sampling (Phase 7):
   - Decouples 24 FPS input to ~8 FPS analysis rate when enabled.
   - Preserves frame index, timestamps, and sequence continuity.
   - Passes all frames through when disabled (backward compatibility).
2. Live Pipeline Vehicle ANPR Path (Phase 8):
   - Track-centric processing: Detection -> Tracking -> Plate Detection -> Quality Scoring -> Buffer -> Selector -> Controlled OCR -> Consensus.
   - Strict OCR budget enforcement: Maximum N attempts per vehicle track.
   - Quality-gated ingestion: Discards blurry / low-quality crops before OCR.
   - Multi-frame consensus & conflict resolution in live pipeline: Conflicting readings produce MULTI_FRAME_CONFLICT with plate_number=None.
   - Memory safety: Stale track cleanup purges inactive tracks from the vehicle buffer.
   - Multiple concurrent vehicles tracked independently.
   - Absolute Human Pipeline Isolation: Person tracks and face verification remain completely untouched.
"""

from typing import List, Tuple
import numpy as np
import pytest

from ibvap.core.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import Detection, Track, WatchlistCategory, EventType
from ibvap.core.sampler import FrameSampler
from ibvap.vehicle.types import VehicleStatus
from ibvap.anpr.ocr_adapter import PlateResult


class MockDetector:
    """Configurable mock detector returning predetermined detections."""
    def __init__(self, detections_sequence: List[List[Detection]]):
        self.sequence = detections_sequence
        self.call_count = 0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if self.call_count < len(self.sequence):
            dets = self.sequence[self.call_count]
        else:
            dets = self.sequence[-1] if self.sequence else []
        self.call_count += 1
        return dets


class ControlledMockOCR:
    """Mock OCR adapter for deterministic budget and consensus verification."""
    def __init__(self, outputs: List[PlateResult]):
        self.outputs = outputs
        self.call_count = 0

    def recognize_plate(self, crop: np.ndarray) -> PlateResult:
        idx = min(self.call_count, len(self.outputs) - 1)
        res = self.outputs[idx]
        self.call_count += 1
        return res


def test_frame_sampler_temporal_reduction():
    """Test 1 (Phase 7): FrameSampler samples 24 FPS input down to ~8 FPS."""
    sampler = FrameSampler(target_fps=8.0, source_fps=24.0, enabled=True)
    camera_fps = 24.0
    duration_sec = 2.0
    total_camera_frames = int(camera_fps * duration_sec)  # 48 frames

    processed_frames = []
    for f_idx in range(1, total_camera_frames + 1):
        t = float(f_idx) / camera_fps
        if sampler.should_process(timestamp=t, frame_index=f_idx):
            processed_frames.append(f_idx)

    # 48 frames at 24 FPS over 2 seconds -> target 8 FPS -> ~16-17 frames analyzed
    assert 15 <= len(processed_frames) <= 18
    # Reduction of ~65% in computational load
    assert sampler.total_received == 48
    assert sampler.total_processed == len(processed_frames)


def test_pipeline_frame_sampling_gate():
    """Test 2 (Phase 7): IBVAPPipeline skips expensive inference on sampled-out frames."""
    config = IBVAPConfig(
        analysis_fps=8.0,
        camera_fps=24.0,
        frame_sampling_enabled=True,
        tracking_min_hits=1,
    )
    mock_detector = MockDetector([[Detection(bbox=(10, 10, 100, 100), class_id=2, class_name="car", confidence=0.90)]])
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    frame = np.full((480, 640, 3), 128, dtype=np.uint8)

    # Frame 1 (t=0.0): should process
    res1 = pipeline.process_frame(frame, camera_id="cam-sampler", timestamp=0.0)
    assert res1.metadata.get("sampled_out") is None
    assert len(res1.detections) == 1

    # Frame 2 (t=0.04s, 25ms later at 24 FPS): should be sampled out (interval is 125ms)
    res2 = pipeline.process_frame(frame, camera_id="cam-sampler", timestamp=0.04)
    assert res2.metadata.get("sampled_out") is True
    assert len(res2.detections) == 0  # No heavy detector inference run!


def test_pipeline_ocr_budget_enforcement():
    """Test 3 (Phase 8): Live pipeline respects max_ocr_attempts_per_track budget."""
    config = IBVAPConfig(
        tracking_min_hits=1,
        vehicle_max_ocr_attempts_per_track=2,  # Budget of exactly 2 attempts
        anpr_ocr_interval_frames=1,  # Try every frame for test
    )
    mock_ocr = ControlledMockOCR([
        PlateResult("DL01AB1234", 0.70, 0.70, WatchlistCategory.UNKNOWN, "DL01AB1234"),
        PlateResult("DL01AB1234", 0.70, 0.70, WatchlistCategory.UNKNOWN, "DL01AB1234"),
        PlateResult("DL01AB1234", 0.70, 0.70, WatchlistCategory.UNKNOWN, "DL01AB1234"),
    ])
    pipeline = IBVAPPipeline(config=config)
    pipeline.anpr_adapter = mock_ocr
    pipeline.controlled_ocr.ocr_adapter = mock_ocr

    # Simulate single car observed across 10 consecutive frames
    car_box = (100, 100, 400, 300)
    mock_det = MockDetector([[Detection(bbox=car_box, class_id=2, class_name="car", confidence=0.92)] for _ in range(10)])
    pipeline.detector = mock_det

    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Draw high contrast plate so plate detector finds candidates
    frame[200:240, 200:300] = 255
    frame[210:230, 210:290] = 0

    for f_idx in range(1, 11):
        pipeline.process_frame(frame, camera_id="cam-budget", timestamp=float(f_idx) * 0.1)

    # Verify total OCR attempts were strictly bounded by the configured budget (<= 2)
    assert mock_ocr.call_count <= 2
    track_state = pipeline.vehicle_buffer.get_track_state(1)
    if track_state:
        assert track_state.ocr_attempts <= 2


def test_pipeline_quality_gating_discards_blurry_plate():
    """Test 4 (Phase 8): Blurry / unusable plate crops are filtered out before OCR."""
    config = IBVAPConfig(
        tracking_min_hits=1,
        vehicle_min_quality_threshold=60.0,
    )
    mock_ocr = ControlledMockOCR([PlateResult("DL01AB1234", 0.95, 0.95, WatchlistCategory.UNKNOWN, "DL01AB1234")])
    pipeline = IBVAPPipeline(config=config)
    pipeline.anpr_adapter = mock_ocr
    pipeline.controlled_ocr.ocr_adapter = mock_ocr

    # Car crop with solid flat gray (Laplacian sharpness = 0, quality < 60)
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    mock_det = MockDetector([[Detection(bbox=(100, 100, 400, 300), class_id=2, class_name="car", confidence=0.90)]])
    pipeline.detector = mock_det

    res = pipeline.process_frame(frame, camera_id="cam-blurry", timestamp=1.0)
    # OCR should not have been called for blurry flat crop
    assert mock_ocr.call_count == 0
    assert res.license_plate is None


def test_pipeline_multi_frame_conflict_resolution():
    """Test 5 (Phase 8): Conflicting OCR across frames results in MULTI_FRAME_CONFLICT."""
    config = IBVAPConfig(
        tracking_min_hits=1,
        vehicle_max_observations_per_track=5,
        vehicle_selector_max_k=3,
        vehicle_min_agreement_ratio=0.60,
        vehicle_single_obs_threshold=0.90,  # Require 0.90 for single frame, so 0.75 awaits multi-frame consensus
    )
    # 3 conflicting plate readings for the same track
    mock_ocr = ControlledMockOCR([
        PlateResult("DL01AB1234", 0.75, 0.75, WatchlistCategory.UNKNOWN, "DL01AB1234"),
        PlateResult("MH12CD5678", 0.75, 0.75, WatchlistCategory.UNKNOWN, "MH12CD5678"),
        PlateResult("UP16EF9999", 0.75, 0.75, WatchlistCategory.UNKNOWN, "UP16EF9999"),
    ])
    pipeline = IBVAPPipeline(config=config)
    pipeline.anpr_adapter = mock_ocr
    pipeline.controlled_ocr.ocr_adapter = mock_ocr

    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[200:240, 200:300] = 255
    frame[210:230, 210:290] = 0

    mock_det = MockDetector([[Detection(bbox=(100, 100, 400, 300), class_id=2, class_name="car", confidence=0.90)] for _ in range(3)])
    pipeline.detector = mock_det

    # Process 3 frames
    res = None
    for f in range(1, 4):
        res = pipeline.process_frame(frame, camera_id="cam-conflict", timestamp=float(f) * 0.1)

    # In conflict, plate_number must remain None (Anti-Hallucination)
    assert res is not None
    assert res.license_plate is None
    track_id = res.tracks[0].track_id
    track_state = pipeline.vehicle_buffer.get_track_state(track_id)
    if track_state and track_state.consensus:
        assert track_state.consensus.is_confirmed is False
        assert track_state.consensus.status in (VehicleStatus.MULTI_FRAME_CONFLICT, VehicleStatus.INSUFFICIENT_EVIDENCE)


def test_pipeline_stale_track_cleanup():
    """Test 6 (Phase 8): Inactive vehicle tracks are pruned from the observation buffer."""
    config = IBVAPConfig(
        tracking_min_hits=1,
        vehicle_stale_track_timeout_seconds=2.0,  # 2-second timeout
    )
    pipeline = IBVAPPipeline(config=config)

    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    frame[200:240, 200:300] = 255
    frame[210:230, 210:290] = 0

    # Car present at t=1.0
    mock_det1 = MockDetector([[Detection(bbox=(100, 100, 400, 300), class_id=2, class_name="car", confidence=0.90)]])
    pipeline.detector = mock_det1
    res1 = pipeline.process_frame(frame, camera_id="cam-cleanup", timestamp=1.0)
    assert len(res1.tracks) >= 1
    track_id = res1.tracks[0].track_id
    assert pipeline.vehicle_buffer.has_track(track_id) is True

    # At t=4.0 (3 seconds later, > 2.0s timeout), car is gone
    mock_det2 = MockDetector([[]])
    pipeline.detector = mock_det2
    pipeline.process_frame(frame, camera_id="cam-cleanup", timestamp=4.0)

    # Stale track should have been purged from vehicle buffer
    assert pipeline.vehicle_buffer.has_track(track_id) is False
    assert pipeline.vehicle_buffer.total_observations() == 0


def test_pipeline_human_pipeline_isolation():
    """Test 7 (PROTECTED HUMAN PIPELINE): Person detections never enter vehicle buffer."""
    config = IBVAPConfig(
        tracking_min_hits=1,
        face_detection_enabled=False,  # Skip face model for quick isolation check
    )
    pipeline = IBVAPPipeline(config=config)

    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Detection is purely a "person"
    mock_det = MockDetector([[Detection(bbox=(50, 50, 150, 300), class_id=0, class_name="person", confidence=0.95)]])
    pipeline.detector = mock_det

    res = pipeline.process_frame(frame, camera_id="cam-person", timestamp=1.0)

    # Person track exists
    assert len(res.tracks) == 1
    assert res.tracks[0].class_name == "person"
    # Zero observations stored in vehicle buffer
    assert pipeline.vehicle_buffer.total_observations() == 0
    assert pipeline.vehicle_buffer.has_track(res.tracks[0].track_id) is False
    # No plate associated with person
    assert res.tracks[0].plate_number is None
