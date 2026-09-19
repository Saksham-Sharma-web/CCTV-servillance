"""
Unit Tests for VehicleTrackBuffer (Phase 3).
Verifies:
1. Empty buffer behavior.
2. Adding a single observation.
3. Retrieving observations.
4. Multiple tracks remain completely isolated.
5. Hard per-track maximum observation limit is strictly enforced.
6. Lowest-quality observation is evicted when capacity is reached and a better candidate arrives.
7. Inferior observation is rejected when buffer is full.
8. Best observation retrieval accurately returns the highest-scoring candidate.
9. Removing a track completely purges its state from memory.
10. Cleanup removes stale tracks beyond the timeout.
11. Cleanup preserves active tracks within the timeout window.
12. None / invalid observation inputs fail safely without raising exceptions.
13. Memory safety: buffer never stores full video frames.
14. Stress memory test: adding 100 observations to a track respects the limit (<= max_observations).
15. Same-frame observation duplicate handling.
"""

import numpy as np
import pytest

from ibvap.vehicle.types import (
    VehicleObservation,
    VehicleTrackState,
    VehicleStatus,
    PlateQualityReport,
)
from ibvap.vehicle.buffer import VehicleTrackBuffer


def make_test_observation(
    track_id: int = 101,
    frame_index: int = 1,
    timestamp: float = 10.0,
    quality_score: float = 50.0,
    bbox: tuple = (10, 20, 130, 56),
    crop: bool = True,
) -> VehicleObservation:
    """Helper to construct synthetic observations with configurable quality."""
    plate_crop = np.zeros((36, 120, 3), dtype=np.uint8) if crop else None
    quality = PlateQualityReport(
        overall_score=quality_score,
        sharpness_score=quality_score,
        is_acceptable=(quality_score >= 45.0),
    )
    return VehicleObservation(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=timestamp,
        plate_bbox=bbox,
        plate_crop=plate_crop,
        quality=quality,
    )


def test_empty_buffer_behavior():
    """Test 1: Empty buffer responds safely to queries."""
    buffer = VehicleTrackBuffer(max_observations_per_track=5)

    assert buffer.get_observations(999) == []
    assert buffer.get_best_observation(999) is None
    assert buffer.get_track_state(999) is None
    assert buffer.has_track(999) is False
    assert buffer.active_track_ids() == []
    assert buffer.observation_count(999) == 0
    assert buffer.total_observations() == 0


def test_add_single_observation():
    """Test 2 & 3: Adding one observation registers track and updates state."""
    buffer = VehicleTrackBuffer(max_observations_per_track=5)
    obs = make_test_observation(track_id=101, frame_index=1, timestamp=10.0, quality_score=75.0)

    inserted = buffer.add_observation(obs, camera_id="cam-01", vehicle_class="car")

    assert inserted is True
    assert buffer.has_track(101) is True
    assert buffer.active_track_ids() == [101]
    assert buffer.observation_count(101) == 1
    assert buffer.total_observations() == 1

    observations = buffer.get_observations(101)
    assert len(observations) == 1
    assert observations[0].track_id == 101
    assert observations[0].frame_index == 1

    state = buffer.get_track_state(101)
    assert state is not None
    assert state.track_id == 101
    assert state.camera_id == "cam-01"
    assert state.vehicle_class == "car"
    assert state.status == VehicleStatus.VEHICLE_TRACKED


def test_multiple_tracks_remain_isolated():
    """Test 4: Tracks with distinct IDs do not cross-pollinate observations."""
    buffer = VehicleTrackBuffer(max_observations_per_track=5)

    obs1 = make_test_observation(track_id=101, frame_index=1, quality_score=60.0)
    obs2 = make_test_observation(track_id=102, frame_index=1, quality_score=80.0)
    obs3 = make_test_observation(track_id=103, frame_index=1, quality_score=90.0)

    buffer.add_observation(obs1)
    buffer.add_observation(obs2)
    buffer.add_observation(obs3)

    assert set(buffer.active_track_ids()) == {101, 102, 103}
    assert buffer.observation_count(101) == 1
    assert buffer.observation_count(102) == 1
    assert buffer.observation_count(103) == 1
    assert buffer.total_observations() == 3

    best101 = buffer.get_best_observation(101)
    best102 = buffer.get_best_observation(102)
    assert best101.quality.overall_score == 60.0
    assert best102.quality.overall_score == 80.0


def test_max_observations_enforced():
    """Test 5: Buffer caps observations strictly at max_observations_per_track."""
    buffer = VehicleTrackBuffer(max_observations_per_track=3)

    # Insert 5 observations with ascending quality
    for i in range(1, 6):
        obs = make_test_observation(
            track_id=101,
            frame_index=i,
            timestamp=10.0 + i,
            quality_score=50.0 + i * 5.0,  # 55, 60, 65, 70, 75
        )
        buffer.add_observation(obs)

    # Total must not exceed 3
    assert buffer.observation_count(101) == 3

    # Retained observations should be the top 3 highest quality: 65, 70, 75
    scores = sorted([o.quality.overall_score for o in buffer.get_observations(101)])
    assert scores == [65.0, 70.0, 75.0]


def test_lowest_quality_evicted_by_better_candidate():
    """Test 6 & 7: Inferior candidate evicted when full; low-scoring candidate rejected."""
    buffer = VehicleTrackBuffer(max_observations_per_track=3)

    # Fill buffer with quality 60, 70, 80
    buffer.add_observation(make_test_observation(track_id=101, frame_index=1, quality_score=60.0))
    buffer.add_observation(make_test_observation(track_id=101, frame_index=2, quality_score=70.0))
    buffer.add_observation(make_test_observation(track_id=101, frame_index=3, quality_score=80.0))
    assert buffer.observation_count(101) == 3

    # Attempt to add a candidate with quality 40 (worse than all existing)
    worse_obs = make_test_observation(track_id=101, frame_index=4, quality_score=40.0)
    accepted = buffer.add_observation(worse_obs)
    assert accepted is False
    assert buffer.observation_count(101) == 3
    scores = sorted([o.quality.overall_score for o in buffer.get_observations(101)])
    assert scores == [60.0, 70.0, 80.0]

    # Add a candidate with quality 95 (better than the lowest 60.0)
    better_obs = make_test_observation(track_id=101, frame_index=5, quality_score=95.0)
    accepted = buffer.add_observation(better_obs)
    assert accepted is True
    assert buffer.observation_count(101) == 3

    # 60.0 should have been replaced by 95.0
    scores_after = sorted([o.quality.overall_score for o in buffer.get_observations(101)])
    assert scores_after == [70.0, 80.0, 95.0]


def test_get_best_observation_returns_highest_quality():
    """Test 8: Best observation retrieval consistently returns highest quality."""
    buffer = VehicleTrackBuffer(max_observations_per_track=5)

    buffer.add_observation(make_test_observation(track_id=101, frame_index=1, quality_score=55.0))
    buffer.add_observation(make_test_observation(track_id=101, frame_index=2, quality_score=88.5))
    buffer.add_observation(make_test_observation(track_id=101, frame_index=3, quality_score=72.0))

    best = buffer.get_best_observation(101)
    assert best is not None
    assert best.frame_index == 2
    assert best.quality.overall_score == 88.5


def test_remove_track():
    """Test 9: Removing a track completely purges its records from memory."""
    buffer = VehicleTrackBuffer(max_observations_per_track=5)

    buffer.add_observation(make_test_observation(track_id=101, frame_index=1, quality_score=70.0))
    buffer.add_observation(make_test_observation(track_id=102, frame_index=1, quality_score=80.0))

    removed = buffer.remove_track(101)
    assert removed is not None
    assert removed.track_id == 101
    assert buffer.has_track(101) is False
    assert buffer.has_track(102) is True
    assert buffer.get_observations(101) == []
    assert buffer.get_best_observation(101) is None
    assert buffer.total_observations() == 1

    # Removing non-existent track returns None
    assert buffer.remove_track(999) is None


def test_cleanup_stale_tracks():
    """Test 10 & 11: Cleanup removes stale tracks without affecting active tracks."""
    buffer = VehicleTrackBuffer(stale_track_timeout_seconds=4.0)

    # Track 101 seen at timestamp 10.0 (stale if now >= 15.0)
    buffer.add_observation(make_test_observation(track_id=101, timestamp=10.0))
    # Track 102 seen at timestamp 13.0 (stale if now >= 17.1)
    buffer.add_observation(make_test_observation(track_id=102, timestamp=13.0))
    # Track 103 seen at timestamp 14.5 (active at now = 15.0, diff = 0.5s <= 4.0s)
    buffer.add_observation(make_test_observation(track_id=103, timestamp=14.5))

    # Run cleanup at current_time = 15.0
    removed_ids = buffer.cleanup_stale_tracks(current_time=15.0)

    # Track 101 diff=5.0s > 4.0s -> removed
    # Track 102 diff=2.0s <= 4.0s -> retained
    # Track 103 diff=0.5s <= 4.0s -> retained
    assert removed_ids == [101]
    assert buffer.has_track(101) is False
    assert buffer.has_track(102) is True
    assert buffer.has_track(103) is True


def test_invalid_observation_handling():
    """Test 12: None or non-VehicleObservation inputs fail safely."""
    buffer = VehicleTrackBuffer()

    assert buffer.add_observation(None) is False
    assert buffer.add_observation("not_an_observation") is False  # type: ignore
    assert buffer.total_observations() == 0


def test_memory_safety_never_stores_full_frame():
    """Test 13: Buffer only holds plate crops or metadata, never full source frames."""
    buffer = VehicleTrackBuffer(max_observations_per_track=5)

    # Full frame simulation: 720p HD frame
    hd_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Bounded candidate plate crop
    plate_crop = hd_frame[200:236, 400:520]  # shape (36, 120, 3)

    obs = VehicleObservation(
        track_id=101,
        frame_index=1,
        timestamp=10.0,
        plate_bbox=(400, 200, 520, 236),
        plate_crop=plate_crop,
    )

    buffer.add_observation(obs)
    retrieved = buffer.get_observations(101)[0]

    assert retrieved.plate_crop.shape == (36, 120, 3)
    assert retrieved.plate_crop.shape != (720, 1280, 3)


def test_stress_observation_limit():
    """
    Test 14 (Memory Stress Test):
    Add 100 observations to a single track with limit=5.
    Verify retained observations never exceed 5.
    """
    buffer = VehicleTrackBuffer(max_observations_per_track=5)

    for i in range(100):
        obs = make_test_observation(
            track_id=200,
            frame_index=i,
            timestamp=10.0 + (i * 0.1),
            quality_score=float(i % 100),
        )
        buffer.add_observation(obs)

    # Hard limit verification
    assert buffer.observation_count(200) == 5
    assert len(buffer.get_observations(200)) == 5

    # Verify best observation is the highest score seen (99.0)
    best = buffer.get_best_observation(200)
    assert best is not None
    assert best.quality.overall_score == 99.0


def test_same_frame_duplicate_update():
    """Test 15: An observation from the same frame updates existing slot if higher quality."""
    buffer = VehicleTrackBuffer(max_observations_per_track=5)

    obs_initial = make_test_observation(track_id=101, frame_index=10, quality_score=50.0)
    buffer.add_observation(obs_initial)
    assert buffer.observation_count(101) == 1

    # Same frame, lower quality -> rejected
    obs_lower = make_test_observation(track_id=101, frame_index=10, quality_score=40.0)
    assert buffer.add_observation(obs_lower) is False
    assert buffer.observation_count(101) == 1
    assert buffer.get_best_observation(101).quality.overall_score == 50.0

    # Same frame, higher quality -> replaces existing slot
    obs_higher = make_test_observation(track_id=101, frame_index=10, quality_score=85.0)
    assert buffer.add_observation(obs_higher) is True
    assert buffer.observation_count(101) == 1
    assert buffer.get_best_observation(101).quality.overall_score == 85.0


def test_clear_method():
    """Test 16: clear() resets all tracks and observation counts."""
    buffer = VehicleTrackBuffer()
    buffer.add_observation(make_test_observation(track_id=1, frame_index=1))
    buffer.add_observation(make_test_observation(track_id=2, frame_index=1))
    assert buffer.total_observations() == 2

    buffer.clear()
    assert buffer.total_observations() == 0
    assert buffer.active_track_ids() == []
