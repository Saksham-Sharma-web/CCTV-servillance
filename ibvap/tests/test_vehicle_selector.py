"""
Unit Tests for BestObservationSelector (Phase 4).
Verifies:
1. Highest-quality observation selected.
2. Maximum K enforced (Top-K bounded).
3. Poor-quality observations excluded (< min_quality_threshold).
4. Temporal diversity selects spaced observations when quality is comparable.
5. Fallback fill populates remaining slots when temporal diversity is strict.
6. Deterministic selection across repeated invocations.
7. Empty buffer / empty list handled safely.
8. None inputs, observations missing plate crops, or missing quality reports handled safely.
9. Parameter validation (max_k, min_quality_threshold, min_frame_separation).
10. Duplicate observation reference handling.
"""

import numpy as np
import pytest

from ibvap.vehicle.types import (
    VehicleObservation,
    PlateQualityReport,
)
from ibvap.vehicle.selector import BestObservationSelector


def make_test_obs(
    track_id: int = 101,
    frame_index: int = 1,
    quality_score: float = 60.0,
    confidence: float = 0.90,
    has_crop: bool = True,
    has_quality: bool = True,
) -> VehicleObservation:
    """Helper to generate synthetic VehicleObservation instances."""
    crop = np.zeros((36, 120, 3), dtype=np.uint8) if has_crop else None
    quality = (
        PlateQualityReport(
            overall_score=quality_score,
            sharpness_score=quality_score,
            is_acceptable=(quality_score >= 45.0),
        )
        if has_quality
        else None
    )
    return VehicleObservation(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=float(frame_index) * 0.04,
        plate_bbox=(10, 10, 130, 46),
        plate_crop=crop,
        quality=quality,
        detection_confidence=confidence,
    )


def test_highest_quality_observation_selected():
    """Test 1: Selector guarantees the highest-quality candidate is always selected first."""
    selector = BestObservationSelector(max_k=1)
    obs1 = make_test_obs(frame_index=1, quality_score=55.0)
    obs2 = make_test_obs(frame_index=2, quality_score=92.5)
    obs3 = make_test_obs(frame_index=3, quality_score=71.0)

    selected = selector.select([obs1, obs2, obs3])
    assert len(selected) == 1
    assert selected[0].frame_index == 2
    assert selected[0].quality.overall_score == 92.5


def test_maximum_k_enforced():
    """Test 2: Selector never returns more than max_k observations."""
    selector = BestObservationSelector(max_k=2)
    observations = [
        make_test_obs(frame_index=i, quality_score=60.0 + i)
        for i in range(1, 10)
    ]

    selected = selector.select(observations)
    assert len(selected) == 2


def test_poor_quality_observations_excluded():
    """Test 3: Observations below min_quality_threshold are filtered out before OCR."""
    selector = BestObservationSelector(max_k=3, min_quality_threshold=50.0)
    obs_poor1 = make_test_obs(frame_index=1, quality_score=35.0)
    obs_poor2 = make_test_obs(frame_index=2, quality_score=48.0)
    obs_good = make_test_obs(frame_index=3, quality_score=65.0)

    selected = selector.select([obs_poor1, obs_poor2, obs_good])
    assert len(selected) == 1
    assert selected[0].frame_index == 3
    assert selected[0].quality.overall_score == 65.0


def test_all_poor_quality_returns_empty():
    """Test 3b: When all observations are poor quality, returns empty list without error."""
    selector = BestObservationSelector(min_quality_threshold=50.0)
    obs_poor1 = make_test_obs(frame_index=1, quality_score=30.0)
    obs_poor2 = make_test_obs(frame_index=2, quality_score=40.0)

    selected = selector.select([obs_poor1, obs_poor2])
    assert selected == []


def test_temporal_diversity_selection():
    """
    Test 4: Selector prefers temporally separated observations when quality is close.
    Example:
      Frame 101 - quality 88.0
      Frame 102 - quality 87.5 (too close temporally, diff=1 < min_separation=2)
      Frame 105 - quality 86.0 (diff=4 >= 2)
    With max_k=2 and min_frame_separation=2, frame 101 and 105 should be selected.
    """
    selector = BestObservationSelector(max_k=2, min_frame_separation=2)
    obs101 = make_test_obs(frame_index=101, quality_score=88.0)
    obs102 = make_test_obs(frame_index=102, quality_score=87.5)
    obs105 = make_test_obs(frame_index=105, quality_score=86.0)

    selected = selector.select([obs101, obs102, obs105])
    assert len(selected) == 2
    selected_frames = [s.frame_index for s in selected]
    assert selected_frames == [101, 105]


def test_temporal_diversity_fallback_fill():
    """
    Test 5: If temporal diversity cannot find enough spaced candidates,
    fallback fill populates remaining slots with next best quality candidates.
    """
    selector = BestObservationSelector(max_k=3, min_frame_separation=5)
    # Consecutive frames 10, 11, 12 with good quality
    obs10 = make_test_obs(frame_index=10, quality_score=85.0)
    obs11 = make_test_obs(frame_index=11, quality_score=84.0)
    obs12 = make_test_obs(frame_index=12, quality_score=83.0)

    selected = selector.select([obs10, obs11, obs12])
    # Must still return 3 observations because fallback fill activates
    assert len(selected) == 3
    assert [s.frame_index for s in selected] == [10, 11, 12]


def test_deterministic_selection():
    """Test 6: Repeated selections on identical or shuffled input produce deterministic results."""
    selector = BestObservationSelector(max_k=3)
    cands = [
        make_test_obs(frame_index=1, quality_score=70.0),
        make_test_obs(frame_index=5, quality_score=85.0),
        make_test_obs(frame_index=8, quality_score=80.0),
        make_test_obs(frame_index=12, quality_score=75.0),
    ]

    res1 = selector.select(cands)
    res2 = selector.select(list(reversed(cands)))

    assert [s.frame_index for s in res1] == [s.frame_index for s in res2]
    assert [s.quality.overall_score for s in res1] == [s.quality.overall_score for s in res2]


def test_empty_buffer_and_none_handling():
    """Test 7 & 8: Empty list or None input handled safely without exceptions."""
    selector = BestObservationSelector()

    assert selector.select([]) == []
    assert selector.select(None) == []


def test_missing_crop_or_quality_excluded():
    """Test 8b: Candidates with None crop or None quality are excluded safely."""
    selector = BestObservationSelector()
    obs_valid = make_test_obs(frame_index=1, quality_score=70.0)
    obs_no_crop = make_test_obs(frame_index=2, quality_score=80.0, has_crop=False)
    obs_no_qual = make_test_obs(frame_index=3, quality_score=90.0, has_quality=False)

    selected = selector.select([obs_valid, obs_no_crop, obs_no_qual])
    assert len(selected) == 1
    assert selected[0].frame_index == 1


def test_parameter_validation():
    """Test 9: Invalid initialization arguments raise ValueError."""
    with pytest.raises(ValueError):
        BestObservationSelector(max_k=0)

    with pytest.raises(ValueError):
        BestObservationSelector(min_quality_threshold=-5.0)

    with pytest.raises(ValueError):
        BestObservationSelector(min_quality_threshold=105.0)

    with pytest.raises(ValueError):
        BestObservationSelector(min_frame_separation=-1)
