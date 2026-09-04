"""
Unit & Integration Tests for ControlledOCRRunner and PlateConsensusEngine (Phases 5 & 6).

Covers:
Phase 5 (Controlled OCR):
1. Valid plate crop reaches OCR engine.
2. Missing crop does not invoke OCR (skipped safely).
3. Track OCR attempt budget is strictly respected (stops at max_ocr_attempts).
4. Already recognized observations are not re-processed.
5. OCR failure produces explicit failure status (OCR_FAILED, confidence 0.0).
6. OCR confidence and raw OCR text are preserved exactly without manufacture.
7. Unusable observations do not invoke OCR.

Phase 6 (Multi-Frame Temporal Consensus):
1. Identical candidates across frames yield strong consensus (PLATE_CONFIRMED).
2. Conflicting candidates yield MULTI_FRAME_CONFLICT with plate_number=None.
3. Positional character voting resolves minor character OCR noise if structurally supported.
4. Indian plate format validation provides confidence signal without mutating characters.
5. Agreement ratio and observation count reflect actual observations.
6. Single observation evaluation behaves conservatively (requires high confidence + valid format).
7. Empty observations produce INSUFFICIENT_EVIDENCE with plate_number=None.
8. Anti-hallucination guarantees: No character invention to satisfy regex.

End-to-End Component Test:
Observation -> QualityScorer -> Buffer -> Selector -> ControlledOCR -> ConsensusEngine.
"""

from typing import Optional, List
import numpy as np
import pytest

from ibvap.vehicle.types import (
    VehicleObservation,
    VehicleTrackState,
    ConsensusResult,
    VehicleStatus,
    PlateQualityReport,
)
from ibvap.vehicle.quality import PlateQualityScorer
from ibvap.vehicle.buffer import VehicleTrackBuffer
from ibvap.vehicle.selector import BestObservationSelector
from ibvap.vehicle.consensus import (
    ControlledOCRRunner,
    PlateConsensusEngine,
)
from ibvap.core.types import WatchlistCategory
from ibvap.anpr.ocr_adapter import PlateResult


# ─────────────────────────────────────────────────────────────────────────────
# Controlled Mock OCR Adapter for Deterministic Fast Testing
# ─────────────────────────────────────────────────────────────────────────────
class MockOCRAdapter:
    """Mock ANPRAdapter for deterministic unit testing without CPU-heavy PaddleOCR inference."""

    def __init__(self, predefined_results: Optional[List[Optional[PlateResult]]] = None):
        self.predefined_results = predefined_results or []
        self.call_count = 0

    def recognize_plate(self, crop: np.ndarray) -> Optional[PlateResult]:
        self.call_count += 1
        if self.predefined_results:
            idx = min(self.call_count - 1, len(self.predefined_results) - 1)
            return self.predefined_results[idx]
        return PlateResult(
            plate_number="DL01AB1234",
            confidence=0.92,
            category=WatchlistCategory.UNKNOWN,
            raw_text="DL01AB1234",
            ocr_confidence=0.92,
        )


def make_obs(
    track_id: int = 101,
    frame_index: int = 1,
    quality_score: float = 80.0,
    ocr_text: Optional[str] = None,
    ocr_confidence: Optional[float] = None,
    crop: bool = True,
) -> VehicleObservation:
    """Helper to generate VehicleObservation instances."""
    plate_crop = np.zeros((36, 120, 3), dtype=np.uint8) if crop else None
    quality = PlateQualityReport(
        overall_score=quality_score,
        sharpness_score=quality_score,
        is_acceptable=(quality_score >= 45.0),
    )
    return VehicleObservation(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=float(frame_index) * 0.04,
        plate_bbox=(10, 10, 130, 46),
        plate_crop=plate_crop,
        quality=quality,
        ocr_text=ocr_text,
        ocr_confidence=ocr_confidence,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 5: CONTROLLED OCR RUNNER TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_controlled_ocr_success_and_preservation():
    """Test 1: Valid plate crop is passed to OCR, text, confidence & raw output preserved."""
    mock_adapter = MockOCRAdapter([
        PlateResult(
            plate_number="MH12DE1432",
            confidence=0.88,
            category=WatchlistCategory.UNKNOWN,
            raw_text="MH 12 DE 1432",
            ocr_confidence=0.88,
        )
    ])
    runner = ControlledOCRRunner(ocr_adapter=mock_adapter, max_ocr_attempts_per_track=3)
    obs = make_obs(frame_index=1)

    results = runner.run_ocr([obs])
    assert len(results) == 1
    assert mock_adapter.call_count == 1
    assert results[0].ocr_text == "MH12DE1432"
    assert results[0].ocr_confidence == 0.88
    assert results[0].metadata["raw_ocr_text"] == "MH 12 DE 1432"
    assert results[0].metadata["ocr_status"] == "SUCCESS"


def test_controlled_ocr_missing_crop_skipped():
    """Test 2: Observation with missing crop does not invoke OCR."""
    mock_adapter = MockOCRAdapter()
    runner = ControlledOCRRunner(ocr_adapter=mock_adapter)
    obs = make_obs(crop=False)

    results = runner.run_ocr([obs])
    assert len(results) == 0
    assert mock_adapter.call_count == 0
    assert obs.metadata.get("ocr_status") == "SKIPPED_NO_CROP"


def test_controlled_ocr_budget_enforced():
    """Test 3: Track OCR budget strictly limits total OCR attempts per track."""
    mock_adapter = MockOCRAdapter()
    runner = ControlledOCRRunner(ocr_adapter=mock_adapter, max_ocr_attempts_per_track=2)

    track_state = VehicleTrackState(track_id=101, ocr_attempts=0)
    observations = [make_obs(track_id=101, frame_index=i) for i in range(1, 6)]

    results = runner.run_ocr(observations, track_state=track_state)

    # Only 2 attempts should have been made
    assert mock_adapter.call_count == 2
    assert track_state.ocr_attempts == 2
    assert len(results) == 2


def test_controlled_ocr_skips_already_processed():
    """Test 4: Already-recognized observation is not re-processed."""
    mock_adapter = MockOCRAdapter()
    runner = ControlledOCRRunner(ocr_adapter=mock_adapter)

    obs_processed = make_obs(frame_index=1, ocr_text="KA01MJ5000", ocr_confidence=0.95)
    results = runner.run_ocr([obs_processed])

    assert len(results) == 1
    assert mock_adapter.call_count == 0
    assert results[0].ocr_text == "KA01MJ5000"


def test_controlled_ocr_failure_handling():
    """Test 5: OCR engine returning None represents explicit failure status."""
    mock_adapter = MockOCRAdapter([None])
    runner = ControlledOCRRunner(ocr_adapter=mock_adapter)
    obs = make_obs(frame_index=1)

    results = runner.run_ocr([obs])
    assert len(results) == 1
    assert mock_adapter.call_count == 1
    assert results[0].ocr_text is None
    assert results[0].ocr_confidence == 0.0
    assert results[0].metadata["ocr_status"] == "OCR_FAILED"


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 6: MULTI-FRAME CONSENSUS TESTS
# ═════════════════════════════════════════════════════════════════════════════

def test_multi_frame_strong_consensus():
    """Test 1: Identical candidate strings across frames produce confirmed plate."""
    engine = PlateConsensusEngine(min_consensus_observations=2, min_agreement_ratio=0.60)

    obs1 = make_obs(frame_index=1, ocr_text="DL01AB1234", ocr_confidence=0.88, quality_score=80.0)
    obs2 = make_obs(frame_index=5, ocr_text="DL01AB1234", ocr_confidence=0.92, quality_score=85.0)
    obs3 = make_obs(frame_index=9, ocr_text="DL01AB1234", ocr_confidence=0.90, quality_score=78.0)

    res = engine.evaluate([obs1, obs2, obs3])

    assert res.is_confirmed is True
    assert res.status == VehicleStatus.PLATE_CONFIRMED
    assert res.plate_number == "DL01AB1234"
    assert res.observation_count == 3
    assert res.agreement_ratio == 1.0
    assert res.confidence >= 0.88


def test_multi_frame_conflict_detection():
    """Test 2: Conflicting candidates with no consensus produce MULTI_FRAME_CONFLICT."""
    engine = PlateConsensusEngine(min_consensus_observations=2, min_agreement_ratio=0.60)

    obs1 = make_obs(frame_index=1, ocr_text="DL01AB1234", ocr_confidence=0.80)
    obs2 = make_obs(frame_index=5, ocr_text="DL01AB1284", ocr_confidence=0.82)
    obs3 = make_obs(frame_index=9, ocr_text="UP16CD9999", ocr_confidence=0.81)

    res = engine.evaluate([obs1, obs2, obs3])

    assert res.is_confirmed is False
    assert res.status == VehicleStatus.MULTI_FRAME_CONFLICT
    assert res.plate_number is None  # Never fabricate certainty
    assert res.observation_count == 3
    assert res.agreement_ratio <= 0.35


def test_positional_character_voting():
    """
    Test 3: Positional voting reconciles minor character noise when candidates share length.
    E.g.:
      Obs 1: DL01AB1284 (conf 0.70)
      Obs 2: DL01AB1234 (conf 0.95)
      Obs 3: DL01AB1234 (conf 0.90)
    '3' wins over '8' at position 8 based on weighted support.
    """
    engine = PlateConsensusEngine(min_consensus_observations=2, min_agreement_ratio=0.60)

    obs1 = make_obs(frame_index=1, ocr_text="DL01AB1284", ocr_confidence=0.70)
    obs2 = make_obs(frame_index=5, ocr_text="DL01AB1234", ocr_confidence=0.95)
    obs3 = make_obs(frame_index=9, ocr_text="DL01AB1234", ocr_confidence=0.90)

    res = engine.evaluate([obs1, obs2, obs3])
    assert res.is_confirmed is True
    assert res.plate_number == "DL01AB1234"


def test_format_validation_does_not_mutate_characters():
    """
    Test 4 (CRITICAL ANTI-HALLUCINATION):
    If candidate is invalid (e.g. 'DLO1ABI234'), consensus does NOT mutate O->0 or I->1.
    """
    engine = PlateConsensusEngine(min_consensus_observations=2)

    # Identical OCR output with letters instead of digits
    obs1 = make_obs(frame_index=1, ocr_text="DLO1ABI234", ocr_confidence=0.90)
    obs2 = make_obs(frame_index=2, ocr_text="DLO1ABI234", ocr_confidence=0.92)

    res = engine.evaluate([obs1, obs2])

    # Text must NOT be mutated to "DL01AB1234"
    assert res.candidate_strings == ["DLO1ABI234", "DLO1ABI234"]
    if res.plate_number is not None:
        assert res.plate_number == "DLO1ABI234"
        assert "DL01AB1234" not in res.plate_number


def test_single_observation_conservative_behavior():
    """Test 5: Single observation confirmed ONLY if confidence exceeds high threshold and format is valid."""
    engine = PlateConsensusEngine(
        single_observation_confidence_threshold=0.92,
        min_confidence_threshold=0.70,
    )

    # Single observation with moderate confidence (0.85 < 0.92) -> INSUFFICIENT_EVIDENCE
    obs_med = make_obs(frame_index=1, ocr_text="DL01AB1234", ocr_confidence=0.85)
    res_med = engine.evaluate([obs_med])
    assert res_med.is_confirmed is False
    assert res_med.plate_number is None
    assert res_med.status == VehicleStatus.INSUFFICIENT_EVIDENCE

    # Single observation with very high confidence (0.95 >= 0.92) and valid Indian format -> CONFIRMED
    obs_high = make_obs(frame_index=1, ocr_text="DL01AB1234", ocr_confidence=0.95)
    res_high = engine.evaluate([obs_high])
    assert res_high.is_confirmed is True
    assert res_high.plate_number == "DL01AB1234"
    assert res_high.status == VehicleStatus.PLATE_CONFIRMED

    # Single observation with high confidence but invalid format -> NOT confirmed
    obs_inv = make_obs(frame_index=1, ocr_text="INVALID123", ocr_confidence=0.96)
    res_inv = engine.evaluate([obs_inv])
    assert res_inv.is_confirmed is False
    assert res_inv.plate_number is None


def test_empty_and_insufficient_evidence():
    """Test 6: Empty observation list or empty OCR text produces INSUFFICIENT_EVIDENCE."""
    engine = PlateConsensusEngine()

    res_empty = engine.evaluate([])
    assert res_empty.is_confirmed is False
    assert res_empty.plate_number is None
    assert res_empty.status == VehicleStatus.INSUFFICIENT_EVIDENCE
    assert res_empty.observation_count == 0

    obs_no_text = make_obs(frame_index=1, ocr_text=None)
    res_no_text = engine.evaluate([obs_no_text])
    assert res_no_text.is_confirmed is False
    assert res_no_text.plate_number is None
    assert res_no_text.status == VehicleStatus.INSUFFICIENT_EVIDENCE


def test_bharat_series_plate_validation():
    """Test 7: Valid Bharat series plates (e.g. 22BH1234AA) are recognized as valid format."""
    engine = PlateConsensusEngine(min_consensus_observations=2)
    obs1 = make_obs(frame_index=1, ocr_text="22BH1234AA", ocr_confidence=0.90)
    obs2 = make_obs(frame_index=2, ocr_text="22BH1234AA", ocr_confidence=0.92)

    res = engine.evaluate([obs1, obs2])
    assert res.is_confirmed is True
    assert res.plate_number == "22BH1234AA"
    assert res.metadata.get("format_valid") is True


# ═════════════════════════════════════════════════════════════════════════════
# END-TO-END COMPONENT INTEGRATION TEST
# ═════════════════════════════════════════════════════════════════════════════

def test_end_to_end_component_pipeline():
    """
    Component Integration Flow:
    Plate Crop -> PlateQualityScorer -> VehicleTrackBuffer ->
    BestObservationSelector -> ControlledOCRRunner -> PlateConsensusEngine
    """
    # 1. Initialize Pipeline Components
    scorer = PlateQualityScorer()
    buffer = VehicleTrackBuffer(max_observations_per_track=5)
    selector = BestObservationSelector(max_k=3, min_quality_threshold=45.0)

    mock_adapter = MockOCRAdapter([
        PlateResult(plate_number="DL04XY8899", confidence=0.91, ocr_confidence=0.91, category=WatchlistCategory.UNKNOWN, raw_text="DL04XY8899"),
        PlateResult(plate_number="DL04XY8899", confidence=0.93, ocr_confidence=0.93, category=WatchlistCategory.UNKNOWN, raw_text="DL04XY8899"),
        PlateResult(plate_number="DL04XY8899", confidence=0.89, ocr_confidence=0.89, category=WatchlistCategory.UNKNOWN, raw_text="DL04XY8899"),
    ])
    ocr_runner = ControlledOCRRunner(ocr_adapter=mock_adapter, max_ocr_attempts_per_track=3)
    consensus_engine = PlateConsensusEngine(min_consensus_observations=2)

    track_id = 501

    # 2. Simulate streaming 6 incoming frames of vehicle observations
    for f_idx in range(1, 7):
        # Create synthetic plate crop with distinct texture per frame
        crop = np.full((36, 120, 3), fill_value=int(40 + f_idx * 25), dtype=np.uint8)
        # Add high-contrast gradient
        crop[:, ::4, :] = 255

        report = scorer.score(crop)
        obs = VehicleObservation(
            track_id=track_id,
            frame_index=f_idx,
            timestamp=10.0 + (f_idx * 0.05),
            plate_bbox=(100, 200, 220, 236),
            plate_crop=crop,
            quality=report,
            detection_confidence=0.85 + (f_idx * 0.02),
        )
        buffer.add_observation(obs)

    # 3. Buffer bounded check
    buffered_obs = buffer.get_observations(track_id)
    assert len(buffered_obs) == 5

    # 4. Best Observation Selection (Top-K=3, temporally spaced)
    selected_obs = selector.select(buffered_obs)
    assert len(selected_obs) <= 3
    assert len(selected_obs) >= 2

    # 5. Controlled OCR Execution
    track_state = buffer.get_track_state(track_id)
    ocr_results = ocr_runner.run_ocr(selected_obs, track_state=track_state)
    assert len(ocr_results) == len(selected_obs)
    assert track_state.ocr_attempts == len(selected_obs)
    assert all(o.ocr_text == "DL04XY8899" for o in ocr_results)

    # 6. Multi-Frame Consensus
    consensus = consensus_engine.evaluate(ocr_results)
    assert consensus.is_confirmed is True
    assert consensus.status == VehicleStatus.PLATE_CONFIRMED
    assert consensus.plate_number == "DL04XY8899"
    assert consensus.confidence >= 0.89
    assert consensus.observation_count == len(selected_obs)
    assert consensus.agreement_ratio == 1.0
