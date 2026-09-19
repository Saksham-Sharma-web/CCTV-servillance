"""
Unit Tests for Vehicle ANPR Data Contracts (Phase 1).
Verifies:
1. Construction of VehicleStatus, PlateQualityReport, VehicleObservation, ConsensusResult, VehicleTrackState.
2. Required and default fields.
3. Optional fields and handling of None.
4. Enum transitions and transparent failure/insufficient-evidence states.
5. JSON-safe dictionary serialization without raw numpy pixel array leaking.
6. Memory safety: VehicleObservation does not require or store full video frames.
7. Strict non-coupling to human/face biometric pipeline.
"""

import json
import numpy as np
import pytest

from ibvap.core.types import WatchlistCategory
from ibvap.vehicle.types import (
    VehicleStatus,
    PlateQualityReport,
    VehicleObservation,
    ConsensusResult,
    VehicleTrackState,
)


def test_vehicle_status_enum_values_and_semantics():
    """Verifies all defined failure, tentative, and confirmed states."""
    expected_states = {
        "NO_VEHICLE",
        "VEHICLE_TENTATIVE",
        "VEHICLE_TRACKED",
        "PLATE_NOT_LOCATED",
        "PLATE_TOO_SMALL",
        "PLATE_QUALITY_INSUFFICIENT",
        "OBSERVATION_COLLECTED",
        "OCR_RECOGNITION",
        "OCR_CONFIDENCE_LOW",
        "MULTI_FRAME_CONFLICT",
        "PLATE_CONFIRMED",
        "INSUFFICIENT_EVIDENCE",
        "TRACK_EXPIRED",
    }
    actual_states = {s.value for s in VehicleStatus}
    assert actual_states == expected_states

    # Statuses must be string comparable
    assert VehicleStatus.PLATE_CONFIRMED == "PLATE_CONFIRMED"
    assert VehicleStatus.INSUFFICIENT_EVIDENCE == "INSUFFICIENT_EVIDENCE"


def test_plate_quality_report_construction_and_serialization():
    """Verifies PlateQualityReport data contract and serialization."""
    report = PlateQualityReport(
        overall_score=82.5,
        sharpness_score=85.0,
        resolution_score=90.0,
        aspect_ratio_score=88.0,
        contrast_score=75.0,
        luminance_score=80.0,
        is_acceptable=True,
        details={"laplacian_var": 142.3}
    )

    assert report.overall_score == 82.5
    assert report.is_acceptable is True
    assert report.details["laplacian_var"] == 142.3

    d = report.to_dict()
    assert d["overall_score"] == 82.5
    assert d["is_acceptable"] is True
    assert d["details"]["laplacian_var"] == 142.3

    # Must be JSON-serializable
    serialized = json.dumps(d)
    assert "82.5" in serialized


def test_vehicle_observation_memory_safety_and_serialization():
    """
    Verifies VehicleObservation handles bounded crops and metadata without
    requiring full source frames or leaking raw numpy pixel arrays into JSON dicts.
    """
    # Small synthetic crop representing a 120x36 license plate
    synthetic_plate_crop = np.zeros((36, 120, 3), dtype=np.uint8)

    quality = PlateQualityReport(
        overall_score=78.0,
        sharpness_score=80.0,
        is_acceptable=True
    )

    obs = VehicleObservation(
        track_id=184,
        frame_index=105,
        timestamp=100.25,
        plate_bbox=(40, 150, 160, 186),
        global_plate_bbox=(440, 450, 560, 486),
        plate_crop=synthetic_plate_crop,
        detection_confidence=0.92,
        quality=quality,
        ocr_text="DL01AB1234",
        ocr_confidence=0.95
    )

    assert obs.track_id == 184
    assert obs.frame_index == 105
    assert obs.crop_shape == (36, 120, 3)

    # Serialization check: crop array MUST NOT be serialized into dict
    d = obs.to_dict()
    assert d["track_id"] == 184
    assert d["frame_index"] == 105
    assert d["crop_dimensions"] == [120, 36]  # [width, height]
    assert "plate_crop" not in d

    # Verify JSON serialization succeeds cleanly
    json_str = json.dumps(d)
    assert '"track_id": 184' in json_str
    assert '"crop_dimensions": [120, 36]' in json_str


def test_vehicle_observation_without_image_crop():
    """Verifies VehicleObservation can exist purely as metadata without any image array."""
    obs = VehicleObservation(
        track_id=99,
        frame_index=10,
        timestamp=12.0,
        plate_bbox=(10, 20, 50, 40),
        plate_crop=None
    )

    assert obs.plate_crop is None
    assert obs.crop_shape is None

    d = obs.to_dict()
    assert d["crop_dimensions"] is None
    json_str = json.dumps(d)
    assert '"crop_dimensions": null' in json_str


def test_consensus_result_insufficient_evidence_state():
    """
    Anti-Hallucination verification:
    Ensures ConsensusResult explicitly supports unconfirmed/insufficient-evidence states
    without fabricating a plate number.
    """
    consensus = ConsensusResult(
        plate_number=None,
        confidence=0.0,
        observation_count=2,
        agreement_ratio=0.0,
        candidate_strings=[],
        status=VehicleStatus.INSUFFICIENT_EVIDENCE,
        is_confirmed=False,
        category=WatchlistCategory.UNKNOWN
    )

    assert consensus.plate_number is None
    assert consensus.is_confirmed is False
    assert consensus.status == VehicleStatus.INSUFFICIENT_EVIDENCE

    d = consensus.to_dict()
    assert d["plate_number"] is None
    assert d["is_confirmed"] is False
    assert d["status"] == "INSUFFICIENT_EVIDENCE"
    assert json.dumps(d) is not None


def test_consensus_result_confirmed_state():
    """Verifies confirmed consensus state with high agreement ratio."""
    consensus = ConsensusResult(
        plate_number="DL01AB1234",
        confidence=0.945,
        observation_count=3,
        agreement_ratio=1.0,
        candidate_strings=["DL01AB1234", "DL01AB1234", "DL01AB1234"],
        status=VehicleStatus.PLATE_CONFIRMED,
        is_confirmed=True,
        category=WatchlistCategory.WHITELIST,
        metadata={"format_valid": True}
    )

    assert consensus.plate_number == "DL01AB1234"
    assert consensus.is_confirmed is True
    assert consensus.agreement_ratio == 1.0
    assert consensus.category == WatchlistCategory.WHITELIST

    d = consensus.to_dict()
    assert d["plate_number"] == "DL01AB1234"
    assert d["status"] == "PLATE_CONFIRMED"
    assert d["category"] == "WHITELIST"
    assert d["metadata"]["format_valid"] is True


def test_vehicle_track_state_lifecycle_and_properties():
    """Verifies VehicleTrackState accumulation, properties, and serialization."""
    track_state = VehicleTrackState(
        track_id=184,
        camera_id="CAM_01",
        vehicle_class="car",
        status=VehicleStatus.VEHICLE_TRACKED,
        first_seen=100.0,
        last_seen=101.5,
        total_frames_tracked=12
    )

    assert track_state.track_id == 184
    assert track_state.camera_id == "CAM_01"
    assert track_state.has_confirmed_plate is False
    assert track_state.confirmed_plate_number is None
    assert len(track_state.observations) == 0

    # Add an observation
    obs = VehicleObservation(
        track_id=184,
        frame_index=5,
        timestamp=100.6,
        plate_bbox=(20, 80, 100, 110)
    )
    track_state.observations.append(obs)
    track_state.best_observation = obs
    assert len(track_state.observations) == 1

    # Attach confirmed consensus
    track_state.consensus = ConsensusResult(
        plate_number="HR26DK8392",
        confidence=0.92,
        is_confirmed=True,
        status=VehicleStatus.PLATE_CONFIRMED
    )
    track_state.status = VehicleStatus.PLATE_CONFIRMED

    assert track_state.has_confirmed_plate is True
    assert track_state.confirmed_plate_number == "HR26DK8392"

    d = track_state.to_dict()
    assert d["track_id"] == 184
    assert d["observation_count"] == 1
    assert d["confirmed_plate_number"] == "HR26DK8392"
    assert d["status"] == "PLATE_CONFIRMED"

    # Verify JSON serialization
    json_str = json.dumps(d)
    assert '"HR26DK8392"' in json_str


def test_strict_isolation_from_human_pipeline():
    """
    CRITICAL SAFETY TEST:
    Verifies VehicleTrackState and VehicleObservation have ZERO human/biometric fields.
    Guarantees no human identity fields, face similarity, or facial embeddings can be
    inadvertently attached to a vehicle track.
    """
    forbidden_human_fields = {
        "identity_id",
        "identity_name",
        "identity_confidence",
        "face_similarity",
        "face_embedding",
        "landmarks",
        "body_similarity",
        "body_role",
    }

    vehicle_state = VehicleTrackState(track_id=1)
    obs = VehicleObservation(track_id=1, frame_index=1, timestamp=1.0, plate_bbox=(0, 0, 10, 10))
    consensus = ConsensusResult()

    for field_name in forbidden_human_fields:
        assert not hasattr(vehicle_state, field_name), f"VehicleTrackState must not have human field '{field_name}'"
        assert not hasattr(obs, field_name), f"VehicleObservation must not have human field '{field_name}'"
        assert not hasattr(consensus, field_name), f"ConsensusResult must not have human field '{field_name}'"
