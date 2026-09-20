"""
Test Suite for IBVAP Unknown Person Tracking & Cross-Camera Re-Identification.
Validates:
1. Face quality gate rejection
2. Evidence accumulation
3. Initial unknown person registration and immediate alert emission
4. Same-camera track caching (no duplicate alerts)
5. Cross-camera re-identification (Cam 1 -> Cam 4) with trajectory history
6. Vector Top-K search and similarity margin verification
7. SQLite persistence and vector index reconstruction
8. Threshold evaluation mode
9. Pipeline integration
"""

import os
import time
import tempfile
import numpy as np
import pytest

from ibvap.core.config import IBVAPConfig
from ibvap.core.types import (
    Detection,
    Track,
    EventType,
    AnalyticsEvent,
    UnknownPersonRecord,
    UnknownPersonSighting,
)
from ibvap.tracking.unknown_storage import SQLiteUnknownStorage
from ibvap.tracking.unknown_person import (
    UnknownPersonManager,
    UnknownMatchResult,
    evaluate_unknown_thresholds,
)
from ibvap.tracking.cross_camera import CrossCameraTracker
from ibvap.core.pipeline import IBVAPPipeline


@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_unknown_persons.db")


@pytest.fixture
def custom_config(temp_db_path):
    cfg = IBVAPConfig()
    cfg.unknown_person_tracking_enabled = True
    cfg.unknown_face_similarity_threshold = 0.65
    cfg.unknown_face_high_confidence_threshold = 0.75
    cfg.unknown_similarity_margin = 0.05
    cfg.unknown_min_quality_score = 0.50
    cfg.unknown_max_representatives_per_person = 5
    cfg.unknown_evidence_min_hits = 2
    cfg.unknown_storage_db_path = temp_db_path
    cfg.unknown_db_write_debounce_seconds = 0.1
    return cfg


def _make_unit_vector(dim: int = 512, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _make_noisy_vector(base_vec: np.ndarray, target_sim: float = 0.85, seed: int = 99) -> np.ndarray:
    rng = np.random.RandomState(seed)
    rand = rng.randn(len(base_vec)).astype(np.float32)
    rand = rand - np.dot(rand, base_vec) * base_vec
    rand = rand / np.linalg.norm(rand)
    v = target_sim * base_vec + np.sqrt(max(0.0, 1.0 - target_sim**2)) * rand
    return (v / np.linalg.norm(v)).astype(np.float32)


def _make_face_crop(width: int = 80, height: int = 80, blur: bool = False) -> np.ndarray:
    if blur:
        return np.full((height, width, 3), 128, dtype=np.uint8)
    rng = np.random.RandomState(123)
    return rng.randint(50, 200, size=(height, width, 3), dtype=np.uint8)


class TestUnknownPersonTracking:

    def test_quality_gate_rejection(self, custom_config, temp_db_path):
        mgr = UnknownPersonManager(config=custom_config)

        # 1. Empty crop
        is_good, q, reason = mgr.validate_face_quality(None)
        assert not is_good
        assert "Empty" in reason

        # 2. Too small crop (< 24x24)
        tiny_crop = np.zeros((16, 16, 3), dtype=np.uint8)
        is_good, q, reason = mgr.validate_face_quality(tiny_crop)
        assert not is_good
        assert "small" in reason

        # 3. Blurred / uniform crop
        blur_crop = _make_face_crop(80, 80, blur=True)
        is_good, q, reason = mgr.validate_face_quality(blur_crop)
        assert not is_good
        assert "blurred" in reason

        # 4. Good textured crop
        good_crop = _make_face_crop(80, 80, blur=False)
        is_good, q, reason = mgr.validate_face_quality(good_crop)
        assert is_good
        assert q >= custom_config.unknown_min_quality_score

    def test_evidence_accumulation_and_registration(self, custom_config, temp_db_path):
        mgr = UnknownPersonManager(config=custom_config)
        emb = _make_unit_vector(512, seed=10)
        crop = _make_face_crop(80, 80)
        now = time.time()

        # Observation 1: evidence min hits is 2 -> should be COLLECTING_EVIDENCE
        res1 = mgr.match_or_register(
            face_embedding=emb,
            camera_id="camera-01",
            track_id=1,
            timestamp=now,
            bbox=(100, 100, 200, 200),
            face_crop=crop,
        )
        assert res1.decision == "COLLECTING_EVIDENCE"
        assert res1.unknown_id is None
        assert res1.event_to_emit is None

        # Observation 2: evidence threshold reached -> NEW_UNKNOWN
        res2 = mgr.match_or_register(
            face_embedding=emb,
            camera_id="camera-01",
            track_id=1,
            timestamp=now + 0.1,
            bbox=(102, 102, 202, 202),
            face_crop=crop,
        )
        assert res2.decision == "NEW_UNKNOWN"
        assert res2.unknown_id is not None
        assert res2.unknown_id.startswith(("UNK-P-", "P"))
        assert res2.event_to_emit is not None
        assert res2.event_to_emit.event_type == EventType.UNKNOWN_PERSON
        assert res2.event_to_emit.identity_id == res2.unknown_id
        assert "timestamp_iso" in res2.event_to_emit.metadata
        assert res2.event_to_emit.metadata["camera_id"] == "camera-01"

        # Observation 3 on SAME camera and SAME track: should use track cache
        res3 = mgr.match_or_register(
            face_embedding=emb,
            camera_id="camera-01",
            track_id=1,
            timestamp=now + 0.2,
            bbox=(105, 105, 205, 205),
            face_crop=crop,
        )
        assert res3.decision == "EXISTING_TRACK"
        assert res3.unknown_id == res2.unknown_id
        assert res3.event_to_emit is None  # No redundant alert

    def test_cross_camera_reidentification(self, custom_config, temp_db_path):
        mgr = UnknownPersonManager(config=custom_config)
        person_a_emb = _make_unit_vector(512, seed=42)
        crop = _make_face_crop(80, 80)
        t0 = time.time()

        # 1. Register Person A on camera-01 (needs 2 hits)
        mgr.match_or_register(person_a_emb, "camera-01", 1, t0, (50, 50, 150, 150), crop)
        res_cam1 = mgr.match_or_register(person_a_emb, "camera-01", 1, t0 + 0.1, (52, 52, 152, 152), crop)
        assert res_cam1.decision == "NEW_UNKNOWN"
        uid = res_cam1.unknown_id

        # 2. Person A appears on camera-04 with slight noise (e.g. cosine sim ~0.85)
        noisy_emb = _make_noisy_vector(person_a_emb, target_sim=0.85, seed=101)
        sim = float(np.dot(person_a_emb, noisy_emb))
        assert sim >= custom_config.unknown_face_similarity_threshold

        t_cam4 = t0 + 15.0  # 15 seconds later
        res_cam4 = mgr.match_or_register(
            face_embedding=noisy_emb,
            camera_id="camera-04",
            track_id=7,  # New local track ID on camera-04
            timestamp=t_cam4,
            bbox=(200, 100, 300, 250),
            face_crop=crop,
        )

        assert res_cam4.decision == "REIDENTIFIED"
        assert res_cam4.unknown_id == uid
        assert res_cam4.is_cross_camera_transition is True
        assert res_cam4.previous_camera == "camera-01"
        assert res_cam4.current_camera == "camera-04"
        assert res_cam4.time_gap_seconds == pytest.approx(15.0, abs=0.1)

        # Verify emitted event
        ev = res_cam4.event_to_emit
        assert ev is not None
        assert ev.event_type == EventType.PERSON_REIDENTIFIED
        assert ev.identity_id == uid
        assert ev.camera_id == "camera-04"
        assert ev.metadata["previous_camera"] == "camera-01"
        assert ev.metadata["current_camera"] == "camera-04"
        assert ev.metadata["time_gap_seconds"] == pytest.approx(15.0, abs=0.1)
        assert ev.metadata["camera_sequence"] == ["camera-01", "camera-04"]
        assert "timestamp_iso" in ev.metadata

        # Verify trajectory in storage
        traj = mgr.get_trajectory(uid)
        assert len(traj) == 2
        assert traj[0]["camera_id"] == "camera-01"
        assert traj[1]["camera_id"] == "camera-04"

    def test_top_k_vector_search_and_margin(self, custom_config, temp_db_path):
        mgr = UnknownPersonManager(config=custom_config)
        crop = _make_face_crop(80, 80)
        t = time.time()

        # Register Person 1
        emb1 = _make_unit_vector(512, seed=1)
        mgr.match_or_register(emb1, "camera-01", 1, t, (10, 10, 50, 50), crop)
        r1 = mgr.match_or_register(emb1, "camera-01", 1, t + 0.1, (10, 10, 50, 50), crop)

        # Register Person 2 (completely different)
        emb2 = _make_unit_vector(512, seed=2)
        mgr.match_or_register(emb2, "camera-01", 2, t, (60, 60, 100, 100), crop)
        r2 = mgr.match_or_register(emb2, "camera-01", 2, t + 0.1, (60, 60, 100, 100), crop)

        # Query with an embedding midway between emb1 and emb2 (ambiguous)
        mid_emb = 0.5 * emb1 + 0.5 * emb2
        mid_emb = mid_emb / np.linalg.norm(mid_emb)

        sim1 = float(np.dot(emb1, mid_emb))
        sim2 = float(np.dot(emb2, mid_emb))
        margin = abs(sim1 - sim2)
        assert margin < custom_config.unknown_similarity_margin

        res_ambiguous = mgr.match_or_register(
            face_embedding=mid_emb,
            camera_id="camera-02",
            track_id=5,
            timestamp=t + 1.0,
            bbox=(20, 20, 80, 80),
            face_crop=crop,
        )
        # Because margin is below threshold (0.05), it should be flagged UNCERTAIN
        assert res_ambiguous.confidence_level == "UNCERTAIN"

    def test_sqlite_persistence_and_cache_reload(self, custom_config, temp_db_path):
        crop = _make_face_crop(80, 80)
        t = time.time()
        emb = _make_unit_vector(512, seed=55)

        # Create manager, register unknown person
        mgr1 = UnknownPersonManager(config=custom_config)
        mgr1.match_or_register(emb, "camera-01", 1, t, (10, 10, 50, 50), crop)
        res = mgr1.match_or_register(emb, "camera-01", 1, t + 0.1, (10, 10, 50, 50), crop)
        uid = res.unknown_id

        # Instantiate brand new manager pointing to same SQLite database
        mgr2 = UnknownPersonManager(config=custom_config)
        assert len(mgr2.list_unknown_persons()) == 1

        rec = mgr2.get_unknown_person(uid)
        assert rec is not None
        assert rec.unknown_id == uid
        assert rec.first_camera_id == "camera-01"

        # Search against new manager's vector index
        candidates = mgr2.storage.search_top_k(emb, k=1)
        assert len(candidates) == 1
        assert candidates[0][0] == uid
        assert candidates[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_threshold_evaluation_mode(self):
        # 3 distinct individuals with 3 embeddings each
        emb_by_person = {
            "p1": [
                _make_unit_vector(512, seed=1),
                _make_noisy_vector(_make_unit_vector(512, seed=1), target_sim=0.90, seed=11),
                _make_noisy_vector(_make_unit_vector(512, seed=1), target_sim=0.80, seed=12),
            ],
            "p2": [
                _make_unit_vector(512, seed=2),
                _make_noisy_vector(_make_unit_vector(512, seed=2), target_sim=0.90, seed=21),
                _make_noisy_vector(_make_unit_vector(512, seed=2), target_sim=0.80, seed=22),
            ],
            "p3": [
                _make_unit_vector(512, seed=3),
                _make_noisy_vector(_make_unit_vector(512, seed=3), target_sim=0.90, seed=31),
                _make_noisy_vector(_make_unit_vector(512, seed=3), target_sim=0.80, seed=32),
            ],
        }

        eval_res = evaluate_unknown_thresholds(emb_by_person, thresholds=[0.60, 0.65, 0.70, 0.75, 0.80])
        assert len(eval_res) == 5
        for th in [0.60, 0.65, 0.70, 0.75, 0.80]:
            assert th in eval_res
            assert "precision" in eval_res[th]
            assert "recall" in eval_res[th]
            assert "f1_score" in eval_res[th]

    def test_cross_camera_tracker_integration(self, custom_config):
        tracker = CrossCameraTracker(config=custom_config)
        now = time.time()

        # Track on camera 1 with unknown identity UNK-P-001
        t1 = Track(
            track_id=1,
            bbox=(10, 10, 50, 50),
            class_name="person",
            confidence=0.9,
            center=(30, 30),
            identity_id="UNK-P-001",
            identity_name="Unknown (UNK-P-001)",
        )
        res1 = tracker.associate_tracks("camera-01", [t1], timestamp=now)
        gid1 = res1[0].global_track_id
        assert gid1 is not None

        # Track on camera 4 with same unknown identity UNK-P-001
        t2 = Track(
            track_id=9,
            bbox=(100, 100, 150, 150),
            class_name="person",
            confidence=0.9,
            center=(125, 125),
            identity_id="UNK-P-001",
            identity_name="Unknown (UNK-P-001)",
        )
        res2 = tracker.associate_tracks("camera-04", [t2], timestamp=now + 5.0)
        gid2 = res2[0].global_track_id

        # Must map to the exact same global entity!
        assert gid2 == gid1
        seq = tracker.get_camera_sequence(gid1)
        assert seq == ["camera-01", "camera-04"]

    def test_pipeline_unknown_tracking_flow(self, custom_config):
        from unittest.mock import MagicMock
        from ibvap.detection.base import BaseObjectDetector
        from ibvap.face.matcher_adapter import VerificationResult
        from ibvap.face.detector import FaceDetection

        class MockDetector(BaseObjectDetector):
            def detect(self, frame):
                return [Detection(bbox=(50, 50, 200, 300), class_id=0, class_name="person", confidence=0.9)]

        pipeline = IBVAPPipeline(config=custom_config, detector=MockDetector())

        fake_face = FaceDetection(box=(10, 10, 80, 80), confidence=0.95, landmarks=[], quality_status="GOOD_FACE")
        pipeline.face_detector.detect_faces = MagicMock(return_value=[fake_face])

        person_emb = _make_unit_vector(512, seed=77)
        mock_verif = VerificationResult(
            face_decision="UNKNOWN",
            face_similarity=0.2,
            face_embedding=person_emb,
            aligned_face=_make_face_crop(80, 80),
        )
        pipeline.identity_verifier.verify = MagicMock(return_value=mock_verif)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1 on camera-01 (evidence accumulating)
        r1 = pipeline.process_frame(frame, camera_id="camera-01", timestamp=100.0)

        # Frame 2 on camera-01 (evidence reaches 2 -> confirmed UNKNOWN_PERSON)
        r2 = pipeline.process_frame(frame, camera_id="camera-01", timestamp=100.1)
        unknown_events = [e for e in r2.events if e.event_type == EventType.UNKNOWN_PERSON]
        assert len(unknown_events) == 1
        uid = unknown_events[0].identity_id
        assert uid.startswith(("UNK-P-", "P"))
        assert unknown_events[0].metadata["camera_id"] == "camera-01"
        assert "timestamp_iso" in unknown_events[0].metadata

        # Frame 3 on camera-01: track already has identity bound, no new UNKNOWN_PERSON alert
        r3 = pipeline.process_frame(frame, camera_id="camera-01", timestamp=100.2)
        assert not any(e.event_type == EventType.UNKNOWN_PERSON for e in r3.events)

        # Person appears on camera-04 at timestamp 120.0 with slightly varied face crop/embedding
        noisy_emb = _make_noisy_vector(person_emb, target_sim=0.90, seed=78)
        mock_verif_cam4 = VerificationResult(
            face_decision="UNKNOWN",
            face_similarity=0.2,
            face_embedding=noisy_emb,
            aligned_face=_make_face_crop(80, 80),
        )
        pipeline.identity_verifier.verify = MagicMock(return_value=mock_verif_cam4)

        r4 = pipeline.process_frame(frame, camera_id="camera-04", timestamp=120.0)
        reid_events = [e for e in r4.events if e.event_type == EventType.PERSON_REIDENTIFIED]
        assert len(reid_events) == 1
        assert reid_events[0].identity_id == uid
        assert reid_events[0].metadata["previous_camera"] == "camera-01"
        assert reid_events[0].metadata["current_camera"] == "camera-04"
        assert reid_events[0].metadata["camera_sequence"] == ["camera-01", "camera-04"]
        assert "timestamp_iso" in reid_events[0].metadata
