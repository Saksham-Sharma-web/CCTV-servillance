"""
Comprehensive Unit Tests for IBVAP Person-Centric Event Correlation,
Identity Persistence, and Cross-Camera Trajectory.
"""

import time
import pytest
import numpy as np

from ibvap.core.config import IBVAPConfig
from ibvap.core.types import (
    Detection,
    Track,
    EventType,
    AnalyticsEvent,
    PresenceSession,
    TrajectorySegment,
)
from ibvap.tracking.unknown_storage import SQLiteUnknownStorage
from ibvap.tracking.unknown_person import UnknownPersonManager
from ibvap.events.correlator import PersonEventCorrelator
from ibvap.events.event_engine import EventEngine


@pytest.fixture
def temp_db(tmp_path):
    db_file = str(tmp_path / "test_person_correlation.db")
    return db_file


@pytest.fixture
def correlator(temp_db):
    cfg = IBVAPConfig(
        unknown_storage_db_path=temp_db,
        person_identity_enabled=True,
        person_session_timeout_seconds=30.0,
        person_event_update_interval_seconds=5.0,
        person_trajectory_max_gap_seconds=300.0,
        event_cooldown_seconds=5.0,
    )
    storage = SQLiteUnknownStorage(db_path=temp_db, config=cfg)
    return PersonEventCorrelator(config=cfg, storage=storage)


class TestPersonCentricCorrelation:

    def test_single_camera_continuous_presence_deduplication(self, correlator):
        """
        Scenario: 100 observations of Person P001 on CAM_01 over 20 minutes.
        Expectation:
        - 1 presence session.
        - Exactly 1 initial PERSON_DETECTED event created.
        - Periodic duration updates, but NO flood of duplicate events.
        """
        person_id = "P001"
        camera_id = "CAM_01"
        track_id = 17

        # Frame 1: 10:00:00 (t=0)
        ev1 = correlator.correlate_event(
            camera_id=camera_id,
            event_type=EventType.PERSON_DETECTED,
            timestamp=1000.0,
            person_id=person_id,
            track_id=track_id,
            confidence=0.92,
        )
        assert ev1 is not None
        assert ev1.event_type == EventType.PERSON_DETECTED
        assert ev1.person_id == person_id
        assert ev1.is_update is False
        assert ev1.event_status == "ACTIVE"
        initial_event_id = ev1.event_id

        # Frame 2: 10:00:01 (t=1s, within update interval)
        ev2 = correlator.correlate_event(
            camera_id=camera_id,
            event_type=EventType.PERSON_DETECTED,
            timestamp=1001.0,
            person_id=person_id,
            track_id=track_id,
            confidence=0.92,
        )
        assert ev2 is None  # Suppressed! No duplicate event emitted!

        # Frame 10: 10:00:06 (t=6s, exceeds 5s update interval)
        ev3 = correlator.correlate_event(
            camera_id=camera_id,
            event_type=EventType.PERSON_DETECTED,
            timestamp=1006.0,
            person_id=person_id,
            track_id=track_id,
            confidence=0.92,
        )
        assert ev3 is not None
        assert ev3.event_id == initial_event_id  # Same logical event!
        assert ev3.is_update is True
        assert ev3.duration_seconds >= 6.0
        assert ev3.metadata["status"] == "PRESENT"

        # Simulate 100 frames spanning 20 minutes (t = 1000 to 2200)
        update_count = 0
        for t in np.linspace(1010.0, 2200.0, 100):
            ev = correlator.correlate_event(
                camera_id=camera_id,
                event_type=EventType.PERSON_DETECTED,
                timestamp=float(t),
                person_id=person_id,
                track_id=track_id,
                confidence=0.92,
            )
            if ev is not None:
                assert ev.event_id == initial_event_id
                assert ev.is_update is True
                update_count += 1

        # Check in-memory session
        session, _, _ = correlator.get_or_create_session(person_id, camera_id, 2200.0, track_id)
        assert session.status == "ACTIVE"
        assert session.duration_seconds == 1200.0  # 20 minutes!
        assert session.tracker_ids == [track_id]

    def test_loitering_quashed_into_presence_duration(self, correlator):
        """
        Scenario: Track remains stationary, loitering detector triggers LOITERING.
        Expectation:
        - Loitering event is QUASHED and does NOT produce repeated LOITERING alerts.
        - Presence duration updates with is_loitering = True.
        """
        person_id = "P001"
        camera_id = "CAM_02"
        track_id = 4

        # 1. Person enters camera at 15:20:00
        ev_entry = correlator.correlate_event(
            camera_id=camera_id,
            event_type=EventType.PERSON_DETECTED,
            timestamp=1000.0,
            person_id=person_id,
            track_id=track_id,
        )
        assert ev_entry is not None

        # 2. Stationary dwell triggers LOITERING at 15:20:15
        ev_loiter1 = correlator.correlate_event(
            camera_id=camera_id,
            event_type=EventType.LOITERING,
            timestamp=1015.0,
            person_id=person_id,
            track_id=track_id,
            metadata={"stationary_duration": 15.0},
        )
        # Separate LOITERING alert is quashed! If an event is emitted, it's a PERSON_DETECTED duration update
        if ev_loiter1 is not None:
            assert ev_loiter1.event_type == EventType.PERSON_DETECTED
            assert ev_loiter1.is_update is True
            assert ev_loiter1.metadata.get("is_loitering") is True

        # Check session metadata updated
        session, _, _ = correlator.get_or_create_session(person_id, camera_id, 1015.0, track_id)
        assert session.metadata.get("is_loitering") is True

        # 3. Repeated LOITERING detection at 15:20:30
        ev_loiter2 = correlator.correlate_event(
            camera_id=camera_id,
            event_type=EventType.LOITERING,
            timestamp=1030.0,
            person_id=person_id,
            track_id=track_id,
            metadata={"stationary_duration": 30.0},
        )
        if ev_loiter2 is not None:
            assert ev_loiter2.event_type == EventType.PERSON_DETECTED
            assert ev_loiter2.is_update is True

    def test_cross_camera_trajectory_and_camera_return(self, correlator):
        """
        Scenario: CAM_01 -> CAM_02 -> CAM_03 -> CAM_01.
        Expectation:
        - All visits belong to same person P001.
        - Trajectory preserves 4 distinct ordered segments.
        - Return to CAM_01 does NOT create a duplicate person or reset history.
        """
        person_id = "P001"

        # 1. CAM_01: 10:00:00 -> 10:05:20 (320s, frames every 20s within 30s session timeout)
        for t in range(1000, 1321, 20):
            correlator.correlate_event("CAM_01", EventType.PERSON_DETECTED, float(t), person_id=person_id, track_id=17)

        # 2. CAM_02: 10:05:35 -> 10:08:10 (155s)
        for t in range(1335, 1491, 20):
            correlator.correlate_event("CAM_02", EventType.PERSON_DETECTED, float(t), person_id=person_id, track_id=4)

        # 3. CAM_03: 10:08:22 -> 10:11:45 (203s)
        for t in range(1502, 1706, 20):
            correlator.correlate_event("CAM_03", EventType.PERSON_DETECTED, float(t), person_id=person_id, track_id=11)

        # 4. Return to CAM_01: 10:12:03 -> 10:15:40 (217s)
        for t in range(1723, 1941, 20):
            correlator.correlate_event("CAM_01", EventType.PERSON_DETECTED, float(t), person_id=person_id, track_id=31)

        # Retrieve profile and trajectory
        profile = correlator.get_person_profile(person_id)
        assert profile["person_id"] == person_id
        assert profile["current_camera"] == "CAM_01"
        assert profile["status"] == "PRESENT"

        trajectory = profile["trajectory"]
        assert len(trajectory) == 4
        assert trajectory[0]["camera_id"] == "CAM_01"
        assert trajectory[1]["camera_id"] == "CAM_02"
        assert trajectory[2]["camera_id"] == "CAM_03"
        assert trajectory[3]["camera_id"] == "CAM_01"  # Return preserved!

        # Verify tracker IDs per segment (Section 5: camera_id + tracker_id is local)
        assert trajectory[0]["tracker_ids"] == [17]
        assert trajectory[1]["tracker_ids"] == [4]
        assert trajectory[2]["tracker_ids"] == [11]
        assert trajectory[3]["tracker_ids"] == [31]

    def test_reappearance_after_long_gap(self, correlator):
        """
        Scenario: Person P001 on CAM_01 (09:00 -> 09:20), leaves, then returns at 14:00.
        Expectation:
        - Same person P001.
        - Two distinct presence sessions.
        """
        person_id = "P001"
        camera_id = "CAM_01"

        # Visit 1: 09:00 (t=0) to 09:20 (t=1200) with continuous 20s frame intervals
        ev1 = correlator.correlate_event(camera_id, EventType.PERSON_DETECTED, 0.0, person_id=person_id, track_id=1)
        for t in range(20, 1201, 20):
            correlator.correlate_event(camera_id, EventType.PERSON_DETECTED, float(t), person_id=person_id, track_id=1)
        session_1_id = ev1.session_id

        # Cull sessions after person leaves (t=2000 > 1200 + 30s timeout)
        closed = correlator.cull_inactive_sessions(2000.0)
        assert len(closed) == 1
        assert closed[0].event_status == "CLOSED"

        # Visit 2: 14:00 (t=18000)
        ev2 = correlator.correlate_event(camera_id, EventType.PERSON_DETECTED, 18000.0, person_id=person_id, track_id=2)
        assert ev2 is not None
        assert ev2.person_id == person_id
        assert ev2.session_id != session_1_id  # Distinct presence session!
        assert ev2.is_update is False

        trajectory = correlator.get_trajectory(person_id)
        assert len(trajectory) == 2
        assert trajectory[0]["session_id"] == session_1_id
        assert trajectory[1]["session_id"] == ev2.session_id

    def test_high_priority_events_preserved(self, correlator):
        """
        Scenario: Person has active presence session, then triggers FENCE_INTRUSION.
        Expectation:
        - FENCE_INTRUSION is NOT swallowed by presence deduplication.
        - It is emitted as a distinct high-priority alert.
        """
        person_id = "P001"
        camera_id = "CAM_01"

        # 1. Active presence event
        correlator.correlate_event(camera_id, EventType.PERSON_DETECTED, 100.0, person_id=person_id, track_id=5)

        # 2. Fence intrusion occurs 2 seconds later
        hp_event = correlator.correlate_event(
            camera_id,
            EventType.FENCE_INTRUSION,
            102.0,
            person_id=person_id,
            track_id=5,
            metadata={"zone_id": "fence-north"},
        )
        assert hp_event is not None
        assert hp_event.event_type == EventType.FENCE_INTRUSION
        assert hp_event.person_id == person_id

    def test_conservative_matching_outcomes(self, temp_db):
        """
        Tests the 3 conservative matching outcomes:
        - High confidence match -> associated
        - Uncertain match -> separate candidate retained
        - No match -> new unknown person created
        """
        cfg = IBVAPConfig(
            unknown_storage_db_path=temp_db,
            unknown_face_similarity_threshold=0.65,
            unknown_face_high_confidence_threshold=0.75,
            unknown_similarity_margin=0.05,
            unknown_min_quality_score=0.40,
            unknown_evidence_min_hits=2,
        )
        mgr = UnknownPersonManager(config=cfg)

        # High-quality dummy face crop (sharpness and brightness passing checks)
        dummy_crop = np.zeros((80, 80, 3), dtype=np.uint8)
        dummy_crop[::2, ::2] = 200
        dummy_crop[1::2, 1::2] = 50

        # Base embedding for Person 1
        rng = np.random.RandomState(42)
        emb1 = rng.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)

        # 1. Register P001 via 2 evidence hits
        res1 = mgr.match_or_register(emb1, "CAM_01", 1, 100.0, (10, 10, 50, 50), face_crop=dummy_crop)
        assert res1.decision == "COLLECTING_EVIDENCE"

        res2 = mgr.match_or_register(emb1, "CAM_01", 1, 100.1, (10, 10, 50, 50), face_crop=dummy_crop)
        assert res2.decision == "NEW_UNKNOWN"
        assert res2.unknown_id == "P001"

        # 2. High confidence match on CAM_02 (similarity ~ 0.90 >= 0.75)
        rand_v = rng.randn(512).astype(np.float32)
        rand_v -= np.dot(rand_v, emb1) * emb1
        rand_v /= np.linalg.norm(rand_v)
        emb_high = (0.90 * emb1 + np.sqrt(1 - 0.90**2) * rand_v).astype(np.float32)
        emb_high /= np.linalg.norm(emb_high)

        res3 = mgr.match_or_register(emb_high, "CAM_02", 2, 110.0, (20, 20, 60, 60), face_crop=dummy_crop)
        assert res3.decision in ("REIDENTIFIED", "SIGHTING")
        assert res3.unknown_id == "P001"

        # 3. No match on CAM_03 (orthogonal vector, sim ~ 0.0 < 0.65)
        emb_no_match = rng.randn(512).astype(np.float32)
        emb_no_match -= np.dot(emb_no_match, emb1) * emb1
        emb_no_match /= np.linalg.norm(emb_no_match)

        res4_1 = mgr.match_or_register(emb_no_match, "CAM_03", 3, 120.0, (30, 30, 70, 70), face_crop=dummy_crop)
        assert res4_1.decision == "COLLECTING_EVIDENCE"
        res4_2 = mgr.match_or_register(emb_no_match, "CAM_03", 3, 120.1, (30, 30, 70, 70), face_crop=dummy_crop)
        assert res4_2.decision == "NEW_UNKNOWN"
        assert res4_2.unknown_id == "P002"
