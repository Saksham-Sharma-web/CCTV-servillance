"""
Unit tests for Strict Separation of Tracking ID and Biometric Identity ID.
Verifies that track_id != identity_id and tracking functions independently of face matching.
"""

import pytest
import numpy as np
from ibvap.core.types import Detection, Track
from ibvap.core.config import IBVAPConfig
from ibvap.tracking.tracker import PersistentTracker
from ibvap.face.matcher_adapter import IdentityVerifierAdapter, AuthorizedPerson


def test_track_id_distinct_from_identity_id():
    config = IBVAPConfig(tracking_min_hits=1)
    tracker = PersistentTracker(config=config)

    # Person detected
    det = [Detection(bbox=(100, 100, 150, 200), class_id=0, class_name="person", confidence=0.9)]
    tracks = tracker.update(det)
    assert len(tracks) == 1
    
    track = tracks[0]
    initial_track_id = track.track_id
    assert isinstance(initial_track_id, int), "track_id must be an integer tracking identifier."
    assert track.identity_id is None, "identity_id must initially be None before face verification."

    # Associate biometric identity (e.g. user_42 matched)
    tracker.update_track_identity(
        track_id=initial_track_id,
        identity_id="user_42",
        identity_name="Alice Smith",
        confidence=0.94
    )

    # Next frame update
    det_next = [Detection(bbox=(103, 101, 153, 201), class_id=0, class_name="person", confidence=0.9)]
    tracks_next = tracker.update(det_next)
    assert len(tracks_next) == 1

    updated_track = tracks_next[0]
    assert updated_track.track_id == initial_track_id, "Visual track_id must persist unaffected."
    assert updated_track.identity_id == "user_42", "Biometric identity_id must be associated."
    assert str(updated_track.track_id) != updated_track.identity_id, "track_id and identity_id are distinct concepts."


def test_tracker_operates_without_face_identification():
    config = IBVAPConfig(tracking_min_hits=1)
    tracker = PersistentTracker(config=config)

    # Two persons: Person A has face verified, Person B has no face / unidentified
    dets_f1 = [
        Detection(bbox=(50, 50, 100, 150), class_id=0, class_name="person", confidence=0.88),
        Detection(bbox=(400, 400, 450, 500), class_id=0, class_name="person", confidence=0.85),
    ]
    tracks_f1 = tracker.update(dets_f1)
    assert len(tracks_f1) == 2
    id_a = tracks_f1[0].track_id
    id_b = tracks_f1[1].track_id

    # Identify Person A only
    tracker.update_track_identity(id_a, identity_id="officer_07", identity_name="Bob Jones", confidence=0.89)

    # Frame 2
    dets_f2 = [
        Detection(bbox=(54, 52, 104, 152), class_id=0, class_name="person", confidence=0.88),
        Detection(bbox=(404, 402, 454, 502), class_id=0, class_name="person", confidence=0.85),
    ]
    tracks_f2 = tracker.update(dets_f2)
    assert len(tracks_f2) == 2

    # Verify Person B is still tracked cleanly despite having no identity_id
    track_b = next(t for t in tracks_f2 if t.track_id == id_b)
    assert track_b.identity_id is None, "Person B must remain unidentified."
    assert track_b.track_id == id_b, "Person B must maintain its visual track_id across frames."


def test_biometric_adapter_cosine_verification():
    adapter = IdentityVerifierAdapter()

    # Register known person with 512-D normalized synthetic embedding
    synthetic_emb = np.random.randn(512).astype(np.float32)
    synthetic_emb /= np.linalg.norm(synthetic_emb)

    adapter.register_person(
        identity_id="employee_101",
        name="John Doe",
        embedding=synthetic_emb,
        role="STAFF"
    )

    # Test registry search with exact matching embedding
    adapter.similarity_threshold = 0.65
    person = adapter.authorized_registry["employee_101"]
    sim = float(np.dot(synthetic_emb, person.embedding))
    assert sim >= 0.99
