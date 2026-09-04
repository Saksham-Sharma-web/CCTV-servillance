"""
Unit tests for Persistent Multi-Object Tracker.
Verifies ID persistence, multi-object separation, and occlusion recovery.
"""

import pytest
from ibvap.core.types import Detection
from ibvap.core.config import IBVAPConfig
from ibvap.tracking.tracker import PersistentTracker


def test_tracking_same_object_id_persistence():
    config = IBVAPConfig(tracking_min_hits=2, tracking_iou_threshold=0.3)
    tracker = PersistentTracker(config=config)

    # Frame 1: Person at (100, 100, 150, 200)
    det1 = [Detection(bbox=(100, 100, 150, 200), class_id=0, class_name="person", confidence=0.9)]
    tracks1 = tracker.update(det1)
    assert len(tracks1) >= 1
    track_id = tracks1[0].track_id

    # Frame 2: Person slightly moved to (104, 102, 154, 202)
    det2 = [Detection(bbox=(104, 102, 154, 202), class_id=0, class_name="person", confidence=0.9)]
    tracks2 = tracker.update(det2)
    assert len(tracks2) == 1
    assert tracks2[0].track_id == track_id, "Track ID must remain identical across consecutive frames."

    # Frame 3: Person slightly moved to (108, 104, 158, 204)
    det3 = [Detection(bbox=(108, 104, 158, 204), class_id=0, class_name="person", confidence=0.9)]
    tracks3 = tracker.update(det3)
    assert len(tracks3) == 1
    assert tracks3[0].track_id == track_id, "Track ID must persist on frame 3."


def test_tracking_multiple_distinct_objects():
    config = IBVAPConfig(tracking_min_hits=1)
    tracker = PersistentTracker(config=config)

    # Two distinct detections far apart: Person in top-left, Car in bottom-right
    dets = [
        Detection(bbox=(50, 50, 100, 150), class_id=0, class_name="person", confidence=0.92),
        Detection(bbox=(500, 500, 700, 600), class_id=2, class_name="car", confidence=0.88),
    ]

    tracks = tracker.update(dets)
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id, "Distinct objects must receive different track IDs."
    
    classes = {t.class_name for t in tracks}
    assert "person" in classes
    assert "car" in classes


def test_tracking_temporary_occlusion_survival():
    config = IBVAPConfig(tracking_min_hits=1, tracking_max_lost_frames=5)
    tracker = PersistentTracker(config=config)

    # Frame 1: Person appears
    det = [Detection(bbox=(200, 200, 250, 300), class_id=0, class_name="person", confidence=0.9)]
    tracks1 = tracker.update(det)
    initial_id = tracks1[0].track_id

    # Frame 2: Person moves
    det = [Detection(bbox=(205, 202, 255, 302), class_id=0, class_name="person", confidence=0.9)]
    tracks2 = tracker.update(det)
    assert tracks2[0].track_id == initial_id

    # Frame 3 & 4: Detection missed / temporary occlusion
    tracks_empty1 = tracker.update([])
    tracks_empty2 = tracker.update([])

    # Frame 5: Person reappears near predicted location
    det = [Detection(bbox=(215, 208, 265, 308), class_id=0, class_name="person", confidence=0.9)]
    tracks5 = tracker.update(det)
    assert len(tracks5) >= 1
    assert tracks5[0].track_id == initial_id, "Track ID must survive temporary detection loss."
