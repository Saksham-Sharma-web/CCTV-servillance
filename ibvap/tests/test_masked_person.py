"""
Unit tests for MaskedPersonDetector.
Verifies lower-face texture entropy and occlusion analysis on synthetic face crops.
"""

import pytest
import numpy as np
import cv2
from ibvap.appearance.masked_person import MaskedPersonDetector


def test_masked_person_uniform_mask():
    detector = MaskedPersonDetector()

    # Create synthetic face image with uniform lower half (surgical mask simulation)
    face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    # Upper half has eyes / skin gradient
    cv2.randn(face_crop[:50, :], (180, 180, 180), (30, 30, 30))
    # Lower half has solid surgical blue mask (very low texture entropy)
    face_crop[50:, :] = (230, 200, 100)  # Solid cyan/blue color

    res = detector.analyze_face(face_crop)
    assert res.is_masked is True
    assert res.concealment_type == "MASKED"
    assert res.confidence >= 0.70


def test_unmasked_person_natural_texture():
    detector = MaskedPersonDetector()

    # Create synthetic face with natural skin variations across both halves
    face_crop = np.zeros((100, 100, 3), dtype=np.uint8)
    # Random normal distribution simulating facial details (mouth, lips, chin, stubble)
    cv2.randn(face_crop, (150, 150, 150), (45, 45, 45))

    res = detector.analyze_face(face_crop)
    assert res.is_masked is False
    assert res.concealment_type == "UNMASKED"


def test_invalid_face_crop_handling():
    detector = MaskedPersonDetector()
    res = detector.analyze_face(np.zeros((10, 10, 3), dtype=np.uint8))
    assert res.is_masked is False
    assert res.concealment_type == "UNKNOWN"


def test_alert_manager_masked_person_severity():
    from ibvap.events.alert_manager import AlertManager, AlertSeverity
    from ibvap.core.types import AnalyticsEvent, EventType

    manager = AlertManager()
    event = AnalyticsEvent(event_type=EventType.MASKED_PERSON, track_id=1, confidence=0.85)
    severity = manager.classify_severity(event)
    assert severity == AlertSeverity.CRITICAL


def test_pipeline_masked_person_event_generation():
    from ibvap.core.pipeline import IBVAPPipeline
    from ibvap.core.config import IBVAPConfig
    from ibvap.core.types import Detection, EventType
    from ibvap.detection.object_detector import MockDetector
    from ibvap.face.detector import FaceDetection

    # Mock detector returning a person
    mock_det = MockDetector()
    mock_det.set_detections([
        Detection(bbox=(50, 50, 200, 350), class_id=0, class_name="person", confidence=0.90)
    ])

    config = IBVAPConfig(
        tracking_min_hits=1,
        mask_detection_enabled=True,
        mask_temporal_confirmation_frames=2,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False,
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_det)

    # Synthetic frame where the upper person crop contains a simulated surgical mask
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Put surgical mask pattern in the person region (y: 50..350, x: 50..200)
    # Upper face (eyes/forehead): natural skin tone
    frame[50:110, 75:175] = (160, 180, 200)
    # Lower face (mask): uniform blue/cyan color
    frame[110:160, 75:175] = (230, 200, 100)

    # Mock the face detector on pipeline to return a valid face covering this patch
    class MockFaceDetector:
        active_detector_type = "mock_face"
        def detect_faces(self, crop):
            return [
                FaceDetection(
                    box=(25, 0, 125, 110),
                    confidence=0.88,
                    landmarks=np.array([[45, 30], [80, 30], [62, 55], [50, 80], [75, 80]]),
                    quality_status="GOOD_FACE"
                )
            ]

    pipeline.face_detector = MockFaceDetector()

    # Process frame 1 (Single image / immediate confirmation in test mode)
    res1 = pipeline.process_frame(frame, camera_id="cam-01", timestamp=100.0)
    assert len(res1.tracks) == 1
    track = res1.tracks[0]
    assert track.is_masked is True
    assert track.concealment_type == "MASKED"

    # Verify event generation
    masked_events = [e for e in res1.events if e.event_type == EventType.MASKED_PERSON]
    assert len(masked_events) == 1
    assert masked_events[0].track_id == track.track_id
    assert masked_events[0].confidence >= 0.70

