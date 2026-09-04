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
