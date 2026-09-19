"""
Unit Tests for PlateQualityScorer (Phase 2).
Verifies:
1. High-quality synthetic plate crop produces reasonable component scores and is_acceptable=True.
2. Heavy blur produces lower sharpness score.
3. Extremely small crop produces low resolution score.
4. Distorted aspect ratio produces lower aspect ratio score.
5. Low contrast produces lower contrast score.
6. Severely dark crop produces lower luminance score.
7. Severely bright crop produces lower luminance score.
8. Empty / zero-sized crop gracefully returns safe report with is_acceptable=False without crashing.
9. None input gracefully returns safe report with is_acceptable=False without crashing.
10. Output is always a valid PlateQualityReport instance.
11. All component scores remain within [0.0, 100.0].
12. Overall score remains within [0.0, 100.0].
13. Invalid or malformed inputs cannot accidentally become acceptable.
14. score_observation correctly attaches report to VehicleObservation.
"""

import cv2
import numpy as np
import pytest

from ibvap.vehicle.types import PlateQualityReport, VehicleObservation
from ibvap.vehicle.quality import PlateQualityScorer


def create_synthetic_plate(
    width: int = 120,
    height: int = 36,
    text: str = "DL01AB1234",
    bg_color: tuple = (255, 255, 255),
    text_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Generates a sharp, realistic synthetic license plate crop."""
    plate = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Draw border
    cv2.rectangle(plate, (0, 0), (width - 1, height - 1), (40, 40, 40), 1)
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = height / 42.0
    thickness = max(1, int(font_scale * 2.0))
    sz, _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = max(2, int((width - sz[0]) / 2))
    ty = max(height - 4, int((height + sz[1]) / 2))
    cv2.putText(plate, text, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return plate


def test_high_quality_plate_crop():
    """Test 1: High quality synthetic plate produces high component scores and acceptable status."""
    scorer = PlateQualityScorer()
    plate = create_synthetic_plate(width=130, height=40, text="DL01AB1234")

    report = scorer.score(plate)

    assert isinstance(report, PlateQualityReport)
    assert report.is_acceptable is True
    assert report.overall_score >= 60.0
    assert report.sharpness_score >= 50.0
    assert report.resolution_score >= 80.0
    assert report.aspect_ratio_score >= 70.0
    assert report.contrast_score >= 50.0
    assert report.luminance_score >= 70.0

    # Check details payload
    assert "laplacian_variance" in report.details
    assert report.details["width"] == 130
    assert report.details["height"] == 40


def test_blurry_crop_produces_low_sharpness():
    """Test 2: Heavily blurred plate crop produces significantly lower sharpness score than sharp plate."""
    scorer = PlateQualityScorer()
    sharp_plate = create_synthetic_plate(width=130, height=40, text="MH12AB9999")
    blurry_plate = cv2.GaussianBlur(sharp_plate, (15, 15), 5.0)

    sharp_report = scorer.score(sharp_plate)
    blurry_report = scorer.score(blurry_plate)

    assert blurry_report.sharpness_score < sharp_report.sharpness_score
    assert blurry_report.sharpness_score < 30.0
    assert blurry_report.overall_score < sharp_report.overall_score


def test_extremely_small_crop_produces_poor_resolution():
    """Test 3: Plate below minimum resolution thresholds produces low resolution score."""
    scorer = PlateQualityScorer()
    tiny_plate = np.full((12, 32, 3), 200, dtype=np.uint8)

    report = scorer.score(tiny_plate)

    assert report.resolution_score < 30.0
    assert report.overall_score < 60.0


def test_bad_aspect_ratio_produces_low_aspect_score():
    """Test 4: Square or vertical boxes (bad aspect ratio) score lower on aspect_ratio_score."""
    scorer = PlateQualityScorer()
    # Canonical ratio plate (~3.25)
    normal_plate = np.full((36, 120, 3), 200, dtype=np.uint8)
    # Square box (ratio = 1.0)
    square_box = np.full((60, 60, 3), 200, dtype=np.uint8)
    # Tall vertical box (ratio = 0.33)
    tall_box = np.full((120, 40, 3), 200, dtype=np.uint8)

    normal_report = scorer.score(normal_plate)
    square_report = scorer.score(square_box)
    tall_report = scorer.score(tall_box)

    assert normal_report.aspect_ratio_score > square_report.aspect_ratio_score
    assert square_report.aspect_ratio_score > tall_report.aspect_ratio_score
    assert tall_report.aspect_ratio_score < 20.0


def test_low_contrast_produces_low_contrast_score():
    """Test 5: Flat uniform crop with near-zero intensity variance scores low on contrast."""
    scorer = PlateQualityScorer()
    sharp_plate = create_synthetic_plate(width=120, height=36, text="KA05MJ1111")
    # Uniform gray plate (std ~ 0)
    flat_plate = np.full((36, 120, 3), 128, dtype=np.uint8)

    sharp_report = scorer.score(sharp_plate)
    flat_report = scorer.score(flat_plate)

    assert flat_report.contrast_score < 5.0
    assert flat_report.contrast_score < sharp_report.contrast_score


def test_severely_dark_crop_produces_lower_luminance():
    """Test 6: Severely underexposed / black plate crop produces lower luminance score."""
    scorer = PlateQualityScorer()
    normal_plate = np.full((36, 120, 3), 150, dtype=np.uint8)
    dark_plate = np.full((36, 120, 3), 10, dtype=np.uint8)

    normal_report = scorer.score(normal_plate)
    dark_report = scorer.score(dark_plate)

    assert dark_report.luminance_score < normal_report.luminance_score
    assert dark_report.luminance_score <= 40.0


def test_severely_bright_crop_produces_lower_luminance():
    """Test 7: Blown-out / overexposed plate crop produces lower luminance score."""
    scorer = PlateQualityScorer()
    normal_plate = np.full((36, 120, 3), 150, dtype=np.uint8)
    glare_plate = np.full((36, 120, 3), 252, dtype=np.uint8)

    normal_report = scorer.score(normal_plate)
    glare_report = scorer.score(glare_plate)

    assert glare_report.luminance_score < normal_report.luminance_score
    assert glare_report.luminance_score <= 40.0


def test_empty_and_zero_sized_crop_handling():
    """Test 8: Empty or 0-dimension crops return safe unaccepted report without raising an exception."""
    scorer = PlateQualityScorer()
    empty_crop = np.empty((0, 0, 3), dtype=np.uint8)

    report = scorer.score(empty_crop)

    assert isinstance(report, PlateQualityReport)
    assert report.is_acceptable is False
    assert report.overall_score == 0.0
    assert "error" in report.details


def test_none_input_handling():
    """Test 9: None input returns safe unaccepted report without raising an exception."""
    scorer = PlateQualityScorer()

    report = scorer.score(None)

    assert isinstance(report, PlateQualityReport)
    assert report.is_acceptable is False
    assert report.overall_score == 0.0
    assert report.details.get("error") == "crop_is_none"


def test_grayscale_and_single_channel_crops():
    """Test 10: Supports 2D grayscale and single-channel 3D crops gracefully."""
    scorer = PlateQualityScorer()
    gray_crop = np.full((36, 120), 180, dtype=np.uint8)
    report_2d = scorer.score(gray_crop)
    assert isinstance(report_2d, PlateQualityReport)
    assert report_2d.overall_score > 0.0

    gray_3d = np.full((36, 120, 1), 180, dtype=np.uint8)
    report_3d = scorer.score(gray_3d)
    assert isinstance(report_3d, PlateQualityReport)
    assert report_3d.overall_score > 0.0


def test_score_ranges_are_strictly_bounded():
    """Test 11 & 12: All scores strictly stay in [0.0, 100.0] across diverse test images."""
    scorer = PlateQualityScorer()

    test_images = [
        create_synthetic_plate(width=140, height=45, text="UP16AB1234"),
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.full((200, 200, 3), 255, dtype=np.uint8),
        np.random.randint(0, 256, (30, 90, 3), dtype=np.uint8),
    ]

    for img in test_images:
        report = scorer.score(img)
        assert 0.0 <= report.overall_score <= 100.0
        assert 0.0 <= report.sharpness_score <= 100.0
        assert 0.0 <= report.resolution_score <= 100.0
        assert 0.0 <= report.aspect_ratio_score <= 100.0
        assert 0.0 <= report.contrast_score <= 100.0
        assert 0.0 <= report.luminance_score <= 100.0


def test_invalid_input_cannot_be_acceptable():
    """Test 13: Invalid inputs can never yield is_acceptable=True."""
    scorer = PlateQualityScorer()

    invalid_inputs = [
        None,
        np.empty((0, 0), dtype=np.uint8),
        np.zeros((1, 1, 3), dtype=np.uint8),
        "not_an_image",
        np.zeros((10, 10, 5), dtype=np.uint8),  # 5 channels
    ]

    for inp in invalid_inputs:
        report = scorer.score(inp)
        assert report.is_acceptable is False
        assert report.overall_score == 0.0


def test_score_observation_convenience_method():
    """Test 14: score_observation correctly updates VehicleObservation.quality."""
    scorer = PlateQualityScorer()
    plate = create_synthetic_plate(width=120, height=36, text="HR26DK8392")
    obs = VehicleObservation(
        track_id=42,
        frame_index=15,
        timestamp=20.5,
        plate_bbox=(10, 10, 130, 46),
        plate_crop=plate
    )

    assert obs.quality is None
    ret_report = scorer.score_observation(obs)

    assert obs.quality is not None
    assert obs.quality == ret_report
    assert obs.quality.is_acceptable is True
    assert obs.quality.overall_score >= 60.0
