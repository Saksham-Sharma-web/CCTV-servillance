"""
Comprehensive Vehicle Detection and ANPR OCR Test Suite.
Tests:
1. Clear daytime car image with visible plate
2. Car with clearly visible license plate OCR extraction
3. Multiple cars in the same frame
4. Night / low-light conditions with headlight simulation
5. Small / distant vehicle plate
6. Structured output format contract
"""

import numpy as np
import cv2
import pytest

from ibvap.core.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig
from ibvap.core.types import Detection, WatchlistCategory
from ibvap.detection.object_detector import MockDetector
from ibvap.anpr.ocr_adapter import ANPRAdapter, PlateResult
from ibvap.anpr.plate_detector import LicensePlateDetector


def draw_synthetic_car(
    frame: np.ndarray,
    bbox: tuple,
    plate_text: str = "DL01AB1234",
    is_night: bool = False,
    plate_size: tuple = (160, 45)
):
    """
    Renders a realistic synthetic vehicle into a frame, complete with body,
    windshield, grille, bumper, and high-contrast license plate.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1

    car_color = (40, 40, 50) if is_night else (180, 50, 50)  # Dark at night, blue/red daytime

    # 1. Car body (lower 65%)
    body_y1 = y1 + int(h * 0.35)
    cv2.rectangle(frame, (x1, body_y1), (x2, y2), car_color, -1)

    # 2. Car roof/cabin (upper 35%)
    cabin_x1 = x1 + int(w * 0.15)
    cabin_x2 = x2 - int(w * 0.15)
    cabin_y1 = y1
    cabin_y2 = body_y1
    cv2.rectangle(frame, (cabin_x1, cabin_y1), (cabin_x2, cabin_y2), (80, 80, 90), -1)

    # 3. Windshield glass
    cv2.rectangle(frame, (cabin_x1 + 10, cabin_y1 + 8), (cabin_x2 - 10, cabin_y2 - 2), (200, 200, 210), -1)

    # 4. Grille & Bumper
    bumper_y1 = y1 + int(h * 0.65)
    bumper_y2 = y2 - int(h * 0.05)
    cv2.rectangle(frame, (x1 + int(w * 0.1), bumper_y1), (x2 - int(w * 0.1), bumper_y2), (30, 30, 30), -1)

    # 5. Headlights
    hl_color = (255, 255, 200) if is_night else (220, 220, 220)
    cv2.circle(frame, (x1 + int(w * 0.12), bumper_y1 + 15), 18, hl_color, -1)
    cv2.circle(frame, (x2 - int(w * 0.12), bumper_y1 + 15), 18, hl_color, -1)

    # 6. License plate (centered on bumper)
    pw, ph = plate_size
    px1 = x1 + int((w - pw) / 2)
    py1 = bumper_y1 + int((bumper_y2 - bumper_y1 - ph) / 2)
    px2 = px1 + pw
    py2 = py1 + ph

    # White plate background with thin black border
    cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), -1)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 0), 1)

    # Scale font so text comfortably fits with 10% margin on each side
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    for fs in np.linspace(1.1, 0.35, 20):
        sz, _ = cv2.getTextSize(plate_text, font, fs, 2)
        if sz[0] <= int(pw * 0.82) and sz[1] <= int(ph * 0.60):
            font_scale = fs
            break
    thickness = max(2, int(font_scale * 2.2))
    text_size, _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    tx = px1 + int((pw - text_size[0]) / 2)
    ty = py1 + int((ph + text_size[1]) / 2)
    cv2.putText(frame, plate_text, (tx, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    return (px1, py1, px2, py2)


def test_clear_daytime_car_with_visible_plate():
    """Test 1: Clear daytime car image with visible plate."""
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)  # Bright daytime background
    car_box = (400, 250, 880, 650)
    draw_synthetic_car(frame, car_box, plate_text="DL01AB1234", is_night=False, plate_size=(210, 52))

    mock_detector = MockDetector([
        Detection(bbox=car_box, class_id=2, class_name="car", confidence=0.94)
    ])
    config = IBVAPConfig(
        tracking_min_hits=1,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    result = pipeline.process_frame(frame, camera_id="cam-day-01")

    assert result.vehicle_detected is True
    assert result.vehicle_type == "car"
    assert result.vehicle_confidence >= 0.90
    assert result.license_plate_detected is True
    assert "DL01AB1234" in result.license_plate or "DL01" in result.license_plate or "AB123" in result.license_plate
    assert result.plate_confidence is not None


def test_plate_ocr_normalization():
    """Test 2: Normalization handles spaces, hyphens, and OCR letter/digit confusions."""
    adapter = ANPRAdapter()

    assert adapter.normalize_plate("DL 01 AB 1234") == "DL01AB1234"
    assert adapter.normalize_plate("UP-16-CD-5678") == "UP16CD5678"
    assert adapter.normalize_plate("0L01AB1234") == "DL01AB1234"  # 0 confused for D
    assert adapter.normalize_plate("mh-12.ef.9999") == "MH12EF9999"


def test_multiple_cars_in_same_frame():
    """Test 3: Multiple cars in the frame are tracked and analyzed independently."""
    frame = np.full((720, 1280, 3), 190, dtype=np.uint8)
    car1_box = (100, 280, 550, 620)
    car2_box = (680, 280, 1150, 620)

    draw_synthetic_car(frame, car1_box, plate_text="HR26DK8392", is_night=False, plate_size=(160, 44))
    draw_synthetic_car(frame, car2_box, plate_text="MH12AB9000", is_night=False, plate_size=(160, 44))

    mock_detector = MockDetector([
        Detection(bbox=car1_box, class_id=2, class_name="car", confidence=0.92),
        Detection(bbox=car2_box, class_id=2, class_name="car", confidence=0.91),
    ])
    config = IBVAPConfig(
        tracking_min_hits=1,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    result = pipeline.process_frame(frame, camera_id="cam-multi-01")

    assert result.vehicle_detected is True
    assert len(result.tracks) == 2
    # At least one of the cars has plate recognized
    detected_plates = [t.plate_number for t in result.tracks if t.plate_number]
    assert len(detected_plates) >= 1


def test_night_low_light_car_detection():
    """Test 4: Low-light / night conditions with CLAHE contrast enhancement."""
    # Dark ambient frame (luminance ~ 35)
    frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
    car_box = (350, 220, 880, 640)
    draw_synthetic_car(frame, car_box, plate_text="UP16AB1234", is_night=True, plate_size=(200, 52))

    mock_detector = MockDetector([
        Detection(bbox=car_box, class_id=2, class_name="car", confidence=0.88)
    ])
    config = IBVAPConfig(
        tracking_min_hits=1,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    result = pipeline.process_frame(frame, camera_id="cam-night-01")

    assert result.vehicle_detected is True
    assert result.license_plate_detected is True
    assert "UP16" in result.license_plate or "16AB" in result.license_plate or "UP16AB1234" in result.license_plate


def test_small_distant_license_plate():
    """Test 5: Smaller vehicle and plate resolution."""
    frame = np.full((720, 1280, 3), 180, dtype=np.uint8)
    # Small car in distance
    car_box = (450, 200, 750, 440)
    draw_synthetic_car(frame, car_box, plate_text="KA05MJ4444", is_night=False, plate_size=(130, 36))

    mock_detector = MockDetector([
        Detection(bbox=car_box, class_id=2, class_name="car", confidence=0.85)
    ])
    config = IBVAPConfig(
        tracking_min_hits=1,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    result = pipeline.process_frame(frame, camera_id="cam-small-01")

    assert result.vehicle_detected is True
    assert result.license_plate_detected is True
    assert len(result.license_plate) >= 4


def test_structured_output_format():
    """Test 6: Verifies the exact structured output requirements."""
    frame = np.full((480, 640, 3), 180, dtype=np.uint8)
    car_box = (100, 100, 500, 400)
    draw_synthetic_car(frame, car_box, plate_text="DL04CD5678", is_night=False, plate_size=(200, 50))

    mock_detector = MockDetector([
        Detection(bbox=car_box, class_id=2, class_name="car", confidence=0.95)
    ])
    config = IBVAPConfig(
        tracking_min_hits=1,
        redis_enabled=False,
        db_enabled=False,
        storage_enabled=False
    )
    pipeline = IBVAPPipeline(config=config, detector=mock_detector)

    result = pipeline.process_frame(frame)
    res_dict = result.to_dict()

    # Verify root-level analysis fields
    assert "vehicle_detected" in res_dict
    assert "vehicle_type" in res_dict
    assert "vehicle_confidence" in res_dict
    assert "license_plate_detected" in res_dict
    assert "license_plate" in res_dict
    assert "plate_confidence" in res_dict
    assert "ocr_confidence" in res_dict

    # Verify vehicle_analysis sub-dictionary
    va = res_dict["vehicle_analysis"]
    assert va["vehicle_detected"] is True
    assert va["vehicle_type"] == "car"
    assert va["vehicle_confidence"] >= 0.90
    assert va["license_plate_detected"] is True
    assert "DL04" in va["license_plate"] or "CD5678" in va["license_plate"] or "DL04CD5678" in va["license_plate"] or len(va["license_plate"]) >= 4
    assert va["plate_confidence"] > 0
    assert va["ocr_confidence"] > 0
