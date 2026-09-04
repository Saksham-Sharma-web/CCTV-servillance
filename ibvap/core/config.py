"""
IBVAP Configuration System.
Provides central, validated, and environment-configurable parameters for all analytics modules.
Avoids hardcoded magic numbers.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os


@dataclass
class IBVAPConfig:
    # ── Object Detection ───────────────────────────────────────
    detector_model_name: str = "yolov8n.pt"
    detection_confidence: float = 0.35
    detection_iou_threshold: float = 0.45
    target_classes: List[str] = field(default_factory=lambda: [
        "person",
        "car",
        "suv",
        "van",
        "vehicle",
        "motorcycle",
        "bus",
        "truck",
        "bicycle",
        "backpack",
        "handbag",
        "suitcase"
    ])

    # ── Object Tracking ─────────────────────────────────────────
    tracking_iou_threshold: float = 0.30
    tracking_max_lost_frames: int = 30
    tracking_min_hits: int = 3

    # ── Face Detection & Biometric Verification ────────────────
    face_detection_enabled: bool = True
    face_detection_confidence: float = 0.60
    face_verification_similarity_threshold: float = 0.65  # Calibrated operating threshold
    face_verification_interval_frames: int = 15  # Do not run face matching every single frame
    face_min_width: int = 24  # Minimum acceptable face width in px
    face_min_height: int = 24  # Minimum acceptable face height in px
    face_blur_threshold: float = 15.0  # Laplacian variance threshold for blur
    face_min_brightness: float = 25.0  # Minimum mean luminance (0-255)
    face_max_brightness: float = 240.0  # Maximum mean luminance (0-255)
    face_enable_tiling: bool = False  # Enable tiled detection for small faces
    face_tile_size: int = 480
    face_tile_overlap: float = 0.2
    require_high_accuracy: bool = False  # If True, refuse Haar degradation

    # ── Body Appearance & Re-ID Support ─────────────────────────
    body_support_enabled: bool = True
    body_weights: Dict[str, float] = field(default_factory=lambda: {
        "most_recent": 0.20,
        "recent": 0.10,
        "old": 0.00
    })

    # ── Frame Sampling & Ingestion Rate (Phase 7) ───────────────
    analysis_fps: float = 8.0  # INITIAL ENGINEERING DEFAULT: ~8 FPS analysis
    camera_fps: float = 24.0   # Nominal CCTV capture FPS
    frame_sampling_enabled: bool = False  # Disabled by default in pipeline for unit test compatibility

    # ── ANPR / License Plate Recognition ─────────────────────────
    anpr_enabled: bool = True
    anpr_ocr_interval_frames: int = 10  # Throttle OCR to run once every N frames per vehicle
    anpr_min_plate_aspect_ratio: float = 1.3
    anpr_max_plate_aspect_ratio: float = 6.0

    # ── Track-Centric Vehicle ANPR Subsystem (Phases 1–6) ────────
    vehicle_max_observations_per_track: int = 5
    vehicle_min_quality_threshold: float = 45.0
    vehicle_selector_max_k: int = 3
    vehicle_max_ocr_attempts_per_track: int = 3
    vehicle_min_consensus_observations: int = 2
    vehicle_min_agreement_ratio: float = 0.60
    vehicle_min_confidence_threshold: float = 0.70
    vehicle_single_obs_threshold: float = 0.70
    vehicle_stale_track_timeout_seconds: float = 5.0

    # ── Virtual Fence & Intrusion ───────────────────────────────
    fence_cooldown_seconds: float = 5.0

    # ── Suspicious Activity Analytics ───────────────────────────
    loitering_duration_seconds: float = 10.0
    loitering_distance_radius_px: float = 50.0
    sudden_acceleration_threshold_px: float = 85.0
    unattended_object_duration_seconds: float = 15.0
    unattended_object_proximity_px: float = 120.0

    # ── Night Movement Analytics ────────────────────────────────
    night_brightness_threshold: float = 50.0  # Mean grayscale luminance (0-255)
    night_movement_cooldown_seconds: float = 8.0

    # ── Event Engine ────────────────────────────────────────────
    event_deduplication_window_seconds: float = 3.0
    event_cooldown_seconds: float = 3.0

    # ── Model Directories & Local Weights ───────────────────────
    models_dir: str = field(
        default_factory=lambda: os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
    )
    detector_model_path: Optional[str] = None
    yunet_model_path: Optional[str] = None

    # ── Optional External Integration & Persistence ─────────────
    # All disabled by default so the package runs standalone in memory
    redis_enabled: bool = False
    redis_host: str = os.getenv("VALKEY_HOST", os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = int(os.getenv("VALKEY_PORT", os.getenv("REDIS_PORT", "6379")))
    redis_password: str = os.getenv("VALKEY_PASSWORD", os.getenv("REDIS_PASSWORD", ""))
    redis_channel: str = "cctv:alerts"

    db_enabled: bool = False
    db_connection_url: str = os.getenv("DATABASE_URL", "")

    storage_enabled: bool = False
    storage_dir: str = os.path.join(os.getcwd(), "cctv_snapshots")


default_config = IBVAPConfig()
