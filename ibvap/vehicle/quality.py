"""
Plate Quality Scorer Module.
Evaluates the visual quality of candidate license plate crops to determine whether
they justify expensive downstream OCR operations.

Pure CPU-bound heuristic evaluation based on:
1. Sharpness (Laplacian variance)
2. Resolution / Dimension adequacy
3. Aspect Ratio geometry
4. Contrast (pixel dynamic range)
5. Luminance (exposure dynamic range)

All weights and thresholds are initial engineering defaults and require real-world validation.
"""

from typing import Optional, Dict, Any
import logging
import cv2
import numpy as np

from .types import PlateQualityReport, VehicleObservation

logger = logging.getLogger("ibvap.vehicle.quality")


class PlateQualityScorer:
    """
    Evaluates visual quality of a candidate license plate crop.
    Computes normalized component scores in range [0.0, 100.0] and returns a PlateQualityReport.
    """

    def __init__(
        self,
        weight_sharpness: float = 0.35,
        weight_resolution: float = 0.25,
        weight_aspect_ratio: float = 0.15,
        weight_contrast: float = 0.15,
        weight_luminance: float = 0.10,
        min_acceptable_score: float = 45.0,
        sharpness_variance_scale: float = 500.0,
        target_aspect_ratio: float = 3.2,
        optimal_width: float = 120.0,
        optimal_height: float = 36.0,
    ):
        """
        Initializes the scorer with configurable weights and target parameters.

        NOTE: All default weights and thresholds are INITIAL ENGINEERING DEFAULTS
        and REQUIRE REAL-WORLD CALIBRATION against live camera footage.
        """
        self.weight_sharpness = weight_sharpness
        self.weight_resolution = weight_resolution
        self.weight_aspect_ratio = weight_aspect_ratio
        self.weight_contrast = weight_contrast
        self.weight_luminance = weight_luminance
        self.min_acceptable_score = min_acceptable_score

        self.sharpness_variance_scale = sharpness_variance_scale
        self.target_aspect_ratio = target_aspect_ratio
        self.optimal_width = optimal_width
        self.optimal_height = optimal_height

    def score(self, plate_crop: Optional[np.ndarray]) -> PlateQualityReport:
        """
        Evaluates the quality of a candidate license plate BGR or grayscale image crop.

        Args:
            plate_crop: numpy.ndarray, shape (height, width, 3) or (height, width), uint8.

        Returns:
            PlateQualityReport containing component scores and acceptability flag.
        """
        # ── 1. Input Sanity & Failure Handling ───────────────────────
        if plate_crop is None:
            return PlateQualityReport(
                overall_score=0.0,
                sharpness_score=0.0,
                resolution_score=0.0,
                aspect_ratio_score=0.0,
                contrast_score=0.0,
                luminance_score=0.0,
                is_acceptable=False,
                details={"error": "crop_is_none"}
            )

        if not isinstance(plate_crop, np.ndarray) or plate_crop.size == 0:
            return PlateQualityReport(
                overall_score=0.0,
                sharpness_score=0.0,
                resolution_score=0.0,
                aspect_ratio_score=0.0,
                contrast_score=0.0,
                luminance_score=0.0,
                is_acceptable=False,
                details={"error": "empty_or_invalid_array"}
            )

        if plate_crop.ndim not in (2, 3):
            return PlateQualityReport(
                overall_score=0.0,
                sharpness_score=0.0,
                resolution_score=0.0,
                aspect_ratio_score=0.0,
                contrast_score=0.0,
                luminance_score=0.0,
                is_acceptable=False,
                details={"error": f"unsupported_ndim_{plate_crop.ndim}"}
            )

        if plate_crop.ndim == 3 and plate_crop.shape[2] not in (1, 3, 4):
            return PlateQualityReport(
                overall_score=0.0,
                sharpness_score=0.0,
                resolution_score=0.0,
                aspect_ratio_score=0.0,
                contrast_score=0.0,
                luminance_score=0.0,
                is_acceptable=False,
                details={"error": f"unsupported_channels_{plate_crop.shape[2]}"}
            )

        h, w = plate_crop.shape[:2]
        if h < 2 or w < 2:
            return PlateQualityReport(
                overall_score=0.0,
                sharpness_score=0.0,
                resolution_score=0.0,
                aspect_ratio_score=0.0,
                contrast_score=0.0,
                luminance_score=0.0,
                is_acceptable=False,
                details={"error": f"too_small_dimensions_{w}x{h}"}
            )

        # ── 2. Color Normalization (Read-Only) ───────────────────────
        if plate_crop.ndim == 3 and plate_crop.shape[2] == 3:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        elif plate_crop.ndim == 3 and plate_crop.shape[2] == 4:
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGRA2GRAY)
        elif plate_crop.ndim == 3 and plate_crop.shape[2] == 1:
            gray = plate_crop[:, :, 0]
        else:
            gray = plate_crop


        # ── 3. Component Metric 1: Sharpness (Laplacian Variance) ────
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        # Scale linearly with ceiling at saturation scale (e.g. 500.0)
        sharpness_score = min(100.0, max(0.0, (laplacian_var / max(1.0, self.sharpness_variance_scale)) * 100.0))

        # ── 4. Component Metric 2: Resolution / Size Adequacy ────────
        # Evaluates height and width independently against optimal dimensions (120x36)
        # Plates below 10px height or 30px width degrade severely
        h_ratio = min(1.0, max(0.0, (h - 8.0) / max(1.0, self.optimal_height - 8.0)))
        w_ratio = min(1.0, max(0.0, (w - 24.0) / max(1.0, self.optimal_width - 24.0)))
        resolution_score = min(100.0, max(0.0, (0.5 * h_ratio + 0.5 * w_ratio) * 100.0))

        # ── 5. Component Metric 3: Aspect Ratio Geometry ─────────────
        ar = float(w) / float(max(1, h))
        # Canonical plate aspect ratio ~ 3.2. Allow deviation within range [1.2, 5.5]
        ar_diff = abs(ar - self.target_aspect_ratio)
        aspect_ratio_score = min(100.0, max(0.0, (1.0 - (ar_diff / 2.5)) * 100.0))

        # ── 6. Component Metric 4: Contrast (Standard Deviation) ──────
        contrast_std = float(np.std(gray))
        # Ideal high-contrast plate has std >= 45.0
        contrast_score = min(100.0, max(0.0, (contrast_std / 45.0) * 100.0))

        # ── 7. Component Metric 5: Luminance / Exposure Balance ──────
        mean_lum = float(np.mean(gray))
        # Ideal luminance between 60 and 200.
        # Below 60 (dark / night without IR): gentle penalty with floor at 20.0
        # Above 200 (overexposure / headlight glare): gentle penalty with floor at 20.0
        if 60.0 <= mean_lum <= 200.0:
            luminance_score = 100.0
        elif mean_lum < 60.0:
            luminance_score = max(0.0, 20.0 + (mean_lum / 60.0) * 80.0)
        else:
            luminance_score = max(0.0, 20.0 + ((255.0 - mean_lum) / 55.0) * 80.0)

        # ── 8. Overall Weighted Score ────────────────────────────────
        weight_sum = (
            self.weight_sharpness
            + self.weight_resolution
            + self.weight_aspect_ratio
            + self.weight_contrast
            + self.weight_luminance
        )
        norm_factor = 1.0 / weight_sum if weight_sum > 0 else 1.0

        overall_score = (
            self.weight_sharpness * sharpness_score
            + self.weight_resolution * resolution_score
            + self.weight_aspect_ratio * aspect_ratio_score
            + self.weight_contrast * contrast_score
            + self.weight_luminance * luminance_score
        ) * norm_factor

        # Bound check [0.0, 100.0]
        overall_score = min(100.0, max(0.0, overall_score))

        # ── 9. Acceptability Decision ────────────────────────────────
        is_acceptable = overall_score >= self.min_acceptable_score

        details: Dict[str, Any] = {
            "width": int(w),
            "height": int(h),
            "aspect_ratio": round(ar, 3),
            "laplacian_variance": round(laplacian_var, 2),
            "contrast_std": round(contrast_std, 2),
            "mean_luminance": round(mean_lum, 2),
        }

        return PlateQualityReport(
            overall_score=round(overall_score, 2),
            sharpness_score=round(sharpness_score, 2),
            resolution_score=round(resolution_score, 2),
            aspect_ratio_score=round(aspect_ratio_score, 2),
            contrast_score=round(contrast_score, 2),
            luminance_score=round(luminance_score, 2),
            is_acceptable=is_acceptable,
            details=details,
        )

    def score_observation(self, observation: VehicleObservation) -> PlateQualityReport:
        """
        Convenience method that scores a VehicleObservation, updates its quality field,
        and returns the resulting PlateQualityReport.
        """
        report = self.score(observation.plate_crop)
        observation.quality = report
        return report
