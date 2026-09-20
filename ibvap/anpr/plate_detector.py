"""
License Plate Detector.
Multi-strategy candidate extraction combining bright-rectangle scanning,
adaptive morphology, Sobel edge gradients, adaptive thresholding, and
candidate bumper ROIs to locate license plate regions within a detected
vehicle bounding box.
"""

from typing import List, Tuple, Optional, Dict
import logging
import cv2
import numpy as np

from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.anpr.plate_detector")


def _box_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / float(max(1, union))


class LicensePlateDetector:
    """
    Locates license plate candidates within vehicle crops using multi-strategy detection.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.min_ar = self.config.anpr_min_plate_aspect_ratio
        self.max_ar = self.config.anpr_max_plate_aspect_ratio
        # Configurable brightness threshold for bright-rectangle plate scan.
        # Indian license plates are typically white/bright rectangles (mean pixel > 160).
        self.bright_plate_threshold = getattr(
            self.config, 'anpr_bright_plate_threshold', 160
        )
        # Diagnostic info from last detection call (for debugging/benchmarking)
        self._last_detection_info: List[Dict] = []

    def detect_plates(self, vehicle_bgr_crop: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Locates candidate license plate regions inside a vehicle crop.

        Args:
            vehicle_bgr_crop: BGR numpy image of the vehicle.

        Returns:
            List of ((px1, py1, px2, py2), plate_crop) in coordinates relative to vehicle_bgr_crop.
        """
        self._last_detection_info = []

        if vehicle_bgr_crop is None or vehicle_bgr_crop.size == 0:
            return []

        vh, vw = vehicle_bgr_crop.shape[:2]
        if vh < 20 or vw < 30:
            return []

        # Plates are usually in the lower 75% of the vehicle (front or rear bumper)
        roi_y1 = int(vh * 0.20)
        roi = vehicle_bgr_crop[roi_y1:, :]
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        candidates_raw: List[Tuple[int, int, int, int]] = []
        source_tags: Dict[Tuple[int, int, int, int], str] = {}

        # ── Strategy 0: Bright-Rectangle Plate Scan ─────────────────
        # License plates are bright rectangles relative to vehicle bodies.
        # In night, rainy, or shadow conditions, plate luminance may be ~100-140
        # rather than 160+. We use an adaptive threshold guided by Otsu's bimodal
        # separation, bounded between [60, self.bright_plate_threshold].
        otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_thresh = min(self.bright_plate_threshold, max(60, int(otsu_val)))
        scan_thresholds = [adaptive_thresh]
        if self.bright_plate_threshold != adaptive_thresh and self.bright_plate_threshold <= 200:
            scan_thresholds.append(self.bright_plate_threshold)

        bk = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        for sthresh in scan_thresholds:
            _, bright_mask = cv2.threshold(gray, sthresh, 255, cv2.THRESH_BINARY)
            bright_closed = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, bk)
            contours_bright, _ = cv2.findContours(
                bright_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours_bright:
                x, y, w, h = cv2.boundingRect(cnt)
                if h < 10 or w < 25:
                    continue
                ar = float(w) / float(h)
                area = w * h
                area_ratio = area / float(roi_w * roi_h)

                if 1.5 <= ar <= 5.5 and 0.005 <= area_ratio <= 0.30:
                    pad_x = max(3, int(w * 0.08))
                    pad_y = max(2, int(h * 0.10))
                    px1 = max(0, x - pad_x)
                    py1 = max(0, roi_y1 + y - pad_y)
                    px2 = min(vw, x + w + pad_x)
                    py2 = min(vh, roi_y1 + y + h + pad_y)
                    box = (px1, py1, px2, py2)
                    candidates_raw.append(box)
                    if box not in source_tags:
                        source_tags[box] = "bright_rectangle"

        # ── Strategy 1: Adaptive Morphology & Sobel Edge Gradient ────────
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        grad_x = cv2.Sobel(blurred, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)

        # Dynamic kernel size adapting to vehicle crop resolution
        kw = max(3, min(25, int(vw * 0.07)))
        kh = max(2, int(kw / 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        closed = cv2.morphologyEx(abs_grad_x, cv2.MORPH_CLOSE, kernel)
        _, thresh1 = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours1:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0 or w <= 0:
                continue
            ar = float(w) / float(h)
            area = w * h
            area_ratio = area / float(roi_w * roi_h)

            if (self.min_ar * 0.75) <= ar <= (self.max_ar * 1.25) and 0.001 <= area_ratio <= 0.40:
                pad_x = max(2, int(w * 0.05))
                pad_y = max(2, int(h * 0.08))
                px1 = max(0, x - pad_x)
                py1 = max(0, roi_y1 + y - pad_y)
                px2 = min(vw, x + w + pad_x)
                py2 = min(vh, roi_y1 + y + h + pad_y)
                box = (px1, py1, px2, py2)
                candidates_raw.append(box)
                if box not in source_tags:
                    source_tags[box] = "sobel_morphology"

        # ── Strategy 2: Adaptive Thresholding for High-Contrast Plates ──
        thresh2 = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9
        )
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours2:
            x, y, w, h = cv2.boundingRect(cnt)
            if h < 6 or w < 16:
                continue
            ar = float(w) / float(h)
            area = w * h
            area_ratio = area / float(roi_w * roi_h)

            if (self.min_ar * 0.8) <= ar <= (self.max_ar * 1.2) and 0.001 <= area_ratio <= 0.35:
                pad_x = max(2, int(w * 0.05))
                pad_y = max(2, int(h * 0.08))
                px1 = max(0, x - pad_x)
                py1 = max(0, roi_y1 + y - pad_y)
                px2 = min(vw, x + w + pad_x)
                py2 = min(vh, roi_y1 + y + h + pad_y)
                box = (px1, py1, px2, py2)
                candidates_raw.append(box)
                if box not in source_tags:
                    source_tags[box] = "adaptive_threshold"

        # ── Strategy 3: Deduplicate overlapping candidate boxes ─────────
        unique_boxes: List[Tuple[int, int, int, int]] = []
        unique_sources: Dict[Tuple[int, int, int, int], str] = {}
        for box in candidates_raw:
            if not any(_box_iou(box, ub) > 0.45 for ub in unique_boxes):
                unique_boxes.append(box)
                unique_sources[box] = source_tags.get(box, "unknown")

        # ── Strategy 4: Candidate Bumper ROI Fallback ────────────────────
        # If morphology found fewer than 2 candidates, include canonical bumper ROIs
        # where plates reside, so downstream OCR text detection can scan them
        if len(unique_boxes) < 2:
            # Lower-center bumper region
            b1 = (int(vw * 0.15), int(vh * 0.45), int(vw * 0.85), int(vh * 0.95))
            # Lower-third bumper region
            b2 = (int(vw * 0.10), int(vh * 0.60), int(vw * 0.90), min(vh, int(vh * 0.98)))
            for fallback_box in (b1, b2):
                if not any(_box_iou(fallback_box, ub) > 0.50 for ub in unique_boxes):
                    unique_boxes.append(fallback_box)
                    unique_sources[fallback_box] = "bumper_fallback"

        # Build candidate output list
        candidates: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        for bx1, by1, bx2, by2 in unique_boxes:
            crop = vehicle_bgr_crop[by1:by2, bx1:bx2]
            if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 20:
                candidates.append(((bx1, by1, bx2, by2), crop))

        # Score candidates: multi-signal ranking combining geometry, brightness,
        # contrast, and edge density to prioritize real license plates over
        # headlights, reflections, and other false positives.
        # Lower score = better candidate.
        def candidate_score(item):
            box, crop = item
            bw = max(1, box[2] - box[0])
            bh = max(1, box[3] - box[1])
            ar = bw / float(bh)
            ar_diff = abs(ar - 3.2)
            area_ratio = (bw * bh) / float(max(1, vw * vh))

            # Analyze crop visual properties
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            brightness = float(np.mean(g))
            contrast = float(np.std(g))
            edges = cv2.Canny(g, 50, 150)
            edge_density = float(np.count_nonzero(edges)) / float(max(1, g.size))

            # ── Composite score (lower = better) ──
            score = ar_diff  # AR proximity to ideal plate (3.2)

            # Brightness: white/bright plates (85-220) are ideal.
            # In night/rain/shadow scenes, plate crops with bumper borders can have luminance ~85-130.
            # Dark candidates (headlight surrounds, deep shadows) get penalized.
            # Overexposed candidates (pure glare) get penalized.
            if 85.0 <= brightness <= 220.0:
                score -= 0.5  # Bonus for plate-like brightness
            elif brightness < 50.0:
                score += 1.0  # Heavy penalty for very dark candidates
            elif brightness < 85.0:
                score += 0.3  # Moderate penalty for dim candidates
            elif brightness > 230.0:
                score += 0.3  # Penalty for overexposed/washed out

            # Contrast: plates have dark text on light background.
            # Low-contrast regions (solid color, uniform glare) are not plates.
            if contrast >= 25.0:
                score -= 0.3  # Bonus for high contrast (text presence)
            elif contrast < 15.0:
                score += 0.8  # Heavy penalty for flat/featureless regions

            # Edge density: real plates have character-like internal edges.
            # Smooth regions (glare, reflections, solid body panels) get penalized.
            # This prevents bright headlights from outscoring real plates.
            if edge_density >= 0.08:
                score -= 0.4  # Strong bonus for text-like edge patterns
            elif edge_density < 0.03:
                score += 0.8  # Penalty for smooth regions

            # Size penalties
            if area_ratio > 0.35:
                score += 5.0  # Unreasonably large
            if bh < 25:
                score += 0.3  # Small noise candidate penalty (real readable plates are >= 25px high)
            if bw < 30:
                score += 0.3  # Very narrow candidate

            return score

        candidates.sort(key=candidate_score)
        top_candidates = candidates[:3]

        # Populate diagnostic info for debugging/benchmarking
        self._last_detection_info = []
        for box, crop in top_candidates:
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
            self._last_detection_info.append({
                "bbox": box,
                "source": unique_sources.get(box, "unknown"),
                "width": box[2] - box[0],
                "height": box[3] - box[1],
                "aspect_ratio": round(float(box[2] - box[0]) / max(1, box[3] - box[1]), 3),
                "brightness": round(float(np.mean(g)), 2),
                "contrast": round(float(np.std(g)), 2),
                "edge_density": round(
                    float(np.count_nonzero(cv2.Canny(g, 50, 150))) / max(1, g.size), 4
                ),
                "score": round(candidate_score((box, crop)), 4),
            })

        logger.debug(
            f"[PlateDetector] Located {len(top_candidates)} plate candidates: "
            f"{[c[0] for c in top_candidates]}"
        )
        return top_candidates


# Alias for modular architecture compatibility
PlateDetector = LicensePlateDetector
