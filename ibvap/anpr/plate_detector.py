"""
License Plate Detector.
Multi-strategy candidate extraction combining adaptive morphology, Sobel edge gradients,
adaptive thresholding, and candidate bumper ROIs to locate license plate regions
within a detected vehicle bounding box.
"""

from typing import List, Tuple, Optional
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

    def detect_plates(self, vehicle_bgr_crop: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], np.ndarray]]:
        """
        Locates candidate license plate regions inside a vehicle crop.

        Args:
            vehicle_bgr_crop: BGR numpy image of the vehicle.

        Returns:
            List of ((px1, py1, px2, py2), plate_crop) in coordinates relative to vehicle_bgr_crop.
        """
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

        # ── Strategy 1: Adaptive Morphology & Sobel Edge Gradient ────────
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
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
                candidates_raw.append((px1, py1, px2, py2))

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
                candidates_raw.append((px1, py1, px2, py2))

        # ── Strategy 3: Deduplicate overlapping candidate boxes ─────────
        unique_boxes: List[Tuple[int, int, int, int]] = []
        for box in candidates_raw:
            if not any(_box_iou(box, ub) > 0.45 for ub in unique_boxes):
                unique_boxes.append(box)

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

        # Build candidate output list
        candidates: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []
        for bx1, by1, bx2, by2 in unique_boxes:
            crop = vehicle_bgr_crop[by1:by2, bx1:bx2]
            if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 20:
                candidates.append(((bx1, by1, bx2, by2), crop))

        # Score candidates: prioritize boxes whose aspect ratio is close to standard plate (~3.2)
        def candidate_score(item):
            box = item[0]
            bw = max(1, box[2] - box[0])
            bh = max(1, box[3] - box[1])
            ar = bw / float(bh)
            ar_diff = abs(ar - 3.2)
            area_ratio = (bw * bh) / float(max(1, vw * vh))
            penalty = 5.0 if area_ratio > 0.35 else 0.0
            return ar_diff + penalty

        candidates.sort(key=candidate_score)
        top_candidates = candidates[:3]

        logger.debug(
            f"[PlateDetector] Located {len(top_candidates)} plate candidates: "
            f"{[c[0] for c in top_candidates]}"
        )
        return top_candidates
