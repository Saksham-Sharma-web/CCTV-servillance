"""
License Plate Detector.
Applies morphological filtering, Sobel edge gradients, and contour analysis
to locate license plate regions within a detected vehicle bounding box.
"""

from typing import List, Tuple, Optional
import cv2
import numpy as np

from ..core.config import IBVAPConfig, default_config


class LicensePlateDetector:
    """
    Locates license plate candidates within vehicle crops.
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
        if vh < 30 or vw < 50:
            return []

        # Plates are usually in the lower 70% of the vehicle
        roi_y1 = int(vh * 0.25)
        roi = vehicle_bgr_crop[roi_y1:, :]
        roi_h, roi_w = roi.shape[:2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Bilateral filter to smooth noise while preserving plate edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)

        # 2. Sobel horizontal gradient to find vertical contrast edges (letters/numbers)
        grad_x = cv2.Sobel(blurred, ddepth=cv2.CV_16S, dx=1, dy=0, ksize=3)
        abs_grad_x = cv2.convertScaleAbs(grad_x)

        # 3. Morphological closing to join characters into a single plate band
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(abs_grad_x, cv2.MORPH_CLOSE, kernel)

        # 4. Otsu thresholding
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 5. Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates: List[Tuple[Tuple[int, int, int, int], np.ndarray]] = []

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h <= 0 or w <= 0:
                continue

            ar = float(w) / float(h)
            area = w * h
            area_ratio = area / float(roi_w * roi_h)

            # Valid plate aspect ratio and reasonable relative area
            if self.min_ar <= ar <= self.max_ar and 0.005 <= area_ratio <= 0.25:
                px1 = max(0, x - 2)
                py1 = max(0, roi_y1 + y - 2)
                px2 = min(vw, x + w + 2)
                py2 = min(vh, roi_y1 + y + h + 2)

                plate_crop = vehicle_bgr_crop[py1:py2, px1:px2]
                if plate_crop.size > 0:
                    candidates.append(((px1, py1, px2, py2), plate_crop))

        # Sort candidates by area descending (largest candidate first)
        candidates.sort(key=lambda c: (c[0][2] - c[0][0]) * (c[0][3] - c[0][1]), reverse=True)
        return candidates[:3]  # Return at most top 3 candidates
