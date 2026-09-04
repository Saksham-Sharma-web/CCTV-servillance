"""
ANPR OCR Adapter.
Self-contained, in-process optical character recognition engine for vehicle license plates.
Utilizes EasyOCR (PyTorch native) with multi-stage image preprocessing, text token aggregation,
alphanumeric normalization, and watchlist cross-referencing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import re
import logging
import cv2
import numpy as np
import torch

from ..core.types import WatchlistCategory
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.anpr.ocr")


@dataclass
class PlateResult:
    plate_number: str
    confidence: float
    ocr_confidence: float = 0.0
    category: WatchlistCategory = WatchlistCategory.UNKNOWN
    raw_text: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None


class ANPRAdapter:
    """
    Self-contained ANPR OCR engine.
    Extracts, normalizes, and validates license plate text from image crops.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.reader = None
        self._initialized = False
        self.watchlist: Dict[str, WatchlistCategory] = {}

    def _ensure_ocr_engine(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            import easyocr
            use_gpu = torch.cuda.is_available()
            logger.info(f"[ANPR] Initializing EasyOCR engine (GPU={use_gpu})...")
            self.reader = easyocr.Reader(["en"], gpu=use_gpu)
            logger.info("[ANPR] EasyOCR engine initialized successfully.")
        except Exception as e:
            logger.error(f"[ANPR] Failed to initialize EasyOCR engine: {e}")
            self.reader = None

    def add_watchlist_entry(self, plate_number: str, category: WatchlistCategory):
        clean_plate = self.normalize_plate(plate_number)
        if clean_plate:
            self.watchlist[clean_plate] = category

    @staticmethod
    def normalize_plate(raw_text: str) -> str:
        """
        Cleans OCR text to standard alphanumeric uppercase license plate string.
        Applies common OCR character confusion heuristics for license plates.
        """
        if not raw_text:
            return ""

        # Remove spaces, hyphens, dots, and special characters
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        if not cleaned:
            return ""

        # Common OCR corrections for state codes (e.g. DL, UP, MH, HR, KA, etc.)
        if len(cleaned) >= 4:
            chars = list(cleaned)
            # Leading letter slot confusion: '0' or 'O' -> 'D'
            if chars[0] in ('0', 'O') and chars[1] in ('L', 'P', 'H', 'R', 'A', 'K'):
                chars[0] = 'D'
            elif chars[0] == '1' and chars[1] in ('L', 'P'):
                chars[0] = 'D'
            # If leading D or U was slightly cut off (e.g. 'L01' -> 'DL01', 'P16' -> 'UP16')
            elif chars[0] == 'L' and chars[1].isdigit():
                chars.insert(0, 'D')
            elif chars[0] == 'P' and chars[1].isdigit():
                chars.insert(0, 'U')
            cleaned = "".join(chars)

        return cleaned

    def preprocess_plate_crop(self, plate_crop: np.ndarray) -> List[np.ndarray]:
        """
        Generates enhanced image variants to ensure readability across varying lighting,
        tilt, contrast, and resolution conditions.
        """
        h, w = plate_crop.shape[:2]
        variants: List[np.ndarray] = []

        # 1. Target resolution scaling (ensure height >= 64px for optimal OCR)
        target_h = max(64, min(160, int(h * 2.0))) if h < 64 else h
        scale = float(target_h) / float(max(1, h))
        target_w = max(120, int(w * scale))
        resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Add neutral border padding so characters touching boundaries are recognized cleanly
        padded = cv2.copyMakeBorder(resized, 10, 10, 16, 16, cv2.BORDER_REPLICATE)

        # 2. Variant A: Bilateral denoising + CLAHE contrast enhancement
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        variant_a = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        variants.append(variant_a)

        # 3. Variant B: Adaptive Otsu thresholding for low-contrast / shadow plates
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variant_b = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)
        variants.append(variant_b)

        # 4. Variant C: Inverted binary for white-on-dark or yellow-on-dark plates
        inv_otsu = cv2.bitwise_not(otsu)
        variant_c = cv2.cvtColor(inv_otsu, cv2.COLOR_GRAY2BGR)
        variants.append(variant_c)

        return variants

    def recognize_plate(self, plate_crop: np.ndarray) -> Optional[PlateResult]:
        """
        Executes OCR on candidate plate crop and returns structured PlateResult.
        """
        self._ensure_ocr_engine()
        if plate_crop is None or plate_crop.size == 0 or self.reader is None:
            return None

        h, w = plate_crop.shape[:2]
        logger.debug(f"[ANPR] OCR input dimensions: {w}x{h}")

        try:
            image_variants = self.preprocess_plate_crop(plate_crop)

            best_result: Optional[PlateResult] = None
            best_confidence = 0.0

            for v_idx, img_var in enumerate(image_variants):
                results = self.reader.readtext(img_var)
                if not results:
                    continue

                # Sort detected text boxes: top-to-bottom, then left-to-right
                # Box format: [ [tl, tr, br, bl], text, confidence ]
                def sort_key(item):
                    box_pts = item[0]
                    cy = (box_pts[0][1] + box_pts[2][1]) / 2.0
                    cx = (box_pts[0][0] + box_pts[2][0]) / 2.0
                    return (int(cy // 25), cx)

                sorted_tokens = sorted(results, key=sort_key)
                raw_texts = [token[1] for token in sorted_tokens]
                confidences = [float(token[2]) for token in sorted_tokens]

                combined_raw = " ".join(raw_texts)
                avg_conf = float(np.mean(confidences)) if confidences else 0.0

                clean_plate = self.normalize_plate(combined_raw)

                logger.debug(
                    f"[ANPR] Variant {v_idx} Raw OCR: '{combined_raw}' (conf: {round(avg_conf, 3)}) -> Normalized: '{clean_plate}'"
                )

                # Valid license plate standard format: 4 to 12 alphanumeric characters
                if 4 <= len(clean_plate) <= 12 and avg_conf > best_confidence:
                    category = self.watchlist.get(clean_plate, WatchlistCategory.UNKNOWN)
                    best_confidence = avg_conf
                    best_result = PlateResult(
                        plate_number=clean_plate,
                        confidence=round(avg_conf, 4),
                        ocr_confidence=round(avg_conf, 4),
                        category=category,
                        raw_text=combined_raw
                    )

                    # If high confidence match achieved, no need to test remaining variants
                    if avg_conf >= 0.70:
                        break

            if best_result:
                logger.info(
                    f"[ANPR] Plate Recognized: '{best_result.plate_number}' "
                    f"(Raw: '{best_result.raw_text}', Conf: {best_result.confidence}, Category: {best_result.category.value})"
                )
            return best_result

        except Exception as e:
            logger.error(f"[ANPR] Error during OCR recognition: {e}")
            return None
