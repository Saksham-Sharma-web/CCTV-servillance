"""
ANPR OCR Adapter.
Self-contained, in-process optical character recognition engine for vehicle license plates.
Utilizes PaddleOCR (PP-OCRv4) with multi-stage image preprocessing, text token aggregation,
alphanumeric normalization, and watchlist cross-referencing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
import re
import logging
import cv2
import numpy as np

# PyTorch must be imported prior to Paddle on Windows to prevent DLL collisions (shm.dll)
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
    Self-contained ANPR OCR engine using PaddleOCR PP-OCRv4.
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
            # Set Paddle environment flags for clean inference
            os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
            logger.info("[ANPR] Initializing PaddleOCR recognition engine (en_PP-OCRv4_mobile_rec)...")
            import paddlex
            model_name = getattr(self.config, "anpr_ocr_model", "en_PP-OCRv4_mobile_rec")
            self.reader = paddlex.create_model(model_name)
            logger.info(f"[ANPR] PaddleOCR engine initialized successfully with '{model_name}'.")
        except Exception as e1:
            logger.warning(f"[ANPR] Could not load '{model_name}' via paddlex: {e1}. Trying fallback...")
            try:
                import paddlex
                self.reader = paddlex.create_model("PP-OCRv4_mobile_rec")
                logger.info("[ANPR] PaddleOCR fallback model 'PP-OCRv4_mobile_rec' initialized successfully.")
            except Exception as e2:
                try:
                    from paddleocr import PaddleOCR
                    self.reader = PaddleOCR(use_angle_cls=False, lang="en")
                    logger.info("[ANPR] Legacy PaddleOCR engine initialized successfully.")
                except Exception as e3:
                    logger.error(f"[ANPR] Failed to initialize PaddleOCR engine: {e1} | {e2} | {e3}")
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

        chars = list(cleaned)
        # Common OCR corrections for state codes (e.g. DL, UP, MH, HR, KA, etc.)
        if len(chars) >= 4:
            # Leading letter slot confusion: '0' or 'O' -> 'D'
            if chars[0] in ('0', 'O') and chars[1] in ('L', 'P', 'H', 'R', 'A', 'K', 'J', 'M'):
                chars[0] = 'D'
            elif chars[0] == '1' and chars[1] in ('L', 'P'):
                chars[0] = 'D'
            # If leading D or U was slightly cut off (e.g. 'L01' -> 'DL01', 'P16' -> 'UP16')
            elif chars[0] == 'L' and chars[1].isdigit():
                chars.insert(0, 'D')
            elif chars[0] == 'P' and chars[1].isdigit():
                chars.insert(0, 'U')

            # Digits slot confusion in standard plates (indices 2, 3 following 2 state letters)
            if len(chars) >= 4 and chars[0].isalpha() and chars[1].isalpha():
                for idx in (2, 3):
                    if idx < len(chars):
                        if chars[idx] in ('O', 'Q'):
                            chars[idx] = '0'
                        elif chars[idx] in ('I', 'L'):
                            chars[idx] = '1'
                        elif chars[idx] == 'Z':
                            chars[idx] = '2'
                        elif chars[idx] == 'S':
                            chars[idx] = '5'
                        elif chars[idx] == 'B':
                            chars[idx] = '8'

            # Trailing digits slot confusion (last 4 characters in 8-10 char plate)
            if len(chars) >= 8:
                for idx in range(len(chars) - 4, len(chars)):
                    if chars[idx] in ('O', 'Q'):
                        chars[idx] = '0'
                    elif chars[idx] in ('I', 'L'):
                        chars[idx] = '1'
                    elif chars[idx] == 'Z':
                        chars[idx] = '2'
                    elif chars[idx] == 'S':
                        chars[idx] = '5'
                    elif chars[idx] == 'B':
                        chars[idx] = '8'

        return "".join(chars)

    def preprocess_plate_crop(self, plate_crop: np.ndarray) -> List[np.ndarray]:
        """
        Generates enhanced image variants to ensure readability across varying lighting,
        tilt, contrast, and resolution conditions.
        """
        h, w = plate_crop.shape[:2]
        variants: List[np.ndarray] = []

        # Target resolution scaling (PP-OCR models perform optimally with ~48-64px height)
        target_h = 48 if h < 48 else (64 if h > 96 else h)
        scale = float(target_h) / float(max(1, h))
        target_w = max(96, int(w * scale))
        resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        # Add neutral border padding so characters touching boundaries are recognized cleanly
        padded = cv2.copyMakeBorder(resized, 8, 8, 12, 12, cv2.BORDER_REPLICATE)

        # Variant 1 (Primary): Natural padded BGR image
        variants.append(padded)

        # Variant 2: Mild unsharp masking (enhances stroke edges for slightly blurred plates)
        blurred = cv2.GaussianBlur(padded, (0, 0), 1.5)
        sharpened = cv2.addWeighted(padded, 1.4, blurred, -0.4, 0)
        variants.append(sharpened)

        # Variant 3: Bilateral denoising + CLAHE contrast enhancement (optimal for night/shadow plates)
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        denoised = cv2.bilateralFilter(gray, 7, 50, 50)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        variant_c = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        variants.append(variant_c)

        return variants

    def recognize_plate(self, plate_crop: np.ndarray) -> Optional[PlateResult]:
        """
        Executes PaddleOCR on candidate plate crop and returns structured PlateResult.
        """
        self._ensure_ocr_engine()
        if plate_crop is None or plate_crop.size == 0 or self.reader is None:
            return None

        h, w = plate_crop.shape[:2]
        logger.debug(f"[ANPR] OCR input dimensions: {w}x{h}")

        try:
            image_variants = self.preprocess_plate_crop(plate_crop)
            if not image_variants:
                return None

            best_result: Optional[PlateResult] = None
            best_confidence = 0.0

            # Batch predict variants using PaddleX recognition model
            if hasattr(self.reader, "predict"):
                predictions = list(self.reader.predict(image_variants))
                for v_idx, pred in enumerate(predictions):
                    raw_text = pred.get("rec_text", "")
                    rec_score = float(pred.get("rec_score", 0.0))
                    clean_plate = self.normalize_plate(raw_text)

                    logger.debug(
                        f"[ANPR] Variant {v_idx} PaddleOCR: '{raw_text}' (score: {round(rec_score, 3)}) -> Normalized: '{clean_plate}'"
                    )

                    # Valid license plate format: 3 to 12 alphanumeric characters
                    if 3 <= len(clean_plate) <= 12 and rec_score > best_confidence:
                        category = self.watchlist.get(clean_plate, WatchlistCategory.UNKNOWN)
                        best_confidence = rec_score
                        best_result = PlateResult(
                            plate_number=clean_plate,
                            confidence=round(rec_score, 4),
                            ocr_confidence=round(rec_score, 4),
                            category=category,
                            raw_text=raw_text
                        )
                        if rec_score >= 0.85:
                            break

            elif hasattr(self.reader, "ocr"):
                # Fallback for legacy PaddleOCR
                for v_idx, img_var in enumerate(image_variants):
                    res = self.reader.ocr(img_var, det=False, rec=True)
                    if not res:
                        continue
                    items = res[0] if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list) else res
                    for item in items:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            raw_text, rec_score = str(item[0]), float(item[1])
                            clean_plate = self.normalize_plate(raw_text)
                            if 3 <= len(clean_plate) <= 12 and rec_score > best_confidence:
                                category = self.watchlist.get(clean_plate, WatchlistCategory.UNKNOWN)
                                best_confidence = rec_score
                                best_result = PlateResult(
                                    plate_number=clean_plate,
                                    confidence=round(rec_score, 4),
                                    ocr_confidence=round(rec_score, 4),
                                    category=category,
                                    raw_text=raw_text
                                )
                                if rec_score >= 0.85:
                                    break

            if best_result:
                logger.info(
                    f"[ANPR] Plate Recognized: '{best_result.plate_number}' "
                    f"(Raw: '{best_result.raw_text}', Conf: {best_result.confidence}, Category: {best_result.category.value})"
                )
            return best_result

        except Exception as e:
            logger.error(f"[ANPR] Error during PaddleOCR recognition: {e}")
            return None
