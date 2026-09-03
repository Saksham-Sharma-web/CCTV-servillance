"""
ANPR OCR Adapter.
Reuses the existing PaddleOCR / OCREngine infrastructure from id-verification/verification/ocr/ocr_engine.py.
Handles plate text extraction, character normalization, and watchlist cross-referencing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import os
import sys
import re
import logging
import cv2
import numpy as np

from ..core.types import WatchlistCategory
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.anpr.ocr")


@dataclass
class PlateResult:
    plate_number: str
    confidence: float
    category: WatchlistCategory = WatchlistCategory.UNKNOWN
    raw_text: str = ""


class ANPRAdapter:
    """
    Adapter wrapping existing OCREngine from id-verification service.
    Normalizes plate alphanumeric strings and performs watchlist category lookup.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.ocr_engine = None
        self._initialized = False
        self.watchlist: Dict[str, WatchlistCategory] = {}

    def _ensure_ocr_engine(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            id_verif_path = os.path.join(repo_root, "id-verification")
            if id_verif_path not in sys.path:
                sys.path.insert(0, id_verif_path)

            from verification.ocr.ocr_engine import OCREngine
            self.ocr_engine = OCREngine()
            logger.info("Existing OCREngine successfully loaded into ANPRAdapter.")
        except Exception as e:
            logger.warning(f"Could not load existing OCREngine: {e}. OCR will operate in fallback mode.")
            self.ocr_engine = None

    def add_watchlist_entry(self, plate_number: str, category: WatchlistCategory):
        clean_plate = self.normalize_plate(plate_number)
        if clean_plate:
            self.watchlist[clean_plate] = category

    @staticmethod
    def normalize_plate(raw_text: str) -> str:
        """
        Cleans OCR text to standard alphanumeric uppercase license plate string.
        """
        if not raw_text:
            return ""
        cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()
        # Common OCR character confusions in plates
        # E.g., if plate has standard state codes like DL, UP, MH, HR, KA
        return cleaned

    def preprocess_plate_crop(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Preprocesses license plate crop for optimal OCR accuracy.
        """
        h, w = plate_crop.shape[:2]
        # Resize if too small (PaddleOCR works best with height >= 48px)
        if h < 64:
            scale = 64.0 / float(h)
            new_w = max(100, int(w * scale))
            plate_crop = cv2.resize(plate_crop, (new_w, 64), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    def recognize_plate(self, plate_crop: np.ndarray) -> Optional[PlateResult]:
        """
        Executes OCR on candidate plate crop and returns structured PlateResult.
        """
        self._ensure_ocr_engine()
        if plate_crop is None or plate_crop.size == 0 or self.ocr_engine is None:
            return None

        try:
            preprocessed = self.preprocess_plate_crop(plate_crop)
            res = self.ocr_engine.extract_text(preprocessed)

            raw_text = res.get("raw_text", "")
            conf = float(res.get("confidence", 0.0))

            clean_plate = self.normalize_plate(raw_text)

            # Valid license plate should typically be between 4 and 12 characters
            if len(clean_plate) >= 4:
                category = self.watchlist.get(clean_plate, WatchlistCategory.UNKNOWN)
                return PlateResult(
                    plate_number=clean_plate,
                    confidence=conf if conf > 0 else 0.85,
                    category=category,
                    raw_text=raw_text
                )
            return None
        except Exception as e:
            logger.error(f"Error during ANPR OCR recognition: {e}")
            return None
