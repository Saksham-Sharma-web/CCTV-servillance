"""
ANPR OCR Adapter.
Self-contained, in-process optical character recognition engine for vehicle license plates.
Utilizes PaddleOCR (PP-OCRv4) with multi-stage image preprocessing, text token aggregation,
alphanumeric normalization, and watchlist cross-referencing.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any
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


INDIAN_STATES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "CG", "DD", "DL", "DN", "GA", "GJ",
    "HR", "HP", "JH", "JK", "KA", "KL", "LA", "LD", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}


class ANPRAdapter:
    """
    Self-contained ANPR OCR engine using PaddleOCR PP-OCRv4.
    Extracts, normalizes, and validates license plate text from image crops.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.reader = None
        self._initialized = False
        self.actual_device: str = "cpu"
        self.watchlist: Dict[str, WatchlistCategory] = {}
        # Preload OCR engine at initialization to prevent first-frame latency spikes
        self._ensure_ocr_engine()

    def detect_runtime(self) -> Dict[str, Any]:
        """Detects PaddlePaddle runtime capabilities and device support."""
        from ..core.device import get_paddle_runtime_info
        req_dev = getattr(self.config, "device", "auto")
        return get_paddle_runtime_info(req_dev)

    def select_device(self, requested: Optional[str] = None) -> str:
        """Centralized device selection for OCR engine ('gpu:0' vs 'cpu')."""
        from ..core.device import get_paddle_device
        req = requested if requested is not None else getattr(self.config, "device", "auto")
        return get_paddle_device(req)

    def _ensure_ocr_engine(self):
        if self._initialized:
            return
        self._initialized = True

        from ..core.device import ensure_cpu_thread_health
        req_dev = getattr(self.config, "device", "auto")
        target_dev = self.select_device(req_dev)
        self.actual_device = "cpu"

        # Set Paddle environment flags for clean inference
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        logger.info(f"[ANPR] Initializing PaddleOCR recognition engine (en_PP-OCRv4_mobile_rec) requested='{req_dev}', target='{target_dev}'...")

        import paddlex
        model_name = getattr(self.config, "anpr_ocr_model", "en_PP-OCRv4_mobile_rec")

        # Attempt 1: Try CUDA GPU if selected
        if target_dev.startswith("gpu"):
            try:
                import paddle
                paddle.device.set_device(target_dev)
                self.reader = paddlex.create_model(model_name, device=target_dev)
                # Warm up model to prime runtime graphs on CUDA
                dummy = np.zeros((48, 120, 3), dtype=np.uint8)
                if hasattr(self.reader, "predict"):
                    _ = list(self.reader.predict([dummy]))
                if hasattr(paddle.device, "synchronize"):
                    paddle.device.synchronize()
                elif hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "synchronize"):
                    paddle.device.cuda.synchronize()
                self.actual_device = target_dev
                ensure_cpu_thread_health()
                logger.info(f"[ANPR] PaddleOCR engine successfully initialized and warmed up on CUDA ({target_dev}).")
                return
            except Exception as e_gpu:
                logger.warning(f"[ANPR] Failed to initialize PaddleOCR on CUDA ({target_dev}): {e_gpu}. Safely falling back to CPU...")
                target_dev = "cpu"
                try:
                    import paddle
                    paddle.device.set_device("cpu")
                except Exception:
                    pass

        # Attempt 2: CPU execution (primary for CPU mode or fallback from CUDA)
        try:
            self.reader = paddlex.create_model(model_name, device="cpu")
            dummy = np.zeros((48, 120, 3), dtype=np.uint8)
            if hasattr(self.reader, "predict"):
                _ = list(self.reader.predict([dummy]))
            self.actual_device = "cpu"
            ensure_cpu_thread_health()
            logger.info(f"[ANPR] PaddleOCR engine initialized and warmed up with '{model_name}' on CPU.")
            return
        except Exception as e1:
            logger.warning(f"[ANPR] Could not load '{model_name}' on CPU: {e1}. Trying fallback model...")

        # Attempt 3: Fallback model PP-OCRv4_mobile_rec on CPU
        try:
            self.reader = paddlex.create_model("PP-OCRv4_mobile_rec", device="cpu")
            dummy = np.zeros((48, 120, 3), dtype=np.uint8)
            if hasattr(self.reader, "predict"):
                _ = list(self.reader.predict([dummy]))
            self.actual_device = "cpu"
            ensure_cpu_thread_health()
            logger.info("[ANPR] PaddleOCR fallback model 'PP-OCRv4_mobile_rec' initialized successfully on CPU.")
            return
        except Exception as e2:
            pass

        # Attempt 4: Legacy PaddleOCR package
        try:
            from paddleocr import PaddleOCR
            self.reader = PaddleOCR(use_angle_cls=False, lang="en")
            dummy = np.zeros((48, 120, 3), dtype=np.uint8)
            if hasattr(self.reader, "ocr"):
                _ = self.reader.ocr(dummy, det=False, rec=True)
            self.actual_device = "cpu"
            ensure_cpu_thread_health()
            logger.info("[ANPR] Legacy PaddleOCR engine initialized successfully on CPU.")
        except Exception as e3:
            logger.error(f"[ANPR] Failed to initialize PaddleOCR engine: {e1} | {e2} | {e3}")
            self.reader = None
            self.actual_device = "none"


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

        # Handle HSRP 'IND' emblem or stray edge prefix artifacts
        if cleaned.startswith("IND") and len(cleaned) > 5:
            cleaned = cleaned[3:]
        elif len(cleaned) >= 6 and cleaned[:2] not in INDIAN_STATES and cleaned[1:3] in INDIAN_STATES:
            cleaned = cleaned[1:]

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

    def _predict_single_variant(self, image: np.ndarray) -> Tuple[str, float, str]:
        """Runs OCR on a single image variant and returns (clean_plate, rec_score, raw_text)."""
        raw_text = ""
        rec_score = 0.0
        if hasattr(self.reader, "predict"):
            preds = list(self.reader.predict([image]))
            if getattr(self, "actual_device", "cpu").startswith("gpu"):
                try:
                    import paddle
                    if hasattr(paddle.device, "synchronize"):
                        paddle.device.synchronize()
                    elif hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "synchronize"):
                        paddle.device.cuda.synchronize()
                except Exception:
                    pass
            if preds:
                raw_text = preds[0].get("rec_text", "")
                rec_score = float(preds[0].get("rec_score", 0.0))
        elif hasattr(self.reader, "ocr"):
            res = self.reader.ocr(image, det=False, rec=True)
            if getattr(self, "actual_device", "cpu").startswith("gpu"):
                try:
                    import paddle
                    if hasattr(paddle.device, "synchronize"):
                        paddle.device.synchronize()
                    elif hasattr(paddle.device, "cuda") and hasattr(paddle.device.cuda, "synchronize"):
                        paddle.device.cuda.synchronize()
                except Exception:
                    pass
            if res:
                items = res[0] if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list) else res
                for item in items:
                    if isinstance(item, (tuple, list)) and len(item) >= 2:
                        raw_text, rec_score = str(item[0]), float(item[1])
                        break
        clean_plate = self.normalize_plate(raw_text)
        return clean_plate, rec_score, raw_text


    STANDARD_INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")
    BHARAT_SERIES_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")

    @classmethod
    def validate_format(cls, plate_text: str) -> bool:
        """
        Validates whether plate text structurally matches legitimate Indian registration formats.
        Used as a validation signal for early exit.
        """
        if not plate_text:
            return False
        if len(plate_text) >= 2 and plate_text[:2] in INDIAN_STATES:
            if cls.STANDARD_INDIAN_PLATE_PATTERN.match(plate_text):
                return True
        if cls.BHARAT_SERIES_PATTERN.match(plate_text):
            return True
        return False

    def is_sufficient(self, clean_plate: str, score: float) -> bool:
        """
        Determines if an OCR candidate result is sufficiently confident and valid to exit early.
        Uses project validation rules and configurable thresholds:
        - Must be valid plate length (3 to 12 chars).
        - If format matches standard Indian registration and score >= vehicle_min_confidence_threshold (0.70) -> sufficient.
        - If score >= vehicle_single_obs_threshold (0.70) and 4 <= len <= 10 -> sufficient.
        - If score >= 0.85 -> sufficient.
        """
        if not clean_plate or len(clean_plate) < 3 or len(clean_plate) > 12:
            return False
        min_conf = getattr(self.config, "vehicle_min_confidence_threshold", 0.70)
        if self.validate_format(clean_plate) and score >= min_conf:
            return True
        single_obs_conf = getattr(self.config, "vehicle_single_obs_threshold", 0.70)
        if score >= single_obs_conf and 4 <= len(clean_plate) <= 10:
            return True
        if score >= 0.85:
            return True
        return False

    def recognize_plate(self, plate_crop: np.ndarray) -> Optional[PlateResult]:
        """
        Executes PaddleOCR on candidate plate crop with early-exit pipeline:
        1. Natural padded crop -> OCR
        2. If is_sufficient (confidence & format valid) -> Early exit immediately!
        3. Only if insufficient -> Enhanced crop (unsharp mask) -> OCR
        4. If still insufficient -> Final fallback (bilateral + CLAHE) -> OCR
        """
        self._ensure_ocr_engine()
        if plate_crop is None or plate_crop.size == 0 or self.reader is None:
            return None

        h, w = plate_crop.shape[:2]
        logger.debug(f"[ANPR] OCR input dimensions: {w}x{h}")

        try:
            # 1. Target resolution scaling & natural padded crop (Variant 1)
            target_h = 48 if h < 48 else (64 if h > 96 else h)
            scale = float(target_h) / float(max(1, h))
            target_w = max(96, int(w * scale))
            resized = cv2.resize(plate_crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            padded = cv2.copyMakeBorder(resized, 8, 8, 12, 12, cv2.BORDER_REPLICATE)

            # Pass 1: Natural padded crop
            clean_plate, rec_score, raw_text = self._predict_single_variant(padded)
            logger.debug(f"[ANPR] Variant 1 (Natural): '{raw_text}' (score: {round(rec_score, 3)}) -> Normalized: '{clean_plate}'")

            best_plate = clean_plate
            best_score = rec_score
            best_raw = raw_text

            # EARLY EXIT: If natural crop yields acceptable confidence/validation, return immediately
            if self.is_sufficient(clean_plate, rec_score):
                cat = self.watchlist.get(clean_plate, WatchlistCategory.UNKNOWN)
                return PlateResult(
                    plate_number=clean_plate,
                    confidence=round(rec_score, 4),
                    ocr_confidence=round(rec_score, 4),
                    category=cat,
                    raw_text=raw_text
                )

            # Pass 2 (Enhanced Retry 1): Unsharp mask
            blurred = cv2.GaussianBlur(padded, (0, 0), 1.5)
            sharpened = cv2.addWeighted(padded, 1.4, blurred, -0.4, 0)
            clean_2, score_2, raw_2 = self._predict_single_variant(sharpened)
            logger.debug(f"[ANPR] Variant 2 (Unsharp): '{raw_2}' (score: {round(score_2, 3)}) -> Normalized: '{clean_2}'")

            if 3 <= len(clean_2) <= 12 and score_2 > best_score:
                best_plate = clean_2
                best_score = score_2
                best_raw = raw_2

            if self.is_sufficient(best_plate, best_score):
                cat = self.watchlist.get(best_plate, WatchlistCategory.UNKNOWN)
                return PlateResult(
                    plate_number=best_plate,
                    confidence=round(best_score, 4),
                    ocr_confidence=round(best_score, 4),
                    category=cat,
                    raw_text=best_raw
                )

            # Pass 3 (Final Fallback): Bilateral + CLAHE
            gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
            denoised = cv2.bilateralFilter(gray, 7, 50, 50)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
            enhanced = clahe.apply(denoised)
            variant_c = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            clean_3, score_3, raw_3 = self._predict_single_variant(variant_c)
            logger.debug(f"[ANPR] Variant 3 (CLAHE): '{raw_3}' (score: {round(score_3, 3)}) -> Normalized: '{clean_3}'")

            if 3 <= len(clean_3) <= 12 and score_3 > best_score:
                best_plate = clean_3
                best_score = score_3
                best_raw = raw_3

            if 3 <= len(best_plate) <= 12 and best_score > 0.0:
                cat = self.watchlist.get(best_plate, WatchlistCategory.UNKNOWN)
                return PlateResult(
                    plate_number=best_plate,
                    confidence=round(best_score, 4),
                    ocr_confidence=round(best_score, 4),
                    category=cat,
                    raw_text=best_raw
                )

            return None

        except Exception as e:
            logger.error(f"[ANPR] Error during PaddleOCR recognition: {e}")
            return None


# Alias for modular architecture compatibility
OCRAdapter = ANPRAdapter
