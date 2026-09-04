"""
High-Accuracy Face Detection & Quality Validation Module.
Uses OpenCV YuNet with 5-point facial landmark extraction, adaptive multi-scale
resolution handling, quality assessment (blur, brightness, size), and configurable fallbacks.
Preserves backward-compatible (x1, y1, x2, y2, confidence) output contract.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import os
import logging
import cv2
import numpy as np

from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.face.detector")


@dataclass
class FaceDetection:
    """
    Rich face detection result containing coordinates, confidence,
    5 facial landmarks, detector origin, and quality status.
    Implements 5-element sequence indexing/iteration for 100% backward
    compatibility with legacy (x1, y1, x2, y2, confidence) callers.
    """
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[np.ndarray] = None  # (5, 2) array: [re, le, nt, rcm, lcm]
    detector: str = "yunet"
    quality_status: str = "GOOD_FACE"  # "GOOD_FACE", "LOW_QUALITY_FACE", "NO_FACE"
    quality_metrics: Dict[str, Any] = field(default_factory=dict)

    # ── Backward Compatibility with legacy tuple unpacking ─────────────
    def __iter__(self):
        return iter((self.box[0], self.box[1], self.box[2], self.box[3], float(self.confidence)))

    def __getitem__(self, idx):
        return (self.box[0], self.box[1], self.box[2], self.box[3], float(self.confidence))[idx]

    def __len__(self):
        return 5

    @property
    def width(self) -> int:
        return max(0, self.box[2] - self.box[0])

    @property
    def height(self) -> int:
        return max(0, self.box[3] - self.box[1])

    @property
    def photo_region(self) -> List[float]:
        """Returns [x, y, w, h] format for ID verification contract."""
        return [self.box[0], self.box[1], self.width, self.height]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": list(self.box),
            "confidence": round(float(self.confidence), 4),
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "detector": self.detector,
            "quality_status": self.quality_status,
            "quality_metrics": self.quality_metrics,
            "width": self.width,
            "height": self.height,
        }


def _download_yunet_model(target_path: str) -> bool:
    """
    Downloads the official OpenCV Zoo YuNet ONNX model (~232 KB)
    from GitHub LFS media storage if not already present.
    """
    import urllib.request
    url = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        temp_path = target_path + ".tmp"
        req = urllib.request.Request(url, headers={"User-Agent": "IBVAP/1.0 (OpenCV FaceDetectorYN)"})
        with urllib.request.urlopen(req, timeout=15) as response, open(temp_path, "wb") as out_file:
            data = response.read()
            if len(data) < 200_000:
                logger.warning(f"Downloaded YuNet file too small ({len(data)} bytes), likely not binary.")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
            out_file.write(data)
        if os.path.exists(temp_path):
            os.replace(temp_path, target_path)
            logger.info(f"YuNet model successfully downloaded to {target_path} ({len(data)} bytes).")
            return True
    except Exception as e:
        logger.warning(f"Could not auto-download YuNet model: {e}")
        if os.path.exists(target_path + ".tmp"):
            try:
                os.remove(target_path + ".tmp")
            except Exception:
                pass
    return False


class OpenCVFaceDetector:
    """
    Detects faces in BGR images using OpenCV YuNet (with 5-point landmarks)
    or OpenCV Haar cascades as a secondary fallback.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, yunet_model_path: Optional[str] = None):
        self.config = config or default_config
        self.confidence_threshold = self.config.face_detection_confidence
        self.yunet = None
        self.haar_cascade = None
        self.active_detector_type = "none"

        # Resolve YuNet model path
        models_dir = getattr(
            self.config,
            "models_dir",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        )
        default_model_path = os.path.join(models_dir, "face_detection_yunet_2023mar.onnx")
        candidate_paths = [
            yunet_model_path,
            getattr(self.config, "yunet_model_path", None),
            default_model_path,
            os.path.join(os.getcwd(), "ibvap", "models", "face_detection_yunet_2023mar.onnx"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "face_detection_yunet_2023mar.onnx")
        ]
        # Check for valid candidate path with proper binary size (>200 KB)
        resolved_path = next((p for p in candidate_paths if p and os.path.exists(p) and os.path.getsize(p) > 200_000), None)

        # If model is not found locally, auto-download from OpenCV Zoo
        if not resolved_path:
            logger.info("YuNet ONNX model not found locally. Initiating auto-download...")
            if _download_yunet_model(default_model_path):
                resolved_path = default_model_path

        if resolved_path:
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model=resolved_path,
                    config="",
                    input_size=(320, 320),
                    score_threshold=self.confidence_threshold,
                    nms_threshold=0.3,
                    top_k=5000,
                )
                self.active_detector_type = "yunet"
                logger.info(f"OpenCV YuNet face detector loaded from {resolved_path}")
            except Exception as e:
                logger.error(f"Could not load YuNet model from {resolved_path}: {e}")

        if self.yunet is None:
            if getattr(self.config, "require_high_accuracy", False):
                logger.critical("YuNet model missing and require_high_accuracy is True. Haar fallback disabled.")
                self.active_detector_type = "NO_FACE_DETECTOR_AVAILABLE"
            elif hasattr(cv2, "CascadeClassifier"):
                try:
                    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    self.haar_cascade = cv2.CascadeClassifier(cascade_path)
                    self.active_detector_type = "haar"
                    logger.warning(
                        "YuNet model missing. OpenCV Haar cascade initialized as development fallback. "
                        "Note: Haar detections do not provide 5-point landmarks."
                    )
                except Exception as e:
                    logger.error(f"Could not load Haar cascade: {e}")
                    self.active_detector_type = "NO_FACE_DETECTOR_AVAILABLE"

    def validate_face_quality(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int],
        landmarks: Optional[np.ndarray] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Validates the geometric and visual quality of a face crop candidate.
        Distinguishes: GOOD_FACE, LOW_QUALITY_FACE, NO_FACE.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box
        fw, fh = x2 - x1, y2 - y1

        # 1. Geometry checks
        if fw <= 0 or fh <= 0 or x1 >= w or y1 >= h or x2 <= 0 or y2 <= 0:
            return "NO_FACE", {"reason": "invalid_coordinates", "box": box}

        # 2. Minimum dimension check
        min_w = getattr(self.config, "face_min_width", 24)
        min_h = getattr(self.config, "face_min_height", 24)
        if fw < min_w or fh < min_h:
            return "LOW_QUALITY_FACE", {
                "reason": "face_too_small",
                "width": fw,
                "height": fh,
                "min_width": min_w,
                "min_height": min_h
            }

        # 3. Crop extraction & visual checks
        crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        if crop.size == 0:
            return "NO_FACE", {"reason": "empty_crop"}

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # 4. Blur check via Laplacian variance
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_thresh = getattr(self.config, "face_blur_threshold", 30.0)
        if blur_score < blur_thresh:
            return "LOW_QUALITY_FACE", {
                "reason": "blurred",
                "blur_score": round(blur_score, 2),
                "threshold": blur_thresh
            }

        # 5. Brightness / exposure check
        mean_brightness = float(np.mean(gray))
        min_bright = getattr(self.config, "face_min_brightness", 25.0)
        max_bright = getattr(self.config, "face_max_brightness", 240.0)
        if mean_brightness < min_bright:
            return "LOW_QUALITY_FACE", {
                "reason": "too_dark",
                "brightness": round(mean_brightness, 2),
                "min_brightness": min_bright
            }
        if mean_brightness > max_bright:
            return "LOW_QUALITY_FACE", {
                "reason": "overexposed",
                "brightness": round(mean_brightness, 2),
                "max_brightness": max_bright
            }

        return "GOOD_FACE", {
            "blur_score": round(blur_score, 2),
            "brightness": round(mean_brightness, 2),
            "width": fw,
            "height": fh
        }

    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detects faces in BGR image using YuNet with 5-point facial landmarks
        and adaptive multi-scale sizing. Returns rich FaceDetection objects.
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        detections: List[FaceDetection] = []

        # ── Strategy 1: YuNet Inference ──────────────────────────────
        if self.yunet is not None:
            try:
                # YuNet performs best when max image dimension is around 640px
                target_max_dim = 640.0
                max_dim = max(h, w)
                if max_dim > target_max_dim:
                    scale = target_max_dim / float(max_dim)
                    nw, nh = int(round(w * scale)), int(round(h * scale))
                    resized_img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
                else:
                    scale = 1.0
                    nw, nh = w, h
                    resized_img = image

                self.yunet.setInputSize((nw, nh))
                _, raw_faces = self.yunet.detect(resized_img)

                # Fallback: if no face found at 640px, retry at higher resolution (up to 1024px)
                if (raw_faces is None or len(raw_faces) == 0) and max_dim > target_max_dim:
                    scale2 = min(1.0, 1024.0 / float(max_dim))
                    nw2, nh2 = int(round(w * scale2)), int(round(h * scale2))
                    if (nw2, nh2) != (nw, nh):
                        resized_img2 = cv2.resize(image, (nw2, nh2), interpolation=cv2.INTER_AREA) if scale2 < 1.0 else image
                        self.yunet.setInputSize((nw2, nh2))
                        _, raw_faces2 = self.yunet.detect(resized_img2)
                        if raw_faces2 is not None and len(raw_faces2) > 0:
                            raw_faces = raw_faces2
                            scale = scale2

                # Fallback to tiled detection for small faces if enabled and no faces found on downscaled
                if (raw_faces is None or len(raw_faces) == 0) and getattr(self.config, "face_enable_tiling", False) and max_dim > target_max_dim:
                    raw_faces = self._detect_tiled(image)
                    scale = 1.0  # Tiled results are already in original coordinate space

                if raw_faces is not None:
                    for f in raw_faces:
                        # Raw coordinates on resized image
                        rx, ry, rfw, rfh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                        conf = float(f[14])

                        if conf < self.confidence_threshold:
                            continue

                        import math
                        if math.isinf(rx) or math.isnan(rx) or math.isinf(rfw) or math.isnan(rfw) or math.isinf(ry) or math.isnan(ry) or math.isinf(rfh) or math.isnan(rfh):
                            continue
                        
                        # Map back to original image coordinates
                        inv_scale = 1.0 / scale
                        x1 = max(0, min(w - 1, int(round(rx * inv_scale))))
                        y1 = max(0, min(h - 1, int(round(ry * inv_scale))))
                        x2 = max(x1 + 1, min(w, int(round((rx + rfw) * inv_scale))))
                        y2 = max(y1 + 1, min(h, int(round((ry + rfh) * inv_scale))))
                        box = (x1, y1, x2, y2)

                        # Extract 5 landmarks: [re, le, nt, rcm, lcm]
                        raw_lm = f[4:14].reshape((5, 2)) * inv_scale
                        # Clamp landmarks inside image
                        raw_lm[:, 0] = np.clip(raw_lm[:, 0], 0, w - 1)
                        raw_lm[:, 1] = np.clip(raw_lm[:, 1], 0, h - 1)

                        quality_status, metrics = self.validate_face_quality(image, box, raw_lm)

                        detections.append(FaceDetection(
                            box=box,
                            confidence=conf,
                            landmarks=raw_lm,
                            detector="yunet",
                            quality_status=quality_status,
                            quality_metrics=metrics
                        ))
            except Exception as e:
                logger.warning(f"YuNet detection error: {e}")

        # ── Strategy 2: Haar Cascade Fallback ────────────────────────
        elif self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                rects = self.haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(getattr(self.config, "face_min_width", 24), getattr(self.config, "face_min_height", 24)),
                )
                if rects is None or len(rects) == 0:
                    # More permissive retry with smaller scale factor and lower minNeighbors
                    rects = self.haar_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.05,
                        minNeighbors=3,
                        minSize=(getattr(self.config, "face_min_width", 24), getattr(self.config, "face_min_height", 24)),
                    )
                for (rx, ry, rfw, rfh) in rects:
                    x1 = max(0, min(w - 1, int(rx)))
                    y1 = max(0, min(h - 1, int(ry)))
                    x2 = max(x1 + 1, min(w, int(rx + rfw)))
                    y2 = max(y1 + 1, min(h, int(ry + rfh)))
                    box = (x1, y1, x2, y2)

                    quality_status, metrics = self.validate_face_quality(image, box)
                    # Conservative estimated confidence for Haar (do NOT fabricate 0.85)
                    estimated_conf = 0.55

                    detections.append(FaceDetection(
                        box=box,
                        confidence=estimated_conf,
                        landmarks=None,
                        detector="haar",
                        quality_status=quality_status,
                        quality_metrics=metrics
                    ))
            except Exception as e:
                logger.error(f"Haar cascade detection error: {e}")

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _detect_tiled(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Tiled detection for finding small faces in high-resolution frames."""
        h, w = image.shape[:2]
        tile_size = getattr(self.config, "face_tile_size", 480)
        overlap = getattr(self.config, "face_tile_overlap", 0.2)
        step = int(tile_size * (1.0 - overlap))

        all_faces = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                tw = min(tile_size, w - x)
                th = min(tile_size, h - y)
                if tw < 64 or th < 64:
                    continue
                tile = image[y:y+th, x:x+tw]
                self.yunet.setInputSize((tw, th))
                _, tile_faces = self.yunet.detect(tile)
                if tile_faces is not None:
                    for tf in tile_faces:
                        # Shift tile coordinates back to original frame
                        tf_copy = tf.copy()
                        tf_copy[0] += x
                        tf_copy[1] += y
                        # Landmarks
                        for i in range(4, 14, 2):
                            tf_copy[i] += x
                            tf_copy[i+1] += y
                        all_faces.append(tf_copy)

        return np.array(all_faces) if all_faces else None

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Legacy interface: returns list of FaceDetection items.
        Because FaceDetection implements sequence indexing (__getitem__ / __iter__),
        callers can unpack as: (x1, y1, x2, y2, conf) = face
        while rich callers can access face.landmarks, face.detector, face.quality_status.
        """
        return self.detect_faces(image)
