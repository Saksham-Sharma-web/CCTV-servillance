"""
Face Detector using OpenCV YuNet with Haar Cascade Fallback.
Provides high-speed CPU face detection on raw BGR frames.
"""

from typing import List, Tuple, Optional
import os
import logging
import cv2
import numpy as np

from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.face.detector")


class OpenCVFaceDetector:
    """
    Detects faces in BGR images using OpenCV YuNet (if onnx model present)
    or OpenCV Haar cascades as a universal fallback.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, yunet_model_path: Optional[str] = None):
        self.config = config or default_config
        self.confidence_threshold = self.config.face_detection_confidence
        self.yunet = None
        self.haar_cascade = None

        # Try YuNet first if model exists
        candidate_paths = [
            yunet_model_path,
            getattr(self.config, "yunet_model_path", None),
            os.path.join(getattr(self.config, "models_dir", ""), "face_detection_yunet_2023mar.onnx")
        ]
        resolved_path = next((p for p in candidate_paths if p and os.path.exists(p)), None)

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
                logger.info(f"OpenCV YuNet face detector loaded from {resolved_path}")
            except Exception as e:
                logger.warning(f"Could not load YuNet model: {e}")

        # Universal fallback: Haar cascade
        if self.yunet is None and hasattr(cv2, "CascadeClassifier"):
            try:
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                self.haar_cascade = cv2.CascadeClassifier(cascade_path)
                logger.info("OpenCV Haar cascade face detector initialized as universal fallback.")
            except Exception as e:
                logger.warning(f"Could not load Haar cascade: {e}")

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detects faces in the given BGR image or crop.

        Args:
            image: numpy.ndarray BGR format

        Returns:
            List of (x1, y1, x2, y2, confidence)
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]
        faces: List[Tuple[int, int, int, int, float]] = []

        # 1. Try YuNet if available
        if self.yunet is not None:
            try:
                self.yunet.setInputSize((w, h))
                _, raw_faces = self.yunet.detect(image)
                if raw_faces is not None:
                    for f in raw_faces:
                        x, y, fw, fh = int(f[0]), int(f[1]), int(f[2]), int(f[3])
                        conf = float(f[14])
                        if conf >= self.confidence_threshold:
                            x1 = max(0, min(w - 1, x))
                            y1 = max(0, min(h - 1, y))
                            x2 = max(x1 + 1, min(w, x + fw))
                            y2 = max(y1 + 1, min(h, y + fh))
                            faces.append((x1, y1, x2, y2, conf))
                    return faces
            except Exception as e:
                logger.warning(f"YuNet detection error: {e}")

        # 2. Haar Cascade fallback
        if self.haar_cascade is not None:
            try:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Equalize histogram for contrast
                gray = cv2.equalizeHist(gray)
                rects = self.haar_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
                for (x, y, fw, fh) in rects:
                    x1 = max(0, min(w - 1, int(x)))
                    y1 = max(0, min(h - 1, int(y)))
                    x2 = max(x1 + 1, min(w, int(x + fw)))
                    y2 = max(y1 + 1, min(h, int(y + fh)))
                    faces.append((x1, y1, x2, y2, 0.85))
            except Exception as e:
                logger.error(f"Haar cascade detection error: {e}")

        return faces
