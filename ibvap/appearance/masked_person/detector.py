"""
Masked Person & Face Concealment Detection Subsystem.
Analyzes lower-face facial landmarks, occlusion entropy, and texture uniformity
to detect medical masks, balaclavas, scarves, and intentional concealment.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2

from ibvap.face.detector import FaceDetection


@dataclass
class MaskDetectionResult:
    """
    Result of face mask / concealment analysis.
    """
    is_masked: bool
    confidence: float
    concealment_type: str  # "MASKED", "UNMASKED", "UNKNOWN"
    lower_face_entropy: float
    details: Dict[str, Any]


class MaskedPersonDetector:
    """
    Evaluates facial crops and landmarks for lower-face concealment.
    Uses structural analysis of the mouth/chin region compared to the periocular (eyes) region.
    """

    def __init__(
        self,
        entropy_threshold: float = 4.2,  # Low texture entropy indicates uniform cloth mask
        color_uniformity_threshold: float = 18.0,  # Std deviation of lower face patch
    ):
        self.entropy_threshold = entropy_threshold
        self.color_uniformity_threshold = color_uniformity_threshold

    def analyze_face(
        self,
        face_crop: np.ndarray,
        face_detection: Optional[FaceDetection] = None,
    ) -> MaskDetectionResult:
        """
        Analyzes a cropped face image to determine if the lower face is masked.

        Args:
            face_crop: BGR image crop of the face.
            face_detection: Optional FaceDetection instance with landmarks.

        Returns:
            MaskDetectionResult indicating whether the person is wearing a mask.
        """
        if face_crop is None or face_crop.size == 0 or face_crop.shape[0] < 20 or face_crop.shape[1] < 20:
            return MaskDetectionResult(
                is_masked=False,
                confidence=0.0,
                concealment_type="UNKNOWN",
                lower_face_entropy=0.0,
                details={"reason": "Face crop too small or invalid"},
            )

        h, w = face_crop.shape[:2]

        # Lower face region: from 55% to 95% of face height, 15% to 85% of width
        y1 = int(0.55 * h)
        y2 = int(0.95 * h)
        x1 = int(0.15 * w)
        x2 = int(0.85 * w)

        lower_patch = face_crop[y1:y2, x1:x2]
        if lower_patch.size == 0:
            return MaskDetectionResult(
                is_masked=False,
                confidence=0.0,
                concealment_type="UNKNOWN",
                lower_face_entropy=0.0,
                details={"reason": "Invalid lower patch"},
            )

        # 1. Evaluate grayscale entropy (texture complexity)
        gray_lower = cv2.cvtColor(lower_patch, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_lower], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        non_zero = hist_norm[hist_norm > 0]
        entropy = -np.sum(non_zero * np.log2(non_zero))

        # 2. Evaluate color variance / standard deviation
        lower_std = float(np.std(gray_lower))

        # 3. Check landmark signals if available
        # In YuNet landmarks: [re, le, nt, rcm, lcm]
        # If nose and eyes are present but mouth corners are clipped or occluded
        landmark_signal = False
        if face_detection is not None and face_detection.landmarks is not None:
            lm = face_detection.landmarks
            if len(lm) >= 5:
                # Mouth corners: index 3, 4
                rcm, lcm = lm[3], lm[4]
                # If mouth points collapse or are outside valid bounding box
                if (rcm[0] == lcm[0] and rcm[1] == lcm[1]) or rcm[1] <= lm[2][1]:
                    landmark_signal = True

        # Decision rule: High uniformity (low std) or low entropy indicates mask coverage
        is_masked = False
        confidence = 0.50

        if entropy < self.entropy_threshold or lower_std < self.color_uniformity_threshold or landmark_signal:
            is_masked = True
            confidence = min(0.95, 0.70 + (self.entropy_threshold - entropy) * 0.1)
            concealment = "MASKED"
        else:
            is_masked = False
            confidence = min(0.95, 0.70 + (entropy - self.entropy_threshold) * 0.1)
            concealment = "UNMASKED"

        return MaskDetectionResult(
            is_masked=is_masked,
            confidence=round(float(confidence), 2),
            concealment_type=concealment,
            lower_face_entropy=round(float(entropy), 2),
            details={
                "lower_face_std": round(lower_std, 2),
                "entropy": round(float(entropy), 2),
                "landmark_signal": landmark_signal,
            },
        )
