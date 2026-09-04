"""
Biometric Identity Verifier & Supporting Body Appearance Engine.
Provides 5-point facial landmark affine alignment to 160x160 canonical template,
InceptionResnetV1 (VGGFace2) 512-D L2-normalized biometric embeddings,
multi-reference identity registry with reference-age semantics ("most_recent", "recent", "old"),
and controlled supporting body/appearance Re-ID feature extraction.

CRITICAL INVARIANT:
FACE = PRIMARY BIOMETRIC VERIFIER
BODY = SUPPORTING OBSERVATION ONLY
A body/clothing match NEVER turns NO_FACE or UNKNOWN into a verified identity.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import os
import sys
import logging
import cv2
import numpy as np
from PIL import Image

from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.face.matcher")

# Canonical 5-point facial landmark coordinates for 160x160 face alignment
# Order: [right_eye (subject right, image left), left_eye (subject left, image right),
#         nose, right_mouth (image left), left_mouth (image right)]
CANONICAL_FACE_TEMPLATE_160 = np.array([
    [54.6, 73.9],   # Right eye
    [105.4, 73.9],  # Left eye
    [80.0, 99.8],   # Nose tip
    [59.2, 123.7],  # Right mouth corner
    [100.8, 123.7]  # Left mouth corner
], dtype=np.float32)


def align_face_160(
    image: np.ndarray,
    landmarks: Optional[np.ndarray] = None,
    box: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    """
    Performs 5-point landmark affine alignment into a canonical 160x160 normalized face image.
    If landmarks are missing or degenerate, falls back to aspect-preserving centered crop of bbox.
    """
    if image is None or image.size == 0:
        return np.zeros((160, 160, 3), dtype=np.uint8)

    h, w = image.shape[:2]

    # Strategy 1: 5-point affine similarity transform
    if landmarks is not None and len(landmarks) == 5:
        try:
            src_pts = np.array(landmarks, dtype=np.float32)
            M, inliers = cv2.estimateAffinePartial2D(src_pts, CANONICAL_FACE_TEMPLATE_160)
            if M is not None:
                aligned = cv2.warpAffine(
                    image, M, (160, 160),
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0)
                )
                return aligned
        except Exception as e:
            logger.debug(f"Affine alignment fallback to bbox crop: {e}")

    # Strategy 2: Bounding box crop with aspect-preserving resize & center padding
    if box is not None:
        x1, y1, x2, y2 = box
        crop = image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    else:
        crop = image

    if crop.size == 0:
        return np.zeros((160, 160, 3), dtype=np.uint8)

    ch, cw = crop.shape[:2]
    scale = 160.0 / max(ch, cw)
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((160, 160, 3), dtype=np.uint8)
    px = (160 - nw) // 2
    py = (160 - nh) // 2
    canvas[py:py+nh, px:px+nw] = resized
    return canvas


class BodyAppearanceExtractor:
    """
    Extracts a 256-D L2-normalized color and spatial appearance descriptor from a person crop.
    Deconstructs person crop into upper torso (clothing) and lower body zones.
    Computes HSV color distribution + edge gradient orientation histograms per zone.
    CPU-friendly, executes in < 2ms, zero heavy GPU dependencies.
    """

    @staticmethod
    def extract(person_bgr: np.ndarray) -> Optional[np.ndarray]:
        if person_bgr is None or person_bgr.size < 100:
            return None

        h, w = person_bgr.shape[:2]
        if h < 20 or w < 10:
            return None

        # Scale down large person crops for ultra-fast CPU histogramming (< 2ms)
        if max(h, w) > 256:
            scale = 256.0 / float(max(h, w))
            person_bgr = cv2.resize(person_bgr, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
            h, w = person_bgr.shape[:2]

        # Split into upper torso (top 60%) and lower body (bottom 40%)
        split_y = int(h * 0.6)
        upper_zone = person_bgr[0:split_y, :]
        lower_zone = person_bgr[split_y:h, :]

        feature_parts = []
        for zone in (upper_zone, lower_zone):
            if zone.size == 0:
                feature_parts.append(np.zeros(128, dtype=np.float32))
                continue

            # 1. Color distribution in HSV
            hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
            # Hue: 16 bins, Saturation: 8 bins, Value: 8 bins = 32 bins
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            s_hist = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            v_hist = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            color_feat = np.concatenate([h_hist, s_hist, v_hist])
            c_norm = np.linalg.norm(color_feat)
            if c_norm > 0:
                color_feat /= c_norm

            # 2. Gradient / texture distribution
            gray = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            ang_hist, _ = np.histogram(ang, bins=16, range=(0, 360), weights=mag)
            g_norm = np.linalg.norm(ang_hist)
            if g_norm > 0:
                ang_hist = ang_hist.astype(np.float32) / g_norm

            # Combined zone feature (32 color + 16 grad = 48 -> padded to 128)
            z_feat = np.pad(np.concatenate([color_feat, ang_hist]), (0, 128 - 48))
            feature_parts.append(z_feat)

        full_vec = np.concatenate(feature_parts).astype(np.float32)  # 256-D
        v_norm = np.linalg.norm(full_vec)
        if v_norm > 0:
            full_vec /= v_norm
        return full_vec


@dataclass
class PersonReference:
    """A single reference photograph enrolled for an identity."""
    source_path: str
    reference_age: str  # "most_recent", "recent", "old"
    face_embedding: np.ndarray  # 512-D L2-normalized
    body_embedding: Optional[np.ndarray] = None  # 256-D L2-normalized
    face_confidence: float = 1.0
    quality_status: str = "GOOD_FACE"


@dataclass
class AuthorizedPerson:
    """An enrolled identity holding one or more biometric reference photos."""
    identity_id: str
    name: str
    role: str = "EMPLOYEE"
    references: List[PersonReference] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None  # Primary/best embedding for legacy compatibility


@dataclass
class VerificationResult:
    """Structured decision output from IdentityVerifierAdapter."""
    identity: Optional[str] = None
    identity_id: Optional[str] = None
    face_decision: str = "NO_FACE_DETECTED"  # MATCH, UNKNOWN, NO_FACE_DETECTED, INSUFFICIENT_FACE_QUALITY
    face_confidence: float = 0.0
    face_similarity: float = 0.0
    best_reference_path: Optional[str] = None
    best_reference_age: Optional[str] = None
    body_status: str = "BODY_NOT_DETECTED"  # BODY_SUPPORTING, BODY_INCONSISTENT, BODY_IGNORED, BODY_NOT_DETECTED
    body_similarity: float = 0.0
    body_role: str = "SUPPORTING ONLY"
    all_reference_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    matched_person: Optional[AuthorizedPerson] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity": self.identity,
            "identity_id": self.identity_id,
            "face_decision": self.face_decision,
            "face_confidence": round(self.face_confidence, 4),
            "face_similarity": round(self.face_similarity, 4),
            "best_reference_path": self.best_reference_path,
            "best_reference_age": self.best_reference_age,
            "body_status": self.body_status,
            "body_similarity": round(self.body_similarity, 4),
            "body_role": self.body_role,
            "references_analyzed": len(self.all_reference_comparisons),
        }


class IdentityVerifierAdapter:
    """
    Biometric face recognizer using InceptionResnetV1 (pretrained on VGGFace2),
    canonical 5-point landmark affine alignment, and controlled supporting body Re-ID.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.similarity_threshold = self.config.face_verification_similarity_threshold
        self.authorized_registry: Dict[str, AuthorizedPerson] = {}
        self._facenet = None
        self._torch = None
        self._initialized = False

    def _ensure_facenet(self):
        """Lazy one-time model initialization and caching."""
        if self._initialized:
            return
        self._initialized = True
        try:

            import torch
            from facenet_pytorch import InceptionResnetV1
            self._torch = torch
            self._facenet = InceptionResnetV1(pretrained="vggface2").eval()
            logger.info("InceptionResnetV1 (VGGFace2) loaded once and cached for biometric verification.")
        except Exception as e:
            logger.error(f"Could not load InceptionResnetV1: {e}")
            self._facenet = None

    def extract_face_embedding(self, aligned_160_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 512-D L2-normalized embedding from 160x160 aligned face image.
        """
        if aligned_160_bgr is None or aligned_160_bgr.size == 0:
            return None

        self._ensure_facenet()
        if self._facenet is None:
            return None

        try:
            rgb = cv2.cvtColor(aligned_160_bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (160, 160):
                rgb = cv2.resize(rgb, (160, 160))
            tensor = self._torch.from_numpy(rgb).permute(2, 0, 1).float()
            # Normalize to [-1, 1]
            tensor = (tensor - 127.5) / 128.0
            tensor = tensor.unsqueeze(0)

            with self._torch.no_grad():
                emb_tensor = self._facenet(tensor)
                emb = emb_tensor[0].cpu().numpy().astype(np.float32)

            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as e:
            logger.error(f"InceptionResnetV1 embedding extraction error: {e}")
            return None

    def register_reference(
        self,
        name: str,
        image_path: str,
        reference_age: str = "most_recent",
        identity_id: Optional[str] = None,
        role: str = "AUTHORIZED",
        detector: Optional[Any] = None
    ) -> Tuple[bool, str]:
        """
        Registers an authorized person from a reference photograph.
        CRITICAL: Reference images MUST NOT bypass face detection!
        Steps:
        1. Load image
        2. Detect faces
        3. Validate face quality
        4. Align face via 5 landmarks to 160x160
        5. Extract 512-D embedding
        6. Extract optional body appearance embedding
        """
        if not os.path.exists(image_path):
            return False, f"File not found: {image_path}"

        # Load image with auto-orientation (respect EXIF from smartphone cameras)
        img = None
        try:
            from PIL import Image, ImageOps
            with Image.open(image_path) as pil_img:
                pil_img = ImageOps.exif_transpose(pil_img)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            pass

        if img is None:
            img = cv2.imread(image_path)

        if img is None or img.size == 0:
            return False, f"Failed to read image at: {image_path}"

        # Initialize detector if not supplied
        if detector is None:
            from .detector import OpenCVFaceDetector
            detector = OpenCVFaceDetector(self.config)

        # 1. Face detection (with multi-rotation search for phone photos)
        faces = detector.detect_faces(img)
        if not faces:
            for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
                rotated = cv2.rotate(img, rot)
                rot_faces = detector.detect_faces(rotated)
                if rot_faces:
                    img = rotated
                    faces = rot_faces
                    logger.info(f"Detected face in {image_path} after auto-rotation ({rot}).")
                    break

        if not faces:
            logger.warning(f"Registration rejected: REFERENCE_FACE_NOT_FOUND in {image_path}")
            return False, "REFERENCE_FACE_NOT_FOUND"

        best_face = faces[0]
        if best_face.quality_status == "NO_FACE":
            return False, "REFERENCE_FACE_NOT_FOUND"
        if best_face.quality_status == "LOW_QUALITY_FACE":
            logger.warning(f"Registration rejected: INSUFFICIENT_FACE_QUALITY in {image_path}")
            return False, "INSUFFICIENT_FACE_QUALITY"

        # 2. 5-point landmark affine alignment
        aligned_face = align_face_160(img, landmarks=best_face.landmarks, box=best_face.box)

        # 3. Face embedding
        face_emb = self.extract_face_embedding(aligned_face)
        if face_emb is None:
            return False, "FACE_EMBEDDING_FAILED"

        # 4. Body appearance embedding (if body available)
        body_emb = None
        if getattr(self.config, "body_support_enabled", True):
            # If face takes only portion of image, use full image as person context
            body_emb = BodyAppearanceExtractor.extract(img)

        # 5. Add to registry
        pid = identity_id or f"ID-{name.upper().replace(' ', '_')}"
        if pid not in self.authorized_registry:
            self.authorized_registry[pid] = AuthorizedPerson(
                identity_id=pid,
                name=name,
                role=role,
                references=[],
                embedding=face_emb
            )

        ref_entry = PersonReference(
            source_path=image_path,
            reference_age=reference_age,
            face_embedding=face_emb,
            body_embedding=body_emb,
            face_confidence=best_face.confidence,
            quality_status=best_face.quality_status
        )
        self.authorized_registry[pid].references.append(ref_entry)
        self.authorized_registry[pid].embedding = face_emb  # Keep most recent primary

        logger.info(
            f"Successfully enrolled reference for '{name}' ({pid}) from {image_path} "
            f"[age={reference_age}, conf={best_face.confidence:.3f}]"
        )
        return True, "SUCCESS"

    def register_person(
        self,
        identity_id: str,
        name: str,
        face_bgr_image: Optional[np.ndarray] = None,
        embedding: Optional[np.ndarray] = None,
        role: str = "EMPLOYEE"
    ) -> bool:
        """
        Legacy registration interface. Preserved for backward compatibility.
        """
        if embedding is not None:
            norm_emb = embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
            person = AuthorizedPerson(
                identity_id=identity_id,
                name=name,
                role=role,
                embedding=np.array(norm_emb, dtype=np.float32)
            )
            person.references.append(PersonReference(
                source_path="direct_embedding",
                reference_age="most_recent",
                face_embedding=person.embedding
            ))
            self.authorized_registry[identity_id] = person
            return True

        if face_bgr_image is not None and face_bgr_image.size > 0:
            aligned = align_face_160(face_bgr_image)
            emb = self.extract_face_embedding(aligned)
            if emb is not None:
                person = AuthorizedPerson(
                    identity_id=identity_id,
                    name=name,
                    role=role,
                    embedding=emb
                )
                person.references.append(PersonReference(
                    source_path="direct_crop",
                    reference_age="most_recent",
                    face_embedding=emb
                ))
                self.authorized_registry[identity_id] = person
                return True

        return False

    def verify(
        self,
        target_image: np.ndarray,
        face_detection: Optional[Any] = None,
        person_crop: Optional[np.ndarray] = None
    ) -> VerificationResult:
        """
        Comprehensive Biometric Verification with Hard Invariant Enforcement:
        1. If no face detected or face quality insufficient -> NO_FACE_DETECTED / INSUFFICIENT_FACE_QUALITY
        2. Extract face embedding from aligned 160x160 face
        3. Match against all enrolled references
        4. If face similarity < threshold -> UNKNOWN (Identity is None)
        5. If face similarity >= threshold -> MATCH
        6. Extract body appearance -> SUPPORTING SIGNAL ONLY.
        """
        # ── Invariant Check 1: No Face Detected ───────────────────────
        if face_detection is None:
            # Body may be present, but face is NOT detected
            body_sim = 0.0
            if person_crop is not None and person_crop.size > 0 and len(self.authorized_registry) > 0:
                target_body = BodyAppearanceExtractor.extract(person_crop)
                if target_body is not None:
                    # Find highest body sim for informational reporting
                    for p in self.authorized_registry.values():
                        for ref in p.references:
                            if ref.body_embedding is not None and ref.reference_age != "old":
                                b_sim = float(np.dot(target_body, ref.body_embedding))
                                if b_sim > body_sim:
                                    body_sim = b_sim

            return VerificationResult(
                face_decision="NO_FACE_DETECTED",
                identity=None,
                identity_id=None,
                face_confidence=0.0,
                face_similarity=0.0,
                body_status="BODY_DETECTED" if body_sim > 0.60 else "BODY_NOT_DETECTED",
                body_similarity=body_sim,
                body_role="SUPPORTING ONLY",
            )

        # ── Invariant Check 2: Insufficient Face Quality ──────────────
        quality_status = getattr(face_detection, "quality_status", "GOOD_FACE")
        if quality_status == "NO_FACE":
            return VerificationResult(
                face_decision="NO_FACE_DETECTED",
                identity=None,
                identity_id=None
            )
        if quality_status == "LOW_QUALITY_FACE":
            return VerificationResult(
                face_decision="INSUFFICIENT_FACE_QUALITY",
                identity=None,
                identity_id=None,
                face_confidence=float(getattr(face_detection, "confidence", 0.0))
            )

        # ── 5-Point Affine Alignment & Embedding ──────────────────────
        landmarks = getattr(face_detection, "landmarks", None)
        box = getattr(face_detection, "box", None)
        aligned_face = align_face_160(target_image, landmarks=landmarks, box=box)
        target_face_emb = self.extract_face_embedding(aligned_face)

        if target_face_emb is None:
            return VerificationResult(face_decision="NO_FACE_DETECTED")

        # ── Body Appearance Extraction (Supporting only) ─────────────
        target_body_emb = None
        if person_crop is not None and person_crop.size > 0 and getattr(self.config, "body_support_enabled", True):
            target_body_emb = BodyAppearanceExtractor.extract(person_crop)

        # ── Multi-Reference Comparison ────────────────────────────────
        best_face_sim = -1.0
        best_person = None
        best_ref = None
        best_body_sim = 0.0
        comparisons = []

        for pid, person in self.authorized_registry.items():
            for ref in person.references:
                f_sim = float(np.dot(target_face_emb, ref.face_embedding))
                b_sim = 0.0
                if target_body_emb is not None and ref.body_embedding is not None:
                    b_sim = float(np.dot(target_body_emb, ref.body_embedding))

                comp_info = {
                    "identity_id": person.identity_id,
                    "name": person.name,
                    "reference_path": ref.source_path,
                    "reference_age": ref.reference_age,
                    "face_similarity": round(f_sim, 4),
                    "body_similarity": round(b_sim, 4) if ref.reference_age != "old" else 0.0,
                    "body_role": "SUPPORTING ONLY" if ref.reference_age != "old" else "IGNORED"
                }
                comparisons.append(comp_info)

                if f_sim > best_face_sim:
                    best_face_sim = f_sim
                    best_person = person
                    best_ref = ref
                    best_body_sim = b_sim

        face_conf = float(getattr(face_detection, "confidence", 1.0))

        # ── Decision Engine: Face is Primary, Body is Supporting ──────
        if best_face_sim >= self.similarity_threshold and best_person is not None and best_ref is not None:
            # Face MATCH verified!
            # Evaluate body appearance role
            if best_ref.reference_age == "old":
                body_status = "BODY_IGNORED"
            elif target_body_emb is not None and best_ref.body_embedding is not None:
                if best_body_sim >= 0.70:
                    body_status = "BODY_SUPPORTING"
                else:
                    body_status = "BODY_INCONSISTENT"  # Disagreement does NOT reject face match
            else:
                body_status = "BODY_NOT_DETECTED"

            return VerificationResult(
                identity=best_person.name,
                identity_id=best_person.identity_id,
                face_decision="MATCH",
                face_confidence=face_conf,
                face_similarity=max(0.0, best_face_sim),
                best_reference_path=best_ref.source_path,
                best_reference_age=best_ref.reference_age,
                body_status=body_status,
                body_similarity=best_body_sim if best_ref.reference_age != "old" else 0.0,
                body_role="SUPPORTING ONLY",
                all_reference_comparisons=comparisons,
                matched_person=best_person
            )
        else:
            # Face similarity does NOT satisfy threshold -> UNKNOWN
            # HARD INVARIANT: Body similarity is NEVER used to override a face mismatch
            return VerificationResult(
                identity=None,
                identity_id=None,
                face_decision="UNKNOWN",
                face_confidence=face_conf,
                face_similarity=max(0.0, best_face_sim),
                best_reference_path=best_ref.source_path if best_ref else None,
                best_reference_age=best_ref.reference_age if best_ref else None,
                body_status="BODY_SUPPORTING_BUT_FACE_UNKNOWN" if best_body_sim >= 0.70 else "BODY_NOT_DETECTED",
                body_similarity=best_body_sim if best_ref and best_ref.reference_age != "old" else 0.0,
                body_role="SUPPORTING ONLY",
                all_reference_comparisons=comparisons,
                matched_person=None
            )

    def verify_crop(
        self,
        face_bgr_crop: np.ndarray,
        person_bgr_crop: Optional[np.ndarray] = None
    ) -> Tuple[Optional[AuthorizedPerson], float]:
        """
        Legacy interface for pipeline.py and external callers.
        Returns: (Matched AuthorizedPerson or None, similarity_score)
        """
        if face_bgr_crop is None or face_bgr_crop.size == 0 or len(self.authorized_registry) == 0:
            return None, 0.0

        aligned = align_face_160(face_bgr_crop)
        emb = self.extract_face_embedding(aligned)
        if emb is None:
            return None, 0.0

        best_person = None
        best_sim = -1.0

        for person in self.authorized_registry.values():
            for ref in person.references:
                sim = float(np.dot(emb, ref.face_embedding))
                if sim > best_sim:
                    best_sim = sim
                    best_person = person

        if best_sim >= self.similarity_threshold:
            return best_person, best_sim
        else:
            return None, max(0.0, best_sim)


def calibrate_threshold(genuine_pairs: List[float], impostor_pairs: List[float]) -> Dict[str, Any]:
    """
    Calibrates face verification similarity threshold using genuine and impostor pair distributions.
    Computes false acceptance rate (FAR) and false rejection rate (FRR) across thresholds.
    """
    if not genuine_pairs or not impostor_pairs:
        return {"error": "Insufficient pairs for calibration"}

    gen = np.array(genuine_pairs, dtype=np.float32)
    imp = np.array(impostor_pairs, dtype=np.float32)

    gen_mean, gen_std = float(np.mean(gen)), float(np.std(gen))
    imp_mean, imp_std = float(np.mean(imp)), float(np.std(imp))

    best_thresh = 0.65
    min_diff = 1.0
    eer = 0.0

    thresholds = np.linspace(0.40, 0.90, 51)
    curve = []
    for th in thresholds:
        far = float(np.mean(imp >= th))
        frr = float(np.mean(gen < th))
        curve.append({"threshold": round(float(th), 3), "FAR": round(far, 4), "FRR": round(frr, 4)})
        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_thresh = float(th)
            eer = (far + frr) / 2.0

    return {
        "recommended_threshold": round(best_thresh, 3),
        "equal_error_rate": round(eer, 4),
        "genuine_distribution": {"mean": round(gen_mean, 4), "std": round(gen_std, 4), "min": round(float(np.min(gen)), 4), "max": round(float(np.max(gen)), 4)},
        "impostor_distribution": {"mean": round(imp_mean, 4), "std": round(imp_std, 4), "min": round(float(np.min(imp)), 4), "max": round(float(np.max(imp)), 4)},
        "roc_operating_curve": curve
    }
