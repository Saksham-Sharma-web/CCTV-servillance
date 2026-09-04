"""
Biometric Identity Verifier Adapter.
Wraps the existing FaceMatcher engine from id-verification/verification/face/face_matcher.py.
DO NOT duplicate FaceMatcher or InceptionResnetV1 logic here.
CRITICAL: Keeps identity_id strictly decoupled from tracking track_id.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import os
import sys
import logging
import cv2
import numpy as np
from PIL import Image

from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.face.matcher")


@dataclass
class AuthorizedPerson:
    identity_id: str
    name: str
    role: str = "EMPLOYEE"
    embedding: Optional[np.ndarray] = None


class IdentityVerifierAdapter:
    """
    Thin adapter interfacing with VibeestaBackend's existing FaceMatcher.
    Maintains an in-memory or database-backed registry of authorized biometric face embeddings.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.similarity_threshold = self.config.face_verification_similarity_threshold
        self.authorized_registry: Dict[str, AuthorizedPerson] = {}
        self.face_matcher = None
        self._facenet = None
        self._initialized = False

    def _ensure_face_matcher(self):
        if self._initialized:
            return
        self._initialized = True
        try:
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            id_verif_path = os.path.join(repo_root, "id-verification")
            if id_verif_path not in sys.path:
                sys.path.insert(0, id_verif_path)

            from verification.face.face_matcher import FaceMatcher
            self.face_matcher = FaceMatcher()
            logger.info("Existing FaceMatcher from id-verification successfully loaded into adapter.")
            return
        except Exception as e:
            self.face_matcher = None

        # Fallback to direct facenet_pytorch
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
            self._facenet = InceptionResnetV1(pretrained="vggface2").eval()
            logger.info("Standalone InceptionResnetV1 loaded for biometric identity verification.")
        except Exception as e:
            logger.warning(f"Could not load InceptionResnetV1: {e}. Running in lightweight fallback mode.")

    def _extract_embedding_from_crop(self, face_bgr_image: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr_image is None or face_bgr_image.size == 0:
            return None

        self._ensure_face_matcher()
        if self.face_matcher is not None:
            try:
                rgb = cv2.cvtColor(face_bgr_image, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                res = self.face_matcher.extract_face_embedding(pil_img)
                if res and "embedding" in res:
                    emb = np.array(res["embedding"], dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
            except Exception as e:
                logger.error(f"FaceMatcher error: {e}")

        if self._facenet is not None:
            try:
                import torch
                rgb = cv2.cvtColor(face_bgr_image, cv2.COLOR_BGR2RGB)
                resized = cv2.resize(rgb, (160, 160))
                tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
                # Standardize to [-1, 1]
                tensor = (tensor - 127.5) / 128.0
                tensor = tensor.unsqueeze(0)
                with torch.no_grad():
                    emb_tensor = self._facenet(tensor)
                    emb = emb_tensor[0].cpu().numpy()
                    norm = np.linalg.norm(emb)
                    return emb / norm if norm > 0 else emb
            except Exception as e:
                logger.error(f"InceptionResnetV1 extraction error: {e}")

        return None

    def register_person(self, identity_id: str, name: str, face_bgr_image: Optional[np.ndarray] = None, embedding: Optional[np.ndarray] = None, role: str = "EMPLOYEE") -> bool:
        """
        Registers an authorized person using either a precomputed 512-D embedding or a reference face image.
        """
        if embedding is not None:
            norm_emb = embedding / np.linalg.norm(embedding) if np.linalg.norm(embedding) > 0 else embedding
            self.authorized_registry[identity_id] = AuthorizedPerson(
                identity_id=identity_id,
                name=name,
                role=role,
                embedding=np.array(norm_emb, dtype=np.float32)
            )
            return True

        if face_bgr_image is not None:
            emb = self._extract_embedding_from_crop(face_bgr_image)
            if emb is not None:
                self.authorized_registry[identity_id] = AuthorizedPerson(
                    identity_id=identity_id,
                    name=name,
                    role=role,
                    embedding=emb
                )
                logger.info(f"Registered authorized person '{name}' ({identity_id}) via face embedding.")
                return True

        return False

    def verify_crop(self, face_bgr_crop: np.ndarray) -> Tuple[Optional[AuthorizedPerson], float]:
        """
        Extracts embedding from face crop and searches registry for best cosine match.

        Returns:
            (Matched AuthorizedPerson or None, similarity_score)
        """
        if face_bgr_crop is None or face_bgr_crop.size == 0 or len(self.authorized_registry) == 0:
            return None, 0.0

        emb = self._extract_embedding_from_crop(face_bgr_crop)
        if emb is None:
            return None, 0.0

        best_person = None
        best_sim = -1.0

        for person in self.authorized_registry.values():
            if person.embedding is None:
                continue
            sim = float(np.dot(emb, person.embedding))
            if sim > best_sim:
                best_sim = sim
                best_person = person

        if best_sim >= self.similarity_threshold:
            return best_person, best_sim
        else:
            return None, max(0.0, best_sim)
