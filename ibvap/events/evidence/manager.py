"""
Forensic Evidence Management Subsystem.
Extracts incident snapshots, target crops, computes SHA-256 cryptographic hashes,
and generates structured tamper-evident audit packages.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import os
import time
import json
import hashlib
import logging
import cv2
import numpy as np

from ibvap.core.types import AnalyticsEvent
from ibvap.core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.events.evidence")


@dataclass
class EvidencePackage:
    """
    Forensic evidence record containing image artifacts, cryptographic hashes, and telemetry.
    """
    evidence_id: str
    event_id: str
    camera_id: str
    timestamp: float
    full_frame_path: Optional[str]
    crop_path: Optional[str]
    sha256_hash: str
    metadata_path: Optional[str]


class EvidenceManager:
    """
    Captures, packages, and stores digital forensic evidence for security events.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, evidence_dir: str = "output/evidence"):
        self.config = config or default_config
        self.evidence_dir = getattr(self.config, "storage_dir", evidence_dir) or evidence_dir
        os.makedirs(self.evidence_dir, exist_ok=True)

    def package_evidence(
        self,
        event: AnalyticsEvent,
        full_frame: np.ndarray,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[EvidencePackage]:
        """
        Extracts evidence images, computes SHA-256 fingerprint, and writes audit package.
        """
        if full_frame is None or full_frame.size == 0:
            return None

        try:
            timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(event.timestamp))
            etype = event.event_type.value if hasattr(event.event_type, "value") else str(event.event_type)
            base_name = f"{event.camera_id}_{etype}_{timestamp_str}_{event.event_id[:8]}"

            # 1. Save Full Frame
            full_frame_path = os.path.join(self.evidence_dir, f"{base_name}_full.jpg")
            cv2.imwrite(full_frame_path, full_frame)

            # 2. Compute SHA-256 Hash of saved full frame
            with open(full_frame_path, "rb") as f:
                sha256_hash = hashlib.sha256(f.read()).hexdigest()

            # 3. Extract and save Target Crop if bbox provided
            crop_path = None
            if target_bbox is not None:
                h, w = full_frame.shape[:2]
                x1, y1, x2, y2 = target_bbox
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    crop_img = full_frame[y1:y2, x1:x2]
                    crop_path = os.path.join(self.evidence_dir, f"{base_name}_crop.jpg")
                    cv2.imwrite(crop_path, crop_img)

            # 4. Write forensic audit JSON metadata
            meta_path = os.path.join(self.evidence_dir, f"{base_name}_meta.json")
            evidence_data = {
                "evidence_id": base_name,
                "event_id": event.event_id,
                "camera_id": event.camera_id,
                "timestamp": event.timestamp,
                "event_type": etype,
                "track_id": event.track_id,
                "identity_id": event.identity_id,
                "confidence": event.confidence,
                "sha256_hash": sha256_hash,
                "full_frame_path": full_frame_path,
                "crop_path": crop_path,
                "metadata": event.metadata,
            }
            with open(meta_path, "w") as f:
                json.dump(evidence_data, f, indent=2)

            event.snapshot_path = full_frame_path

            return EvidencePackage(
                evidence_id=base_name,
                event_id=event.event_id,
                camera_id=event.camera_id,
                timestamp=event.timestamp,
                full_frame_path=full_frame_path,
                crop_path=crop_path,
                sha256_hash=sha256_hash,
                metadata_path=meta_path,
            )
        except Exception as e:
            logger.error(f"Failed to package evidence: {e}")
            return None
