"""
IBVAP Unknown Person Manager.
Orchestrates cross-camera human tracking, unknown identity assignment,
biometric matching, quality gating, evidence accumulation, and transition reporting.
"""

from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import datetime
import os
import time
import uuid
import logging
import numpy as np
import cv2

from ..core.types import (
    UnknownPersonRecord,
    UnknownPersonSighting,
    AnalyticsEvent,
    EventType,
)
from ..core.config import IBVAPConfig, default_config
from .unknown_storage import SQLiteUnknownStorage

logger = logging.getLogger("ibvap.tracking.unknown_person")


@dataclass
class PendingObservation:
    """Internal observation collected before confirming a new unknown identity."""
    embedding: np.ndarray
    quality: float
    timestamp: float
    bbox: Tuple[int, int, int, int]
    body_embedding: Optional[np.ndarray] = None


@dataclass
class UnknownMatchResult:
    """Detailed result of an unknown person matching or registration attempt."""
    decision: str  # "NEW_UNKNOWN", "REIDENTIFIED", "SIGHTING", "COLLECTING_EVIDENCE", "QUALITY_REJECTED", "EXISTING_TRACK"
    unknown_id: Optional[str] = None
    record: Optional[UnknownPersonRecord] = None
    sighting: Optional[UnknownPersonSighting] = None
    similarity: float = 0.0
    body_similarity: float = 0.0
    best_similarity: float = 0.0
    second_best_similarity: float = 0.0
    similarity_margin: float = 0.0
    confidence_level: str = "LOW"  # "HIGH", "UNCERTAIN", "LOW"
    is_cross_camera_transition: bool = False
    previous_camera: Optional[str] = None
    current_camera: str = "camera-01"
    time_gap_seconds: float = 0.0
    event_to_emit: Optional[AnalyticsEvent] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class UnknownPersonManager:
    """
    Manages unknown person re-identification and cross-camera trajectory tracking.
    """

    def __init__(
        self,
        config: Optional[IBVAPConfig] = None,
        storage: Optional[SQLiteUnknownStorage] = None
    ):
        self.config = config or default_config
        self.storage = storage or SQLiteUnknownStorage(config=self.config)

        # Active camera-local track cache: (camera_id, track_id) -> unknown_id
        self._track_to_identity: Dict[Tuple[str, int], str] = {}
        # Last sighting time per (camera_id, track_id) to debounce database writes
        self._track_last_write: Dict[Tuple[str, int], float] = {}

        # Pending observations for new identities (evidence accumulation)
        # (camera_id, track_id) -> list of PendingObservation
        self._pending_evidence: Dict[Tuple[str, int], List[PendingObservation]] = {}

        # Uncertain candidate temporal tracking:
        # (camera_id, track_id) -> list of (candidate_unknown_id, similarity)
        self._uncertain_candidates: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}

    # ── Face Quality Gating ──────────────────────────────────────
    def validate_face_quality(
        self,
        face_crop: Optional[np.ndarray],
        face_detection: Optional[Any] = None
    ) -> Tuple[bool, float, str]:
        """
        Validates face dimensions, sharpness, brightness, and detection confidence
        before allowing embedding generation or identity registration.
        """
        if face_crop is None or face_crop.size == 0:
            return False, 0.0, "Empty face crop"

        fh, fw = face_crop.shape[:2]
        if fw < 24 or fh < 24:
            return False, 0.0, f"Face dimensions too small ({fw}x{fh} < 24x24)"

        # 1. Detection confidence
        conf = 1.0
        if face_detection is not None:
            conf = float(getattr(face_detection, "confidence", 1.0))
            if conf < 0.40:
                return False, conf, f"Face detection confidence too low ({conf:.2f} < 0.40)"

        # 2. Brightness / luminance check
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY) if face_crop.ndim == 3 else face_crop
        mean_brightness = float(np.mean(gray))
        if mean_brightness < 15.0 or mean_brightness > 245.0:
            return False, 0.2, f"Face extreme lighting (brightness={mean_brightness:.1f})"

        # 3. Blur / Sharpness check via Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if laplacian_var < 10.0:
            return False, 0.3, f"Face blurred (Laplacian variance={laplacian_var:.1f} < 10.0)"

        # Compute normalized composite quality score [0.0, 1.0]
        size_score = min(1.0, (fw * fh) / (80.0 * 80.0))
        blur_score = min(1.0, laplacian_var / 150.0)
        conf_score = min(1.0, conf)
        quality_score = float(0.3 * size_score + 0.4 * blur_score + 0.3 * conf_score)

        min_quality = getattr(self.config, "unknown_min_quality_score", 0.40)
        if quality_score < min_quality:
            return False, quality_score, f"Quality score below threshold ({quality_score:.2f} < {min_quality:.2f})"

        return True, quality_score, "Quality check passed"

    # ── Primary Matching & Registration ──────────────────────────
    def match_or_register(
        self,
        face_embedding: np.ndarray,
        camera_id: str,
        track_id: int,
        timestamp: float,
        bbox: Tuple[int, int, int, int],
        face_crop: Optional[np.ndarray] = None,
        face_detection: Optional[Any] = None,
        body_embedding: Optional[np.ndarray] = None,
    ) -> UnknownMatchResult:
        """
        Matches a face embedding against known unknown identities or registers a new identity.
        Implements:
        - Quality gating
        - Camera-local track ID reuse (avoid repeated matching)
        - Top-K vector search with similarity margin check
        - Cross-camera transition detection and event emission
        - Evidence accumulation before registering a new unknown person
        """
        now = timestamp
        iso_timestamp = datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).isoformat()
        track_key = (camera_id, track_id)

        # 1. Quality Gating
        is_quality_valid, quality_score, reason = self.validate_face_quality(face_crop, face_detection)
        if not is_quality_valid:
            logger.debug(f"Face quality rejected for {camera_id} track {track_id}: {reason}")
            return UnknownMatchResult(
                decision="QUALITY_REJECTED",
                current_camera=camera_id,
                metrics={
                    "camera_id": camera_id,
                    "track_id": track_id,
                    "quality_score": quality_score,
                    "rejection_reason": reason,
                }
            )

        # 2. Check Active Local Track Cache (Avoid repeated matching on every frame)
        if track_key in self._track_to_identity:
            cached_uid = self._track_to_identity[track_key]
            record = self.storage.get_record(cached_uid)
            debounce_sec = getattr(self.config, "unknown_db_write_debounce_seconds", 2.0)
            last_write = self._track_last_write.get(track_key, 0.0)

            # Persist sighting only if debounce interval has elapsed
            if now - last_write >= debounce_sec:
                self._track_last_write[track_key] = now
                sighting = UnknownPersonSighting(
                    sighting_id=f"sight-{uuid.uuid4().hex[:8]}",
                    unknown_id=cached_uid,
                    camera_id=camera_id,
                    track_id=track_id,
                    timestamp=now,
                    timestamp_iso=iso_timestamp,
                    bbox=bbox,
                    similarity=1.0,
                    face_quality=quality_score,
                    metadata={"cache_hit": True}
                )
                if record:
                    record.last_seen_timestamp = now
                    record.last_camera_id = camera_id
                    record.updated_at_iso = iso_timestamp
                    self.storage.update_person(record, new_sighting=sighting)

            return UnknownMatchResult(
                decision="EXISTING_TRACK",
                unknown_id=cached_uid,
                record=record,
                similarity=1.0,
                confidence_level="HIGH",
                current_camera=camera_id,
                metrics={"cache_hit": True, "unknown_id": cached_uid}
            )

        # 3. Vector Top-K Search across Unknown Identities
        candidates = self.storage.search_top_k(face_embedding, k=5)
        best_uid: Optional[str] = None
        best_sim: float = 0.0
        second_sim: float = 0.0

        if candidates:
            best_uid, best_sim = candidates[0]
            if len(candidates) > 1:
                second_sim = candidates[1][1]

        margin = best_sim - second_sim

        # 4. Evaluate Thresholds & Confidence
        base_threshold = getattr(self.config, "unknown_face_similarity_threshold", 0.65)
        high_threshold = getattr(self.config, "unknown_face_high_confidence_threshold", 0.75)
        margin_threshold = getattr(self.config, "unknown_similarity_margin", 0.05)

        is_match = False
        confidence_level = "LOW"

        if best_sim >= high_threshold and margin >= margin_threshold:
            # High-confidence direct match
            is_match = True
            confidence_level = "HIGH"
        elif best_sim >= base_threshold:
            # Match is within similarity range; verify ambiguity margin
            if margin >= margin_threshold:
                is_match = True
                confidence_level = "HIGH"
            else:
                # Ambiguous candidate (second best is very close)
                confidence_level = "UNCERTAIN"
                # Check body embedding as secondary signal if available
                body_sim = 0.0
                if body_embedding is not None and best_uid:
                    body_sim = self._compute_body_similarity(best_uid, body_embedding)
                    if body_sim >= 0.70:
                        is_match = True
                        confidence_level = "HIGH"
                        logger.info(f"Uncertain face match confirmed via body Re-ID (body_sim={body_sim:.3f})")

                if not is_match:
                    # Collect temporal evidence across consecutive frames
                    is_match = self._check_temporal_consistency(track_key, best_uid, best_sim)
                    if is_match:
                        confidence_level = "HIGH"
        else:
            confidence_level = "LOW"
            is_match = False

        metrics = {
            "camera_id": camera_id,
            "track_id": track_id,
            "timestamp": now,
            "face_quality": quality_score,
            "best_similarity": best_sim,
            "second_best_similarity": second_sim,
            "similarity_margin": margin,
            "candidate_count": len(candidates),
            "confidence_level": confidence_level,
            "threshold_used": base_threshold,
        }

        # ── Branch 1: Matched an Existing Unknown Person ─────────────
        if is_match and best_uid:
            record = self.storage.get_record(best_uid)
            if record is None:
                # Cache discrepancy fallback
                return self._accumulate_and_register_new(
                    track_key, face_embedding, camera_id, track_id,
                    now, iso_timestamp, bbox, quality_score, body_embedding,
                    metrics
                )

            # Check for cross-camera transition
            is_transition = False
            prev_cam = record.last_camera_id
            time_gap = now - record.last_seen_timestamp

            if prev_cam != camera_id:
                is_transition = True
                if not record.camera_sequence or record.camera_sequence[-1] != camera_id:
                    record.camera_sequence.append(camera_id)
                record.last_camera_id = camera_id

            record.last_seen_timestamp = now
            record.updated_at_iso = iso_timestamp

            # Create sighting record
            sighting = UnknownPersonSighting(
                sighting_id=f"sight-{uuid.uuid4().hex[:8]}",
                unknown_id=best_uid,
                camera_id=camera_id,
                track_id=track_id,
                timestamp=now,
                timestamp_iso=iso_timestamp,
                bbox=bbox,
                similarity=best_sim,
                face_quality=quality_score,
                metadata={
                    "confidence_level": confidence_level,
                    "is_transition": is_transition,
                    "previous_camera": prev_cam,
                    "time_gap_seconds": time_gap,
                }
            )

            # Update representative embedding if observation has high quality and reasonable diversity
            new_rep = None
            if quality_score >= 0.70 and best_sim < 0.92:
                new_rep = face_embedding

            self.storage.update_person(record, new_sighting=sighting, new_representative=new_rep)

            # Bind local track to global identity
            self._track_to_identity[track_key] = best_uid
            self._track_last_write[track_key] = now

            event: Optional[AnalyticsEvent] = None
            if is_transition:
                # Emit PERSON_REIDENTIFIED event detailing cross-camera trajectory
                event = AnalyticsEvent(
                    camera_id=camera_id,
                    timestamp=now,
                    event_type=EventType.PERSON_REIDENTIFIED,
                    track_id=track_id,
                    identity_id=best_uid,
                    confidence=best_sim,
                    metadata={
                        "unknown_id": best_uid,
                        "previous_camera": prev_cam,
                        "current_camera": camera_id,
                        "time_gap_seconds": round(time_gap, 2),
                        "camera_sequence": list(record.camera_sequence),
                        "similarity": round(best_sim, 4),
                        "timestamp_iso": iso_timestamp,
                        "message": (
                            f"Unknown person {best_uid} re-identified at {camera_id} "
                            f"(previously seen at {prev_cam} {time_gap:.1f}s ago)"
                        )
                    }
                )
                logger.info(f"Cross-camera re-identification: {best_uid} {prev_cam} -> {camera_id}")

            metrics["decision"] = "REIDENTIFIED" if is_transition else "SIGHTING"
            metrics["matched_unknown_id"] = best_uid

            return UnknownMatchResult(
                decision="REIDENTIFIED" if is_transition else "SIGHTING",
                unknown_id=best_uid,
                record=record,
                sighting=sighting,
                similarity=best_sim,
                best_similarity=best_sim,
                second_best_similarity=second_sim,
                similarity_margin=margin,
                confidence_level=confidence_level,
                is_cross_camera_transition=is_transition,
                previous_camera=prev_cam,
                current_camera=camera_id,
                time_gap_seconds=time_gap,
                event_to_emit=event,
                metrics=metrics,
            )

        # ── Branch 2: Uncertain Match Candidate ──────────────────────
        # Section 4: UNCERTAIN MATCH -> do not merge -> retain as separate candidate
        if confidence_level == "UNCERTAIN" and not is_match:
            metrics["decision"] = "UNCERTAIN_MATCH"
            return UnknownMatchResult(
                decision="UNCERTAIN_MATCH",
                unknown_id=None,
                record=None,
                similarity=best_sim,
                best_similarity=best_sim,
                second_best_similarity=second_sim,
                similarity_margin=margin,
                confidence_level="UNCERTAIN",
                current_camera=camera_id,
                event_to_emit=None,
                metrics=metrics,
            )

        # ── Branch 3: Candidate for New Unknown Person Identity (NO MATCH) ──
        return self._accumulate_and_register_new(
            track_key, face_embedding, camera_id, track_id,
            now, iso_timestamp, bbox, quality_score, body_embedding,
            metrics
        )

    def _accumulate_and_register_new(
        self,
        track_key: Tuple[str, int],
        face_embedding: np.ndarray,
        camera_id: str,
        track_id: int,
        timestamp: float,
        iso_timestamp: str,
        bbox: Tuple[int, int, int, int],
        quality_score: float,
        body_embedding: Optional[np.ndarray],
        metrics: Dict[str, Any]
    ) -> UnknownMatchResult:
        """
        Accumulates observations before assigning a brand new persistent unknown identity.
        Prevents single-frame transient noise from polluting the database.
        """
        evidence_needed = getattr(self.config, "unknown_evidence_min_hits", 2)

        if track_key not in self._pending_evidence:
            self._pending_evidence[track_key] = []

        self._pending_evidence[track_key].append(
            PendingObservation(
                embedding=face_embedding,
                quality=quality_score,
                timestamp=timestamp,
                bbox=bbox,
                body_embedding=body_embedding,
            )
        )

        obs_list = self._pending_evidence[track_key]

        if len(obs_list) < evidence_needed:
            metrics["decision"] = "COLLECTING_EVIDENCE"
            metrics["evidence_count"] = len(obs_list)
            metrics["evidence_needed"] = evidence_needed
            return UnknownMatchResult(
                decision="COLLECTING_EVIDENCE",
                confidence_level="UNCERTAIN",
                current_camera=camera_id,
                metrics=metrics,
            )

        # Evidence threshold reached -> Confirm new unknown person!
        del self._pending_evidence[track_key]

        # Generate clean persistent person ID: P001, P002, etc.
        existing_count = len(self.storage._records) + 1
        new_uid = f"P{existing_count:03d}"
        while self.storage.get_record(new_uid) is not None:
            existing_count += 1
            new_uid = f"P{existing_count:03d}"

        best_obs = max(obs_list, key=lambda o: o.quality)

        # Select prototype and up to N representatives
        rep_embs = [
            o.embedding for o in obs_list if o is not best_obs
        ][:getattr(self.config, "unknown_max_representatives_per_person", 5)]

        new_record = UnknownPersonRecord(
            unknown_id=new_uid,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp,
            first_camera_id=camera_id,
            last_camera_id=camera_id,
            prototype_embedding=best_obs.embedding,
            representative_embeddings=rep_embs,
            sightings=[],
            camera_sequence=[camera_id],
            created_at_iso=iso_timestamp,
            updated_at_iso=iso_timestamp,
            metadata={
                "first_track_id": track_id,
                "initial_quality": best_obs.quality,
                "evidence_observations": len(obs_list),
            }
        )

        initial_sighting = UnknownPersonSighting(
            sighting_id=f"sight-{uuid.uuid4().hex[:8]}",
            unknown_id=new_uid,
            camera_id=camera_id,
            track_id=track_id,
            timestamp=timestamp,
            timestamp_iso=iso_timestamp,
            bbox=bbox,
            similarity=0.0,
            face_quality=best_obs.quality,
            snapshot_path=None,
            metadata={"evidence_count": len(obs_list)}
        )

        self.storage.save_new_person(new_record, initial_sighting=initial_sighting)
        self._track_to_identity[track_key] = new_uid
        self._track_last_write[track_key] = timestamp

        # Immediately emit UNKNOWN_PERSON event
        event = AnalyticsEvent(
            camera_id=camera_id,
            timestamp=timestamp,
            event_type=EventType.UNKNOWN_PERSON,
            track_id=track_id,
            identity_id=new_uid,
            confidence=best_obs.quality,
            metadata={
                "unknown_id": new_uid,
                "camera_id": camera_id,
                "track_id": track_id,
                "timestamp_iso": iso_timestamp,
                "face_quality": round(best_obs.quality, 4),
                "message": f"New unknown person detected: {new_uid} at {camera_id} ({iso_timestamp})",
            }
        )

        metrics["decision"] = "NEW_UNKNOWN"
        metrics["registered_unknown_id"] = new_uid
        logger.info(f"Registered new unknown person: {new_uid} on {camera_id} at {iso_timestamp}")

        return UnknownMatchResult(
            decision="NEW_UNKNOWN",
            unknown_id=new_uid,
            record=new_record,
            sighting=initial_sighting,
            similarity=0.0,
            confidence_level="HIGH",
            current_camera=camera_id,
            event_to_emit=event,
            metrics=metrics,
        )

    def _check_temporal_consistency(
        self,
        track_key: Tuple[str, int],
        candidate_uid: str,
        similarity: float
    ) -> bool:
        """Requires 2 consistent observations before confirming an uncertain candidate."""
        if track_key not in self._uncertain_candidates:
            self._uncertain_candidates[track_key] = []

        history = self._uncertain_candidates[track_key]
        history.append((candidate_uid, similarity))

        # Check if last 2 observations point to same candidate
        if len(history) >= 2:
            last_two = history[-2:]
            if last_two[0][0] == candidate_uid and last_two[1][0] == candidate_uid:
                del self._uncertain_candidates[track_key]
                return True
        return False

    def _compute_body_similarity(self, unknown_id: str, query_body_emb: np.ndarray) -> float:
        """Compares secondary body embedding against stored sightings if available."""
        record = self.storage.get_record(unknown_id)
        if not record or "body_embedding" not in record.metadata:
            return 0.0
        try:
            target_body = np.array(record.metadata["body_embedding"], dtype=np.float32)
            q = query_body_emb.flatten().astype(np.float32)
            n1 = np.linalg.norm(target_body)
            n2 = np.linalg.norm(q)
            if n1 > 0 and n2 > 0:
                return float(np.dot(target_body / n1, q / n2))
        except Exception:
            pass
        return 0.0

    # ── Query & Trajectory API ───────────────────────────────────
    def get_unknown_person(self, unknown_id: str) -> Optional[UnknownPersonRecord]:
        """Retrieves master unknown record."""
        return self.storage.get_record(unknown_id)

    def get_trajectory(self, unknown_id: str) -> List[Dict[str, Any]]:
        """
        Returns chronological trajectory of camera visits and sightings with timestamps.
        """
        if hasattr(self.storage, "get_ordered_trajectory_segments"):
            segments = self.storage.get_ordered_trajectory_segments(unknown_id)
            if segments:
                return [s.to_dict() for s in segments]

        sightings = self.storage.get_sightings(unknown_id)
        trajectory = []
        for s in sightings:
            trajectory.append({
                "camera_id": s.camera_id,
                "timestamp": s.timestamp,
                "timestamp_iso": s.timestamp_iso,
                "track_id": s.track_id,
                "bbox": list(s.bbox),
                "similarity": s.similarity,
                "face_quality": s.face_quality,
            })
        return trajectory

    def get_person_profile(self, person_id: str) -> Dict[str, Any]:
        """
        Returns complete operator profile for a persistent person.
        """
        record = self.storage.get_record(person_id)
        trajectory = self.get_trajectory(person_id)
        total_duration = sum(s.get("duration_seconds", 0.0) for s in trajectory)
        cameras = list(dict.fromkeys(s.get("camera_id") for s in trajectory))

        return {
            "person_id": person_id,
            "identity_status": "unknown" if record else "not_found",
            "first_seen": record.created_at_iso if record else "",
            "last_seen": record.updated_at_iso if record else "",
            "cameras_visited": cameras,
            "current_camera": record.last_camera_id if record else (cameras[-1] if cameras else None),
            "total_sessions": len(trajectory),
            "total_presence_duration": round(total_duration, 2),
            "trajectory": trajectory,
        }

    def list_unknown_persons(self) -> List[UnknownPersonRecord]:
        """Returns all tracked unknown persons."""
        return self.storage.list_all_records()

    def clear(self):
        """Clears all caches and storage."""
        self._track_to_identity.clear()
        self._track_last_write.clear()
        self._pending_evidence.clear()
        self._uncertain_candidates.clear()
        self.storage.clear()


# ── Offline Threshold Evaluation ─────────────────────────────────
def evaluate_unknown_thresholds(
    embeddings_by_person: Dict[str, List[np.ndarray]],
    thresholds: Optional[List[float]] = None
) -> Dict[float, Dict[str, Any]]:
    """
    Evaluates different cosine similarity thresholds (e.g. 0.60, 0.65, 0.70, 0.75, 0.80)
    across a set of labeled embeddings to measure true matches, false matches, and false splits.
    """
    if thresholds is None:
        thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]

    results = {}
    person_ids = list(embeddings_by_person.keys())

    # Build all intra-person pairs (same person) and inter-person pairs (different person)
    same_pairs = []
    diff_pairs = []

    for i, pid in enumerate(person_ids):
        embs = embeddings_by_person[pid]
        for a in range(len(embs)):
            for b in range(a + 1, len(embs)):
                sim = float(np.dot(embs[a] / np.linalg.norm(embs[a]), embs[b] / np.linalg.norm(embs[b])))
                same_pairs.append(sim)

        for other_pid in person_ids[i + 1:]:
            other_embs = embeddings_by_person[other_pid]
            for ea in embs:
                for eb in other_embs:
                    sim = float(np.dot(ea / np.linalg.norm(ea), eb / np.linalg.norm(eb)))
                    diff_pairs.append(sim)

    for th in thresholds:
        true_positives = sum(1 for s in same_pairs if s >= th)
        false_negatives = sum(1 for s in same_pairs if s < th)  # False split
        false_positives = sum(1 for s in diff_pairs if s >= th)  # False merge
        true_negatives = sum(1 for s in diff_pairs if s < th)

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        results[th] = {
            "threshold": th,
            "same_person_evals": len(same_pairs),
            "diff_person_evals": len(diff_pairs),
            "true_matches (TP)": true_positives,
            "false_splits (FN)": false_negatives,
            "false_merges (FP)": false_positives,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }

    return results
