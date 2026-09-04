"""
Controlled OCR & Multi-Frame Consensus Engine (Phases 5 & 6).
Provides:
1. ControlledOCRRunner: Safely executes OCR on selected plate observations within strict budget limits.
2. PlateConsensusEngine: Temporal multi-frame consensus reconciling candidate readings into a ConsensusResult.

Anti-Hallucination Invariant:
When evidence is ambiguous, conflicting, or below threshold, the engine returns
is_confirmed=False and plate_number=None (or unconfirmed candidate).
Characters are NEVER invented or altered to satisfy Indian plate regex.
"""

from typing import List, Dict, Optional, Tuple, Any
import logging
import re

from .types import (
    VehicleObservation,
    VehicleTrackState,
    ConsensusResult,
    VehicleStatus,
)
from ..core.types import WatchlistCategory
from ..anpr.ocr_adapter import ANPRAdapter, INDIAN_STATES

logger = logging.getLogger("ibvap.vehicle.consensus")

# Standard Indian License Plate Registration Patterns (Validation Signal Only)
STANDARD_INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$")
BHARAT_SERIES_PATTERN = re.compile(r"^[0-9]{2}BH[0-9]{4}[A-Z]{1,2}$")


class ControlledOCRRunner:
    """
    Executes controlled optical character recognition on selected plate observations.
    Enforces per-track OCR attempt budgets to prevent runaway CPU utilization.
    """

    def __init__(
        self,
        ocr_adapter: Optional[ANPRAdapter] = None,
        max_ocr_attempts_per_track: int = 3,
    ):
        """
        Initializes the controlled OCR runner.

        Args:
            ocr_adapter: Reusable ANPRAdapter instance (PaddleOCR wrapper).
            max_ocr_attempts_per_track: Hard budget on heavy OCR invocations per track (default: 3).
                                       NOTE: INITIAL ENGINEERING DEFAULT. REQUIRES REAL-WORLD VALIDATION.
        """
        self.ocr_adapter = ocr_adapter or ANPRAdapter()
        self.max_ocr_attempts = max_ocr_attempts_per_track

    def run_ocr(
        self,
        observations: List[VehicleObservation],
        track_state: Optional[VehicleTrackState] = None,
    ) -> List[VehicleObservation]:
        """
        Executes OCR on un-recognized observations while respecting track attempt limits.

        Args:
            observations: List of candidate VehicleObservation instances to recognize.
            track_state: Optional VehicleTrackState tracking total ocr_attempts.

        Returns:
            List[VehicleObservation]: The observations updated with OCR fields.
        """
        if not observations:
            return []

        processed: List[VehicleObservation] = []

        for obs in observations:
            # Check track OCR budget limit
            if track_state is not None and track_state.ocr_attempts >= self.max_ocr_attempts:
                logger.debug(
                    f"[ControlledOCR] Track #{track_state.track_id} exceeded max OCR budget ({self.max_ocr_attempts}). Skipping."
                )
                break

            # Skip observations without plate crops
            if obs.plate_crop is None or obs.plate_crop.size == 0:
                obs.metadata["ocr_status"] = "SKIPPED_NO_CROP"
                continue

            # Skip observations that already have OCR results to prevent redundant compute
            if obs.ocr_text is not None and obs.ocr_confidence is not None:
                processed.append(obs)
                continue

            # Increment track attempt counter
            if track_state is not None:
                track_state.ocr_attempts += 1

            # Execute existing PaddleOCR engine
            try:
                plate_res = self.ocr_adapter.recognize_plate(obs.plate_crop)
                if plate_res and plate_res.plate_number:
                    obs.ocr_text = plate_res.plate_number
                    obs.ocr_confidence = float(plate_res.ocr_confidence)
                    obs.metadata["raw_ocr_text"] = plate_res.raw_text
                    obs.metadata["ocr_category"] = (
                        plate_res.category.value
                        if hasattr(plate_res.category, "value")
                        else str(plate_res.category)
                    )
                    obs.metadata["ocr_status"] = "SUCCESS"
                else:
                    obs.ocr_text = None
                    obs.ocr_confidence = 0.0
                    obs.metadata["ocr_status"] = "OCR_FAILED"
            except Exception as e:
                logger.error(f"[ControlledOCR] Error recognizing plate crop: {e}")
                obs.ocr_text = None
                obs.ocr_confidence = 0.0
                obs.metadata["ocr_status"] = f"ERROR_{type(e).__name__}"

            processed.append(obs)

        return processed


class PlateConsensusEngine:
    """
    Reconciles multiple OCR observations across a vehicle track into a validated ConsensusResult.
    Applies character-level voting, temporal agreement analysis, and conservative verification.
    """

    def __init__(
        self,
        min_consensus_observations: int = 2,
        min_agreement_ratio: float = 0.60,
        min_confidence_threshold: float = 0.70,
        single_observation_confidence_threshold: float = 0.92,
    ):
        """
        Initializes the consensus engine.

        All thresholds are INITIAL ENGINEERING DEFAULTS and REQUIRE REAL-WORLD VALIDATION.
        """
        self.min_consensus_observations = min_consensus_observations
        self.min_agreement_ratio = min_agreement_ratio
        self.min_confidence_threshold = min_confidence_threshold
        self.single_obs_threshold = single_observation_confidence_threshold

    def evaluate(self, observations: Optional[List[VehicleObservation]]) -> ConsensusResult:
        """
        Evaluates a collection of plate observations and produces an evidence-based ConsensusResult.

        Args:
            observations: List of VehicleObservation instances (with populated ocr_text).

        Returns:
            ConsensusResult: Structured consensus report.
        """
        if not observations:
            return ConsensusResult(
                plate_number=None,
                confidence=0.0,
                observation_count=0,
                agreement_ratio=0.0,
                candidate_strings=[],
                status=VehicleStatus.INSUFFICIENT_EVIDENCE,
                is_confirmed=False,
                category=WatchlistCategory.UNKNOWN,
                metadata={"reason": "no_observations"},
            )

        # ── 1. Extract Valid Candidates ──────────────────────────────
        valid_candidates: List[Tuple[str, float, float]] = []  # (text, ocr_conf, quality_score)
        all_candidate_strings: List[str] = []

        for obs in observations:
            if not isinstance(obs, VehicleObservation):
                continue
            if obs.ocr_text and len(obs.ocr_text) >= 3:
                clean_text = obs.ocr_text.strip().upper()
                ocr_conf = float(obs.ocr_confidence) if obs.ocr_confidence is not None else 0.5
                quality = (
                    float(obs.quality.overall_score)
                    if obs.quality and obs.quality.overall_score is not None
                    else 50.0
                )
                valid_candidates.append((clean_text, ocr_conf, quality))
                all_candidate_strings.append(clean_text)

        obs_count = len(valid_candidates)

        if obs_count == 0:
            return ConsensusResult(
                plate_number=None,
                confidence=0.0,
                observation_count=0,
                agreement_ratio=0.0,
                candidate_strings=[],
                status=VehicleStatus.INSUFFICIENT_EVIDENCE,
                is_confirmed=False,
                category=WatchlistCategory.UNKNOWN,
                metadata={"reason": "no_valid_ocr_text"},
            )

        # ── 2. Single Observation Evaluation ─────────────────────────
        if obs_count == 1:
            text, conf, quality = valid_candidates[0]
            format_valid = self._validate_format(text)

            # Conservative single-frame rule: confirm ONLY if confidence >= threshold AND format is valid
            is_confirmed = (conf >= self.single_obs_threshold) and format_valid
            status = (
                VehicleStatus.PLATE_CONFIRMED
                if is_confirmed
                else (
                    VehicleStatus.OCR_CONFIDENCE_LOW
                    if conf < self.min_confidence_threshold
                    else VehicleStatus.INSUFFICIENT_EVIDENCE
                )
            )

            return ConsensusResult(
                plate_number=text if is_confirmed else None,
                confidence=round(conf, 4),
                observation_count=1,
                agreement_ratio=1.0,
                candidate_strings=all_candidate_strings,
                status=status,
                is_confirmed=is_confirmed,
                category=WatchlistCategory.UNKNOWN,
                metadata={
                    "single_observation": True,
                    "format_valid": format_valid,
                    "candidate_text": text,
                },
            )

        # ── 3. Multi-Frame String Frequencies & Weights ──────────────
        # Weighted frequency tally: weight = conf * (0.5 + 0.5 * (quality / 100))
        string_weights: Dict[str, float] = {}
        string_counts: Dict[str, int] = {}
        string_confs: Dict[str, List[float]] = {}

        for text, conf, quality in valid_candidates:
            w = conf * (0.5 + 0.5 * (quality / 100.0))
            string_weights[text] = string_weights.get(text, 0.0) + w
            string_counts[text] = string_counts.get(text, 0) + 1
            string_confs.setdefault(text, []).append(conf)

        # Top candidate by total weighted score
        top_string = max(string_weights.keys(), key=lambda s: string_weights[s])
        matching_count = string_counts[top_string]
        agreement_ratio = float(matching_count) / float(obs_count)
        avg_conf_for_top = sum(string_confs[top_string]) / float(matching_count)

        # ── 4. Character-Level Positional Voting ──────────────────────
        # If candidates share equal string length, perform positional voting
        char_voted_string, char_vote_conf = self._positional_voting(valid_candidates)
        if char_voted_string and char_voted_string != top_string:
            # Check if char-voted string has higher structural support
            char_format_valid = self._validate_format(char_voted_string)
            top_format_valid = self._validate_format(top_string)
            if char_format_valid and not top_format_valid:
                top_string = char_voted_string
                avg_conf_for_top = char_vote_conf

        # ── 5. Format Validation Signal (Never Mutate) ────────────────
        is_format_valid = self._validate_format(top_string)
        format_factor = 1.0 if is_format_valid else 0.85
        final_confidence = min(1.0, max(0.0, avg_conf_for_top * format_factor))

        # ── 6. Consensus & Conflict Decision Logic ───────────────────
        # Conflict detection: If multiple distinct strings exist and top agreement ratio < threshold
        is_conflict = (len(string_counts) > 1) and (agreement_ratio < self.min_agreement_ratio)

        if is_conflict:
            return ConsensusResult(
                plate_number=None,
                confidence=round(final_confidence, 4),
                observation_count=obs_count,
                agreement_ratio=round(agreement_ratio, 4),
                candidate_strings=all_candidate_strings,
                status=VehicleStatus.MULTI_FRAME_CONFLICT,
                is_confirmed=False,
                category=WatchlistCategory.UNKNOWN,
                metadata={
                    "conflict_details": string_counts,
                    "top_candidate": top_string,
                    "format_valid": is_format_valid,
                },
            )

        # Check confirmation criteria
        is_confirmed = (
            agreement_ratio >= self.min_agreement_ratio
            and final_confidence >= self.min_confidence_threshold
            and obs_count >= self.min_consensus_observations
        )

        status = (
            VehicleStatus.PLATE_CONFIRMED
            if is_confirmed
            else (
                VehicleStatus.OCR_CONFIDENCE_LOW
                if final_confidence < self.min_confidence_threshold
                else VehicleStatus.INSUFFICIENT_EVIDENCE
            )
        )

        return ConsensusResult(
            plate_number=top_string if is_confirmed else None,
            confidence=round(final_confidence, 4),
            observation_count=obs_count,
            agreement_ratio=round(agreement_ratio, 4),
            candidate_strings=all_candidate_strings,
            status=status,
            is_confirmed=is_confirmed,
            category=WatchlistCategory.UNKNOWN,
            metadata={
                "candidate_counts": string_counts,
                "format_valid": is_format_valid,
                "unconfirmed_candidate": top_string if not is_confirmed else None,
            },
        )

    def _validate_format(self, plate_text: str) -> bool:
        """
        Validates whether plate text structurally matches legitimate Indian registration formats.
        Used solely as a confidence weighting signal; NEVER mutates characters.
        """
        if not plate_text:
            return False

        # State code check (first 2 characters must be valid Indian state/UT code)
        if len(plate_text) >= 2 and plate_text[:2] in INDIAN_STATES:
            if STANDARD_INDIAN_PLATE_PATTERN.match(plate_text):
                return True

        # Bharat series format check (e.g. 22BH1234AA)
        if BHARAT_SERIES_PATTERN.match(plate_text):
            return True

        return False

    def _positional_voting(
        self, candidates: List[Tuple[str, float, float]]
    ) -> Tuple[Optional[str], float]:
        """
        Performs positional character-by-character weighted voting across equal-length candidates.
        """
        lengths = [len(c[0]) for c in candidates]
        # Only vote positionally if majority share identical length
        mode_len = max(set(lengths), key=lengths.count)
        if lengths.count(mode_len) < 2:
            return None, 0.0

        matching_cands = [c for c in candidates if len(c[0]) == mode_len]

        voted_chars: List[str] = []
        pos_confs: List[float] = []

        for pos in range(mode_len):
            char_scores: Dict[str, float] = {}
            for text, conf, qual in matching_cands:
                char = text[pos]
                w = conf * (0.5 + 0.5 * (qual / 100.0))
                char_scores[char] = char_scores.get(char, 0.0) + w

            best_char = max(char_scores.keys(), key=lambda c: char_scores[c])
            voted_chars.append(best_char)
            total_w = sum(char_scores.values())
            pos_conf = char_scores[best_char] / total_w if total_w > 0 else 0.0
            pos_confs.append(pos_conf)

        voted_string = "".join(voted_chars)
        avg_pos_conf = sum(pos_confs) / float(len(pos_confs)) if pos_confs else 0.0
        return voted_string, avg_pos_conf
