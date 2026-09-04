"""
Best Observation Selector Module (Phase 4).
Selects the strongest candidate license plate observations from a vehicle track buffer
to feed downstream OCR processing.

Key Invariants:
1. Deterministic quality-ranked selection.
2. Filters out low-quality observations before expensive OCR.
3. Enforces temporal diversity: avoids selecting identical consecutive frames when quality is comparable.
4. Hard upper bound on selected observations (Top-K, default K=2 or 3).
5. Zero OCR / Zero image modification.
"""

from typing import List, Optional
import logging

from .types import VehicleObservation

logger = logging.getLogger("ibvap.vehicle.selector")


class BestObservationSelector:
    """
    Selects top-K highest-quality, temporally diverse observations from buffered plate observations.
    """

    def __init__(
        self,
        max_k: int = 3,
        min_quality_threshold: float = 45.0,
        min_frame_separation: int = 2,
    ):
        """
        Initializes the selector.

        Args:
            max_k: Maximum number of observations to select for OCR (default: 3).
                   NOTE: INITIAL ENGINEERING DEFAULT. REQUIRES REAL-WORLD VALIDATION.
            min_quality_threshold: Observations with quality.overall_score below this are excluded (default: 45.0).
            min_frame_separation: Preferred minimum separation in frame_index between selected observations
                                  to ensure temporal viewpoint diversity (default: 2).
        """
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1, got {max_k}")
        if min_quality_threshold < 0.0 or min_quality_threshold > 100.0:
            raise ValueError(f"min_quality_threshold must be in [0.0, 100.0], got {min_quality_threshold}")
        if min_frame_separation < 0:
            raise ValueError(f"min_frame_separation must be >= 0, got {min_frame_separation}")

        self.max_k = max_k
        self.min_quality_threshold = min_quality_threshold
        self.min_frame_separation = min_frame_separation

    def select(self, observations: Optional[List[VehicleObservation]]) -> List[VehicleObservation]:
        """
        Selects up to top-K highest quality observations from the input list.

        Args:
            observations: List of VehicleObservation instances.

        Returns:
            List[VehicleObservation]: Bounded list (length <= max_k) of selected observations,
                                     ordered by quality descending.
        """
        if not observations:
            return []

        # ── 1. Quality & Usability Filtering ─────────────────────────
        usable: List[VehicleObservation] = []
        for obs in observations:
            if not isinstance(obs, VehicleObservation):
                continue
            # Must have a non-empty image crop
            if obs.plate_crop is None or obs.plate_crop.size == 0:
                continue
            # Must have computed quality
            if obs.quality is None:
                continue
            # Must meet minimum quality threshold
            if obs.quality.overall_score < self.min_quality_threshold:
                continue
            usable.append(obs)

        if not usable:
            return []

        # ── 2. Deterministic Ranking ─────────────────────────────────
        # Primary: overall_score (descending)
        # Secondary: detection_confidence (descending)
        # Tertiary: frame_index (ascending for stable sort)
        sorted_candidates = sorted(
            usable,
            key=lambda o: (
                o.quality.overall_score if o.quality else 0.0,
                float(o.detection_confidence),
                -o.frame_index,
            ),
            reverse=True,
        )

        # ── 3. Temporal Diversity Selection ──────────────────────────
        # Always take the absolute highest quality candidate first
        selected: List[VehicleObservation] = [sorted_candidates[0]]

        # Try to pick remaining candidates that maintain temporal separation
        if self.min_frame_separation > 0:
            for cand in sorted_candidates[1:]:
                if len(selected) >= self.max_k:
                    break
                min_dist = min(abs(cand.frame_index - s.frame_index) for s in selected)
                if min_dist >= self.min_frame_separation:
                    selected.append(cand)

        # ── 4. Fallback Fill if slots remain ─────────────────────────
        # If temporal diversity was too strict, fill remaining slots up to max_k
        # with the next best quality candidates that are not already selected
        if len(selected) < self.max_k:
            selected_ids = {id(s) for s in selected}
            for cand in sorted_candidates:
                if len(selected) >= self.max_k:
                    break
                if id(cand) not in selected_ids:
                    selected.append(cand)
                    selected_ids.add(id(cand))

        return selected[:self.max_k]
