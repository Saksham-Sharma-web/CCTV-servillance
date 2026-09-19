"""
Bounded Vehicle Observation Buffer Module.
Maintains a small, bounded set of high-quality license plate observations per vehicle track.

Key Invariants:
1. Hard per-track upper bound: never accumulates unbounded frame history.
2. Quality-aware eviction: retains only the highest-quality candidate plate crops.
3. Memory safety: retains only candidate plate crops, NEVER full video frames.
4. Clean track lifecycle: supports track registration, retrieval, and stale track cleanup.
5. Zero OCR / zero recognition: stores evidence only; never attempts character recognition.
"""

from typing import Dict, List, Optional, Tuple
import logging
import time

from .types import VehicleObservation, VehicleTrackState, VehicleStatus, PlateQualityReport

logger = logging.getLogger("ibvap.vehicle.buffer")


class VehicleTrackBuffer:
    """
    In-memory bounded buffer storing candidate plate observations indexed by vehicle track ID.
    Enforces quality-aware retention and bounded memory limits.
    """

    def __init__(
        self,
        max_observations_per_track: int = 5,
        stale_track_timeout_seconds: float = 3.75,
        max_active_tracks: int = 100,
    ):
        """
        Initializes the bounded buffer.

        Args:
            max_observations_per_track: Maximum observations retained per vehicle track (default: 5).
            stale_track_timeout_seconds: Seconds of inactivity before a track is considered stale (default: 3.75s).
            max_active_tracks: Maximum number of distinct active tracks held in buffer (default: 100).
        """
        if max_observations_per_track < 1:
            raise ValueError(f"max_observations_per_track must be >= 1, got {max_observations_per_track}")
        if stale_track_timeout_seconds <= 0:
            raise ValueError(f"stale_track_timeout_seconds must be > 0, got {stale_track_timeout_seconds}")
        if max_active_tracks < 1:
            raise ValueError(f"max_active_tracks must be >= 1, got {max_active_tracks}")

        self.max_observations = max_observations_per_track
        self.stale_timeout = stale_track_timeout_seconds
        self.max_active_tracks = max_active_tracks

        # Storage: track_id -> VehicleTrackState
        self._tracks: Dict[int, VehicleTrackState] = {}

    def add_observation(
        self,
        observation: Optional[VehicleObservation],
        camera_id: str = "camera-01",
        vehicle_class: str = "vehicle",
    ) -> bool:
        """
        Inserts a candidate plate observation into the track's bounded buffer.
        If the buffer is at capacity, the observation with the lowest quality score is evicted
        if and only if the new observation has a strictly higher quality score.

        Args:
            observation: VehicleObservation instance to buffer.
            camera_id: Identifier of the camera stream.
            vehicle_class: Vehicle classification string (e.g. 'car', 'truck').

        Returns:
            bool: True if observation was retained/inserted; False if dropped or invalid.
        """
        if observation is None or not isinstance(observation, VehicleObservation):
            logger.debug("[VehicleBuffer] Rejected invalid or None observation.")
            return False

        track_id = observation.track_id

        # If track not yet registered, ensure capacity and initialize state
        if track_id not in self._tracks:
            if len(self._tracks) >= self.max_active_tracks:
                # Evict oldest stale track to keep global track count strictly bounded
                self._evict_oldest_track()

            self._tracks[track_id] = VehicleTrackState(
                track_id=track_id,
                camera_id=camera_id,
                vehicle_class=vehicle_class,
                status=VehicleStatus.VEHICLE_TRACKED,
                first_seen=observation.timestamp,
                last_seen=observation.timestamp,
                observations=[],
                best_observation=None,
                total_frames_tracked=1,
            )

        track_state = self._tracks[track_id]
        track_state.last_seen = max(track_state.last_seen, observation.timestamp)
        track_state.total_frames_tracked += 1

        obs_score = observation.quality.overall_score if observation.quality else 0.0

        # Check for same-frame observation update to prevent duplicate slots from single frame
        for idx, existing in enumerate(track_state.observations):
            if existing.frame_index == observation.frame_index:
                existing_score = existing.quality.overall_score if existing.quality else 0.0
                if obs_score > existing_score:
                    track_state.observations[idx] = observation
                    self._recalculate_best_observation(track_state)
                    return True
                return False

        # If buffer has available capacity, simply append
        if len(track_state.observations) < self.max_observations:
            track_state.observations.append(observation)
            self._recalculate_best_observation(track_state)
            return True

        # Buffer is at capacity: find existing observation with lowest quality score
        min_idx, min_obs = min(
            enumerate(track_state.observations),
            key=lambda item: item[1].quality.overall_score if item[1].quality else 0.0,
        )
        min_score = min_obs.quality.overall_score if min_obs.quality else 0.0

        # Evict lowest-quality observation if the new candidate is strictly better
        if obs_score > min_score:
            track_state.observations[min_idx] = observation
            self._recalculate_best_observation(track_state)
            return True

        # New candidate does not exceed the quality of retained observations; safely drop
        return False

    def get_observations(self, track_id: int) -> List[VehicleObservation]:
        """
        Returns a shallow copy of currently buffered observations for the given track_id.
        Returns empty list if track does not exist.
        """
        if track_id not in self._tracks:
            return []
        return list(self._tracks[track_id].observations)

    def get_best_observation(self, track_id: int) -> Optional[VehicleObservation]:
        """
        Returns the highest-quality VehicleObservation currently buffered for the track.
        Returns None if track does not exist or has no observations.
        """
        if track_id not in self._tracks:
            return None
        return self._tracks[track_id].best_observation

    def get_track_state(self, track_id: int) -> Optional[VehicleTrackState]:
        """
        Returns the VehicleTrackState object associated with track_id, or None.
        """
        return self._tracks.get(track_id, None)

    def has_track(self, track_id: int) -> bool:
        """Returns True if track_id exists in the buffer."""
        return track_id in self._tracks

    def active_track_ids(self) -> List[int]:
        """Returns a list of all active track IDs currently in buffer."""
        return list(self._tracks.keys())

    def observation_count(self, track_id: int) -> int:
        """Returns the number of retained observations for a specific track ID."""
        if track_id not in self._tracks:
            return 0
        return len(self._tracks[track_id].observations)

    def total_observations(self) -> int:
        """Returns total observations held across all active tracks."""
        return sum(len(state.observations) for state in self._tracks.values())

    def remove_track(self, track_id: int) -> Optional[VehicleTrackState]:
        """
        Removes a track and its buffered observations from memory.
        Returns the removed VehicleTrackState if present, otherwise None.
        """
        return self._tracks.pop(track_id, None)

    def cleanup_stale_tracks(
        self,
        current_time: Optional[float] = None,
        timeout_seconds: Optional[float] = None,
    ) -> List[int]:
        """
        Removes tracks that have not been observed within timeout_seconds.

        Args:
            current_time: Epoch timestamp in seconds (defaults to time.time()).
            timeout_seconds: Stale threshold in seconds (defaults to self.stale_timeout).

        Returns:
            List of track IDs that were pruned.
        """
        now = current_time if current_time is not None else time.time()
        timeout = timeout_seconds if timeout_seconds is not None else self.stale_timeout

        stale_ids: List[int] = []
        for track_id, state in list(self._tracks.items()):
            if (now - state.last_seen) > timeout:
                stale_ids.append(track_id)

        for tid in stale_ids:
            self._tracks.pop(tid, None)

        return stale_ids

    def clear(self) -> None:
        """Clears all track states and observations."""
        self._tracks.clear()

    # ── Internal Helpers ───────────────────────────────────────────

    def _recalculate_best_observation(self, track_state: VehicleTrackState) -> None:
        """Updates the best_observation pointer on a VehicleTrackState."""
        if not track_state.observations:
            track_state.best_observation = None
            return

        track_state.best_observation = max(
            track_state.observations,
            key=lambda obs: obs.quality.overall_score if obs.quality else 0.0,
        )

    def _evict_oldest_track(self) -> Optional[int]:
        """Evicts the track with the earliest last_seen timestamp to respect global track capacity."""
        if not self._tracks:
            return None

        oldest_id, _ = min(
            self._tracks.items(),
            key=lambda item: item[1].last_seen,
        )
        self._tracks.pop(oldest_id, None)
        return oldest_id
