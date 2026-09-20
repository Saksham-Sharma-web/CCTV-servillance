"""
IBVAP Person-Centric Event Correlation Layer.
Consolidates raw detections into persistent person identities, presence sessions,
and cross-camera trajectories. Suppresses duplicate events, quashes repeated loitering
alerts into live duration updates, and preserves high-priority security events.
"""

from typing import List, Dict, Tuple, Optional, Any, Set
import time
import datetime
import uuid
import logging
import threading

from ..core.types import (
    PresenceSession,
    TrajectorySegment,
    AnalyticsEvent,
    EventType,
    Track,
)
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.events.correlator")


# High-priority security events that must never be swallowed into routine presence
HIGH_PRIORITY_EVENTS: Set[EventType] = {
    EventType.FENCE_INTRUSION,
    EventType.REGION_INTRUSION,
    EventType.BORDER_CROSSING,
    EventType.LINE_CROSSING,
    EventType.TRESPASSING,
    EventType.UNAUTHORIZED_VEHICLE,
    EventType.VEHICLE_RESTRICTED_ZONE,
    EventType.MASKED_PERSON,
    EventType.BLACKLISTED_VEHICLE,
    EventType.WATCHLIST_VEHICLE,
}


class PersonEventCorrelator:
    """
    Centralized event correlation, deduplication, and trajectory tracking engine.
    Maintains thread-safe in-memory state of active presence sessions and events.
    """

    def __init__(
        self,
        config: Optional[IBVAPConfig] = None,
        storage: Optional[Any] = None,
    ):
        self.config = config or default_config
        self.storage = storage

        self.session_timeout = getattr(self.config, "person_session_timeout_seconds", 30.0)
        self.update_interval = getattr(self.config, "person_event_update_interval_seconds", 5.0)
        self.max_trajectory_gap = getattr(self.config, "person_trajectory_max_gap_seconds", 300.0)

        # Thread safety reentrant lock across cameras
        self._lock = threading.RLock()

        # In-memory presence state:
        # (person_id, camera_id) -> PresenceSession
        self._active_sessions: Dict[Tuple[str, str], PresenceSession] = {}

        # In-memory active events:
        # event_key -> AnalyticsEvent
        self._active_events: Dict[str, AnalyticsEvent] = {}

        # Last emitted timestamp for durable updates:
        # event_key -> float
        self._last_emitted_event_time: Dict[str, float] = {}

        # Last seen camera per person for cross-camera transitions:
        # person_id -> (camera_id, timestamp)
        self._person_last_camera: Dict[str, Tuple[str, float]] = {}

        # Ordered trajectory segments per person:
        # person_id -> List[TrajectorySegment]
        self._person_trajectories: Dict[str, List[TrajectorySegment]] = {}

        # Local tracking association:
        # (camera_id, track_id) -> person_id
        self._track_to_person: Dict[Tuple[str, int], str] = {}

    # ── Identity & Session Management ─────────────────────────────

    def bind_track_person(self, camera_id: str, track_id: int, person_id: str):
        """Binds a camera-local track ID to a global persistent person identity."""
        with self._lock:
            self._track_to_person[(camera_id, track_id)] = person_id

    def get_person_for_track(self, camera_id: str, track_id: int) -> Optional[str]:
        """Resolves global person_id for a local camera track."""
        with self._lock:
            return self._track_to_person.get((camera_id, track_id))

    def get_or_create_session(
        self,
        person_id: str,
        camera_id: str,
        timestamp: float,
        track_id: Optional[int] = None,
    ) -> Tuple[PresenceSession, bool, bool]:
        """
        Retrieves active session or creates a new one.
        Returns (session, is_new_session, is_camera_transition).
        """
        iso_ts = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).isoformat()
        session_key = (person_id, camera_id)
        is_new = False
        is_transition = False

        if session_key in self._active_sessions:
            session = self._active_sessions[session_key]
            # Check if session timed out
            if (timestamp - session.last_seen) <= self.session_timeout:
                session.last_seen = timestamp
                session.duration_seconds = max(0.0, timestamp - session.first_seen)
                session.updated_at_iso = iso_ts
                if track_id is not None and track_id not in session.tracker_ids:
                    session.tracker_ids.append(track_id)
                return session, False, False
            else:
                # Timed out -> close old session and record segment
                self._close_session(session, timestamp)

        # Create new presence session
        is_new = True
        sess_id = f"sess_{uuid.uuid4().hex[:8]}"

        # Check cross-camera transition
        prev_cam_info = self._person_last_camera.get(person_id)
        if prev_cam_info is not None:
            prev_cam, prev_time = prev_cam_info
            if prev_cam != camera_id and (timestamp - prev_time) <= self.max_trajectory_gap:
                is_transition = True
                # Close active session on previous camera if still open
                prev_session_key = (person_id, prev_cam)
                if prev_session_key in self._active_sessions:
                    prev_sess = self._active_sessions.pop(prev_session_key)
                    self._close_session(prev_sess, timestamp)
                logger.info(
                    f"Cross-camera transition detected for {person_id}: {prev_cam} -> {camera_id} "
                    f"(gap={timestamp - prev_time:.1f}s)"
                )

        new_session = PresenceSession(
            session_id=sess_id,
            person_id=person_id,
            camera_id=camera_id,
            first_seen=timestamp,
            last_seen=timestamp,
            duration_seconds=0.0,
            status="ACTIVE",
            tracker_ids=[track_id] if track_id is not None else [],
            created_at_iso=iso_ts,
            updated_at_iso=iso_ts,
            metadata={"is_transition": is_transition},
        )
        self._active_sessions[session_key] = new_session
        self._person_last_camera[person_id] = (camera_id, timestamp)

        if self.storage and hasattr(self.storage, "save_session"):
            try:
                self.storage.save_session(new_session)
            except Exception as e:
                logger.warning(f"Failed to persist session {sess_id}: {e}")

        return new_session, is_new, is_transition

    def _close_session(self, session: PresenceSession, current_time: float):
        """Closes a presence session, adds segment to trajectory, and persists."""
        session.status = "CLOSED"
        session.duration_seconds = max(0.0, session.last_seen - session.first_seen)
        iso_ts = datetime.datetime.fromtimestamp(session.last_seen, tz=datetime.timezone.utc).isoformat()
        session.updated_at_iso = iso_ts

        segment = TrajectorySegment(
            camera_id=session.camera_id,
            entry_time=session.first_seen,
            exit_time=session.last_seen,
            duration_seconds=session.duration_seconds,
            tracker_ids=list(session.tracker_ids),
            session_id=session.session_id,
            entry_time_iso=session.created_at_iso,
            exit_time_iso=session.updated_at_iso,
        )

        if session.person_id not in self._person_trajectories:
            self._person_trajectories[session.person_id] = []
        self._person_trajectories[session.person_id].append(segment)

        if self.storage and hasattr(self.storage, "update_session"):
            try:
                self.storage.update_session(session)
            except Exception as e:
                logger.warning(f"Failed to update closed session {session.session_id}: {e}")

    # ── Centralized Event Correlation ─────────────────────────────

    def correlate_event(
        self,
        camera_id: str,
        event_type: EventType,
        timestamp: float,
        person_id: Optional[str] = None,
        track_id: Optional[int] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        snapshot_path: Optional[str] = None,
    ) -> Optional[AnalyticsEvent]:
        """
        Centralized event correlation and deduplication.
        - Resolves person identity
        - Finds or updates presence session
        - Quashes repeated loitering alerts into live presence duration
        - Emits a single PERSON_DETECTED event and periodically updates duration
        - Emits high-priority events (virtual fence, masked person) directly
        """
        with self._lock:
            meta = dict(metadata or {})
            resolved_pid = person_id
            if resolved_pid is None and track_id is not None:
                resolved_pid = self._track_to_person.get((camera_id, track_id))

            # Non-person event (e.g. vehicle, general motion) without identity:
            if resolved_pid is None:
                return self._correlate_anonymous_event(
                    camera_id=camera_id,
                    event_type=event_type,
                    timestamp=timestamp,
                    track_id=track_id,
                    confidence=confidence,
                    metadata=meta,
                    snapshot_path=snapshot_path,
                )

            # Get or update active presence session
            session, is_new_session, is_transition = self.get_or_create_session(
                person_id=resolved_pid,
                camera_id=camera_id,
                timestamp=timestamp,
                track_id=track_id,
            )

            # ── 1. Quash Loitering as Repeated Alert ──────────────────
            # Section 13 & 14: Loitering becomes Presence Duration!
            if event_type == EventType.LOITERING:
                # Update session metadata
                session.metadata["is_loitering"] = True
                session.metadata["loitering_duration"] = round(session.duration_seconds, 1)

                # Update active PERSON_DETECTED event if one exists
                event_key = f"{resolved_pid}:{camera_id}:PERSON_DETECTED:{session.session_id}"
                if event_key in self._active_events:
                    ev = self._active_events[event_key]
                    ev.last_seen = timestamp
                    ev.duration_seconds = session.duration_seconds
                    ev.metadata["status"] = "PRESENT"
                    ev.metadata["presence_duration"] = int(session.duration_seconds)
                    ev.metadata["is_loitering"] = True

                    last_emit = self._last_emitted_event_time.get(event_key, 0.0)
                    if timestamp - last_emit >= self.update_interval:
                        self._last_emitted_event_time[event_key] = timestamp
                        ev.is_update = True
                        return ev
                # Suppress separate LOITERING alert
                return None

            # ── 2. High-Priority Security Events ─────────────────────
            # Section 24: Never swallow virtual fence, restricted area, masked person, etc.
            if event_type in HIGH_PRIORITY_EVENTS or "INTRUSION" in str(event_type):
                sub_key = meta.get("zone_id") or meta.get("line_id") or str(track_id)
                hp_key = f"{resolved_pid}:{camera_id}:{event_type.value}:{sub_key}"
                last_hp_time = self._last_emitted_event_time.get(hp_key, 0.0)
                cooldown = getattr(self.config, "event_cooldown_seconds", 5.0)

                if (timestamp - last_hp_time) >= cooldown:
                    self._last_emitted_event_time[hp_key] = timestamp
                    hp_event = AnalyticsEvent(
                        event_id=f"evt_{camera_id}_{int(timestamp * 1000)}",
                        camera_id=camera_id,
                        timestamp=timestamp,
                        event_type=event_type,
                        track_id=track_id,
                        identity_id=resolved_pid,
                        person_id=resolved_pid,
                        session_id=session.session_id,
                        confidence=confidence,
                        metadata=meta,
                        snapshot_path=snapshot_path,
                        first_seen=session.first_seen,
                        last_seen=timestamp,
                        duration_seconds=session.duration_seconds,
                        event_status="ACTIVE",
                        is_update=False,
                    )
                    return hp_event
                return None

            # ── 3. Routine Person Observation / Sighting ──────────────
            # Section 15 & 16: PERSON_DETECTED is the primary event
            # Event correlation key: person_id + camera_id + event_type + active_session_id
            event_key = f"{resolved_pid}:{camera_id}:PERSON_DETECTED:{session.session_id}"

            if event_key in self._active_events:
                # UPDATE existing event
                ev = self._active_events[event_key]
                ev.last_seen = timestamp
                ev.duration_seconds = session.duration_seconds
                ev.metadata["status"] = "PRESENT"
                ev.metadata["presence_duration"] = int(session.duration_seconds)
                ev.confidence = max(ev.confidence, confidence)

                last_emit = self._last_emitted_event_time.get(event_key, 0.0)
                if timestamp - last_emit >= self.update_interval:
                    self._last_emitted_event_time[event_key] = timestamp
                    ev.is_update = True
                    return ev
                # Debounced frame observation within interval
                return None

            # CREATE new event for this person/camera session
            event_id = f"evt_{camera_id}_{int(timestamp * 1000)}"
            session.event_id = event_id

            meta["status"] = "PRESENT"
            meta["presence_duration"] = 0
            meta["person_id"] = resolved_pid
            meta["is_transition"] = is_transition
            meta["identity_status"] = meta.get("identity_status", "unknown")

            new_ev = AnalyticsEvent(
                event_id=event_id,
                camera_id=camera_id,
                timestamp=timestamp,
                event_type=event_type,
                track_id=track_id,
                identity_id=resolved_pid,
                person_id=resolved_pid,
                session_id=session.session_id,
                confidence=confidence,
                metadata=meta,
                snapshot_path=snapshot_path,
                first_seen=timestamp,
                last_seen=timestamp,
                duration_seconds=0.0,
                event_status="ACTIVE",
                is_update=False,
            )
            self._active_events[event_key] = new_ev
            self._last_emitted_event_time[event_key] = timestamp
            return new_ev

    def _correlate_anonymous_event(
        self,
        camera_id: str,
        event_type: EventType,
        timestamp: float,
        track_id: Optional[int],
        confidence: float,
        metadata: Dict[str, Any],
        snapshot_path: Optional[str],
    ) -> Optional[AnalyticsEvent]:
        """Deduplicates events that do not yet have a resolved person identity."""
        sub_key = metadata.get("plate_number") or metadata.get("zone_id") or str(track_id or "")
        key = f"anon:{camera_id}:{event_type.value}:{sub_key}"
        last_time = self._last_emitted_event_time.get(key, 0.0)
        cooldown = getattr(self.config, "event_deduplication_window_seconds", 3.0)

        if (timestamp - last_time) >= cooldown:
            self._last_emitted_event_time[key] = timestamp
            return AnalyticsEvent(
                event_id=f"evt_{camera_id}_{int(timestamp * 1000)}",
                camera_id=camera_id,
                timestamp=timestamp,
                event_type=event_type,
                track_id=track_id,
                confidence=confidence,
                metadata=metadata,
                snapshot_path=snapshot_path,
                first_seen=timestamp,
                last_seen=timestamp,
                duration_seconds=0.0,
                event_status="ACTIVE",
                is_update=False,
            )
        return None

    # ── Session Culling & Event Closure ───────────────────────────

    def cull_inactive_sessions(self, current_time: float) -> List[AnalyticsEvent]:
        """
        Inspects all active sessions. Closes sessions exceeding session_timeout.
        Returns final closure events for any closed active events.
        """
        closed_events: List[AnalyticsEvent] = []

        with self._lock:
            for (pid, cid), session in list(self._active_sessions.items()):
                if (current_time - session.last_seen) > self.session_timeout:
                    # Close session
                    self._close_session(session, current_time)
                    del self._active_sessions[(pid, cid)]

                    # Close corresponding active event
                    event_key = f"{pid}:{cid}:PERSON_DETECTED:{session.session_id}"
                    if event_key in self._active_events:
                        ev = self._active_events.pop(event_key)
                        ev.event_status = "CLOSED"
                        ev.last_seen = session.last_seen
                        ev.duration_seconds = session.duration_seconds
                        ev.metadata["status"] = "LEFT"
                        ev.metadata["presence_duration"] = int(session.duration_seconds)
                        ev.is_update = True
                        closed_events.append(ev)

        return closed_events

    # ── Trajectory & Profile Queries ──────────────────────────────

    def get_trajectory(self, person_id: str) -> List[Dict[str, Any]]:
        """Returns chronological trajectory segments for a person."""
        with self._lock:
            # Check in-memory segments
            in_mem = list(self._person_trajectories.get(person_id, []))
            # Also include any currently active session as an ongoing segment
            for (pid, cid), sess in self._active_sessions.items():
                if pid == person_id:
                    in_mem.append(
                        TrajectorySegment(
                            camera_id=cid,
                            entry_time=sess.first_seen,
                            exit_time=sess.last_seen,
                            duration_seconds=sess.duration_seconds,
                            tracker_ids=list(sess.tracker_ids),
                            session_id=sess.session_id,
                            entry_time_iso=sess.created_at_iso,
                            exit_time_iso=sess.updated_at_iso,
                        )
                    )

            if in_mem:
                in_mem.sort(key=lambda s: s.entry_time)
                return [s.to_dict() for s in in_mem]

            # Fallback to storage
            if self.storage and hasattr(self.storage, "get_ordered_trajectory_segments"):
                segments = self.storage.get_ordered_trajectory_segments(person_id)
                return [s.to_dict() for s in segments]

            return []

    def get_person_profile(self, person_id: str) -> Dict[str, Any]:
        """Returns complete operator profile for a persistent person."""
        with self._lock:
            trajectory = self.get_trajectory(person_id)
            total_duration = sum(s["duration_seconds"] for s in trajectory)
            cameras_visited = list(dict.fromkeys(s["camera_id"] for s in trajectory))

            current_cam = None
            status = "LEFT"
            for (pid, cid), sess in self._active_sessions.items():
                if pid == person_id:
                    current_cam = cid
                    status = "PRESENT"
                    break

            if current_cam is None and trajectory:
                current_cam = trajectory[-1]["camera_id"]

            return {
                "person_id": person_id,
                "identity_status": "unknown",
                "current_camera": current_cam,
                "status": status,
                "first_seen": trajectory[0]["entry_time_iso"] if trajectory else "",
                "last_seen": trajectory[-1]["exit_time_iso"] if trajectory else "",
                "cameras_visited": cameras_visited,
                "total_sessions": len(trajectory),
                "total_presence_duration": round(total_duration, 2),
                "trajectory": trajectory,
            }

    def clear(self):
        """Clears all in-memory states."""
        with self._lock:
            self._active_sessions.clear()
            self._active_events.clear()
            self._last_emitted_event_time.clear()
            self._person_last_camera.clear()
            self._person_trajectories.clear()
            self._track_to_person.clear()
