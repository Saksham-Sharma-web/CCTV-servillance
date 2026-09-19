"""
IBVAP Cross-Camera Tracking Subsystem.
Maintains persistent cross-camera identity association (global_track_id) across distinct video streams.

CRITICAL ARCHITECTURAL INVARIANT:
- Cross-camera tracking is strictly READ-ONLY regarding camera configurations.
- Cross-camera tracking MUST NEVER modify, transfer, infer, or copy regions, borders,
  virtual lines, or event rules between cameras.
- Each camera's physical view and operational boundaries remain solely determined by the admin.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import time
import uuid

from ..core.types import Track, AnalyticsEvent, EventType
from ..core.config import IBVAPConfig, default_config


@dataclass
class CrossCameraEntity:
    """
    Global tracked entity spanning multiple camera views.
    """
    global_track_id: str
    primary_class: str
    identity_id: Optional[str] = None
    identity_name: Optional[str] = None
    plate_number: Optional[str] = None
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    current_camera_id: Optional[str] = None
    current_track_id: Optional[int] = None
    camera_sequence: List[str] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CrossCameraTracker:
    """
    Manages global multi-camera object association.
    Associates local tracks from disparate camera streams using:
    1. Biometric verification identity (identity_id)
    2. ANPR license plate number (plate_number)
    3. Spatial-temporal handoff heuristics

    Guaranteed Invariant:
    Never interacts with, reads for alteration, or modifies CameraConfig.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, association_timeout_seconds: float = 300.0):
        self.config = config or default_config
        self.association_timeout = association_timeout_seconds

        # Map: global_track_id -> CrossCameraEntity
        self.entities: Dict[str, CrossCameraEntity] = {}

        # Lookup indexes for rapid cross-camera correlation
        # identity_id -> global_track_id
        self._identity_index: Dict[str, str] = {}
        # plate_number -> global_track_id
        self._plate_index: Dict[str, str] = {}
        # (camera_id, local_track_id) -> global_track_id
        self._local_track_index: Dict[Tuple[str, int], str] = {}

    def associate_tracks(
        self,
        camera_id: str,
        tracks: List[Track],
        timestamp: Optional[float] = None
    ) -> List[Track]:
        """
        Assigns global_track_id to tracks in the current camera stream.
        Maintains cross-camera movement history without touching camera configs.
        """
        now = timestamp if timestamp is not None else time.time()
        self._cleanup_stale_entities(now)

        for track in tracks:
            local_key = (camera_id, track.track_id)
            matched_global_id: Optional[str] = None

            # 1. Existing local binding in current camera session
            if local_key in self._local_track_index:
                matched_global_id = self._local_track_index[local_key]

            # 2. Association by Biometric Identity (Person)
            elif track.identity_id and track.identity_id in self._identity_index:
                matched_global_id = self._identity_index[track.identity_id]

            # 3. Association by License Plate (Vehicle)
            elif track.plate_number and track.plate_number in self._plate_index:
                matched_global_id = self._plate_index[track.plate_number]

            # If match found, update existing entity
            if matched_global_id and matched_global_id in self.entities:
                entity = self.entities[matched_global_id]
                entity.last_seen_timestamp = now
                entity.current_camera_id = camera_id
                entity.current_track_id = track.track_id
                if not entity.camera_sequence or entity.camera_sequence[-1] != camera_id:
                    entity.camera_sequence.append(camera_id)
                if track.identity_id and not entity.identity_id:
                    entity.identity_id = track.identity_id
                    entity.identity_name = track.identity_name
                    self._identity_index[track.identity_id] = matched_global_id
                if track.plate_number and not entity.plate_number:
                    entity.plate_number = track.plate_number
                    self._plate_index[track.plate_number] = matched_global_id

                track.global_track_id = matched_global_id
                self._local_track_index[local_key] = matched_global_id

            else:
                # Create new CrossCameraEntity
                new_gid = f"global-{camera_id}-{track.track_id}-{uuid.uuid4().hex[:6]}"
                new_entity = CrossCameraEntity(
                    global_track_id=new_gid,
                    primary_class=track.class_name,
                    identity_id=track.identity_id,
                    identity_name=track.identity_name,
                    plate_number=track.plate_number,
                    first_seen_timestamp=now,
                    last_seen_timestamp=now,
                    current_camera_id=camera_id,
                    current_track_id=track.track_id,
                    camera_sequence=[camera_id],
                    confidence=track.confidence,
                )
                self.entities[new_gid] = new_entity
                self._local_track_index[local_key] = new_gid
                if track.identity_id:
                    self._identity_index[track.identity_id] = new_gid
                if track.plate_number:
                    self._plate_index[track.plate_number] = new_gid

                track.global_track_id = new_gid

        return tracks

    def get_camera_sequence(self, global_track_id: str) -> List[str]:
        """Returns the chronological sequence of cameras this entity traversed."""
        if global_track_id in self.entities:
            return list(self.entities[global_track_id].camera_sequence)
        return []

    def get_entity(self, global_track_id: str) -> Optional[CrossCameraEntity]:
        return self.entities.get(global_track_id)

    def _cleanup_stale_entities(self, now: float):
        """Removes entities that have not been seen for longer than association_timeout."""
        cutoff = now - self.association_timeout
        stale_gids = [
            gid for gid, ent in self.entities.items()
            if ent.last_seen_timestamp < cutoff
        ]
        for gid in stale_gids:
            ent = self.entities.pop(gid)
            if ent.identity_id and self._identity_index.get(ent.identity_id) == gid:
                del self._identity_index[ent.identity_id]
            if ent.plate_number and self._plate_index.get(ent.plate_number) == gid:
                del self._plate_index[ent.plate_number]

        # Cull stale local keys
        for key in list(self._local_track_index.keys()):
            if self._local_track_index[key] in stale_gids:
                del self._local_track_index[key]

    def clear(self):
        self.entities.clear()
        self._identity_index.clear()
        self._plate_index.clear()
        self._local_track_index.clear()
