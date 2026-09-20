"""
IBVAP Local SQLite Storage & Vector Index for Unknown Persons.
Provides standalone, zero-dependency persistence for cross-camera unknown person tracking.
Maintains an in-memory vector index for microsecond-latency Top-K cosine similarity search.
"""

from typing import List, Dict, Tuple, Optional, Any
import os
import time
import json
import sqlite3
import logging
import numpy as np

from ..core.types import UnknownPersonRecord, UnknownPersonSighting, PresenceSession, TrajectorySegment
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.tracking.unknown_storage")


class SQLiteUnknownStorage:
    """
    SQLite persistence layer and fast in-memory vector cache for unknown persons.
    """

    def __init__(self, db_path: Optional[str] = None, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        
        # Robust path normalization: anchor relative paths to the CCTV-servillance root
        if db_path is None:
            config_path = getattr(self.config, "unknown_storage_db_path", "data/unknown_persons.db")
            if not os.path.isabs(config_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.db_path = os.path.normpath(os.path.join(base_dir, config_path))
            else:
                self.db_path = config_path
        else:
            if not os.path.isabs(db_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                self.db_path = os.path.normpath(os.path.join(base_dir, db_path))
            else:
                self.db_path = db_path

        # In-memory vector cache:
        # matrix of shape (M, 512), float32, L2-normalized
        self._embedding_matrix: Optional[np.ndarray] = None
        # parallel list of unknown_ids for each row in matrix
        self._embedding_owners: List[str] = []
        # parallel list of embedding types ("prototype" or "representative")
        self._embedding_types: List[str] = []

        # In-memory entity records cache: unknown_id -> UnknownPersonRecord
        self._records: Dict[str, UnknownPersonRecord] = {}

        self._init_db()
        self._load_cache()

    def _get_connection(self) -> sqlite3.Connection:
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables for unknown persons and sightings."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unknown_persons (
                    unknown_id TEXT PRIMARY KEY,
                    first_seen_timestamp REAL NOT NULL,
                    last_seen_timestamp REAL NOT NULL,
                    first_camera_id TEXT NOT NULL,
                    last_camera_id TEXT NOT NULL,
                    prototype_embedding BLOB NOT NULL,
                    representative_embeddings BLOB,
                    camera_sequence TEXT NOT NULL,
                    created_at_iso TEXT,
                    updated_at_iso TEXT,
                    metadata TEXT
                );
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS unknown_person_sightings (
                    sighting_id TEXT PRIMARY KEY,
                    unknown_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    track_id INTEGER NOT NULL,
                    timestamp REAL NOT NULL,
                    timestamp_iso TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    similarity REAL DEFAULT 0.0,
                    face_quality REAL DEFAULT 1.0,
                    snapshot_path TEXT,
                    metadata TEXT,
                    FOREIGN KEY (unknown_id) REFERENCES unknown_persons(unknown_id)
                );
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sightings_unknown_id 
                ON unknown_person_sightings(unknown_id);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sightings_timestamp 
                ON unknown_person_sightings(timestamp);
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS person_presence_sessions (
                    session_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    first_seen REAL NOT NULL,
                    last_seen REAL NOT NULL,
                    duration_seconds REAL NOT NULL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    tracker_ids TEXT NOT NULL,
                    event_id TEXT,
                    created_at_iso TEXT,
                    updated_at_iso TEXT,
                    metadata TEXT,
                    FOREIGN KEY (person_id) REFERENCES unknown_persons(unknown_id)
                );
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_person_cam 
                ON person_presence_sessions(person_id, camera_id, status);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_last_seen 
                ON person_presence_sessions(last_seen);
            """)

            # Safe migrations for unknown_persons table
            for col_def in [
                "identity_status TEXT DEFAULT 'unknown'",
                "total_sessions INTEGER DEFAULT 1",
                "total_presence_duration REAL DEFAULT 0.0",
                "current_camera_id TEXT",
                "current_session_id TEXT"
            ]:
                try:
                    cursor.execute(f"ALTER TABLE unknown_persons ADD COLUMN {col_def};")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            conn.commit()

    def _load_cache(self):
        """Loads all unknown persons from SQLite into in-memory vector matrix."""
        self._records.clear()
        self._embedding_owners.clear()
        self._embedding_types.clear()
        all_embeddings = []

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM unknown_persons")
            rows = cursor.fetchall()
            for r in rows:
                unknown_id = r["unknown_id"]
                proto_bytes = r["prototype_embedding"]
                proto_emb = np.frombuffer(proto_bytes, dtype=np.float32).copy()

                rep_embs = []
                if r["representative_embeddings"]:
                    rep_data = np.frombuffer(r["representative_embeddings"], dtype=np.float32)
                    if len(rep_data) > 0 and len(rep_data) % 512 == 0:
                        rep_embs = [rep_data[i:i+512].copy() for i in range(0, len(rep_data), 512)]

                cam_seq = json.loads(r["camera_sequence"]) if r["camera_sequence"] else []
                meta = json.loads(r["metadata"]) if r["metadata"] else {}

                record = UnknownPersonRecord(
                    unknown_id=unknown_id,
                    first_seen_timestamp=float(r["first_seen_timestamp"]),
                    last_seen_timestamp=float(r["last_seen_timestamp"]),
                    first_camera_id=r["first_camera_id"],
                    last_camera_id=r["last_camera_id"],
                    prototype_embedding=proto_emb,
                    representative_embeddings=rep_embs,
                    sightings=[],
                    camera_sequence=cam_seq,
                    created_at_iso=r["created_at_iso"] or "",
                    updated_at_iso=r["updated_at_iso"] or "",
                    metadata=meta
                )
                self._records[unknown_id] = record

                # Add prototype to vector index
                all_embeddings.append(proto_emb)
                self._embedding_owners.append(unknown_id)
                self._embedding_types.append("prototype")

                # Add representatives to vector index
                for rep in rep_embs:
                    all_embeddings.append(rep)
                    self._embedding_owners.append(unknown_id)
                    self._embedding_types.append("representative")

        if all_embeddings:
            self._embedding_matrix = np.vstack(all_embeddings).astype(np.float32)
        else:
            self._embedding_matrix = None

        logger.info(f"Loaded {len(self._records)} unknown identities ({len(self._embedding_owners)} embeddings) into vector cache.")

    def search_top_k(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """
        Fast cosine similarity vector search across all cached unknown embeddings.
        Returns Top-K unique unknown individuals sorted by descending similarity score:
        [(unknown_id, max_similarity), ...]
        """
        if self._embedding_matrix is None or len(self._embedding_owners) == 0:
            return []

        q = query_embedding.astype(np.float32).flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Cosine similarity via matrix multiplication
        sims = np.dot(self._embedding_matrix, q)

        # Aggregate highest similarity per unique unknown_id
        best_per_person: Dict[str, float] = {}
        for idx, score in enumerate(sims):
            uid = self._embedding_owners[idx]
            f_score = float(score)
            if uid not in best_per_person or f_score > best_per_person[uid]:
                best_per_person[uid] = f_score

        # Sort descending by similarity
        sorted_candidates = sorted(best_per_person.items(), key=lambda x: x[1], reverse=True)
        return sorted_candidates[:k]

    def save_new_person(self, record: UnknownPersonRecord, initial_sighting: Optional[UnknownPersonSighting] = None):
        """Persists a brand new unknown person record and initial sighting."""
        self._records[record.unknown_id] = record

        # Add to vector cache
        new_embs = [record.prototype_embedding]
        self._embedding_owners.append(record.unknown_id)
        self._embedding_types.append("prototype")

        for rep in record.representative_embeddings:
            new_embs.append(rep)
            self._embedding_owners.append(record.unknown_id)
            self._embedding_types.append("representative")

        if self._embedding_matrix is not None:
            self._embedding_matrix = np.vstack([self._embedding_matrix] + new_embs).astype(np.float32)
        else:
            self._embedding_matrix = np.vstack(new_embs).astype(np.float32)

        proto_bytes = record.prototype_embedding.astype(np.float32).tobytes()
        rep_bytes = b"".join(r.astype(np.float32).tobytes() for r in record.representative_embeddings)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO unknown_persons (
                    unknown_id, first_seen_timestamp, last_seen_timestamp,
                    first_camera_id, last_camera_id, prototype_embedding,
                    representative_embeddings, camera_sequence, created_at_iso,
                    updated_at_iso, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                record.unknown_id,
                record.first_seen_timestamp,
                record.last_seen_timestamp,
                record.first_camera_id,
                record.last_camera_id,
                proto_bytes,
                rep_bytes,
                json.dumps(record.camera_sequence),
                record.created_at_iso,
                record.updated_at_iso,
                json.dumps(record.metadata),
            ))

            if initial_sighting:
                record.sightings.append(initial_sighting)
                cursor.execute("""
                    INSERT OR REPLACE INTO unknown_person_sightings (
                        sighting_id, unknown_id, camera_id, track_id,
                        timestamp, timestamp_iso, bbox, similarity,
                        face_quality, snapshot_path, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    initial_sighting.sighting_id,
                    initial_sighting.unknown_id,
                    initial_sighting.camera_id,
                    initial_sighting.track_id,
                    initial_sighting.timestamp,
                    initial_sighting.timestamp_iso,
                    json.dumps(list(initial_sighting.bbox)),
                    float(initial_sighting.similarity),
                    float(initial_sighting.face_quality),
                    initial_sighting.snapshot_path,
                    json.dumps(initial_sighting.metadata),
                ))

            conn.commit()

    def update_person(
        self,
        record: UnknownPersonRecord,
        new_sighting: Optional[UnknownPersonSighting] = None,
        new_representative: Optional[np.ndarray] = None
    ):
        """Updates an existing unknown person record, sightings, and representative embeddings."""
        self._records[record.unknown_id] = record

        if new_sighting:
            record.sightings.append(new_sighting)

        # If a new representative embedding is provided, update vector cache
        if new_representative is not None:
            max_reps = getattr(self.config, "unknown_max_representatives_per_person", 5)
            if len(record.representative_embeddings) < max_reps:
                record.representative_embeddings.append(new_representative)
            else:
                # Replace the oldest representative (FIFO)
                record.representative_embeddings.pop(0)
                record.representative_embeddings.append(new_representative)

            # Rebuild vector cache to maintain exact indices
            self._rebuild_vector_cache()

        proto_bytes = record.prototype_embedding.astype(np.float32).tobytes()
        rep_bytes = b"".join(r.astype(np.float32).tobytes() for r in record.representative_embeddings)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE unknown_persons SET
                    last_seen_timestamp = ?,
                    last_camera_id = ?,
                    representative_embeddings = ?,
                    camera_sequence = ?,
                    updated_at_iso = ?,
                    metadata = ?
                WHERE unknown_id = ?;
            """, (
                record.last_seen_timestamp,
                record.last_camera_id,
                rep_bytes,
                json.dumps(record.camera_sequence),
                record.updated_at_iso,
                json.dumps(record.metadata),
                record.unknown_id
            ))

            if new_sighting:
                cursor.execute("""
                    INSERT OR REPLACE INTO unknown_person_sightings (
                        sighting_id, unknown_id, camera_id, track_id,
                        timestamp, timestamp_iso, bbox, similarity,
                        face_quality, snapshot_path, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (
                    new_sighting.sighting_id,
                    new_sighting.unknown_id,
                    new_sighting.camera_id,
                    new_sighting.track_id,
                    new_sighting.timestamp,
                    new_sighting.timestamp_iso,
                    json.dumps(list(new_sighting.bbox)),
                    float(new_sighting.similarity),
                    float(new_sighting.face_quality),
                    new_sighting.snapshot_path,
                    json.dumps(new_sighting.metadata),
                ))

            conn.commit()

    def _rebuild_vector_cache(self):
        """Reconstructs the embedding matrix from the in-memory records dictionary."""
        all_embeddings = []
        self._embedding_owners.clear()
        self._embedding_types.clear()

        for uid, rec in self._records.items():
            all_embeddings.append(rec.prototype_embedding)
            self._embedding_owners.append(uid)
            self._embedding_types.append("prototype")

            for rep in rec.representative_embeddings:
                all_embeddings.append(rep)
                self._embedding_owners.append(uid)
                self._embedding_types.append("representative")

        if all_embeddings:
            self._embedding_matrix = np.vstack(all_embeddings).astype(np.float32)
        else:
            self._embedding_matrix = None

    def get_record(self, unknown_id: str) -> Optional[UnknownPersonRecord]:
        """Retrieves a cached record by unknown_id."""
        return self._records.get(unknown_id)

    def get_sightings(self, unknown_id: str) -> List[UnknownPersonSighting]:
        """Loads all historical sightings for an unknown person from SQLite."""
        sightings = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM unknown_person_sightings 
                WHERE unknown_id = ? 
                ORDER BY timestamp ASC;
            """, (unknown_id,))
            for r in cursor.fetchall():
                bbox_list = json.loads(r["bbox"]) if r["bbox"] else [0, 0, 0, 0]
                meta = json.loads(r["metadata"]) if r["metadata"] else {}
                sightings.append(
                    UnknownPersonSighting(
                        sighting_id=r["sighting_id"],
                        unknown_id=r["unknown_id"],
                        camera_id=r["camera_id"],
                        track_id=r["track_id"],
                        timestamp=float(r["timestamp"]),
                        timestamp_iso=r["timestamp_iso"],
                        bbox=tuple(bbox_list),
                        similarity=float(r["similarity"]),
                        face_quality=float(r["face_quality"]),
                        snapshot_path=r["snapshot_path"],
                        metadata=meta
                    )
                )
        return sightings

    def list_all_records(self) -> List[UnknownPersonRecord]:
        """Returns all tracked unknown person records."""
        return list(self._records.values())

    def save_session(self, session: PresenceSession):
        """Persists a new presence session to SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO person_presence_sessions (
                    session_id, person_id, camera_id, first_seen, last_seen,
                    duration_seconds, status, tracker_ids, event_id,
                    created_at_iso, updated_at_iso, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                session.session_id,
                session.person_id,
                session.camera_id,
                session.first_seen,
                session.last_seen,
                session.duration_seconds,
                session.status,
                json.dumps(session.tracker_ids),
                session.event_id,
                session.created_at_iso,
                session.updated_at_iso,
                json.dumps(session.metadata),
            ))
            conn.commit()

    def update_session(self, session: PresenceSession):
        """Updates an existing presence session in SQLite."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE person_presence_sessions SET
                    last_seen = ?,
                    duration_seconds = ?,
                    status = ?,
                    tracker_ids = ?,
                    event_id = ?,
                    updated_at_iso = ?,
                    metadata = ?
                WHERE session_id = ?;
            """, (
                session.last_seen,
                session.duration_seconds,
                session.status,
                json.dumps(session.tracker_ids),
                session.event_id,
                session.updated_at_iso,
                json.dumps(session.metadata),
                session.session_id,
            ))
            conn.commit()

    def get_active_session(self, person_id: str, camera_id: str) -> Optional[PresenceSession]:
        """Retrieves currently active presence session for a person on a camera."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM person_presence_sessions
                WHERE person_id = ? AND camera_id = ? AND status = 'ACTIVE'
                ORDER BY last_seen DESC LIMIT 1;
            """, (person_id, camera_id))
            row = cursor.fetchone()
            if not row:
                return None
            return PresenceSession(
                session_id=row["session_id"],
                person_id=row["person_id"],
                camera_id=row["camera_id"],
                first_seen=float(row["first_seen"]),
                last_seen=float(row["last_seen"]),
                duration_seconds=float(row["duration_seconds"]),
                status=row["status"],
                tracker_ids=json.loads(row["tracker_ids"]) if row["tracker_ids"] else [],
                event_id=row["event_id"],
                created_at_iso=row["created_at_iso"] or "",
                updated_at_iso=row["updated_at_iso"] or "",
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
            )

    def list_sessions_for_person(self, person_id: str) -> List[PresenceSession]:
        """Loads all presence sessions for a person in chronological order."""
        sessions = []
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM person_presence_sessions
                WHERE person_id = ?
                ORDER BY first_seen ASC;
            """, (person_id,))
            for r in cursor.fetchall():
                sessions.append(
                    PresenceSession(
                        session_id=r["session_id"],
                        person_id=r["person_id"],
                        camera_id=r["camera_id"],
                        first_seen=float(r["first_seen"]),
                        last_seen=float(r["last_seen"]),
                        duration_seconds=float(r["duration_seconds"]),
                        status=r["status"],
                        tracker_ids=json.loads(r["tracker_ids"]) if r["tracker_ids"] else [],
                        event_id=r["event_id"],
                        created_at_iso=r["created_at_iso"] or "",
                        updated_at_iso=r["updated_at_iso"] or "",
                        metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                    )
                )
        return sessions

    def get_ordered_trajectory_segments(self, person_id: str) -> List[TrajectorySegment]:
        """Returns ordered camera presence segments for a person."""
        sessions = self.list_sessions_for_person(person_id)
        segments = []
        for s in sessions:
            segments.append(
                TrajectorySegment(
                    camera_id=s.camera_id,
                    entry_time=s.first_seen,
                    exit_time=s.last_seen,
                    duration_seconds=s.duration_seconds,
                    tracker_ids=list(s.tracker_ids),
                    session_id=s.session_id,
                    entry_time_iso=s.created_at_iso,
                    exit_time_iso=s.updated_at_iso,
                )
            )
        return segments

    def clear(self):
        """Clears in-memory cache and SQLite tables."""
        self._records.clear()
        self._embedding_owners.clear()
        self._embedding_types.clear()
        self._embedding_matrix = None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM person_presence_sessions;")
            cursor.execute("DELETE FROM unknown_person_sightings;")
            cursor.execute("DELETE FROM unknown_persons;")
            conn.commit()
