"""
PostgreSQL Event Logger.
Persists AnalyticsEvent records into cctv_analytics_events table.
Gracefully handles database disconnections without interrupting real-time video processing.
"""

from typing import List, Optional
import json
import logging

from ..core.types import AnalyticsEvent
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.integration.db")


class DatabaseEventLogger:
    """
    Asynchronously or synchronously writes events to PostgreSQL.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.enabled = self.config.db_enabled and bool(self.config.db_connection_url)
        self.conn = None

        if self.enabled:
            self._connect()

    def _connect(self):
        try:
            import psycopg2
            self.conn = psycopg2.connect(self.config.db_connection_url)
            self.conn.autocommit = True
            logger.info("Connected to PostgreSQL for CCTV analytics logging.")
        except Exception as e:
            logger.warning(f"Could not connect to PostgreSQL: {e}. DB logging disabled.")
            self.conn = None

    def log_events(self, events: List[AnalyticsEvent]):
        """
        Inserts events into cctv_analytics_events table.
        """
        if not self.enabled or self.conn is None or not events:
            return

        query = """
            INSERT INTO cctv_analytics_events (
                camera_id, event_type, track_id, identity_id, confidence, metadata, snapshot_url
            ) VALUES (%s, %s, %s, %s, %s, %s, %s);
        """

        try:
            with self.conn.cursor() as cur:
                for ev in events:
                    event_type_str = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                    cur.execute(query, (
                        ev.camera_id,
                        event_type_str,
                        ev.track_id,
                        ev.identity_id,
                        float(ev.confidence),
                        json.dumps(ev.metadata),
                        ev.snapshot_path,
                    ))
        except Exception as e:
            logger.error(f"Error inserting events into PostgreSQL: {e}")
            # Try to reconnect on failure
            try:
                self._connect()
            except Exception:
                pass
