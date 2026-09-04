"""
IBVAP Event Engine Subsystem.
Standardizes, deduplicates, and debounces analytics alerts.
Prevents alert flooding across consecutive video frames.
"""

from typing import List, Dict, Tuple, Optional
import time
import logging

from ibvap.core.types import AnalyticsEvent, EventType
from ibvap.core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.events.engine")


class EventEngine:
    """
    Central event aggregator and debouncer.
    Filters raw candidate events against temporal deduplication windows.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.dedup_window = self.config.event_deduplication_window_seconds
        # Key: (camera_id, event_type, track_id, sub_key) -> float (last emitted timestamp)
        self.last_emitted: Dict[Tuple[str, str, Optional[int], Optional[str]], float] = {}

    def filter_and_emit(self, raw_events: List[AnalyticsEvent]) -> List[AnalyticsEvent]:
        """
        Deduplicates candidate events against recent history.

        Args:
            raw_events: List of candidate AnalyticsEvent instances.

        Returns:
            Filtered list of unique, debounced AnalyticsEvents.
        """
        if not raw_events:
            return []

        ref_time = raw_events[0].timestamp
        emitted_events: List[AnalyticsEvent] = []

        # Cull stale records older than 5x the deduplication window
        cutoff = ref_time - (self.dedup_window * 5.0)
        for key in list(self.last_emitted.keys()):
            if self.last_emitted[key] < cutoff:
                del self.last_emitted[key]

        for ev in raw_events:
            event_type_str = ev.event_type.value if isinstance(ev.event_type, EventType) else str(ev.event_type)
            # Secondary key for zones or plate numbers
            sub_key = ev.metadata.get("zone_id") or ev.metadata.get("plate_number") or None

            dedup_key = (ev.camera_id, event_type_str, ev.track_id, sub_key)
            last_time = self.last_emitted.get(dedup_key, 0.0)

            if (ev.timestamp - last_time) >= self.dedup_window:
                self.last_emitted[dedup_key] = ev.timestamp
                emitted_events.append(ev)
                logger.info(
                    f"[{ev.camera_id}] EMIT {event_type_str} | Track: {ev.track_id} | Identity: {ev.identity_id}"
                )

        return emitted_events

    def reset(self):
        self.last_emitted.clear()
