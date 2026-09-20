"""
IBVAP Event Engine.
Standardizes, deduplicates, and debounces analytics alerts.
Prevents alert flooding across consecutive video frames.
"""

from typing import List, Dict, Tuple, Optional, Any
import time
import logging

from ..core.types import AnalyticsEvent, EventType
from ..core.config import IBVAPConfig, default_config
from .correlator import PersonEventCorrelator

logger = logging.getLogger("ibvap.events")


class EventEngine:
    """
    Central event aggregator, debouncer, and person-centric correlation engine.
    Suppresses duplicate frame-level observations, quashes loitering into live duration,
    and preserves high-priority security alerts.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None, storage: Optional[Any] = None):
        self.config = config or default_config
        self.dedup_window = self.config.event_deduplication_window_seconds
        # Key: (camera_id, event_type, track_id, sub_key) -> float (last emitted timestamp)
        self.last_emitted: Dict[Tuple[str, str, Optional[int], Optional[str]], float] = {}

        # Person-centric event correlation layer
        self.correlator = PersonEventCorrelator(config=self.config, storage=storage)

    def filter_and_emit(self, raw_events: List[AnalyticsEvent]) -> List[AnalyticsEvent]:
        """
        Correlates and deduplicates candidate events against in-memory presence sessions.

        Args:
            raw_events: List of candidate AnalyticsEvent instances.

        Returns:
            Filtered list of consolidated, debounced AnalyticsEvents.
        """
        if not raw_events:
            return []

        ref_time = raw_events[0].timestamp
        emitted_events: List[AnalyticsEvent] = []

        # Check for any timed-out sessions that should be closed
        closed_events = self.correlator.cull_inactive_sessions(ref_time)
        emitted_events.extend(closed_events)

        # If person identity correlation is enabled, route through PersonEventCorrelator
        if getattr(self.config, "person_identity_enabled", True):
            for ev in raw_events:
                pid = ev.person_id or ev.identity_id or ev.metadata.get("unknown_id")
                corr_event = self.correlator.correlate_event(
                    camera_id=ev.camera_id,
                    event_type=ev.event_type,
                    timestamp=ev.timestamp,
                    person_id=pid,
                    track_id=ev.track_id,
                    confidence=ev.confidence,
                    metadata=ev.metadata,
                    snapshot_path=ev.snapshot_path,
                )
                if corr_event is not None:
                    emitted_events.append(corr_event)
            return emitted_events

        # Fallback legacy debouncing
        cutoff = ref_time - (self.dedup_window * 5.0)
        for key in list(self.last_emitted.keys()):
            if self.last_emitted[key] < cutoff:
                del self.last_emitted[key]

        for ev in raw_events:
            event_type_str = ev.event_type.value if isinstance(ev.event_type, EventType) else str(ev.event_type)
            sub_key = (
                ev.metadata.get("zone_id")
                or ev.metadata.get("plate_number")
                or ev.metadata.get("unknown_id")
                or None
            )

            dedup_key = (ev.camera_id, event_type_str, ev.track_id, sub_key)
            last_time = self.last_emitted.get(dedup_key, 0.0)

            if (ev.timestamp - last_time) >= self.dedup_window:
                self.last_emitted[dedup_key] = ev.timestamp
                emitted_events.append(ev)
                logger.debug(
                    f"[{ev.camera_id}] EMIT {event_type_str} | Track: {ev.track_id} | Identity: {ev.identity_id}"
                )

        return emitted_events

    def reset(self):
        self.last_emitted.clear()
        self.correlator.clear()

