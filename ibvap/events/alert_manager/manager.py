"""
Alert Manager Subsystem.
Prioritizes, categorizes, routes, and dispatches surveillance alerts by severity level.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Optional, Any
import time
import logging

from ibvap.core.types import AnalyticsEvent, EventType

logger = logging.getLogger("ibvap.events.alert_manager")


class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class Alert:
    """
    Standardized dispatched alert wrapper.
    """
    alert_id: str
    severity: AlertSeverity
    event: AnalyticsEvent
    dispatched_at: float = field(default_factory=time.time)
    acknowledged: bool = False


class AlertManager:
    """
    Dispatches analytics events to configured listeners based on severity rules.
    """

    def __init__(self):
        self._handlers: Dict[AlertSeverity, List[Callable[[Alert], None]]] = {
            AlertSeverity.INFO: [],
            AlertSeverity.WARNING: [],
            AlertSeverity.CRITICAL: [],
        }
        self._active_alerts: List[Alert] = []

    def register_handler(self, severity: AlertSeverity, handler: Callable[[Alert], None]):
        """Registers a callback handler for a specific severity tier."""
        self._handlers[severity].append(handler)

    def classify_severity(self, event: AnalyticsEvent) -> AlertSeverity:
        """Determines alert severity based on event type and metadata."""
        etype = event.event_type
        if etype == EventType.FENCE_INTRUSION:
            return AlertSeverity.CRITICAL
        elif etype in (EventType.BLACKLISTED_VEHICLE, EventType.SUSPICIOUS_ACTIVITY):
            return AlertSeverity.CRITICAL
        elif etype in (EventType.LOITERING, EventType.NIGHT_MOVEMENT, EventType.LINE_CROSSING):
            return AlertSeverity.WARNING
        return AlertSeverity.INFO

    def dispatch(self, event: AnalyticsEvent) -> Alert:
        """Classifies and dispatches an event to registered listeners."""
        severity = self.classify_severity(event)
        alert = Alert(alert_id=event.event_id, severity=severity, event=event)
        self._active_alerts.append(alert)

        handlers = self._handlers.get(severity, [])
        for handler in handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")

        return alert

    def acknowledge(self, alert_id: str) -> bool:
        """Marks an active alert as acknowledged."""
        for a in self._active_alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                return True
        return False

    def get_unacknowledged(self) -> List[Alert]:
        """Returns all unacknowledged alerts."""
        return [a for a in self._active_alerts if not a.acknowledged]
