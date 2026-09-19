"""
IBVAP API Subsystem.
Provides REST and WebSocket endpoints for camera telemetry, alert queries, and system status.
Compatible with FastAPI / Starlette if installed, or direct dictionary formatting.
"""

from typing import Dict, Any, List, Optional
import time
import json

from ibvap.core.types import PipelineResult, AnalyticsEvent


class IBVAPApiRouter:
    """
    API integration router for exposing surveillance analytics to external gateways.
    """

    def __init__(self, pipeline_ref: Optional[Any] = None):
        self.pipeline_ref = pipeline_ref
        self._cached_alerts: List[Dict[str, Any]] = []

    def get_health(self) -> Dict[str, Any]:
        """Returns engine health status and timestamp."""
        return {
            "status": "HEALTHY",
            "timestamp": time.time(),
            "subsystems": {
                "ingestion": "UP",
                "detection": "UP",
                "tracking": "UP",
                "face": "UP",
                "vehicle": "UP",
                "behavior": "UP",
                "appearance": "UP",
                "events": "UP",
                "output": "UP",
            }
        }

    def format_telemetry(self, result: PipelineResult) -> Dict[str, Any]:
        """Converts PipelineResult into standard JSON-serializable telemetry dictionary."""
        return result.to_dict()

    def record_alert(self, event: AnalyticsEvent):
        """Buffers an alert for external polling."""
        payload = event.to_dict() if hasattr(event, "to_dict") else asdict(event)
        self._cached_alerts.append(payload)
        if len(self._cached_alerts) > 500:
            self._cached_alerts.pop(0)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Returns the most recent alert events."""
        return self._cached_alerts[-limit:]

    def register_fastapi_routes(self, app):
        """
        Optional helper to mount standard REST routes onto an existing FastAPI app.
        """
        try:
            from fastapi import APIRouter
            router = APIRouter(prefix="/api/ibvap", tags=["IBVAP"])

            @router.get("/health")
            async def health():
                return self.get_health()

            @router.get("/alerts")
            async def alerts(limit: int = 50):
                return self.get_recent_alerts(limit=limit)

            app.include_router(router)
            return True
        except ImportError:
            return False
