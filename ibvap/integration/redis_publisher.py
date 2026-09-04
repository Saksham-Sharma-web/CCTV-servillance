"""
Redis / Valkey Pub/Sub Alert Publisher.
Publishes real-time JSON event alerts to Valkey/Redis channel 'cctv:alerts'.
The existing VibeestaBackend Node.js server subscribes to this channel and emits to Socket.IO clients.
Gracefully degrades if Redis is unavailable.
"""

from typing import List, Optional
import json
import logging

from ..core.types import AnalyticsEvent
from ..core.config import IBVAPConfig, default_config

logger = logging.getLogger("ibvap.integration.redis")


class RedisAlertPublisher:
    """
    Publishes events to Redis/Valkey pub/sub channel.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.enabled = self.config.redis_enabled
        self.channel = self.config.redis_channel
        self.client = None

        if self.enabled:
            self._connect()

    def _connect(self):
        try:
            import redis
            self.client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password or None,
                socket_timeout=0.5,
                socket_connect_timeout=0.5,
                decode_responses=True
            )
            self.client.ping()
            logger.info(f"Connected to Redis/Valkey at {self.config.redis_host}:{self.config.redis_port}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis/Valkey: {e}. Running with Redis publishing disabled.")
            self.client = None

    def publish_events(self, events: List[AnalyticsEvent]):
        """
        Publishes a list of events to the configured Redis pub/sub channel.
        """
        if not self.enabled or self.client is None or not events:
            return

        for ev in events:
            try:
                payload = json.dumps(ev.to_dict())
                self.client.publish(self.channel, payload)
            except Exception as e:
                logger.error(f"Failed to publish event to Redis channel '{self.channel}': {e}")
                # Don't crash frame processing
