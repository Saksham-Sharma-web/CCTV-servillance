"""
Thread-safe RingBuffer and FrameQueue for frame ingestion.
Decouples high-frequency capture threads from inference loops.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class IngestionFrame:
    """Represents a single captured frame with temporal and sequence metadata."""
    frame: np.ndarray
    frame_index: int
    timestamp: float
    camera_id: str


class FrameRingBuffer:
    """
    Fixed-capacity, thread-safe circular frame buffer.
    Drops oldest frames when capacity is exceeded to prevent latency buildup.
    """

    def __init__(self, capacity: int = 30):
        if capacity < 1:
            raise ValueError("Capacity must be at least 1")
        self.capacity = capacity
        self._buffer = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._dropped_count = 0
        self._pushed_count = 0

    def push(self, frame: np.ndarray, frame_index: int, camera_id: str = "default", timestamp: Optional[float] = None) -> bool:
        """Push a new frame into the buffer, discarding oldest if full."""
        ts = timestamp if timestamp is not None else time.time()
        item = IngestionFrame(frame=frame, frame_index=frame_index, timestamp=ts, camera_id=camera_id)
        with self._lock:
            if len(self._buffer) >= self.capacity:
                self._dropped_count += 1
            self._buffer.append(item)
            self._pushed_count += 1
            self._not_empty.notify()
            return True

    def pop(self, timeout: Optional[float] = None) -> Optional[IngestionFrame]:
        """Pop the oldest available frame. Blocks up to timeout if empty."""
        with self._not_empty:
            if not self._buffer:
                if timeout is not None and timeout <= 0:
                    return None
                if not self._not_empty.wait(timeout=timeout):
                    return None
            if self._buffer:
                return self._buffer.popleft()
            return None

    def pop_latest(self) -> Optional[IngestionFrame]:
        """Pop the newest frame and discard any intermediate backlog to achieve zero-latency."""
        with self._lock:
            if not self._buffer:
                return None
            item = self._buffer.pop()
            dropped = len(self._buffer)
            self._dropped_count += dropped
            self._buffer.clear()
            return item

    def clear(self):
        with self._lock:
            self._buffer.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "size": len(self._buffer),
                "capacity": self.capacity,
                "pushed": self._pushed_count,
                "dropped": self._dropped_count,
            }
