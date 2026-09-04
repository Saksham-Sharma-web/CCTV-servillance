"""
Frame Sampler Module (Phase 7).
Decouples high-frame-rate camera ingestion (e.g. 24-30 FPS) from downstream
analytical computer vision processing (e.g. ~8 FPS).

Key Invariants:
1. Prevents high-frequency camera ingestion from saturating CPU with heavy inference.
2. Preserves frame indices, original source timestamps, and sequence continuity.
3. Configurable target analysis FPS (Initial Engineering Default: 8.0 FPS).
4. Thread-safe, lightweight, sub-microsecond evaluation.
"""

from typing import Optional
import time
import logging

logger = logging.getLogger("ibvap.core.sampler")


class FrameSampler:
    """
    Manages temporal frame sampling between high-FPS camera ingestion and analytical processing.
    """

    def __init__(
        self,
        target_fps: float = 8.0,
        source_fps: float = 24.0,
        enabled: bool = True,
    ):
        """
        Initializes the frame sampler.

        Args:
            target_fps: Desired analytical processing rate in frames per second (default: 8.0).
                        NOTE: INITIAL ENGINEERING DEFAULT. REQUIRES REAL-WORLD VALIDATION.
            source_fps: Nominal incoming camera capture rate in FPS (default: 24.0).
            enabled: If False, should_process() always returns True (e.g. for unit test compatibility).
        """
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}")
        if source_fps <= 0:
            raise ValueError(f"source_fps must be > 0, got {source_fps}")

        self.target_fps = float(target_fps)
        self.source_fps = float(source_fps)
        self.enabled = enabled

        self.frame_interval = 1.0 / self.target_fps
        self.stride = max(1, int(round(self.source_fps / self.target_fps)))

        self.last_processed_time: Optional[float] = None
        self.total_received: int = 0
        self.total_processed: int = 0

    def should_process(
        self,
        timestamp: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> bool:
        """
        Determines whether the current frame should be processed by analytical pipelines.

        Args:
            timestamp: Epoch timestamp of the current frame in seconds.
            frame_index: Sequential 1-indexed frame number from camera capture.

        Returns:
            bool: True if frame should be analyzed; False if it should be skipped.
        """
        self.total_received += 1

        if not self.enabled:
            self.total_processed += 1
            return True

        # First frame is always analyzed
        if self.last_processed_time is None or self.total_received == 1:
            self.last_processed_time = timestamp if timestamp is not None else time.time()
            self.total_processed += 1
            return True

        current_time = timestamp if timestamp is not None else time.time()

        # Check temporal elapsed interval
        elapsed = current_time - self.last_processed_time
        if elapsed >= (self.frame_interval - 0.002):  # 2ms tolerance for timer jitter
            self.last_processed_time = current_time
            self.total_processed += 1
            return True

        # Secondary fallback: check frame index stride if timestamps are identical/synthesized
        if frame_index is not None and self.stride > 1:
            if frame_index % self.stride == 0:
                self.last_processed_time = current_time
                self.total_processed += 1
                return True

        return False

    def reset(self) -> None:
        """Resets sampling state across camera reconnects."""
        self.last_processed_time = None
        self.total_received = 0
        self.total_processed = 0
