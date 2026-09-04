"""
Unit tests for Ingestion subsystem:
- FrameRingBuffer
- FrameSampler
"""

import pytest
import numpy as np
from ibvap.ingestion.buffer import FrameRingBuffer
from ibvap.ingestion.sampler import FrameSampler


def test_frame_ring_buffer_fifo_and_drop():
    buffer = FrameRingBuffer(capacity=3)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    # Push 3 frames
    buffer.push(frame, frame_index=1, camera_id="c1", timestamp=1.0)
    buffer.push(frame, frame_index=2, camera_id="c1", timestamp=2.0)
    buffer.push(frame, frame_index=3, camera_id="c1", timestamp=3.0)
    assert buffer.size == 3
    assert buffer.stats["dropped"] == 0

    # Push 4th frame -> overflows, drops frame 1
    buffer.push(frame, frame_index=4, camera_id="c1", timestamp=4.0)
    assert buffer.size == 3
    assert buffer.stats["dropped"] == 1

    # Pop oldest should now be frame 2
    f = buffer.pop()
    assert f.frame_index == 2
    assert buffer.size == 2


def test_frame_ring_buffer_pop_latest():
    buffer = FrameRingBuffer(capacity=5)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    buffer.push(frame, frame_index=1)
    buffer.push(frame, frame_index=2)
    buffer.push(frame, frame_index=3)

    # pop_latest gets newest (3) and clears backlog
    latest = buffer.pop_latest()
    assert latest.frame_index == 3
    assert buffer.size == 0


def test_frame_sampler_stride():
    # 24 FPS source downsampled to 8 FPS target -> stride of 3
    sampler = FrameSampler(target_fps=8.0, source_fps=24.0, enabled=True)
    assert sampler.stride == 3

    # Frame 1 always processes
    assert sampler.should_process(timestamp=100.0, frame_index=1) is True
    # Frame 2 within same timestamp interval should skip
    assert sampler.should_process(timestamp=100.01, frame_index=2) is False
