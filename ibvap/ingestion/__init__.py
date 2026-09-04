"""
IBVAP Ingestion Subsystem.
Provides discovery, authenticated streaming, ring buffering, and frame sampling.
"""

from .buffer import FrameRingBuffer, IngestionFrame
from .sampler import FrameSampler
from .discovery import ONVIFDiscovery, discover
from .connection import connect
from .rtsp_client import RTSPStreamClient, resolve_rtsp_url

__all__ = [
    "FrameRingBuffer",
    "IngestionFrame",
    "FrameSampler",
    "ONVIFDiscovery",
    "discover",
    "connect",
    "RTSPStreamClient",
    "resolve_rtsp_url",
]
