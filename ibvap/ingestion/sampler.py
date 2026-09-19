"""
Ingestion Frame Sampler.
Controls temporal downsampling between camera ingestion (24-30 FPS) and analytical processing.
"""

from ibvap.core.sampler import FrameSampler

__all__ = ["FrameSampler"]
