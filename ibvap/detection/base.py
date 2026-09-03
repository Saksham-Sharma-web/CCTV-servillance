"""
Abstract Base Object Detector Interface.
Ensures the rest of IBVAP is completely decoupled from any specific detection model or vendor.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
from ..core.types import Detection


class BaseObjectDetector(ABC):
    """
    Abstract contract for object detectors.
    Accepts an uncompressed BGR numpy array and returns standard Detection instances.
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a single BGR frame.

        Args:
            frame: numpy.ndarray, shape (height, width, 3), dtype uint8, color format BGR.

        Returns:
            List of Detection instances with bounding boxes in (x1, y1, x2, y2) coordinates.
        """
        pass
