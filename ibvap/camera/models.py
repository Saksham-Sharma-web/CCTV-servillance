from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import numpy as np

class CameraStatus(Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    ERROR = "ERROR"
    STOPPED = "STOPPED"

class SourceType(Enum):
    ONVIF = "onvif"
    RTSP = "rtsp"
    USB = "usb"
    FILE = "file"
    PHONE = "phone"

@dataclass
class CameraConfig:
    """Configuration for a specific camera source."""
    id: str
    name: str
    location: str
    source_type: SourceType
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StreamHealth:
    """Metrics regarding the health of a camera stream."""
    is_active: bool = False
    fps_current: float = 0.0
    dropped_frames: int = 0
    reconnect_count: int = 0
    last_frame_timestamp: float = 0.0
    error_message: Optional[str] = None

@dataclass
class FramePacket:
    """A single frame packet coming from the stream manager."""
    camera_id: str
    timestamp: float
    frame: np.ndarray  # BGR frame (H, W, 3)
    frame_index: int

@dataclass
class CameraSession:
    """Represents a live session of an active camera."""
    config: CameraConfig
    status: CameraStatus = CameraStatus.STOPPED
    health: StreamHealth = field(default_factory=StreamHealth)
