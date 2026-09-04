import logging
from typing import Dict, List, Optional
from ibvap.camera.models import CameraConfig, CameraSession, CameraStatus
from ibvap.camera.rtsp import RTSPSource

logger = logging.getLogger(__name__)

class CameraManager:
    """
    Manages the lifecycle of multiple camera sources.
    """
    def __init__(self):
        self.configs: Dict[str, CameraConfig] = {}
        self.sources: Dict[str, RTSPSource] = {}
        self.sessions: Dict[str, CameraSession] = {}

    def add_camera(self, config: CameraConfig) -> bool:
        """Registers a camera configuration."""
        if config.id in self.configs:
            logger.warning(f"Camera {config.id} already exists.")
            return False
            
        self.configs[config.id] = config
        self.sessions[config.id] = CameraSession(config=config, status=CameraStatus.STOPPED)
        logger.info(f"Registered camera: {config.name} ({config.id})")
        return True

    def remove_camera(self, camera_id: str):
        """Stops and removes a camera."""
        if camera_id in self.configs:
            self.stop_camera(camera_id)
            del self.configs[camera_id]
            del self.sessions[camera_id]
            logger.info(f"Removed camera: {camera_id}")

    def start_camera(self, camera_id: str) -> bool:
        """Starts the ingestion stream for a registered camera."""
        if camera_id not in self.configs:
            logger.error(f"Cannot start unknown camera: {camera_id}")
            return False

        if camera_id in self.sources:
            logger.warning(f"Camera {camera_id} is already running.")
            return True

        config = self.configs[camera_id]
        
        # Determine source (for now we assume RTSP or USB)
        source = RTSPSource(config=config)
        self.sources[camera_id] = source
        self.sessions[camera_id].status = CameraStatus.CONNECTING
        self.sessions[camera_id].health = source.health
        
        source.start()
        logger.info(f"Started stream for camera: {camera_id}")
        return True

    def stop_camera(self, camera_id: str):
        """Stops the ingestion stream."""
        if camera_id in self.sources:
            self.sources[camera_id].stop()
            del self.sources[camera_id]
            self.sessions[camera_id].status = CameraStatus.STOPPED
            logger.info(f"Stopped stream for camera: {camera_id}")

    def get_source(self, camera_id: str) -> Optional[RTSPSource]:
        """Returns the active RTSPSource if running."""
        return self.sources.get(camera_id)

    def list_cameras(self) -> List[CameraSession]:
        """Returns a list of all camera sessions and their statuses."""
        # Update session status based on health
        for cid, source in self.sources.items():
            if source.health.is_active:
                self.sessions[cid].status = CameraStatus.ONLINE
            else:
                self.sessions[cid].status = CameraStatus.CONNECTING
                
        return list(self.sessions.values())

    def shutdown(self):
        """Stops all running cameras."""
        logger.info("Shutting down CameraManager...")
        for cid in list(self.sources.keys()):
            self.stop_camera(cid)
