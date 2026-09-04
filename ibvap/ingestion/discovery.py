"""
ONVIF WS-Discovery Subsystem.
Scans local broadcast domains for ONVIF-compliant CCTV cameras and video transmitters.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("ibvap.ingestion.discovery")


class ONVIFDiscovery:
    """
    Manages dynamic network discovery of ONVIF IP camera devices.
    """

    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def discover(self, timeout: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Discovers ONVIF cameras on the local network via multicast WS-Discovery.
        """
        t = timeout if timeout is not None else self.timeout
        try:
            from wsdiscovery import WSDiscovery
        except ImportError:
            logger.warning("wsdiscovery not installed. Cannot perform network scan.")
            return []

        wsd = WSDiscovery()
        devices = []
        try:
            wsd.start()
            services = wsd.searchServices(timeout=t)
            for s in services:
                devices.append(dict(s.__dict__))
            return devices
        except Exception as e:
            logger.error(f"Error during WS-Discovery: {e}")
            return []
        finally:
            try:
                wsd.stop()
            except Exception:
                pass


def discover(timeout: float = 5.0) -> List[Dict[str, Any]]:
    """Convenience function matching root discovery.py signature."""
    scanner = ONVIFDiscovery(timeout=timeout)
    return scanner.discover()
