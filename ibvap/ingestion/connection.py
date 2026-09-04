"""
Camera Connection Manager.
Handles authenticated ONVIF camera initial handshake and device management.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger("ibvap.ingestion.connection")


async def connect(device: Dict[str, Any], username: str = "cam", passwd: str = "12345678"):
    """
    Connects and authenticates with an ONVIF camera device.
    """
    try:
        from onvif import ONVIFCamera
    except ImportError:
        raise ImportError("onvif-zeep is required for ONVIF camera connections")

    xaddrs = device.get("_xAddrs", [])
    if not xaddrs:
        raise ValueError("Device dictionary does not contain _xAddrs")

    s = xaddrs[0].split("/")[2]
    host, _, port = s.partition(":")
    port_int = int(port) if port else 80

    cam = None
    try:
        cam = ONVIFCamera(host=host, port=port_int, user=username, passwd=passwd, encrypt=True)
        await cam._devicemgmt_with_time()
        await cam.update_xaddrs()
        logger.info(f"Connection to {host}:{port_int} successful!")
        return cam
    except Exception as e:
        if cam is not None:
            try:
                await cam.close()
            except Exception:
                pass
        raise e
