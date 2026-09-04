import asyncio
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)

async def connect_and_get_rtsp(device: Dict, username: str = "cam", passwd: str = "12345678") -> Optional[str]:
    """
    Connects to an ONVIF device and extracts the authenticated RTSP stream URI.
    Requires onvif-zeep.
    """
    xaddrs = device.get("xAddrs", "")
    if not xaddrs:
        return None

    try:
        parts = xaddrs.split("/")
        s = parts[2] if len(parts) > 2 else xaddrs
        host, _, port_str = s.partition(":")
        port = int(port_str) if port_str.isdigit() else 80
    except Exception as e:
        logger.error(f"Error parsing xAddrs {xaddrs}: {e}")
        return None

    try:
        from onvif import ONVIFCamera
        cam = ONVIFCamera(
            host=host,
            port=port,
            user=username,
            passwd=passwd,
            encrypt=True
        )
        await cam._devicemgmt_with_time()
        await cam.update_xaddrs()
        
        # Get profiles
        media = await cam.create_media_service()
        profiles = await media.GetProfiles()
        if not profiles:
            await cam.close()
            return None

        profile = profiles[0]
        req = media.create_type("GetStreamUri")
        req.ProfileToken = profile.token
        req.StreamSetup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"}
        }

        response = await media.GetStreamUri(req)
        uri = response.Uri

        # Inject credentials if not present
        if "@" not in uri and username:
            uri = uri.replace("rtsp://", f"rtsp://{username}:{passwd}@")
            
        await cam.close()
        return uri
    except AttributeError:
        logger.warning(f"ONVIF connection failed for {host}:{port}: Camera rejected credentials.")
        return None
    except Exception as e:
        logger.error(f"ONVIF connection failed for {host}:{port}: {e}")
        return None
