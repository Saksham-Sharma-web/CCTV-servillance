import json
import logging
from typing import List, Dict
from wsdiscovery import WSDiscovery

logger = logging.getLogger(__name__)

def discover(timeout: int = 3) -> List[Dict]:
    """
    Discovers ONVIF devices on the local network using WS-Discovery.
    Returns a list of device dictionaries containing xAddrs.
    """
    wsd = WSDiscovery()
    devices = []
    try:
        wsd.start()
        services = wsd.searchServices(timeout=timeout)
        if services:
            for s in services:
                try:
                    data = dict(s.__dict__)
                    xaddrs = data.get("_xAddrs") or []
                    xaddr = xaddrs[0] if len(xaddrs) > 0 else ""
                    if not xaddr:
                        continue
                        
                    device = {
                        "instance_id": str(data.get("_instanceId", "0")),
                        "xAddrs": xaddr,
                        "message_num": data.get("_messageNumber", 0),
                        "metadataVersion": data.get("_metadataVersion", 1),
                        "epr": str(data.get("_epr", "")),
                    }
                    devices.append(device)
                except Exception as ex:
                    logger.error(f"Error parsing service: {ex}")
    except Exception as e:
        logger.error(f"WS-Discovery search error: {e}")
    finally:
        try:
            wsd.stop()
        except Exception:
            pass
            
    return devices
