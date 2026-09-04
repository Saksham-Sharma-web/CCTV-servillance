from wsdiscovery import WSDiscovery
import json


def discover(timeout=3):
    """Takes a timeout value in seconds and returns a list of available devices as a JSON string."""
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
                    print(f"[discovery] Error parsing service: {ex}")
    except Exception as e:
        print(f"[discovery] WS-Discovery search error: {e}")
    finally:
        try:
            wsd.stop()
        except Exception:
            pass
    return json.dumps(devices)


if __name__ == "__main__":
    print(discover())
