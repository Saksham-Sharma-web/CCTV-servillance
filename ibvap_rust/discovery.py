from wsdiscovery import WSDiscovery, QName
import json


def discover(timeout = 5):
    ''' IT takes a timeout value in seconds and returns a list of available devices'''
    wsd = WSDiscovery()
    try:
        wsd.start()
        services = wsd.searchServices(timeout=timeout)
        if not services:
            print("No Services")
        devices = []
        for i in services:
            data = dict(i.__dict__)
            device = {
                "instance_id" : str(data["_instanceId"]),
                "xAddrs": data["_xAddrs"][0],
                "message_num" : data["_messageNumber"],
                "metadataVersion" : data["_metadataVersion"],
                "epr" : str(data["_epr"])
            }
            devices.append(device)

        # print(devices)
        return json.dumps(devices)



    except Exception as e:
        print(e)

    finally:
        wsd.stop()

if __name__ == "__main__":
    print(discover())
