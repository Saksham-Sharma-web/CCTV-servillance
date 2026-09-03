from wsdiscovery import WSDiscovery, QName



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
            device = dict(i.__dict__)
            devices.append(device)

        # print(devices)
        return devices



    except Exception as e:
        print(e)

    finally:
        wsd.stop()


if __name__ == "__main__":
    print(len(discover()))
