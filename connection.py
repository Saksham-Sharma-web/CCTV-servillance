from onvif import ONVIFCamera

async def connect(device,username="cam",passwd = "12345678"):
    xaddrs = device["_xAddrs"]
    s = xaddrs[0].split("/")[2]
    # print(s)
    host,p,port = s.partition(":")
    # print(host,port)

    try :
        cam = ONVIFCamera(host=host,port=int(port),user=username,passwd=passwd,encrypt=True)
        await cam._devicemgmt_with_time()
        await cam.update_xaddrs()

        print(f"connection to {host} at {port} successful!")

        return cam
    except Exception as e:
        await cam.close()
        raise




