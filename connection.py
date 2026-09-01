from onvif import ONVIFCamera

def connect(device,username="cam",passwd = "12345678"):
    xaddrs = device["_xAddrs"]
    s = xaddrs[0].split("/")[2]
    host,p,port = s.partition(":")

    try :
        cam = ONVIFCamera(host=host,port=int(port),user=username,passwd=passwd,adjust_time=True,encrypt=False)

        print(f"connection to {host} at {port} successful!")
        print(cam)
        return cam
    except Exception as e:
        print(e)




