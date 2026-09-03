import asyncio
import discovery
import connection
import cv2
import numpy as np

async def connect_camera(device):
    try:
        return await connection.connect(device)
    except Exception as e:
        print(f"failed on {device['_xAddrs']}: {e}")
        return None

async def rtsp_url(cam,username = "cam",passwd = "12345678"):
    '''Takes the device, username and password. Default Values for username = "cam" and passwd = "12345678"'''
    media = await cam.create_media_service()
    profiles = await media.GetProfiles()
    main_profile = profiles[0].token


    req = media.create_type('GetStreamUri')
    req.ProfileToken = main_profile
    req.StreamSetup = {
        'Stream': 'RTP-Unicast',
        'Transport': {'Protocol': 'RTSP'}
    }


    stream_response = await media.GetStreamUri(req)
    raw_rtsp_url = stream_response.Uri


    rtsp_url = raw_rtsp_url.replace("rtsp://", f"rtsp://{username}:{passwd}@")
    return rtsp_url






async def main():
    devices = discovery.discover()
    cams = await asyncio.gather(*(connect_camera(d) for d in devices))
    details = []
    urls=[]
    for c in cams:
        if c:
            details.append(c.__dict__)
            url = await rtsp_url(c)
            urls.append(url)
            # cap = cv2.VideoCapture(url)
            # while True:
            #     ret,frame = cap.read()
            #     if not ret:
            #         print("Failed to recieve Frames")
            #         break
            #     print(frame.shape)
            #     cv2.imshow("camera",frame)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         break
            await c.close()

    return  details,urls

if __name__ == "__main__":
    check= asyncio.run(main())

    # print(l)

