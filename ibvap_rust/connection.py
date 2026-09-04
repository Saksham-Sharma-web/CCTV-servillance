import asyncio
import json
import cv2
import numpy as np

import discovery



# ============================================================
# Camera connection
# ============================================================

from onvif import ONVIFCamera

async def connect_camera(device,username="cam",passwd = "12345678"):
    xaddrs = device["xAddrs"]
    s = xaddrs.split("/")[2]
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


async def get_rtsp_url(cam, username="cam", passwd="12345678"):

    media = await cam.create_media_service()

    profiles = await media.GetProfiles()

    profile = profiles[0]

    req = media.create_type("GetStreamUri")

    req.ProfileToken = profile.token

    req.StreamSetup = {
        "Stream": "RTP-Unicast",
        "Transport": {
            "Protocol": "RTSP"
        }
    }

    response = await media.GetStreamUri(req)

    return response.Uri.replace(
        "rtsp://",
        f"rtsp://{username}:{passwd}@"
    )


# ============================================================
# Discovery
# ============================================================

async def main():

    devices = json.loads(
        discovery.discover()
    )

    results = []

    for device in devices:

        cam = await connect_camera(device)

        if cam is None:
            continue

        try:

            xaddr = device["xAddrs"]

            host = xaddr.split("/")[2].split(":")[0]

            rtsp = await get_rtsp_url(cam)

            results.append({
                "id": device["instance_id"],
                "name": f"Camera {host}",
                "ip": host,
                "rtsp": rtsp
            })

        except Exception as e:

            print(
                f"Failed processing camera: {e}"
            )

        finally:

            await cam.close()

    return results


# ============================================================
# LIVE STREAM
# ============================================================

class CameraStream:

    def __init__(self):

        self.cap = None
        self.rtsp_url = None


    def start(self, rtsp_url):

        self.stop()

        self.rtsp_url = rtsp_url

        self.cap = cv2.VideoCapture(
            rtsp_url,
            cv2.CAP_FFMPEG
        )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            1
        )

        if not self.cap.isOpened():

            self.cap = None

            raise RuntimeError(
                "Could not open RTSP stream"
            )


    def read(self):

        if self.cap is None:

            return None

        ret, frame = self.cap.read()

        if not ret:

            return None


        # ----------------------------------------------------
        # AI PROCESSING GOES HERE
        # ----------------------------------------------------
        #
        # Example:
        #
        # frame = ai.process(frame)
        #
        # Keep the frame as NumPy ndarray.
        #
        # ----------------------------------------------------


        # OpenCV uses BGR.
        # Slint wants RGB.

        frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        height, width, channels = frame.shape

        return (
            frame.tobytes(),
            width,
            height
        )


    def stop(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None


# One persistent stream object.
camera_stream = CameraStream()


def start_stream(rtsp_url):

    camera_stream.start(rtsp_url)


def read_frame():

    return camera_stream.read()


def stop_stream():

    camera_stream.stop()
