import asyncio
import json

import discovery
import connection


async def connect_camera(device):

    try:
        print(device)
        return await connection.connect_camera(device)
    except Exception as e:
        print(f"Camera connection failed: {e}")
        return None


async def get_rtsp_url(cam, username="cam", passwd="12345678"):
    media = await cam.create_media_service()
    profiles = await media.GetProfiles()

    profile = profiles[0]

    req = media.create_type("GetStreamUri")
    req.ProfileToken = profile.token
    req.StreamSetup = {
        "Stream": "RTP-Unicast",
        "Transport": {"Protocol": "RTSP"}
    }

    response = await media.GetStreamUri(req)

    return response.Uri.replace(
        "rtsp://",
        f"rtsp://{username}:{passwd}@"
    )


async def main():

    devices = json.loads(discovery.discover())

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

        finally:

            await cam.close()

    return results


if __name__ == "__main__":
    result = asyncio.run(main())

    # IMPORTANT:
    # stdout should contain ONLY the JSON response
    print(json.dumps(result))
