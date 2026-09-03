import stream
import discovery
import ibvap.core.pipeline as pipeline
import asyncio
import cv2


async def survillance():
    devices = discovery.discover()
    cams = await asyncio.gather(*(stream.connect_camera(d) for d in devices))
    processor = pipeline.IBVAPPipeline()
    for c in cams:
        if c:
            url = await stream.rtsp_url(c)
            cap = cv2.VideoCapture(url)
            print("Processing started!")
            print("Press Ctrl+C to stop")
            while True:
                ret,frame = cap.read()

                if not ret:
                    print("Failed to Recieve frames")
                    break
                result = processor.process_frame(frame)
                print(result)
                print()
                if KeyboardInterrupt:
                    break
            await c.close()



if __name__ == "__main__":
    asyncio.run(survillance())




