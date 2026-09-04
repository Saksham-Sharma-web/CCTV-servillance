import stream
import discovery
import ibvap.core.pipeline as pipeline
import asyncio
import cv2


async def survillance():
    devices = discovery.discover()
    cams = await asyncio.gather(*(stream.connect_camera(d) for d in devices))
    processor = pipeline.IBVAPPipeline()
    processor.register_reference_image(
        name="Akshat",
        image_path=r"C:\ibvap\akshat.jpeg",
        reference_age="most_recent"
    )
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
 #               if KeyboardInterrupt:
#               break
            await c.close()


def test_images(path):
    processor = pipeline.IBVAPPipeline()

    processor.register_reference_image(
        name="Akshat",
        image_path=r"C:\ibvap\akshat.jpeg",
        reference_age="most_recent"
    )

    frame = cv2.imread(path,cv2.IMREAD_COLOR)
    result = None
    if frame is not None:
        result = processor.process_frame(frame)
    return result






if __name__ == "__main__":
    print(test_images("C:\ibvap\akshat.jpeg"))

