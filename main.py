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
            print(f"Connecting to RTSP Stream: {url}")
            import os
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            print("Processing started! Display window opened.")
            print("Press 'q' in the video window or Ctrl+C to stop.")

            PROCESS_EVERY_N_FRAMES = 24  # Process AI every 24 frames
            frame_counter = 0
            last_result = None

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        print("Failed to receive frames or stream interrupted.")
                        break

                    frame_counter += 1

                    # Run heavy AI detection periodically
                    if frame_counter % PROCESS_EVERY_N_FRAMES == 0 or last_result is None:
                        last_result = processor.process_frame(frame)

                    # Render live visual AI overlay smoothly at full FPS
                    if last_result is not None:
                        annotated = processor.draw_debug(frame, last_result)
                    else:
                        annotated = frame

                    cv2.imshow("IBVAP CCTV Live Surveillance (Press 'q' to quit)", annotated)

                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                        break
            finally:
                cap.release()
                cv2.destroyAllWindows()
                await c.close()


def test_images(path):
    processor = pipeline.IBVAPPipeline()
    frame = cv2.imread(path,cv2.IMREAD_COLOR)
    result = None
    if frame is not None:
        result = processor.process_frame(frame)
    return result






if __name__ == "__main__":
    print(test_images("test.png"))
