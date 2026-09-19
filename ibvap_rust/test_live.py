import sys
import time
from live_streaming import LiveCameraStream

def test():
    print("starting stream...")
    # use webcam
    stream = LiveCameraStream("cam1", "0")
    start = time.time()
    frames = 0
    while time.time() - start < 5:
        res = stream.next_frame()
        if res:
            frames += 1
            bgr, w, h, ev = res
            print(f"Got frame: {w}x{h}, events: {ev}")
        else:
            print("No frame")
    
    print(f"Total frames in 5s: {frames}")
    stream.release()

if __name__ == "__main__":
    test()
