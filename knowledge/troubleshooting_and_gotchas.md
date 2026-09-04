# IBVAP Troubleshooting & Engineering Gotchas

This document records key technical edge cases, design pitfalls, and performance recommendations discovered during development and debugging.

---

## 1. Frame-1 Throttling Bug

### The Issue
ANPR OCR and face recognition are computationally expensive compared to YOLO bounding box detection. To prevent framerate degradation, IBVAP implements frame throttling via `IBVAPConfig.anpr_ocr_interval_frames` (default: 10).

Previously, the condition was written as:
```python
need_ocr_check = (
    track.plate_number is None
    and (frame_index - track.last_ocr_check_frame) >= self.config.anpr_ocr_interval_frames
)
```
When evaluating static images or single frames, `frame_index = 1` and `last_ocr_check_frame = 0`.
Because `1 - 0 = 1 < 10`, `need_ocr_check` evaluated to `False`. The pipeline skipped OCR on frame 1 and on the first 9 frames of any video feed.

### The Fix
Always allow immediate execution when `last_ocr_check_frame == 0` (indicating the track has never been checked):
```python
need_ocr_check = (
    track.plate_number is None
    and (track.last_ocr_check_frame == 0 or (frame_index - track.last_ocr_check_frame) >= self.config.anpr_ocr_interval_frames)
)
```

---

## 2. Character Boundary Clipping in OCR

### The Issue
When license plate crops tightly frame the characters without outer whitespace, deep-learning OCR models (such as EasyOCR and CRNNs) often drop or misread the first and last letters (e.g. `DL01AB1234` recognized as `L01AB123`). This happens because the receptive fields of the convolutional feature extractors extend beyond the bounding box edges.

### The Fix
Before passing candidate crops to EasyOCR, apply border replication padding via OpenCV:
```python
padded = cv2.copyMakeBorder(resized, top=10, bottom=10, left=16, right=16, borderType=cv2.BORDER_REPLICATE)
```
This guarantees an artificial safety margin so boundary characters are fully extracted.

---

## 3. Synthetic vs. Photorealistic Testing

### The Issue
YOLOv8 is trained on real-world photographic images (COCO dataset). Testing with simplified cartoon or flat-color rectangle drawings often fails to trigger vehicle detections because natural automotive textures, shadows, highlights, and geometries are absent.

### Recommendation
* For unit testing without neural model weights, use `MockDetector([Detection(...)])`.
* For end-to-end integration tests, use real photographic vehicle images (e.g., standard CCTV samples or photos of cars, vans, trucks, and buses).

---

## 4. PyTorch & EasyOCR Warnings on Python 3.13

### The Issue
When running EasyOCR on CPU with PyTorch, deprecation warnings regarding dynamic quantization (`torch.ao.quantization`) or pinned memory may appear in the terminal logs:
```text
UserWarning: torch.quantize_per_tensor ... are deprecated and will be removed in a future PyTorch release.
UserWarning: 'pin_memory' argument is set as true but no accelerator is found...
```

### Impact & Solution
* These are non-fatal upstream PyTorch/EasyOCR library warnings and do not affect recognition accuracy or execution flow.
* If running in production with an NVIDIA GPU, installing CUDA-enabled PyTorch will automatically eliminate these CPU warnings and accelerate OCR inference by $5\times$ to $10\times$.

---

## 5. Standalone Execution vs. ONVIF Camera Streaming

### The Issue
`stream.py` and `discovery.py` rely on `wsdiscovery` and `onvif-zeep-async` for local network camera discovery over WS-Discovery. If those packages are absent from the local Python environment, importing `stream` or `discovery` at the module root will crash the entire script.

### The Fix
In `main.py`, imports for `stream` and `discovery` are kept lazy inside `async def survillance()`. This ensures `test_images()` and the core `ibvap` package run 100% standalone without network or ONVIF hardware dependencies.
