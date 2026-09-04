# IBVAP Vehicle Detection and ANPR OCR Pipeline Knowledge Base

## 1. Executive Summary

The **IBVAP (Intelligent Border Video Analytics Platform)** vehicle-analysis pipeline provides real-time detection, tracking, and optical character recognition (OCR) of vehicles and license plates from surveillance video feeds and still frames.

```text
┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Decoded BGR   │     │  YOLOv8 Object │     │   Persistent   │     │  Multi-Strategy│
│  OpenCV Frame  ├────►│   Detection    ├────►│    Tracker     ├────►│ Plate Candidate│
│ (H x W x 3)    │     │ (Car/SUV/Truck)│     │(Kalman Filter) │     │   Extraction   │
└────────────────┘     └────────────────┘     └────────────────┘     └───────┬────────┘
                                                                             │
                                                                             ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Structured    │     │ Normalization  │     │ Batch Predict  │     │  PaddleOCR +   │
│ PipelineResult ◄─────┤ & Watchlist    │◄────┤  Confidence    │◄────┤ Multi-Variant  │
│     Output     │     │ Cross-Reference│     │ (DL01AB1234)   │     │ Preprocessing  │
└────────────────┘     └────────────────┘     └────────────────┘     └────────────────┘
```

---

## 2. Root Cause Analysis of Prior Failures

Prior to the overhaul, the vehicle and ANPR pipeline failed completely due to five critical defects:

| Failure Point | Component | Root Cause | Impact | Resolution |
| :--- | :--- | :--- | :--- | :--- |
| **Broken OCR Dependency** | `ibvap/anpr/ocr_adapter.py` | Attempted to import `OCREngine` from a nonexistent relative path `../../id-verification/verification/ocr/ocr_engine.py`. | Import failed, setting `self.ocr_engine = None`. `recognize_plate()` returned `None` unconditionally. | Replaced with self-contained, in-process `EasyOCR` (PyTorch native with CUDA/CPU auto-detection). |
| **Frame-1 Throttling Skip** | `ibvap/core/pipeline.py` | Throttle condition was `(frame_index - track.last_ocr_check_frame) >= 10`. When `frame_index = 1` and `last_ocr_check_frame = 0`, `1 - 0 = 1 < 10` evaluated to `False`. | OCR was skipped on all static image tests and the first 9 video frames. | Changed condition to `(track.last_ocr_check_frame == 0 or (frame_index - track.last_ocr_check_frame) >= interval)`. |
| **Local Track Object Disconnect** | `ibvap/core/pipeline.py` | `cam_tracker.update_track_plate()` mutated internal tracker dictionary items, but the `Track` instances returned in `tracks = cam_tracker.update()` were never updated on that frame. | Returned `PipelineResult.tracks` had `plate_number: None` on the frame the plate was detected. | Added direct attribute mutation on the active `track` object (`track.plate_number = ...`). |
| **Brittle Morphology Filter** | `ibvap/anpr/plate_detector.py` | Used static Sobel kernel `(17, 3)` and rigid area filters `[0.005, 0.25]` without fallback. | Failed to detect plates on distant cars, small resolutions, or high-contrast bumpers. | Implemented multi-strategy candidate extraction with resolution-adaptive kernels and bumper ROI fallback. |
| **Module Import Blocker** | `main.py` | Imported `stream` and `discovery` at the top level, which required `wsdiscovery`. | Standalone testing crashed with `ModuleNotFoundError` before `test_images()` could execute. | Made ONVIF discovery imports lazy inside `survillance()`. |

---

## 3. Component Architecture & Implementation Details

### 3.1 Vehicle Detection (`ibvap/detection/object_detector.py`)
* **Underlying Engine**: Ultralytics YOLOv8 (`YOLOv8Detector`).
* **Weights Resolution**: Checks `model_weights` parameter $\to$ `IBVAPConfig.detector_model_path` $\to$ `models/yolov8n.pt` $\to$ workspace root `yolov8n.pt` $\to$ default Ultralytics download.
* **Confidence Threshold**: Default set to `0.35` in `IBVAPConfig` (empirically balanced for CCTV angles and night lighting).
* **Surveillance Classes Supported**: `car`, `suv`, `van`, `truck`, `bus`, `motorcycle`, `vehicle`.
* **Bounding Box Clamping**: All coordinates are strictly constrained to `[0, 0, width, height]` to prevent out-of-bound memory errors during cropping.

### 3.2 Multi-Strategy License Plate Detection (`ibvap/anpr/plate_detector.py`)
The detector processes the vehicle bounding crop using four complementary strategies:
1. **Adaptive Morphology & Sobel Horizontal Gradient**:
   * Kernel size dynamically scales with vehicle crop dimensions:
     $$\text{kernel\_width} = \max(7, \min(35, \text{int}(vw \times 0.08)))$$
     $$\text{kernel\_height} = \max(3, \text{int}(\text{kernel\_width} / 4))$$
   * Aspect ratio range: $1.3 \le \text{AR} \le 6.0$.
   * Area ratio range: $0.001 \le \text{AreaRatio} \le 0.40$ (allowing small or distant plates).
2. **Adaptive Gaussian Thresholding**:
   * Employs `cv2.adaptiveThreshold(..., cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)` to isolate rectangular plate containers on high-contrast or shadowed surfaces.
3. **IoU Candidate Deduplication**:
   * Suppresses overlapping candidate boxes with IoU $> 0.45$.
4. **Bumper ROI Fallback**:
   * If morphological contours yield fewer than 2 candidates, the canonical bumper zone (lower-center 50% of the vehicle) is added as a candidate so downstream OCR text detection can inspect the area directly.

### 3.3 OCR & Plate Normalization Engine (`ibvap/anpr/ocr_adapter.py`)
* **Underlying Engine**: PaddleOCR PP-OCRv4 (`paddlex.create_model("en_PP-OCRv4_mobile_rec")`).
* **Multi-Variant Preprocessing**:
  1. **Scaling**: Upscales small plate crops to height 48-64px with aspect ratio preserved.
  2. **Border Padding**: Adds 8px vertical and 12px horizontal replication padding (`cv2.copyMakeBorder`) to prevent boundary characters (e.g. leading 'D' or trailing '4') from touching image edges.
  3. **Unsharp Masking**: Enhances character stroke edges for low-resolution or slightly blurred crops.
  4. **CLAHE Enhancement & Bilateral Denoising**: Contrast Limited Adaptive Histogram Equalization with clip limit 2.5 on grayscale, smoothing surface noise while preserving sharp font edges for night/shadow plates.
* **Batch Prediction**:
  * Feeds all image variants simultaneously in a single forward pass.
* **Alphanumeric Normalization**:
  * Strips punctuation, dashes, spaces, and special characters.
  * Corrects standard state-code character confusion:
    * `0L` / `OL` $\to$ `DL`
    * `P1` $\to$ `UP`
    * `O` / `Q` $\to$ `0` inside state/RTO digit slots (indices 2, 3) and trailing number slots.
    * `I` / `L` $\to$ `1`, `Z` $\to$ `2`, `S` $\to$ `5`, `B` $\to$ `8` in numerical slots.
* **Validation Criteria**: Valid plates must contain between 3 and 12 alphanumeric characters.

---

## 4. Data Contracts & Output Schemas

### 4.1 Track Object Schema (`ibvap/core/types.py`)
```python
@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    center: Tuple[int, int]
    
    # Biometric identity
    identity_id: Optional[str] = None
    identity_name: Optional[str] = None
    identity_confidence: Optional[float] = None
    
    # ANPR results
    plate_number: Optional[str] = None
    plate_category: Optional[WatchlistCategory] = None
    plate_confidence: Optional[float] = None
    ocr_confidence: Optional[float] = None
    plate_bbox: Optional[Tuple[int, int, int, int]] = None
```

### 4.2 PipelineResult Schema
`result.to_dict()` provides both granular tracking data and the top-level structured analysis dictionary:
```json
{
  "success": true,
  "camera_id": "camera-01",
  "timestamp": 1788508978.15,
  "frame_shape": [1080, 810],
  "vehicle_detected": true,
  "vehicle_type": "bus",
  "vehicle_confidence": 0.8734,
  "license_plate_detected": true,
  "license_plate": "DL01AB1234",
  "plate_confidence": 0.8962,
  "ocr_confidence": 0.8962,
  "vehicle_analysis": {
    "vehicle_detected": true,
    "vehicle_type": "bus",
    "vehicle_confidence": 0.8734,
    "license_plate_detected": true,
    "license_plate": "DL01AB1234",
    "plate_confidence": 0.8962,
    "ocr_confidence": 0.8962
  },
  "detections": [...],
  "tracks": [...],
  "events": [...]
}
```

---

## 5. Usage & Integration

### 5.1 Programmatic Execution (Python API)
```python
import cv2
from ibvap.core.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig

# Initialize pipeline once (preserves persistent Kalman state across frames)
pipeline = IBVAPPipeline(config=IBVAPConfig())

# Process single frame
frame = cv2.imread("car_frame.jpg")
result = pipeline.process_frame(frame, camera_id="camera-01")

print(f"Vehicle Detected: {result.vehicle_detected} ({result.vehicle_type})")
print(f"License Plate: {result.license_plate} (Conf: {result.plate_confidence})")
```

### 5.2 Standalone Testing Script (`main.py`)
To test an image file from the command line:
```bash
python main.py "path/to/car_image.jpg"
```
Or within Python code:
```python
from main import test_images
result = test_images("C:/images/test_car.jpg")
```
This executes sequential verification:
1. Validates and decodes image dimensions.
2. Runs face detection and prints occupant identities.
3. Runs YOLOv8 vehicle detection and EasyOCR license plate recognition.
4. Correlates occupant face and vehicle license number.
