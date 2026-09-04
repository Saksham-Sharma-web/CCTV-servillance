# IBVAP: Intelligent Border Video Analytics Platform

A self-contained, source-agnostic computer-vision and behavioral analytics engine in Python.

IBVAP receives already-decoded OpenCV BGR video frames (`numpy.ndarray`) and executes object detection, persistent multi-object tracking, selective face detection/biometric verification, license plate recognition (ANPR), virtual fence boundary intrusion detection, loitering analysis, night-time movement detection, and debounced event generation.

---

## 1. Core Architecture & Philosophy

```text
                  Upstream Video Source
             (RTSP, WebRTC, Webcam, File)
                         │
                         │ cv2.VideoCapture / PyAV / ffmpeg
                         ▼
             OpenCV numpy.ndarray (BGR, uint8)
                         │
                         ▼
              ┌─────────────────────┐
              │   IBVAPPipeline     │
              │                     │
              │ • Object Detection  │
              │ • Object Tracking   │
              │ • Face Biometrics   │
              │ • ANPR OCR          │
              │ • Virtual Fence     │
              │ • Behavioral Rules  │
              │ • Event Engine      │
              └──────────┬──────────┘
                         │
                         ▼
             Structured PipelineResult
```

### Source-Agnostic Principle
* **IBVAP owns**: Frame validation, computer vision, tracking, behavioral analytics, event debouncing, structured results.
* **Integrating application owns**: Camera connections, RTSP, video decoding, codecs, frame acquisition, network streaming, persistence/databases.
* **No Server Required**: IBVAP is a pure Python library. It does not run HTTP servers, background daemons, or database queries.

---

## 2. Installation

Install dependencies from the standalone `requirements.txt`:

```bash
pip install -r ibvap/requirements.txt
```

### Core Dependencies:
* `opencv-python>=4.8.0`
* `numpy>=1.24.0`
* `scipy>=1.10.0`
* `ultralytics>=8.0.0`
* `torch>=2.0.0`
* `torchvision>=0.15.0`
* `pillow>=9.5.0`

---

## 3. Quickstart & Usage

```python
import cv2
import numpy as np
from ibvap import IBVAPPipeline

# 1. Instantiate once (maintains persistent tracking across frames)
pipeline = IBVAPPipeline()

# 2. Caller obtains the frame (from webcam, RTSP, file, etc.)
frame = cv2.imread("sample_frame.jpg")

# 3. Process frame through the pipeline
result = pipeline.process_frame(
    frame=frame,
    camera_id="camera-01"
)

# 4. Access structured results
print(f"Status: {result.success}")
print(f"Detections: {len(result.detections)}")
print(f"Tracks: {len(result.tracks)}")
print(f"Events: {len(result.events)}")

# Optional: Export to standard dictionary
result_dict = result.to_dict()
```

### Webcam Integration Example

```python
import cv2
from ibvap import IBVAPPipeline

pipeline = IBVAPPipeline()
cap = cv2.VideoCapture(0)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Pass raw OpenCV BGR frame directly
        result = pipeline.process_frame(frame, camera_id="laptop-camera")

        for track in result.tracks:
            print(f"Track {track.track_id}: {track.class_name} at {track.bbox}")

        for event in result.events:
            print(f"ALERT [{event.event_type}]: Track {event.track_id}")

        # Optional: Render visual debug overlays on a copy of the frame
        annotated = pipeline.draw_debug(frame, result)
        cv2.imshow("Surveillance Feed", annotated)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
```

---

## 4. Input Contract & Validation

`pipeline.process_frame()` enforces strict validation at the library boundary:

| Property | Requirement | Error on Failure |
| :--- | :--- | :--- |
| **Type** | `numpy.ndarray` (cannot be `None`) | `ValueError: Input frame cannot be None` |
| **Dimensions** | Exactly 3 dimensions: `(height, width, 3)` | `ValueError: Invalid frame dimensions` |
| **Channels** | Exactly 3 channels (OpenCV BGR color order) | `ValueError: Invalid channel count` |
| **Data Type** | `numpy.uint8` (0 - 255) | `ValueError: Invalid frame dtype` |
| **Shape** | Non-empty (`height > 0`, `width > 0`) | `ValueError: Invalid frame shape` |
| **Immutability** | Caller's input frame is **never modified** in place | Guaranteed |

---

## 5. Output Data Model

`process_frame()` returns a typed `PipelineResult` dataclass (convertible via `.to_dict()`):

```json
{
  "success": true,
  "camera_id": "camera-01",
  "timestamp": 1725345600.12,
  "frame_shape": [720, 1280],
  "detections": [
    {
      "bbox": [120, 80, 240, 420],
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.942
    }
  ],
  "tracks": [
    {
      "track_id": 1,
      "bbox": [120, 80, 240, 420],
      "class_name": "person",
      "confidence": 0.942,
      "center": [180, 250],
      "identity_id": "emp_042",
      "identity_name": "John Doe",
      "identity_confidence": 0.887,
      "plate_number": null,
      "plate_category": null,
      "plate_confidence": null
    }
  ],
  "events": [
    {
      "event_id": "b182b8a4-0c20-410a-9d22-1d59526ff342",
      "camera_id": "camera-01",
      "timestamp": 1725345600.12,
      "event_type": "FACE_MATCHED",
      "track_id": 1,
      "identity_id": "emp_042",
      "confidence": 0.887,
      "metadata": {
        "name": "John Doe",
        "role": "EMPLOYEE",
        "similarity": 0.887
      }
    }
  ],
  "metadata": {
    "frame_index": 124,
    "camera_id": "camera-01"
  }
}
```

---

## 6. Key Concepts

### Persistent Visual Tracking (`track_id`)
* `track_id` is assigned and maintained by `PersistentTracker`.
* Tracks survive frame-to-frame motion and brief occlusions using spatial IoU matching and Kalman velocity predictions.
* **Camera Isolation**: Tracking state is isolated per `camera_id`. Processing frames for `camera-01` and `camera-02` through the same pipeline instance will maintain separate tracker states without ID collisions.

### Tracking vs. Biometric Identity (`track_id != identity_id`)
* `track_id` represents visual spatio-temporal continuity.
* `identity_id` represents verified biometric identity.
* If a person is unrecognized or face recognition fails, `track_id` is preserved (`identity_id: None`).
* Once verified, `identity_id` is bound to the track and face matching is throttled to save compute.

### ANPR / License Plate Recognition
* Operates selectively on vehicle tracks (`car`, `motorcycle`, `bus`, `truck`).
* Throttled per vehicle track so expensive OCR does not execute on every frame.
* Detected plates are cross-referenced against custom watchlists (`WHITELIST`, `BLACKLIST`, `WATCHLIST`).

### Virtual Fences & Behavioral Rules
* Virtual line-crossing and polygon restricted zones generate `FENCE_INTRUSION` events only on boundary transitions.
* Loitering analysis monitors dwell time within localized spatial radii.
* Night movement detects confirmed motion under low ambient frame illumination.
* All events are debounced by `EventEngine` to eliminate duplicate alert spam.

---

## 7. Configuration

All thresholds can be configured via `IBVAPConfig`:

```python
from ibvap import IBVAPPipeline, IBVAPConfig

config = IBVAPConfig(
    detection_confidence=0.50,
    tracking_iou_threshold=0.35,
    face_verification_interval_frames=15,
    anpr_ocr_interval_frames=10,
    loitering_duration_seconds=10.0,
    fence_cooldown_seconds=5.0,
    night_brightness_threshold=50.0,
)

pipeline = IBVAPPipeline(config=config)
```

---

## 8. Directory Structure

```text
ibvap/
├── __init__.py                # Package exports (IBVAPPipeline, IBVAPConfig, types)
├── pipeline.py                # Re-export entrypoint
├── requirements.txt           # Standalone Python dependencies
├── README.md                  # Documentation
├── core/
│   ├── config.py              # Central validated configuration
│   ├── types.py               # Dataclasses (Detection, Track, AnalyticsEvent, PipelineResult)
│   └── pipeline.py            # Master orchestration engine
├── detection/
│   ├── base.py                # BaseObjectDetector abstract interface
│   └── object_detector.py     # YOLOv8Detector & MockDetector
├── tracking/
│   └── tracker.py             # PersistentTracker (IoU matching & velocity prediction)
├── face/
│   ├── detector.py            # OpenCVFaceDetector (YuNet + Haar cascade fallback)
│   └── matcher_adapter.py     # IdentityVerifierAdapter (Cosine similarity face matching)
├── anpr/
│   ├── plate_detector.py      # LicensePlateDetector (Morphology / aspect ratio candidate extraction)
│   └── ocr_adapter.py         # ANPRAdapter (OCR text extraction & watchlist validation)
├── analytics/
│   ├── virtual_fence.py       # Line-crossing and polygon boundary intrusion detection
│   ├── suspicious_activity.py # Loitering, sudden acceleration, unattended object detection
│   └── night_movement.py      # Low-light luminance & motion activity detection
├── events/
│   └── event_engine.py        # Temporal deduplication & debouncing engine
├── visualization/
│   └── debug_renderer.py      # Debug overlay renderer (bounding boxes, tracks, alerts)
├── models/                    # Designated local weight storage
└── tests/                     # Comprehensive pytest test suite
```
