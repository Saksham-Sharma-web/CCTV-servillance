# IBVAP: Multi-Reference Face & Supporting Body Appearance Testing Guide

This guide explains how to configure, access, and test the IBVAP face verification pipeline with **multiple reference photographs per person** across the three reference-age categories: **`most_recent`**, **`recent`**, and **`old`**.

---

## 1. Core Biometric Principles & Semantics

### The Golden Biometric Invariant
```text
NO VALID FACE ──► NO FACE EMBEDDING ──► NO IDENTITY
FACE = PRIMARY BIOMETRIC VERIFIER
BODY = STRICTLY SUPPORTING OBSERVATION
```
- **Body appearance will NEVER create a face identity** when no face is detected.
- **Body appearance will NEVER turn a face mismatch (`UNKNOWN`) into a `MATCH`**, even if the person is wearing the exact same clothing.
- A person/torso crop is **never** fed to the facial recognition model.

### Reference Age Categories

| Category | Meaning | Body Weight | Body Role | Impact on Face Recognition |
| :--- | :--- | :---: | :--- | :--- |
| **`most_recent`** | Represents the person's current / latest appearance. Clothing may be similar. | `0.20` | `BODY_SUPPORTING` | Standard threshold ($0.65$) |
| **`recent`** | Reasonably recent photo. Clothing or style may differ. | `0.10` | Supporting with lower weight | Standard threshold ($0.65$) |
| **`old`** | Older photo. Hairstyle, weight, and clothing have likely changed. | `0.00` | `BODY_IGNORED` | Standard threshold ($0.65$) |

> [!IMPORTANT]
> **Reference Age does NOT lower the face similarity threshold.**
> Face verification remains strictly conservative. `reference_age` only governs how much the supplementary body/appearance signal can contribute.

---

## 2. Quickstart: Testing via `test_images()` in Python

You can invoke `test_images()` directly from Python or a script with arbitrary image paths and reference age tags:

```python
from main import test_images

results = test_images(
    references=[
        # (Name, File Path, Reference Age)
        ("Amit", r"C:\ibvap\amit_front_today.jpg", "most_recent"),
        ("Amit", r"C:\ibvap\amit_last_month.jpg", "recent"),
        ("Amit", r"C:\ibvap\amit_college_days.jpg", "old"),
        ("Rahul", r"C:\ibvap\rahul_recent.jpg", "most_recent"),
    ],
    target=r"C:\ibvap\live_target.jpg",
    debug=True  # Generates diagnostic overlay at output/debug_analysis.jpg
)
```

---

## 3. Command-Line (CLI) Quick Tests

### Option A: One-Liner PowerShell / Command Prompt
Test multiple references directly from your terminal:

```powershell
python -c "from main import test_images; res = test_images(references=[('Amit', r'C:\ibvap\akshat.jpeg', 'most_recent'), ('Amit', r'C:\ibvap\akshat.jpeg', 'recent'), ('Amit', r'C:\ibvap\akshat.jpeg', 'old')], target=r'C:\ibvap\akshat.jpeg', debug=True)"
```

### Option B: Run Default Verification
```powershell
python main.py
```
*(Automatically tests `akshat.jpeg` against the enrolled reference with debug overlay saved to `output/debug_analysis.jpg`)*

---

## 4. How to Use in the Live Surveillance Pipeline (`IBVAPPipeline`)

If you are running the live CCTV surveillance pipeline or building a custom script:

```python
import cv2
from ibvap.core.pipeline import IBVAPPipeline
from ibvap.core.config import IBVAPConfig

# 1. Initialize Pipeline
config = IBVAPConfig()
pipeline = IBVAPPipeline(config=config)

# 2. Register Multiple Reference Photos for a Person
# Every photo automatically undergoes YuNet face detection & quality validation
ok1, msg1 = pipeline.register_reference_image(
    name="Amit",
    image_path=r"C:\ibvap\amit_front.jpg",
    reference_age="most_recent"
)

ok2, msg2 = pipeline.register_reference_image(
    name="Amit",
    image_path=r"C:\ibvap\amit_old.jpg",
    reference_age="old"
)

if not ok1:
    print(f"Failed to register reference: {msg1}")  # e.g. REFERENCE_FACE_NOT_FOUND

# 3. Process Live CCTV Video Frames
frame = cv2.imread(r"C:\ibvap\live_cctv_frame.jpg")
result = pipeline.process_frame(frame, camera_id="camera-01")

# 4. Check Biometric Match Events
for event in result.events:
    if event.event_type.value == "FACE_MATCHED":
        print(f"Match: {event.metadata['name']} (Similarity: {event.metadata['similarity']:.2f})")
        print(f"Body Status: {event.metadata['body_status']}")
```

---

## 5. Understanding the Console Output

When `test_images()` executes, it produces structured console output:

```text
============================================================
IBVAP FACE ANALYSIS
============================================================

Target:
C:\ibvap\target.jpeg
Target Dimensions: 899x1599 px

------------------------------------------------------------
MATCH
------------------------------------------------------------

Identity: Amit

Reference:
C:\ibvap\amit_front_today.jpg

Reference Age:
most_recent

Face:
DETECTED

Face Confidence:
0.96

Face Similarity:
0.89

Body:
DETECTED

Body Similarity:
0.84

Body Role:
SUPPORTING ONLY

Final Face Decision:
MATCH

------------------------------------------------------------
INDIVIDUAL REFERENCE COMPARISONS:
------------------------------------------------------------

Person: Amit
  Reference 1:
    Path: C:\ibvap\amit_front_today.jpg
    Age:  most_recent
    Face similarity: 0.89
    Body similarity: 0.84
  Reference 2:
    Path: C:\ibvap\amit_last_month.jpg
    Age:  recent
    Face similarity: 0.84
    Body similarity: 0.71
  Reference 3:
    Path: C:\ibvap\amit_college_days.jpg
    Age:  old
    Face similarity: 0.82
    Body similarity: ignored
  --> Best face similarity: 0.89
  --> Face evidence: STRONG
  --> Body evidence: SUPPORTING
============================================================
```

### Interpretation of Status Values

- **`MATCH`**: Best face cosine similarity is $\ge 0.65$. The person is verified.
- **`UNKNOWN`**: Face detected, but highest similarity is $< 0.65$. The identity is `None`. (Body similarity will **not** override this).
- **`NO_FACE_DETECTED`**: No valid face detected in the target image. Identity is `None`.
- **`INSUFFICIENT_FACE_QUALITY`**: Face found, but failed blur (Laplacian variance $< 15.0$) or size checks.
- **`MULTIPLE_FACES_DETECTED`**: More than one person's face is visible in the frame.

### Interpretation of Body Signals

- **`BODY_SUPPORTING`**: Target body appearance matches reference ($\ge 0.70$) for `most_recent` or `recent` reference.
- **`BODY_INCONSISTENT`**: Face matches, but clothing differs. The face match is **still accepted** (clothing change does not auto-reject).
- **`BODY_IGNORED`**: Reference was tagged `old`. Body similarity is calculated as $0.00$ and disregarded.
- **`BODY_NOT_DETECTED`**: No person body context available.

---

## 6. Diagnostic Visualization (Debug Mode)

When `debug=True` is passed to `test_images()`:
- The system renders:
  - **Green Box**: High-quality detected face.
  - **Yellow Dots**: 5 facial landmarks (right eye, left eye, nose tip, right mouth, left mouth).
  - **Blue Box**: Detected person body context zone for appearance extraction.
  - **Annotations**: Detector confidence, quality status, and similarity.
- Saved to: `output/debug_analysis.jpg`.

---

## 7. Running Automated Unit & Integration Tests

The dedicated test suite covers all 11 required face/body scenarios and benchmarks:

```powershell
# Run the face appearance and multi-reference test suite
pytest ibvap/tests/test_face_appearance.py -v -s

# Run the complete IBVAP repository test suite (all 38 tests)
pytest ibvap/tests -v
```

### Key Automated Test Scenarios

1. `test_case_01_single_person_most_recent`: Verifies `most_recent` reference produces `BODY_SUPPORTING`.
2. `test_case_02_same_person_recent`: Verifies `recent` reference functions with lower body weighting.
3. `test_case_03_same_person_old`: Verifies `old` reference triggers `BODY_IGNORED`.
4. `test_case_04_different_person_unknown`: Verifies unfamiliar faces output `UNKNOWN` with `identity=None`.
5. `test_case_05_no_face_image`: Verifies blank/background images produce `NO_FACE_DETECTED`.
6. `test_case_06_body_only_image_strict_invariant`: Verifies a headless body crop produces `NO_FACE_DETECTED` and **zero** face identity.
7. `test_case_11_same_clothing_different_person`: Verifies that when two people wear identical clothing, body similarity ($\approx 0.90$) does **not** override a face mismatch.
8. `test_threshold_calibration`: Verifies genuine vs impostor distribution analysis and FAR/FRR curve generation.
9. `test_resource_benchmarks`: Verifies YuNet detection ($< 50$ms), FaceNet embedding ($< 90$ms), and body appearance descriptor ($< 2$ms).

---

## 8. Common Troubleshooting & Gotchas

1. **"REFERENCE_FACE_NOT_FOUND" during registration**:
   - The reference photo does not contain a recognizable face or the face is turned away past $65^\circ$. Provide a clearer frontal reference image.
2. **"INSUFFICIENT_FACE_QUALITY"**:
   - The face is heavily motion-blurred (Laplacian variance $< 15.0$) or smaller than $24 \times 24$px.
3. **Paths on Windows**:
   - Always prefix Windows paths with `r` (raw string), e.g., `r"C:\ibvap\photo.jpg"`, or use forward slashes `"C:/ibvap/photo.jpg"`.
