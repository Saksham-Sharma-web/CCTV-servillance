# IBVAP Admin-Controlled Per-Camera Analytics & Spatial Architecture

## 1. Executive Summary & Core Objective

The **IBVAP (Intelligent Border Video Analytics Platform)** per-camera spatial architecture establishes a strictly isolated, administrator-governed system for regions, borders, virtual tripwire lines, and event rules.

### Problem Solved
Prior to this architectural update, spatial boundaries (lines and polygons) were registered in a flat global registry (`VirtualFenceAnalytics.boundaries`) that was evaluated across all video streams indiscriminately. If a virtual line or restricted polygon was configured for one camera, other camera streams would inadvertently evaluate tracks against that same boundary. Furthermore, there was no native support for virtual line crossing directions (e.g. distinguishing entries from exits), nor was there a formal data model for camera-specific event rules or cross-camera tracking isolation.

### Core Invariant
**REGIONS, BORDERS, VIRTUAL LINES, AND EVENT RULES ARE CONTROLLED WHOLLY AND SOLELY BY THE ADMINISTRATOR ON A PER-CAMERA BASIS.**
- IBVAP **never** automatically creates, infers, assigns, copies, or transfers a region from one camera to another.
- The administrator-defined camera configuration (`CameraConfig`) is the **single source of truth** for all spatial and camera-specific analytics.
- If a camera has no region or boundary configured, boundary processing is completely skipped and **zero** spatial alerts are generated.
- Unconfigured cameras fail safely: IBVAP never defaults to treating the entire frame as a region.

```
                            ADMINISTRATOR
                                  │
                                  ▼
                         Camera Configuration
                     (CameraConfig / CameraManager)
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
     Regions                   Borders                Virtual Lines
  (Polygons >= 3 pts)   (Perimeter Polylines)     (ENTRY, EXIT, BOTH)
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  │
                                  ▼
                     Camera-Specific Event Rules
              (Enabled event types, direction, classes)
                                  │
                                  ▼
                        AI Detection & Tracking
                (YOLOv8 + Isolated PersistentTracker)
                                  │
                                  ▼
                      Cross-Camera Association
             (CrossCameraTracker: Read-Only on Config)
                                  │
                                  ▼
                          Event Generation
             (Filtered strictly against CameraConfig)
                                  │
                                  ▼
                           Alert & Dispatch
                   (Debounced by EventEngine)
```

---

## 2. Conceptual Hierarchy & Data Contracts

The architecture is implemented in `ibvap/core/camera_config.py` with integration in `ibvap/core/pipeline.py` and `ibvap/analytics/virtual_fence.py`.

### 2.1. Line Direction Enum (`LineDirection`)
Defines the allowed movement direction across a 2D virtual line:
- `ENTRY`: Only triggers when an entity crosses in the forward direction (Left-to-Right relative to the directed line segment $A \rightarrow B$). Reverse crossings trigger a `DIRECTION_VIOLATION` event.
- `EXIT`: Only triggers when an entity crosses in the reverse direction (Right-to-Left relative to $A \rightarrow B$).
- `BIDIRECTIONAL` (or `ANY` / `BOTH`): Triggers in either direction.
- `LEFT_TO_RIGHT` & `RIGHT_TO_LEFT`: Explicit geometric aliases.

### 2.2. Region Dataclass (`Region`)
Represents an admin-defined polygonal area for a specific camera view:
```python
@dataclass
class Region:
    region_id: str
    name: str
    camera_id: str                  # Enforced: must match parent CameraConfig
    polygon: List[Tuple[int, int]]  # At least 3 vertices: [(x1, y1), (x2, y2), (x3, y3), ...]
    region_type: RegionType = RegionType.RESTRICTED
    target_classes: List[str] = field(default_factory=lambda: ["person", "car", "motorcycle", "truck"])
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2.3. Border Dataclass (`Border`)
Represents an admin-defined boundary polyline or perimeter segment:
```python
@dataclass
class Border:
    border_id: str
    name: str
    camera_id: str                  # Enforced: must match parent CameraConfig
    coordinates: List[Tuple[int, int]]  # At least 2 points: [(x1, y1), (x2, y2), ...]
    target_classes: List[str] = field(default_factory=lambda: ["person", "car", "motorcycle", "truck"])
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2.4. VirtualLine Dataclass (`VirtualLine`)
Represents an admin-defined directional tripwire:
```python
@dataclass
class VirtualLine:
    line_id: str
    name: str
    camera_id: str                  # Enforced: must match parent CameraConfig
    coordinates: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1, y1), (x2, y2))
    direction: LineDirection = LineDirection.BIDIRECTIONAL
    target_classes: List[str] = field(default_factory=lambda: ["person", "car", "motorcycle", "truck"])
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2.5. CameraEventRule Dataclass (`CameraEventRule`)
Controls whether and under what conditions an event is emitted:
```python
@dataclass
class CameraEventRule:
    rule_id: str
    name: str
    camera_id: str                  # Enforced: must match parent CameraConfig
    event_type: Union[EventType, str]
    region_id: Optional[str] = None
    border_id: Optional[str] = None
    line_id: Optional[str] = None
    direction: Optional[LineDirection] = None
    target_classes: Optional[List[str]] = None
    min_confidence: float = 0.0
    cooldown_seconds: Optional[float] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### 2.6. CameraConfig & CameraManager
- **`CameraConfig`**: Aggregates all regions, borders, virtual lines, event rules, and detection rules for one camera.
- **`CameraManager`**: Validates registration:
  - Verifies that every region's `camera_id` equals `config.camera_id`.
  - Verifies that every border's `camera_id` equals `config.camera_id`.
  - Verifies that every line's `camera_id` equals `config.camera_id`.
  - Verifies that every event rule's `camera_id` equals `config.camera_id`, and that referenced `region_id`, `border_id`, or `line_id` exist.
  - Stores isolated deep copies to guarantee external callers cannot mutate configurations in-place.

---

## 3. Directional Virtual Line Mathematics

To determine line-crossing transitions and directions deterministically, IBVAP utilizes 2D computational geometry in `ibvap/analytics/virtual_fence.py`.

### 3.1. Line Segment Intersection
Segment $(P_1, P_2)$ representing track centroid motion from $P_1 = (x_{prev}, y_{prev})$ to $P_2 = (x_{curr}, y_{curr})$ intersects line fence $(Q_1, Q_2)$ if and only if orientation tests alternate:
$$\text{ccw}(A, B, C) = (C_y - A_y) \cdot (B_x - A_x) > (B_y - A_y) \cdot (C_x - A_x)$$
$$\text{intersects} = (\text{ccw}(P_1, Q_1, Q_2) \neq \text{ccw}(P_2, Q_1, Q_2)) \land (\text{ccw}(P_1, P_2, Q_1) \neq \text{ccw}(P_1, P_2, Q_2))$$

### 3.2. Directional Orientation Determination
Given directed line segment from $A = (x_1, y_1)$ to $B = (x_2, y_2)$ with vector $\vec{D} = (x_2 - x_1, y_2 - y_1)$, the 2D cross product of vector $\vec{D}$ and vector $\vec{AP}$ indicates which side of the line point $P$ lies on:
$$\text{Cross}(A, B, P) = (x_2 - x_1) \cdot (P_y - y_1) - (y_2 - y_1) \cdot (P_x - x_1)$$
- If $\text{Cross}(A, B, P) > 0$: Point $P$ is on the **LEFT** side of vector $\vec{AB}$.
- If $\text{Cross}(A, B, P) < 0$: Point $P$ is on the **RIGHT** side of vector $\vec{AB}$.

When a track crosses the line:
- $\text{Cross}(P_{prev}) > 0 \land \text{Cross}(P_{curr}) \le 0 \implies$ **`ENTRY`** (Left-to-Right).
- $\text{Cross}(P_{prev}) < 0 \land \text{Cross}(P_{curr}) \ge 0 \implies$ **`EXIT`** (Right-to-Left).

If an admin configured `LineDirection.ENTRY` and a track moves in reverse (Exit direction), IBVAP generates a **`DIRECTION_VIOLATION`** event (`WRONG_WAY_CROSSING`) rather than a valid entry crossing.

---

## 4. Cross-Camera Tracking Invariance (`CrossCameraTracker`)

Cross-camera tracking provides intelligence across multiple camera feeds:
- Multi-camera entity tracing (masked persons, security suspects).
- Trajectory and camera sequence recording (`camera_sequence: ["cam-01", "cam-02"]`).
- Persistent global track assignment (`global_track_id`).

### Architectural Invariant
`CrossCameraTracker` maintains an association between local `track_id` and global `global_track_id` using biometric face embeddings (`identity_id`), license plates (`plate_number`), or spatiotemporal proximity.
- **`CrossCameraTracker` has strictly READ-ONLY access with respect to camera configuration.**
- It does **not** have access to mutate, alter, or share `CameraConfig`.
- When a tracked person enters Camera 2, IBVAP evaluates the person against Camera 2's admin rules. Camera 1's configuration is completely uninfluenced by Camera 2's existence.

---

## 5. Primary Pipeline Execution Flow

In `IBVAPPipeline.process_frame(frame, camera_id="camera-01", timestamp=None)`:

1. **Step 0: Load Camera Configuration**:
   Retrieves `cam_config = self.camera_manager.get_camera_config(camera_id)`.
   If unconfigured, `cam_config` is `None`.
2. **Step 1: Object Detection**:
   Runs YOLOv8 detector. Applies per-camera detection rules (confidence thresholds, target classes, min bbox area) if defined in `cam_config.detection_rules`.
3. **Step 2: Multi-Object Tracking**:
   Runs camera-isolated Kalman tracker (`self.get_tracker(camera_id)`).
4. **Step 2b: Cross-Camera Association**:
   Runs `self.cross_camera_tracker.associate_tracks(camera_id, tracks, timestamp=now)` to assign `global_track_id`.
5. **Step 3 & 4: Biometrics & ANPR**:
   Evaluates face verification and license plate recognition per track.
6. **Step 5: Behavioral & Spatial Analytics**:
   Evaluates `self.virtual_fence.process_tracks(tracks, camera_id, now, camera_config=cam_config)`.
   - If `cam_config` has no regions, borders, or lines, **spatial processing is skipped completely**.
7. **Step 5b: Camera-Specific Event Rule Filtering**:
   If `cam_config.enabled_event_types` is set, non-whitelisted event types are dropped.
   If `cam_config.event_rules` exist, events are validated against rule constraints (`min_confidence`, `target_classes`, `direction`).
8. **Step 6 & 7: Debounce & Integration**:
   Passes candidate events through `EventEngine` for deduplication and dispatches to Redis / DB.

---

## 6. Verification & Test Suite Matrix

The architecture is covered by automated unit and integration tests:

| Test File | Key Test Cases | Purpose |
| :--- | :--- | :--- |
| `ibvap/tests/test_camera_architecture.py` | `test_camera_manager_validation_and_rejection` | Verifies rejection of mismatched `camera_id` in regions, borders, lines, rules. |
| `ibvap/tests/test_camera_architecture.py` | `test_strict_camera_isolation_no_configuration_leakage` | Verifies Camera 1 (Restricted) vs Camera 2 (Normal) isolation with zero leakage. |
| `ibvap/tests/test_camera_architecture.py` | `test_directional_virtual_line_crossing` | Verifies forward crossing (`LINE_CROSSING`) vs reverse wrong-way (`DIRECTION_VIOLATION`). |
| `ibvap/tests/test_camera_architecture.py` | `test_cross_camera_tracking_does_not_modify_configuration` | Verifies `global_track_id` continuity across cameras without config modification. |
| `ibvap/tests/test_camera_architecture.py` | `test_missing_camera_configuration_fails_safely` | Verifies unconfigured cameras generate 0 spatial alerts and never crash. |
| `ibvap/tests/test_camera_architecture.py` | `test_camera_specific_event_rules_filtering` | Verifies rule suppression, confidence thresholds, and class filtering. |
| `ibvap/tests/test_virtual_fence.py` | `test_polygon_intrusion_and_debouncing`, `test_line_crossing_intrusion` | Verifies polygon intrusion and debouncing. |
| `ibvap/tests/test_pipeline_e2e.py` | `test_pipeline_end_to_end_frame_processing` | Verifies full pipeline frame ingestion and event emission. |

**Regression Result**: 119/119 tests passing (`pytest ibvap/tests`).
