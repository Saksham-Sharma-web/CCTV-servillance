# Administrator Guide: Per-Camera Configuration & Spatial Analytics

This guide explains how an **Administrator** can define and manage camera-specific configurations in the **IBVAP** (Intelligent Border Video Analytics Platform) architecture.

---

## 1. Core Principles

1. **Administrator as the Single Source of Truth**:
   Every region, border, virtual line, and event rule is decided wholly and solely by the administrator on a per-camera basis.
2. **Strict Camera Isolation**:
   Configurations are isolated. If **Camera 1** has a restricted area and **Camera 2** does not, Camera 2 will **never** inherit, infer, copy, or use Camera 1's configuration.
3. **No Automatic Inferences**:
   IBVAP never automatically creates default regions and never assumes an unconfigured camera's entire frame is a restricted zone.
4. **Read-Only Cross-Camera Intelligence**:
   Tracked persons or vehicles moving across multiple cameras are correlated via `global_track_id`, but cross-camera tracking **never alters or transfers camera configurations**.

---

## 2. Configuration Hierarchy

```
ADMINISTRATOR
  ↓
CameraConfig (camera_id)
  ├── Regions (Polygonal Restricted / Monitored Areas)
  ├── Borders (Perimeter Polylines)
  ├── Virtual Lines (Directional Tripwires: ENTRY, EXIT, BIDIRECTIONAL)
  ├── CameraEventRules (Enabled event types, direction filtering, class filters)
  └── DetectionRules (Confidence thresholds, target classes)
```

---

## 3. Quick Start: Configuring Camera 1 as Restricted & Camera 2 as Normal

Here is how an administrator sets up **Camera 1** with a restricted polygon and entry line, while setting **Camera 2** as a normal camera with zero boundaries:

```python
from ibvap.core.pipeline import IBVAPPipeline
from ibvap.core.camera_config import (
    CameraConfig,
    Region,
    Border,
    VirtualLine,
    LineDirection,
    RegionType,
    CameraEventRule,
)

# 1. Initialize the IBVAP Pipeline
pipeline = IBVAPPipeline()

# ═════════════════════════════════════════════════════════════════════════════
# CAMERA 1: RESTRICTED SECURITY AREA
# ═════════════════════════════════════════════════════════════════════════════
cam1 = CameraConfig(camera_id="camera-01", name="Perimeter Gate 1")

# Add a restricted polygon region (at least 3 coordinates: [(x, y), ...])
cam1.regions["restricted-vault"] = Region(
    region_id="restricted-vault",
    name="Restricted Vault Zone",
    camera_id="camera-01",
    polygon=[(100, 100), (400, 100), (400, 400), (100, 400)],
    region_type=RegionType.RESTRICTED,
    target_classes=["person", "car", "motorcycle"]
)

# Add a directional virtual line (ENTRY only)
cam1.virtual_lines["entry-tripwire"] = VirtualLine(
    line_id="entry-tripwire",
    name="North Entry Tripwire",
    camera_id="camera-01",
    coordinates=((250, 0), (250, 500)),  # ((x1, y1), (x2, y2))
    direction=LineDirection.ENTRY,       # Only triggers on forward entry!
    target_classes=["person"]
)

# Add a perimeter border line
cam1.borders["north-fence"] = Border(
    border_id="north-fence",
    name="North Perimeter Fence",
    camera_id="camera-01",
    coordinates=[(0, 100), (640, 100)],
    target_classes=["person"]
)

# Register Camera 1 configuration
pipeline.set_camera_config(cam1)


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA 2: NORMAL CAMERA (No regions, no borders, no tripwires)
# ═════════════════════════════════════════════════════════════════════════════
cam2 = CameraConfig(camera_id="camera-02", name="Office Lobby")
pipeline.set_camera_config(cam2)
```

---

## 4. How Event Processing Behaves

When video frames are ingested:

```python
# Video Stream 1 arrives from Camera 1
result_cam1 = pipeline.process_frame(frame_camera1, camera_id="camera-01")

# Video Stream 2 arrives from Camera 2
result_cam2 = pipeline.process_frame(frame_camera2, camera_id="camera-02")
```

### The Behavior:
- **Camera 1**:
  - Person enters `(150, 150)` $\rightarrow$ Inside `restricted-vault` polygon $\rightarrow$ **`REGION_INTRUSION`** alert generated.
  - Person crosses `x = 250` in the entry direction $\rightarrow$ **`LINE_CROSSING`** alert generated.
- **Camera 2**:
  - Person enters `(150, 150)` on Camera 2 $\rightarrow$ No regions configured on Camera 2 $\rightarrow$ **Zero spatial alerts generated**.
  - Object detection and tracking run normally, but boundary evaluation is completely skipped.
  - Camera 2 **never** receives Camera 1's restricted zone.

---

## 5. Virtual Lines & Directional Tripwires

The administrator can define tripwire lines with specific directional constraints using `LineDirection`:

| Direction | Behavior |
| :--- | :--- |
| `LineDirection.ENTRY` | Triggers **`LINE_CROSSING`** only when moving in the forward (Left-to-Right / A-to-B) direction. Crossing in reverse triggers a **`DIRECTION_VIOLATION`** event. |
| `LineDirection.EXIT` | Triggers **`LINE_CROSSING`** only when moving in the exit (Right-to-Left / B-to-A) direction. |
| `LineDirection.BIDIRECTIONAL` | Triggers **`LINE_CROSSING`** in both directions. |

### Example: Setting an Exit-Only Tripwire

```python
cam_exit = CameraConfig(camera_id="camera-exit-gate", name="Exit Gate")
cam_exit.virtual_lines["gate-exit"] = VirtualLine(
    line_id="gate-exit",
    name="Exit Lane Only",
    camera_id="camera-exit-gate",
    coordinates=((300, 0), (300, 720)),
    direction=LineDirection.EXIT,
    target_classes=["car", "truck", "motorcycle"]
)
pipeline.set_camera_config(cam_exit)
```

---

## 6. Camera-Specific Event Rules & Filtering

The administrator can control which events fire for each camera using `CameraEventRule`:

```python
# Rule: Suppress person alerts in a specific zone (only alert on vehicles)
rule = CameraEventRule(
    rule_id="vehicles-only-rule",
    name="Ignore Persons in Delivery Bay",
    camera_id="camera-01",
    event_type="REGION_INTRUSION",
    region_id="restricted-vault",
    target_classes=["car", "truck", "van"],  # Persons are filtered out!
    min_confidence=0.75,                     # Minimum AI confidence required
    cooldown_seconds=5.0,                    # Debounce duplicate alerts
    enabled=True
)
pipeline.add_camera_event_rule("camera-01", rule)
```

To restrict a camera to only emit certain event types:

```python
# Camera will only emit FACE_MATCHED and PLATE_DETECTED alerts
cam1.enabled_event_types = {"FACE_MATCHED", "PLATE_DETECTED"}
pipeline.set_camera_config(cam1)
```

---

## 7. Dynamic Management at Runtime

The administrator can dynamically query, add, modify, or delete configurations without restarting the pipeline:

```python
# Retrieve active configuration for a camera
config = pipeline.get_camera_config("camera-01")
print(f"Active regions for Camera 1: {list(config.regions.keys())}")

# Dynamically add a new region
new_zone = Region(
    region_id="loading-dock",
    name="Loading Dock",
    camera_id="camera-01",
    polygon=[(20, 20), (80, 20), (80, 80), (20, 80)],
    region_type=RegionType.MONITORED
)
pipeline.add_camera_region("camera-01", new_zone)

# Dynamically remove a boundary
pipeline.remove_boundary("loading-dock", camera_id="camera-01")
```

---

## 8. JSON / Dictionary Serialization

Configurations can be loaded directly from JSON or dictionaries (e.g. from an administrative web UI or database):

```python
camera_dict = {
    "camera_id": "camera-east-01",
    "name": "East Gate",
    "regions": {
        "zone-1": {
            "region_id": "zone-1",
            "name": "Restricted Zone",
            "camera_id": "camera-east-01",
            "polygon": [[100, 100], [300, 100], [300, 300], [100, 300]],
            "region_type": "RESTRICTED",
            "target_classes": ["person", "car"],
            "metadata": {}
        }
    },
    "borders": {},
    "virtual_lines": {
        "line-1": {
            "line_id": "line-1",
            "name": "Inbound Line",
            "camera_id": "camera-east-01",
            "coordinates": [[200, 0], [200, 480]],
            "direction": "ENTRY",
            "target_classes": ["person"],
            "metadata": {}
        }
    },
    "event_rules": {},
    "detection_rules": None,
    "enabled_event_types": None,
    "metadata": {}
}

# Converting back to CameraConfig
regions = {
    k: Region(polygon=[tuple(p) for p in v["polygon"]], **{k2: v2 for k2, v2 in v.items() if k2 != "polygon"})
    for k, v in camera_dict["regions"].items()
}
virtual_lines = {
    k: VirtualLine(
        coordinates=(tuple(v["coordinates"][0]), tuple(v["coordinates"][1])),
        **{k2: v2 for k2, v2 in v.items() if k2 != "coordinates"}
    )
    for k, v in camera_dict["virtual_lines"].items()
}

cam_config = CameraConfig(
    camera_id=camera_dict["camera_id"],
    name=camera_dict["name"],
    regions=regions,
    virtual_lines=virtual_lines,
)
pipeline.set_camera_config(cam_config)
```

---

## 9. Visual Debug Overlay

When using `pipeline.draw_debug(frame, result)`:
- Frames from **Camera 1** will display Camera 1's configured zones, directional line arrows, and HUD alerts.
- Frames from **Camera 2** will display **only detections and track IDs**, with no boundary overlays.
- Camera 1's zones are never rendered on Camera 2.
