# CCTV Surveillance & IBVAP Knowledge Base

This directory maintains comprehensive architecture documentation, engineering decisions, troubleshooting guides, and knowledge items for the CCTV Surveillance platform and the **IBVAP** (Intelligent Border Video Analytics Platform) computer vision engine.

---

## Knowledge Index

| Document | Topic | Description |
| :--- | :--- | :--- |
| [IBVAP Track-Centric ANPR Architecture (Phases 1–6)](file:///c:/CCTV-servillance/knowledge/ibvap_track_centric_anpr_phases_1_to_6.md) | ANPR / Performance Architecture | Comprehensive specification of the 6-phase track-centric vehicle ANPR pipeline: data contracts, quality scoring, bounded buffer, best observation selection, controlled OCR, and multi-frame consensus. |
| [IBVAP Admin Camera & Spatial Architecture](file:///c:/CCTV-servillance/knowledge/ibvap_admin_camera_spatial_architecture.md) | Camera Config & Spatial Analytics | Core architectural specification of admin-controlled per-camera regions, borders, directional virtual lines, camera-isolated event rules, and cross-camera tracking invariance. |
| [Administrator Camera Configuration Guide](file:///c:/CCTV-servillance/ADMIN_CAMERA_CONFIGURATION_GUIDE.md) | User Guide / Configuration | Complete administrator guide for configuring per-camera restricted regions, perimeter borders, directional virtual lines (entry/exit), and camera-isolated event rules. |
| [IBVAP Vehicle Analysis & ANPR Pipeline (Legacy)](file:///c:/CCTV-servillance/knowledge/ibvap_vehicle_anpr_pipeline.md) | Computer Vision / ANPR | Complete end-to-end guide on single-frame vehicle detection, license plate candidate localization, EasyOCR text extraction, character normalization, tracking, and output schemas. |
| [Troubleshooting & Gotchas](file:///c:/CCTV-servillance/knowledge/troubleshooting_and_gotchas.md) | Debugging & Maintenance | Known pitfalls, frame-1 throttling gotchas, boundary margin clipping, CPU/GPU runtime optimizations, and testing strategies. |

---

## Platform Summary

* **IBVAP Core**: Source-agnostic Python analytics engine for CCTV streams and static frames.
* **Tracking & State**: Camera-isolated Kalman filter multi-object tracking (`PersistentTracker`).
* **Object Detection**: Pluggable YOLOv8 object detector (`YOLOv8Detector`) filtering for surveillance classes.
* **ANPR Pipeline**: Multi-strategy license plate localization + self-contained in-process EasyOCR engine.
* **Biometrics**: Face detection and cosine similarity feature verification.
* **Behavioral Analytics**: Virtual fence intrusion, loitering, sudden acceleration, night motion.
