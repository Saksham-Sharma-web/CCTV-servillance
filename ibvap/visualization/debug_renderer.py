"""
Visual Debug Renderer for IBVAP.
Renders high-visibility overlays on BGR frames:
- Bounding boxes colored by class/state
- Persistent visual track_id
- Biometric identity_id tag (separated clearly from track_id)
- License plate number & watchlist badge
- Virtual fence lines and polygons
- Active alert Heads-Up Display (HUD) banner
"""

from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

from ..core.types import PipelineResult, Track, VirtualBoundary, ZoneType, WatchlistCategory

CLASS_COLORS = {
    "person": (0, 255, 128),      # Neon Green
    "car": (255, 200, 0),         # Cyan / Yellow-Blue
    "motorcycle": (255, 128, 0),  # Orange
    "bus": (255, 100, 100),       # Light Blue
    "truck": (200, 150, 255),     # Pink-Purple
    "bicycle": (0, 220, 255),     # Yellow
    "backpack": (180, 105, 255),  # Hot Pink
    "handbag": (180, 105, 255),
    "suitcase": (180, 105, 255),
}

DEFAULT_COLOR = (200, 200, 200)
ALERT_COLOR = (0, 0, 255)       # Red
INTRUSION_ZONE_COLOR = (0, 0, 255)
NORMAL_ZONE_COLOR = (0, 255, 0)


class DebugRenderer:
    """
    Renders analytics debug overlays on raw BGR frames.
    """

    def __init__(self):
        pass

    def render(
        self,
        frame: np.ndarray,
        result: PipelineResult,
        boundaries: Optional[Dict[str, VirtualBoundary]] = None
    ) -> np.ndarray:
        """
        Draws visual annotations onto a copy of the input frame.
        """
        if frame is None or frame.size == 0:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Draw Virtual Boundaries
        if boundaries:
            for b in boundaries.values():
                if b.zone_type == ZoneType.LINE and len(b.coordinates) >= 2:
                    p1, p2 = b.coordinates[0], b.coordinates[1]
                    cv2.line(annotated, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(
                        annotated, f"LINE: {b.name}",
                        (p1[0], max(20, p1[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                    )
                elif b.zone_type == ZoneType.POLYGON and len(b.coordinates) >= 3:
                    pts = np.array(b.coordinates, dtype=np.int32).reshape((-1, 1, 2))
                    # Draw semi-transparent overlay
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, [pts], (0, 180, 255))
                    cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
                    cv2.polylines(annotated, [pts], True, (0, 180, 255), 2, cv2.LINE_AA)
                    first_pt = b.coordinates[0]
                    cv2.putText(
                        annotated, f"ZONE: {b.name}",
                        (first_pt[0], max(20, first_pt[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2, cv2.LINE_AA
                    )

        # 2. Draw Tracked Objects
        for track in result.tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Determine color and labels
            if track.class_name == "person":
                if track.identity_id:
                    # Authorized Match (>= 60%): Green Box
                    color = (0, 255, 128)
                    pct = (track.identity_confidence * 100.0) if track.identity_confidence is not None else 100.0
                    label_parts = [f"PERSON #{track.track_id}", f"MATCH: {track.identity_name} ({pct:.1f}%)"]
                else:
                    # Unknown Person (< 60%): Red Box
                    color = (0, 0, 255)
                    if track.identity_confidence is not None:
                        pct = track.identity_confidence * 100.0
                        label_parts = [f"PERSON #{track.track_id}", f"UNKNOWN PERSON ({pct:.1f}%)"]
                    else:
                        label_parts = [f"PERSON #{track.track_id}", "UNKNOWN PERSON"]
            else:
                color = CLASS_COLORS.get(track.class_name, DEFAULT_COLOR)
                label_parts = [f"{track.class_name.upper()} #{track.track_id}"]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw centroid trail
            if len(track.history) > 1:
                for i in range(1, len(track.history)):
                    cv2.line(annotated, track.history[i - 1], track.history[i], color, 1, cv2.LINE_AA)

            if track.plate_number:
                cat_str = f"[{track.plate_category.value}]" if track.plate_category else ""
                label_parts.append(f"PLATE:{track.plate_number} {cat_str}")

            label = " | ".join(label_parts)

            # Background pill for text readability
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            pill_y1 = max(0, y1 - text_h - 8)
            pill_y2 = y1
            pill_x2 = min(w, x1 + text_w + 10)

            cv2.rectangle(annotated, (x1, pill_y1), (pill_x2, pill_y2), color, -1)
            # Text color: Black on Green, White on Red
            text_color = (0, 0, 0) if color == (0, 255, 128) else (255, 255, 255)
            cv2.putText(
                annotated, label,
                (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA
            )

        # 3. Draw Heads-Up Display (HUD) for Active Events
        if result.events:
            banner_h = min(120, 28 * len(result.events) + 12)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.65, annotated, 0.35, 0, annotated)

            for i, ev in enumerate(result.events[:4]):
                event_name = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                detail = ""
                if ev.track_id:
                    detail += f"Track #{ev.track_id} "
                if ev.identity_id:
                    detail += f"Identity: {ev.identity_id} "
                if "zone_name" in ev.metadata:
                    detail += f"Zone: {ev.metadata['zone_name']} "
                if "plate_number" in ev.metadata:
                    detail += f"Plate: {ev.metadata['plate_number']} "

                hud_text = f"🚨 ALERT [{event_name}] {detail}"
                cv2.putText(
                    annotated, hud_text,
                    (15, 22 + (i * 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 50, 255), 2, cv2.LINE_AA
                )
                cv2.putText(
                    annotated, hud_text,
                    (15, 22 + (i * 26)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
                )

        return annotated
