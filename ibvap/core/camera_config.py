"""
IBVAP Admin Camera Configuration System.
Provides strict, admin-controlled, per-camera configuration models and validation.
Enforces the invariant:
- Region, border, line, and event rules are controlled wholly and solely by the admin.
- IBVAP never automatically creates, infers, assigns, copies, or transfers configurations across cameras.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional, Set, Union
import copy
import logging

from .types import EventType

logger = logging.getLogger("ibvap.camera_config")


class LineDirection(str, Enum):
    """
    Directional constraint for virtual line crossings.
    """
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    BIDIRECTIONAL = "BIDIRECTIONAL"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"

    @classmethod
    def from_str(cls, value: Union[str, "LineDirection"]) -> "LineDirection":
        if isinstance(value, cls):
            return value
        v = str(value).strip().upper()
        mapping = {
            "ENTRY": cls.ENTRY,
            "ENTERING": cls.ENTRY,
            "IN": cls.ENTRY,
            "EXIT": cls.EXIT,
            "EXITING": cls.EXIT,
            "OUT": cls.EXIT,
            "BOTH": cls.BIDIRECTIONAL,
            "BIDIRECTIONAL": cls.BIDIRECTIONAL,
            "ANY": cls.BIDIRECTIONAL,
            "LEFT_TO_RIGHT": cls.LEFT_TO_RIGHT,
            "LTR": cls.LEFT_TO_RIGHT,
            "RIGHT_TO_LEFT": cls.RIGHT_TO_LEFT,
            "RTL": cls.RIGHT_TO_LEFT,
        }
        if v in mapping:
            return mapping[v]
        raise ValueError(
            f"Invalid LineDirection '{value}'. Valid options are: "
            f"{[e.value for e in cls]} (or aliases 'both', 'entering', 'exiting', 'any')."
        )


class RegionType(str, Enum):
    RESTRICTED = "RESTRICTED"
    MONITORED = "MONITORED"
    LOITERING = "LOITERING"
    HAZARD = "HAZARD"
    CUSTOM = "CUSTOM"


@dataclass
class Region:
    """
    Admin-defined polygonal area for a specific camera view.
    Every region belongs to exactly one camera configuration.
    """
    region_id: str
    name: str
    camera_id: str
    polygon: List[Tuple[int, int]]  # Minimum 3 vertices: [(x1, y1), (x2, y2), (x3, y3), ...]
    region_type: RegionType = RegionType.RESTRICTED
    target_classes: List[str] = field(
        default_factory=lambda: ["person", "car", "motorcycle", "truck", "bus", "van", "suv"]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.region_id or not isinstance(self.region_id, str):
            raise ValueError("region_id must be a non-empty string.")
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if not self.polygon or len(self.polygon) < 3:
            raise ValueError(
                f"Region '{self.region_id}' polygon must contain at least 3 vertices, got {len(self.polygon) if self.polygon else 0}."
            )
        if isinstance(self.region_type, str):
            try:
                self.region_type = RegionType(self.region_type.upper())
            except ValueError:
                self.region_type = RegionType.CUSTOM

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "polygon": [list(pt) for pt in self.polygon],
            "region_type": self.region_type.value if hasattr(self.region_type, "value") else str(self.region_type),
            "target_classes": list(self.target_classes),
            "metadata": self.metadata,
        }


@dataclass
class Border:
    """
    Admin-defined border line or boundary polyline for a specific camera view.
    Every border belongs to exactly one camera configuration.
    """
    border_id: str
    name: str
    camera_id: str
    coordinates: List[Tuple[int, int]]  # At least 2 points
    target_classes: List[str] = field(
        default_factory=lambda: ["person", "car", "motorcycle", "truck", "bus", "van", "suv"]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.border_id or not isinstance(self.border_id, str):
            raise ValueError("border_id must be a non-empty string.")
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if not self.coordinates or len(self.coordinates) < 2:
            raise ValueError(
                f"Border '{self.border_id}' coordinates must contain at least 2 points, got {len(self.coordinates) if self.coordinates else 0}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "border_id": self.border_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "coordinates": [list(pt) for pt in self.coordinates],
            "target_classes": list(self.target_classes),
            "metadata": self.metadata,
        }


@dataclass
class VirtualLine:
    """
    Admin-defined directional virtual tripwire line for a specific camera view.
    Every virtual line belongs to exactly one camera configuration.
    """
    line_id: str
    name: str
    camera_id: str
    coordinates: Tuple[Tuple[int, int], Tuple[int, int]]  # ((x1, y1), (x2, y2))
    direction: LineDirection = LineDirection.BIDIRECTIONAL
    target_classes: List[str] = field(
        default_factory=lambda: ["person", "car", "motorcycle", "truck", "bus", "van", "suv"]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.line_id or not isinstance(self.line_id, str):
            raise ValueError("line_id must be a non-empty string.")
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if not self.coordinates or len(self.coordinates) < 2:
            raise ValueError(
                f"VirtualLine '{self.line_id}' coordinates must contain exactly 2 endpoints ((x1, y1), (x2, y2))."
            )
        self.direction = LineDirection.from_str(self.direction)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "line_id": self.line_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "coordinates": [list(self.coordinates[0]), list(self.coordinates[1])],
            "direction": self.direction.value,
            "target_classes": list(self.target_classes),
            "metadata": self.metadata,
        }


@dataclass
class CameraEventRule:
    """
    Admin-configured event triggering rule for a specific camera.
    Controls whether, when, and under what constraints events are produced.
    """
    rule_id: str
    name: str
    camera_id: str
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

    def __post_init__(self):
        if not self.rule_id or not isinstance(self.rule_id, str):
            raise ValueError("rule_id must be a non-empty string.")
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")
        if self.direction is not None:
            self.direction = LineDirection.from_str(self.direction)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "camera_id": self.camera_id,
            "event_type": self.event_type.value if hasattr(self.event_type, "value") else str(self.event_type),
            "region_id": self.region_id,
            "border_id": self.border_id,
            "line_id": self.line_id,
            "direction": self.direction.value if self.direction else None,
            "target_classes": self.target_classes,
            "min_confidence": self.min_confidence,
            "cooldown_seconds": self.cooldown_seconds,
            "enabled": self.enabled,
            "metadata": self.metadata,
        }


@dataclass
class DetectionRule:
    """
    Per-camera detection constraints (e.g. target classes, confidence thresholds).
    """
    camera_id: str
    target_classes: Optional[List[str]] = None
    confidence_threshold: Optional[float] = None
    min_bbox_area: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "target_classes": self.target_classes,
            "confidence_threshold": self.confidence_threshold,
            "min_bbox_area": self.min_bbox_area,
            "metadata": self.metadata,
        }


@dataclass
class CameraConfig:
    """
    Independent camera configuration defined by the administrator.
    This is the single source of truth for region-based and camera-specific analytics.
    """
    camera_id: str
    name: Optional[str] = None
    regions: Dict[str, Region] = field(default_factory=dict)
    borders: Dict[str, Border] = field(default_factory=dict)
    virtual_lines: Dict[str, VirtualLine] = field(default_factory=dict)
    event_rules: Dict[str, CameraEventRule] = field(default_factory=dict)
    detection_rules: Optional[DetectionRule] = None
    enabled_event_types: Optional[Set[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.camera_id or not isinstance(self.camera_id, str):
            raise ValueError("camera_id must be a non-empty string.")

    @property
    def has_spatial_boundaries(self) -> bool:
        """Returns True if any region, border, or virtual line is configured."""
        return bool(self.regions or self.borders or self.virtual_lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name or self.camera_id,
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "borders": {k: v.to_dict() for k, v in self.borders.items()},
            "virtual_lines": {k: v.to_dict() for k, v in self.virtual_lines.items()},
            "event_rules": {k: v.to_dict() for k, v in self.event_rules.items()},
            "detection_rules": self.detection_rules.to_dict() if self.detection_rules else None,
            "enabled_event_types": list(self.enabled_event_types) if self.enabled_event_types is not None else None,
            "metadata": self.metadata,
        }


class CameraManager:
    """
    Administrator Camera Configuration Registry.
    Guarantees:
    - Every region, border, virtual line belongs to exactly one camera.
    - Configuration from Camera A never leaks into Camera B.
    - Missing configuration fails safely (zero assumptions, no default regions).
    - Immutable isolated retrieval (no accidental external mutation).
    """

    def __init__(self):
        self._configs: Dict[str, CameraConfig] = {}

    def register_camera(self, config: CameraConfig) -> None:
        """
        Registers or updates a camera configuration with strict validation.
        """
        if not isinstance(config, CameraConfig):
            raise TypeError(f"Expected CameraConfig, got {type(config).__name__}.")

        cid = config.camera_id
        if not cid or not isinstance(cid, str):
            raise ValueError("camera_id must be a non-empty string.")

        # Validation Rule 1: Every region must match config.camera_id
        for rid, reg in config.regions.items():
            if reg.camera_id != cid:
                raise ValueError(
                    f"Configuration conflict: Region '{rid}' has camera_id '{reg.camera_id}', "
                    f"which does not match camera configuration '{cid}'."
                )

        # Validation Rule 2: Every border must match config.camera_id
        for bid, bor in config.borders.items():
            if bor.camera_id != cid:
                raise ValueError(
                    f"Configuration conflict: Border '{bid}' has camera_id '{bor.camera_id}', "
                    f"which does not match camera configuration '{cid}'."
                )

        # Validation Rule 3: Every virtual line must match config.camera_id
        for lid, line in config.virtual_lines.items():
            if line.camera_id != cid:
                raise ValueError(
                    f"Configuration conflict: VirtualLine '{lid}' has camera_id '{line.camera_id}', "
                    f"which does not match camera configuration '{cid}'."
                )

        # Validation Rule 4: Every event rule must reference this camera
        for r_id, rule in config.event_rules.items():
            if rule.camera_id != cid:
                raise ValueError(
                    f"Configuration conflict: Event rule '{r_id}' has camera_id '{rule.camera_id}', "
                    f"which does not match camera configuration '{cid}'."
                )
            if rule.region_id and rule.region_id not in config.regions:
                raise ValueError(
                    f"Event rule '{r_id}' references non-existent region '{rule.region_id}' on camera '{cid}'."
                )
            if rule.border_id and rule.border_id not in config.borders:
                raise ValueError(
                    f"Event rule '{r_id}' references non-existent border '{rule.border_id}' on camera '{cid}'."
                )
            if rule.line_id and rule.line_id not in config.virtual_lines:
                raise ValueError(
                    f"Event rule '{r_id}' references non-existent virtual line '{rule.line_id}' on camera '{cid}'."
                )

        # Validation Rule 5: DetectionRule must match config.camera_id
        if config.detection_rules and config.detection_rules.camera_id != cid:
            raise ValueError(
                f"Configuration conflict: DetectionRule camera_id '{config.detection_rules.camera_id}' "
                f"does not match camera configuration '{cid}'."
            )

        # Store isolated deep copy to prevent mutation leakage
        self._configs[cid] = copy.deepcopy(config)
        logger.info(f"Admin registered camera configuration for '{cid}'.")

    def get_camera_config(self, camera_id: str) -> Optional[CameraConfig]:
        """
        Retrieves camera configuration.
        Returns None if not configured. Never invents default regions.
        """
        cfg = self._configs.get(camera_id)
        return copy.deepcopy(cfg) if cfg is not None else None

    def has_camera_config(self, camera_id: str) -> bool:
        return camera_id in self._configs

    def ensure_camera_exists(self, camera_id: str, name: Optional[str] = None) -> CameraConfig:
        """
        Creates an empty camera configuration if one does not exist yet.
        Note: Contains NO default regions, borders, or lines.
        """
        if camera_id not in self._configs:
            self._configs[camera_id] = CameraConfig(camera_id=camera_id, name=name or camera_id)
        return self._configs[camera_id]

    def add_region(self, camera_id: str, region: Region) -> None:
        if region.camera_id != camera_id:
            raise ValueError(
                f"Region '{region.region_id}' camera_id '{region.camera_id}' does not match target camera '{camera_id}'."
            )
        cam = self.ensure_camera_exists(camera_id)
        cam.regions[region.region_id] = copy.deepcopy(region)

    def add_border(self, camera_id: str, border: Border) -> None:
        if border.camera_id != camera_id:
            raise ValueError(
                f"Border '{border.border_id}' camera_id '{border.camera_id}' does not match target camera '{camera_id}'."
            )
        cam = self.ensure_camera_exists(camera_id)
        cam.borders[border.border_id] = copy.deepcopy(border)

    def add_virtual_line(self, camera_id: str, line: VirtualLine) -> None:
        if line.camera_id != camera_id:
            raise ValueError(
                f"VirtualLine '{line.line_id}' camera_id '{line.camera_id}' does not match target camera '{camera_id}'."
            )
        cam = self.ensure_camera_exists(camera_id)
        cam.virtual_lines[line.line_id] = copy.deepcopy(line)

    def add_event_rule(self, camera_id: str, rule: CameraEventRule) -> None:
        if rule.camera_id != camera_id:
            raise ValueError(
                f"Event rule '{rule.rule_id}' camera_id '{rule.camera_id}' does not match target camera '{camera_id}'."
            )
        cam = self.ensure_camera_exists(camera_id)
        if rule.region_id and rule.region_id not in cam.regions:
            raise ValueError(
                f"Event rule '{rule.rule_id}' references non-existent region '{rule.region_id}' on camera '{camera_id}'."
            )
        if rule.border_id and rule.border_id not in cam.borders:
            raise ValueError(
                f"Event rule '{rule.rule_id}' references non-existent border '{rule.border_id}' on camera '{camera_id}'."
            )
        if rule.line_id and rule.line_id not in cam.virtual_lines:
            raise ValueError(
                f"Event rule '{rule.rule_id}' references non-existent virtual line '{rule.line_id}' on camera '{camera_id}'."
            )
        cam.event_rules[rule.rule_id] = copy.deepcopy(rule)

    def remove_camera(self, camera_id: str) -> bool:
        if camera_id in self._configs:
            del self._configs[camera_id]
            return True
        return False

    def list_cameras(self) -> List[str]:
        return list(self._configs.keys())

    def clear(self) -> None:
        self._configs.clear()
