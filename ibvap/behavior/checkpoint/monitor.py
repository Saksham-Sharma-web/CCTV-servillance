"""
Checkpoint & Virtual Gate Passage Monitoring Subsystem.
Tracks directional entry/exit passage, maintains inbound/outbound counts, and flags unauthorized reverse crossings.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import time
import uuid

from ibvap.core.types import Track, AnalyticsEvent, EventType
from ibvap.core.camera_config import LineDirection
from ibvap.core.config import IBVAPConfig, default_config
from ibvap.analytics.virtual_fence import lines_intersect, determine_line_crossing_direction


@dataclass
class CheckpointGate:
    """
    Defines a virtual checkpoint line / turnstile / barrier.
    """
    gate_id: str
    name: str
    line_start: Tuple[int, int]
    line_end: Tuple[int, int]
    allowed_direction: LineDirection = LineDirection.BIDIRECTIONAL
    camera_id: str = "camera-01"


@dataclass
class PassageRecord:
    """
    Log of an individual entity crossing a checkpoint.
    """
    gate_id: str
    track_id: int
    direction: LineDirection
    timestamp: float
    is_violation: bool
    identity_id: Optional[str] = None
    plate_number: Optional[str] = None


class CheckpointMonitor:
    """
    Monitors checkpoint line crossings, passage statistics, and barrier direction compliance.
    """

    def __init__(self, config: Optional[IBVAPConfig] = None):
        self.config = config or default_config
        self.gates: Dict[str, CheckpointGate] = {}
        # Track statistics
        self.inbound_counts: Dict[str, int] = {}
        self.outbound_counts: Dict[str, int] = {}
        self.violations_counts: Dict[str, int] = {}
        self.passage_history: List[PassageRecord] = []
        # Map: (gate_id, track_id) -> last crossing timestamp
        self._last_crossings: Dict[Tuple[str, int], float] = {}

    def add_gate(self, gate: CheckpointGate):
        """Registers a checkpoint gate."""
        self.gates[gate.gate_id] = gate
        self.inbound_counts.setdefault(gate.gate_id, 0)
        self.outbound_counts.setdefault(gate.gate_id, 0)
        self.violations_counts.setdefault(gate.gate_id, 0)

    def remove_gate(self, gate_id: str):
        if gate_id in self.gates:
            del self.gates[gate_id]

    def process(
        self,
        tracks: List[Track],
        camera_id: str = "camera-01",
        timestamp: Optional[float] = None,
    ) -> List[AnalyticsEvent]:
        """
        Evaluates tracks crossing active checkpoint gates.
        """
        now = timestamp if timestamp is not None else time.time()
        events: List[AnalyticsEvent] = []

        applicable_gates = [
            g for g in self.gates.values()
            if g.camera_id == camera_id
        ]
        if not applicable_gates:
            return events

        active_ids = {t.track_id for t in tracks}
        for (gid, tid) in list(self._last_crossings.keys()):
            if tid not in active_ids:
                del self._last_crossings[(gid, tid)]

        for track in tracks:
            if len(track.history) < 2:
                continue

            prev_pos = track.history[-2]
            curr_pos = track.history[-1]

            for gate in applicable_gates:
                key = (gate.gate_id, track.track_id)
                last_cross = self._last_crossings.get(key, -1e9)
                if now - last_cross < self.config.event_cooldown_seconds:
                    continue

                if lines_intersect(prev_pos, curr_pos, gate.line_start, gate.line_end):
                    crossing_dir = determine_line_crossing_direction(
                        prev_pos, curr_pos, gate.line_start, gate.line_end
                    )

                    is_inbound = crossing_dir in (LineDirection.ENTRY, LineDirection.LEFT_TO_RIGHT)

                    is_violation = False
                    if gate.allowed_direction in (LineDirection.ENTRY, LineDirection.LEFT_TO_RIGHT) and not is_inbound:
                        is_violation = True
                    elif gate.allowed_direction in (LineDirection.EXIT, LineDirection.RIGHT_TO_LEFT) and is_inbound:
                        is_violation = True

                    # Update counts
                    if is_inbound:
                        self.inbound_counts[gate.gate_id] += 1
                    else:
                        self.outbound_counts[gate.gate_id] += 1

                    if is_violation:
                        self.violations_counts[gate.gate_id] += 1

                    record = PassageRecord(
                        gate_id=gate.gate_id,
                        track_id=track.track_id,
                        direction=crossing_dir,
                        timestamp=now,
                        is_violation=is_violation,
                        identity_id=track.identity_id,
                        plate_number=track.plate_number,
                    )
                    self.passage_history.append(record)
                    self._last_crossings[key] = now

                    # Generate appropriate event
                    event_type = EventType.CHECKPOINT_VIOLATION if is_violation else EventType.LINE_CROSSING

                    event = AnalyticsEvent(
                        event_id=str(uuid.uuid4()),
                        camera_id=camera_id,
                        timestamp=now,
                        event_type=event_type,
                        track_id=track.track_id,
                        identity_id=track.identity_id,
                        confidence=0.95,
                        metadata={
                            "rule": "checkpoint_crossing",
                            "gate_id": gate.gate_id,
                            "gate_name": gate.name,
                            "direction": crossing_dir.value,
                            "is_violation": is_violation,
                            "inbound_total": self.inbound_counts[gate.gate_id],
                            "outbound_total": self.outbound_counts[gate.gate_id],
                            "plate_number": track.plate_number,
                        },
                    )
                    events.append(event)

        return events

    def get_stats(self, gate_id: str) -> Dict[str, int]:
        """Returns passage telemetry for a specific checkpoint gate."""
        return {
            "inbound": self.inbound_counts.get(gate_id, 0),
            "outbound": self.outbound_counts.get(gate_id, 0),
            "violations": self.violations_counts.get(gate_id, 0),
        }
