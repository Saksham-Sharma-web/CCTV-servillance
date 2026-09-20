"""
Unit tests for EvidenceManager and AlertManager.
"""

import os
import json
import pytest
import numpy as np
import uuid

from ibvap.core.types import AnalyticsEvent, EventType
from ibvap.events.alert_manager import AlertManager, AlertSeverity
from ibvap.events.evidence import EvidenceManager, EvidencePackage


def test_alert_manager_classification_and_dispatch():
    manager = AlertManager()
    dispatched = []

    manager.register_handler(AlertSeverity.CRITICAL, lambda a: dispatched.append(a))

    # 1. Critical event (Intrusion)
    ev_critical = AnalyticsEvent(
        event_id=str(uuid.uuid4()),
        camera_id="cam1",
        timestamp=100.0,
        event_type=EventType.FENCE_INTRUSION,
        track_id=1,
        identity_id=None,
        confidence=0.92,
        metadata={},
    )
    alert = manager.dispatch(ev_critical)
    assert alert.severity == AlertSeverity.CRITICAL
    assert len(dispatched) == 1
    assert alert.acknowledged is False

    # Acknowledge
    assert manager.acknowledge(alert.alert_id) is True
    assert len(manager.get_unacknowledged()) == 0


def test_evidence_manager_packaging(tmp_path):
    manager = EvidenceManager(evidence_dir=str(tmp_path))

    frame = np.ones((200, 200, 3), dtype=np.uint8) * 128
    bbox = (50, 50, 150, 150)

    event = AnalyticsEvent(
        event_id="test-evidence-1234",
        camera_id="cam-front",
        timestamp=1234567.0,
        event_type=EventType.FENCE_INTRUSION,
        track_id=42,
        identity_id="EMP-001",
        confidence=0.95,
        metadata={"zone": "Restricted-A"},
    )

    pkg = manager.package_evidence(event, frame, target_bbox=bbox)
    assert pkg is not None
    assert os.path.exists(pkg.full_frame_path)
    assert os.path.exists(pkg.crop_path)
    assert os.path.exists(pkg.metadata_path)
    assert len(pkg.sha256_hash) == 64  # Valid SHA-256 hex string

    # Verify JSON metadata
    with open(pkg.metadata_path, "r") as f:
        data = json.load(f)
    assert data["event_id"] == "test-evidence-1234"
    assert data["sha256_hash"] == pkg.sha256_hash
    assert data["track_id"] == 42
