"""
Verification suite for IBVAP 9-Subsystem Architecture.
Ensures every subsystem package, module, and export is correctly wired and importable.
"""

import pytest


def test_import_all_9_subsystems():
    import ibvap
    from ibvap import (
        ingestion,
        detection,
        tracking,
        face,
        vehicle,
        behavior,
        appearance,
        events,
        output,
    )

    assert ingestion is not None
    assert detection is not None
    assert tracking is not None
    assert face is not None
    assert vehicle is not None
    assert behavior is not None
    assert appearance is not None
    assert events is not None
    assert output is not None


def test_ingestion_subsystem():
    from ibvap.ingestion import (
        FrameRingBuffer,
        IngestionFrame,
        FrameSampler,
        ONVIFDiscovery,
        RTSPStreamClient,
    )
    buffer = FrameRingBuffer(capacity=5)
    assert buffer.capacity == 5
    assert buffer.size == 0


def test_detection_subsystem():
    from ibvap.detection import (
        BaseObjectDetector,
        YOLOv8Detector,
        PersonDetector,
        VehicleDetector,
        GeneralObjectDetector,
    )
    from ibvap.detection.person import PersonDetector as PD
    from ibvap.detection.vehicle import VehicleDetector as VD
    from ibvap.detection.object import GeneralObjectDetector as OD

    assert PersonDetector is PD
    assert VehicleDetector is VD
    assert GeneralObjectDetector is OD


def test_tracking_subsystem():
    from ibvap.tracking import (
        PersistentTracker,
        PersonTracker,
        VehicleTracker,
        CrossCameraReID,
    )
    from ibvap.tracking.person_tracker import PersonTracker as PT
    from ibvap.tracking.vehicle_tracker import VehicleTracker as VT
    from ibvap.tracking.reidentification import CrossCameraReID as CR

    assert PersonTracker is PT
    assert VehicleTracker is VT
    assert CrossCameraReID is CR


def test_face_subsystem():
    from ibvap.face import (
        OpenCVFaceDetector,
        FaceDetector,
        IdentityVerifierAdapter,
        FaceRecognizer,
    )
    from ibvap.face.face_detection import FaceDetector as FD
    from ibvap.face.face_recognition import FaceRecognizer as FR

    assert FaceDetector is FD
    assert FaceRecognizer is FR


def test_vehicle_subsystem():
    from ibvap.vehicle import (
        PlateDetector,
        PlateOCREngine,
        VehicleTrackBuffer,
        BestObservationSelector,
        PlateConsensusEngine,
    )
    from ibvap.vehicle.plate_detection import PlateDetector as PD
    from ibvap.vehicle.ocr import PlateOCREngine as PO

    assert PlateDetector is PD
    assert PlateOCREngine is PO


def test_behavior_subsystem():
    from ibvap.behavior import (
        IntrusionDetector,
        LoiteringDetector,
        RouteDeviationDetector,
        CheckpointMonitor,
        CrowdDetector,
    )
    from ibvap.behavior.route_deviation import RouteDeviationDetector as RD
    from ibvap.behavior.checkpoint import CheckpointMonitor as CM
    from ibvap.behavior.intrusion import IntrusionDetector as ID
    from ibvap.behavior.loitering import LoiteringDetector as LD
    from ibvap.behavior.crowd import CrowdDetector as CD

    assert RouteDeviationDetector is RD
    assert CheckpointMonitor is CM
    assert IntrusionDetector is ID
    assert LoiteringDetector is LD
    assert CrowdDetector is CD


def test_appearance_subsystem():
    from ibvap.appearance import (
        MaskedPersonDetector,
        MaskDetectionResult,
        BodyAppearanceExtractor,
    )
    from ibvap.appearance.masked_person import MaskedPersonDetector as MP

    assert MaskedPersonDetector is MP


def test_events_subsystem():
    from ibvap.events import (
        EventEngine,
        AlertManager,
        AlertSeverity,
        EvidenceManager,
    )
    from ibvap.events.event_engine import EventEngine as EE
    from ibvap.events.alert_manager import AlertManager as AM
    from ibvap.events.evidence import EvidenceManager as EM

    assert EventEngine is EE
    assert AlertManager is AM
    assert EvidenceManager is EM


def test_output_subsystem():
    from ibvap.output import (
        SurveillanceDashboard,
        DebugRenderer,
        IBVAPApiRouter,
    )
    from ibvap.output.dashboard import SurveillanceDashboard as SD
    from ibvap.output.api import IBVAPApiRouter as AR

    assert SurveillanceDashboard is SD
    assert IBVAPApiRouter is AR
