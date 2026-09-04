"""
Face Recognition Subsystem.
"""

from .recognizer import (
    FaceRecognizer,
    IdentityVerifierAdapter,
    AuthorizedPerson,
    PersonReference,
    align_face_160,
    calibrate_threshold,
    CANONICAL_FACE_TEMPLATE_160,
)

__all__ = [
    "FaceRecognizer",
    "IdentityVerifierAdapter",
    "AuthorizedPerson",
    "PersonReference",
    "align_face_160",
    "calibrate_threshold",
    "CANONICAL_FACE_TEMPLATE_160",
]
