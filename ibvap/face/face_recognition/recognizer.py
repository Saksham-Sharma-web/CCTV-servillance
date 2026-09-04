"""
Face Recognition Subsystem.
Extracts 512D deep facial embeddings, calculates cosine similarity, and verifies identities against registered references.
"""

from ibvap.face.matcher_adapter import (
    IdentityVerifierAdapter,
    AuthorizedPerson,
    PersonReference,
    align_face_160,
    calibrate_threshold,
    CANONICAL_FACE_TEMPLATE_160,
)

# Provide FaceRecognizer as an alias for IdentityVerifierAdapter
FaceRecognizer = IdentityVerifierAdapter

__all__ = [
    "FaceRecognizer",
    "IdentityVerifierAdapter",
    "AuthorizedPerson",
    "PersonReference",
    "align_face_160",
    "calibrate_threshold",
    "CANONICAL_FACE_TEMPLATE_160",
]
