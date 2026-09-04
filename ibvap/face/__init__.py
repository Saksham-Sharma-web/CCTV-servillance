from .detector import OpenCVFaceDetector
from .matcher_adapter import IdentityVerifierAdapter, AuthorizedPerson

__all__ = [
    "OpenCVFaceDetector",
    "IdentityVerifierAdapter",
    "AuthorizedPerson",
]
