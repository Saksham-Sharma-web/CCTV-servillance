"""
Face Detection Subsystem.
Modular wrapper around YuNet ONNX, Haar cascades, facial landmarks, and quality verification.
"""

from ibvap.face.detector import OpenCVFaceDetector, FaceDetection

# Provide FaceDetector as an alias for OpenCVFaceDetector
FaceDetector = OpenCVFaceDetector

__all__ = ["FaceDetector", "OpenCVFaceDetector", "FaceDetection"]
