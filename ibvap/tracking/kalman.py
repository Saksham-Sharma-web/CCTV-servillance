"""
Kalman Filter for 2D Bounding Box Tracking.
State space: [x_center, y_center, area, aspect_ratio, vx, vy, va]
Pure NumPy implementation for high performance and zero external C++ dependencies.
"""

import numpy as np
from typing import Tuple


def convert_bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Converts [x1, y1, x2, y2] to [x_center, y_center, area, aspect_ratio] column vector.
    """
    x1, y1, x2, y2 = bbox
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))
    x = float(x1) + w / 2.0
    y = float(y1) + h / 2.0
    s = w * h
    r = w / h
    return np.array([[x], [y], [s], [r]], dtype=np.float32)


def convert_x_to_bbox(x: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Converts [x_center, y_center, area, aspect_ratio, ...] to [x1, y1, x2, y2].
    """
    xc = float(x[0, 0])
    yc = float(x[1, 0])
    s = max(1.0, float(x[2, 0]))
    r = max(0.01, float(x[3, 0]))

    w = np.sqrt(s * r)
    h = s / max(1e-5, w)

    x1 = int(round(xc - w / 2.0))
    y1 = int(round(yc - h / 2.0))
    x2 = int(round(xc + w / 2.0))
    y2 = int(round(yc + h / 2.0))
    return (x1, y1, x2, y2)


class KalmanBoxTracker:
    """
    Represents the internal state of individual tracked objects observed over time.
    """
    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int], class_name: str, confidence: float):
        # State vector: [x, y, s, r, vx, vy, vs] (7x1)
        self.dim_x = 7
        self.dim_z = 4

        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        self.x[:4] = convert_bbox_to_z(bbox)

        # State Transition Matrix F (Constant Velocity Model)
        self.F = np.eye(self.dim_x, dtype=np.float32)
        self.F[0, 4] = 1.0  # x = x + vx
        self.F[1, 5] = 1.0  # y = y + vy
        self.F[2, 6] = 1.0  # s = s + vs

        # Measurement Matrix H (we observe [x, y, s, r])
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Covariance Matrix P
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0

        # Process Noise Covariance Q
        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        self.Q[2, 2] *= 0.01

        # Measurement Noise Covariance R
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[2:, 2:] *= 10.0

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.class_name = class_name
        self.confidence = confidence

        self.time_since_update = 0
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

        # Persistent metadata attached to this track
        self.identity_id = None
        self.identity_confidence = None
        self.identity_name = None
        self.last_face_check_frame = 0

        self.plate_number = None
        self.plate_category = None
        self.plate_confidence = None
        self.ocr_confidence = None
        self.plate_bbox = None
        self.last_ocr_check_frame = 0

        self.stationary_since = None
        self.centroid_history = []
        cx = int(round(self.x[0, 0]))
        cy = int(round(self.x[1, 0]))
        self.centroid_history.append((cx, cy))

    def predict(self) -> Tuple[int, int, int, int]:
        """
        Advances the state vector and returns the predicted bounding box.
        """
        if (self.x[2, 0] + self.x[6, 0]) <= 0:
            self.x[6, 0] = 0.0

        # x = F * x
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        pred_box = convert_x_to_bbox(self.x)
        self.history.append(pred_box)
        return pred_box

    def update(self, bbox: Tuple[int, int, int, int], confidence: float):
        """
        Updates the state vector with an observed bounding box.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.confidence = confidence

        z = convert_bbox_to_z(bbox)

        # y = z - H * x
        y = z - np.dot(self.H, self.x)
        # S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        # K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # x = x + K * y
        self.x = self.x + np.dot(K, y)
        # P = (I - K * H) * P
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

        cx = int(round(self.x[0, 0]))
        cy = int(round(self.x[1, 0]))
        self.centroid_history.append((cx, cy))
        if len(self.centroid_history) > 60:
            self.centroid_history.pop(0)

    def get_state(self) -> Tuple[int, int, int, int]:
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.x)

    @property
    def velocity(self) -> Tuple[float, float]:
        """
        Returns (vx, vy) estimated velocity in pixels per frame.
        """
        return (float(self.x[4, 0]), float(self.x[5, 0]))
