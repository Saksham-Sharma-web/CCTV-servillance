"""
IoU and Hungarian Bipartite Matching for Object Tracking.
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes1: List[Tuple[int, int, int, int]], bboxes2: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Computes Intersection over Union (IoU) matrix between two sets of bounding boxes.
    bboxes: (x1, y1, x2, y2)
    Returns:
        iou_matrix of shape (len(bboxes1), len(bboxes2))
    """
    if not bboxes1 or not bboxes2:
        return np.empty((len(bboxes1), len(bboxes2)), dtype=np.float32)

    b1 = np.array(bboxes1, dtype=np.float32)
    b2 = np.array(bboxes2, dtype=np.float32)

    # b1: (N, 4), b2: (M, 4)
    x11, y11, x12, y12 = b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3]
    x21, y21, x22, y22 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    # Intersection coordinates
    inter_x1 = np.maximum(x11[:, None], x21[None, :])
    inter_y1 = np.maximum(y11[:, None], y21[None, :])
    inter_x2 = np.minimum(x12[:, None], x22[None, :])
    inter_y2 = np.minimum(y12[:, None], y22[None, :])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    # Areas
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)
    union = area1[:, None] + area2[None, :] - intersection

    iou = np.where(union > 0, intersection / union, 0.0)
    return iou.astype(np.float32)


def associate_detections_to_trackers(
    detections: List[Tuple[int, int, int, int]],
    trackers: List[Tuple[int, int, int, int]],
    iou_threshold: float = 0.3
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns detections to tracked objects using the Hungarian algorithm.

    Returns:
        matches: ndarray of shape (K, 2) where col 0 is detection_idx and col 1 is tracker_idx
        unmatched_detections: 1D ndarray of detection indices with no match
        unmatched_trackers: 1D ndarray of tracker indices with no match
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers))

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)
            matched_indices = np.stack((row_ind, col_ind), axis=1)
    else:
        matched_indices = np.empty(shape=(0, 2), dtype=int)

    unmatched_detections = []
    for d in range(len(detections)):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(len(trackers)):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matches with low IoU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)
