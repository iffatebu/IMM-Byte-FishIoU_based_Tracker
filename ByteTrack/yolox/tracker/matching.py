# -*- coding: utf-8 -*-
"""
This script is replacing IoU with FishIoU [Spcifically customized FishIoU, removed cIoU (hurting since vertical bars)]
"""

import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
#from yolox.tracker import kalman_filter
from yolox.tracker.imm_kalman_filter_CA import IMMKalmanFilter as KalmanFilter, IMMDiagnosticLogger
import time

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious

def fish_iou_single(b1, b2):
    """
    FishIoU between two boxes
    b1, b2: [x1, y1, x2, y2] format

    Returns FishIoU score (higher = better match)
    """
    # Standard IoU
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    area2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union_area = area1 + area2 - inter_area

    iou = inter_area / union_area \
          if union_area > 0 else 0.0

    # Center distance (normalized)
    cx1 = (b1[0] + b1[2]) / 2
    cy1 = (b1[1] + b1[3]) / 2
    cx2 = (b2[0] + b2[2]) / 2
    cy2 = (b2[1] + b2[3]) / 2

    # Enclosing box diagonal
    enclose_x1 = min(b1[0], b2[0])
    enclose_y1 = min(b1[1], b2[1])
    enclose_x2 = max(b1[2], b2[2])
    enclose_y2 = max(b1[3], b2[3])
    d_diag_sq = ((enclose_x2 - enclose_x1)**2 +
                 (enclose_y2 - enclose_y1)**2)

    dc = ((cx1-cx2)**2 + (cy1-cy2)**2) / \
         (d_diag_sq + 1e-7)

    # Scale factor (small fish penalty)
    min_area = min(area1, area2)
    s = 1 - np.exp(-min_area / 1000.0)

    # Central region IoU (head emphasis)
    # Asymmetric: more front (head) less back (tail)
    alpha, beta, gamma = 0.15, 0.30, 0.25

    w1, h1 = b1[2]-b1[0], b1[3]-b1[1]
    w2, h2 = b2[2]-b2[0], b2[3]-b2[1]

    c1 = [b1[0] + alpha*w1,
          b1[1] + beta*h1,
          b1[2] - gamma*w1,
          b1[3] - beta*h1]

    c2 = [b2[0] + alpha*w2,
          b2[1] + beta*h2,
          b2[2] - gamma*w2,
          b2[3] - beta*h2]

    # IoU of central regions
    ci_x1 = max(c1[0], c2[0])
    ci_y1 = max(c1[1], c2[1])
    ci_x2 = min(c1[2], c2[2])
    ci_y2 = min(c1[3], c2[3])

    ci_w = max(0, ci_x2 - ci_x1)
    ci_h = max(0, ci_y2 - ci_y1)
    ci_area = ci_w * ci_h

    ca1 = max(0, (c1[2]-c1[0])) * \
              max(0, (c1[3]-c1[1]))
    ca2 = max(0, (c2[2]-c2[0])) * \
              max(0, (c2[3]-c2[1]))
    cu  = ca1 + ca2 - ci_area

    ciou = ci_area / cu if cu > 0 else 0.0

    # Aspect ratio consistency
    r1 = w1 / (h1 + 1e-7)
    r2 = w2 / (h2 + 1e-7)
    ar = min(r1, r2) / (max(r1, r2) + 1e-7)

    # Area consistency 
    aa = min(area1, area2) / \
         (max(area1, area2) + 1e-7)

    # Final FishIoU 
    # fish_iou = (1 * iou   +
    #             0.5 * ciou  +
    #             0.1 * ar    +
    #             0.1 * aa    -
    #             0.3 * s * dc)
    
    fish_iou = (1 * iou   +
                0.2 * ar    +
                0.2 * aa    -
                0.6 * s * dc)
    fish_iou = np.clip(fish_iou, 0.0, 1.0)
    return float(fish_iou)


def fish_iou_batch(atlbrs, btlbrs):
    """
    Compute FishIoU for all pairs
    atlbrs: list of [x1,y1,x2,y2] (tracks)
    btlbrs: list of [x1,y1,x2,y2] (detections)
    Returns: (N, M) matrix of FishIoU scores
    """
    n = len(atlbrs)
    m = len(btlbrs)

    if n == 0 or m == 0:
        return np.zeros((n, m))

    result = np.zeros((n, m))
    for i, b1 in enumerate(atlbrs):
        for j, b2 in enumerate(btlbrs):
            result[i, j] = fish_iou_single(b1, b2)

    return result


# REPLACE THIS EXISTING FUNCTION

def iou_distance(atracks, btracks):
    """
    MODIFIED: uses FishIoU instead of standard IoU
    Drop in replacement:  same inputs/outputs
    """
    if (len(atracks) > 0 and
            isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and
            isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        # These are STrack objects
        # tlbr = [x1, y1, x2, y2] format
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    # ORIGINAL LINE (commented out):
    # ious = ious(atlbrs, btlbrs)

    # REPLACEMENT:
    #print(f"tracks: {atlbrs}, detections: {btlbrs}")
    ious = fish_iou_batch(atlbrs, btlbrs)

    # Convert similarity ? distance
    # (Hungarian minimizes cost)
    cost_matrix = 1 - ious

    return cost_matrix

# def iou_distance(atracks, btracks):
#     """
#     Compute cost based on IoU
#     :type atracks: list[STrack]
#     :type btracks: list[STrack]

#     :rtype cost_matrix np.ndarray
#     """

#     if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
#         atlbrs = atracks
#         btlbrs = btracks
#     else:
#         atlbrs = [track.tlbr for track in atracks]
#         btlbrs = [track.tlbr for track in btracks]
#     _ious = ious(atlbrs, btlbrs)
#     cost_matrix = 1 - _ious

#     return cost_matrix

def v_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost