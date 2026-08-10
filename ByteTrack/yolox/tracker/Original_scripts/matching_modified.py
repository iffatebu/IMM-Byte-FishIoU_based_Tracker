# -*- coding: utf-8 -*-
"""
Script: matching.py 
- Added 2 new function named diou and diou_distance for DIoU 
"""


import cv2
import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
# from yolox.tracker import kalman_filter
from yolox.tracker.imm_kalman_filter_CA import IMMKalmanFilter as KalmanFilter, IMMDiagnosticLogger
import time


# >>>> new add function
def _center_tlbr(tlbr):
    return np.array([(tlbr[0] + tlbr[2]) / 2.0,
                     (tlbr[1] + tlbr[3]) / 2.0], dtype=np.float64)

# >>>> new add function
def direction_cost(tracks, detections, lambda_=0.2):
    """OC-SORT OCM as an additive soft penalty (observation-centric).
    0 when a detection lies along the track's heading, up to lambda_ when it
    lies against it. Add this to an IoU cost matrix. Uses only observations,
    never the Kalman prediction."""
    N, M = len(tracks), len(detections)
    cost = np.zeros((N, M), dtype=np.float64)
    if N == 0 or M == 0:
        return cost

    vel = np.array([t.velocity for t in tracks], dtype=np.float64)        # (N,2)
    has_dir = np.linalg.norm(vel, axis=1) > 1e-6                          # (N,)

    # reference = the track's oldest observation in the window (~DELTA_T ago)
    ref = np.array([t.obs_history[0] if len(t.obs_history) else
                    _center_tlbr(t.tlbr) for t in tracks], dtype=np.float64)  # (N,2)
    det_c = np.array([_center_tlbr(d.tlbr) for d in detections],
                     dtype=np.float64)                                    # (M,2)

    diff = det_c[None, :, :] - ref[:, None, :]                           # (N,M,2)
    norm = np.linalg.norm(diff, axis=2, keepdims=True) + 1e-6
    unit = diff / norm
    cos = np.clip((unit * vel[:, None, :]).sum(axis=2), -1.0, 1.0)        # (N,M)
    dcost = (1.0 - cos) / 2.0          # 0 aligned, 0.5 perpendicular, 1 opposite
    dcost[~has_dir, :] = 0.0           # new/short/stationary tracks: no penalty
    return lambda_ * dcost



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

# Newly added:
def diou(atlbrs, btlbrs):
    """Pairwise Distance-IoU between two sets of boxes in tlbr (x1, y1, x2, y2).
    Returns an (N, M) matrix. DIoU = IoU - center_dist^2 / enclosing_diag^2,
    so values lie in [-1, 1]."""
    A = np.ascontiguousarray(atlbrs, dtype=np.float64)
    B = np.ascontiguousarray(btlbrs, dtype=np.float64)
    out = np.zeros((len(A), len(B)), dtype=np.float64)
    if out.size == 0:
        return out

    area_a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1])
    area_b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])

    # intersection
    lt = np.maximum(A[:, None, :2], B[None, :, :2])
    rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter
    iou = inter / np.clip(union, 1e-7, None)

    # squared distance between box centers
    ca = np.stack([(A[:, 0] + A[:, 2]) / 2, (A[:, 1] + A[:, 3]) / 2], axis=1)
    cb = np.stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2], axis=1)
    center_d2 = ((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1)

    # squared diagonal of the smallest enclosing box
    enc_lt = np.minimum(A[:, None, :2], B[None, :, :2])
    enc_rb = np.maximum(A[:, None, 2:], B[None, :, 2:])
    enc_wh = np.clip(enc_rb - enc_lt, 0.0, None)
    enc_d2 = enc_wh[..., 0] ** 2 + enc_wh[..., 1] ** 2

    return iou - center_d2 / np.clip(enc_d2, 1e-7, None)


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


def iou_distance(atracks, btracks):
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
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

# Newly added:
def diou_distance(atracks, btracks):
    """DIoU-based cost matrix, drop-in replacement for iou_distance."""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) \
            or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _diou = diou(atlbrs, btlbrs)
    cost_matrix = 1 - _diou
    return cost_matrix


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