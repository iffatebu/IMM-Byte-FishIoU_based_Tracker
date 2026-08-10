# -*- coding: utf-8 -*-
"""
Script: IMMKalmanFilter

imm_kalman_filter.py  —  True 12D CA + 8D CV  IMM Filter
=========================================================
Drop-in replacement for ByteTrack's kalman_filter.py.

MODELS
------
CV (Constant Velocity)     8D  [x,y,a,h, vx,vy,va,vh]
   x_new  = x + vx*dt
   vx_new = vx              (velocity constant)

CA (Constant Acceleration) 12D [x,y,a,h, vx,vy,va,vh, ax,ay,aa,ah]
   x_new  = x + vx*dt + 0.5*ax*dt²   <- ax USED in prediction
   vx_new = vx + ax*dt                <- velocity UPDATED by ax
   ax_new = ax                        (acceleration constant)

DIMENSION MISMATCH SOLUTION
-----------------------------
CV=8D, CA=12D, ByteTrack needs 8D output.
Solution: both models run independently in their own space.
Fusion happens in 4D measurement space (both project to [x,y,a,h]).
Output to ByteTrack is always 8D (CV-based, corrected by fusion).

ONE-LINE SWAP in byte_tracker.py:
  from yolox.tracker.imm_kalman_filter import IMMKalmanFilter as KalmanFilter
"""

import numpy as np
import scipy.linalg
import csv, os, numpy as np
from collections import defaultdict

chi2inv95 = {1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919}
DT = 1 / 5.0


class IMMKalmanFilter:

    _Pi = np.array([[0.75, 0.25],   # Transition matrix
                    [0.40, 0.60]])

    # _Pi = np.array([[0.95, 0.05],   # Transition matrix
    #             [0.40, 0.60]])

    _std_weight_position = 1. / 20  # used both in Q and R
    _std_weight_velocity = 1. / 160 # Process noise variance (Q): originally 1. / 160
    _std_weight_accel    = 0.06 #0.06 # noise for accelration; earlier value 1/160

    def __init__(self):
        dt = DT # Single Frame

        # CV F (8x8): x = x + vx*dt, vx unchanged
        self.F_CV = np.eye(8)
        for i in range(4):
            self.F_CV[i, 4+i] = dt
        self.H_CV = np.eye(4, 8)

        # CA F (12x12): x = x + vx*dt + 0.5*ax*dt^2, vx = vx + ax*dt, ax unchanged
        self.F_CA = np.eye(12)
        for i in range(4):
            self.F_CA[i,   4+i] = dt
            self.F_CA[i,   8+i] = 0.5 * dt * dt
            self.F_CA[4+i, 8+i] = dt
        self.H_CA = np.eye(4, 12)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def initiate(self, measurement):
        h = float(measurement[3])

        # CV 8D
        mean_CV = np.r_[measurement, np.zeros(4)] # 
        std_CV  = [2*self._std_weight_position*h,
                   2*self._std_weight_position*h, 
                   1e-2,
                   2*self._std_weight_position*h,
                   10*self._std_weight_velocity*h,
                   10*self._std_weight_velocity*h, 
                   1e-5,
                   10*self._std_weight_velocity*h]
        cov_CV  = np.diag(np.square(std_CV))

        # CA 12D — same pos/vel init, accel=0 with large uncertainty
        mean_CA = np.r_[measurement, np.zeros(4), np.zeros(4)]
        std_CA  = [2*self._std_weight_position*h,
                   2*self._std_weight_position*h, 
                   1e-2,
                   2*self._std_weight_position*h,
                   10*self._std_weight_velocity*h,
                   10*self._std_weight_velocity*h, 
                   1e-5,
                   10*self._std_weight_velocity*h,
                   10*self._std_weight_accel*h,
                   10*self._std_weight_accel*h, 
                   1e-5,
                   10*self._std_weight_accel*h]
        cov_CA  = np.diag(np.square(std_CA))

        imm = {'mean_CV': mean_CV, 'cov_CV': cov_CV,
               'mean_CA': mean_CA, 'cov_CA': cov_CA,
               'mu': np.array([0.7, 0.3]), 'h_box': h}

        return self._fuse8(imm), imm

    def predict(self, mean, covariance):
        imm   = covariance
        h     = float(imm['h_box'])
        mu    = imm['mu']
        c_bar = self._Pi.T @ mu                         # (2,)

        # mixing weights mu_mix[i,j]
        mu_mix = (self._Pi * mu[:, None]) / c_bar[None, :]

        x_CV = imm['mean_CV']; P_CV = imm['cov_CV']    # 8D
        x_CA = imm['mean_CA']; P_CA = imm['cov_CA']    # 12D

        # Cross-model projections for mixing
        x_CA8 = x_CA[:8];   P_CA8 = P_CA[:8, :8]       # CA->8D
        x_CV12 = np.r_[x_CV, np.zeros(4)]              # CV->12D
        P_CV12 = np.zeros((12,12)); P_CV12[:8,:8] = P_CV
        # P_CV12[8:, 8:] = P_CA[8:, 8:]   # CV is agnostic about accel: inherit CA's accel uncertainty

        # Mixed state for CV (8D)
        xm_CV = mu_mix[0,0]*x_CV + mu_mix[1,0]*x_CA8
        Pm_CV = self._mix_cov([mu_mix[0,0], mu_mix[1,0]],
                               [x_CV, x_CA8], [P_CV, P_CA8], xm_CV, 8)

        # Mixed state for CA (12D)
        xm_CA = mu_mix[0,1]*x_CV12 + mu_mix[1,1]*x_CA
        Pm_CA = self._mix_cov([mu_mix[0,1], mu_mix[1,1]],
                               [x_CV12, x_CA], [P_CV12, P_CA], xm_CA, 12)

        # Kalman predict
        Q_CV = self._Q_CV(h); Q_CA = self._Q_CA(h)

        x_CV_p = self.F_CV @ xm_CV
        P_CV_p = self.F_CV @ Pm_CV @ self.F_CV.T + Q_CV

        # CA predict uses KINEMATIC equations — ax baked into F_CA
        x_CA_p = self.F_CA @ xm_CA
        P_CA_p = self.F_CA @ Pm_CA @ self.F_CA.T + Q_CA

        imm_new = {'mean_CV': x_CV_p, 'cov_CV': P_CV_p,
                   'mean_CA': x_CA_p, 'cov_CA': P_CA_p,
                   'mu': c_bar, 'h_box': h}

        return self._fuse8(imm_new), imm_new

    def update(self, mean, covariance, measurement):
        imm   = covariance
        h     = float(measurement[3])
        z     = np.asarray(measurement, dtype=float)
        c_bar = imm['mu']
        R     = self._R(h)

        #>>>>>>>>>>> add debugging
        # predicted box (xyah) the filter expects:
        zh_CV = self.H_CV @ imm['mean_CV']        # [x, y, a, h] predicted
        # measured box (xyah) the detector gave:
        z     = np.asarray(measurement, float)    # [x, y, a, h] observed

        def _xyah_to_tlbr(b):
            x, y, a, h = b
            w = a * h
            return np.array([x - w/2, y - h/2, x + w/2, y + h/2])

        def _iou(b1, b2):
            t1, t2 = _xyah_to_tlbr(b1), _xyah_to_tlbr(b2)
            xx1, yy1 = max(t1[0], t2[0]), max(t1[1], t2[1])
            xx2, yy2 = min(t1[2], t2[2]), min(t1[3], t2[3])
            iw, ih = max(0.0, xx2 - xx1), max(0.0, yy2 - yy1)
            inter = iw * ih
            a1 = (t1[2]-t1[0]) * (t1[3]-t1[1])
            a2 = (t2[2]-t2[0]) * (t2[3]-t2[1])
            return inter / (a1 + a2 - inter + 1e-7)

        #<<<<< new add
        # # -- temp diagnostic ----------------------
        # print(f"[NOISE] h={h:.1f}")
        # print(f"  R  diag = {np.diag(R).round(2)}")
        # print(f"  Q_CV[0,0]={self._Q_CV(h)[0,0]:.4f}  Q_CV[4,4]={self._Q_CV(h)[4,4]:.4f}")
        # print(f"  Q_CA[0,0]={self._Q_CA(h)[0,0]:.4f}  Q_CA[8,8]={self._Q_CA(h)[8,8]:.4f}")
        # print(f"  nu_CV={np.round(self.H_CV @ imm['mean_CV'] - np.asarray(measurement),2)}")

        # # >>>>>>> new add
        # print(f"  pred_vs_meas IoU = {_iou(zh_CV, z):.3f}  "
        #       f"center_shift = {np.linalg.norm((z[:2]-zh_CV[:2])):.1f}px")
        # -----------------------------------------

        # ── CV update (8D) ──────────────────────────────────────────
        x_CV = imm['mean_CV']; P_CV = imm['cov_CV']
        zh_CV = self.H_CV @ x_CV                        # (4,)
        S_CV  = self.H_CV @ P_CV @ self.H_CV.T + R      # (4,4)
        nu_CV = z - zh_CV                               # (4,) innovation
        K_CV  = self._gain(P_CV, self.H_CV, S_CV)       # (8,4)
        xu_CV = x_CV + K_CV @ nu_CV
        Pu_CV = self._sym(P_CV - K_CV @ S_CV @ K_CV.T)

        # ── CA update (12D) ──────────────────────────────────────────
        x_CA = imm['mean_CA']; P_CA = imm['cov_CA']
        zh_CA = self.H_CA @ x_CA                        # (4,)
        S_CA  = self.H_CA @ P_CA @ self.H_CA.T + R      # (4,4)
        nu_CA = z - zh_CA                               # (4,) innovation
        K_CA  = self._gain(P_CA, self.H_CA, S_CA)       # (12,4)
        xu_CA = x_CA + K_CA @ nu_CA
        Pu_CA = self._sym(P_CA - K_CA @ S_CA @ K_CA.T)

        # ── Likelihood Λ_j = N(z; ẑ_j, S_j) ────────────────────────
        # Both evaluated in 4D measurement space — directly comparable
        L_CV = self._likelihood(nu_CV, S_CV)
        L_CA = self._likelihood(nu_CA, S_CA)

        # ── mu update (Bayes) ────────────────────────────────────────
        # # mu_tilde_j = Λ_j × c̄_j
        # mu_tilde = np.array([L_CV, L_CA]) * c_bar
        # c = mu_tilde.sum()
        # mu_new = mu_tilde / c if c > 1e-300 else c_bar.copy()
        #>>>>>>>>>> Replace with
        log_L = np.array([
        self._likelihood(nu_CV, S_CV),
        self._likelihood(nu_CA, S_CA)])
        log_mu = log_L + np.log(np.clip(c_bar, 1e-300, None))
        log_mu -= log_mu.max()          # shift for numerical stability
        mu_tilde = np.exp(log_mu)
        mu_new = mu_tilde / mu_tilde.sum()


        # ── State fusion ──────────────────────────────────────────────
        # Fuse in 8D: use CA[:8] for the CA contribution
        xu_CA8 = xu_CA[:8]
        x_fused = mu_new[0]*xu_CV + mu_new[1]*xu_CA8

        # Fused covariance (spread-of-means formula)
        P_fused = np.zeros((8,8))
        for j, (xj, Pj) in enumerate([(xu_CV, Pu_CV),
                                        (xu_CA8, Pu_CA[:8,:8])]):
            d = xj - x_fused
            P_fused += mu_new[j] * (Pj + np.outer(d, d))
        P_fused = self._sym(P_fused)

        imm_new = {'mean_CV': xu_CV, 'cov_CV': Pu_CV,
                   'mean_CA': xu_CA, 'cov_CA': Pu_CA,
                   'mu': mu_new, 'h_box': h}

        # return x_fused, imm_new
        # >>>>>>>>>>change to
        return x_fused, imm_new, (nu_CV, nu_CA, L_CV, L_CA)

    def multi_predict(self, multi_mean, multi_covariance):
        ms, cs = [], []
        for m, c in zip(multi_mean, multi_covariance):
            m2, c2 = self.predict(m, c)
            ms.append(m2); cs.append(c2)
        return np.array(ms), cs

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        imm = covariance
        h   = float(imm['h_box'])
        mu  = imm['mu']
        R   = self._R(h)

        zh_CV = self.H_CV @ imm['mean_CV']
        S_CV  = self.H_CV @ imm['cov_CV'] @ self.H_CV.T + R

        zh_CA = self.H_CA @ imm['mean_CA']
        S_CA  = self.H_CA @ imm['cov_CA'] @ self.H_CA.T + R

        zh = mu[0]*zh_CV + mu[1]*zh_CA
        S  = mu[0]*S_CV  + mu[1]*S_CA

        meas = np.asarray(measurements)
        if only_position:
            zh = zh[:2]; S = S[:2,:2]; meas = meas[:,:2]

        chol = np.linalg.cholesky(S)
        nu   = meas - zh
        w    = scipy.linalg.solve_triangular(chol, nu.T, lower=True,
                                              check_finite=False)
        d    = np.sum(w*w, axis=0)
        if metric == 'gaussian':
            d = -2.0 * np.log(np.maximum(d, 1e-300))
        return d

    @staticmethod
    def get_mu(covariance):
        """Returns [mu_CV, mu_CA] — read per track."""
        return covariance['mu']

    # ==================================================================
    # PRIVATE
    # ==================================================================

    def _Q_CV(self, h):
        sp = self._std_weight_position * h
        sv = self._std_weight_velocity * h
        return np.diag(np.square([sp,sp,1e-2,sp, sv,sv,1e-5,sv]))

    def _Q_CA(self, h):
        sp = self._std_weight_position * h
        sv = self._std_weight_velocity * h
        sa = self._std_weight_accel    * h
        return np.diag(np.square([sp,sp,1e-2,sp,
                                   sv,sv,1e-5,sv,
                                   sa,sa,1e-5,sa]))

    def _R(self, h):
        sp = self._std_weight_position * h
        return np.diag(np.square([sp, sp, 0.1, sp]))

    def _gain(self, P, H, S):
        PHt = P @ H.T
        cf, lo = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        return scipy.linalg.cho_solve((cf, lo), PHt.T,
                                       check_finite=False).T

    def _likelihood(self, nu, S):
        k = nu.shape[0]
        try:
            chol    = np.linalg.cholesky(S)
            w       = scipy.linalg.solve_triangular(chol, nu, lower=True,
                                                     check_finite=False)
            maha    = float(w @ w)
            log_det = 2.0 * np.sum(np.log(np.diag(chol)))
        except np.linalg.LinAlgError:
            maha = float(nu @ np.linalg.pinv(S) @ nu)
            sg, log_det = np.linalg.slogdet(S)
            if sg <= 0: return 1e-300
        
        return -0.5*(k*np.log(2*np.pi) + log_det + maha)
        # return float(np.exp(np.clip(log_L, -500, 0)))

    def _mix_cov(self, weights, means, covs, x_mix, ndim):
        P = np.zeros((ndim, ndim))
        for w, x, C in zip(weights, means, covs):
            d = x - x_mix
            P += w * (C + np.outer(d, d))
        return P

    def _fuse8(self, imm):
        mu = imm['mu']
        return mu[0]*imm['mean_CV'] + mu[1]*imm['mean_CA'][:8]

    @staticmethod
    def _sym(P):
        return (P + P.T) * 0.5


class IMMDiagnosticLogger:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir   = out_dir
        self.frame_log = []          # one row per track per frame
        self.track_acc = defaultdict(list)  # mu history per track_id

    def log_update(self, frame_id, track_id, imm_before, imm_after,
                   nu_CV, nu_CA, L_CV, L_CA):
        mu     = imm_after['mu']
        # print(f"[LOG] frame={frame_id} track={track_id} mu=[{mu[0]:.3f},{mu[1]:.3f}]")
        ax, ay = float(imm_after['mean_CA'][8]), float(imm_after['mean_CA'][9])
        P_fused_trace = float(np.trace(
            mu[0]*imm_after['cov_CV'] +
            mu[1]*imm_after['cov_CA'][:8,:8]
        ))

        row = {
            'frame':       frame_id,
            'track_id':    track_id,
            'mu_CV':       round(mu[0], 4),
            'mu_CA':       round(mu[1], 4),
            'dominant':    'CV' if mu[0] > mu[1] else 'CA',
            'nu_CV_norm':  round(float(np.linalg.norm(nu_CV)), 4),
            'nu_CA_norm':  round(float(np.linalg.norm(nu_CA)), 4),
            'L_CV':        float(L_CV),
            'L_CA':        float(L_CA),
            'ax':          round(ax, 5),
            'ay':          round(ay, 5),
            'P_trace':     round(P_fused_trace, 4),
        }
        self.frame_log.append(row)
        self.track_acc[track_id].append(mu[0])

    def summarize_tracks(self):
        rows = []
        for tid, mu_history in self.track_acc.items():
            arr     = np.array(mu_history)
            flips   = int(np.sum(np.abs(np.diff(arr > 0.5))))
            rows.append({
                'track_id':      tid,
                'n_frames':      len(arr),
                'mean_mu_CV':    round(float(arr.mean()), 3),
                'mean_mu_CA':    round(float(1 - arr.mean()), 3),
                'mu_flips':      flips,
                'cv_dominant':   arr.mean() > 0.6,
                'needs_CT':      flips > len(arr) * 0.3,  # >30% frames flip
            })
        return rows

    def save(self, video_name):
        # frame-level CSV
        frame_path = os.path.join(self.out_dir, f'{video_name}_frames.csv')
        if self.frame_log:
            keys = list(self.frame_log[0].keys())
            with open(frame_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(self.frame_log)

        # track-level summary CSV
        track_path = os.path.join(self.out_dir, f'{video_name}_tracks.csv')
        summaries  = self.summarize_tracks()
        if summaries:
            keys = list(summaries[0].keys())
            with open(track_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(summaries)

        print(f"[IMM] saved ? {frame_path}")
        print(f"[IMM] saved ? {track_path}")