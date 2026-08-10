# -*- coding: utf-8 -*-
"""
imm_kalman_filter.py  -  CV + CT  IMM Filter
=============================================
Drop-in replacement for ByteTrack's kalman_filter.py.

MODELS
------
CV (Constant Velocity)   8D  [x,y,a,h, vx,vy,va,vh]
   x_new  = x + vx*dt
   vx_new = vx

CT (Constant Turn)       9D [x,y,a,h, vx,vy,va,vh, omega]
   x_new  = x + (sin(w*dt)/w)*vx - ((1-cos(w*dt))/w)*vy
   y_new  = y + ((1-cos(w*dt))/w)*vx + (sin(w*dt)/w)*vy
   vx_new =  cos(w*dt)*vx - sin(w*dt)*vy
   vy_new =  sin(w*dt)*vx + cos(w*dt)*vy
   w_new  = omega  (turn rate constant)
   a,h,va,vh use simple dt (not coupled to turn)

   Degenerates to CV when omega -> 0  (handled numerically)

DIMENSION MISMATCH SOLUTION
-----------------------------
CV=8D, CT=10D, ByteTrack needs 8D output.
Fusion output is always 8D (pos+vel).
omega is CT-internal only.

ONE-LINE SWAP in byte_tracker.py:
  from yolox.tracker.imm_kalman_filter import IMMKalmanFilter as KalmanFilter
"""

import numpy as np
import scipy.linalg
import csv
import os
from collections import defaultdict

chi2inv95 = {
    1: 3.8415, 2: 5.9915, 3: 7.8147, 4: 9.4877,
    5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507, 9: 16.919
}
DT = 1 #1 / 5.0


class IMMKalmanFilter:

    _Pi = np.array([[0.85, 0.15],   # prob of staying CV | switching to CT
                    [0.15, 0.85]])  # prob of switching to CV | staying CT

    _std_weight_position = 1. / 20
    _std_weight_velocity = 1. / 160
    _std_weight_turn     = 0.1     # turn-rate noise (rad/frame) — NOT h-scaled

    def __init__(self):
        dt = DT

        # CV F (8x8): x += vx*dt, vx unchanged
        self.F_CV = np.eye(8)
        for i in range(4):
            self.F_CV[i, 4+i] = dt

        # CV H (4x8): observe [x,y,a,h]
        self.H_CV = np.eye(4, 8)

        # CT H (4x10): observe [x,y,a,h] — omega not observed
        self.H_CT = np.eye(4, 9)

        # CT F is built dynamically in _build_F_CT(omega)

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    def initiate(self, measurement):
        h = float(measurement[3])

        # -- CV 8D ------------------------------------------------------
        mean_CV = np.r_[measurement, np.zeros(4)]
        std_CV  = [2*self._std_weight_position*h,
                   2*self._std_weight_position*h,
                   1e-2,
                   2*self._std_weight_position*h,
                   10*self._std_weight_velocity*h,
                   10*self._std_weight_velocity*h,
                   1e-5,
                   10*self._std_weight_velocity*h]
        cov_CV  = np.diag(np.square(std_CV))

        # -- CT 10D — same pos/vel init, omega=0 with uncertainty -------
        mean_CT = np.r_[measurement, np.zeros(4), 0.0]   # omega=0, 8x8 matrix
        std_CT  = [2*self._std_weight_position*h,
                   2*self._std_weight_position*h,
                   1e-2,
                   2*self._std_weight_position*h,
                   10*self._std_weight_velocity*h,
                   10*self._std_weight_velocity*h,
                   1e-5,
                   10*self._std_weight_velocity*h,
                   0.5]    # omega initial std — small, fish start straight
        cov_CT  = np.diag(np.square(std_CT))

        imm = {'mean_CV': mean_CV, 'cov_CV': cov_CV,
               'mean_CT': mean_CT, 'cov_CT': cov_CT,
               'mu': np.array([0.5, 0.5]), 'h_box': h}

        return self._fuse8(imm), imm

    def predict(self, mean, covariance):
        imm   = covariance
        h     = float(imm['h_box'])
        mu    = imm['mu']
        c_bar = self._Pi.T @ mu                         # (2,)

        # mixing weights  mu_mix[i,j] = P(model_i -> model_j)
        mu_mix = (self._Pi * mu[:, None]) / c_bar[None, :]

        x_CV = imm['mean_CV']; P_CV = imm['cov_CV']    # 8D
        x_CT = imm['mean_CT']; P_CT = imm['cov_CT']    # 10D

        # -- cross-model projections for mixing -------------------------
        x_CT8  = x_CT[:8];    P_CT8  = P_CT[:8, :8]    # CT -> 8D (drop omega)
        x_CV9 = np.r_[x_CV, np.zeros(1)]              # CV -> 9D (pad omega=0)
        P_CV9 = np.zeros((9, 9))
        P_CV9[:8, :8] = P_CV

        # -- mixed input for CV (8D) ------------------------------------
        xm_CV = mu_mix[0,0]*x_CV + mu_mix[1,0]*x_CT8
        Pm_CV = self._mix_cov([mu_mix[0,0], mu_mix[1,0]],
                               [x_CV, x_CT8], [P_CV, P_CT8], xm_CV, 8)

        # -- mixed input for CT (10D) -----------------------------------
        xm_CT = mu_mix[0,1]*x_CV9 + mu_mix[1,1]*x_CT
        Pm_CT = self._mix_cov([mu_mix[0,1], mu_mix[1,1]],
                               [x_CV9, x_CT], [P_CV9, P_CT], xm_CT, 9)

        # -- KF predict -------------------------------------------------
        Q_CV  = self._Q_CV(h)
        Q_CT  = self._Q_CT(h)
        F_CT  = self._build_F_CT(float(xm_CT[8]))      # built with current omega

        x_CV_p = self.F_CV @ xm_CV
        P_CV_p = self.F_CV @ Pm_CV @ self.F_CV.T + Q_CV

        x_CT_p = F_CT @ xm_CT
        P_CT_p = F_CT @ Pm_CT @ F_CT.T + Q_CT

        imm_new = {'mean_CV': x_CV_p, 'cov_CV': P_CV_p,
                   'mean_CT': x_CT_p, 'cov_CT': P_CT_p,
                   'mu': c_bar, 'h_box': h}

        return self._fuse8(imm_new), imm_new

    def update(self, mean, covariance, measurement):
        imm   = covariance
        h     = float(measurement[3])
        z     = np.asarray(measurement, dtype=float)
        c_bar = imm['mu']
        R     = self._R(h)

        # -- CV update (8D) ---------------------------------------------
        x_CV  = imm['mean_CV']; P_CV = imm['cov_CV']
        zh_CV = self.H_CV @ x_CV                        # (4,)
        S_CV  = self.H_CV @ P_CV @ self.H_CV.T + R      # (4,4)
        nu_CV = z - zh_CV                               # (4,) innovation
        K_CV  = self._gain(P_CV, self.H_CV, S_CV)       # (8,4)
        xu_CV = x_CV + K_CV @ nu_CV
        Pu_CV = self._sym(P_CV - K_CV @ S_CV @ K_CV.T)

        # -- CT update (10D) --------------------------------------------
        x_CT  = imm['mean_CT']; P_CT = imm['cov_CT']
        zh_CT = self.H_CT @ x_CT                        # (4,)
        S_CT  = self.H_CT @ P_CT @ self.H_CT.T + R      # (4,4)
        nu_CT = z - zh_CT                               # (4,) innovation
        K_CT  = self._gain(P_CT, self.H_CT, S_CT)       # (10,4)
        xu_CT = x_CT + K_CT @ nu_CT
        Pu_CT = self._sym(P_CT - K_CT @ S_CT @ K_CT.T)

        # at end of update(), after xu_CT is computed
        # estimate omega from velocity rotation
        vx = float(xu_CT[4])
        vy = float(xu_CT[5])
        speed_sq = vx**2 + vy**2

        if speed_sq > 1.0:   # only estimate if fish is actually moving
            # omega from centripetal acceleration estimate
            # use cross product of position innovation and velocity
            omega_est = (nu_CT[0]*vy - nu_CT[1]*vx) / (speed_sq * DT)
            omega_est = float(np.clip(omega_est, -2.0, 2.0))  # bound to ±2 rad/frame

            # soft inject — blend with current estimate
            alpha = 0.3   # how fast to trust new omega estimate
            xu_CT = xu_CT.copy()
            xu_CT[8] = (1 - alpha)*xu_CT[8] + alpha*omega_est

        # -- mu update — log-space Bayes --------------------------------
        log_L = np.array([
            self._likelihood(nu_CV, S_CV),
            self._likelihood(nu_CT, S_CT)
        ])
        log_mu  = log_L + np.log(np.clip(c_bar, 1e-300, None))
        log_mu -= log_mu.max()                          # numerical stability
        mu_tilde = np.exp(log_mu)
        mu_new   = mu_tilde / mu_tilde.sum()

        # -- state fusion (8D output) -----------------------------------
        xu_CT8   = xu_CT[:8]
        x_fused  = mu_new[0]*xu_CV + mu_new[1]*xu_CT8

        # -- covariance fusion (spread-of-means) ------------------------
        P_fused = np.zeros((8, 8))
        for j, (xj, Pj) in enumerate([(xu_CV,  Pu_CV),
                                       (xu_CT8, Pu_CT[:8, :8])]):
            d = xj - x_fused
            P_fused += mu_new[j] * (Pj + np.outer(d, d))
        P_fused = self._sym(P_fused)

        L_CV = float(np.exp(np.clip(log_L[0], -500, 0)))
        L_CT = float(np.exp(np.clip(log_L[1], -500, 0)))

        print(f"  L_CV={self._likelihood(nu_CV, S_CV):.4f}  "
        f"  L_CT={self._likelihood(nu_CT, S_CT):.4f}  "
        f"  diff={self._likelihood(nu_CV,S_CV)-self._likelihood(nu_CT,S_CT):.6f}")
        print(f"  nu_CV={np.round(nu_CV,3)}  nu_CT={np.round(nu_CT,3)}")
        print(f"  S_CV[0,0]={S_CV[0,0]:.2f}  S_CT[0,0]={S_CT[0,0]:.2f}")

        imm_new = {'mean_CV': xu_CV, 'cov_CV': Pu_CV,
                   'mean_CT': xu_CT, 'cov_CT': Pu_CT,
                   'mu': mu_new, 'h_box': h}

        return x_fused, imm_new, (nu_CV, nu_CT, L_CV, L_CT)

    def multi_predict(self, multi_mean, multi_covariance):
        ms, cs = [], []
        for m, c in zip(multi_mean, multi_covariance):
            m2, c2 = self.predict(m, c)
            ms.append(m2)
            cs.append(c2)
        return np.array(ms), cs

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False, metric='maha'):
        imm = covariance
        h   = float(imm['h_box'])
        mu  = imm['mu']
        R   = self._R(h)

        zh_CV = self.H_CV @ imm['mean_CV']
        S_CV  = self.H_CV @ imm['cov_CV'] @ self.H_CV.T + R

        zh_CT = self.H_CT @ imm['mean_CT']
        S_CT  = self.H_CT @ imm['cov_CT'] @ self.H_CT.T + R

        zh = mu[0]*zh_CV + mu[1]*zh_CT
        S  = mu[0]*S_CV  + mu[1]*S_CT

        meas = np.asarray(measurements)
        if only_position:
            zh = zh[:2]; S = S[:2, :2]; meas = meas[:, :2]

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
        """Returns [mu_CV, mu_CT]: readable per track"""
        return covariance['mu']

    # ==================================================================
    # PRIVATE
    # ==================================================================

    def _build_F_CT(self, omega):
        """
        Build 10x10 CT transition matrix for given turn rate omega (rad/frame).
        Degenerates to CV when |omega| < 1e-5.
        """
        dt = DT
        F  = np.eye(9)

        if abs(omega) < 1e-5:
            # near-zero turn rate — reduce to CV
            for i in range(4):
                F[i, 4+i] = dt
            return F

        sin_odt = np.sin(omega * dt)
        cos_odt = np.cos(omega * dt)

        # x,y coupled through vx,vy via turn
        F[0, 4] =  sin_odt / omega
        F[0, 5] = -(1.0 - cos_odt) / omega
        F[1, 4] =  (1.0 - cos_odt) / omega
        F[1, 5] =  sin_odt / omega

        # a, h use simple dt (not coupled to turn)
        F[2, 6] = dt
        F[3, 7] = dt

        # velocity rotates with turn rate
        F[4, 4] =  cos_odt
        F[4, 5] = -sin_odt
        F[5, 4] =  sin_odt
        F[5, 5] =  cos_odt

        # va, vh unchanged
        F[6, 6] = 1.0
        F[7, 7] = 1.0

        # omega unchanged (index 8)
        F[8, 8] = 1.0

        return F

    def _Q_CV(self, h):
        sp = self._std_weight_position * h
        sv = self._std_weight_velocity * h
        return np.diag(np.square([sp, sp, 1e-2, sp,
                                   sv, sv, 1e-5, sv]))

    def _Q_CT(self, h):
        sp = self._std_weight_position * h
        sv = self._std_weight_velocity * h
        st = self._std_weight_turn          # NOT h-scaled — omega is in radians
        return np.diag(np.square([sp, sp, 1e-2, sp,
                                   sv, sv, 1e-5, sv,
                                   st]))    # 9D

    def _R(self, h):
        sp = self._std_weight_position * h
        return np.diag(np.square([sp, sp, 0.1, sp]))

    def _gain(self, P, H, S):
        PHt = P @ H.T
        cf, lo = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        return scipy.linalg.cho_solve((cf, lo), PHt.T,
                                       check_finite=False).T

    def _likelihood(self, nu, S):
        """Returns log-likelihood (used directly in log-space mu update)."""
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
            if sg <= 0:
                return -1e10
        return -0.5 * (k * np.log(2 * np.pi) + log_det + maha)

    def _mix_cov(self, weights, means, covs, x_mix, ndim):
        P = np.zeros((ndim, ndim))
        for w, x, C in zip(weights, means, covs):
            d = x - x_mix
            P += w * (C + np.outer(d, d))
        return P

    def _fuse8(self, imm):
        mu = imm['mu']
        return mu[0]*imm['mean_CV'] + mu[1]*imm['mean_CT'][:8]

    @staticmethod
    def _sym(P):
        return (P + P.T) * 0.5


# ==================================================================
# DIAGNOSTIC LOGGER
# ==================================================================

class IMMDiagnosticLogger:
    def __init__(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir   = out_dir
        self.frame_log = []
        self.track_acc = defaultdict(list)

    def log_update(self, frame_id, track_id, imm_before, imm_after,
                   nu_CV, nu_CT, L_CV, L_CT):
        mu    = imm_after['mu']
        omega = float(imm_after['mean_CT'][8])
        print(f"[LOG] frame={frame_id} track={track_id} "
              f"mu=[{mu[0]:.3f},{mu[1]:.3f}] "
              f"dominant={'CV' if mu[0]>mu[1] else 'CT'} "
              f"omega={omega:.4f}")

        P_fused_trace = float(np.trace(
            mu[0]*imm_after['cov_CV'] +
            mu[1]*imm_after['cov_CT'][:8, :8]
        ))

        row = {
            'frame':        frame_id,
            'track_id':     track_id,
            'mu_CV':        round(mu[0], 4),
            'mu_CT':        round(mu[1], 4),
            'dominant':     'CV' if mu[0] > mu[1] else 'CT',
            'nu_CV_norm':   round(float(np.linalg.norm(nu_CV)), 4),
            'nu_CT_norm':   round(float(np.linalg.norm(nu_CT)), 4),
            'L_CV':         float(L_CV),
            'L_CT':         float(L_CT),
            'omega':        round(omega, 5),
            'P_trace':      round(P_fused_trace, 4),
        }
        self.frame_log.append(row)
        self.track_acc[track_id].append(mu[0])

    def summarize_tracks(self):
        rows = []
        for tid, mu_history in self.track_acc.items():
            arr   = np.array(mu_history)
            flips = int(np.sum(np.abs(np.diff(arr > 0.5))))
            rows.append({
                'track_id':    tid,
                'n_frames':    len(arr),
                'mean_mu_CV':  round(float(arr.mean()), 3),
                'mean_mu_CT':  round(float(1 - arr.mean()), 3),
                'mu_flips':    flips,
                'cv_dominant': arr.mean() > 0.6,
                'ct_dominant': arr.mean() < 0.4,
                'mixed':       0.4 <= arr.mean() <= 0.6,
            })
        return rows

    def save(self, video_name):
        frame_path = os.path.join(self.out_dir, f'{video_name}_frames.csv')
        if self.frame_log:
            keys = list(self.frame_log[0].keys())
            with open(frame_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(self.frame_log)

        track_path = os.path.join(self.out_dir, f'{video_name}_tracks.csv')
        summaries  = self.summarize_tracks()
        if summaries:
            keys = list(summaries[0].keys())
            with open(track_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(summaries)

        print(f"[IMM] saved -> {frame_path}")
        print(f"[IMM] saved -> {track_path}")