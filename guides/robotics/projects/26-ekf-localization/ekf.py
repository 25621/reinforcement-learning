"""The EKF and the UKF for a planar robot, written side by side.

The two differ in exactly one idea:

  EKF -- push the MEAN through the nonlinear function, and push the COVARIANCE
         through a linear approximation of it (the Jacobian).  Cheap.  Wrong
         whenever the function bends appreciably across the width of your
         uncertainty.

  UKF -- pick a handful of representative points ("sigma points") spread over
         the current uncertainty, push each one through the REAL nonlinear
         function, and rebuild a mean and covariance from where they land.  No
         Jacobian is ever needed.  Same O(n^3) cost for these tiny states.

Both must wrap angles in three separate places, and forgetting any one of them
produces a filter that works fine for a while and then explodes.  Experiment 3
measures exactly what "explodes" costs.
"""

import numpy as np

from world import (wrap, move, motion_jacobians, motion_noise,
                   range_bearing, measurement_jacobian)


# ---------------------------------------------------------------------- EKF
class EKF:
    def __init__(self, x, P):
        self.x = np.asarray(x, dtype=float).copy()
        self.P = np.asarray(P, dtype=float).copy()

    def predict(self, u, dt, alpha=None):
        G, V = motion_jacobians(self.x, u, dt)
        M = motion_noise(u) if alpha is None else motion_noise(u, alpha)
        self.x = move(self.x, u, dt)
        self.P = G @ self.P @ G.T + V @ M @ V.T
        self.P = 0.5 * (self.P + self.P.T)
        return self.x, self.P

    def update(self, z, lm, R, wrap_bearing=True):
        """Fold in one range/bearing measurement.  Returns (innovation, S)."""
        z_hat = range_bearing(self.x, lm)
        H = measurement_jacobian(self.x, lm)
        y = np.asarray(z, dtype=float) - z_hat
        if wrap_bearing:
            # THE line.  A bearing of +179 deg and one of -179 deg are 2 deg
            # apart, but subtracting them gives 358 deg.  Without this, the
            # filter is handed a huge fake innovation every time the robot's
            # heading passes through pi -- see experiment 3.
            y[1] = wrap(y[1])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        return y, S


# ---------------------------------------------------------------------- UKF
def _nearest_pd(P, floor=1e-12):
    """Clip any negative eigenvalues to zero (plus a floor).

    Needed because the UKF's covariance update, P <- P - K S K', is a
    DIFFERENCE of two matrices, and round-off can leave the result with a tiny
    negative eigenvalue.  The EKF avoids this by using the Joseph form (a sum of
    two positive terms) but there is no Joseph form for the UKF, so the fix has
    to be applied afterwards.  Skipping it is not a theoretical worry: without
    this function, experiment 6 crashes on a Cholesky failure partway through.
    """
    P = 0.5 * (P + P.T)
    w, V = np.linalg.eigh(P)
    if w.min() > floor:
        return P
    return (V * np.maximum(w, floor)) @ V.T


def _sigma_points(x, P, alpha=0.9, beta=2.0, kappa=0.0):
    """Spread 2n+1 points over the current uncertainty.

    alpha controls how far out they sit.  The textbook default alpha = 1e-3 is
    meant for high-dimensional states; with n = 3 it makes the scaling factor
    (n + lambda) about 3e-6, which is numerically miserable.  alpha = 0.9 keeps
    the points at a sensible distance and the weights well conditioned.
    """
    n = len(x)
    lam = alpha ** 2 * (n + kappa) - n
    A = np.linalg.cholesky(_nearest_pd((n + lam) * P))
    pts = np.empty((2 * n + 1, n))
    pts[0] = x
    for i in range(n):
        pts[1 + i] = x + A[:, i]
        pts[1 + n + i] = x - A[:, i]
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = lam / (n + lam) + (1.0 - alpha ** 2 + beta)
    return pts, wm, wc


def _angular_mean(angles, w):
    """Average angles the only way that is correct: on the unit circle.

    Averaging 179 deg and -179 deg arithmetically gives 0 deg -- pointing the
    exact opposite way.  Converting to (cos, sin), averaging, and converting
    back gives 180 deg, which is right.
    """
    return np.arctan2(np.sum(w * np.sin(angles)), np.sum(w * np.cos(angles)))


class UKF:
    def __init__(self, x, P):
        self.x = np.asarray(x, dtype=float).copy()
        self.P = np.asarray(P, dtype=float).copy()

    def predict(self, u, dt, alpha=None):
        pts, wm, wc = _sigma_points(self.x, self.P)
        prop = np.array([move(p, u, dt) for p in pts])
        xm = np.array([wm @ prop[:, 0], wm @ prop[:, 1],
                       _angular_mean(prop[:, 2], wm)])
        d = prop - xm
        d[:, 2] = wrap(d[:, 2])
        P = (d * wc[:, None]).T @ d
        _, V = motion_jacobians(self.x, u, dt)
        M = motion_noise(u) if alpha is None else motion_noise(u, alpha)
        self.x, self.P = xm, _nearest_pd(P + V @ M @ V.T)
        return self.x, self.P

    def update(self, z, lm, R):
        pts, wm, wc = _sigma_points(self.x, self.P)
        zs = np.array([range_bearing(p, lm) for p in pts])
        zm = np.array([wm @ zs[:, 0], _angular_mean(zs[:, 1], wm)])
        dz = zs - zm
        dz[:, 1] = wrap(dz[:, 1])
        dx = pts - self.x
        dx[:, 2] = wrap(dx[:, 2])
        S = (dz * wc[:, None]).T @ dz + R
        C = (dx * wc[:, None]).T @ dz                      # cross-covariance
        K = C @ np.linalg.inv(S)
        y = np.asarray(z, dtype=float) - zm
        y[1] = wrap(y[1])
        self.x = self.x + K @ y
        self.x[2] = wrap(self.x[2])
        self.P = _nearest_pd(self.P - K @ S @ K.T)
        return y, S


# -------------------------------------------------------------- diagnostics
def pose_nees(x_est, x_true, P):
    e = np.asarray(x_est) - np.asarray(x_true)
    e[2] = wrap(e[2])
    return float(e @ np.linalg.solve(P, e))
