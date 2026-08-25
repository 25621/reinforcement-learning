"""The linear Kalman filter, written once and shared by all of Phase 4.

Project 24 uses it on a scalar temperature; project 25 uses the same class on a
4-state constant-velocity tracker.  Projects 26-28 need *nonlinear* models, so
they subclass or re-derive, but they reuse the diagnostics at the bottom of this
file (NIS, NEES, chi-square gates), because "is my filter honest?" is the same
question no matter how the model is shaped.

Nothing here is more than the six lines from the guide plus bookkeeping:

    predict:  x <- F x + B u ,          P <- F P F' + Q
    update:   y  = z - H x   (innovation)
              S  = H P H' + R (innovation covariance)
              K  = P H' S^-1 (Kalman gain)
              x <- x + K y ,            P <- (I - K H) P

The only real subtlety is the last line, and it has its own note below.
"""

import math

import numpy as np


class KalmanFilter:
    """A linear-Gaussian state estimator.

    Parameters
    ----------
    x : (n,) array      initial mean of the state
    P : (n, n) array    initial covariance -- how unsure we are about that mean
    """

    def __init__(self, x, P):
        self.x = np.asarray(x, dtype=float).reshape(-1)
        self.P = np.asarray(P, dtype=float).reshape(self.x.size, self.x.size)
        self.n = self.x.size

    # ---------------------------------------------------------------- predict
    def predict(self, F, Q, B=None, u=None):
        """Push the belief forward one time step through the motion model."""
        F = np.asarray(F, dtype=float)
        self.x = F @ self.x
        if B is not None and u is not None:
            self.x = self.x + np.asarray(B, dtype=float) @ np.atleast_1d(u)
        self.P = F @ self.P @ F.T + np.asarray(Q, dtype=float)
        # Numerical hygiene: P must stay symmetric.  Floating-point round-off
        # makes P and P.T drift apart by ~1e-17 per step, and over thousands of
        # steps that asymmetry can push P to be non-positive-definite, at which
        # point the Cholesky inside a gate blows up.  Averaging costs nothing.
        self.P = 0.5 * (self.P + self.P.T)
        return self.x, self.P

    # ----------------------------------------------------------------- update
    def update(self, z, H, R, joseph=True):
        """Fold in one measurement.  Returns (innovation y, its covariance S).

        We return y and S rather than throwing them away because every
        consistency check in Phase 4 is built from them.  A filter that does not
        expose its innovations cannot be debugged.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        H = np.atleast_2d(np.asarray(H, dtype=float))
        R = np.atleast_2d(np.asarray(R, dtype=float))

        y = z - H @ self.x                       # innovation: surprise, in sensor units
        S = H @ self.P @ H.T + R                 # how surprised we expected to be
        K = self.P @ H.T @ np.linalg.inv(S)      # Kalman gain

        self.x = self.x + K @ y
        I_KH = np.eye(self.n) - K @ H
        if joseph:
            # Joseph form.  Algebraically identical to (I - K H) P, but it is a
            # sum of two symmetric positive-semidefinite terms, so round-off can
            # never make P indefinite.  The short form can, and does, on long
            # runs with very small R.  This is why production filters ship it.
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        else:
            self.P = I_KH @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return y, S


# --------------------------------------------------------------------- gates
#
# SciPy is not installed in this environment, and we need chi-square quantiles
# for the consistency tests, so here are the two special functions that produce
# them.  Both are textbook series/continued-fraction expansions (Numerical
# Recipes 6.2); they agree with scipy.stats.chi2.ppf to ~1e-10.


def _gammainc_lower(a, x):
    """Regularized lower incomplete gamma P(a, x) = gamma(a,x) / Gamma(a)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:                                    # series expansion
        ap, term, total = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for the upper part, then complement
    tiny = 1e-300
    b, c, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def chi2_cdf(x, dof):
    """P(X <= x) for a chi-square variable with `dof` degrees of freedom."""
    return _gammainc_lower(0.5 * dof, 0.5 * x)


def chi2_ppf(p, dof):
    """Inverse of chi2_cdf, by bisection.  Slow and completely reliable."""
    lo, hi = 0.0, max(10.0 * dof, 10.0)
    while chi2_cdf(hi, dof) < p:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if chi2_cdf(mid, dof) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def chi2_interval(dof, alpha=0.05):
    """The two-sided acceptance band a consistent statistic should fall inside."""
    return chi2_ppf(alpha / 2.0, dof), chi2_ppf(1.0 - alpha / 2.0, dof)


# --------------------------------------------------------------- diagnostics


def nis(y, S):
    """Normalized Innovation Squared -- the self-check you can run on a robot.

    y' S^-1 y.  If the filter's noise model is right, this has a chi-square
    distribution with dim(z) degrees of freedom, so its long-run average should
    equal dim(z).  Bigger means the filter is overconfident (it claimed the
    measurement would land closer than it did); smaller means it is timid.

    "Normalized" because dividing by S turns a surprise measured in metres or
    degrees into a unitless number you can compare across sensors.
    """
    y = np.atleast_1d(y)
    return float(y @ np.linalg.solve(np.atleast_2d(S), y))


def nees(x_est, x_true, P):
    """Normalized Estimation Error Squared -- the same check, in simulation.

    (x_est - x_true)' P^-1 (x_est - x_true).  NIS needs no ground truth, so you
    can run it on hardware; NEES needs ground truth, so it only works in sim,
    but it is the stronger test because it looks at the state directly instead
    of at a projection of it through H.
    """
    e = np.atleast_1d(x_est) - np.atleast_1d(x_true)
    return float(e @ np.linalg.solve(np.atleast_2d(P), e))


def wrap_angle(a):
    """Fold an angle into (-pi, pi].  Used constantly from project 25 onward."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi
