"""Piecewise-polynomial trajectories that minimise a chosen derivative.

The problem: given m waypoints and a duration for each leg, find one smooth
curve through all of them.  "Smooth" is made precise by picking a derivative
and asking for the curve whose total squared value of that derivative is
smallest:

    minimise   sum_i  integral_0^{T_i}  ( d^r p_i / dt^r )^2  dt
    subject to passing through the waypoints, and joining smoothly

r = 4 is **minimum snap**.  "Snap" is the fourth derivative of position -- the
sequence is position, velocity, acceleration, jerk, snap, and yes, the joke
names for the fifth and sixth are crackle and pop.  Snap is the right choice
for a quadrotor for a concrete reason, not an aesthetic one: the flatness
algebra in `quad.py` shows the required *torque* is a function of snap, and
torque is what the four motors are differentially limited on.  Minimising
snap therefore minimises exactly the demand that saturates the motors.

Everything below is one equality-constrained quadratic program per axis, and
the three axes are completely independent -- which is itself a consequence of
flatness.
"""

import numpy as np


def _deriv_coefs(deg, r):
    """Row of d^r/dt^r applied to [1, t, t^2, ...] -- the k! / (k-r)! factors."""
    out = np.zeros(deg + 1)
    for k in range(r, deg + 1):
        f = 1.0
        for j in range(r):
            f *= (k - j)
        out[k] = f
    return out


def _basis(deg, T, r):
    """Row vector b with b @ c = the r-th derivative of the polynomial at T."""
    d = _deriv_coefs(deg, r)
    row = np.zeros(deg + 1)
    for k in range(r, deg + 1):
        row[k] = d[k] * (T ** (k - r))
    return row


def _Q(deg, T, r):
    """Cost matrix: integral of (d^r p/dt^r)^2 over [0, T], as c' Q c."""
    Q = np.zeros((deg + 1, deg + 1))
    d = _deriv_coefs(deg, r)
    for i in range(r, deg + 1):
        for j in range(r, deg + 1):
            e = i + j - 2 * r + 1
            Q[i, j] = d[i] * d[j] * (T ** e) / e
    return Q


def poly_traj(way, times, deg=7, r=4, cont=4, bc=3):
    """Solve the QP for one axis.  `way` is a 1-D array of waypoint values.

    `cont` is how many derivative orders are forced to match at each internal
    join, and it is the parameter that decides whether there is an
    optimisation at all.  Force enough of them and the system becomes square:
    the constraints alone pin every coefficient, the cost never gets a vote,
    and "minimum snap" and "minimum jerk" produce the *same* curve.  Leaving
    a few orders free is what gives the objective something to choose.

    `bc` is how many derivatives are pinned to zero at the very start and end
    (start from rest, arrive at rest).
    """
    m = len(way) - 1                      # number of segments
    n = deg + 1
    Qb = np.zeros((m * n, m * n))
    for i in range(m):
        Qb[i * n:(i + 1) * n, i * n:(i + 1) * n] = _Q(deg, times[i], r)

    A, b = [], []

    def row():
        return np.zeros(m * n)

    for i in range(m):                    # both ends of every segment
        z = row(); z[i * n:(i + 1) * n] = _basis(deg, 0.0, 0)
        A.append(z); b.append(way[i])
        z = row(); z[i * n:(i + 1) * n] = _basis(deg, times[i], 0)
        A.append(z); b.append(way[i + 1])
    for i in range(m - 1):                # continuity at internal joins
        for d in range(1, cont + 1):
            z = row()
            z[i * n:(i + 1) * n] = _basis(deg, times[i], d)
            z[(i + 1) * n:(i + 2) * n] = -_basis(deg, 0.0, d)
            A.append(z); b.append(0.0)
    for d in range(1, bc + 1):            # start and finish at rest
        z = row(); z[0:n] = _basis(deg, 0.0, d)
        A.append(z); b.append(0.0)
        z = row(); z[(m - 1) * n:m * n] = _basis(deg, times[-1], d)
        A.append(z); b.append(0.0)

    A = np.asarray(A)
    b = np.asarray(b)
    # KKT system of  min c'Qc  s.t.  Ac = b.  The Lagrange multipliers come
    # out alongside the coefficients and are simply discarded.
    K = np.block([[2 * Qb, A.T], [A, np.zeros((len(A), len(A)))]])
    rhs = np.concatenate([np.zeros(m * n), b])
    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    return sol[:m * n].reshape(m, n)


class Trajectory:
    """A 3-D piecewise polynomial with derivatives up to snap."""

    def __init__(self, waypoints, times, deg=7, r=4, cont=4, bc=3):
        self.way = np.asarray(waypoints, float)
        self.times = np.asarray(times, float)
        self.t_end = float(np.sum(self.times))
        self.edges = np.concatenate([[0.0], np.cumsum(self.times)])
        self.deg = deg
        self.C = [poly_traj(self.way[:, k], self.times, deg, r, cont, bc)
                  for k in range(3)]

    def _seg(self, t):
        t = min(max(t, 0.0), self.t_end - 1e-9)
        i = int(np.searchsorted(self.edges, t, side="right") - 1)
        i = min(max(i, 0), len(self.times) - 1)
        return i, t - self.edges[i]

    def at(self, t, d=0):
        i, tt = self._seg(t)
        row = _basis(self.deg, tt, d)
        return np.array([row @ self.C[k][i] for k in range(3)])

    def ref(self, t, yaw=0.0):
        return dict(pos=self.at(t, 0), vel=self.at(t, 1), acc=self.at(t, 2),
                    jerk=self.at(t, 3), snap=self.at(t, 4), yaw=yaw)

    def sample(self, ts, dmax=4):
        """All derivatives 0..dmax at all times, vectorised.

        Evaluating one time at a time is fine for a plot and far too slow for
        a simulation loop: a 12 s flight at 2 ms is 6000 steps, each rebuilding
        fifteen little basis rows in Python.  Doing it once as arrays turns
        three seconds into a few milliseconds.
        """
        ts = np.asarray(ts, float)
        idx = np.clip(np.searchsorted(self.edges, ts, side="right") - 1,
                      0, len(self.times) - 1)
        tt = ts - self.edges[idx]
        pw = tt[:, None] ** np.arange(self.deg + 1)[None, :]
        out = np.empty((dmax + 1, len(ts), 3))
        for d in range(dmax + 1):
            fac = _deriv_coefs(self.deg, d)
            shift = np.zeros((len(ts), self.deg + 1))
            if d <= self.deg:
                shift[:, d:] = pw[:, :self.deg + 1 - d]
            basis = shift * fac[None, :]
            for k in range(3):
                out[d, :, k] = np.einsum("ij,ij->i", basis, self.C[k][idx])
        return out

    def peak(self, d, n=2000):
        ts = np.linspace(0, self.t_end, n)
        return float(np.max([np.linalg.norm(self.at(t, d)) for t in ts]))

    def cost(self, d=4, n=2000):
        ts = np.linspace(0, self.t_end, n)
        vals = np.array([np.sum(self.at(t, d) ** 2) for t in ts])
        return float(np.trapezoid(vals, ts))


# ------------------------------------------------------------------ timing
def allocate(way, total=None, mode="length", speed=2.0, alpha=0.5):
    """Split the total flight time between the legs.

    Three rules worth comparing:
      "uniform"  -- every leg gets the same time.  A 5 m leg and a 0.5 m leg
                    both get 2 s, so the long one demands 10x the speed.
      "length"   -- time proportional to leg length.  Constant average speed.
      "sqrt"     -- time proportional to length^alpha with alpha < 1.  Short
                    legs get relatively MORE time than their length suggests,
                    which is what you want because a short leg between two
                    long ones is usually a sharp corner, and corners cost
                    acceleration, not distance.
    """
    d = np.linalg.norm(np.diff(way, axis=0), axis=1)
    if mode == "uniform":
        wgt = np.ones_like(d)
    elif mode == "sqrt":
        wgt = d ** alpha
    else:
        wgt = d
    wgt = wgt / wgt.sum()
    T = total if total is not None else float(d.sum() / speed)
    return wgt * T
