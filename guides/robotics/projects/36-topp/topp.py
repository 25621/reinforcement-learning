"""Time-optimal path parameterisation -- the shared library for project 36.

The input is a GEOMETRIC PATH: a curve q(s) through joint space, where s runs
from 0 to 1 and says *where* you are but nothing about *when*.  The output is a
TRAJECTORY: the same curve plus a schedule s(t) saying how fast to run along it.

Splitting the problem this way is the whole trick.  Planning a trajectory
directly means searching over positions AND velocities at once, which is a much
bigger space.  Fixing the path first turns the timing question into a problem
with ONE unknown function of ONE variable -- and that one is solvable exactly.

The algorithm is TOPP-RA (Pham & Pham, 2018): "RA" for Reachability Analysis.
"""

import math

import numpy as np


# ------------------------------------------------------------------ the path
class CubicPath:
    """A C2-continuous cubic spline through waypoints, parameterised on [0, 1].

    "C2" means position, velocity and acceleration are all continuous.  We need
    the second derivative because the acceleration constraint below contains
    q''(s); a piecewise-straight path has q'' = 0 everywhere except at the
    corners, where it is infinite, and no timing law can survive that.  Fitting
    a spline is how the corner gets a finite (if large) curvature.
    """

    def __init__(self, waypoints):
        P = np.asarray(waypoints, float)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        seg = np.maximum(seg, 1e-9)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        self.total = t[-1]
        self.t = t / t[-1]
        self.P = P
        self.n = P.shape[1]
        self.M = np.stack([self._second_derivs(self.t, P[:, j])
                           for j in range(self.n)], axis=1)

    @staticmethod
    def _second_derivs(t, y):
        """Natural cubic spline: solve the tridiagonal system for y''."""
        n = len(t)
        h = np.diff(t)
        A = np.zeros((n, n))
        r = np.zeros(n)
        A[0, 0] = A[-1, -1] = 1.0            # natural: y'' = 0 at both ends
        for i in range(1, n - 1):
            A[i, i - 1] = h[i - 1]
            A[i, i] = 2.0 * (h[i - 1] + h[i])
            A[i, i + 1] = h[i]
            r[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] -
                          (y[i] - y[i - 1]) / h[i - 1])
        return np.linalg.solve(A, r)

    def _locate(self, s):
        s = np.clip(np.atleast_1d(np.asarray(s, float)), 0.0, 1.0)
        i = np.clip(np.searchsorted(self.t, s) - 1, 0, len(self.t) - 2)
        return s, i

    def eval(self, s):
        s, i = self._locate(s)
        h = self.t[i + 1] - self.t[i]
        a = (self.t[i + 1] - s) / h
        b = (s - self.t[i]) / h
        return (a[:, None] * self.P[i] + b[:, None] * self.P[i + 1] +
                ((a ** 3 - a)[:, None] * self.M[i] +
                 (b ** 3 - b)[:, None] * self.M[i + 1]) * (h ** 2)[:, None] / 6.0)

    def d1(self, s):
        s, i = self._locate(s)
        h = self.t[i + 1] - self.t[i]
        a = (self.t[i + 1] - s) / h
        b = (s - self.t[i]) / h
        return ((self.P[i + 1] - self.P[i]) / h[:, None] +
                ((-(3 * a ** 2 - 1))[:, None] * self.M[i] +
                 (3 * b ** 2 - 1)[:, None] * self.M[i + 1]) * h[:, None] / 6.0)

    def d2(self, s):
        s, i = self._locate(s)
        h = self.t[i + 1] - self.t[i]
        a = (self.t[i + 1] - s) / h
        b = (s - self.t[i]) / h
        return a[:, None] * self.M[i] + b[:, None] * self.M[i + 1]


# ------------------------------------------------------------------ 2-D LP
def lp2(cvec, A, b, lo, hi, tol=1e-9):
    """Maximise cvec . z over {A z <= b} intersected with the box [lo, hi].

    Only two variables, so there is no need for a simplex implementation: the
    optimum of a linear objective over a polygon is always at a corner, and
    with m constraints there are at most m*(m-1)/2 corners.  We build them all,
    throw away the infeasible ones, and take the best.  With m around 16 that
    is a few hundred arithmetic operations -- far cheaper than calling a
    general LP solver, and it cannot fail to converge.
    """
    A = np.vstack([A, np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0],
                                [0.0, -1.0]])])
    b = np.concatenate([b, [hi[0], -lo[0], hi[1], -lo[1]]])
    m = len(b)
    best, bestz = -np.inf, None
    for i in range(m):
        for j in range(i + 1, m):
            M = np.array([A[i], A[j]])
            det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
            if abs(det) < 1e-12:
                continue
            rhs = np.array([b[i], b[j]])
            z = np.array([(rhs[0] * M[1, 1] - rhs[1] * M[0, 1]) / det,
                          (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) / det])
            if np.all(A @ z <= b + tol):
                v = float(cvec @ z)
                if v > best:
                    best, bestz = v, z
    return best, bestz


# ------------------------------------------------------------------ TOPP-RA
class Limits:
    """Velocity and acceleration bounds, per joint."""

    def __init__(self, vmax, amax):
        self.vmax = np.asarray(vmax, float)
        self.amax = np.asarray(amax, float)

    def stage(self, qd, qdd):
        """Return (A, b) so that A [u, x] <= b encodes the limits at one s.

        With x = sdot^2 and u = sddot, the chain rule gives

            joint velocity      qdot_j     = q'_j * sdot     ->  x <= (v_j/|q'_j|)^2
            joint acceleration  qddot_j    = q'_j * u + q''_j * x

        The velocity limit is a bound on x alone, the acceleration limit is a
        line in the (u, x) plane.  Both are LINEAR in (u, x) -- which is the
        reason for using x = sdot^2 rather than sdot itself, and the reason the
        problem is exactly solvable instead of merely approximately.
        """
        rows, rhs = [], []
        for j in range(len(qd)):
            rows.append([qd[j], qdd[j]])
            rhs.append(self.amax[j])
            rows.append([-qd[j], -qdd[j]])
            rhs.append(self.amax[j])
        return np.array(rows), np.array(rhs)

    def xmax(self, qd):
        with np.errstate(divide="ignore"):
            return float(np.min(np.where(np.abs(qd) > 1e-9,
                                         (self.vmax / np.maximum(np.abs(qd), 1e-9)) ** 2,
                                         np.inf)))


class TorqueLimits:
    """Torque bounds turned into (u, x) constraints via the arm's dynamics.

    tau = M(q) qddot + C(q, qdot) qdot + g(q), and along a fixed path

        qdot  = q' sdot,   qddot = q' u + q'' x

    so   tau = [M q'] u + [M q'' + C~ q'] x + g(q),  which is again LINEAR in
    (u, x).  The important consequence: the bound depends on the CONFIGURATION,
    so the same path is traversable faster in some poses than in others -- a
    fact a constant acceleration limit cannot express.
    """

    def __init__(self, model, taumax, vmax, dyn):
        self.model = model
        self.taumax = np.asarray(taumax, float)
        self.vmax = np.asarray(vmax, float)
        self.dyn = dyn

    def stage_at(self, q, qd, qdd):
        M = self.dyn.mass_matrix(self.model, q)
        g = self.dyn.gravity_torque(self.model, q)
        C = self.dyn.coriolis_matrix(self.model, q, qd)
        a = M @ qd
        bb = M @ qdd + C @ qd
        rows, rhs = [], []
        for j in range(len(q)):
            rows.append([a[j], bb[j]])
            rhs.append(self.taumax[j] - g[j])
            rows.append([-a[j], -bb[j]])
            rhs.append(self.taumax[j] + g[j])
        return np.array(rows), np.array(rhs)

    def xmax(self, qd):
        return float(np.min((self.vmax / np.maximum(np.abs(qd), 1e-9)) ** 2))


def topp_ra(path, limits, N=200, u_bound=2000.0, x_start=0.0, x_end=0.0,
            torque=None):
    """Time-optimal parameterisation by reachability analysis.

    Two sweeps over the discretised path:

      BACKWARD -- compute, for every grid point, the largest squared speed from
        which it is still possible to reach the end while obeying every limit.
        This is the "controllable set".  Going backwards is what makes braking
        distance automatic: a point 10 cm before a hairpin inherits the hairpin's
        speed limit plus however much you can shed in 10 cm.

      FORWARD -- start at the required initial speed and, at every step, go as
        fast as the controllable set allows.  Because the backward pass has
        already guaranteed that every allowed speed is survivable, being greedy
        here is not just safe, it is optimal.

    A beginner might ask why two passes are needed when a single forward pass
    already respects every limit at every step.  Because a forward-only pass is
    greedy about the present and blind to the future: it will happily accelerate
    into a corner it then cannot slow down for, and discover the problem when it
    is too late to fix.  The backward pass is what converts local limits into a
    global speed plan.
    """
    ss = np.linspace(0.0, 1.0, N + 1)
    ds = 1.0 / N
    qs = path.eval(ss)
    qd = path.d1(ss)
    qdd = path.d2(ss)

    stages = []
    xmaxs = np.empty(N + 1)
    for i in range(N + 1):
        if torque is not None:
            A, b = torque.stage_at(qs[i], qd[i], qdd[i])
            xmaxs[i] = torque.xmax(qd[i])
        else:
            A, b = limits.stage(qd[i], qdd[i])
            xmaxs[i] = limits.xmax(qd[i])
        stages.append((A, b))

    # ---- backward pass: controllable sets
    xbar = np.zeros(N + 1)
    xbar[N] = min(x_end, xmaxs[N])
    for i in range(N - 1, -1, -1):
        A, b = stages[i]
        # x_{i+1} = x_i + 2 u ds  must land inside [0, xbar_{i+1}]
        Aex = np.vstack([A, [2.0 * ds, 1.0], [-2.0 * ds, -1.0]])
        bex = np.concatenate([b, [xbar[i + 1], 0.0]])
        val, _ = lp2(np.array([0.0, 1.0]), Aex, bex,
                     lo=[-u_bound, 0.0], hi=[u_bound, max(xmaxs[i], 1e-12)])
        xbar[i] = max(0.0, val) if np.isfinite(val) else 0.0

    # ---- forward pass: go as fast as the controllable set permits
    x = np.zeros(N + 1)
    u = np.zeros(N)
    x[0] = min(x_start, xbar[0])
    feasible = True
    for i in range(N):
        A, b = stages[i]
        # fix x = x_i, maximise the reachable x_{i+1}
        Aex = np.vstack([A, [2.0 * ds, 1.0], [-2.0 * ds, -1.0],
                         [0.0, 1.0], [0.0, -1.0]])
        bex = np.concatenate([b, [xbar[i + 1], 0.0], [x[i], -x[i]]])
        val, z = lp2(np.array([2.0 * ds, 1.0]), Aex, bex,
                     lo=[-u_bound, 0.0], hi=[u_bound, max(xmaxs[i], 1e-12)])
        if z is None:
            feasible = False
            break
        u[i] = z[0]
        x[i + 1] = min(max(0.0, x[i] + 2.0 * ds * z[0]), xbar[i + 1])

    sdot = np.sqrt(np.maximum(x, 0.0))
    # trapezoid in ds/sdot, written so a zero speed does not divide by zero
    dt = np.zeros(N)
    for i in range(N):
        denom = sdot[i] + sdot[i + 1]
        dt[i] = 2.0 * ds / denom if denom > 1e-9 else 0.0
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return dict(s=ss, x=x, sdot=sdot, u=u, t=t, duration=float(t[-1]),
                xbar=xbar, xmax=xmaxs, q=qs, qd=qd, qdd=qdd,
                # the endpoints are meant to be at rest; every point in
                # between must have positive speed or the robot has stalled
                feasible=feasible and bool(np.all(sdot[1:-1] > 1e-6)))


def uniform_scaling(path, limits, N=200):
    """The naive baseline: run the path at a single constant sdot.

    Pick the largest constant speed that violates nothing anywhere.  This is
    what "just slow the whole thing down until it stops complaining" means, and
    it is what a great many production systems actually ship.  It is safe, it
    is one line, and experiment 2 measures exactly how much time it wastes.
    """
    ss = np.linspace(0.0, 1.0, N + 1)
    qd = path.d1(ss)
    qdd = path.d2(ss)
    lo, hi = 1e-6, 1e4
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        ok = (np.all(np.abs(qd) * mid <= limits.vmax + 1e-9) and
              np.all(np.abs(qdd) * mid ** 2 <= limits.amax + 1e-9))
        if ok:
            lo = mid
        else:
            hi = mid
    sdot = lo
    return dict(sdot=sdot, duration=1.0 / sdot,
                t=ss / sdot, s=ss, q=path.eval(ss))


def trapezoid_scaling(path, limits, N=400):
    """The honest baseline: one trapezoid speed profile for the whole path.

    `uniform_scaling` cheats -- it runs at a constant speed, which means it
    starts and stops instantaneously.  A real robot has to accelerate away from
    rest and brake back to it.  So here we use the standard textbook profile:
    accelerate at a constant rate, cruise, brake at a constant rate.

    The trick that keeps it simple is TIME SCALING.  Build the profile once in
    "canonical" units, then stretch it in time by a factor.  Stretching by k
    divides every speed by k and every acceleration by k^2, so finding the
    fastest legal version is a one-dimensional search on k -- no iteration over
    the path shape at all.

    What it cannot do is speed up on the easy stretches, because there is only
    ONE number for the whole path.  Whatever the tightest corner demands, the
    long straight gets too.  That is the waste TOPP removes.
    """
    ss = np.linspace(0.0, 1.0, N + 1)
    qd = path.d1(ss)
    qdd = path.d2(ss)

    # Canonical triangular profile over s in [0, 1] with peak sdot = 1 and
    # |sddot| = 1: accelerate for tau in [0,1], decelerate for tau in [1,2].
    taus = np.linspace(0.0, 2.0, 2 * N + 1)
    sdot_c = np.where(taus <= 1.0, taus, 2.0 - taus)
    sddot_c = np.where(taus <= 1.0, 1.0, -1.0)
    s_c = np.where(taus <= 1.0, 0.5 * taus ** 2, 1.0 - 0.5 * (2.0 - taus) ** 2)

    def ok(T):
        k = 2.0 / T                      # tau = k t
        sd = sdot_c * k
        sdd = sddot_c * k * k
        qd_i = np.stack([np.interp(s_c, ss, qd[:, j])
                         for j in range(qd.shape[1])], 1)
        qdd_i = np.stack([np.interp(s_c, ss, qdd[:, j])
                          for j in range(qdd.shape[1])], 1)
        vel = qd_i * sd[:, None]
        acc = qd_i * sdd[:, None] + qdd_i * (sd ** 2)[:, None]
        return (np.all(np.abs(vel) <= limits.vmax + 1e-9) and
                np.all(np.abs(acc) <= limits.amax + 1e-9))

    lo, hi = 1e-3, 1e4
    while not ok(hi):
        hi *= 2
        if hi > 1e9:
            break
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return dict(duration=hi, s=s_c, t=taus / (2.0 / hi),
                sdot=sdot_c * (2.0 / hi))


def verify(path, res, limits, torque=None, model=None, dyn=None):
    """Re-derive joint velocity, acceleration (and torque) from the schedule
    and report the worst violation, as a fraction of the limit."""
    s, sdot, u = res["s"], res["sdot"], res["u"]
    qd = path.d1(s)
    qdd = path.d2(s)
    x = sdot ** 2
    vel = qd * sdot[:, None]
    acc = qd[:-1] * u[:, None] + qdd[:-1] * x[:-1, None]
    out = dict(
        vel_ratio=float(np.max(np.abs(vel) / limits.vmax)),
        acc_ratio=float(np.max(np.abs(acc) / limits.amax)))
    if torque is not None:
        worst = 0.0
        for i in range(len(u)):
            q = path.eval(np.array([s[i]]))[0]
            qq = qd[i] * sdot[i]
            aa = qd[i] * u[i] + qdd[i] * x[i]
            tau = (dyn.mass_matrix(model, q) @ aa +
                   dyn.coriolis_matrix(model, q, qq) @ qq +
                   dyn.gravity_torque(model, q))
            worst = max(worst, float(np.max(np.abs(tau) / torque.taumax)))
        out["torque_ratio"] = worst
    return out


def active_fraction(path, res, limits, tol=0.02):
    """Fraction of the path where at least one limit is at its bound.

    Time-optimal control theory predicts this should be 1.0: if nothing were
    saturated anywhere, you could go a little faster there, so the answer would
    not have been optimal.  Measuring it is the cheapest possible check that
    the solver really did find the optimum.
    """
    s, sdot, u = res["s"], res["sdot"], res["u"]
    qd = path.d1(s)
    qdd = path.d2(s)
    x = sdot ** 2
    hits = 0
    for i in range(len(u)):
        v = np.abs(qd[i] * sdot[i]) / limits.vmax
        a = np.abs(qd[i] * u[i] + qdd[i] * x[i]) / limits.amax
        if max(v.max(), a.max()) > 1.0 - tol:
            hits += 1
    return hits / len(u)
