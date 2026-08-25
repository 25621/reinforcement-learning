"""CHOMP from scratch -- signed distance fields and covariant gradient descent.

CHOMP = Covariant Hamiltonian Optimization for Motion Planning (Ratliff et al.,
2009).  Two of those words carry the whole idea:

  * "Optimization"  -- the trajectory is not searched for, it is IMPROVED.  You
    hand CHOMP a complete (probably terrible) trajectory and it walks downhill.
  * "Covariant"     -- the downhill direction is measured with a ruler that
    understands trajectories, not one that treats the N waypoints as N
    unrelated points.  This is the part that makes it work, and experiment 2
    measures how much it is worth.

("Hamiltonian" refers to the Hamiltonian-Jacobi view of the update used in the
original derivation; nothing in the code below needs it.)

Imported by project 36 (TOPP) for its smooth test paths.
"""

import math

import numpy as np


# ------------------------------------------------------------------ the SDF
def _dt_1d(f):
    """Exact 1D squared-distance transform (Felzenszwalb & Huttenlocher 2004).

    It computes, for every index q, min_p ( (q-p)^2 + f[p] ).  Doing that
    naively costs O(n^2); this does it in O(n) by tracking the lower envelope
    of the parabolas rooted at each p.
    """
    n = len(f)
    d = np.empty(n)
    v = np.zeros(n, dtype=int)
    z = np.empty(n + 1)
    k = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def edt(mask):
    """Euclidean distance (in cells) from every cell to the nearest True cell."""
    big = 1e12
    f = np.where(mask, 0.0, big)
    out = np.empty_like(f)
    for i in range(f.shape[0]):
        out[i] = _dt_1d(f[i])
    for j in range(f.shape[1]):
        out[:, j] = _dt_1d(out[:, j])
    return np.sqrt(out)


class SDF:
    """A Signed Distance Field over a rectangle.

    At every point it stores the distance to the nearest obstacle surface,
    POSITIVE outside an obstacle and NEGATIVE inside it.  Two things make this
    the right data structure for trajectory optimization:

      1. it is defined everywhere, including deep inside an obstacle, so a
         trajectory that starts in collision still gets told which way is out;
      2. its gradient points directly away from the nearest surface, so
         "push the trajectory out of collision" becomes one subtraction.

    A plain occupancy grid (project 27) has neither property: it answers
    "blocked?" with a yes or a no, and a yes carries no direction and no
    magnitude, so gradient descent has nothing to descend.
    """

    def __init__(self, occ, lo, hi):
        self.occ = occ
        self.lo = np.asarray(lo, float)
        self.hi = np.asarray(hi, float)
        h, w = occ.shape
        self.res = (self.hi[0] - self.lo[0]) / (w - 1)
        d_out = edt(occ)          # distance to nearest obstacle cell
        d_in = edt(~occ)          # distance to nearest free cell
        self.field = (d_out - d_in) * self.res
        gy, gx = np.gradient(self.field, self.res)
        self.gx, self.gy = gx, gy

    def _idx(self, pts):
        p = np.atleast_2d(np.asarray(pts, float))
        fx = (p[:, 0] - self.lo[0]) / self.res
        fy = (p[:, 1] - self.lo[1]) / self.res
        h, w = self.occ.shape
        fx = np.clip(fx, 0, w - 1.001)
        fy = np.clip(fy, 0, h - 1.001)
        return fx, fy

    def _bilinear(self, arr, fx, fy):
        x0 = np.floor(fx).astype(int)
        y0 = np.floor(fy).astype(int)
        tx = fx - x0
        ty = fy - y0
        return (arr[y0, x0] * (1 - tx) * (1 - ty) +
                arr[y0, x0 + 1] * tx * (1 - ty) +
                arr[y0 + 1, x0] * (1 - tx) * ty +
                arr[y0 + 1, x0 + 1] * tx * ty)

    def value(self, pts):
        fx, fy = self._idx(pts)
        return self._bilinear(self.field, fx, fy)

    def grad(self, pts):
        fx, fy = self._idx(pts)
        return np.stack([self._bilinear(self.gx, fx, fy),
                         self._bilinear(self.gy, fx, fy)], axis=1)


def build_sdf(circles, lo=(0.0, 0.0), hi=(10.0, 10.0), cells=241):
    xs = np.linspace(lo[0], hi[0], cells)
    ys = np.linspace(lo[1], hi[1], cells)
    X, Y = np.meshgrid(xs, ys)
    occ = np.zeros_like(X, dtype=bool)
    for cx, cy, r in np.atleast_2d(circles):
        occ |= (X - cx) ** 2 + (Y - cy) ** 2 <= r * r
    return SDF(occ, lo, hi)


# ------------------------------------------------------------------ cost
def obstacle_cost(d, eps):
    """CHOMP's hinge cost on the signed distance d.

    It has three pieces, and the reason for each is worth stating:

        d < 0        c = -d + eps/2     grows without limit inside an obstacle
        0 <= d < eps c = (d-eps)^2/(2 eps)   a soft buffer, smoothly reaching 0
        d >= eps     c = 0              far away, no opinion at all

    The middle piece is what makes the whole thing differentiable.  A cost that
    jumped from "0" to "huge" at the surface would have an infinite gradient at
    the surface and none anywhere else -- useless for gradient descent.  The
    quadratic ramp gives a gradient that grows steadily as the robot gets
    close, so it starts turning away BEFORE it touches anything.
    """
    d = np.asarray(d, float)
    c = np.zeros_like(d)
    inside = d < 0
    buf = (d >= 0) & (d < eps)
    c[inside] = -d[inside] + 0.5 * eps
    c[buf] = (d[buf] - eps) ** 2 / (2.0 * eps)
    return c


def obstacle_cost_deriv(d, eps):
    d = np.asarray(d, float)
    g = np.zeros_like(d)
    g[d < 0] = -1.0
    buf = (d >= 0) & (d < eps)
    g[buf] = (d[buf] - eps) / eps
    return g


def smoothness_matrices(n, q0, qN, order=1):
    """Build A, b for the smoothness cost  0.5 xi^T A xi + xi^T b + const.

    `order` is which derivative is being penalised: 1 = velocity (short paths),
    2 = acceleration (paths that do not jerk about).  K is the finite-
    difference operator and e carries the fixed endpoints, so that
    K xi + e is literally the stack of differences along the trajectory.
    """
    if order == 1:
        K = np.zeros((n + 1, n))
        e0 = np.zeros((n + 1, len(q0)))
        for i in range(n + 1):
            if i < n:
                K[i, i] = 1.0
            if i > 0:
                K[i, i - 1] = -1.0
        e0[0] = -q0
        e0[n] = qN
    else:
        K = np.zeros((n + 2, n))
        e0 = np.zeros((n + 2, len(q0)))
        for i in range(n + 2):
            for j, w in ((i - 2, 1.0), (i - 1, -2.0), (i, 1.0)):
                if 0 <= j < n:
                    K[i, j] = w
        e0[0] = q0
        e0[1] = -2.0 * q0
        e0[n] = -2.0 * qN
        e0[n + 1] = qN
    A = K.T @ K
    b = K.T @ e0
    return A, b, K, e0


def chomp(sdf, q0, qN, n=80, iters=300, eta=200.0, lam=1.0, eps=0.5,
          covariant=True, order=1, init=None, track=False, rng=None,
          noise=0.0):
    """Optimise a trajectory of n interior waypoints from q0 to qN.

    `covariant=False` switches off the A^{-1} step and does ordinary gradient
    descent on the same cost.  That is the control for experiment 2.
    """
    q0 = np.asarray(q0, float)
    qN = np.asarray(qN, float)
    if init is None:
        ts = np.linspace(0, 1, n + 2)[1:-1][:, None]
        xi = q0[None, :] * (1 - ts) + qN[None, :] * ts
    else:
        xi = np.asarray(init, float).copy()
        n = len(xi)
    if noise and rng is not None:
        xi = xi + rng.normal(0, noise, xi.shape)

    A, b, _, _ = smoothness_matrices(n, q0, qN, order)
    Ainv = np.linalg.inv(A + 1e-9 * np.eye(n))
    hist = []

    for it in range(iters):
        full = np.vstack([q0, xi, qN])
        # central differences give velocity and acceleration at each waypoint
        vel = 0.5 * (full[2:] - full[:-2])
        acc = full[2:] - 2.0 * full[1:-1] + full[:-2]
        speed = np.linalg.norm(vel, axis=1, keepdims=True)
        speed = np.maximum(speed, 1e-9)
        that = vel / speed                        # unit tangent

        d = sdf.value(xi)
        c = obstacle_cost(d, eps)
        gd = obstacle_cost_deriv(d, eps)[:, None] * sdf.grad(xi)

        # project the cost gradient onto the direction PERPENDICULAR to travel
        proj = gd - that * np.sum(gd * that, axis=1, keepdims=True)
        # curvature term: rewards taking the corner wide rather than speeding up
        kappa = (acc - that * np.sum(acc * that, axis=1, keepdims=True)) / speed ** 2
        g_obs = speed * (proj - c[:, None] * kappa)

        g_smooth = A @ xi + b
        g = g_obs + lam * g_smooth

        step = (Ainv @ g) if covariant else g
        xi = xi - (1.0 / eta) * step

        if track:
            hist.append(trajectory_report(sdf, q0, xi, qN, eps, lam, order))
    out = np.vstack([q0, xi, qN])
    return (out, hist) if track else out


def trajectory_report(sdf, q0, xi, qN, eps, lam, order=1):
    n = len(xi)
    A, b, K, e0 = smoothness_matrices(n, q0, qN, order)
    f_smooth = 0.5 * float(np.sum((K @ xi + e0) ** 2))
    d = sdf.value(xi)
    f_obs = float(np.sum(obstacle_cost(d, eps)))
    full = np.vstack([q0, xi, qN])
    length = float(np.sum(np.linalg.norm(np.diff(full, axis=0), axis=1)))
    return dict(smooth=f_smooth, obs=f_obs, total=f_obs + lam * f_smooth,
                min_d=float(d.min()), length=length,
                collision=bool((d < 0).any()))


def path_min_clearance(sdf, path, upsample=8):
    """Smallest signed distance anywhere along the path, checked between
    waypoints too -- a trajectory can be safe at every waypoint and still cut
    a corner between two of them."""
    p = np.asarray(path, float)
    pts = []
    for a, b in zip(p[:-1], p[1:]):
        for t in np.linspace(0, 1, upsample, endpoint=False):
            pts.append(a + t * (b - a))
    pts.append(p[-1])
    return float(sdf.value(np.asarray(pts)).min())


# ------------------------------------------------------------------ STOMP
def stomp(sdf, q0, qN, n=80, iters=300, k=12, sigma=0.35, eps=0.5, lam=1.0,
          order=1, rng=None, init=None, track=False):
    """STOMP: the same cost, optimised WITHOUT a gradient.

    Stochastic Trajectory Optimization for Motion Planning (Kalakrishnan et
    al., 2011).  Each round it jiggles the current trajectory k different ways,
    scores each, and moves toward the cheap ones with a softmax weighting.

    Why would anyone give up the gradient?  Because the gradient only knows
    the cost immediately around the current trajectory.  A noisy sample can
    land on the far side of an obstacle and report back that it is cheaper
    there -- information no local derivative could have supplied.  The price
    is k cost evaluations per iteration instead of one.

    The noise is drawn with covariance A^{-1}, the same matrix CHOMP uses in
    its update.  That makes the jiggles SMOOTH (neighbouring waypoints move
    together) instead of white noise that would shred the trajectory.
    """
    rng = rng or np.random.default_rng(0)
    q0 = np.asarray(q0, float)
    qN = np.asarray(qN, float)
    if init is None:
        ts = np.linspace(0, 1, n + 2)[1:-1][:, None]
        xi = q0[None, :] * (1 - ts) + qN[None, :] * ts
    else:
        xi = np.asarray(init, float).copy()
        n = len(xi)
    A, b, K, e0 = smoothness_matrices(n, q0, qN, order)
    Ainv = np.linalg.inv(A + 1e-9 * np.eye(n))
    scale = Ainv / np.max(np.abs(Ainv))
    L = np.linalg.cholesky(scale + 1e-9 * np.eye(n))
    hist = []

    def cost(x):
        d = sdf.value(x)
        return (float(np.sum(obstacle_cost(d, eps))) +
                lam * 0.5 * float(np.sum((K @ x + e0) ** 2)))

    for it in range(iters):
        base = cost(xi)
        noises = [L @ rng.normal(0, sigma, xi.shape) for _ in range(k)]
        costs = np.array([cost(xi + e) for e in noises])
        spread = max(1e-9, float(np.ptp(costs)))
        w = np.exp(-10.0 * (costs - costs.min()) / spread)
        w /= w.sum()
        delta = sum(wi * e for wi, e in zip(w, noises))
        cand = xi + delta
        if cost(cand) < base:
            xi = cand
        if track:
            hist.append(trajectory_report(sdf, q0, xi, qN, eps, lam, order))
    out = np.vstack([q0, xi, qN])
    return (out, hist) if track else out
