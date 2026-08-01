"""A 2D pose-graph optimizer: sparse Gauss-Newton with robust kernels.

Shared with project 30, which builds the general factor-graph machinery on top
of this.

A POSE GRAPH is the whole trajectory written as a graph.  Every node is a robot
pose; every edge is a measurement of the relative pose between two nodes.  Two
kinds of edge:

  ODOMETRY edges join consecutive poses.  There are n-1 of them and they always
  exist.  On their own they define exactly one trajectory: start at the origin
  and chain them.  Optimizing a graph with only odometry edges changes nothing.

  LOOP-CLOSURE edges join poses that are far apart in TIME but close in SPACE --
  "I have been here before".  These are the only edges that carry new
  information, because they are the only ones that constrain a quantity the
  chain did not already determine.

Solving means finding the poses that best satisfy all edges at once.  Since
there are more equations than unknowns once a loop closes, "best" means least
squares, and the problem is nonlinear because rotations are.
"""

import numpy as np

from scanmatch import wrap, pose_to_T, T_to_pose, between, compose


def _J_between(a, b):
    """Derivatives of between(a, b) with respect to a and b.

    between(a, b) = R(a)' (b_xy - a_xy), wrap(b_th - a_th)
    """
    ca, sa = np.cos(a[2]), np.sin(a[2])
    R = np.array([[ca, -sa], [sa, ca]])
    d = b[:2] - a[:2]
    dR = np.array([[-sa, ca], [-ca, -sa]])          # d(R')/d(theta_a)
    Ja = np.zeros((3, 3))
    Ja[:2, :2] = -R.T
    Ja[:2, 2] = dR @ d
    Ja[2, 2] = -1.0
    Jb = np.zeros((3, 3))
    Jb[:2, :2] = R.T
    Jb[2, 2] = 1.0
    return Ja, Jb


# ------------------------------------------------------------ robust kernels
#
# A kernel decides how much a large residual is allowed to pull.  Plain least
# squares uses rho(e) = e^2/2, so a residual ten times too big pulls a HUNDRED
# times as hard -- one wrong loop closure can outvote a thousand good edges.
# A robust kernel flattens rho out past some threshold so that beyond it, extra
# wrongness stops adding extra pull.
#
# Each returns the WEIGHT to apply to that edge's squared error, which is
# rho'(e)/e -- the standard "iteratively reweighted least squares" trick: solve
# a weighted linear problem, recompute the weights, repeat.


def kernel_l2(e2, delta):
    return np.ones_like(e2)


def kernel_huber(e2, delta):
    """Named after Peter Huber (1964).  Quadratic near zero, LINEAR beyond
    delta, so a far-out residual pulls with constant force instead of growing
    force.  It never fully ignores an outlier, only stops letting it shout."""
    e = np.sqrt(np.maximum(e2, 1e-12))
    return np.where(e <= delta, 1.0, delta / e)


def kernel_cauchy(e2, delta):
    """From the Cauchy distribution, which has such heavy tails that its mean
    does not exist -- exactly the shape you want when you believe some of your
    data is arbitrarily wrong.  The weight FALLS towards zero for large
    residuals, so a bad edge is genuinely switched off, not merely quietened."""
    return 1.0 / (1.0 + e2 / delta ** 2)


def kernel_gm(e2, delta):
    """Geman-McClure.  Even more aggressive than Cauchy: weight ~ 1/e^4 far
    out.  Powerful and easy to get stuck with, because an edge it switches off
    early can never argue its way back in."""
    d2 = delta ** 2
    return (d2 / (d2 + e2)) ** 2


KERNELS = {"l2": kernel_l2, "huber": kernel_huber,
           "cauchy": kernel_cauchy, "gm": kernel_gm}


class PoseGraph:
    def __init__(self):
        self.nodes = []          # list of (x, y, theta)
        self.edges = []          # (i, j, measured relative pose, information)

    def add_node(self, pose):
        self.nodes.append(np.asarray(pose, dtype=float).copy())
        return len(self.nodes) - 1

    def add_edge(self, i, j, rel, info):
        self.edges.append((i, j, np.asarray(rel, float).copy(),
                           np.asarray(info, float).copy()))

    def poses(self):
        return np.array(self.nodes)

    # ------------------------------------------------------------- residuals
    def residuals(self, nodes=None):
        nodes = self.nodes if nodes is None else nodes
        out = np.empty((len(self.edges), 3))
        for k, (i, j, rel, info) in enumerate(self.edges):
            r = between(nodes[i], nodes[j]) - rel
            r[2] = wrap(r[2])
            out[k] = r
        return out

    def chi2(self, nodes=None, weights=None):
        r = self.residuals(nodes)
        tot = 0.0
        for k, (i, j, rel, info) in enumerate(self.edges):
            w = 1.0 if weights is None else weights[k]
            tot += w * float(r[k] @ info @ r[k])
        return tot

    # ------------------------------------------------------------- optimize
    def optimize(self, iters=30, kernel="l2", delta=1.0, fix_first=True,
                 lm=1e-6, verbose=False):
        """Gauss-Newton with a small Levenberg damping term.

        Named for Carl Friedrich Gauss and Isaac Newton: it is Newton's method
        with the second-derivative (Hessian) replaced by J'J, which is cheap
        and is a good approximation exactly when the residuals are small --
        i.e. when the answer is nearly right, which is the situation after the
        first couple of iterations.

        `fix_first` pins node 0.  Without it the problem is singular: the whole
        trajectory can be translated and rotated freely without changing a
        single relative measurement, so J'J has three zero eigenvalues (the
        "gauge freedom") and the solve fails.  Pinning one pose picks a frame.
        """
        n = len(self.nodes)
        kern = KERNELS[kernel]
        hist = []
        for it in range(iters):
            H = np.zeros((3 * n, 3 * n))
            g = np.zeros(3 * n)
            r_all = self.residuals()
            e2 = np.array([float(r @ info @ r)
                           for r, (_, _, _, info) in zip(r_all, self.edges)])
            w_all = kern(e2, delta)
            for k, (i, j, rel, info) in enumerate(self.edges):
                r = r_all[k]
                Ja, Jb = _J_between(self.nodes[i], self.nodes[j])
                w = w_all[k]
                Wi = w * info
                si, sj = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
                H[si, si] += Ja.T @ Wi @ Ja
                H[si, sj] += Ja.T @ Wi @ Jb
                H[sj, si] += Jb.T @ Wi @ Ja
                H[sj, sj] += Jb.T @ Wi @ Jb
                g[si] -= Ja.T @ Wi @ r
                g[sj] -= Jb.T @ Wi @ r
            if fix_first:
                H[0:3, :] = 0.0
                H[:, 0:3] = 0.0
                H[0:3, 0:3] = np.eye(3)
                g[0:3] = 0.0
            H += lm * np.eye(3 * n)
            try:
                dx = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            for i in range(n):
                self.nodes[i] = self.nodes[i] + dx[3 * i:3 * i + 3]
                self.nodes[i][2] = wrap(self.nodes[i][2])
            c = self.chi2(weights=w_all)
            hist.append(c)
            if verbose:
                print(f"    it {it:2d}  chi2 {c:12.4f}  |dx| {np.linalg.norm(dx):.3e}")
            if np.linalg.norm(dx) < 1e-9:
                break
        return np.array(hist), w_all


def chain(poses0, rels):
    """Build an initial guess by chaining relative poses from a starting pose."""
    out = [np.asarray(poses0, float)]
    for r in rels:
        out.append(compose(out[-1], r))
    return np.array(out)


def align_poses(est, truth):
    """Rigidly align a whole trajectory (x, y AND heading) onto the truth.

    A SLAM answer is only defined up to a global rigid transform -- pinning
    node 0 picks *a* frame, but the optimizer is free to rotate everything
    about it, and a small residual rotation at the start becomes metres of
    displacement at the far end of the map.  Comparing or DRAWING raw
    coordinates therefore scores the frame choice rather than the map, which is
    why every SLAM benchmark aligns first.
    """
    est = np.asarray(est)
    e, t = est[:, :2], np.asarray(truth)[:, :2]
    ec, tc = e - e.mean(0), t - t.mean(0)
    U, S, Vt = np.linalg.svd(ec.T @ tc)
    R = U @ np.diag([1.0, np.linalg.det(U @ Vt)]) @ Vt
    out = np.empty_like(est)
    out[:, :2] = ec @ R + t.mean(0)
    out[:, 2] = wrap(est[:, 2] + np.arctan2(R[0, 1], R[0, 0]))
    return out


def align_and_ate(est, truth):
    """Absolute Trajectory Error after aligning the two trajectories rigidly.

    A pose graph fixes node 0, so its answer lives in whatever frame node 0
    happened to be in.  Comparing raw coordinates would score the frame choice,
    not the map.  Aligning first (Umeyama / Kabsch, as in project 19) removes
    that and leaves only the shape error, which is what SLAM is judged on.
    """
    e, t = np.asarray(est)[:, :2], np.asarray(truth)[:, :2]
    ec, tc = e - e.mean(0), t - t.mean(0)
    U, S, Vt = np.linalg.svd(ec.T @ tc)
    R = U @ np.diag([1.0, np.linalg.det(U @ Vt)]) @ Vt
    aligned = ec @ R + t.mean(0)
    return float(np.sqrt(np.mean(np.sum((aligned - t) ** 2, axis=1)))), aligned
