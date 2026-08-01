"""Scan matching in 2D: given two laser scans, how did the robot move between?

Three matchers, in increasing order of how much they know about the world.

  point-to-point ICP  -- pull every point onto its nearest neighbour
  point-to-line ICP   -- pull every point onto the LINE its neighbours lie on
  correlative matcher -- brute-force search over a grid of candidate poses

Project 19 built the 3D versions of the first two and measured point-to-plane
converging in 5 iterations where point-to-point needed 58.  The same argument
holds one dimension down and is re-measured here, because the reason is
geometric, not incidental: a laser scan samples a *surface*, and no two scans
sample the same POINTS on it.  Asking a point to land on the exact point its
neighbour hit is asking for something that is not true; asking it to land on
the same surface is asking for something that is.
"""

import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def pose_to_T(x):
    c, s = np.cos(x[2]), np.sin(x[2])
    return np.array([[c, -s, x[0]], [s, c, x[1]], [0.0, 0.0, 1.0]])


def T_to_pose(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def transform(pts, x):
    c, s = np.cos(x[2]), np.sin(x[2])
    R = np.array([[c, -s], [s, c]])
    return pts @ R.T + np.array([x[0], x[1]])


def compose(a, b):
    """Pose a followed by pose b, both as (x, y, theta)."""
    return T_to_pose(pose_to_T(a) @ pose_to_T(b))


def invert(a):
    return T_to_pose(np.linalg.inv(pose_to_T(a)))


def between(a, b):
    """The relative pose that takes frame a to frame b (b expressed in a)."""
    return T_to_pose(np.linalg.inv(pose_to_T(a)) @ pose_to_T(b))


def scan_to_points(ranges, angles, max_range):
    """Polar laser readings -> Cartesian points in the sensor frame.

    Readings at max range are dropped: they mean "nothing was hit", which is
    information about free space but NOT a surface, and feeding them to a
    matcher invents walls at exactly 8 m in every direction.
    """
    good = ranges < max_range - 1e-6
    r, a = ranges[good], angles[good]
    return np.stack([r * np.cos(a), r * np.sin(a)], axis=1)


def _nearest(src, dst):
    """Index of the closest dst point for each src point, and the distance."""
    d2 = ((src[:, None, 0] - dst[None, :, 0]) ** 2 +
          (src[:, None, 1] - dst[None, :, 1]) ** 2)
    idx = np.argmin(d2, axis=1)
    return idx, np.sqrt(d2[np.arange(len(src)), idx])


def _normals(pts, win=3, max_gap=0.5):
    """A surface normal per point, using the order the beams arrived in.

    A laser scan is not an unordered cloud: beam i and beam i+1 point at
    neighbouring directions, so consecutive points are neighbours ON THE
    SURFACE.  Fitting a line to a short run of consecutive points is therefore
    far more reliable than fitting one to the k nearest points in space, which
    happily reaches across a doorway and joins two different walls.

    Where the gap between consecutive points is large -- a depth discontinuity,
    the edge of an object -- there is no local surface to speak of, and the
    normal is marked invalid by returning a zero vector.  Those points are then
    ignored by the point-to-line term, which is the correct thing to do: they
    carry no reliable direction.
    """
    n = len(pts)
    out = np.zeros((n, 2))
    if n < 2 * win + 1:
        return out
    for i in range(n):
        lo, hi = max(0, i - win), min(n, i + win + 1)
        seg = pts[lo:hi]
        gaps = np.linalg.norm(np.diff(seg, axis=0), axis=1)
        if len(gaps) < 2 or gaps.max() > max_gap:
            continue                       # a break in the surface: no normal
        c = seg - seg.mean(axis=0)
        C = c.T @ c
        w, V = np.linalg.eigh(C)
        if w[1] < 1e-12 or w[0] / w[1] > 0.2:
            continue                       # too round to call it a line
        out[i] = V[:, 0]                   # smallest-variance direction
    return out


def icp(src, dst, x0=(0.0, 0.0, 0.0), mode="point", max_iter=40, tol=1e-5,
        max_pair=1.0, trim=0.85):
    """Align `src` onto `dst`.  Returns (pose, info, iterations, rms, overlap).

    `overlap` is the fraction of source points that ended up within 15 cm of a
    target point.  It is the single most useful number for deciding whether two
    scans are of the SAME PLACE, and the residual is not: a matcher handed two
    unrelated scans will happily line up whichever few points it can and report
    a small residual over those few, while leaving most of the scan unexplained.
    Experiment 4 uses it as the acceptance test for loop closures.

    `info` is the 3x3 information matrix of the fit -- the inverse covariance,
    up to the noise scale.  Its smallest eigenvalue is the number that warns
    you when the geometry is degenerate; experiment 3 is built on it.
    """
    x = np.array(x0, dtype=float)
    nrm = _normals(dst) if mode == "line" else None
    rms = np.inf
    it = 0
    info = np.zeros((3, 3))
    overlap = 0.0
    for it in range(1, max_iter + 1):
        cur = transform(src, x)
        idx, d = _nearest(cur, dst)
        overlap = float(np.mean(d < 0.15))
        keep = d < max_pair
        if trim < 1.0 and keep.sum() > 10:
            # Trimmed ICP: throw away the worst (1 - trim) of the pairs.  Two
            # scans of a moving world never overlap completely, and a single
            # point matched across a gap can drag the whole fit.
            thr = np.quantile(d[keep], trim)
            keep &= d <= thr
        if keep.sum() < 6:
            break
        p, q = cur[keep], dst[idx[keep]]
        # Jacobian of a transformed point with respect to (dx, dy, dtheta),
        # linearized about the current estimate.
        J = np.zeros((len(p), 2, 3))
        J[:, 0, 0] = 1.0
        J[:, 1, 1] = 1.0
        J[:, 0, 2] = -p[:, 1]
        J[:, 1, 2] = p[:, 0]
        r = q - p
        if mode == "line":
            nn = nrm[idx[keep]]
            valid = np.any(nn != 0.0, axis=1)
            if valid.sum() < 6:
                break
            nn, J, r = nn[valid], J[valid], r[valid]
            Jl = np.einsum("ni,nij->nj", nn, J)
            rl = np.einsum("ni,ni->n", nn, r)
            H = Jl.T @ Jl
            g = Jl.T @ rl
            rms_new = float(np.sqrt(np.mean(rl ** 2)))
        else:
            Jf = J.reshape(-1, 3)
            rf = r.reshape(-1)
            H = Jf.T @ Jf
            g = Jf.T @ rf
            rms_new = float(np.sqrt(np.mean(np.sum(r ** 2, axis=1))))
        info = H
        try:
            dx = np.linalg.solve(H + 1e-9 * np.eye(3), g)
        except np.linalg.LinAlgError:
            break
        # The increment is applied on the LEFT, not the right.  The Jacobian
        # above was written for the ALREADY-TRANSFORMED points, so (dx, dy,
        # dtheta) describes a small motion in the TARGET frame, and the target
        # frame is what left-composition means.  Applying it on the right --
        # composing it as a motion in the source frame -- looks almost the same
        # for small dtheta and quietly stops ICP from correcting rotation at
        # all.  That version left the rotation error exactly where the initial
        # guess put it, which made scan matching WORSE than the wheel odometry
        # that seeded it.
        x = compose(np.array([dx[0], dx[1], dx[2]]), x)
        rms = rms_new
        if np.linalg.norm(dx[:2]) < tol and abs(dx[2]) < tol:
            break
    return x, info, it, rms, overlap


def icp_multi(src, dst, seeds, mode="line", **kw):
    """Run ICP from several starting rotations and keep the best fit.

    ICP only ever walks downhill, so it can only find the answer if it starts
    in the right valley.  Between two poses that are close in space but far
    apart in TIME -- a loop closure -- there is no good starting guess at all,
    because the odometry between them has drifted by exactly the amount we are
    trying to measure.  Trying a handful of rotations and keeping whichever
    converges best is the cheapest way to cover the valleys that matter, since
    for a scan match the rotation is what decides which valley you are in.
    """
    best = None
    for s0 in seeds:
        res = icp(src, dst, x0=s0, mode=mode, **kw)
        # rank by OVERLAP first: a fit that explains more of the scan is a
        # better answer than one with a smaller residual over fewer points.
        if best is None or (res[4], -res[3]) > (best[4], -best[3]):
            best = res
    return best


def correlative(src, dst, x0=(0.0, 0.0, 0.0), win_xy=0.5, win_th=0.25,
                res_xy=0.05, res_th=0.02, sigma=0.15):
    """Brute-force search over a grid of candidate poses.

    No derivatives, no nearest-neighbour iteration: score every candidate by
    how well the transformed scan lands on a blurred version of the reference,
    and keep the best.  It cannot fall into a local minimum inside the search
    window, which is exactly what ICP does; it pays for that with cost that
    grows as the cube of the window size.  Real systems use it to INITIALIZE
    ICP, and that pairing is measured in experiment 2.
    """
    lo = dst.min(axis=0) - 1.0
    hi = dst.max(axis=0) + 1.0
    n = np.maximum(((hi - lo) / res_xy).astype(int) + 1, 2)
    grid = np.zeros((n[1], n[0]))
    ci = ((dst - lo) / res_xy).astype(int)
    ci[:, 0] = np.clip(ci[:, 0], 0, n[0] - 1)
    ci[:, 1] = np.clip(ci[:, 1], 0, n[1] - 1)
    grid[ci[:, 1], ci[:, 0]] = 1.0
    # blur, so a near miss still scores; a 1-cell-wide target would make the
    # search surface a field of spikes with nothing in between to climb
    k = int(np.ceil(3 * sigma / res_xy))
    xs = np.arange(-k, k + 1) * res_xy
    g1 = np.exp(-0.5 * (xs / sigma) ** 2)
    g1 /= g1.sum()
    grid = np.apply_along_axis(lambda m: np.convolve(m, g1, mode="same"), 0, grid)
    grid = np.apply_along_axis(lambda m: np.convolve(m, g1, mode="same"), 1, grid)

    best, best_s = np.array(x0, dtype=float), -np.inf
    ths = np.arange(x0[2] - win_th, x0[2] + win_th + 1e-9, res_th)
    offs = np.arange(-win_xy, win_xy + 1e-9, res_xy)
    for th in ths:
        c, s = np.cos(th), np.sin(th)
        rot = src @ np.array([[c, s], [-s, c]])
        for dx in offs:
            for dy in offs:
                p = rot + np.array([x0[0] + dx, x0[1] + dy])
                ci = ((p - lo) / res_xy).astype(int)
                ok = ((ci[:, 0] >= 0) & (ci[:, 0] < n[0]) &
                      (ci[:, 1] >= 0) & (ci[:, 1] < n[1]))
                sc = grid[ci[ok, 1], ci[ok, 0]].sum()
                if sc > best_s:
                    best_s, best = sc, np.array([x0[0] + dx, x0[1] + dy, th])
    return best, best_s
