"""Iterative Closest Point, both flavours, from scratch.

ICP is two steps in a loop, and neither is hard:

    1. for every point in cloud A, find the closest point in cloud B
    2. compute the rigid motion that best lines up those pairs; apply it

The loop is not guaranteed to reach the right answer -- only to stop getting
worse.  Step 1 is a guess about which point corresponds to which, and if the
guess is wrong the motion in step 2 is wrong too, and it can settle into a
wrong alignment that no further iteration escapes.  Everything interesting
about ICP is about that failure and how to see it coming.
"""

import numpy as np


# --------------------------------------------------------------------------
# nearest neighbours
# --------------------------------------------------------------------------

def nearest(query, target, chunk=2000):
    """Index of, and distance to, the closest target point for every query point.

    Brute force, in chunks so the distance matrix never has to exist all at
    once.  With a few thousand points per cloud this is a handful of
    milliseconds and beats the setup cost of a tree; production ICP on
    100k-point clouds uses a k-d tree instead, which is the same idea with
    the search space cut in half at every level.
    """
    idx = np.empty(len(query), dtype=np.int64)
    dst = np.empty(len(query))
    t = target.astype(np.float32)
    t2 = (t * t).sum(1)
    for i in range(0, len(query), chunk):
        q = query[i:i + chunk].astype(np.float32)
        # |q - t|^2 = |q|^2 - 2 q.t + |t|^2; the |q|^2 term does not affect
        # the argmin, so it is left out
        d = t2[None, :] - 2.0 * (q @ t.T)
        j = np.argmin(d, axis=1)
        idx[i:i + chunk] = j
        dst[i:i + chunk] = np.linalg.norm(query[i:i + chunk] - target[j], axis=1)
    return idx, dst


def voxel_downsample(pts, size, extra=None):
    """Keep one point per cube of side `size` (the mean of the points in it).

    Two reasons this matters more than "it is faster": it makes the point
    DENSITY uniform, and ICP silently weights its answer by density, so a
    cloud with 50,000 points on a nearby wall and 500 on a far one is really
    an alignment of the near wall with the far one along for the ride.
    """
    key = np.floor(pts / size).astype(np.int64)
    key -= key.min(axis=0)
    span = key.max(axis=0) + 1
    # Fold the three cell indices into one integer, so the grouping is a
    # plain 1-D unique.  (`np.unique(..., axis=0)` also works but its
    # inverse-index shape has changed between NumPy versions -- this is one
    # line and cannot surprise you.)
    flat = (key[:, 0] * span[1] + key[:, 1]) * span[2] + key[:, 2]
    _, inv = np.unique(flat, return_inverse=True)
    inv = inv.reshape(-1)
    n = inv.max() + 1
    out = np.zeros((n, 3))
    cnt = np.zeros(n)
    np.add.at(out, inv, pts)
    np.add.at(cnt, inv, 1.0)
    out /= cnt[:, None]
    if extra is None:
        return out
    ex = np.zeros((n, extra.shape[1]))
    np.add.at(ex, inv, extra)
    return out, ex / cnt[:, None]


def estimate_normals(pts, k=16):
    """Surface normal at every point, from the local plane through its k
    nearest neighbours.

    The normal is the direction in which those neighbours vary LEAST -- the
    smallest eigenvector of their covariance.  On a flat patch the points
    spread out in two directions and not at all in the third, and the third
    is the normal.
    """
    idx, _ = _knn(pts, k)
    nb = pts[idx]                                   # (N, k, 3)
    nb = nb - nb.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb) / k
    w, v = np.linalg.eigh(cov)
    nrm = v[:, :, 0]                                # smallest eigenvalue first
    # planarity: how flat the neighbourhood is, in [0, 1].  Points on an edge
    # or in a noisy corner get a low score and can be dropped.
    plan = 1.0 - w[:, 0] / np.maximum(w[:, 2], 1e-12)
    return nrm, plan


def _knn(pts, k, chunk=1500):
    out = np.empty((len(pts), k), dtype=np.int64)
    p = pts.astype(np.float32)
    p2 = (p * p).sum(1)
    for i in range(0, len(pts), chunk):
        q = p[i:i + chunk]
        d = p2[None, :] - 2.0 * (q @ p.T)
        out[i:i + chunk] = np.argpartition(d, k, axis=1)[:, :k]
    return out, None


# --------------------------------------------------------------------------
# the two ways to solve one step
# --------------------------------------------------------------------------

def kabsch(P, Q, w=None):
    """The rotation and translation that best map points P onto points Q.

    Closed form, via the SVD -- this is the Kabsch algorithm (Wolfgang
    Kabsch, 1976), also called the orthogonal Procrustes solution.
    "Procrustes" is the Greek innkeeper who stretched or trimmed guests to
    fit his bed; here the fit is restricted to rotation, so nothing gets
    stretched.
    """
    if w is None:
        w = np.ones(len(P))
    w = w / w.sum()
    cp = (P * w[:, None]).sum(0)
    cq = (Q * w[:, None]).sum(0)
    H = ((P - cp) * w[:, None]).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def point_to_plane_step(P, Q, N, w=None):
    """One point-to-plane step, plus how well-determined it was.

    Point-to-point asks each source point to land ON its matched target
    point.  Point-to-plane only asks it to land on the target's local
    SURFACE -- it is free to slide along that surface.  That is the right
    thing to ask for, because two scans of a flat wall taken from different
    places have no reason to sample the same physical points, and forcing
    them together fights the geometry instead of using it.

    Returns (R, t, eigenvalues of the 6x6 system).  The smallest eigenvalue
    is the honest observability warning: near zero means some motion of the
    cloud changes nothing, so that motion is not determined by the data.
    """
    if w is None:
        w = np.ones(len(P))
    A = np.concatenate([np.cross(P, N), N], axis=1)         # (M, 6)
    b = -np.einsum("ij,ij->i", P - Q, N)
    Aw = A * w[:, None]
    H = Aw.T @ A
    g = Aw.T @ b
    ev = np.linalg.eigvalsh(H)
    # H can be singular -- that IS the degenerate case experiment 6 is about,
    # and it also happens transiently when a diverging run matches everything
    # to a handful of points.  A least-squares solve returns the
    # minimum-norm step instead of raising, which is the right behaviour:
    # move only in the directions the data actually constrains.
    x = np.linalg.lstsq(H + 1e-9 * max(float(ev[-1]), 1e-12) * np.eye(6), g,
                        rcond=None)[0]
    om, t = x[:3], x[3:]
    th = np.linalg.norm(om)
    if th < 1e-12:
        R = np.eye(3)
    else:
        k = om / th
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    return R, t, ev


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def icp(src, dst, R0=None, t0=None, mode="point", dst_normals=None,
        max_iter=50, tol=1e-6, trim=1.0, max_dist=np.inf, history=False):
    """Align `src` onto `dst`.

    mode  : "point" (point-to-point) or "plane" (point-to-plane)
    trim  : keep only this fraction of pairs, the closest ones.  Trimmed ICP
            is the standard defence against partial overlap: a source point
            with no true partner produces a long, wrong pair, and dropping
            the worst pairs drops exactly those.
    max_dist : also reject any pair longer than this.

    Returns dict with R, t, rms (final mean pair distance), iters, and the
    smallest eigenvalue of the last point-to-plane system if mode="plane".
    """
    R = np.eye(3) if R0 is None else np.array(R0, float)
    t = np.zeros(3) if t0 is None else np.array(t0, float)
    if mode == "plane" and dst_normals is None:
        dst_normals, _ = estimate_normals(dst)

    prev = np.inf
    hist = []
    ev = None
    it = 0
    for it in range(1, max_iter + 1):
        P = src @ R.T + t
        idx, d = nearest(P, dst)
        keep = np.ones(len(P), bool)
        if trim < 1.0:
            thr = np.quantile(d, trim)
            keep &= d <= thr
        if np.isfinite(max_dist):
            keep &= d <= max_dist
        if keep.sum() < 10:
            break
        Pk, Qk = P[keep], dst[idx[keep]]
        if mode == "point":
            dR, dt = kabsch(Pk, Qk)
        else:
            dR, dt, ev = point_to_plane_step(Pk, Qk, dst_normals[idx[keep]])
        R = dR @ R
        t = dR @ t + dt
        rms = float(np.sqrt((d[keep] ** 2).mean()))
        if history:
            hist.append((it, rms, R.copy(), t.copy()))
        if abs(prev - rms) < tol:
            break
        prev = rms

    P = src @ R.T + t
    _, d = nearest(P, dst)
    out = dict(R=R, t=t, rms=float(np.sqrt((np.sort(d)[:int(len(d) * trim)] ** 2).mean())),
               iters=it, min_eig=(float(ev[0]) if ev is not None else None),
               cond=(float(ev[-1] / ev[0]) if (ev is not None and ev[0] > 1e-300)
                     else (float("inf") if ev is not None else None)))
    if history:
        out["history"] = hist
    return out


def pose_error(R, t, R_true, t_true):
    """Rotation error in degrees, translation error in millimetres."""
    dR = R_true.T @ R
    c = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c))), float(np.linalg.norm(t - t_true) * 1000)
