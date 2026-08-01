"""Zhang's camera calibration, from scratch: DLT homography, the closed-form
intrinsics, and a Levenberg-Marquardt refinement.

The three stages exist for three different reasons, and it is worth knowing
which is doing what:

  1. homography_dlt   -- per view, fit the 3x3 map from board plane to image.
                         Linear, so it needs no starting guess.
  2. intrinsics_zhang -- turn N homographies into ONE camera matrix K in
                         closed form.  Also linear.  It ignores distortion,
                         so its answer is only a starting point.
  3. refine           -- non-linear least squares over EVERYTHING at once
                         (K, distortion, and every board pose), minimizing the
                         real reprojection error in pixels.

Stage 3 is what people quote.  Stages 1-2 exist only to hand stage 3 a start
close enough that it converges -- a non-linear solver launched from a random
guess lands in a nonsense minimum.
"""

import numpy as np

from camera import Camera, rodrigues, rot_to_rvec, orthonormalize


# --------------------------------------------------------------------------
# 1. homography by the Direct Linear Transform
# --------------------------------------------------------------------------

def _normalize_2d(pts):
    """Hartley normalization: shift to the centroid, scale to mean distance
    sqrt(2).  Without it the DLT matrix has columns of wildly different
    magnitude (pixels ~ 500, products of pixels ~ 250000) and the smallest
    singular vector is dominated by round-off rather than by the data."""
    c = pts.mean(axis=0)
    d = np.linalg.norm(pts - c, axis=1).mean()
    s = np.sqrt(2.0) / max(d, 1e-12)
    T = np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1.0]])
    q = (np.hstack([pts, np.ones((len(pts), 1))]) @ T.T)[:, :2]
    return q, T


def homography_dlt(src, dst):
    """Fit H with dst ~ H src, both (N,2).  N >= 4."""
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    s, Ts = _normalize_2d(src)
    d, Td = _normalize_2d(dst)
    n = len(s)
    A = np.zeros((2 * n, 9))
    A[0::2, 0:2] = s
    A[0::2, 2] = 1
    A[0::2, 6:8] = -d[:, 0:1] * s
    A[0::2, 8] = -d[:, 0]
    A[1::2, 3:5] = s
    A[1::2, 5] = 1
    A[1::2, 6:8] = -d[:, 1:2] * s
    A[1::2, 8] = -d[:, 1]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Ts
    return H / H[2, 2]


# --------------------------------------------------------------------------
# 2. Zhang's closed-form intrinsics
# --------------------------------------------------------------------------

def _v_ij(H, i, j):
    """The row of the constraint matrix that encodes h_i^T B h_j."""
    return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[1, i] * H[1, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j],
    ])


def intrinsics_zhang(Hs):
    """N homographies -> K, in closed form.

    The trick: each homography's first two columns are the images of two
    PERPENDICULAR unit vectors on the board, and "perpendicular" plus "equal
    length" are two linear constraints on the symmetric matrix
    B = K^-T K^-1.  Six unknowns in B, two constraints per view, so three
    views suffice in principle.
    """
    V = []
    for H in Hs:
        V.append(_v_ij(H, 0, 1))
        V.append(_v_ij(H, 0, 0) - _v_ij(H, 1, 1))
    V = np.asarray(V)
    _, _, Vt = np.linalg.svd(V)
    b = Vt[-1]
    B11, B12, B22, B13, B23, B33 = b
    den = B11 * B22 - B12 * B12
    if abs(den) < 1e-20:
        raise np.linalg.LinAlgError("degenerate view set (B is singular)")
    v0 = (B12 * B13 - B11 * B23) / den
    lam = B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11
    if lam / B11 <= 0 or lam * B11 / den <= 0:
        raise np.linalg.LinAlgError("degenerate view set (no real focal length)")
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / den)
    gamma = -B12 * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha * alpha / lam
    return np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])


def extrinsics_from_H(K, H):
    """Recover the board pose (R_cb, t_cb) from one homography."""
    Ki = np.linalg.inv(K)
    h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]
    lam = 1.0 / np.linalg.norm(Ki @ h1)
    r1 = lam * (Ki @ h1)
    r2 = lam * (Ki @ h2)
    r3 = np.cross(r1, r2)
    t = lam * (Ki @ h3)
    R = orthonormalize(np.stack([r1, r2, r3], axis=1))
    if t[2] < 0:                                   # board must be IN FRONT
        R = -R
        R[:, 2] *= -1
        t = -t
    return R, t


# --------------------------------------------------------------------------
# 3. non-linear refinement (Levenberg-Marquardt, hand-rolled)
# --------------------------------------------------------------------------

def _unpack(p, n_views, n_dist):
    cam = Camera(p[0], p[1], p[2], p[3], p[4:4 + n_dist])
    poses = []
    for i in range(n_views):
        q = p[4 + n_dist + 6 * i: 4 + n_dist + 6 * i + 6]
        poses.append((rodrigues(q[:3]), q[3:]))
    return cam, poses


def _residuals(p, obj, obs, n_dist):
    cam, poses = _unpack(p, len(obs), n_dist)
    out = []
    for (R, t), uv in zip(poses, obs):
        pred = cam.project(obj @ R.T + t)
        out.append((pred - uv).reshape(-1))
    return np.concatenate(out)


def refine(cam0, poses0, obj, obs, n_dist=5, iters=60, verbose=False):
    """Levenberg-Marquardt over intrinsics + distortion + all board poses.

    LM interpolates between two solvers: Gauss-Newton (fast, but happily
    diverges) and gradient descent (slow, but safe).  The damping term
    `lam * diag(J^T J)` decides which one you get -- raise it after a bad
    step, lower it after a good one.  It is named after Kenneth Levenberg
    and Donald Marquardt, who proposed the damping and the scaling of it.
    """
    p = np.concatenate([[cam0.fx, cam0.fy, cam0.cx, cam0.cy],
                        cam0.dist[:n_dist]] +
                       [np.concatenate([rot_to_rvec(R), t]) for R, t in poses0])
    r = _residuals(p, obj, obs, n_dist)
    cost = float(r @ r)
    lam = 1e-3
    for it in range(iters):
        # numerical Jacobian: one extra projection per parameter.  With ~100
        # parameters that is cheap, and it keeps the code honest -- no chance
        # of a hand-derived derivative silently disagreeing with the model.
        J = np.zeros((r.size, p.size))
        for k in range(p.size):
            step = 1e-6 * max(1.0, abs(p[k]))
            pk = p.copy()
            pk[k] += step
            J[:, k] = (_residuals(pk, obj, obs, n_dist) - r) / step
        JTJ = J.T @ J
        g = J.T @ r
        improved = False
        for _ in range(12):
            A = JTJ + lam * np.diag(np.maximum(np.diag(JTJ), 1e-12))
            try:
                dp = np.linalg.solve(A, -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            rn = _residuals(p + dp, obj, obs, n_dist)
            cn = float(rn @ rn)
            if cn < cost:
                p, r, cost = p + dp, rn, cn
                lam = max(lam * 0.3, 1e-12)
                improved = True
                break
            lam *= 10
        if verbose:
            print(f"    it {it:2d}  rms={np.sqrt(cost / r.size * 2):.5f} px  lam={lam:.1e}")
        if not improved or np.linalg.norm(dp) < 1e-12:
            break
    cam, poses = _unpack(p, len(obs), n_dist)
    return cam, poses, rms_from_residuals(r)


def rms_from_residuals(r):
    """RMS reprojection error in pixels (2 residuals per corner)."""
    return float(np.sqrt((r @ r) / (r.size / 2)))


def solve_pose(cam, obj, uv, R0=None, t0=None, iters=40):
    """Fit ONE board pose with the camera held fixed (6 parameters).

    Needed whenever you want to score a camera on views it was not trained
    on: the board poses of those views are nuisance parameters that must
    still be fitted, but the intrinsics must not move, or the "held-out"
    test quietly re-fits the very thing it was supposed to test.
    """
    if R0 is None:
        H = homography_dlt(obj[:, :2], cam.pixel_to_normalized(uv) *
                           np.array([cam.fx, cam.fy]) + np.array([cam.cx, cam.cy]))
        R0, t0 = extrinsics_from_H(cam.K, H)
    q = np.concatenate([rot_to_rvec(R0), t0])

    def res(qq):
        R = rodrigues(qq[:3])
        return (cam.project(obj @ R.T + qq[3:]) - uv).reshape(-1)

    r = res(q)
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(iters):
        J = np.zeros((r.size, 6))
        for k in range(6):
            step = 1e-7 * max(1.0, abs(q[k]))
            qk = q.copy(); qk[k] += step
            J[:, k] = (res(qk) - r) / step
        JTJ, g = J.T @ J, J.T @ r
        for _ in range(10):
            try:
                dq = np.linalg.solve(JTJ + lam * np.diag(np.maximum(np.diag(JTJ), 1e-12)), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            rn = res(q + dq)
            cn = float(rn @ rn)
            if cn < cost:
                q, r, cost, lam = q + dq, rn, cn, max(lam * 0.3, 1e-12)
                break
            lam *= 10
        else:
            break
    return rodrigues(q[:3]), q[3:], rms_from_residuals(r)


def parameter_std(cam, poses, obj, obs, n_dist=5):
    """One-sigma uncertainty of every intrinsic parameter, in pixels.

    This is the number a reprojection error CANNOT give you.  The residual
    says "the model fits the corners I saw"; this says "given the noise, how
    far could fx be from the value I reported and still fit them just as
    well".  It is the diagonal of  sigma^2 (J^T J)^-1, the standard
    least-squares covariance, where sigma is the residual noise level.

    Returns (std_vector, condition_number).  A huge condition number means
    some direction in parameter space is invisible to the data -- exactly
    the failure the fronto-parallel trap produces.  The condition number is
    computed after scaling every column of J to unit length, otherwise it
    would mostly report that focal lengths are measured in hundreds and
    distortion coefficients in hundredths.
    """
    p = np.concatenate([[cam.fx, cam.fy, cam.cx, cam.cy], cam.dist[:n_dist]] +
                       [np.concatenate([rot_to_rvec(R), t]) for R, t in poses])
    r = _residuals(p, obj, obs, n_dist)
    J = np.zeros((r.size, p.size))
    for k in range(p.size):
        step = 1e-6 * max(1.0, abs(p[k]))
        pk = p.copy()
        pk[k] += step
        J[:, k] = (_residuals(pk, obj, obs, n_dist) - r) / step
    dof = max(r.size - p.size, 1)
    sigma2 = float(r @ r) / dof
    scale = np.maximum(np.linalg.norm(J, axis=0), 1e-300)
    Jn = J / scale
    ev = np.linalg.svd(Jn, compute_uv=False)
    cond = float(ev[0] / max(ev[-1], 1e-300))
    # rcond=0: keep every direction, however weakly the data sees it.  With
    # a larger rcond a pseudo-inverse silently DELETES the unobservable
    # direction and then reports a comfortingly small uncertainty for a
    # parameter the data never constrained.
    cov = (np.linalg.pinv(Jn.T @ Jn, rcond=0.0) / np.outer(scale, scale)) * sigma2
    return np.sqrt(np.maximum(np.diag(cov)[:4 + n_dist], 0.0)), cond


def reprojection_rms(cam, poses, obj, obs):
    return rms_from_residuals(np.concatenate(
        [(cam.project(obj @ R.T + t) - uv).reshape(-1) for (R, t), uv in zip(poses, obs)]))


# --------------------------------------------------------------------------
# the whole pipeline in one call
# --------------------------------------------------------------------------

def calibrate(obj, obs, n_dist=5, iters=60, verbose=False):
    """obj: (M,3) board points with z=0.  obs: list of (M,2) detected corners.

    Returns (Camera, poses, rms_pixels).
    """
    Hs = [homography_dlt(obj[:, :2], uv) for uv in obs]
    K0 = intrinsics_zhang(Hs)
    cam0 = Camera.from_K(K0)
    poses0 = [extrinsics_from_H(K0, H) for H in Hs]
    return refine(cam0, poses0, obj, obs, n_dist=n_dist, iters=iters, verbose=verbose)
