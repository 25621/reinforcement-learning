"""A monocular visual-odometry front end, and the corridor it drives through.

The loop, per frame pair:

    track features  ->  essential matrix  ->  R and a UNIT t  ->  accumulate

The word to notice is UNIT.  From two images of a rigid scene you can recover
which way the camera moved and how it turned, but not how far it went: a
camera that moves 1 m through a room and one that moves 2 m through a room
twice the size produce pixel-for-pixel identical images.  Scale has to come
from somewhere outside the images, and where it comes from is the subject of
half this project.
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "16-camera-calibration"))

from camera import Camera, rodrigues, rot_to_rvec                      # noqa: E402
from render import Plane, render, speckle_texture                      # noqa: E402


# --------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------

def corridor(length=120.0, half_width=1.4, height=1.1):
    """A straight textured corridor along +z: two walls, floor and ceiling.

    Camera convention (from project 16): +z forward, +x right, +y DOWN.  So
    the floor is at +y and the ceiling at -y.
    """
    tex = [speckle_texture(256, 256, seed=s, scale=3, lo=45, hi=235) for s in range(4)]
    rep_z = length / 3.0                    # a fresh patch of texture every 3 m
    return [
        Plane([-half_width, -height, -5], [0, 0, length], [0, 2 * height, 0],
              tex[0], "wall_left", repeat=(rep_z, 1)),
        Plane([half_width, -height, -5], [0, 0, length], [0, 2 * height, 0],
              tex[1], "wall_right", repeat=(rep_z, 1)),
        Plane([-half_width, height, -5], [0, 0, length], [2 * half_width, 0, 0],
              tex[2], "floor", repeat=(rep_z, 1)),
        Plane([-half_width, -height, -5], [0, 0, length], [2 * half_width, 0, 0],
              tex[3], "ceiling", repeat=(rep_z, 1)),
    ]


def trajectory(n_frames, step=0.5, sway=0.25, yaw_amp=4.0, period=28.0,
               speed_var=0.4):
    """A camera driving down the corridor with a gentle weave and a varying speed.

    Straight-line motion at constant speed would be a poor test on two counts.
    With no rotation the rotation half of the estimate is never exercised.  And
    with constant speed, "assume the robot always moves 0.5 m per frame" would
    be exactly right, which would make experiment 2 (where does scale come
    from?) prove nothing.  `speed_var` makes the step length swing between
    0.3 m and 0.7 m, the way a real vehicle does.
    """
    poses = []
    s = 0.0
    for i in range(n_frames):
        x = sway * np.sin(2 * np.pi * s / period)
        yaw = np.radians(yaw_amp) * np.cos(2 * np.pi * s / period)
        R = rodrigues([0, yaw, 0])                     # yaw about the DOWN axis
        t = np.array([x, 0.0, s])
        poses.append((R, t))
        s += step * (1.0 + speed_var * np.sin(2 * np.pi * i / 37.0))
    return poses


def render_sequence(cam, planes, poses, supersample=1, noise=2.0):
    imgs = []
    for k, (R, t) in enumerate(poses):
        img, _, _ = render(cam, planes, R, t, supersample=supersample,
                           seed=k, noise=noise, background=(12, 12, 14))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    return imgs


# --------------------------------------------------------------------------
# the front end
# --------------------------------------------------------------------------

def track(prev_img, img, prev_pts=None, max_corners=600, quality=0.01, min_dist=8):
    """Find corners in the previous image and follow them into this one.

    Lucas-Kanade optical flow assumes a feature's neighbourhood only shifts a
    little between frames, so it is fast but needs small motion; the pyramid
    (track on a shrunken image first, refine on the full one) is what lets it
    survive motions of tens of pixels.
    """
    if prev_pts is None or len(prev_pts) < max_corners // 3:
        prev_pts = cv2.goodFeaturesToTrack(prev_img, max_corners, quality, min_dist)
        if prev_pts is None:
            return None, None
        prev_pts = prev_pts.reshape(-1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(
        prev_img, img, prev_pts.astype(np.float32).reshape(-1, 1, 2), None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    st = st.reshape(-1).astype(bool)
    return prev_pts[st], p1.reshape(-1, 2)[st]


def essential_8pt(cam, p0, p1):
    """The eight-point algorithm, written out: point pairs in, E out.

    Every correspondence gives one linear equation in the nine entries of E
    (x1' E x0 = 0), so eight of them determine E up to scale and the answer
    is the smallest singular vector of the stacked equations.  Two details
    matter and both are in the code below:

    * work in NORMALIZED image coordinates (K and the lens distortion undone
      first).  On raw pixels the numbers span 1 to 500 and their products
      span 1 to 250,000, and the smallest singular vector then describes
      round-off rather than geometry.
    * force the answer to be a legal essential matrix afterwards.  A true E
      has singular values (s, s, 0); the raw least-squares fit does not, so
      we push it to the nearest matrix that does.

    NO outlier rejection here.  That is deliberate -- experiment 3 needs a
    non-robust estimator to compare RANSAC against, and OpenCV does not
    offer one (both of its options are already robust).
    """
    x0 = cam.pixel_to_normalized(p0)
    x1 = cam.pixel_to_normalized(p1)
    A = np.stack([x1[:, 0] * x0[:, 0], x1[:, 0] * x0[:, 1], x1[:, 0],
                  x1[:, 1] * x0[:, 0], x1[:, 1] * x0[:, 1], x1[:, 1],
                  x0[:, 0], x0[:, 1], np.ones(len(x0))], axis=1)
    _, _, Vt = np.linalg.svd(A)
    E = Vt[-1].reshape(3, 3)
    U, s, Vt2 = np.linalg.svd(E)
    m = (s[0] + s[1]) / 2
    return U @ np.diag([m, m, 0.0]) @ Vt2


def ransac_essential(cam, p0, p1, iters=200, thresh=1.5e-3, rng=None):
    """RANSAC around the eight-point solver.

    RANSAC = RANdom SAmple Consensus (Fischler and Bolles, 1981).  Fit the
    model to a small random subset, count how many of ALL the points agree
    with it, repeat, keep the model with the largest agreeing set, and refit
    on that set.  The insight is that a subset containing no outliers gives a
    good model, and it only takes ONE such subset out of many tries -- so
    instead of trying to down-weight bad data you simply hope to miss it.
    """
    rng = rng or np.random.default_rng(0)
    x0 = np.concatenate([cam.pixel_to_normalized(p0), np.ones((len(p0), 1))], axis=1)
    x1 = np.concatenate([cam.pixel_to_normalized(p1), np.ones((len(p1), 1))], axis=1)
    n = len(p0)
    best, best_inl = None, None
    for _ in range(iters):
        sel = rng.choice(n, 8, replace=False)
        try:
            E = essential_8pt(cam, p0[sel], p1[sel])
        except np.linalg.LinAlgError:
            continue
        # Sampson distance: a first-order estimate of how far the pair is
        # from satisfying x1' E x0 = 0, measured in image units rather than
        # in the meaningless units of the raw algebraic residual.
        Ex0 = x0 @ E.T
        Etx1 = x1 @ E
        num = np.einsum("ij,ij->i", x1, Ex0) ** 2
        den = Ex0[:, 0] ** 2 + Ex0[:, 1] ** 2 + Etx1[:, 0] ** 2 + Etx1[:, 1] ** 2
        d = num / np.maximum(den, 1e-12)
        inl = d < thresh ** 2
        if best_inl is None or inl.sum() > best_inl.sum():
            best, best_inl = E, inl
    if best_inl is None or best_inl.sum() < 8:
        return None, None
    return essential_8pt(cam, p0[best_inl], p1[best_inl]), best_inl


def relative_motion(cam, p0, p1, ransac=True, thresh_px=1.0, solver="opencv",
                    rng=None):
    """Recover (R, unit t) between two views from tracked point pairs.

    The essential matrix E encodes the epipolar constraint: a point seen in
    one image must lie on a known LINE in the other, and that line depends
    only on how the camera moved.  Five point pairs are enough to pin E down;
    everything above five is used to outvote the mismatches.

    Returns (R, t_unit, inlier_fraction) or None.
    """
    if len(p0) < 8:
        return None
    if solver == "ours":
        if ransac:
            E, inl = ransac_essential(cam, p0, p1, rng=rng)
            mask = None if inl is None else inl.astype(np.uint8).reshape(-1, 1)
        else:
            E, mask = essential_8pt(cam, p0, p1), None
        # Our E lives in normalized coordinates with the lens distortion
        # already removed, so recoverPose must be handed matching points:
        # undistorted pixels, i.e. normalized coordinates put back through K.
        p0 = cam.pixel_to_normalized(p0) * [cam.fx, cam.fy] + [cam.cx, cam.cy]
        p1 = cam.pixel_to_normalized(p1) * [cam.fx, cam.fy] + [cam.cx, cam.cy]
    else:
        E, mask = cv2.findEssentialMat(p0, p1, cam.K, method=cv2.RANSAC,
                                       prob=0.999, threshold=thresh_px)
    if E is None or np.asarray(E).shape != (3, 3):
        return None
    n, R, t, mask2 = cv2.recoverPose(E, p0, p1, cam.K, mask=mask.copy() if mask is not None else None)
    inl = float(mask2.astype(bool).mean()) if mask2 is not None else 1.0
    return R, t.reshape(3), inl


def integrate(steps, scales):
    """Chain per-step motions into a trajectory.

    `cv2.recoverPose` returns (R, t) describing where the 3D POINTS went:
    x2 = R x1 + t.  The camera moved the other way, so camera 2 sits at
    -R^T t inside camera 1's frame, and the world pose accumulates as

        R_w  <-  R_w R^T           (new orientation)
        p_w  <-  p_w + s R_w_new (-t)

    Getting the direction of this backwards is the classic first bug: the
    trajectory then runs backwards down the corridor while every per-step
    number still looks healthy.  Drift lives here too: every step's error is
    applied to everything that follows it.
    """
    R = np.eye(3)
    p = np.zeros(3)
    out = [(R.copy(), p.copy())]
    for (Rs, ts), s in zip(steps, scales):
        R = R @ Rs.T
        p = p + s * (R @ (-ts))
        out.append((R.copy(), p.copy()))
    return out


def align_yaw_only(est, gt):
    """Trajectory error, quoted the usual way.

    Both trajectories start at the origin with the same orientation here, so
    no alignment is needed -- this just returns the per-frame position error
    and the classic 'drift as a percentage of distance travelled'.
    """
    e = np.array([p for _, p in est])
    g = np.array([p for _, p in gt])
    n = min(len(e), len(g))
    err = np.linalg.norm(e[:n] - g[:n], axis=1)
    dist = np.linalg.norm(np.diff(g[:n], axis=0), axis=1).cumsum()
    dist = np.concatenate([[0.0], dist])
    return err, dist
