"""A simulated eye-in-hand calibration rig.

Everything a real calibration session has, except the robot:

* a 7-DoF arm (project 02's ``arm7``) with a camera bolted somewhere near the
  tool -- the offset ``X_TRUE`` that the solver is supposed to rediscover
* a square fiducial tag lying on a table
* a pinhole camera that PROJECTS the tag's four corners into pixels, with
  Gaussian noise added in PIXELS (which is where the noise really lives) rather
  than in the pose
* ``cv2.solvePnP`` to turn those noisy pixels back into a measured tag pose,
  exactly as an AprilTag pipeline would

Doing it this way matters.  Adding noise directly to the tag POSE would give
every direction the same error; real pixel noise does not -- the distance to
the tag is far less certain than its sideways position, because a small change
in depth barely changes the picture.  That asymmetry is a real property of
cameras and it survives into the calibration result.
"""

import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in ("01-transform-calculator", "02-urdf-visualizer", "03-forward-kinematics-from-scratch",
             "04-jacobian-from-scratch", "05-damped-least-squares-ik"):
    sys.path.insert(0, os.path.join(HERE, "..", _rel))

import transforms as tf  # noqa: E402
from fk import fk_all  # noqa: E402
from ik import ik  # noqa: E402

# ---- the unknown we are trying to recover ---------------------------------
# A camera bracket: 4 cm forward, 3 cm to one side, 6 cm up from the tool
# frame, rotated 45 degrees about the mount and tipped slightly.
X_TRUE = tf.T_from_Rp(tf.rpy_to_R([0.05, -0.10, 0.7854]), np.array([0.040, -0.030, 0.060]))

# ---- the world -------------------------------------------------------------
TAG_SIZE = 0.08  # metres, edge to edge
T_BASE_TAG = tf.T_from_Rp(tf.rpy_to_R([0.0, 0.0, 0.35]), np.array([0.45, 0.02, 0.04]))

# ---- the camera ------------------------------------------------------------
W, H = 640, 480
K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
DIST = np.zeros(5)

# Tag corners in the tag's own frame (planar, z = 0), in the order solvePnP wants.
_s = TAG_SIZE / 2
TAG_CORNERS = np.array([[-_s, _s, 0], [_s, _s, 0], [_s, -_s, 0], [-_s, -_s, 0]], dtype=np.float64)


def look_at(eye, target, roll=0.0):
    """A camera pose whose +z axis points from ``eye`` at ``target``.

    ``roll`` spins the camera about its own line of sight.  It changes nothing
    about what is visible, and everything about how well the calibration is
    conditioned -- see the axis-diversity experiment.
    """
    z = np.asarray(target, float) - np.asarray(eye, float)
    z = z / np.linalg.norm(z)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, z)) > 0.98:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z]) @ tf.Rz(roll)
    return tf.T_from_Rp(R, eye)


def project_tag(T_cam_tag, rng=None, pixel_noise=0.0):
    """Project the four tag corners into the image; optionally add pixel noise."""
    rvec, _ = cv2.Rodrigues(T_cam_tag[:3, :3])
    tvec = T_cam_tag[:3, 3].reshape(3, 1)
    px, _ = cv2.projectPoints(TAG_CORNERS, rvec, tvec, K, DIST)
    px = px.reshape(-1, 2)
    if pixel_noise > 0 and rng is not None:
        px = px + rng.normal(0.0, pixel_noise, px.shape)
    return px


def detect_tag(px):
    """Pixels -> measured tag pose, the way an AprilTag pipeline does it."""
    ok, rvec, tvec = cv2.solvePnP(TAG_CORNERS, px.astype(np.float64), K, DIST,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return tf.T_from_Rp(R, tvec.flatten())


def in_view(px, margin=12):
    return bool(np.all(px[:, 0] > margin) and np.all(px[:, 0] < W - margin)
                and np.all(px[:, 1] > margin) and np.all(px[:, 1] < H - margin))


# ---------------------------------------------------------------------------
# generating a calibration session
# ---------------------------------------------------------------------------
def _try_pose(robot, T_base_cam, q_seed):
    """Turn a wanted CAMERA pose into a reachable ARM pose, or return None."""
    T_base_ee = T_base_cam @ tf.T_inv(X_TRUE)
    q, info = ik(robot, q_seed, T_base_ee, lam=1e-2, max_iters=300, clamp_limits=True)
    if not info["ok"]:
        return None
    T_ee = fk_all(robot, q)["tool0"]
    T_cam_tag = tf.T_inv(T_ee @ X_TRUE) @ T_BASE_TAG
    if T_cam_tag[2, 3] < 0.15:  # behind or too close to the lens
        return None
    px = project_tag(T_cam_tag)
    if not in_view(px):
        return None
    return q, T_ee, T_cam_tag


def collect(robot, n, rng, degenerate=False, q_seed=None, max_tries=400):
    """Drive the arm to ``n`` viewpoints that all see the tag.

    ``degenerate=True`` produces the failure case on purpose: every viewpoint
    sits on one circle at a fixed height and tilt, so the arm only ever turns
    about the world's vertical axis.
    """
    if q_seed is None:
        q_seed = np.array([0.0, 0.90, 0.0, 2.00, 0.0, -1.00, 0.0])
    target = T_BASE_TAG[:3, 3]
    qs, T_ees, T_cts = [], [], []
    tries = 0
    while len(qs) < n and tries < max_tries:
        tries += 1
        if degenerate:
            az = rng.uniform(0, 2 * np.pi)
            r, h, roll = 0.22, 0.42, 0.0  # fixed radius, height and roll
        else:
            az = rng.uniform(0, 2 * np.pi)
            r = rng.uniform(0.05, 0.30)
            h = rng.uniform(0.32, 0.55)
            roll = rng.uniform(-np.pi, np.pi)
        eye = target + np.array([r * np.cos(az), r * np.sin(az), h])
        got = _try_pose(robot, look_at(eye, target, roll), q_seed)
        if got is None:
            continue
        q, T_ee, T_ct = got
        qs.append(q)
        T_ees.append(T_ee)
        T_cts.append(T_ct)
    return qs, T_ees, T_cts


def observe(T_cts, rng, pixel_noise=0.3):
    """Run every true tag pose through project -> add pixel noise -> solvePnP."""
    out, pxs = [], []
    for T_ct in T_cts:
        px = project_tag(T_ct, rng, pixel_noise)
        pxs.append(px)
        m = detect_tag(px)
        out.append(m if m is not None else T_ct)
    return out, pxs
