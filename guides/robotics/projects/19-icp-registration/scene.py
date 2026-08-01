"""The scene that gets scanned, and how a depth image becomes a point cloud.

ICP needs geometry that varies in three directions.  A single wall is the
classic counter-example: two scans of it can slide past each other freely
and every alignment looks equally good.  So the scene here has a floor, a
backdrop and three boxes at three different angles -- and experiment 6
deliberately scans a bare wall to show what happens when that variety is
missing.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "16-camera-calibration"))

from camera import default_camera, look_at, invert_pose, rodrigues       # noqa: E402
from render import Plane, render, speckle_texture                        # noqa: E402


def _tex(seed):
    return speckle_texture(160, 160, seed=seed, scale=4, lo=70, hi=220)


def box(center, size, yaw=0.0, seed=0):
    """Five faces of an axis-aligned box (the bottom is never visible)."""
    c = np.asarray(center, float)
    sx, sy, sz = np.asarray(size, float) / 2
    R = rodrigues([0, 0, np.radians(yaw)])
    faces = [
        # (origin offset, edge1, edge2)
        ([-sx, -sy, sz], [2 * sx, 0, 0], [0, 2 * sy, 0]),      # top
        ([-sx, -sy, -sz], [2 * sx, 0, 0], [0, 0, 2 * sz]),     # -y
        ([-sx, sy, -sz], [2 * sx, 0, 0], [0, 0, 2 * sz]),      # +y
        ([-sx, -sy, -sz], [0, 2 * sy, 0], [0, 0, 2 * sz]),     # -x
        ([sx, -sy, -sz], [0, 2 * sy, 0], [0, 0, 2 * sz]),      # +x
    ]
    out = []
    for i, (o, e1, e2) in enumerate(faces):
        out.append(Plane(c + R @ np.asarray(o, float), R @ np.asarray(e1, float),
                         R @ np.asarray(e2, float), _tex(seed + i), "box"))
    return out


def room():
    """A tabletop: floor, one backdrop wall, three boxes.  World z is up.

    Both scan viewpoints sit at y < -1.2, i.e. in front of the backdrop and
    above the floor, so neither camera is ever inside a surface.  (An earlier
    version put a wall between the camera and the scene and produced two
    scans with almost nothing in common -- ICP cannot align what does not
    overlap, and it fails without saying so.)
    """
    planes = [
        Plane([-1.5, -1.5, 0.0], [3.0, 0, 0], [0, 3.0, 0], _tex(1), "floor"),
        Plane([-1.5, 1.2, 0.0], [3.0, 0, 0], [0, 0, 1.4], _tex(2), "backdrop"),
    ]
    planes += box([0.10, 0.35, 0.15], [0.30, 0.30, 0.30], yaw=20, seed=10)
    planes += box([0.62, -0.25, 0.11], [0.22, 0.44, 0.22], yaw=-35, seed=20)
    planes += box([-0.35, -0.05, 0.09], [0.40, 0.24, 0.18], yaw=55, seed=30)
    return planes


VIEW_A = (0.55, -1.35, 1.05)
VIEW_B = (0.28, -1.42, 1.14)      # about 12 deg and 30 cm away from VIEW_A
LOOK_AT = (0.05, 0.05, 0.12)


def bare_wall():
    """One flat, richly textured wall -- geometrically degenerate for ICP."""
    return [Plane([-1.4, 1.4, -0.2], [2.8, 0, 0], [0, 0, 2.0], _tex(7), "wall")]


def scan(planes, eye, target=LOOK_AT, cam=None, supersample=1,
         noise_m=0.0, rng=None, step=3):
    """Take a depth scan from `eye`, return points in the CAMERA's own frame
    plus the camera pose that produced them.

    Points come back in the camera frame because that is what a real sensor
    gives you: a scan knows nothing about where it was taken.  Recovering the
    transform between two such frames is exactly the job of registration.
    """
    cam = cam or default_camera()
    R_wc, t_wc = look_at(np.asarray(eye, float), np.asarray(target, float))
    _, depth, ids = render(cam, planes, R_wc, t_wc, supersample=supersample,
                           background=(0, 0, 0))
    H, W = depth.shape
    vv, uu = np.mgrid[0:H:step, 0:W:step]
    z = depth[::step, ::step]
    ok = np.isfinite(z)
    uv = np.stack([uu[ok], vv[ok]], axis=1).astype(float)
    xn = cam.pixel_to_normalized(uv)
    zz = z[ok][:, None]
    if noise_m > 0:
        rng = rng or np.random.default_rng(0)
        # depth sensors are noisy ALONG THE RAY, and (as project 18 measured)
        # the noise grows with the square of the distance
        zz = zz * (1.0 + rng.normal(0, noise_m, zz.shape) * zz)
    pts = np.concatenate([xn * zz, zz], axis=1)
    return pts, R_wc, t_wc


def relative_pose(R_a, t_a, R_b, t_b):
    """The transform that maps a point expressed in scan A's camera frame
    into scan B's camera frame."""
    R_ba = R_b.T @ R_a
    t_ba = R_b.T @ (t_a - t_b)
    return R_ba, t_ba
