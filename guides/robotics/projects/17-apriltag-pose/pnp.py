"""Pose from a planar target -- and the second answer nobody warns you about.

"PnP" is short for **Perspective-n-Point**: given n points whose 3D positions
you know (here the four corners of a tag, in the tag's own frame) and where
they landed in the image, recover the camera-to-object pose.  "Perspective"
because the projection is a perspective one -- distant things look smaller --
which is exactly what makes the problem solvable at all: an orthographic
camera would give the same picture at every distance.

For a PLANAR target the problem has, in general, **two** solutions that
reproject almost equally well.  This is not a bug in any solver; it is a
property of the geometry.  A tag tilted 20 degrees toward you and the same
tag tilted 20 degrees away from you produce nearly the same quadrilateral,
and the nearer the tag is to facing you square-on, the more nearly identical
those two quadrilaterals become.  This module returns BOTH, so the projects
can measure how often the better-fitting one is the wrong one.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "16-camera-calibration"))

from calib import homography_dlt, extrinsics_from_H, solve_pose      # noqa: E402
from camera import rodrigues                                          # noqa: E402


def _mirror_pose(R, t):
    """The twin of a planar pose: the same tag tilted the other way.

    Reflect the tag's normal about the line of sight.  Geometrically: keep
    the tag where it is and where it points *along* the viewing ray, but flip
    which way it leans.  That lands exactly in the basin of the second
    minimum, so a local optimizer started here finds it.
    """
    n = R[:, 2]                                   # tag normal in camera frame
    v = t / np.linalg.norm(t)                     # direction to the tag
    n2 = 2.0 * np.dot(n, v) * v - n               # mirror n about v
    axis = np.cross(n, n2)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return R.copy(), t.copy()
    axis = axis / s
    ang = np.arctan2(s, float(np.dot(n, n2)))
    return rodrigues(axis * ang) @ R, t.copy()


def pnp_planar(cam, obj, uv, both=True):
    """Pose of a planar target from its image points.

    Returns a list of (R, t, rms_pixels), best fit first.  With `both=True`
    the list has two entries -- the second is the ambiguous twin.
    """
    obj = np.asarray(obj, float).reshape(-1, 3)
    uv = np.asarray(uv, float).reshape(-1, 2)

    # Work in IDEAL normalized coordinates: undo K and the lens distortion
    # first, so the homography step sees a clean pinhole camera.  Skipping
    # this is the single most common way to get a subtly wrong tag pose.
    xy = cam.pixel_to_normalized(uv)
    H = homography_dlt(obj[:, :2], xy)
    R, t = extrinsics_from_H(np.eye(3), H)

    sols = [solve_pose(cam, obj, uv, R, t)]
    if both:
        R2, t2 = _mirror_pose(R, t)
        sols.append(solve_pose(cam, obj, uv, R2, t2))
        # if the optimizer slid back into the same minimum, say so honestly
        if np.linalg.norm(sols[0][0] - sols[1][0]) < 1e-6:
            sols = [sols[0]]
    return sorted(sols, key=lambda s: s[2])


def pnp_multi(cam, objs, dets, init=None):
    """One pose for a rigid plate carrying several tags.

    `objs` and `dets` are dicts id -> (4,3) object points / (4,2) pixels, all
    expressed in the SAME plate frame.  Stacking every visible corner into
    one solve is what kills the ambiguity: the tags sit at different offsets
    on the plate, so the wrong tilt can no longer explain all of them at once.
    """
    ids = sorted(set(objs) & set(dets))
    if not ids:
        return None
    O = np.concatenate([objs[i] for i in ids])
    U = np.concatenate([dets[i] for i in ids])
    if init is None:
        first = pnp_planar(cam, objs[ids[0]], dets[ids[0]])[0]
        init = (first[0], first[1])
    return solve_pose(cam, O, U, init[0], init[1])
