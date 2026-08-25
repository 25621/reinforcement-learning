"""A tabletop of coloured objects, rendered top-down with colour and depth.

Top-down is not laziness -- it is the setup the whole project assumes.  A
gripper coming straight down onto a table only needs to choose *where* in the
image to close and *how* to rotate the wrist, so the grasp is three numbers
(x, y, angle) plus a width, instead of a full 6-DoF pose.  That reduction is
what makes "top-down grasping" the standard first bin-picking system.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for _p in ("16-camera-calibration", "22-object-6-dof-pose"):
    sys.path.insert(0, os.path.join(_PROJ, _p))

from camera import Camera, rodrigues                                    # noqa: E402
from mesh import box_mesh, cylinder_mesh, mug_mesh                      # noqa: E402

CAM = Camera(fx=520.0, fy=520.0, cx=159.5, cy=159.5, dist=(0, 0, 0, 0, 0),
             width=320, height=320)
TABLE_Z = 0.60                        # camera is 60 cm above the table
GRIPPER_MAX_WIDTH = 0.075             # metres between the open fingers

# name -> (mesh maker, colour name, RGB, height above the table of its centre)
CATALOG = {
    "mug": (lambda: mug_mesh(r=0.032, h=0.080), (205, 60, 45)),
    "box": (lambda: box_mesh(0.075, 0.050, 0.045), (55, 95, 200)),
    "can": (lambda: cylinder_mesh(r=0.026, h=0.098), (60, 160, 85)),
    "block": (lambda: box_mesh(0.042, 0.042, 0.042), (225, 180, 45)),
    "bar": (lambda: box_mesh(0.115, 0.022, 0.022), (150, 90, 190)),
}
COLOR_NAME = {"mug": "red", "box": "blue", "can": "green",
              "block": "yellow", "bar": "purple"}


def _rasterize(verts, faces, R, t, cam, img, zbuf, idbuf, obj_id, color,
               light=(0.25, -0.35, -0.9)):
    """Same z-buffer rasterizer as project 22, but writing into buffers that
    are shared between several objects and a table, so occlusion works."""
    P = verts @ R.T + t
    uv = cam.project(P)
    H, W = cam.height, cam.width
    light = np.asarray(light, float)
    light = light / np.linalg.norm(light)
    col = np.asarray(color, float)
    for tri in faces:
        p = P[tri]
        if np.any(p[:, 2] <= 1e-4):
            continue
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        # No back-face culling -- see the note in project 22's mesh.py.  The
        # z-buffer already keeps the nearest surface, and culling by winding
        # deleted the top face of every procedurally generated cylinder.
        a = uv[tri]
        x0 = max(int(np.floor(a[:, 0].min())), 0)
        x1 = min(int(np.ceil(a[:, 0].max())) + 1, W)
        y0 = max(int(np.floor(a[:, 1].min())), 0)
        y1 = min(int(np.ceil(a[:, 1].max())) + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        d = ((a[1, 1] - a[2, 1]) * (a[0, 0] - a[2, 0]) +
             (a[2, 0] - a[1, 0]) * (a[0, 1] - a[2, 1]))
        if abs(d) < 1e-9:
            continue
        l0 = ((a[1, 1] - a[2, 1]) * (xs - a[2, 0]) + (a[2, 0] - a[1, 0]) * (ys - a[2, 1])) / d
        l1 = ((a[2, 1] - a[0, 1]) * (xs - a[2, 0]) + (a[0, 0] - a[2, 0]) * (ys - a[2, 1])) / d
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        z = l0 * p[0, 2] + l1 * p[1, 2] + l2 * p[2, 2]
        sub = zbuf[y0:y1, x0:x1]
        better = inside & (z < sub)
        if not better.any():
            continue
        shade = 0.35 + 0.65 * abs(float(n @ light))
        sub[better] = z[better]
        img[y0:y1, x0:x1][better] = col * shade
        idbuf[y0:y1, x0:x1][better] = obj_id


def render_scene(items, cam=CAM, rng=None, table_color=(150, 145, 138),
                 clutter_noise=6.0):
    """`items` is a list of (name, x, y, yaw_deg).  Returns rgb, depth, ids.

    ids: 0 = table, 1..n = the objects in the order given, -1 = nothing.
    """
    H, W = cam.height, cam.width
    img = np.zeros((H, W, 3), np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    ids = np.full((H, W), -1, np.int32)

    # the table, as a plane at z = TABLE_Z in the camera frame
    vv, uu = np.mgrid[0:H, 0:W]
    xn = cam.pixel_to_normalized(np.stack([uu.reshape(-1), vv.reshape(-1)], 1))
    zbuf[:] = TABLE_Z
    ids[:] = 0
    rng = rng or np.random.default_rng(0)
    grain = rng.normal(0, 4.0, (H, W, 1))
    img[:] = np.asarray(table_color, float) + grain

    for k, (name, x, y, yaw) in enumerate(items, start=1):
        maker, color = CATALOG[name]
        v, f = maker()
        half_h = (v[:, 2].max() - v[:, 2].min()) / 2
        R = rodrigues([0, 0, np.radians(yaw)])
        # the object sits ON the table, so its centre is half its height up,
        # i.e. half its height CLOSER to a camera looking down
        t = np.array([x, y, TABLE_Z - half_h])
        _rasterize(v, f, R, t, cam, img, zbuf, ids, k, color)

    if clutter_noise:
        img = img + rng.normal(0, clutter_noise, img.shape)
    depth = zbuf.copy()
    return np.clip(img, 0, 255).astype(np.uint8), depth, ids


def random_scene(rng, n=4, names=None):
    """n objects placed without overlapping, at random angles."""
    names = names or list(CATALOG)
    chosen = list(rng.choice(names, size=min(n, len(names)), replace=False))
    placed = []
    for name in chosen:
        for _ in range(200):
            x = rng.uniform(-0.075, 0.075)
            y = rng.uniform(-0.075, 0.075)
            if all((x - px) ** 2 + (y - py) ** 2 > 0.075 ** 2 for px, py, _ in placed):
                placed.append((x, y, name))
                break
    return [(n_, x, y, float(rng.uniform(0, 180))) for x, y, n_ in placed]


def deproject(depth, cam=CAM):
    """Depth image -> 3D point per pixel, in the camera frame."""
    H, W = depth.shape
    vv, uu = np.mgrid[0:H, 0:W]
    xn = cam.pixel_to_normalized(np.stack([uu.reshape(-1), vv.reshape(-1)], 1))
    z = depth.reshape(-1, 1)
    return np.concatenate([xn * z, z], axis=1).reshape(H, W, 3)
