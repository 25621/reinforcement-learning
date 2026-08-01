"""Three small objects, a triangle rasterizer, and the two pose metrics.

Why build meshes and a rasterizer instead of using the plane renderer of
project 16?  Because pose estimation needs objects with a definite shape and
a definite 3D extent -- the metric (ADD) is an average over the object's own
points, so "the object" has to be a real set of 3D points, not a plane.

The three objects are chosen for what their SYMMETRY does to the problem:

    block     no symmetry at all              -> pose is fully determined
    cylinder  continuous symmetry about z     -> its rotation about that axis
                                                 is not observable from any
                                                 image, ever
    mug       a cylinder with a handle        -> the handle breaks the
                                                 symmetry, so pose IS
                                                 determined -- but only when
                                                 the handle is visible
"""

import numpy as np


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------

def box_mesh(sx=0.06, sy=0.045, sz=0.030):
    x, y, z = sx / 2, sy / 2, sz / 2
    v = np.array([[-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
                  [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]], float)
    f = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3]])
    return v, f


def cylinder_mesh(r=0.030, h=0.085, n=22, cap=True):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    top = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, h / 2)], axis=1)
    bot = np.stack([r * np.cos(a), r * np.sin(a), np.full(n, -h / 2)], axis=1)
    v = np.concatenate([top, bot, [[0, 0, h / 2], [0, 0, -h / 2]]])
    f = []
    for i in range(n):
        j = (i + 1) % n
        f += [[i, n + i, n + j], [i, n + j, j]]
        if cap:
            f += [[2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return v, np.array(f)


def mug_mesh(r=0.030, h=0.085, n=22):
    """A cylinder plus a handle: a half-ring of small quads on the +x side."""
    v, f = cylinder_mesh(r, h, n)
    v = list(v)
    f = list(f)
    base = len(v)
    m, rr, tube = 10, 0.024, 0.007
    for i in range(m + 1):
        th = np.pi * (i / m) - np.pi / 2
        cx, cz = r * 0.85 + rr * np.cos(th), rr * np.sin(th)
        for s in (-1, 1):
            v.append([cx, s * tube, cz])
    for i in range(m):
        a0, a1 = base + 2 * i, base + 2 * i + 1
        b0, b1 = base + 2 * i + 2, base + 2 * i + 3
        f += [[a0, b0, b1], [a0, b1, a1]]
    return np.array(v, float), np.array(f)


def ell_mesh(a=(0.075, 0.028, 0.026), b=(0.026, 0.055, 0.026)):
    """Two boxes joined into an L.  This is the only object here with NO
    symmetry at all -- every rotation of it looks different.

    It exists because of what experiment 3 discovers: a plain box is not
    asymmetric.  A box with three different side lengths still looks
    identical after a 180-degree turn about any of its three axes, so it has
    FOUR indistinguishable poses, and a network trained to regress one of
    them with a squared-error loss learns nothing at all.  The L breaks all
    four.
    """
    v1, f1 = box_mesh(*a)
    v2, f2 = box_mesh(*b)
    off = np.array([a[0] / 2 - b[0] / 2, b[1] / 2 + a[1] / 2, 0.0])
    v = np.concatenate([v1, v2 + off])
    f = np.concatenate([f1, f2 + len(v1)])
    return v - v.mean(0), f


OBJECTS = {"block": box_mesh, "cylinder": cylinder_mesh, "mug": mug_mesh,
           "ell": ell_mesh}
SYMMETRIC = {"block": True, "cylinder": True, "mug": False, "ell": False}


def model_points(verts, n=256, seed=0):
    """A fixed cloud of points ON the object, used by the metrics.

    ADD is an average over the object's own points, so the metric depends on
    which points you pick.  Everyone uses the same fixed sample for a given
    object, and so do we.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(verts), min(n, len(verts)), replace=len(verts) < n)
    return verts[idx]


# --------------------------------------------------------------------------
# rasterizer
# --------------------------------------------------------------------------

def render_mesh(verts, faces, R, t, cam, bg=None, light=(0.3, -0.5, -0.8),
                color=(210, 175, 120), rng=None):
    """Render one posed mesh with a z-buffer and flat Lambert shading.

    One loop over triangles; each triangle only touches the pixels inside its
    own bounding box, so the cost follows the object's size on screen rather
    than the image size.
    """
    P = verts @ R.T + t
    uv = cam.project(P)
    H, W = cam.height, cam.width
    img = (np.zeros((H, W, 3), np.float32) if bg is None else bg.astype(np.float32).copy())
    zbuf = np.full((H, W), np.inf, np.float32)
    mask = np.zeros((H, W), bool)

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
        # No back-face culling.  It is only ever an optimization, and it is
        # wrong here: the winding of the procedurally generated end caps is
        # not guaranteed, so culling silently deleted the top face of every
        # cylinder and left a "mug" that was 288 pixels of handle.  The
        # z-buffer already keeps the nearest surface, which is the part that
        # actually matters.
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
        # Back faces are already culled, so |n . light| is safe here and
        # gives far more contrast than clamping at zero -- a surface facing
        # away from the light still has to be visible in the image, or the
        # network has nothing to learn the pose from.
        shade = 0.25 + 0.75 * abs(float(n @ light))
        sub[better] = z[better]
        img[y0:y1, x0:x1][better] = col * shade
        mask[y0:y1, x0:x1][better] = True

    if rng is not None:
        img = img + rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8), mask, zbuf


# --------------------------------------------------------------------------
# the metrics
# --------------------------------------------------------------------------

def add(points, R_pred, t_pred, R_true, t_true):
    """ADD -- Average Distance of model points.

    Move the object's own points by the predicted pose and by the true pose,
    then average how far each point ended up from where it should have been.
    This is the right way to score a pose because it answers the question the
    robot cares about ("how far off is the object's surface?") instead of
    mixing degrees and millimetres into one meaningless number.
    """
    a = points @ R_pred.T + t_pred
    b = points @ R_true.T + t_true
    return float(np.linalg.norm(a - b, axis=1).mean())


def add_s(points, R_pred, t_pred, R_true, t_true):
    """ADD-S -- the same, but each point is matched to its CLOSEST partner.

    The "S" is for Symmetric.  A featureless cylinder rotated 90 degrees about
    its axis is the *same object in the same place*; plain ADD would call that
    a large error, because it compares point i with point i.  ADD-S compares
    point i with whichever point is nearest, so an indistinguishable rotation
    scores zero -- which is the honest answer.
    """
    a = points @ R_pred.T + t_pred
    b = points @ R_true.T + t_true
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d.min(axis=1)).mean())


def diameter(points):
    """Largest distance between any two model points.  The standard success
    threshold for ADD is 10% of this."""
    d = ((points[:, None, :] - points[None, :, :]) ** 2).sum(-1)
    return float(np.sqrt(d.max()))
