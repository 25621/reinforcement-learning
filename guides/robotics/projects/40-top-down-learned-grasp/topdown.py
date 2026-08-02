"""A top-down depth camera over flat objects, and a truth oracle for grasps.

The objects are *prisms*: project 39's polygons, extruded upwards.  That choice
is the whole reason this project can exist on a laptop.  Because the footprint
is a polygon we know exactly, project 39's force-closure test can tell us,
with no simulator and no guessing, whether any proposed grasp would really
hold.  That gives us millions of free training labels -- which is exactly how
Dex-Net was built, only with 3D meshes and a slower analytic model.

Three things live here:

  render      -- a top-down depth image, with the noise a real depth camera has
  label       -- the truth oracle: does this grasp hold?  (uses project 39)
  baselines   -- the two hand-written grasp scorers the learned net competes with
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJ, "39-analytic-2d-grasp"))

from grasp2d import (SHAPES, centroid, cone_generators, force_closure,        # noqa: E402
                     quality)

# ---------------------------------------------------------------------------
# the world
# ---------------------------------------------------------------------------

PX_PER_M = 1000.0        # 1 pixel = 1 mm
IMG = 240                # 24 cm x 24 cm of table
TABLE_Z = 0.60           # camera 60 cm above the table
GRIPPER_MAX = 0.075      # metres between the open fingers
GRIPPER_MIN = 0.008
FINGER_W = 0.010         # each fingertip is a 10 mm x 6 mm pad
FINGER_T = 0.006
MU = 0.4
FN = 20.0                # newtons the gripper squeezes with
DENSITY = 700.0          # kg / m^3, roughly a block of wood
SAFETY = 1.5             # the grasp must beat the object's weight by this much

# The five shapes of project 39 are the TRAINING objects.  The three below are
# only ever shown at test time, so "novel object" means an outline the network
# has genuinely never seen -- not a new pose of a familiar one.
NOVEL = {
    "tee": np.array([(-0.042, 0.014), (0.042, 0.014), (0.042, -0.010),
                     (0.010, -0.010), (0.010, -0.040), (-0.010, -0.040),
                     (-0.010, -0.010), (-0.042, -0.010)]),
    "arrow": np.array([(-0.030, -0.030), (0.040, 0.000), (-0.030, 0.030),
                       (-0.014, 0.000)]),
    "slab": np.array([(-0.050, -0.014), (0.050, -0.028), (0.050, 0.028),
                      (-0.050, 0.014)]),
}
TRAIN_SHAPES = dict(SHAPES)
TEST_SHAPES = dict(NOVEL)


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


class Scene:
    """A few prisms dropped on the table without overlapping."""

    def __init__(self, rng, shapes, n=3, height=(0.025, 0.055)):
        self.polys, self.h, self.names = [], [], []
        keys = list(shapes)
        tries = 0
        while len(self.polys) < n and tries < 300:
            tries += 1
            k = keys[rng.integers(len(keys))]
            P = shapes[k] - centroid(shapes[k])
            P = P @ _rot(rng.uniform(0, 2 * np.pi)).T
            c = rng.uniform(-0.085, 0.085, 2)
            Q = P + c
            if np.abs(Q).max() > 0.112:
                continue
            if any(_polys_close(Q, R, 0.006) for R in self.polys):
                continue
            self.polys.append(Q)
            self.h.append(float(rng.uniform(*height)))
            self.names.append(k)

    def mass(self, i):
        P = self.polys[i]
        a = 0.5 * abs(np.sum(P[:, 0] * np.roll(P[:, 1], -1) -
                             np.roll(P[:, 0], -1) * P[:, 1]))
        return DENSITY * a * self.h[i]


def _polys_close(A, B, gap):
    """Cheap separation test: bounding circles, then vertex-to-polygon."""
    ca, cb = A.mean(0), B.mean(0)
    ra = np.linalg.norm(A - ca, axis=1).max()
    rb = np.linalg.norm(B - cb, axis=1).max()
    if np.linalg.norm(ca - cb) > ra + rb + gap:
        return False
    return True


def _fill(P, out, value):
    """Rasterize a polygon (in metres) into an image, painting `value`."""
    pix = (P * PX_PER_M + IMG / 2).astype(np.int32)
    cv2.fillPoly(out, [pix.reshape(-1, 1, 2)], value)


def render(scene, rng, noise=0.0, dropout=0.0):
    """Top-down depth image (metres from the camera) plus a per-pixel object id.

    The camera is treated as ORTHOGRAPHIC -- a pixel maps to a fixed square of
    table no matter how tall the object under it is.  A real camera 60 cm up
    sees a 5 cm tall object about 8% too big.  Dropping that keeps the truth
    labels exactly aligned with the pixels, so the experiment measures noise
    and learning rather than a projection bug.

    `noise` is the axial noise of a structured-light sensor.  `dropout` is the
    other half of real depth data: pixels where the sensor returns nothing.
    Both fall on the OBJECT EDGES hardest, because that is where a real sensor
    mixes the near surface and the far surface into one wrong reading -- the
    classic "flying pixel".  Edges are also exactly where the fingers go, so
    this noise attacks the measurement the grasp depends on.
    """
    depth = np.full((IMG, IMG), TABLE_Z, np.float32)
    ids = np.full((IMG, IMG), -1, np.int32)
    order = np.argsort(scene.h)          # paint short things first
    for i in order:
        buf = np.zeros((IMG, IMG), np.uint8)
        _fill(scene.polys[i], buf, 1)
        m = buf.astype(bool)
        depth[m] = TABLE_Z - scene.h[i]
        ids[m] = i
    if noise > 0 or dropout > 0:
        edge = cv2.dilate((cv2.Laplacian(depth, cv2.CV_32F) != 0).astype(np.uint8),
                          np.ones((3, 3), np.uint8), iterations=2).astype(bool)
        sigma = noise * (1.0 + 4.0 * edge)
        depth = depth + rng.normal(0, 1, depth.shape).astype(np.float32) * sigma
        if dropout > 0:
            hole = rng.random(depth.shape) < dropout * (1.0 + 8.0 * edge)
            depth[hole] = TABLE_Z       # a dropped pixel reads as "table"
    return depth, ids


# ---------------------------------------------------------------------------
# grasp candidates
# ---------------------------------------------------------------------------

def sample_candidates(scene, rng, n=40):
    """Random (centre, angle) pairs, biased to land on the objects.

    A uniform sample over the whole image would be ~90% empty table, and a
    classifier trained on that learns "is there anything here", not "will this
    grip".  Biasing towards object pixels is not cheating: at run time an
    off-the-shelf depth segmentation gives you the same object pixels for free
    (this is what project 23 built), so the network is only ever asked the
    question it will actually be asked.
    """
    pts = []
    for _ in range(n):
        i = rng.integers(len(scene.polys))
        P = scene.polys[i]
        lo, hi = P.min(0), P.max(0)
        for _ in range(30):
            c = rng.uniform(lo, hi)
            if _inside(P, c):
                break
        pts.append((c[0], c[1], rng.uniform(0, np.pi)))
    return np.array(pts)


def _inside(P, q):
    A, B = P, np.roll(P, -1, axis=0)
    e = B - A
    r = q[None, :] - A
    cr = e[:, 0] * r[:, 1] - e[:, 1] * r[:, 0]
    return bool(np.all(cr > 0) or np.all(cr < 0))


def _exit(P, c, d):
    """Where the ray c + t*d (t > 0) leaves polygon P, and the inward normal."""
    A, B = P, np.roll(P, -1, axis=0)
    e = B - A
    den = d[0] * e[:, 1] - d[1] * e[:, 0]
    ok = np.abs(den) > 1e-12
    rx, ry = A[:, 0] - c[0], A[:, 1] - c[1]
    t = np.where(ok, (rx * e[:, 1] - ry * e[:, 0]) / np.where(ok, den, 1), -1.0)
    u = np.where(ok, (rx * d[1] - ry * d[0]) / np.where(ok, den, 1), -1.0)
    good = ok & (t > 1e-9) & (u >= -1e-9) & (u <= 1 + 1e-9)
    if not good.any():
        return None, None
    k = int(np.argmin(np.where(good, t, np.inf)))
    L = np.linalg.norm(e[k])
    n = np.array([-e[k, 1], e[k, 0]]) / L
    # ensure the normal points INTO the polygon
    if n @ d > 0:
        n = -n
    return c + t[k] * d, n


def _finger_box(p, n, width=FINGER_W, thick=FINGER_T):
    """The rectangle a fingertip occupies, sitting just outside contact p."""
    t = np.array([-n[1], n[0]])
    base = p - n * (thick * 0.5)
    return np.array([base + t * width / 2, base - t * width / 2,
                     base - t * width / 2 - n * thick,
                     base + t * width / 2 - n * thick])


def _seg_hit(p, p2, q, q2):
    r, s = p2 - p, q2 - q
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-15:
        return False
    t = ((q - p)[0] * s[1] - (q - p)[1] * s[0]) / den
    u = ((q - p)[0] * r[1] - (q - p)[1] * r[0]) / den
    return 0 <= t <= 1 and 0 <= u <= 1


def _overlap(A, B):
    """Do two simple polygons intersect?  Exact, and safe for non-convex shapes.

    A separating-axis test would be shorter but only correct for CONVEX
    polygons, and half the objects here (the L, the T, the arrow) are not
    convex -- it would report collisions inside their notches that are not
    there, and quietly throw away good grasps.
    """
    if any(_inside(B, a) for a in A) or any(_inside(A, b) for b in B):
        return True
    for i in range(len(A)):
        for j in range(len(B)):
            if _seg_hit(A[i], A[(i + 1) % len(A)], B[j], B[(j + 1) % len(B)]):
                return True
    return False


def label(scene, g):
    """The truth oracle.  Returns (success, info dict).

    A grasp holds when four separate things go right, and the point of writing
    them out is that a learned network has to discover all four from pixels:

      1. the fingers close on the SAME object and within the gripper's span
      2. the two contacts pass project 39's force-closure test
      3. the grasp is strong enough for this object's weight
      4. the fingers do not run into a neighbouring object on the way down
    """
    c = np.array(g[:2])
    d = np.array([np.cos(g[2]), np.sin(g[2])])
    obj = next((i for i, P in enumerate(scene.polys) if _inside(P, c)), None)
    info = dict(obj=obj, width=np.nan, q=0.0, fc=False, collide=False,
                weight=np.nan, reason="off object")
    if obj is None:
        return False, info
    P = scene.polys[obj]
    p1, n1 = _exit(P, c, d)
    p2, n2 = _exit(P, c, -d)
    if p1 is None or p2 is None:
        return False, info
    w = float(np.linalg.norm(p1 - p2))
    info["width"] = w
    if w > GRIPPER_MAX or w < GRIPPER_MIN:
        info["reason"] = "too wide" if w > GRIPPER_MAX else "too narrow"
        return False, info
    W = cone_generators(np.stack([p1, p2]), np.stack([n1, n2]), MU, centroid(P))
    info["fc"] = force_closure(W)
    info["q"] = quality(W) if info["fc"] else 0.0
    weight = scene.mass(obj) * 9.81
    info["weight"] = weight
    for k, (p, n) in enumerate(((p1, n1), (p2, n2))):
        box = _finger_box(p, n)
        for j, Q in enumerate(scene.polys):
            if j == obj:
                continue
            if _overlap(box, Q) and scene.h[j] > scene.h[obj] - 0.004:
                info["collide"] = True
    if info["collide"]:
        info["reason"] = "finger hits a neighbour"
        return False, info
    if not info["fc"]:
        info["reason"] = "not force closed"
        return False, info
    if FN * info["q"] < SAFETY * weight:
        info["reason"] = "too weak for the weight"
        return False, info
    info["reason"] = "ok"
    return True, info


# ---------------------------------------------------------------------------
# patches
# ---------------------------------------------------------------------------

PATCH = 32
PATCH_M = 0.104          # the patch spans a little more than the gripper


def patch(depth, g):
    """A square of depth, rotated so the fingers close left-to-right.

    Two conventions, both borrowed from Dex-Net, both load-bearing:

      * ROTATE the crop.  Without it the network has to learn the same grasp
        at every angle separately, and it has to encode the angle somewhere.
        Rotating puts every candidate in the same frame, so one filter covers
        all orientations -- roughly a 16x saving in what has to be learned.
      * SUBTRACT the centre depth.  The network then sees "how much does the
        surface rise and fall around here", not "how far away is the table".
        The absolute distance is a camera-mounting fact, not a grasp fact, and
        leaving it in makes the net fail the moment you raise the camera.
    """
    px = g[0] * PX_PER_M + IMG / 2
    py = g[1] * PX_PER_M + IMG / 2
    side = PATCH_M * PX_PER_M
    M = cv2.getRotationMatrix2D((float(px), float(py)),
                                float(np.degrees(g[2])), 1.0)
    M[0, 2] += side / 2 - px
    M[1, 2] += side / 2 - py
    crop = cv2.warpAffine(depth, M, (int(side), int(side)),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=TABLE_Z)
    small = cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    return (small - small[PATCH // 2, PATCH // 2]).astype(np.float32)


# ---------------------------------------------------------------------------
# the two hand-written scorers the network competes with
# ---------------------------------------------------------------------------

def score_depth_heuristic(depth, cands):
    """The guide's `best_top_down_grasp` score, one candidate at a time.

    Reward the two finger sides for being at the same height as each other and
    lower than the grasp centre: that is what "closing across a ridge" looks
    like in a depth image.  It is three lines and it needs no training, which
    is why it is the baseline everyone should have to beat.
    """
    out = np.zeros(len(cands))
    half = 0.5 * GRIPPER_MAX * PX_PER_M
    for k, g in enumerate(cands):
        u = g[0] * PX_PER_M + IMG / 2
        v = g[1] * PX_PER_M + IMG / 2
        dx, dy = np.cos(g[2]), np.sin(g[2])
        pts = [(u - half * dx, v - half * dy), (u + half * dx, v + half * dy)]
        zz = []
        for (a, b) in pts:
            ai, bi = int(round(a)), int(round(b))
            if not (0 <= ai < IMG and 0 <= bi < IMG):
                zz = None
                break
            zz.append(depth[bi, ai])
        if zz is None:
            out[k] = -1e9
            continue
        z0 = depth[int(round(v)), int(round(u))]
        out[k] = -abs(zz[0] - zz[1]) - 0.1 * abs(z0 - 0.5 * (zz[0] + zz[1]))
    return out


def observed_geometry(depth, min_height=0.006):
    """Object mask and per-pixel surface normal, estimated from the depth image.

    This is the front half of the classical pipeline: before you can run an
    analytic grasp test on a real scene you have to *recover* the geometry the
    test needs.  Everything the sensor got wrong enters here.
    """
    above = (TABLE_Z - depth) > min_height
    above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8))
    n, lab = cv2.connectedComponents(above, 8)
    sm = cv2.GaussianBlur(above.astype(np.float32), (9, 9), 3.0)
    gx = cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=5)
    return lab, gx, gy


def score_analytic_observed(depth, cands, mu=MU):
    """The classical pipeline, run on what the camera actually saw.

    Same force-closure test as project 39 -- but the contact points come from
    walking the observed mask and the normals come from the observed mask's
    gradient.  When the depth image is clean this is nearly the truth oracle.
    When it is noisy, every error goes straight into the normals, and a normal
    that is 25 degrees wrong flips a force-closure verdict on its own.
    """
    lab, gx, gy = observed_geometry(depth)
    H = IMG
    out = np.full(len(cands), -1e9)     # "could not even measure this one"
    step = 1.0
    for k, g in enumerate(cands):
        u = g[0] * PX_PER_M + IMG / 2
        v = g[1] * PX_PER_M + IMG / 2
        d = np.array([np.cos(g[2]), np.sin(g[2])])
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < H and 0 <= vi < H) or lab[vi, ui] == 0:
            continue
        me = lab[vi, ui]
        cs, ns = [], []
        ok = True
        for s in (+1, -1):
            for t in np.arange(1, GRIPPER_MAX * PX_PER_M / 2 + 6, step):
                a, b = u + s * t * d[0], v + s * t * d[1]
                ai, bi = int(round(a)), int(round(b))
                if not (0 <= ai < H and 0 <= bi < H) or lab[bi, ai] != me:
                    aa, bb = u + s * (t - 1) * d[0], v + s * (t - 1) * d[1]
                    cs.append(np.array([aa, bb]))
                    gi, gj = int(round(bb)), int(round(aa))
                    # the smoothed mask rises towards the interior, so its
                    # gradient IS the inward contact normal
                    nv = np.array([gx[gi, gj], gy[gi, gj]])
                    nn = np.linalg.norm(nv)
                    ns.append(nv / nn if nn > 1e-9 else np.array([1.0, 0.0]))
                    break
            else:
                ok = False
        if not ok or len(cs) != 2:
            continue
        p1, p2 = cs[0] / PX_PER_M, cs[1] / PX_PER_M
        if np.linalg.norm(p1 - p2) > GRIPPER_MAX:
            continue
        ref = 0.5 * (p1 + p2)
        W = cone_generators(np.stack([p1, p2]), np.stack(ns), mu, ref)
        if force_closure(W):
            out[k] = quality(W)
        else:
            # A pass/fail score cannot rank anything: on a hard scene every
            # candidate fails and argmax picks the first one, which is random
            # selection wearing a lab coat.  So failing grasps get a small
            # NEGATIVE score equal to how far outside the friction cone they
            # are -- "least bad" is still useful information.
            e = np.linalg.norm(p2 - p1)
            u = (p2 - p1) / max(e, 1e-9)
            a = np.arctan(mu)
            m = min(a - np.arccos(np.clip(u @ ns[0], -1, 1)),
                    a - np.arccos(np.clip(-u @ ns[1], -1, 1)))
            out[k] = 1e-3 * min(m, 0.0)
    return out


def score_depth_heuristic_fixed(depth, cands):
    """The same three lines, with the second term's sign corrected.

    The guide's version rewards the grasp centre for being at the SAME height
    as the two finger sites.  For a top-down grasp you want the opposite: the
    fingers should land on the table (far from the camera) while the centre
    sits on the object (near the camera).  Flipping that one term is the
    difference between "find a flat patch" and "find something to straddle".
    """
    out = np.zeros(len(cands))
    half = 0.5 * GRIPPER_MAX * PX_PER_M
    for k, g in enumerate(cands):
        u = g[0] * PX_PER_M + IMG / 2
        v = g[1] * PX_PER_M + IMG / 2
        dx, dy = np.cos(g[2]), np.sin(g[2])
        zz = []
        for s in (-1, 1):
            a, b = u + s * half * dx, v + s * half * dy
            ai, bi = int(round(a)), int(round(b))
            if not (0 <= ai < IMG and 0 <= bi < IMG):
                zz = None
                break
            zz.append(depth[bi, ai])
        if zz is None:
            out[k] = -1e9
            continue
        z0 = depth[int(round(v)), int(round(u))]
        out[k] = -abs(zz[0] - zz[1]) + 0.5 * (0.5 * (zz[0] + zz[1]) - z0)
    return out
