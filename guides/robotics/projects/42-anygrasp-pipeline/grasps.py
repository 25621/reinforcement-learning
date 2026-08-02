"""Turning a point cloud into 6-DoF grasp candidates, and scoring them.

This is the shape of every modern grasp detector -- GPD, GraspNet, AnyGrasp --
with the learned part shrunk to something a laptop can train:

    point cloud -> surface normals -> sample grasp poses -> filter -> score

The one thing to hold on to: a candidate is a *pose*, six numbers, and it is
generated from the geometry rather than searched over.  Scoring every 6-DoF
pose in a workspace is hopeless (project 40 could enumerate a whole image
because top-down grasping is only four numbers); sampling poses that sit on
the observed surface cuts the space to something you can actually enumerate.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from pick import GRIP_HALF_WIDTH, grasp_frame                                 # noqa: E402

CLOSE_HALF = 0.036        # half the free space between the open fingers
FINGER_Y = 0.012          # half-depth of a fingertip
FINGER_Z = 0.024          # half-length of a fingertip along the approach
FINGER_X = 0.006          # half-thickness of a fingertip
MU = 0.4


# ---------------------------------------------------------------------------
# normals
# ---------------------------------------------------------------------------

def normals(P, radius=0.014, viewpoint=None):
    """Surface normal at every point, from the local neighbourhood's flatness.

    Take the points within `radius`, compute their covariance, and take the
    eigenvector with the SMALLEST eigenvalue.  Intuition: the neighbourhood is
    a little patch of surface, so it spreads out a lot in two directions and
    barely at all in the third — and "barely at all" is the direction sticking
    out of the surface.

    The sign is genuinely ambiguous from geometry alone (a plane has two
    faces), so we resolve it the only way a single camera can: the normal must
    point back toward the camera, because that is the side we saw.
    """
    n = len(P)
    D = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    N = np.zeros_like(P)
    for i in range(n):
        idx = np.flatnonzero(D[i] < radius ** 2)
        if len(idx) < 5:
            N[i] = np.array([0.0, 0.0, 1.0])
            continue
        Q = P[idx] - P[idx].mean(0)
        w, V = np.linalg.eigh(Q.T @ Q)
        N[i] = V[:, 0]
    if viewpoint is not None:
        flip = np.einsum("ij,ij->i", N, viewpoint[None, :] - P) < 0
        N[flip] *= -1
    return N


# ---------------------------------------------------------------------------
# candidate generation
# ---------------------------------------------------------------------------

def sample_candidates(P, N, rng, n_points=90, n_angles=6, max_up=0.35):
    """Grasp poses generated from the observed surface.

    For each sampled surface point we take the approach direction to be
    straight into the surface, then try several rotations of the fingers about
    that approach.  The gripper's roll about its own approach axis is the one
    degree of freedom the surface normal does not determine, so it is the one
    we enumerate.

    Returns the surviving candidates plus a funnel counting who died where —
    experiment 2 plots that funnel, because knowing which filter kills your
    candidates is most of debugging a grasp pipeline.
    """
    funnel = dict(sampled=0, approach_too_flat=0, empty=0, too_wide=0,
                  finger_collision=0, below_table=0, kept=0)
    out = []
    if len(P) < 20:
        return out, funnel
    idx = rng.choice(len(P), size=min(n_points, len(P)), replace=False)
    for i in idx:
        p, nrm = P[i], N[i]
        a = -nrm
        funnel["sampled"] += n_angles
        # A table-top robot cannot come at an object from below, and a nearly
        # horizontal approach means the wrist sweeps across the table.
        if a[2] > -max_up:
            funnel["approach_too_flat"] += n_angles
            continue
        u = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(u) < 1e-6:
            u = np.array([1.0, 0.0, 0.0])
        u = u / np.linalg.norm(u)
        v = np.cross(a, u)
        for t in np.linspace(0, np.pi, n_angles, endpoint=False):
            c = np.cos(t) * u + np.sin(t) * v
            g = _fit(P, p, a, c, funnel)
            if g is not None:
                out.append(g)
    funnel["kept"] = len(out)
    return out, funnel


def _fit(P, p, a, c, funnel):
    """Place the gripper on the surface at p, then check it fits."""
    R = grasp_frame(a, c)
    # Start 12 mm inside the surface so the fingers straddle it rather than
    # closing on thin air just outside the object.
    g = p + a * 0.012
    L = (P - g) @ R
    inside = (np.abs(L[:, 0]) < CLOSE_HALF) & (np.abs(L[:, 1]) < FINGER_Y) & \
             (np.abs(L[:, 2]) < FINGER_Z)
    if inside.sum() < 8:
        funnel["empty"] += 1
        return None
    Q = L[inside]
    # Recentre: the fingers should close symmetrically about the material, and
    # sit at the middle of it along the approach.
    g = g + R @ np.array([Q[:, 0].mean(), 0.0, Q[:, 2].mean()])
    L = (P - g) @ R
    inside = (np.abs(L[:, 0]) < CLOSE_HALF) & (np.abs(L[:, 1]) < FINGER_Y) & \
             (np.abs(L[:, 2]) < FINGER_Z)
    Q = L[inside]
    if len(Q) < 8:
        funnel["empty"] += 1
        return None
    width = Q[:, 0].max() - Q[:, 0].min()
    if width > 2 * CLOSE_HALF - 0.004:
        funnel["too_wide"] += 1
        return None
    if g[2] < 0.012:
        funnel["below_table"] += 1
        return None
    if finger_collision(P, g, R):
        funnel["finger_collision"] += 1
        return None
    return dict(pos=g, approach=R[:, 2], closing=R[:, 0], R=R,
                pts=Q, width=float(width), n_inside=int(len(Q)))


def finger_collision(P, g, R, clear=0.004):
    """Would the fingers or the palm hit something on the way in?

    This is the step that separates a grasp *detector* from a grasp *planner*.
    The detector says "these two surfaces face each other"; the planner also
    has to get the metal there.  In clutter this filter rejects more candidates
    than the geometry test does — experiment 5 measures exactly how many.
    """
    L = (P - g) @ R
    for sgn in (-1, 1):
        x0 = sgn * (CLOSE_HALF + clear)
        hit = (np.abs(L[:, 0] - sgn * (CLOSE_HALF + FINGER_X)) < FINGER_X + clear) & \
              (np.abs(L[:, 1]) < FINGER_Y + clear) & (np.abs(L[:, 2]) < FINGER_Z)
        if hit.any():
            return True
    # the palm, a slab sitting just behind the finger bases
    palm = (np.abs(L[:, 0]) < 0.032) & (np.abs(L[:, 1]) < 0.020) & \
           (L[:, 2] < -FINGER_Z + 0.002) & (L[:, 2] > -FINGER_Z - 0.030)
    return bool(palm.any())


# ---------------------------------------------------------------------------
# the hand-written scorer
# ---------------------------------------------------------------------------

def antipodal_score(g, N_all, P_all, mu=MU):
    """Project 39's antipodal test, lifted into 3D and run on measured normals.

    Split the points between the fingers into a left half and a right half.
    For the grasp to hold, the left surface must face right and the right
    surface must face left, each within the [friction cone](angle atan(mu)).
    That is exactly the 2D condition, applied to the two point *sets* rather
    than two exact contact points, because a measured surface does not give you
    a single contact.
    """
    R = g["R"]
    L = (P_all - g["pos"]) @ R
    inside = (np.abs(L[:, 0]) < CLOSE_HALF) & (np.abs(L[:, 1]) < FINGER_Y) & \
             (np.abs(L[:, 2]) < FINGER_Z)
    if inside.sum() < 8:
        return -1.0
    Nl = N_all[inside] @ R          # normals, in the grasp's own frame
    Ql = L[inside]
    lo, hi = Ql[:, 0].min(), Ql[:, 0].max()
    band = 0.25 * (hi - lo) + 0.003
    left = Ql[:, 0] < lo + band
    right = Ql[:, 0] > hi - band
    if left.sum() < 3 or right.sum() < 3:
        return -1.0
    # the outward normal of the left surface should point in -x, the right in +x
    cl = -Nl[left, 0].mean() / max(np.linalg.norm(Nl[left].mean(0)), 1e-9)
    cr = Nl[right, 0].mean() / max(np.linalg.norm(Nl[right].mean(0)), 1e-9)
    lim = np.cos(np.arctan(mu))
    margin = min(cl, cr) - lim
    # a narrow grasp is a safer grasp: more finger travel left before it slips
    width = hi - lo
    return float(margin - 0.6 * width)
