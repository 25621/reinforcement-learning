"""A square of cloth, simulated by position-based dynamics, and a pick-and-place arm.

Why write the simulator instead of using MuJoCo.  Cloth is not a rigid body,
and a rigid-body engine models it (if at all) as hundreds of small bodies with
hundreds of joints -- slow, fiddly, and mostly a demonstration of the engine's
solver.  Fifty lines of [position-based dynamics](/shared/glossary/#position-based-dynamics)
give a cloth that folds convincingly and runs a whole episode in under a
second, which is what makes the experiments in this project affordable.

The idea behind PBD, which is unusual enough to be worth a sentence: it never
computes a force.  It guesses where every particle would drift to, and then
*moves the positions* until the constraints hold -- this edge is 14 mm long,
this point is above the table -- and reads the velocity back from how far the
position actually ended up moving.  A stiff spring plus a large time step
overshoots and explodes; a projection cannot overshoot, because it stops when
the constraint is satisfied.  That is why nearly every real-time cloth in
games and robotics is built this way.
"""

import numpy as np

GRID = 15                 # particles per side
SIZE = 0.20               # metres
SPACING = SIZE / (GRID - 1)
DT = 0.006
GRAV = np.array([0.0, 0.0, -9.81])
ITERS = 6                 # constraint projections per step
DAMP = 0.02
GROUND_MU = 0.55
SELF_R = 0.45 * SPACING


def _pairs():
    """Structural, shear and bend constraints, as index arrays."""
    idx = np.arange(GRID * GRID).reshape(GRID, GRID)
    a, b = [], []

    def add(i0, j0, i1, j1):
        a.append(idx[i0, j0].ravel())
        b.append(idx[i1, j1].ravel())

    r = np.arange(GRID)
    # structural: the threads of the weave
    add(r[:-1, None], r[None, :], r[1:, None], r[None, :])
    add(r[:, None], r[None, :-1], r[:, None], r[None, 1:])
    # shear: the diagonals.  Without them the sheet folds like a paper fan --
    # a grid of hinges with nothing resisting a parallelogram distortion.
    add(r[:-1, None], r[None, :-1], r[1:, None], r[None, 1:])
    add(r[:-1, None], r[None, 1:], r[1:, None], r[None, :-1])
    # bend: every SECOND particle, which is what makes the cloth resist
    # creasing rather than resist stretching
    add(r[:-2, None], r[None, :], r[2:, None], r[None, :])
    add(r[:, None], r[None, :-2], r[:, None], r[None, 2:])
    return np.concatenate(a), np.concatenate(b)


IA, IB = _pairs()
CNT = np.maximum(np.bincount(IA, minlength=GRID * GRID) +
                 np.bincount(IB, minlength=GRID * GRID), 1).astype(float)


class Cloth:
    def __init__(self, rng=None, yaw=0.0, centre=(0.0, 0.0), stiff=1.0,
                 self_collide=True):
        u = (np.arange(GRID) - (GRID - 1) / 2) * SPACING
        X, Y = np.meshgrid(u, u, indexing="ij")
        c, s = np.cos(yaw), np.sin(yaw)
        self.p = np.stack([c * X - s * Y + centre[0],
                           s * X + c * Y + centre[1],
                           np.full_like(X, 0.002)], -1).reshape(-1, 3)
        self.v = np.zeros_like(self.p)
        self.rest = np.linalg.norm(self.p[IA] - self.p[IB], axis=1)
        self.w = np.ones(len(self.p))
        self.stiff = stiff
        self.self_collide = self_collide
        # which particle pairs are close enough in the WEAVE that touching is
        # normal and should not be pushed apart
        gi, gj = np.divmod(np.arange(GRID * GRID), GRID)
        d = np.abs(gi[:, None] - gi[None, :]) + np.abs(gj[:, None] - gj[None, :])
        self.near = d <= 2

    # ---------------------------------------------------------------- physics
    def step(self, pinned=None, target=None):
        w = self.w.copy()
        if pinned is not None and len(pinned):
            w[pinned] = 0.0
        self.v += DT * GRAV
        self.v *= (1.0 - DAMP)
        q = self.p + DT * self.v
        if pinned is not None and len(pinned):
            q[pinned] = target
        for _ in range(ITERS):
            self._project(q, w)
        # Self-collision every few steps rather than every step.  Layers move
        # slowly compared with the constraint solver, so checking a third as
        # often is visually identical and three times cheaper.
        self.k = getattr(self, "k", 0) + 1
        if self.self_collide and self.k % 3 == 0:
            self._self(q, w)
        # the table
        below = q[:, 2] < 0.0
        q[below, 2] = 0.0
        newv = (q - self.p) / DT
        # Coulomb friction on the table, applied as a straight scaling of the
        # in-plane velocity of whatever is touching it.  Without it a fold
        # slides back open the moment the gripper lets go.
        newv[below, :2] *= (1.0 - GROUND_MU)
        self.v = newv
        self.p = q

    def _project(self, q, w):
        d = q[IA] - q[IB]
        L = np.linalg.norm(d, axis=1)
        L = np.where(L < 1e-9, 1e-9, L)
        corr = self.stiff * (L - self.rest) / L
        wa, wb = w[IA], w[IB]
        denom = np.where(wa + wb < 1e-9, 1.0, wa + wb)
        c = (corr / denom)[:, None] * d
        # Jacobi rather than Gauss-Seidel: accumulate every constraint's
        # correction, then apply the average.  Gauss-Seidel converges faster
        # per iteration but has to visit constraints one at a time, which in
        # Python is a hundred times slower than one array operation.
        n = len(q)
        ca = -(wa[:, None] * c)
        cb = +(wb[:, None] * c)
        # np.bincount, not np.add.at: they compute the same scatter-add, but
        # add.at is a Python-level loop under the hood and is roughly fifty
        # times slower here, which is the difference between a two-second
        # episode and a two-minute one.
        acc = np.stack([np.bincount(IA, ca[:, k], n) +
                        np.bincount(IB, cb[:, k], n) for k in range(3)], 1)
        q += acc / CNT[:, None]

    def _self(self, q, w):
        """Keep layers of cloth from passing through each other.

        Without this the top layer of a fold sinks straight through the bottom
        one and settles on the table, so a two-layer fold and a one-layer sheet
        end up looking identical from above -- and the metric this project uses
        is exactly a view from above.
        """
        D = q[:, None, :] - q[None, :, :]
        dist = np.sqrt((D ** 2).sum(-1) + 1e-12)
        hit = (dist < SELF_R) & (~self.near)
        if not hit.any():
            return
        push = np.where(hit, (SELF_R - dist) / dist, 0.0)
        corr = 0.5 * (push[:, :, None] * D).sum(1)
        q += corr * (w[:, None] > 0)

    # ------------------------------------------------------------------ views
    def mask(self, res=64, extent=0.16):
        """The cloth's footprint, seen from directly above.

        Each little quad of the mesh is filled in, not just the particles --
        a particle mask would be 225 dots and would measure the resolution of
        the grid rather than the shape of the cloth.
        """
        import cv2
        img = np.zeros((res, res), np.uint8)
        P = self.p.reshape(GRID, GRID, 3)
        uv = ((P[:, :, :2] + extent) / (2 * extent) * res).astype(np.int32)
        quads = np.stack([uv[:-1, :-1], uv[1:, :-1], uv[1:, 1:], uv[:-1, 1:]],
                         axis=2).reshape(-1, 4, 1, 2)
        cv2.fillPoly(img, list(quads), 1)
        # Close one-pixel pinholes.  Rounding each quad's corners to whole
        # pixels leaves hairline gaps along the shared edges of neighbouring
        # quads, and those gaps would be counted as "no cloth here" by the IoU.
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return img.astype(bool)

    def layers(self, res=64, extent=0.16):
        """How many layers of cloth sit over each pixel (0, 1, 2, ...)."""
        import cv2
        img = np.zeros((res, res), np.int32)
        P = self.p.reshape(GRID, GRID, 3)
        uv = ((P[:, :, :2] + extent) / (2 * extent) * res).astype(np.int32)
        for i in range(GRID - 1):
            for j in range(GRID - 1):
                one = np.zeros((res, res), np.uint8)
                quad = np.stack([uv[i, j], uv[i + 1, j], uv[i + 1, j + 1],
                                 uv[i, j + 1]]).astype(float)
                # Shrink each quad towards its own centre before filling it.
                # Neighbouring quads share an edge, so at full size they
                # overlap along it and a perfectly flat sheet would be counted
                # as four layers thick.
                quad = (quad.mean(0) + 0.8 * (quad - quad.mean(0)))
                cv2.fillPoly(one, [quad.astype(np.int32)[:, None, :]], 1)
                img += one
        return img


def iou(a, b):
    """Intersection over union of two footprints, in [0, 1]."""
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / max(union, 1))


# ---------------------------------------------------------------------------
# the arm: one pick-and-place primitive
# ---------------------------------------------------------------------------

def nearest_particle(cloth, xy):
    """Which particle the gripper would actually pinch.

    Real cloth grasping picks up whatever is on TOP at that spot, so among the
    particles near the requested point we take the highest one.  Reaching for
    the lowest layer is exactly the failure that makes a fold come undone.
    """
    d = np.linalg.norm(cloth.p[:, :2] - np.asarray(xy)[None, :], axis=1)
    close = np.flatnonzero(d < 1.5 * SPACING)
    if len(close) == 0:
        return int(np.argmin(d))
    return int(close[np.argmax(cloth.p[close, 2])])


def pick_place(cloth, pick_xy, place_xy, height=0.09, n_lift=35, n_move=60,
               n_drop=35, n_settle=80):
    """Grab the top layer at `pick_xy`, arc it over to `place_xy`, let go.

    The arc matters.  Dragging the grasped corner along the table instead
    scrapes the whole sheet sideways; lifting it clear means only the part
    that has to move, moves.  This is the standard "quasi-static pick and
    place" action primitive, and reducing a whole manipulation to two points
    on an image is what makes it learnable from a handful of examples.
    """
    k = nearest_particle(cloth, pick_xy)
    p0 = cloth.p[k].copy()
    p1 = np.array([place_xy[0], place_xy[1], 0.004])
    pin = np.array([k])
    for t in range(n_lift):
        a = (t + 1) / n_lift
        cloth.step(pin, p0 + np.array([0, 0, height * a]))
    top0 = p0 + np.array([0, 0, height])
    top1 = p1 + np.array([0, 0, height])
    for t in range(n_move):
        a = (t + 1) / n_move
        cloth.step(pin, top0 + a * (top1 - top0))
    for t in range(n_drop):
        a = (t + 1) / n_drop
        cloth.step(pin, top1 + a * (p1 - top1))
    for _ in range(n_settle):
        cloth.step()
    return k


def settle(cloth, n=180):
    for _ in range(n):
        cloth.step()


# ---------------------------------------------------------------------------
# goals
# ---------------------------------------------------------------------------

def corner_xy(cloth, which):
    """World position of one of the four corners of the sheet."""
    P = cloth.p.reshape(GRID, GRID, 3)
    return {0: P[0, 0, :2], 1: P[0, -1, :2], 2: P[-1, -1, :2],
            3: P[-1, 0, :2]}[which].copy()


def expert_folds(cloth):
    """Two folds that quarter the sheet: corner 0 onto 2, then 1 onto 3."""
    return [(corner_xy(cloth, 0), corner_xy(cloth, 2)),
            (corner_xy(cloth, 1), corner_xy(cloth, 3))]


def extreme_fold(cloth, axis=0):
    """Fold the most extreme point along a diagonal onto the opposite extreme.

    Defined from what is VISIBLE (the particles' positions in the plane), not
    from which corner of the weave a point belongs to.  That distinction is the
    whole reason this version exists: a square sheet looks identical under a
    quarter turn, so "pick corner number 0" is not a question a camera can
    answer, and a policy trained on it would be learning to guess.  "Pick the
    point furthest along this diagonal" is answerable from the image, so it is
    something a policy can actually be graded on.
    """
    key = cloth.p[:, 0] + cloth.p[:, 1] if axis == 0 else \
        cloth.p[:, 0] - cloth.p[:, 1]
    return cloth.p[int(np.argmin(key)), :2].copy(), \
        cloth.p[int(np.argmax(key)), :2].copy()


def expert_plan(cloth, n_folds=2):
    """The two folds an expert would make, read off the current shape."""
    out = []
    tmp = Cloth(yaw=0.0)
    tmp.p = cloth.p.copy()
    tmp.v = cloth.v.copy()
    tmp.stiff, tmp.self_collide = cloth.stiff, cloth.self_collide
    for k in range(n_folds):
        pk, pl = extreme_fold(tmp, axis=k % 2)
        out.append((pk, pl))
        pick_place(tmp, pk, pl)
    return out


def expert_result(cloth, n_folds=2, res=64, extent=0.16):
    """Run the expert on a COPY of this cloth and return its footprint.

    Grading against the expert's own result on the same starting sheet, rather
    than against a drawn rectangle, keeps the metric honest: it asks "did you
    fold it the way it can be folded" and not "did you reach a shape the
    physics cannot reach from here".
    """
    ref = Cloth(yaw=0.0, stiff=cloth.stiff, self_collide=cloth.self_collide)
    ref.p = cloth.p.copy()
    ref.v = cloth.v.copy()
    for k in range(n_folds):
        pk, pl = extreme_fold(ref, axis=k % 2)
        pick_place(ref, pk, pl)
    return ref.mask(res, extent), ref
