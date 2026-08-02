"""Force closure for 2D polygons, from the friction cone up.

Everything here is one idea repeated: a contact can only push along a limited
set of directions, each of those directions produces a *wrench* on the object,
and the grasp is good exactly when those wrenches can add up to cancel
anything the world throws at the object.

The three objects you need:

  Contact   -- a point on the boundary plus the inward normal there.
  wrench    -- a (fx, fy, torque) triple: what one unit force at one contact
               does to the object.
  hull      -- the convex hull of all the contact wrenches.  Force closure is
               the statement "the origin is strictly inside that hull".

No SciPy in this environment, so the two geometric primitives (is the origin
inside a hull in 3D, and how far from the boundary is it) are written out by
hand.  They are short, because the number of wrenches is small.
"""

import numpy as np

# ---------------------------------------------------------------------------
# polygons
# ---------------------------------------------------------------------------


def _poly(pts):
    P = np.asarray(pts, dtype=float)
    # make sure the winding is counter-clockwise, so that "left of the edge
    # direction" is reliably "inside".  Shoelace formula: positive area = CCW.
    area = 0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1])
    if area < 0:
        P = P[::-1]
    return P


def _regular(n, r, phase=0.0):
    a = np.arange(n) * 2 * np.pi / n + phase
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


# Five shapes, chosen so that the easy cases and the awkward cases are both
# represented.  Sizes are in metres and are typical of what a small parallel
# jaw gripper handles.
SHAPES = {
    "rect":     _poly([(-0.045, -0.020), (0.045, -0.020), (0.045, 0.020), (-0.045, 0.020)]),
    "hex":      _poly(_regular(6, 0.033, phase=0.1)),
    "triangle": _poly([(-0.045, -0.026), (0.045, -0.026), (0.0, 0.052)]),
    "ell":      _poly([(-0.040, -0.030), (0.040, -0.030), (0.040, 0.000),
                       (-0.008, 0.000), (-0.008, 0.045), (-0.040, 0.045)]),
    "wedge":    _poly([(-0.048, -0.018), (0.048, -0.006), (0.048, 0.006), (-0.048, 0.018)]),
}


def centroid(P):
    """Area centroid of a simple polygon -- the point torques are measured about."""
    x, y = P[:, 0], P[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = 0.5 * cr.sum()
    cx = np.sum((x + x1) * cr) / (6 * a)
    cy = np.sum((y + y1) * cr) / (6 * a)
    return np.array([cx, cy])


def perimeter_samples(P, n):
    """`n` contact candidates spaced evenly along the boundary.

    Returns points, inward normals, and the edge index each sample came from.
    The inward normal of a CCW edge (a -> b) is the edge direction rotated by
    +90 degrees: if the edge runs left-to-right along the bottom of the shape,
    the interior is above it.
    """
    A = P
    B = np.roll(P, -1, axis=0)
    seg = B - A
    L = np.linalg.norm(seg, axis=1)
    s = np.linspace(0, L.sum(), n, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    idx = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(P) - 1)
    t = (s - cum[idx]) / L[idx]
    pts = A[idx] + t[:, None] * seg[idx]
    d = seg[idx] / L[idx][:, None]
    nrm = np.stack([-d[:, 1], d[:, 0]], 1)
    return pts, nrm, idx


def inside(P, q):
    """Is point q strictly inside the CCW polygon P?  (cross product per edge)"""
    A, B = P, np.roll(P, -1, axis=0)
    e = B - A
    r = q[None, :] - A
    return bool(np.all(e[:, 0] * r[:, 1] - e[:, 1] * r[:, 0] > 0))


# ---------------------------------------------------------------------------
# wrenches
# ---------------------------------------------------------------------------

# The torque of a force has units N*m while the force has units N, so the
# three numbers in a wrench are not comparable until you divide the torque by
# some length.  LAMBDA is that length: with it, a wrench of "1" in the torque
# slot means the same amount of grip as a wrench of "1" in a force slot for an
# object about LAMBDA across.  Experiment 6 measures how much your answer
# depends on this arbitrary choice.
LAMBDA = 0.05


def cone_generators(pts, nrm, mu, ref, lam=LAMBDA):
    """Wrenches of the two edges of each contact's friction cone.

    A contact with friction coefficient `mu` can push anywhere inside a cone of
    half-angle atan(mu) around the inward normal -- push at a steeper angle and
    the finger slides.  Any force inside the cone is a positive combination of
    the two edge directions, so the two edges carry all the information: work
    with them and the whole continuous cone comes along for free.

    Returns an array of shape (2 * n_contacts, 3): rows (fx, fy, torque/lam).
    """
    pts = np.atleast_2d(pts)
    nrm = np.atleast_2d(nrm)
    a = np.arctan(mu)
    out = []
    for sgn in (+1, -1):
        c, s = np.cos(sgn * a), np.sin(sgn * a)
        f = np.stack([c * nrm[:, 0] - s * nrm[:, 1],
                      s * nrm[:, 0] + c * nrm[:, 1]], 1)
        r = pts - ref[None, :]
        tau = (r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]) / lam
        out.append(np.concatenate([f, tau[:, None]], 1))
    # interleave so that the two edges of contact k are rows 2k and 2k+1
    W = np.empty((2 * len(pts), 3))
    W[0::2], W[1::2] = out[0], out[1]
    return W


# ---------------------------------------------------------------------------
# the force-closure test
# ---------------------------------------------------------------------------

def force_closure(W, eps=1e-9):
    """True when the wrenches in W positively span the whole plane's wrench space.

    "Positively span" means: every wrench you might need can be written as a
    sum of the contact wrenches with NON-NEGATIVE coefficients.  Non-negative
    because a finger can push but not pull -- that one-sidedness is the whole
    difficulty.

    The test is a search for a counter-example.  If the grasp is NOT
    force-closed, some direction d exists in which no contact can push at all:
    d . w <= 0 for every wrench w.  Then a disturbance along d is unopposed.
    In 3D such a "separating" direction, if one exists, can always be rotated
    until it is perpendicular to two of the wrenches, so it is enough to try
    d = w_i x w_j for every pair.  That is an exact test, not a sampled one.
    """
    m = len(W)
    # A flat set of wrenches can never surround the origin.  Two exactly
    # opposed frictionless fingers are the classic case: they can push along
    # one line and nothing else, so the wrenches are all multiples of one
    # vector and the pair loop below has no plane to test.  Rank first.
    if m < 4 or np.linalg.matrix_rank(W, tol=1e-9) < 3:
        return False
    for i in range(m):
        for j in range(i + 1, m):
            d = np.cross(W[i], W[j])
            nd = np.linalg.norm(d)
            if nd < 1e-12:
                continue
            p = W @ (d / nd)
            if np.all(p >= -1e-9) or np.all(p <= 1e-9):
                return False
    return True


def force_closure_sampled(W, n=4000, rng=None):
    """The lazy version: try random directions and hope to trip over a bad one.

    Kept because it is what most people write first, and because experiment 4
    measures what it misses.  It can only ever *disprove* force closure; not
    finding a bad direction in n tries is not a proof that none exists.
    """
    rng = rng or np.random.default_rng(0)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return bool(np.all((W @ d.T).max(axis=0) > 0))


def antipodal(p1, n1, p2, n2, mu):
    """The two-finger shortcut: is the line between the contacts inside both cones?

    For exactly two frictional point contacts in 2D this is equivalent to force
    closure, and it is one line instead of a hull.  "Antipodal" is Greek for
    "feet opposite" (the antipodes are the point on the far side of the Earth):
    the two contacts must face each other.
    """
    a = np.arctan(mu)
    d = p2 - p1
    L = np.linalg.norm(d)
    if L < 1e-9:
        return False
    d = d / L
    return (np.arccos(np.clip(d @ n1, -1, 1)) < a - 1e-12 and
            np.arccos(np.clip(-d @ n2, -1, 1)) < a - 1e-12)


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------

def _facets(W):
    """Every facet of the convex hull of W, as (unit normal, offset) pairs.

    Brute force: each triple of points spans a plane; it is a facet if every
    other point is on one side of it.  With at most a few dozen wrenches this
    is milliseconds, and it avoids depending on a hull library.
    """
    m = len(W)
    out = []
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                nvec = np.cross(W[j] - W[i], W[k] - W[i])
                nn = np.linalg.norm(nvec)
                if nn < 1e-12:
                    continue
                nvec = nvec / nn
                off = nvec @ W[i]
                s = W @ nvec - off
                if np.all(s <= 1e-9):
                    out.append((nvec, off))
                elif np.all(s >= -1e-9):
                    out.append((-nvec, -off))
    return out


def quality(W):
    """Ferrari-Canny: the radius of the biggest ball of disturbance the grasp resists.

    Named after Carlo Ferrari and John Canny, who proposed it in 1992.  Take
    the convex hull of the contact wrenches; that hull is the set of wrenches
    the grasp can apply when the fingers together push with one unit of normal
    force.  The distance from the origin to the NEAREST face of that hull is
    the disturbance in the WORST direction that the grasp can still hold.  A
    grasp is only as good as its weakest direction, which is why the metric is
    a minimum and not an average.

    Returns 0.0 when the grasp is not force-closed (the origin is on or outside
    the hull, so some direction is unopposed).
    """
    fc = _facets(W)
    if not fc:
        return 0.0
    # every facet is stored with its OUTWARD normal, so `off` is exactly the
    # signed distance from the origin to that face: positive means inside.
    q = min(off for _, off in fc)
    return float(max(q, 0.0))


def max_resistible(W, u):
    """How big a disturbance in direction u the grasp survives (unit normal force).

    This is the hull's radius in direction u: how far you can walk from the
    origin along u before leaving the hull.  `quality` is the minimum of this
    over all directions u, so plotting it (experiment 5) shows what the single
    quality number is summarising.
    """
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    best = np.inf
    for nvec, off in _facets(W):
        d = nvec @ u
        if d > 1e-12:
            best = min(best, off / d)
    return float(max(best, 0.0)) if np.isfinite(best) else 0.0


# ---------------------------------------------------------------------------
# grasp search
# ---------------------------------------------------------------------------

def candidate_pairs(P, n_samples=90, max_width=0.075, min_width=0.008):
    """All two-finger candidates: sampled boundary points, paired by distance."""
    pts, nrm, eid = perimeter_samples(P, n_samples)
    i, j = np.triu_indices(n_samples, k=1)
    d = np.linalg.norm(pts[i] - pts[j], axis=1)
    keep = (d <= max_width) & (d >= min_width)
    return pts, nrm, eid, i[keep], j[keep], d[keep]


def score_pairs(P, mu, n_samples=90, max_width=0.075, lam=LAMBDA):
    """Force closure and quality for every two-finger candidate on a polygon."""
    ref = centroid(P)
    pts, nrm, eid, i, j, w = candidate_pairs(P, n_samples, max_width)
    fc = np.zeros(len(i), bool)
    q = np.zeros(len(i))
    for k, (a, b) in enumerate(zip(i, j)):
        W = cone_generators(pts[[a, b]], nrm[[a, b]], mu, ref, lam)
        if force_closure(W):
            fc[k] = True
            q[k] = quality(W)
    return dict(pts=pts, nrm=nrm, i=i, j=j, width=w, fc=fc, q=q, ref=ref)


def best_pair(P, mu, **kw):
    r = score_pairs(P, mu, **kw)
    if not r["fc"].any():
        return None, r
    k = int(np.argmax(r["q"]))
    return k, r
