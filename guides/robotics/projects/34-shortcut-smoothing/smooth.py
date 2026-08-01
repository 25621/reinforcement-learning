"""Post-processing a sampled path -- the shared library for project 34.

Everything here takes a path (a list of configurations) and a `seg_free(a, b)`
oracle, and gives back a shorter or smoother path.  Nothing here knows whether
the configurations are 2 numbers or 7, which is the point: the same three
routines run on the plane from project 32 and on the arm from project 33.

Imported by project 36 (TOPP), which needs a smooth path to time-parameterise.
"""

import math

import numpy as np


def path_cost(path):
    if path is None or len(path) < 2:
        return math.inf
    p = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def resample(path, spacing):
    """Re-space a path so consecutive points are `spacing` apart.

    Why bother, when the path already describes the same geometry?  Because a
    shortcut must be allowed to START AND END IN THE MIDDLE of a segment.  A
    raw RRT path has its corners as waypoints and nothing in between, so an
    algorithm that only picks existing waypoints can only cut corners, never
    slide along an edge.  Re-spacing turns "pick two waypoints" into "pick two
    points anywhere on the path" without writing any extra code.
    """
    p = np.asarray(path, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0:
        return p.copy()
    n = max(2, int(np.ceil(s[-1] / spacing)) + 1)
    ts = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(ts, s, p[:, j]) for j in range(p.shape[1])], 1)


def shortcut(path, seg_free, rng, iters=200, spacing=None, track=False):
    """Classic random shortcutting.

    Repeat: pick two points at random along the path; if the straight line
    between them is collision-free, throw away everything in between.

    It is a stunningly simple algorithm and it works because of an asymmetry:
    a successful shortcut is a permanent improvement, and a failed one costs
    only one collision check.  You are buying lottery tickets whose losses are
    tiny and whose wins never expire.
    """
    pts = list(resample(path, spacing)) if spacing else [np.asarray(p, float)
                                                        for p in path]
    hist = [(0, path_cost(pts))]
    tries = 0
    for it in range(iters):
        if len(pts) < 3:
            break
        i = int(rng.integers(0, len(pts) - 2))
        j = int(rng.integers(i + 2, len(pts)))
        tries += 1
        if seg_free(pts[i], pts[j]):
            pts = pts[:i + 1] + pts[j:]
        if track:
            hist.append((it + 1, path_cost(pts)))
    return (pts, hist) if track else pts


def partial_shortcut(path, seg_free, rng, iters=200, spacing=None, track=False):
    """Shortcut ONE joint at a time, leaving the others alone.

    On a 7-joint arm a full shortcut has to be legal for all seven joints at
    once, so it is rejected as soon as any single joint's motion would clip an
    obstacle.  A partial shortcut straightens joint 3 while joints 1, 2 and
    4-7 keep their existing (safe) motion, so it can still make progress in
    exactly the situations where the full version always fails.

    A beginner may ask why this is not simply worse -- surely straightening one
    joint removes less length than straightening all seven?  Per successful
    attempt, yes.  But the ACCEPTANCE RATE is far higher, and total improvement
    is (rate x size).  Experiment 3 measures which term wins.
    """
    pts = resample(path, spacing) if spacing else np.asarray(path, float).copy()
    pts = np.asarray(pts, float)
    d = pts.shape[1]
    hist = [(0, path_cost(pts))]
    for it in range(iters):
        if len(pts) < 3:
            break
        i = int(rng.integers(0, len(pts) - 2))
        j = int(rng.integers(i + 2, len(pts)))
        k = int(rng.integers(0, d))
        cand = pts.copy()
        cand[i:j + 1, k] = np.linspace(pts[i, k], pts[j, k], j - i + 1)
        if all(seg_free(cand[m], cand[m + 1]) for m in range(i, j)):
            pts = cand
        if track:
            hist.append((it + 1, path_cost(pts)))
    out = [p for p in pts]
    return (out, hist) if track else out


def blend_corners(path, radius, seg_free=None, samples=8):
    """Round every corner with a circular-ish arc of the given radius.

    A shortcut path is made of straight lines meeting at sharp corners.  A
    sharp corner is a problem no length metric can see: to follow it exactly
    the robot has to change direction instantly, which needs infinite
    acceleration, so in practice it has to stop dead at every corner.
    Replacing each corner with a short arc costs a little extra length and
    buys back the ability to drive through it at speed -- which is what
    project 36 then exploits.
    """
    p = np.asarray(path, float)
    if len(p) < 3:
        return p.copy()
    out = [p[0]]
    for i in range(1, len(p) - 1):
        a, b, c = p[i - 1], p[i], p[i + 1]
        la = np.linalg.norm(b - a)
        lc = np.linalg.norm(c - b)
        r = min(radius, 0.45 * la, 0.45 * lc)
        if r <= 1e-9:
            out.append(b)
            continue
        pa = b + (a - b) * (r / la)
        pc = b + (c - b) * (r / lc)
        arc = [(1 - t) ** 2 * pa + 2 * (1 - t) * t * b + t ** 2 * pc
               for t in np.linspace(0, 1, samples)]      # quadratic Bezier
        if seg_free is not None and not all(
                seg_free(arc[m], arc[m + 1]) for m in range(len(arc) - 1)):
            out.append(b)
            continue
        out.extend(arc)
    out.append(p[-1])
    return np.asarray(out)


def max_turn(path):
    """Sharpest direction change along a path, in radians.

    pi means the path doubles back on itself; 0 means perfectly straight.
    """
    p = np.asarray(path, float)
    if len(p) < 3:
        return 0.0
    v = np.diff(p, axis=0)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    u = v / n
    dots = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return float(np.max(np.arccos(dots)))


def min_turn_radius(path, eps=1e-9):
    """Tightest bend along the path, as a radius in metres.

    For each triple of consecutive points we fit the unique circle through
    them; its radius is R = abc / (4 * area) for a triangle with sides a, b, c.
    A perfectly straight stretch has zero area and infinite radius; a hairpin
    has almost zero radius.

    This is the metric that matters for driving, and it is the one that
    "sharpest turn angle" gets wrong: the angle between consecutive segments
    depends on how finely the path happens to be sampled, while the radius
    does not.
    """
    p = np.asarray(path, float)
    if len(p) < 3:
        return math.inf
    best = math.inf
    for i in range(len(p) - 2):
        a = np.linalg.norm(p[i + 1] - p[i])
        b = np.linalg.norm(p[i + 2] - p[i + 1])
        c = np.linalg.norm(p[i + 2] - p[i])
        if min(a, b, c) < eps:
            continue
        s = 0.5 * (a + b + c)
        area2 = max(0.0, s * (s - a) * (s - b) * (s - c))
        area = math.sqrt(area2)
        if area < eps:
            continue                      # collinear: infinite radius
        best = min(best, a * b * c / (4.0 * area))
    return best


def turn_profile(path):
    p = np.asarray(path, float)
    v = np.diff(p, axis=0)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    u = v / n
    dots = np.clip(np.sum(u[:-1] * u[1:], axis=1), -1.0, 1.0)
    return np.arccos(dots)


def side_signature(path, centres):
    """Which way did the path go round each obstacle?  (2D only.)

    For every obstacle centre we accumulate the signed angle the path sweeps
    around it.  A path that passes above a circle sweeps roughly +pi; one that
    passes below sweeps roughly -pi.  Two paths with different signatures
    cannot be deformed into each other without crossing the obstacle -- they
    are in different HOMOTOPY CLASSES.  ("Homotopy" is the mathematical word
    for continuous deformation: two paths are homotopic if you can slide one
    onto the other without ever cutting through an obstacle.)
    """
    p = np.asarray(path, float)
    sig = []
    for c in np.atleast_2d(centres):
        d = p - c
        ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        sig.append(np.sign(ang[-1] - ang[0]) if abs(ang[-1] - ang[0]) > 0.6
                   else 0.0)
    return tuple(sig)
