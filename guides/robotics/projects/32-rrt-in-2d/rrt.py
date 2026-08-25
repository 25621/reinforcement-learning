"""Sampling-based planning in the plane -- the shared library for project 32.

The environment is a 2D box with circular and rectangular obstacles, and the
"robot" is a point.  That sounds like a toy, and it is, but every line here
survives unchanged into a 7-joint arm (project 33): the only thing that
changes is what `in_collision` means and how many numbers are in a
configuration.  That is the whole selling point of sampling-based planning --
it never looks at the shape of the obstacles, only at a yes/no answer.

Imported by project 34 (shortcut smoothing).
"""

import math
import time

import numpy as np


class Env:
    """A 2D world.  `lo`/`hi` are the box; obstacles are circles and boxes."""

    def __init__(self, lo=(0.0, 0.0), hi=(10.0, 10.0), circles=None, rects=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = np.asarray(hi, dtype=float)
        self.circles = np.asarray(circles, dtype=float).reshape(-1, 3) \
            if circles is not None else np.zeros((0, 3))
        self.rects = np.asarray(rects, dtype=float).reshape(-1, 4) \
            if rects is not None else np.zeros((0, 4))
        self.n_checks = 0

    def sample(self, rng):
        return self.lo + rng.random(2) * (self.hi - self.lo)

    def points_free(self, pts):
        """Vectorised: which of these points are outside every obstacle?

        Doing this for a whole batch at once is the single biggest speed-up in
        the file, because the collision check is called far more often than
        anything else -- see experiment 7.
        """
        pts = np.atleast_2d(pts)
        ok = np.all((pts >= self.lo) & (pts <= self.hi), axis=1)
        if len(self.circles):
            d2 = ((pts[:, None, :] - self.circles[None, :, :2]) ** 2).sum(-1)
            ok &= np.all(d2 > self.circles[None, :, 2] ** 2, axis=1)
        if len(self.rects):
            r = self.rects
            inside = ((pts[:, None, 0] >= r[None, :, 0]) &
                      (pts[:, None, 0] <= r[None, :, 2]) &
                      (pts[:, None, 1] >= r[None, :, 1]) &
                      (pts[:, None, 1] <= r[None, :, 3]))
            ok &= ~np.any(inside, axis=1)
        return ok

    def free(self, q):
        self.n_checks += 1
        return bool(self.points_free(np.asarray(q, float)[None, :])[0])

    def segment_free(self, a, b, res=0.05):
        """Is the straight segment a->b collision-free?

        We do NOT have a true continuous check; we sample the segment every
        `res` metres and test those points.  That is what every real planner
        does too, and it is a real approximation -- a thin obstacle narrower
        than `res` can be stepped straight through.  Experiment 3 of project 33
        measures exactly that failure.
        """
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        n = int(max(2, math.ceil(np.linalg.norm(b - a) / res) + 1))
        ts = np.linspace(0.0, 1.0, n)[:, None]
        pts = a[None, :] * (1 - ts) + b[None, :] * ts
        self.n_checks += n
        return bool(self.points_free(pts).all())


# ---------------------------------------------------------------- worlds
def world_blobs(rng, n=14, r_lo=0.5, r_hi=1.2):
    circles = []
    while len(circles) < n:
        c = rng.uniform(1.0, 9.0, 2)
        r = rng.uniform(r_lo, r_hi)
        if np.linalg.norm(c - np.array([0.5, 0.5])) < r + 0.6:
            continue
        if np.linalg.norm(c - np.array([9.5, 9.5])) < r + 0.6:
            continue
        circles.append([c[0], c[1], r])
    return Env(circles=circles)


def world_narrow(gap=1.0, thickness=3.0):
    """A thick wall across the middle with a single corridor of width `gap`.

    The corridor's width is the one number that decides whether a sampling
    planner works, because the chance of a uniform sample landing inside it is
    proportional to its AREA.  Halve the width and you halve the chance.

    The wall is deliberately thick.  A thin wall is easy even when the gap is
    narrow, because the planner can step across it in one move from a node on
    either side.  A thick wall forces several consecutive nodes to land inside
    the corridor, and the probability of that is the gap fraction raised to the
    number of nodes needed -- which is why narrow passages fail exponentially,
    not gradually.
    """
    y0 = 5.0 - thickness / 2.0
    y1 = 5.0 + thickness / 2.0
    half = gap / 2.0
    rects = [[0.0, y0, 5.0 - half, y1], [5.0 + half, y0, 10.0, y1]]
    return Env(rects=rects)


def world_trap():
    """A C-shaped trap around the start: greedy behaviour walks into it."""
    rects = [[1.0, 1.0, 6.0, 1.5], [1.0, 1.0, 1.5, 6.0], [1.0, 5.5, 6.0, 6.0]]
    return Env(rects=rects)


# ---------------------------------------------------------------- planners
class Tree:
    def __init__(self, root, cap=200000):
        self.pts = np.empty((cap, len(root)))
        self.pts[0] = root
        self.parent = [-1]
        self.cost = [0.0]
        self.n = 1

    def add(self, q, parent, cost):
        self.pts[self.n] = q
        self.parent.append(parent)
        self.cost.append(cost)
        self.n += 1
        return self.n - 1

    def nearest(self, q):
        d = np.linalg.norm(self.pts[:self.n] - q, axis=1)
        return int(np.argmin(d)), float(d.min())

    def near(self, q, r):
        d = np.linalg.norm(self.pts[:self.n] - q, axis=1)
        return np.nonzero(d <= r)[0]

    def path_to(self, idx):
        out = []
        while idx != -1:
            out.append(self.pts[idx].copy())
            idx = self.parent[idx]
        return out[::-1]


def steer(a, b, step):
    """Move from a toward b, but no further than `step`.

    This is why the tree is "rapidly-exploring": every extension is a bounded
    step, so the tree grows outward at a controlled rate instead of jumping.
    """
    d = b - a
    n = np.linalg.norm(d)
    if n <= step:
        return b.copy()
    return a + (step / n) * d


def rrt(env, start, goal, rng, step=0.5, goal_bias=0.05, max_iters=8000,
        goal_tol=0.3, res=0.05, stop_at_goal=True, record_every=0):
    """Plain RRT.  Returns (tree, path, stats).

    `stop_at_goal=False` keeps sampling after the first solution and reports
    the best goal connection found so far.  That is the fair way to compare
    against RRT*: otherwise "RRT does not improve" would just be a restatement
    of "RRT stopped".  Experiment 4 uses it.

    "Rapidly-exploring Random Tree": random because samples are drawn
    uniformly, tree because every node has exactly one parent, and
    rapidly-exploring because of a subtle bias -- the nearest node to a uniform
    sample is almost always a node on the FRONTIER, since frontier nodes own
    the biggest share of the space.  The tree therefore pulls itself outward
    into unexplored regions without anyone telling it to.
    """
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    t0 = time.perf_counter()
    env.n_checks = 0
    tree = Tree(start)
    extended = 0
    best_goal, best_cost, first_it = -1, math.inf, 0
    history = []
    for it in range(max_iters):
        target = goal if rng.random() < goal_bias else env.sample(rng)
        ni, _ = tree.nearest(target)
        new = steer(tree.pts[ni], target, step)
        if not env.segment_free(tree.pts[ni], new, res):
            if record_every and (it + 1) % record_every == 0:
                history.append((it + 1, best_cost))
            continue
        extended += 1
        idx = tree.add(new, ni, tree.cost[ni] + np.linalg.norm(new - tree.pts[ni]))
        if np.linalg.norm(new - goal) <= goal_tol and \
                env.segment_free(new, goal, res):
            gc = tree.cost[idx] + np.linalg.norm(goal - new)
            if gc < best_cost:
                best_goal = tree.add(goal, idx, gc)
                best_cost = gc
                if not first_it:
                    first_it = it + 1
            if stop_at_goal:
                return tree, tree.path_to(best_goal), dict(
                    iters=it + 1, nodes=tree.n, extended=extended,
                    checks=env.n_checks, time=time.perf_counter() - t0,
                    cost=best_cost, found=True, history=history)
        if record_every and (it + 1) % record_every == 0:
            history.append((it + 1, best_cost))
    return tree, (tree.path_to(best_goal) if best_goal >= 0 else None), dict(
        iters=max_iters, nodes=tree.n, extended=extended, checks=env.n_checks,
        time=time.perf_counter() - t0, cost=best_cost, found=best_goal >= 0,
        history=history, first_iter=first_it)


def rrt_star(env, start, goal, rng, step=0.5, goal_bias=0.05, max_iters=8000,
             goal_tol=0.3, res=0.05, gamma=6.0, record_every=0):
    """RRT* -- same tree, plus two extra moves per sample.

    1. CHOOSE PARENT: among all nodes within a shrinking radius of the new
       node, attach to the one that gives the cheapest total cost, not simply
       the nearest one.
    2. REWIRE: check whether any of those same neighbours would now be cheaper
       if they went THROUGH the new node, and re-parent them if so.

    A beginner should ask why (2) is needed when (1) already picked the best
    parent.  They fix different things.  (1) makes the NEW node cheap given the
    tree as it stands.  (2) lets an OLD node profit from information that did
    not exist when it was added -- the tree keeps improving paths it committed
    to earlier.  Without rewiring, an early bad decision is permanent, which is
    exactly why plain RRT never converges to the optimum.

    The radius shrinks like (log n / n)^(1/d).  That is the rate at which you
    can keep the neighbour count roughly constant while the samples get denser
    -- cheap enough to run forever, wide enough to keep finding improvements.
    """
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    d = len(start)
    t0 = time.perf_counter()
    env.n_checks = 0
    tree = Tree(start)
    best_goal, best_cost = -1, math.inf
    history = []
    for it in range(max_iters):
        target = goal if rng.random() < goal_bias else env.sample(rng)
        ni, _ = tree.nearest(target)
        new = steer(tree.pts[ni], target, step)
        if not env.segment_free(tree.pts[ni], new, res):
            continue
        r = min(step * 3.0, gamma * (math.log(tree.n + 1) / (tree.n + 1)) ** (1.0 / d))
        cand = tree.near(new, r)
        if len(cand) == 0:
            cand = np.array([ni])
        # 1. choose the cheapest legal parent
        best_p, best_c = ni, tree.cost[ni] + np.linalg.norm(new - tree.pts[ni])
        order = np.argsort([tree.cost[c] + np.linalg.norm(new - tree.pts[c])
                            for c in cand])
        for k in order:
            c = int(cand[k])
            cc = tree.cost[c] + np.linalg.norm(new - tree.pts[c])
            if cc >= best_c:
                break
            if env.segment_free(tree.pts[c], new, res):
                best_p, best_c = c, cc
                break
        idx = tree.add(new, best_p, best_c)
        # 2. rewire the neighbourhood through the new node
        for c in cand:
            c = int(c)
            if c == best_p:
                continue
            cc = best_c + np.linalg.norm(tree.pts[c] - new)
            if cc < tree.cost[c] - 1e-9 and env.segment_free(new, tree.pts[c], res):
                tree.parent[c] = idx
                tree.cost[c] = cc
        if np.linalg.norm(new - goal) <= goal_tol and \
                env.segment_free(new, goal, res):
            gc = best_c + np.linalg.norm(goal - new)
            if gc < best_cost:
                best_goal = tree.add(goal, idx, gc)
                best_cost = gc
        if record_every and (it + 1) % record_every == 0:
            history.append((it + 1, best_cost if best_goal >= 0 else math.inf))
    path = tree.path_to(best_goal) if best_goal >= 0 else None
    return tree, path, dict(iters=max_iters, nodes=tree.n, checks=env.n_checks,
                            time=time.perf_counter() - t0, cost=best_cost,
                            found=best_goal >= 0, history=history)


def path_cost(path):
    if path is None or len(path) < 2:
        return math.inf
    p = np.asarray(path)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
