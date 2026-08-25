"""Grid maps and best-first search -- the shared library for Phase 5, project 31.

Everything in this file is deliberately one algorithm with knobs, not five
algorithms.  Dijkstra, A*, weighted A* and greedy best-first search differ only
in how the priority `f` is built from `g` (cost so far) and `h` (estimate of
cost to go):

    Dijkstra        f = g              (h switched off)
    A*              f = g + h
    weighted A*     f = g + eps*h      (eps > 1)
    greedy          f = h              (g switched off)

So `search()` takes `heuristic` and `weight` and covers all four.  That is not
a code-golf trick -- it is the actual relationship between them, and seeing it
in one function is the point.

Imported by project 34 (shortcut smoothing) and project 38 (footstep planning).
"""

import heapq
import math
import time

import numpy as np

SQRT2 = math.sqrt(2.0)


# ---------------------------------------------------------------- maps
def blob_map(h, w, rng, n_blobs=26, r_lo=3, r_hi=9, border=2):
    """Open field with round obstacles scattered in it.

    This is the easy case for a heuristic: the straight line to the goal is
    almost always nearly correct, so the estimate is informative everywhere.
    """
    grid = np.zeros((h, w), dtype=bool)
    yy, xx = np.mgrid[0:h, 0:w]
    for _ in range(n_blobs):
        cy = rng.integers(border, h - border)
        cx = rng.integers(border, w - border)
        r = rng.integers(r_lo, r_hi)
        grid |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    return grid


def maze_map(h, w, rng):
    """Perfect maze (exactly one route between any two cells).

    Built with the standard depth-first "recursive backtracker" on a lattice of
    odd coordinates: walk to a random unvisited neighbour two cells away and
    knock out the wall between.  A maze is the hard case for a heuristic --
    the straight-line distance to the goal says almost nothing about the real
    distance, because the real route is forced to zig-zag.
    """
    grid = np.ones((h, w), dtype=bool)
    start = (1, 1)
    grid[start] = False
    stack = [start]
    while stack:
        y, x = stack[-1]
        nbrs = []
        for dy, dx in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            ny, nx = y + dy, x + dx
            if 1 <= ny < h - 1 and 1 <= nx < w - 1 and grid[ny, nx]:
                nbrs.append((ny, nx, dy, dx))
        if not nbrs:
            stack.pop()
            continue
        ny, nx, dy, dx = nbrs[rng.integers(len(nbrs))]
        grid[y + dy // 2, x + dx // 2] = False
        grid[ny, nx] = False
        stack.append((ny, nx))
    return grid


def free_cells(grid):
    ys, xs = np.nonzero(~grid)
    return list(zip(ys.tolist(), xs.tolist()))


def random_query(grid, rng, min_sep=None):
    """Pick a random free start and a random free goal that are far apart."""
    cells = free_cells(grid)
    h, w = grid.shape
    if min_sep is None:
        min_sep = 0.5 * math.hypot(h, w)
    for _ in range(400):
        a = cells[rng.integers(len(cells))]
        b = cells[rng.integers(len(cells))]
        if math.dist(a, b) >= min_sep:
            return a, b
    return cells[0], cells[-1]


# ---------------------------------------------------------------- heuristics
def h_zero(a, b):
    """No estimate at all.  Plugging this into A* gives you Dijkstra."""
    return 0.0


def h_manhattan(a, b):
    """|dy| + |dx| -- the distance if you may only step N/S/E/W.

    Named after the street grid of Manhattan, where you cannot cut across a
    block: you walk so far east, then so far north.  Also called the L1 norm,
    because it is the p = 1 case of the general Lp formula
    (sum |v_i|^p)^(1/p).
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def h_euclidean(a, b):
    """Straight-line distance -- the L2 norm (p = 2 in the same Lp formula)."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def h_chebyshev(a, b):
    """max(|dy|, |dx|) -- the L-infinity norm, the p -> infinity limit.

    This is the distance if a diagonal step were as cheap as a straight one
    (a king on a chessboard).  Named after Pafnuty Chebyshev.
    """
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def h_octile(a, b):
    """The exactly-correct distance on an empty 8-connected grid.

    Take as many diagonal steps as you can (each costs sqrt(2) and eats one
    unit of BOTH dy and dx), then walk the leftover in a straight line:

        d = (sqrt(2) - 1) * min(dy, dx) + max(dy, dx)

    "Octile" because there are eight directions to move in.  On an empty grid
    this is not an estimate, it is the answer -- which is exactly what you
    want a heuristic to be.
    """
    dy = abs(a[0] - b[0])
    dx = abs(a[1] - b[1])
    return (SQRT2 - 1.0) * min(dy, dx) + max(dy, dx)


HEURISTICS = {
    "zero": h_zero,
    "manhattan": h_manhattan,
    "euclidean": h_euclidean,
    "chebyshev": h_chebyshev,
    "octile": h_octile,
}

# Neighbour offsets and their step costs.
NBR4 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]
NBR8 = NBR4 + [(-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


# ---------------------------------------------------------------- search
def search(grid, start, goal, heuristic="octile", weight=1.0, conn=8,
           tie_break="low_h", greedy=False, max_expansions=None):
    """One best-first search that covers Dijkstra, A*, weighted A* and greedy.

    Returns a dict with the path, its cost, and the counters that matter:
    `expanded` (nodes popped off the open list and settled) and `generated`
    (nodes ever pushed).  `expanded` is the honest measure of search effort --
    it is how many times the algorithm asked "what is next to this cell?".

    tie_break:
      "none"    -- ties in f are broken by insertion order.
      "low_h"   -- among equal f, prefer the node closer to the goal.  This is
                   free (h is already computed) and usually the single
                   cheapest speed-up on an open map.
    """
    hf = HEURISTICS[heuristic] if isinstance(heuristic, str) else heuristic
    nbrs = NBR8 if conn == 8 else NBR4
    h, w = grid.shape
    t0 = time.perf_counter()

    g = {start: 0.0}
    parent = {start: None}
    closed = np.zeros(grid.shape, dtype=bool)
    counter = 0
    h0 = hf(start, goal)
    open_heap = [((0.0 if greedy else 0.0) + weight * h0, h0, counter, start)]
    expanded = 0
    generated = 1

    while open_heap:
        f, _, _, node = heapq.heappop(open_heap)
        if closed[node]:
            continue
        closed[node] = True
        expanded += 1
        if node == goal:
            path = []
            n = node
            while n is not None:
                path.append(n)
                n = parent[n]
            path.reverse()
            return dict(path=path, cost=g[goal], expanded=expanded,
                        generated=generated, time=time.perf_counter() - t0,
                        found=True, closed=closed)
        if max_expansions and expanded > max_expansions:
            break
        y, x = node
        gn = g[node]
        for dy, dx, step in nbrs:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w) or grid[ny, nx] or closed[ny, nx]:
                continue
            # No corner cutting: a diagonal step is only legal if both of the
            # straight steps that bracket it are also free.  Without this the
            # robot slips through the gap between two obstacles that touch at
            # a corner -- a zero-width gap in the real world.
            if dy and dx and (grid[y + dy, x] or grid[y, x + dx]):
                continue
            nb = (ny, nx)
            ng = gn + step
            if ng < g.get(nb, math.inf) - 1e-12:
                g[nb] = ng
                parent[nb] = node
                hv = hf(nb, goal)
                counter += 1
                generated += 1
                pri = weight * hv if greedy else ng + weight * hv
                # tie_break "low_h": h rides in the second slot of the tuple,
                # so equal-f nodes are ordered by how close they look to goal.
                second = hv if tie_break == "low_h" else counter
                heapq.heappush(open_heap, (pri, second, counter, nb))

    return dict(path=None, cost=math.inf, expanded=expanded,
                generated=generated, time=time.perf_counter() - t0,
                found=False, closed=closed)


def path_length(path):
    """True Euclidean length of a cell path (diagonals counted as sqrt(2))."""
    if not path or len(path) < 2:
        return 0.0
    p = np.asarray(path, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def line_of_sight(grid, a, b, samples_per_cell=3):
    """Is the straight segment a->b free?  Used to measure how much longer a
    grid path is than the path a robot could actually drive."""
    n = int(max(2, samples_per_cell * math.dist(a, b)))
    ys = np.linspace(a[0], b[0], n)
    xs = np.linspace(a[1], b[1], n)
    yi = np.clip(np.round(ys).astype(int), 0, grid.shape[0] - 1)
    xi = np.clip(np.round(xs).astype(int), 0, grid.shape[1] - 1)
    return not grid[yi, xi].any()


def shortcut_los(grid, path, rng, iters=200):
    """Any-angle post-process: repeatedly try to replace a slice of the grid
    path with a straight segment.  Used in experiment 7 to measure how much of
    a grid path's length is an artefact of the grid rather than the obstacles.
    """
    pts = [tuple(p) for p in path]
    for _ in range(iters):
        if len(pts) < 3:
            break
        i = rng.integers(0, len(pts) - 2)
        j = rng.integers(i + 2, len(pts))
        if line_of_sight(grid, pts[i], pts[j]):
            pts = pts[:i + 1] + pts[j:]
    return pts
