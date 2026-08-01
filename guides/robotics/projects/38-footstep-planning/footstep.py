"""Footstep planning over a lattice -- the shared library for project 38.

A walking robot has an enormous continuous state (every joint, every joint
rate, the floating base) and a plan that lasts many seconds.  Planning all of
that at once is hopeless.  Footstep planning throws almost all of it away and
keeps one thing: WHERE THE FEET GO.

    state  = (position of the stance foot, which foot it is)
    action = a discrete choice of where to put the other foot next

That turns walking into a graph search, which project 31's A* already solves.
Everything the robot does between two footfalls -- swinging the leg, shifting
its weight, bending its knees -- is delegated to a lower layer that is assumed
to be able to execute any step the planner declares legal.  Experiment 7 is
about what happens when that assumption is not quite true.

The plane here is: x forward, y sideways.  The robot always faces +x, so there
is no heading in the state.  Adding one is a fourth dimension and changes
nothing conceptually.
"""

import heapq
import math
import time

import numpy as np


# ------------------------------------------------------------------ terrain
class Terrain:
    """A set of axis-aligned rectangular stepping stones."""

    def __init__(self, stones):
        self.stones = np.asarray(stones, float).reshape(-1, 4)  # x0,y0,x1,y1

    def supports(self, x, y):
        """Is the point (x, y) on some stone?  Vectorised over arrays."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        s = self.stones
        hit = ((x[:, None] >= s[None, :, 0]) & (x[:, None] <= s[None, :, 2]) &
               (y[:, None] >= s[None, :, 1]) & (y[:, None] <= s[None, :, 3]))
        return np.any(hit, axis=1)

    def supports_one(self, x, y):
        return bool(self.supports(x, y)[0])

    def area(self):
        s = self.stones
        return float(np.sum((s[:, 2] - s[:, 0]) * (s[:, 3] - s[:, 1])))


def flat_ground(x0=-0.5, x1=6.0, y0=-1.0, y1=1.0):
    return Terrain([[x0, y0, x1, y1]])


def stepping_stones(rng, n=26, x_end=4.0, size=0.16, y_spread=0.55,
                    jitter=0.10):
    """A staggered line of stones, left and right alternating, plus noise.

    A purely random scatter is a bad test: it is either trivially crossable or
    obviously impossible.  Staggering the stones roughly where a walking robot
    would want to put its feet makes the terrain solvable but not by accident,
    which is where the planner has something to do.
    """
    stones = [[-0.55, -0.65, -0.05, 0.65]]        # the starting platform
    x = 0.15
    side = 1
    while x < x_end:
        y = side * (0.16 + rng.uniform(0, y_spread * 0.5))
        xx = x + rng.uniform(-jitter, jitter)
        stones.append([xx - size / 2, y - size / 2, xx + size / 2, y + size / 2])
        x += rng.uniform(0.20, 0.34)
        side *= -1
    stones.append([x_end, -0.65, x_end + 0.6, 0.65])     # the far platform
    return Terrain(stones)


# ------------------------------------------------------------------ actions
class Robot:
    """Kinematic limits of one step, and the cost of taking it."""

    def __init__(self, dx_min=-0.12, dx_max=0.50, dy_min=0.12, dy_max=0.42,
                 w_len=1.0, w_step=0.35, w_lat=0.6, w_quad=0.0,
                 y_nominal=0.20):
        self.dx_min, self.dx_max = dx_min, dx_max
        self.dy_min, self.dy_max = dy_min, dy_max
        self.w_len, self.w_step, self.w_lat = w_len, w_step, w_lat
        self.w_quad = w_quad
        self.y_nominal = y_nominal

    def actions(self, n_x=7, n_y=4):
        """The lattice of candidate steps, in the stance foot's frame.

        `dy` is always POSITIVE here and gets its sign from which foot is
        swinging, which is what stops the planner from crossing the legs: a
        left foot may only ever land to the left of the right foot.
        """
        dxs = np.linspace(self.dx_min, self.dx_max, n_x)
        dys = np.linspace(self.dy_min, self.dy_max, n_y)
        return [(float(dx), float(dy)) for dx in dxs for dy in dys]

    def step_cost(self, dx, dy):
        """Four terms, and each one pushes the gait in a different direction.

          w_step  a flat charge for taking a step at all -> fewer, longer strides
          w_len   proportional to step length            -> also fewer strides
          w_quad  proportional to length SQUARED         -> shorter strides
          w_lat   deviation from a nominal stance width  -> narrower stance

        The quadratic term is the one that makes this interesting.  With only
        the linear terms, "shortest total length" and "fewest steps" both want
        the longest stride the legs allow, so every weight setting produces the
        same gait.  Real walking energy grows faster than linearly with stride
        length -- doubling the stride costs much more than twice as much -- and
        it is that superlinearity that gives an interior optimum.
        """
        return (self.w_step +
                self.w_len * math.hypot(dx, dy) +
                self.w_quad * (dx * dx + dy * dy) +
                self.w_lat * abs(abs(dy) - self.y_nominal))


# ------------------------------------------------------------------ A*
def plan(terrain, robot, start_xy, start_foot, goal_x, res=0.05,
         actions=None, heuristic="linear", max_expansions=400000,
         y_limit=1.1):
    """A* over the footstep lattice.

    Three heuristics, because the difference between them is a real lesson.

    "none"    h = 0.  This is Dijkstra.

    "linear"  h = (remaining forward distance / dx_max) * (cheapest step).
        Every step advances at most `dx_max` and costs at least
        `w_step + w_len * dx_max`, so this can never overestimate: it is
        ADMISSIBLE, and A* with an admissible heuristic returns an optimal
        path.  It is also CONSISTENT (h never drops by more than the cost of
        the step that caused the drop), which is the stronger property A*
        needs when it refuses to revisit closed states -- as this
        implementation does.

    "ceil"    h = ceil(remaining / dx_max) * (cheapest step).
        Tempting, because you cannot take a fractional step, and still
        admissible.  But it is NOT consistent: a tiny step that happens to
        cross a ceiling boundary makes h fall by a whole step's worth while
        costing almost nothing.  Experiment 2 measures what that does.
    """
    acts = actions if actions is not None else robot.actions()
    cheapest = robot.w_step + robot.w_len * robot.dx_max

    def h(x):
        rem = max(0.0, goal_x - x)
        if heuristic in (False, "none"):
            return 0.0
        if heuristic == "ceil":
            return math.ceil(rem / robot.dx_max) * cheapest
        return (rem / robot.dx_max) * cheapest

    def key(x, y, foot):
        return (int(round(x / res)), int(round(y / res)), foot)

    t0 = time.perf_counter()
    s0 = (float(start_xy[0]), float(start_xy[1]), start_foot)
    g = {key(*s0): 0.0}
    parent = {key(*s0): None}
    state = {key(*s0): s0}
    open_heap = [(h(s0[0]), 0, key(*s0))]
    closed = set()
    counter = 0
    expanded = 0

    while open_heap:
        f, _, k = heapq.heappop(open_heap)
        if k in closed:
            continue
        closed.add(k)
        expanded += 1
        x, y, foot = state[k]
        if x >= goal_x:
            path = []
            kk = k
            while kk is not None:
                path.append(state[kk])
                kk = parent[kk]
            return dict(path=path[::-1], cost=g[k], expanded=expanded,
                        time=time.perf_counter() - t0, found=True)
        if expanded > max_expansions:
            break
        gx = g[k]
        for dx, dy in acts:
            # the swinging foot is the other one; it lands on its own side
            new_foot = "R" if foot == "L" else "L"
            sy = dy if new_foot == "L" else -dy
            # Snap the landing point onto the lattice.  Without this the state
            # and its dictionary key drift apart: two slightly different
            # continuous positions round to the same key, the first one to
            # arrive claims it, and the search silently loses the other -- an
            # aliasing bug that makes a FINER action set perform WORSE.
            nx = round((x + dx) / res) * res
            ny = round((y + sy) / res) * res
            if abs(ny) > y_limit:
                continue
            if not terrain.supports_one(nx, ny):
                continue
            nk = key(nx, ny, new_foot)
            if nk in closed:
                continue
            ng = gx + robot.step_cost(dx, dy)
            if ng < g.get(nk, math.inf) - 1e-12:
                g[nk] = ng
                parent[nk] = k
                state[nk] = (nx, ny, new_foot)
                counter += 1
                heapq.heappush(open_heap, (ng + h(nx), counter, nk))

    return dict(path=None, cost=math.inf, expanded=expanded,
                time=time.perf_counter() - t0, found=False)


def greedy(terrain, robot, start_xy, start_foot, goal_x, actions=None,
           max_steps=200):
    """The baseline everyone tries first: always take the step that gets you
    furthest forward.  It has no way to back out of a dead end, and stepping
    stones are full of dead ends."""
    acts = actions if actions is not None else robot.actions()
    x, y, foot = float(start_xy[0]), float(start_xy[1]), start_foot
    path = [(x, y, foot)]
    for _ in range(max_steps):
        if x >= goal_x:
            return dict(path=path, found=True, cost=len(path))
        best = None
        for dx, dy in acts:
            new_foot = "R" if foot == "L" else "L"
            sy = dy if new_foot == "L" else -dy
            nx, ny = x + dx, y + sy
            if not terrain.supports_one(nx, ny):
                continue
            if best is None or nx > best[0]:
                best = (nx, ny, new_foot)
        if best is None:
            return dict(path=path, found=False, cost=math.inf)
        x, y, foot = best
        path.append((x, y, foot))
    return dict(path=path, found=False, cost=math.inf)


def path_stats(path, robot):
    if not path or len(path) < 2:
        return dict(steps=0, mean_len=0.0, max_len=0.0, mean_lat=0.0)
    steps = []
    for a, b in zip(path[:-1], path[1:]):
        steps.append((b[0] - a[0], b[1] - a[1]))
    ln = [math.hypot(*s) for s in steps]
    return dict(steps=len(steps), mean_len=float(np.mean(ln)),
                max_len=float(np.max(ln)),
                mean_forward=float(np.mean([s[0] for s in steps])),
                mean_lat=float(np.mean([abs(s[1]) for s in steps])))


# ------------------------------------------------------------------ dynamics
def capture_points(path, step_time=0.6, com_height=0.8, g=9.81):
    """Where the robot would have to step to come to a stop after each step.

    Under the LINEAR INVERTED PENDULUM model, the body is a point mass at a
    fixed height balancing over the stance foot, and its horizontal motion
    obeys  xddot = omega^2 (x - p),  with p the stance foot and

        omega = sqrt(g / height)

    The combination  xi = x + xdot / omega  is the CAPTURE POINT (also called
    the divergent component of motion).  It is special because it is the ONLY
    part of the state that runs away: put the next foot exactly there and the
    body comes to rest over it; put it anywhere else and it keeps going.

    Note what this ignores -- swing-leg mass, ankle torque, arm motion, the
    fact that the height is not really constant.  It is a deliberately crude
    model, and it is still enough to catch plans the kinematic planner would
    happily hand over.
    """
    omega = math.sqrt(g / com_height)
    c = math.cosh(omega * step_time)
    s = math.sinh(omega * step_time)
    out = []
    for a, b in zip(path[:-1], path[1:]):
        p = np.array([a[0], a[1]])           # stance foot for this step
        target = np.array([b[0], b[1]])      # where the swing foot will land
        # Simplification: take the body to start directly over the stance foot.
        # Then  x(t) = p + (v0/omega) sinh(omega t), so landing the body on the
        # next foot after `step_time` fixes v0 completely.
        v0 = omega * (target - p) / s
        v_T = v0 * c
        xi = target + v_T / omega            # capture point at the end
        out.append(dict(v0=float(np.linalg.norm(v0)),
                        vT=float(np.linalg.norm(v_T)),
                        capture=xi))
    return out


def dynamic_check(terrain, path, step_time=0.6, com_height=0.8, v_max=1.6):
    """How many steps in this plan are dynamically uncomfortable?

    Two tests per step:
      * the body speed the LIP needs to cover the step in `step_time`;
      * whether the capture point at the end of the step lands on a stone --
        if it does not, the robot is committed: it cannot stop there even if
        it wants to.
    """
    cps = capture_points(path, step_time, com_height)
    fast = sum(1 for c in cps if c["vT"] > v_max)
    uncapturable = sum(1 for c in cps
                       if not terrain.supports_one(c["capture"][0],
                                                   c["capture"][1]))
    return dict(steps=len(cps), too_fast=fast, uncapturable=uncapturable,
                max_speed=max((c["vT"] for c in cps), default=0.0))
