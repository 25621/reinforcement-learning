"""The Dynamic Window Approach, a costmap, and the layered navigation stack.

The "dynamic window" is two ideas stacked in one name:

  * *window*  -- instead of asking "where should I go", ask "which (v, omega)
                 pair should I send in the next 100 ms".  The search space is
                 two numbers, not a path.
  * *dynamic* -- the window is not the robot's full speed range.  It is only
                 the part reachable within one control period given the
                 acceleration limits.  A robot at 1.0 m/s that can decelerate
                 at 2 m/s^2 cannot be at 0 m/s in 100 ms, so 0 is not in the
                 window and there is no point scoring it.

Everything is vectorised: all candidate velocity pairs are rolled out at once
as arrays, which is what makes a 30x30 grid of candidates cheap enough to run
at 10 Hz in pure Python.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "35-chomp-from-scratch"))
from chomp import edt as _edt_cells   # noqa: E402  exact Felzenszwalb DT


# ------------------------------------------------------------------ costmap
class Costmap:
    """An occupancy grid plus the two derived layers a planner actually uses.

    Why not just use the occupancy grid?  Because a planner that only knows
    "occupied / free" will happily drive a path that grazes a wall by 1 mm,
    and a real robot is not a point.  The two extra layers fix two different
    holes:

      * `dist`     -- distance in metres from every free cell to the nearest
                      obstacle.  Turns "did I hit something" into "how close
                      am I", which is what a cost function can be smooth in.
      * `inflated` -- cells within the robot's radius of an obstacle, marked
                      as lethal.  This is what lets the rest of the stack
                      pretend the robot is a point: inflate the obstacles by
                      the robot instead of shrinking nothing.
    """

    def __init__(self, occ, res=0.1, origin=(0.0, 0.0), robot_radius=0.22):
        self.occ = np.asarray(occ, bool)
        self.res = res
        self.origin = np.asarray(origin, float)
        self.robot_radius = robot_radius
        # Project 35 already wrote the exact distance transform; reuse it
        # rather than shipping a second, approximate one.
        self.dist = _edt_cells(self.occ) * res
        self.inflated = self.dist <= robot_radius
        self.h, self.w = self.occ.shape

    def world_to_cell(self, xy):
        c = (np.asarray(xy, float) - self.origin) / self.res
        return np.floor(c).astype(int)[..., ::-1]      # (row, col)

    def cell_to_world(self, rc):
        rc = np.asarray(rc, float)
        xy = rc[..., ::-1] * self.res + self.origin
        return xy + self.res * 0.5

    def clearance(self, xy):
        """Distance to the nearest obstacle, -1 outside the map."""
        rc = self.world_to_cell(xy)
        r, c = rc[..., 0], rc[..., 1]
        bad = (r < 0) | (r >= self.h) | (c < 0) | (c >= self.w)
        r = np.clip(r, 0, self.h - 1)
        c = np.clip(c, 0, self.w - 1)
        d = self.dist[r, c]
        return np.where(bad, -1.0, d)


# ------------------------------------------------------------------ maps
def room_map(res=0.1, kind="rooms"):
    """A 16 x 12 m indoor scene, in one of three flavours."""
    W, H = int(16 / res), int(12 / res)
    occ = np.zeros((H, W), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True

    def box(x0, y0, x1, y1):
        occ[int(y0 / res):int(y1 / res), int(x0 / res):int(x1 / res)] = True

    if kind == "rooms":
        box(5.0, 0.0, 5.4, 4.4)          # wall with a door at y = 4.4-6.4
        box(5.0, 6.4, 5.4, 12.0)
        box(10.6, 0.0, 11.0, 7.6)        # second wall, door at the other end
        box(2.0, 7.0, 3.6, 8.6)          # furniture
        box(13.0, 2.0, 14.4, 3.4)
        box(7.5, 9.0, 9.0, 10.0)
    elif kind == "trap":
        # A deep U with its MOUTH facing the start and its closed back facing
        # the goal.  A greedy local planner drives straight in (every step
        # gets it closer to the goal), then finds nothing but wall -- and its
        # 2 m horizon cannot see that the way out is 4 m backwards.
        box(11.6, 2.0, 12.0, 9.0)       # back wall, between robot and goal
        box(7.0, 2.0, 12.0, 2.4)        # bottom arm
        box(7.0, 8.6, 12.0, 9.0)        # top arm
    elif kind == "clutter":
        rng = np.random.default_rng(3)
        for _ in range(26):
            cx = rng.uniform(1.5, 14.5)
            cy = rng.uniform(1.5, 10.5)
            box(cx - 0.35, cy - 0.35, cx + 0.35, cy + 0.35)
    return occ


# ------------------------------------------------------------------ A* glue
def astar_path(cmap, start_xy, goal_xy, grid_search):
    """Global plan on the INFLATED map, returned in world coordinates.

    Planning on the inflated map instead of the raw one is what makes the
    global path something the local planner can actually follow: a route
    through a 30 cm gap is not a route if the robot is 44 cm wide.
    """
    s = tuple(cmap.world_to_cell(start_xy))
    g = tuple(cmap.world_to_cell(goal_xy))
    res = grid_search(cmap.inflated, s, g, heuristic="octile")
    if res["path"] is None:
        return None
    return cmap.cell_to_world(np.asarray(res["path"], float))


def carrot(path, xy, dist=2.0):
    """The local goal: the point on the global path `dist` ahead of the robot.

    This is the whole interface between the two planners, and it is worth
    naming what it hides.  The local planner never sees the goal 30 m away.
    It sees a point 2 m away that the global planner promises is on a route
    to the goal.  So the local planner can be short-sighted and greedy, and
    the stack still finishes -- the long-range thinking has been delegated.
    """
    d = np.linalg.norm(path - xy, axis=1)
    i = int(np.argmin(d))
    j = i
    while j < len(path) - 1 and np.linalg.norm(path[j] - xy) < dist:
        j += 1
    return path[j], i


# ------------------------------------------------------------------ the DWA
class DWAParams:
    def __init__(self, v_max=1.0, w_max=2.0, a_max=1.2, alpha_max=2.5,
                 nv=13, nw=27, sim_time=2.0, sim_dt=0.15,
                 w_head=1.0, w_clear=1.6, w_vel=0.35,
                 admissible=True, robot_radius=0.22, obs_margin=0.05):
        self.__dict__.update(locals())
        del self.__dict__["self"]


def dwa_step(state, v0, w0, goal_xy, cmap, p, dt_ctrl=0.1, moving=None,
             predict=False):
    """Score every reachable (v, omega) pair and return the best.

    `moving` is a list of (x, y, vx, vy, radius) for dynamic obstacles.
    `predict=False` is the honest baseline: the planner rolls out its own
    motion into the future but treats every moving obstacle as frozen where
    it is right now.  That is what a plain costmap-based DWA does, because
    the costmap is a snapshot.  `predict=True` moves each obstacle along its
    current velocity while rolling out -- the minimum viable prediction, and
    the seed of project 53.
    """
    # --- the window: reachable in one control period, clipped to the limits
    vs = np.linspace(max(0.0, v0 - p.a_max * dt_ctrl),
                     min(p.v_max, v0 + p.a_max * dt_ctrl), p.nv)
    ws = np.linspace(max(-p.w_max, w0 - p.alpha_max * dt_ctrl),
                     min(p.w_max, w0 + p.alpha_max * dt_ctrl), p.nw)
    V, W = np.meshgrid(vs, ws, indexing="ij")
    V, W = V.ravel(), W.ravel()

    # --- roll every candidate forward with a constant (v, omega): an arc
    n_steps = int(round(p.sim_time / p.sim_dt))
    x, y, th = state
    xs = np.empty((n_steps + 1, V.size))
    ys = np.empty((n_steps + 1, V.size))
    xs[0], ys[0] = x, y
    TH = np.full(V.size, th)
    X = np.full(V.size, x)
    Y = np.full(V.size, y)
    for k in range(n_steps):
        X = X + V * np.cos(TH) * p.sim_dt
        Y = Y + V * np.sin(TH) * p.sim_dt
        TH = TH + W * p.sim_dt
        xs[k + 1], ys[k + 1] = X, Y

    # --- clearance along each rollout, and the static-obstacle veto
    pts = np.stack([xs, ys], axis=-1)
    cl = cmap.clearance(pts.reshape(-1, 2)).reshape(xs.shape)
    min_cl = cl.min(axis=0)
    ok = min_cl > (p.robot_radius + p.obs_margin)

    # --- moving obstacles
    if moving:
        t = np.arange(n_steps + 1)[:, None] * p.sim_dt
        for (ox, oy, ovx, ovy, orad) in moving:
            cx = ox + (ovx * t if predict else 0.0)
            cy = oy + (ovy * t if predict else 0.0)
            d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
            dm = d.min(axis=0)
            ok &= dm > (p.robot_radius + orad + p.obs_margin)
            min_cl = np.minimum(min_cl, dm - orad)

    # --- the admissible-velocity rule
    if p.admissible:
        # Keep only speeds you could still stop from before hitting the
        # nearest obstacle on this rollout.  v <= sqrt(2 * a * d).  Without
        # this, a robot can commit to a speed at which no future braking
        # command can save it -- it is already too late at the moment it
        # chooses.
        d_stop = np.maximum(min_cl - p.robot_radius, 0.0)
        ok &= V <= np.sqrt(2.0 * p.a_max * d_stop) + 1e-9

    if not ok.any():
        return 0.0, 0.0, None, dict(feasible=0)

    # --- score: three terms, each normalised to [0, 1] over the survivors
    gx, gy = goal_xy
    end_d = np.sqrt((xs[-1] - gx) ** 2 + (ys[-1] - gy) ** 2)
    head = -end_d
    clear = np.clip(min_cl, 0.0, 1.5)
    vel = V

    def norm(a):
        a = a[ok]
        rng = a.max() - a.min()
        return (a - a.min()) / rng if rng > 1e-12 else np.zeros_like(a)

    score = (p.w_head * norm(head) + p.w_clear * norm(clear)
             + p.w_vel * norm(vel))
    idx = np.flatnonzero(ok)[int(np.argmax(score))]
    best_traj = np.stack([xs[:, idx], ys[:, idx]], axis=1)
    return (float(V[idx]), float(W[idx]), best_traj,
            dict(feasible=int(ok.sum()), n_cand=int(V.size),
                 min_clear=float(min_cl[idx])))
