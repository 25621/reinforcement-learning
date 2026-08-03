"""Pedestrians that move on their own, and planners that try to get past them.

Project 47's local planner treats every obstacle as a thing that sits still.
A person is not a thing that sits still, and three separate consequences
follow -- each of which needs its own piece of machinery here:

  * **They move.**  A costmap is a snapshot, so by the time the planner has
    finished thinking, the person is somewhere else.  Fixed by predicting.
  * **They have personal space.**  Passing a human at 5 cm is technically a
    success and socially a failure.  Fixed by a proxemic cost, not a
    constraint.
  * **They avoid you back.**  This is the one that quietly breaks
    evaluations: if the simulated humans step aside, a robot that does
    nothing at all scores well, and the benchmark is measuring the humans.

The crowd uses the **social force model** (Helbing & Molnar, 1995).  The name
is literal: each person is a particle, and their walking is modelled as a sum
of forces -- one pulling towards their goal, one pushing away from each other
person, one pushing away from walls.  It is not a claim about psychology; it
is the simplest model that reproduces the lane formation and door-clogging
that real crowds show.
"""

import math
import os
import sys

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "47-dwa-local-planner"))
sys.path.insert(0, os.path.join(_PROJ, "46-pure-pursuit"))

from dwa import Costmap                                       # noqa: E402,F401

PERSONAL_SPACE = 0.45          # m -- Hall's "intimate distance" boundary
ROBOT_R = 0.22
PED_R = 0.25


class Crowd:
    """A group of pedestrians walking between goals, using social forces."""

    def __init__(self, starts, goals, rng, v_pref=1.2, tau=0.5,
                 A=2.1, B=0.35, react_to_robot=True, robot_A=None):
        self.p = np.asarray(starts, float).copy()
        self.g = np.asarray(goals, float).copy()
        self.v = np.zeros_like(self.p)
        self.rng = rng
        self.v_pref = v_pref
        self.tau = tau                 # how fast they return to preferred speed
        self.A, self.B = A, B          # repulsion strength and its length scale
        self.react = react_to_robot
        self.robot_A = A if robot_A is None else robot_A
        self.n = len(self.p)

    def step(self, dt, robot_xy=None, bounds=(0.4, 15.6, 0.4, 11.6)):
        d = self.g - self.p
        dn = np.linalg.norm(d, axis=1, keepdims=True)
        # The driving force: "I want to be going v_pref towards my goal, and
        # I get there within about tau seconds."
        want = self.v_pref * d / np.maximum(dn, 1e-9)
        F = (want - self.v) / self.tau

        # Person-person repulsion, exponential in the gap between their edges.
        diff = self.p[:, None, :] - self.p[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, 1e9)
        mag = self.A * np.exp((2 * PED_R - dist) / self.B)
        F += np.einsum("ij,ijk->ik", mag / np.maximum(dist, 1e-9), diff)

        if robot_xy is not None and self.react:
            dr = self.p - np.asarray(robot_xy, float)
            dd = np.linalg.norm(dr, axis=1, keepdims=True)
            F += (self.robot_A * np.exp((PED_R + ROBOT_R - dd) / self.B)
                  * dr / np.maximum(dd, 1e-9))

        x0, x1, y0, y1 = bounds
        for k, (lo, hi) in enumerate([(x0, x1), (y0, y1)]):
            F[:, k] += 3.0 * np.exp((lo - self.p[:, k]) / 0.3)
            F[:, k] -= 3.0 * np.exp((self.p[:, k] - hi) / 0.3)

        self.v += F * dt
        sp = np.linalg.norm(self.v, axis=1, keepdims=True)
        self.v = np.where(sp > 1.6, self.v * 1.6 / np.maximum(sp, 1e-9), self.v)
        self.p = self.p + self.v * dt
        # Reached the goal?  Turn around, so the scene keeps flowing.
        done = np.linalg.norm(self.g - self.p, axis=1) < 0.6
        if done.any():
            self.g[done] = self.g[done] * 0 + self._flip(self.p[done])
        return self.p.copy()

    def _flip(self, p):
        out = p.copy()
        out[:, 0] = 16.0 - p[:, 0]
        out[:, 1] = 12.0 - p[:, 1]
        return out


# ------------------------------------------------------------------ planners
def predict(crowd_p, crowd_v, ts, mode="cv", sigma_rate=0.0):
    """Where each person will be at each future time, plus how sure we are.

    `mode`:
      "static" -- they stay where they are.  What a plain costmap believes.
      "cv"     -- constant velocity.  Two lines of code, and the baseline that
                  most published predictors only beat by a few centimetres
                  over a two-second horizon.
      "oracle" -- handled by the caller, which passes in the true future.

    `sigma_rate` grows a circle of uncertainty around each prediction at so
    many metres per second.  It is the knob that produces the freezing robot:
    make it large enough and every future is blocked, so the only safe action
    is to stop -- for ever.
    """
    ts = np.asarray(ts, float)[:, None, None]
    if mode == "static":
        pred = np.broadcast_to(crowd_p[None], (len(ts), *crowd_p.shape)).copy()
    else:
        pred = crowd_p[None] + crowd_v[None] * ts
    rad = np.full(pred.shape[:2], PED_R) + sigma_rate * ts[:, :, 0]
    return pred, rad


def social_dwa(state, v0, w0, goal_xy, cmap, pred, rad, p,
               w_social=0.0, social_r=PERSONAL_SPACE):
    """DWA with a personal-space term added to the cost.

    The extra term is deliberately a COST and not a constraint.  Making
    personal space a hard constraint means that in a corridor where every
    option intrudes, no option is legal and the robot stops -- which is worse
    manners than squeezing past.  As a cost it says "prefer not to", and the
    planner can trade it against making progress.
    """
    dt_ctrl = p.dt_ctrl
    vs = np.linspace(max(0.0, v0 - p.a_max * dt_ctrl),
                     min(p.v_max, v0 + p.a_max * dt_ctrl), p.nv)
    ws = np.linspace(max(-p.w_max, w0 - p.alpha_max * dt_ctrl),
                     min(p.w_max, w0 + p.alpha_max * dt_ctrl), p.nw)
    V, W = np.meshgrid(vs, ws, indexing="ij")
    V, W = V.ravel(), W.ravel()

    n_steps = pred.shape[0] - 1
    x, y, th = state
    xs = np.empty((n_steps + 1, V.size))
    ys = np.empty((n_steps + 1, V.size))
    X = np.full(V.size, x); Y = np.full(V.size, y); TH = np.full(V.size, th)
    xs[0], ys[0] = X, Y
    for k in range(n_steps):
        X = X + V * np.cos(TH) * p.sim_dt
        Y = Y + V * np.sin(TH) * p.sim_dt
        TH = TH + W * p.sim_dt
        xs[k + 1], ys[k + 1] = X, Y

    pts = np.stack([xs, ys], axis=-1)
    cl = cmap.clearance(pts.reshape(-1, 2)).reshape(xs.shape)
    min_cl = cl.min(axis=0)
    ok = min_cl > (ROBOT_R + 0.05)

    # gap to each person at each rollout step: (T, n_ped, n_cand)
    dx = xs[:, None, :] - pred[:, :, 0:1]
    dy = ys[:, None, :] - pred[:, :, 1:2]
    dist = np.sqrt(dx * dx + dy * dy) - rad[:, :, None] - ROBOT_R
    gap = dist.min(axis=(0, 1))
    ok &= gap > 0.02
    # How deep into anybody's personal space this rollout goes.
    intrusion = np.clip(social_r - dist, 0.0, None).sum(axis=(0, 1))

    if not ok.any():
        return 0.0, 0.0, None, dict(feasible=0)

    gx, gy = goal_xy
    end_d = np.sqrt((xs[-1] - gx) ** 2 + (ys[-1] - gy) ** 2)

    def norm(a):
        a = a[ok]
        r = a.max() - a.min()
        return (a - a.min()) / r if r > 1e-12 else np.zeros_like(a)

    score = (p.w_head * norm(-end_d)
             + p.w_clear * norm(np.clip(np.minimum(min_cl, gap), 0.0, 1.5))
             + p.w_vel * norm(V)
             - w_social * norm(intrusion))
    idx = np.flatnonzero(ok)[int(np.argmax(score))]
    return (float(V[idx]), float(W[idx]),
            np.stack([xs[:, idx], ys[:, idx]], axis=1),
            dict(feasible=int(ok.sum()), gap=float(gap[idx])))


class Params:
    def __init__(self, v_max=1.0, w_max=2.0, a_max=1.2, alpha_max=2.5,
                 nv=11, nw=25, sim_time=2.4, sim_dt=0.2, dt_ctrl=0.1,
                 w_head=1.0, w_clear=1.0, w_vel=0.5):
        self.__dict__.update(locals())
        del self.__dict__["self"]
