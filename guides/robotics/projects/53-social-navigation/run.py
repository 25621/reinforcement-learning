"""Project 53 -- navigating among people who move, and who move around you.

Six experiments:
  1. one crossing, four planners
  2. crowd density
  3. the freezing robot: prediction uncertainty, swept
  4. do the humans avoid you back?  (the evaluation trap)
  5. personal space: what politeness costs
  6. how good does the prediction have to be?
"""

import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "47-dwa-local-planner"))
sys.path.insert(0, os.path.join(_PROJ, "46-pure-pursuit"))
sys.path.insert(0, os.path.join(_PROJ, "31-a-star-on-a-grid"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from social import (Crowd, PED_R, PERSONAL_SPACE, Params, ROBOT_R,  # noqa: E402
                    predict, social_dwa)
from dwa import Costmap, astar_path, carrot                         # noqa: E402
from robot import DiffDrive                                         # noqa: E402
from grid import search as grid_search                              # noqa: E402
from plot_style import COLORS, use_style, save                      # noqa: E402

import matplotlib.pyplot as plt                                     # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []

W_M, H_M = 16.0, 12.0
START = (1.0, 6.0, 0.0)
GOAL = np.array([15.0, 6.0])
SEEDS = range(5)


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


def empty_map():
    occ = np.zeros((120, 160), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    return Costmap(occ, res=0.1, robot_radius=ROBOT_R)


def make_crowd(n, rng, react=True):
    """Half the crowd walks towards the robot, half crosses its path."""
    starts, goals = [], []
    for i in range(n):
        if i % 2 == 0:                       # head-on traffic
            y = rng.uniform(3.5, 8.5)
            starts.append([rng.uniform(9.0, 14.0), y])
            goals.append([1.0, rng.uniform(3.5, 8.5)])
        else:                                # crossing traffic
            x = rng.uniform(5.0, 12.0)
            top = rng.random() < 0.5
            starts.append([x, 11.0 if top else 1.0])
            goals.append([x + rng.uniform(-2, 2), 1.0 if top else 11.0])
    return Crowd(np.array(starts), np.array(goals), rng, react_to_robot=react)


def episode(n_ped=8, seed=0, mode="cv", sigma_rate=0.0, w_social=0.0,
            react=True, t_max=45.0, keep=False, oracle=False):
    """One traverse of the room.  Returns the trace and every metric."""
    rng = np.random.default_rng(seed)
    cmap = empty_map()
    p = Params()
    crowd = make_crowd(n_ped, rng, react=react)
    for _ in range(30):                       # let the crowd get moving
        crowd.step(0.1)
    gpath = astar_path(cmap, np.array(START[:2]), GOAL, grid_search)
    rb = DiffDrive(*START, v_max=p.v_max, w_max=p.w_max, a_max=p.a_max,
                   alpha_max=p.alpha_max)

    ts = np.arange(0.0, p.sim_time + 1e-9, p.sim_dt)
    t, stopped, collided, intrusions, n_ctrl = 0.0, 0, 0, 0, 0
    min_gap = 1e9
    trace, crowd_hist = [], []
    n_sub = 5
    sub_dt = p.dt_ctrl / n_sub

    while t < t_max:
        xy = rb.state[:2]
        trace.append(xy.copy())
        if keep:
            crowd_hist.append(crowd.p.copy())
        if np.linalg.norm(xy - GOAL) < 0.4:
            break
        tgt = carrot(gpath, xy, 2.5)[0]
        if oracle:
            # Roll a COPY of the crowd forward to get the true future.  Not
            # implementable on a robot; here as the ceiling any predictor
            # could reach.
            ghost = Crowd(crowd.p.copy(), crowd.g.copy(),
                          np.random.default_rng(0), react_to_robot=False)
            ghost.v = crowd.v.copy()
            fut = [crowd.p.copy()]
            for _ in range(len(ts) - 1):
                for _ in range(4):
                    ghost.step(p.sim_dt / 4)
                fut.append(ghost.p.copy())
            pred = np.asarray(fut)
            rad = np.full(pred.shape[:2], PED_R)
        else:
            pred, rad = predict(crowd.p, crowd.v, ts, mode=mode,
                                sigma_rate=sigma_rate)
        v, w, traj, info = social_dwa(rb.state, rb.v, rb.w, tgt, cmap, pred,
                                      rad, p, w_social=w_social)
        n_ctrl += 1
        if traj is None or v < 0.05:
            stopped += 1
        for _ in range(n_sub):
            rb.step(v, w, sub_dt)
            crowd.step(sub_dt, robot_xy=rb.state[:2])
        t += p.dt_ctrl
        d = np.linalg.norm(crowd.p - rb.state[:2], axis=1) - PED_R - ROBOT_R
        min_gap = min(min_gap, float(d.min()))
        if d.min() < 0.0:
            collided += 1
        if d.min() < PERSONAL_SPACE:
            intrusions += 1

    trace = np.asarray(trace)
    reached = len(trace) and np.linalg.norm(trace[-1] - GOAL) < 0.6
    return dict(trace=trace, crowd_hist=crowd_hist, cmap=cmap,
                success=bool(reached and collided == 0),
                reached=bool(reached), t=t,
                collided=collided > 0, collision_steps=collided,
                min_gap=min_gap,
                intrusion_frac=intrusions / max(n_ctrl, 1),
                frozen_frac=stopped / max(n_ctrl, 1),
                path_len=float(np.sum(np.linalg.norm(np.diff(trace, axis=0),
                                                     axis=1)))
                if len(trace) > 1 else 0.0)


def agg(**kw):
    seeds = kw.pop("seeds", SEEDS)
    rs = [episode(seed=s, **kw) for s in seeds]
    return dict(n=len(rs),
                success=float(np.mean([r["success"] for r in rs])),
                collision=float(np.mean([r["collided"] for r in rs])),
                time=float(np.mean([r["t"] for r in rs])),
                min_gap=float(np.mean([r["min_gap"] for r in rs])),
                intrusion=float(np.mean([r["intrusion_frac"] for r in rs])),
                frozen=float(np.mean([r["frozen_frac"] for r in rs])),
                path_len=float(np.mean([r["path_len"] for r in rs])))


def show(ax, cmap):
    ax.imshow(np.where(cmap.occ, 0.4, 1.0), cmap="gray", vmin=0, vmax=1,
              origin="lower", extent=[0, W_M, 0, H_M])
    ax.set_aspect("equal"); ax.grid(False)


# ================================================================ 1. a scene
def exp1():
    print("[1] one crossing, four planners")
    arms = [("static costmap", dict(mode="static")),
            ("constant-velocity prediction", dict(mode="cv")),
            ("prediction + personal space", dict(mode="cv", w_social=1.2)),
            ("prediction + 0.35 m/s uncertainty",
             dict(mode="cv", sigma_rate=0.35))]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (lab, kw) in zip(axes, arms):
        r = episode(n_ped=10, seed=3, keep=True, **kw)
        show(ax, r["cmap"])
        ch = np.asarray(r["crowd_hist"])
        for i in range(ch.shape[1]):
            ax.plot(ch[:, i, 0], ch[:, i, 1], color="0.7", lw=0.8)
            ax.add_patch(plt.Circle(ch[0, i], PED_R, color="0.75"))
        ax.plot(r["trace"][:, 0], r["trace"][:, 1], color=COLORS[0], lw=2.0)
        ax.plot(*GOAL, "X", color="k", ms=9)
        ax.set_title(f"{lab}\n{r['t']:.0f} s, min gap {r['min_gap']:.2f} m",
                     fontsize=8)
        rec("1_scene", planner=lab, time_s=round(r["t"], 1),
            min_gap_m=round(r["min_gap"], 3),
            intrusion_frac=round(r["intrusion_frac"], 3),
            frozen_frac=round(r["frozen_frac"], 3),
            reached=int(r["reached"]), collided=int(r["collided"]))
    save(fig, os.path.join(OUT, "scene.png"))


# ================================================= 2. density
def exp2():
    print("[2] crowd density")
    dens = [2, 8, 14, 22]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.4))
    for mode, ws, c, lab in [("static", 0.0, COLORS[1], "static costmap"),
                             ("cv", 0.0, COLORS[0], "CV prediction"),
                             ("cv", 1.2, COLORS[2], "CV + personal space")]:
        S, C, T = [], [], []
        for n in dens:
            a = agg(n_ped=n, mode=mode, w_social=ws)
            S.append(a["success"]); C.append(a["collision"]); T.append(a["time"])
            rec("2_density", planner=lab, n_ped=n, **{
                k: round(v, 3) for k, v in a.items() if k != "n"})
        axes[0].plot(dens, S, "o-", color=c, label=lab)
        axes[1].plot(dens, C, "o-", color=c, label=lab)
        axes[2].plot(dens, T, "o-", color=c, label=lab)
    axes[0].set_ylabel("success rate"); axes[1].set_ylabel("collision rate")
    axes[2].set_ylabel("time to goal [s]")
    for ax in axes:
        ax.set_xlabel("pedestrians"); ax.legend(fontsize=7)
    axes[0].set_ylim(-0.05, 1.05); axes[1].set_ylim(-0.05, 1.05)
    save(fig, os.path.join(OUT, "density.png"))


# ================================================= 3. the freezing robot
def exp3():
    print("[3] the freezing robot")
    sigmas = [0.0, 0.15, 0.3, 0.45, 0.7, 1.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    for n, c in zip([6, 14], COLORS):
        S, F, C, T = [], [], [], []
        for s in sigmas:
            a = agg(n_ped=n, mode="cv", sigma_rate=s)
            S.append(a["success"]); F.append(a["frozen"])
            C.append(a["collision"]); T.append(a["time"])
            rec("3_freeze", n_ped=n, sigma_rate=s, **{
                k: round(v, 3) for k, v in a.items() if k != "n"})
        axes[0].plot(sigmas, C, "o-", color=c, label=f"{n} people: collision")
        axes[0].plot(sigmas, F, "s--", color=c, label=f"{n} people: frozen")
        axes[1].plot(sigmas, T, "o-", color=c, label=f"{n} people")
    axes[0].set_ylabel("rate"); axes[0].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("time to goal [s]")
    for ax in axes:
        ax.set_xlabel("uncertainty growth [m per second of horizon]")
        ax.legend(fontsize=7)
    axes[0].set_title("too little and you hit; too much and you stop")
    save(fig, os.path.join(OUT, "freezing.png"))


# ================================================= 4. do the humans move?
def exp4():
    print("[4] do the humans avoid you back?")
    dens = [6, 12, 20]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    x = np.arange(len(dens))
    for i, (react, c, lab) in enumerate(
            [(True, COLORS[0], "humans avoid the robot"),
             (False, COLORS[1], "humans ignore the robot")]):
        for j, (mode, hatch, sub) in enumerate(
                [("static", "", "static costmap"), ("cv", "//", "CV prediction")]):
            vals = []
            for n in dens:
                a = agg(n_ped=n, mode=mode, react=react)
                vals.append(a["success"])
                rec("4_react", humans_react=int(react), planner=sub, n_ped=n,
                    **{k: round(v, 3) for k, v in a.items() if k != "n"})
            ax.bar(x + (i * 2 + j) * 0.2 - 0.3, vals, 0.2, color=c,
                   hatch=hatch, label=f"{lab}, {sub}" if True else None)
    ax.set_xticks(x); ax.set_xticklabels([str(d) for d in dens])
    ax.set_xlabel("pedestrians"); ax.set_ylabel("success rate")
    ax.legend(fontsize=6); ax.set_title("who is doing the avoiding?")
    save(fig, os.path.join(OUT, "reciprocity.png"))


# ================================================= 5. personal space
def exp5():
    print("[5] what politeness costs")
    ws = [0.0, 0.4, 0.8, 1.5, 3.0, 6.0]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    I, T, G, S = [], [], [], []
    for w in ws:
        a = agg(n_ped=12, mode="cv", w_social=w)
        I.append(a["intrusion"]); T.append(a["time"]); G.append(a["min_gap"])
        S.append(a["success"])
        rec("5_personal", w_social=w, **{
            k: round(v, 3) for k, v in a.items() if k != "n"})
    axes[0].plot(ws, I, "o-", color=COLORS[0], label="fraction of time inside\n"
                 "someone's personal space")
    axes[0].plot(ws, S, "s-", color=COLORS[2], label="success rate")
    axes[0].legend(fontsize=7); axes[0].set_ylim(-0.05, 1.05)
    axes[1].plot(ws, T, "o-", color=COLORS[1], label="time to goal [s]")
    ax2 = axes[1].twinx()
    ax2.plot(ws, G, "s--", color=COLORS[3])
    ax2.set_ylabel("mean closest approach [m]", color=COLORS[3])
    ax2.grid(False)
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("personal-space cost weight")
    save(fig, os.path.join(OUT, "personal_space.png"))


# ================================================= 6. prediction quality
def exp6():
    print("[6] how good does the prediction have to be?")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    arms = [("static (no prediction)", dict(mode="static")),
            ("constant velocity", dict(mode="cv")),
            ("oracle (the true future)", dict(oracle=True))]
    dens = [6, 12, 20]
    for (lab, kw), c in zip(arms, COLORS):
        S = []
        for n in dens:
            a = agg(n_ped=n, **kw)
            S.append(a["success"])
            rec("6_predictor", predictor=lab, n_ped=n, **{
                k: round(v, 3) for k, v in a.items() if k != "n"})
        ax.plot(dens, S, "o-", color=c, label=lab)
    ax.set_xlabel("pedestrians"); ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=7)
    ax.set_title("what a perfect predictor would be worth")
    save(fig, os.path.join(OUT, "predictor.png"))


if __name__ == "__main__":
    t0 = time.time()
    exp1()
    exp2()
    exp3()
    exp4()
    exp5()
    exp6()
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(ROWS)
    print(f"done in {time.time() - t0:.1f}s")
