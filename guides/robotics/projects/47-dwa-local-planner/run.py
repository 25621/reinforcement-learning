"""Project 47 -- the Dynamic Window Approach, and why it needs a global plan.

Six experiments:
  1. one navigation run, with the candidate rollouts drawn
  2. the layered stack: DWA alone vs DWA under an A* global plan
  3. the three weights -- what "safe" and "fast" actually cost
  4. the rollout horizon and the admissible-velocity rule
  5. how many candidates are worth sampling
  6. moving obstacles: a snapshot costmap vs a one-line prediction
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
sys.path.insert(0, os.path.join(_PROJ, "46-pure-pursuit"))
sys.path.insert(0, os.path.join(_PROJ, "31-a-star-on-a-grid"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from dwa import (Costmap, DWAParams, astar_path, carrot, dwa_step,   # noqa: E402
                 room_map)
from robot import DiffDrive                                          # noqa: E402
from grid import search as grid_search                               # noqa: E402
from plot_style import COLORS, use_style, save                       # noqa: E402

import matplotlib.pyplot as plt                                      # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


# ------------------------------------------------------------------ driver
def navigate(cmap, start, goal, p, use_global=True, dt_ctrl=0.1, t_max=90.0,
             moving=None, predict=False, carrot_d=2.0, goal_tol=0.35,
             keep_traj=False):
    """Run one navigation episode.  Returns the trace and the verdict."""
    gpath = None
    if use_global:
        gpath = astar_path(cmap, start[:2], goal, grid_search)
        if gpath is None:
            return dict(success=False, reason="no global plan", xy=np.zeros((0, 2)))
    rb = DiffDrive(start[0], start[1], start[2], v_max=p.v_max, w_max=p.w_max,
                   a_max=p.a_max, alpha_max=p.alpha_max)
    obs = [list(o) for o in (moving or [])]
    xy, trajs, clears, obs_hist = [], [], [], []
    t, collided, stuck, tcalls, tsum = 0.0, False, 0, 0, 0.0
    n_sub = max(int(round(dt_ctrl / 0.02)), 1)
    sub_dt = dt_ctrl / n_sub

    while t < t_max:
        st = rb.state
        xy.append(st[:2].copy())
        if np.linalg.norm(st[:2] - goal) < goal_tol:
            break
        tgt = goal if not use_global else carrot(gpath, st[:2], carrot_d)[0]
        t0 = time.perf_counter()
        v, w, traj, info = dwa_step(st, rb.v, rb.w, tgt, cmap, p,
                                    dt_ctrl=dt_ctrl,
                                    moving=[tuple(o) for o in obs],
                                    predict=predict)
        tsum += time.perf_counter() - t0
        tcalls += 1
        if traj is None:                      # no admissible command at all
            stuck += 1
            v, w = 0.0, 0.0
        if keep_traj:
            trajs.append(traj)
        for _ in range(n_sub):
            rb.step(v, w, sub_dt)
            for o in obs:                     # obstacles move too
                o[0] += o[2] * sub_dt
                o[1] += o[3] * sub_dt
        t += dt_ctrl
        c = float(cmap.clearance(rb.state[:2]))
        for o in obs:
            c = min(c, math.hypot(rb.x - o[0], rb.y - o[1]) - o[4])
        clears.append(c)
        if obs:
            obs_hist.append([(o[0], o[1]) for o in obs])
        if c < p.robot_radius * 0.5:          # centre well inside an obstacle
            collided = True
            break

    xy = np.asarray(xy)
    reached = len(xy) > 0 and np.linalg.norm(xy[-1] - goal) < goal_tol + 0.15
    why = "ok" if (reached and not collided) else (
        "collision" if collided else "timeout")
    return dict(success=bool(reached and not collided), collided=collided,
                why=why,
                xy=xy, gpath=gpath, trajs=trajs, obs_hist=obs_hist,
                t=t, stuck_frac=stuck / max(tcalls, 1),
                path_len=float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))
                if len(xy) > 1 else 0.0,
                min_clear=float(np.min(clears)) if clears else 0.0,
                ms_per_call=1000.0 * tsum / max(tcalls, 1))


def show_map(ax, cmap):
    ax.imshow(np.where(cmap.occ, 0.35, 1.0), cmap="gray", vmin=0, vmax=1,
              origin="lower", extent=[0, cmap.w * cmap.res, 0,
                                      cmap.h * cmap.res])
    ax.set_aspect("equal"); ax.grid(False)


# ================================================================= 1. a run
def exp1():
    print("[1] one navigation run")
    cmap = Costmap(room_map(kind="rooms"))
    start, goal = (1.2, 1.2, 0.0), np.array([14.6, 10.6])
    p = DWAParams()
    r = navigate(cmap, start, goal, p, keep_traj=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    show_map(axes[0], cmap)
    axes[0].plot(r["gpath"][:, 0], r["gpath"][:, 1], "--", color=COLORS[1],
                 lw=1.6, label="A* global plan")
    axes[0].plot(r["xy"][:, 0], r["xy"][:, 1], color=COLORS[0], lw=2.0,
                 label="DWA trace")
    k = len(r["trajs"]) // 3
    st = r["xy"][k]
    # Draw the whole candidate fan at one instant.
    tgt = carrot(r["gpath"], st, 2.0)[0]
    all_tr = _fan(cmap, r, k, p)
    for tr in all_tr:
        axes[0].plot(tr[:, 0], tr[:, 1], color=COLORS[4], lw=0.5, alpha=0.5)
    axes[0].plot(r["trajs"][k][:, 0], r["trajs"][k][:, 1], color=COLORS[2],
                 lw=2.2, label="chosen rollout")
    axes[0].plot(*tgt, "*", color=COLORS[3], ms=13, label="carrot (2 m ahead)")
    axes[0].plot(*goal, "X", color="k", ms=10)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_title("A* plans the route; DWA picks the next 100 ms")

    im = axes[1].imshow(np.minimum(cmap.dist, 1.5), origin="lower",
                        extent=[0, cmap.w * cmap.res, 0, cmap.h * cmap.res],
                        cmap="magma")
    axes[1].contour(np.linspace(0, cmap.w * cmap.res, cmap.w),
                    np.linspace(0, cmap.h * cmap.res, cmap.h),
                    cmap.inflated.astype(float), levels=[0.5],
                    colors=[COLORS[5]], linewidths=1.0)
    axes[1].set_aspect("equal"); axes[1].grid(False)
    axes[1].set_title("clearance layer; cyan = inflated (lethal) boundary")
    fig.colorbar(im, ax=axes[1], label="distance to obstacle [m]")
    save(fig, os.path.join(OUT, "overview.png"))
    rec("1_run", success=int(r["success"]), time_s=round(r["t"], 1),
        path_len=round(r["path_len"], 2), min_clear=round(r["min_clear"], 3),
        ms_per_call=round(r["ms_per_call"], 2))
    return cmap


def _fan(cmap, r, k, p):
    """Re-roll the candidate set at step k, purely for the figure."""
    from dwa import dwa_step as _s
    st = np.array([r["xy"][k][0], r["xy"][k][1], 0.0])
    if k + 1 < len(r["xy"]):
        d = r["xy"][k + 1] - r["xy"][k]
        st[2] = math.atan2(d[1], d[0])
    out = []
    vs = np.linspace(0.2, p.v_max, 5)
    ws = np.linspace(-p.w_max, p.w_max, 13)
    for v in vs:
        for w in ws:
            x, y, th = st
            pts = [(x, y)]
            for _ in range(int(p.sim_time / p.sim_dt)):
                x += v * math.cos(th) * p.sim_dt
                y += v * math.sin(th) * p.sim_dt
                th += w * p.sim_dt
                pts.append((x, y))
            out.append(np.asarray(pts))
    return out


# ================================================= 2. the layered stack
def exp2():
    print("[2] DWA alone vs DWA + A*")
    maps = {k: Costmap(room_map(kind=k)) for k in ("rooms", "trap", "clutter")}
    rng = np.random.default_rng(7)
    p = DWAParams()
    queries = {}
    for name, cmap in maps.items():
        if name == "trap":
            queries[name] = [((2.0, 5.5, 0.0), np.array([14.6, 5.5]))]
            queries[name] += [((2.0, y, 0.0), np.array([14.6, y]))
                              for y in (4.0, 5.0, 6.5, 7.5)]
        else:
            qs = []
            free = np.argwhere(~cmap.inflated)
            while len(qs) < 12:
                a, b = free[rng.integers(0, len(free), 2)]
                pa, pb = cmap.cell_to_world(a), cmap.cell_to_world(b)
                if np.linalg.norm(pa - pb) < 8.0:
                    continue
                if astar_path(cmap, pa, pb, grid_search) is None:
                    continue
                qs.append(((pa[0], pa[1], rng.uniform(-math.pi, math.pi)), pb))
            queries[name] = qs

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, (name, cmap) in zip(axes, maps.items()):
        for use_global, c, lab in [(False, COLORS[1], "DWA alone"),
                                   (True, COLORS[0], "DWA + A*")]:
            ok, lens, times, coll, tout = 0, [], [], 0, 0
            for st, gl in queries[name]:
                r = navigate(cmap, st, gl, p, use_global=use_global, t_max=90)
                ok += int(r["success"])
                coll += int(r["why"] == "collision")
                tout += int(r["why"] == "timeout")
                if r["success"]:
                    lens.append(r["path_len"]); times.append(r["t"])
            rec("2_layers", map=name, planner=lab, n=len(queries[name]),
                success=ok, collided=coll, timed_out=tout,
                rate=round(ok / len(queries[name]), 3),
                mean_len=round(float(np.mean(lens)), 2) if lens else None,
                mean_time=round(float(np.mean(times)), 1) if times else None)
            if name == "trap":
                r = navigate(cmap, queries[name][0][0], queries[name][0][1],
                             p, use_global=use_global, t_max=90)
                ax.plot(r["xy"][:, 0], r["xy"][:, 1], color=c, lw=2.0,
                        label=lab)
        show_map(ax, cmap)
        ax.set_title(name)
        if name == "trap":
            ax.plot(2.0, 5.5, "o", color="k", ms=7)
            ax.plot(14.6, 5.5, "X", color="k", ms=10)
            ax.legend(fontsize=8, loc="lower left")
    save(fig, os.path.join(OUT, "layers.png"))
    return maps, queries


# ================================================= 3. the weights
def exp3(maps, queries):
    print("[3] the three weights")
    cmap = maps["clutter"]
    qs = queries["clutter"]
    combos = [("balanced", 1.0, 1.6, 0.35),
              ("goal only", 1.0, 0.0, 0.0),
              ("timid (clearance x4)", 1.0, 6.4, 0.35),
              ("greedy speed", 1.0, 0.4, 2.0),
              ("no goal term", 0.0, 1.6, 1.0)]
    res = []
    for name, wh, wc, wv in combos:
        p = DWAParams(w_head=wh, w_clear=wc, w_vel=wv)
        ok, lens, times, cl, coll, stuck = 0, [], [], [], 0, []
        for st, gl in qs:
            r = navigate(cmap, st, gl, p, t_max=90)
            ok += int(r["success"])
            coll += int(r["why"] == "collision")
            cl.append(r["min_clear"]); stuck.append(r["stuck_frac"])
            if r["success"]:
                lens.append(r["path_len"]); times.append(r["t"])
        row = (name, ok / len(qs),
               float(np.mean(lens)) if lens else float("nan"),
               float(np.mean(times)) if times else float("nan"),
               float(np.mean(cl)))
        res.append(row)
        rec("3_weights", weights=name, w_head=wh, w_clear=wc, w_vel=wv,
            rate=round(row[1], 3), mean_len=round(row[2], 2),
            mean_time=round(row[3], 1), mean_min_clear=round(row[4], 3),
            collided=coll, timed_out=len(qs) - ok - coll,
            frozen_frac=round(float(np.mean(stuck)), 3))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    names = [r[0] for r in res]
    for ax, col, lab in [(axes[0], 1, "success rate"),
                         (axes[1], 3, "mean time to goal [s]"),
                         (axes[2], 4, "mean min clearance [m]")]:
        ax.barh(names, [r[col] for r in res], color=COLORS[0])
        ax.set_title(lab)
        ax.invert_yaxis()
    axes[2].axvline(0.22, color=COLORS[1], ls=":", label="robot radius")
    axes[2].legend(fontsize=7)
    save(fig, os.path.join(OUT, "weights.png"))


# ================================================= 4. horizon + admissibility
def exp4(maps, queries):
    print("[4] rollout horizon and the admissible-velocity rule")
    cmap = maps["clutter"]
    qs = queries["clutter"]
    horizons = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 4.0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for adm, c, lab in [(True, COLORS[0], "with admissible check"),
                        (False, COLORS[1], "without")]:
        rates, times = [], []
        for T in horizons:
            p = DWAParams(sim_time=T, admissible=adm)
            ok, tt = 0, []
            for st, gl in qs:
                r = navigate(cmap, st, gl, p, t_max=90)
                ok += int(r["success"])
                if r["success"]:
                    tt.append(r["t"])
            rates.append(ok / len(qs))
            times.append(float(np.mean(tt)) if tt else float("nan"))
            rec("4_horizon", sim_time=T, admissible=int(adm),
                rate=round(rates[-1], 3),
                mean_time=round(times[-1], 1) if tt else None,
                braking_dist=round(1.0 ** 2 / (2 * 1.2), 2))
        axes[0].plot(horizons, rates, "o-", color=c, label=lab)
        axes[1].plot(horizons, times, "o-", color=c, label=lab)
    for ax, lab in [(axes[0], "success rate"), (axes[1], "mean time [s]")]:
        ax.set_xlabel("rollout horizon [s]"); ax.set_title(lab)
        ax.axvline(1.0 / 1.2, color="0.5", ls=":", lw=1.0)
        ax.legend(fontsize=7)
    axes[0].set_ylim(0, 1.05)
    save(fig, os.path.join(OUT, "horizon.png"))


# ================================================= 5. sample count
def exp5(maps, queries):
    print("[5] how many candidates")
    cmap = maps["clutter"]
    qs = queries["clutter"]
    sizes = [(3, 5), (5, 9), (7, 15), (13, 27), (21, 41), (31, 61)]
    xs, rates, ms, lens = [], [], [], []
    for nv, nw in sizes:
        p = DWAParams(nv=nv, nw=nw)
        ok, mm, ll = 0, [], []
        for st, gl in qs:
            r = navigate(cmap, st, gl, p, t_max=90)
            ok += int(r["success"]); mm.append(r["ms_per_call"])
            if r["success"]:
                ll.append(r["path_len"])
        xs.append(nv * nw); rates.append(ok / len(qs))
        ms.append(float(np.mean(mm)))
        lens.append(float(np.mean(ll)) if ll else float("nan"))
        rec("5_samples", nv=nv, nw=nw, n_candidates=nv * nw,
            rate=round(rates[-1], 3), ms_per_call=round(ms[-1], 3),
            mean_len=round(lens[-1], 2),
            # The window is only 2*a*dt wide, so each extra sample resolves
            # a very small velocity difference -- this is the number that
            # explains why the sweep flattens so fast.
            dv_per_sample=round(2 * 1.2 * 0.1 / max(nv - 1, 1), 4),
            dw_per_sample=round(2 * 2.5 * 0.1 / max(nw - 1, 1), 4))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.plot(xs, rates, "o-", color=COLORS[0], label="success rate")
    ax2 = ax.twinx()
    ax2.plot(xs, ms, "s--", color=COLORS[1], label="ms per call")
    ax2.set_ylabel("ms per DWA call"); ax2.grid(False)
    ax.set_xscale("log"); ax.set_xlabel("candidate (v, omega) pairs")
    ax.set_ylabel("success rate"); ax.set_ylim(0, 1.05)
    ax.axhline(1.0 / 0.1, color="none")
    ax.legend(loc="lower right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax.set_title("diminishing returns, and the 100 ms budget")
    save(fig, os.path.join(OUT, "samples.png"))


# ================================================= 6. moving obstacles
def exp6():
    print("[6] moving obstacles: snapshot vs prediction")
    # A deliberately clean scene: an empty room, the robot crossing left to
    # right, and pedestrians launched so their paths CROSS the robot's near
    # the middle.  In a cluttered map most obstacles never come near the
    # robot, so the encounters that matter get averaged away with dozens of
    # non-events -- the comparison then measures the map, not the planner.
    occ = np.zeros((120, 160), bool)
    occ[0, :] = occ[-1, :] = occ[:, 0] = occ[:, -1] = True
    cmap = Costmap(occ)
    rng = np.random.default_rng(11)
    p = DWAParams()
    speeds = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    N = 30
    scen = []
    for _ in range(N):
        y0 = rng.uniform(3.0, 9.0)
        st = (1.2, y0, 0.0)
        gl = np.array([14.6, y0 + rng.uniform(-1.5, 1.5)])
        # Three crossers, aimed at points the robot will pass through.
        meet_x = rng.uniform(5.0, 11.0, 3)
        side = rng.choice([-1.0, 1.0], 3)
        scen.append((st, gl, meet_x, side, rng.uniform(0.7, 1.3, 3)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.3]})
    for pred, c, lab in [(False, COLORS[1], "frozen snapshot"),
                         (True, COLORS[0], "constant-velocity prediction")]:
        rates, cols, times = [], [], []
        for sp in speeds:
            ok, coll, tt = 0, 0, []
            for st, gl, meet_x, side, lead in scen:
                mv = []
                for i in range(3):
                    # Time for the robot to reach meet_x at ~0.9 m/s.
                    t_meet = (meet_x[i] - st[0]) / 0.9 * lead[i]
                    oy = st[1] + side[i] * max(sp, 0.05) * t_meet
                    mv.append((meet_x[i], float(np.clip(oy, 0.5, 11.5)),
                               0.0, -side[i] * sp, 0.30))
                r = navigate(cmap, st, gl, p, moving=mv, predict=pred,
                             t_max=60)
                ok += int(r["success"]); coll += int(r["collided"])
                if r["success"]:
                    tt.append(r["t"])
            rates.append(ok / len(scen)); cols.append(coll / len(scen))
            times.append(float(np.mean(tt)) if tt else float("nan"))
            rec("6_moving", predict=int(pred), obs_speed=sp, n=len(scen),
                rate=round(rates[-1], 3), collision_rate=round(cols[-1], 3),
                mean_time=round(times[-1], 1) if tt else None)
        axes[0].plot(speeds, cols, "o-", color=c, label=lab)
        axes[1].plot(speeds, times, "o-", color=c, label=lab)
    axes[0].set_ylabel("collision rate"); axes[0].set_ylim(-0.03, 1.03)
    axes[1].set_ylabel("mean time to goal [s]")
    for ax in axes[:2]:
        ax.set_xlabel("pedestrian speed [m/s]"); ax.legend(fontsize=7)
        ax.axvline(1.0, color="0.5", ls=":", lw=1.0)
    axes[0].set_title("dotted = the robot's own top speed")
    axes[1].set_title("what avoidance costs in time")

    # One illustrative episode.
    st, gl, meet_x, side, lead = scen[0]
    sp = 0.75
    mv = []
    for i in range(3):
        t_meet = (meet_x[i] - st[0]) / 0.9 * lead[i]
        oy = st[1] + side[i] * sp * t_meet
        mv.append((meet_x[i], float(np.clip(oy, 0.5, 11.5)), 0.0,
                   -side[i] * sp, 0.30))
    show_map(axes[2], cmap)
    for pred, c, lab in [(False, COLORS[1], "snapshot"),
                         (True, COLORS[0], "prediction")]:
        r = navigate(cmap, st, gl, p, moving=mv, predict=pred, t_max=60)
        axes[2].plot(r["xy"][:, 0], r["xy"][:, 1], color=c, lw=1.8, label=lab)
        if pred and r["obs_hist"]:
            oh = np.asarray(r["obs_hist"])
            for i in range(oh.shape[1]):
                axes[2].plot(oh[:, i, 0], oh[:, i, 1], ":", color="0.45",
                             lw=1.2)
                axes[2].add_patch(plt.Circle(oh[0, i], 0.30, color="0.7"))
    axes[2].legend(fontsize=8, loc="upper left")
    axes[2].set_title("one crossing episode (grey = pedestrians)")
    save(fig, os.path.join(OUT, "moving.png"))


if __name__ == "__main__":
    t0 = time.time()
    exp1()
    maps, queries = exp2()
    exp3(maps, queries)
    exp4(maps, queries)
    exp5(maps, queries)
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
