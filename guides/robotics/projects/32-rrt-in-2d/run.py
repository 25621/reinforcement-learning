"""Project 32 -- RRT in 2D: what random sampling buys, and what it costs.

Seven experiments:

  1. how the tree actually grows, and the bias nobody programmed in
  2. goal bias: 0% never aims, 100% is greedy, and both are bad
  3. step size: the knob that trades node count against rejection rate
  4. RRT never improves; RRT* does -- measured against a true optimum
  5. the narrow passage, and why it is THE failure mode of sampling
  6. run-to-run variance: one run tells you almost nothing
  7. where the time goes: collision checks and the nearest-neighbour query

Runs in about four minutes.  NumPy and Matplotlib only.
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
sys.path.insert(0, os.path.join(_PROJ, "31-a-star-on-a-grid"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from rrt import (Env, world_blobs, world_narrow, rrt, rrt_star, steer,   # noqa: E402
                 path_cost, Tree)
from grid import search, shortcut_los, path_length                       # noqa: E402
from plot_style import COLORS, use_style, save                           # noqa: E402

import matplotlib.pyplot as plt                                          # noqa: E402
from matplotlib.patches import Circle, Rectangle                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []
START = np.array([0.5, 0.5])
GOAL = np.array([9.5, 9.5])


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def draw_env(ax, env):
    for cx, cy, r in env.circles:
        ax.add_patch(Circle((cx, cy), r, color="#4A4A4A"))
    for x0, y0, x1, y1 in env.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#4A4A4A"))
    ax.set_xlim(env.lo[0], env.hi[0])
    ax.set_ylim(env.lo[1], env.hi[1])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def draw_tree(ax, tree, color="#BFDCEF", lw=0.5):
    segs = []
    for i in range(1, tree.n):
        p = tree.parent[i]
        if p >= 0:
            segs.append((tree.pts[p], tree.pts[i]))
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw))


def reference_optimum(env, cells=400, rng=None):
    """A near-optimal cost, for judging how good a sampled path is.

    We rasterise the environment onto a fine grid, run A* from project 31, and
    then straighten the result with line-of-sight shortcutting.  A grid path is
    a few percent too long by construction (project 31, experiment 7), and the
    shortcutting removes almost all of that -- so this number is a tight upper
    bound on the true optimum, which is exactly what we need.
    """
    rng = rng or np.random.default_rng(0)
    xs = np.linspace(env.lo[0], env.hi[0], cells)
    ys = np.linspace(env.lo[1], env.hi[1], cells)
    X, Y = np.meshgrid(xs, ys)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    occ = ~env.points_free(pts).reshape(cells, cells)
    scale = (env.hi[0] - env.lo[0]) / (cells - 1)

    def to_cell(p):
        return (int(round((p[1] - env.lo[1]) / scale)),
                int(round((p[0] - env.lo[0]) / scale)))

    s, g = to_cell(START), to_cell(GOAL)
    occ[s] = occ[g] = False
    res = search(occ, s, g, heuristic="octile")
    if not res["found"]:
        return math.inf
    sm = shortcut_los(occ, res["path"], rng, iters=800)
    return path_length(sm) * scale


# =====================================================================  1
def exp1_growth(rng):
    banner("1. How the tree grows, and the bias nobody programmed in")

    env = world_blobs(np.random.default_rng(5))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    for ax, n in zip(axes, (100, 400, 1500, 6000)):
        r = np.random.default_rng(7)
        tree, path, st = rrt(env, START, GOAL, r, step=0.5, goal_bias=0.0,
                             max_iters=n, goal_tol=0.3, stop_at_goal=False)
        draw_env(ax, env)
        draw_tree(ax, tree)
        ax.plot(*START, "o", color=COLORS[2], ms=6)
        ax.plot(*GOAL, "*", color=COLORS[3], ms=12)
        ax.set_title(f"{n} samples -> {tree.n} nodes")
    save(fig, os.path.join(OUT, "tree_growth.png"))

    # The "rapidly-exploring" claim, measured: how far from the tree's centre
    # of mass does a new node land, early versus late?
    r = np.random.default_rng(7)
    tree, path, st = rrt(env, START, GOAL, r, step=0.5, goal_bias=0.0,
                         max_iters=6000, goal_tol=0.3, stop_at_goal=False)
    pts = tree.pts[:tree.n]
    print(f"  {st['iters']} samples -> {tree.n} nodes "
          f"({100*st['extended']/st['iters']:.1f}% of samples extended the tree)")
    print(f"  {st['checks']} collision checks, {st['time']:.2f} s")
    record(1, "growth", samples=st["iters"], nodes=int(tree.n),
           extend_rate_pct=round(100 * st["extended"] / st["iters"], 2),
           checks=st["checks"], seconds=round(st["time"], 3))

    # Coverage: what fraction of the free space is within `step` of some node?
    probe = env.lo + np.random.default_rng(1).random((4000, 2)) * (env.hi - env.lo)
    probe = probe[env.points_free(probe)]
    for n in (100, 400, 1500, 6000):
        sub = pts[:min(n, tree.n)]
        d = np.full(len(probe), np.inf)
        for c0 in range(0, len(sub), 512):          # chunked: keep memory small
            blk = sub[c0:c0 + 512]
            dd = np.linalg.norm(probe[:, None, :] - blk[None, :, :], axis=2)
            d = np.minimum(d, dd.min(axis=1))
        cov = 100 * np.mean(d < 0.5)
        print(f"  after {n:5d} nodes: {cov:5.1f}% of free space is within one "
              f"step of the tree")
        record(1, f"coverage_{n}", pct=round(float(cov), 2))


# =====================================================================  2
def exp2_goal_bias(rng):
    banner("2. Goal bias: 0% never aims, 100% is greedy, and both are bad")

    envs = [world_blobs(np.random.default_rng(100 + k)) for k in range(6)]
    biases = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.9, 1.0]
    print(f"  {'bias':>6s} {'success':>8s} {'mean iters':>11s} "
          f"{'mean nodes':>11s} {'mean cost':>10s}")
    rows = []
    for b in biases:
        it, nd, ct, ok = [], [], [], 0
        for k, env in enumerate(envs):
            for s in range(5):
                r = np.random.default_rng(1000 * k + s)
                tree, path, st = rrt(env, START, GOAL, r, step=0.5,
                                     goal_bias=b, max_iters=6000)
                if st["found"]:
                    ok += 1
                    it.append(st["iters"])
                    nd.append(st["nodes"])
                    ct.append(st["cost"])
        n = len(envs) * 5
        rows.append((b, 100 * ok / n, np.mean(it) if it else math.nan,
                     np.mean(nd) if nd else math.nan,
                     np.mean(ct) if ct else math.nan))
        print(f"  {b:6.2f} {100*ok/n:7.0f}% {rows[-1][2]:11.0f} "
              f"{rows[-1][3]:11.0f} {rows[-1][4]:10.3f}")
        record(2, f"bias_{b}", success_pct=round(100 * ok / n, 1),
               mean_iters=round(float(rows[-1][2]), 1),
               mean_nodes=round(float(rows[-1][3]), 1),
               mean_cost=round(float(rows[-1][4]), 4))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    bb = [r[0] for r in rows]
    ax.plot(bb, [r[2] for r in rows], "o-", color=COLORS[0], label="samples used")
    ax.set_xlabel("goal bias (probability the sample IS the goal)")
    ax.set_ylabel("samples until success")
    ax2 = ax.twinx()
    ax2.plot(bb, [r[1] for r in rows], "s--", color=COLORS[1], label="success %")
    ax2.set_ylabel("success rate (%)")
    ax2.grid(False)
    ax.set_title("Aiming at the goal helps -- until it is all you do")
    save(fig, os.path.join(OUT, "goal_bias.png"))


# =====================================================================  3
def exp3_step_size(rng):
    banner("3. Step size: node count against rejection rate")

    envs = [world_blobs(np.random.default_rng(200 + k)) for k in range(6)]
    steps = [0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 2.0, 3.0]
    print(f"  {'step':>6s} {'success':>8s} {'nodes':>8s} {'rejected %':>11s} "
          f"{'checks':>9s} {'cost':>8s} {'ms':>8s}")
    rows = []
    for s in steps:
        nd, rej, ck, ct, tm, ok = [], [], [], [], [], 0
        for k, env in enumerate(envs):
            for sd in range(4):
                r = np.random.default_rng(3000 * k + sd)
                tree, path, st = rrt(env, START, GOAL, r, step=s,
                                     goal_bias=0.05, max_iters=12000,
                                     goal_tol=max(0.3, s))
                if st["found"]:
                    ok += 1
                    nd.append(st["nodes"])
                    rej.append(100 * (1 - st["extended"] / st["iters"]))
                    ck.append(st["checks"])
                    ct.append(st["cost"])
                    tm.append(st["time"] * 1e3)
        n = len(envs) * 4
        rows.append((s, 100 * ok / n, np.mean(nd), np.mean(rej), np.mean(ck),
                     np.mean(ct), np.mean(tm)))
        print(f"  {s:6.2f} {100*ok/n:7.0f}% {np.mean(nd):8.0f} "
              f"{np.mean(rej):11.1f} {np.mean(ck):9.0f} {np.mean(ct):8.3f} "
              f"{np.mean(tm):8.1f}")
        record(3, f"step_{s}", success_pct=round(100 * ok / n, 1),
               nodes=round(float(np.mean(nd)), 1),
               rejected_pct=round(float(np.mean(rej)), 2),
               checks=round(float(np.mean(ck)), 0),
               cost=round(float(np.mean(ct)), 4),
               ms=round(float(np.mean(tm)), 2))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot([r[0] for r in rows], [r[6] for r in rows], "o-", color=COLORS[0])
    ax.set_xlabel("step size (m)")
    ax.set_ylabel("time to first solution (ms)")
    ax2 = ax.twinx()
    ax2.plot([r[0] for r in rows], [r[3] for r in rows], "s--", color=COLORS[1])
    ax2.set_ylabel("samples rejected by collision (%)")
    ax2.grid(False)
    ax.set_xscale("log")
    ax.set_title("Small steps: many nodes.  Big steps: everything is rejected.")
    save(fig, os.path.join(OUT, "step_size.png"))


# =====================================================================  4
def exp4_rrt_star(rng):
    banner("4. RRT never improves; RRT* does")

    env = world_blobs(np.random.default_rng(5))
    opt = reference_optimum(env)
    print(f"  near-optimal cost from A* + line-of-sight shortcutting: {opt:.3f}")
    record(4, "reference_optimum", cost=round(opt, 4))

    # Both planners keep sampling to the same budget, and we take the best
    # goal connection either of them has found at each checkpoint.  Stopping
    # RRT at its first solution would make "RRT does not improve" a tautology.
    iters = [500, 1000, 2000, 4000, 8000, 16000]
    n_seeds = 8
    N_MAX = iters[-1]
    rrt_hist, star_hist = [], []
    for s in range(n_seeds):
        _, _, st = rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                       goal_bias=0.05, max_iters=N_MAX, stop_at_goal=False,
                       record_every=250)
        rrt_hist.append(dict(st["history"]))
        _, _, st2 = rrt_star(env, START, GOAL, np.random.default_rng(s),
                             step=0.5, goal_bias=0.05, max_iters=N_MAX,
                             record_every=250)
        star_hist.append(dict(st2["history"]))

    def at(hists, N):
        vals = [h[N] for h in hists if math.isfinite(h.get(N, math.inf))]
        return np.mean(vals) if vals else math.nan

    rrt_curve = [at(rrt_hist, N) for N in iters]
    star_curve = [at(star_hist, N) for N in iters]
    for N, a, b in zip(iters, rrt_curve, star_curve):
        print(f"  {N:6d} samples: RRT {a:7.3f} ({100*(a/opt-1):5.1f}% over)   "
              f"RRT* {b:7.3f} ({100*(b/opt-1):5.1f}% over)")
        record(4, f"iters_{N}", rrt=round(float(a), 4),
               rrt_star=round(float(b), 4),
               rrt_excess_pct=round(100 * (a / opt - 1), 2),
               star_excess_pct=round(100 * (b / opt - 1), 2))
    print(f"  over 16000 samples RRT improved by "
          f"{100*(1 - rrt_curve[-1]/rrt_curve[0]):.1f}% and RRT* by "
          f"{100*(1 - star_curve[-1]/star_curve[0]):.1f}%")
    record(4, "improvement_500_to_16000",
           rrt_pct=round(100 * (1 - rrt_curve[-1] / rrt_curve[0]), 2),
           star_pct=round(100 * (1 - star_curve[-1] / star_curve[0]), 2))

    _, _, sth = rrt_star(env, START, GOAL, np.random.default_rng(0), step=0.5,
                         goal_bias=0.05, max_iters=16000, record_every=250)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.7))
    axes[0].plot(iters, rrt_curve, "o-", color=COLORS[1], label="RRT")
    axes[0].plot(iters, star_curve, "s-", color=COLORS[0], label="RRT*")
    axes[0].axhline(opt, color=COLORS[2], ls="--", label="near-optimal")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("samples")
    axes[0].set_ylabel("path cost (m)")
    axes[0].set_title("More samples do not make RRT better")
    axes[0].legend()
    _, _, rth = rrt(env, START, GOAL, np.random.default_rng(0), step=0.5,
                    goal_bias=0.05, max_iters=16000, stop_at_goal=False,
                    record_every=250)
    for hh, c, nm in ((rth["history"], COLORS[1], "RRT"),
                      (sth["history"], COLORS[0], "RRT*")):
        h = [x for x in hh if math.isfinite(x[1])]
        axes[1].plot([x[0] for x in h], [x[1] for x in h], color=c, label=nm)
    axes[1].axhline(opt, color=COLORS[2], ls="--", label="near-optimal")
    axes[1].set_xlabel("samples")
    axes[1].set_ylabel("best cost so far (m)")
    axes[1].set_title("One seed each, both still sampling")
    axes[1].legend()
    save(fig, os.path.join(OUT, "rrt_star.png"))

    # cost of the improvement
    t_rrt = np.mean([rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                         goal_bias=0.05, max_iters=4000)[2]["time"]
                     for s in range(4)])
    t_star = np.mean([rrt_star(env, START, GOAL, np.random.default_rng(s),
                               step=0.5, goal_bias=0.05, max_iters=4000)[2]["time"]
                      for s in range(4)])
    print(f"  4000 samples: RRT {t_rrt*1e3:.0f} ms, RRT* {t_star*1e3:.0f} ms "
          f"({t_star/t_rrt:.1f}x)")
    record(4, "cost_of_star", rrt_ms=round(t_rrt * 1e3, 1),
           star_ms=round(t_star * 1e3, 1), ratio=round(t_star / t_rrt, 2))


# =====================================================================  5
def exp5_narrow(rng):
    banner("5. The narrow passage")

    gaps = [1.0, 0.6, 0.4, 0.25, 0.15, 0.10, 0.07, 0.05]
    n_seeds = 20
    print(f"  wall 4 m thick, corridor width varies; 20 seeds, 6000 samples each")
    print(f"  {'gap (m)':>8s} {'corridor % of free area':>24s} {'success':>8s} "
          f"{'mean samples':>13s}")
    rows = []
    for g in gaps:
        env = world_narrow(g, thickness=4.0)
        ok, it = 0, []
        for s in range(n_seeds):
            _, _, st = rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                           goal_bias=0.05, max_iters=6000)
            if st["found"]:
                ok += 1
                it.append(st["iters"])
        rows.append((g, 100 * ok / n_seeds, np.mean(it) if it else math.nan))
        print(f"  {g:8.2f} {100*g*4/(100-4*10+g*4):24.3f} "
              f"{100*ok/n_seeds:7.0f}% {rows[-1][2]:13.0f}")
        record(5, f"gap_{g}", success_pct=round(100 * ok / n_seeds, 1),
               corridor_area_pct=round(100 * g * 4 / (100 - 4 * 10 + g * 4), 4),
               mean_samples=round(float(rows[-1][2]), 1) if it else "")

    # does throwing samples at it help?
    env = world_narrow(0.05, thickness=4.0)
    print("  gap 0.05 m, more budget:")
    for N in (6000, 20000, 60000):
        ok = sum(rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                     goal_bias=0.05, max_iters=N)[2]["found"]
                 for s in range(12))
        print(f"    {N:6d} samples -> {100*ok/12:.0f}% success")
        record(5, f"gap005_budget_{N}", success_pct=round(100 * ok / 12, 1))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    axes[0].plot([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0])
    axes[0].set_xlabel("gap width (m)")
    axes[0].set_ylabel("success within 6000 samples (%)")
    axes[0].set_title("Narrow the corridor and the planner degrades")
    axes[0].set_xscale("log")
    env = world_narrow(0.25, thickness=4.0)
    tree, path, st = rrt(env, START, GOAL, np.random.default_rng(3), step=0.5,
                         goal_bias=0.05, max_iters=6000)
    draw_env(axes[1], env)
    draw_tree(axes[1], tree)
    if path:
        p = np.asarray(path)
        axes[1].plot(p[:, 0], p[:, 1], color=COLORS[1], lw=2)
    axes[1].plot(*START, "o", color=COLORS[2], ms=6)
    axes[1].plot(*GOAL, "*", color=COLORS[3], ms=12)
    axes[1].set_title(f"gap 0.25 m: {tree.n} nodes to thread it")
    save(fig, os.path.join(OUT, "narrow_passage.png"))


# =====================================================================  6
def exp6_variance(rng):
    banner("6. Run-to-run variance: one run tells you almost nothing")

    env = world_blobs(np.random.default_rng(5))
    opt = reference_optimum(env)
    costs, iters = [], []
    for s in range(200):
        _, _, st = rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                       goal_bias=0.05, max_iters=8000)
        if st["found"]:
            costs.append(st["cost"])
            iters.append(st["iters"])
    costs = np.array(costs)
    iters = np.array(iters)
    print(f"  {len(costs)}/200 runs succeeded")
    print(f"  cost   : min {costs.min():.2f}  median {np.median(costs):.2f}  "
          f"max {costs.max():.2f}  (optimum {opt:.2f})")
    print(f"  the best run is {100*(costs.min()/opt-1):.0f}% over optimal, "
          f"the worst is {100*(costs.max()/opt-1):.0f}%")
    print(f"  samples: min {iters.min()}  median {int(np.median(iters))}  "
          f"max {iters.max()}  ({iters.max()/iters.min():.0f}x spread)")
    record(6, "cost_spread", n=len(costs), min=round(float(costs.min()), 3),
           median=round(float(np.median(costs)), 3),
           max=round(float(costs.max()), 3), optimum=round(opt, 3))
    record(6, "sample_spread", min=int(iters.min()),
           median=int(np.median(iters)), max=int(iters.max()))

    # best-of-k: the cheapest way to buy quality from a random planner
    print(f"  {'k':>3s} {'expected best-of-k cost':>24s} {'% over optimal':>15s}")
    for k in (1, 2, 5, 10, 20):
        best = [np.min(np.random.default_rng(9000 + t).choice(costs, k))
                for t in range(400)]
        print(f"  {k:>3d} {np.mean(best):24.3f} "
              f"{100*(np.mean(best)/opt-1):15.1f}")
        record(6, f"best_of_{k}", cost=round(float(np.mean(best)), 4),
               excess_pct=round(100 * (float(np.mean(best)) / opt - 1), 2))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5))
    axes[0].hist(costs, bins=30, color=COLORS[0])
    axes[0].axvline(opt, color=COLORS[2], ls="--", label="near-optimal")
    axes[0].set_xlabel("path cost (m)")
    axes[0].set_ylabel("runs")
    axes[0].set_title("200 runs, identical problem, only the seed differs")
    axes[0].legend()
    axes[1].hist(iters, bins=30, color=COLORS[1])
    axes[1].set_xlabel("samples used")
    axes[1].set_title("Planning TIME is even more variable than cost")
    save(fig, os.path.join(OUT, "variance.png"))


# =====================================================================  7
def exp7_where_time_goes(rng):
    banner("7. Where the time goes")

    env = world_blobs(np.random.default_rng(5))
    r = np.random.default_rng(7)
    tree, path, st = rrt(env, START, GOAL, r, step=0.5, goal_bias=0.0,
                         max_iters=6000, stop_at_goal=False)

    # time one collision check and one nearest-neighbour query in isolation
    pts = np.random.default_rng(0).random((2000, 2)) * 10
    t0 = time.perf_counter()
    for i in range(2000):
        env.segment_free(pts[i], pts[i] + 0.5, 0.05)
    t_seg = (time.perf_counter() - t0) / 2000

    print(f"  {'tree nodes':>11s} {'nearest-neighbour query (us)':>30s}")
    nn_rows = []
    for n in (100, 500, 2000, 8000, 32000, 128000):
        t = Tree(np.zeros(2), cap=n + 1)
        t.pts[:n] = np.random.default_rng(1).random((n, 2)) * 10
        t.n = n
        q = np.array([5.0, 5.0])
        t0 = time.perf_counter()
        for _ in range(300):
            t.nearest(q)
        us = (time.perf_counter() - t0) / 300 * 1e6
        nn_rows.append((n, us))
        print(f"  {n:11d} {us:30.1f}")
        record(7, f"nn_{n}", microseconds=round(us, 2))

    total = st["time"]
    est_seg = t_seg * st["iters"]      # one segment test per sample drawn
    print(f"  one segment check: {t_seg*1e6:.1f} us")
    print(f"  a 6000-sample run: {total*1e3:.0f} ms total, of which roughly "
          f"{100*est_seg/total:.0f}% is collision checking")
    print(f"  {st['checks']} point-collision tests were done "
          f"({st['checks']/st['iters']:.1f} per sample)")
    record(7, "budget", total_ms=round(total * 1e3, 1),
           seg_check_us=round(t_seg * 1e6, 2),
           collision_pct=round(100 * est_seg / total, 1),
           point_tests=st["checks"])

    # nearest-neighbour is linear per query, so quadratic over a whole run
    ns = np.array([r0[0] for r0 in nn_rows], dtype=float)
    us = np.array([r0[1] for r0 in nn_rows], dtype=float)
    big = ns >= 2000        # small n is dominated by NumPy call overhead
    slope = np.polyfit(np.log(ns[big]), np.log(us[big]), 1)[0]
    print(f"  nearest-neighbour query time scales as n^{slope:.2f} for "
          f"n >= 2000 (brute force is linear, so 1.0 is the theory)")
    record(7, "nn_exponent", exponent=round(float(slope), 3))

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.loglog(ns, us, "o-", color=COLORS[0])
    ax.set_xlabel("nodes in the tree")
    ax.set_ylabel("one nearest-neighbour query (us)")
    ax.set_title(f"Brute-force nearest neighbour: n^{slope:.2f}\n"
                 "(a k-d tree would make this log n -- and that is what OMPL uses)")
    save(fig, os.path.join(OUT, "time_budget.png"))


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_growth(rng)
    exp2_goal_bias(rng)
    exp3_step_size(rng)
    exp4_rrt_star(rng)
    exp5_narrow(rng)
    exp6_variance(rng)
    exp7_where_time_goes(rng)

    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\nwrote {os.path.join(OUT, 'results.csv')}  ({len(RESULTS)} rows)")


if __name__ == "__main__":
    main()
