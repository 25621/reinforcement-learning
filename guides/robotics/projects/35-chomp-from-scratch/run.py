"""Project 35 -- CHOMP from scratch.

Seven experiments:

  1. the signed distance field, and a straight line pushed out of collision
  2. what "covariant" buys: the same cost, two different rulers
  3. the local minimum you can see coming, and the one you cannot
  4. initialisation decides everything -- including whether to use a planner
  5. STOMP: the same cost, no gradient, and where that pays off
  6. the clearance dial, and what it costs in length
  7. resolution: waypoints, grid cells, and where A^-1 gets its smoothing from

Runs in about six minutes.  NumPy and Matplotlib only.
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
sys.path.insert(0, os.path.join(_PROJ, "32-rrt-in-2d"))
sys.path.insert(0, os.path.join(_PROJ, "34-shortcut-smoothing"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from chomp import (build_sdf, chomp, stomp, obstacle_cost, smoothness_matrices,  # noqa: E402
                   trajectory_report, path_min_clearance)
from rrt import Env, rrt                                                   # noqa: E402
from smooth import shortcut, resample, max_turn                            # noqa: E402
from plot_style import COLORS, use_style, save                             # noqa: E402

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import Circle                                      # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []
Q0 = np.array([0.5, 0.5])
QN = np.array([9.5, 9.5])

# The default scene.  The big obstacle sits slightly off the straight line so
# the gradient has a side to prefer; experiment 3 removes that nudge.
SCENE = [[4.4, 5.4, 1.6], [3.0, 7.5, 0.9], [7.6, 3.4, 1.0]]


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def draw_scene(ax, circles, sdf=None, field=False):
    if field and sdf is not None:
        ax.imshow(sdf.field, origin="lower", extent=[0, 10, 0, 10],
                  cmap="RdBu", vmin=-2, vmax=4)
    for cx, cy, r in np.atleast_2d(circles):
        ax.add_patch(Circle((cx, cy), r, color="#4A4A4A",
                            fill=not field, ec="#4A4A4A", lw=1.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def random_scene(rng, n=5):
    cs = []
    while len(cs) < n:
        c = rng.uniform(1.5, 8.5, 2)
        r = rng.uniform(0.7, 1.5)
        if np.linalg.norm(c - Q0) < r + 0.8 or np.linalg.norm(c - QN) < r + 0.8:
            continue
        cs.append([c[0], c[1], r])
    return cs


def curvature_cost(path):
    p = np.asarray(path, float)
    acc = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.sum(acc ** 2))


# =====================================================================  1
def exp1_sdf(rng):
    banner("1. The signed distance field, and a line pushed out of collision")

    t0 = time.perf_counter()
    sdf = build_sdf(SCENE, cells=241)
    print(f"  241x241 SDF built in {(time.perf_counter()-t0)*1e3:.0f} ms "
          f"(exact Felzenszwalb distance transform, two 1-D passes)")
    print(f"  field ranges {sdf.field.min():.2f} m (inside) to "
          f"{sdf.field.max():.2f} m (far outside)")
    record(1, "sdf", ms=round((time.perf_counter() - t0) * 1e3, 1),
           min_m=round(float(sdf.field.min()), 3),
           max_m=round(float(sdf.field.max()), 3))

    path, hist = chomp(sdf, Q0, QN, n=80, iters=400, eta=200.0, lam=1.0,
                       eps=0.5, track=True)
    first_free = next((i + 1 for i, h in enumerate(hist)
                       if not h["collision"]), None)
    r = hist[-1]
    print(f"  straight-line start: length "
          f"{hist[0]['length']:.3f} m, deepest penetration "
          f"{hist[0]['min_d']:.3f} m")
    print(f"  collision-free after {first_free} iterations")
    print(f"  final: length {r['length']:.3f} m, clearance "
          f"{r['min_d']:.3f} m, smoothness cost {r['smooth']:.4f}")
    record(1, "chomp_run", iters_to_free=first_free,
           start_length=round(hist[0]["length"], 4),
           start_min_d=round(hist[0]["min_d"], 4),
           final_length=round(r["length"], 4),
           final_min_d=round(r["min_d"], 4),
           final_smooth=round(r["smooth"], 5))

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.9))
    im = axes[0].imshow(sdf.field, origin="lower", extent=[0, 10, 0, 10],
                        cmap="RdBu", vmin=-2, vmax=4)
    axes[0].contour(np.linspace(0, 10, 241), np.linspace(0, 10, 241),
                    sdf.field, levels=[0.0], colors="k", linewidths=1.2)
    axes[0].set_title("signed distance field\n(black = obstacle surface)")
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    cost_grid = obstacle_cost(sdf.field, 0.5)
    im2 = axes[1].imshow(cost_grid, origin="lower", extent=[0, 10, 0, 10],
                         cmap="magma_r")
    axes[1].set_title("obstacle cost c(d), eps = 0.5 m\n(zero everywhere it is safe)")
    axes[1].set_aspect("equal")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].grid(False)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    draw_scene(axes[2], SCENE)
    for k, it in enumerate((0, 5, 20, 60, 150, 399)):
        p, _ = chomp(sdf, Q0, QN, n=80, iters=it + 1, eta=200.0, track=True)
        axes[2].plot(p[:, 0], p[:, 1], color=COLORS[k % 7], lw=1.4,
                     label=f"iter {it}")
    axes[2].legend(fontsize=7)
    axes[2].set_title("the trajectory climbing out")
    save(fig, os.path.join(OUT, "sdf_and_run.png"))


# =====================================================================  2
def exp2_covariant(rng):
    banner("2. What 'covariant' buys")

    sdf = build_sdf(SCENE, cells=241)
    print(f"  {'1/step':>8s}  {'covariant: iters to free':>26s} "
          f"{'length':>8s} {'curvature':>10s} |  "
          f"{'plain: iters to free':>21s} {'length':>8s} {'curvature':>10s}")
    rows = []
    for eta in (500.0, 200.0, 100.0, 50.0, 20.0):
        line = [eta]
        for cov in (True, False):
            p, h = chomp(sdf, Q0, QN, n=80, iters=1500, eta=eta, lam=1.0,
                         eps=0.5, covariant=cov, track=True)
            # hist[i] is the state AFTER iteration i+1, so add one to report
            # a count of iterations rather than an index.
            ff = next((i + 1 for i, x in enumerate(h)
                       if not x["collision"]), None)
            line += [ff, h[-1]["length"], curvature_cost(p)]
        rows.append(line)
        print(f"  {eta:8.0f}  {str(line[1]):>26s} {line[2]:8.3f} "
              f"{line[3]:10.4f} |  {str(line[4]):>21s} {line[5]:8.3f} "
              f"{line[6]:10.4f}")
        record(2, f"eta_{eta}", cov_iters=line[1], cov_length=round(line[2], 4),
               cov_curv=round(line[3], 5), plain_iters=line[4],
               plain_length=round(line[5], 4), plain_curv=round(line[6], 5))

    # how many cost evaluations each needs at its own best setting
    best_cov = min((r for r in rows if r[1] is not None), key=lambda r: r[1])
    plain_ok = [r for r in rows if r[4] is not None]
    if plain_ok:
        best_pl = min(plain_ok, key=lambda r: r[4])
        print(f"  best covariant: {best_cov[1]} iterations; "
              f"best plain: {best_pl[4]} -> {best_pl[4]/best_cov[1]:.0f}x more")
        record(2, "best_case", covariant=best_cov[1], plain=best_pl[4],
               ratio=round(best_pl[4] / best_cov[1], 1))
    else:
        print("  plain gradient descent never reached a collision-free "
              "trajectory at any step size tried")
        record(2, "best_case", covariant=best_cov[1], plain="never")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for cov, c, nm in ((True, COLORS[0], "covariant (A^-1 step)"),
                       (False, COLORS[1], "plain gradient")):
        _, h = chomp(sdf, Q0, QN, n=80, iters=1500, eta=200.0, covariant=cov,
                     track=True)
        axes[0].semilogy([x["total"] for x in h], color=c, label=nm)
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("total cost")
    axes[0].set_title("Same cost function, same step size")
    axes[0].legend(fontsize=8)
    draw_scene(axes[1], SCENE)
    for cov, c, nm in ((True, COLORS[0], "covariant"),
                       (False, COLORS[1], "plain")):
        p = chomp(sdf, Q0, QN, n=80, iters=1500, eta=200.0, covariant=cov)
        axes[1].plot(p[:, 0], p[:, 1], color=c, lw=1.6, label=nm)
    axes[1].legend(fontsize=8)
    axes[1].set_title("...and two very different answers")
    save(fig, os.path.join(OUT, "covariant.png"))


# =====================================================================  3
def exp3_local_minima(rng):
    banner("3. Local minima")

    # (a) the one you can see coming: an obstacle centred on the straight line
    sym = [[5.0, 5.0, 1.6]]
    sdf = build_sdf(sym, cells=241)
    p, h = chomp(sdf, Q0, QN, n=80, iters=1500, eta=200.0, track=True)
    print(f"  obstacle centred exactly on the straight line:")
    print(f"    after 1500 iterations, deepest penetration "
          f"{h[-1]['min_d']:.3f} m, still in collision: {h[-1]['collision']}")
    record(3, "symmetric_trap", min_d=round(h[-1]["min_d"], 4),
           collision=h[-1]["collision"])

    # nudge the obstacle and it works instantly
    print(f"  {'offset (m)':>11s} {'escapes?':>9s} {'final clearance':>16s}")
    for off in (0.0, 0.01, 0.05, 0.2, 0.5):
        s = [[5.0 - off / math.sqrt(2), 5.0 + off / math.sqrt(2), 1.6]]
        sd = build_sdf(s, cells=161)
        _, hh = chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0, track=True)
        ok = not hh[-1]["collision"]
        print(f"  {off:11.2f} {str(ok):>9s} {hh[-1]['min_d']:16.3f}")
        record(3, f"offset_{off}", escapes=ok, min_d=round(hh[-1]["min_d"], 4))

    # (b) the one you cannot: random scenes, straight-line init
    n_scenes = 30
    ok, lens = 0, []
    for s in range(n_scenes):
        sc = random_scene(np.random.default_rng(600 + s), n=5)
        sd = build_sdf(sc, cells=161)
        pp = chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0)
        if path_min_clearance(sd, pp) > 0:
            ok += 1
            lens.append(float(np.sum(np.linalg.norm(np.diff(pp, axis=0), axis=1))))
    print(f"  random 5-obstacle scenes, straight-line start: "
          f"{100*ok/n_scenes:.0f}% end collision-free "
          f"({ok}/{n_scenes}), mean length {np.mean(lens):.3f} m")
    record(3, "random_scenes", success_pct=round(100 * ok / n_scenes, 1),
           scenes=n_scenes, mean_length=round(float(np.mean(lens)), 4))

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    draw_scene(axes[0], sym)
    axes[0].plot(p[:, 0], p[:, 1], color=COLORS[1], lw=2)
    axes[0].set_title("Perfectly symmetric: the sideways force cancels\n"
                      "and CHOMP pushes straight into the wall")
    s = [[5.0 - 0.2 / math.sqrt(2), 5.0 + 0.2 / math.sqrt(2), 1.6]]
    sd = build_sdf(s, cells=241)
    p2 = chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0)
    draw_scene(axes[1], s)
    axes[1].plot(p2[:, 0], p2[:, 1], color=COLORS[0], lw=2)
    axes[1].set_title("Move it 20 cm and the tie is broken")
    save(fig, os.path.join(OUT, "local_minima.png"))


# =====================================================================  4
def exp4_initialisation(rng):
    banner("4. Initialisation decides everything")

    n_scenes = 30
    print(f"  {'initialisation':<28s} {'success':>8s} {'mean length':>12s} "
          f"{'mean ms':>9s}")
    summary = {}
    for nm in ("straight line", "straight line x5 random restarts",
               "RRT path, then CHOMP"):
        ok, lens, ms = 0, [], []
        for s in range(n_scenes):
            sc = random_scene(np.random.default_rng(600 + s), n=5)
            sd = build_sdf(sc, cells=161)
            t0 = time.perf_counter()
            best = None
            if nm == "straight line":
                cands = [chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0)]
            elif nm.startswith("straight line x5"):
                cands = [chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0,
                               rng=np.random.default_rng(k), noise=0.6)
                         for k in range(5)]
            else:
                env = Env(circles=sc)
                _, raw, st = rrt(env, Q0, QN, np.random.default_rng(s),
                                 step=0.5, goal_bias=0.05, max_iters=8000)
                if not st["found"]:
                    continue
                init = resample(raw, 12.0 / 81)[1:-1]
                if len(init) != 80:
                    idx = np.linspace(0, len(init) - 1, 80)
                    init = np.stack([np.interp(idx, np.arange(len(init)),
                                               init[:, j]) for j in range(2)], 1)
                cands = [chomp(sd, Q0, QN, iters=1500, eta=200.0, init=init)]
            for cnd in cands:
                if path_min_clearance(sd, cnd) > 0:
                    L = float(np.sum(np.linalg.norm(np.diff(cnd, axis=0), axis=1)))
                    if best is None or L < best:
                        best = L
            ms.append((time.perf_counter() - t0) * 1e3)
            if best is not None:
                ok += 1
                lens.append(best)
        summary[nm] = (100 * ok / n_scenes, np.mean(lens), np.mean(ms))
        print(f"  {nm:<28s} {100*ok/n_scenes:7.0f}% {np.mean(lens):12.3f} "
              f"{np.mean(ms):9.0f}")
        record(4, nm, success_pct=round(100 * ok / n_scenes, 1),
               mean_length=round(float(np.mean(lens)), 4),
               mean_ms=round(float(np.mean(ms)), 1))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(summary)
    ax.barh(names, [summary[n][0] for n in names],
            color=[COLORS[0], COLORS[3], COLORS[2]])
    ax.set_xlabel("scenes ending collision-free (%)")
    ax.set_xlim(0, 105)
    ax.set_title("A local optimiser is only as good as what you hand it")
    save(fig, os.path.join(OUT, "initialisation.png"))


# =====================================================================  5
def exp5_stomp(rng):
    banner("5. STOMP: the same cost, no gradient")

    sym = [[5.0, 5.0, 1.6]]
    sd = build_sdf(sym, cells=241)
    escapes = 0
    for s in range(10):
        p = stomp(sd, Q0, QN, n=80, iters=400, k=12, sigma=0.5,
                  rng=np.random.default_rng(s))
        escapes += path_min_clearance(sd, p) > 0
    print(f"  the symmetric trap that stops CHOMP dead: STOMP escapes "
          f"{escapes}/10 times")
    record(5, "symmetric_trap", stomp_escapes=escapes, of=10, chomp_escapes=0)

    n_scenes = 20
    print(f"  {'method':<12s} {'success':>8s} {'mean length':>12s} "
          f"{'cost evals':>11s} {'mean ms':>9s}")
    for nm in ("CHOMP", "STOMP"):
        ok, lens, ms, ev = 0, [], [], []
        for s in range(n_scenes):
            sc = random_scene(np.random.default_rng(600 + s), n=5)
            sdd = build_sdf(sc, cells=161)
            t0 = time.perf_counter()
            if nm == "CHOMP":
                p = chomp(sdd, Q0, QN, n=80, iters=1500, eta=200.0)
                ev.append(1500)
            else:
                p = stomp(sdd, Q0, QN, n=80, iters=300, k=12, sigma=0.5,
                          rng=np.random.default_rng(s))
                ev.append(300 * 13)
            ms.append((time.perf_counter() - t0) * 1e3)
            if path_min_clearance(sdd, p) > 0:
                ok += 1
                lens.append(float(np.sum(np.linalg.norm(np.diff(p, axis=0),
                                                        axis=1))))
        print(f"  {nm:<12s} {100*ok/n_scenes:7.0f}% "
              f"{np.mean(lens) if lens else math.nan:12.3f} "
              f"{np.mean(ev):11.0f} {np.mean(ms):9.0f}")
        record(5, nm, success_pct=round(100 * ok / n_scenes, 1),
               mean_length=round(float(np.mean(lens)), 4) if lens else "",
               cost_evals=round(float(np.mean(ev))),
               mean_ms=round(float(np.mean(ms)), 1))

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    draw_scene(ax, sym)
    p = chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0)
    ax.plot(p[:, 0], p[:, 1], color=COLORS[1], lw=2, label="CHOMP (stuck)")
    for s in range(3):
        p2 = stomp(sd, Q0, QN, n=80, iters=400, k=12, sigma=0.5,
                   rng=np.random.default_rng(s))
        ax.plot(p2[:, 0], p2[:, 1], color=COLORS[0], lw=1.4,
                label="STOMP" if s == 0 else None)
    ax.legend(fontsize=8)
    ax.set_title("Noise breaks a tie a gradient cannot")
    save(fig, os.path.join(OUT, "stomp.png"))


# =====================================================================  6
def exp6_epsilon(rng):
    banner("6. The clearance dial")

    sdf = build_sdf(SCENE, cells=241)
    print(f"  {'eps (m)':>8s} {'length':>8s} {'true clearance':>15s} "
          f"{'clearance / eps':>16s}")
    rows = []
    for eps in (0.05, 0.1, 0.25, 0.5, 0.8, 1.2):
        p = chomp(sdf, Q0, QN, n=80, iters=1500, eta=200.0, eps=eps)
        L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        cl = path_min_clearance(sdf, p)
        rows.append((eps, L, cl))
        print(f"  {eps:8.2f} {L:8.3f} {cl:15.3f} {cl/eps:16.2f}")
        record(6, f"eps_{eps}", length=round(L, 4), clearance=round(cl, 4),
               ratio=round(cl / eps, 3))
    base = rows[0][1]
    print(f"  going from eps = 0.05 m to eps = 1.2 m buys "
          f"{rows[-1][2]-rows[0][2]:.2f} m of clearance for "
          f"{100*(rows[-1][1]/base-1):+.1f}% length")
    record(6, "tradeoff", clearance_gain=round(rows[-1][2] - rows[0][2], 3),
           length_pct=round(100 * (rows[-1][1] / base - 1), 2))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    axes[0].plot([r[0] for r in rows], [r[2] for r in rows], "o-",
                 color=COLORS[0], label="achieved clearance")
    axes[0].plot([r[0] for r in rows], [r[0] for r in rows], "--",
                 color=COLORS[2], label="eps (what you asked for)")
    axes[0].set_xlabel("eps (m)")
    axes[0].set_ylabel("metres")
    axes[0].legend(fontsize=8)
    axes[0].set_title("You get roughly what you ask for")
    draw_scene(axes[1], SCENE)
    for k, eps in enumerate((0.05, 0.25, 0.5, 1.2)):
        p = chomp(sdf, Q0, QN, n=80, iters=1500, eta=200.0, eps=eps)
        axes[1].plot(p[:, 0], p[:, 1], color=COLORS[k], lw=1.5,
                     label=f"eps={eps}")
    axes[1].legend(fontsize=7)
    axes[1].set_title("Wider buffer, wider berth")
    save(fig, os.path.join(OUT, "epsilon.png"))


# =====================================================================  7
def exp7_resolution(rng):
    banner("7. Resolution, and where the smoothing comes from")

    print(f"  {'waypoints':>10s} {'length':>8s} {'clearance':>10s} "
          f"{'ms':>8s}")
    sdf = build_sdf(SCENE, cells=241)
    for n in (20, 40, 80, 160, 320):
        t0 = time.perf_counter()
        p = chomp(sdf, Q0, QN, n=n, iters=1500, eta=200.0)
        ms = (time.perf_counter() - t0) * 1e3
        L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        cl = path_min_clearance(sdf, p)
        print(f"  {n:10d} {L:8.3f} {cl:10.3f} {ms:8.0f}")
        record(7, f"waypoints_{n}", length=round(L, 4), clearance=round(cl, 4),
               ms=round(ms, 1))

    print(f"  {'SDF cells':>10s} {'cell size (m)':>14s} {'length':>8s} "
          f"{'clearance':>10s} {'build ms':>9s}")
    for cells in (41, 81, 161, 241, 401):
        t0 = time.perf_counter()
        sd = build_sdf(SCENE, cells=cells)
        bms = (time.perf_counter() - t0) * 1e3
        p = chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0)
        ref = build_sdf(SCENE, cells=401)
        L = float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))
        cl = path_min_clearance(ref, p)
        print(f"  {cells:10d} {10.0/(cells-1):14.4f} {L:8.3f} {cl:10.3f} "
              f"{bms:9.0f}")
        record(7, f"cells_{cells}", cell_m=round(10.0 / (cells - 1), 5),
               length=round(L, 4), true_clearance=round(cl, 4),
               build_ms=round(bms, 1))

    # what A^-1 actually does: apply it to a single spike
    n = 60
    A, b, K, e0 = smoothness_matrices(n, np.zeros(2), np.zeros(2), 1)
    Ainv = np.linalg.inv(A)
    spike = np.zeros((n, 2))
    spike[n // 2, 0] = 1.0
    spread = Ainv @ spike
    width = int(np.sum(np.abs(spread[:, 0]) > 0.05 * np.abs(spread).max()))
    print(f"  a one-waypoint kick, passed through A^-1, spreads over {width} "
          f"waypoints instead of 1")
    print(f"  that is the entire trick: A^-1 turns 'move this point' into "
          f"'move this stretch of trajectory'")
    record(7, "ainv_spread", waypoints=width, of=n)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    axes[0].plot(spike[:, 0], color=COLORS[1], label="raw gradient (one point)")
    axes[0].plot(spread[:, 0] / spread[:, 0].max(), color=COLORS[0],
                 label="after A^-1 (normalised)")
    axes[0].set_xlabel("waypoint index")
    axes[0].set_title("A^-1 is a smoothing filter")
    axes[0].legend(fontsize=8)
    draw_scene(axes[1], SCENE)
    for k, cells in enumerate((41, 81, 241)):
        sd = build_sdf(SCENE, cells=cells)
        p = chomp(sd, Q0, QN, n=80, iters=1500, eta=200.0)
        axes[1].plot(p[:, 0], p[:, 1], color=COLORS[k], lw=1.5,
                     label=f"{cells} cells")
    axes[1].legend(fontsize=7)
    axes[1].set_title("A coarse field rounds off the obstacles")
    save(fig, os.path.join(OUT, "resolution.png"))


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_sdf(rng)
    exp2_covariant(rng)
    exp3_local_minima(rng)
    exp4_initialisation(rng)
    exp5_stomp(rng)
    exp6_epsilon(rng)
    exp7_resolution(rng)

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
