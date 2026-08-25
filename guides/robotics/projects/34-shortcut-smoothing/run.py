"""Project 34 -- shortcut smoothing: how much of a sampled path is waste.

Seven experiments:

  1. before and after, and the shape of the improvement curve
  2. the two settings that matter: how many tries, and how finely you re-space
  3. a 7-joint arm: full shortcuts against one-joint-at-a-time shortcuts
  4. what shortcutting CANNOT fix: the route it was handed
  5. the bill: collision checks spent planning against checks spent smoothing
  6. the corners it leaves behind, and what they cost a real robot
  7. the practical question: RRT + shortcut, or RRT*, for the same time?

Runs in about five minutes.  Needs mujoco (for experiment 3), NumPy, Matplotlib.
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
sys.path.insert(0, os.path.join(_PROJ, "32-rrt-in-2d"))
sys.path.insert(0, os.path.join(_PROJ, "33-rrt-connect-for-an-arm"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from smooth import (shortcut, partial_shortcut, resample, blend_corners,  # noqa: E402
                    max_turn, turn_profile, min_turn_radius, side_signature,
                    path_cost)
from rrt import Env, world_blobs, rrt, rrt_star                            # noqa: E402
from grid import search, shortcut_los, path_length                         # noqa: E402
from plot_style import COLORS, use_style, save                             # noqa: E402

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import Circle, Rectangle                           # noqa: E402

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


def two_route_world():
    """One fat obstacle, nudged off-centre: two ways round, one of them worse.

    The nudge matters.  A perfectly centred obstacle gives two routes of equal
    length, and then "which one did you get" is a question with no consequence.
    Off-centre, one route is genuinely shorter, so the planner's coin flip
    costs you something measurable.
    """
    return Env(circles=[[4.6, 5.4, 2.6], [8.8, 1.2, 1.0], [1.2, 8.8, 1.0]])


def reference_optimum(env, cells=400):
    xs = np.linspace(env.lo[0], env.hi[0], cells)
    ys = np.linspace(env.lo[1], env.hi[1], cells)
    X, Y = np.meshgrid(xs, ys)
    occ = ~env.points_free(np.stack([X.ravel(), Y.ravel()], 1)).reshape(cells, cells)
    scale = (env.hi[0] - env.lo[0]) / (cells - 1)
    s = (int(round(START[1] / scale)), int(round(START[0] / scale)))
    g = (int(round(GOAL[1] / scale)), int(round(GOAL[0] / scale)))
    occ[s] = occ[g] = False
    res = search(occ, s, g, heuristic="octile")
    sm = shortcut_los(occ, res["path"], np.random.default_rng(0), iters=800)
    return path_length(sm) * scale


# =====================================================================  1
def exp1_before_after(rng):
    banner("1. Before and after")

    env = world_blobs(np.random.default_rng(5))
    opt = reference_optimum(env)
    r = np.random.default_rng(11)
    _, raw, st = rrt(env, START, GOAL, r, step=0.5, goal_bias=0.05,
                     max_iters=8000)

    def seg(a, b):
        return env.segment_free(a, b, 0.02)

    sm, hist = shortcut(raw, seg, np.random.default_rng(1), iters=300,
                        spacing=0.25, track=True)
    print(f"  raw RRT path      : {path_cost(raw):7.3f} m "
          f"({len(raw)} waypoints)")
    print(f"  after shortcutting: {path_cost(sm):7.3f} m "
          f"({len(sm)} waypoints)")
    print(f"  near-optimal      : {opt:7.3f} m")
    print(f"  the shortcut removed {100*(1-path_cost(sm)/path_cost(raw)):.1f}% "
          f"of the length and closed "
          f"{100*(path_cost(raw)-path_cost(sm))/(path_cost(raw)-opt):.0f}% of "
          f"the gap to optimal")
    record(1, "single_path", raw=round(path_cost(raw), 4),
           smoothed=round(path_cost(sm), 4), optimum=round(opt, 4),
           reduction_pct=round(100 * (1 - path_cost(sm) / path_cost(raw)), 2),
           gap_closed_pct=round(100 * (path_cost(raw) - path_cost(sm)) /
                                (path_cost(raw) - opt), 1),
           waypoints_before=len(raw), waypoints_after=len(sm))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, p, t in ((axes[0], raw, f"raw RRT: {path_cost(raw):.2f} m"),
                     (axes[1], sm, f"shortcut: {path_cost(sm):.2f} m")):
        draw_env(ax, env)
        a = np.asarray(p)
        ax.plot(a[:, 0], a[:, 1], color=COLORS[0], lw=2)
        ax.plot(a[:, 0], a[:, 1], ".", color=COLORS[1], ms=3)
        ax.plot(*START, "o", color=COLORS[2], ms=6)
        ax.plot(*GOAL, "*", color=COLORS[3], ms=12)
        ax.set_title(t)
    axes[2].plot([h[0] for h in hist], [h[1] for h in hist], color=COLORS[0])
    axes[2].axhline(opt, color=COLORS[2], ls="--", label="near-optimal")
    axes[2].set_xlabel("shortcut attempts")
    axes[2].set_ylabel("path length (m)")
    axes[2].set_title("Almost all of the gain is in the first few tries")
    axes[2].legend()
    save(fig, os.path.join(OUT, "before_after.png"))

    # where in the curve does the improvement happen?
    total = hist[0][1] - hist[-1][1]
    for k in (5, 10, 25, 50, 100, 200, 300):
        got = hist[0][1] - hist[min(k, len(hist) - 1)][1]
        print(f"  after {k:3d} attempts: {100*got/total:5.1f}% of the total "
              f"improvement")
        record(1, f"attempts_{k}", pct_of_total_gain=round(100 * got / total, 2))


# =====================================================================  2
def exp2_knobs(rng):
    banner("2. Attempts and re-spacing")

    envs = [world_blobs(np.random.default_rng(300 + k)) for k in range(8)]
    raws = []
    for env in envs:
        _, p, _ = rrt(env, START, GOAL, np.random.default_rng(2), step=0.5,
                      goal_bias=0.05, max_iters=8000)
        raws.append((env, p))

    print(f"  {'attempts':>9s} {'mean length':>12s} {'reduction %':>12s} "
          f"{'mean ms':>9s}")
    rows = []
    for iters in (10, 25, 50, 100, 200, 400, 800):
        lens, ms = [], []
        for env, p in raws:
            def seg(a, b, e=env):
                return e.segment_free(a, b, 0.02)
            t0 = time.perf_counter()
            sm = shortcut(p, seg, np.random.default_rng(1), iters=iters,
                          spacing=0.25)
            ms.append((time.perf_counter() - t0) * 1e3)
            lens.append(path_cost(sm) / path_cost(p))
        rows.append((iters, np.mean(lens), np.mean(ms)))
        print(f"  {iters:>9d} {np.mean(lens):12.4f} "
              f"{100*(1-np.mean(lens)):12.2f} {np.mean(ms):9.1f}")
        record(2, f"attempts_{iters}", length_ratio=round(float(np.mean(lens)), 4),
               reduction_pct=round(100 * (1 - float(np.mean(lens))), 2),
               ms=round(float(np.mean(ms)), 2))

    print(f"\n  {'spacing (m)':>12s} {'reduction %':>12s} {'points':>8s} "
          f"{'mean ms':>9s}")
    for spacing in (None, 2.0, 1.0, 0.5, 0.25, 0.1):
        lens, ms, npts = [], [], []
        for env, p in raws:
            def seg(a, b, e=env):
                return e.segment_free(a, b, 0.02)
            t0 = time.perf_counter()
            sm = shortcut(p, seg, np.random.default_rng(1), iters=200,
                          spacing=spacing)
            ms.append((time.perf_counter() - t0) * 1e3)
            lens.append(path_cost(sm) / path_cost(p))
            npts.append(len(resample(p, spacing)) if spacing else len(p))
        lbl = "raw waypoints" if spacing is None else f"{spacing:.2f}"
        print(f"  {lbl:>12s} {100*(1-np.mean(lens)):12.2f} "
              f"{np.mean(npts):8.0f} {np.mean(ms):9.1f}")
        record(2, f"spacing_{lbl}",
               reduction_pct=round(100 * (1 - float(np.mean(lens))), 2),
               points=round(float(np.mean(npts)), 1),
               ms=round(float(np.mean(ms)), 2))

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot([r[0] for r in rows], [100 * (1 - r[1]) for r in rows], "o-",
            color=COLORS[0])
    ax.set_xscale("log")
    ax.set_xlabel("shortcut attempts")
    ax.set_ylabel("length removed (%)")
    ax.set_title("Diminishing returns, and they set in early")
    save(fig, os.path.join(OUT, "attempts.png"))


# =====================================================================  3
def exp3_arm(rng):
    banner("3. A 7-joint arm: full shortcuts against one-joint shortcuts")

    from arm import Arm, rrt_connect, path_cost as arm_cost
    Q_OPEN = np.array([0.7055, -0.8423, 0.7681, -1.8944, -1.2008, 0.0832, -1.0332])
    Q_SHELF = np.array([0.1193, 0.2686, -2.9000, -1.8640, 1.1785, 1.0603, 0.6185])
    arm = Arm()

    paths = []
    for s in range(8):
        _, p, st = rrt_connect(arm, Q_OPEN, Q_SHELF, np.random.default_rng(s),
                               step=0.4, max_iters=8000, res=0.05)
        if st["found"]:
            paths.append(p)
    print(f"  {len(paths)} RRT-Connect plans, mean raw cost "
          f"{np.mean([path_cost(p) for p in paths]):.3f} rad")

    def seg(a, b):
        return arm.segment_free(a, b, 0.02)

    print(f"  {'method':<26s} {'length ratio':>13s} {'reduction %':>12s} "
          f"{'accepted %':>11s} {'ms':>8s}")
    summary = {}
    for nm, fn in (("full shortcut", shortcut),
                   ("one joint at a time", partial_shortcut)):
        costs, ms, acc = [], [], []
        for p in paths:
            arm.n_checks = 0
            t0 = time.perf_counter()
            sm, hist = fn(p, seg, np.random.default_rng(1), iters=200,
                          spacing=0.15, track=True)
            ms.append((time.perf_counter() - t0) * 1e3)
            costs.append(path_cost(sm) / path_cost(p))
            improved = sum(1 for k in range(1, len(hist))
                           if hist[k][1] < hist[k - 1][1] - 1e-9)
            acc.append(100 * improved / (len(hist) - 1))
        summary[nm] = (np.mean(costs), np.mean(acc))
        print(f"  {nm:<26s} {np.mean(costs):13.4f} "
              f"{100*(1-np.mean(costs)):12.2f} {np.mean(acc):11.1f} "
              f"{np.mean(ms):8.0f}")
        record(3, nm, length_ratio=round(float(np.mean(costs)), 4),
               reduction_pct=round(100 * (1 - float(np.mean(costs))), 2),
               accept_pct=round(float(np.mean(acc)), 2),
               ms=round(float(np.mean(ms)), 1))

    # the two combined, which is what real implementations ship
    costs = []
    for p in paths:
        sm = shortcut(p, seg, np.random.default_rng(1), iters=100, spacing=0.15)
        sm = partial_shortcut(sm, seg, np.random.default_rng(2), iters=100)
        costs.append(path_cost(sm) / path_cost(p))
    print(f"  {'both, 100 tries each':<26s} {np.mean(costs):10.4f} "
          f"{100*(1-np.mean(costs)):12.2f}")
    record(3, "both", length_ratio=round(float(np.mean(costs)), 4),
           reduction_pct=round(100 * (1 - float(np.mean(costs))), 2))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    p = paths[0]
    for nm, fn, c in (("raw", None, COLORS[2]),
                      ("full shortcut", shortcut, COLORS[0]),
                      ("one joint at a time", partial_shortcut, COLORS[1])):
        if fn is None:
            _, hist = shortcut(p, seg, np.random.default_rng(1), iters=0,
                               spacing=0.15, track=True)
            ax.axhline(hist[0][1], color=c, ls="--", label="raw")
            continue
        _, hist = fn(p, seg, np.random.default_rng(1), iters=200,
                     spacing=0.15, track=True)
        ax.plot([h[0] for h in hist], [h[1] for h in hist], color=c, label=nm)
    ax.set_xlabel("attempts")
    ax.set_ylabel("path cost (rad)")
    ax.set_title("Smoothing a 7-dimensional path")
    ax.legend()
    save(fig, os.path.join(OUT, "arm_shortcut.png"))


# =====================================================================  4
def exp4_homotopy(rng):
    banner("4. What shortcutting cannot fix: the route it was handed")

    env = two_route_world()
    centres = env.circles[:, :2]

    def seg(a, b):
        return env.segment_free(a, b, 0.02)

    groups = {}
    raws, smoothed, sigs = [], [], []
    for s in range(160):
        _, p, st = rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                       goal_bias=0.05, max_iters=8000)
        if not st["found"]:
            continue
        sm = shortcut(p, seg, np.random.default_rng(1), iters=300, spacing=0.25)
        sig = side_signature(sm, centres[:1])
        raws.append(path_cost(p))
        smoothed.append(path_cost(sm))
        sigs.append(sig)
        groups.setdefault(sig, []).append((path_cost(sm), np.asarray(sm)))
        # did smoothing ever move a path to the other side?
    print(f"  {len(raws)} runs")
    for sig, items in sorted(groups.items()):
        cs = [i[0] for i in items]
        side = "below-right" if sig[0] < 0 else "above-left"
        print(f"  route {side:<12s}: {len(items):3d} runs "
              f"({100*len(items)/len(raws):4.0f}%), smoothed cost "
              f"{np.mean(cs):.3f} +- {np.std(cs):.3f} m")
        record(4, f"route_{side}", runs=len(items),
               pct=round(100 * len(items) / len(raws), 1),
               mean_cost=round(float(np.mean(cs)), 4),
               std=round(float(np.std(cs)), 4))
    best = min(np.mean([i[0] for i in v]) for v in groups.values())
    worst = max(np.mean([i[0] for i in v]) for v in groups.values())
    print(f"  the good route is {100*(worst/best-1):.1f}% shorter than the bad "
          f"one, and NO amount of shortcutting turns one into the other")
    record(4, "route_gap_pct", pct=round(100 * (worst / best - 1), 2))

    # check directly: does the smoothed path ever change class?
    changed = 0
    for s in range(60):
        _, p, st = rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                       goal_bias=0.05, max_iters=8000)
        if not st["found"]:
            continue
        a = side_signature(resample(p, 0.25), centres[:1])
        b = side_signature(shortcut(p, seg, np.random.default_rng(1),
                                    iters=500, spacing=0.25), centres[:1])
        changed += (a != b)
    print(f"  paths that changed route during smoothing: {changed}/60")
    record(4, "class_changes", changed=changed, of=60)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    draw_env(axes[0], env)
    for sig, items in groups.items():
        c = COLORS[0] if sig[0] > 0 else COLORS[1]
        for _, a in items[:12]:
            axes[0].plot(a[:, 0], a[:, 1], color=c, lw=1.0, alpha=0.7)
    axes[0].plot(*START, "o", color=COLORS[2], ms=6)
    axes[0].plot(*GOAL, "*", color=COLORS[3], ms=12)
    axes[0].set_title("Smoothed paths, coloured by which way they went")
    axes[1].hist(smoothed, bins=30, color=COLORS[0])
    axes[1].set_xlabel("smoothed path length (m)")
    axes[1].set_ylabel("runs")
    axes[1].set_title("Two humps = two routes, not two amounts of luck")
    save(fig, os.path.join(OUT, "homotopy.png"))


# =====================================================================  5
def exp5_bill(rng):
    banner("5. The bill: checks spent planning against checks spent smoothing")

    envs = [world_blobs(np.random.default_rng(500 + k)) for k in range(8)]
    plan_ck, plan_ms, sm_ck, sm_ms, gain = [], [], [], [], []
    for env in envs:
        _, p, st = rrt(env, START, GOAL, np.random.default_rng(2), step=0.5,
                       goal_bias=0.05, max_iters=8000)
        plan_ck.append(st["checks"])
        plan_ms.append(st["time"] * 1e3)

        def seg(a, b, e=env):
            return e.segment_free(a, b, 0.02)

        env.n_checks = 0
        t0 = time.perf_counter()
        sm = shortcut(p, seg, np.random.default_rng(1), iters=200, spacing=0.25)
        sm_ms.append((time.perf_counter() - t0) * 1e3)
        sm_ck.append(env.n_checks)
        gain.append(100 * (1 - path_cost(sm) / path_cost(p)))
    print(f"  planning : {np.mean(plan_ck):8.0f} point checks, "
          f"{np.mean(plan_ms):6.1f} ms")
    print(f"  smoothing: {np.mean(sm_ck):8.0f} point checks, "
          f"{np.mean(sm_ms):6.1f} ms  -> {np.mean(gain):.1f}% shorter")
    print(f"  smoothing costs {np.mean(sm_ms)/np.mean(plan_ms):.2f}x the "
          f"planning time")
    record(5, "bill", plan_checks=round(float(np.mean(plan_ck))),
           plan_ms=round(float(np.mean(plan_ms)), 2),
           smooth_checks=round(float(np.mean(sm_ck))),
           smooth_ms=round(float(np.mean(sm_ms)), 2),
           ratio=round(float(np.mean(sm_ms) / np.mean(plan_ms)), 3),
           gain_pct=round(float(np.mean(gain)), 2))

    # why smoothing checks are expensive: a shortcut spans a long distance
    print("  a shortcut attempt spans a long stretch of path, so ONE attempt "
          "costs many more point tests than one RRT extension does.")


# =====================================================================  6
def exp6_corners(rng):
    banner("6. The corners it leaves behind")

    env = world_blobs(np.random.default_rng(5))

    def seg(a, b):
        return env.segment_free(a, b, 0.02)

    _, raw, _ = rrt(env, START, GOAL, np.random.default_rng(11), step=0.5,
                    goal_bias=0.05, max_iters=8000)
    sm = shortcut(raw, seg, np.random.default_rng(1), iters=300, spacing=0.25)
    print(f"  sharpest turn: raw {math.degrees(max_turn(resample(raw,0.25))):.0f} deg, "
          f"after shortcutting {math.degrees(max_turn(sm)):.0f} deg")

    print(f"  {'blend radius':>13s} {'length':>8s} {'tightest bend (m)':>18s} "
          f"{'speed through it':>17s}")
    rows = []
    V, A = 1.0, 2.0     # a robot that wants 1 m/s and can pull 2 m/s^2 sideways
    for radius in (0.0, 0.1, 0.2, 0.4, 0.8):
        b = sm if radius == 0 else blend_corners(sm, radius, seg)
        L = path_cost(b)
        # Sampling the path finely first, so the radius measures the PATH and
        # not how many points we happen to have stored.
        R = min_turn_radius(resample(b, 0.02))
        # On a bend of radius R the sideways acceleration at speed v is v^2/R,
        # so the fastest safe speed is sqrt(A * R).
        v = min(V, math.sqrt(A * R))
        rows.append((radius, L, R, v))
        print(f"  {radius:13.2f} {L:8.3f} {R:18.4f} {v:17.2f}")
        record(6, f"blend_{radius}", length=round(L, 4),
               tightest_bend_m=round(R, 5), corner_speed_mps=round(v, 3))
    print("  A sharp corner forces the robot to a crawl; rounding it costs "
          f"{100*(rows[3][1]/rows[0][1]-1):+.2f}% length and takes the corner "
          f"speed from {rows[0][3]:.2f} to {rows[3][3]:.2f} m/s.")
    record(6, "blend_tradeoff",
           extra_length_pct=round(100 * (rows[3][1] / rows[0][1] - 1), 3),
           speed_before=round(rows[0][3], 3), speed_after=round(rows[3][3], 3))

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))
    draw_env(axes[0], env)
    a = np.asarray(sm)
    axes[0].plot(a[:, 0], a[:, 1], color=COLORS[1], lw=1.6, label="shortcut")
    b = blend_corners(sm, 0.4, seg)
    axes[0].plot(b[:, 0], b[:, 1], color=COLORS[0], lw=1.6, label="blended")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Rounded corners")
    axes[1].plot([r[0] for r in rows], [r[3] for r in rows], "o-",
                 color=COLORS[0], label="safe corner speed (m/s)")
    ax2 = axes[1].twinx()
    ax2.plot([r[0] for r in rows], [100 * (r[1] / rows[0][1] - 1) for r in rows],
             "s--", color=COLORS[1])
    ax2.set_ylabel("extra length (%)", color=COLORS[1])
    ax2.grid(False)
    axes[1].set_xlabel("blend radius (m)")
    axes[1].set_ylabel("safe corner speed (m/s)")
    axes[1].set_title("Smoothness is bought with length")
    save(fig, os.path.join(OUT, "corners.png"))


# =====================================================================  7
def exp7_vs_rrt_star(rng):
    banner("7. RRT + shortcut, or RRT*, for the same time?")

    env = world_blobs(np.random.default_rng(5))
    opt = reference_optimum(env)

    def seg(a, b):
        return env.segment_free(a, b, 0.02)

    budgets_ms = [10, 25, 50, 100, 250, 500, 1000]
    print(f"  near-optimal {opt:.3f} m")
    print(f"  {'budget (ms)':>12s} {'RRT+shortcut':>14s} {'RRT*':>10s} "
          f"{'RRT*+shortcut':>15s}")
    rows = []
    for B in budgets_ms:
        a, b, c = [], [], []
        for s in range(6):
            # RRT once, then spend the rest of the budget shortcutting
            t0 = time.perf_counter()
            _, p, st = rrt(env, START, GOAL, np.random.default_rng(s), step=0.5,
                           goal_bias=0.05, max_iters=8000)
            left = B / 1e3 - (time.perf_counter() - t0)
            pts = p
            n = 0
            while time.perf_counter() - t0 < B / 1e3 and n < 4000:
                pts = shortcut(pts, seg, np.random.default_rng(1000 + n),
                               iters=10, spacing=0.25 if n == 0 else None)
                n += 10
            a.append(path_cost(pts))
            # RRT* with as many samples as fit in the budget
            iters, last = 100, None
            while True:
                t1 = time.perf_counter()
                _, p2, st2 = rrt_star(env, START, GOAL, np.random.default_rng(s),
                                      step=0.5, goal_bias=0.05, max_iters=iters)
                elapsed = time.perf_counter() - t1
                if elapsed <= B / 1e3:
                    last = (p2, st2)
                if elapsed > B / 1e3 or iters > 40000:
                    break
                iters = int(iters * 1.6)
            # If even the smallest run overran the budget, report it anyway and
            # let the table show RRT* losing on the tightest budgets.
            p2, st2 = last if last is not None else (p2, st2)
            b.append(st2["cost"] if st2["found"] else math.inf)
            if p2 is not None:
                c.append(path_cost(shortcut(p2, seg, np.random.default_rng(1),
                                            iters=100, spacing=0.25)))
        rows.append((B, np.mean(a), np.mean(b), np.mean(c) if c else math.nan))
        print(f"  {B:12d} {np.mean(a):14.3f} {np.mean(b):10.3f} "
              f"{np.mean(c) if c else math.nan:15.3f}")
        record(7, f"budget_{B}ms", rrt_shortcut=round(float(np.mean(a)), 4),
               rrt_star=round(float(np.mean(b)), 4),
               rrt_star_shortcut=round(float(np.mean(c)) if c else "", 4)
               if c else "", optimum=round(opt, 4))

    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0],
            label="RRT + shortcut")
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "s-", color=COLORS[1],
            label="RRT*")
    ax.plot([r[0] for r in rows], [r[3] for r in rows], "^-", color=COLORS[3],
            label="RRT* + shortcut")
    ax.axhline(opt, color=COLORS[2], ls="--", label="near-optimal")
    ax.set_xscale("log")
    ax.set_xlabel("time budget (ms)")
    ax.set_ylabel("path length (m)")
    ax.set_title("Same clock, three pipelines")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "vs_rrt_star.png"))


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_before_after(rng)
    exp2_knobs(rng)
    exp3_arm(rng)
    exp4_homotopy(rng)
    exp5_bill(rng)
    exp6_corners(rng)
    exp7_vs_rrt_star(rng)

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
