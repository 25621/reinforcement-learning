"""Project 33 -- RRT-Connect for a 7-joint arm in MuJoCo.

Seven experiments:

  1. the scene, a plan, and what a 7-dimensional path looks like
  2. RRT against RRT-Connect: the bidirectional speed-up, measured
  3. collision-check resolution: the setting that silently ships broken plans
  4. where the time actually goes (spoiler: it is not the search)
  5. planning to a goal SET instead of a goal POINT
  6. C-space is not workspace: a straight joint path is a curved hand path
  7. dimensionality: why nobody puts a grid on a 7-joint arm

Runs in about six minutes.  Needs mujoco, NumPy and Matplotlib.
"""

import csv
import math
import os
import sys
import time

import numpy as np

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from arm import Arm, rrt, rrt_connect, path_cost, path_free                # noqa: E402
from plot_style import COLORS, use_style, save                             # noqa: E402

import matplotlib.pyplot as plt                                            # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

# Two poses found by inverse kinematics (project 05's method, re-implemented
# on MuJoCo's Jacobian in arm.py).  Q_OPEN puts the hand low and to the left,
# past the post; Q_SHELF puts it inside the shelf.  The straight joint-space
# line between them is blocked -- verified in experiment 1.
Q_OPEN = np.array([0.7055, -0.8423, 0.7681, -1.8944, -1.2008, 0.0832, -1.0332])
Q_SHELF = np.array([0.1193, 0.2686, -2.9000, -1.8640, 1.1785, 1.0603, 0.6185])
Q_START, Q_GOAL = Q_OPEN, Q_SHELF


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def render(arm, q, w=420, h=340, cam=None):
    arm.data.qpos[:] = q
    mujoco.mj_forward(arm.model, arm.data)
    r = mujoco.Renderer(arm.model, h, w)
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = [0.25, 0.0, 0.45]
    c.distance = 1.9
    c.azimuth = 215 if cam is None else cam
    c.elevation = -22
    r.update_scene(arm.data, c)
    img = r.render()
    r.close()
    return img


def interpolate(path, n=60):
    """Resample a piecewise-linear joint path to n evenly spaced points."""
    p = np.asarray(path)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    ts = np.linspace(0, s[-1], n)
    return np.stack([np.interp(ts, s, p[:, j]) for j in range(p.shape[1])], 1)


# =====================================================================  1
def exp1_scene(rng):
    banner("1. The scene, and a plan through it")

    arm = Arm()
    print(f"  {arm.nq} joints, limits span "
          f"{np.round(arm.hi - arm.lo, 2)} rad")
    print(f"  start free: {arm.free(Q_START)}   goal free: {arm.free(Q_GOAL)}")
    direct = arm.segment_free(Q_START, Q_GOAL, res=0.02)
    print(f"  straight joint-space line from start to goal is free: {direct} "
          f"<- the whole reason a planner is needed")
    record(1, "direct_line_free", value=bool(direct))

    _, path, st = rrt_connect(arm, Q_START, Q_GOAL, np.random.default_rng(0),
                              step=0.4, res=0.05)
    print(f"  RRT-Connect: {st['iters']} iterations, {st['nodes']} nodes, "
          f"{st['checks']} collision checks, {st['time']*1e3:.0f} ms")
    print(f"  path: {len(path)} waypoints, cost {path_cost(path):.3f} rad")
    record(1, "plan", iters=st["iters"], nodes=st["nodes"],
           checks=st["checks"], ms=round(st["time"] * 1e3, 1),
           waypoints=len(path), cost=round(path_cost(path), 4))

    qs = interpolate(path, 6)
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4))
    for ax, q in zip(axes.ravel(), qs):
        ax.imshow(render(arm, q))
        ax.axis("off")
    fig.suptitle("RRT-Connect plan: six frames along the path", y=0.98)
    save(fig, os.path.join(OUT, "plan_frames.png"))

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    dense = interpolate(path, 200)
    for j in range(arm.nq):
        ax.plot(np.linspace(0, 1, 200), dense[:, j], color=COLORS[j % 7],
                label=f"j{j+1}")
    ax.set_xlabel("fraction along path")
    ax.set_ylabel("joint angle (rad)")
    ax.set_title("The same plan as seven numbers.\n"
                 "The corners are where the tree changed direction.")
    ax.legend(ncol=4, fontsize=7)
    save(fig, os.path.join(OUT, "joint_path.png"))

    # a tool-path plot, to show the hand does not travel in a straight line
    tp = np.array([arm.tool_pos(q) for q in dense])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.plot(tp[:, 1], tp[:, 2], color=COLORS[0], label="planned hand path")
    ax.plot([tp[0, 1], tp[-1, 1]], [tp[0, 2], tp[-1, 2]], "--",
            color=COLORS[1], label="straight line in space")
    ax.set_xlabel("y (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal")
    ax.set_title("Where the hand went")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "tool_path.png"))


# =====================================================================  2
def exp2_connect_vs_rrt(rng):
    banner("2. RRT against RRT-Connect")

    arm = Arm()
    n = 10
    BUDGET = 8000
    all_rows = {}
    for dirname, qa, qb in (("into the shelf", Q_OPEN, Q_SHELF),
                            ("out of the shelf", Q_SHELF, Q_OPEN)):
        rows = {"RRT": [], "RRT-Connect": []}
        for s in range(n):
            _, p1, s1 = rrt(arm, qa, qb, np.random.default_rng(s), step=0.4,
                            goal_bias=0.05, max_iters=BUDGET, res=0.05)
            _, p2, s2 = rrt_connect(arm, qa, qb, np.random.default_rng(s),
                                    step=0.4, max_iters=BUDGET, res=0.05)
            rows["RRT"].append((s1, path_cost(p1) if p1 is not None else math.inf))
            rows["RRT-Connect"].append(
                (s2, path_cost(p2) if p2 is not None else math.inf))
        all_rows[dirname] = rows

        print(f"\n  direction: {dirname}   ({BUDGET} samples, {n} seeds)")
        print(f"  {'planner':<13s} {'success':>8s} {'median ms':>10s} "
              f"{'median nodes':>13s} {'median checks':>14s} {'median cost':>12s}")
        for nm, rs in rows.items():
            ok = [x for x in rs if x[0]["found"]]
            if not ok:
                print(f"  {nm:<13s} {0:7.0f}%  (never solved within the budget)")
                record(2, f"{dirname} | {nm}", success_pct=0.0)
                continue
            ms = np.median([x[0]["time"] * 1e3 for x in ok])
            nd = np.median([x[0]["nodes"] for x in ok])
            ck = np.median([x[0]["checks"] for x in ok])
            ct = np.median([x[1] for x in ok])
            print(f"  {nm:<13s} {100*len(ok)/n:7.0f}% {ms:10.0f} {nd:13.0f} "
                  f"{ck:14.0f} {ct:12.3f}")
            record(2, f"{dirname} | {nm}", success_pct=round(100 * len(ok) / n, 1),
                   median_ms=round(float(ms), 1), median_nodes=float(nd),
                   median_checks=float(ck), median_cost=round(float(ct), 4))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for ax, (dirname, rows) in zip(axes, all_rows.items()):
        for k, (nm, rs) in enumerate(rows.items()):
            t = [x[0]["time"] * 1e3 for x in rs if x[0]["found"]]
            if t:
                ax.hist(t, bins=10, alpha=0.75, color=COLORS[k], label=nm)
            else:
                ax.plot([], [], color=COLORS[k], label=f"{nm}: 0 of {n} solved")
        ax.set_xlabel("planning time (ms)")
        ax.set_ylabel("runs")
        ax.set_title(f"reaching {dirname}")
        ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "connect_vs_rrt.png"))


# =====================================================================  3
def exp3_resolution(rng):
    banner("3. Collision-check resolution: the setting that ships broken plans")

    arm = Arm()
    print(f"  {'res (rad)':>10s} {'success':>8s} {'median ms':>10s} "
          f"{'checks':>9s} {'passes fine re-check':>21s}")
    rows = []
    for res in (0.4, 0.2, 0.1, 0.05, 0.02, 0.01):
        ok, valid, times, checks = 0, 0, [], []
        for s in range(12):
            _, p, st = rrt_connect(arm, Q_START, Q_GOAL,
                                   np.random.default_rng(100 + s), step=0.4,
                                   max_iters=30000, res=res)
            if not st["found"]:
                continue
            ok += 1
            times.append(st["time"] * 1e3)
            checks.append(st["checks"])
            arm.n_checks = 0
            if path_free(arm, p, res=0.005):
                valid += 1
        rows.append((res, 100 * ok / 12, np.median(times), np.median(checks),
                     100 * valid / max(1, ok)))
        print(f"  {res:10.3f} {100*ok/12:7.0f}% {np.median(times):10.0f} "
              f"{np.median(checks):9.0f} {100*valid/max(1,ok):20.0f}%")
        record(3, f"res_{res}", success_pct=round(100 * ok / 12, 1),
               median_ms=round(float(np.median(times)), 1),
               median_checks=float(np.median(checks)),
               valid_pct=round(100 * valid / max(1, ok), 1))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot([r[0] for r in rows], [r[4] for r in rows], "o-", color=COLORS[0],
            label="plans that survive a fine re-check")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "s--", color=COLORS[2],
            label="plans found at all")
    ax.set_xscale("log")
    ax.set_xlabel("collision-check spacing (rad)")
    ax.set_ylabel("percent")
    ax.set_ylim(-5, 105)
    ax2 = ax.twinx()
    ax2.plot([r[0] for r in rows], [r[2] for r in rows], "^:", color=COLORS[1])
    ax2.set_ylabel("median planning time (ms)", color=COLORS[1])
    ax2.grid(False)
    ax.set_title("A coarse check is fast and returns plans that are not safe")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, os.path.join(OUT, "resolution.png"))


# =====================================================================  4
def exp4_time_budget(rng):
    banner("4. Where the time actually goes")

    arm = Arm()
    r = np.random.default_rng(0)
    qs = np.array([arm.sample(r) for _ in range(4000)])
    t0 = time.perf_counter()
    for q in qs:
        arm.collides(q)
    t_check = (time.perf_counter() - t0) / len(qs)

    t0 = time.perf_counter()
    for q in qs[:2000]:
        arm.tool_pos(q)
    t_fk = (time.perf_counter() - t0) / 2000

    _, p, st = rrt_connect(arm, Q_START, Q_GOAL, np.random.default_rng(0),
                           step=0.4, max_iters=30000, res=0.05)
    est = t_check * st["checks"]
    print(f"  one collision query (mj_kinematics + mj_collision): "
          f"{t_check*1e6:.1f} us")
    print(f"  forward kinematics alone:                          "
          f"{t_fk*1e6:.1f} us")
    print(f"  a plan: {st['checks']} collision queries in {st['time']*1e3:.0f} "
          f"ms total -> {100*est/st['time']:.0f}% of the run is collision "
          f"checking")
    print(f"  the tree itself has only {st['nodes']} nodes: "
          f"{st['checks']/st['nodes']:.0f} collision queries per node kept")
    record(4, "budget", check_us=round(t_check * 1e6, 2),
           fk_us=round(t_fk * 1e6, 2), checks=st["checks"],
           total_ms=round(st["time"] * 1e3, 1),
           collision_pct=round(100 * est / st["time"], 1), nodes=st["nodes"],
           checks_per_node=round(st["checks"] / st["nodes"], 1))

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    other = max(0.0, st["time"] - est)
    ax.barh(["collision checking", "everything else"], [est * 1e3, other * 1e3],
            color=[COLORS[1], COLORS[0]])
    ax.set_xlabel("milliseconds in one plan")
    ax.set_title("Sampling planners are collision checkers\nwith a search "
                 "loop attached")
    save(fig, os.path.join(OUT, "time_budget.png"))


# =====================================================================  5
def exp5_goal_set(rng):
    banner("5. Planning to a goal SET instead of a goal POINT")

    arm = Arm()
    target = arm.tool_pos(Q_GOAL)
    print(f"  hand target: {np.round(target, 3)} m")

    # Collect several distinct IK solutions -- the arm has 7 joints and the
    # target constrains only 3 numbers, so a whole 4-dimensional family of
    # answers exists.  Different seeds fall into different parts of it.
    r = np.random.default_rng(0)
    goals = []
    tries = 0
    while len(goals) < 8 and tries < 400:
        tries += 1
        q0 = arm.sample(r)
        q, err = arm.ik(target, q0)
        if err < 5e-3 and arm.free(q):
            if all(np.linalg.norm(q - g) > 0.5 for g in goals):
                goals.append(q)
    print(f"  {len(goals)} distinct collision-free IK solutions found in "
          f"{tries} attempts")
    record(5, "ik_solutions", found=len(goals), attempts=tries)

    print(f"  {'goals offered':>14s} {'success':>8s} {'median ms':>10s} "
          f"{'median cost':>12s}")
    rows = []
    for k in (1, 2, 4, 8):
        ms, cost, ok = [], [], 0
        for s in range(12):
            rr = np.random.default_rng(200 + s)
            sub = goals[:k]
            # planning to a set = plan to each member, keep the best.  A real
            # implementation grows ONE goal tree seeded with all of them, which
            # is cheaper still; this version keeps the comparison honest by
            # charging the full cost of every attempt.
            best, tot = math.inf, 0.0
            for g in sub:
                _, p, st = rrt_connect(arm, Q_START, g, rr, step=0.4,
                                       max_iters=8000, res=0.05)
                tot += st["time"] * 1e3
                if st["found"]:
                    best = min(best, path_cost(p))
            if math.isfinite(best):
                ok += 1
                ms.append(tot)
                cost.append(best)
        rows.append((k, 100 * ok / 12, np.median(ms), np.median(cost)))
        print(f"  {k:>14d} {100*ok/12:7.0f}% {np.median(ms):10.0f} "
              f"{np.median(cost):12.3f}")
        record(5, f"goals_{k}", success_pct=round(100 * ok / 12, 1),
               median_ms=round(float(np.median(ms)), 1),
               median_cost=round(float(np.median(cost)), 4))

    fig, axes = plt.subplots(1, min(4, len(goals)), figsize=(11, 3.0))
    for ax, q in zip(np.atleast_1d(axes), goals[:4]):
        ax.imshow(render(arm, q, w=300, h=260))
        ax.axis("off")
    fig.suptitle("Four different arm poses, one identical hand position", y=1.0)
    save(fig, os.path.join(OUT, "goal_set.png"))


# =====================================================================  6
def exp6_cspace_vs_workspace(rng):
    banner("6. C-space is not workspace")

    arm = Arm()
    r = np.random.default_rng(3)
    ratios, curv = [], []
    for _ in range(120):
        qa = arm.sample_free(r)
        qb = arm.sample_free(r)
        ts = np.linspace(0, 1, 40)
        qs = qa[None, :] + ts[:, None] * (qb - qa)[None, :]
        p = np.array([arm.tool_pos(q) for q in qs])
        arclen = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        chord = np.linalg.norm(p[-1] - p[0])
        if chord > 0.05:
            ratios.append(arclen / chord)
        # how far the hand strays from the straight line joining the ends
        d = p - p[0]
        u = (p[-1] - p[0]) / max(1e-9, chord)
        perp = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
        curv.append(perp.max())
    ratios = np.array(ratios)
    curv = np.array(curv)
    print(f"  a STRAIGHT line in joint space makes the hand travel "
          f"{np.median(ratios):.2f}x the straight-line distance (median), "
          f"up to {ratios.max():.2f}x")
    print(f"  the hand strays up to {curv.max()*100:.0f} cm from the straight "
          f"line (median {np.median(curv)*100:.0f} cm)")
    record(6, "joint_line_bows", median_ratio=round(float(np.median(ratios)), 3),
           max_ratio=round(float(ratios.max()), 3),
           median_deviation_cm=round(float(np.median(curv)) * 100, 2),
           max_deviation_cm=round(float(curv.max()) * 100, 2))

    # And the reverse: how often is a straight WORKSPACE line even reachable?
    ok_joint, ok_task = 0, 0
    trials = 60
    for _ in range(trials):
        qa = arm.sample_free(r)
        qb = arm.sample_free(r)
        if arm.segment_free(qa, qb, 0.05):
            ok_joint += 1
        pa, pb = arm.tool_pos(qa), arm.tool_pos(qb)
        q = qa.copy()
        good = True
        for t in np.linspace(0, 1, 25)[1:]:
            q, err = arm.ik(pa + t * (pb - pa), q, iters=60)
            if err > 1e-2 or not arm.free(q):
                good = False
                break
        ok_task += good
    print(f"  straight JOINT-space move is collision-free {100*ok_joint/trials:.0f}% "
          f"of the time")
    print(f"  straight TASK-space move (IK at every step) succeeds "
          f"{100*ok_task/trials:.0f}% of the time")
    record(6, "straight_moves", joint_pct=round(100 * ok_joint / trials, 1),
           task_pct=round(100 * ok_task / trials, 1), trials=trials)

    fig, ax = plt.subplots(figsize=(5.8, 3.4))
    ax.hist(ratios, bins=30, color=COLORS[0])
    ax.axvline(1.0, color=COLORS[1], ls="--", label="hand goes straight")
    ax.set_xlabel("hand path length / straight-line distance")
    ax.set_ylabel("random joint-space segments")
    ax.set_title("Straight in joint space is bent in real space")
    ax.legend()
    save(fig, os.path.join(OUT, "cspace_vs_workspace.png"))


# =====================================================================  7
def exp7_dimensionality(rng):
    banner("7. Dimensionality: why nobody grids a 7-joint arm")

    arm = Arm()
    lo0, hi0 = arm.lo.copy(), arm.hi.copy()
    print(f"  {'joints free':>12s} {'median ms':>10s} {'median nodes':>13s} "
          f"{'success':>8s}   grid cells at 10 steps per joint")
    rows = []
    for k in (2, 3, 4, 5, 6, 7):
        # Lock the joints past k at a fixed pose by shrinking the sampling box
        # to a point there.  Then plan between random free configurations of
        # the remaining k joints, so every k gets a comparable workload.
        arm.lo, arm.hi = lo0.copy(), hi0.copy()
        arm.lo[k:] = Q_OPEN[k:]
        arm.hi[k:] = Q_OPEN[k:]
        r = np.random.default_rng(400 + k)
        ms, nd, ok, trials = [], [], 0, 10
        for s in range(trials):
            qa = arm.sample_free(r)
            qb = arm.sample_free(r)
            _, p, st = rrt_connect(arm, qa, qb, np.random.default_rng(s),
                                   step=0.4, max_iters=6000, res=0.05)
            if st["found"]:
                ok += 1
                ms.append(st["time"] * 1e3)
                nd.append(st["nodes"])
        med = np.median(ms) if ms else math.nan
        mnd = np.median(nd) if nd else math.nan
        rows.append((k, med, mnd, 100 * ok / trials))
        print(f"  {k:>12d} {med:10.1f} {mnd:13.0f} {100*ok/trials:7.0f}%   "
              f"10^{k} = {10**k:,}")
        record(7, f"dof_{k}", median_ms=round(float(med), 2),
               median_nodes=float(mnd) if nd else "",
               success_pct=round(100 * ok / trials, 1), grid_cells=10 ** k)
    arm.lo, arm.hi = lo0, hi0

    # Each row plans between its OWN random pairs, so one row being a little
    # out of order is sampling noise, not a trend.  Compare 3 joints to 7.
    growth = rows[-1][1] / rows[1][1]
    print(f"  going from 3 to 7 joints costs {growth:.1f}x more planning time; "
          f"a grid would cost {10**7/10**3:,}x more cells")
    record(7, "growth_3_to_7", planner_time_ratio=round(float(growth), 2),
           grid_cell_ratio=10 ** 4)

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.semilogy([r[0] for r in rows], [r[1] for r in rows], "o-",
                color=COLORS[0], label="RRT-Connect time (ms)")
    ax.semilogy([r[0] for r in rows], [10.0 ** r[0] for r in rows], "s--",
                color=COLORS[1], label="cells in a 10-per-joint grid")
    ax.set_xlabel("joints allowed to move")
    ax.set_ylabel("log scale")
    ax.set_title("Sampling grows gently with dimension; a grid explodes")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "dimensionality.png"))

    print("  a 10-per-joint grid on 7 joints is 10 million cells -- and 10 "
          "steps per joint is a 36-degree resolution, far too coarse to plan "
          "with.  At 1 degree it would be 360^7 = 7.8e17 cells.")
    record(7, "grid_blowup", cells_10_per_joint=10 ** 7,
           cells_1_degree=360 ** 7)


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_scene(rng)
    exp2_connect_vs_rrt(rng)
    exp3_resolution(rng)
    exp4_time_budget(rng)
    exp5_goal_set(rng)
    exp6_cspace_vs_workspace(rng)
    exp7_dimensionality(rng)

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
