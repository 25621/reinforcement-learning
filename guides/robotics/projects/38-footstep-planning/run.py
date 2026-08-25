"""Project 38 -- footstep planning: walking as a graph search.

Seven experiments:

  1. the terrain, the plan, and what a footstep lattice looks like
  2. the heuristic, on a graph that is not a grid
  3. how finely to chop the step space
  4. the greedy baseline, and exactly where it walks into a dead end
  5. terrain difficulty: the cliff between "easy" and "impossible"
  6. the cost function IS the gait
  7. kinematically fine, dynamically committed: a capture-point audit

Runs in about three minutes.  NumPy and Matplotlib only.
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
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from footstep import (Terrain, Robot, stepping_stones, flat_ground, plan,     # noqa: E402
                      greedy, path_stats, capture_points, dynamic_check)
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402
from matplotlib.patches import Rectangle                                      # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

START = (-0.2, -0.20)
START_FOOT = "R"
GOAL_X = 3.9
STONE = 0.20


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def draw_terrain(ax, terrain):
    for x0, y0, x1, y1 in terrain.stones:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#C9CFD6",
                               ec="#8A939C", lw=0.6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(-0.9, 0.9)
    ax.grid(False)


def draw_path(ax, path, lw=1.2):
    if not path:
        return
    for i, (x, y, foot) in enumerate(path):
        c = COLORS[0] if foot == "L" else COLORS[1]
        ax.add_patch(Rectangle((x - 0.055, y - 0.032), 0.11, 0.064, color=c,
                               alpha=0.9))
    p = np.array([(x, y) for x, y, _ in path])
    ax.plot(p[:, 0], p[:, 1], color="#4A4A4A", lw=lw, alpha=0.6, zorder=0)


def make_terrain(seed, size=STONE):
    return stepping_stones(np.random.default_rng(seed), size=size)


# =====================================================================  1
def exp1_plan(rng):
    banner("1. The terrain, and a plan across it")

    robot = Robot()
    acts = robot.actions(13, 7)
    print(f"  robot: forward step {robot.dx_min:+.2f} to {robot.dx_max:+.2f} m, "
          f"sideways {robot.dy_min:.2f} to {robot.dy_max:.2f} m")
    print(f"  action lattice: 13 forward x 7 sideways = {len(acts)} candidate "
          f"steps per state")

    terr = make_terrain(5)
    res = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts)
    st = path_stats(res["path"], robot)
    print(f"  {len(terr.stones)} stones, total support area "
          f"{terr.area():.2f} m^2 out of a {5.4*1.3:.1f} m^2 corridor "
          f"({100*terr.area()/(5.4*1.3):.0f}%)")
    print(f"  planned in {res['time']*1e3:.0f} ms, {res['expanded']} states "
          f"expanded, cost {res['cost']:.3f}")
    print(f"  {st['steps']} steps, mean length {st['mean_len']:.3f} m "
          f"(forward {st['mean_forward']:.3f} m), longest {st['max_len']:.3f} m")
    record(1, "plan", stones=len(terr.stones), area=round(terr.area(), 3),
           ms=round(res["time"] * 1e3, 1), expanded=res["expanded"],
           cost=round(res["cost"], 4), **{k: round(v, 4)
                                          for k, v in st.items()})

    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    draw_terrain(ax, terr)
    draw_path(ax, res["path"])
    ax.set_title(f"{st['steps']} steps across {len(terr.stones)} stones "
                 f"(blue = left foot, orange = right)")
    ax.set_xticks([])
    ax.set_yticks([])
    save(fig, os.path.join(OUT, "plan.png"))

    # what the reachable set from one foot actually looks like
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for dx, dy in acts:
        ax.plot(dx, dy, ".", color=COLORS[0], ms=5)
        ax.plot(dx, -dy, ".", color=COLORS[1], ms=5)
    ax.plot(0, 0, "s", color="#4A4A4A", ms=9)
    ax.set_xlabel("forward offset from the stance foot (m)")
    ax.set_ylabel("sideways offset (m)")
    ax.set_aspect("equal")
    ax.set_title("The action set: where the other foot may go\n"
                 "(the empty band down the middle is what stops the legs "
                 "crossing)")
    save(fig, os.path.join(OUT, "actions.png"))


# =====================================================================  2
def exp2_heuristic(rng):
    banner("2. The heuristic, on a graph that is not a grid")

    robot = Robot()
    acts = robot.actions(13, 7)
    print(f"  {'terrain':>7s} | {'Dijkstra':>9s} | {'A* linear h':>12s} "
          f"{'cost = opt?':>12s} | {'A* ceil h':>10s} {'cost = opt?':>12s}")
    rows = []
    for seed in range(10):
        terr = make_terrain(seed)
        a = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts,
                 heuristic="none")
        b = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts,
                 heuristic="linear")
        c = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts,
                 heuristic="ceil")
        if not a["found"]:
            continue
        okb = abs(a["cost"] - b["cost"]) < 1e-6
        okc = abs(a["cost"] - c["cost"]) < 1e-6
        rows.append((seed, a["expanded"], b["expanded"], okb,
                     c["expanded"], okc, a["cost"], c["cost"]))
        print(f"  {seed:>7d} | {a['expanded']:>9d} | {b['expanded']:>12d} "
              f"{str(okb):>12s} | {c['expanded']:>10d} {str(okc):>12s}")
        record(2, f"seed_{seed}", dijkstra=a["expanded"], astar=b["expanded"],
               astar_optimal=okb, ceil_expanded=c["expanded"],
               ceil_optimal=okc, optimal_cost=round(a["cost"], 5),
               ceil_cost=round(c["cost"], 5))
    n = len(rows)
    sp_b = float(np.mean([r[1] / r[2] for r in rows]))
    sp_c = float(np.mean([r[1] / r[4] for r in rows]))
    exc = [100 * (r[7] / r[6] - 1) for r in rows if not r[5]]
    print(f"  consistent (linear) heuristic: {sp_b:.2f}x fewer expansions, "
          f"optimal on {sum(r[3] for r in rows)}/{n} terrains")
    print(f"  inconsistent (ceil) heuristic: {sp_c:.2f}x fewer expansions, "
          f"optimal on {sum(r[5] for r in rows)}/{n} terrains"
          + (f", and when it is wrong it is {np.mean(exc):.1f}% over "
             f"(worst {max(exc):.1f}%)" if exc else ""))
    record(2, "summary", linear_speedup=round(sp_b, 3),
           linear_optimal=int(sum(r[3] for r in rows)),
           ceil_speedup=round(sp_c, 3),
           ceil_optimal=int(sum(r[5] for r in rows)), of=n,
           ceil_mean_excess_pct=round(float(np.mean(exc)), 3) if exc else "",
           ceil_worst_excess_pct=round(float(max(exc)), 3) if exc else "")
    print("  Two separate lessons here.")
    print("  First, the speed-up from a good heuristic is smaller than project")
    print("  31's 15x on an open grid, for the same reason its maze was:")
    print("  most branches here die immediately because there is no stone")
    print("  under them, so Dijkstra was never going to explore them either.")
    print("  Second, ADMISSIBLE is not enough once you refuse to reopen closed")
    print("  states.  The ceiling heuristic never overestimates, and it still")
    print("  returns paths that are not optimal, because it can drop by a whole")
    print("  step's worth across a step that cost almost nothing.")

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    idx = np.arange(len(rows))
    ax.bar(idx - 0.27, [r[1] for r in rows], 0.27, color=COLORS[6],
           label="Dijkstra")
    ax.bar(idx, [r[2] for r in rows], 0.27, color=COLORS[0],
           label="A*, consistent h")
    ax.bar(idx + 0.27, [r[4] for r in rows], 0.27, color=COLORS[1],
           label="A*, inconsistent h")
    ax.set_xticks(idx)
    ax.set_xticklabels([str(r[0]) for r in rows])
    ax.set_xlabel("terrain seed")
    ax.set_ylabel("states expanded")
    ax.legend(fontsize=8)
    ax.set_title("Cheaper search, and what it can cost you")
    save(fig, os.path.join(OUT, "heuristic.png"))


# =====================================================================  3
def exp3_granularity(rng):
    banner("3. How finely to chop the step space")

    robot = Robot()
    print(f"  {'lattice':>10s} {'actions':>8s} {'solved':>8s} {'mean cost':>10s} "
          f"{'mean expanded':>14s} {'mean ms':>9s}")
    rows = []
    for nx, ny in ((3, 2), (5, 3), (7, 4), (13, 7), (21, 9), (31, 13)):
        acts = robot.actions(nx, ny)
        ok, costs, exp, ms = 0, [], [], []
        for seed in range(12):
            terr = make_terrain(seed)
            r = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts)
            exp.append(r["expanded"])
            ms.append(r["time"] * 1e3)
            if r["found"]:
                ok += 1
                costs.append(r["cost"])
        rows.append((f"{nx}x{ny}", len(acts), 100 * ok / 12,
                     np.mean(costs) if costs else math.nan, np.mean(exp),
                     np.mean(ms)))
        print(f"  {rows[-1][0]:>10s} {len(acts):>8d} {100*ok/12:7.0f}% "
              f"{rows[-1][3]:10.3f} {np.mean(exp):14.0f} {np.mean(ms):9.0f}")
        record(3, f"lattice_{nx}x{ny}", actions=len(acts),
               success_pct=round(100 * ok / 12, 1),
               mean_cost=round(float(rows[-1][3]), 4) if costs else "",
               mean_expanded=round(float(np.mean(exp)), 1),
               mean_ms=round(float(np.mean(ms)), 1))
    print("  Two different failures at the two ends: too coarse and there is")
    print("  no legal step onto the next stone at all; too fine and every")
    print("  state has hundreds of children, so the search gets slower without")
    print("  finding anything better.")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([r[1] for r in rows], [r[2] for r in rows], "o-", color=COLORS[0],
            label="solved (%)")
    ax.set_xscale("log")
    ax.set_xlabel("candidate steps per state")
    ax.set_ylabel("terrains solved (%)")
    ax2 = ax.twinx()
    ax2.plot([r[1] for r in rows], [r[5] for r in rows], "s--", color=COLORS[1])
    ax2.set_ylabel("mean planning time (ms)", color=COLORS[1])
    ax2.grid(False)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("More choices: more terrains solved, more time per terrain")
    save(fig, os.path.join(OUT, "granularity.png"))


# =====================================================================  4
def exp4_greedy(rng):
    banner("4. The greedy baseline")

    robot = Robot()
    acts = robot.actions(13, 7)
    a_ok, g_ok, g_reach = 0, 0, []
    n = 24
    for seed in range(n):
        terr = make_terrain(seed)
        r = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts)
        g = greedy(terr, robot, START, START_FOOT, GOAL_X, actions=acts)
        a_ok += r["found"]
        g_ok += g["found"]
        if r["found"] and not g["found"]:
            g_reach.append(g["path"][-1][0] / GOAL_X)
    print(f"  A*     solved {a_ok}/{n} = {100*a_ok/n:.0f}%")
    print(f"  greedy solved {g_ok}/{n} = {100*g_ok/n:.0f}%")
    if g_reach:
        print(f"  on the {len(g_reach)} terrains A* solved and greedy did not, "
              f"greedy got {100*np.mean(g_reach):.0f}% of the way across "
              f"before running out of stones")
    record(4, "greedy_vs_astar", astar=a_ok, greedy=g_ok, of=n,
           greedy_mean_progress_pct=round(100 * float(np.mean(g_reach)), 1)
           if g_reach else "")
    print("  Greedy fails for a reason worth naming: reaching forward as far")
    print("  as possible now often lands on the FAR EDGE of a stone, and from")
    print("  the far edge the next stone is out of range.  Search wins because")
    print("  it is willing to take a short step in order to set up a long one.")

    # show one
    for seed in range(n):
        terr = make_terrain(seed)
        r = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts)
        g = greedy(terr, robot, START, START_FOOT, GOAL_X, actions=acts)
        if r["found"] and not g["found"]:
            fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.0))
            for ax, path, title in ((axes[0], r["path"], "A*: crossed"),
                                    (axes[1], g["path"],
                                     "greedy: stuck, no legal next step")):
                draw_terrain(ax, terr)
                draw_path(ax, path)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
            save(fig, os.path.join(OUT, "greedy.png"))
            break


# =====================================================================  5
def exp5_difficulty(rng):
    banner("5. Terrain difficulty")

    robot = Robot()
    acts = robot.actions(13, 7)
    print(f"  {'stone size (m)':>15s} {'support area':>13s} {'solved':>8s} "
          f"{'mean cost':>10s} {'mean expanded':>14s}")
    rows = []
    for size in (0.34, 0.28, 0.24, 0.20, 0.17, 0.14, 0.11, 0.08):
        ok, costs, exp, areas = 0, [], [], []
        for seed in range(20):
            terr = make_terrain(seed, size=size)
            r = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts)
            exp.append(r["expanded"])
            areas.append(terr.area())
            if r["found"]:
                ok += 1
                costs.append(r["cost"])
        rows.append((size, 100 * ok / 20, np.mean(costs) if costs else math.nan,
                     np.mean(exp), np.mean(areas)))
        print(f"  {size:15.2f} {np.mean(areas):12.2f}m2 {100*ok/20:7.0f}% "
              f"{rows[-1][2]:10.3f} {np.mean(exp):14.0f}")
        record(5, f"size_{size}", success_pct=round(100 * ok / 20, 1),
               area=round(float(np.mean(areas)), 3),
               mean_cost=round(float(rows[-1][2]), 4) if costs else "",
               mean_expanded=round(float(np.mean(exp)), 1))

    flat = flat_ground()
    r = plan(flat, robot, START, START_FOOT, GOAL_X, actions=acts)
    print(f"  control -- flat ground: solved={r['found']}, cost "
          f"{r['cost']:.3f}, {r['expanded']} expanded, "
          f"{path_stats(r['path'], robot)['steps']} steps")
    record(5, "flat_ground", solved=r["found"], cost=round(r["cost"], 4),
           expanded=r["expanded"],
           steps=path_stats(r["path"], robot)["steps"])
    print("  On flat ground the answer is the trivial one -- march forward at")
    print("  the longest stride the cost function likes -- and it is the cost")
    print("  function, not the terrain, that decides the gait.  That is")
    print("  experiment 6.")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0])
    ax.set_xlabel("stone size (m)")
    ax.set_ylabel("terrains solved (%)")
    ax.set_title("A cliff, not a slope")
    save(fig, os.path.join(OUT, "difficulty.png"))


# =====================================================================  6
def exp6_cost_shapes_gait(rng):
    banner("6. The cost function IS the gait")

    acts = Robot().actions(13, 7)
    flat = flat_ground()
    print("  Flat ground, so nothing but the cost function can decide the gait.")
    print(f"  {'w_step':>7s} {'w_len':>6s} {'w_quad':>7s} {'w_lat':>6s} | "
          f"{'steps':>6s} {'mean stride (m)':>16s} {'mean sideways (m)':>18s}")
    rows = []
    for w_step, w_len, w_quad, w_lat in ((0.35, 1.0, 0.0, 0.6),
                                         (0.35, 1.0, 1.0, 0.6),
                                         (0.35, 1.0, 3.0, 0.6),
                                         (0.35, 1.0, 8.0, 0.6),
                                         (2.00, 1.0, 3.0, 0.6),
                                         (0.35, 1.0, 3.0, 0.0),
                                         (0.35, 1.0, 3.0, 4.0)):
        robot = Robot(w_step=w_step, w_len=w_len, w_quad=w_quad, w_lat=w_lat)
        r = plan(flat, robot, START, START_FOOT, GOAL_X, actions=acts)
        st = path_stats(r["path"], robot)
        rows.append((w_step, w_quad, w_lat, st))
        print(f"  {w_step:7.2f} {w_len:6.2f} {w_quad:7.2f} {w_lat:6.2f} | "
              f"{st['steps']:6d} {st['mean_forward']:16.3f} "
              f"{st['mean_lat']:18.3f}")
        record(6, f"step{w_step}_quad{w_quad}_lat{w_lat}", steps=st["steps"],
               mean_forward=round(st["mean_forward"], 4),
               mean_lat=round(st["mean_lat"], 4),
               mean_len=round(st["mean_len"], 4))
    print("  Rows 1-4 hold everything fixed but the quadratic length penalty:")
    print("  the stride shrinks from 0.46 m to 0.26 m and the step count rises")
    print("  from 9 to 16.  Row 5 raises the flat per-step charge instead,")
    print("  which pushes straight back the other way.")
    print("  Rows 6-7 vary only the stance-width penalty -- and note that they")
    print("  changed the STRIDE too (0.37 m against 0.46 m).  The knobs are NOT")
    print("  independent: a wider stance makes each step longer through the")
    print("  quadratic term, which then changes what the forward part wants.")
    print("  Tuning a gait cost is not four separate dials; it is one surface.")

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 5.4))
    for ax, (w_quad, lbl) in zip(axes, ((0.0, "w_quad = 0: longest legal stride"),
                                        (3.0, "w_quad = 3: shorter, more frequent"),
                                        (8.0, "w_quad = 8: short, frequent steps"))):
        robot = Robot(w_quad=w_quad)
        r = plan(flat, robot, START, START_FOOT, 2.4, actions=acts)
        draw_terrain(ax, flat)
        draw_path(ax, r["path"])
        ax.set_xlim(-0.7, 2.8)
        st = path_stats(r["path"], robot)
        ax.set_title(f"{lbl}: {st['steps']} steps, "
                     f"{st['mean_forward']:.2f} m stride")
        ax.set_xticks([])
        ax.set_yticks([])
    save(fig, os.path.join(OUT, "gait.png"))


# =====================================================================  7
def exp7_dynamics(rng):
    banner("7. Kinematically fine, dynamically committed")

    robot = Robot()
    acts = robot.actions(13, 7)
    terr = make_terrain(5)
    r = plan(terr, robot, START, START_FOOT, GOAL_X, actions=acts)

    print("  (a) how the step DURATION changes things -- it barely does")
    print(f"  {'step time (s)':>14s} {'peak body speed':>16s} "
          f"{'capture point off-stone':>24s}")
    rows = []
    for T in (0.35, 0.5, 0.65, 0.8, 1.0, 1.4):
        d = dynamic_check(terr, r["path"], step_time=T)
        rows.append((T, d["max_speed"], d["uncapturable"], d["steps"]))
        print(f"  {T:14.2f} {d['max_speed']:15.2f}m/s "
              f"{d['uncapturable']:>13d}/{d['steps']:<10d}")
        record(7, f"step_time_{T}", max_speed=round(d["max_speed"], 3),
               uncapturable=d["uncapturable"], steps=d["steps"])
    print("  That flatness is a real property of the model, not a bug.  Under")
    print("  the linear inverted pendulum the body position diverges like")
    print("  exp(omega t), so however long you take, you arrive at the next")
    print("  foot travelling at very nearly omega x (step length).  The capture")
    print("  point therefore sits about ONE STRIDE beyond the foot you are")
    print("  about to plant, no matter how slowly you walk.  Walking is")
    print("  controlled falling; you cannot stop inside the current step.")

    print("\n  (b) what DOES change things: stride length and body height")
    print(f"  {'max stride (m)':>15s} {'steps':>6s} {'peak speed':>11s} "
          f"{'capture point off-stone':>24s}")
    for dx_max in (0.50, 0.40, 0.30, 0.24, 0.20):
        rb = Robot(dx_max=dx_max)
        rr = plan(terr, rb, START, START_FOOT, GOAL_X, actions=rb.actions(13, 7))
        if not rr["found"]:
            print(f"  {dx_max:15.2f} {'-':>6s} {'-':>11s} "
                  f"{'no plan exists with this stride':>24s}")
            record(7, f"stride_{dx_max}", found=False)
            continue
        d = dynamic_check(terr, rr["path"], step_time=0.6)
        print(f"  {dx_max:15.2f} {d['steps']:6d} {d['max_speed']:10.2f}m/s "
              f"{d['uncapturable']:>13d}/{d['steps']:<10d}")
        record(7, f"stride_{dx_max}", found=True, steps=d["steps"],
               max_speed=round(d["max_speed"], 3),
               uncapturable=d["uncapturable"])

    print(f"  {'body height (m)':>16s} {'omega (1/s)':>12s} {'peak speed':>11s} "
          f"{'capture point off-stone':>24s}")
    for h in (0.5, 0.8, 1.1, 1.5):
        d = dynamic_check(terr, r["path"], step_time=0.6, com_height=h)
        om = math.sqrt(9.81 / h)
        print(f"  {h:16.2f} {om:12.2f} {d['max_speed']:10.2f}m/s "
              f"{d['uncapturable']:>13d}/{d['steps']:<10d}")
        record(7, f"height_{h}", omega=round(om, 3),
               max_speed=round(d["max_speed"], 3),
               uncapturable=d["uncapturable"], steps=d["steps"])

    print("\n  (c) the control: the SAME check on flat ground")
    flat = flat_ground()
    rf = plan(flat, robot, START, START_FOOT, GOAL_X, actions=acts)
    df = dynamic_check(flat, rf["path"], step_time=0.6)
    print(f"  flat ground: peak speed {df['max_speed']:.2f} m/s, capture point "
          f"off-stone {df['uncapturable']}/{df['steps']}")
    record(7, "flat_control", max_speed=round(df["max_speed"], 3),
           uncapturable=df["uncapturable"], steps=df["steps"])
    print(f"  {df['uncapturable']} out of {df['steps']}.  The body speed is")
    print("  essentially identical -- it is the same walking.  What changed is")
    print("  that on flat ground there is")
    print("  always something under the capture point.  The problem is not the")
    print("  gait; it is that the plan crosses GAPS at a speed which leaves the")
    print("  robot no legal place to abort.  That is invisible to a planner")
    print("  that only ever asked 'is there a stone here?'")
    print("  Production systems bolt a dynamic filter onto the step cost, or")
    print("  plan over (footstep, step duration) pairs so speed is a decision.")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.6))
    axes[0].plot([r0[0] for r0 in rows], [r0[1] for r0 in rows], "o-",
                 color=COLORS[1], label="peak body speed (m/s)")
    axes[0].plot([r0[0] for r0 in rows],
                 [r0[2] / r0[3] for r0 in rows], "s--", color=COLORS[0],
                 label="fraction of steps with no abort")
    axes[0].set_xlabel("step time (s)")
    axes[0].set_ylim(0, 2.6)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Walking slower does not buy safety here")

    draw_terrain(axes[1], terr)
    draw_path(axes[1], r["path"])
    cps = capture_points(r["path"], step_time=0.6)
    pts = np.array([c["capture"] for c in cps])
    axes[1].plot(pts[:, 0], pts[:, 1], "x", color=COLORS[3], ms=6,
                 label="capture point (0.6 s/step)")
    axes[1].legend(fontsize=7, loc="upper right")
    axes[1].set_title("Where the robot would have to step to stop")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    save(fig, os.path.join(OUT, "dynamics.png"))


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_plan(rng)
    exp2_heuristic(rng)
    exp3_granularity(rng)
    exp4_greedy(rng)
    exp5_difficulty(rng)
    exp6_cost_shapes_gait(rng)
    exp7_dynamics(rng)

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
