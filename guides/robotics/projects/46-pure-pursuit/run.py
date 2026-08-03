"""Project 46 -- pure pursuit, and what the look-ahead distance actually buys.

Seven experiments:
  1. one lap, and where the error lives
  2. the look-ahead sweep -- two failure modes, one knob
  3. the circle test: does pure pursuit really cut corners?
  4. corner cutting, measured on a single 90-degree corner
  5. look-ahead has to scale with speed
  6. the control period is the other half of the stability story
  7. pure pursuit vs Stanley vs a naive heading controller
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

from robot import (Path, path_circle, path_corner, path_racetrack,      # noqa: E402
                   path_slalom, pure_pursuit, stanley, heading_p, simulate)
from plot_style import COLORS, use_style, save                          # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


def draw_path(ax, path, **kw):
    kw.setdefault("color", "0.55")
    kw.setdefault("lw", 1.2)
    kw.setdefault("ls", "--")
    ax.plot(path.pts[:, 0], path.pts[:, 1], **kw)


# ================================================================= 1. one lap
def exp1():
    print("[1] one lap of the racetrack")
    track = path_racetrack(straight=8.0, R=2.0)
    r = simulate(track, pure_pursuit, v=1.0, L=1.0, ctrl_hz=20.0, t_max=80,
                 pose_sigma=0.02, robot_kw=dict(tau=0.06))

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    draw_path(ax, track)
    ax.plot(r["xy"][:, 0], r["xy"][:, 1], color=COLORS[0], lw=1.6)
    # Sketch the look-ahead geometry at one instant.
    k = len(r["xy"]) // 6
    x, y, th = r["xy"][k, 0], r["xy"][k, 1], r["th"][k]
    every = max(len(r["xy"]) // max(len(r["tgt"]), 1), 1)
    tgt = r["tgt"][min(k // every, len(r["tgt"]) - 1)]
    ax.add_patch(plt.Circle((x, y), 1.0, fill=False, color=COLORS[1], lw=1.0))
    ax.plot([x, tgt[0]], [y, tgt[1]], color=COLORS[1], lw=1.4)
    ax.arrow(x, y, 0.9 * math.cos(th), 0.9 * math.sin(th), head_width=0.22,
             color=COLORS[3], length_includes_head=True)
    ax.plot([x], [y], "o", color=COLORS[3], ms=6)
    ax.set_aspect("equal")
    ax.set_title("racetrack, L = 1.0 m, v = 1.0 m/s")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    axes[1].plot(r["t"], r["e"], color=COLORS[0])
    axes[1].axhline(0, color="0.6", lw=0.8)
    JOIN_S = [0.0, 8.0, 8.0 + 2 * math.pi, 16.0 + 2 * math.pi]
    for xline in JOIN_S:
        axes[1].axvline(xline, color=COLORS[1], lw=0.8, ls=":")
    axes[1].set_title("cross-track error (dotted = curvature jumps)")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("e [m]")

    axes[2].plot(r["t"], r["w"], color=COLORS[2])
    axes[2].set_title("commanded turn rate")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("omega [rad/s]")
    save(fig, os.path.join(OUT, "overview.png"))

    # Split the error by what the path is doing there.
    track_pts = track.pts
    seg = []
    hint = 0
    for p in r["xy"]:
        hint = track.closest(p, hint=hint, window=200)
        seg.append(hint)
    seg = np.asarray(seg)
    is_arc = np.abs(track_pts[seg, 0]) > 4.0 - 1e-6
    s_arr = track.s[seg]
    # A "join" is within 0.6 m of arc length of a curvature jump.
    joins = np.array(JOIN_S + [track.length])
    ds = np.min(np.abs(s_arr[:, None] - joins[None, :]), axis=1)
    near_join = ds < 0.6
    rec("1_error_by_segment", where="straight_interior",
        mean_abs_e=round(float(np.mean(np.abs(r["e"][~is_arc & ~near_join]))), 4))
    rec("1_error_by_segment", where="arc_interior",
        mean_abs_e=round(float(np.mean(np.abs(r["e"][is_arc & ~near_join]))), 4))
    rec("1_error_by_segment", where="curvature_join",
        mean_abs_e=round(float(np.mean(np.abs(r["e"][near_join]))), 4))
    rec("1_lap", mean_abs_e=round(r["mean_abs_e"], 4),
        max_abs_e=round(r["max_abs_e"], 4), lap_time=round(r["t_done"], 2))
    return track


# ================================================= 2. the look-ahead sweep
def exp2(track):
    print("[2] look-ahead sweep")
    Ls = [0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.4, 3.0, 4.0]
    out = []
    for L in Ls:
        r = simulate(track, pure_pursuit, v=1.0, L=L, ctrl_hz=20.0, t_max=90,
                     pose_sigma=0.02, robot_kw=dict(tau=0.06))
        out.append((L, r["mean_abs_e"], r["max_abs_e"], r["w_rate"]))
        rec("2_lookahead", L=L, mean_abs_e=round(r["mean_abs_e"], 4),
            max_abs_e=round(r["max_abs_e"], 4), w_rate=round(r["w_rate"], 3),
            diverged=int(r["diverged"]))
    out = np.asarray(out)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(out[:, 0], out[:, 1], "o-", color=COLORS[0], label="mean |e|")
    axes[0].plot(out[:, 0], out[:, 2], "s-", color=COLORS[1], label="max |e|")
    axes[0].set_xlabel("look-ahead L [m]"); axes[0].set_ylabel("error [m]")
    axes[0].set_title("tracking error"); axes[0].legend()
    axes[1].plot(out[:, 0], out[:, 3], "o-", color=COLORS[2])
    axes[1].set_xlabel("look-ahead L [m]")
    axes[1].set_ylabel("mean |d omega / dt| [rad/s^2]")
    axes[1].set_title("steering roughness")
    for L, c in zip([0.25, 1.0, 3.0], COLORS):
        r = simulate(track, pure_pursuit, v=1.0, L=L, ctrl_hz=20.0, t_max=90,
                     pose_sigma=0.02, robot_kw=dict(tau=0.06))
        axes[2].plot(r["xy"][:, 0], r["xy"][:, 1], color=c, lw=1.4,
                     label=f"L = {L}")
    draw_path(axes[2], track)
    axes[2].set_aspect("equal"); axes[2].legend(loc="center")
    axes[2].set_title("the two failure modes")
    save(fig, os.path.join(OUT, "lookahead.png"))
    best = out[int(np.argmin(out[:, 1])), 0]
    rec("2_best", best_L=float(best),
        mean_abs_e=round(float(out[:, 1].min()), 4))


# ================================================= 3. the circle test
def exp3():
    print("[3] does pure pursuit cut a constant-curvature arc?")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    for R, c in zip([2.0, 4.0, 8.0], COLORS):
        Ls, offs = [], []
        for L in [0.2, 0.4, 0.8, 1.2, 1.8, 2.5]:
            circ = path_circle(R=R)
            r = simulate(circ, pure_pursuit, v=1.0, L=L, ctrl_hz=200.0,
                         dt=0.005, t_max=200, laps=2.0)
            # Steady state = the last half lap only.
            tail = r["e"][len(r["e"]) // 2:]
            off = float(np.mean(tail))
            Ls.append(L); offs.append(off)
            sagitta = L * L / (8.0 * R)
            rec("3_circle", R=R, L=L, steady_offset=round(off, 5),
                sagitta_prediction=round(sagitta, 5))
        axes[0].plot(Ls, np.abs(offs), "o-", color=c, label=f"R = {R} m")
        axes[0].plot(Ls, [L * L / (8 * R) for L in Ls], ":", color=c)
    axes[0].set_xlabel("look-ahead L [m]")
    axes[0].set_ylabel("|steady-state offset| [m]")
    axes[0].set_title("solid = measured, dotted = the chord/sagitta guess")
    axes[0].set_yscale("log"); axes[0].legend()

    circ = path_circle(R=4.0)
    for L, c in zip([0.4, 1.2, 2.5], COLORS):
        r = simulate(circ, pure_pursuit, v=1.0, L=L, ctrl_hz=200.0, dt=0.005,
                     t_max=200, laps=2.0)
        axes[1].plot(r["t"], r["e"], color=c, lw=1.4, label=f"L = {L}")
    axes[1].axhline(0, color="0.6", lw=0.8)
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("e [m]")
    axes[1].set_title("R = 4 m: the transient dies, the offset does not exist")
    axes[1].legend()
    save(fig, os.path.join(OUT, "circle.png"))


# ================================================= 4. corner cutting
def exp4():
    print("[4] corner cutting on a single right angle")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    corner = path_corner()
    res = []
    for L in [0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]:
        r = simulate(corner, pure_pursuit, v=1.0, L=L, ctrl_hz=100.0,
                     t_max=40)
        # Distance from the corner vertex to the closest point of the trace,
        # which for an inside cut is exactly how much was cut.
        d = np.linalg.norm(r["xy"] - np.array([0.0, 0.0]), axis=1)
        cut = float(np.min(d))
        res.append((L, cut, float(r["diverged"])))
        rec("4_corner", L=L, cut_distance=round(cut, 4),
            max_abs_e=round(r["max_abs_e"], 4), diverged=int(r["diverged"]),
            omega_needed=round(1.0 / max(L / math.sqrt(2.0), 1e-6), 2))
        if L in (0.2, 1.0, 3.0):
            axes[0].plot(r["xy"][:, 0], r["xy"][:, 1], lw=1.5, label=f"L = {L}")
    draw_path(axes[0], corner)
    axes[0].set_aspect("equal"); axes[0].legend()
    axes[0].set_title("the corner is where L is spent")
    axes[0].set_xlabel("x [m]"); axes[0].set_ylabel("y [m]")
    res = np.asarray(res)
    ok = res[:, 2] < 0.5
    axes[1].plot(res[ok, 0], res[ok, 1], "o-", color=COLORS[0],
                 label="measured cut")
    axes[1].plot(res[~ok, 0], res[~ok, 1], "x", color=COLORS[3], ms=9,
                 label="lost the path")
    axes[1].plot(res[:, 0], res[:, 0] / math.sqrt(2), ":", color=COLORS[1],
                 label="L / sqrt(2)")
    axes[1].set_xlabel("look-ahead L [m]"); axes[1].set_ylabel("cut [m]")
    axes[1].set_title("cut grows LINEARLY in L"); axes[1].legend()
    save(fig, os.path.join(OUT, "corner.png"))


# ================================================= 5. speed
def exp5(track):
    print("[5] look-ahead must scale with speed")
    speeds = [0.5, 1.0, 1.5, 2.0, 2.5]
    Ls = [0.2, 0.35, 0.5, 0.75, 1.0, 1.4, 2.0, 2.8]
    grid = np.zeros((len(speeds), len(Ls)))
    div = np.zeros((len(speeds), len(Ls)), dtype=bool)
    for a, v in enumerate(speeds):
        for b, L in enumerate(Ls):
            r = simulate(track, pure_pursuit, v=v, L=L, ctrl_hz=20.0,
                         t_max=120, pose_sigma=0.02, robot_kw=dict(tau=0.08))
            grid[a, b] = r["mean_abs_e"]
            div[a, b] = r["diverged"]
            rec("5_speed_L", v=v, L=L, mean_abs_e=round(r["mean_abs_e"], 4),
                diverged=int(r["diverged"]))
    best_L = [Ls[int(np.argmin(grid[a]))] for a in range(len(speeds))]
    # The other boundary: the SMALLEST look-ahead that still keeps the path.
    min_stable = []
    for a, v in enumerate(speeds):
        s_ok = [Ls[b] for b in range(len(Ls)) if not div[a][b]]
        min_stable.append(min(s_ok) if s_ok else float("nan"))
        rec("5_boundary", v=v, best_L=best_L[a], min_stable_L=min_stable[-1])

    # One adaptive rule against the single best fixed L over the whole range.
    fixed_best = Ls[int(np.argmin(grid.mean(axis=0)))]
    ad, fx = [], []
    for v in speeds:
        ra = simulate(track, pure_pursuit, v=v, L=None, adaptive=(0.25, 0.35),
                      ctrl_hz=20.0, t_max=120, pose_sigma=0.02,
                      robot_kw=dict(tau=0.08))
        rf = simulate(track, pure_pursuit, v=v, L=fixed_best, ctrl_hz=20.0,
                      t_max=120, pose_sigma=0.02, robot_kw=dict(tau=0.08))
        ad.append(ra["mean_abs_e"]); fx.append(rf["mean_abs_e"])
        rec("5_adaptive", v=v, adaptive_L=round(0.25 * v + 0.35, 2),
            adaptive_e=round(ra["mean_abs_e"], 4),
            fixed_L=fixed_best, fixed_e=round(rf["mean_abs_e"], 4))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    im = axes[0].imshow(grid, aspect="auto", origin="lower", cmap="viridis")
    axes[0].set_xticks(range(len(Ls)))
    axes[0].set_xticklabels([str(x) for x in Ls])
    axes[0].set_yticks(range(len(speeds)))
    axes[0].set_yticklabels([str(x) for x in speeds])
    axes[0].set_xlabel("look-ahead L [m]"); axes[0].set_ylabel("speed [m/s]")
    axes[0].set_title("mean |e| [m]  (* best, x = lost the path)")
    axes[0].grid(False)
    for a in range(len(speeds)):
        axes[0].plot(int(np.argmin(grid[a])), a, "*", color="w", ms=13)
        for b in range(len(Ls)):
            if div[a, b]:
                axes[0].plot(b, a, "x", color="r", ms=8)
    fig.colorbar(im, ax=axes[0])
    axes[1].plot(speeds, fx, "o-", color=COLORS[1],
                 label=f"best single fixed L = {fixed_best}")
    axes[1].plot(speeds, ad, "s-", color=COLORS[0],
                 label="adaptive L = 0.25 v + 0.35")
    axes[1].set_xlabel("speed [m/s]"); axes[1].set_ylabel("mean |e| [m]")
    axes[1].set_title("one rule beats one number"); axes[1].legend()
    save(fig, os.path.join(OUT, "speed.png"))
    rec("5_best_L_per_speed", **{f"v{v}": L for v, L in zip(speeds, best_L)})


# ================================================= 6. control period
def exp6(track):
    print("[6] control rate and stability")
    rates = [50.0, 20.0, 10.0, 5.0, 2.5]
    Ls = [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.2, 1.6, 2.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ax, ps, lab in [(axes[0], 0.0, "perfect pose"),
                        (axes[1], 0.02, "2 cm pose noise")]:
        for a, hz in enumerate(rates):
            dv = []
            for b, L in enumerate(Ls):
                r = simulate(track, pure_pursuit, v=1.5, L=L, ctrl_hz=hz,
                             t_max=120, pose_sigma=ps, robot_kw=dict(tau=0.08))
                rec("6_rate", pose_sigma=ps, ctrl_hz=hz, L=L,
                    max_abs_e=round(r["max_abs_e"], 4),
                    w_rate=round(r["w_rate"], 3),
                    diverged=int(r["diverged"]))
                dv.append(r["diverged"])
            # The boundary is the smallest L above which NOTHING diverges --
            # not simply the first L that happens to survive, which a single
            # lucky run can fake.
            last_bad = max([b for b, d in enumerate(dv) if d], default=-1)
            row_min = Ls[last_bad + 1] if last_bad + 1 < len(Ls) else None
            rec("6_min_stable_L", pose_sigma=ps, ctrl_hz=hz,
                min_stable_L=row_min)
            ax.plot([hz], [row_min if row_min else np.nan], "o",
                    color=COLORS[0], ms=8)
        ys = [r["min_stable_L"] for r in ROWS
              if r["experiment"] == "6_min_stable_L" and r["pose_sigma"] == ps]
        ax.plot(rates, ys, "-", color=COLORS[0])
        ax.set_xscale("log")
        ax.set_xlabel("control rate [Hz]")
        ax.set_ylabel("smallest look-ahead that keeps the path [m]")
        ax.set_ylim(0, 2.2)
        ax.set_title(lab)
    for L, hz, c in [(0.5, 20.0, COLORS[0]), (0.5, 2.5, COLORS[1]),
                     (1.2, 2.5, COLORS[2])]:
        r = simulate(track, pure_pursuit, v=1.5, L=L, ctrl_hz=hz, t_max=120,
                     pose_sigma=0.02, robot_kw=dict(tau=0.08))
        axes[2].plot(r["xy"][:, 0], r["xy"][:, 1], color=c, lw=1.3,
                     label=f"L={L}, {hz:g} Hz")
    draw_path(axes[2], track)
    axes[2].set_aspect("equal"); axes[2].legend(loc="center", fontsize=7)
    axes[2].set_title("a short look-ahead needs a fast loop")
    save(fig, os.path.join(OUT, "rate.png"))


# ================================================= 7. three trackers
def exp7():
    print("[7] pure pursuit vs Stanley vs heading-only")
    sl = path_slalom()
    # Tune each controller's one gain on the nominal case, so the comparison
    # is between two tuned controllers and not between a tuned one and a
    # straw man.  This is the step that gets skipped in most blog posts.
    GRIDS = [
        ("pure pursuit", pure_pursuit,
         [("L", L) for L in [0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.5]]),
        ("Stanley", stanley,
         [("k_e", k) for k in [0.5, 1.0, 1.6, 2.5, 4.0, 6.0, 9.0]]),
        ("heading P", heading_p,
         [("k", k) for k in [1.0, 2.5, 5.0, 9.0, 14.0]]),
    ]

    def build(name, gname, g):
        if gname == "L":
            return dict(L=g, tracker_kw={})
        return dict(L=0.8, tracker_kw={gname: g})

    tuned = {}
    for tune_ps, tag in [(0.0, "clean"), (0.05, "noisy")]:
        for name, tf, grid in GRIDS:
            best, bestv = grid[0][1], 1e9
            for gname, g in grid:
                r = simulate(sl, tf, v=1.0, ctrl_hz=20.0, t_max=60, lost=3.0,
                             pose_sigma=tune_ps, **build(name, gname, g))
                rec("7_tune", tuned_on=tag, tracker=name, gain=g,
                    mean_abs_e=round(r["mean_abs_e"], 4))
                if r["mean_abs_e"] < bestv and not r["diverged"]:
                    best, bestv = g, r["mean_abs_e"]
            tuned[(tag, name)] = (grid[0][0], best)
            print(f"    tuned-on-{tag}: {name} {grid[0][0]}={best}")

    # The cross-table: does a gain tuned on a clean simulator survive noise?
    for tag in ("clean", "noisy"):
        for name, tf, grid in GRIDS:
            gname, g = tuned[(tag, name)]
            for ev_ps, ev in [(0.0, "clean"), (0.05, "noisy")]:
                r = simulate(sl, tf, v=1.0, ctrl_hz=20.0, t_max=60, lost=3.0,
                             pose_sigma=ev_ps, **build(name, gname, g))
                rec("7_transfer", tuned_on=tag, evaluated_on=ev, tracker=name,
                    gain=g, mean_abs_e=round(r["mean_abs_e"], 4),
                    w_rate=round(r["w_rate"], 3))

    trackers = [(name, tf, build(name, *tuned[("clean", name)]))
                for name, tf, _ in GRIDS]
    scen = [("nominal", dict(), None, 0.0),
            ("start 1.2 m off the path", dict(), (0.0, 1.2, 0.0), 0.0),
            ("pose noise 5 cm", dict(), None, 0.05),
            ("120 ms actuator lag", dict(tau=0.12), None, 0.0)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    for k, (sname, rkw, start, ps) in enumerate(scen):
        for (tname, tf, kw), c in zip(trackers, COLORS):
            r = simulate(sl, tf, v=1.0, ctrl_hz=20.0, t_max=60, lost=3.0,
                         start=start, robot_kw=rkw, pose_sigma=ps, **kw)
            rec("7_trackers", scenario=sname, tracker=tname,
                mean_abs_e=round(r["mean_abs_e"], 4),
                max_abs_e=round(r["max_abs_e"], 4),
                w_rate=round(r["w_rate"], 3), finished=int(r["finished"]))
            axes[k].plot(r["xy"][:, 0], r["xy"][:, 1], color=c, lw=1.3,
                         label=tname)
        draw_path(axes[k], sl)
        axes[k].set_title(sname, fontsize=9)
        axes[k].set_xlim(0, 24); axes[k].set_ylim(-2.6, 2.6)
        axes[k].set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]"); axes[0].legend(fontsize=7)
    save(fig, os.path.join(OUT, "trackers.png"))


if __name__ == "__main__":
    t0 = time.time()
    track = exp1()
    exp2(track)
    exp3()
    exp4()
    exp5(track)
    exp6(track)
    exp7()
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(ROWS)
    print(f"done in {time.time() - t0:.1f}s -> {OUT}/results.csv")
