"""Project 13 -- MPC for a unicycle.

Six experiments on a from-scratch nonlinear MPC (see ``mpc.py``):

  1. tracking a figure-8, and where the error actually lives
  2. how far ahead is far enough?  a horizon sweep
  3. MPC against pure pursuit, at three speeds
  4. a tight turn-rate limit: anticipating it vs hitting it
  5. a robot that does not do what it is told (wheel slip and a steering bias)
  6. what one solve costs, and what happens when you cannot afford it

Runs in about two minutes on a CPU.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "01-transform-calculator"))

import matplotlib.pyplot as plt  # noqa: E402

from mpc import (UnicycleMPC, unicycle_step, figure_eight, reference_window,  # noqa: E402
                 pure_pursuit, wrap)
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

DT = 0.1  # control period
SIM_SUB = 10  # physics sub-steps per control period
PERIOD = 20.0
LAPS = 1.6


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<56s} {value:>12.5f} {unit}")


def roll(controller, T=PERIOD * LAPS, x0=None, slip=1.0, w_bias=0.0, period=PERIOD,
         w_lim=(-2.5, 2.5), v_lim=(-0.4, 1.6)):
    """Closed-loop simulation.  ``controller(t, x, u_prev) -> u``.

    ``slip`` and ``w_bias`` corrupt what the robot ACTUALLY does relative to what
    it was told, so the controller's model and the world can be made to differ.
    """
    x = figure_eight(0.0, period=period).copy() if x0 is None else np.asarray(x0, float).copy()
    steps = int(T / DT)
    X = np.zeros((steps, 3))
    U = np.zeros((steps, 2))
    times = np.zeros(steps)
    u_prev = np.zeros(2)
    for k in range(steps):
        t = k * DT
        t0 = time.perf_counter()
        u = controller(t, x, u_prev)
        times[k] = time.perf_counter() - t0
        u = np.array([np.clip(u[0], *v_lim), np.clip(u[1], *w_lim)])
        X[k], U[k] = x, u
        u_real = np.array([u[0] * slip, u[1] + w_bias])
        for _ in range(SIM_SUB):
            x = unicycle_step(x, u_real, DT / SIM_SUB)[0]
        u_prev = u
    return np.arange(steps) * DT, X, U, times


def errors(ts, X, period=PERIOD):
    """Position error, split into ALONG-path and ACROSS-path components.

    The split matters for a nonholonomic robot: lagging behind on the path
    (along) is harmless and self-correcting, while being off to the side
    (across) is what actually puts a wheel in the ditch, and it is the one the
    robot cannot fix by simply sliding over.
    """
    ref = figure_eight(ts, period=period)
    d = X[:, :2] - ref[:, :2]
    ct, st = np.cos(ref[:, 2]), np.sin(ref[:, 2])
    along = d[:, 0] * ct + d[:, 1] * st
    across = -d[:, 0] * st + d[:, 1] * ct
    return np.linalg.norm(d, axis=1), along, across, np.abs(wrap(X[:, 2] - ref[:, 2]))


def best_pursuit(period=PERIOD, v_nom=1.0, w_lim=(-2.5, 2.5), T=None):
    """Pure pursuit with its ONE tuning knob swept, so the comparison is fair.

    The look-ahead distance is the whole of pure pursuit's tuning: too short and
    it wobbles, too long and it cuts corners.  Handing the MPC a carefully tuned
    horizon while leaving the baseline at an arbitrary default would make the
    result about the tuning, not about the method.
    """
    best = (np.inf, None, None)
    for la in (0.25, 0.4, 0.6, 0.9, 1.3, 1.8):
        ts, X, U, _ = roll(lambda t, x, u_prev, la=la: pure_pursuit(
            x, t, lookahead=la, v_nom=v_nom, w_lim=w_lim, period=period),
            T=T if T is not None else period * LAPS, period=period, w_lim=w_lim)
        _, _, across, _ = errors(ts, X, period=period)
        rms = float(np.sqrt(np.mean(across ** 2)))
        if rms < best[0]:
            best = (rms, la, (ts, X, U))
    return best


def mpc_controller(mpc, period=PERIOD):
    def f(t, x, u_prev):
        ref = reference_window(t, mpc.N, mpc.dt, period=period)
        u, _ = mpc.solve(x, ref, u_prev)
        return u
    return f


# ---------------------------------------------------------------------------
# 1. Tracking the figure-8
# ---------------------------------------------------------------------------
def exp1_track():
    print("[1] tracking a figure-8")
    mpc = UnicycleMPC(N=15, dt=DT)
    ts, X, U, times = roll(mpc_controller(mpc))
    err, along, across, herr = errors(ts, X)

    record("1-track", "position error RMS", 1e3 * float(np.sqrt(np.mean(err ** 2))), "mm")
    record("1-track", "worst position error", 1e3 * float(err.max()), "mm")
    record("1-track", "across-path error RMS", 1e3 * float(np.sqrt(np.mean(across ** 2))), "mm")
    record("1-track", "along-path error RMS", 1e3 * float(np.sqrt(np.mean(along ** 2))), "mm")
    record("1-track", "heading error RMS", float(np.degrees(np.sqrt(np.mean(herr ** 2)))), "deg")
    record("1-track", "median solve time", 1e3 * float(np.median(times)), "ms")
    record("1-track", "worst solve time", 1e3 * float(times.max()), "ms")
    record("1-track", "solve time as a share of the 100 ms tick",
           100 * float(np.median(times)) / DT, "%")

    ref = figure_eight(ts)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    axes[0].plot(ref[:, 0], ref[:, 1], "--", color=COLORS[6], lw=1.4, label="reference")
    axes[0].plot(X[:, 0], X[:, 1], color=COLORS[0], label="MPC")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("The path")
    axes[0].legend(fontsize=8)
    axes[1].plot(ts, 1e3 * across, color=COLORS[1], label="across path")
    axes[1].plot(ts, 1e3 * along, color=COLORS[2], label="along path")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("error (mm)")
    axes[1].set_title("Where the error lives")
    axes[1].legend(fontsize=8)
    axes[2].plot(ts, U[:, 0], color=COLORS[0], label="v (m/s)")
    axes[2].plot(ts, U[:, 1], color=COLORS[3], label="omega (rad/s)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("The commands it chose")
    axes[2].legend(fontsize=8)
    save(fig, os.path.join(OUT, "track.png"))


# ---------------------------------------------------------------------------
# 2. Horizon
# ---------------------------------------------------------------------------
def exp2_horizon():
    print("[2] how far ahead is far enough?")
    Ns = [2, 3, 5, 8, 12, 15, 20, 30]
    errs, tms = [], []
    keep = {}
    for N in Ns:
        mpc = UnicycleMPC(N=N, dt=DT)
        ts, X, U, times = roll(mpc_controller(mpc))
        _, _, across, _ = errors(ts, X)
        errs.append(1e3 * float(np.sqrt(np.mean(across ** 2))))
        tms.append(1e3 * float(np.median(times)))
        if N in (2, 5, 15):
            keep[N] = X
        record("2-horizon", f"N={N} ({N * DT:.1f} s ahead): across-path RMS", errs[-1], "mm")
        record("2-horizon", f"N={N}: median solve time", tms[-1], "ms")
    best = Ns[int(np.argmin(errs))]
    record("2-horizon", "best horizon", best, "steps")
    record("2-horizon", "  in seconds of look-ahead", best * DT, "s")
    record("2-horizon", "error at N=2 vs at the best N", errs[0] / min(errs), "x")
    record("2-horizon", "error at N=30 vs at the best N", errs[-1] / min(errs), "x")
    record("2-horizon", "solve time at N=30 vs N=2", tms[-1] / tms[0], "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].loglog(Ns, errs, "o-", color=COLORS[0], label="across-path error")
    ax2 = axes[0].twinx()
    ax2.loglog(Ns, tms, "s--", color=COLORS[1], label="solve time")
    ax2.set_ylabel("median solve time (ms)", color=COLORS[1])
    axes[0].set_xlabel("horizon N (steps)")
    axes[0].set_ylabel("across-path error RMS (mm)", color=COLORS[0])
    axes[0].set_title("Longer plans track better -- up to a point")
    ref = figure_eight(np.arange(int(PERIOD * LAPS / DT)) * DT)
    axes[1].plot(ref[:, 0], ref[:, 1], "--", color=COLORS[6], lw=1.4, label="reference")
    for k, (N, X) in enumerate(sorted(keep.items())):
        axes[1].plot(X[:, 0], X[:, 1], color=COLORS[k], label=f"N = {N}")
    axes[1].set_aspect("equal")
    axes[1].set_title("A myopic plan cuts the corners")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "horizon.png"))


# ---------------------------------------------------------------------------
# 3. MPC vs pure pursuit
# ---------------------------------------------------------------------------
def exp3_vs_pursuit():
    print("[3] MPC vs pure pursuit")
    periods = [30.0, 20.0, 12.0]
    rows = []
    keep = {}
    for p in periods:
        mpc = UnicycleMPC(N=15, dt=DT)
        ts, Xm, _, _ = roll(mpc_controller(mpc, period=p), T=p * LAPS, period=p)
        _, _, am, _ = errors(ts, Xm, period=p)
        v_nom = 2 * np.pi * 2.0 / p * 1.15  # roughly the path's own speed
        pp_rms, la, (ts2, Xp, _) = best_pursuit(period=p, v_nom=v_nom, T=p * LAPS)
        record("3-pursuit", f"lap {p:.0f} s: best pure-pursuit look-ahead", la, "m")
        rows.append((p, 1e3 * float(np.sqrt(np.mean(am ** 2))), 1e3 * pp_rms))
        record("3-pursuit", f"lap {p:.0f} s: MPC across-path RMS", rows[-1][1], "mm")
        record("3-pursuit", f"lap {p:.0f} s: pure-pursuit across-path RMS", rows[-1][2], "mm")
        record("3-pursuit", f"lap {p:.0f} s: MPC advantage", rows[-1][2] / rows[-1][1], "x")
        if p == 12.0:
            keep = {"MPC": Xm, "pure pursuit": Xp, "reference": figure_eight(ts, period=p)}

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    x = np.arange(len(rows))
    axes[0].bar(x - 0.18, [r[1] for r in rows], 0.36, color=COLORS[0], label="MPC")
    axes[0].bar(x + 0.18, [r[2] for r in rows], 0.36, color=COLORS[1], label="pure pursuit")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r[0]:.0f} s lap" for r in rows])
    axes[0].set_ylabel("across-path error RMS (mm)")
    axes[0].set_yscale("log")
    axes[0].set_title("Same information, different amount of thinking")
    axes[0].legend(fontsize=8)
    axes[1].plot(keep["reference"][:, 0], keep["reference"][:, 1], "--", color=COLORS[6], lw=1.4)
    axes[1].plot(keep["MPC"][:, 0], keep["MPC"][:, 1], color=COLORS[0], label="MPC")
    axes[1].plot(keep["pure pursuit"][:, 0], keep["pure pursuit"][:, 1], color=COLORS[1],
                 label="pure pursuit")
    axes[1].set_aspect("equal")
    axes[1].set_title("The fastest lap (12 s)")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "vs_pursuit.png"))


# ---------------------------------------------------------------------------
# 4. A tight turn-rate limit
# ---------------------------------------------------------------------------
def exp4_limits():
    print("[4] a tight turn-rate limit")
    lims = [2.5, 1.2, 0.8, 0.6, 0.45]
    mpc_e, pp_e = [], []
    for wl in lims:
        mpc = UnicycleMPC(N=15, dt=DT, w_lim=(-wl, wl))
        ts, Xm, Um, _ = roll(mpc_controller(mpc), w_lim=(-wl, wl))
        _, _, am, _ = errors(ts, Xm)
        pp_rms, la, _ = best_pursuit(v_nom=1.0, w_lim=(-wl, wl))
        mpc_e.append(1e3 * float(np.sqrt(np.mean(am ** 2))))
        pp_e.append(1e3 * pp_rms)
        sat = 100.0 * float(np.mean(np.abs(np.abs(Um[:, 1]) - wl) < 1e-6))
        record("4-limits", f"omega limit {wl:.2f} rad/s: MPC across-path RMS", mpc_e[-1], "mm")
        record("4-limits", f"omega limit {wl:.2f} rad/s: MPC at the limit", sat, "% of ticks")
        record("4-limits", f"omega limit {wl:.2f} rad/s: pure-pursuit across-path RMS", pp_e[-1], "mm")
    record("4-limits", "MPC advantage at the loosest limit", pp_e[0] / mpc_e[0], "x")
    record("4-limits", "MPC advantage at the tightest limit", pp_e[-1] / mpc_e[-1], "x")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.semilogy(lims, mpc_e, "o-", color=COLORS[0], label="MPC")
    ax.semilogy(lims, pp_e, "s-", color=COLORS[1], label="pure pursuit")
    ax.invert_xaxis()
    ax.set_xlabel("turn-rate limit (rad/s), tighter to the right")
    ax.set_ylabel("across-path error RMS (mm)")
    ax.set_title("Knowing the limit in advance is worth more than reacting to it")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "limits.png"))


# ---------------------------------------------------------------------------
# 5. A robot that does not do what it is told
# ---------------------------------------------------------------------------
def exp5_mismatch():
    print("[5] wheel slip and a steering bias")
    cases = [("perfect", 1.0, 0.0), ("15% slip", 0.85, 0.0), ("30% slip", 0.70, 0.0),
             ("steering bias 0.15 rad/s", 1.0, 0.15), ("slip + bias", 0.80, 0.15)]
    rows = []
    for label, slip, bias in cases:
        mpc = UnicycleMPC(N=15, dt=DT)
        ts, X, U, _ = roll(mpc_controller(mpc), slip=slip, w_bias=bias)
        _, along, across, _ = errors(ts, X)
        rows.append((label, 1e3 * float(np.sqrt(np.mean(across ** 2))),
                     1e3 * float(np.sqrt(np.mean(along ** 2)))))
        record("5-mismatch", f"{label}: across-path RMS", rows[-1][1], "mm")
        record("5-mismatch", f"{label}: along-path RMS", rows[-1][2], "mm")
        record("5-mismatch", f"{label}: mean commanded speed", float(np.mean(U[:, 0])), "m/s")
    record("5-mismatch", "across-path cost of 30% slip", rows[2][1] / rows[0][1], "x")
    record("5-mismatch", "along-path cost of 30% slip", rows[2][2] / rows[0][2], "x")

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], 0.36, color=COLORS[1], label="across path")
    ax.bar(x + 0.18, [r[2] for r in rows], 0.36, color=COLORS[2], label="along path")
    ax.set_xticks(x)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7)
    ax.set_ylabel("error RMS (mm)")
    ax.set_title("Re-planning absorbs the error the robot can fix; the rest becomes lag")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "mismatch.png"))


# ---------------------------------------------------------------------------
# 6. The compute budget
# ---------------------------------------------------------------------------
def exp6_budget():
    print("[6] what one solve costs")
    rows = []
    for iters in (1, 2, 4, 8):
        mpc = UnicycleMPC(N=15, dt=DT, gn_iters=iters)
        ts, X, U, times = roll(mpc_controller(mpc))
        _, _, across, _ = errors(ts, X)
        rows.append((iters, 1e3 * float(np.sqrt(np.mean(across ** 2))),
                     1e3 * float(np.median(times))))
        record("6-budget", f"{iters} Gauss-Newton iterations: across-path RMS", rows[-1][1], "mm")
        record("6-budget", f"{iters} Gauss-Newton iterations: median solve time", rows[-1][2], "ms")

    # Cold start: the warm start is what makes one iteration enough.
    mpc = UnicycleMPC(N=15, dt=DT, gn_iters=1)
    orig_solve = mpc.solve

    def cold(x0, ref, u_prev):
        mpc.U = np.zeros((mpc.N, 2))
        return orig_solve(x0, ref, u_prev)

    mpc.solve = cold
    ts, X, _, _ = roll(mpc_controller(mpc))
    _, _, across_cold, _ = errors(ts, X)
    record("6-budget", "1 iteration WITHOUT the warm start: across-path RMS",
           1e3 * float(np.sqrt(np.mean(across_cold ** 2))), "mm")
    record("6-budget", "  vs the same budget WITH a warm start",
           float(np.sqrt(np.mean(across_cold ** 2))) / (rows[0][1] / 1e3), "x worse")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot([r[2] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0])
    for it, e, t in rows:
        ax.annotate(f"{it} iter", (t, e), fontsize=8, textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("median solve time (ms)")
    ax.set_ylabel("across-path error RMS (mm)")
    ax.set_title("Buying accuracy with compute, on a 100 ms tick")
    save(fig, os.path.join(OUT, "budget.png"))


def main():
    t0 = time.perf_counter()
    exp1_track()
    exp2_horizon()
    exp3_vs_pursuit()
    exp4_limits()
    exp5_mismatch()
    exp6_budget()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
