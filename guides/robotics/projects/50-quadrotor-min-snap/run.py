"""Project 50 -- minimum-snap trajectories for a quadrotor, and flying them.

Six experiments:
  1. one flight through eight waypoints
  2. which derivative should you minimise?
  3. when the objective does not matter at all
  4. how to split the time between the legs
  5. how fast can you fly it before the motors run out
  6. what the flatness feedforward is worth
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

from minsnap import Trajectory, allocate                       # noqa: E402
from quad import (F_MAX, G, MASS, GeoController, Quad,         # noqa: E402
                  flat_to_state)
from plot_style import COLORS, use_style, save                 # noqa: E402

import matplotlib.pyplot as plt                                # noqa: E402
from mpl_toolkits.mplot3d import Axes3D                        # noqa: E402,F401

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []

WAY = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.5], [3.0, 2.0, 2.0],
                [1.5, 3.5, 1.5], [-1.0, 3.0, 2.2], [-2.0, 1.0, 1.5],
                [-1.0, -1.0, 1.0], [0.0, 0.0, 1.0]])
# The largest upward acceleration four motors can produce, minus gravity.
A_LIMIT = 4 * F_MAX / MASS - G


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


def fly(traj, dt=0.002, ff=True, wind=np.zeros(3), mass_scale=1.0,
        ctrl_kw=None):
    """Simulate one flight and report tracking and motor use."""
    ts = np.arange(0.0, traj.t_end, dt)
    S = traj.sample(ts)
    q = Quad(p=S[0, 0])
    c = GeoController(feedforward=ff, **(ctrl_kw or {}))
    errs = np.empty(len(ts))
    fmot = np.empty((len(ts), 4))
    for k in range(len(ts)):
        ref = dict(pos=S[0, k], vel=S[1, k], acc=S[2, k], jerk=S[3, k],
                   snap=S[4, k], yaw=0.0)
        fs = flat_to_state(ref["pos"], ref["vel"], ref["acc"], ref["jerk"],
                           ref["snap"])
        ref["w"], ref["wdot"] = fs["w"], fs["wdot"]
        f, M = c(q, ref)
        fmot[k] = q.step(f * mass_scale, M, dt, wind=wind)
        errs[k] = float(np.linalg.norm(q.p - ref["pos"]))
    return dict(ts=ts, pos=S[0], errs=errs, fmot=fmot,
                mean_err=float(np.mean(errs)), max_err=float(np.max(errs)),
                sat_frac=q.sat_steps / max(q.steps, 1),
                peak_motor=float(np.max(fmot)),
                # Where each waypoint was actually crossed.
                wp_err=_wp_err(traj, S[0], ts))


def _wp_err(traj, pos, ts):
    out = []
    for e, w in zip(traj.edges, traj.way):
        k = int(np.clip(np.searchsorted(ts, e), 0, len(ts) - 1))
        out.append(float(np.linalg.norm(pos[k] - w)))
    return float(np.max(out))


def draw3d(ax, traj, S=None, label=None, color=COLORS[0]):
    if S is None:
        S = traj.sample(np.linspace(0, traj.t_end, 800))
    ax.plot(S[0][:, 0], S[0][:, 1], S[0][:, 2], color=color, lw=1.8,
            label=label)
    ax.plot(traj.way[:, 0], traj.way[:, 1], traj.way[:, 2], "o", color="k",
            ms=4)


# ================================================================ 1. a flight
def exp1():
    print("[1] one flight")
    T = allocate(WAY, total=12.0, mode="length")
    tr = Trajectory(WAY, T)
    r = fly(tr)
    fig = plt.figure(figsize=(14, 4.0))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw3d(ax, tr)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("8 waypoints, 12 s")
    ts = np.linspace(0, tr.t_end, 1500)
    S = tr.sample(ts)
    ax = fig.add_subplot(1, 3, 2)
    for d, lab, sc in [(1, "speed [m/s]", 1), (2, "accel [m/s^2]", 1),
                       (3, "jerk / 10", 0.1)]:
        ax.plot(ts, np.linalg.norm(S[d], axis=1) * sc, label=lab)
    ax.legend(fontsize=7); ax.set_xlabel("t [s]")
    ax.set_title("the derivatives the motors feel")
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(r["ts"], r["errs"] * 1000, color=COLORS[0])
    ax2 = ax.twinx()
    ax2.plot(r["ts"], r["fmot"], lw=0.6, color=COLORS[1], alpha=0.6)
    ax2.axhline(F_MAX, color=COLORS[3], ls=":")
    ax2.set_ylabel("motor thrust [N]", color=COLORS[1]); ax2.grid(False)
    ax.set_xlabel("t [s]"); ax.set_ylabel("tracking error [mm]")
    ax.set_title("tracking error and the four motors")
    save(fig, os.path.join(OUT, "overview.png"))
    rec("1_flight", total_time_s=12.0,
        peak_speed=round(tr.peak(1), 2), peak_accel=round(tr.peak(2), 2),
        peak_jerk=round(tr.peak(3), 1), peak_snap=round(tr.peak(4), 1),
        mean_err_mm=round(r["mean_err"] * 1000, 2),
        max_err_mm=round(r["max_err"] * 1000, 2),
        worst_waypoint_mm=round(r["wp_err"] * 1000, 2),
        peak_motor_N=round(r["peak_motor"], 2),
        motor_limit_N=F_MAX, sat_frac=round(r["sat_frac"], 4))
    return tr


# ================================================= 2. which derivative
def exp2():
    print("[2] which derivative to minimise")
    T = allocate(WAY, total=8.0, mode="length")
    fig = plt.figure(figsize=(14, 4.0))
    ax3 = fig.add_subplot(1, 3, 1, projection="3d")
    axes = [fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3)]
    rows = []
    for (r_, lab), c in zip([(2, "min acceleration"), (3, "min jerk"),
                             (4, "min snap (r = 4)")], COLORS):
        tr = Trajectory(WAY, T, deg=7, r=r_, cont=4, bc=3)
        f = fly(tr)
        draw3d(ax3, tr, label=lab, color=c)
        axes[0].plot(np.linspace(0, tr.t_end, 800),
                     np.linalg.norm(tr.sample(np.linspace(0, tr.t_end, 800))[4],
                                    axis=1), color=c, label=lab)
        rows.append((lab, tr.peak(2), tr.peak(3), tr.peak(4), f["peak_motor"],
                     f["sat_frac"], f["mean_err"] * 1000))
        rec("2_objective", objective=lab, total_time_s=8.0,
            peak_accel=round(tr.peak(2), 2), peak_jerk=round(tr.peak(3), 1),
            peak_snap=round(tr.peak(4), 1),
            snap_cost=round(tr.cost(4), 1),
            peak_motor_N=round(f["peak_motor"], 3),
            sat_frac=round(f["sat_frac"], 4),
            mean_err_mm=round(f["mean_err"] * 1000, 2))
    ax3.legend(fontsize=7); ax3.set_title("all three pass the waypoints")
    axes[0].set_xlabel("t [s]"); axes[0].set_ylabel("|snap| [m/s^4]")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
    axes[0].set_title("snap along the flight")
    labs = [r[0] for r in rows]
    x = np.arange(len(labs))
    axes[1].bar(x - 0.2, [r[4] for r in rows], 0.4, color=COLORS[0],
                label="peak motor [N]")
    axes[1].bar(x + 0.2, [r[6] for r in rows], 0.4, color=COLORS[1],
                label="mean error [mm]")
    axes[1].set_xticks(x); axes[1].set_xticklabels(
        ["accel", "jerk", "snap"], fontsize=8)
    axes[1].legend(fontsize=7); axes[1].set_title("what it costs to fly")
    save(fig, os.path.join(OUT, "objective.png"))


# ================================================= 3. when it doesn't matter
def exp3():
    print("[3] the continuity order decides whether the objective matters")
    T = allocate(WAY, total=8.0, mode="length")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    diffs = []
    for cont in [2, 3, 4, 5, 6]:
        curves = {}
        for r_ in (2, 4):
            tr = Trajectory(WAY, T, deg=7, r=r_, cont=cont, bc=3)
            curves[r_] = tr.sample(np.linspace(0, tr.t_end, 800))[0]
            if r_ == 4:
                snap4 = tr.peak(4)
            else:
                snap2 = tr.peak(4)
        d = float(np.max(np.linalg.norm(curves[2] - curves[4], axis=1)))
        diffs.append(d)
        # Degrees of freedom: 8 coefficients per segment, minus the
        # constraints.  When this hits zero the cost has nothing left to pick.
        m = len(WAY) - 1
        dof = 8 * m - (2 * m + cont * (m - 1) + 2 * 3)
        rec("3_continuity", cont_order=cont, free_parameters=dof,
            max_curve_difference_m=round(d, 6),
            peak_snap_min_accel=round(snap2, 1),
            peak_snap_min_snap=round(snap4, 1))
    ax.plot([2, 3, 4, 5, 6], diffs, "o-", color=COLORS[0])
    ax.set_yscale("symlog", linthresh=1e-9)
    ax.set_xlabel("derivative orders forced to match at each join")
    ax.set_ylabel("max difference between\nmin-accel and min-snap curves [m]")
    ax.set_title("force enough continuity and the objective stops mattering")
    save(fig, os.path.join(OUT, "continuity.png"))


# ================================================= 4. time allocation
def exp4():
    print("[4] time allocation")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    totals = [6.0, 8.0, 10.0, 14.0]
    for mode, c in zip(["uniform", "length", "sqrt"], COLORS):
        pk, sat = [], []
        for tot in totals:
            T = allocate(WAY, total=tot, mode=mode)
            tr = Trajectory(WAY, T)
            f = fly(tr)
            pk.append(f["peak_motor"]); sat.append(f["sat_frac"])
            rec("4_timing", mode=mode, total_time_s=tot,
                shortest_leg_s=round(float(T.min()), 2),
                longest_leg_s=round(float(T.max()), 2),
                peak_accel=round(tr.peak(2), 2),
                peak_snap=round(tr.peak(4), 1),
                peak_motor_N=round(f["peak_motor"], 3),
                sat_frac=round(f["sat_frac"], 4),
                mean_err_mm=round(f["mean_err"] * 1000, 2),
                worst_waypoint_mm=round(f["wp_err"] * 1000, 2))
        axes[0].plot(totals, pk, "o-", color=c, label=mode)
        axes[1].plot(totals, sat, "o-", color=c, label=mode)
    axes[0].axhline(F_MAX, color="0.4", ls=":", label="motor limit")
    axes[0].set_ylabel("peak motor thrust [N]")
    axes[1].set_ylabel("fraction of the flight saturated")
    for ax in axes:
        ax.set_xlabel("total flight time [s]"); ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "timing.png"))


# ================================================= 5. how fast
def exp5():
    print("[5] how fast can it be flown")
    totals = [16.0, 12.0, 9.0, 7.0, 6.0, 5.0, 4.5, 4.0, 3.5]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    pk_a, errs, sats = [], [], []
    for tot in totals:
        T = allocate(WAY, total=tot, mode="length")
        tr = Trajectory(WAY, T)
        f = fly(tr)
        pk_a.append(tr.peak(2)); errs.append(f["max_err"])
        sats.append(f["sat_frac"])
        rec("5_aggressive", total_time_s=tot, peak_speed=round(tr.peak(1), 2),
            peak_accel=round(tr.peak(2), 2),
            accel_limit=round(A_LIMIT, 2),
            peak_motor_N=round(f["peak_motor"], 2),
            sat_frac=round(f["sat_frac"], 4),
            mean_err_mm=round(f["mean_err"] * 1000, 1),
            max_err_mm=round(f["max_err"] * 1000, 1),
            worst_waypoint_mm=round(f["wp_err"] * 1000, 1))
    axes[0].plot(pk_a, np.array(errs) * 1000, "o-", color=COLORS[0])
    axes[0].axvline(A_LIMIT, color=COLORS[1], ls=":",
                    label=f"4*Fmax/m - g = {A_LIMIT:.1f}")
    axes[0].set_xlabel("peak commanded acceleration [m/s^2]")
    axes[0].set_ylabel("max tracking error [mm]")
    axes[0].set_yscale("log"); axes[0].legend(fontsize=7)
    axes[1].plot(totals, sats, "o-", color=COLORS[2])
    axes[1].set_xlabel("total flight time [s]")
    axes[1].set_ylabel("fraction of the flight saturated")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.set_title("faster ->" if ax is axes[1] else "")
    save(fig, os.path.join(OUT, "aggressive.png"))


# ================================================= 6. the feedforward
def exp6():
    print("[6] what the flatness feedforward is worth")
    totals = [16.0, 12.0, 9.0, 7.0, 6.0, 5.0]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for ff, c, lab in [(True, COLORS[0], "with flatness feedforward"),
                       (False, COLORS[1], "PD feedback only")]:
        errs = []
        for tot in totals:
            T = allocate(WAY, total=tot, mode="length")
            tr = Trajectory(WAY, T)
            f = fly(tr, ff=ff)
            errs.append(f["mean_err"])
            rec("6_feedforward", feedforward=int(ff), total_time_s=tot,
                peak_accel=round(tr.peak(2), 2),
                mean_err_mm=round(f["mean_err"] * 1000, 2),
                max_err_mm=round(f["max_err"] * 1000, 2),
                worst_waypoint_mm=round(f["wp_err"] * 1000, 2),
                sat_frac=round(f["sat_frac"], 4))
        ax.plot(totals, np.array(errs) * 1000, "o-", color=c, label=lab)
    ax.set_xlabel("total flight time [s]"); ax.invert_xaxis()
    ax.set_ylabel("mean tracking error [mm]"); ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("faster flights are to the right")
    save(fig, os.path.join(OUT, "feedforward.png"))


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
