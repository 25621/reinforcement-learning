"""Project 36 -- TOPP: turning a path into a schedule.

Seven experiments:

  1. the speed profile, and why it looks like a bang-bang controller
  2. TOPP against one global speed limit for the whole path
  3. which limit binds where, and the saturation check for optimality
  4. the headline: the SHORTEST path is not the FASTEST path
  5. grid resolution: how many points before the answer stops moving
  6. torque limits instead of acceleration limits, on a real 2-link arm
  7. verification, and what happens if you shave 10% off the answer

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
sys.path.insert(0, os.path.join(_PROJ, "32-rrt-in-2d"))
sys.path.insert(0, os.path.join(_PROJ, "34-shortcut-smoothing"))
sys.path.insert(0, os.path.join(_PROJ, "10-inverse-dynamics-from-scratch"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from topp import (CubicPath, Limits, TorqueLimits, topp_ra, uniform_scaling,  # noqa: E402
                  trapezoid_scaling, verify, active_fraction)
from rrt import Env, world_blobs, rrt                                      # noqa: E402
from smooth import shortcut, blend_corners, resample, min_turn_radius, path_cost  # noqa: E402
from plot_style import COLORS, use_style, save                             # noqa: E402
import dynamics as dyn                                                     # noqa: E402

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import Circle                                      # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

VMAX = np.array([1.0, 1.0])
AMAX = np.array([2.0, 2.0])
LIM = Limits(VMAX, AMAX)

SHAPES = {
    "long straight + one corner": [[0, 0], [4, 0], [4.2, 0.1], [4.3, 0.4],
                                   [4.3, 4]],
    "gentle S": [[0, 0], [2, 0.5], [4, 0], [6, 0.5], [8, 0]],
    "zig-zag": [[0, 0], [1, 1], [2, -1], [3, 1], [4, -1], [5, 0]],
    "straight line": [[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]],
}


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# =====================================================================  1
def exp1_profile(rng):
    banner("1. The speed profile")

    p = CubicPath(np.array(SHAPES["long straight + one corner"], float))
    t0 = time.perf_counter()
    r = topp_ra(p, LIM, N=200)
    ms = (time.perf_counter() - t0) * 1e3
    print(f"  duration {r['duration']:.3f} s, solved in {ms:.0f} ms "
          f"(200 grid points, two sweeps)")
    print(f"  feasible: {r['feasible']}")
    print(f"  worst violation of limits: "
          f"{max(verify(p, r, LIM).values()):.4f} (1.0 = exactly at the limit)")
    print(f"  fraction of the path with some limit saturated: "
          f"{100*active_fraction(p, r, LIM):.1f}%")
    record(1, "profile", duration=round(r["duration"], 4), solve_ms=round(ms, 1),
           feasible=r["feasible"],
           worst_ratio=round(max(verify(p, r, LIM).values()), 5),
           saturated_pct=round(100 * active_fraction(p, r, LIM), 2))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.8), constrained_layout=True)
    q = p.eval(r["s"])
    axes[0].plot(q[:, 0], q[:, 1], color=COLORS[0])
    sc = axes[0].scatter(q[:, 0], q[:, 1], c=r["sdot"], cmap="viridis", s=8)
    axes[0].set_aspect("equal")
    axes[0].set_title("the path, coloured by speed")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)
    fig.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.03,
                 label="sdot (path units/s)")

    axes[1].plot(r["s"], np.sqrt(np.minimum(r["xmax"], 20)), color=COLORS[1],
                 ls="--", label="velocity limit curve")
    axes[1].plot(r["s"], np.sqrt(np.maximum(r["xbar"], 0)), color=COLORS[2],
                 ls=":", label="controllable set (backward pass)")
    axes[1].plot(r["s"], r["sdot"], color=COLORS[0], label="chosen profile")
    axes[1].set_xlabel("path coordinate s")
    axes[1].set_ylabel("sdot (path units per second)")
    axes[1].set_title("The profile rides whichever\nceiling is lowest")
    axes[1].legend(fontsize=7)

    axes[2].plot(r["s"][:-1], r["u"], color=COLORS[0])
    axes[2].set_xlabel("path coordinate s")
    axes[2].set_ylabel("sddot")
    axes[2].set_title("...and the acceleration slams\nbetween extremes "
                      "(that is 'bang-bang')")
    save(fig, os.path.join(OUT, "profile.png"))


# =====================================================================  2
def exp2_vs_global(rng):
    banner("2. TOPP against one global speed limit")

    print(f"  {'path':<28s} {'TOPP (s)':>9s} {'trapezoid (s)':>14s} "
          f"{'saving':>8s} {'constant-speed (s)':>19s}")
    rows = []
    for nm, wp in SHAPES.items():
        p = CubicPath(np.array(wp, float))
        r = topp_ra(p, LIM, N=200)
        z = trapezoid_scaling(p, LIM)
        u = uniform_scaling(p, LIM)
        rows.append((nm, r["duration"], z["duration"], u["duration"]))
        print(f"  {nm:<28s} {r['duration']:9.3f} {z['duration']:14.3f} "
              f"{100*(1-r['duration']/z['duration']):7.1f}% "
              f"{u['duration']:19.3f}")
        record(2, nm, topp=round(r["duration"], 4),
               trapezoid=round(z["duration"], 4),
               saving_pct=round(100 * (1 - r["duration"] / z["duration"]), 2),
               constant_speed=round(u["duration"], 4))
    print("  note the constant-speed column: it beats the trapezoid by exactly")
    print("  2x every time, because a triangular profile's AVERAGE speed is")
    print("  half its peak.  It is also physically impossible -- it starts and")
    print("  stops instantly.  TOPP beats the honest baseline, not that one.")

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r[1] for r in rows], 0.4, color=COLORS[0], label="TOPP")
    ax.barh(y + 0.2, [r[2] for r in rows], 0.4, color=COLORS[1],
            label="one trapezoid for the whole path")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel("traversal time (s)")
    ax.legend(fontsize=8)
    ax.set_title("The waste is the easy stretches held back by the hard one")
    save(fig, os.path.join(OUT, "vs_global.png"))


# =====================================================================  3
def exp3_which_limit(rng):
    banner("3. Which limit binds where")

    p = CubicPath(np.array(SHAPES["zig-zag"], float))
    r = topp_ra(p, LIM, N=200)
    s, sdot, u = r["s"], r["sdot"], r["u"]
    qd, qdd = p.d1(s), p.d2(s)
    x = sdot ** 2
    vel_r, acc_r = [], []
    for i in range(len(u)):
        vel_r.append(np.max(np.abs(qd[i] * sdot[i]) / VMAX))
        acc_r.append(np.max(np.abs(qd[i] * u[i] + qdd[i] * x[i]) / AMAX))
    vel_r = np.array(vel_r)
    acc_r = np.array(acc_r)
    tol = 0.02
    v_hit = vel_r > 1 - tol
    a_hit = acc_r > 1 - tol
    print(f"  velocity limit active on {100*v_hit.mean():5.1f}% of the path")
    print(f"  acceleration limit active on {100*a_hit.mean():5.1f}%")
    print(f"  at least one active on {100*(v_hit|a_hit).mean():5.1f}%  "
          f"<- theory says 100%: if nothing were saturated somewhere, you "
          f"could go faster there")
    print(f"  both at once on {100*(v_hit&a_hit).mean():5.1f}%")
    record(3, "which_limit", velocity_pct=round(100 * float(v_hit.mean()), 2),
           acceleration_pct=round(100 * float(a_hit.mean()), 2),
           either_pct=round(100 * float((v_hit | a_hit).mean()), 2),
           both_pct=round(100 * float((v_hit & a_hit).mean()), 2))

    for nm, wp in SHAPES.items():
        pp = CubicPath(np.array(wp, float))
        rr = topp_ra(pp, LIM, N=200)
        print(f"  {nm:<28s} saturated {100*active_fraction(pp, rr, LIM):5.1f}% "
              f"of the time")
        record(3, f"saturation_{nm}",
               pct=round(100 * active_fraction(pp, rr, LIM), 2))

    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    ax.plot(s[:-1], vel_r, color=COLORS[0], label="velocity / limit")
    ax.plot(s[:-1], acc_r, color=COLORS[1], label="acceleration / limit")
    ax.axhline(1.0, color=COLORS[2], ls="--", label="the limit")
    ax.set_xlabel("path coordinate s")
    ax.set_ylabel("fraction of limit used")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.set_title("Something is always flat out")
    save(fig, os.path.join(OUT, "which_limit.png"))


# =====================================================================  4
def exp4_short_vs_fast(rng):
    banner("4. The shortest path is not the fastest path")

    env = world_blobs(np.random.default_rng(5))
    S, G = np.array([0.5, 0.5]), np.array([9.5, 9.5])

    def seg(a, b):
        return env.segment_free(a, b, 0.02)

    lim = Limits([1.5, 1.5], [2.0, 2.0])
    print(f"  {'path':<26s} {'length (m)':>11s} {'tightest bend':>14s} "
          f"{'time (s)':>9s}")
    rows = []
    for s in range(8):
        _, raw, st = rrt(env, S, G, np.random.default_rng(s), step=0.5,
                         goal_bias=0.05, max_iters=8000)
        if not st["found"]:
            continue
        sm = shortcut(raw, seg, np.random.default_rng(1), iters=300,
                      spacing=0.25)
        variants = [("raw RRT", np.asarray(resample(raw, 0.25))),
                    ("shortcut", np.asarray(resample(sm, 0.25))),
                    ("shortcut + blended", np.asarray(
                        resample(blend_corners(sm, 0.5, seg), 0.25)))]
        row = {}
        for nm, wp in variants:
            path = CubicPath(wp)
            r = topp_ra(path, lim, N=200)
            row[nm] = (path_cost(wp), min_turn_radius(wp), r["duration"])
        rows.append(row)

    agg = {}
    for nm in ("raw RRT", "shortcut", "shortcut + blended"):
        L = np.mean([r[nm][0] for r in rows])
        R = np.median([r[nm][1] for r in rows])
        T = np.mean([r[nm][2] for r in rows])
        agg[nm] = (L, R, T)
        print(f"  {nm:<26s} {L:11.3f} {R:14.4f} {T:9.3f}")
        record(4, nm, length=round(float(L), 4), tightest_bend=round(float(R), 5),
               time_s=round(float(T), 4))
    a, b = agg["shortcut"], agg["shortcut + blended"]
    print(f"  the blended path is {100*(b[0]/a[0]-1):+.2f}% in LENGTH but "
          f"{100*(b[2]/a[2]-1):+.1f}% in TIME")
    print(f"  shortcutting alone: {100*(a[0]/agg['raw RRT'][0]-1):+.1f}% length,"
          f" {100*(a[2]/agg['raw RRT'][2]-1):+.1f}% time")
    record(4, "blend_vs_shortcut",
           length_pct=round(100 * (b[0] / a[0] - 1), 3),
           time_pct=round(100 * (b[2] / a[2] - 1), 2))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for cx, cy, rr in env.circles:
        axes[0].add_patch(Circle((cx, cy), rr, color="#4A4A4A"))
    _, raw, _ = rrt(env, S, G, np.random.default_rng(0), step=0.5,
                    goal_bias=0.05, max_iters=8000)
    sm = shortcut(raw, seg, np.random.default_rng(1), iters=300, spacing=0.25)
    bl = blend_corners(sm, 0.5, seg)
    for nm, wp, c in (("shortcut", np.asarray(sm), COLORS[1]),
                      ("blended", np.asarray(bl), COLORS[0])):
        axes[0].plot(wp[:, 0], wp[:, 1], color=c, lw=1.8, label=nm)
    axes[0].set_xlim(0, 10)
    axes[0].set_ylim(0, 10)
    axes[0].set_aspect("equal")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].grid(False)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Two paths, nearly the same length")
    for nm, wp, c in (("shortcut", np.asarray(resample(sm, 0.25)), COLORS[1]),
                      ("blended", np.asarray(resample(bl, 0.25)), COLORS[0])):
        path = CubicPath(wp)
        r = topp_ra(path, lim, N=200)
        axes[1].plot(r["t"], r["sdot"] * path.total, color=c,
                     label=f"{nm}: {r['duration']:.2f} s")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("speed along the path (m/s)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("...and very different schedules")
    save(fig, os.path.join(OUT, "short_vs_fast.png"))


# =====================================================================  5
def exp5_resolution(rng):
    banner("5. Grid resolution")

    p = CubicPath(np.array(SHAPES["zig-zag"], float))
    print(f"  {'N':>6s} {'duration (s)':>13s} {'vs N=800':>10s} "
          f"{'worst violation':>16s} {'solve ms':>9s}")
    ref = topp_ra(p, LIM, N=800)["duration"]
    rows = []
    for N in (25, 50, 100, 200, 400, 800):
        t0 = time.perf_counter()
        r = topp_ra(p, LIM, N=N)
        ms = (time.perf_counter() - t0) * 1e3
        v = max(verify(p, r, LIM).values())
        rows.append((N, r["duration"], ms, v))
        print(f"  {N:6d} {r['duration']:13.4f} "
              f"{100*(r['duration']/ref-1):9.2f}% {v:16.4f} {ms:9.0f}")
        record(5, f"N_{N}", duration=round(r["duration"], 5),
               vs_ref_pct=round(100 * (r["duration"] / ref - 1), 3),
               worst_ratio=round(v, 5), solve_ms=round(ms, 1))
    print("  Coarser grids report a SHORTER time than they can actually")
    print("  deliver: they simply do not look between the grid points, so")
    print("  they never see the corner that would have slowed them down.")

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0])
    ax.axhline(ref, color=COLORS[2], ls="--", label="N = 800")
    ax.set_xscale("log")
    ax.set_xlabel("grid points N")
    ax.set_ylabel("reported duration (s)")
    ax.legend(fontsize=8)
    ax.set_title("Convergence, from below")
    save(fig, os.path.join(OUT, "resolution.png"))


# =====================================================================  6
def exp6_torque(rng):
    banner("6. Torque limits instead of acceleration limits")

    urdf = os.path.join(_PROJ, "10-inverse-dynamics-from-scratch", "models",
                        "arm2.urdf")
    model = dyn.Model(urdf)
    print(f"  2-link arm from project 10: torque limits {model.tau_max} Nm, "
          f"joint speed limits {model.qd_max} rad/s")

    wp = np.array([[-1.0, -1.2], [-0.3, -0.6], [0.4, -1.1], [1.0, -0.4]])
    p = CubicPath(wp)
    tl = TorqueLimits(model, model.tau_max, model.qd_max, dyn)

    r_t = topp_ra(p, Limits(model.qd_max, np.ones(2)), N=150, torque=tl)
    print(f"  time-optimal under TORQUE limits: {r_t['duration']:.3f} s")
    record(6, "torque_limited", duration=round(r_t["duration"], 4))

    # What a constant acceleration bound gives, calibrated at one pose.  For
    # joint j on its own, tau_j = M_jj * qddot_j, so the acceleration the motor
    # can produce there is tau_max_j / M_jj.  This is exactly the calculation
    # someone does when they are asked for "the acceleration limit" of an arm.
    q_mid = p.eval(np.array([0.5]))[0]
    M = dyn.mass_matrix(model, q_mid)
    a_equiv = model.tau_max / np.diag(M)
    g_mid = dyn.gravity_torque(model, q_mid)
    print(f"  the same torques at the mid-path pose allow "
          f"{np.round(a_equiv, 1)} rad/s^2 -- so a fixed acceleration bound "
          f"'equivalent' to the torque bound would be that")
    print(f"  but gravity alone already costs {np.round(np.abs(g_mid), 2)} Nm "
          f"there, {np.round(100*np.abs(g_mid)/model.tau_max, 1)}% of the "
          f"budget, before any acceleration is asked for")
    record(6, "gravity_share_pct",
           values=list(np.round(100 * np.abs(g_mid) / model.tau_max, 2)))
    r_a = topp_ra(p, Limits(model.qd_max, a_equiv), N=150)
    print(f"  time-optimal under that FIXED acceleration bound: "
          f"{r_a['duration']:.3f} s")
    v = verify(p, r_a, Limits(model.qd_max, a_equiv), torque=tl, model=model,
               dyn=dyn)
    print(f"  ...but replaying it and computing the real torques gives a worst "
          f"case of {v['torque_ratio']:.2f}x the motor limit")
    record(6, "fixed_accel_bound", a_equiv=list(np.round(a_equiv, 3)),
           duration=round(r_a["duration"], 4),
           torque_ratio=round(v["torque_ratio"], 4))
    if v["torque_ratio"] > 1.0:
        print("  -> the fixed bound is NOT conservative: gravity and the arm's "
              "changing inertia make the same acceleration cost different "
              "torques at different poses.")
    else:
        print("  -> on this path the fixed bound happened to stay legal, but "
              "it is legal by luck, not by construction.")

    # how much the available acceleration varies along the path
    ss = np.linspace(0, 1, 60)
    qs = p.eval(ss)
    avail = np.array([model.tau_max / np.diag(dyn.mass_matrix(model, q))
                      for q in qs])
    grav = np.array([np.abs(dyn.gravity_torque(model, q)) for q in qs])
    for j in range(2):
        print(f"  joint {j+1}: available acceleration {avail[:, j].min():7.1f} "
              f"to {avail[:, j].max():7.1f} rad/s^2 "
              f"({avail[:, j].max()/max(avail[:, j].min(),1e-9):.2f}x); "
              f"gravity torque {grav[:, j].min():.2f} to {grav[:, j].max():.2f} Nm")
        record(6, f"joint{j+1}_variation",
               a_min=round(float(avail[:, j].min()), 3),
               a_max=round(float(avail[:, j].max()), 3),
               g_min=round(float(grav[:, j].min()), 3),
               g_max=round(float(grav[:, j].max()), 3))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot(ss, avail[:, 0], color=COLORS[0], label="joint 1")
    axes[0].plot(ss, avail[:, 1], color=COLORS[1], label="joint 2")
    axes[0].set_xlabel("path coordinate s")
    axes[0].set_ylabel("available acceleration (rad/s^2)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("The 'acceleration limit' is not a constant")
    axes[1].plot(r_t["t"], r_t["sdot"], color=COLORS[0], label="torque limits")
    axes[1].plot(r_a["t"], r_a["sdot"], color=COLORS[1],
                 label="fixed acceleration bound")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("sdot")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Two different schedules for one path")
    save(fig, os.path.join(OUT, "torque.png"))


# =====================================================================  7
def exp7_verify(rng):
    banner("7. Verification, and the cost of shaving 10% off")

    print(f"  {'path':<28s} {'vel ratio':>10s} {'acc ratio':>10s}")
    for nm, wp in SHAPES.items():
        p = CubicPath(np.array(wp, float))
        r = topp_ra(p, LIM, N=200)
        v = verify(p, r, LIM)
        print(f"  {nm:<28s} {v['vel_ratio']:10.4f} {v['acc_ratio']:10.4f}")
        record(7, f"verify_{nm}", vel_ratio=round(v["vel_ratio"], 5),
               acc_ratio=round(v["acc_ratio"], 5))

    p = CubicPath(np.array(SHAPES["zig-zag"], float))
    r = topp_ra(p, LIM, N=200)
    print(f"\n  now speed the SAME trajectory up by a factor k "
          f"(time scaling: speeds x k, accelerations x k^2)")
    print(f"  {'k':>6s} {'duration (s)':>13s} {'vel ratio':>10s} "
          f"{'acc ratio':>10s} {'legal?':>7s}")
    rows = []
    for k in (0.9, 1.0, 1.05, 1.1, 1.25, 1.5):
        v = verify(p, r, LIM)
        vr, ar = v["vel_ratio"] * k, v["acc_ratio"] * k * k
        legal = vr <= 1.0001 and ar <= 1.0001
        rows.append((k, r["duration"] / k, vr, ar, legal))
        print(f"  {k:6.2f} {r['duration']/k:13.3f} {vr:10.4f} {ar:10.4f} "
              f"{str(legal):>7s}")
        record(7, f"speedup_{k}", duration=round(r["duration"] / k, 4),
               vel_ratio=round(vr, 5), acc_ratio=round(ar, 5), legal=legal)
    print("  A 10% speed-up asks for 21% more acceleration, because")
    print("  acceleration scales with the SQUARE of the time scaling.  That is")
    print("  why 'just run it a bit faster' is a bigger request than it sounds.")

    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ax.plot([r0[0] for r0 in rows], [r0[2] for r0 in rows], "o-",
            color=COLORS[0], label="velocity used")
    ax.plot([r0[0] for r0 in rows], [r0[3] for r0 in rows], "s-",
            color=COLORS[1], label="acceleration used")
    ax.axhline(1.0, color=COLORS[2], ls="--", label="the limit")
    ax.set_xlabel("time-scaling factor k (k > 1 = faster)")
    ax.set_ylabel("fraction of limit")
    ax.legend(fontsize=8)
    ax.set_title("Speed scales linearly; acceleration scales as k^2")
    save(fig, os.path.join(OUT, "speedup.png"))


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_profile(rng)
    exp2_vs_global(rng)
    exp3_which_limit(rng)
    exp4_short_vs_fast(rng)
    exp5_resolution(rng)
    exp6_torque(rng)
    exp7_verify(rng)

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
