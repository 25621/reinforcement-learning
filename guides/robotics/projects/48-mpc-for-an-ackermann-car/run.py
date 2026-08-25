"""Project 48 -- kinematic-bicycle MPC on a racetrack, and where the model breaks.

Six experiments:
  1. one lap
  2. how far ahead should the horizon look
  3. the model the controller believes vs the car it is driving
  4. MPC vs pure pursuit, and what the extra compute buys
  5. the track-boundary constraint
  6. the solver takes time, and the car keeps moving while it thinks
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

import car                                                      # noqa: E402
from car import (MPC, Track, dyn_step_np, dyn_to_kin, kin_step_np,  # noqa: E402
                 kin_to_dyn, make_ref, pure_pursuit_steer)
from plot_style import COLORS, use_style, save                  # noqa: E402

import matplotlib.pyplot as plt                                 # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []
MPC_CACHE = {}


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


def get_mpc(**kw):
    key = tuple(sorted(kw.items()))
    if key not in MPC_CACHE:
        MPC_CACHE[key] = MPC(**kw)
    MPC_CACHE[key].reset()
    return MPC_CACHE[key]


# ------------------------------------------------------------------ driver
def lap(track, v_prof, controller="mpc", plant="dyn", N=20, dt=0.1,
        half_width=None, delay=0, compensate=False, laps=1.0, mu=car.MU,
        Ld=None, t_max=90.0, a_min=car.A_MIN, soft=False,
        d_rate=car.DELTA_RATE):
    """Drive `laps` laps and report what happened."""
    j0 = 0
    p0 = track.pts[0]
    xk = np.array([p0[0], p0[1], track.psi[0], float(v_prof[0]), 0.0])
    x = kin_to_dyn(xk) if plant == "dyn" else xk
    ctrl = (get_mpc(N=N, dt=dt, half_width=half_width, a_min=a_min,
                    soft=soft, d_rate=d_rate)
            if controller == "mpc" else None)

    u_prev = np.zeros(2)
    hist = []
    buf = [np.array(x) for _ in range(delay + 1)]
    s_travel, t, off, hint, n_fail, n_call = 0.0, 0.0, 0, 0, 0, 0
    prev_xy = np.array(x[:2])
    solve_t = []

    while t < t_max and s_travel < track.length * laps:
        xs = buf[0]                              # the state the controller sees
        xk_seen = dyn_to_kin(xs) if plant == "dyn" else xs
        if compensate and delay > 0:
            # Roll the model forward by the delay, so the controller plans
            # from where the car WILL be, not where it was.  Costs one model
            # evaluation and is the standard fix.
            for _ in range(delay):
                xk_seen = kin_step_np(xk_seen, u_prev, dt)
        hint, lat = track.project(xk_seen[:2], hint=hint)
        s_here = track.s[hint]
        t0 = time.perf_counter()
        if controller == "mpc":
            ref = make_ref(track, s_here, v_prof, N, dt)
            u, _ = ctrl(xk_seen, ref, u_prev)
            n_call += 1
            n_fail += int(not ctrl.last_ok)
        else:
            # Pure pursuit for steering + a P controller on the speed profile.
            v_ref = float(np.interp(np.mod(s_here, track.length), track.s,
                                    v_prof, period=track.length))
            d_tgt = pure_pursuit_steer(xk_seen, track,
                                       Ld if Ld else max(4.0, 0.6 * xk_seen[3]),
                                       hint=hint)
            u = np.array([np.clip(2.0 * (v_ref - xk_seen[3]), car.A_MIN, car.A_MAX),
                          np.clip((d_tgt - xk_seen[4]) / dt,
                                  -car.DELTA_RATE, car.DELTA_RATE)])
        solve_t.append((time.perf_counter() - t0) * 1000.0)

        if plant == "dyn":
            x = dyn_step_np(x, u, dt, mu=mu)
        else:
            x = kin_step_np(x, u, dt)
        buf.append(np.array(x))
        buf = buf[-(delay + 1):]
        u_prev = u
        s_travel += float(np.linalg.norm(x[:2] - prev_xy))
        prev_xy = np.array(x[:2])
        t += dt
        hint2, lat_true = track.project(x[:2], hint=hint)
        speed = x[3] if plant == "dyn" else x[3]
        hist.append((t, x[0], x[1], speed, lat_true,
                     x[6] if plant == "dyn" else x[4], u[0]))
        off += int(abs(lat_true) > track.half_width)
        if abs(lat_true) > 4 * track.half_width:
            break

    h = np.asarray(hist)
    off_frac = off / max(len(h), 1)
    # A lap only counts if the car covered the distance AND stayed between the
    # kerbs.  Distance alone would score a car that cut straight across the
    # infield as a finisher, which is how a metric quietly stops measuring
    # what it is named after.
    covered = s_travel >= track.length * laps - 2.0
    finished = bool(covered and off_frac < 1e-9)
    return dict(hist=h, lap_time=t if finished else float("nan"),
                finished=finished, covered=covered,
                mean_abs_lat=float(np.mean(np.abs(h[:, 4]))),
                max_abs_lat=float(np.max(np.abs(h[:, 4]))),
                off_frac=off_frac,
                mean_speed=float(np.mean(h[:, 3])),
                ms_per_step=float(np.mean(solve_t)),
                solver_fail_frac=n_fail / max(n_call, 1),
                max_ms=float(np.max(solve_t)))


def draw_track(ax, track):
    n = np.column_stack([-np.sin(track.psi), np.cos(track.psi)])
    for sgn, c in [(1, "0.6"), (-1, "0.6")]:
        b = track.pts + sgn * track.half_width * n
        ax.plot(np.append(b[:, 0], b[0, 0]), np.append(b[:, 1], b[0, 1]),
                color=c, lw=1.0)
    ax.plot(np.append(track.pts[:, 0], track.pts[0, 0]),
            np.append(track.pts[:, 1], track.pts[0, 1]), "--", color="0.75",
            lw=0.8)
    ax.set_aspect("equal"); ax.grid(False)


# ================================================================= 1. one lap
def exp1(track, vp):
    print("[1] one lap")
    r = lap(track, vp, N=20)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    draw_track(axes[0], track)
    sc = axes[0].scatter(r["hist"][:, 1], r["hist"][:, 2], c=r["hist"][:, 3],
                         s=7, cmap="viridis")
    fig.colorbar(sc, ax=axes[0], label="speed [m/s]")
    axes[0].set_title(f"one lap, {r['lap_time']:.1f} s")
    axes[1].plot(r["hist"][:, 0], r["hist"][:, 3], color=COLORS[0],
                 label="actual")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("speed [m/s]")
    axes[1].set_title("speed follows the curvature profile")
    ax2 = axes[1].twinx()
    ax2.plot(r["hist"][:, 0], np.degrees(r["hist"][:, 5]), color=COLORS[1],
             lw=1.0)
    ax2.set_ylabel("steering [deg]", color=COLORS[1]); ax2.grid(False)
    axes[2].plot(r["hist"][:, 0], r["hist"][:, 4], color=COLORS[2])
    axes[2].axhline(track.half_width, color=COLORS[1], ls=":")
    axes[2].axhline(-track.half_width, color=COLORS[1], ls=":")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("lateral offset [m]")
    axes[2].set_title("dotted = track edge")
    save(fig, os.path.join(OUT, "overview.png"))
    rec("1_lap", achieved_speed_range=round(float(r["hist"][:, 3].max()
                                                  - r["hist"][:, 3].min()), 2),
        lap_time=round(r["lap_time"], 2),
        mean_abs_lat=round(r["mean_abs_lat"], 3),
        max_abs_lat=round(r["max_abs_lat"], 3),
        mean_speed=round(r["mean_speed"], 2),
        ms_per_step=round(r["ms_per_step"], 1),
        max_ms=round(r["max_ms"], 1))


# ================================================= 2. horizon
def exp2(track, vp):
    print("[2] horizon length vs braking power")
    Ns = [3, 5, 8, 12, 20, 35]
    # The speed profile drops abruptly at every corner entry (it is computed
    # point by point, with no backward smoothing).  So the MPC has to do the
    # braking itself, and the horizon is exactly the warning it gets.  The
    # prediction: the horizon must cover the braking TIME, dv / a_brake.
    dv = float(np.max(vp) - np.min(vp))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    for a_brake, c in zip([-3.0, -6.0, -12.0], COLORS):
        lats, offs, ms, need = [], [], [], dv / abs(a_brake)
        for N in Ns:
            r = lap(track, vp, N=N, a_min=a_brake)
            lats.append(r["max_abs_lat"]); ms.append(r["ms_per_step"])
            offs.append(r["off_frac"])
            rec("2_horizon", a_brake=a_brake, N=N, horizon_s=round(N * 0.1, 1),
                braking_time_needed_s=round(need, 2),
                lap_time=round(r["lap_time"], 2) if r["finished"] else None,
                mean_abs_lat=round(r["mean_abs_lat"], 3),
                max_abs_lat=round(r["max_abs_lat"], 3),
                off_track_frac=round(r["off_frac"], 3),
                ms_per_step=round(r["ms_per_step"], 1),
                finished=int(r["finished"]))
        lab = f"brake {abs(a_brake):.0f} m/s^2"
        axes[0].plot(np.array(Ns) * 0.1, lats, "o-", color=c, label=lab)
        axes[0].axvline(need, color=c, ls=":", lw=1.0)
        axes[1].plot(np.array(Ns) * 0.1, offs, "o-", color=c, label=lab)
        axes[1].axvline(need, color=c, ls=":", lw=1.0)
        axes[2].plot(np.array(Ns) * 0.1, ms, "o-", color=c, label=lab)
    axes[0].axhline(track.half_width, color="0.5", ls="--", lw=0.8)
    axes[0].set_ylabel("max |lateral offset| [m]"); axes[0].set_yscale("log")
    axes[0].set_title("dotted = predicted braking time dv / a")
    axes[1].set_ylabel("fraction of the lap off the track")
    axes[2].set_ylabel("solve time [ms]"); axes[2].set_yscale("log")
    axes[2].axhline(100.0, color="0.4", ls=":", label="100 ms budget")
    for ax in axes:
        ax.set_xlabel("horizon [s]"); ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "horizon.png"))
    rec("2_prediction", profile_speed_drop=round(dv, 2),
        need_at_3=round(dv / 3, 2), need_at_6=round(dv / 6, 2),
        need_at_12=round(dv / 12, 2))
    # If braking room is not what sets the threshold, maybe steering speed is.
    # Same sweep, same brakes, only the steering rate limit changes.
    for d_rate in [0.4, 1.2, 4.0]:
        for N in [3, 5, 8, 12]:
            rr = lap(track, vp, N=N, d_rate=d_rate)
            rec("2_steer_rate", d_rate=d_rate, N=N, horizon_s=round(N * .1, 1),
                off_track_frac=round(rr["off_frac"], 3),
                max_abs_lat=round(rr["max_abs_lat"], 3),
                finished=int(rr["finished"]))


# ================================================= 3. model mismatch
def exp3(track):
    print("[3] the model the controller believes vs the car it drives")
    a_lats = [3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for plant, c, lab in [("kin", COLORS[0], "kinematic plant (no tyres)"),
                          ("dyn", COLORS[1], "dynamic plant (real tyres)")]:
        maxe, offs, times = [], [], []
        for al in a_lats:
            vp = track.speed_profile(a_lat=al, v_max=20.0)
            r = lap(track, vp, N=20, plant=plant)
            maxe.append(r["max_abs_lat"]); offs.append(r["off_frac"])
            times.append(r["lap_time"])
            rec("3_mismatch", plant=plant, a_lat_target=al,
                lap_time=round(r["lap_time"], 2) if r["finished"] else None,
                max_abs_lat=round(r["max_abs_lat"], 3),
                off_track_frac=round(r["off_frac"], 3),
                mean_speed=round(r["mean_speed"], 2),
                finished=int(r["finished"]))
        axes[0].plot(a_lats, maxe, "o-", color=c, label=lab)
        axes[1].plot(a_lats, offs, "o-", color=c, label=lab)
        axes[2].plot(a_lats, times, "o-", color=c, label=lab)
    limit = car.MU * car.G
    for ax, lab in [(axes[0], "max |lateral offset| [m]"),
                    (axes[1], "fraction of the lap off the track"),
                    (axes[2], "lap time [s]")]:
        ax.axvline(limit, color=COLORS[3], ls=":",
                   label=f"mu*g = {limit:.1f}")
        ax.set_xlabel("lateral acceleration the profile asks for [m/s^2]")
        ax.set_title(lab); ax.legend(fontsize=7)
    axes[0].axhline(track.half_width, color="0.5", ls="--", lw=0.8)
    save(fig, os.path.join(OUT, "mismatch.png"))


# ================================================= 4. MPC vs pure pursuit
def exp4(track):
    print("[4] MPC vs pure pursuit")
    a_lats = [3.0, 4.5, 6.0, 7.5, 9.0]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    for ctrl, c, lab in [("mpc", COLORS[0], "MPC (N = 20)"),
                         ("pp", COLORS[1], "pure pursuit + speed P")]:
        maxe, times, ms = [], [], []
        for al in a_lats:
            vp = track.speed_profile(a_lat=al, v_max=20.0)
            r = lap(track, vp, N=20, controller=ctrl, plant="dyn")
            maxe.append(r["max_abs_lat"]); times.append(r["lap_time"])
            ms.append(r["ms_per_step"])
            rec("4_vs_pp", controller=lab, a_lat_target=al,
                lap_time=round(r["lap_time"], 2) if r["finished"] else None,
                mean_abs_lat=round(r["mean_abs_lat"], 3),
                max_abs_lat=round(r["max_abs_lat"], 3),
                off_track_frac=round(r["off_frac"], 3),
                ms_per_step=round(r["ms_per_step"], 3))
        axes[0].plot(a_lats, maxe, "o-", color=c, label=lab)
        axes[1].plot(a_lats, times, "o-", color=c, label=lab)
        axes[2].bar([lab], [np.mean(ms)], color=c)
    axes[0].set_ylabel("max |lateral offset| [m]")
    axes[0].axhline(track.half_width, color="0.5", ls="--", lw=0.8)
    axes[1].set_ylabel("lap time [s]")
    axes[2].set_ylabel("mean solve time [ms]"); axes[2].set_yscale("log")
    for ax in axes[:2]:
        ax.set_xlabel("a_lat target [m/s^2]"); ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "vs_pp.png"))

    # The comparison above is on a 8 m wide track, where nothing constrains
    # the car and the MPC's machinery has nothing to do.  Narrow the track
    # to 2.4 m and the picture should change: pure pursuit has no concept of
    # an edge, while a soft-constrained MPC is told exactly where it is.
    narrow = Track(half_width=1.2)
    vpn = narrow.speed_profile(a_lat=6.0, v_max=20.0)
    for ctrl, soft, hw, lab in [("mpc", False, None, "MPC, no bounds"),
                                ("mpc", True, 1.0, "MPC, soft bounds"),
                                ("pp", False, None, "pure pursuit")]:
        r_ = lap(narrow, vpn, N=20, controller=ctrl, soft=soft,
                 half_width=hw, plant="dyn")
        rec("4_narrow", track_half_width=1.2, controller=lab,
            lap_time=round(r_["lap_time"], 2) if r_["finished"] else None,
            mean_abs_lat=round(r_["mean_abs_lat"], 3),
            max_abs_lat=round(r_["max_abs_lat"], 3),
            off_track_frac=round(r_["off_frac"], 3),
            ms_per_step=round(r_["ms_per_step"], 2))


# ================================================= 5. the boundary constraint
def exp5(track):
    print("[5] the track-boundary constraint: none vs hard vs soft")
    hw = track.half_width * 0.85
    arms = [(None, False, COLORS[2], "no constraint"),
            (hw, False, COLORS[1], "hard constraint"),
            (hw, True, COLORS[0], "soft (penalised) constraint")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8),
                             gridspec_kw={"width_ratios": [1.2, 1, 1]})
    draw_track(axes[0], track)
    a_lats = [6.0, 7.5, 9.0, 11.0]
    for h, soft, c, lab in arms:
        offs, ms, fails = [], [], []
        for al in a_lats:
            vp = track.speed_profile(a_lat=al, v_max=20.0)
            r = lap(track, vp, N=20, half_width=h, soft=soft, plant="dyn")
            offs.append(r["off_frac"]); ms.append(r["ms_per_step"])
            fails.append(r["solver_fail_frac"])
            rec("5_bounds", constraint=lab, a_lat_target=al,
                lap_time=round(r["lap_time"], 2) if r["finished"] else None,
                max_abs_lat=round(r["max_abs_lat"], 3),
                off_track_frac=round(r["off_frac"], 3),
                solver_fail_frac=round(r["solver_fail_frac"], 3),
                ms_per_step=round(r["ms_per_step"], 1))
            if al == 9.0:
                axes[0].plot(r["hist"][:, 1], r["hist"][:, 2], color=c, lw=1.5,
                             label=lab)
        axes[1].plot(a_lats, offs, "o-", color=c, label=lab)
        axes[2].plot(a_lats, ms, "o-", color=c, label=lab)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title("a_lat 9 m/s^2 -- near what the tyres can give")
    axes[1].set_ylabel("fraction of the lap off the track")
    axes[2].set_ylabel("solve time [ms]")
    axes[2].axhline(100.0, color="0.4", ls=":", label="100 ms budget")
    for ax in axes[1:]:
        ax.set_xlabel("a_lat target [m/s^2]"); ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "bounds.png"))


# ================================================= 6. latency
def exp6(track):
    print("[6] latency, and what compensation is worth")
    vp = track.speed_profile(a_lat=6.0, v_max=20.0)
    delays = [0, 1, 2, 3]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharey=True)
    # Run it on BOTH plants.  Compensation rolls the state forward with the
    # controller's own kinematic model, so on the kinematic plant the
    # prediction is exact and compensation should be nearly free.  On the
    # dynamic plant the same prediction is wrong in the same way the
    # controller is wrong, which is the interesting case.
    for ax, plant, title in [(axes[0], "kin", "kinematic plant (model exact)"),
                             (axes[1], "dyn", "dynamic plant (model wrong)")]:
        for comp, c, lab in [(False, COLORS[1], "uses the stale state"),
                             (True, COLORS[0], "rolls the state forward")]:
            lats = []
            for d in delays:
                r_ = lap(track, vp, N=20, delay=d, compensate=comp,
                         plant=plant)
                lats.append(r_["mean_abs_lat"])
                rec("6_latency", plant=plant, compensated=int(comp),
                    delay_steps=d, delay_ms=d * 100,
                    lap_time=round(r_["lap_time"], 2) if r_["finished"] else None,
                    mean_abs_lat=round(r_["mean_abs_lat"], 3),
                    max_abs_lat=round(r_["max_abs_lat"], 3),
                    off_track_frac=round(r_["off_frac"], 3),
                    finished=int(r_["finished"]))
            ax.plot([d * 100 for d in delays], lats, "o-", color=c, label=lab)
        ax.set_xlabel("measurement delay [ms]"); ax.set_title(title)
        ax.set_yscale("log"); ax.legend(fontsize=7)
    axes[0].set_ylabel("mean |lateral offset| [m]")
    save(fig, os.path.join(OUT, "latency.png"))


if __name__ == "__main__":
    t0 = time.time()
    track = Track()
    vp = track.speed_profile(a_lat=6.0, v_max=20.0)
    print(f"track {track.length:.0f} m, tightest radius "
          f"{1 / np.abs(track.kappa).max():.1f} m")
    exp1(track, vp)
    exp2(track, vp)
    exp3(track)
    exp4(track)
    exp5(track)
    exp6(track)
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(ROWS)
    print(f"done in {time.time() - t0:.1f}s")
