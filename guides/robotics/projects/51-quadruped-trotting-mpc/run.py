"""Project 51 -- a convex-MPC trotting controller on a simulated quadruped.

Five experiments:
  1. one trot
  2. how far ahead the MPC has to look
  3. the friction cone, and what happens on ice
  4. is the MPC earning its keep? (a fixed-force controller as the control)
  5. gaits, and a push test
"""

import csv
import math
import os
import sys
import time

import mujoco
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from mpc import ConvexMPC, stance_torque, swing_torque             # noqa: E402
from quadruped import (HIP, LEGS, STAND_H, Gait, Robot, leg_ik,    # noqa: E402
                       L_ABD, raibert_step, swing_traj)
from plot_style import COLORS, use_style, save                     # noqa: E402

import matplotlib.pyplot as plt                                    # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []
MPC_DT = 0.03


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


def trot(v_des=(0.5, 0.0), t_max=5.0, N=10, mu_ctrl=0.6, friction=0.9,
         cone=True, gait=None, controller="mpc", payload=0.0, terrain=None,
         push=None, seed=0, record=False):
    """Walk for `t_max` seconds and report how well it went."""
    rb = Robot(friction=friction, payload=payload, terrain=terrain, seed=seed)
    gait = gait or Gait()
    v_des = np.asarray([v_des[0], v_des[1], 0.0], float)
    m = ConvexMPC(N=N, dt=MPC_DT, mass=rb.total_mass, mu=mu_ctrl,
                  friction_cone=cone)

    n_steps = int(t_max / rb.dt)
    every = max(int(round(MPC_DT / rb.dt)), 1)
    f_cmd = np.zeros((4, 3))
    swing_from = rb.foot_pos()
    swing_to = swing_from.copy()
    was_stance = gait.in_stance(0.0)
    log = {"t": [], "v": [], "p": [], "rpy": [], "f": [], "st": [],
           "slip": [], "tau": []}
    solve_ms, n_solve = 0.0, 0
    fell_at = None
    p_ref = rb.p.copy()

    for k in range(n_steps):
        t = k * rb.dt
        st = gait.in_stance(t)
        if push is not None and 0.0 <= (t - push[0]) < rb.dt:
            rb.data.qvel[0:3] += np.asarray(push[1:], float)

        # New swing legs get a fresh foothold from the Raibert rule.
        for i in range(4):
            if was_stance[i] and not st[i]:
                swing_from[i] = rb.foot_pos()[i]
                hip_w = rb.p + rb.R @ HIP[i]
                swing_to[i] = raibert_step(rb.v, v_des, 0.0, hip_w,
                                           gait.period * gait.duty)
        was_stance = st

        if k % every == 0:
            p_ref = p_ref + v_des * MPC_DT
            p_ref[2] = STAND_H
            x0 = np.concatenate([rb.rpy, rb.p, rb.w, rb.v, [-9.81]])
            xr = np.zeros((m.N, 13))
            for j in range(m.N):
                xr[j, 3:6] = p_ref + v_des * MPC_DT * (j + 1)
                xr[j, 5] = STAND_H
                xr[j, 9:12] = v_des
                xr[j, 12] = -9.81
            ct = np.array([gait.in_stance(t + MPC_DT * (j + 1))
                           for j in range(m.N)])
            r_feet = rb.foot_pos() - rb.p
            t0 = time.perf_counter()
            if controller == "mpc":
                f_cmd = m.solve(x0, xr, r_feet, ct)
            else:
                # The control: share the body weight equally among the feet
                # that are down, straight up, with no optimisation at all.
                ns = max(int(st.sum()), 1)
                f_cmd = np.zeros((4, 3))
                for i in range(4):
                    if st[i]:
                        f_cmd[i] = [0.0, 0.0, rb.total_mass * 9.81 / ns]
            solve_ms += (time.perf_counter() - t0) * 1000.0
            n_solve += 1

        tau = np.zeros(12)
        sw = gait.swing_frac(t)
        for i in range(4):
            J = rb.foot_jac(i)
            if st[i]:
                tau[3 * i:3 * i + 3] = stance_torque(J, f_cmd[i])
            else:
                p_des = swing_traj(swing_from[i], swing_to[i], sw[i])
                hip_w = rb.p + rb.R @ HIP[i]
                local = rb.R.T @ (p_des - hip_w)
                side = -1.0 if LEGS[i].endswith("R") else 1.0
                q_des = leg_ik(local, side)
                q = rb.data.qpos[7 + 3 * i:10 + 3 * i]
                dq = rb.data.qvel[6 + 3 * i:9 + 3 * i]
                tau[3 * i:3 * i + 3] = swing_torque(q, dq, q_des)
        rb.step(tau)

        if k % 5 == 0:
            fp = rb.foot_pos()
            log["t"].append(t); log["v"].append(rb.v.copy())
            log["p"].append(rb.p.copy()); log["rpy"].append(rb.rpy.copy())
            log["f"].append(f_cmd.copy()); log["st"].append(st.copy())
            log["tau"].append(np.abs(tau).max())
        if rb.fallen():
            fell_at = t
            break

    for key in ("t", "v", "p", "rpy", "f", "st", "tau"):
        log[key] = np.asarray(log[key])
    survived = fell_at is None
    # Only judge tracking once the gait has started; the first half second is
    # the robot getting off the spot, not the controller's steady behaviour.
    warm = log["t"] > 1.0
    return dict(log=log, survived=survived, fell_at=fell_at,
                vx_err=float(np.mean(np.abs(log["v"][warm, 0] - v_des[0])))
                if warm.any() else float("nan"),
                vy_err=float(np.mean(np.abs(log["v"][warm, 1] - v_des[1])))
                if warm.any() else float("nan"),
                mean_vx=float(np.mean(log["v"][warm, 0])) if warm.any() else 0.0,
                height_err=float(np.mean(np.abs(log["p"][warm, 2] - STAND_H)))
                if warm.any() else float("nan"),
                rp_rms=float(np.sqrt(np.mean(log["rpy"][warm, :2] ** 2)))
                if warm.any() else float("nan"),
                peak_tau=float(np.max(log["tau"])) if len(log["tau"]) else 0.0,
                distance=float(log["p"][-1, 0] - log["p"][0, 0])
                if len(log["p"]) else 0.0,
                ms_per_solve=solve_ms / max(n_solve, 1))


# ================================================================= 1. a trot
def exp1():
    print("[1] one trot")
    r = trot(v_des=(0.6, 0.0), t_max=6.0)
    lg = r["log"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.0))
    axes[0, 0].plot(lg["t"], lg["v"][:, 0], color=COLORS[0], label="v_x")
    axes[0, 0].plot(lg["t"], lg["v"][:, 1], color=COLORS[1], label="v_y")
    axes[0, 0].axhline(0.6, color="0.5", ls=":", label="commanded")
    axes[0, 0].set_ylabel("body velocity [m/s]"); axes[0, 0].legend(fontsize=7)
    axes[0, 1].plot(lg["t"], lg["p"][:, 2], color=COLORS[2])
    axes[0, 1].axhline(STAND_H, color="0.5", ls=":")
    axes[0, 1].set_ylabel("body height [m]")
    axes[1, 0].plot(lg["t"], np.degrees(lg["rpy"][:, 0]), color=COLORS[0],
                    label="roll")
    axes[1, 0].plot(lg["t"], np.degrees(lg["rpy"][:, 1]), color=COLORS[1],
                    label="pitch")
    axes[1, 0].set_ylabel("body attitude [deg]"); axes[1, 0].legend(fontsize=7)
    sel = lg["t"] < 2.0
    for i, n in enumerate(LEGS):
        axes[1, 1].plot(lg["t"][sel], lg["f"][sel, i, 2],
                        color=COLORS[i], lw=1.2, label=f"{n} f_z")
        axes[1, 1].fill_between(lg["t"][sel], -20 - 8 * i, -12 - 8 * i,
                                where=lg["st"][sel, i], color=COLORS[i],
                                alpha=0.5, step="mid")
    axes[1, 1].set_ylabel("vertical force [N]  /  contact bars")
    axes[1, 1].legend(fontsize=6, ncol=2)
    for ax in axes.ravel():
        ax.set_xlabel("t [s]")
    save(fig, os.path.join(OUT, "trot.png"))
    rec("1_trot", commanded_vx=0.6, mean_vx=round(r["mean_vx"], 3),
        vx_err=round(r["vx_err"], 3), vy_err=round(r["vy_err"], 3),
        height_err_mm=round(r["height_err"] * 1000, 1),
        roll_pitch_rms_deg=round(math.degrees(r["rp_rms"]), 2),
        peak_torque_Nm=round(r["peak_tau"], 1),
        ms_per_solve=round(r["ms_per_solve"], 2),
        survived=int(r["survived"]))


# ================================================= 2. horizon
def exp2():
    print("[2] horizon length")
    Ns = [1, 2, 3, 5, 8, 12, 16, 20]
    out = []
    for N in Ns:
        rs = [trot(v_des=(0.6, 0.0), t_max=5.0, N=N, seed=s) for s in (0,)]
        r = rs[0]
        out.append((N, r["vx_err"], r["rp_rms"], r["ms_per_solve"],
                    r["survived"]))
        rec("2_horizon", N=N, horizon_s=round(N * MPC_DT, 2),
            gait_cycles=round(N * MPC_DT / 0.34, 2),
            vx_err=round(r["vx_err"], 3),
            roll_pitch_rms_deg=round(math.degrees(r["rp_rms"]), 2),
            ms_per_solve=round(r["ms_per_solve"], 2),
            survived=int(r["survived"]))
    out = np.asarray(out, float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(out[:, 0] * MPC_DT, out[:, 1], "o-", color=COLORS[0])
    axes[0].set_ylabel("mean |v_x error| [m/s]")
    axes[1].plot(out[:, 0] * MPC_DT, np.degrees(out[:, 2]), "o-",
                 color=COLORS[1])
    axes[1].set_ylabel("roll/pitch RMS [deg]")
    axes[2].plot(out[:, 0] * MPC_DT, out[:, 3], "o-", color=COLORS[2])
    axes[2].set_ylabel("ms per solve"); axes[2].set_yscale("log")
    axes[2].axhline(MPC_DT * 1000, color="0.4", ls=":",
                    label="the 30 ms control period")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.set_xlabel("horizon [s]")
        ax.axvline(0.34, color="0.6", ls="--", lw=0.8)
    axes[0].set_title("dashed line = one gait cycle")
    save(fig, os.path.join(OUT, "horizon.png"))


# ================================================= 3. friction
def exp3():
    print("[3] friction: the ground's, and the controller's belief about it")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
    # Panel A: vary the GROUND, hold the controller's assumption fixed and
    # conservative.  Panel B: vary what the CONTROLLER believes, hold the
    # ground fixed.  Sweeping both together (which is the tempting thing to
    # do) cannot tell you which one caused the failure.
    mus = [0.2, 0.3, 0.4, 0.6, 0.9]
    errs, alive = [], []
    for mu in mus:
        r = trot(v_des=(0.6, 0.0), t_max=5.0, friction=mu, mu_ctrl=0.35)
        errs.append(r["vx_err"] if r["survived"] else np.nan)
        alive.append(int(r["survived"]))
        rec("3_ground", ground_mu=mu, controller_mu=0.35,
            vx_err=round(r["vx_err"], 3), mean_vx=round(r["mean_vx"], 3),
            distance_m=round(r["distance"], 2), survived=int(r["survived"]),
            fell_at_s=round(r["fell_at"], 2) if r["fell_at"] else None)
    axes[0].plot(mus, alive, "o-", color=COLORS[0], label="survived")
    axes[0].plot(mus, errs, "s-", color=COLORS[1], label="|v_x error| [m/s]")
    axes[0].set_xlabel("ground friction (controller assumes 0.35)")
    axes[0].set_ylim(-0.1, 1.1); axes[0].legend(fontsize=7)
    axes[0].set_title("varying the ground")

    cms = [0.2, 0.35, 0.5, 0.7, 1.0, 3.0]
    errs, alive = [], []
    for cm in cms:
        r = trot(v_des=(0.6, 0.0), t_max=5.0, friction=0.9, mu_ctrl=cm)
        errs.append(r["vx_err"] if r["survived"] else np.nan)
        alive.append(int(r["survived"]))
        rec("3_belief", ground_mu=0.9, controller_mu=cm,
            vx_err=round(r["vx_err"], 3), mean_vx=round(r["mean_vx"], 3),
            distance_m=round(r["distance"], 2), survived=int(r["survived"]),
            fell_at_s=round(r["fell_at"], 2) if r["fell_at"] else None)
    axes[1].plot(cms, alive, "o-", color=COLORS[0], label="survived")
    axes[1].plot(cms, errs, "s-", color=COLORS[1], label="|v_x error| [m/s]")
    axes[1].axvline(0.9, color="0.5", ls=":", label="the true ground mu")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("friction the CONTROLLER assumes (ground is 0.9)")
    axes[1].set_ylim(-0.1, 1.1); axes[1].legend(fontsize=7)
    axes[1].set_title("varying the belief")

    for cone, c, lab in [(True, COLORS[0], "with cone constraint"),
                         (False, COLORS[1], "cone removed entirely")]:
        alive = []
        for mu in mus:
            r = trot(v_des=(0.6, 0.0), t_max=5.0, friction=mu,
                     mu_ctrl=0.35, cone=cone)
            alive.append(int(r["survived"]))
            rec("3_cone", cone=int(cone), ground_mu=mu,
                vx_err=round(r["vx_err"], 3), distance_m=round(r["distance"], 2),
                survived=int(r["survived"]),
                fell_at_s=round(r["fell_at"], 2) if r["fell_at"] else None)
        axes[2].plot(mus, alive, "o-", color=c, label=lab)
    axes[2].set_xlabel("ground friction"); axes[2].set_ylim(-0.1, 1.1)
    axes[2].legend(fontsize=7); axes[2].set_title("the constraint itself")
    save(fig, os.path.join(OUT, "friction.png"))


# ================================================= 4. is the MPC earning it
def exp4():
    print("[4] MPC vs a fixed weight-sharing controller")
    speeds = [0.0, 0.3, 0.6, 0.9, 1.2]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    for ctrl, c, lab in [("mpc", COLORS[0], "convex MPC"),
                         ("fixed", COLORS[1], "equal weight share (no QP)")]:
        errs, rp, alive = [], [], []
        for v in speeds:
            r = trot(v_des=(v, 0.0), t_max=5.0, controller=ctrl)
            errs.append(r["vx_err"]); rp.append(math.degrees(r["rp_rms"]))
            alive.append(int(r["survived"]))
            rec("4_vs_fixed", controller=lab, commanded_vx=v,
                mean_vx=round(r["mean_vx"], 3), vx_err=round(r["vx_err"], 3),
                roll_pitch_rms_deg=round(math.degrees(r["rp_rms"]), 2),
                height_err_mm=round(r["height_err"] * 1000, 1),
                survived=int(r["survived"]),
                ms_per_solve=round(r["ms_per_solve"], 3))
        axes[0].plot(speeds, errs, "o-", color=c, label=lab)
        axes[1].plot(speeds, rp, "o-", color=c, label=lab)
        axes[2].plot(speeds, alive, "o-", color=c, label=lab)
    axes[0].set_ylabel("mean |v_x error| [m/s]")
    axes[1].set_ylabel("roll/pitch RMS [deg]")
    axes[2].set_ylabel("survived"); axes[2].set_ylim(-0.1, 1.1)
    for ax in axes:
        ax.set_xlabel("commanded forward speed [m/s]"); ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "vs_fixed.png"))


# ================================================= 5. gaits and a push
def exp5():
    print("[5] gaits, and a push test")
    gaits = [("trot (diagonal pairs)", Gait(0.34, 0.5, (0.0, 0.5, 0.5, 0.0))),
             ("pace (lateral pairs)", Gait(0.34, 0.5, (0.0, 0.5, 0.0, 0.5))),
             ("bound (front/rear pairs)", Gait(0.34, 0.5, (0.0, 0.0, 0.5, 0.5))),
             ("walk (duty 0.75)", Gait(0.5, 0.75, (0.0, 0.5, 0.25, 0.75)))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    names, errs, rolls = [], [], []
    for lab, g in gaits:
        r = trot(v_des=(0.6, 0.0), t_max=5.0, gait=g)
        names.append(lab.split(" ")[0]); errs.append(r["vx_err"])
        rolls.append(math.degrees(np.sqrt(np.mean(r["log"]["rpy"][:, 0] ** 2))))
        rec("5_gait", gait=lab, duty=g.duty, period_s=g.period,
            mean_vx=round(r["mean_vx"], 3), vx_err=round(r["vx_err"], 3),
            roll_rms_deg=round(rolls[-1], 2),
            roll_pitch_rms_deg=round(math.degrees(r["rp_rms"]), 2),
            survived=int(r["survived"]),
            fell_at_s=round(r["fell_at"], 2) if r["fell_at"] else None)
    x = np.arange(len(names))
    axes[0].bar(x - 0.2, errs, 0.4, color=COLORS[0], label="|v_x error| [m/s]")
    axes[0].bar(x + 0.2, np.array(rolls) / 10, 0.4, color=COLORS[1],
                label="roll RMS / 10 [deg]")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, fontsize=8)
    axes[0].legend(fontsize=7); axes[0].set_title("the same MPC, four gaits")

    # A push is not one number.  Landing it early in a stride (both diagonal
    # pairs about to swap) is a different problem from landing it mid-stance,
    # so sweeping magnitude at ONE instant produces a survival curve that
    # jumps around and means nothing.  Sweep both.
    # v = 0.3 m/s, not 0.4.  The controller has speed bands where it falls
    # over unprompted (see experiment 4's survival column), and running the
    # push test inside one of those measures the baseline, not the push --
    # the first version of this experiment reported 0/4 recoveries at a
    # 0.1 m/s shove, which no 14 kg robot should notice.
    phases = np.linspace(0.0, 0.34, 5)[:4]
    for mag, c in zip([0.2, 0.4, 0.7, 1.0], COLORS):
        surv, rolls = [], []
        for ph in phases:
            r = trot(v_des=(0.3, 0.0), t_max=5.0,
                     push=(2.0 + float(ph), 0.0, mag, 0.0))
            lg = r["log"]
            after = lg["t"] > 2.0 + ph
            mroll = float(np.degrees(np.max(np.abs(lg["rpy"][after, 0])))) \
                if after.any() else float("nan")
            surv.append(int(r["survived"])); rolls.append(mroll)
            rec("5_push", push_mps=mag, push_phase_s=round(float(ph), 3),
                recovered=int(r["survived"]), max_roll_deg=round(mroll, 1),
                max_vy=round(float(np.max(np.abs(lg["v"][after, 1]))), 2)
                if after.any() else None)
        rec("5_push_summary", push_mps=mag, n_phases=len(phases),
            recovered=int(np.sum(surv)),
            rate=round(float(np.mean(surv)), 2),
            mean_max_roll_deg=round(float(np.nanmean(rolls)), 1))
        axes[1].plot(phases, rolls, "o-", color=c, label=f"{mag} m/s sideways")
    axes[1].set_xlabel("when in the stride the push lands [s]")
    axes[1].set_ylabel("peak roll after the push [deg]")
    axes[1].legend(fontsize=7)
    axes[1].set_title("the same shove, four points in the stride")
    save(fig, os.path.join(OUT, "gaits.png"))


if __name__ == "__main__":
    t0 = time.time()
    exp1()
    exp2()
    exp3()
    exp4()
    exp5()
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(ROWS)
    print(f"done in {time.time() - t0:.1f}s")
