"""Project 52 -- learning to walk, and what the reward and the randomisation buy.

Six experiments:
  1. training, and what the policy learned
  2. the reward terms, removed one at a time
  3. the observation, removed one part at a time
  4. domain randomisation: insurance, and its premium
  5. the learned policy against project 51's convex MPC
  6. ground it never saw
"""

import csv
import importlib.util
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "51-quadruped-trotting-mpc"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

import ars                                                      # noqa: E402
from env import WalkEnv                                         # noqa: E402
from plot_style import COLORS, use_style, save                  # noqa: E402

import matplotlib.pyplot as plt                                 # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
ROWS = []

ITERS = 110
CMDS = ((0.2, 0.0), (0.4, 0.0), (0.6, 0.0), (0.8, 0.0))
N_OBS = 50


def rec(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))
    print("   ", exp, " ".join(f"{k}={v}" for k, v in kw.items()))


def _mpc():
    """Load project 51's controller by path (both projects have a run.py)."""
    p = os.path.join(_PROJ, "51-quadruped-trotting-mpc", "run.py")
    spec = importlib.util.spec_from_file_location("r51", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def learn(tag, envkw=None, randomize=False, iters=ITERS, seed=0):
    t0 = time.time()
    W, norm, curve = ars.train(envkw or {}, N_OBS, iters=iters, ndir=16, top=8,
                               alpha=0.03, sigma=0.05, seed=seed,
                               v_cmds=CMDS, randomize=randomize,
                               log=lambda s: print(s))
    print(f"    [{tag}] {time.time() - t0:.0f}s  final return {curve[-1]:.0f}")
    return W, norm, curve


def evaluate(policy, cmds=CMDS, envkw=None, n=1):
    """Average tracking over the command set, on a fixed nominal robot."""
    envkw = dict(envkw or {})
    env = WalkEnv(seed=99, **envkw)
    rows = []
    for c in cmds:
        for s in range(n):
            r = env.rollout(policy, c)
            rows.append((c[0], r["mean_vx"], r["ret"], r["survived"]))
    a = np.asarray(rows, float)
    return dict(track_err=float(np.mean(np.abs(a[:, 1] - a[:, 0]))),
                ret=float(np.mean(a[:, 2])),
                survive=float(np.mean(a[:, 3])),
                per_cmd={c[0]: float(np.mean(a[a[:, 0] == c[0], 1]))
                         for c in cmds})


# ================================================================ 1. training
def exp1():
    print("[1] training")
    W, norm, curve = learn("baseline")
    np.savez(os.path.join(OUT, "policy.npz"), W=W,
             mean=norm.mean, m2=norm.m2, n=norm.n)
    p = ars.policy_fn(W, norm)
    ev = evaluate(p)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.5))
    axes[0].plot(curve, color=COLORS[0])
    axes[0].set_xlabel("ARS iteration"); axes[0].set_ylabel("mean return")
    axes[0].set_title(f"{len(curve)} iterations, {16 * 2 * len(curve)} episodes")
    cmds = [c[0] for c in CMDS]
    axes[1].plot(cmds, [ev["per_cmd"][c] for c in cmds], "o-", color=COLORS[0],
                 label="achieved")
    axes[1].plot(cmds, cmds, ":", color="0.5", label="commanded")
    axes[1].set_xlabel("commanded v_x [m/s]"); axes[1].set_ylabel("achieved")
    axes[1].legend(fontsize=8); axes[1].set_title("velocity tracking")
    env = WalkEnv(seed=99)
    o = env.reset((0.6, 0.0))
    tr = []
    for _ in range(env.ep_len):
        a = p(o)
        o, r, d = env.step(a)
        tr.append(np.concatenate([[env.rb.p[2]], env.rb.data.qpos[7:10]]))
        if d:
            break
    tr = np.asarray(tr)
    tt = np.arange(len(tr)) * 0.02
    axes[2].plot(tt, tr[:, 1], color=COLORS[0], label="FR abduction")
    axes[2].plot(tt, tr[:, 2], color=COLORS[1], label="FR hip")
    axes[2].plot(tt, tr[:, 3], color=COLORS[2], label="FR knee")
    axes[2].set_xlabel("t [s]"); axes[2].set_ylabel("joint angle [rad]")
    axes[2].legend(fontsize=7); axes[2].set_title("the gait it invented")
    save(fig, os.path.join(OUT, "training.png"))
    rec("1_train", iterations=len(curve), episodes=16 * 2 * len(curve),
        params=12 * N_OBS, final_return=round(curve[-1], 1),
        track_err=round(ev["track_err"], 3), survive=ev["survive"],
        **{f"vx_at_{c}": round(ev["per_cmd"][c], 3) for c in cmds})
    return W, norm, ev


# ================================================= 2. reward ablations
def exp2(base_ev):
    print("[2] reward ablations")
    offs = [("upright",), ("effort", "smooth"), ("drift",)]
    names, errs, surv = ["full reward"], [base_ev["track_err"]], \
        [base_ev["survive"]]
    for off in offs:
        W, norm, curve = learn("-".join(off), envkw=dict(reward_off=off))
        ev = evaluate(ars.policy_fn(W, norm))
        names.append("no " + "+".join(off))
        errs.append(ev["track_err"]); surv.append(ev["survive"])
        rec("2_reward", removed="+".join(off), final_return=round(curve[-1], 1),
            track_err=round(ev["track_err"], 3), survive=ev["survive"],
            mean_return=round(ev["ret"], 1))
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, errs, 0.4, color=COLORS[0], label="tracking error [m/s]")
    ax.bar(x + 0.2, surv, 0.4, color=COLORS[1], label="survival rate")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=7, rotation=18)
    ax.legend(fontsize=7); ax.set_title("what each reward term is holding up")
    save(fig, os.path.join(OUT, "reward.png"))


# ================================================= 3. observation ablations
def exp3(base_ev):
    print("[3] observation ablations")
    drops = [("clock",), ("lin_vel",), ("joint_vel",), ("command",)]
    names, errs = ["full observation"], [base_ev["track_err"]]
    for d in drops:
        W, norm, curve = learn("-".join(d), envkw=dict(obs_drop=d))
        ev = evaluate(ars.policy_fn(W, norm), envkw=dict(obs_drop=d))
        names.append("no " + "+".join(d))
        errs.append(ev["track_err"])
        rec("3_obs", removed="+".join(d), final_return=round(curve[-1], 1),
            track_err=round(ev["track_err"], 3), survive=ev["survive"])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh(names, errs, color=COLORS[0])
    ax.invert_yaxis(); ax.set_xlabel("velocity tracking error [m/s]")
    ax.set_title("what the policy actually needs to see")
    save(fig, os.path.join(OUT, "observation.png"))


# ================================================= 4. domain randomization
def exp4(W_spec, norm_spec):
    print("[4] domain randomization")
    W_dr, norm_dr, _ = learn("randomized", randomize=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    frictions = [0.35, 0.5, 0.7, 0.9, 1.1]
    payloads = [0.0, 1.0, 2.0, 3.0, 4.0]
    for (W, nm), c, lab in [((W_spec, norm_spec), COLORS[1], "specialist"),
                            ((W_dr, norm_dr), COLORS[0], "randomized")]:
        p = ars.policy_fn(W, nm)
        ef, ep = [], []
        for mu in frictions:
            ev = evaluate(p, envkw=dict(friction=mu))
            ef.append(ev["track_err"])
            rec("4_dr", policy=lab, axis="friction", value=mu,
                track_err=round(ev["track_err"], 3), survive=ev["survive"])
        for pl in payloads:
            ev = evaluate(p, envkw=dict(payload=pl))
            ep.append(ev["track_err"])
            rec("4_dr", policy=lab, axis="payload_kg", value=pl,
                track_err=round(ev["track_err"], 3), survive=ev["survive"])
        axes[0].plot(frictions, ef, "o-", color=c, label=lab)
        axes[1].plot(payloads, ep, "o-", color=c, label=lab)
    axes[0].set_xlabel("ground friction"); axes[1].set_xlabel("payload [kg]")
    for ax in axes:
        ax.set_ylabel("tracking error [m/s]"); ax.legend(fontsize=7)
    axes[0].axvspan(0.45, 1.15, color=COLORS[0], alpha=0.08)
    axes[1].axvspan(0.0, 2.5, color=COLORS[0], alpha=0.08)
    axes[0].set_title("shaded = the training range")
    save(fig, os.path.join(OUT, "randomization.png"))
    return W_dr, norm_dr


# ================================================= 5. against the MPC
def exp5(W, norm):
    print("[5] learned policy vs convex MPC")
    m51 = _mpc()
    p = ars.policy_fn(W, norm)
    env = WalkEnv(seed=99)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5))
    cmds = [0.2, 0.4, 0.6, 0.8, 1.0]
    lp, mp, lt, mt = [], [], [], []
    for v in cmds:
        r = env.rollout(p, (v, 0.0))
        t0 = time.perf_counter()
        for _ in range(200):
            p(env.obs())
        lms = (time.perf_counter() - t0) / 200 * 1000
        rm = m51.trot(v_des=(v, 0.0), t_max=5.0)
        lp.append(abs(r["mean_vx"] - v)); mp.append(abs(rm["mean_vx"] - v))
        lt.append(lms); mt.append(rm["ms_per_solve"])
        rec("5_vs_mpc", commanded_vx=v,
            learned_vx=round(r["mean_vx"], 3),
            learned_err=round(abs(r["mean_vx"] - v), 3),
            learned_survived=int(r["survived"]),
            learned_ms=round(lms, 4),
            mpc_vx=round(rm["mean_vx"], 3),
            mpc_err=round(abs(rm["mean_vx"] - v), 3),
            mpc_survived=int(rm["survived"]),
            mpc_ms=round(rm["ms_per_solve"], 2))
    axes[0].plot(cmds, lp, "o-", color=COLORS[0], label="learned policy")
    axes[0].plot(cmds, mp, "o-", color=COLORS[1], label="convex MPC")
    axes[0].set_xlabel("commanded v_x [m/s]")
    axes[0].set_ylabel("|velocity error| [m/s]"); axes[0].legend(fontsize=7)
    axes[1].bar(["learned", "MPC"], [np.mean(lt), np.mean(mt)],
                color=[COLORS[0], COLORS[1]])
    axes[1].set_yscale("log"); axes[1].set_ylabel("ms per control step")
    axes[1].set_title(f"{np.mean(mt) / max(np.mean(lt), 1e-9):.0f}x apart")
    save(fig, os.path.join(OUT, "vs_mpc.png"))
    return m51


# ================================================= 6. unseen ground
def exp6(W, norm, W_dr, norm_dr, m51):
    print("[6] ground it never saw")
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    heights = [0.0, 0.01, 0.02, 0.035, 0.05]
    for (Wx, nmx), c, lab in [((W, norm), COLORS[1], "specialist policy"),
                              ((W_dr, norm_dr), COLORS[0], "randomized policy")]:
        px = ars.policy_fn(Wx, nmx)
        errs = []
        for h in heights:
            ev = evaluate(px, envkw=dict(terrain=None if h == 0 else h))
            errs.append(ev["track_err"])
            rec("6_terrain", controller=lab, bump_height_m=h,
                track_err=round(ev["track_err"], 3), survive=ev["survive"])
        ax.plot(heights, errs, "o-", color=c, label=lab)
    errs = []
    for h in heights:
        r = m51.trot(v_des=(0.5, 0.0), t_max=5.0,
                     terrain=None if h == 0 else h)
        errs.append(abs(r["mean_vx"] - 0.5))
        rec("6_terrain", controller="convex MPC", bump_height_m=h,
            track_err=round(abs(r["mean_vx"] - 0.5), 3),
            survive=float(r["survived"]))
    ax.plot(heights, errs, "o-", color=COLORS[2], label="convex MPC")
    ax.set_xlabel("random bump height [m]")
    ax.set_ylabel("velocity tracking error [m/s]")
    ax.legend(fontsize=7); ax.set_title("all three trained/tuned on flat ground")
    save(fig, os.path.join(OUT, "terrain.png"))


if __name__ == "__main__":
    t0 = time.time()
    W, norm, ev = exp1()
    exp2(ev)
    exp3(ev)
    W_dr, norm_dr = exp4(W, norm)
    m51 = exp5(W, norm)
    exp6(W, norm, W_dr, norm_dr, m51)
    keys = []
    for r in ROWS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(ROWS)
    print(f"done in {time.time() - t0:.1f}s")
