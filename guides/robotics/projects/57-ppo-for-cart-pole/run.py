"""PPO on the cart-pole, graded against a controller that cannot be beaten.

  1. does it learn, and how close to optimal does it get?
  2. the clipped objective, switched off  (the loss, verified)
  3. the GAE lambda knob
  4. advantage normalisation and the entropy bonus
  5. outside the linearisation, where LQR's guarantee expires
  6. when the model is wrong
"""

import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cp_env as E         # noqa: E402
import ppo as P            # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

LONG = 1_000_000
SHORT = 400_000
ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:20s} {key:44s} {value}  {unit}", flush=True)


def job(cfg):
    torch.set_num_threads(2)
    kw = dict(cfg)
    tag = kw.pop("tag")
    t0 = time.time()
    net, hist, diag = P.train_ppo(eval_every=5, **kw)
    return dict(tag=tag, cfg=cfg, hist=hist,
                kl=float(np.mean(diag["kl"][-10:])),
                clipfrac=float(np.mean(diag["clipfrac"][-10:])),
                state=net.state_dict(), secs=time.time() - t0)


def net_from(state):
    net = P.ActorCritic(E.BatchCartPole.obs_dim, E.BatchCartPole.act_dim)
    net.load_state_dict(state)
    net.eval()
    return net


def main():
    torch.set_num_threads(2)
    lqr_ctrl, K = E.lqr_controller()
    lqr = E.episode_cost(lqr_ctrl, n_ep=200, seed=4242)
    record("0-reference", "LQR gain K", np.round(K, 2).tolist())
    record("0-reference", "LQR mean episode cost", round(lqr["cost"], 4))
    record("0-reference", "do-nothing mean episode cost",
           round(E.episode_cost(lambda o, s: np.zeros(len(s)), n_ep=200,
                                seed=4242)["cost"], 1))
    record("0-reference", "batched physics vs project 09", E.verify())

    jobs = []
    for s in (0, 1, 2):
        jobs.append(dict(tag=f"main-s{s}", seed=s, total_steps=LONG))
    for clip in (0.2, None):
        for ep in (1, 4, 10):
            for s in (0, 1):
                jobs.append(dict(tag=f"clip{clip}-ep{ep}-s{s}", seed=s,
                                 total_steps=SHORT, clip=clip, epochs=ep))
    for lam in (0.0, 0.5, 0.95, 1.0):
        for s in (0, 1):
            jobs.append(dict(tag=f"lam{lam}-s{s}", seed=s, total_steps=SHORT,
                             lam=lam))
    for name, kw in [("advnorm-off", dict(adv_norm=False)),
                     ("ent-0.01", dict(ent_coef=0.01)),
                     ("rewnorm-off", dict(reward_norm=False)),
                     ("costcap-off", dict(plant=dict(cap=False)))]:
        for s in (0, 1):
            jobs.append(dict(tag=f"{name}-s{s}", seed=s, total_steps=SHORT, **kw))
    for s in (0, 1):
        jobs.append(dict(tag=f"wide-s{s}", seed=s, total_steps=LONG,
                         init_angle=1.0))

    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        res = list(ex.map(job, jobs))
    R = {r["tag"]: r for r in res}
    print(f"[{time.time() - T0:6.1f}s] {len(res)} PPO runs finished", flush=True)

    def cost_of(tag):
        return float(R[tag]["hist"][-1][1])

    # -- 1. the learning curve ---------------------------------------------
    hists = [R[f"main-s{s}"]["hist"] for s in (0, 1, 2)]
    finals = [h[-1][1] for h in hists]
    record("1-learning", "PPO final cost (3 seeds)",
           f"{np.mean(finals):.3f} +- {np.std(finals):.3f}")
    record("1-learning", "PPO / LQR cost ratio", round(np.mean(finals) / lqr["cost"], 2), "x")
    record("1-learning", "environment steps", LONG)
    record("1-learning", "wall clock per run", round(np.mean([R[f'main-s{s}']['secs']
                                                             for s in (0, 1, 2)]), 1), "s")
    first_stand = []
    for h in hists:
        idx = np.where(h[:, 2] == 0)[0]
        first_stand.append(h[idx[0], 0] if len(idx) else np.nan)
    record("1-learning", "steps until the pole stops falling",
           f"{np.nanmean(first_stand):.0f}")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    for s, h in zip((0, 1, 2), hists):
        ax.plot(h[:, 0], h[:, 1], label=f"PPO seed {s}")
    ax.axhline(lqr["cost"], c="g", ls="--", label=f"LQR (optimal) {lqr['cost']:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("episode cost (lower is better)")
    ax.set_title("PPO finds its way to within a small factor of optimal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curve.png"), dpi=110)
    plt.close(fig)

    # -- 2. the clip --------------------------------------------------------
    tbl = {}
    for clip in (0.2, None):
        for ep in (1, 4, 10):
            c = [cost_of(f"clip{clip}-ep{ep}-s{s}") for s in (0, 1)]
            kl = np.mean([R[f"clip{clip}-ep{ep}-s{s}"]["kl"] for s in (0, 1)])
            tbl[(clip, ep)] = (float(np.mean(c)), float(kl))
            record("2-clipping", f"clip={clip}, {ep:2d} epochs/batch: cost",
                   f"{np.mean(c):.3f}")
            record("2-clipping", f"clip={clip}, {ep:2d} epochs/batch: mean KL",
                   round(float(kl), 5))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    eps = [1, 4, 10]
    axes[0].plot(eps, [tbl[(0.2, e)][0] for e in eps], "o-", label="clipped (PPO)")
    axes[0].plot(eps, [tbl[(None, e)][0] for e in eps], "s--", label="no clip (vanilla PG)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("gradient epochs per batch of data")
    axes[0].set_ylabel("episode cost")
    axes[0].set_title("Clipping is free insurance -- until you reuse data")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].plot(eps, [tbl[(0.2, e)][1] for e in eps], "o-", label="clipped")
    axes[1].plot(eps, [tbl[(None, e)][1] for e in eps], "s--", label="no clip")
    axes[1].set_xlabel("gradient epochs per batch")
    axes[1].set_ylabel("KL(old || new) per minibatch")
    axes[1].set_title("...because that is when the policy runs away")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "clipping.png"), dpi=110)
    plt.close(fig)

    # -- 3. GAE lambda ------------------------------------------------------
    lams, lcost = [], []
    for lam in (0.0, 0.5, 0.95, 1.0):
        c = float(np.mean([cost_of(f"lam{lam}-s{s}") for s in (0, 1)]))
        lams.append(lam)
        lcost.append(c)
        record("3-gae-lambda", f"lambda={lam}: cost", round(c, 3))
    fig, ax = plt.subplots(figsize=(6.2, 4))
    ax.plot(lams, lcost, "o-")
    ax.set_yscale("log")
    ax.set_xlabel("GAE lambda")
    ax.set_ylabel("episode cost")
    ax.set_title("Trusting the critic (0) vs trusting the rewards (1)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gae_lambda.png"), dpi=110)
    plt.close(fig)

    # -- 4. the other switches ---------------------------------------------
    base = float(np.mean([cost_of(f"lam0.95-s{s}") for s in (0, 1)]))
    record("4-switches", "baseline (400k steps)", round(base, 3))
    abl = {"baseline": base}
    for name in ("advnorm-off", "ent-0.01", "rewnorm-off", "costcap-off"):
        c = float(np.mean([cost_of(f"{name}-s{s}") for s in (0, 1)]))
        abl[name] = c
        record("4-switches", f"{name}: cost", round(c, 3))
    fig, ax = plt.subplots(figsize=(6.6, 4))
    ax.bar(list(abl.keys()), list(abl.values()),
           color=["tab:green", "tab:orange", "tab:blue", "tab:red", "tab:purple"])
    ax.set_yscale("log")
    ax.set_ylabel("episode cost")
    ax.set_title("Which parts of the recipe are load-bearing?")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "switches.png"), dpi=110)
    plt.close(fig)

    # -- 5. outside the linearisation --------------------------------------
    wide_net = net_from(R["wide-s0"]["state"])
    narrow_net = net_from(R["main-s0"]["state"])
    angles = [0.1, 0.3, 0.6, 0.9, 1.2]
    curves = {"LQR": [], "PPO (trained wide)": [], "PPO (trained narrow)": []}
    for ang in angles:
        for name, ctrl in [("LQR", lqr_ctrl),
                           ("PPO (trained wide)", P.make_controller(wide_net)),
                           ("PPO (trained narrow)", P.make_controller(narrow_net))]:
            c = E.episode_cost(ctrl, n_ep=100, seed=777, init_angle=ang, init_rate=0.5)
            curves[name].append(c["cost"])
            record("5-nonlinear", f"start angle {ang:.1f} rad, {name}: cost",
                   round(c["cost"], 3))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for name, ys in curves.items():
        ax.plot(angles, ys, "o-", label=name)
    ax.set_yscale("log")
    ax.set_xlabel("initial pole angle (radians from upright)")
    ax.set_ylabel("episode cost")
    ax.set_title("LQR is optimal for the model it was given -- and only near it")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nonlinear.png"), dpi=110)
    plt.close(fig)

    # -- 6. when the model is wrong ----------------------------------------
    for name, wrong in [("pole 4x heavier", dict(m=0.4)),
                        ("pole half as long", dict(l=0.25)),
                        ("cart 3x heavier", dict(M=3.0))]:
        bad_ctrl, _ = E.lqr_controller(**{**dict(M=1.0, m=0.1, l=0.5), **wrong})
        c_bad = E.episode_cost(bad_ctrl, n_ep=200, seed=4242)
        record("6-model-error", f"LQR designed on '{name}', run on the real plant",
               f"cost {c_bad['cost']:.3f}, fell {c_bad['fell']:.2f}")
    record("6-model-error", "PPO (no model at all)",
           f"cost {np.mean(finals):.3f}")

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
