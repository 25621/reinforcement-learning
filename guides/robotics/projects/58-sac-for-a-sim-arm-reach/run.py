"""SAC on an arm reach: six experiments, all about alpha and what it buys.

  1. does it learn, and how does it compare to the classical controller?
  2. the entropy temperature alpha, swept
  3. automatic alpha and the target entropy it chases
  4. the reward, which turns out to matter more than alpha
  5. the replay ratio: gradient steps per environment step
  6. the control nobody runs: just clone the classical controller
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE),
                                "54-behavior-cloning-on-a-sim-arm"))

import reach as RE        # noqa: E402
import sac as S           # noqa: E402
import nets               # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

BASE = dict(total_steps=50_000, hidden=64, n_env=16, bs=128, utd=0.5,
            lr=1e-3, alpha=0.01, eval_every=5000)
ROWS = []
T0 = time.time()


def record(exp, key, value, unit=""):
    ROWS.append(dict(experiment=exp, key=key, value=value, unit=unit))
    print(f"[{time.time() - T0:6.1f}s] {exp:22s} {key:42s} {value}  {unit}", flush=True)


def job(cfg):
    torch.set_num_threads(2)
    kw = dict(BASE)
    tag = cfg.pop("tag")
    kw.update(cfg)
    t0 = time.time()
    actor, hist, n_upd = S.train_sac(**kw)
    return dict(tag=tag, cfg=cfg, hist=hist, secs=time.time() - t0,
                updates=n_upd, state=actor.state_dict())


def main():
    torch.set_num_threads(2)

    # -- references ---------------------------------------------------------
    ik = RE.evaluate(RE.ik_controller(), n_ep=200, seed=4242)
    record("0-reference", "batched arm vs project 54's arm", RE.verify())
    record("0-reference", "damped-least-squares IK: mean final distance",
           round(ik["dist"] * 1000, 2), "mm")
    record("0-reference", "damped-least-squares IK: success", ik["success"])
    for name, ctrl in [("do nothing", lambda o: np.zeros((len(o), 2))),
                       ("random actions",
                        lambda o: np.random.uniform(-1, 1, (len(o), 2)))]:
        r = RE.evaluate(ctrl, n_ep=200, seed=4242)
        record("0-reference", f"{name}: mean final distance",
               round(r["dist"] * 1000, 1), "mm")

    jobs = [dict(tag=f"main-s{s}", seed=s) for s in (0, 1, 2)]
    for a in (0.002, 0.05, 0.2):
        jobs.append(dict(tag=f"alpha{a}", seed=0, alpha=a))
    for te in (-1.0, -4.0):
        jobs.append(dict(tag=f"auto{te}", seed=0, auto_alpha=True,
                         target_entropy=te))
    jobs.append(dict(tag="auto-2.0", seed=0, auto_alpha=True, target_entropy=-2.0))
    for mode in ("dense", "sparse"):
        jobs.append(dict(tag=f"rew-{mode}", seed=0, env_kw=dict(reward_mode=mode)))
    for u in (0.25, 1.0):
        jobs.append(dict(tag=f"utd{u}", seed=0, utd=u))

    ctx = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        res = list(ex.map(job, jobs))
    R = {r["tag"]: r for r in res}
    print(f"[{time.time() - T0:6.1f}s] {len(res)} SAC runs finished", flush=True)

    def final(tag, col=1):
        return float(R[tag]["hist"][-1][col])

    # -- 1. the learning curve ---------------------------------------------
    hists = [R[f"main-s{s}"]["hist"] for s in (0, 1, 2)]
    d = [h[-1][1] for h in hists]
    sc = [h[-1][2] for h in hists]
    record("1-learning", "SAC final distance (3 seeds)",
           f"{np.mean(d) * 1000:.2f} +- {np.std(d) * 1000:.2f} mm")
    record("1-learning", "SAC final success (3 seeds)",
           f"{np.mean(sc):.3f} +- {np.std(sc):.3f}")
    record("1-learning", "SAC / IK distance ratio",
           round(float(np.mean(d) / ik["dist"]), 1), "x")
    record("1-learning", "environment steps", BASE["total_steps"])
    record("1-learning", "gradient updates", R["main-s0"]["updates"])
    record("1-learning", "wall clock per run",
           round(float(np.mean([R[f'main-s{s}']['secs'] for s in (0, 1, 2)])), 1), "s")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for s, h in zip((0, 1, 2), hists):
        axes[0].plot(h[:, 0], h[:, 1] * 1000, label=f"seed {s}")
    axes[0].axhline(ik["dist"] * 1000, c="g", ls="--", label="IK controller")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("environment steps")
    axes[0].set_ylabel("final tip error (mm)")
    axes[0].set_title("SAC learns to reach")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    for s, h in zip((0, 1, 2), hists):
        axes[1].plot(h[:, 0], h[:, 3], label=f"seed {s}")
    axes[1].set_xlabel("environment steps")
    axes[1].set_ylabel("policy entropy (nats)")
    axes[1].set_title("...and gets more decisive as it does")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curve.png"), dpi=110)
    plt.close(fig)

    # -- 2. alpha -----------------------------------------------------------
    alphas = [0.002, 0.01, 0.05, 0.2]
    tags = {0.002: "alpha0.002", 0.01: "main-s0", 0.05: "alpha0.05", 0.2: "alpha0.2"}
    dists, succ, ents = [], [], []
    for a in alphas:
        t = tags[a]
        dists.append(final(t, 1))
        succ.append(final(t, 2))
        ents.append(final(t, 3))
        record("2-alpha", f"alpha={a}: final distance", f"{final(t, 1) * 1000:.2f} mm")
        record("2-alpha", f"alpha={a}: success", round(final(t, 2), 3))
        record("2-alpha", f"alpha={a}: policy entropy", round(final(t, 3), 2))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(alphas, np.array(dists) * 1000, "o-", label="final error (mm)")
    ax.set_xscale("log")
    ax.set_xlabel("entropy temperature alpha")
    ax.set_ylabel("final tip error (mm)")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(alphas, succ, "s--", c="gray", label="success")
    ax2.set_ylabel("success rate")
    ax.set_title("Too much entropy and the policy never commits")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "alpha.png"), dpi=110)
    plt.close(fig)

    # -- 3. automatic alpha -------------------------------------------------
    for te, tag in [(-1.0, "auto-1.0"), (-2.0, "auto-2.0"), (-4.0, "auto-4.0")]:
        record("3-auto-alpha", f"target entropy {te}: final distance",
               f"{final(tag, 1) * 1000:.2f} mm")
        record("3-auto-alpha", f"target entropy {te}: success", round(final(tag, 2), 3))
        record("3-auto-alpha", f"target entropy {te}: alpha ended at",
               round(final(tag, 4), 5))
    best_fixed = min(dists)
    record("3-auto-alpha", "best fixed alpha vs best auto",
           f"{best_fixed * 1000:.2f} mm vs "
           f"{min(final('auto-1.0', 1), final('auto-2.0', 1), final('auto-4.0', 1)) * 1000:.2f} mm")

    # -- 4. the reward ------------------------------------------------------
    rew = {"distance + on-target bonus": ("main-s0", 0), "distance only": ("rew-dense", 0),
           "on-target bonus only (sparse)": ("rew-sparse", 0)}
    fig, ax = plt.subplots(figsize=(7, 4))
    names, vals, ss = [], [], []
    for name, (tag, _) in rew.items():
        record("4-reward", f"{name}: final distance", f"{final(tag, 1) * 1000:.2f} mm")
        record("4-reward", f"{name}: success", round(final(tag, 2), 3))
        names.append(name.replace(" (", "\n("))
        vals.append(final(tag, 1) * 1000)
        ss.append(final(tag, 2))
    ax.bar(names, vals, color=["tab:green", "tab:orange", "tab:red"])
    for i, s in enumerate(ss):
        ax.text(i, vals[i], f"success {s:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("final tip error (mm)")
    ax.set_yscale("log")
    ax.set_title("Same algorithm, same budget, three reward functions")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "reward.png"), dpi=110)
    plt.close(fig)

    # -- 5. replay ratio ----------------------------------------------------
    for u, tag in [(0.25, "utd0.25"), (0.5, "main-s0"), (1.0, "utd1.0")]:
        record("5-replay-ratio", f"utd={u}: final distance", f"{final(tag, 1) * 1000:.2f} mm")
        record("5-replay-ratio", f"utd={u}: updates", R[tag]["updates"])
        record("5-replay-ratio", f"utd={u}: wall clock", round(R[tag]["secs"], 1), "s")

    # -- 6. the control: clone the classical controller ---------------------
    ikc = RE.ik_controller()
    bc_pts, bc_dist = [], []
    for n_ep in (10, 40, 160):
        rng = np.random.default_rng(7)
        env = RE.BatchReach(n_ep, seed=7)
        O, Y = [], []
        for _ in range(RE.EP_LEN):
            o = env.obs()
            a = ikc(o)
            O.append(o)
            Y.append(a)
            env.step(a)
        O = np.concatenate(O).astype(np.float32)
        Y = np.concatenate(Y).astype(np.float32)
        net, norm, _ = nets.train_bc(O, Y, epochs=200, seed=0, hidden=64)

        def pol(obs, net=net, norm=norm):
            with torch.no_grad():
                return net(torch.tensor(norm(obs.astype(np.float32)),
                                        dtype=torch.float32)).numpy()
        r = RE.evaluate(pol, n_ep=200, seed=4242)
        bc_pts.append(len(O))
        bc_dist.append(r["dist"])
        record("6-clone-the-expert",
               f"BC on {len(O)} IK transitions: final distance",
               f"{r['dist'] * 1000:.2f} mm")
        record("6-clone-the-expert", f"BC on {len(O)} IK transitions: success",
               round(r["success"], 3))
    record("6-clone-the-expert", "SAC needed", f"{BASE['total_steps']} interactions "
           f"for {np.mean(d) * 1000:.2f} mm")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(bc_pts, np.array(bc_dist) * 1000, "o-", label="behaviour cloning of IK")
    ax.axhline(np.mean(d) * 1000, c="tab:red", ls="--",
               label=f"SAC after {BASE['total_steps']} steps")
    ax.axhline(ik["dist"] * 1000, c="g", ls=":", label="IK itself")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("transitions used")
    ax.set_ylabel("final tip error (mm)")
    ax.set_title("If you already have a controller, copying it is cheap")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "clone_control.png"), dpi=110)
    plt.close(fig)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment", "key", "value", "unit"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nDONE in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
