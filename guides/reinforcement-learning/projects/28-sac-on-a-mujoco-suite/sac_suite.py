"""Project 28 — SAC across a MuJoCo suite, with one hyperparameter setting.

The claim SAC's reputation rests on is not "it gets a high score". It is:

    ONE configuration — one learning rate, one network size, one temperature
    rule — learns on bodies from a 6-joint runner to a 17-joint humanoid, with
    no per-task tuning.

That claim is testable on a laptop budget. What is NOT testable on a laptop
budget is the *final score*: the published numbers come from 1M-10M environment
steps, and this file runs 30k-50k. So this project measures the thing the budget
can actually settle (does one config learn on every body?) and reports the gap to
the published numbers honestly rather than pretending a 10-minute run reproduces
a 3-day one.

Run:
    python3 sac_suite.py      # ~9 min on 12 cores; 5 envs x 2 seeds in parallel
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-ddpg-on-pendulum"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import cc_lib as cc
import plot_style as ps

OUT = HERE / "outputs"
HIDDEN = 128  # see project 27: halves the update cost on CPU, same width for every body

# ONE seed per body, deliberately. The alternative within this budget was 2 seeds at
# half the steps, and that was tried first: at 40k steps Walker2d's curve is flat and
# Ant's ends *below* where it started, so the run answered nothing. Five bodies as five
# single-seed processes also map cleanly onto six physical cores, where ten processes
# would contend and halve each one's speed. The trade is real and worth naming: there
# are no error bars below. Treat the curves as "does this body learn at all", not as a
# benchmark table.
SEEDS = [0]

# Step budgets differ ONLY because the simulators differ in cost: Humanoid's 348-dim
# observation and 17 actuators make one SAC step several times slower than Hopper's, so
# an equal-steps suite would spend all its wall-clock on one body. Every *algorithm*
# setting is identical across the five bodies — that is the experiment. Published
# references are approximate, from the SAC papers and Spinning Up's benchmarks, at
# 1M-3M steps (12-40x this budget).
SUITE = [
    # env_id,           steps,   random,  published (1M-3M steps)
    ("HalfCheetah-v5",  60_000,   -280,   "~11,000"),
    ("Hopper-v5",       60_000,     18,   "~3,300"),
    ("Walker2d-v5",     60_000,      2,   "~4,500"),
    ("Ant-v5",          40_000,    -60,   "~4,500"),
    ("Humanoid-v5",     25_000,    120,   "~5,000"),
]


def run_one(args):
    env_id, steps, seed = args
    # Evaluation is not free here, and it is easy to let it dominate. A MuJoCo episode
    # runs 1000 steps, so `eval_every=2500, eval_episodes=3` spends 3000 env steps per
    # eval — on an 80k-step run that is 96k evaluation steps, MORE than the training
    # itself. Halving the frequency and dropping to 2 episodes cuts that to 32k. It does
    # not touch the learned policy at all (evaluation runs on a separate env, consumes
    # no training RNG, and takes no gradients); it only makes the curve coarser.
    cfg = cc.sac_config(env_id=env_id, seed=seed, total_steps=steps, hidden=HIDDEN,
                        eval_every=5_000, eval_episodes=2)
    hist, _ = cc.train(cfg)
    return env_id, seed, hist


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(e, n, s) for e, n, _, _ in SUITE for s in SEEDS]
    print(f"launching {len(jobs)} runs across {len(SUITE)} bodies...", flush=True)
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run_one, jobs))

    runs = {}
    for env_id, seed, h in results:
        runs.setdefault(env_id, {})[seed] = h

    # ---- one panel per body ----
    fig, axes = ps.plt.subplots(1, 5, figsize=(17.5, 3.5), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for i, (env_id, _, rand, pub) in enumerate(SUITE):
        ax = ps.style_axes(axes[i])
        steps = np.asarray(runs[env_id][SEEDS[0]]["steps"])
        c = np.stack([runs[env_id][s]["eval_return"] for s in SEEDS])
        ax.plot(steps, c.mean(0), color=ps.SERIES[i], lw=2.0)
        if len(SEEDS) > 1:
            ax.fill_between(steps, c.min(0), c.max(0), color=ps.SERIES[i],
                            alpha=0.15, lw=0)
        ax.axhline(rand, color=ps.INK_MUTED, ls="--", lw=1.0)
        ax.set_title(f"{env_id.replace('-v5', '')}  ({pub} at 1M+)",
                     color=ps.INK, fontsize=10.5, loc="left")
        ax.set_xlabel("env steps", color=ps.INK_SECONDARY, fontsize=9)
        if i == 0:
            ax.set_ylabel("evaluation return", color=ps.INK_SECONDARY, fontsize=9)
        ax.annotate("random policy", (steps[0], rand), fontsize=7,
                    color=ps.INK_MUTED, xytext=(2, 4), textcoords="offset points")
    fig.suptitle("One SAC configuration, five bodies, one seed each — 10 minutes of CPU",
                 color=ps.INK, fontsize=13, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "suite_curves.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'suite_curves.png'}")

    # ---- entropy / alpha: the same rule adapts per body ----
    fig, axes = ps.plt.subplots(1, 2, figsize=(12.5, 3.8), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax = ps.style_axes(axes[0])
    for i, (env_id, _, _, _) in enumerate(SUITE):
        h = runs[env_id][SEEDS[0]]
        ax.plot(h["steps"], h["alpha"], color=ps.SERIES[i], lw=1.6,
                label=env_id.replace("-v5", ""))
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Temperature alpha, tuned automatically", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("env steps", color=ps.INK_SECONDARY, fontsize=9)

    ax = ps.style_axes(axes[1])
    for i, (env_id, _, _, _) in enumerate(SUITE):
        h = runs[env_id][SEEDS[0]]
        act_dim = {"HalfCheetah-v5": 6, "Hopper-v5": 3, "Walker2d-v5": 6,
                   "Ant-v5": 8, "Humanoid-v5": 17}[env_id]
        ax.plot(h["steps"], np.asarray(h["entropy"]) / act_dim, color=ps.SERIES[i], lw=1.6,
                label=env_id.replace("-v5", ""))
    ax.axhline(-1.0, color=ps.INK, ls="--", lw=1.2)
    ax.annotate("target entropy (-1 per action dim)", (0.02, 0.06),
                xycoords="axes fraction", fontsize=8, color=ps.INK_SECONDARY)
    ax.set_title("Policy entropy per action dimension", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("env steps", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_entropy.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'alpha_entropy.png'}")

    # ---- the honest table ----
    print(f"\n{'body':16s} {'steps':>7s} {'random':>8s} {'SAC (ours)':>12s} "
          f"{'published':>11s} {'we reach':>9s}")
    for env_id, n, rand, pub in SUITE:
        c = np.stack([runs[env_id][s]["eval_return"] for s in SEEDS])
        final = c[:, -2:].mean()
        pub_n = float(pub.replace("~", "").replace(",", ""))
        frac = 100 * max(final - rand, 0) / (pub_n - rand)
        print(f"{env_id:16s} {n:7,d} {rand:8.0f} {final:12.0f} {pub:>11s} {frac:8.0f}%")
    print(f"\nslowest run: {max(h['wall_total'] for _, _, h in results):.0f}s")


if __name__ == "__main__":
    main()
