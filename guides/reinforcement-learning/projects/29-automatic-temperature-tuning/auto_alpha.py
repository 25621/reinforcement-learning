"""Project 29 — automatic temperature tuning: does auto-alpha beat a tuned constant?

The temperature `alpha` weighs the entropy bonus against reward. It is the one
SAC hyperparameter that is genuinely hard to set by hand, because the right value
depends on the *scale of the rewards*, and that scale differs per task — and even
per training stage within one task, as returns grow.

The experiment is a grid, and it is designed to be falsifiable:

    4 fixed temperatures  x  2 tasks     -> is there ONE constant that wins on both?
    auto-tuned alpha      x  2 tasks     -> does the automatic rule match the best
                                            per-task constant, without being told
                                            which task it is on?

If some fixed alpha won everywhere, auto-tuning would be a solution to a problem
nobody has. The result below is the reason SAC v2 exists.

Run:
    python3 auto_alpha.py     # ~6 min on 12 cores; 10 runs in parallel
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
STEPS = 30_000
TASKS = ["Hopper-v5", "HalfCheetah-v5"]
FIXED = [0.01, 0.05, 0.2, 1.0]
SEED = 0
HIDDEN = 128  # as in projects 27/28: same width for every arm, half the CPU cost of 256


def run_one(args):
    env_id, alpha = args
    common = dict(env_id=env_id, seed=SEED, total_steps=STEPS, hidden=HIDDEN,
                  eval_every=2_000, eval_episodes=3)
    if alpha == "auto":
        cfg = cc.sac_config(auto_alpha=True, alpha=0.2, **common)
    else:
        # identical SAC, one field changed: the temperature is frozen
        cfg = cc.sac_config(auto_alpha=False, alpha=alpha, **common)
    hist, _ = cc.train(cfg)
    return env_id, alpha, hist


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(t, a) for t in TASKS for a in FIXED + ["auto"]]
    print(f"launching {len(jobs)} runs...", flush=True)
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run_one, jobs))

    runs = {}
    for env_id, alpha, h in results:
        runs.setdefault(env_id, {})[alpha] = h

    labels = [f"fixed a={a}" for a in FIXED] + ["auto-tuned"]
    keys = FIXED + ["auto"]

    # ---- 1. return per temperature, per task ----
    fig, axes = ps.plt.subplots(1, 2, figsize=(13, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for j, env_id in enumerate(TASKS):
        ax = ps.style_axes(axes[j])
        for i, (k, lab) in enumerate(zip(keys, labels)):
            h = runs[env_id][k]
            style = (dict(color=ps.INK, lw=2.6) if k == "auto"
                     else dict(color=ps.SERIES[i], lw=1.5))
            ax.plot(h["steps"], h["eval_return"], label=lab, **style)
        ax.set_title(env_id.replace("-v5", ""), color=ps.INK, fontsize=11, loc="left")
        ax.set_xlabel("env steps", color=ps.INK_SECONDARY, fontsize=9)
        if j == 0:
            ax.set_ylabel("evaluation return", color=ps.INK_SECONDARY, fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("The best fixed temperature is task-dependent. Auto-tuning is not.",
                 color=ps.INK, fontsize=13, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "temperature_sweep.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'temperature_sweep.png'}")

    # ---- 2. what auto-alpha actually does: alpha and entropy over time ----
    fig, axes = ps.plt.subplots(1, 2, figsize=(13, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax = ps.style_axes(axes[0])
    for j, env_id in enumerate(TASKS):
        h = runs[env_id]["auto"]
        ax.plot(h["steps"], h["alpha"], color=ps.SERIES[j], lw=2.0,
                label=f"{env_id.replace('-v5', '')} (auto)")
    for i, a in enumerate(FIXED):
        ax.axhline(a, color=ps.INK_MUTED, ls=":", lw=0.9)
        ax.annotate(f"fixed {a}", (0.99, a), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7, color=ps.INK_MUTED)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("alpha finds a different value on each task", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("env steps", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("temperature alpha", color=ps.INK_SECONDARY, fontsize=9)

    ax = ps.style_axes(axes[1])
    for j, env_id in enumerate(TASKS):
        act_dim = {"Hopper-v5": 3, "HalfCheetah-v5": 6}[env_id]
        h = runs[env_id]["auto"]
        ax.plot(h["steps"], np.asarray(h["entropy"]) / act_dim, color=ps.SERIES[j],
                lw=2.0, label=f"{env_id.replace('-v5', '')} (auto)")
        for i, a in enumerate(FIXED):
            hf = runs[env_id][a]
            ax.plot(hf["steps"], np.asarray(hf["entropy"]) / act_dim,
                    color=ps.SERIES[j], lw=0.9, alpha=0.35)
    ax.axhline(-1.0, color=ps.INK, ls="--", lw=1.4)
    ax.annotate("target: -1 per action dim", (0.02, 0.04), xycoords="axes fraction",
                fontsize=8, color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("...and lands entropy on target (thin lines: fixed alpha, adrift)",
                 color=ps.INK, fontsize=11, loc="left")
    ax.set_xlabel("env steps", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("entropy / action dim", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "alpha_entropy_trace.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'alpha_entropy_trace.png'}")

    # ---- the table ----
    print(f"\n=== final return after {STEPS:,} steps ===")
    print(f"{'temperature':14s} " + "".join(f"{t.replace('-v5',''):>16s}" for t in TASKS))
    finals = {}
    for k, lab in zip(keys, labels):
        row = []
        for t in TASKS:
            f = float(np.mean(runs[t][k]["eval_return"][-3:]))
            finals[(t, k)] = f
            row.append(f)
        print(f"{lab:14s} " + "".join(f"{v:16.0f}" for v in row))

    print("\n=== the point ===")
    for t in TASKS:
        best_fixed = max(FIXED, key=lambda a: finals[(t, a)])
        print(f"{t.replace('-v5',''):14s} best fixed alpha = {best_fixed:<5} "
              f"({finals[(t, best_fixed)]:7.0f})   auto = {finals[(t, 'auto')]:7.0f}")
    print("\n=== final entropy per action dim (target = -1.0) ===")
    for t in TASKS:
        act_dim = {"Hopper-v5": 3, "HalfCheetah-v5": 6}[t]
        row = [f"{np.mean(runs[t][k]['entropy'][-3:]) / act_dim:+6.2f}" for k in keys]
        print(f"{t.replace('-v5',''):14s} " + "  ".join(
            f"{lab}={v}" for lab, v in zip(["a=.01", "a=.05", "a=.2", "a=1", "auto"], row)))
    print(f"\nslowest run: {max(h['wall_total'] for _, _, h in results):.0f}s")


if __name__ == "__main__":
    main()
