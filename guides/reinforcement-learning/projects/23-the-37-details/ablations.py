"""Turn each of PPO's implementation details off, one at a time, and measure the damage.

Project 22 built a PPO with every detail switched on and every detail behind a
named flag. This project flips them.

The point is not that "details matter" — everyone says that. The point is the
SHAPE of the result: a couple of these are load-bearing walls whose removal
collapses the agent, most are worth a modest and real improvement, and at least
one is worth nothing at all on this task. You cannot tell which is which from
the paper, and you cannot tell from the source code either. You have to run it.

Each ablation is the full PPO with exactly one thing changed, three seeds, the
same 300k steps of LunarLander. The baseline is project 22's config.

Runtime: ~9 min on 12 CPU cores (`python3 ablations.py all`).
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "22-ppo-from-scratch"))
sys.path.insert(0, str(HERE.parent / "19-reinforce-on-cartpole"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import plot_style as ps
from ppo import PPOConfig, train_ppo

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SEEDS = [0, 1, 2]
# The baseline is project 22's tuned LunarLander PPO at a shorter budget, so that a
# 39-run ablation fits in ten minutes. At 300k steps the full agent is climbing
# steeply and has not yet saturated, which is exactly where an ablation is most
# legible: a detail that matters shows up as a curve that never left the ground.
#
# Three seeds is thin, and it is what the compute allows. PPO's seed-to-seed spread
# on LunarLander at this budget is LARGE — comparable to the effect of most of the
# details being tested — so the figure plots the individual seeds rather than hiding
# them behind a mean, and no claim below rests on a gap smaller than that spread.
# Under-powered ablations are the norm in published RL, not the exception; the honest
# response is to say which differences survive the noise and which do not.
BASE = PPOConfig(env_id="LunarLander-v3", total_steps=300_000, n_envs=8, n_steps=256,
                 n_epochs=10, n_minibatches=8, lr=3e-4, ent_coef=0.01,
                 gamma=0.999, gae_lambda=0.98, norm_reward=True)

# Each entry: label -> (detail number in the blog post, the single change).
# Every one of these is "PPO, but with one line deleted".
ABLATIONS = {
    "PPO (all details on)":         (None, {}),
    "no reward scaling (#30)":      (30,   dict(norm_reward=False)),
    "no ratio clipping (#8)":       (8,    dict(clip_ratio=False)),
    "no advantage norm (#7)":       (7,    dict(norm_adv=False)),
    "no GAE, n-step only (#5)":     (5,    dict(gae=False)),
    "no grad clipping (#11)":       (11,   dict(max_grad_norm=1e9)),
    "no entropy bonus (#10)":       (10,   dict(ent_coef=0.0)),
    "no value clipping (#9)":       (9,    dict(clip_vloss=False)),
    "no orthogonal init (#2)":      (2,    dict(ortho_init=False)),
    "Adam eps 1e-8 (#3)":           (3,    dict(adam_eps=1e-8)),
    "no LR annealing (#4)":         (4,    dict(anneal_lr=False)),
    "1 epoch = A2C (#6)":           (6,    dict(n_epochs=1, n_minibatches=1)),
    # Not an ablation of one detail — the fourth corner of a 2x2. Details #30 and
    # #11 are NOT independent, and the interaction is invisible to any experiment
    # that removes one thing at a time. See `cmd_interaction`.
    "no reward scaling AND no grad clip": (None, dict(norm_reward=False, max_grad_norm=1e9)),
}

# The 2x2 is assembled from four arms that are ALREADY being trained above, so it
# costs nothing extra: {reward scaling on/off} x {global grad clip on/off}.
INTERACTION = [
    ("PPO (all details on)",              "scaling ON",  "clip ON"),
    ("no reward scaling (#30)",           "scaling OFF", "clip ON"),
    ("no grad clipping (#11)",            "scaling ON",  "clip OFF"),
    ("no reward scaling AND no grad clip", "scaling OFF", "clip OFF"),
]


def _job(args):
    label, changes, seed = args
    cfg = replace(BASE, **changes)
    r = train_ppo(cfg, seed=seed, threads=1)
    r.pop("agent")
    return label, seed, r


def cmd_run():
    jobs = [(label, changes, s) for label, (_, changes) in ABLATIONS.items() for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(_job, jobs))

    by_label = {}
    for label, seed, r in res:
        by_label.setdefault(label, []).append(r)

    np.savez(OUT / "ablations.npz",
             labels=list(ABLATIONS),
             steps=by_label["PPO (all details on)"][0]["steps"],
             curves=np.stack([np.stack([r["curve"] for r in by_label[l]]) for l in ABLATIONS]),
             finals=np.stack([np.array([r["final"] for r in by_label[l]]) for l in ABLATIONS]),
             kl=np.stack([np.array([r["kl"].mean() for r in by_label[l]]) for l in ABLATIONS]))

    base = np.mean([r["final"] for r in by_label["PPO (all details on)"]])
    print(f"\n{'ablation':36s} {'final return':>14s} {'Δ vs full PPO':>14s}  {'mean KL':>8s}")
    print("-" * 78)
    rows = sorted(ABLATIONS, key=lambda l: np.mean([r["final"] for r in by_label[l]]), reverse=True)
    for l in rows:
        f = np.array([r["final"] for r in by_label[l]])
        kl = np.mean([r["kl"].mean() for r in by_label[l]])
        print(f"{l:36s} {f.mean():8.1f} ± {f.std():4.0f} {f.mean() - base:+14.1f}  {kl:8.4f}")

    print("\nThe 2x2 that one-at-a-time ablation cannot see:")
    print(f"{'':22s} {'grad clip ON':>14s} {'grad clip OFF':>14s}")
    for scaling in ("scaling ON", "scaling OFF"):
        cells = []
        for clip in ("clip ON", "clip OFF"):
            label = next(l for l, s, c in INTERACTION if s == scaling and c == clip)
            cells.append(np.mean([r["final"] for r in by_label[label]]))
        print(f"reward {scaling:15s} {cells[0]:14.1f} {cells[1]:14.1f}")


def cmd_plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT / "ablations.npz")
    labels = list(d["labels"])
    finals = d["finals"]
    base_i = labels.index("PPO (all details on)")
    base = finals[base_i].mean()

    order = np.argsort(finals.mean(1))
    fig, ax = ps.new_axes(9.0, 5.6)
    y = np.arange(len(order))
    means = finals.mean(1)[order]
    names = [labels[i] for i in order]
    colors = [ps.SERIES[2] if m < base - 40 else (ps.SERIES[0] if i == base_i else ps.SERIES[1])
              for m, i in zip(means, order)]
    ax.barh(y, means, color=colors, height=0.66)
    # Two seeds is too few to hide behind an error bar — plot them.
    for yi, i in enumerate(order):
        ax.plot(finals[i], [yi] * finals.shape[1], "o", color=ps.INK, markersize=3.5,
                markerfacecolor="none", markeredgewidth=1.0, zorder=5)
    ax.axvline(base, color=ps.INK, linestyle="--", linewidth=1.2)
    ax.text(base + 4, len(order) - 0.4, "full PPO", color=ps.INK, fontsize=8)
    ax.axvline(0, color=ps.BASELINE, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ps.finish(fig, ax, "One detail removed at a time — LunarLander, 300k steps\n"
                       "(bars are the mean of 3 seeds; circles are the seeds themselves)",
              "final return (20 evaluation episodes)", "", OUT / "ablation_ladder.png")

    # The 2x2. This is the figure that a one-at-a-time ablation cannot produce.
    fig, ax = ps.new_axes(6.6, 4.2)
    grid = np.zeros((2, 2))
    for label, scaling, clip in INTERACTION:
        i = 0 if scaling == "scaling ON" else 1
        j = 0 if clip == "clip ON" else 1
        grid[i, j] = finals[labels.index(label)].mean()
    im = ax.imshow(grid, cmap="RdYlGn", vmin=grid.min() - 40, vmax=grid.max() + 10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center",
                    color=ps.INK, fontsize=15, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["global grad clip ON\n(detail #11)",
                                               "grad clip OFF"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["reward scaling ON\n(detail #30)",
                                               "reward scaling OFF"], fontsize=9)
    ax.grid(False)
    ps.finish(fig, ax, "Two details, neither of which is safe alone",
              "", "", OUT / "interaction.png")

    # The two catastrophes deserve their curves shown, not just their endpoints.
    fig, ax = ps.new_axes(7.8, 4.4)
    show = ["PPO (all details on)", "no ratio clipping (#8)", "no advantage norm (#7)",
            "1 epoch = A2C (#6)"]
    for i, name in enumerate(show):
        j = labels.index(name)
        c = d["curves"][j]
        ax.fill_between(d["steps"], c.min(0), c.max(0), color=ps.SERIES[i], alpha=0.12, linewidth=0)
        ax.plot(d["steps"], c.mean(0), color=ps.SERIES[i], linewidth=2.2, label=name)
    ax.axhline(200, color=ps.INK_MUTED, linestyle=":", linewidth=1)
    ax.axhline(0, color=ps.BASELINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ps.finish(fig, ax, "The load-bearing details, as learning curves",
              "environment steps", "episode return", OUT / "load_bearing.png")


if __name__ == "__main__":
    # A fork-safety guard, and it is not optional. `ProcessPoolExecutor` forks this
    # process, and forking a process that already has live OpenMP threads deadlocks:
    # the children inherit the lock state but not the threads holding it, and then
    # wait forever. ANY torch op in the parent — the ratio check below, a figure, a
    # throughput probe — is enough to start those threads. Pinning the parent to one
    # thread keeps the OpenMP pool from ever being created; each worker sets its own
    # thread count once it is safely inside its own process.
    torch.set_num_threads(1)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("run", "all"):
        cmd_run()
    if cmd in ("plot", "all"):
        cmd_plot()
