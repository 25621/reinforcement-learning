r"""Project 49 — the noisy TV: build the trap on purpose, then watch curiosity fall in.

Same maze as projects 46 and 47, with one addition: a television in the corner of
the right-hand room. Stand next to it and part of your observation is replaced by a
fresh sheet of random static — a new, never-before-seen picture, every single step,
forever.

To a curiosity agent that measures novelty as "how badly did I predict this?", the
TV is a perpetual motion machine of surprise. It cannot be predicted, because there
is nothing to predict: the static is noise, not structure. So the error never falls,
so the bonus never fades, so the agent sits down and watches television.

Three bonuses, one maze, one question — who keeps working?

    RND               fingerprints the raw observation. Static -> new fingerprint
                      every step -> permanently high error.  PREDICTION: trapped.
    pixel prediction  predicts the raw next frame. Same story.  PREDICTION: trapped.
    ICM               predicts the next FEATURE vector, where the features are
                      trained to name the action you took. Static is the same
                      whatever you do, so it cannot help name the action, so it is
                      thrown out of the features. PREDICTION: immune.

    python3 noisy_tv.py     # ~8 min: 3 bonuses x 3 seeds
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "46-rnd-on-atari"))
sys.path.insert(0, str(HERE.parent / "47-icm"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import explore_lib as el  # noqa: E402
import plot_style as ps  # noqa: E402
from icm import ICM, PixelForward  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 300_000
SEEDS = (0, 1, 2)

ARMS = {
    "rnd": lambda c, a, s: el.RND(c, a, s),
    "pixel": lambda c, a, s: PixelForward(c, a, s),
    "icm": lambda c, a, s: ICM(c, a, s),
}
LABELS = {"rnd": "RND", "pixel": "pixel prediction", "icm": "ICM"}


def probe(bonus):
    return el.bonus_map(bonus, tv=True).reshape(-1)


def run_one(args):
    arm, seed = args
    h = el.train(make_bonus=ARMS[arm], total_steps=STEPS, seed=seed, tv=True,
                 verbose=False, probe=probe)
    print(f"  {LABELS[arm]:17s} seed {seed}: {h['tv_frac'][-1] * 100:5.1f}% of all steps spent "
          f"at the TV | success {h['success'][-1]:.2f} | coverage {h['coverage'][-1]}/122",
          flush=True)
    return arm, seed, h


def main(replot=False):
    OUT.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    torch.set_num_threads(1)

    if replot:                       # redraw the figures from the saved curves, no retraining
        p = dict(np.load(CKPT / "curves.npz"))
        plot_tv(p, OUT / "noisy_tv.png")
        plot_maps(p, OUT / "bonus_maps_tv.png")
        summarize(p)
        return

    jobs = [(arm, s) for arm in ARMS for s in SEEDS]
    print(f"[1/2] {len(jobs)} runs x {STEPS:,} steps in the maze WITH a television")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=9) as ex:
        results = list(ex.map(run_one, jobs))
    print(f"  ({time.time() - t0:.0f}s)")

    p = {}
    for arm, seed, h in results:
        for k in ("success", "coverage", "tv_frac", "bonus"):
            p[f"{arm}_{seed}_{k}"] = h[k]
        p[f"{arm}_{seed}_probe"] = np.stack(h["probe"])
        p["steps"] = h["steps"]
    np.savez(CKPT / "curves.npz", **p)

    print("[2/2] figures")
    plot_tv(p, OUT / "noisy_tv.png")
    plot_maps(p, OUT / "bonus_maps_tv.png")
    summarize(p)


def summarize(p):
    print(f"\n  an agent that IGNORED the TV would spend {el.tv_floor_share() * 100:.1f}% of its "
          f"steps beside it (the TV's share of the floor):\n")
    print(f"    {'':17s} {'TV time':>9s} {'solved':>8s} {'coverage':>9s}")
    for arm in ARMS:
        tv = [p[f"{arm}_{s}_tv_frac"][-1] * 100 for s in SEEDS]
        su = [p[f"{arm}_{s}_success"][-1] for s in SEEDS]
        cv = [p[f"{arm}_{s}_coverage"][-1] for s in SEEDS]
        print(f"    {LABELS[arm]:17s} {np.mean(tv):8.1f}% {np.mean(su):8.2f} {np.mean(cv):9.0f}")
    # The single cleanest statement in the whole project:
    hooked = [(a, s) for a in ARMS for s in SEEDS if p[f"{a}_{s}_tv_frac"][-1] > 0.20]
    solved = [(a, s) for a in ARMS for s in SEEDS if p[f"{a}_{s}_success"][-1] > 0.5]
    print(f"\n    runs that spent >20% of their life at the TV: {len(hooked)} "
          f"— of which solved the maze: {sum(1 for x in hooked if x in solved)}")
    print(f"    runs that solved the maze: {len(solved)}, TV time "
          f"{[round(p[f'{a}_{s}_tv_frac'][-1] * 100, 1) for a, s in solved]}%")


def plot_tv(p, path):
    fig, axes = ps.plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    steps = np.array(p["steps"]) / 1000
    for ax, key, ylab, title, scale in [
        (axes[0], "tv_frac", "share of all steps spent next to the TV", "Curiosity in the raw pixels cannot look away", 100),
        (axes[1], "success", "episodes solved (fraction)", "Nobody who got hooked ever solved it", 1),
    ]:
        for i, arm in enumerate(ARMS):
            runs = np.stack([p[f"{arm}_{s}_{key}"] for s in SEEDS]) * scale
            for r in runs:
                ax.plot(steps, r, color=ps.SERIES[i], lw=0.9, alpha=0.3)
            ax.plot(steps, runs.mean(0), color=ps.SERIES[i], lw=2.4, label=LABELS[arm])
        if key == "tv_frac":
            ax.axhline(el.tv_floor_share() * 100, color=ps.INK_MUTED, ls="--", lw=1.3,
                       label="the TV's share of the floor (what ignoring it looks like)")
        ps.style_axes(ax)
        ax.set_title(title, color=ps.INK, fontsize=11.5, loc="left")
        ax.set_xlabel("environment steps (thousands)", color=ps.INK_SECONDARY, fontsize=10)
        ax.set_ylabel(ylab, color=ps.INK_SECONDARY, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


def plot_maps(p, path):
    """What each bonus pays for arriving in each cell, at the end of training.

    Every map is divided by its own median cell, so the question is "how much MORE than
    an ordinary cell does this pay?" — comparing raw values across bonuses with different
    units would be meaningless. The colour scale is capped at 4x because one cell would
    otherwise flatten everything else: for pixel prediction, the KEY pays 26x, which is
    the most important thing in this figure and the reason that arm solved the maze.
    """
    env = el.KeyDoorRoom(tv=True)
    fig, axes = ps.plt.subplots(1, 3, figsize=(12.6, 3.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, arm in zip(axes, ARMS):
        heat = p[f"{arm}_{SEEDS[0]}_probe"][-1].reshape(env.h, env.w)
        med = np.nanmedian(heat)
        rel = heat / (med + 1e-12)
        im = ax.imshow(rel, cmap="magma", vmin=0, vmax=4.0)
        tv = np.nanmax(rel[el.TV[1] - 1:el.TV[1] + 2, el.TV[0] - 1:el.TV[0] + 2])
        key = rel[el.KEY[1], el.KEY[0]]
        for (x, y), name, col in [(el.TV, "TV", "#7ee6ff"), (el.KEY, "key", "#ffe27a")]:
            ax.scatter([x], [y], s=95, facecolor="none", edgecolor=col, lw=2.0)
            ax.annotate(name, (x, y - 1.1), color=col, ha="center", fontsize=9)
        ax.set_title(f"{LABELS[arm]}\nTV pays {tv:.1f}x  ·  key pays {key:.1f}x",
                     color=ps.INK, fontsize=10.5)
        ax.set_xticks([]), ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.85, extend="max")
    fig.suptitle("What each bonus pays, in multiples of an ordinary cell "
                 f"(after {STEPS // 1000}k steps; scale capped at 4x)",
                 color=ps.INK, fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main(replot="--plot" in sys.argv)
