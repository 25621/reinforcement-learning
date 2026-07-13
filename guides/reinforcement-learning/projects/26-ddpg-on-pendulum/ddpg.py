"""Project 26 — DDPG on Pendulum: verify it learns, then watch it wobble.

Three experiments, in the order the guide asks for them:

  main      5 seeds of DDPG on Pendulum. Does it learn? (Yes.) Do all the seeds
            agree? (No — and that disagreement is the point of the whole phase.)
  qbias     The critic's predicted Q at the start of each episode, against what
            the episode actually paid. DDPG's critic is not merely wrong; it is
            wrong in a *direction*, and the direction is up. This is the
            overestimation bias that TD3's twin critics exist to kill.
  explore   Exploration is bolted onto DDPG from outside, and it arrives in TWO
            pieces that tutorials rarely separate: the Gaussian action noise, and
            the `start_steps` warmup of uniform-random actions before the policy
            takes over. Turning the noise off alone changes nothing on this task.
            Turning off *both* is what breaks it. This experiment separates them,
            because "DDPG explores with action noise" turns out to be, on
            Pendulum, mostly false.

Run:
    python3 ddpg.py           # ~8 min on 12 cores (all three, in parallel)
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import cc_lib as cc
import plot_style as ps

OUT = HERE / "outputs"
SEEDS = [0, 1, 2, 3, 4]
# Pendulum is solved (and flat) well before 15k steps; 20k cost 10.5 min of wall clock
# across 14 parallel runs and bought nothing but a longer plateau.
STEPS = 15_000


# (act_noise, start_steps, label). The last row is the one that matters: it removes
# BOTH sources of exploration, which is the only way to find out which one was working.
EXPLORE = [
    (0.1, 1_000, "noise 0.1 + warmup (standard DDPG)"),
    (0.3, 1_000, "noise 0.3 + warmup"),
    (0.0, 1_000, "no noise, warmup kept"),
    (0.0, 0, "no noise, NO warmup"),
]


def run_one(args):
    """One training run in one process. Must be top-level so it can be pickled."""
    kind, seed, act_noise, start_steps = args
    cfg = cc.ddpg_config(
        env_id="Pendulum-v1",
        seed=seed,
        total_steps=STEPS,
        act_noise=act_noise,
        start_steps=start_steps,
        update_after=max(start_steps, 1_000),
        eval_every=1_000,
        log_q_bias=(kind == "main"),
    )
    hist, _ = cc.train(cfg)
    return kind, seed, (act_noise, start_steps), hist


def main():
    OUT.mkdir(exist_ok=True)

    jobs = [("main", s, 0.1, 1_000) for s in SEEDS]
    # The explore sweep reuses the standard runs above for its first row rather than
    # repeating them: same config, same seeds, so re-running would waste 3 processes.
    jobs += [("explore", s, n, w) for n, w, _ in EXPLORE[1:] for s in SEEDS[:3]]

    print(f"launching {len(jobs)} runs on {min(len(jobs), 12)} cores...", flush=True)
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run_one, jobs))
    print("done training", flush=True)

    main_runs = {s: h for k, s, _, h in results if k == "main"}
    explore_runs = {}
    for k, s, key, h in results:
        if k == "explore":
            explore_runs.setdefault(key, {})[s] = h
    explore_runs[(0.1, 1_000)] = {s: main_runs[s] for s in SEEDS[:3]}

    # ---- 1. learning curves, all 5 seeds ----
    fig, ax = ps.new_axes(7.6, 4.4)
    steps = np.asarray(main_runs[SEEDS[0]]["steps"])
    curves = np.stack([main_runs[s]["eval_return"] for s in SEEDS])
    for i, s in enumerate(SEEDS):
        ax.plot(steps, curves[i], color=ps.SERIES[i % len(ps.SERIES)],
                lw=1.1, alpha=0.85, label=f"seed {s}")
    ax.plot(steps, curves.mean(0), color=ps.INK, lw=2.4, label="mean")
    ax.axhline(-150, color=ps.INK_MUTED, ls="--", lw=1.0)
    ax.annotate("-150 ~ 'solved'", (steps[-1], -150), textcoords="offset points",
                xytext=(-8, 8), ha="right", color=ps.INK_MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    ps.finish(fig, ax, "DDPG on Pendulum: it learns, and the seeds disagree",
              "environment steps", "evaluation return", OUT / "learning_curves.png")

    # ---- 2. overestimation: predicted Q vs realized return ----
    # Plotted as a scatter against the y=x line, not as two time series: Pendulum
    # resets to a random angle, so both quantities swing by hundreds from episode to
    # episode and a time-series plot buries the bias in start-state noise. What we
    # care about is not the level, it is which side of the diagonal the points fall on.
    fig, axes = ps.plt.subplots(1, 2, figsize=(13, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    ax = ps.style_axes(axes[0])
    all_true = []
    for i, s in enumerate(SEEDS):
        h = main_runs[s]
        t = np.asarray(h["q_bias_steps"])
        pred, true = np.asarray(h["q_pred"]), np.asarray(h["q_true"])
        m = t > 5_000  # after the critic has found the right order of magnitude
        all_true.append(true[m])
        ax.scatter(true[m], pred[m], s=9, alpha=0.55,
                   color=ps.SERIES[i % len(ps.SERIES)], lw=0, label=f"seed {s}")
    # Data-driven limits: a fixed [-800, 50] window would squeeze every point into one
    # corner, because after 5k steps the returns live in a narrow band.
    lo = float(np.percentile(np.concatenate(all_true), 1)) - 20
    lim = [lo, 20]
    ax.plot(lim, lim, color=ps.INK, lw=1.4, ls="--")
    ax.annotate("y = x: a perfectly\ncalibrated critic", (0.52, 0.13),
                xycoords="axes fraction", fontsize=8, color=ps.INK_SECONDARY)
    ax.annotate("above the line =\nthe critic promises\nmore than it delivers",
                (0.06, 0.72), xycoords="axes fraction", fontsize=8.5, color=ps.INK)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Every seed sits above the diagonal", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("what the episode actually paid (discounted return)",
                  color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("what the critic predicted, Q(s0, a0)",
                  color=ps.INK_SECONDARY, fontsize=9)

    ax = ps.style_axes(axes[1])
    # Start at 5k. Before that the critic is simply untrained and its bias is in the
    # hundreds, which would flatten the steady-state bias — the actual finding — into
    # an invisible line along the x-axis.
    for i, s in enumerate(SEEDS):
        h = main_runs[s]
        t = np.asarray(h["q_bias_steps"])
        bias = np.asarray(h["q_pred"]) - np.asarray(h["q_true"])
        k = 15
        if len(bias) > k:
            sm = np.convolve(bias, np.ones(k) / k, mode="valid")
            ts = t[k - 1:]
            m = ts > 5_000
            ax.plot(ts[m], sm[m], color=ps.SERIES[i % len(ps.SERIES)], lw=1.5,
                    label=f"seed {s}")
    ax.axhline(0, color=ps.INK, lw=1.4, ls="--")
    ax.annotate("zero bias = an honest critic", (0.03, 0.06),
                xycoords="axes fraction", fontsize=8, color=ps.INK_SECONDARY)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("...and stays there (bias after 5k steps, smoothed)",
                 color=ps.INK, fontsize=11, loc="left")
    ax.set_xlabel("environment steps", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylabel("predicted Q  -  actual return", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "q_overestimation.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'q_overestimation.png'}")

    # ---- 3. where does DDPG's exploration actually come from? ----
    fig, ax = ps.new_axes(7.6, 4.4)
    # Explicit colors: the two "no noise" rows must not both land on red, or the one
    # row that actually fails becomes indistinguishable from the one that does not.
    colors = [ps.SERIES[0], ps.SERIES[1], ps.SERIES[4], ps.SERIES[2]]
    for i, (n, w, label) in enumerate(EXPLORE):
        runs = explore_runs[(n, w)]
        c = np.stack([runs[s]["eval_return"] for s in sorted(runs)])
        style = (dict(color=colors[3], lw=2.6) if w == 0
                 else dict(color=colors[i], lw=1.5))
        ax.plot(steps, c.mean(0), label=label, **style)
        ax.fill_between(steps, c.min(0), c.max(0),
                        color=style["color"], alpha=0.13, lw=0)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ps.finish(fig, ax,
              "The action noise is not what explores. The random warmup is.",
              "environment steps", "evaluation return (3 seeds, min-max band)",
              OUT / "exploration_ablation.png")

    # ---- the numbers the README quotes ----
    final = curves[:, -3:].mean(1)
    print("\n=== DDPG on Pendulum, final return (mean of last 3 evals) ===")
    for s, f in zip(SEEDS, final):
        print(f"  seed {s}: {f:8.1f}")
    print(f"  mean {final.mean():.1f}   best {final.max():.1f}   worst {final.min():.1f}"
          f"   spread {final.max() - final.min():.1f}")

    print("\n=== overestimation (mean over episodes after step 5k) ===")
    for s in SEEDS[:3]:
        h = main_runs[s]
        t = np.asarray(h["q_bias_steps"])
        pred, true = np.asarray(h["q_pred"]), np.asarray(h["q_true"])
        m = t > 5_000
        print(f"  seed {s}: predicted {pred[m].mean():8.1f}   actual {true[m].mean():8.1f}"
              f"   bias {pred[m].mean() - true[m].mean():+7.1f}")

    print("\n=== where the exploration comes from ===")
    for n, w, label in EXPLORE:
        runs = explore_runs[(n, w)]
        c = np.stack([runs[s]["eval_return"] for s in sorted(runs)])
        f = c[:, -3:].mean(1)
        print(f"  {label:36s} final {f.mean():8.1f}   "
              f"(seeds: {'  '.join(f'{v:7.1f}' for v in f)})")

    wall = max(h["wall_total"] for _, _, _, h in results)
    print(f"\nslowest run: {wall:.0f}s")


if __name__ == "__main__":
    main()
