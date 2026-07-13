"""Project 27 — TD3 on HalfCheetah: the three fixes, and what a small budget can prove.

TD3 is DDPG plus three changes. Every tutorial lists them. Almost none say *how
much each one is worth*, which is the only question that matters when you are
deciding whether to keep a line of code.

Two commands, each under 10 minutes:

    python3 td3.py              # headline: TD3 vs DDPG on HalfCheetah, 4 seeds, 80k steps
    python3 td3.py ablation     # remove one fix at a time (5 variants x 2 seeds, 60k)

Read this before trusting any number below it. TD3's published win over DDPG is
measured at 1,000,000 steps with 10 seeds. This runs 80,000 with 4. That is 8% of
the samples and 40% of the seeds, and deep RL's seed-to-seed variance is *enormous*
— large enough, at this budget, to swallow the difference between two algorithms
whole. So this project is built to measure the thing a small budget CAN settle
decisively — the **mechanism**, i.e. whether each critic tells the truth about the
returns it predicts — and to be honest about the thing it cannot: the scoreboard.

Networks are 128 units wide, not the usual 256. That is a budget decision, not a
scientific one: it nearly doubles the update rate on a CPU, which buys twice the
steps. Every variant gets the same width, so the comparison stays fair.
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
ENV = "HalfCheetah-v5"
HIDDEN = 128

# Two bodies, and the second one is the whole reason this project is interesting.
# HalfCheetah CANNOT fall over — a bad policy just runs slowly — so an over-optimistic
# critic is never punished. Hopper CAN fall, which ends the episode, so an overconfident
# action is punished immediately. If TD3's fixes cure overestimation, the place to look
# for the cure working is the body where the disease is actually present.
BODIES = [("HalfCheetah-v5", 60_000), ("Hopper-v5", 40_000)]
MAIN_SEEDS = [0, 1, 2]

# Pinned per algorithm. The return panels list TD3 first and the bias panels list DDPG
# first (so the overestimating one is drawn underneath), and with positional colors that
# silently painted TD3 blue on the left and green on the right — the same algorithm in
# two colours on one figure.
COLORS = {"TD3": ps.SERIES[0], "DDPG": ps.SERIES[1]}

ABL_STEPS, ABL_SEEDS = 60_000, [0, 1]
ABLATIONS = ["TD3 -twin", "TD3 -delay", "TD3 -smoothing"]


def build(name, steps, seed, env_id=ENV):
    """Every variant is the same Config with different flags. That is the lesson."""
    common = dict(env_id=env_id, seed=seed, total_steps=steps, hidden=HIDDEN,
                  eval_every=5_000, eval_episodes=3, log_q_bias=True)
    if name == "DDPG":
        return cc.ddpg_config(**common)
    if name == "TD3":
        return cc.td3_config(**common)
    if name == "TD3 -twin":
        return cc.td3_config(twin_critics=False, **common)
    if name == "TD3 -delay":
        return cc.td3_config(policy_delay=1, **common)
    if name == "TD3 -smoothing":
        return cc.td3_config(target_noise=0.0, **common)
    raise ValueError(name)


def run_one(args):
    name, steps, seed, env_id = args
    hist, _ = cc.train(build(name, steps, seed, env_id))
    return (name, env_id), seed, hist


def collect(jobs):
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run_one, jobs))
    runs = {}
    for key, seed, h in results:
        runs.setdefault(key, {})[seed] = h
    return runs, results


def curves(runs, key):
    rs = runs[key]
    seeds = sorted(rs)
    c = np.stack([rs[s]["eval_return"] for s in seeds])
    return np.asarray(rs[seeds[0]]["steps"]), c


def bias_of(h, after):
    t = np.asarray(h["q_bias_steps"])
    m = t > after
    if not m.sum():
        return np.nan
    return np.asarray(h["q_pred"])[m].mean() - np.asarray(h["q_true"])[m].mean()


def main_experiment():
    jobs = [(n, steps, s, env) for env, steps in BODIES
            for n in ("TD3", "DDPG") for s in MAIN_SEEDS]
    print(f"launching {len(jobs)} runs across {len(BODIES)} bodies...", flush=True)
    runs, results = collect(jobs)

    # One row per body: the scoreboard on the left, the mechanism on the right.
    fig, axes = ps.plt.subplots(len(BODIES), 2, figsize=(13, 4.2 * len(BODIES)), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    for r, (env_id, steps_n) in enumerate(BODIES):
        short = env_id.replace("-v5", "")

        ax = ps.style_axes(axes[r][0])
        for name in ["TD3", "DDPG"]:
            steps, c = curves(runs, (name, env_id))
            for j in range(c.shape[0]):
                ax.plot(steps, c[j], color=COLORS[name], lw=0.8, alpha=0.35)
            ax.plot(steps, c.mean(0), color=COLORS[name], lw=2.6, label=f"{name} (mean)")
        ax.legend(frameon=False, fontsize=10, loc="upper left")
        ax.set_title(f"{short} — return (thin lines: individual seeds)",
                     color=ps.INK, fontsize=11, loc="left")
        ax.set_xlabel("environment steps", color=ps.INK_SECONDARY, fontsize=9)
        ax.set_ylabel("evaluation return", color=ps.INK_SECONDARY, fontsize=9)

        ax = ps.style_axes(axes[r][1])
        for name in ["DDPG", "TD3"]:
            for j, s in enumerate(sorted(runs[(name, env_id)])):
                h = runs[(name, env_id)][s]
                t = np.asarray(h["q_bias_steps"])
                bias = np.asarray(h["q_pred"]) - np.asarray(h["q_true"])
                k = 15
                if len(bias) > k:
                    sm = np.convolve(bias, np.ones(k) / k, mode="valid")
                    ts = t[k - 1:]
                    m = ts > steps_n // 5  # skip the untrained-critic transient
                    ax.plot(ts[m], sm[m], color=COLORS[name], lw=1.4, alpha=0.85,
                            label=name if j == 0 else None)
        ax.axhline(0, color=ps.INK, lw=1.4, ls="--")
        ax.annotate("above 0 = the critic promises more than it pays",
                    (0.03, 0.92), xycoords="axes fraction", fontsize=8,
                    color=ps.INK_SECONDARY)
        ax.legend(frameon=False, fontsize=10, loc="lower left")
        ax.set_title(f"{short} — critic bias", color=ps.INK, fontsize=11, loc="left")
        ax.set_xlabel("environment steps", color=ps.INK_SECONDARY, fontsize=9)
        ax.set_ylabel("predicted Q  -  actual return",
                      color=ps.INK_SECONDARY, fontsize=9)

    fig.suptitle("HalfCheetah cannot fall, so optimism is never punished. Hopper can.",
                 color=ps.INK, fontsize=13, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "td3_vs_ddpg.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'td3_vs_ddpg.png'}")

    for env_id, steps_n in BODIES:
        print(f"\n=== {env_id}: final return after {steps_n:,} steps "
              f"({len(MAIN_SEEDS)} seeds) ===")
        for name in ["TD3", "DDPG"]:
            _, c = curves(runs, (name, env_id))
            f = c[:, -3:].mean(1)
            print(f"{name:6s} mean {f.mean():7.0f}   std {f.std():6.0f}   "
                  f"seeds " + "  ".join(f"{v:6.0f}" for v in f))
        print(f"--- critic bias (predicted Q - actual return), second half ---")
        for name in ["TD3", "DDPG"]:
            b = [bias_of(runs[(name, env_id)][s], steps_n // 2)
                 for s in sorted(runs[(name, env_id)])]
            sign = "OVER" if np.mean(b) > 0 else "under"
            print(f"{name:6s} mean {np.mean(b):+8.1f}  ({sign}-estimates)   "
                  f"seeds " + "  ".join(f"{v:+7.1f}" for v in b))

    print(f"\nslowest run: {max(h['wall_total'] for _, _, h in results):.0f}s")


def ablation_experiment():
    # Run on Hopper, not HalfCheetah. On HalfCheetah the seed-to-seed spread at this
    # budget (std ~1100 on a mean of ~2000) is larger than any difference between the
    # variants, so a 2-seed ablation there measures noise and nothing else.
    env_id, steps_n = "Hopper-v5", 40_000
    order = ["TD3"] + ABLATIONS + ["DDPG"]
    jobs = [(n, steps_n, s, env_id) for n in order for s in ABL_SEEDS]
    print(f"launching {len(jobs)} runs of {steps_n:,} steps on {env_id}...", flush=True)
    runs, results = collect(jobs)

    fig, ax = ps.new_axes(7.8, 4.4)
    for i, name in enumerate(order):
        steps, c = curves(runs, (name, env_id))
        if name == "TD3":
            style = dict(lw=2.6, color=ps.INK)          # the reference
        elif name == "DDPG":
            style = dict(lw=1.6, color=ps.INK_MUTED, ls="--")  # the other reference
        else:
            style = dict(lw=1.5, color=ps.SERIES[ABLATIONS.index(name)])
        ax.plot(steps, c.mean(0), label=name, **style)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ps.finish(fig, ax,
              f"Removing one fix at a time ({env_id.replace('-v5','')}, "
              f"{len(ABL_SEEDS)} seeds)",
              "environment steps", "evaluation return (mean)", OUT / "ablation.png")

    print(f"\n=== ablation on {env_id}: final return after {steps_n:,} steps ===")
    print(f"{'variant':16s} {'mean':>7s} {'critic bias':>12s}   seeds")
    for name in order:
        _, c = curves(runs, (name, env_id))
        f = c[:, -3:].mean(1)
        b = np.mean([bias_of(runs[(name, env_id)][s], steps_n // 2)
                     for s in sorted(runs[(name, env_id)])])
        print(f"{name:16s} {f.mean():7.0f} {b:+12.1f}   "
              + "  ".join(f"{v:6.0f}" for v in f))
    print(f"\nslowest run: {max(h['wall_total'] for _, _, h in results):.0f}s")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == "ablation":
        ablation_experiment()
    else:
        main_experiment()
