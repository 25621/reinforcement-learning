"""Project 31 — PPO vs SAC on the same task, measured on both axes that matter.

"SAC is more sample-efficient than PPO" is true and is half a sentence. The other
half is that samples are not the resource you are always short of. There are two
different questions hiding in the word "efficient":

    Which learns more per ENVIRONMENT STEP?   -> matters when steps are expensive
                                                 (a real robot, a medical trial,
                                                 anything with a physical clock)
    Which learns more per SECOND?             -> matters when steps are cheap
                                                 (a fast simulator you can fork
                                                 across 64 cores)

They have different answers, and an RL paper that reports only the first is
answering a question its readers may not have asked. This file runs the same two
tasks under both algorithms and plots the same runs against both x-axes.

The asymmetry that drives everything: SAC does one gradient update per env step
(expensive per step, few steps needed). PPO does a handful of updates per
thousands of env steps (cheap per step, many steps needed).

Run:
    python3 sample_efficiency.py     # ~8 min on 12 cores
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-ddpg-on-pendulum"))
sys.path.insert(0, str(HERE.parent / "22-ppo-from-scratch"))
sys.path.insert(0, str(HERE.parent / "19-reinforce-on-cartpole"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import cc_lib as cc
import plot_style as ps
from ppo import PPOConfig, train_ppo

OUT = HERE / "outputs"
TASKS = ["HalfCheetah-v5"]
SEEDS = [0, 1, 2]
SAC_STEPS = 40_000
PPO_STEPS = 250_000
SAC_HIDDEN = 128

# The threshold for "reached a decent policy". Set well above what a random policy
# scores (-280) and below what either algorithm tops out at here, so both can cross it.
THRESHOLD = {"HalfCheetah-v5": 800}


def ppo_config(env_id):
    """PPO tuned for continuous control — CleanRL's `ppo_continuous_action` settings.

    These are NOT the CartPole settings from project 22. Continuous control needs
    observation and reward normalization (MuJoCo observations are unbounded joint
    velocities), a longer rollout, more epochs, and no entropy bonus.
    """
    return PPOConfig(
        env_id=env_id, total_steps=PPO_STEPS,
        n_envs=4, n_steps=512,          # 2048-step batch, as in the paper
        n_epochs=10, n_minibatches=32,
        lr=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_coef=0.2, ent_coef=0.0, vf_coef=0.5,
        hidden=64,
        norm_obs=True, norm_reward=True,  # load-bearing on MuJoCo; see project 22
    )


def run_one(args):
    algo, env_id, seed = args
    if algo == "SAC":
        cfg = cc.sac_config(env_id=env_id, seed=seed, total_steps=SAC_STEPS,
                            hidden=SAC_HIDDEN, eval_every=2_000, eval_episodes=5)
        hist, _ = cc.train(cfg)
        return algo, env_id, seed, dict(
            steps=np.asarray(hist["steps"]),
            curve=np.asarray(hist["eval_return"]),
            wall_curve=np.asarray(hist["wall_time"]),
            wall=hist["wall_total"],
        )
    r = train_ppo(ppo_config(env_id), seed=seed, threads=1)
    # PPO's per-step cost is constant (fixed rollout, fixed epoch count), so wall-clock
    # interpolates linearly across the curve. SAC's is measured directly.
    steps = r["steps"]
    return algo, env_id, seed, dict(
        steps=steps,
        curve=r["curve"],
        wall_curve=r["wall"] * steps / steps[-1],
        wall=r["wall"],
        final=r["final"],
    )


def first_crossing(steps, curve, xs, thr):
    """Where a smoothed curve first reaches the threshold. None if it never does."""
    k = 5
    if len(curve) >= k:
        sm = np.convolve(curve, np.ones(k) / k, mode="valid")
        xs_sm, steps_sm = xs[k - 1:], steps[k - 1:]
    else:
        sm, xs_sm, steps_sm = curve, xs, steps
    hit = np.nonzero(sm >= thr)[0]
    if len(hit) == 0:
        return None, None
    return steps_sm[hit[0]], xs_sm[hit[0]]


def main():
    OUT.mkdir(exist_ok=True)
    jobs = [(a, t, s) for t in TASKS for a in ("SAC", "PPO") for s in SEEDS]
    print(f"launching {len(jobs)} runs "
          f"(SAC {SAC_STEPS:,} steps, PPO {PPO_STEPS:,} steps)...", flush=True)
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(run_one, jobs))

    runs = {}
    for algo, env_id, seed, r in results:
        runs.setdefault(env_id, {}).setdefault(algo, {})[seed] = r

    def agg(env_id, algo, key):
        rs = runs[env_id][algo]
        n = min(len(rs[s][key]) for s in SEEDS)
        return np.stack([rs[s][key][:n] for s in SEEDS])

    # ---- the two-axis figure: same runs, plotted against samples and against seconds ----
    env_id = TASKS[0]
    fig, axes = ps.plt.subplots(1, 2, figsize=(13, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    colors = {"SAC": ps.SERIES[0], "PPO": ps.SERIES[1]}

    for col, xaxis in enumerate(["steps", "wall_curve"]):
        ax = ps.style_axes(axes[col])
        for algo in ("SAC", "PPO"):
            x = agg(env_id, algo, xaxis).mean(0)
            c = agg(env_id, algo, "curve")
            ax.plot(x, c.mean(0), color=colors[algo], lw=2.2, label=algo)
            ax.fill_between(x, c.min(0), c.max(0), color=colors[algo], alpha=0.15, lw=0)
        ax.axhline(THRESHOLD[env_id], color=ps.INK_MUTED, ls="--", lw=1.0)
        ax.annotate(f"threshold {THRESHOLD[env_id]}", (0.02, 0.04),
                    xycoords="axes fraction", fontsize=8, color=ps.INK_MUTED)
        ax.set_ylabel("evaluation return", color=ps.INK_SECONDARY, fontsize=9)
        if xaxis == "steps":
            ax.set_xscale("log")
            ax.set_xlabel("environment steps (log scale)",
                          color=ps.INK_SECONDARY, fontsize=9)
            ax.set_title("per SAMPLE  —  SAC wins by a mile", color=ps.INK,
                         fontsize=12, loc="left")
            ax.legend(frameon=False, fontsize=10, loc="upper left")
        else:
            ax.set_xlabel("wall-clock seconds (1 CPU core)",
                          color=ps.INK_SECONDARY, fontsize=9)
            ax.set_title("per SECOND  —  the gap narrows", color=ps.INK,
                         fontsize=12, loc="left")
    fig.suptitle(f"The same runs on {env_id.replace('-v5','')}, two x-axes",
                 color=ps.INK, fontsize=14, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "sample_vs_wallclock.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'sample_vs_wallclock.png'}")

    # ---- the numbers ----
    print("\n=== cost to reach the threshold return ===")
    for env_id in TASKS:
        thr = THRESHOLD[env_id]
        print(f"\n{env_id}  (threshold = {thr})")
        print(f"  {'algo':5s} {'env steps':>12s} {'seconds':>10s} {'final return':>14s}"
              f" {'steps/s':>9s}")
        ref = {}
        for algo in ("SAC", "PPO"):
            steps = agg(env_id, algo, "steps").mean(0)
            wall = agg(env_id, algo, "wall_curve").mean(0)
            curve = agg(env_id, algo, "curve").mean(0)
            s_at, t_at = first_crossing(steps, curve, wall, thr)
            final = curve[-3:].mean()
            total_wall = np.mean([runs[env_id][algo][s]["wall"] for s in SEEDS])
            tput = steps[-1] / total_wall
            ref[algo] = (s_at, t_at)
            s_txt = f"{s_at:,.0f}" if s_at else "not reached"
            t_txt = f"{t_at:,.0f}" if t_at else "-"
            print(f"  {algo:5s} {s_txt:>12s} {t_txt:>10s} {final:14.0f} {tput:9.0f}")
        if ref["SAC"][0] and ref["PPO"][0]:
            print(f"  -> SAC needs {ref['PPO'][0] / ref['SAC'][0]:.0f}x fewer SAMPLES; "
                  f"PPO needs {ref['SAC'][1] / ref['PPO'][1]:.1f}x "
                  f"{'fewer' if ref['PPO'][1] < ref['SAC'][1] else 'more'} SECONDS")

    print("\n=== throughput (the whole explanation) ===")
    for env_id in TASKS:
        for algo in ("SAC", "PPO"):
            steps = agg(env_id, algo, "steps").mean(0)
            total_wall = np.mean([runs[env_id][algo][s]["wall"] for s in SEEDS])
            print(f"  {env_id:16s} {algo:4s} {steps[-1] / total_wall:7.0f} env-steps/s "
                  f"({steps[-1]:,.0f} steps in {total_wall:.0f}s)")


if __name__ == "__main__":
    main()
