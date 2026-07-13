"""Project 32 — PETS / random-shooting MPC on Pendulum.

Three questions, in the order a skeptic would ask them:

  1. Does planning through a learned model actually control the pendulum, and how
     many real environment steps does it need compared to SAC (Phase 5's best)?
  2. The ensemble is the "P" in PETS. Does it earn its keep, or is one network fine?
  3. The model is trained on one-step transitions but the planner rolls it 15 steps.
     How badly does the error compound over those 15 steps?

Experiment 3 is the one that matters beyond this project: the answer is what forces
MBPO (project 34) to keep its rollouts short and TD-MPC2 (project 37) to bolt a value
function onto the end of a 3-step horizon.

  python3 pets.py        # ~4 min on 12 hyperthreads
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-ddpg-on-pendulum"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import cc_lib as cc  # noqa: E402  — Phase 5's SAC, used here as the model-free baseline
import mbrl_lib as M  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"

# Three seeds, not five. This box has 6 physical cores, and the arms below already
# fill them; a fourth and fifth seed would run *concurrently with* the others and slow
# every process down by more than the extra seeds are worth. Measured: the same PETS
# run takes 98s alone and 361s with ten processes competing for six cores.
SEEDS = [0, 1, 2]

# 1 random warmup episode + 9 planned episodes = 2,000 real steps. Pendulum episodes
# are a fixed 200 steps, so "episode" and "200 env steps" are the same axis here.
N_WARMUP = 1
N_EPISODES = 9

PLAN = M.PlanConfig(horizon=15, n_candidates=400, n_iters=1, gamma=0.99)


def pets_run(seed, n_members=5, want_model=False):
    cfg = M.MBRLConfig(
        seed=seed,
        n_members=n_members,
        n_warmup_episodes=N_WARMUP,
        n_episodes=N_EPISODES,
        plan=PLAN,
    )
    h = M.run_mpc(cfg, M.RandomShooting)
    out = {
        "seed": seed,
        "n_members": n_members,
        "steps": h["steps"],
        "return": h["return"],
        "hold_mse": h["hold_mse"],
        "wall_total": h["wall_total"],
    }
    # Experiment 3 needs a trained model. Ship this one home rather than training a
    # third one from scratch — that is 6 minutes of the budget saved for free. Weights
    # pickle fine across the process boundary; the live module would not.
    if want_model:
        out["state_dict"] = h["model"].state_dict()
        obs, _, _, _ = h["buffer"].arrays()
        out["obs"] = obs.copy()
    return out


def sac_run(seed):
    """Phase 5's SAC, unchanged, on the same task — the model-free yardstick."""
    cfg = cc.sac_config(
        env_id="Pendulum-v1",
        seed=seed,
        total_steps=15_000,
        hidden=128,
        eval_every=500,
        eval_episodes=3,
    )
    hist, _ = cc.train(cfg)
    return {"seed": seed, "steps": hist["steps"], "return": hist["eval_return"]}


# --------------------------------------------------------------------------
# Experiment 3: how fast does the model's error compound?
# --------------------------------------------------------------------------
def compounding_error(model, seed=0, max_k=30):
    """Ask a trained model to predict k steps ahead, and check it against reality.

    The model is only ever *trained* on one-step transitions. But every planner in this
    phase rolls it forward 15 steps and trusts the result. So: from real states along a
    real trajectory, feed the model the actions that were actually taken, let it imagine
    k steps with no correction, and measure how far its imagined state has drifted from
    the state the real pendulum reached.

    Also recorded: how much the 5 ensemble members disagree with *each other* at step k.
    That number needs no ground truth at all — which is the point. It is the model's own
    estimate of how lost it is, available at planning time, when the truth is not.
    """
    import gymnasium as gym

    env = gym.make("Pendulum-v1")
    env.action_space.seed(seed + 99)
    gen = torch.Generator().manual_seed(seed + 4242)

    # A fresh on-policy trajectory: the states the planner actually visits are the
    # states whose predictions actually matter.
    planner = M.RandomShooting(1, 2.0, PLAN)
    o, _ = env.reset(seed=seed + 99)
    real_obs, real_acts = [o.copy()], []
    for _ in range(200):
        a, _ = planner.plan(model, torch.as_tensor(o, dtype=torch.float32), gen)
        a = np.clip(a, -2.0, 2.0)
        o, r, term, trunc, _ = env.step(a)
        real_acts.append(np.asarray(a, dtype=np.float32))
        real_obs.append(o.copy())
        if term or trunc:
            break
    env.close()

    real_obs = np.array(real_obs, dtype=np.float32)
    real_acts = np.array(real_acts, dtype=np.float32)
    T = len(real_acts)

    starts = np.arange(0, T - max_k)
    E = model.n_members
    # Every start state, imagined forward in lock-step: (E, n_starts, obs_dim)
    o_img = torch.as_tensor(real_obs[starts]).unsqueeze(0).expand(E, -1, -1).contiguous()

    err = np.zeros(max_k + 1)
    disagree = np.zeros(max_k + 1)
    with torch.no_grad():
        for k in range(1, max_k + 1):
            a = torch.as_tensor(real_acts[starts + k - 1]).unsqueeze(0).expand(E, -1, -1)
            o_img, _ = model.predict(o_img, a, sample=False)
            truth = torch.as_tensor(real_obs[starts + k])
            pred = o_img.mean(0)  # the ensemble's consensus prediction
            err[k] = (pred - truth).pow(2).sum(-1).sqrt().mean().item()
            # Spread of the members around their own mean — no ground truth used.
            disagree[k] = o_img.std(0).mean().item()
    return err, disagree


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    # ---- everything that needs a core, launched at once ----
    # PETS (3 seeds), the 1-member ablation (3 seeds), and SAC (3 seeds) are all
    # independent, so they go into one pool together instead of three pools in
    # sequence. The 5-member arm of the ablation IS the PETS run — same config — so it
    # is read off these results rather than recomputed.
    print("=== PETS vs SAC (sample efficiency) ===", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        pets_futs = [ex.submit(pets_run, s, 5, s == 0) for s in SEEDS]
        solo_futs = [ex.submit(pets_run, s, 1) for s in SEEDS]
        sac_futs = [ex.submit(sac_run, s) for s in SEEDS]
        pets = [f.result() for f in pets_futs]
        solo = [f.result() for f in solo_futs]
        sacs = [f.result() for f in sac_futs]

    pets_ret = np.array([p["return"] for p in pets])          # (seeds, episodes)
    pets_steps = np.array(pets[0]["steps"])
    sac_ret = np.array([s["return"] for s in sacs])
    sac_steps = np.array(sacs[0]["steps"])

    for p in pets:
        print(f"  PETS seed {p['seed']}: final {p['return'][-1]:8.1f}  "
              f"({p['wall_total']:.0f}s)", flush=True)
    print(f"  PETS mean final return : {pets_ret[:, -1].mean():8.1f}")
    print(f"  SAC  mean final return : {sac_ret[:, -1].mean():8.1f}")

    def first_step_reaching(steps, ret, thresh=-300):
        """Env steps needed before the mean return first clears `thresh`."""
        m = ret.mean(0)
        idx = np.where(m >= thresh)[0]
        return steps[idx[0]] if len(idx) else None

    p_at = first_step_reaching(pets_steps, pets_ret)
    s_at = first_step_reaching(sac_steps, sac_ret)
    print(f"\n  env steps to mean return >= -300:")
    print(f"    PETS : {p_at}")
    print(f"    SAC  : {s_at}")
    if p_at and s_at:
        print(f"    PETS is {s_at / p_at:.1f}x more sample-efficient")

    fig, ax = ps.new_axes(7.6, 4.4)
    ax.plot(sac_steps, sac_ret.mean(0), color=ps.SERIES[2], lw=2, label="SAC (model-free)")
    ax.fill_between(sac_steps, sac_ret.min(0), sac_ret.max(0), color=ps.SERIES[2], alpha=0.15)
    ax.plot(pets_steps, pets_ret.mean(0), color=ps.SERIES[0], lw=2,
            marker="o", ms=4, label="PETS (random-shooting MPC)")
    ax.fill_between(pets_steps, pets_ret.min(0), pets_ret.max(0), color=ps.SERIES[0], alpha=0.15)
    ax.axhline(-150, color=ps.INK_MUTED, ls="--", lw=1)
    ax.text(15_000, -150, " solved", color=ps.INK_MUTED, fontsize=8, va="center", ha="right")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "PETS reaches a working policy with a fraction of the real steps",
              "environment steps (log scale)", "episode return", OUT / "learning_curves.png")

    # ---- experiment 2: does the ensemble matter? ----
    print("\n=== ensemble ablation (1 member vs 5) ===", flush=True)
    abl = {1: solo, 5: pets}

    fig, ax = ps.new_axes(7.2, 4.2)
    for i, n in enumerate((1, 5)):
        r = np.array([a["return"] for a in abl[n]])
        st = np.array(abl[n][0]["steps"])
        label = f"{n} member" + ("s (PETS)" if n > 1 else " (no uncertainty)")
        ax.plot(st, r.mean(0), color=ps.SERIES[i * 2], lw=2, marker="o", ms=4, label=label)
        ax.fill_between(st, r.min(0), r.max(0), color=ps.SERIES[i * 2], alpha=0.15)
        print(f"  {n} member(s): final return {r[:, -1].mean():8.1f} "
              f"(seeds: {', '.join(f'{v:.0f}' for v in r[:, -1])})  "
              f"holdout_mse {np.mean([a['hold_mse'][-1] for a in abl[n]]):.4f}")
    ax.axhline(-150, color=ps.INK_MUTED, ls="--", lw=1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "The ensemble is what makes the plan trustworthy",
              "environment steps", "episode return", OUT / "ensemble_ablation.png")

    # ---- experiment 3: compounding error ----
    print("\n=== compounding model error ===", flush=True)
    trained = M.GaussianEnsemble(5, 3, 1, hidden=200, n_layers=3)
    trained.load_state_dict(pets[0]["state_dict"])  # the seed-0 model, already trained
    err, disagree = compounding_error(trained, seed=0, max_k=30)
    ks = np.arange(len(err))
    for k in (1, 5, 10, 15, 20, 30):
        print(f"  k={k:2d} steps ahead:  state error {err[k]:6.3f}   "
              f"ensemble disagreement {disagree[k]:6.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)

    axes[0].plot(ks[1:], err[1:], color=ps.SERIES[0], lw=2)
    axes[0].axvline(PLAN.horizon, color=ps.SERIES[2], ls="--", lw=1.2)
    axes[0].text(PLAN.horizon + 0.4, err[1:].max() * 0.5,
                 f"planning\nhorizon = {PLAN.horizon}", color=ps.SERIES[2], fontsize=8)
    axes[0].set_title("Imagined state drifts from reality", color=ps.INK, fontsize=11, loc="left")
    axes[0].set_xlabel("steps imagined without correction (k)", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_ylabel("distance from the true state", color=ps.INK_SECONDARY, fontsize=10)

    axes[1].plot(ks[1:], disagree[1:], color=ps.SERIES[4], lw=2)
    axes[1].axvline(PLAN.horizon, color=ps.SERIES[2], ls="--", lw=1.2)
    axes[1].set_title("The ensemble knows it, without being told", color=ps.INK, fontsize=11, loc="left")
    axes[1].set_xlabel("steps imagined without correction (k)", color=ps.INK_SECONDARY, fontsize=10)
    axes[1].set_ylabel("disagreement between members", color=ps.INK_SECONDARY, fontsize=10)

    fig.tight_layout()
    fig.savefig(OUT / "compounding_error.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'compounding_error.png'}")


if __name__ == "__main__":
    main()
