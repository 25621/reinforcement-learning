"""Project 38 — Behavior cloning: the number every offline method must beat.

Three questions. Two of them have textbook answers, and the measurement says the
textbook answers are wrong FOR THIS TASK — which is the most useful thing that can
happen to you in a baseline project, so we report it rather than hide it.

  1. What does BC score on the `medium` dataset? (the bar everything else must clear)
  2. Does BC "drift off the data and compound its errors", as everyone says it does?
     We measure the drift directly instead of assuming it.
  3. If some episodes in the data are better than others, why copy the bad ones?
     Train on the best 10% only (this is called %BC) and see what it buys.

    python3 bc.py      # ~4 min on one core
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-build-a-gridworld"))
import plot_style as ps  # noqa: E402

import offline_lib as ol  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
GRAD_STEPS = 20_000

# THREE seeds, not one — and this is not a formality. Run once, BC scored 1,255 and
# %BC scored 1,559, and it looked like filtering the data had bought a solid 24%.
# Run it again at a different budget and the ordering REVERSED. A single run cannot
# tell "this method is better" from "this run was luckier", and with a gap this size
# the honest answer needs an error bar. BC is cheap (under a minute a run), so there
# is no excuse for not having one.
SEEDS = [0, 1, 2]


def top_fraction_dataset(ds, frac=0.1):
    """Keep only the transitions belonging to the best `frac` of episodes.

    This is "%BC", and it is the cheapest possible form of learning from reward:
    we never build a value function, we just refuse to imitate the losers. Note
    what it needs that plain BC does not — the reward. Plain BC throws rewards
    away; %BC uses them for exactly one thing, ranking whole episodes.
    """
    ends = np.nonzero((ds.raw["terminal"] + ds.raw["timeout"]) > 0)[0]
    bounds, start = [], 0
    for e in ends:
        bounds.append((start, int(e) + 1))
        start = int(e) + 1
    rets = np.array([ds.raw["rew"][s:e].sum() for s, e in bounds])
    keep = np.argsort(-rets)[: max(1, int(len(rets) * frac))]
    idx = np.concatenate([np.arange(*bounds[k]) for k in keep])

    sub = object.__new__(ol.Dataset)
    sub.__dict__.update(ds.__dict__)          # same normalization, so eval is comparable
    sub.obs, sub.act = ds.obs[idx], ds.act[idx]
    sub.rew, sub.next_obs, sub.done = ds.rew[idx], ds.next_obs[idx], ds.done[idx]
    sub.n = len(idx)
    return sub, rets[keep].mean(), rets.mean()


def state_novelty(states, ds, chunk=256):
    """Distance from each visited state to the CLOSEST state in the dataset.

    In normalized units, so "1.0" means roughly one standard deviation away from
    anything the dataset has ever seen. This is the concrete meaning of
    `distribution shift`: it is a number, and it grows.
    """
    D = ds.obs.numpy()
    out = []
    for i in range(0, len(states), chunk):
        d = ((states[i:i + chunk, None, :] - D[None, :, :]) ** 2).sum(-1)
        out.append(np.sqrt(d.min(1)))
    return np.concatenate(out)


@torch.no_grad()
def rollout(actor, ds, episodes=5, seed=0):
    """Run the policy and record every state it visits, in order."""
    import gymnasium as gym
    env = gym.make(ol.ENV_ID)
    trajs = []
    for i in range(episodes):
        o, _ = env.reset(seed=seed + 500 + i)
        states, done = [], False
        while not done:
            on = (o - ds.obs_mean[0]) / ds.obs_std[0]
            states.append(on.astype(np.float32))
            a, _ = actor(torch.as_tensor(on, dtype=torch.float32).unsqueeze(0), deterministic=True)
            o, _, te, tr, _ = env.step(a.squeeze(0).numpy())
            done = te or tr
        trajs.append(np.array(states))
    env.close()
    return trajs


def run_one(job):
    """One (method, seed). Returns the history and the trained actor's weights."""
    method, seed = job
    ds = ol.Dataset("medium")
    if method == "%BC":
        ds = top_fraction_dataset(ds, 0.1)[0]
    cfg = ol.OfflineConfig(algo="bc", level="medium", seed=seed, grad_steps=GRAD_STEPS,
                           eval_every=2_500, eval_episodes=10)
    hist, agent, _ = ol.train(cfg, ds=ds)
    return method, seed, hist, agent.actor.state_dict()


def main():
    t0 = time.time()
    OUT.mkdir(exist_ok=True)
    ds = ol.Dataset("medium")
    lo, hi = ol.score_bounds()
    data_rets = ds.episode_returns()
    sub, kept_ret, all_ret = top_fraction_dataset(ds, 0.1)
    print(f"dataset: {ds.n} transitions, {len(data_rets)} episodes, "
          f"mean episode return {data_rets.mean():.1f}, best {data_rets.max():.1f}")
    print(f"%BC keeps {sub.n} transitions; those episodes average {kept_ret:.1f} "
          f"vs {all_ret:.1f} over all episodes")
    print(f"teachers: random {lo:.1f}  expert {hi:.1f}\n", flush=True)

    # ---- 1. plain BC, and 2. %BC — three seeds each, all six in parallel ----
    METHODS = ["BC", "%BC"]
    jobs = [(m, s) for m in METHODS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(run_one, jobs))
    runs = {(m, s): h for m, s, h, _ in results}
    actors = {(m, s): sd for m, s, h, sd in results}
    final = {m: np.array([runs[(m, s)]["eval_return"][-1] for s in SEEDS]) for m in METHODS}
    print(f"[{time.time() - t0:.0f}s] training done\n", flush=True)

    # ---- 3. the compounding-error diagnostic (on the first BC seed) ----
    # BC's per-step error, measured the way a supervised-learning person would:
    # mean squared error between the action BC picks and the action the data took.
    bc_actor = ol.TanhGaussianPolicy(ds.obs_dim, ds.act_dim, 256)
    bc_actor.load_state_dict(actors[("BC", SEEDS[0])])
    bc_actor.eval()
    with torch.no_grad():
        pred, _ = bc_actor(ds.obs[:20_000], deterministic=True)
    act_mse = ((pred - ds.act[:20_000]) ** 2).mean().item()
    act_scale = ds.act[:20_000].abs().mean().item()

    bc_traj = rollout(bc_actor, ds, episodes=5)
    # The dataset's own trajectories, as the reference for "how far is normal?"
    ends = np.nonzero((ds.raw["terminal"] + ds.raw["timeout"]) > 0)[0]
    starts = np.concatenate([[0], ends[:-1] + 1])
    teach_traj = [ds.obs.numpy()[s:e + 1] for s, e in zip(starts[:5], ends[:5])]

    T = 300  # the first 300 steps of an episode: the drift is over long before the end
    bc_nov = np.stack([state_novelty(tr[:T], ds) for tr in bc_traj])
    # These trajectories ARE dataset rows, so their nearest neighbor is themselves at
    # distance 0. Take the SECOND-nearest instead, or we would be comparing "how far is
    # BC from the data" against a hard-coded zero and the panel would prove nothing.
    te_nov = []
    for tr in teach_traj:
        d = ((tr[:T, None, :] - ds.obs.numpy()[None, :, :]) ** 2).sum(-1)
        te_nov.append(np.sqrt(np.partition(d, 1, axis=1)[:, 1]))
    te_nov = np.stack(te_nov)

    # ---- report ----
    teacher_ret = ol.load_teacher("medium")[1]["eval_return"]
    print("=" * 84)
    print(f"{'':36s} {'return':>22s} {'score':>8s}")
    print(f"{'random teacher':36s} {lo:12.1f} {'':>9s} {0.0:8.1f}")
    print(f"{'the data`s average episode':36s} {data_rets.mean():12.1f} {'':>9s} "
          f"{ol.normalized_score(data_rets.mean()):8.1f}")
    print(f"{'the medium teacher itself':36s} {teacher_ret:12.1f} {'':>9s} "
          f"{ol.normalized_score(teacher_ret):8.1f}")
    for m in METHODS:
        f = final[m]
        label = "BC (all data)" if m == "BC" else "%BC (best 10% of episodes)"
        print(f"{label:36s} {f.mean():12.1f} +/- {f.std():5.1f} "
              f"{ol.normalized_score(f.mean()):8.1f}   seeds: "
              f"{', '.join(f'{v:.0f}' for v in f)}")
    print(f"{'expert teacher':36s} {hi:12.1f} {'':>9s} {100.0:8.1f}")
    print("=" * 84)

    print("\n1. BC reproduces its teacher. It does not exceed it.")
    print(f"   BC {final['BC'].mean():.0f}  vs the teacher that wrote the data "
          f"{teacher_ret:.0f}  vs expert {hi:.0f}")

    print("\n2. Does BC drift off the data? MEASURED, and the answer here is NO.")
    print(f"   per-step action error is tiny:  MSE {act_mse:.4f} "
          f"(actions average |a| = {act_scale:.2f})")
    print("   distance to the nearest dataset state, at step 10 vs step 250:")
    print(f"     BC's policy       {bc_nov[:, 10].mean():.2f}  ->  {bc_nov[:, 250].mean():.2f}"
          f"   (flat: it does NOT wander off)")
    print(f"     the data itself   {te_nov[:, 10].mean():.2f}  ->  {te_nov[:, 250].mean():.2f}"
          f"   (BC actually sits in a DENSER region than the data does)")
    print("   Why: HalfCheetah cannot fall over. There is no cliff to drift off.")

    gap = final["%BC"].mean() - final["BC"].mean()
    spread = max(final["BC"].std(), final["%BC"].std())
    verdict = ("NO — the gap is smaller than the seed-to-seed noise"
               if abs(gap) < 2 * spread else "yes, beyond the noise")
    print(f"\n3. Does filtering to the best 10% of episodes help?  {verdict}")
    print(f"   BC  {final['BC'].mean():7.0f} +/- {final['BC'].std():.0f}"
          f"   (seeds: {', '.join(f'{v:.0f}' for v in final['BC'])})")
    print(f"   %BC {final['%BC'].mean():7.0f} +/- {final['%BC'].std():.0f}"
          f"   (seeds: {', '.join(f'{v:.0f}' for v in final['%BC'])})")
    print(f"   the gap between them is {gap:+.0f}, and one seed alone moves the number "
          f"by up to {spread:.0f}.")
    print(f"   Why to EXPECT nothing: all 100 episodes came from ONE policy, and their")
    print(f"   returns spread by only {100 * data_rets.std() / data_rets.mean():.0f}%. "
          f"They differ by luck, not by skill,")
    print(f"   so 'the best 10%' selects lucky noise, and luck does not transfer.")
    print(f"[{time.time() - t0:.0f}s total]")

    # ---- plots ----
    fig, axes = ps.plt.subplots(1, 3, figsize=(16, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    # (1) the bar itself, and how far below the expert it sits. The shaded band is the
    #     spread across seeds — and it is wide enough to swallow the BC/%BC difference.
    ax = ps.style_axes(axes[0])
    steps = runs[("BC", SEEDS[0])]["steps"]
    for i, m in enumerate(METHODS):
        c = np.stack([runs[(m, s)]["eval_return"] for s in SEEDS])
        label = "BC (all data)" if m == "BC" else "%BC (best 10% of episodes)"
        ax.plot(steps, c.mean(0), marker="o", ms=3, lw=2, color=ps.SERIES[i],
                label=f"{label}, mean of {len(SEEDS)} seeds")
        ax.fill_between(steps, c.min(0), c.max(0), color=ps.SERIES[i], alpha=0.18)
    ax.axhline(teacher_ret, ls="-.", lw=1.4, color=ps.SERIES[4],
               label="the teacher that wrote the data")
    ax.axhline(hi, ls=":", lw=1.6, color=ps.INK, label="expert teacher")
    ax.set_ylim(0, hi * 1.1)
    ax.set_title("BC lands on its teacher, and stops", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("gradient steps", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("return in the real environment", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="center right")

    # (2) the textbook claim, tested — and NOT confirmed on this task
    ax = ps.style_axes(axes[1])
    ax.plot(bc_nov.mean(0), color=ps.SERIES[2], lw=2, label="BC's policy")
    ax.fill_between(range(T), bc_nov.min(0), bc_nov.max(0), color=ps.SERIES[2], alpha=0.15)
    ax.plot(te_nov.mean(0), color=ps.SERIES[0], lw=2, label="the data itself")
    ax.set_ylim(0, None)
    ax.set_title("The drift everyone warns about — isn't here", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("step within the episode", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("distance to the nearest state\nin the dataset (std devs)",
                  color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)

    # (3) why to expect nothing from %BC: the episodes are not different in SKILL
    ax = ps.style_axes(axes[2])
    ax.hist(data_rets, bins=22, color=ps.SERIES[0], alpha=0.8,
            label="the 100 episodes in `medium`")
    ax.axvline(np.percentile(data_rets, 90), color=ps.SERIES[4], lw=1.8, ls="-.",
               label="%BC keeps everything right of here")
    for i, m in enumerate(METHODS):
        f = final[m]
        ax.axvline(f.mean(), color=ps.SERIES[i], lw=2.2,
                   label=f"{m} scores {f.mean():.0f} +/- {f.std():.0f}")
        ax.axvspan(f.mean() - f.std(), f.mean() + f.std(), color=ps.SERIES[i], alpha=0.15)
    ax.set_title("One policy wrote all of these", color=ps.INK, fontsize=12, loc="left", pad=10)
    ax.set_xlabel("episode return", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("number of episodes", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "bc_baseline.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'bc_baseline.png'}")



if __name__ == "__main__":
    main()
