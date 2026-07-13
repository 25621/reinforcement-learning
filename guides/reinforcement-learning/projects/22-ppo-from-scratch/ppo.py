"""PPO, written out in full, with every implementation detail behind a named flag.

This is the reference implementation for the rest of the phase: project 23
ablates its flags one at a time, project 24 swaps its network for a CNN, and
project 25 uses it as the yardstick TRPO is measured against.

The algorithm is five lines (see `ppo_losses`). The other two hundred are the
reason it works, and every one of them is a flag on `PPOConfig` so that project
23 can turn it off and watch what happens.

The single idea:

    A2C (project 21) throws a rollout away after one gradient step, because the
    policy has moved and the data is now off-policy. That is enormously wasteful
    — the data cost a walk through the environment; the gradient step cost a
    microsecond. PPO reuses the batch for several epochs, and pays for the
    resulting off-policy-ness with an importance ratio π_new/π_old. Left alone,
    that ratio would let a single batch drag the policy arbitrarily far. So PPO
    clips it: beyond ±ε, the objective goes flat, its gradient vanishes, and the
    batch stops pulling. Trust region by way of a `clamp`.

Runtime: ~8 min on 12 CPU cores (`python3 ppo.py all`).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19-reinforce-on-cartpole"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import plot_style as ps
from pg_lib import ActorCritic, evaluate, make_vec_env, rollout, set_seed, space_dims

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


@dataclass
class PPOConfig:
    """Every knob, with the value the papers and CleanRL actually use.

    The flags below the fold are the ones project 23 ablates. They default to
    ON, which is what "PPO" means in practice — a PPO with these switched off is
    an algorithm that shares a name with the one in the paper and nothing else.
    """
    env_id: str = "CartPole-v1"
    total_steps: int = 500_000    # CleanRL's budget: 976 updates, not 146
    n_envs: int = 4               # detail: vectorized envs
    n_steps: int = 128            # rollout length per env
    n_epochs: int = 4             # how many times each batch is reused
    n_minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 64

    # --- the ablatable details ---
    anneal_lr: bool = True        # linear decay of the learning rate to 0
    gae: bool = True              # GAE(λ) vs plain n-step returns
    norm_adv: bool = True         # per-minibatch advantage normalization
    clip_vloss: bool = True       # the clipped value loss
    ortho_init: bool = True       # orthogonal init, policy head gain 0.01
    adam_eps: float = 1e-5        # NOT torch's 1e-8 default
    clip_ratio: bool = True       # the clipped surrogate itself
    norm_obs: bool = False        # running obs normalization (continuous control)
    norm_reward: bool = False     # running reward scaling (continuous control)
    target_kl: float = None       # early-stop the epoch loop if KL blows past this

    @property
    def batch_size(self):
        return self.n_envs * self.n_steps

    @property
    def minibatch_size(self):
        return self.batch_size // self.n_minibatches


def ppo_losses(cfg, new_logp, old_logp, entropy, v_pred, v_old, adv, ret):
    """The five lines, plus the two that people forget.

    ratio = exp(new_logp - old_logp)
        In log-space because the probabilities themselves underflow. At the very
        first epoch of a batch the policy has not moved, so ratio ≡ 1 exactly —
        a useful assertion when debugging: if your first-epoch ratio is not 1.0,
        your stored log-probs are wrong.

    The clipped objective takes the *pessimistic* branch:
        -min(ratio·A, clip(ratio)·A)
        For A > 0 it caps how much the probability of a good action can be raised;
        for A < 0 it caps how much a bad action's can be cut. `min` (not `max`)
        is what makes it a lower bound on the true objective — PPO always
        believes the less flattering of the two stories.
    """
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()

    if cfg.norm_adv:
        # Per-minibatch, not per-batch. Advantages have no natural scale — only
        # their signs and relative sizes carry information — so standardizing
        # them makes one learning rate work across environments whose rewards
        # differ by orders of magnitude.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    if cfg.clip_ratio:
        pg1 = -adv * ratio
        pg2 = -adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
        pg_loss = torch.max(pg1, pg2).mean()
    else:
        pg_loss = (-adv * ratio).mean()          # plain importance-weighted PG: no guard rail

    if cfg.clip_vloss:
        # The value function gets its own trust region: it may not move further
        # than clip_coef from what it predicted when the data was collected.
        v_unclipped = (v_pred - ret) ** 2
        v_clipped = v_old + torch.clamp(v_pred - v_old, -cfg.clip_coef, cfg.clip_coef)
        v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - ret) ** 2).mean()
    else:
        v_loss = 0.5 * ((v_pred - ret) ** 2).mean()

    ent_loss = -entropy.mean()
    loss = pg_loss + cfg.vf_coef * v_loss + cfg.ent_coef * ent_loss

    with torch.no_grad():
        # http://joschu.net/blog/kl-approx.html — the k3 estimator. The naive
        # (-log_ratio).mean() is unbiased but can go negative, which makes it
        # useless as a "how far have I moved" alarm.
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clipfrac = ((ratio - 1.0).abs() > cfg.clip_coef).float().mean()
    return loss, dict(pg=pg_loss.item(), v=v_loss.item(), ent=-ent_loss.item(),
                      kl=approx_kl.item(), clipfrac=clipfrac.item())


def train_ppo(cfg, seed=0, threads=1, log_every=5, agent=None, envs=None, verbose=False):
    torch.set_num_threads(threads)
    close_envs = envs is None
    if envs is None:
        envs = make_vec_env(cfg.env_id, cfg.n_envs, seed, gamma=cfg.gamma,
                            norm_obs=cfg.norm_obs, norm_reward=cfg.norm_reward,
                            clip_action=True)
    obs_dim, act_dim, cont = space_dims(envs)
    set_seed(seed)
    if agent is None:
        agent = ActorCritic(obs_dim, act_dim, continuous=cont, hidden=cfg.hidden,
                            ortho=cfg.ortho_init, policy_gain=0.01)
    optim = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=cfg.adam_eps)

    obs, _ = envs.reset(seed=seed)
    n_updates = cfg.total_steps // cfg.batch_size
    steps, curve, kls, clipfracs = [], [], [], []
    recent = []
    t0 = time.time()

    for update in range(n_updates):
        if cfg.anneal_lr:
            # Linear decay to exactly zero at the end of training. The intuition
            # is that PPO's guarantee is local — a big step is only safe while
            # the policy is bad — so the step size should shrink as the policy
            # gets good and there is more to lose.
            frac = 1.0 - update / n_updates
            optim.param_groups[0]["lr"] = frac * cfg.lr

        ro, obs = rollout(envs, agent, obs, cfg.n_steps, cfg.gamma, cfg.gae_lambda,
                          use_gae=cfg.gae)
        recent.extend(ro.ep_returns)

        idx = np.arange(cfg.batch_size)
        stop = False
        for epoch in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                _, new_logp, entropy, v_pred = agent.act(ro.obs[mb], ro.actions[mb])
                loss, info = ppo_losses(cfg, new_logp, ro.logps[mb], entropy, v_pred,
                                        ro.values[mb], ro.advantages[mb], ro.returns[mb])
                optim.zero_grad()
                loss.backward()
                # One clip over ALL parameters jointly (not per-tensor): it
                # rescales the whole update direction rather than distorting it.
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optim.step()
            if cfg.target_kl is not None and info["kl"] > cfg.target_kl:
                stop = True
                break

        kls.append(info["kl"])
        clipfracs.append(info["clipfrac"])
        if (update + 1) % log_every == 0:
            steps.append((update + 1) * cfg.batch_size)
            curve.append(np.mean(recent[-40:]) if recent else np.nan)
            if verbose:
                print(f"  step {(update + 1) * cfg.batch_size:7d}  return {curve[-1]:7.1f}  "
                      f"kl {info['kl']:.4f}  clipfrac {info['clipfrac']:.2f}", flush=True)

    obs_rms = None
    if cfg.norm_obs:
        e = envs.envs[0]
        while not hasattr(e, "obs_rms"):
            e = e.env
        obs_rms = e.obs_rms
    if close_envs:
        envs.close()
    final, final_std = evaluate(agent, cfg.env_id, n_episodes=20, seed=9999,
                                norm_obs=cfg.norm_obs, obs_rms=obs_rms,
                                clip_action=True)
    return dict(steps=np.asarray(steps), curve=np.asarray(curve), final=final,
                final_std=final_std, kl=np.asarray(kls), clipfrac=np.asarray(clipfracs),
                wall=time.time() - t0, agent=agent)


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------

CARTPOLE = PPOConfig()

# LunarLander needs three changes from the CartPole defaults, and each one is a
# lesson rather than a knob-twiddle:
#
#   gamma 0.999, lambda 0.98 -- the +100 landing bonus arrives hundreds of steps
#       after the thruster firings that earned it. At gamma=0.99 the effective
#       horizon is ~100 steps and the agent literally cannot see the landing pad
#       from where it is; it learns to hover instead.
#
#   norm_reward=True -- and this one is NOT optional, for a reason that has
#       nothing to do with the reward and everything to do with detail #11.
#       LunarLander's returns run into the hundreds, so the CRITIC's gradient norm
#       is ~50 while the ACTOR's is ~0.13. Global gradient clipping (#11) clips
#       them JOINTLY, so the clip rescales everything by 0.5/50 = 0.01 and the
#       actor's effective learning rate becomes 3e-6. The policy cannot move.
#       Normalizing the reward brings the critic's gradient to ~1.0, the clip
#       factor to ~0.5, and the agent to life. Project 23 measures this.
#
#   10 epochs, 8 minibatches -- more gradient steps per unit of collected data,
#       which is the whole reason PPO exists.
LUNAR = PPOConfig(env_id="LunarLander-v3", total_steps=1_000_000, n_envs=8, n_steps=256,
                  n_epochs=10, n_minibatches=8, lr=3e-4, ent_coef=0.01,
                  gamma=0.999, gae_lambda=0.98, norm_reward=True)


def _job(args):
    cfg, seed = args
    r = train_ppo(cfg, seed, threads=2)
    r.pop("agent")
    return r


def run(cfg, name, seeds=(0, 1, 2)):
    with ProcessPoolExecutor(max_workers=3) as pool:
        res = list(pool.map(_job, [(cfg, s) for s in seeds]))
    np.savez(OUT / f"{name}.npz",
             steps=res[0]["steps"],
             curves=np.stack([r["curve"] for r in res]),
             kl=np.stack([r["kl"] for r in res]),
             clipfrac=np.stack([r["clipfrac"] for r in res]),
             finals=np.array([r["final"] for r in res]))
    for s, r in zip(seeds, res):
        print(f"  {name} seed {s}: eval {r['final']:7.1f} ± {r['final_std']:5.1f}  "
              f"({r['wall']:.0f}s, mean KL {r['kl'].mean():.4f}, "
              f"clipfrac {r['clipfrac'].mean():.2f})")
    print(f"  {name} mean final: {np.mean([r['final'] for r in res]):.1f}")
    return res


def cmd_cartpole():
    print("CartPole-v1:")
    run(CARTPOLE, "cartpole")


def cmd_lunar():
    print("LunarLander-v3:")
    run(LUNAR, "lunarlander")


def cmd_ratio_check():
    """A one-off sanity check that belongs in every PPO you ever write.

    At the first minibatch of the first epoch, the policy that scores the data is
    the policy that collected it, so the importance ratio must be exactly 1 and
    the KL exactly 0. If they are not, the stored log-probs disagree with the
    network — the single most common silent PPO bug, and it does not crash, it
    just quietly trains a worse agent.
    """
    cfg = replace(CARTPOLE, total_steps=CARTPOLE.batch_size)
    envs = make_vec_env(cfg.env_id, cfg.n_envs, 0)
    obs_dim, act_dim, cont = space_dims(envs)
    set_seed(0)
    agent = ActorCritic(obs_dim, act_dim, continuous=cont)
    obs, _ = envs.reset(seed=0)
    ro, _ = rollout(envs, agent, obs, cfg.n_steps, cfg.gamma, cfg.gae_lambda)
    _, new_logp, _, _ = agent.act(ro.obs, ro.actions)
    ratio = (new_logp - ro.logps).exp()
    print(f"  first-epoch ratio: mean {ratio.mean():.10f}  max deviation from 1: "
          f"{(ratio - 1).abs().max():.2e}")
    print(f"  first-epoch approx KL: {((ratio - 1) - (new_logp - ro.logps)).mean():.2e}")
    envs.close()


def cmd_plot():
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for name, ax, thresh, title in [("cartpole", axes[0], 475, "CartPole-v1"),
                                    ("lunarlander", axes[1], 200, "LunarLander-v3")]:
        ps.style_axes(ax)
        d = np.load(OUT / f"{name}.npz")
        c, x = d["curves"], d["steps"]
        ax.fill_between(x, c.min(0), c.max(0), color=ps.SERIES[0], alpha=0.15, linewidth=0)
        ax.plot(x, c.mean(0), color=ps.SERIES[0], linewidth=2.2)
        ax.axhline(thresh, color=ps.INK_MUTED, linestyle=":", linewidth=1)
        ax.text(x[0], thresh * 1.02, f"solved ({thresh})", color=ps.INK_MUTED, fontsize=8)
        ax.set_title(f"{title} — final {d['finals'].mean():.0f}", color=ps.INK,
                     fontsize=11, loc="left")
        ax.set_xlabel("environment steps", color=ps.INK_SECONDARY, fontsize=9)
        ax.set_ylabel("episode return", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ppo_curves.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'ppo_curves.png'}")

    # What the clip is actually doing, over training.
    d = np.load(OUT / "lunarlander.npz")
    fig, ax = ps.new_axes(7.4, 3.6)
    ax2 = ax.twinx()
    x = np.arange(d["kl"].shape[1])
    ax.plot(x, d["kl"].mean(0), color=ps.SERIES[0], linewidth=2.0, label="approx KL per update")
    ax2.plot(x, d["clipfrac"].mean(0), color=ps.SERIES[2], linewidth=2.0,
             label="fraction of samples clipped")
    ax2.set_ylabel("clip fraction", color=ps.SERIES[2], fontsize=9)
    ax2.tick_params(colors=ps.INK_MUTED, labelsize=9)
    ax2.spines["top"].set_visible(False)
    ax.set_ylabel("approx KL", color=ps.SERIES[0], fontsize=9)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], frameon=False, fontsize=8, loc="upper right")
    ps.finish(fig, ax, "The clip is not decoration — it fires on a fifth of all samples",
              "update", "", OUT / "clipping.png")


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
    if cmd in ("check", "all"):
        cmd_ratio_check()
    if cmd in ("cartpole", "all"):
        cmd_cartpole()
    if cmd in ("lunar", "all"):
        cmd_lunar()
    if cmd in ("plot", "all"):
        cmd_plot()
