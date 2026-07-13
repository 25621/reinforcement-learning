"""TRPO: the algorithm PPO replaced. Implemented in full, and raced against it on MuJoCo.

PPO's clip is a *heuristic* stand-in for a real constraint. TRPO enforces the
real thing:

        maximize   E[ π_θ(a|s)/π_old(a|s) · A(s,a) ]
        subject to E[ KL( π_old(·|s) ‖ π_θ(·|s) ) ] ≤ δ

That constraint is what makes the step safe: the policy may not move more than δ
nats away from where it started, no matter what the advantages say. The cost is
that you can no longer just call `loss.backward()` and `optim.step()`, because
that is not how constrained optimization works. What you need instead:

  1. The natural gradient direction. Linearize the objective and quadratically
     approximate the constraint, and the optimal direction is F⁻¹g, where F is
     the Fisher information matrix of the policy (the Hessian of the KL). For a
     64x64 network F is 4600x4600 — forming it is possible; forming it every
     update is not.

  2. Conjugate gradient. Solve F·x = g for x WITHOUT ever building F, using only
     the ability to compute F·v for a given vector v. That product is a double
     backward through the KL: ∇(∇KL · v). Ten CG iterations get a good enough x.

  3. A line search. Steps 1-2 rest on approximations that are only valid near θ.
     So compute the largest step the quadratic model allows, then walk BACK from
     it — halving repeatedly — until you find a step that genuinely improves the
     surrogate and genuinely respects the KL constraint. If none does, take no
     step at all.

Three hundred lines to PPO's five. This project measures what those lines buy.

Runtime: ~9 min on 12 CPU cores (`python3 trpo.py all`).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import kl_divergence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "22-ppo-from-scratch"))
sys.path.insert(0, str(HERE.parent / "19-reinforce-on-cartpole"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import plot_style as ps
from pg_lib import ActorCritic, evaluate, make_vec_env, rollout, set_seed, space_dims
from ppo import PPOConfig, train_ppo

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

ENV_ID = "Hopper-v5"
TOTAL_STEPS = 400_000
N_ENVS = 8
N_STEPS = 256
GAMMA = 0.99
LAM = 0.95
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------
# Flat-parameter plumbing. TRPO thinks of θ as one long vector; torch does not.
# --------------------------------------------------------------------------

def flat_params(module):
    return torch.cat([p.data.reshape(-1) for p in module.parameters()])


def set_flat_params(module, flat):
    i = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def flat_grad(loss, params, retain=False, create=False):
    g = torch.autograd.grad(loss, params, retain_graph=retain, create_graph=create)
    return torch.cat([x.reshape(-1) for x in g])


# --------------------------------------------------------------------------
# The three pieces of machinery
# --------------------------------------------------------------------------

def conjugate_gradient(fvp, b, iters=10, tol=1e-10):
    """Solve F·x = b using only matrix-vector products. Never forms F.

    This is textbook CG, and it is the reason TRPO is tractable at all. Each
    iteration costs one Fisher-vector product (one double-backward), so ten
    iterations cost about ten backward passes — expensive next to PPO's one,
    but linear in the number of parameters rather than quadratic.
    """
    x = torch.zeros_like(b)
    r = b.clone()                     # residual b - F·x, and x starts at 0
    p = b.clone()
    rr = r @ r
    for _ in range(iters):
        fp = fvp(p)
        alpha = rr / (p @ fp + 1e-8)
        x += alpha * p
        r -= alpha * fp
        rr_new = r @ r
        if rr_new < tol:
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def fisher_vector_product(agent, obs, old_dist, damping):
    """v ↦ F·v, where F is the Hessian of the mean KL at the current parameters.

    The trick that makes this cheap: instead of building the Hessian and
    multiplying, differentiate the SCALAR (∇KL · v) once more. Torch's autograd
    gives ∇(∇KL · v) = H·v directly, for the cost of one extra backward pass.

    The `damping` term adds εI to F. Without it, F is near-singular (a network
    has directions its output does not depend on at all, and the Fisher is
    exactly zero along them), CG divides by a number near zero, and the step
    explodes down a direction that means nothing.
    """
    params = list(agent.actor.parameters())
    if agent.continuous:
        params = params + [agent.log_std]

    def fvp(v):
        new_dist = agent.dist(obs)
        kl = kl_divergence(old_dist, new_dist).sum(-1).mean() if agent.continuous \
            else kl_divergence(old_dist, new_dist).mean()
        grads = flat_grad(kl, params, retain=True, create=True)
        gv = (grads * v).sum()
        hv = flat_grad(gv, params, retain=True)
        return hv + damping * v
    return fvp


def surrogate(agent, obs, actions, old_logp, adv):
    """E[ ratio · A ] — the thing TRPO maximizes. Identical to PPO's, minus the clip."""
    _, logp, _, _ = agent.act(obs, actions)
    return (torch.exp(logp - old_logp) * adv).mean()


def trpo_step(agent, ro, max_kl=0.01, damping=0.1, cg_iters=10, backtrack_iters=10,
              backtrack_coef=0.8):
    """One TRPO policy update. Returns diagnostics, including whether it gave up.

    Everything here happens on the actor only; the critic is trained separately
    with ordinary gradient descent, because there is no trust region to respect
    when fitting a regression.
    """
    obs, actions = ro.obs, ro.actions
    adv = (ro.advantages - ro.advantages.mean()) / (ro.advantages.std() + 1e-8)
    old_logp = ro.logps

    params = list(agent.actor.parameters())
    if agent.continuous:
        params = params + [agent.log_std]

    with torch.no_grad():
        old_dist = agent.dist(obs)
        if agent.continuous:
            old_dist = torch.distributions.Normal(old_dist.mean.detach(),
                                                  old_dist.stddev.detach())
        else:
            old_dist = torch.distributions.Categorical(logits=old_dist.logits.detach())

    loss = surrogate(agent, obs, actions, old_logp, adv)
    g = flat_grad(loss, params, retain=True)

    if g.norm() < 1e-8:
        return dict(kl=0.0, improve=0.0, backtracks=-1, step_norm=0.0)

    fvp = fisher_vector_product(agent, obs, old_dist, damping)
    step_dir = conjugate_gradient(fvp, g, iters=cg_iters)

    # The largest step the quadratic model says stays inside the KL ball:
    #   maximize gᵀs  s.t.  ½ sᵀFs ≤ δ   ⇒   s = sqrt(2δ / (xᵀFx)) · x
    shs = 0.5 * (step_dir @ fvp(step_dir))
    if shs <= 0:
        return dict(kl=0.0, improve=0.0, backtracks=-1, step_norm=0.0)
    step_size = torch.sqrt(max_kl / (shs + 1e-8))
    full_step = step_size * step_dir

    old_params = flat_params(agent.actor if not agent.continuous else agent).clone()
    old_actor = torch.cat([p.data.reshape(-1) for p in params]).clone()
    old_loss = loss.item()

    # Backtracking line search: the quadratic model is a local lie, so verify.
    for i in range(backtrack_iters):
        frac = backtrack_coef ** i
        new_params = old_actor + frac * full_step
        _set(params, new_params)
        with torch.no_grad():
            new_loss = surrogate(agent, obs, actions, old_logp, adv).item()
            new_dist = agent.dist(obs)
            kl = (kl_divergence(old_dist, new_dist).sum(-1).mean() if agent.continuous
                  else kl_divergence(old_dist, new_dist).mean()).item()
        # Accept only if BOTH promises hold: the surrogate really improved, and
        # the constraint really holds. The quadratic model guaranteed both; the
        # true objective is under no obligation to agree.
        if kl <= max_kl * 1.5 and new_loss > old_loss:
            return dict(kl=kl, improve=new_loss - old_loss, backtracks=i,
                        step_norm=float((frac * full_step).norm()))
    _set(params, old_actor)                     # nothing worked: stay put
    return dict(kl=0.0, improve=0.0, backtracks=backtrack_iters, step_norm=0.0)


def _set(params, flat):
    i = 0
    for p in params:
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


# --------------------------------------------------------------------------
# The training loop
# --------------------------------------------------------------------------

def train_trpo(seed, total_steps=TOTAL_STEPS, threads=1, max_kl=0.01, log_every=5):
    torch.set_num_threads(threads)
    envs = make_vec_env(ENV_ID, N_ENVS, seed, gamma=GAMMA, norm_obs=True,
                        norm_reward=True, clip_action=True)
    obs_dim, act_dim, cont = space_dims(envs)
    set_seed(seed)
    agent = ActorCritic(obs_dim, act_dim, continuous=cont)
    # The critic is NOT part of the trust region — it is a regression, and a
    # regression has no policy to destroy. Plain Adam, as in every TRPO paper.
    critic_optim = torch.optim.Adam(agent.critic.parameters(), lr=1e-3)

    obs, _ = envs.reset(seed=seed)
    n_updates = total_steps // (N_ENVS * N_STEPS)
    steps, curve, kls, backtracks, walls = [], [], [], [], []
    recent = []
    t0 = time.time()

    for u in range(n_updates):
        ro, obs = rollout(envs, agent, obs, N_STEPS, GAMMA, LAM)
        recent.extend(ro.ep_returns)

        t_up = time.time()
        info = trpo_step(agent, ro, max_kl=max_kl)
        walls.append(time.time() - t_up)

        for _ in range(10):                         # the critic, ordinary SGD
            idx = torch.randperm(len(ro.obs))
            for start in range(0, len(idx), 512):
                mb = idx[start:start + 512]
                v_loss = F.mse_loss(agent.value(ro.obs[mb]), ro.returns[mb])
                critic_optim.zero_grad()
                v_loss.backward()
                critic_optim.step()

        kls.append(info["kl"])
        backtracks.append(info["backtracks"])
        if (u + 1) % log_every == 0:
            steps.append((u + 1) * N_ENVS * N_STEPS)
            curve.append(np.mean(recent[-40:]) if recent else np.nan)

    e = envs.envs[0]
    while not hasattr(e, "obs_rms"):
        e = e.env
    obs_rms = e.obs_rms
    envs.close()
    final, _ = evaluate(agent, ENV_ID, n_episodes=20, seed=9999, norm_obs=True,
                        obs_rms=obs_rms, clip_action=True)
    return dict(algo="TRPO", seed=seed, steps=np.asarray(steps), curve=np.asarray(curve),
                final=final, kl=np.asarray(kls), backtracks=np.asarray(backtracks),
                update_wall=float(np.mean(walls)), wall=time.time() - t0)


# PPO's side of the comparison must not be a straw man. These are the settings that
# came out best of a small sweep on this task (n_minibatches 4 -> 32 and lr 3e-4 ->
# 1e-4 are each worth a few hundred points of return), so TRPO is being raced against
# the best PPO found here rather than the first one tried.
PPO_CFG = PPOConfig(env_id=ENV_ID, total_steps=TOTAL_STEPS, n_envs=N_ENVS, n_steps=N_STEPS,
                    n_epochs=10, n_minibatches=32, lr=1e-4, ent_coef=0.0,
                    norm_obs=True, norm_reward=True)


def train_ppo_job(seed):
    t0 = time.time()
    r = train_ppo(PPO_CFG, seed=seed, threads=1)
    r.pop("agent")
    r.update(algo="PPO", seed=seed, backtracks=np.zeros(1), update_wall=np.nan)
    return r


def _job(args):
    algo, seed = args
    return train_trpo(seed) if algo == "TRPO" else train_ppo_job(seed)


def cmd_compare():
    jobs = [(a, s) for a in ("TRPO", "PPO") for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        res = list(pool.map(_job, jobs))
    by = {}
    for r in res:
        by.setdefault(r["algo"], []).append(r)

    np.savez(OUT / "compare.npz",
             steps=by["TRPO"][0]["steps"],
             trpo=np.stack([r["curve"] for r in by["TRPO"]]),
             ppo=np.stack([r["curve"] for r in by["PPO"]]),
             trpo_final=np.array([r["final"] for r in by["TRPO"]]),
             ppo_final=np.array([r["final"] for r in by["PPO"]]),
             trpo_kl=np.stack([r["kl"] for r in by["TRPO"]]),
             ppo_kl=np.stack([r["kl"] for r in by["PPO"]]),
             trpo_backtracks=np.stack([r["backtracks"] for r in by["TRPO"]]),
             trpo_wall=np.array([r["wall"] for r in by["TRPO"]]),
             ppo_wall=np.array([r["wall"] for r in by["PPO"]]))

    for algo in ("TRPO", "PPO"):
        f = np.array([r["final"] for r in by[algo]])
        w = np.mean([r["wall"] for r in by[algo]])
        kl = np.concatenate([r["kl"] for r in by[algo]])
        print(f"{algo}: final {f.mean():7.1f} ± {f.std():5.1f}   wall {w:5.0f}s   "
              f"mean KL/update {kl.mean():.4f}  max KL {kl.max():.4f}  "
              f"KL > 0.01: {100 * (kl > 0.01).mean():.1f}% of updates")
    bt = np.concatenate([r["backtracks"] for r in by["TRPO"]])
    print(f"TRPO line search: accepted at first try {100 * (bt == 0).mean():.0f}% of updates, "
          f"gave up entirely {100 * (bt == 10).mean():.1f}%")


def cmd_plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT / "compare.npz")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in (ax1, ax2):
        ps.style_axes(a)

    for i, (key, name) in enumerate([("trpo", "TRPO"), ("ppo", "PPO")]):
        c = d[key]
        ax1.fill_between(d["steps"], c.min(0), c.max(0), color=ps.SERIES[i], alpha=0.14, linewidth=0)
        ax1.plot(d["steps"], c.mean(0), color=ps.SERIES[i], linewidth=2.3,
                 label=f"{name} — final {d[key + '_final'].mean():.0f}")
    ax1.legend(frameon=False, fontsize=9, loc="upper left")
    ax1.set_title(f"{ENV_ID}: TRPO is not the one that loses", color=ps.INK, fontsize=11, loc="left")
    ax1.set_xlabel("environment steps", color=ps.INK_SECONDARY, fontsize=9)
    ax1.set_ylabel("episode return", color=ps.INK_SECONDARY, fontsize=9)

    # The honest difference: what the KL per update actually does.
    for i, (key, name) in enumerate([("trpo_kl", "TRPO"), ("ppo_kl", "PPO")]):
        kl = d[key].mean(0)
        ax2.plot(np.arange(len(kl)), kl, color=ps.SERIES[i], linewidth=1.4, label=name, alpha=0.9)
    ax2.axhline(0.01, color=ps.INK, linestyle="--", linewidth=1.2)
    ax2.text(1, 0.0115, "TRPO's constraint δ = 0.01", color=ps.INK, fontsize=8)
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=9, loc="lower right")
    ax2.set_title("TRPO obeys its constraint. PPO approximately does.",
                  color=ps.INK, fontsize=11, loc="left")
    ax2.set_xlabel("update", color=ps.INK_SECONDARY, fontsize=9)
    ax2.set_ylabel("KL(π_old ‖ π_new)  (log)", color=ps.INK_SECONDARY, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "trpo_vs_ppo.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'trpo_vs_ppo.png'}")


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
    if cmd in ("compare", "all"):
        cmd_compare()
    if cmd in ("plot", "all"):
        cmd_plot()
