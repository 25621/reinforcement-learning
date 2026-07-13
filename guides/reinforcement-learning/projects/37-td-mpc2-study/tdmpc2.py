r"""Project 37 — TD-MPC2: a short plan with a learned value stapled to the end.

Project 32 showed the problem in a single plot: a learned model is excellent for 3 steps,
usable for 10, and fiction by 30. PETS responds by planning 15 steps anyway and hoping.
TD-MPC2 responds by *not planning far at all*:

    score(a_0..a_H) = SUM_{t<H} gamma^t * R(z_t, a_t)  +  gamma^H * Q(z_H, pi(z_H))
                      \_______ imagined, 3 steps _______/    \___ learned, everything after ___/

Plan 3 steps with the model — the range where it is still honest — and hand the entire
rest of the future to a learned Q-function, which was trained on real data and never has
to imagine anything. Planning supplies precise short-term decisions; the value supplies
cheap long-term foresight. Neither could do the job alone.

The second idea is where the latent comes from. The encoder is trained ONLY by the
reward, value and consistency losses — never by reconstructing the observation. Nothing
in this file asks the model to predict what the pendulum will look like. It only has to
predict what the pendulum will be *worth*. A model that models everything wastes its
capacity on the wallpaper; this one models only what it will be quizzed on.

Experiments:
  1. The 2-D sweep that IS the paper's argument: planning horizon x {value bootstrap
     on, off}. Without a value, the agent needs a long horizon and pays for it in model
     error. With one, a 3-step horizon wins.
  2. TD-MPC2 vs SAC (model-free) vs PETS (model, no value) on one axis.

  python3 tdmpc2.py     # ~7 min on 12 hyperthreads
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
torch.set_num_threads(1)

LATENT = 32
HIDDEN = 128


def mlp(i, h, o, out_act=None):
    layers = [nn.Linear(i, h), nn.ELU(), nn.Linear(h, h), nn.ELU(), nn.Linear(h, o)]
    if out_act:
        layers.append(out_act)
    return nn.Sequential(*layers)


class TDMPC(nn.Module):
    """Encoder + latent dynamics + reward + twin Q + policy. All trained together.

    "Trained together" is the point. The encoder's gradients come from the reward head,
    the Q heads and the consistency loss — so the latent space is shaped by what the
    *decisions* need. In PETS the model is fit in a vacuum, by maximum likelihood over
    states, and then the planner has to live with whatever it got.
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.encoder = mlp(obs_dim, HIDDEN, LATENT, nn.LayerNorm(LATENT))
        self.dynamics = mlp(LATENT + act_dim, HIDDEN, LATENT, nn.LayerNorm(LATENT))
        self.reward = mlp(LATENT + act_dim, HIDDEN, 1)
        self.q1 = mlp(LATENT + act_dim, HIDDEN, 1)
        self.q2 = mlp(LATENT + act_dim, HIDDEN, 1)
        self.pi = mlp(LATENT, HIDDEN, act_dim, nn.Tanh())

    def next(self, z, a):
        za = torch.cat([z, a], dim=-1)
        return self.dynamics(za), self.reward(za).squeeze(-1)

    def Q(self, z, a, min_of_two=True):
        za = torch.cat([z, a], dim=-1)
        q1, q2 = self.q1(za).squeeze(-1), self.q2(za).squeeze(-1)
        # The twin-critic minimum, straight from TD3/SAC (Phase 5). Project 26 measured
        # exactly why it is here: a single critic, maximized over actions, is a machine
        # for finding its own most optimistic mistakes.
        return torch.min(q1, q2) if min_of_two else (q1, q2)


@dataclass
class Cfg:
    seed: int = 0
    total_steps: int = 5_000
    start_steps: int = 500
    horizon: int = 3            # <-- the variable under test in experiment 1
    use_value: bool = True      # <-- and so is this
    n_candidates: int = 64
    n_iters: int = 3
    n_elites: int = 8
    n_policy_traj: int = 12     # candidates seeded by the policy instead of by noise
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 128
    seq_len: int = 3            # subsequence length for the consistency loss
    tau: float = 0.01
    rho: float = 0.5            # down-weight losses at deeper imagined steps
    utd: int = 1
    eval_every: int = 1_000
    eval_episodes: int = 2      # each eval episode re-plans 200 times; they are not free
    horizon_std: float = 0.6


class SeqBuffer:
    """Stores transitions but samples *subsequences*.

    The consistency loss needs (o_t, a_t, r_t, o_{t+1}, a_{t+1}, ...) — a stretch of
    consecutive time — because it trains the model to stay accurate when its own output
    is fed back into its own input. A shuffled buffer of independent transitions, which
    is all SAC ever needed, cannot express that.
    """

    def __init__(self, obs_dim, act_dim, capacity, seq_len):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.valid = np.zeros((capacity,), dtype=bool)  # can a sequence start here?
        self.seq_len = seq_len
        self.capacity = capacity
        self.size = 0

    def add(self, o, a, r, o2, ep_step, ep_len_ok):
        i = self.size
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i] = o, a, r, o2
        self.valid[i] = ep_len_ok
        self.size = min(i + 1, self.capacity - 1)

    def sample(self, batch_size, rng):
        # Only start a sequence where the next `seq_len` entries are from the same episode.
        idx = rng.integers(0, max(1, self.size - self.seq_len), size=batch_size)
        ok = self.valid[idx]
        idx = idx[ok] if ok.any() else idx
        seq = np.stack([idx + k for k in range(self.seq_len)], axis=1)  # (B, L)
        return (
            torch.as_tensor(self.obs[seq]),
            torch.as_tensor(self.act[seq]),
            torch.as_tensor(self.rew[seq]),
            torch.as_tensor(self.next_obs[seq]),
        )


@torch.no_grad()
def plan(model, obs, cfg, act_dim, act_limit, gen, prev_mean=None):
    """CEM over a SHORT horizon, with Q(z_H, pi(z_H)) at the end.

    Two differences from project 33's CEM, and both matter:

    * The rollout happens in LATENT space. `obs` is encoded exactly once; after that the
      search never sees an observation again, it just iterates `dynamics`.
    * A slice of the population is seeded by the learned policy rather than by noise.
      The policy is a decent guess at a good action sequence, so this gives CEM a warm
      body to refine instead of making it rediscover competence from scratch every step.
      It also means the planner can never be much worse than the policy alone.
    """
    z0 = model.encoder(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    H, N, P = cfg.horizon, cfg.n_candidates, cfg.n_policy_traj

    # --- the policy's own suggestion, rolled out in imagination ---
    pi_actions = torch.zeros(H, P, act_dim)
    z = z0.expand(P, -1)
    for t in range(H):
        pi_actions[t] = model.pi(z) * act_limit
        z, _ = model.next(z, pi_actions[t] / act_limit)

    mean = torch.zeros(H, act_dim) if prev_mean is None else prev_mean
    std = torch.full((H, act_dim), cfg.horizon_std * act_limit)

    for _ in range(cfg.n_iters):
        noise = torch.randn(H, N, act_dim, generator=gen)
        sampled = (mean.unsqueeze(1) + std.unsqueeze(1) * noise).clamp(-act_limit, act_limit)
        actions = torch.cat([sampled, pi_actions], dim=1)  # (H, N+P, act_dim)
        M = actions.shape[1]

        z = z0.expand(M, -1)
        G, disc = torch.zeros(M), 1.0
        for t in range(H):
            z, r = model.next(z, actions[t] / act_limit)
            G += disc * r
            disc *= cfg.gamma
        if cfg.use_value:
            # THE line. Everything after step H is handed to a learned value, which was
            # fit on real returns and does not have to imagine anything.
            G += disc * model.Q(z, model.pi(z))

        elite_idx = G.topk(cfg.n_elites).indices
        elites = actions[:, elite_idx]           # (H, n_elites, act_dim)
        elite_G = G[elite_idx]

        # Softmax-weighted refit (TD-MPC's variant of CEM): elites are weighted by how
        # good they are, not counted equally. A plan that is twice as good pulls the
        # mean twice as hard.
        w = torch.softmax(0.5 * (elite_G - elite_G.max()), dim=0).view(1, -1, 1)
        mean = (w * elites).sum(1)
        std = ((w * (elites - mean.unsqueeze(1)) ** 2).sum(1)).sqrt().clamp(0.05, 2.0)

    return mean[0].numpy(), mean


def train(cfg: Cfg, progress=False):
    import gymnasium as gym

    torch.manual_seed(cfg.seed)  # before the nets exist
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 7)

    env, eval_env = gym.make("Pendulum-v1"), gym.make("Pendulum-v1")
    env.action_space.seed(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    model = TDMPC(obs_dim, act_dim)
    import copy

    target = copy.deepcopy(model)
    optim = torch.optim.Adam(
        [p for n, p in model.named_parameters() if not n.startswith("pi.")], lr=cfg.lr
    )
    pi_optim = torch.optim.Adam(model.pi.parameters(), lr=cfg.lr)
    buf = SeqBuffer(obs_dim, act_dim, 100_000, cfg.seq_len)

    hist = {"steps": [], "eval_return": []}
    o, _ = env.reset(seed=cfg.seed)
    ep_step = 0
    t0 = time.time()
    prev_mean = None

    for t in range(1, cfg.total_steps + 1):
        if t <= cfg.start_steps:
            a = env.action_space.sample()
            prev_mean = None
        else:
            a, prev_mean = plan(model, o, cfg, act_dim, act_limit, gen, prev_mean)
            a = np.clip(a + 0.1 * act_limit * rng.standard_normal(act_dim),
                        -act_limit, act_limit)
            # Warm start the next decision with this one, shifted forward a step.
            prev_mean = torch.cat([prev_mean[1:], torch.zeros(1, act_dim)], 0)

        o2, r, term, trunc, _ = env.step(a)
        buf.add(o, a, r, o2, ep_step, ep_step < 200 - cfg.seq_len)
        o = o2
        ep_step += 1
        if term or trunc:
            o, _ = env.reset()
            ep_step = 0
            prev_mean = None

        if t >= cfg.start_steps:
            for _ in range(cfg.utd):
                update(model, target, optim, pi_optim, buf, cfg, rng, act_limit)

        if t % cfg.eval_every == 0:
            ret = evaluate(eval_env, model, cfg, act_dim, act_limit, gen)
            hist["steps"].append(t)
            hist["eval_return"].append(ret)
            if progress:
                print(f"  [H={cfg.horizon} value={cfg.use_value} seed={cfg.seed}] "
                      f"step {t:5d}  return {ret:8.1f}  ({time.time() - t0:.0f}s)",
                      flush=True)

    env.close()
    eval_env.close()
    hist["wall"] = time.time() - t0
    return hist


def update(model, target, optim, pi_optim, buf, cfg, rng, act_limit):
    obs, act, rew, next_obs = buf.sample(cfg.batch_size, rng)
    L = obs.shape[1]

    z = model.encoder(obs[:, 0])
    consistency = torch.zeros(())
    reward_loss = torch.zeros(())
    q_loss = torch.zeros(())

    for k in range(L):
        a = act[:, k] / act_limit

        # --- Q, before the latent moves on: TD target from the target network ---
        with torch.no_grad():
            z_next_real = target.encoder(next_obs[:, k])
            td_target = rew[:, k] + cfg.gamma * target.Q(z_next_real, target.pi(z_next_real))
        q1, q2 = model.Q(z, a, min_of_two=False)
        w = cfg.rho ** k  # deeper imagined steps are less trustworthy, so weigh them less
        q_loss = q_loss + w * (F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target))

        # --- one imagined step, and the two losses that shape the latent ---
        z, r_pred = model.next(z, a)
        reward_loss = reward_loss + w * F.mse_loss(r_pred, rew[:, k])
        with torch.no_grad():
            # The target for the imagined latent is the TARGET ENCODER's view of the
            # state that really came next. Not the observation — the *latent*. This is
            # the consistency loss, and it is the only thing tying `dynamics` to reality.
            # It is also self-referential, which is why the target network is essential:
            # without it, the encoder could satisfy this loss perfectly by mapping every
            # observation to the same constant vector, and it would.
            z_target = target.encoder(next_obs[:, k])
        consistency = consistency + w * F.mse_loss(z, z_target)

    loss = 2.0 * consistency + 0.5 * reward_loss + 0.1 * q_loss
    optim.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optim.step()

    # --- the policy: plain deterministic policy gradient, exactly as in project 26 ---
    z0 = model.encoder(obs[:, 0]).detach()
    pi_loss = -model.Q(z0, model.pi(z0)).mean()
    pi_optim.zero_grad(set_to_none=True)
    pi_loss.backward()
    pi_optim.step()

    with torch.no_grad():
        for p, pt in zip(model.parameters(), target.parameters()):
            pt.mul_(1 - cfg.tau).add_(cfg.tau * p)


def evaluate(env, model, cfg, act_dim, act_limit, gen, episodes=None):
    episodes = episodes or cfg.eval_episodes
    total = 0.0
    for i in range(episodes):
        o, _ = env.reset(seed=cfg.seed + 5000 + i)
        done, prev = False, None
        while not done:
            a, prev = plan(model, o, cfg, act_dim, act_limit, gen, prev)
            prev = torch.cat([prev[1:], torch.zeros(1, act_dim)], 0)
            o, r, term, trunc, _ = env.step(np.clip(a, -act_limit, act_limit))
            total += r
            done = term or trunc
    return total / episodes


def job(args):
    horizon, use_value, seed = args
    cfg = Cfg(seed=seed, horizon=horizon, use_value=use_value)
    h = train(cfg, progress=(seed == 0 and horizon == 3))
    return {"horizon": horizon, "use_value": use_value, "seed": seed,
            "steps": h["steps"], "return": h["eval_return"], "wall": h["wall"]}


# Two seeds, not three. This sweep is 8 configurations wide (4 horizons x value on/off),
# so a third seed means 8 more processes on 6 cores — and the effect being measured here
# is large enough that a third seed would buy less than it costs.
SEEDS = [0, 1]
HORIZONS = [1, 3, 5, 10]


def main():
    OUT.mkdir(exist_ok=True)

    print("=== horizon x value-bootstrap sweep ===", flush=True)
    jobs = [(h, v, s) for h in HORIZONS for v in (True, False) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(job, jobs))

    def pick(h, v):
        r = [x for x in res if x["horizon"] == h and x["use_value"] == v]
        return np.array([x["return"] for x in r]), np.array(r[0]["steps"])

    print(f"\n  {'horizon':>8s} {'with value':>13s} {'no value':>13s}")
    final_v, final_nv = [], []
    for h in HORIZONS:
        rv, _ = pick(h, True)
        rn, _ = pick(h, False)
        fv, fn = rv[:, -1].mean(), rn[:, -1].mean()
        final_v.append(fv)
        final_nv.append(fn)
        print(f"  {h:8d} {fv:13.1f} {fn:13.1f}")

    fig, ax = ps.new_axes(7.4, 4.3)
    ax.plot(HORIZONS, final_v, color=ps.SERIES[0], lw=2, marker="o", ms=6,
            label="with the learned value at the end (TD-MPC2)")
    ax.plot(HORIZONS, final_nv, color=ps.SERIES[2], lw=2, marker="o", ms=6,
            label="pure planning, no value (PETS-style)")
    ax.axhline(-150, color=ps.INK_MUTED, ls="--", lw=1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "A value function buys the foresight a long rollout cannot",
              "planning horizon (imagined steps)", "final return",
              OUT / "horizon_value_sweep.png")

    # learning curves at the best setting
    fig, ax = ps.new_axes(7.4, 4.3)
    for h, c in zip([3, 10], [ps.SERIES[0], ps.SERIES[4]]):
        r, st = pick(h, True)
        ax.plot(st, r.mean(0), color=c, lw=2, label=f"TD-MPC2, horizon {h} (with value)")
        ax.fill_between(st, r.min(0), r.max(0), color=c, alpha=0.12)
    r, st = pick(10, False)
    ax.plot(st, r.mean(0), color=ps.SERIES[2], lw=2, ls="--",
            label="horizon 10, no value")
    ax.fill_between(st, r.min(0), r.max(0), color=ps.SERIES[2], alpha=0.12)
    ax.axhline(-150, color=ps.INK_MUTED, ls="--", lw=1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "Short plan + learned value beats a long plan alone",
              "environment steps", "evaluation return", OUT / "learning_curves.png")


if __name__ == "__main__":
    main()
