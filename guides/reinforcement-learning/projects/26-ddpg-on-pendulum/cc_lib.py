"""The reusable continuous-control core for Phase 5.

Phase 4 parameterized the policy and learned it on-policy: collect a rollout,
take a gradient step, throw the data away. Phase 5 goes back to a replay buffer
and a Q-function — DQN's recipe — and has to solve the one problem that makes
that recipe fail on continuous actions:

    Q-learning needs   max_a Q(s, a).
    With 6 continuous joint torques, that max is an optimization problem,
    not a lookup over 4 buttons. You cannot afford to solve it every step.

Every algorithm in this phase is one answer to that sentence: *learn a network
that outputs the maximizing action directly*. The actor IS the argmax.

    DDPG  = replay + target nets + a deterministic actor trained through the critic
    TD3   = DDPG + twin critics + delayed actor + target policy smoothing
    SAC   = twin critics + a *stochastic* actor + an entropy bonus in the target

Written as three separate files those look like three algorithms. They are not.
They are one algorithm with flags, which is why this file has exactly one agent
class and one `Config`, and each project flips a field:

    project 26  everything off                      -> DDPG
    project 27  twin + delay + smoothing            -> TD3   (and ablates each)
    project 28  stochastic + entropy                -> SAC
    project 29  auto_alpha on/off                   -> the temperature study
    project 30  tanh_correction on/off              -> the reparameterization audit
    project 31  SAC vs PPO                          -> the sample-efficiency study

If a project could be written by flipping a flag and it instead copy-pasted the
agent, the reader would learn that these are different algorithms. They would be
learning something false.

Everything is CPU-only on purpose. These networks are two hidden layers; a GPU's
kernel-launch overhead would cost more than the arithmetic it saves, and this
box's card (sm_61) is too old for the installed torch anyway — `torch.cuda` is
never touched. One torch thread per process, because the parallelism that pays
here is running N independent seeds on N cores, not splitting one 256x256 matmul
across them.
"""

import math
import random
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")

# One thread per process. Measured on this 12-core box: a 256x256 SAC update runs
# at 106 updates/s on 1 thread and 156 on 4 — a 1.5x speedup for 4x the cores.
# Running 6 seeds as 6 single-threaded processes beats 1 seed on 6 threads by a
# mile, and the seeds are what we actually need.
torch.set_num_threads(1)

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


def set_seed(seed):
    """Seed every RNG that can move a result.

    Call this BEFORE building a network. Torch seeds its default generator from
    OS entropy at import, so a net constructed before `manual_seed` has unseeded
    weights and the run is irreproducible — the symptom (curves that differ
    between identical invocations) looks like environment noise, not a bug, which
    is what makes it expensive. This cost the repo real time in Phase 3.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        layers += [act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


# --------------------------------------------------------------------------
# Replay buffer
# --------------------------------------------------------------------------
class ReplayBuffer:
    """A ring buffer of transitions, preallocated as numpy arrays.

    Preallocated, not a `deque` of tuples: at 50k transitions the deque version
    spends more time in Python object overhead and per-sample `np.stack` than the
    network spends on gradients. Same algorithm, ~3x slower loop.
    """

    def __init__(self, obs_dim, act_dim, capacity):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i] = o
        self.act[i] = a
        self.rew[i] = r
        self.next_obs[i] = o2
        self.done[i] = d
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.obs[idx]),
            torch.as_tensor(self.act[idx]),
            torch.as_tensor(self.rew[idx]),
            torch.as_tensor(self.next_obs[idx]),
            torch.as_tensor(self.done[idx]),
        )


# --------------------------------------------------------------------------
# Actors
# --------------------------------------------------------------------------
class DeterministicActor(nn.Module):
    """DDPG/TD3's actor: state -> the single action it thinks is best.

    `tanh` bounds the output to (-1, 1) and it is then rescaled to the env's real
    action range. This actor has no notion of randomness — exploration has to be
    bolted on from outside, as additive noise. That bolt-on is exactly what SAC
    removes.
    """

    def __init__(self, obs_dim, act_dim, hidden, act_limit):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, act_dim], out_act=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class SquashedGaussianActor(nn.Module):
    """SAC's actor: state -> a *distribution* over actions.

    The network emits a mean and a log-std. An action is sampled with the
    reparameterization trick (`u = mu + sigma * eps`, so the randomness is an
    input, not an operation, and gradients flow through it), then squashed
    through `tanh` to land inside the action bounds.

    The squashing is where SAC hides its most famous silent bug. Bending a
    density through `tanh` changes it, so the log-probability needs a
    change-of-variables correction:

        log p(a) = log p(u) - SUM log(1 - tanh(u)^2)

    Get it wrong and nothing crashes: SAC just optimizes the wrong entropy, the
    temperature chases a target that does not mean what you think, and the agent
    is quietly worse. Project 30 audits this term numerically; `tanh_correction`
    exists so it can turn it off and measure the damage.

    The correction is computed as `2 * (log(2) - u - softplus(-2u))` rather than
    the literal `log(1 - tanh(u)^2)`. The two are equal in exact arithmetic; the
    literal one is not: at |u| > 6, `tanh(u)^2` rounds to exactly 1.0 in float32,
    `log(0) = -inf`, and the loss becomes NaN the first time the actor saturates.
    """

    def __init__(self, obs_dim, act_dim, hidden, act_limit, tanh_correction=True):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], out_act=nn.ReLU)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.act_limit = act_limit
        self.tanh_correction = tanh_correction

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        if deterministic:
            u = mu  # evaluation: act at the mode, no sampling
        else:
            u = mu + std * torch.randn_like(mu)  # reparameterized sample

        logp = None
        if with_logprob:
            # log N(u; mu, std), summed over action dims
            logp = (
                -0.5 * (((u - mu) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))
            ).sum(-1)
            if self.tanh_correction:
                # numerically stable log(1 - tanh(u)^2), summed over action dims
                logp = logp - (
                    2 * (math.log(2) - u - F.softplus(-2 * u))
                ).sum(-1)

        a = torch.tanh(u) * self.act_limit
        return a, logp


class Critic(nn.Module):
    """Q(s, a) -> a scalar. Takes the action as *input* (it cannot enumerate them)."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


def soft_update(net, target, tau):
    """Polyak averaging: target <- tau * net + (1 - tau) * target.

    The target network is a slow-moving copy of the critic, and it is the only
    reason this whole family is stable. Bootstrapping a network off its own
    instantaneous output is chasing a target that moves every time you step
    toward it; the slow copy makes it hold roughly still.
    """
    with torch.no_grad():
        for p, pt in zip(net.parameters(), target.parameters()):
            pt.mul_(1 - tau).add_(tau * p)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
@dataclass
class Config:
    """One config for DDPG, TD3 and SAC. The algorithm IS the flag settings."""

    env_id: str = "Pendulum-v1"
    seed: int = 0
    total_steps: int = 20_000
    start_steps: int = 1_000        # uniform-random actions before the policy takes over
    update_after: int = 1_000       # collect this much before the first gradient step
    update_every: int = 1           # env steps per gradient update (the UTD ratio, inverted)
    batch_size: int = 256
    hidden: int = 256
    buffer_size: int = 200_000
    gamma: float = 0.99
    tau: float = 0.005              # polyak coefficient for the target nets
    lr: float = 3e-4

    # --- the three TD3 fixes, individually switchable (project 27 ablates them) ---
    twin_critics: bool = False      # min over two critics in the target
    policy_delay: int = 1           # actor updates once per this many critic updates
    target_noise: float = 0.0       # std of noise added to the target action
    noise_clip: float = 0.5         # that noise is clipped to +/- this

    # --- DDPG/TD3 exploration (SAC needs none of this) ---
    act_noise: float = 0.1          # std of Gaussian noise added to the behaviour action

    # --- SAC ---
    stochastic_actor: bool = False  # squashed-Gaussian actor instead of deterministic
    entropy_bonus: bool = False     # subtract alpha * logp inside the TD target
    alpha: float = 0.2              # the temperature
    auto_alpha: bool = False        # tune alpha by gradient descent on the dual (project 29)
    target_entropy: float = None    # defaults to -act_dim
    tanh_correction: bool = True    # the log-prob change-of-variables term (project 30)

    # --- bookkeeping ---
    eval_every: int = 2_000
    eval_episodes: int = 5
    log_q_bias: bool = False        # track critic Q vs realized return (project 26)


def ddpg_config(**kw):
    return Config(**kw)


def td3_config(**kw):
    base = dict(twin_critics=True, policy_delay=2, target_noise=0.2, noise_clip=0.5)
    base.update(kw)
    return Config(**base)


def sac_config(**kw):
    base = dict(
        twin_critics=True,
        stochastic_actor=True,
        entropy_bonus=True,
        auto_alpha=True,
        act_noise=0.0,
    )
    base.update(kw)
    return Config(**base)


# --------------------------------------------------------------------------
# The agent
# --------------------------------------------------------------------------
class Agent:
    def __init__(self, cfg, obs_dim, act_dim, act_limit):
        self.cfg = cfg
        self.act_dim = act_dim
        self.act_limit = act_limit

        if cfg.stochastic_actor:
            self.actor = SquashedGaussianActor(
                obs_dim, act_dim, cfg.hidden, act_limit, cfg.tanh_correction
            )
        else:
            self.actor = DeterministicActor(obs_dim, act_dim, cfg.hidden, act_limit)

        self.q1 = Critic(obs_dim, act_dim, cfg.hidden)
        self.q2 = Critic(obs_dim, act_dim, cfg.hidden) if cfg.twin_critics else None

        import copy

        self.q1_targ = copy.deepcopy(self.q1)
        self.q2_targ = copy.deepcopy(self.q2) if cfg.twin_critics else None
        # SAC does not need a target actor: its next action comes from the current
        # (stochastic) policy. DDPG/TD3 do, because their target action IS an argmax.
        self.actor_targ = None if cfg.stochastic_actor else copy.deepcopy(self.actor)

        q_params = list(self.q1.parameters())
        if cfg.twin_critics:
            q_params += list(self.q2.parameters())
        self.q_optim = torch.optim.Adam(q_params, lr=cfg.lr)
        self.pi_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)

        # The temperature. Optimized in log-space so it can never go negative.
        self.target_entropy = (
            cfg.target_entropy if cfg.target_entropy is not None else -float(act_dim)
        )
        self.log_alpha = torch.tensor(math.log(cfg.alpha), requires_grad=cfg.auto_alpha)
        self.alpha_optim = (
            torch.optim.Adam([self.log_alpha], lr=cfg.lr) if cfg.auto_alpha else None
        )
        self.updates = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item() if self.cfg.entropy_bonus else 0.0

    @torch.no_grad()
    def act(self, obs, deterministic=False, rng=None):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.cfg.stochastic_actor:
            a, _ = self.actor(o, deterministic=deterministic, with_logprob=False)
            a = a.squeeze(0).numpy()
        else:
            a = self.actor(o).squeeze(0).numpy()
            if not deterministic and self.cfg.act_noise > 0:
                a = a + self.cfg.act_noise * self.act_limit * rng.standard_normal(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    def update(self, batch):
        cfg = self.cfg
        o, a, r, o2, d = batch

        # ---- critic ----
        with torch.no_grad():
            if cfg.stochastic_actor:
                a2, logp2 = self.actor(o2)
            else:
                a2 = self.actor_targ(o2)
                if cfg.target_noise > 0:
                    # target policy smoothing: the critic should not be able to
                    # exploit a razor-thin peak in its own Q surface
                    eps = (torch.randn_like(a2) * cfg.target_noise * self.act_limit).clamp(
                        -cfg.noise_clip * self.act_limit, cfg.noise_clip * self.act_limit
                    )
                    a2 = (a2 + eps).clamp(-self.act_limit, self.act_limit)
                logp2 = None

            q_targ = self.q1_targ(o2, a2)
            if cfg.twin_critics:
                q_targ = torch.min(q_targ, self.q2_targ(o2, a2))
            if cfg.entropy_bonus:
                q_targ = q_targ - self.log_alpha.exp() * logp2.unsqueeze(-1)
            backup = r + cfg.gamma * (1 - d) * q_targ

        q1 = self.q1(o, a)
        q_loss = F.mse_loss(q1, backup)
        if cfg.twin_critics:
            q_loss = q_loss + F.mse_loss(self.q2(o, a), backup)

        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optim.step()

        self.updates += 1
        info = {"q_loss": q_loss.item(), "q_mean": q1.mean().item(), "alpha": self.alpha}

        # ---- actor (possibly delayed) ----
        if self.updates % cfg.policy_delay == 0:
            # Freeze the critic's params: we want dQ/da, not dQ/dphi. Without this
            # the actor's backward pass computes critic grads we then throw away —
            # correct, but ~25% of the update's time spent on nothing.
            for p in self.q1.parameters():
                p.requires_grad_(False)

            if cfg.stochastic_actor:
                pi, logp = self.actor(o)
                q_pi = self.q1(o, pi)
                if cfg.twin_critics:
                    for p in self.q2.parameters():
                        p.requires_grad_(False)
                    q_pi = torch.min(q_pi, self.q2(o, pi))
                    for p in self.q2.parameters():
                        p.requires_grad_(True)
                pi_loss = (self.log_alpha.exp().detach() * logp.unsqueeze(-1) - q_pi).mean()
            else:
                pi = self.actor(o)
                # THE deterministic policy gradient: maximize the critic's score of
                # the actor's action. dJ/dtheta = dQ/da * da/dtheta, by chain rule,
                # which autograd does for free once you write `-Q(s, mu(s)).mean()`.
                pi_loss = -self.q1(o, pi).mean()
                logp = None

            self.pi_optim.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.pi_optim.step()

            for p in self.q1.parameters():
                p.requires_grad_(True)

            info["pi_loss"] = pi_loss.item()
            if logp is not None:
                info["entropy"] = -logp.mean().item()

            # ---- temperature ----
            if cfg.auto_alpha:
                # Dual gradient descent: push alpha up when the policy is more
                # deterministic than the target entropy, down when it is more random.
                alpha_loss = -(
                    self.log_alpha * (logp.detach() + self.target_entropy)
                ).mean()
                self.alpha_optim.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_optim.step()

            # ---- targets ----
            soft_update(self.q1, self.q1_targ, cfg.tau)
            if cfg.twin_critics:
                soft_update(self.q2, self.q2_targ, cfg.tau)
            if self.actor_targ is not None:
                soft_update(self.actor, self.actor_targ, cfg.tau)

        return info


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def evaluate(env, agent, episodes, seed):
    """Deterministic evaluation: no exploration noise, no sampling."""
    total = 0.0
    for i in range(episodes):
        o, _ = env.reset(seed=seed + 10_000 + i)
        done = False
        while not done:
            a = agent.act(o, deterministic=True)
            o, r, term, trunc, _ = env.step(a)
            total += r
            done = term or trunc
    return total / episodes


def discounted_return(rewards, gamma):
    g = 0.0
    for r in reversed(rewards):
        g = r + gamma * g
    return g


def train(cfg, progress=False):
    """Train one agent. Returns a history dict — this is the only entry point."""
    set_seed(cfg.seed)  # BEFORE the nets are built
    rng = np.random.default_rng(cfg.seed)

    env = gym.make(cfg.env_id)
    eval_env = gym.make(cfg.env_id)
    # `env.reset(seed=...)` seeds the ENVIRONMENT's RNG. It does NOT seed the
    # ACTION SPACE's RNG, which is what `action_space.sample()` draws from — and
    # that is where the first `start_steps` warmup actions come from. Miss this
    # line and two runs with the same seed produce different results (measured:
    # -1314.79 vs -1247.73 on Pendulum), which looks exactly like ordinary RL
    # variance and is therefore nearly impossible to notice. The warmup is the
    # variable under test in project 26's exploration ablation, so an unseeded
    # warmup would have quietly invalidated that experiment.
    env.action_space.seed(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = Agent(cfg, obs_dim, act_dim, act_limit)
    buf = ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)

    hist = {
        "steps": [], "eval_return": [], "wall_time": [],
        "q_mean": [], "alpha": [], "entropy": [],
        "q_bias_steps": [], "q_pred": [], "q_true": [],
    }
    o, _ = env.reset(seed=cfg.seed)
    ep_obs, ep_acts, ep_rews = [], [], []
    t_start = time.time()
    last_info = {}

    for t in range(1, cfg.total_steps + 1):
        if t <= cfg.start_steps:
            a = env.action_space.sample()
        else:
            a = agent.act(o, rng=rng)

        o2, r, term, trunc, _ = env.step(a)
        # `term` only. A timeout (`trunc`) is not the world ending — the value of
        # the next state is still real, so it must NOT be zeroed in the bootstrap.
        # Conflating the two teaches the critic that the world ends at 1000 steps.
        buf.add(o, a, r, o2, float(term))

        if cfg.log_q_bias:
            ep_obs.append(o.copy())
            ep_acts.append(np.asarray(a, dtype=np.float32).copy())
            ep_rews.append(r)

        o = o2
        if term or trunc:
            if cfg.log_q_bias and len(ep_rews) > 0:
                # What the critic *predicted* at the start of this episode, versus
                # what the episode actually paid. The gap is overestimation bias,
                # measured rather than asserted.
                with torch.no_grad():
                    q_pred = agent.q1(
                        torch.as_tensor(np.array(ep_obs[:1]), dtype=torch.float32),
                        torch.as_tensor(np.array(ep_acts[:1]), dtype=torch.float32),
                    ).item()
                hist["q_bias_steps"].append(t)
                hist["q_pred"].append(q_pred)
                hist["q_true"].append(discounted_return(ep_rews, cfg.gamma))
            ep_obs, ep_acts, ep_rews = [], [], []
            o, _ = env.reset()

        if t >= cfg.update_after and t % cfg.update_every == 0:
            for _ in range(cfg.update_every):
                last_info = agent.update(buf.sample(cfg.batch_size, rng))

        if t % cfg.eval_every == 0:
            ret = evaluate(eval_env, agent, cfg.eval_episodes, cfg.seed)
            hist["steps"].append(t)
            hist["eval_return"].append(ret)
            hist["wall_time"].append(time.time() - t_start)
            hist["q_mean"].append(last_info.get("q_mean", float("nan")))
            hist["alpha"].append(last_info.get("alpha", 0.0))
            hist["entropy"].append(last_info.get("entropy", float("nan")))
            if progress:
                print(
                    f"  [{cfg.env_id} seed={cfg.seed}] step {t:6d}  "
                    f"return {ret:8.1f}  alpha {last_info.get('alpha', 0):.3f}  "
                    f"({time.time() - t_start:.0f}s)",
                    flush=True,
                )

    hist["wall_total"] = time.time() - t_start
    env.close()
    eval_env.close()
    return hist, agent
