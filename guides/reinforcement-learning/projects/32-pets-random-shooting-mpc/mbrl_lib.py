"""The reusable model-based core for Phase 6.

Phases 4 and 5 were model-free: the agent learned a policy (or a Q-function) from
samples and never asked "what would happen if I did X?" Phase 6 learns the answer
to that question — a *dynamics model* f(s, a) -> (s', r) — and then spends it in
one of three ways. This file holds the two that are cheap enough to build from
scratch on a CPU:

    PLAN WITH IT   (projects 32, 33)  At every step, imagine many action sequences,
                                      score them with the model, and execute the
                                      first action of the best one.
    TRAIN ON IT    (project 34)       Generate fake transitions, mix them into a
                                      replay buffer, let SAC drink from it.

(Project 37, TD-MPC2, does both — but over a *learned latent space* rather than over
raw observations, so it shares these ideas and none of this code.)

The parts every project here needs are the same, so they live here once:

    GaussianEnsemble   the model itself: N networks, each predicting a *distribution*
    train_model        one fit of that ensemble to a replay buffer
    RandomShooting     the dumb planner (project 32)
    CEM                the smart planner (project 33)

`Planner` is an interface with one method, `plan(model, obs)`. Project 32 and
project 33 differ by *which object gets passed in* and nothing else — that is the
whole point of project 33, and it would be invisible if each project re-implemented
its own MPC loop.

Two implementation choices are load-bearing for the 10-minute budget:

1. The ensemble is ONE module, not a list of N modules. Weights are stored stacked
   as (n_members, in, out) and applied with `torch.baddbmm`, so all N members run in
   a single batched matmul. A Python list of 5 MLPs is the same arithmetic at ~4x
   the wall time, because on tensors this small the cost is per-op dispatch overhead,
   not FLOPs.
2. Planning replicates each candidate action sequence once per ensemble member and
   keeps the members on the leading axis, so the rollout is `horizon` batched matmuls
   total — not `horizon * n_members` separate ones.

CPU-only on purpose (this box's GPU is sm_61, too old for the installed torch), and
one torch thread per process: the parallelism that pays is N seeds on N cores.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cpu")
torch.set_num_threads(1)


def set_seed(seed):
    """Seed every RNG that can move a result — BEFORE any network is built.

    Torch seeds its default generator from OS entropy at import, so a net
    constructed before `manual_seed` has unseeded weights and the run cannot be
    reproduced. The symptom (curves that differ between identical invocations)
    looks exactly like ordinary RL variance, which is what makes it expensive.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class EnsembleLinear(nn.Module):
    """`nn.Linear` for N networks at once.

    Weight is (n, in, out) instead of (in, out), and the input carries the member
    on its leading axis: (n, batch, in) -> (n, batch, out). One `baddbmm` does all
    n members. This is the only reason a 5-member ensemble costs ~1.2x a single
    network here rather than 5x.
    """

    def __init__(self, n_members, in_dim, out_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_members, in_dim, out_dim))
        self.bias = nn.Parameter(torch.zeros(n_members, 1, out_dim))
        # Truncated-normal init, per-member, matching the fan-in convention PETS uses.
        std = 1.0 / (2.0 * math.sqrt(in_dim))
        with torch.no_grad():
            self.weight.normal_(0.0, std).clamp_(-2 * std, 2 * std)

    def forward(self, x):
        return torch.baddbmm(self.bias, x, self.weight)


class GaussianEnsemble(nn.Module):
    """The dynamics model: N networks, each predicting a Gaussian over (delta_s, r).

    Three design choices, each of which PETS argues for and each of which you can
    feel the absence of:

    * It predicts **delta_s = s' - s**, not s' directly. Consecutive states are
      nearly identical, so predicting s' means learning the identity function plus a
      small correction, and the small correction is the entire signal. Predicting the
      delta hands the identity part to arithmetic and lets the network spend its
      capacity on what actually changes.
    * It predicts a **variance**, not just a mean — "the pole will be here, and I am
      *this sure*". This is *aleatoric* uncertainty: the noise the model believes is
      inherent to the transition itself.
    * There are **N of them**, trained on different bootstrap samples of the data.
      Where the data is thick they agree; where it is thin they disagree wildly. That
      disagreement is *epistemic* uncertainty: what the model does not know. A planner
      that averages over the members automatically distrusts its own predictions in
      the places it has never visited, which is what stops MPC from happily planning a
      route through a region where the model has hallucinated free reward.

    `max_logvar` / `min_logvar` are learned bounds, softly enforced. Without them the
    Gaussian NLL below has a degenerate escape hatch: drive the variance to zero on
    the easy points, and the NLL runs to negative infinity without the mean improving
    at all.
    """

    def __init__(self, n_members, obs_dim, act_dim, hidden=200, n_layers=3):
        super().__init__()
        self.n_members = n_members
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.out_dim = obs_dim + 1  # delta_obs, then reward

        dims = [obs_dim + act_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList(
            [EnsembleLinear(n_members, dims[i], dims[i + 1]) for i in range(n_layers)]
        )
        self.head = EnsembleLinear(n_members, hidden, 2 * self.out_dim)

        self.max_logvar = nn.Parameter(torch.full((1, 1, self.out_dim), 0.5))
        self.min_logvar = nn.Parameter(torch.full((1, 1, self.out_dim), -10.0))

        # Input normalization. Held as buffers so they travel with the module.
        self.register_buffer("in_mu", torch.zeros(1, 1, obs_dim + act_dim))
        self.register_buffer("in_std", torch.ones(1, 1, obs_dim + act_dim))

    def fit_normalizer(self, obs, act):
        x = np.concatenate([obs, act], axis=-1)
        self.in_mu[:] = torch.as_tensor(x.mean(0), dtype=torch.float32)
        std = torch.as_tensor(x.std(0), dtype=torch.float32)
        self.in_std[:] = torch.clamp(std, min=1e-6)

    def forward(self, obs, act):
        """(n, B, obs), (n, B, act) -> mean (n, B, obs+1), logvar (n, B, obs+1).

        The mean's first `obs_dim` entries are the *delta*; the last is the reward.
        """
        x = torch.cat([obs, act], dim=-1)
        x = (x - self.in_mu) / self.in_std
        for layer in self.layers:
            x = F.silu(layer(x))
        out = self.head(x)
        mean, logvar = out.chunk(2, dim=-1)
        # Soft bounds: inside the range this is a no-op, outside it saturates smoothly.
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def loss(self, obs, act, target):
        """Gaussian negative log-likelihood, summed over members.

        Per element:  0.5 * [ (target - mu)^2 / var + log var ]

        Read the two terms as a bargain the model strikes with itself. The first says
        "be accurate"; the second says "don't hedge". Widening the variance shrinks the
        first term but grows the second, so the model can only buy forgiveness for a bad
        prediction by admitting, in the loss, that it was unsure — and it pays for that
        admission. The result is a variance that actually tracks the model's error
        instead of being a free parameter.
        """
        mean, logvar = self(obs, act)
        inv_var = torch.exp(-logvar)
        mse = ((mean - target) ** 2 * inv_var).mean(dim=(1, 2))
        var_term = logvar.mean(dim=(1, 2))
        loss = (mse + var_term).sum()
        # Nudge the learned bounds inward; keeps them from drifting off to infinity.
        loss = loss + 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()
        return loss

    @torch.no_grad()
    def predict(self, obs, act, sample=True, generator=None):
        """One imagined step. (n, B, obs), (n, B, act) -> next_obs, reward.

        `sample=True` draws from each member's predicted Gaussian instead of taking
        its mean. This is PETS's "trajectory sampling": the particle inherits both
        kinds of uncertainty — which member you asked (epistemic) and the noise that
        member believes in (aleatoric). Planning on means alone throws away exactly the
        signal the ensemble was built to produce.
        """
        mean, logvar = self(obs, act)
        if sample:
            std = torch.exp(0.5 * logvar)
            noise = torch.randn(mean.shape, generator=generator, device=mean.device)
            mean = mean + std * noise
        next_obs = obs + mean[..., : self.obs_dim]
        reward = mean[..., self.obs_dim]
        return next_obs, reward


# --------------------------------------------------------------------------
# Data + training
# --------------------------------------------------------------------------
class TransitionBuffer:
    """Every real transition ever seen. The model trains on all of it, every time."""

    def __init__(self, obs_dim, act_dim, capacity=100_000):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.size = 0
        self.capacity = capacity

    def add(self, o, a, r, o2):
        i = self.size
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i] = o, a, r, o2
        self.size = min(i + 1, self.capacity - 1)

    def arrays(self):
        n = self.size
        return self.obs[:n], self.act[:n], self.rew[:n], self.next_obs[:n]


def train_model(model, buf, epochs=40, batch_size=256, lr=1e-3, holdout=0.1, rng=None):
    """Fit the ensemble to everything in the buffer. Returns (train_nll, holdout_mse).

    Each member sees its own bootstrap resample of the data — sampling with
    replacement, so member 3 might see a transition twice and never see another. If
    all members saw identical data in an identical order they would converge to nearly
    identical functions, their disagreement would collapse to zero, and the ensemble
    would be an expensive way to have one model. The bootstrap is what keeps them
    genuinely different, and their difference is the uncertainty estimate.
    """
    rng = rng or np.random.default_rng(0)
    obs, act, rew, next_obs = buf.arrays()
    n = len(obs)

    idx = rng.permutation(n)
    n_hold = max(1, int(n * holdout))
    hold_idx, train_idx = idx[:n_hold], idx[n_hold:]

    model.fit_normalizer(obs[train_idx], act[train_idx])
    target = np.concatenate([next_obs - obs, rew], axis=-1).astype(np.float32)

    obs_t = torch.as_tensor(obs)
    act_t = torch.as_tensor(act)
    tgt_t = torch.as_tensor(target)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    E = model.n_members
    n_train = len(train_idx)

    # The bootstrap: one resample of the training indices per member, drawn once and
    # reused across epochs (so a member's dataset is a fixed dataset, not a new one
    # every epoch — otherwise every member converges to the mean of all the data).
    boot = np.stack([rng.choice(train_idx, size=n_train, replace=True) for _ in range(E)])

    last = 0.0
    for _ in range(epochs):
        perm = rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            sel = perm[start : start + batch_size]
            if len(sel) < 2:
                continue
            b = boot[:, sel]  # (E, B) — a different slice of data per member
            bi = torch.as_tensor(b, dtype=torch.long)
            loss = model.loss(obs_t[bi], act_t[bi], tgt_t[bi])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            last = loss.item()

    with torch.no_grad():
        hi = torch.as_tensor(hold_idx, dtype=torch.long)
        ho = obs_t[hi].unsqueeze(0).expand(E, -1, -1)
        ha = act_t[hi].unsqueeze(0).expand(E, -1, -1)
        mean, _ = model(ho, ha)
        # Holdout error of the ensemble MEAN — the number that says whether the model
        # is any good, and the one to watch instead of the training NLL (which can fall
        # happily while the model overfits).
        hold_mse = ((mean.mean(0) - tgt_t[hi]) ** 2).mean().item()
    return last, hold_mse


# --------------------------------------------------------------------------
# Planners
# --------------------------------------------------------------------------
@dataclass
class PlanConfig:
    horizon: int = 15
    n_candidates: int = 400     # action sequences scored per decision
    n_iters: int = 1            # CEM refinement rounds (1 == random shooting)
    n_elites: int = 40
    gamma: float = 0.99
    alpha: float = 0.1          # CEM: how much of the old distribution to keep
    init_std: float = 1.0       # CEM: starting spread, as a fraction of the action range


class Planner:
    """Score action sequences with the model; return the first action of the best one.

    This is Model Predictive Control (MPC). Note what is NOT here: a policy network.
    The agent has no learned habit of what to do — every single timestep it re-imagines
    the future from scratch and re-decides. The "policy" is an emergent side effect of
    a model plus a search.
    """

    def __init__(self, act_dim, act_limit, cfg: PlanConfig):
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.cfg = cfg

    @torch.no_grad()
    def _score(self, model, obs, actions, generator):
        """Roll `actions` (C, H, act_dim) through the model. -> (C,) predicted return.

        Every candidate is evaluated once per ensemble member, with the member on the
        leading axis, so the whole population moves forward in lock-step: `horizon`
        batched matmuls for the entire search, regardless of how many candidates or
        members there are. The candidate's score is the mean over members — which is
        where "trust the model only where the ensemble agrees" quietly happens, because
        a candidate that looks great to one member and catastrophic to another averages
        out to unremarkable and loses to a candidate they all like.
        """
        E = model.n_members
        C, H, _ = actions.shape
        o = obs.view(1, 1, -1).expand(E, C, -1).contiguous()
        returns = torch.zeros(E, C)
        disc = 1.0
        for t in range(H):
            a = actions[:, t].unsqueeze(0).expand(E, -1, -1)
            o, r = model.predict(o, a, sample=True, generator=generator)
            returns += disc * r
            disc *= self.cfg.gamma
        return returns.mean(0)

    def plan(self, model, obs, generator=None):
        raise NotImplementedError

    def reset(self):
        pass


class RandomShooting(Planner):
    """Guess a lot, keep the best guess. The whole algorithm is two lines.

    Sample `n_candidates` action sequences uniformly at random, score them all, take
    the first action of the winner. It is not clever and it does not need to be — with
    a good model and a 1-D action, uniform guessing covers the space well enough. Its
    weakness is dimensional: the number of sequences needed to stumble onto a good one
    by luck grows exponentially in horizon x action_dim. Project 33 is what you do when
    that stops working.
    """

    @torch.no_grad()
    def plan(self, model, obs, generator=None):
        cfg = self.cfg
        actions = (
            torch.rand(
                cfg.n_candidates, cfg.horizon, self.act_dim, generator=generator
            )
            * 2
            - 1
        ) * self.act_limit
        returns = self._score(model, obs, actions, generator)
        best = returns.argmax()
        return actions[best, 0].numpy(), returns[best].item()


class CEM(Planner):
    """Cross-Entropy Method: guess, keep the winners, guess again near the winners.

    Random shooting throws darts blindfolded. CEM throws a handful, looks at where the
    best ones landed, moves the dartboard there, and throws again — `n_iters` times.
    Concretely it keeps a Gaussian over action sequences, and each round:

        1. sample `n_candidates` sequences from it
        2. score them with the model
        3. keep the top `n_elites` ("the elites")
        4. refit the Gaussian's mean and std to just those elites

    Step 4 is the name: fitting a distribution to a set of good samples is minimizing a
    cross-entropy between them, the same quantity the cross-entropy *loss* minimizes
    when it fits a classifier to correct labels. Same idea, different target — here the
    "labels" are the action sequences that scored well.

    The mean carries over between timesteps (`self.mean`), shifted forward by one step.
    The plan you committed to last tick is an excellent first guess for this tick — the
    world only moved 0.05 seconds — and warm-starting like this buys most of an extra
    CEM iteration for free.
    """

    def __init__(self, act_dim, act_limit, cfg: PlanConfig):
        super().__init__(act_dim, act_limit, cfg)
        self.mean = None

    def reset(self):
        self.mean = None

    @torch.no_grad()
    def plan(self, model, obs, generator=None):
        cfg = self.cfg
        H, A = cfg.horizon, self.act_dim

        if self.mean is None:
            mean = torch.zeros(H, A)
        else:
            # Warm start: shift last tick's plan one step earlier, pad with a zero.
            mean = torch.cat([self.mean[1:], torch.zeros(1, A)], dim=0)
        std = torch.full((H, A), cfg.init_std * self.act_limit)

        best_ret = float("-inf")
        best_first = None
        for _ in range(cfg.n_iters):
            noise = torch.randn(cfg.n_candidates, H, A, generator=generator)
            actions = (mean + std * noise).clamp(-self.act_limit, self.act_limit)
            returns = self._score(model, obs, actions, generator)

            elite_idx = returns.topk(cfg.n_elites).indices
            elites = actions[elite_idx]

            # Refit. `alpha` is momentum: keep a little of the old distribution so a
            # single unlucky batch of elites cannot yank the plan somewhere silly.
            new_mean = elites.mean(0)
            new_std = elites.std(0).clamp(min=1e-3)
            mean = cfg.alpha * mean + (1 - cfg.alpha) * new_mean
            std = cfg.alpha * std + (1 - cfg.alpha) * new_std

            top = returns[elite_idx[0]].item()
            if top > best_ret:
                best_ret = top
                best_first = actions[elite_idx[0], 0].clone()

        self.mean = mean
        return best_first.numpy(), best_ret


# --------------------------------------------------------------------------
# The MPC training loop
# --------------------------------------------------------------------------
@dataclass
class MBRLConfig:
    env_id: str = "Pendulum-v1"
    seed: int = 0
    n_members: int = 5
    hidden: int = 200
    n_layers: int = 3
    model_epochs: int = 40
    model_lr: float = 1e-3
    batch_size: int = 256
    n_warmup_episodes: int = 1   # random episodes before the model is worth anything
    n_episodes: int = 12
    plan: PlanConfig = None


def run_mpc(cfg: MBRLConfig, planner_cls, progress=False):
    """The whole PETS/CEM-MPC loop: collect -> refit the model -> plan with it.

    Each iteration is one real episode. Notice there is no policy gradient, no critic,
    no target network — nothing from Phases 4 or 5. The only thing being *learned* is
    the model; the behaviour is produced fresh by search, every step.
    """
    import gymnasium as gym

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 777)

    env = gym.make(cfg.env_id)
    # Seeds the ACTION SPACE's RNG, which `action_space.sample()` draws from. Seeding
    # only `env.reset(seed=...)` leaves the warmup episodes' random actions unseeded,
    # and two "identical" runs then diverge — which looks exactly like ordinary RL
    # variance and is therefore nearly impossible to catch.
    env.action_space.seed(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    model = GaussianEnsemble(cfg.n_members, obs_dim, act_dim, cfg.hidden, cfg.n_layers)
    buf = TransitionBuffer(obs_dim, act_dim)
    planner = planner_cls(act_dim, act_limit, cfg.plan)

    hist = {
        "episode": [], "steps": [], "return": [], "hold_mse": [],
        "wall": [], "plan_return": [],
    }
    total_steps = 0
    t0 = time.time()

    for ep in range(cfg.n_warmup_episodes + cfg.n_episodes):
        warmup = ep < cfg.n_warmup_episodes
        o, _ = env.reset(seed=cfg.seed + ep)
        planner.reset()
        done = False
        ep_ret = 0.0
        plan_rets = []

        while not done:
            if warmup:
                a = env.action_space.sample()
            else:
                a, pr = planner.plan(model, torch.as_tensor(o, dtype=torch.float32), gen)
                a = np.clip(a, -act_limit, act_limit)
                plan_rets.append(pr)
            o2, r, term, trunc, _ = env.step(a)
            buf.add(o, a, r, o2)
            ep_ret += r
            total_steps += 1
            o = o2
            done = term or trunc

        # Refit on ALL data collected so far, from scratch each time. The model is
        # cheap and the data is precious; there is no reason to be incremental.
        _, hold_mse = train_model(
            model, buf, epochs=cfg.model_epochs, batch_size=cfg.batch_size,
            lr=cfg.model_lr, rng=rng,
        )

        hist["episode"].append(ep)
        hist["steps"].append(total_steps)
        hist["return"].append(ep_ret)
        hist["hold_mse"].append(hold_mse)
        hist["wall"].append(time.time() - t0)
        hist["plan_return"].append(float(np.mean(plan_rets)) if plan_rets else float("nan"))

        if progress:
            tag = "warmup" if warmup else "  mpc "
            print(
                f"  [seed {cfg.seed}] {tag} ep {ep:2d}  steps {total_steps:5d}  "
                f"return {ep_ret:8.1f}  holdout_mse {hold_mse:.4f}  "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

    env.close()
    hist["wall_total"] = time.time() - t0
    hist["model"] = model
    hist["buffer"] = buf
    return hist
