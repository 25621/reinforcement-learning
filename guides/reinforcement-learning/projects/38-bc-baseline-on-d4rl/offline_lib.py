"""The reusable offline-RL core for Phase 7.

Phases 4-6 could always ask the environment one more question. Phase 7 cannot.
You get a file of past experience and that is all you will ever get, which breaks
the one assumption every previous algorithm was quietly built on:

    Q-learning's target is   r + gamma * max_a' Q(s', a').

    Online, that `max` is safe. If it picks a wrong action, the agent TRIES that
    action, the environment says "no", and the error is corrected.

    Offline, nothing corrects it. The `max` deliberately hunts for the action
    with the highest predicted value, and the highest predicted values live
    exactly where the network has never seen data and is free to make things up.
    The agent then learns to love those made-up actions. Nobody ever tells it no.

Every algorithm in this file is one answer to that paragraph:

    bc       don't learn values at all; just copy the data     (project 38)
    naive_q  do the forbidden thing, and watch it explode      (project 39)
    cql      keep the max, but PUNISH it for being optimistic  (project 40)
    iql      never compute a max over unseen actions at all    (project 41)

Written as four files those would look like four algorithms. They are one loop
with a different loss, which is why there is exactly one `OfflineAgent` here and
each project sets `algo=`. If a project could be written by changing a string and
it instead copy-pasted the agent, the reader would learn that these are unrelated
methods. They would be learning something false.

(The Decision Transformer, project 42, genuinely IS a different thing — it has no
value function and no Bellman backup at all — so it lives in its own file.)

Everything is CPU-only on purpose, as in Phase 5: these networks are two hidden
layers, and this box's GPU (sm_61) is too old for the installed torch anyway.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-ddpg-on-pendulum"))
from cc_lib import mlp, set_seed, soft_update  # noqa: E402  (Phase 5's building blocks)

torch.set_num_threads(1)  # N projects on N cores beats one project on N threads

DEVICE = torch.device("cpu")
ENV_ID = "HalfCheetah-v5"  # see make_teachers.py for why this body and not Hopper
LEVELS = ("random", "medium", "expert")
TEACHERS = HERE / "teachers"
DATA = HERE / "data"
N_TRANSITIONS = 100_000
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


# ==========================================================================
# 1. The dataset
# ==========================================================================
def load_teacher(level):
    """Load one frozen behavior policy (see make_teachers.py).

    Returns `None` for the actor when the teacher is the uniform-random one — there
    is no network to load, because "shake the joystick" is not a network.
    """
    ck = torch.load(TEACHERS / f"{level}.pt", weights_only=False)
    if ck["uniform"]:
        return None, ck
    actor = TanhGaussianPolicy(ck["obs_dim"], ck["act_dim"], ck["hidden"], ck["act_limit"])
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    return actor, ck


def build_dataset(level, n=N_TRANSITIONS, seed=0):
    """Roll out a frozen teacher to produce a fixed dataset, and cache it.

    The teacher acts by SAMPLING from its policy, not by taking its best guess.
    That matters: a dataset with no variety would show every algorithm the same
    single trajectory and there would be nothing to learn from and nothing to
    stitch together. Real logged data is noisy for the same reason — the thing
    that generated it was not perfect either. D4RL does exactly this.
    """
    DATA.mkdir(exist_ok=True)
    path = DATA / f"{ENV_ID}-{level}-{n}.npz"
    if path.exists():
        return dict(np.load(path))

    actor, ck = load_teacher(level)
    env = gym.make(ENV_ID)
    rng_seed = seed + 1234 * LEVELS.index(level)
    o, _ = env.reset(seed=rng_seed)
    # Seeding the env does NOT seed the action space — `action_space.sample()` draws
    # from a separate RNG. Miss this line and the `random` dataset is different on
    # every run, which looks like ordinary RL noise and is nearly impossible to spot.
    # Phase 5 lost real time to exactly this bug.
    env.action_space.seed(rng_seed)
    torch.manual_seed(rng_seed)

    obs, act, rew = np.zeros((n, ck["obs_dim"]), np.float32), np.zeros((n, ck["act_dim"]), np.float32), np.zeros(n, np.float32)
    nobs, term, tout = np.zeros((n, ck["obs_dim"]), np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)

    for i in range(n):
        if actor is None:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0),
                             deterministic=False)
            a = a.squeeze(0).numpy()
        o2, r, te, tr, _ = env.step(a)
        obs[i], act[i], rew[i], nobs[i], term[i], tout[i] = o, a, r, o2, float(te), float(tr)
        o = o2
        if te or tr:
            o, _ = env.reset()
    env.close()

    d = {"obs": obs, "act": act, "rew": rew, "next_obs": nobs, "terminal": term, "timeout": tout}
    np.savez_compressed(path, **d)
    return d


class Dataset:
    """A fixed dataset, with the observation normalization every offline method needs.

    Why normalize: HalfCheetah's observations mix joint angles near 0.1 with joint
    velocities that swing past 10. A network fed raw values spends its early
    training just learning to rescale them, and the big-magnitude inputs dominate
    the gradient. Online RL survives this sloppiness because it can always collect
    more data. Offline RL cannot, so we hand it clean inputs.

    The mean and std are computed from the DATASET, never from the environment —
    using anything the dataset does not contain would be cheating, and cheating in
    a way that would not be available in the real applications (medical records,
    logged recommendations) this whole phase is for.
    """

    def __init__(self, level, n=N_TRANSITIONS, seed=0):
        d = build_dataset(level, n, seed)
        self.level = level
        self.obs_mean = d["obs"].mean(0, keepdims=True)
        self.obs_std = d["obs"].std(0, keepdims=True) + 1e-3

        self.obs = torch.as_tensor((d["obs"] - self.obs_mean) / self.obs_std)
        self.next_obs = torch.as_tensor((d["next_obs"] - self.obs_mean) / self.obs_std)
        self.act = torch.as_tensor(d["act"])
        self.rew = torch.as_tensor(d["rew"]).unsqueeze(-1)
        self.done = torch.as_tensor(d["terminal"]).unsqueeze(-1)
        self.raw = d
        self.n = len(self.obs)
        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.act.shape[1]

    def sample(self, batch_size, rng):
        i = torch.as_tensor(rng.integers(0, self.n, size=batch_size))
        return self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i]

    def episode_returns(self):
        """Undiscounted return of every complete episode in the dataset."""
        ends = np.nonzero((self.raw["terminal"] + self.raw["timeout"]) > 0)[0]
        rets, start = [], 0
        for e in ends:
            rets.append(float(self.raw["rew"][start:e + 1].sum()))
            start = e + 1
        return np.array(rets)


# ==========================================================================
# 2. Scoring — the D4RL normalized score
# ==========================================================================
def score_bounds():
    """(random_return, expert_return): the two ends of the ruler."""
    return (torch.load(TEACHERS / "random.pt", weights_only=False)["eval_return"],
            torch.load(TEACHERS / "expert.pt", weights_only=False)["eval_return"])


def true_value_of_the_data(level, gamma=0.99):
    """The EXACT value of the dataset's own actions. No network involved.

    Q(s, a) means "the discounted reward you collect from here, if you take `a` and
    then carry on behaving as you did." Offline, we HAVE the recording — so for every
    row we can just add up what actually happened next:

        G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...

    This is not an estimate. It is the answer, read off the tape. Online RL can never
    do this (the future has not happened yet), and it is what lets projects 39-41 say
    "the critic is wrong by a factor of 9" instead of "the curves look a bit high".
    """
    d = build_dataset(level)
    r = d["rew"]
    ends = np.nonzero((d["terminal"] + d["timeout"]) > 0)[0]
    G, start = np.zeros_like(r), 0
    for e in ends:
        e = int(e) + 1
        g = 0.0
        for t in range(e - 1, start - 1, -1):   # walk backwards through the episode
            g = r[t] + gamma * g
            G[t] = g
        start = e
    return float(G[:start].mean())


def q_ceiling(gamma=0.99):
    """The largest value ANY state-action in this environment can possibly have.

        Q(s, a) = r_t + gamma*r_{t+1} + gamma^2*r_{t+2} + ...
                <= r_max * (1 + gamma + gamma^2 + ...) = r_max / (1 - gamma)

    Even a policy that somehow collected the single best reward in the dataset at
    EVERY step forever could not beat this. It is arithmetic, not an empirical
    observation, and it gives projects 39-41 a line to hold the critic against: any Q
    above it is not "too optimistic", it is describing something that cannot exist.

    Computed over all three datasets, not just the one being trained on, so that every
    project in the phase quotes the SAME number and their tables can be compared.
    """
    r_max = max(float(build_dataset(lv)["rew"].max()) for lv in LEVELS)
    return r_max / (1 - gamma)


def normalized_score(ret):
    """0 = as good as the random teacher, 100 = as good as the expert teacher.

    Raw returns are unreadable across tasks ("is 1,800 good?"). This is D4RL's
    convention and it makes the whole phase comparable on one axis. Note that a
    score CAN exceed 100 — an offline learner that stitches together the best
    pieces of many mediocre episodes can beat the policy that recorded them.
    """
    lo, hi = score_bounds()
    return 100.0 * (ret - lo) / (hi - lo)


# ==========================================================================
# 3. Networks
# ==========================================================================
class TanhGaussianPolicy(nn.Module):
    """State -> a distribution over actions, squashed into the action bounds.

    Same actor as SAC in Phase 5 (`cc_lib.SquashedGaussianActor`), re-declared here
    with one addition it needs offline: `log_prob(s, a)`, the probability the policy
    assigns to an action SOMEONE ELSE chose. Online, a policy is only ever asked
    about its own actions. Offline, every method here is built on grading the
    dataset's actions, so it must be able to score an action it did not pick.
    """

    def __init__(self, obs_dim, act_dim, hidden, act_limit=1.0):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden], out_act=nn.ReLU)
        self.mu_head = nn.Linear(hidden, act_dim)
        self.log_std_head = nn.Linear(hidden, act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False):
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        u = mu if deterministic else mu + log_std.exp() * torch.randn_like(mu)
        a = torch.tanh(u) * self.act_limit
        return a, mu

    def log_prob(self, obs, act):
        """log pi(a | s) for an action the dataset chose, not one we sampled.

        Two traps live in these four lines.

        1. `act` arrives already squashed (it came out of a tanh). To score it we
           must undo the squash — `atanh` — to recover the pre-squash `u`. But
           `atanh(±1) = ±inf`, and a saturated teacher emits actions that round to
           exactly ±1 in float32. So we clamp first. Without the clamp the loss is
           NaN within a few hundred steps and it looks like a learning-rate problem.
        2. Bending a density through `tanh` CHANGES it, so the log-probability needs
           a change-of-variables correction (the same one project 30 audited). Drop
           it and BC still trains and still looks fine — it just quietly optimizes
           the wrong objective, which is the worst kind of bug.
        """
        h = self.net(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        a = (act / self.act_limit).clamp(-0.999999, 0.999999)
        u = torch.atanh(a)
        logp = (-0.5 * (((u - mu) / log_std.exp()) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1)
        # numerically stable log(1 - tanh(u)^2), summed over action dims
        logp = logp - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        return logp.unsqueeze(-1)


class Critic(nn.Module):
    """Q(s, a) -> a scalar."""

    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, hidden, hidden, 1])

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class ValueNet(nn.Module):
    """V(s) -> a scalar. Only IQL has one, and that is the entire point of IQL."""

    def __init__(self, obs_dim, hidden):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 1])

    def forward(self, obs):
        return self.net(obs)


# ==========================================================================
# 4. One agent, four algorithms
# ==========================================================================
@dataclass
class OfflineConfig:
    algo: str = "iql"           # "bc" | "naive_q" | "cql" | "iql"
    level: str = "medium"
    seed: int = 0
    grad_steps: int = 30_000
    batch_size: int = 256
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005          # polyak rate for the target critic
    # Twin critics (take the MIN of two independent Q-nets) are TD3's fix for
    # overestimation ONLINE. Project 39 turns them off to see the unprotected
    # failure, and on to prove the online fix does not rescue the offline setting.
    twin_critics: bool = True

    # --- CQL (project 40) ---
    cql_alpha: float = 5.0      # how hard to push down unseen actions. 0 == naive_q
    cql_n_actions: int = 10     # how many unseen actions to push down per state
    # The penalty is an AVERAGE over states, so it does not need every state in the
    # batch — a random subset estimates the same average, just a little noisier.
    #
    # This matters enormously. The penalty evaluates the critic on `2*n_actions*batch`
    # rows, so at the full batch of 256 it does 20x the work of the TD loss it is
    # bolted onto: 10.7 updates/s, i.e. 47 MINUTES for one run, and this project needs
    # five of them. Measured on this CPU:
    #
    #     cql_batch    256     64     32     16
    #     updates/s   10.7   39.5   54.2   67.8
    #
    # Cost is driven by the PRODUCT n_actions * cql_batch, so we could have cut either.
    # We cut states, not actions, on purpose: `logsumexp` over few actions is a biased
    # estimate of the soft maximum (it systematically underestimates), whereas averaging
    # over few states is merely noisy — and noise averages out over 20,000 updates while
    # bias does not. Keep the 10 actions; charge only 16 states for them.
    cql_batch: int = 16

    # --- IQL (project 41) ---
    expectile: float = 0.7      # 0.5 == plain mean regression; ->1 == "assume the best"
    awr_beta: float = 3.0       # how sharply to prefer good dataset actions
    awr_clip: float = 100.0     # cap on the exponential weight

    eval_every: int = 5_000
    eval_episodes: int = 10


class OfflineAgent:
    def __init__(self, cfg, obs_dim, act_dim):
        self.cfg = cfg
        self.actor = TanhGaussianPolicy(obs_dim, act_dim, cfg.hidden)
        self.pi_optim = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.act_dim = act_dim
        self.needs_q = cfg.algo != "bc"

        if self.needs_q:
            import copy
            self.q1 = Critic(obs_dim, act_dim, cfg.hidden)
            self.q2 = Critic(obs_dim, act_dim, cfg.hidden) if cfg.twin_critics else None
            self.q1_targ = copy.deepcopy(self.q1)
            self.q2_targ = copy.deepcopy(self.q2) if cfg.twin_critics else None
            qp = list(self.q1.parameters())
            if cfg.twin_critics:
                qp += list(self.q2.parameters())
            self.q_optim = torch.optim.Adam(qp, lr=cfg.lr)
        if cfg.algo == "iql":
            self.vnet = ValueNet(obs_dim, cfg.hidden)
            self.v_optim = torch.optim.Adam(self.vnet.parameters(), lr=cfg.lr)

    # ---------------- the four losses ----------------
    def update(self, batch):
        return getattr(self, f"_update_{self.cfg.algo}")(batch)

    def _q_min(self, o, a, target=False):
        """min(Q1, Q2), or just Q1 when twin critics are switched off.

        Taking the MINIMUM of two independently-initialized critics is TD3's cure for
        overestimation: a single critic, maximized over actions, is a machine for
        finding its own most optimistic mistakes, and two of them rarely make the same
        mistake in the same place. Project 39 removes it to show the disease untreated,
        then puts it back to show that this online cure does not cure the offline one.
        """
        q1net = self.q1_targ if target else self.q1
        q = q1net(o, a)
        if self.cfg.twin_critics:
            q2net = self.q2_targ if target else self.q2
            q = torch.min(q, q2net(o, a))
        return q

    def _update_bc(self, batch):
        """Behavior cloning: supervised learning, with the rewards thrown in the bin.

        There is no Q, no target network, no bootstrapping — so there is nothing to
        explode. That safety is exactly what BC trades its ceiling for: it cannot be
        better than the hand that wrote the data, because "be like the data" is the
        whole objective.
        """
        o, a, _, _, _ = batch
        loss = -self.actor.log_prob(o, a).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.pi_optim.step()
        return {"pi_loss": loss.item()}

    def _q_target(self, r, o2, d):
        """The forbidden max: bootstrap through the action the ACTOR wants at s'.

        The actor is trained to maximize Q, so `self.actor(o2)` is, by construction,
        a search for the action Q likes best at s' — a continuous stand-in for
        `max_a' Q(s', a')`. Nothing here checks whether that action appears anywhere
        in the dataset. Nothing here can.
        """
        with torch.no_grad():
            a2, _ = self.actor(o2)
            return r + self.cfg.gamma * (1 - d) * self._q_min(o2, a2, target=True)

    def _update_naive_q(self, batch, cql_alpha=0.0):
        """Offline TD3, with no protection whatsoever. Project 39 watches it burn.

        Note this is a *strong* online algorithm — twin critics, target networks,
        the works. It is not failing because it is a weak implementation. It fails
        because of the fixed dataset, and that is the lesson.
        """
        o, a, r, o2, d = batch
        backup = self._q_target(r, o2, d)
        q1 = self.q1(o, a)
        q_loss = F.mse_loss(q1, backup)
        q2 = None
        if self.cfg.twin_critics:
            q2 = self.q2(o, a)
            q_loss = q_loss + F.mse_loss(q2, backup)

        info = {"q_data": q1.mean().item(), "q_loss": q_loss.item()}

        if cql_alpha > 0:
            pen, q_ood = self._cql_penalty(o, q1, q2)
            q_loss = q_loss + cql_alpha * pen
            info["cql_penalty"] = pen.item()
            info["q_ood"] = q_ood

        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optim.step()

        # actor: pick the action the critic scores highest (DDPG's deterministic
        # policy gradient). This is the step that turns a hallucinating critic into
        # a hallucinating POLICY.
        qparams = [p for net in (self.q1, self.q2) if net is not None
                   for p in net.parameters()]
        for p in qparams:
            p.requires_grad_(False)
        pi, _ = self.actor(o)
        pi_loss = -self._q_min(o, pi).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_optim.step()
        for p in qparams:
            p.requires_grad_(True)

        soft_update(self.q1, self.q1_targ, self.cfg.tau)
        if self.cfg.twin_critics:
            soft_update(self.q2, self.q2_targ, self.cfg.tau)
        info["pi_loss"] = pi_loss.item()
        # Q at the action the POLICY wants — the out-of-distribution action, the one
        # that appears in no dataset row. `pi_loss` is already its negative mean, so
        # this costs nothing to log, and project 39 is entirely about watching it.
        info["q_pi"] = -pi_loss.item()
        return info

    def _cql_penalty(self, o, q1_data, q2_data):
        """CQL's one extra term:  logsumexp over actions  -  Q at the data's action.

        Read it as a tug of war on the Q surface:

            logsumexp_a Q(s, a)   is a soft maximum over MANY actions, most of them
                                  never seen. Minimizing it pushes the whole surface
                                  DOWN, hardest wherever it is highest.
            Q(s, a_data)          is the value of the action really taken. Subtracting
                                  it pulls that one point back UP.

        Net effect: the only actions allowed to keep a high value are the ones the
        dataset actually contains. Everything the network was tempted to daydream
        about gets flattened. That is what "pessimism" means, concretely.

        (The real CQL importance-weights the sampled actions; we sample uniformly
        and from the current policy, which is the common simplified form and shows
        the same behavior.)
        """
        cfg = self.cfg
        # Only the first `cql_batch` states pay the penalty. The batch is already a
        # random sample of the dataset, so its first 64 rows are a random sample too —
        # no shuffling needed.
        B = min(cfg.cql_batch, o.shape[0])
        N = cfg.cql_n_actions
        o, q1_data = o[:B], q1_data[:B]
        o_rep = o.unsqueeze(1).expand(B, N, o.shape[1]).reshape(B * N, -1)

        a_unif = torch.empty(B * N, self.act_dim).uniform_(-1, 1)
        with torch.no_grad():
            a_pi, _ = self.actor(o_rep)
        a_all = torch.cat([a_unif, a_pi], 0)
        o_all = torch.cat([o_rep, o_rep], 0)

        q1_all = self.q1(o_all, a_all).view(B, 2 * N)
        pen = torch.logsumexp(q1_all, dim=1).mean() - q1_data.mean()
        if cfg.twin_critics:
            q2_all = self.q2(o_all, a_all).view(B, 2 * N)
            pen = pen + torch.logsumexp(q2_all, dim=1).mean() - q2_data[:B].mean()
        return pen, q1_all.mean().item()

    def _update_cql(self, batch):
        return self._update_naive_q(batch, cql_alpha=self.cfg.cql_alpha)

    def _update_iql(self, batch):
        """IQL: three losses, and NOT ONE of them evaluates an unseen action.

        Scan the code below for a network being called on an action that did not
        come out of the dataset. There isn't one. That is the whole algorithm.
        """
        cfg = self.cfg
        o, a, r, o2, d = batch

        # (1) V by expectile regression toward the dataset's Q-values.
        #     A normal (mean) regression would make V the value of an AVERAGE action
        #     in the data. We want the value of a GOOD action in the data. So we use
        #     an asymmetric loss: when Q is above V, the error costs `tau`; when Q is
        #     below V, it costs `1 - tau`. With tau=0.7, being too low hurts more
        #     than being too high, so V settles near the upper end of what the data
        #     achieves — an implicit maximum, taken only over actions that exist.
        with torch.no_grad():
            q = self._q_min(o, a, target=True)
        v = self.vnet(o)
        diff = q - v
        weight = torch.where(diff > 0, cfg.expectile, 1 - cfg.expectile)
        v_loss = (weight * diff.pow(2)).mean()
        self.v_optim.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optim.step()

        # (2) Q by plain TD — and look at the target: r + gamma * V(s'). There is no
        #     max, no actor, no action at s' at all. The dangerous operation has been
        #     deleted rather than defended against.
        with torch.no_grad():
            backup = r + cfg.gamma * (1 - d) * self.vnet(o2)
        q1 = self.q1(o, a)
        q_loss = F.mse_loss(q1, backup)
        if cfg.twin_critics:
            q_loss = q_loss + F.mse_loss(self.q2(o, a), backup)
        self.q_optim.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optim.step()

        # (3) Actor by advantage-weighted regression: behavior cloning, but each
        #     dataset action is weighted by exp(beta * advantage). Actions that beat
        #     their state's typical outcome are copied hard; actions that did worse
        #     are nearly ignored. It is BC with a filter, and because it only ever
        #     copies actions that are IN the data, it cannot output a fantasy.
        with torch.no_grad():
            adv = self._q_min(o, a, target=True) - self.vnet(o)
            w = torch.clamp((cfg.awr_beta * adv).exp(), max=cfg.awr_clip)
        pi_loss = -(w * self.actor.log_prob(o, a)).mean()
        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_optim.step()

        soft_update(self.q1, self.q1_targ, cfg.tau)
        if cfg.twin_critics:
            soft_update(self.q2, self.q2_targ, cfg.tau)
        return {"v_loss": v_loss.item(), "q_loss": q_loss.item(), "pi_loss": pi_loss.item(),
                "q_data": q1.mean().item(), "v_mean": v.mean().item(), "adv_mean": adv.mean().item()}


# ==========================================================================
# 5. Evaluation — the ONLY time Phase 7 is allowed to touch the environment
# ==========================================================================
@torch.no_grad()
def evaluate(actor, ds, episodes=10, seed=0, env=None):
    """Run the learned policy in the real environment and report the mean return.

    This is a MEASUREMENT, not training. No gradient, no data collection, nothing
    learned from it. If you were to feed anything seen here back into the agent you
    would no longer be doing offline RL — you would be doing online RL with extra
    steps, and every number in this phase would be a lie.

    Note `deterministic=True`: at evaluation the policy takes its best guess rather
    than sampling. Exploration noise has no job here; there is nothing left to explore.
    """
    close = env is None
    env = env or gym.make(ENV_ID)
    total = 0.0
    for i in range(episodes):
        o, _ = env.reset(seed=seed + 10_000 + i)
        done = False
        while not done:
            on = (o - ds.obs_mean[0]) / ds.obs_std[0]  # the same normalization as training
            a, _ = actor(torch.as_tensor(on, dtype=torch.float32).unsqueeze(0), deterministic=True)
            o, r, te, tr, _ = env.step(a.squeeze(0).numpy())
            total += r
            done = te or tr
    if close:
        env.close()
    return total / episodes


def train(cfg, progress=False, ds=None):
    """Train one offline agent. Returns (history, agent, dataset)."""
    set_seed(cfg.seed)  # BEFORE the networks are built (Phase 3 lost a day to this)
    rng = np.random.default_rng(cfg.seed)
    ds = ds or Dataset(cfg.level)
    agent = OfflineAgent(cfg, ds.obs_dim, ds.act_dim)
    env = gym.make(ENV_ID)

    hist = {"steps": [], "eval_return": [], "score": [], "q_data": [], "q_pi": [],
            "q_steps": [], "q_data_all": [], "q_pi_all": []}
    for t in range(1, cfg.grad_steps + 1):
        info = agent.update(ds.sample(cfg.batch_size, rng))
        if t % 100 == 0 and "q_data" in info:
            # Q-values are logged 50x more often than returns because they move 50x
            # faster — an evaluation costs 10 episodes of physics, but reading a
            # number off the last batch costs nothing.
            hist["q_steps"].append(t)
            hist["q_data_all"].append(info["q_data"])
            hist["q_pi_all"].append(info.get("q_pi", float("nan")))
        if t % cfg.eval_every == 0 or t == cfg.grad_steps:
            ret = evaluate(agent.actor, ds, cfg.eval_episodes, cfg.seed, env)
            hist["steps"].append(t)
            hist["eval_return"].append(ret)
            hist["score"].append(normalized_score(ret))
            hist["q_data"].append(info.get("q_data", float("nan")))
            hist["q_pi"].append(info.get("q_pi", float("nan")))
            if progress:
                print(f"  [{cfg.algo:8s} {cfg.level:6s} seed={cfg.seed}] step {t:6d}  "
                      f"return {ret:8.1f}  score {normalized_score(ret):6.1f}  "
                      f"Q(data) {info.get('q_data', float('nan')):12.1f}", flush=True)
    env.close()
    return hist, agent, ds
