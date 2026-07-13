"""The reusable policy-gradient core for Phase 4.

Phase 3 built its projects around a value function: a table of numbers, one per
action, and a rule that says "take the biggest". Phase 4 throws that away and
parameterizes the policy directly. Almost everything downstream — REINFORCE, the
value baseline, A2C, PPO, TRPO — is the *same four objects* with one of them
swapped:

    policy  -- what maps a state to a distribution over actions   (ActorCritic)
    rollout -- how experience is gathered                          (collect_episodes / rollout)
    weight  -- what each log-prob gets multiplied by               (return, advantage, GAE)
    update  -- how the gradient becomes a parameter change         (SGD step, clipped step, CG step)

Project 19 uses the plainest possible weight (the full Monte-Carlo return) and
the plainest update. Project 20 swaps the weight for an advantage. Project 21
swaps the rollout for parallel environments. Project 22 swaps the update for a
clipped one. Project 25 swaps it for a natural-gradient step. The scaffolding in
this file never changes, which is the point: the algorithms differ by less than
their names suggest.

Everything is CPU-only on purpose. These networks are two 64-unit hidden layers;
a GPU's kernel-launch overhead would cost more than the arithmetic it saves, and
the box's card is too old for the installed torch anyway. `torch.cuda` is never
touched.
"""

import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import AutoresetMode
from torch.distributions import Categorical, Normal

DEVICE = torch.device("cpu")


def set_seed(seed):
    """Seed every RNG that can move a result.

    Call this BEFORE constructing a network. Torch seeds its default generator
    from OS entropy at import, so a net built before `manual_seed` has unseeded
    weights and the whole run is irreproducible — a trap that cost this repo real
    time in Phase 3, because the symptom (a learning curve that moves between
    identical invocations) looks like environment noise rather than a bug.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Orthogonal weight init with a tuned gain — detail #2 of the 37.

    An orthogonal matrix preserves the norm of whatever it multiplies, so the
    forward signal neither explodes nor collapses as it crosses layers. The gain
    differs per layer role and that is not decoration: `std=0.01` on the policy
    head makes the initial logits nearly identical, so the starting policy is
    almost uniform (maximum entropy, no accidental early commitment), while
    `std=1.0` on the value head keeps early value estimates near zero.
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# --------------------------------------------------------------------------
# Environments
# --------------------------------------------------------------------------

def make_env(env_id, seed, idx, gamma=0.99, norm_obs=False, norm_reward=False,
             clip_action=False, max_steps=None):
    """Build one environment thunk for a vector env.

    The wrapper stack here is the *whole* of what people mean by "preprocessing"
    in continuous-control RL, and each layer is one of the 37 details:

      RecordEpisodeStatistics -- bookkeeping; records the true episode return
                                 BEFORE any reward scaling touches it, which is
                                 the only reason the numbers we report mean
                                 anything.
      ClipAction              -- a Gaussian policy can emit 7.3 for an actuator
                                 that saturates at 1.0; clip before the env sees it.
      NormalizeObservation    -- running mean/std over every observation seen.
      NormalizeReward         -- divides rewards by the running std of the
                                 discounted return. NOT mean-centred: subtracting
                                 a constant from every reward would change the
                                 optimal policy in an episodic task (it puts a
                                 price on staying alive).
    """
    def thunk():
        env = gym.make(env_id) if max_steps is None else gym.make(env_id, max_episode_steps=max_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # ClipAction is meaningless (and asserts) on a discrete action space — there
        # is no range to clip to. Guard here rather than at every call site, so a
        # caller can pass clip_action=True for any env and get the right thing.
        if clip_action and isinstance(env.action_space, gym.spaces.Box):
            env = gym.wrappers.ClipAction(env)
        if norm_obs:
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(
                env, lambda o: np.clip(o, -10, 10), env.observation_space)
        if norm_reward:
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda r: float(np.clip(r, -10, 10)))
        env.action_space.seed(seed + idx)
        env.reset(seed=seed + idx)
        return env
    return thunk


def make_vec_env(env_id, n_envs, seed, **kw):
    """N copies of an environment, stepped in lockstep.

    `AutoresetMode.SAME_STEP` is not a stylistic choice. Gymnasium 1.x defaults
    to NEXT_STEP, where a terminated env burns the *following* step on a reset:
    the action is ignored, the reward is 0, and that non-transition lands in the
    rollout buffer as if it were real. Every algorithm here would train on it.
    SAME_STEP resets immediately and hands back the true final observation in
    `info["final_obs"]` — which is exactly what a correct truncation bootstrap
    needs (see `rollout`).
    """
    return gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, i, **kw) for i in range(n_envs)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


def space_dims(env):
    """(obs_dim, act_dim, is_continuous) for a vector or single env."""
    obs_space = getattr(env, "single_observation_space", env.observation_space)
    act_space = getattr(env, "single_action_space", env.action_space)
    obs_dim = int(np.prod(obs_space.shape))
    if isinstance(act_space, gym.spaces.Discrete):
        return obs_dim, int(act_space.n), False
    return obs_dim, int(np.prod(act_space.shape)), True


# --------------------------------------------------------------------------
# The policy (and, from project 20 on, its critic)
# --------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """One network class for every algorithm in the phase.

    Discrete actions get a Categorical over logits; continuous actions get a
    Gaussian with a *state-independent* log-std held as a bare parameter. That
    second choice is worth a sentence: the spread of the action distribution is
    a property of how sure the agent is overall, not of the state it happens to
    be in, and letting it depend on the state is a well-known way to get a policy
    that learns to be maximally random in the states it has not figured out yet.
    Every reference implementation (CleanRL, SB3, spinning-up) does it this way.

    `critic=False` gives project 19's actor-only REINFORCE. Everything after
    project 20 turns the critic on. The two trunks are separate rather than
    shared — a shared trunk saves compute but couples the policy loss and the
    value loss through the same weights, and the value loss (whose scale is the
    return, potentially in the hundreds) then bullies the policy gradient.
    Separate trunks are what the PPO paper's continuous-control results use, and
    they remove an entire class of "why won't it learn" from the picture.
    """

    def __init__(self, obs_dim, act_dim, continuous=False, hidden=64,
                 critic=True, ortho=True, policy_gain=0.01):
        super().__init__()
        self.continuous = continuous
        self.has_critic = critic

        def lin(i, o, std):
            layer = nn.Linear(i, o)
            return layer_init(layer, std) if ortho else layer

        self.actor = nn.Sequential(
            lin(obs_dim, hidden, np.sqrt(2)), nn.Tanh(),
            lin(hidden, hidden, np.sqrt(2)), nn.Tanh(),
            lin(hidden, act_dim, policy_gain),
        )
        if critic:
            self.critic = nn.Sequential(
                lin(obs_dim, hidden, np.sqrt(2)), nn.Tanh(),
                lin(hidden, hidden, np.sqrt(2)), nn.Tanh(),
                lin(hidden, 1, 1.0),
            )
        if continuous:
            # log-std, not std: the parameter is unconstrained, and exp() keeps
            # the std positive without a clamp that would kill its gradient.
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs):
        out = self.actor(obs)
        if self.continuous:
            return Normal(out, self.log_std.exp().expand_as(out))
        return Categorical(logits=out)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, action=None, deterministic=False):
        """Sample (or score) an action. Returns (action, logprob, entropy, value).

        Summing over the action dimension for the continuous case is the
        independence assumption made explicit: a diagonal Gaussian treats each
        actuator as its own decision, so the joint log-prob is the sum of the
        per-actuator log-probs and the joint entropy is the sum of the entropies.
        """
        d = self.dist(obs)
        if action is None:
            action = d.mean if (deterministic and self.continuous) else \
                     (d.probs.argmax(-1) if (deterministic and not self.continuous) else d.sample())
        logp = d.log_prob(action)
        ent = d.entropy()
        if self.continuous:
            logp, ent = logp.sum(-1), ent.sum(-1)
        v = self.value(obs) if self.has_critic else None
        return action, logp, ent, v


# --------------------------------------------------------------------------
# Weights: what multiplies each log-prob
# --------------------------------------------------------------------------

def discounted_returns(rewards, gamma):
    """Reward-to-go: G_t = r_t + γ·G_{t+1}, computed backwards in one pass.

    "Reward-to-go" rather than the whole episode's return, because an action at
    step t cannot possibly have caused a reward at step t-1. Weighting it by
    those rewards anyway adds a term whose expectation is zero — pure variance,
    no signal. This is the cheapest variance reduction in RL and it is free.
    """
    out = np.empty(len(rewards), dtype=np.float64)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def compute_gae(rewards, values, dones, next_values, gamma, lam):
    """Generalized Advantage Estimation, vectorized over parallel envs.

    Shapes: rewards/values/dones (T, N); next_values (T, N) holds V(s_{t+1}) for
    every step, already zeroed where the episode genuinely ended.

    The recursion is the whole idea in two lines:
        δ_t = r_t + γ·V(s_{t+1}) - V(s_t)          the one-step TD residual
        A_t = δ_t + γλ·A_{t+1}                     an exponentially-weighted sum of them

    λ=0 collapses to the one-step advantage (low variance, biased by whatever the
    critic gets wrong); λ=1 collapses to the Monte-Carlo advantage (unbiased,
    high variance). λ≈0.95 is the empirical sweet spot everyone converged on.
    """
    T = len(rewards)
    adv = np.zeros_like(rewards)
    last = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] - values[t]
        # γλ·A_{t+1} is cut at an episode boundary: an advantage cannot flow
        # backwards across a reset into a different life.
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


# --------------------------------------------------------------------------
# Rollouts
# --------------------------------------------------------------------------

def collect_episode(env, agent, gamma, max_steps=10_000):
    """One full episode from a single (non-vector) env — REINFORCE's unit of work.

    REINFORCE cannot update until an episode ends, because its weight is the
    Monte-Carlo return, and the return does not exist until the future has
    happened. That constraint is why projects 21+ move to fixed-length rollouts
    with a bootstrapped value: they can update mid-episode.

    The reset is seeded from numpy's (seeded) generator rather than left to
    default. A bare `env.reset()` lets Gymnasium draw the starting state from OS
    entropy, so two runs of the same script with the same `set_seed` disagree —
    which is invisible in a learning curve and fatal in a variance measurement,
    where the whole result is a number computed from which episodes happened to
    be drawn. Seeding here makes every episode in this phase reproducible.
    """
    obs, _ = env.reset(seed=int(np.random.randint(0, 2**31 - 1)))
    logps, rewards, entropies, observations = [], [], [], []
    for _ in range(max_steps):
        t_obs = torch.as_tensor(obs, dtype=torch.float32)
        action, logp, ent, _ = agent.act(t_obs.unsqueeze(0))
        a = action.squeeze(0)
        obs2, r, term, trunc, _ = env.step(a.numpy() if agent.continuous else int(a))
        observations.append(obs)
        logps.append(logp.squeeze(0))
        entropies.append(ent.squeeze(0))
        rewards.append(float(r))
        obs = obs2
        if term or trunc:
            break
    return {
        "logps": torch.stack(logps),
        "entropies": torch.stack(entropies),
        "rewards": np.asarray(rewards, dtype=np.float64),
        "obs": torch.as_tensor(np.asarray(observations), dtype=torch.float32),
        "returns": discounted_returns(rewards, gamma),
        "ep_return": float(np.sum(rewards)),
        "length": len(rewards),
    }


@dataclass
class Rollout:
    obs: torch.Tensor          # (T*N, obs_dim)
    actions: torch.Tensor      # (T*N, ...)
    logps: torch.Tensor        # (T*N,)
    advantages: torch.Tensor   # (T*N,)
    returns: torch.Tensor      # (T*N,)
    values: torch.Tensor       # (T*N,)
    ep_returns: list           # true episode returns finished during this rollout


def rollout(envs, agent, obs, n_steps, gamma, lam, use_gae=True):
    """Collect `n_steps` from each of N parallel envs and turn it into advantages.

    Returns (Rollout, next_obs) so the caller can keep stepping where it left off:
    the environments are never reset between rollouts, so a single episode may
    span several updates. That is what makes fixed-length rollouts possible at all.

    The truncation bootstrap (detail #5 of the 37, and the one most often skipped)
    lives here. Gymnasium reports two kinds of episode end:

      terminated -- the pole fell, the lander crashed. The future really is worth
                    zero, so V(s_{t+1}) = 0 is correct.
      truncated  -- a step limit fired mid-episode. The pole is still upright! The
                    future is worth plenty, but the env hands you a reset. If you
                    treat this as terminal you teach the agent that balancing for
                    500 steps is followed by the end of the world.

    So the value target bootstraps off `info["final_obs"]` — the real observation
    the agent would have seen — while `done` (used to cut the GAE recursion) is
    set only by genuine termination.
    """
    n_envs = envs.num_envs
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, nextval_buf = [], [], [], [], [], [], []
    ep_returns = []

    for _ in range(n_steps):
        t_obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action, logp, _, value = agent.act(t_obs)
        a_np = action.numpy()
        obs2, reward, term, trunc, info = envs.step(a_np)

        # V(s_{t+1}) for the bootstrap: zero on true termination, and the value of
        # the *pre-reset* observation on truncation.
        with torch.no_grad():
            next_val = agent.value(torch.as_tensor(obs2, dtype=torch.float32)).numpy()
        real_done = term | trunc
        if real_done.any():
            next_val = next_val.copy()
            next_val[term] = 0.0                      # terminated: the future is worth nothing
            if trunc.any() and "final_obs" in info:
                idx = np.nonzero(trunc)[0]
                finals = np.stack([np.asarray(info["final_obs"][i], dtype=np.float32) for i in idx])
                with torch.no_grad():
                    next_val[idx] = agent.value(torch.as_tensor(finals)).numpy()
            for i in np.nonzero(real_done)[0]:
                fi = info.get("final_info")
                if fi is not None and "episode" in fi:
                    mask = fi["episode"].get("_r", None)
                    r_arr = fi["episode"]["r"]
                    if mask is None or mask[i]:
                        ep_returns.append(float(np.asarray(r_arr).reshape(-1)[i]))

        obs_buf.append(obs)
        act_buf.append(a_np)
        logp_buf.append(logp.numpy())
        val_buf.append(value.numpy())
        nextval_buf.append(next_val)
        rew_buf.append(np.asarray(reward, dtype=np.float64))
        # The two kinds of ending are used for two DIFFERENT things, and conflating
        # them is a bug that does not crash:
        #   * the value BOOTSTRAP (in next_val, above) must distinguish them — a
        #     terminated future is worth 0, a truncated one is worth V(final_obs);
        #   * the GAE RECURSION below must NOT — it has to be cut at either kind of
        #     ending, because A_{t+1} refers to the next step of the SAME episode,
        #     and after a reset there is no such step.
        # Recording only `term` here leaks advantage backwards across a truncation
        # into a different episode. On LunarLander (which truncates at 1000 steps,
        # exactly the fate of an agent that learns to hover) that leak was enough to
        # stop A2C learning at all.
        done_buf.append((term | trunc).astype(np.float64))
        obs = obs2

    rewards = np.asarray(rew_buf)
    values = np.asarray(val_buf, dtype=np.float64)
    dones = np.asarray(done_buf)
    next_values = np.asarray(nextval_buf, dtype=np.float64)

    if use_gae:
        adv, ret = compute_gae(rewards, values, dones, next_values, gamma, lam)
    else:
        # λ=1 with no value baseline in the target: the plain n-step return.
        adv, ret = compute_gae(rewards, values, dones, next_values, gamma, 1.0)

    flat = lambda x, dt=torch.float32: torch.as_tensor(
        np.asarray(x).reshape((n_steps * n_envs,) + np.asarray(x).shape[2:]), dtype=dt)

    actions = np.asarray(act_buf)
    return Rollout(
        obs=flat(obs_buf),
        actions=flat(actions, torch.float32 if agent.continuous else torch.long),
        logps=flat(logp_buf),
        advantages=flat(adv),
        returns=flat(ret),
        values=flat(values),
        ep_returns=ep_returns,
    ), obs


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def evaluate(agent, env_id, n_episodes=10, seed=1234, deterministic=False, **kw):
    """Mean episode return over fresh episodes, on an env the agent never trained on.

    Reported on the *raw* reward scale: the eval env deliberately skips the
    reward-normalization wrapper, because a return measured in normalized units
    is not comparable to anything, including the published number you want to
    check yourself against.
    """
    env = make_env(env_id, seed, 0, norm_obs=kw.get("norm_obs", False),
                   clip_action=kw.get("clip_action", False))()
    # If observation normalization is on, the eval env must use the *training*
    # statistics rather than re-estimating its own from ten episodes.
    if kw.get("obs_rms") is not None:
        target = env
        while not isinstance(target, gym.wrappers.NormalizeObservation):
            target = target.env
        target.obs_rms = kw["obs_rms"]
        target.update_running_mean = False

    totals = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action, _, _, _ = agent.act(t_obs, deterministic=deterministic)
            a = action.squeeze(0)
            obs, r, term, trunc, _ = env.step(a.numpy() if agent.continuous else int(a))
            total += float(r)
            done = term or trunc
        totals.append(total)
    env.close()
    return float(np.mean(totals)), float(np.std(totals))
