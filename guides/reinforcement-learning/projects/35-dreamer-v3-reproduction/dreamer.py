"""Project 35 — DreamerV3, in miniature, on a custom environment.

Dreamer's premise, in one sentence: *the agent should learn to act inside its own head.*

    real env  ->  a few thousand frames  ->  world model
    world model  ->  millions of imagined frames  ->  actor + critic

The actor in this file NEVER sees a real frame. Not once. Its every gradient comes from
rollouts imagined inside the learned latent space. The real environment is used for
exactly two things: collecting frames to train the world model, and telling us the score.

The world model is a Recurrent State-Space Model (RSSM), which splits the latent state
into two parts that do different jobs:

    h_t  (deterministic)  a GRU's hidden state — carries information forward reliably.
                          "The ball has been falling on the left for three frames."
    z_t  (stochastic)     a sample from a distribution — captures what could not have
                          been predicted. "The ball started in column 3" — nothing in the
                          past told you that, so it must be *sampled*, not computed.

Splitting them is the whole trick. A purely deterministic model cannot represent
uncertainty and will average over futures (predicting a blurred ball in two places at
once). A purely stochastic one forgets. Together: h remembers, z guesses.

The environment is CATCH — a paddle, a falling ball, 10x5 pixels — and the agent sees
only the raw pixels, never the ball's coordinates. It has to *discover* from images that
there is such a thing as a ball, that it falls, and that the paddle should be underneath
it when it lands.

Experiments:
  1. Does it learn Catch from pixels, training the actor entirely in imagination?
  2. What does the dream actually look like? (Decode an imagined rollout back to pixels
     and put it next to reality.)
  3. DreamerV3's headline claim is that it works everywhere with ONE hyperparameter
     setting. That generality is bought by a handful of stabilizing tricks. Turn each
     one off and see which are load-bearing and which are insurance.

  python3 dreamer.py     # ~8 min on 12 hyperthreads
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
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
torch.set_num_threads(1)

ROWS, COLS = 10, 5
OBS_DIM = ROWS * COLS
N_ACTIONS = 3          # left, stay, right


# ==========================================================================
# The custom environment: Catch
# ==========================================================================
class Catch:
    """A ball falls one row per step. Move the paddle under it. Pixels only.

    Reward is +1 for a catch and -1 for a miss, and it arrives ONLY on the last frame —
    every other step pays exactly zero. So the agent has to connect a reward at step 10
    back to the paddle nudges it made at steps 1-9. That is a credit-assignment problem,
    and it is the reason a world model helps: the model learns the ball's motion from
    pixels (a dense, easy signal available every frame), and the actor then plans
    against that model instead of groping for a signal that shows up once per episode.
    """

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.ball_col = int(self.rng.integers(COLS))
        self.ball_row = 0
        self.paddle = COLS // 2
        return self._obs()

    def _obs(self):
        g = np.zeros((ROWS, COLS), dtype=np.float32)
        g[self.ball_row, self.ball_col] = 1.0
        g[ROWS - 1, self.paddle] = 1.0
        return g.reshape(-1)

    def step(self, a):
        self.paddle = int(np.clip(self.paddle + (a - 1), 0, COLS - 1))
        self.ball_row += 1
        if self.ball_row == ROWS - 1:
            r = 1.0 if self.ball_col == self.paddle else -1.0
            return self._obs(), r, True
        return self._obs(), 0.0, False


# ==========================================================================
# Utility: the DreamerV3 tricks that are worth naming
# ==========================================================================
def symlog(x):
    """Squash big numbers, leave small ones alone. symlog(x) = sign(x) * log(1 + |x|).

    DreamerV3 predicts symlog(reward) instead of reward. Why: the same hyperparameters
    have to work on a game paying rewards of 0.01 and a game paying rewards of 10,000.
    A plain MSE loss on the raw reward would produce gradients 10^6 times bigger in the
    second game, and any learning rate that survived one would explode or stall on the
    other. symlog compresses the scale difference away, so ONE learning rate fits all —
    which is a large part of how "one hyperparameter setting for every task" is achieved.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))


@dataclass
class Cfg:
    seed: int = 0
    total_steps: int = 11_000
    updates_per_episode: int = 3
    batch_size: int = 32
    imag_horizon: int = 8
    n_imag_starts: int = 96    # roots subsampled from the batch; the main CPU cost
    deter: int = 96            # size of h (the GRU state)
    stoch_groups: int = 8      # z is 8 independent categorical variables...
    stoch_classes: int = 8     # ...each with 8 possible values
    hidden: int = 96
    lr: float = 6e-4
    actor_lr: float = 6e-4
    gamma: float = 0.99
    lam: float = 0.95
    # The entropy bonus, and it is not decoration. At 3e-3 (DreamerV3's own value, tuned
    # for far bigger networks and far longer runs) the actor here saturates to "always
    # stay" within a few hundred updates: once a softmax reaches probability 1.0 on one
    # action, the REINFORCE gradient through the other two is ~0 and nothing can revive
    # them. The policy stops sampling, so it stops learning, and the return sits at the
    # random-play score forever while the world-model losses keep dropping beautifully —
    # a failure that looks, on every chart except the return, like success.
    entropy: float = 3e-2
    free_bits: float = 1.0     # <-- ablated
    kl_balance: bool = True    # <-- ablated
    unimix: float = 0.01       # <-- ablated
    use_symlog: bool = True    # <-- ablated
    eval_every: int = 1_000

    @property
    def stoch(self):
        return self.stoch_groups * self.stoch_classes


# ==========================================================================
# The world model
# ==========================================================================
class RSSM(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        S, D, H = cfg.stoch, cfg.deter, cfg.hidden

        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU()
        )
        self.gru_input = nn.Sequential(nn.Linear(S + N_ACTIONS, H), nn.SiLU())
        self.gru = nn.GRUCell(H, D)

        # PRIOR: what z should be, guessed from h alone — i.e. from the past only.
        # This is the network that has to do the imagining, because during a dream there
        # is no observation to look at.
        self.prior_net = nn.Sequential(nn.Linear(D, H), nn.SiLU(), nn.Linear(H, S))
        # POSTERIOR: what z actually is, now that we have SEEN the frame.
        self.post_net = nn.Sequential(nn.Linear(D + H, H), nn.SiLU(), nn.Linear(H, S))

        self.decoder = nn.Sequential(
            nn.Linear(D + S, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, OBS_DIM)
        )
        self.reward_head = nn.Sequential(
            nn.Linear(D + S, H), nn.SiLU(), nn.Linear(H, 1)
        )
        self.cont_head = nn.Sequential(nn.Linear(D + S, H), nn.SiLU(), nn.Linear(H, 1))

    def _dist(self, logits):
        cfg = self.cfg
        logits = logits.view(*logits.shape[:-1], cfg.stoch_groups, cfg.stoch_classes)
        probs = torch.softmax(logits, -1)
        if cfg.unimix > 0:
            # Mix in a little uniform noise. A categorical that has driven some class to
            # probability exactly 0 can never recover it — the gradient there is 0 too.
            # This keeps every option alive at the cost of 1% of the probability mass.
            probs = (1 - cfg.unimix) * probs + cfg.unimix / cfg.stoch_classes
        return probs

    def _sample(self, probs, generator=None):
        """Draw a one-hot z, but let gradients pass through as if it were the probs.

        Sampling is not differentiable — you cannot take the derivative of "a die came up
        4". The straight-through estimator sidesteps this: the FORWARD pass uses the hard
        one-hot sample (so the model really does commit to one option), and the BACKWARD
        pass pretends it used the smooth probabilities. The gradient is biased, and works.
        """
        shape = probs.shape
        flat = probs.reshape(-1, shape[-1])
        idx = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
        onehot = F.one_hot(idx, shape[-1]).float().view(shape)
        return (onehot - probs).detach() + probs

    def observe(self, obs_seq, act_prev_seq, generator=None):
        """Run the model along a REAL sequence, using the frames. Trains everything.

        At each step we compute both distributions over z:
          * the posterior, which is allowed to look at the frame, and
          * the prior, which is not.
        The KL between them is the loss that teaches the prior to predict the future —
        because during imagination, the prior is all the model will have.

        `act_prev_seq[:, t]` is the action taken *before* frame t, i.e. the one that
        CAUSED it. Feeding the action taken *at* frame t instead is an easy and vicious
        bug: the model trains on a timeline shifted one step from the one it runs on at
        inference, so the dynamics never learns what an action actually does, while every
        loss keeps decreasing and nothing ever crashes.
        """
        cfg = self.cfg
        B, L, _ = obs_seq.shape
        h = torch.zeros(B, cfg.deter)
        z = torch.zeros(B, cfg.stoch)

        posts, priors, hs, zs = [], [], [], []
        embed = self.encoder(obs_seq)
        for t in range(L):
            h = self.gru(self.gru_input(torch.cat([z, act_prev_seq[:, t]], -1)), h)
            prior_probs = self._dist(self.prior_net(h))
            post_probs = self._dist(self.post_net(torch.cat([h, embed[:, t]], -1)))
            z = self._sample(post_probs, generator).view(B, cfg.stoch)
            posts.append(post_probs)
            priors.append(prior_probs)
            hs.append(h)
            zs.append(z)
        return (
            torch.stack(hs, 1), torch.stack(zs, 1),
            torch.stack(posts, 1), torch.stack(priors, 1),
        )

    def imagine_step(self, h, z, a_onehot, generator=None):
        """One step of DREAMING: no frame, no encoder, no posterior. Prior only."""
        cfg = self.cfg
        h = self.gru(self.gru_input(torch.cat([z, a_onehot], -1)), h)
        probs = self._dist(self.prior_net(h))
        z = self._sample(probs, generator).view(-1, cfg.stoch)
        return h, z

    def feat(self, h, z):
        return torch.cat([h, z], -1)


class ActorCritic(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        F_ = cfg.deter + cfg.stoch
        self.actor = nn.Sequential(
            nn.Linear(F_, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, N_ACTIONS),
        )
        self.critic = nn.Sequential(
            nn.Linear(F_, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, cfg.hidden), nn.SiLU(),
            nn.Linear(cfg.hidden, 1),
        )


def kl_divergence(p_probs, q_probs):
    """KL(p || q) for a batch of independent categoricals, summed over the groups."""
    p = p_probs.clamp(1e-8, 1.0)
    q = q_probs.clamp(1e-8, 1.0)
    return (p * (p.log() - q.log())).sum(-1).sum(-1)


# ==========================================================================
# World-model loss
# ==========================================================================
def world_model_loss(wm, obs, act_prev, rew_prev, cont, cfg, generator):
    """Fit the model to a real sequence.

    The reward head predicts `rew_prev[t]` — the reward received on ARRIVING at frame t,
    i.e. the payout for the action that got us here. It does not predict the reward for
    the action about to be taken.

    That distinction decides whether the whole project works. Catch pays +1 or -1 based
    on where the paddle is *after* its final move, so the reward is a function of the
    action. If the reward head were asked to predict it from the state alone, it could
    only ever learn the average payout of that state — the imagined reward would be the
    same no matter what the actor did, the actor's gradient would be exactly zero, and
    the agent would never learn to move the paddle. Reward-on-arrival keeps the action
    inside the prediction, because the action is what produced the state being scored.
    """
    h, z, post, prior = wm.observe(obs, act_prev, generator)
    feat = wm.feat(h, z)

    recon = wm.decoder(feat)
    recon_loss = F.mse_loss(recon, obs, reduction="none").sum(-1).mean()

    r_pred = wm.reward_head(feat).squeeze(-1)
    r_target = symlog(rew_prev) if cfg.use_symlog else rew_prev
    reward_loss = F.mse_loss(r_pred, r_target)

    c_pred = wm.cont_head(feat).squeeze(-1)
    cont_loss = F.binary_cross_entropy_with_logits(c_pred, cont)

    if cfg.kl_balance:
        # KL BALANCING. The KL term has two jobs that pull in opposite directions:
        # drag the prior toward the posterior (teach the model to predict), and drag the
        # posterior toward the prior (stop the encoder from smuggling in unpredictable
        # detail). Done with one symmetric term, the second job wins and the posterior
        # collapses onto the lazy prior, destroying the representation. So the two
        # directions get separate weights: mostly train the PRIOR (0.5), only gently
        # regularize the POSTERIOR (0.1). `detach` is what splits them.
        dyn = kl_divergence(post.detach(), prior)      # move the prior
        rep = kl_divergence(post, prior.detach())      # move the posterior
        # FREE BITS: ignore the KL entirely while it is below `free_bits` nats. Without
        # this the model keeps grinding an already-tiny KL toward zero, which it can only
        # achieve by making the latent carry less information — posterior collapse, the
        # classic failure of every latent-variable model.
        dyn = torch.clamp(dyn, min=cfg.free_bits).mean()
        rep = torch.clamp(rep, min=cfg.free_bits).mean()
        kl = 0.5 * dyn + 0.1 * rep
    else:
        kl = kl_divergence(post, prior)
        kl = torch.clamp(kl, min=cfg.free_bits).mean() if cfg.free_bits > 0 else kl.mean()

    loss = recon_loss + reward_loss + cont_loss + kl
    return loss, h.detach(), z.detach(), {
        "recon": recon_loss.item(), "kl": kl.detach().item(),
        "reward": reward_loss.item(),
    }


# ==========================================================================
# Actor-critic, trained ONLY on imagined rollouts
# ==========================================================================
def imagine_and_train(wm, ac, h0, z0, optim_ac, cfg, generator, rng):
    """Dream `imag_horizon` steps from real states in the batch, then learn from the dream.

    Each real state in the batch becomes the root of its own dream, and the actor's ENTIRE
    training signal comes from these dreams — it never sees a real frame.

    `n_imag_starts` subsamples the roots. Every state in the batch could be a root (that is
    what the paper does), but on a CPU the imagination rollout is the single most expensive
    thing in the file, and dreaming from 96 roots instead of 288 costs almost nothing in
    learning while cutting the update time by two thirds.
    """
    h = h0.reshape(-1, h0.shape[-1])
    z = z0.reshape(-1, z0.shape[-1])
    n = h.shape[0]
    if n > cfg.n_imag_starts:
        sel = torch.as_tensor(rng.choice(n, size=cfg.n_imag_starts, replace=False))
        h, z = h[sel], z[sel]

    feats, logps, entropies = [], [], []
    for _ in range(cfg.imag_horizon):
        feat = wm.feat(h, z)
        logits = ac.actor(feat)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        logps.append(dist.log_prob(a))
        entropies.append(dist.entropy())
        feats.append(feat)
        # The world model is FROZEN here. We train the actor against the model, never the
        # model against the actor — otherwise the actor could "improve" by corrupting the
        # model into predicting reward everywhere, and would.
        with torch.no_grad():
            h, z = wm.imagine_step(h, z, F.one_hot(a, N_ACTIONS).float(), generator)
    feats.append(wm.feat(h, z))

    feat_stack = torch.stack(feats)                       # (H+1, N, F)
    with torch.no_grad():
        # r[k] is the reward for ARRIVING at feats[k] — i.e. the payout of action a_{k-1}.
        # So the reward earned by the action taken at step k is r[k+1], and r[0] is the
        # payout of whatever happened before the dream began, which is not ours to claim.
        r = wm.reward_head(feat_stack).squeeze(-1)
        if cfg.use_symlog:
            r = symexp(r)
        c = torch.sigmoid(wm.cont_head(feat_stack).squeeze(-1))
    values = ac.critic(feat_stack).squeeze(-1)

    # lambda-returns through the dream: a geometric blend of "trust the imagined rewards"
    # and "trust the critic", the same bias/variance dial as GAE in Phase 4.
    with torch.no_grad():
        ret = values[-1]
        returns = []
        for t in reversed(range(cfg.imag_horizon)):
            ret = r[t + 1] + cfg.gamma * c[t + 1] * (
                (1 - cfg.lam) * values[t + 1] + cfg.lam * ret
            )
            returns.append(ret)
        returns.reverse()
        returns = torch.stack(returns)                    # (H, N)

    adv = returns - values[:-1].detach()
    # Percentile return normalization: scale the advantage by the spread of returns
    # rather than by a tuned constant. This is the other half of "one hyperparameter
    # setting works everywhere" — a task paying +1 and a task paying +1000 produce
    # advantages of the same size once divided by their own scale.
    scale = torch.quantile(returns, 0.95) - torch.quantile(returns, 0.05)
    adv = adv / torch.clamp(scale, min=1.0)

    logp = torch.stack(logps)
    ent = torch.stack(entropies)
    # REINFORCE with the critic as a baseline (Phase 4, project 20) — but the "episodes"
    # it learns from are dreams.
    actor_loss = -(logp * adv.detach()).mean() - cfg.entropy * ent.mean()
    critic_loss = F.mse_loss(values[:-1], returns)

    loss = actor_loss + critic_loss
    optim_ac.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(ac.parameters(), 100.0)
    optim_ac.step()
    return actor_loss.item(), critic_loss.item()


# ==========================================================================
# Training loop
# ==========================================================================
class EpisodeBuffer:
    def __init__(self, capacity=4000):
        self.eps = []
        self.capacity = capacity

    def add(self, ep):
        self.eps.append(ep)
        if len(self.eps) > self.capacity:
            self.eps.pop(0)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, len(self.eps), size=batch_size)
        return (
            torch.as_tensor(np.stack([self.eps[i]["obs"] for i in idx])),
            torch.as_tensor(np.stack([self.eps[i]["act_prev"] for i in idx])),
            torch.as_tensor(np.stack([self.eps[i]["rew_prev"] for i in idx])),
            torch.as_tensor(np.stack([self.eps[i]["cont"] for i in idx])),
        )


def collect_episode(env, wm, ac, cfg, rng, generator, greedy=False):
    """Play one real episode, carrying the RSSM state along as we go.

    The agent is doing inference, not planning: it runs the posterior forward on each real
    frame and asks the actor what to do with the resulting latent. No search, no rollouts.
    All the imagining happened at training time — which is why Dreamer acts fast even
    though it is a model-based method, unlike the MPC agents of projects 32 and 33 that
    must re-plan from scratch at every single step.

    The episode is stored on the model's timeline, not the environment's:

        obs[t]        the frame at t, INCLUDING the final one after the last action
        act_prev[t]   the action that caused frame t (zeros at t=0)
        rew_prev[t]   the reward collected on arriving at frame t (0 at t=0)
        cont[t]       0 if frame t is terminal, else 1

    Storing the final frame is not bookkeeping pedantry: in Catch the ONLY nonzero reward
    in the whole episode is the one paid on arriving at that last frame. Drop it and the
    reward head is trained exclusively on zeros.
    """
    obs = env.reset()
    h = torch.zeros(1, cfg.deter)
    z = torch.zeros(1, cfg.stoch)
    a_prev = torch.zeros(1, N_ACTIONS)

    O, A_prev, R_prev, C = [obs], [np.zeros(N_ACTIONS, dtype=np.float32)], [np.float32(0.0)], []
    done = False
    total = 0.0
    while not done:
        with torch.no_grad():
            h = wm.gru(wm.gru_input(torch.cat([z, a_prev], -1)), h)
            embed = wm.encoder(torch.as_tensor(obs).unsqueeze(0))
            probs = wm._dist(wm.post_net(torch.cat([h, embed], -1)))
            z = wm._sample(probs, generator).view(1, cfg.stoch)
            logits = ac.actor(wm.feat(h, z))
            if greedy:
                a = int(logits.argmax(-1).item())
            else:
                a = int(torch.distributions.Categorical(logits=logits).sample().item())

        a1h = np.eye(N_ACTIONS, dtype=np.float32)[a]
        a_prev = torch.as_tensor(a1h).unsqueeze(0)
        obs, r, done = env.step(a)
        total += r

        C.append(np.float32(1.0))       # the frame we just LEFT was not terminal
        O.append(obs)
        A_prev.append(a1h)
        R_prev.append(np.float32(r))
    C.append(np.float32(0.0))           # the frame we ended on is

    return {
        "obs": np.array(O, dtype=np.float32),
        "act_prev": np.array(A_prev, dtype=np.float32),
        "rew_prev": np.array(R_prev, dtype=np.float32),
        "cont": np.array(C, dtype=np.float32),
    }, total


def train(cfg: Cfg, progress=False, want_model=False):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed + 11)

    env = Catch(seed=cfg.seed)
    wm = RSSM(cfg)
    ac = ActorCritic(cfg)
    optim_wm = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    optim_ac = torch.optim.Adam(ac.parameters(), lr=cfg.actor_lr)
    buf = EpisodeBuffer()

    hist = {"steps": [], "return": [], "recon": [], "kl": []}
    steps = 0
    recent = []
    t0 = time.time()
    info = {}

    while steps < cfg.total_steps:
        ep, total = collect_episode(env, wm, ac, cfg, rng, gen)
        buf.add(ep)
        ep_len = len(ep["rew_prev"]) - 1
        steps += ep_len
        recent.append(total)

        if len(buf.eps) >= cfg.batch_size:
            for _ in range(cfg.updates_per_episode):
                obs, act_prev, rew_prev, cont = buf.sample(cfg.batch_size, rng)
                loss, h, z, info = world_model_loss(
                    wm, obs, act_prev, rew_prev, cont, cfg, gen
                )
                optim_wm.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
                optim_wm.step()
                imagine_and_train(wm, ac, h, z, optim_ac, cfg, gen, rng)

        if steps % cfg.eval_every < ep_len and len(recent) >= 20:
            avg = float(np.mean(recent[-50:]))
            hist["steps"].append(steps)
            hist["return"].append(avg)
            hist["recon"].append(info.get("recon", float("nan")))
            hist["kl"].append(info.get("kl", float("nan")))
            if progress:
                print(f"  [seed {cfg.seed}] steps {steps:6d}  return {avg:+.2f}  "
                      f"recon {info.get('recon', 0):.3f}  kl {info.get('kl', 0):.2f}  "
                      f"({time.time() - t0:.0f}s)", flush=True)

    hist["wall"] = time.time() - t0
    hist["seed"] = cfg.seed
    if want_model:
        hist["wm"] = wm.state_dict()
        hist["ac"] = ac.state_dict()
    return hist


# ==========================================================================
# Experiment 2: look inside the dream
# ==========================================================================
def visualize_dream(wm_state, ac_state, cfg, seed=0):
    """Show the model a few real frames, then make it hallucinate the rest of the episode.

    Top row: what really happened. Bottom row: what the model *imagined* would happen,
    given only the first 3 frames and the actions taken. After frame 3 the model receives
    NO pixels — every frame it draws is produced by rolling its own prior forward and
    decoding the result.
    """
    wm, ac = RSSM(cfg), ActorCritic(cfg)
    wm.load_state_dict(wm_state)
    ac.load_state_dict(ac_state)
    gen = torch.Generator().manual_seed(seed)
    env = Catch(seed=seed + 500)
    rng = np.random.default_rng(seed)

    ep, _ = collect_episode(env, wm, ac, cfg, rng, gen, greedy=True)
    obs = torch.as_tensor(ep["obs"]).unsqueeze(0)
    act_prev = torch.as_tensor(ep["act_prev"]).unsqueeze(0)
    T = obs.shape[1]
    context = 3

    with torch.no_grad():
        # Watch the first `context` frames for real...
        h, z, _, _ = wm.observe(obs[:, :context], act_prev[:, :context], gen)
        h, z = h[:, -1], z[:, -1]
        # ...then close its eyes. From here the model is fed only the actions; every
        # frame it produces is decoded from a latent it rolled forward itself.
        imagined = []
        for t in range(context, T):
            h, z = wm.imagine_step(h, z, act_prev[:, t], gen)
            # No sigmoid here: the decoder is trained with a plain MSE straight against the
            # 0/1 pixels, so its output IS the pixel value, not a logit. Squashing it
            # through a sigmoid would map 0 -> 0.5 and 1 -> 0.73 and turn a sharp, correct
            # dream into uniform grey mush.
            frame = wm.decoder(wm.feat(h, z)).view(ROWS, COLS).numpy()
            imagined.append(np.clip(frame, 0.0, 1.0))

    real = ep["obs"].reshape(T, ROWS, COLS)
    return real, np.array(imagined), context


# ==========================================================================
def job(args):
    cfg, want_model = args
    return train(cfg, progress=(cfg.seed == 0 and want_model), want_model=want_model)


ABLATIONS = [
    ("DreamerV3 (all tricks on)", {}),
    ("no free bits", dict(free_bits=0.0)),
    ("no KL balancing", dict(kl_balance=False)),
    ("no symlog rewards", dict(use_symlog=False)),
    ("no unimix", dict(unimix=0.0)),
]

SEEDS = [0, 1, 2]


def main():
    OUT.mkdir(exist_ok=True)
    import matplotlib.pyplot as plt

    base = Cfg()
    print("=== DreamerV3 on Catch: does it learn from pixels, in imagination? ===",
          flush=True)

    jobs = []
    for name, over in ABLATIONS:
        for s in SEEDS:
            cfg = replace(base, seed=s, **over)
            jobs.append((cfg, name == ABLATIONS[0][0] and s == 0))
    with ProcessPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(job, jobs))

    grouped = {}
    k = 0
    for name, _ in ABLATIONS:
        grouped[name] = res[k : k + len(SEEDS)]
        k += len(SEEDS)

    print(f"\n  {'variant':28s} {'final return':>13s}  (perfect = +1.0, random ~ -0.5)")
    for name, _ in ABLATIONS:
        r = np.array([h["return"] for h in grouped[name]])
        print(f"  {name:28s} {r[:, -1].mean():+13.2f}  "
              f"(seeds: {', '.join(f'{v:+.2f}' for v in r[:, -1])})")
    print(f"\n  wall per seed: {np.mean([h['wall'] for h in res]):.0f}s")

    # ---- learning curve of the full agent ----
    main_h = grouped[ABLATIONS[0][0]]
    steps = np.array(main_h[0]["steps"])
    r = np.array([h["return"] for h in main_h])
    fig, ax = ps.new_axes(7.4, 4.3)
    ax.plot(steps, r.mean(0), color=ps.SERIES[0], lw=2, label="DreamerV3 (actor trained only in imagination)")
    ax.fill_between(steps, r.min(0), r.max(0), color=ps.SERIES[0], alpha=0.15)
    ax.axhline(1.0, color=ps.INK_MUTED, ls="--", lw=1)
    ax.text(steps[-1], 1.0, " perfect", color=ps.INK_MUTED, fontsize=8, va="bottom", ha="right")
    ax.axhline(-0.5, color=ps.BASELINE, ls=":", lw=1)
    ax.text(steps[-1], -0.5, " random policy", color=ps.INK_MUTED, fontsize=8, va="bottom", ha="right")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "Learning Catch from raw pixels", "real environment steps",
              "episode return", OUT / "learning_curve.png")

    # ---- ablation bars ----
    fig, ax = ps.new_axes(7.6, 4.0)
    names = [n for n, _ in ABLATIONS]
    means = [np.array([h["return"] for h in grouped[n]])[:, -1].mean() for n in names]
    errs = [np.array([h["return"] for h in grouped[n]])[:, -1].std() for n in names]
    colors = [ps.SERIES[0]] + [ps.SERIES[2]] * (len(names) - 1)
    ax.barh([n.replace(" (all tricks on)", "") for n in names][::-1], means[::-1],
            xerr=errs[::-1], color=colors[::-1], height=0.6,
            error_kw=dict(ecolor=ps.INK_MUTED, lw=1))
    ax.grid(axis="y", visible=False)
    ps.finish(fig, ax, "Which of DreamerV3's stabilizers actually carry weight here?",
              "final episode return (higher is better)", "", OUT / "ablations.png")

    # ---- the dream ----
    print("\n=== decoding the dream ===", flush=True)
    full = next(h for h in res if "wm" in h)
    real, imag, context = visualize_dream(full["wm"], full["ac"], replace(base, seed=0))
    T = len(real)
    fig, axes = plt.subplots(2, T, figsize=(1.05 * T, 3.4), dpi=130)
    fig.patch.set_facecolor(ps.SURFACE)
    for t in range(T):
        axes[0, t].imshow(real[t], cmap="magma", vmin=0, vmax=1)
        axes[0, t].set_title(f"t={t}", fontsize=7, color=ps.INK_SECONDARY, pad=3)
        if t < context:
            axes[1, t].imshow(real[t], cmap="Greys", vmin=0, vmax=1, alpha=0.35)
            axes[1, t].set_title("given", fontsize=6, color=ps.INK_MUTED, pad=3)
        else:
            axes[1, t].imshow(imag[t - context], cmap="magma", vmin=0, vmax=1)
        for a in (axes[0, t], axes[1, t]):
            a.set_xticks([])
            a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color(ps.BASELINE)
    axes[0, 0].set_ylabel("REAL", fontsize=8, color=ps.INK)
    axes[1, 0].set_ylabel("DREAMED", fontsize=8, color=ps.INK)
    fig.suptitle("The model is shown 3 frames, then imagines the rest of the episode",
                 fontsize=10, color=ps.INK, y=0.98)
    fig.tight_layout()
    fig.savefig(OUT / "dream.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'dream.png'}")

    # How accurate is the dream, numerically?
    err = np.abs(imag - real[context:]).mean()
    print(f"  mean per-pixel error of the imagined frames: {err:.4f}")


if __name__ == "__main__":
    main()
