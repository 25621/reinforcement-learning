"""Learning inside a dream: a world model you can practise in.

The split of labour, spelled out
--------------------------------
This guide owns the *generative* half of a world model: given a screen and a
button, draw the next screen.  Projects 40 and 41 built exactly that.  What this
project adds is the reason anyone outside video generation cares — you can put
an agent inside it and let it practise.  The learning rule that turns practice
into a better [policy](/shared/glossary/#policy) is reinforcement learning's
subject, covered in [RL Phase 6](../../../reinforcement-learning/#phase-6-model-based-rl);
here it is deliberately the smallest thing that works, so the interesting
variable stays the world model.

Why the world model here is NOT a diffusion model
-------------------------------------------------
Project 41's diffusion world model needs ~30 network passes per frame.  Training
a policy takes hundreds of thousands of imagined frames.  30 x 300,000 passes is
not a ten-minute experiment; it is not a ten-hour one either.

[DreamerV3](/shared/glossary/#dreamerv3) hit the same wall and answered it the
same way: imagine in a small *latent* space with a single cheap step per frame,
and only decode to pixels when a human wants to look.  So the model here is one
forward pass per imagined frame through a network half project 41's width, and
it is deliberately deterministic.  Project 44 measures the resulting speed gap
on the same hardware.

The honest cost of that choice is the thing project 40 measured: a one-shot
regressor cannot represent "the coin could land anywhere", so it smears the coin
instead of picking a spot.  The policy still learns, because *movement* — the
part it actually controls — is deterministic.  Knowing which parts of your world
you are allowed to model badly is most of the craft here.

What the agent is optimising
----------------------------
Reward is 1 when the player steps onto the coin and 0 otherwise, so "return"
means "coins per unit of time".  The world model must therefore predict two
things: the next screen, and whether a coin was just collected.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "40-action-conditioned-video"))
import world_lib as W                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# the world model
# ---------------------------------------------------------------------------

class DreamWorld(nn.Module):
    """(frame, action) -> (next frame, "was a coin collected?").

    One forward pass, no denoising loop.  The action steers through FiLM, the
    same mechanism project 40 used, so the only real difference from project
    40's model is that this one commits to a single answer instead of sampling.
    """

    def __init__(self, base=32, cond=128):
        super().__init__()
        self.act_emb = nn.Embedding(W.N_ACT, cond)
        self.a_mlp = nn.Sequential(nn.Linear(cond, cond), nn.SiLU(),
                                   nn.Linear(cond, cond))
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        self.b1 = W.FiLMBlock(base, base, cond)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.mid = W.FiLMBlock(base * 2, base * 2, cond)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.b2 = W.FiLMBlock(base * 2, base, cond)
        self.out_n = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)
        self.rew = nn.Sequential(nn.Linear(base * 2, 128), nn.SiLU(),
                                 nn.Linear(128, 1))

    def forward(self, frame, action):
        c = self.a_mlp(self.act_emb(action))
        h = self.b1(self.stem(W.channelize(frame)), c)
        m = self.mid(self.down(h), c)
        u = self.b2(torch.cat([self.up(m), h], dim=1), c)
        nxt = self.out(F.silu(self.out_n(u)))[:, 0]
        rew_logit = self.rew(m.mean(dim=(2, 3)))[:, 0]
        return nxt, rew_logit


# ---------------------------------------------------------------------------
# the agent
# ---------------------------------------------------------------------------

class Policy(nn.Module):
    """State in, one of four buttons out, plus a guess at future reward.

    The "state" is four numbers — the player's (row, col) and the coin's —
    read out of the frame by `frame_to_coords`.  See that function for why the
    policy reads coordinates rather than pixels.  Critically, in the dream those
    numbers come from the *world model's* frame, so a drifted dream feeds the
    policy a wrong coin position, and that is where model exploitation shows up.

    The second head (the "critic") predicts how much reward is coming from this
    state.  It is not used to pick actions; it is subtracted from the observed
    return so the learning signal says "better or worse than expected" rather
    than "good or bad in absolute terms" — the cheapest way to stop
    [REINFORCE](/shared/glossary/#reinforce) from thrashing, and also what
    supplies the [bootstrap](/shared/glossary/#model-based-rl) that turns the
    sparse coin reward into a dense one.
    """

    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(4, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, W.N_ACT)
        self.v = nn.Linear(hidden, 1)

    def forward(self, frame):
        h = self.body(W.frame_to_coords(frame))
        return self.pi(h), self.v(h)[:, 0]

    def act(self, frame, generator=None):
        logits, value = self(frame)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), value


# ---------------------------------------------------------------------------
# imagination
# ---------------------------------------------------------------------------

def imagine(world, policy, start, horizon, gamma=0.95, ent_coef=0.03):
    """Roll the policy forward INSIDE the world model and return its loss.

    No real environment is touched.  The world model produces both the next
    screen and the reward, so the whole loop is a few tensor ops -- this is the
    "imagined [rollout](/shared/glossary/#rollout)" that makes model-based RL
    sample-efficient.

    Gradients flow through the policy only.  The world model is frozen scenery:
    letting the policy push gradients into it would let the agent *edit its own
    dream* until everything paid out.
    """
    f = start
    logps, ents, vals, rews = [], [], [], []
    for _ in range(horizon):
        a, logp, ent, v = policy.act(f)
        with torch.no_grad():
            f, r_logit = world(f, a)
            f = f.clamp(0.0, 1.0)
            r = torch.sigmoid(r_logit)
        logps.append(logp); ents.append(ent); vals.append(v); rews.append(r)
    logps = torch.stack(logps); ents = torch.stack(ents)
    vals = torch.stack(vals); rews = torch.stack(rews)

    # Bootstrap: the value of the state we STOPPED at stands in for all the
    # reward that would have come after the horizon.  This is what turns a
    # sparse reward (a coin is rarely inside a short window) into a dense
    # signal: even a rollout that never touches a coin gets a gradient, because
    # the critic says "you ended up closer to one than you started."  Without
    # it, a policy that pins itself against a wall sees all-zero returns and
    # never learns to move.  This is the standard actor-critic fix, and it is
    # why the Policy carries a value head at all.
    with torch.no_grad():
        _, boot = policy(f)
    rets, run = [], boot
    for t in reversed(range(horizon)):
        run = rews[t] + gamma * run
        rets.append(run)
    rets = torch.stack(rets[::-1])

    adv = (rets - vals).detach()
    adv = (adv - adv.mean()) / (adv.std() + 1e-6)
    pg = -(logps * adv).mean()
    vloss = F.mse_loss(vals, rets.detach())
    loss = pg + 0.5 * vloss - ent_coef * ents.mean()
    return loss, float(rews.sum(0).mean())


# ---------------------------------------------------------------------------
# the real thing, for collecting data and for scoring
# ---------------------------------------------------------------------------

class VecGame:
    """A batch of independent GridGames stepped together.

    Batching matters here for a boring but decisive reason: the policy is a
    neural network, and one forward pass on 64 screens costs barely more than
    one forward pass on 1.  Stepping 64 games in lockstep therefore makes the
    model-free baseline ~50x faster to train, which is what makes it affordable
    to give that baseline a fair, well-resourced run.
    """

    def __init__(self, n, seed=0):
        self.envs = [W.GridGame(seed=seed * 1000 + i) for i in range(n)]
        self.n = n

    def reset(self):
        for e in self.envs:
            e.reset()
        return self.frames()

    def frames(self):
        return torch.from_numpy(
            np.stack([W.render(*e.state()) for e in self.envs]))

    def step(self, actions):
        r = np.zeros(self.n, dtype=np.float32)
        for i, e in enumerate(self.envs):
            _, ri, _ = e.step(int(actions[i]))
            r[i] = ri
        return self.frames(), torch.from_numpy(r)


@torch.no_grad()
def evaluate(policy, n_env=64, steps=200, seed=123, greedy=False):
    """Coins per 100 steps in the REAL game -- the only number that counts."""
    vec = VecGame(n_env, seed=seed)
    f = vec.reset()
    total = 0.0
    for _ in range(steps):
        if policy is None:
            a = torch.randint(0, W.N_ACT, (n_env,))
        else:
            logits, _ = policy(f)
            a = (logits.argmax(1) if greedy
                 else torch.distributions.Categorical(logits=logits).sample())
        f, r = vec.step(a)
        total += float(r.sum())
    return 100.0 * total / (n_env * steps)


@torch.no_grad()
def evaluate_scripted(n_env=64, steps=200, seed=123, eps=0.0):
    """The hand-written coin-seeker, which cheats by reading the true state."""
    vec = VecGame(n_env, seed=seed)
    vec.reset()
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(steps):
        for e in vec.envs:
            a = W._greedy_action(e.walls, e.agent, e.coin, rng, eps)
            _, r, _ = e.step(a)
            total += r
    return 100.0 * total / (n_env * steps)


def collect(n_steps, n_env=64, seed=5, eps=0.6):
    """Real experience for training the world model.

    Actions are mostly random on purpose.  A world model has to be right about
    what happens after *any* button, including the ones a good player would
    never press, or the policy will walk straight into the parts of the model
    that were never trained.
    """
    vec = VecGame(n_env, seed=seed)
    vec.reset()
    rng = np.random.default_rng(seed)
    F0, A, R, F1 = [], [], [], []
    per_env = n_steps // n_env
    for _ in range(per_env):
        f0 = vec.frames()
        acts = np.array([
            (rng.integers(W.N_ACT) if rng.random() < eps
             else W._greedy_action(e.walls, e.agent, e.coin, rng, 0.0))
            for e in vec.envs])
        f1, r = vec.step(acts)
        F0.append(f0); A.append(torch.from_numpy(acts)); R.append(r); F1.append(f1)
    return (torch.cat(F0), torch.cat(A), torch.cat(R), torch.cat(F1))
