"""A small diffusion policy, plus the two baselines it has to beat.

The problem it solves is not "the network is too small".  It is that squared
error asks for the *average* correct action, and when two opposite actions are
both correct -- go round the puck clockwise, or anticlockwise -- their average
is a third action that is wrong (drive straight into the puck).

A diffusion policy sidesteps that by never predicting an action directly.  It
learns to *denoise*: given the observation and a blob of pure noise, it removes
a little noise at a time until an action chunk falls out.  Two modes in the
data become two basins the denoiser can fall into, chosen by which random noise
it started from, so it produces one mode or the other rather than the average.

Vocabulary, decoded:

* **diffusion** -- borrowed from physics: adding noise step by step spreads a
  sharp distribution out the way a drop of ink spreads in water.  Training
  learns to run that process backwards.
* **DDPM** -- Denoising Diffusion Probabilistic Model, the standard recipe:
  train the network to predict the noise that was added.
* **epsilon prediction** -- the network outputs the noise (traditionally
  written as the Greek letter epsilon), not the clean action.  Subtracting a
  predicted noise turns out to be easier to learn than producing a clean sample
  in one shot.
* **DDIM** -- Denoising Diffusion *Implicit* Model: the same trained network,
  sampled along a shorter path.  It is what lets a model trained with 50 noise
  levels run with 5 at test time -- and on a robot, test-time steps are
  milliseconds of control latency.
* **action chunk** -- the policy outputs the next H actions in one go instead
  of one.  The name is literal: a chunk of the future.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402


# ---------------------------------------------------------------------------
# data: observations paired with the next H actions
# ---------------------------------------------------------------------------
def chunk_demos(n_demos, horizon=8, seed=0, side_mode="random", noise=0.0):
    """Collect demos and slice them into (obs, next-H-actions) pairs.

    Near the end of an episode there are fewer than H actions left; those are
    padded by repeating the last one, which is what the robot effectively does
    anyway once it has stopped.
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    O, C, sides = [], [], []
    while len(sides) < n_demos:
        side = (1 if rng.random() < 0.5 else -1) if side_mode == "random" else int(side_mode)
        r = A.rollout(env, None, side=side, record=True, noise=noise)
        if not r["success"]:
            continue
        obs, act = r["obs"], r["act"]
        T = len(obs)
        for t in range(T):
            idx = np.clip(np.arange(t, t + horizon), 0, T - 1)
            O.append(obs[t])
            C.append(act[idx])
        sides.append(side)
    return (np.array(O, np.float32), np.array(C, np.float32),
            dict(n=len(sides), sides=np.array(sides)))


# ---------------------------------------------------------------------------
# the denoiser
# ---------------------------------------------------------------------------
def timestep_embedding(t, dim=32):
    """Sinusoidal features for the noise level, as in the original DDPM.

    A single number would work in principle, but a network reads a smooth
    number as "almost the same" for nearby values; sinusoids of many
    frequencies give it a code where noise level 3 and noise level 4 are
    clearly distinguishable while still ordered.
    """
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, -6, half, device=t.device))
    ang = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)


class Denoiser(nn.Module):
    def __init__(self, obs_dim, act_dim, horizon, hidden=256, temb=32):
        super().__init__()
        self.horizon, self.act_dim, self.temb = horizon, act_dim, temb
        d_in = obs_dim + act_dim * horizon + temb
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, act_dim * horizon),
        )

    def forward(self, obs, a_noisy, t):
        h = torch.cat([obs, a_noisy.flatten(1), timestep_embedding(t, self.temb)], 1)
        return self.net(h).view(-1, self.horizon, self.act_dim)


class Diffusion:
    """Cosine noise schedule + epsilon-prediction training + DDIM sampling."""

    def __init__(self, n_train_steps=50):
        self.T = n_train_steps
        s = 0.008
        x = torch.linspace(0, 1, n_train_steps + 1)
        f = torch.cos((x + s) / (1 + s) * np.pi / 2) ** 2
        self.abar = (f / f[0])[1:]                 # alpha-bar_t, t = 1..T
        self.abar = self.abar.clamp(1e-5, 0.9999)

    def add_noise(self, a0, t, eps):
        ab = self.abar[t][:, None, None]
        return ab.sqrt() * a0 + (1 - ab).sqrt() * eps

    @torch.no_grad()
    def sample(self, model, obs, n_steps=10, generator=None):
        """Start from pure noise, walk back down the schedule (DDIM, eta = 0)."""
        B = obs.shape[0]
        a = torch.randn(B, model.horizon, model.act_dim, generator=generator)
        ts = torch.linspace(self.T - 1, 0, n_steps).long()
        for i, t in enumerate(ts):
            tb = torch.full((B,), int(t), dtype=torch.long)
            eps = model(obs, a, tb)
            ab = self.abar[t]
            a0 = (a - (1 - ab).sqrt() * eps) / ab.sqrt()
            a0 = a0.clamp(-1, 1)
            if i == len(ts) - 1:
                a = a0
            else:
                ab_prev = self.abar[ts[i + 1]]
                a = ab_prev.sqrt() * a0 + (1 - ab_prev).sqrt() * eps
        return a


def train_diffusion(O, C, epochs=300, bs=256, lr=1e-3, seed=0, hidden=256,
                    n_train_steps=50, log=None):
    nets.seed_all(seed)
    norm = nets.Norm(O)
    X = torch.tensor(norm(O), dtype=torch.float32)
    Y = torch.tensor(C, dtype=torch.float32)
    horizon, act_dim = Y.shape[1], Y.shape[2]
    model = Denoiser(X.shape[1], act_dim, horizon, hidden=hidden)
    diff = Diffusion(n_train_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            a0 = Y[b]
            t = torch.randint(0, diff.T, (len(b),))
            eps = torch.randn_like(a0)
            pred = model(X[b], diff.add_noise(a0, t, eps), t)
            loss = ((pred - eps) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        if log and (ep + 1) % 100 == 0:
            print(f"  {log} epoch {ep + 1}: denoise loss {tot / len(X):.4f}", flush=True)
    model.eval()
    return model, norm, diff


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def train_mse_chunk(O, C, epochs=300, bs=256, lr=1e-3, seed=0, hidden=256):
    """Ordinary regression, but predicting the whole chunk (the fair control).

    Without this, a diffusion-vs-MLP comparison confounds two changes at once:
    the distribution model AND the action chunking.  This isolates the chunk.
    """
    horizon, act_dim = C.shape[1], C.shape[2]
    net, norm, hist = nets.train_bc(O, C.reshape(len(C), -1), epochs=epochs,
                                    bs=bs, lr=lr, seed=seed, hidden=hidden)
    return net, norm, horizon, act_dim


# ---------------------------------------------------------------------------
# executing a chunk
# ---------------------------------------------------------------------------
class ChunkPolicy:
    """Runs a chunked policy with a receding horizon.

    ``exec_len`` is how many of the H predicted actions are actually executed
    before re-planning.  exec_len = 1 is fully closed loop (re-plan every step);
    exec_len = H is open loop (commit to the whole chunk).  The trade is
    reactivity against consistency: re-planning every step lets the policy
    change its mind mid-way around the puck, which is exactly the flip-flopping
    that multimodal data causes.
    """

    def __init__(self, kind, model, norm, diff=None, horizon=8, exec_len=1,
                 n_steps=10, seed=0):
        self.kind, self.model, self.norm, self.diff = kind, model, norm, diff
        self.horizon, self.exec_len, self.n_steps = horizon, exec_len, n_steps
        self.gen = torch.Generator().manual_seed(seed)
        self.buf = []

    def reset(self):
        self.buf = []

    def __call__(self, obs):
        if not self.buf:
            x = torch.tensor(self.norm(np.asarray(obs, np.float32))[None],
                             dtype=torch.float32)
            with torch.no_grad():
                if self.kind == "diffusion":
                    ch = self.diff.sample(self.model, x, self.n_steps,
                                          generator=self.gen)[0].numpy()
                else:
                    ch = self.model(x)[0].numpy().reshape(self.horizon, -1)
            self.buf = list(ch[:self.exec_len])
        return np.clip(self.buf.pop(0), -1, 1)


def evaluate_chunk(policy, n=60, seed=999, side=None):
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    ok, errs, steps = 0, [], []
    for _ in range(n):
        obs = env.reset()
        policy.reset()
        done = False
        while not done:
            obs, _, done, _ = env.step(policy(obs))
        ok += env.success
        errs.append(env.err)
        if env.success:
            steps.append(env.t)
    return dict(success=ok / n, err=float(np.mean(errs)),
                steps=float(np.mean(steps)) if steps else float("nan"))
