"""A tiny action-conditioned video model, and a planner that only sees pictures.

Project 60 already planned with a learned model, so it is worth being precise
about what is different here.  There, the model predicted the six numbers that
describe the world (joint angles, joint speeds, puck position) and the planner
scored a plan by computing the distance from the puck to the goal.  Both of
those require somebody to have already told the robot what the state variables
are and what the goal means as a number.

Here the model sees **pictures** and the goal is **a picture**.  Nothing tells
it that there is a puck, or that pucks have positions.  That is the world-model
bet: video is a format every task can be written in, so a model that predicts
video could in principle be planned with for any task, without a hand-written
state or a hand-written reward.

What the robot sees
-------------------
A 32x32 two-channel image rendered from above:

* channel 0 -- the arm
* channel 1 -- the puck

Splitting them is deliberate and generous.  A real camera gives one RGB frame
in which the two are tangled, and separating them is a perception problem of
its own.  We hand the planner the *already solved* version so that when the
pixel-space plan fails anyway, the failure cannot be blamed on segmentation.

The goal image is the same render, with the puck at the goal and the arm
wherever it happens to be.  There is no goal marker in the picture: the goal
IS the picture.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402

IMG = 32
EXTENT = 0.42              # metres from the centre of the picture to its edge
CX = CY = (IMG - 1) / 2.0
ACT_DIM = 2

_gx, _gy = np.meshgrid(np.arange(IMG), np.arange(IMG), indexing="xy")
_PX = (_gx - CX) * (2 * EXTENT / IMG)
_PY = (CY - _gy) * (2 * EXTENT / IMG)          # y up, as in the table frame
_PIX = np.stack([_PX, _PY], -1)                # (IMG, IMG, 2) world coords


def _disc(centre, radius, soft=0.010):
    d = np.linalg.norm(_PIX - np.asarray(centre, float), axis=-1)
    return np.clip((radius + soft - d) / soft, 0.0, 1.0)


def _segment(p0, p1, radius, soft=0.008):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    v = p1 - p0
    L2 = float(v @ v) + 1e-12
    t = np.clip(((_PIX - p0) @ v) / L2, 0.0, 1.0)
    proj = p0 + t[..., None] * v
    d = np.linalg.norm(_PIX - proj, axis=-1)
    return np.clip((radius + soft - d) / soft, 0.0, 1.0)


def render(env, puck=None):
    """Two-channel picture of the scene, in [0, 1]."""
    pts = env.arm.points(env.q)
    a = np.zeros((IMG, IMG), np.float32)
    for k in range(env.arm.n):
        a = np.maximum(a, _segment(pts[k], pts[k + 1], 0.012))
    a = np.maximum(a, _disc(pts[-1], A.R_TIP))
    p = _disc(env.puck if puck is None else puck, A.R_PUCK).astype(np.float32)
    return np.stack([a, p]).astype(np.float32)


def goal_image(env):
    """What the camera would show if the puck were already on the goal."""
    return render(env, puck=env.goal)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
class VideoModel(nn.Module):
    """frame_t, action_t -> frame_{t+1}.

    It predicts the CHANGE, not the frame.  Two consecutive frames are almost
    identical -- a decision moves the arm a few pixels -- so a model asked for
    the absolute frame spends all its capacity re-drawing what it was given and
    a copy of the input already scores well.  Predicting the residual makes the
    thing being learned the thing we care about.
    """

    def __init__(self, ch=32, latent=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, ch, 4, 2, 1), nn.GELU(),        # 32 -> 16
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GELU(),   # 16 -> 8
            nn.Conv2d(ch * 2, ch * 2, 4, 2, 1), nn.GELU(),   # 8 -> 4
        )
        self.fc_in = nn.Linear(ch * 2 * 16 + ACT_DIM, latent)
        self.fc_out = nn.Linear(latent, ch * 2 * 16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1), nn.GELU(),   # 4 -> 8
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1), nn.GELU(),       # 8 -> 16
            nn.ConvTranspose2d(ch, 2, 4, 2, 1),                       # 16 -> 32
        )
        self.ch = ch

    def embed(self, x):
        return self.enc(x)

    def forward(self, x, a):
        f = self.enc(x)
        b = f.shape[0]
        z = torch.nn.functional.gelu(
            self.fc_in(torch.cat([f.reshape(b, -1), a], 1)))
        h = self.fc_out(z).reshape(b, -1, 4, 4)
        d = self.dec(torch.cat([f, h], 1))
        return torch.clamp(x + d, 0.0, 1.0)


def collect_play(n_eps, seed=0, kind="mixed"):
    """Play data: frames, actions, next frames -- and no demonstrations.

    Same principle as project 60.  A world model does not need anybody who can
    do the task; it needs the robot to move and something to watch it.
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    X, Aa, Y = [], [], []
    for ep in range(n_eps):
        env.reset()
        scripted = (kind == "mixed") and (ep % 2 == 0)
        side = 1 if rng.random() < 0.5 else -1
        for _ in range(A.EP_LEN):
            if scripted:
                a, _ = A.expert_action(env, side=side)
                a = np.clip(a + rng.normal(0, 0.35, 2), -1, 1)
            else:
                a = rng.uniform(-1, 1, 2)
            X.append(render(env))
            Aa.append(a)
            env.step(a)
            Y.append(render(env))
    return (np.array(X, np.float32), np.array(Aa, np.float32),
            np.array(Y, np.float32))


def train_model(X, Aa, Y, epochs=14, bs=128, lr=2e-3, seed=0, log=None):
    torch.manual_seed(seed)
    m = VideoModel()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xt = torch.tensor(X)
    At = torch.tensor(Aa)
    Yt = torch.tensor(Y)
    n = len(Xt)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            loss = ((m(Xt[b], At[b]) - Yt[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        hist.append(tot / n)
        if log:
            print(f"  {log} epoch {ep + 1:3d}  mse {hist[-1]:.5f}", flush=True)
    m.eval()
    return m, hist


# ---------------------------------------------------------------------------
# the planner
# ---------------------------------------------------------------------------
PUCK_W = None              # set by run.py for the reweighted cost


@torch.no_grad()
def imagine(model, frame, seqs):
    """Roll a batch of action sequences forward through the model."""
    b, h, _ = seqs.shape
    x = torch.tensor(frame)[None].expand(b, -1, -1, -1).contiguous()
    for k in range(h):
        x = model(x, seqs[:, k])
    return x


def cost_pixel(pred, goal, w_puck=1.0):
    """Mean squared pixel error against the goal image."""
    d = (pred - goal[None]) ** 2
    if w_puck != 1.0:
        d = torch.stack([d[:, 0], d[:, 1] * w_puck], 1)
    return d.reshape(len(pred), -1).mean(1)


def cost_puck_only(pred, goal):
    return ((pred[:, 1] - goal[None, 1]) ** 2).reshape(len(pred), -1).mean(1)


@torch.no_grad()
def cost_latent(model, pred, goal):
    """Distance in the model's own encoder features rather than in pixels."""
    f = model.embed(pred).reshape(len(pred), -1)
    g = model.embed(torch.as_tensor(goal)[None]).reshape(1, -1)
    return ((f - g) ** 2).mean(1)


@torch.no_grad()
def cem_plan(model, frame, goal, cost_fn, horizon=6, pop=48, iters=3,
             elites=8, rng=None):
    rng = rng or np.random.default_rng(0)
    mu = np.zeros((horizon, ACT_DIM), np.float32)
    sd = np.full((horizon, ACT_DIM), 0.75, np.float32)
    goal_t = torch.tensor(goal)
    best = mu.copy()
    for _ in range(iters):
        cand = np.clip(mu[None] + sd[None] * rng.normal(size=(pop, horizon,
                                                             ACT_DIM)), -1, 1)
        seqs = torch.tensor(cand.astype(np.float32))
        pred = imagine(model, frame, seqs)
        c = cost_fn(pred, goal_t).numpy()
        idx = np.argsort(c)[:elites]
        mu = cand[idx].mean(0).astype(np.float32)
        sd = (cand[idx].std(0) + 0.05).astype(np.float32)
        best = cand[idx[0]].astype(np.float32)
    return best, mu
