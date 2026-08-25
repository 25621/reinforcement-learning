"""The smallest honest behaviour-cloning stack: a normaliser, an MLP, a loop.

Shared with projects 55, 56, 59 and 61 so that every comparison in the phase is
against the *same* baseline rather than against a re-tuned one.
"""

import numpy as np
import torch
import torch.nn as nn


def seed_all(seed):
    """Seed before anything is built.

    Order matters: the network's initial weights are drawn from the torch RNG,
    so seeding after ``MLP(...)`` gives a different network every run and every
    "identical" comparison silently differs.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


class Norm:
    """Whitening for observations.

    The observation mixes sines and cosines (order 1), velocities (order 10)
    and positions in metres (order 0.1).  Fed raw, the first layer's gradients
    are dominated by whichever entry happens to have the largest units, and the
    metres -- the part that actually says where the puck is -- are the smallest.
    """

    def __init__(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0) + 1e-3      # a floor, not an epsilon: some entries
                                       # barely move and dividing by their true
                                       # spread would amplify pure noise
                                       # (project 52 learned this the hard way)

    def __call__(self, x):
        return (x - self.mu) / self.sd


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, depth=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_bc(obs, act, hidden=256, depth=2, epochs=150, bs=256, lr=1e-3,
             seed=0, val_frac=0.1, weight_decay=1e-4, log=None):
    """Supervised regression from observation to demonstrated action."""
    seed_all(seed)
    obs = np.asarray(obs, np.float32)
    act = np.asarray(act, np.float32)
    n = len(obs)
    idx = np.random.permutation(n)
    n_val = max(1, int(val_frac * n))
    va, tr = idx[:n_val], idx[n_val:]

    norm = Norm(obs[tr])
    X = torch.tensor(norm(obs), dtype=torch.float32)
    Y = torch.tensor(act, dtype=torch.float32)
    Xtr, Ytr, Xva, Yva = X[tr], Y[tr], X[va], Y[va]

    net = MLP(obs.shape[1], act.shape[1], hidden, depth)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    hist = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), bs):
            b = perm[i:i + bs]
            loss = ((net(Xtr[b]) - Ytr[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        with torch.no_grad():
            vl = float(((net(Xva) - Yva) ** 2).mean()) if len(Xva) else float("nan")
        hist.append((tot / max(1, len(Xtr)), vl))
        if log and (ep + 1) % 50 == 0:
            print(f"  {log} epoch {ep + 1:4d}  train {hist[-1][0]:.4f}  val {vl:.4f}",
                  flush=True)
    net.eval()
    return net, norm, np.array(hist)


def make_policy(net, norm):
    """Wrap a trained network as the ``policy(obs) -> action`` the env wants."""
    def policy(obs):
        with torch.no_grad():
            x = torch.tensor(norm(np.asarray(obs, np.float32))[None],
                             dtype=torch.float32)
            return net(x)[0].numpy()
    return policy


def action_mse(net, norm, obs, act):
    with torch.no_grad():
        x = torch.tensor(norm(np.asarray(obs, np.float32)), dtype=torch.float32)
        y = torch.tensor(np.asarray(act, np.float32), dtype=torch.float32)
        return float(((net(x) - y) ** 2).mean())
