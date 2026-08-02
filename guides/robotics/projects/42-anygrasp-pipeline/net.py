"""A miniature PointNet that scores one 6-DoF grasp from the points it would close on.

The input is the little cloud of points that would end up *between* the
fingers, written in the grasp's own frame.  Putting it in the grasp frame is
the 3D version of project 40's rotated crop: the network then never has to
learn the same grasp at forty orientations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)

NPTS = 48
SCALE = np.array([0.036, 0.012, 0.024])      # the closing region's half-extents


def encode(g, rng=None):
    """One candidate -> a fixed-size (3, NPTS) array plus two global numbers.

    Two design notes worth reading, because both look redundant and are not.

    * The points are divided by the closing region's half-extents, so every
      coordinate lands in [-1, 1].  Without it the y axis (12 mm) contributes
      a hundredth as much as the x axis (36 mm) to every dot product, and the
      first layer spends its capacity undoing that.
    * The point COUNT is passed separately, even though the points themselves
      are right there.  Max-pooling is what makes a PointNet indifferent to
      the order and the number of its inputs -- that is the whole trick -- and
      the price is that it cannot tell forty points from four hundred.  When
      "how much material is between the fingers" is exactly what predicts
      success, you have to hand that number over explicitly.
    """
    Q = g["pts"] / SCALE
    n = len(Q)
    rng = rng or np.random
    if n >= NPTS:
        idx = rng.choice(n, NPTS, replace=False)
    else:
        idx = np.concatenate([np.arange(n), rng.choice(n, NPTS - n, replace=True)])
    glob = np.array([min(n, 400) / 200.0, g["width"] / 0.072], np.float32)
    return Q[idx].T.astype(np.float32), glob


class GraspNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = nn.Conv1d(3, 32, 1)
        self.p2 = nn.Conv1d(32, 64, 1)
        self.p3 = nn.Conv1d(64, 96, 1)
        self.f1 = nn.Linear(96 + 2, 64)
        self.f2 = nn.Linear(64, 1)

    def forward(self, x, g):
        # the same little network is applied to every point independently...
        h = F.relu(self.p1(x))
        h = F.relu(self.p2(h))
        h = F.relu(self.p3(h))
        # ...and then max-pooled, which is what makes the answer independent of
        # the order the points arrived in
        h = h.max(dim=2).values
        h = F.relu(self.f1(torch.cat([h, g], 1)))
        return self.f2(h).squeeze(1)


def train(X, G, y, epochs=25, bs=128, lr=2e-3, seed=0, log=None):
    torch.manual_seed(seed)
    net = GraspNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt, Gt = torch.from_numpy(X), torch.from_numpy(G)
    yt = torch.from_numpy(y.astype(np.float32))
    pos = float(yt.mean())
    w = torch.where(yt > 0.5, (1 - pos) / max(pos, 1e-6), 1.0)
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(0, len(Xt), bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xt[k], Gt[k])
            loss = (F.binary_cross_entropy_with_logits(out, yt[k],
                                                       reduction="none")
                    * w[k]).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(k)
        curve.append(tot / len(Xt))
        if log and (ep + 1) % 5 == 0:
            log(f"    epoch {ep + 1}/{epochs}  loss {curve[-1]:.4f}")
    return net, curve


@torch.no_grad()
def score(net, cands, rng=None):
    if not cands:
        return np.zeros(0)
    enc = [encode(g, rng) for g in cands]
    X = torch.from_numpy(np.stack([e[0] for e in enc]))
    G = torch.from_numpy(np.stack([e[1] for e in enc]))
    net.eval()
    s = torch.sigmoid(net(X, G)).numpy()
    net.train()
    return s
