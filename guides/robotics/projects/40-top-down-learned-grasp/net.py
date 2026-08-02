"""The grasp-quality network: a 32x32 depth patch in, one probability out.

This is a miniature GQ-CNN (Dex-Net's network).  It is deliberately small --
about 60 000 weights -- because the interesting question in this project is not
"how big a network" but "does learning from noisy depth beat computing from
noisy depth".  A model that trains in ninety seconds lets us ask that question
at eight noise levels instead of one.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)


class GQCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 5, padding=2)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 32, 3, padding=1)
        self.f1 = nn.Linear(32 * 4 * 4, 96)
        self.f2 = nn.Linear(96, 1)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)      # 32 -> 16
        x = F.max_pool2d(F.relu(self.c2(x)), 2)      # 16 -> 8
        x = F.max_pool2d(F.relu(self.c3(x)), 2)      # 8  -> 4
        x = F.relu(self.f1(x.flatten(1)))
        return self.f2(x).squeeze(1)


def train(X, y, epochs=8, bs=256, lr=2e-3, seed=0, log=None):
    """Plain binary cross-entropy.  Returns the model and the loss curve.

    The labels are balanced by weighting, not by throwing data away: a scene
    has far more bad grasps than good ones, and the bad ones are the ones that
    teach the network where the object edges and the neighbours are.
    """
    torch.manual_seed(seed)
    net = GQCNN()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.from_numpy(X).unsqueeze(1)
    yt = torch.from_numpy(y.astype(np.float32))
    pos = float(yt.mean())
    w = torch.where(yt > 0.5, (1 - pos) / max(pos, 1e-6), 1.0)
    n = len(Xt)
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xt[idx])
            loss = (F.binary_cross_entropy_with_logits(out, yt[idx],
                                                       reduction="none")
                    * w[idx]).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        curve.append(tot / n)
        if log:
            log(f"    epoch {ep + 1}/{epochs}  loss {curve[-1]:.4f}")
    return net, curve


@torch.no_grad()
def score(net, X, bs=1024):
    net.eval()
    out = []
    for i in range(0, len(X), bs):
        t = torch.from_numpy(X[i:i + bs]).unsqueeze(1)
        out.append(torch.sigmoid(net(t)).numpy())
    net.train()
    return np.concatenate(out) if out else np.zeros(0)
