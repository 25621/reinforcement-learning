"""Three policies that differ only in what they are allowed to look at.

  StatePolicy  -- the object's true pose, handed over for free
  PixelPolicy  -- a camera image, plus where the robot's own hand is
  ImageOnly    -- the camera image and nothing else

Keeping the network body identical across all three is deliberate.  Any
difference in the results is then a difference in *information*, not in
architecture, which is the only way the comparison means anything.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(8)

PROP = 6
PRIV = 5
ACT = 5


class Trunk(nn.Module):
    """Shared vision stack: four strided convolutions down to a flat vector."""

    def __init__(self, out=128):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.c2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(32, 48, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.fc = nn.Linear(64 * 4 * 6, out)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        return F.relu(self.fc(x.flatten(1)))


class Policy(nn.Module):
    def __init__(self, mode="pixel", hidden=128):
        super().__init__()
        self.mode = mode
        self.trunk = Trunk(hidden) if mode in ("pixel", "image") else None
        n_in = {"state": PROP + PRIV, "pixel": hidden + PROP,
                "image": hidden}[mode]
        self.f1 = nn.Linear(n_in, 128)
        self.f2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, ACT)

    def forward(self, img, prop, priv):
        if self.mode == "state":
            h = torch.cat([prop, priv], 1)
        elif self.mode == "pixel":
            h = torch.cat([self.trunk(img), prop], 1)
        else:
            h = self.trunk(img)
        h = F.relu(self.f1(h))
        h = F.relu(self.f2(h))
        out = self.head(h)
        # the four motion outputs are squashed into [-1, 1], the gripper
        # output is a logit read as "should the hand be open"
        return torch.cat([torch.tanh(out[:, :4]), out[:, 4:]], 1)


def pack(obs_list):
    img = np.stack([o["img"] for o in obs_list])
    prop = np.stack([o["prop"] for o in obs_list])
    priv = np.stack([o["priv"] for o in obs_list])
    return img, prop, priv


def train(data, mode, epochs=18, bs=64, lr=1e-3, seed=0, log=None):
    """Behaviour cloning: plain supervised regression onto the expert's action.

    Note what is NOT here: no reward, no simulator, no exploration.  That is
    what makes it cheap, and it is also the whole of its weakness -- the
    training set only ever contains states the *expert* visited.
    """
    img, prop, priv, act = data
    torch.manual_seed(seed)
    net = Policy(mode)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    I = torch.from_numpy(img)
    P = torch.from_numpy(prop)
    V = torch.from_numpy(priv)
    A = torch.from_numpy(act)
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(len(A))
        tot = 0.0
        for i in range(0, len(A), bs):
            k = perm[i:i + bs]
            opt.zero_grad()
            out = net(I[k], P[k], V[k])
            loss = F.mse_loss(out[:, :4], A[k, :4]) + \
                0.3 * F.binary_cross_entropy_with_logits(out[:, 4], A[k, 4])
            loss.backward()
            opt.step()
            tot += loss.item() * len(k)
        curve.append(tot / len(A))
        if log and (ep + 1) % 6 == 0:
            log(f"    {mode:6s} epoch {ep + 1}/{epochs}  loss {curve[-1]:.4f}")
    return net, curve


def make_policy(net):
    @torch.no_grad()
    def f(obs):
        net.eval()
        i = torch.from_numpy(obs["img"][None])
        p = torch.from_numpy(obs["prop"][None])
        v = torch.from_numpy(obs["priv"][None])
        a = net(i, p, v)[0].numpy()
        a[4] = 1.0 if a[4] > 0 else 0.0
        return a
    return f
