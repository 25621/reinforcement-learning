"""Pretrain on many robots, then fine-tune on one.

Two details that look like bookkeeping and are not:

* the loss is MASKED, so a two-joint robot contributes nothing to the third
  action slot.  Without the mask the padded zeros act as a training signal --
  the policy learns to output zero for joint three even on the robot that has
  one;
* the input normaliser is FROZEN at fine-tuning time.  Re-fitting it on five
  target demonstrations would move every input under the pretrained weights,
  which is a fast way to destroy the thing you paid for.
"""

import os
import sys

import numpy as np
import torch

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nets                # noqa: E402
import embodiment as EMB   # noqa: E402


def masked_mse(pred, y, m):
    return (((pred - y) ** 2) * m).sum() / m.sum().clamp(min=1.0)


def train(O, Y, M, epochs=250, bs=256, lr=1e-3, seed=0, hidden=256,
          init=None, norm=None, freeze_trunk=False, log=None):
    """Behaviour cloning with padded actions; optionally warm-started."""
    nets.seed_all(seed)
    if norm is None:
        norm = nets.Norm(O)
    net = nets.MLP(O.shape[1], EMB.ACT_DIM, hidden, depth=2)
    if init is not None:
        net.load_state_dict(init)
    if freeze_trunk:
        for i, layer in enumerate(net.net):
            if i < len(net.net) - 1:
                for p in layer.parameters():
                    p.requires_grad_(False)
    X = torch.tensor(norm(O), dtype=torch.float32)
    Yt, Mt = torch.tensor(Y), torch.tensor(M)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), bs):
            j = perm[i:i + bs]
            loss = masked_mse(net(X[j]), Yt[j], Mt[j])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        if log and (ep + 1) % 100 == 0:
            print(f"  {log} epoch {ep + 1}: {float(loss.detach()):.4f}", flush=True)
    net.eval()
    return net, norm


def policy_of(net, norm, drop_emb=False):
    def pol(obs):
        o = np.asarray(obs, np.float32).copy()
        if drop_emb:
            o[-4:] = 0.0
        with torch.no_grad():
            return net(torch.tensor(norm(o)[None], dtype=torch.float32))[0].numpy()
    return pol


def pretrain(names, n_demos=90, seed=0, epochs=250, hidden=256, drop_emb=False):
    """One policy trained on several robots at once."""
    Os, Ys, Ms = [], [], []
    for k, name in enumerate(names):
        O, Y, M, _ = EMB.collect(name, n_demos, seed=100 + k)
        Os.append(O)
        Ys.append(Y)
        Ms.append(M)
    O, Y, M = np.concatenate(Os), np.concatenate(Ys), np.concatenate(Ms)
    if drop_emb:
        O = O.copy()
        O[:, -4:] = 0.0
    net, norm = train(O, Y, M, epochs=epochs, seed=seed, hidden=hidden)
    return net, norm, len(O)
