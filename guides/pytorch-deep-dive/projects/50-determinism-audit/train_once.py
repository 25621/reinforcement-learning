"""One training run, in a fresh process, printing a bit-exact fingerprint.

Why a separate process for every configuration? Because several of the things
that break reproducibility are *process-level*: `PYTHONHASHSEED` is read by the
interpreter at startup, the thread pool is created on first use, and a global
seed set halfway through a program does not undo what already happened. Running
each configuration twice in two fresh processes is the only honest way to ask
"would someone else, on this machine, get my number?".

Usage:  python3 train_once.py '<json config>'
Prints one JSON line.
"""

from __future__ import annotations

import json
import os
import random
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "48-nan-forensics"))
import debug_lib as D  # noqa: E402

CFG = json.loads(sys.argv[1])


# --- the controls, each one independently switchable -----------------------

def apply_controls(cfg):
    if cfg.get("torch_seed", True):
        torch.manual_seed(0)
    if cfg.get("py_np_seed", True):
        random.seed(0)
        np.random.seed(0)
    if cfg.get("deterministic_algos", False):
        torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(cfg.get("threads", 4))


def worker_init(worker_id):
    """Give every DataLoader worker its own reproducible seeds.

    This is the function every tutorial tells you to write, on the grounds that
    PyTorch seeds each worker's *torch* generator but leaves `numpy` and
    Python's `random` alone. Section 3 checks that claim against this PyTorch
    version's source and against a measurement, and finds it is no longer true:
    `torch/utils/data/_utils/worker.py` seeds all three. It is still worth
    passing for any *other* library that keeps global random state.
    """
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


# --- data ------------------------------------------------------------------

class NoisyFeatures(Dataset):
    """A fixed dataset plus a small unseeded-by-default numpy augmentation."""

    def __init__(self, n=1024, d=64, augment=True):
        g = torch.Generator().manual_seed(123)
        self.x = torch.randn(n, d, generator=g)
        self.y = (self.x[:, :8].sum(1) > 0).long()
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x = self.x[i]
        if self.augment:
            x = x + torch.from_numpy(np.random.randn(x.numel()).astype("float32")) * 0.01
        return x, self.y[i]


class Net(nn.Module):
    def __init__(self, d=64, width=256):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(d, width), nn.ReLU(),
                               nn.Linear(width, width), nn.ReLU(),
                               nn.Linear(width, 2))
        self.emb = nn.Embedding(16, d)

    def forward(self, x, ids):
        return self.f(x + self.emb(ids))


def build_vocab(cfg):
    """Map 16 category names to embedding rows.

    Building the mapping by iterating a `set` is a very common shortcut. A
    Python `set` has no order, and since Python 3.3 the hash of a *string* is
    randomised per process unless `PYTHONHASHSEED` is fixed — so this loop
    assigns different ids to the same words in different processes, and the
    model that reads `emb[id]` sees different numbers.
    """
    words = {f"cat{i}" for i in range(16)}
    if cfg.get("sorted_vocab", False):
        words = sorted(words)
    return {w: i for i, w in enumerate(words)}


def main():
    cfg = CFG
    apply_controls(cfg)

    vocab = build_vocab(cfg)
    ids_for_row = torch.tensor([vocab[f"cat{i % 16}"] for i in range(1024)])

    ds = NoisyFeatures(augment=cfg.get("augment", True))
    gen = None
    if cfg.get("loader_generator", True):
        gen = torch.Generator()
        gen.manual_seed(0)
    workers = cfg.get("workers", 0)
    dl = DataLoader(ds, batch_size=64, shuffle=True, generator=gen,
                    num_workers=workers,
                    worker_init_fn=worker_init if cfg.get("worker_init", True) else None)

    model = Net()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    row = 0
    for _ in range(cfg.get("epochs", 3)):
        for xb, yb in dl:
            ids = ids_for_row[row:row + len(xb)]
            row = (row + len(xb)) % 1024
            if len(ids) < len(xb):
                ids = ids_for_row[:len(xb)]
            loss = F.cross_entropy(model(xb, ids), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.detach().item())

    print(json.dumps({
        "fingerprint": D.state_fingerprint(model),
        "loss_fingerprint": D.fingerprint(torch.tensor(losses)),
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "steps": len(losses),
        "param_sum": float(sum(p.detach().double().sum().item()
                               for p in model.parameters())),
        "threads": torch.get_num_threads(),
        "hashseed": os.environ.get("PYTHONHASHSEED", "unset"),
        "losses": losses,
    }))


if __name__ == "__main__":
    main()
