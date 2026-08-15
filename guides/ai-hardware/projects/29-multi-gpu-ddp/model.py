"""A deliberately boring model, so that every number in project 29 is about
communication rather than about clever architecture.

6 linear layers of 512x512 ~= 1.58 M parameters ~= 6.3 MB of fp32 gradients.
That is small enough to train in seconds and big enough that one all-reduce per
step is a real message rather than a rounding error.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d: int = 512, layers: int = 6, n_class: int = 10):
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(d, d), nn.GELU()]
        self.body = nn.Sequential(*blocks)
        self.head = nn.Linear(d, n_class)

    def forward(self, x):
        return self.head(self.body(x))


def make_model(seed: int = 0, **kw) -> MLP:
    """Identical weights on every rank -- data parallelism assumes the replicas
    start equal, and DDP only broadcasts at construction time if we let it."""
    torch.manual_seed(seed)
    return MLP(**kw)


def param_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def fake_batch(batch: int, d: int = 512, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, d, generator=g)
    y = torch.randint(0, 10, (batch,), generator=g)
    return x, y
