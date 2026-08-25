"""Multi-LoRA: train several small adapters, then serve them from one base.

LoRA ("Low-Rank Adaptation") replaces a fine-tune of a big weight matrix W
with a *thin* correction:

    y = xW^T  +  (xA^T)B^T          A: (r, in)   B: (out, r)

`r` is the rank, here 8. For a 896x896 projection that is 14,336 numbers
instead of 802,816 -- 56x smaller. "Low-rank" is the linear-algebra term for
"this matrix can be written as the product of two skinny ones", and the bet
LoRA makes is that whatever a fine-tune needs to change is simple enough to
fit in that shape.

Two classes here, and the difference between them is the whole project:

    LoRALinear       one adapter, used while TRAINING. Every row in the
                     batch belongs to the same tenant.
    MultiLoRALinear  many adapters, used while SERVING. Every row in the
                     batch may belong to a different tenant, and all of them
                     go through one call. This is the "BGMV/SGMV" kernel that
                     S-LoRA and Punica exist to provide -- written here as
                     two batched matrix multiplies.

Why not just swap the adapter in between requests? Because that serialises
tenants: you would run a forward pass for tenant A, then another for tenant
B, and a decode step costs almost the same whether it carries 1 row or 32.
Batching across tenants is the only way one device serves many fine-tunes at
full speed, and it is impossible unless the adapter is selected *per row*.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

RANK = 8
N_ADAPTED_LAYERS = 8          # the last 8 of the model's 24 blocks
TARGETS = ("q_proj", "v_proj")


# ---------------------------------------------------------------------------
# Training-time: one adapter
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = RANK):
        super().__init__()
        self.base = base
        # A is random and B is zero, so the adapter starts as an exact no-op
        # and the model behaves like the untouched base at step 0. Starting
        # both at random would corrupt the model before it has learnt
        # anything.
        self.A = nn.Parameter(torch.randn(rank, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        self.scale = 2.0 / rank

    def forward(self, x):
        return self.base(x) + ((x @ self.A.T) @ self.B.T) * self.scale


def attach_training_lora(model, rank: int = RANK,
                         n_layers: int = N_ADAPTED_LAYERS):
    """Wrap the target projections of the last `n_layers` blocks."""
    params = []
    blocks = model.model.layers
    for blk in blocks[len(blocks) - n_layers:]:
        for name in TARGETS:
            mod = getattr(blk.self_attn, name)
            if isinstance(mod, LoRALinear):
                mod = mod.base
            lora = LoRALinear(mod, rank)
            setattr(blk.self_attn, name, lora)
            params += [lora.A, lora.B]
    return params


def detach_lora(model):
    """Put the plain Linear layers back."""
    blocks = model.model.layers
    for blk in blocks:
        for name in TARGETS:
            mod = getattr(blk.self_attn, name)
            if isinstance(mod, (LoRALinear, MultiLoRALinear)):
                setattr(blk.self_attn, name, mod.base)


def adapter_state(model, n_layers: int = N_ADAPTED_LAYERS):
    blocks = model.model.layers
    out = {}
    for i, blk in enumerate(blocks[len(blocks) - n_layers:]):
        for name in TARGETS:
            mod = getattr(blk.self_attn, name)
            out[f"{i}.{name}.A"] = mod.A.detach().clone()
            out[f"{i}.{name}.B"] = mod.B.detach().clone()
    return out


def adapter_bytes(state) -> int:
    return sum(t.numel() * t.element_size() for t in state.values())


# ---------------------------------------------------------------------------
# Serving-time: many adapters, one forward pass
# ---------------------------------------------------------------------------


class Router:
    """Which adapter each row of the current batch belongs to.

    A module-level slot rather than an extra argument, because the modules
    are buried inside HuggingFace's forward and there is no clean way to
    thread a per-row id down to them. Real engines do the same thing with a
    request-metadata object attached to the batch.
    """

    idx: torch.Tensor | None = None    # (batch,) long, values in [0, n_ad]


class MultiLoRALinear(nn.Module):
    """base + the adapter each row asks for, in one call.

    Index `n_ad` (one past the last real adapter) is a permanently-zero
    adapter. That is how a request with *no* fine-tune rides in the same
    batch as five that have one: it selects the zero adapter, the correction
    it receives is exactly zero, and no branch is needed.
    """

    def __init__(self, base: nn.Linear, stacks, scale: float):
        super().__init__()
        self.base = base
        self.A = stacks[0]          # (n_ad+1, r, in)
        self.B = stacks[1]          # (n_ad+1, out, r)
        self.scale = scale

    def forward(self, x):
        y = self.base(x)
        idx = Router.idx
        if idx is None:
            return y
        b = x.shape[0]
        if x.dim() == 2:                     # (tokens, in) -- rare path
            x = x.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        a = self.A[idx]                      # (b, r, in)
        bb = self.B[idx]                     # (b, out, r)
        h = torch.bmm(x, a.transpose(1, 2))  # (b, t, r)
        d = torch.bmm(h, bb.transpose(1, 2))  # (b, t, out)
        if squeeze:
            d = d.squeeze(1)
        return y + d.view_as(y) * self.scale


def attach_serving_lora(model, states, rank: int = RANK,
                        n_layers: int = N_ADAPTED_LAYERS, scale=2.0 / RANK):
    """Stack N adapters (plus a zero one) into every adapted projection."""
    blocks = model.model.layers
    n_ad = len(states)
    for i, blk in enumerate(blocks[len(blocks) - n_layers:]):
        for name in TARGETS:
            mod = getattr(blk.self_attn, name)
            if isinstance(mod, (LoRALinear, MultiLoRALinear)):
                mod = mod.base
            A = torch.stack([s[f"{i}.{name}.A"] for s in states]
                            + [torch.zeros(rank, mod.in_features)])
            B = torch.stack([s[f"{i}.{name}.B"] for s in states]
                            + [torch.zeros(mod.out_features, rank)])
            setattr(blk.self_attn, name,
                    MultiLoRALinear(mod, (A, B), scale))
    return n_ad
