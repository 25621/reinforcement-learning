"""Shared pieces for project 70 — serving a Mixture-of-Experts model.

The model is `ibm-granite/granite-3.0-1b-a400m-instruct`: a **real** MoE
checkpoint, small enough to run on a CPU.  1.3B total parameters, 400M of them
active per token, 24 layers, **32 experts per layer, top-8 routing**.  Its
router is a plain linear layer, so a forward hook on it gives us exactly what a
serving engine's dispatch code sees: for every token, at every layer, which 8
of the 32 experts it was sent to.

Nothing here trains or changes the model.  We only *watch* the routing, because
that is the quantity that decides whether expert parallelism runs at full speed
or at half speed.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "ibm-granite/granite-3.0-1b-a400m-instruct"
N_THREADS = 6


def add_quantlib_to_path():
    """Project 30's quantlib owns the corpora loader this project reuses."""
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(os.path.dirname(here), "30-quantize-a-7b-model-end-to-end")
    if p not in sys.path:
        sys.path.insert(0, p)


def load(model_id: str = MODEL_ID, threads: int = N_THREADS):
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return tok, model


def moe_config(model):
    cfg = model.config
    return dict(n_layers=cfg.num_hidden_layers,
                n_experts=cfg.num_local_experts,
                top_k=cfg.num_experts_per_tok,
                hidden=cfg.hidden_size,
                ffn=cfg.intermediate_size)


class RouterTap:
    """Record every routing decision the model makes.

    A forward hook on each layer's `block_sparse_moe.router` catches its output
    `(top_k_index, top_k_weights, router_logits)`.  We keep `top_k_index`: the
    expert ids this token was dispatched to.  That is the dispatch list an
    expert-parallel engine would put on the wire.
    """

    def __init__(self, model):
        self.routers = [m for n, m in model.named_modules()
                        if n.endswith("block_sparse_moe.router")]
        self.handles = []
        self.buf: list[list[np.ndarray]] = []   # per forward: per layer, [T, k]

    def __enter__(self):
        for li, r in enumerate(self.routers):
            self.handles.append(r.register_forward_hook(self._make(li)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _make(self, li):
        def hook(_mod, _inp, out):
            idx = out[0].detach().to(torch.int8).numpy()
            while len(self.buf) <= li:
                self.buf.append(None)
            self.buf[li] = idx
        return hook

    def take(self) -> np.ndarray:
        """Return [n_layers, n_tokens, top_k] for the last forward, then reset."""
        arr = np.stack(self.buf, axis=0)
        self.buf = []
        return arr


@torch.no_grad()
def route_corpus(model, tap: RouterTap, chunks) -> np.ndarray:
    """Run prefill over token chunks and return [n_layers, n_tokens, top_k]."""
    parts = []
    for ch in chunks:
        ids = torch.tensor(ch).unsqueeze(0)
        model(input_ids=ids)
        parts.append(tap.take())
    return np.concatenate(parts, axis=1)


# ------------------------------------------------------------- statistics ---

def counts(assign: np.ndarray, n_experts: int) -> np.ndarray:
    """Tokens-per-expert. `assign` is [..., top_k] over some set of tokens."""
    flat = assign.reshape(-1)
    return np.bincount(flat.astype(np.int64), minlength=n_experts)


def imbalance(c: np.ndarray) -> float:
    """max load / mean load.

    This is the number that matters for expert parallelism, because a step ends
    when the *slowest* expert finishes.  1.0 is perfect; 2.0 means the busiest
    expert does twice the average work, so half the hardware is idle at the end
    of every step.
    """
    m = c.mean()
    return float(c.max() / m) if m > 0 else float("nan")


def norm_entropy(c: np.ndarray) -> float:
    """Routing entropy divided by its maximum (log of the expert count).

    1.0 means tokens are spread perfectly evenly across experts; 0.0 means every
    token goes to the same expert.  Entropy is a *whole-distribution* measure,
    while `imbalance` only looks at the single worst expert — a distribution can
    have high entropy and still have one hot expert.
    """
    p = c / max(c.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / math.log(len(c)))


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two expert-usage distributions, in bits.

    Named after Johan Jensen and Claude Shannon: it is the *symmetric*,
    always-finite cousin of the Kullback-Leibler divergence — you compare each
    distribution against the average of the two, so neither one can blow up by
    assigning zero probability where the other has mass.  0 bits means two
    workloads use the experts identically; 1 bit means they share nothing.
    """
    p = a / max(a.sum(), 1)
    q = b / max(b.sum(), 1)
    m = 0.5 * (p + q)

    def kl(x, y):
        mask = x > 0
        return float((x[mask] * np.log2(x[mask] / y[mask])).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dropped_fraction(c: np.ndarray, n_tokens: int, top_k: int,
                     capacity_factor: float) -> float:
    """Fraction of dispatches dropped at a given capacity factor.

    A real engine allocates each expert a fixed-size buffer before it knows the
    routing: `capacity = capacity_factor * tokens * top_k / n_experts`.  Anything
    beyond that is thrown away — the token skips that expert entirely, which is
    a silent quality loss, not an error.
    """
    cap = capacity_factor * n_tokens * top_k / len(c)
    over = np.maximum(c - cap, 0).sum()
    return float(over / max(n_tokens * top_k, 1))


# --------------------------------------------------------------- placement ---

def placements(n_experts: int, n_dev: int, order: np.ndarray | None = None):
    """Three ways to spread experts over devices, as a list of device ids."""
    out = {}
    per = n_experts // n_dev
    out["contiguous"] = np.repeat(np.arange(n_dev), per)
    out["round_robin"] = np.tile(np.arange(n_dev), per)
    if order is not None:                       # greedy longest-first bin-pack
        dev = np.zeros(n_experts, dtype=int)
        loads = np.zeros(n_dev)
        slots = np.full(n_dev, per)
        for e in order:
            cand = [d for d in range(n_dev) if slots[d] > 0]
            d = min(cand, key=lambda d: loads[d])
            dev[e] = d
            loads[d] += 1
            slots[d] -= 1
        out["balanced"] = dev
    return out


def device_loads(c: np.ndarray, dev: np.ndarray, n_dev: int) -> np.ndarray:
    return np.array([c[dev == d].sum() for d in range(n_dev)], dtype=float)


def pct(xs, p):
    if len(xs) == 0:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])
