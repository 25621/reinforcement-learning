"""Shared pieces for project 68 — sessions that outlive a single request.

Project 57 asked "which eviction policy keeps the hit rate up?".  This project
asks the question underneath it: **when a session has to leave GPU memory, what
exactly leaves?**  There are two answers and they cost different things.

  drop      throw the KV cache away.  Free instantly, but the next turn has to
            re-run prefill over the whole conversation.
  offload   copy the KV cache to host memory and free the GPU copy.  Costs a
            copy out and a copy back, but no recomputation.

Everything here works on a real `DynamicCache` from a real model, so the
byte counts and the copy times are measured, not modelled.  The only modelled
number in the project is the PCIe bandwidth of a machine we do not have, and it
is labelled as such everywhere it appears.
"""

from __future__ import annotations

import os
import sys
import time

import torch
from transformers.cache_utils import DynamicCache


def add_ctxlib_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(os.path.dirname(here), "51-needle-in-a-haystack")
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------- cache ops ---

def cache_layers(past: DynamicCache):
    return [ly for ly in past.layers if ly.keys is not None]


def cache_bytes(past: DynamicCache) -> int:
    return sum(ly.keys.numel() * ly.keys.element_size()
               + ly.values.numel() * ly.values.element_size()
               for ly in cache_layers(past))


def clone_cache(past: DynamicCache) -> DynamicCache:
    return DynamicCache(ddp_cache_data=[(ly.keys.clone(), ly.values.clone())
                                        for ly in cache_layers(past)])


def crop(past: DynamicCache, n: int) -> DynamicCache:
    past.crop(n)
    return past


class Offloaded:
    """A session's KV cache parked outside the accelerator.

    On a GPU server this is pinned host memory reached over PCIe.  Here it is
    ordinary RAM, so the *copy* is real and timed while the *link* is not — the
    arithmetic for a real PCIe link is done separately in run.py and labelled.
    """

    def __init__(self, past: DynamicCache):
        self.blobs = [(ly.keys.detach().clone(), ly.values.detach().clone())
                      for ly in cache_layers(past)]
        self.nbytes = sum(k.numel() * k.element_size()
                          + v.numel() * v.element_size()
                          for k, v in self.blobs)

    def restore(self) -> DynamicCache:
        return DynamicCache(ddp_cache_data=[(k.clone(), v.clone())
                                            for k, v in self.blobs])


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    out = fn(*a, **kw)
    return out, time.perf_counter() - t0


# ------------------------------------------------------------- the session ---

class Session:
    """One agent conversation: its token history and, maybe, its live cache."""

    __slots__ = ("sid", "ids", "kv", "parked", "last_used", "turns",
                 "recomputed", "restored", "offloads", "drops")

    def __init__(self, sid: int):
        self.sid = sid
        self.ids: list[int] = []
        self.kv: DynamicCache | None = None
        self.parked: Offloaded | None = None
        self.last_used = 0.0
        self.turns = 0
        self.recomputed = 0
        self.restored = 0
        self.offloads = 0
        self.drops = 0

    @property
    def resident(self) -> bool:
        return self.kv is not None

    def nbytes(self) -> int:
        return cache_bytes(self.kv) if self.kv is not None else 0


@torch.inference_mode()
def prefill(model, ids: list[int], past: DynamicCache | None = None):
    """Run one prefill (or a partial one on top of `past`); return (cache, s)."""
    t0 = time.perf_counter()
    out = model(torch.tensor([ids]), past_key_values=past, use_cache=True,
                logits_to_keep=1)
    return out.past_key_values, time.perf_counter() - t0


@torch.inference_mode()
def decode(model, past: DynamicCache, first_logits, n_new: int, eos=None):
    """Greedy decode `n_new` tokens on top of a cache; return (tokens, s)."""
    nxt = first_logits[:, -1, :].argmax(-1, keepdim=True)
    toks = [int(nxt)]
    t0 = time.perf_counter()
    for _ in range(n_new - 1):
        if eos is not None and toks[-1] == eos:
            break
        out = model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        toks.append(int(nxt))
    return toks, time.perf_counter() - t0


def pct(xs, p):
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return float(s[k])
