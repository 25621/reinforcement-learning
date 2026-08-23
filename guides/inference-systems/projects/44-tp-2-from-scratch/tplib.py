"""Tensor parallelism (TP=2) by hand, on top of Phase 3's batched Qwen2 engine.

The whole trick is in `__init__` and `_layer`:

  * `__init__` SLICES the weight matrices. Rank 0 keeps query heads 0-6 and
    KV head 0; rank 1 keeps query heads 7-13 and KV head 1. The MLP's
    gate/up matrices are split the same way (each rank keeps half of the
    4864 hidden columns).
  * `_layer` computes attention and the MLP with only the local half, then
    calls `dist.all_reduce` so both ranks end up holding the full result.

That is Megatron-style TP in two sentences: **column-parallel in, row-parallel
out, one all-reduce per block**. Everything else (RoPE, the KV pool, masks,
the residual stream) is inherited unchanged from `batchlib.BatchedRunner` --
the residual stream `x` is REPLICATED on both ranks, only the wide inner
matrices are sharded.

Why the split works without changing any math:

  * Attention heads never talk to each other until the output projection,
    so giving each rank its own heads is exact, not approximate.
  * Qwen2's GQA grouping maps query heads 0-6 to KV head 0 and heads 7-13
    to KV head 1 (`repeat_interleave`), so slicing along the head axis also
    slices the KV cache cleanly -- each rank stores HALF the KV cache.
  * The MLP is `down(silu(gate(x)) * up(x))`. Splitting gate/up by output
    column and `down` by input column makes each rank's `down` output a
    partial sum; adding the two partial sums (the all-reduce) is exactly
    the full matrix product.

Used by `tp_run.py` (launched under torchrun) and read by the README.
"""

from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "16-static-vs-continuous"))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import batchlib  # noqa: E402
from batchlib import rms_norm, apply_rope_rows  # noqa: E402


class TPRunner(batchlib.BatchedRunner):
    """BatchedRunner that holds 1/world of every attention head and MLP column.

    Correct for any `world` that divides both the head count (14) and the
    KV-head count (2) -- i.e. world in {1, 2} for Qwen2.5-0.5B. That limit is
    real and worth noticing: **you cannot shard finer than a KV head** without
    duplicating KV cache, which is why TP degree on GQA models is usually
    kept at or below the KV-head count (or KV is replicated above it).
    """

    def __init__(self, model, tok, rank: int, world: int):
        super().__init__(model, tok)
        assert self.n_heads % world == 0 and self.n_kv_heads % world == 0
        self.rank, self.world = rank, world
        hl, kl = self.n_heads // world, self.n_kv_heads // world  # local counts
        dh = self.d_head
        fl = self.d_ff // world

        qs = slice(rank * hl * dh, (rank + 1) * hl * dh)      # q rows
        ks = slice(rank * kl * dh, (rank + 1) * kl * dh)      # k/v rows
        fs = slice(rank * fl, (rank + 1) * fl)                # mlp columns

        for p in self.layers:
            p["wq"], p["bq"] = p["wq"][qs], p["bq"][qs]
            p["wk"], p["bk"] = p["wk"][ks], p["bk"][ks]
            p["wv"], p["bv"] = p["wv"][ks], p["bv"][ks]
            p["wo"] = p["wo"][:, qs]          # row-parallel: input columns
            p["gate"], p["up"] = p["gate"][fs], p["up"][fs]
            p["down"] = p["down"][:, fs]      # row-parallel: input columns

        self.n_heads, self.n_kv_heads, self.d_ff = hl, kl, fl
        self.rep = hl // kl
        # communication meters, reset with reset_comm()
        self.comm_s = 0.0
        self.comm_calls = 0
        self.comm_bytes = 0

    def reset_comm(self):
        self.comm_s, self.comm_calls, self.comm_bytes = 0.0, 0, 0

    def _all_reduce(self, t):
        t0 = time.perf_counter()
        dist.all_reduce(t)
        self.comm_s += time.perf_counter() - t0
        self.comm_calls += 1
        self.comm_bytes += t.numel() * t.element_size()
        return t

    # Same body as batchlib.BatchedRunner._layer with exactly two changes:
    # an all_reduce after the attention output projection and one after the
    # MLP down projection. Those are the only two places where a rank's
    # result is a PARTIAL sum rather than the full answer.
    def _layer(self, i, x, pool, slots, q_pos, kv_len, decode: bool):
        p = self.layers[i]
        b, t, _ = x.shape
        h = rms_norm(x, p["ln1"], self.eps)

        q = (h @ p["wq"].T + p["bq"]).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = (h @ p["wk"].T + p["bk"]).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = (h @ p["wv"].T + p["bv"]).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = apply_rope_rows(q, self.cos, self.sin, q_pos)
        k = apply_rope_rows(k, self.cos, self.sin, q_pos)

        if decode:
            pool.write_rows(i, slots, q_pos[:, 0], k, v)
        else:
            pool.write(i, slots, int(q_pos[0, 0]), k, v)

        upto = int(kv_len.max())
        k_all, v_all = pool.read(i, slots, upto)
        k_all = k_all.repeat_interleave(self.rep, dim=1)
        v_all = v_all.repeat_interleave(self.rep, dim=1)

        import math
        scores = (q @ k_all.transpose(-1, -2)) / math.sqrt(self.d_head)
        cols = torch.arange(upto).view(1, 1, -1)
        mask = (cols > q_pos.unsqueeze(-1)) | (cols >= kv_len.view(-1, 1, 1))
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        o = (w @ v_all).transpose(1, 2).reshape(b, t, self.n_heads * self.d_head)
        # Each rank's o covers only ITS heads, and wo's kept columns match
        # exactly those heads, so `o @ wo.T` is a partial sum of the full
        # projection. The all-reduce completes it on both ranks at once.
        x = x + self._all_reduce(o @ p["wo"].T)

        h = rms_norm(x, p["ln2"], self.eps)
        gate = F.silu(h @ p["gate"].T)
        x = x + self._all_reduce((gate * (h @ p["up"].T)) @ p["down"].T)
        return x

    def param_bytes(self):
        """Per-rank weight bytes, split into sharded vs replicated."""
        shard = rep = 0
        for p in self.layers:
            for k in ("wq", "bq", "wk", "bk", "wv", "bv", "wo", "gate", "up", "down"):
                shard += p[k].numel() * 4
            rep += (p["ln1"].numel() + p["ln2"].numel()) * 4
        rep += (self.embed.numel() + self.norm.numel()) * 4
        if self.lm_head.data_ptr() != self.embed.data_ptr():
            rep += self.lm_head.numel() * 4
        return shard, rep
