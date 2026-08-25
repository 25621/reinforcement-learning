"""Speculative decoding inside a continuous batch.

Phase 3's `16/batchlib.py` can run a heterogeneous batch: every row at its own
position, one new token per row per step. Speculation breaks that last
assumption -- a verification step carries `k+1` query tokens *per row*, and
after verification each row has advanced by a **different** number of tokens.

Two things have to change, and only two:

  1. `SpecSlotKV.write_block` -- write `T` tokens per row, each at that row's
     own position. `batchlib.SlotKV.write_rows` writes exactly one.
  2. `SpecBatchedRunner._layer` -- call it. Everything else in the layer,
     including the two-condition mask (causal AND ownership), already works
     for a multi-token query block, because Phase 3 built it from per-row
     positions rather than one shared sequence length.

Rollback needs no code at all: a slot's length is a number in the scheduler,
so un-accepting three tokens is `lengths[i] -= 3`. The stale keys and values
stay in the pool and are masked out by the ownership condition, exactly as a
production paged engine leaves rejected blocks allocated.
"""

from __future__ import annotations

import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "16-static-vs-continuous"))

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

import batchlib  # noqa: E402


class SpecSlotKV(batchlib.SlotKV):
    """A slot pool that can absorb a whole speculative block per row."""

    def write_block(self, layer, slots, positions, k, v):
        """positions: (B, T) -- row i's token j lands at positions[i, j].

        `write_rows` in Phase 3 takes a (B,) vector because a decode step
        advances every row by exactly one. Here row 3 might be writing
        positions 41..44 while row 4 writes 300..303, so the index has to be
        two-dimensional. Advanced indexing puts the indexed dimensions first,
        which is why k and v are permuted to (B, T, heads, d).
        """
        idx = slots.view(-1, 1)                       # (B, 1) broadcasts over T
        self.k[layer][idx, :, positions, :] = k.permute(0, 2, 1, 3)
        self.v[layer][idx, :, positions, :] = v.permute(0, 2, 1, 3)


class SpecBatchedRunner(batchlib.BatchedRunner):
    """`batchlib.BatchedRunner` plus a multi-token-per-row decode step."""

    # This is `batchlib.BatchedRunner._layer` with exactly one line changed --
    # the KV write. It is spelled out rather than parameterised so that the
    # one difference between "batched decoding" and "batched speculation" is
    # visible in one place.
    def _layer_spec(self, i, x, pool, slots, q_pos, kv_len):
        p = self.layers[i]
        b, t, _ = x.shape
        h = batchlib.rms_norm(x, p["ln1"], self.eps)

        q = (h @ p["wq"].T + p["bq"]).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = (h @ p["wk"].T + p["bk"]).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = (h @ p["wv"].T + p["bv"]).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = batchlib.apply_rope_rows(q, self.cos, self.sin, q_pos)
        k = batchlib.apply_rope_rows(k, self.cos, self.sin, q_pos)

        pool.write_block(i, slots, q_pos, k, v)       # <-- the one change

        upto = int(kv_len.max())
        k_all, v_all = pool.read(i, slots, upto)
        k_all = k_all.repeat_interleave(self.rep, dim=1)
        v_all = v_all.repeat_interleave(self.rep, dim=1)

        scores = (q @ k_all.transpose(-1, -2)) / math.sqrt(self.d_head)
        cols = torch.arange(upto).view(1, 1, -1)
        mask = (cols > q_pos.unsqueeze(-1)) | (cols >= kv_len.view(-1, 1, 1))
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        o = (w @ v_all).transpose(1, 2).reshape(b, t, self.d_model)
        x = x + o @ p["wo"].T

        h = batchlib.rms_norm(x, p["ln2"], self.eps)
        gate = Fn.silu(h @ p["gate"].T)
        x = x + (gate * (h @ p["up"].T)) @ p["down"].T
        return x

    @torch.inference_mode()
    def verify_step(self, pool, slots, block, lengths, count=True):
        """One forward pass carrying `T` query tokens for each of B rows.

        `block[i]` is `[row i's last token] + [row i's k drafts]`.
        `lengths[i]` is how many tokens row i already owns; the block lands at
        positions lengths[i]-1 .. lengths[i]+k-1, so the *first* column is a
        re-run of a token already in the cache. That is deliberate: it is what
        makes the pass return a prediction for the position right after the
        last accepted token, i.e. the free bonus token.

        Returns logits of shape (B, T, vocab).
        """
        slots_t = torch.as_tensor(slots)
        lens = torch.as_tensor(lengths)
        b, t = block.shape
        q_pos = (lens - 1).view(-1, 1) + torch.arange(t).view(1, -1)
        kv_len = lens - 1 + t
        t0 = time.perf_counter()
        x = self.embed[block]
        for i in range(self.n_layers):
            x = self._layer_spec(i, x, pool, slots_t, q_pos, kv_len)
        x = batchlib.rms_norm(x, self.norm, self.eps)
        logits = x @ self.lm_head.T
        dt = time.perf_counter() - t0
        if count:
            useful = b * t
            self.counters.add(useful, 0,
                              sum(self.flops_per_token(int(kv_len[i]))
                                  for i in range(b)) * t, 0.0, dt)
        return logits, dt


# ---------------------------------------------------------------------------
# drafting inside a batch
# ---------------------------------------------------------------------------


def ngram_propose(tokens, k, max_n=4, min_n=2):
    """Prompt lookup for one row. Returns 0..k token ids.

    A batch needs the *same* number of query columns for every row, so a row
    with nothing to propose has to be padded -- see `pad_drafts`. That padding
    is the specific waste speculation adds to batching, and section B counts
    it.
    """
    for n in range(max_n, min_n - 1, -1):
        if len(tokens) <= n:
            continue
        pat = tokens[-n:]
        for i in range(len(tokens) - n - 1, -1, -1):
            if tokens[i:i + n] == pat:
                out = tokens[i + n:i + n + k]
                if out:
                    return out
    return []


def pad_drafts(drafts, k, pad_id=0):
    """Make every row's proposal exactly k long. Padded columns are verified
    like any other and are guaranteed to be rejected, because verification
    stops at the first mismatch and a pad token never matches -- so they cost
    time and can never produce a token."""
    return drafts + [pad_id] * (k - len(drafts)), len(drafts)
