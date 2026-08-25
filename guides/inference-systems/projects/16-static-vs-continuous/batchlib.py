"""A *batched* Qwen2 engine with a slot-based KV cache -- Phase 3's shared toy.

Phase 2 (`09/kvlib.py`) ran one sequence at a time, because the question there
was "what does the cache hold?". Phase 3's question is different: *many*
requests are in flight at once, at different positions in their generation, and
the scheduler decides which of them share each forward pass. So this file adds
the two things a batch needs and a single sequence never does:

  1. **Slots.** A fixed pool of KV space, `n_slots` deep. A request occupies one
     slot from admission to completion. The pool is a real tensor, so "the
     cache is full" is a real condition, not a simulated one.
  2. **Per-row positions.** Row 3 of the batch may be at token 12 while row 4 is
     at token 300. Every mask in here is built from a per-row length, never
     from a single shared `seq_len`.

Everything is instrumented for *waste*: `Counters` records, for every forward
pass, how many token-slots were real and how many were padding, in FLOPs as
well as in count. Project 17 is entirely a reading of those counters.

Shared by projects 16, 17, 19, 20 and 21.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_THREADS = 6


# ---------------------------------------------------------------------------
# The KV slot pool
# ---------------------------------------------------------------------------


class SlotKV:
    """A pool of `n_slots` fixed-size KV lanes.

    This is deliberately *not* the paged cache from project 11. A scheduler
    study wants the simplest possible memory model -- one lane per request,
    lane length `max_len` -- so that "the cache is full" means exactly "all
    lanes are taken". Project 21 is where the difference starts to matter, and
    it says so there.

    Layout per layer: (n_slots, n_kv_heads, max_len, d_head).
    """

    def __init__(self, n_layers, n_slots, n_kv_heads, d_head, max_len,
                 dtype=torch.float32):
        self.n_layers = n_layers
        self.n_slots = n_slots
        self.max_len = max_len
        self.k = [torch.zeros(n_slots, n_kv_heads, max_len, d_head, dtype=dtype)
                  for _ in range(n_layers)]
        self.v = [torch.zeros(n_slots, n_kv_heads, max_len, d_head, dtype=dtype)
                  for _ in range(n_layers)]
        self.free = list(range(n_slots))

    def acquire(self):
        """Take a lane, or return None if the cache is full."""
        return self.free.pop(0) if self.free else None

    def release(self, slot: int):
        self.free.append(slot)

    def n_free(self) -> int:
        return len(self.free)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def write(self, layer, slots: torch.Tensor, start: int, k, v):
        """Write `k`,`v` (B, kvh, T, d) into rows `slots` at positions
        start..start+T-1. All rows are written at the same offset, which is
        true for a prefill batch and for a decode step where the engine has
        already grouped rows by position -- for the general decode case the
        caller uses `write_rows`."""
        t = k.shape[2]
        self.k[layer][slots, :, start:start + t, :] = k
        self.v[layer][slots, :, start:start + t, :] = v

    def write_rows(self, layer, slots: torch.Tensor, positions: torch.Tensor,
                   k, v):
        """Decode-step write: row i goes to slot slots[i] at its *own*
        position positions[i]. This is the operation a heterogeneous batch
        needs and a homogeneous one never does."""
        self.k[layer][slots, :, positions, :] = k[:, :, 0, :]
        self.v[layer][slots, :, positions, :] = v[:, :, 0, :]

    def read(self, layer, slots: torch.Tensor, upto: int):
        """Gather rows `slots`, columns 0..upto-1."""
        return (self.k[layer][slots, :, :upto, :],
                self.v[layer][slots, :, :upto, :])


# ---------------------------------------------------------------------------
# Waste counters
# ---------------------------------------------------------------------------


@dataclass
class Counters:
    """What every forward pass cost, split into useful and wasted.

    `*_slots` count token-positions the batch carried. `*_flops` weight those
    positions by the arithmetic each one actually triggers, which is not
    proportional to the count -- a token at context 300 costs more attention
    than one at context 12. Reporting only counts would understate the waste
    of padding a short sequence into a long batch.
    """

    useful_slots: int = 0
    pad_slots: int = 0
    useful_flops: float = 0.0
    pad_flops: float = 0.0
    forward_passes: int = 0
    model_s: float = 0.0

    def add(self, useful_slots, pad_slots, useful_flops, pad_flops, dt):
        self.useful_slots += useful_slots
        self.pad_slots += pad_slots
        self.useful_flops += useful_flops
        self.pad_flops += pad_flops
        self.forward_passes += 1
        self.model_s += dt

    @property
    def pad_slot_frac(self):
        tot = self.useful_slots + self.pad_slots
        return self.pad_slots / tot if tot else 0.0

    @property
    def pad_flop_frac(self):
        tot = self.useful_flops + self.pad_flops
        return self.pad_flops / tot if tot else 0.0


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def rms_norm(x, weight, eps=1e-6):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).type_as(x) * weight


def rope_tables(d_head: int, max_pos: int, theta: float = 1_000_000.0):
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope_rows(x, cos, sin, pos):
    """x: (b, heads, t, d_head); pos: (b, t) absolute position of every row's
    every token. Per-row positions are the whole point: in a continuous batch
    two rows at the same index of the same forward pass are at different
    absolute positions in their own sequences."""
    c = cos[pos].unsqueeze(1)          # (b, 1, t, d)
    s = sin[pos].unsqueeze(1)
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * c + rot * s


class BatchedRunner:
    """Qwen2 forward pass, written out by hand, over a heterogeneous batch."""

    def __init__(self, model, tok):
        self.tok = tok
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.d_head = cfg.hidden_size // cfg.num_attention_heads
        self.d_model = cfg.hidden_size
        self.d_ff = cfg.intermediate_size
        self.eps = cfg.rms_norm_eps
        self.rep = self.n_heads // self.n_kv_heads
        self.eos_id = cfg.eos_token_id
        self.vocab = cfg.vocab_size

        sd = model.state_dict()
        g = lambda n: sd[n].detach()                    # noqa: E731
        self.embed = g("model.embed_tokens.weight")
        self.layers = []
        for i in range(self.n_layers):
            p = f"model.layers.{i}."
            self.layers.append({
                "ln1": g(p + "input_layernorm.weight"),
                "ln2": g(p + "post_attention_layernorm.weight"),
                "wq": g(p + "self_attn.q_proj.weight"), "bq": g(p + "self_attn.q_proj.bias"),
                "wk": g(p + "self_attn.k_proj.weight"), "bk": g(p + "self_attn.k_proj.bias"),
                "wv": g(p + "self_attn.v_proj.weight"), "bv": g(p + "self_attn.v_proj.bias"),
                "wo": g(p + "self_attn.o_proj.weight"),
                "gate": g(p + "mlp.gate_proj.weight"),
                "up": g(p + "mlp.up_proj.weight"),
                "down": g(p + "mlp.down_proj.weight"),
            })
        self.norm = g("model.norm.weight")
        self.lm_head = sd.get("lm_head.weight", self.embed)

        theta = getattr(cfg, "rope_parameters", None)
        theta = theta["rope_theta"] if theta else getattr(cfg, "rope_theta", 1e6)
        self.cos, self.sin = rope_tables(self.d_head, 32768, theta)
        self.counters = Counters()

    # -- arithmetic model ----------------------------------------------------

    def flops_per_token(self, ctx: int) -> float:
        """FLOPs one token position costs in one forward pass, at context
        length `ctx` (how many keys it attends to). Multiply-add counted as 2.

        Split out because the padding audit needs to price a *wasted* position,
        and a wasted position at context 300 is not the same size as one at
        context 12."""
        d, dh, H, KV = self.d_model, self.d_head, self.n_heads, self.n_kv_heads
        qkv = 2 * d * (H * dh + 2 * KV * dh)
        attn = 2 * 2 * H * dh * ctx           # scores + weighted sum of V
        proj = 2 * d * H * dh
        mlp = 2 * 3 * d * self.d_ff
        return self.n_layers * (qkv + attn + proj + mlp)

    def head_flops(self) -> float:
        return 2 * self.d_model * self.vocab

    # -- one layer -----------------------------------------------------------

    def _layer(self, i, x, pool, slots, q_pos, kv_len, decode: bool):
        """q_pos: (b, t) absolute positions of the queries.
        kv_len: (b,) how many keys each row legitimately owns *after* this
        write. Columns past that are other requests' garbage or not yet
        written, and the mask must drop them."""
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

        scores = (q @ k_all.transpose(-1, -2)) / math.sqrt(self.d_head)
        cols = torch.arange(upto).view(1, 1, -1)
        # Two masks at once: causal (a query cannot see the future) and
        # ownership (a row cannot see columns it does not own -- padding, or
        # a slot's stale bytes from a previous request).
        mask = (cols > q_pos.unsqueeze(-1)) | (cols >= kv_len.view(-1, 1, 1))
        scores = scores.masked_fill(mask.unsqueeze(1), float("-inf"))
        w = torch.softmax(scores, dim=-1)
        o = (w @ v_all).transpose(1, 2).reshape(b, t, self.d_model)
        x = x + o @ p["wo"].T

        h = rms_norm(x, p["ln2"], self.eps)
        gate = F.silu(h @ p["gate"].T)
        x = x + (gate * (h @ p["up"].T)) @ p["down"].T
        return x

    @torch.inference_mode()
    def _forward(self, ids, pool, slots, q_pos, kv_len, decode):
        x = self.embed[ids]
        for i in range(self.n_layers):
            x = self._layer(i, x, pool, slots, q_pos, kv_len, decode)
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T

    # -- the two public operations -------------------------------------------

    @torch.inference_mode()
    def prefill(self, pool, slots, ids, prompt_lens, start=0, count=True):
        """One prefill forward pass over a (possibly padded) batch of prompts.

        `ids` is (B, T) right-padded; `prompt_lens[i]` says how much of row i
        is real. Right-padding is safe here because a causal mask already stops
        real tokens from reading later columns, so the junk rows can never
        contaminate a real one -- they only cost time.

        Returns logits at each row's *last real* position: (B, vocab).
        """
        b, t = ids.shape
        slots_t = torch.as_tensor(slots)
        lens = torch.as_tensor(prompt_lens)
        q_pos = torch.arange(start, start + t).view(1, -1).expand(b, -1)
        kv_len = lens.clamp(max=start + t)
        t0 = time.perf_counter()
        logits = self._forward(ids, pool, slots_t, q_pos, kv_len, decode=False)
        dt = time.perf_counter() - t0
        if count:
            useful = int(lens.clamp(max=start + t).sub(start).clamp(min=0).sum())
            self.counters.add(
                useful, b * t - useful,
                sum(self.flops_per_token(start + j + 1)
                    for i in range(b)
                    for j in range(t) if start + j < int(lens[i])),
                sum(self.flops_per_token(start + j + 1)
                    for i in range(b)
                    for j in range(t) if start + j >= int(lens[i])),
                dt)
        last = (lens - 1 - start).clamp(min=0, max=t - 1)
        return logits[torch.arange(b), last], dt

    @torch.inference_mode()
    def decode_step(self, pool, slots, tokens, lengths, live=None, count=True):
        """One decode step over B rows at B different positions.

        `lengths[i]` is how many tokens row i already has; the new token lands
        at position lengths[i]. `live[i]` is False for a row that has already
        finished but is still occupying the batch -- static batching's
        signature waste, which we have to be able to represent to measure.
        """
        slots_t = torch.as_tensor(slots)
        lens = torch.as_tensor(lengths)
        b = len(slots)
        q_pos = lens.view(-1, 1)
        kv_len = lens + 1
        ids = torch.as_tensor(tokens).view(b, 1)
        t0 = time.perf_counter()
        logits = self._forward(ids, pool, slots_t, q_pos, kv_len, decode=True)
        dt = time.perf_counter() - t0
        if count:
            live = [True] * b if live is None else live
            uf = sum(self.flops_per_token(int(lens[i]) + 1)
                     for i in range(b) if live[i])
            pf = sum(self.flops_per_token(int(lens[i]) + 1)
                     for i in range(b) if not live[i])
            self.counters.add(sum(live), b - sum(live), uf, pf, dt)
        return logits[:, 0, :], dt


# ---------------------------------------------------------------------------
# Requests and workloads
# ---------------------------------------------------------------------------


@dataclass
class Request:
    rid: int
    arrive: float                 # seconds after t=0
    prompt_ids: list
    max_new: int
    priority: int = 0             # project 20
    deadline: float = 0.0         # project 22
    # filled in as it runs
    slot: int | None = None
    tokens: list = field(default_factory=list)
    admit_t: float | None = None
    first_tok_t: float | None = None
    end_t: float | None = None
    token_times: list = field(default_factory=list)

    @property
    def prompt_len(self):
        return len(self.prompt_ids)

    @property
    def ttft(self):
        return None if self.first_tok_t is None else self.first_tok_t - self.arrive

    @property
    def e2e(self):
        return None if self.end_t is None else self.end_t - self.arrive

    def itls(self):
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]


def make_workload(tok, n=48, rate=6.0, seed=0, p_med=64, p_sigma=0.75,
                  o_med=32, o_sigma=0.7, p_max=192, o_max=72):
    """Poisson arrivals, lognormal lengths -- the standard traffic model.

    *Poisson* (after Simeon Poisson) means "events at some average rate, with
    no coordination between them", which is what independent users are. The
    gap between two arrivals is then exponentially distributed, which is why
    the generator below draws `expovariate`.

    *Lognormal* means the **logarithm** of the length is normally distributed.
    That produces the shape real chat traffic has: most requests short, a long
    thin tail of pasted documents, and never a negative length.
    """
    rng = random.Random(seed)
    vocab_sample = [t for t in range(1000, 12000)]
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        plen = int(min(p_max, max(8, rng.lognormvariate(math.log(p_med), p_sigma))))
        olen = int(min(o_max, max(4, rng.lognormvariate(math.log(o_med), o_sigma))))
        ids = [rng.choice(vocab_sample) for _ in range(plen)]
        reqs.append(Request(rid=i, arrive=t, prompt_ids=ids, max_new=olen))
    return reqs


def load_runner(model_id: str = MODEL_ID, n_threads: int = N_THREADS):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return BatchedRunner(model, tok), tok


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[i]


def summarize(reqs, wall_s, label=""):
    done = [r for r in reqs if r.end_t is not None]
    ttfts = [r.ttft for r in done if r.ttft is not None]
    itls = [x for r in done for x in r.itls()]
    out_tokens = sum(len(r.tokens) for r in done)
    return {
        "label": label,
        "requests_done": len(done),
        "wall_s": round(wall_s, 3),
        "out_tokens": out_tokens,
        "throughput_tok_s": round(out_tokens / wall_s, 2) if wall_s else 0.0,
        "ttft_p50": round(pct(ttfts, 50), 3),
        "ttft_p99": round(pct(ttfts, 99), 3),
        "itl_p50": round(pct(itls, 50), 4),
        "itl_p99": round(pct(itls, 99), 4),
        "e2e_p50": round(pct([r.e2e for r in done], 50), 3),
        "e2e_p99": round(pct([r.e2e for r in done], 99), 3),
    }
