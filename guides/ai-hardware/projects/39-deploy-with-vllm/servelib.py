"""A miniature LLM serving engine, written from scratch for AI-hardware Phase 8.

Why from scratch?  vLLM's wheel installs on this machine but its kernels require
compute capability >= 7.0 and the only GPU here is a GTX 1070 Ti (sm_61), so the
real engine cannot start.  Rather than skip the phase, projects 39-44 rebuild the
three ideas that make vLLM fast -- a *paged* KV cache, a *continuous batching*
scheduler, and *speculative decoding* -- on the CPU, where every byte moved is
visible in Python instead of hidden inside a CUDA kernel.

The model code is a hand-written Qwen2 forward pass driven by Hugging Face
weights.  `verify_against_hf()` checks it against the reference implementation.

Shapes and conventions
----------------------
* The KV cache is one big pool per layer, shaped (num_blocks, block_size,
  n_kv_heads, head_dim).  A sequence owns a *list of block ids*; logical position
  `p` of sequence `s` lives at block `s.block_table[p // block_size]`, slot
  `p % block_size`.  This is exactly PagedAttention's mapping, minus the kernel.
* A "batch" here is *ragged*: every sequence may contribute a different number of
  new tokens (a long prefill, or a single decode token).  All the big matrix
  multiplies run on one flat (total_tokens, hidden) tensor, which is what makes
  batching pay: the weights are read from memory once for the whole batch.
"""

import json
import math
import os
import time

import torch
import torch.nn.functional as F

SMALL = "Qwen/Qwen2.5-0.5B-Instruct"   # 494 M params -- the phase workhorse
BIG = "Qwen/Qwen2.5-1.5B-Instruct"     # 1.54 B params -- speculative-decoding target


def setup(threads=12, seed=0):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:
        pass


# --------------------------------------------------------------------- weights
class Weights:
    """Flat, framework-free view of a Qwen2 checkpoint."""

    def __init__(self, name=SMALL):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        cfg = AutoConfig.from_pretrained(name)
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()

        self.n_layer = cfg.num_hidden_layers
        self.n_head = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.d_model = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.eps = cfg.rms_norm_eps
        self.vocab = cfg.vocab_size
        rope = cfg.to_dict()["rope_parameters"]
        self.theta = rope["rope_theta"]

        m = hf.model
        self.embed = m.embed_tokens.weight.data
        self.final_norm = m.norm.weight.data
        self.lm_head = hf.lm_head.weight.data
        self.layers = []
        for blk in m.layers:
            a, mlp = blk.self_attn, blk.mlp
            self.layers.append(dict(
                ln1=blk.input_layernorm.weight.data,
                ln2=blk.post_attention_layernorm.weight.data,
                wq=a.q_proj.weight.data, bq=a.q_proj.bias.data,
                wk=a.k_proj.weight.data, bk=a.k_proj.bias.data,
                wv=a.v_proj.weight.data, bv=a.v_proj.bias.data,
                wo=a.o_proj.weight.data,
                wgate=mlp.gate_proj.weight.data,
                wup=mlp.up_proj.weight.data,
                wdown=mlp.down_proj.weight.data,
            ))
        del hf

        # RoPE tables. "Rotary" = each pair of channels is rotated by an angle
        # proportional to the token's position, so a dot product between two
        # tokens only depends on their *distance*.
        inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self._inv_freq = inv
        self._rope_len = 0
        self._grow_rope(4096)

    def _grow_rope(self, n):
        if n <= self._rope_len:
            return
        n = max(n, 2 * self._rope_len, 512)
        t = torch.arange(n).float()
        f = torch.outer(t, self._inv_freq)
        emb = torch.cat([f, f], dim=-1)
        self.cos, self.sin = emb.cos(), emb.sin()
        self._rope_len = n

    def bytes_of_weights(self):
        n = self.embed.numel() + self.final_norm.numel()
        if self.lm_head.data_ptr() != self.embed.data_ptr():
            n += self.lm_head.numel()
        for L in self.layers:
            n += sum(t.numel() for t in L.values())
        return n * 4  # fp32


def rms_norm(x, w, eps):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps) * w


def rope(x, cos, sin):
    """x: (T, H, D); cos/sin: (T, D)."""
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    d = x.shape[-1] // 2
    rot = torch.cat([-x[..., d:], x[..., :d]], dim=-1)
    return x * c + rot * s


# ------------------------------------------------------------------ paged pool
class KVPool:
    """A pool of fixed-size KV blocks, handed out and returned like memory pages.

    block_size is measured in *tokens*.  vLLM's default is 16; the same number
    here keeps the arithmetic comparable.
    """

    def __init__(self, w, num_blocks, block_size=16, dtype=torch.float32):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.k = [torch.zeros(num_blocks, block_size, w.n_kv, w.head_dim, dtype=dtype)
                  for _ in range(w.n_layer)]
        self.v = [torch.zeros(num_blocks, block_size, w.n_kv, w.head_dim, dtype=dtype)
                  for _ in range(w.n_layer)]
        self.free = list(range(num_blocks))
        self.peak_used = 0

    @property
    def bytes_total(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

    def bytes_per_token(self):
        return self.bytes_total / (self.num_blocks * self.block_size)

    def allocate(self, n):
        if n > len(self.free):
            return None
        out = [self.free.pop() for _ in range(n)]
        self.peak_used = max(self.peak_used, self.num_blocks - len(self.free))
        return out

    def release(self, blocks):
        self.free.extend(blocks)

    @property
    def used(self):
        return self.num_blocks - len(self.free)


class Sequence:
    """One request in flight."""

    __slots__ = ("sid", "prompt_ids", "out_ids", "blocks", "length", "done",
                 "arrival", "first_token_t", "finish_t", "max_new", "prefill_done")

    def __init__(self, sid, prompt_ids, max_new=32, arrival=0.0):
        self.sid = sid
        self.prompt_ids = list(prompt_ids)
        self.out_ids = []
        self.blocks = []
        self.length = 0          # tokens currently *in the cache*
        self.done = False
        self.arrival = arrival
        self.first_token_t = None
        self.finish_t = None
        self.max_new = max_new
        self.prefill_done = False

    @property
    def all_ids(self):
        return self.prompt_ids + self.out_ids


# ---------------------------------------------------------------------- engine
class Engine:
    """Runs ragged batches of sequences against a paged KV cache."""

    def __init__(self, weights, num_blocks=512, block_size=16, gather_stats=False):
        self.w = weights
        self.pool = KVPool(weights, num_blocks, block_size)
        self.block_size = block_size
        self.gather_stats = gather_stats
        self.gather_bytes = 0
        self.gather_time = 0.0

    # --- block bookkeeping -------------------------------------------------
    def ensure_blocks(self, seq, extra):
        need = math.ceil((seq.length + extra) / self.block_size) - len(seq.blocks)
        if need <= 0:
            return True
        got = self.pool.allocate(need)
        if got is None:
            return False
        seq.blocks.extend(got)
        return True

    def free(self, seq):
        self.pool.release(seq.blocks)
        seq.blocks = []

    def _slots(self, seqs, starts, qlens):
        """Physical slot index of every new token: block_id * block_size + offset.

        Computed once per step and reused by all layers -- the block table is a
        property of the *sequence*, not of the layer.
        """
        bs = self.block_size
        out = []
        for s, st, q in zip(seqs, starts, qlens):
            for p in range(st, st + q):
                out.append(s.blocks[p // bs] * bs + p % bs)
        return torch.tensor(out, dtype=torch.long)

    def _table(self, seqs):
        nb = max(len(s.blocks) for s in seqs)
        table = torch.zeros(len(seqs), nb, dtype=torch.long)
        for i, s in enumerate(seqs):
            table[i, :len(s.blocks)] = torch.tensor(s.blocks, dtype=torch.long)
        return table

    def _gather_kv(self, layer, table):
        """Materialise (B, n_kv, max_len, D) from the block table.

        A real PagedAttention kernel skips this copy: it walks the block table
        *inside* the attention kernel and reads each block straight from HBM.  We
        have no such kernel, so we pay a gather and measure it -- that
        measurement is exactly the price of not having the kernel.
        """
        bs, B, nb = self.block_size, table.shape[0], table.shape[1]
        t0 = time.perf_counter() if self.gather_stats else 0.0
        k = self.pool.k[layer][table]          # (B, nb, bs, n_kv, D)
        v = self.pool.v[layer][table]
        k = k.reshape(B, nb * bs, self.w.n_kv, self.w.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, nb * bs, self.w.n_kv, self.w.head_dim).permute(0, 2, 1, 3)
        if self.gather_stats:
            self.gather_time += time.perf_counter() - t0
            self.gather_bytes += 2 * B * nb * bs * self.w.n_kv * self.w.head_dim * 4
        return k, v

    # --- the forward pass --------------------------------------------------
    @torch.no_grad()
    def forward(self, seqs, token_lists, last_only=True):
        """seqs: list[Sequence]; token_lists: list[list[int]] of NEW tokens.

        Returns logits for the last token of every sequence (last_only) or for
        every token fed in.
        """
        w = self.w
        starts = [s.length for s in seqs]
        qlens = [len(t) for t in token_lists]
        for s, q in zip(seqs, qlens):
            if not self.ensure_blocks(s, q):
                raise MemoryError(f"KV pool exhausted: {self.pool.used}/{self.pool.num_blocks} blocks")
        flat = torch.tensor([t for lst in token_lists for t in lst], dtype=torch.long)
        pos = torch.tensor([p for st, ql in zip(starts, qlens) for p in range(st, st + ql)])
        w._grow_rope(int(pos.max()) + 1)
        cos, sin = w.cos[pos], w.sin[pos]

        x = w.embed[flat]                                   # (T, d)
        T = x.shape[0]
        offs = [0]
        for q in qlens:
            offs.append(offs[-1] + q)
        lens = [st + q for st, q in zip(starts, qlens)]
        B, maxq = len(seqs), max(qlens)
        uniform = min(qlens) == maxq
        slots = self._slots(seqs, starts, qlens)
        table = self._table(seqs)
        nslot = table.shape[1] * self.block_size

        # mask[b, i, j]: may query i of sequence b attend to cached slot j?
        # Causal *and* bounded by what this sequence actually owns: slots past
        # its own length hold another request's data, or stale bytes.
        qpos = torch.tensor(starts).unsqueeze(1) + torch.arange(maxq)     # (B, maxq)
        mask = torch.arange(nslot).view(1, 1, -1) <= qpos.unsqueeze(-1)
        if not uniform:
            valid = torch.arange(maxq).view(1, -1) < torch.tensor(qlens).view(-1, 1)
            mask &= valid.unsqueeze(-1)
            # A row that is masked everywhere makes softmax divide by zero (NaN).
            # These rows are padding and get sliced off, but let one slot through
            # so the NaN never exists in the first place.
            mask[..., 0] |= ~valid
        mask = mask.unsqueeze(1)
        for li, L in enumerate(w.layers):
            h = rms_norm(x, L["ln1"], w.eps)
            q = (h @ L["wq"].T + L["bq"]).view(T, w.n_head, w.head_dim)
            k = (h @ L["wk"].T + L["bk"]).view(T, w.n_kv, w.head_dim)
            v = (h @ L["wv"].T + L["bv"]).view(T, w.n_kv, w.head_dim)
            q = rope(q, cos, sin)
            k = rope(k, cos, sin)
            kf = self.pool.k[li].view(-1, w.n_kv, w.head_dim)
            vf = self.pool.v[li].view(-1, w.n_kv, w.head_dim)
            kf.index_copy_(0, slots, k)
            vf.index_copy_(0, slots, v)
            K, V = self._gather_kv(li, table)
            if uniform:
                qb = q.view(B, maxq, w.n_head, w.head_dim).permute(0, 2, 1, 3)
            else:
                qb = torch.zeros(B, w.n_head, maxq, w.head_dim)
                for b in range(B):
                    qb[b, :, :qlens[b]] = q[offs[b]:offs[b + 1]].permute(1, 0, 2)
            o = F.scaled_dot_product_attention(qb, K, V, attn_mask=mask, enable_gqa=True)
            if uniform:
                oflat = o.permute(0, 2, 1, 3).reshape(T, -1)
            else:
                oflat = torch.cat([o[b, :, :qlens[b]].permute(1, 0, 2).reshape(qlens[b], -1)
                                   for b in range(B)], dim=0)
            x = x + oflat @ L["wo"].T
            h = rms_norm(x, L["ln2"], w.eps)
            gate = h @ L["wgate"].T
            up = h @ L["wup"].T
            x = x + (F.silu(gate) * up) @ L["wdown"].T

        for b, s in enumerate(seqs):
            s.length = lens[b]
        if last_only:
            idx = torch.tensor([offs[b + 1] - 1 for b in range(len(seqs))])
            x = x[idx]
        x = rms_norm(x, w.final_norm, w.eps)
        return x @ w.lm_head.T

    # --- convenience -------------------------------------------------------
    def prefill(self, seq):
        logits = self.forward([seq], [seq.prompt_ids])
        seq.prefill_done = True
        return logits[0]

    def decode_step(self, seqs):
        toks = [[s.all_ids[-1]] for s in seqs]
        return self.forward(seqs, toks)


def greedy(logits):
    return int(torch.argmax(logits, dim=-1))


def synthetic_seqs(engine, batch, context, token=1000):
    """Sequences whose caches are *allocated* but never filled by a real prefill.

    Decode-step timing depends on the shapes of the tensors involved, not on the
    numbers inside them, so for pure timing experiments this skips the expensive
    prefill and lets us reach 2048-token contexts in seconds instead of minutes.
    Never use these for quality measurements -- the cache holds zeros.
    """
    seqs = []
    for i in range(batch):
        s = Sequence(i, [token], max_new=1)
        engine.ensure_blocks(s, context)
        s.length = context
        s.out_ids = [token]
        seqs.append(s)
    return seqs


def time_decode(engine, seqs, rounds=3, warmup=1):
    """Median-of-rounds decode step time, warm-up excluded."""
    for _ in range(warmup):
        engine.decode_step(seqs)
        for s in seqs:
            s.length -= 1          # undo, so the context length stays fixed
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        engine.decode_step(seqs)
        ts.append(time.perf_counter() - t0)
        for s in seqs:
            s.length -= 1
    return sorted(ts)[len(ts) // 2]


def measure_peaks(gb=2.0, mnk=1024, rounds=3):
    """The two roofs a decoder can hit on this machine.

    * read bandwidth: how fast a big array can be streamed out of DRAM
    * matmul rate: how fast fp32 GEMM runs once the data is in cache
    Take the best of several rounds -- a shared machine can only slow you down.
    """
    n = int(gb * 1e9 / 4)
    x = torch.empty(n, dtype=torch.float32).uniform_(-1, 1)
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        x.sum()
        best = min(best, time.perf_counter() - t0)
    bw = n * 4 / best / 1e9
    a = torch.randn(mnk, mnk)
    b = torch.randn(mnk, mnk)
    a @ b
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        a @ b
        best = min(best, time.perf_counter() - t0)
    flops = 2 * mnk ** 3 / best / 1e9
    del x, a, b
    return dict(read_GB_s=bw, matmul_GFLOP_s=flops)


# ------------------------------------------------------------- verification
def verify_against_hf(name=SMALL, prompt="The capital of France is", n_new=8):
    """Compare the hand-written engine with Hugging Face's own forward pass."""
    from transformers import AutoModelForCausalLM
    w = Weights(name)
    eng = Engine(w, num_blocks=64)
    ids = w.tok(prompt, return_tensors=None)["input_ids"]
    seq = Sequence(0, ids, max_new=n_new)
    mine = eng.prefill(seq)
    outs = [greedy(mine)]
    seq.out_ids.append(outs[-1])
    for _ in range(n_new - 1):
        lg = eng.decode_step([seq])[0]
        outs.append(greedy(lg))
        seq.out_ids.append(outs[-1])

    # Reference: plain greedy decoding with Hugging Face's own module stack.
    # (We do *not* call .generate(); Qwen ships a generation_config with
    # repetition_penalty=1.05, which silently applies even when do_sample=False
    # and would make the two runs diverge for reasons unrelated to this engine.)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    cur = torch.tensor([ids])
    gen, ref = [], None
    with torch.no_grad():
        for i in range(n_new):
            lg = hf(cur).logits[0, -1]
            if i == 0:
                ref = lg
            nxt = int(lg.argmax())
            gen.append(nxt)
            cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)
    return dict(
        max_abs_logit_diff=float((mine - ref).abs().max()),
        rel_logit_diff=float((mine - ref).norm() / ref.norm()),
        my_tokens=outs, hf_tokens=gen, tokens_match=outs == gen,
        my_text=w.tok.decode(outs), hf_text=w.tok.decode(gen),
    )


# -------------------------------------------------------------------- outputs
def outdir(project_file):
    d = os.path.join(os.path.dirname(os.path.abspath(project_file)), "outputs")
    os.makedirs(d, exist_ok=True)
    return d


def save_findings(project_file, obj):
    p = os.path.join(outdir(project_file), "findings.json")
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return p


def load_findings(project_file):
    with open(os.path.join(outdir(project_file), "findings.json")) as f:
        return json.load(f)


def prompt_ids(tok, text, length=None):
    ids = tok(text, return_tensors=None)["input_ids"]
    if length is not None:
        while len(ids) < length:
            ids = ids + ids
        ids = ids[:length]
    return ids
