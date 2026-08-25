"""A from-scratch Qwen2 forward pass with a *pluggable* KV cache.

Why re-implement a model that `transformers` already runs? Because Phase 2 is
about the cache, and HuggingFace owns its cache. Projects 11-15 need to swap in
a paged cache, a quantized cache, an evicting cache and a disk-backed cache and
still get real text out of a real model. That is only possible if the attention
loop reads from an object *we* define.

So: real Qwen2.5-0.5B-Instruct weights, our own arithmetic, and one small
interface (`KVCache`) that every later project subclasses.

    layer.forward(x):
        q, k, v  = projections(x)          # our code
        k, v     = cache.append(layer, k, v)  # <- the seam
        out      = attention(q, k, v)      # our code

Shared by projects 09, 11, 12, 13, 14 and 15.
"""

from __future__ import annotations

import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_THREADS = 6


# ---------------------------------------------------------------------------
# The cache interface
# ---------------------------------------------------------------------------


class KVCache:
    """What every cache in this phase must be able to do.

    `append` is called once per layer per forward pass. It stores the new keys
    and values and returns *everything* the attention step should look at --
    which is usually "all tokens so far", but an evicting cache (project 14)
    returns fewer, and a paged cache (project 11) gathers them from scattered
    blocks.

    Shapes: k, v are (batch, n_kv_heads, new_tokens, d_head).
    """

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        raise NotImplementedError

    def positions(self, layer: int) -> torch.Tensor | None:
        """Absolute positions of the rows `append` returned, or None if they
        are simply 0, 1, 2, ... (the usual case). Eviction breaks that
        assumption, so the attention mask has to be told."""
        return None

    def reset(self):
        raise NotImplementedError


class ContiguousCache(KVCache):
    """The simplest thing that works: one growing tensor per layer.

    This is what almost every tutorial implements, and what `transformers`
    does by default. It is correct, it is fast, and its memory behaviour is
    the problem project 11 exists to fix.
    """

    def __init__(self, n_layers: int):
        self.n_layers = n_layers
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def append(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]

    def reset(self):
        self.k = [None] * self.n_layers
        self.v = [None] * self.n_layers

    def n_tokens(self) -> int:
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v
                   if t is not None)


class NoCache(KVCache):
    """The control. Stores nothing; the caller must re-feed the whole prefix
    every step, so attention sees the right keys purely because they were
    just recomputed. Used to prove the cache changes speed and not output."""

    def __init__(self, n_layers: int):
        self.n_layers = n_layers

    def append(self, layer, k, v):
        return k, v

    def reset(self):
        pass


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


def rms_norm(x, weight, eps=1e-6):
    var = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(var + eps)).type_as(x) * weight


def rope_tables(d_head: int, max_pos: int, theta: float = 1_000_000.0):
    """Rotary position tables: cos and sin for every position we might see.

    "Rotary" because the position is applied by *rotating* each pair of
    dimensions of q and k by an angle proportional to the token's index. The
    dot product of two rotated vectors then depends on the *difference* of
    their positions, which is exactly what relative-position attention wants.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    t = torch.arange(max_pos).float()
    freqs = torch.outer(t, inv_freq)          # (max_pos, d_head/2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (max_pos, d_head)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin, pos):
    """x: (b, heads, seq, d_head); pos: (seq,) absolute positions."""
    c = cos[pos].unsqueeze(0).unsqueeze(0)
    s = sin[pos].unsqueeze(0).unsqueeze(0)
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * c + rot * s


class Qwen2Runner:
    """Qwen2 decoder, forward pass written out by hand.

    Weights come straight from the HuggingFace checkpoint; only the execution
    is ours. `attn_recorder`, if set, is called with each layer's attention
    weight matrix -- that is how project 14 learns which tokens are 'heavy
    hitters' worth keeping.
    """

    def __init__(self, model, tok):
        self.tok = tok
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.d_head = cfg.hidden_size // cfg.num_attention_heads
        self.d_model = cfg.hidden_size
        self.eps = cfg.rms_norm_eps
        self.rep = self.n_heads // self.n_kv_heads     # GQA repeat factor
        self.eos_id = cfg.eos_token_id

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
        # Qwen2.5-0.5B ties the output head to the input embedding table.
        self.lm_head = sd.get("lm_head.weight", self.embed)

        # transformers 5.x moved rope_theta inside `rope_parameters`.
        theta = getattr(cfg, "rope_parameters", None)
        theta = theta["rope_theta"] if theta else getattr(cfg, "rope_theta", 1e6)
        self.cos, self.sin = rope_tables(self.d_head, 32768, theta)
        self.attn_recorder = None

    # -- one layer -----------------------------------------------------------

    def _layer(self, i, x, cache, pos, kv_pos):
        p = self.layers[i]
        b, t, _ = x.shape
        h = rms_norm(x, p["ln1"], self.eps)

        q = (h @ p["wq"].T + p["bq"]).view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        k = (h @ p["wk"].T + p["bk"]).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = (h @ p["wv"].T + p["bv"]).view(b, t, self.n_kv_heads, self.d_head).transpose(1, 2)

        q = apply_rope(q, self.cos, self.sin, pos)
        k = apply_rope(k, self.cos, self.sin, pos)

        k_all, v_all = cache.append(i, k, v)

        # GQA: 2 kv-heads serve 14 query heads, so each kv head is reused 7x.
        k_all = k_all.repeat_interleave(self.rep, dim=1)
        v_all = v_all.repeat_interleave(self.rep, dim=1)

        scores = (q @ k_all.transpose(-1, -2)) / math.sqrt(self.d_head)
        # Causal mask, written against *absolute* positions so that an
        # evicting cache (whose rows are no longer 0,1,2,...) still works.
        kp = cache.positions(i)
        if kp is None:
            kp = kv_pos
        mask = pos.view(-1, 1) < kp.view(1, -1)
        scores = scores.masked_fill(mask, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        if self.attn_recorder is not None:
            self.attn_recorder(i, w)
        o = (w @ v_all).transpose(1, 2).reshape(b, t, self.d_model)
        x = x + o @ p["wo"].T

        h = rms_norm(x, p["ln2"], self.eps)
        gate = F.silu(h @ p["gate"].T)
        x = x + (gate * (h @ p["up"].T)) @ p["down"].T
        return x

    # -- full forward --------------------------------------------------------

    @torch.inference_mode()
    def forward(self, ids, cache, start_pos=0):
        b, t = ids.shape
        pos = torch.arange(start_pos, start_pos + t)
        kv_pos = torch.arange(0, start_pos + t)
        x = self.embed[ids]
        for i in range(self.n_layers):
            x = self._layer(i, x, cache, pos, kv_pos)
        x = rms_norm(x, self.norm, self.eps)
        return x @ self.lm_head.T

    # -- generation ----------------------------------------------------------

    @torch.inference_mode()
    def generate(self, ids, cache, max_new_tokens=32, stop_on_eos=True,
                 on_step=None):
        """Prefill the prompt, then decode one token at a time."""
        out_ids, step_s = [], []
        t0 = time.perf_counter()
        logits = self.forward(ids, cache, start_pos=0)
        prefill_s = time.perf_counter() - t0
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        out_ids.append(int(nxt))
        pos = ids.shape[1]
        for step in range(1, max_new_tokens):
            if stop_on_eos and int(nxt) == self.eos_id:
                break
            t0 = time.perf_counter()
            logits = self.forward(nxt, cache, start_pos=pos)
            step_s.append(time.perf_counter() - t0)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            out_ids.append(int(nxt))
            pos += 1
            if on_step is not None:
                on_step(step, pos)
        return out_ids, prefill_s, step_s

    @torch.inference_mode()
    def generate_no_cache(self, ids, max_new_tokens=32, stop_on_eos=True):
        """Control: re-run the entire prefix through the model every step."""
        cache = NoCache(self.n_layers)
        out_ids, step_s = [], []
        cur = ids
        prefill_s = 0.0
        for step in range(max_new_tokens):
            t0 = time.perf_counter()
            logits = self.forward(cur, cache, start_pos=0)
            dt = time.perf_counter() - t0
            if step == 0:
                prefill_s = dt
            else:
                step_s.append(dt)
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)
            out_ids.append(int(nxt))
            cur = torch.cat([cur, nxt], dim=1)
            if stop_on_eos and int(nxt) == self.eos_id:
                break
        return out_ids, prefill_s, step_s


# ---------------------------------------------------------------------------
# Helpers shared across the phase
# ---------------------------------------------------------------------------


def load(model_id: str = MODEL_ID, n_threads: int = N_THREADS):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return tok, model


def load_runner(model_id: str = MODEL_ID, n_threads: int = N_THREADS):
    tok, model = load(model_id, n_threads)
    return Qwen2Runner(model, tok), tok, model


def kv_bytes_per_token(n_layers, n_kv_heads, d_head, dtype_bytes=2) -> int:
    """The formula the whole phase runs on: 2 (K and V) x layers x kv-heads
    x head width x bytes per number."""
    return 2 * n_layers * n_kv_heads * d_head * dtype_bytes


def interleaved(fns: dict, rounds: int = 3, warmup: int = 1) -> dict:
    """Time callables round-robin, keep the minimum. This box is shared, so
    running A to completion then B charges any background spike entirely to
    whichever happened to be running."""
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    best = {k: float("inf") for k in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            t0 = time.perf_counter()
            fn()
            best[name] = min(best[name], time.perf_counter() - t0)
    return best


def wikitext_lines(n_chars: int = 200_000) -> str:
    """A real-text corpus for the quality measurements in 13 and 14.

    The `datasets` package is not installed in this environment, so the
    parquet file is pulled straight off the Hub and read with pandas. Same
    bytes, one dependency fewer.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")
    df = pd.read_parquet(path)
    buf = []
    total = 0
    for s in df["text"]:
        if len(s.strip()) < 40:
            continue
        buf.append(s)
        total += len(s)
        if total >= n_chars:
            break
    return "".join(buf)
