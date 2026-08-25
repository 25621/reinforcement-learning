"""Shared inference primitives for Phase 1 of the inference-systems guide.

Everything here is deliberately small and readable: a model loader, a manual
prefill/decode loop with a KV cache, a no-cache control loop, and a timing
helper that interleaves rounds (this machine is shared, so back-to-back
timing of A then B is not a fair comparison).

Imported by projects 01, 02, 06 and 07.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass, field

# Keep BLAS threading under control *before* torch is imported.
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_THREADS = 6


def load(model_id: str = MODEL_ID, n_threads: int = N_THREADS):
    """Load tokenizer + model on CPU in float32 and put it in eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return tok, model


def model_shape(model) -> dict:
    """The handful of config numbers the KV-cache and roofline math needs."""
    c = model.config
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model_id": c.name_or_path,
        "n_params": n_params,
        "weight_bytes_fp32": n_params * 4,
        "n_layers": c.num_hidden_layers,
        "n_heads": c.num_attention_heads,
        "n_kv_heads": c.num_key_value_heads,
        "d_model": c.hidden_size,
        "d_head": c.hidden_size // c.num_attention_heads,
        "vocab_size": c.vocab_size,
    }


def kv_bytes_per_token(shape: dict, dtype_bytes: int = 4) -> int:
    """2 (K and V) x layers x kv_heads x d_head x bytes."""
    return 2 * shape["n_layers"] * shape["n_kv_heads"] * shape["d_head"] * dtype_bytes


# ----------------------------------------------------------------------------
# The two decode loops
# ----------------------------------------------------------------------------


@dataclass
class GenResult:
    token_ids: list = field(default_factory=list)
    text: str = ""
    prefill_s: float = 0.0
    decode_step_s: list = field(default_factory=list)

    @property
    def ttft_s(self) -> float:
        """Time to first token = prefill (there is no queue in a bare loop)."""
        return self.prefill_s

    @property
    def median_itl_s(self) -> float:
        return statistics.median(self.decode_step_s) if self.decode_step_s else 0.0


@torch.inference_mode()
def generate_with_cache(model, input_ids, max_new_tokens=32, eos_id=None,
                        on_token=None) -> GenResult:
    """Prefill once, then one forward pass per new token, reusing the KV cache.

    `on_token(token_id, step_index)` is called for every generated token, which
    is how the streaming server in project 02 pushes bytes to the client.
    """
    res = GenResult()

    t0 = time.perf_counter()
    out = model(input_ids, use_cache=True)
    past = out.past_key_values
    next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
    res.prefill_s = time.perf_counter() - t0

    res.token_ids.append(int(next_id))
    if on_token is not None:
        on_token(int(next_id), 0)

    for step in range(1, max_new_tokens):
        if eos_id is not None and int(next_id) == eos_id:
            break
        t0 = time.perf_counter()
        out = model(next_id, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        res.decode_step_s.append(time.perf_counter() - t0)
        res.token_ids.append(int(next_id))
        if on_token is not None:
            on_token(int(next_id), step)
    return res


@torch.inference_mode()
def generate_no_cache(model, input_ids, max_new_tokens=32, eos_id=None) -> GenResult:
    """The control: no KV cache. Every step re-reads the whole prefix.

    Mathematically identical output, quadratically more work.
    """
    res = GenResult()
    ids = input_ids
    for step in range(max_new_tokens):
        t0 = time.perf_counter()
        out = model(ids, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(-1, keepdim=True)
        dt = time.perf_counter() - t0
        if step == 0:
            res.prefill_s = dt
        else:
            res.decode_step_s.append(dt)
        ids = torch.cat([ids, next_id], dim=1)
        res.token_ids.append(int(next_id))
        if eos_id is not None and int(next_id) == eos_id:
            break
    return res


# ----------------------------------------------------------------------------
# Timing helpers
# ----------------------------------------------------------------------------


def interleaved(fns: dict, rounds: int = 3, warmup: int = 1) -> dict:
    """Time several callables round-robin and keep the MINIMUM per callable.

    Round-robin matters on a shared machine: if another process wakes up for
    ten seconds, timing A fully and then B fully charges the whole disturbance
    to whichever ran during it. Interleaving spreads the noise over both, and
    taking the minimum keeps the least-disturbed sample.
    """
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


def percentile(values, q: float) -> float:
    """Nearest-rank percentile; q in [0, 100]. No numpy dependency needed."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round(q / 100.0 * len(s) + 0.5)) - 1))
    return s[k]
