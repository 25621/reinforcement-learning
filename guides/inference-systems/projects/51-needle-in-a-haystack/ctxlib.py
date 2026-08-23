"""Long-context serving helpers, shared by inference-systems Phase 8.

Phases 2-7 ran on `09/kvlib.py`, a Qwen2 forward pass written out by hand so
that the KV cache could be swapped. Phase 8 needs the opposite trade: prompts
of 8,000-16,000 tokens, where a hand-written attention that materialises the
whole `(heads, T, T)` score matrix would need 6 GB per layer. So the projects
that only need a *fast, memory-sane* forward pass use HuggingFace's own model
with PyTorch's fused `scaled_dot_product_attention`, and this file is the thin
layer that makes that convenient and consistent.

Nothing here re-implements a model. It provides:

    load()             real Qwen2.5-0.5B-Instruct, float32, SDPA attention
    filler_tokens()    a long stretch of real English to pad prompts with
    chat_ids()         the model's own chat template, as token ids
    greedy()           prefill + greedy decode with the two times separated
    kv_bytes_per_token()  the size formula every long-context budget starts from

Shared by projects 51, 52 and 57.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "12")
os.environ.setdefault("MKL_NUM_THREADS", "12")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
N_THREADS = 12


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load(model_id: str = MODEL_ID, threads: int = N_THREADS, layers: int | None = None):
    """Load the tokenizer and model. `layers` truncates the block stack.

    Truncation exists for project 57, which must hold dozens of live sessions
    in RAM at once. A shallower model has a proportionally smaller KV cache
    per token, which is the only thing that project measures. Every project
    that reports *quality* uses the full 24 blocks.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    if layers is not None:
        model.model.layers = model.model.layers[:layers]
        model.config.num_hidden_layers = layers
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def chat_ids(tok, user: str, system: str | None = None) -> torch.Tensor:
    """Wrap a user message in the model's own chat template.

    Skipping this and feeding raw text works, but an instruct model answers
    far better inside the format it was tuned on, and a needle test that
    fails because of a missing `<|im_start|>` measures the harness, not the
    model.
    """
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    out = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    return out["input_ids"]


@torch.inference_mode()
def greedy(model, ids: torch.Tensor, n_new: int = 12, past=None,
           eos_id: int | None = None):
    """Prefill then greedy-decode, timing the two phases separately.

    Returns (new_token_ids, prefill_seconds, decode_seconds, past_key_values).

    `past` lets a caller hand in a cache that was built earlier -- that is
    exactly what project 52's warm path does, and the reason this is written
    out by hand instead of calling `model.generate()`.
    """
    t0 = time.perf_counter()
    out = model(ids, past_key_values=past, use_cache=True, logits_to_keep=1)
    prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    new = [int(nxt)]
    t0 = time.perf_counter()
    for _ in range(n_new - 1):
        if eos_id is not None and int(nxt) == eos_id:
            break
        out = model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        new.append(int(nxt))
    decode_s = time.perf_counter() - t0
    return new, prefill_s, decode_s, past


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


def wikitext(n_chars: int = 800_000) -> str:
    """Real English prose to pad prompts with.

    Random tokens would be cheaper but would make the test meaningless: a
    model attending over noise has nothing to be distracted *by*, and the
    whole point of a haystack is that it looks like something worth reading.

    `datasets` is not installed here, so the parquet file comes straight off
    the Hub and is read with pandas -- same bytes, one dependency fewer.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet",
        repo_type="dataset")
    df = pd.read_parquet(path)
    buf, total = [], 0
    for s in df["text"]:
        if len(s.strip()) < 40:
            continue
        buf.append(s)
        total += len(s)
        if total >= n_chars:
            break
    return "".join(buf)


def filler_tokens(tok, n_chars: int = 800_000) -> list[int]:
    return tok(wikitext(n_chars), add_special_tokens=False).input_ids


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def kv_bytes_per_token(cfg, dtype_bytes: int = 4) -> int:
    """2 (K and V) x layers x kv-heads x head width x bytes per number.

    For Qwen2.5-0.5B in float32: 2 x 24 x 2 x 64 x 4 = 24,576 bytes -- 24 KB
    of cache for every single token of prompt. That number is why long
    context is a *memory* problem before it is a compute problem.
    """
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    d_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * n_kv * d_head * dtype_bytes


def interleaved(fns: dict, rounds: int = 3, warmup: int = 1) -> dict:
    """Time callables round-robin and keep the minimum.

    This box is shared with a desktop session. Running A to completion and
    then B charges any background spike entirely to whichever happened to be
    running; alternating spreads it over both.
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
