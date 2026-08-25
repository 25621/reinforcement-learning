"""Shared quantization toolkit for AI-hardware Phase 7 (projects 34-38).

Everything here is written from scratch on top of plain PyTorch: this machine has
no bitsandbytes, no auto-gptq, no optimum, and a GPU (sm_61) that PyTorch refuses
to launch kernels on. That is a feature for a learning guide -- the arithmetic is
visible instead of hidden behind a CUDA kernel.

Conventions used throughout:
  * A weight matrix W has shape (out_features, in_features), matching nn.Linear.
  * Quantization *groups* run along the INPUT dimension, because the matmul sums
    over that dimension, so every element of a group is multiplied by the same
    activation slice and can share one scale.
  * "fake quant" = quantize then immediately dequantize back to fp32. The values
    are exactly the ones an INT4 kernel would reconstruct, but the tensor stays
    fp32 so we can run it on a CPU that has no INT4 matmul instruction.
"""

import json
import os
import time

import torch

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"          # 494 M params, the "small LLM"
TINY = "HuggingFaceTB/SmolLM2-135M-Instruct"  # 135 M params, for wide sweeps

_WIKITEXT = ("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet")


# ----------------------------------------------------------------- environment
def setup(threads=12, seed=0):
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def load(name=MODEL):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


# ------------------------------------------------------------------------ data
def wikitext_text():
    """WikiText-2 (raw) test split, joined into one long string."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(_WIKITEXT[0], _WIKITEXT[1], repo_type="dataset")
    return "\n\n".join(pd.read_parquet(path)["text"].tolist())


def code_text():
    """Python source from the standard library -- a deliberately different domain."""
    import glob
    chunks = []
    for path in sorted(glob.glob("/usr/lib/python3*/*.py"))[:60]:
        try:
            chunks.append(open(path, encoding="utf-8", errors="ignore").read())
        except OSError:
            pass
    return "\n\n".join(chunks)


def token_batches(tok, text, n_seq, seqlen, skip=0):
    """Cut `text` into n_seq non-overlapping sequences of exactly `seqlen` tokens."""
    need = (skip + n_seq) * seqlen
    ids = tok(text[: 6 * need + 20000], return_tensors="pt").input_ids[0]
    assert ids.numel() >= need, f"need {need} tokens, got {ids.numel()}"
    ids = ids[skip * seqlen: need]
    return ids.view(n_seq, seqlen)


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def perplexity(model, batches, return_logits=False):
    """Token-level perplexity: exp(mean negative log-likelihood of the next token)."""
    total_nll, total_tok, logits_out = 0.0, 0, []
    for i in range(batches.shape[0]):
        x = batches[i: i + 1]
        out = model(x)
        logits = out.logits[:, :-1].float()
        target = x[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum")
        total_nll += float(nll)
        total_tok += target.numel()
        if return_logits:
            logits_out.append(logits[0].argmax(-1))
    ppl = float(torch.exp(torch.tensor(total_nll / total_tok)))
    if return_logits:
        return ppl, torch.cat(logits_out)
    return ppl


def agreement(pred_a, pred_b):
    """Fraction of positions where two models pick the same next token."""
    return float((pred_a == pred_b).float().mean())


def mmlu_items(n=100, seed=0):
    """A fixed random subset of MMLU test questions (real benchmark data)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet",
                           repo_type="dataset")
    df = pd.read_parquet(path)
    df = df.sample(n=n, random_state=seed)
    return [(r.question, list(r.choices), int(r.answer)) for r in df.itertuples()]


@torch.no_grad()
def mmlu_accuracy(model, tok, items):
    """Score A/B/C/D by one forward pass per question, reading the letter logits."""
    letters = ["A", "B", "C", "D"]
    letter_ids = [tok(" " + c, add_special_tokens=False).input_ids[-1] for c in letters]
    correct = 0
    for question, choices, answer in items:
        prompt = question.strip() + "\n"
        for letter, choice in zip(letters, choices):
            prompt += f"{letter}. {choice}\n"
        prompt += "Answer:"
        ids = tok(prompt, return_tensors="pt").input_ids[:, -320:]
        logits = model(ids).logits[0, -1]
        pick = int(torch.tensor([logits[i] for i in letter_ids]).argmax())
        correct += int(pick == answer)
    return correct / len(items)


# ----------------------------------------------------------------- quantizers
def quantize(W, bits, group_size=None, sym=True, per_tensor=False):
    """Round-to-nearest (RTN) uniform quantization of a weight matrix.

    group_size=None  -> one scale per output row  (per-channel)
    group_size=g     -> one scale per g input elements of a row (per-group)
    per_tensor=True  -> exactly one scale for the whole matrix

    Returns (q, scale, zero, shape) so the caller can measure real storage cost.
    """
    out_f, in_f = W.shape
    pad = 0
    if per_tensor:
        X = W.reshape(1, -1)
    elif group_size is None:
        X = W
    else:
        # If in_features is not a multiple of the group size, repeat the last
        # column to fill the final group. Repeating a value already inside the
        # group cannot change that group's min/max, so the scales stay exact.
        pad = (-in_f) % group_size
        Wp = W if pad == 0 else torch.cat([W, W[:, -1:].expand(out_f, pad)], dim=1)
        X = Wp.reshape(out_f * ((in_f + pad) // group_size), group_size)

    if sym:
        qmax = 2 ** (bits - 1) - 1                    # e.g. 7 for INT4
        amax = X.abs().amax(dim=1, keepdim=True)
        scale = (amax / qmax).clamp(min=1e-8)
        zero = torch.zeros_like(scale)
        q = torch.clamp(torch.round(X / scale), -qmax - 1, qmax)
    else:
        qmax = 2 ** bits - 1                          # e.g. 15 for INT4
        lo = X.amin(dim=1, keepdim=True)
        hi = X.amax(dim=1, keepdim=True)
        scale = ((hi - lo) / qmax).clamp(min=1e-8)
        zero = torch.round(-lo / scale)
        q = torch.clamp(torch.round(X / scale) + zero, 0, qmax)
    return q, scale, zero, (out_f, in_f, pad)


def dequantize(q, scale, zero, shape):
    out_f, in_f, pad = shape
    W = ((q - zero) * scale).reshape(out_f, in_f + pad)
    return W[:, :in_f] if pad else W


def fake_quant(W, bits, group_size=None, sym=True, per_tensor=False):
    """Quantize and immediately reconstruct -- the values an INT kernel would see."""
    return dequantize(*quantize(W, bits, group_size, sym, per_tensor))


def bits_per_weight(in_f, bits, group_size, per_tensor=False, scale_bits=16,
                    sym=True):
    """Effective storage cost once the scales (and zero-points) are counted."""
    if per_tensor:
        groups_per_row = 1.0 / in_f
    elif group_size is None:
        groups_per_row = 1.0
    else:
        groups_per_row = -(-in_f // group_size)   # ceil
    extra = scale_bits * (1 if sym else 2) * groups_per_row / in_f
    return bits + extra


# ------------------------------------------------- walking a transformer model
def quantizable_linears(model):
    """Every nn.Linear inside the transformer blocks.

    Embeddings and lm_head are deliberately excluded: they are looked up / read
    one row at a time rather than multiplied densely, and quantizing them costs
    far more quality than it saves. Every production recipe skips them too.
    """
    blocks = model.model.layers
    out = {}
    for bi, block in enumerate(blocks):
        for name, mod in block.named_modules():
            if isinstance(mod, torch.nn.Linear):
                out[f"layers.{bi}.{name}"] = mod
    return out


class QuantizedWeights:
    """Context manager: swap in fake-quantized weights, restore fp32 on exit."""

    def __init__(self, model, fn, names=None):
        self.model = model
        self.fn = fn
        self.names = names
        self.saved = {}

    def __enter__(self):
        for name, mod in quantizable_linears(self.model).items():
            if self.names is not None and name not in self.names:
                continue
            self.saved[name] = mod.weight.data
            mod.weight.data = self.fn(name, mod.weight.data)
        return self

    def __exit__(self, *exc):
        for name, mod in quantizable_linears(self.model).items():
            if name in self.saved:
                mod.weight.data = self.saved[name]
        self.saved.clear()
        return False


# ------------------------------------------------------------- block plumbing
@torch.no_grad()
def block_inputs(model, batches):
    """Capture the hidden states and keyword arguments entering transformer block 0.

    GPTQ needs to run one block at a time, but a modern decoder block also needs
    the rotary tables and the attention mask, which the top-level model builds.
    Rather than rebuild them, we let the model build them and steal them with a
    pre-hook that aborts the forward pass.
    """
    blocks = model.model.layers
    grabbed = {}

    def pre(mod, args, kwargs):
        grabbed["kwargs"] = kwargs
        raise _Stop()

    handle = blocks[0].register_forward_pre_hook(pre, with_kwargs=True)
    hidden = []
    try:
        for i in range(batches.shape[0]):
            try:
                model(batches[i: i + 1])
            except _Stop:
                pass
            hidden.append(model.model.embed_tokens(batches[i: i + 1]))
    finally:
        handle.remove()
    kwargs = dict(grabbed["kwargs"])
    kwargs["past_key_values"] = None
    kwargs["use_cache"] = False
    return hidden, kwargs


class _Stop(Exception):
    pass


@torch.no_grad()
def run_block(block, hidden, kwargs):
    out = block(hidden, **kwargs)
    return out[0] if isinstance(out, tuple) else out


# --------------------------------------------------------------------- output
def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"wrote {path}")


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.dt = time.perf_counter() - self.t
        return False
