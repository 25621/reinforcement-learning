"""quantlib.py -- the shared serving-quantization toolkit for Phase 5.

Phase 2 wrote its own forward pass because the *cache* was the subject and
HuggingFace owns its cache. Phase 5's subject is the *weights and activations*,
which live in ordinary `nn.Linear` modules -- so here the real HuggingFace model
is the right vehicle. It gives us three things for free that a hand-written
runner would not:

  * `nn.Linear.weight` is a plain tensor we can overwrite and restore, so
    "quantize the model" is a context manager rather than a rewrite;
  * `register_forward_pre_hook` taps the *input* of every linear, which is what
    activation quantization and AWQ calibration both need;
  * `torch.ao.quantization.quantize_dynamic` produces a genuinely faster int8
    model on this CPU, so project 32 can report a measured speedup and not only
    a simulated quality number.

Everything here is **fake quantization** unless a function says otherwise:
weights are rounded to the low-precision grid and immediately expanded back to
fp32, so the arithmetic still runs in fp32 but the *numbers* are exactly the
ones a real int4/fp8 kernel would use. That is the standard way to measure
quantization quality without owning the matching kernel, and it is exact --
the only thing it cannot tell you is the speed, which we get separately from
byte counting (`size_report`) and from the real int8 path in project 32.

Shared by projects 30, 32, 33, 34, 35 and 36. Project 31 quantizes the KV
cache instead and plugs into Phase 2's `kvlib` cache seam.
"""

from __future__ import annotations

import glob
import json
import math
import os
import random
import sys
import time
from collections import OrderedDict

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BIG_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
N_THREADS = 6

# The seven linear layers inside one Qwen2 transformer block. `lm_head` and the
# embedding table are deliberately NOT in this list: they are a separate
# serving decision (project 33 measures exactly what leaving them out buys).
PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj",
              "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# Loading and layer bookkeeping
# ---------------------------------------------------------------------------


def load(model_id: str = MODEL_ID, n_threads: int = N_THREADS):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(n_threads)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    model.eval()
    return tok, model


def block_linears(model) -> "OrderedDict[str, nn.Linear]":
    """The quantizable linears, keyed `layers.<i>.<proj>`.

    Filtering matters: a transformer's parameters are ~99% these matrices, and
    they are the only ones a weight-quantization kernel replaces. Sweeping
    `named_modules()` for every `nn.Linear` would also catch `lm_head`, which
    behaves very differently (project 33)."""
    out = OrderedDict()
    for i, layer in enumerate(model.model.layers):
        for proj in PROJ_NAMES:
            mod = getattr(layer.self_attn, proj, None) or getattr(layer.mlp, proj)
            out[f"layers.{i}.{proj}"] = mod
    return out


def group_of(name: str) -> str:
    """Coarse family of a linear -- used by project 33's sensitivity ranking."""
    return name.rsplit(".", 1)[1]


# ---------------------------------------------------------------------------
# The quantizer itself
# ---------------------------------------------------------------------------


def fake_quant(W: torch.Tensor, bits: int, group: int = 0, sym: bool = False,
               per_tensor: bool = False):
    """Round `W` onto a `bits`-wide grid and expand it straight back to fp32.

    `W` is (out_features, in_features). Scales are shared along the *input*
    dimension, in blocks of `group` (0 or -1 means "the whole row", i.e. one
    scale per output channel). That is the layout every weight-only kernel
    uses, because the input dimension is the one the matmul reduces over: all
    the values sharing a scale get multiplied and summed together, so the scale
    factors straight out of the sum.

    `per_tensor=True` collapses it further to a single scale for the whole
    matrix -- the cheapest option, the one torch's default int8 path uses, and
    the one a single large weight ruins for everybody else.

    `sym=True` centres the grid on zero (levels -2^(b-1)..2^(b-1)-1) and stores
    one number per block. `sym=False` (asymmetric) also stores a zero-point, so
    it can shift the grid to cover a lopsided range -- two numbers per block,
    slightly better fit. Weight-only 4-bit deployments almost always use
    asymmetric; int8 activations almost always use symmetric, because a
    symmetric grid lets the hardware skip a correction term.
    """
    out_f, in_f = W.shape
    if per_tensor:
        Wg = W.reshape(1, 1, -1).float()
        g, pad = W.numel(), 0
    else:
        g = in_f if group in (0, -1, None) else int(group)
        pad = (-in_f) % g
        Wp = W if pad == 0 else torch.cat([W, W[:, -1:].expand(out_f, pad)], dim=1)
        Wg = Wp.reshape(out_f, -1, g).float()

    if sym:
        qmax = 2 ** (bits - 1) - 1
        s = (Wg.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-9)
        q = (Wg / s).round().clamp(-qmax - 1, qmax)
        deq = q * s
    else:
        levels = 2 ** bits - 1
        mx, mn = Wg.amax(-1, keepdim=True), Wg.amin(-1, keepdim=True)
        s = ((mx - mn) / levels).clamp_min(1e-9)
        z = (-mn / s).round()
        q = (Wg / s + z).round().clamp(0, levels)
        deq = (q - z) * s

    if per_tensor:
        return deq.reshape(out_f, in_f).to(W.dtype)
    return deq.reshape(out_f, -1)[:, :in_f].to(W.dtype)


def fake_quant_fp8(x: torch.Tensor, fmt: str = "e4m3", scale=None):
    """Fake-quantize to one of the two FP8 formats, using torch's real casts.

    Nothing is simulated here: `torch.float8_e4m3fn` is a genuine 1-byte dtype
    on CPU, so the round-trip reproduces the exact bit pattern an FP8 kernel
    would store. `scale` (optional) is divided out first and multiplied back
    afterwards -- FP8's whole range is only +-448 (e4m3), so serving stacks
    always pair it with a scale that recentres the tensor inside that window.
    """
    dt = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    y = x if scale is None else x / scale
    y = y.to(dt).to(torch.float32)
    return y if scale is None else y * scale


class Quantized:
    """Context manager: swap in fake-quantized weights, put them back on exit.

    Usage:
        with Quantized(model, bits=4, group=128):
            ppl = perplexity(model, chunks)

    `awq_scales` is a per-linear vector `s` over input channels. AWQ works by
    quantizing `W * diag(s)` and dividing `diag(s)` back out afterwards. In a
    real deployment the division is *folded into the previous operation* (the
    RMSNorm weight, or the previous linear's rows), so it costs nothing at
    serving time; here we simply apply both halves, which produces numerically
    the same weights the folded kernel would use.
    """

    def __init__(self, model, bits=4, group=128, sym=False, names=None,
                 skip=(), awq_scales=None, per_layer=None, per_tensor=False):
        self.model = model
        self.per_tensor = per_tensor
        self.lins = block_linears(model)
        if names is not None:
            self.lins = OrderedDict((k, v) for k, v in self.lins.items() if k in names)
        if skip:
            self.lins = OrderedDict((k, v) for k, v in self.lins.items()
                                    if group_of(k) not in skip and k not in skip)
        self.bits, self.group, self.sym = bits, group, sym
        self.awq_scales = awq_scales or {}
        # per_layer: {name: dict(bits=..., group=..., sym=...)} overrides
        self.per_layer = per_layer or {}
        self._saved = {}

    def __enter__(self):
        with torch.no_grad():
            for name, mod in self.lins.items():
                cfg = self.per_layer.get(name, {})
                bits = cfg.get("bits", self.bits)
                if bits >= 16:                      # "leave this one alone"
                    continue
                group = cfg.get("group", self.group)
                sym = cfg.get("sym", self.sym)
                pt = cfg.get("per_tensor", self.per_tensor)
                W = mod.weight.data
                self._saved[name] = W.clone()
                s = self.awq_scales.get(name)
                if s is not None:
                    s = s.to(W.dtype).clamp_min(1e-5)
                    Wq = fake_quant(W * s, bits, group, sym, pt) / s
                else:
                    Wq = fake_quant(W, bits, group, sym, pt)
                mod.weight.data.copy_(Wq)
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for name, W in self._saved.items():
                self.lins[name].weight.data.copy_(W)
        self._saved.clear()
        return False


def quantize_head(model, bits=8, group=0, sym=False):
    """Context manager for the output head alone (`lm_head` / tied embedding).

    Separate from `Quantized` because it is a separate serving decision, and
    because Qwen2.5-0.5B *ties* `lm_head` to the input embedding table: writing
    one writes the other. That tie is why the head is 136M of a 494M model here
    -- large enough that leaving it in fp16 is a real cost, which is exactly
    what project 33 has to price."""

    class _Head:
        def __enter__(self_):
            W = model.lm_head.weight.data
            self_.saved = W.clone()
            with torch.no_grad():
                W.copy_(fake_quant(W, bits, group, sym))
            return self_

        def __exit__(self_, *exc):
            with torch.no_grad():
                model.lm_head.weight.data.copy_(self_.saved)
            return False

    return _Head()


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------


def _quant_act(x, bits, mode):
    """Fake-quantize an activation tensor (..., in_features), symmetric.

    `mode` decides what shares a scale:
      per-tensor  -- one scale for the whole tensor. Cheapest, and what int8
                     serving kernels used before SmoothQuant existed.
      per-token   -- one scale per row (per token). Free on hardware, because
                     the scale factors out of each output element's sum.
      per-channel -- one scale per input channel. Sounds better, and *cannot be
                     used* by a real matmul: the reduction runs along channels,
                     so a per-channel scale does not factor out. It is here as
                     a reference point showing what you are giving up.
    """
    qmax = 2 ** (bits - 1) - 1
    if mode == "per-tensor":
        s = x.abs().amax() / qmax
    elif mode == "per-token":
        s = x.abs().amax(dim=-1, keepdim=True) / qmax
    elif mode == "per-channel":
        s = x.abs().reshape(-1, x.shape[-1]).amax(0) / qmax
    else:
        raise ValueError(mode)
    s = torch.as_tensor(s).clamp_min(1e-9)
    return (x / s).round().clamp(-qmax - 1, qmax) * s


class ActQuant:
    """Context manager: fake-quantize the *input* of every target linear.

    Implemented with forward pre-hooks, which fire with the tuple of positional
    arguments about to enter the module -- returning a new tuple replaces them.
    That is the whole mechanism; no model surgery required.

    `smooth` is a per-linear vector `s`: SmoothQuant divides the activation by
    `s` and multiplies the weight by `s`, moving the hard-to-quantize spikes
    out of the activation and into the weight, which has room for them. Passing
    `smooth` here only does the activation half -- the weight half is applied by
    handing the same dict to `Quantized(..., awq_scales=inverse)`."""

    def __init__(self, model, bits=8, mode="per-tensor", names=None, smooth=None):
        self.lins = block_linears(model)
        if names is not None:
            self.lins = OrderedDict((k, v) for k, v in self.lins.items() if k in names)
        self.bits, self.mode = bits, mode
        self.smooth = smooth or {}
        self.handles = []

    def __enter__(self):
        for name, mod in self.lins.items():
            s = self.smooth.get(name)

            def hook(_m, args, _s=s):
                x = args[0]
                if _s is not None:
                    x = x / _s.to(x.dtype)
                x = _quant_act(x, self.bits, self.mode)
                if _s is not None:
                    x = x * _s.to(x.dtype)
                return (x,) + tuple(args[1:])

            self.handles.append(mod.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False


# ---------------------------------------------------------------------------
# Calibration: what the model actually sees
# ---------------------------------------------------------------------------


@torch.inference_mode()
def act_stats(model, chunks, names=None, sample_rows=0, seed=0):
    """Run calibration data through the model and record, per linear:

      absmean  -- mean |x| per input channel. AWQ's saliency signal: a weight
                  column that always meets large activations matters more.
      absmax   -- max |x| per input channel. SmoothQuant's signal, and the
                  thing an outlier channel blows up.
      rows     -- (optional) a random sample of actual input rows, kept so the
                  AWQ search can score candidate scales against real data
                  instead of a summary statistic.

    One forward pass over the calibration set fills all three, which is why
    calibration costs about the same as one evaluation and not one per config.
    """
    lins = block_linears(model)
    if names is not None:
        lins = OrderedDict((k, v) for k, v in lins.items() if k in names)
    g = torch.Generator().manual_seed(seed)
    stats = {k: {"absmean": None, "absmax": None, "n": 0, "rows": []} for k in lins}
    handles = []

    for name, mod in lins.items():
        def hook(_m, args, _n=name):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            st = stats[_n]
            a_sum = x.abs().sum(0)
            a_max = x.abs().amax(0)
            st["absmean"] = a_sum if st["absmean"] is None else st["absmean"] + a_sum
            st["absmax"] = a_max if st["absmax"] is None else torch.maximum(st["absmax"], a_max)
            st["n"] += x.shape[0]
            if sample_rows and len(st["rows"]) * 32 < sample_rows:
                idx = torch.randint(0, x.shape[0], (32,), generator=g)
                st["rows"].append(x[idx].clone())
        handles.append(mod.register_forward_pre_hook(hook))

    for ch in chunks:
        model(ch.unsqueeze(0) if ch.dim() == 1 else ch)
    for h in handles:
        h.remove()

    for name, st in stats.items():
        st["absmean"] = st["absmean"] / max(st["n"], 1)
        st["rows"] = torch.cat(st["rows"])[:sample_rows] if st["rows"] else None
    return stats


def awq_scales(model, stats, bits=4, group=128, sym=False,
               alphas=(0.0, 0.25, 0.5, 1.0), verbose=False):
    """AWQ: pick a per-input-channel scale that protects the salient columns.

    The idea in one line. Quantization error on weight column `j` gets
    multiplied by activation channel `j` on its way to the output, so a column
    that meets big activations does more damage per unit of rounding error.
    AWQ therefore multiplies that column *up* before quantizing (so it lands on
    a finer part of the grid relative to its own size) and divides it back
    afterwards. `s_j = absmean_j ** alpha`, normalised so the geometric mean is
    1 -- alpha = 0 is plain round-to-nearest, alpha = 1 scales fully with the
    activation size.

    Alpha is *searched*, not derived, exactly as in the paper: for each layer we
    try each alpha and keep the one whose quantized weights reproduce the real
    output best on calibration rows. Searching per layer matters -- the best
    alpha is not the same for `down_proj` as for `q_proj`.
    """
    lins = block_linears(model)
    chosen, report = {}, []
    for name, mod in lins.items():
        st = stats.get(name)
        if st is None or st["rows"] is None:
            continue
        X = st["rows"]                              # (rows, in)
        W = mod.weight.data.float()
        ref = X @ W.T
        denom = ref.pow(2).mean().clamp_min(1e-12)
        best = (None, float("inf"), None)
        for a in alphas:
            if a == 0.0:
                s = torch.ones_like(st["absmean"])
            else:
                s = st["absmean"].clamp_min(1e-5) ** a
                s = s / s.log().mean().exp()        # geometric mean -> 1
            Wq = fake_quant(W * s, bits, group, sym) / s
            err = ((X @ Wq.T - ref).pow(2).mean() / denom).item()
            if err < best[1]:
                best = (a, err, s)
        chosen[name] = best[2]
        report.append({"linear": name, "alpha": best[0], "rel_mse": best[1]})
        if verbose:
            print(f"  {name}: alpha={best[0]} rel_mse={best[1]:.5f}")
    return chosen, report


def smooth_scales(model, stats, alpha=0.5):
    """SmoothQuant: split the dynamic range between activation and weight.

    `s_j = amax(|x_j|)^alpha / amax(|W_:,j|)^(1-alpha)`. Dividing the activation
    by `s` flattens its outlier channels; multiplying the weight column by `s`
    gives that range to the weights, which are dense and well-behaved and can
    absorb it. alpha = 0.5 splits the difference; higher alpha pushes more of
    the problem into the weights.
    """
    lins = block_linears(model)
    out = {}
    for name, mod in lins.items():
        st = stats.get(name)
        if st is None:
            continue
        a = st["absmax"].clamp_min(1e-5)
        w = mod.weight.data.float().abs().amax(0).clamp_min(1e-5)
        s = (a ** alpha) / (w ** (1 - alpha))
        out[name] = (s / s.log().mean().exp()).clamp(1e-2, 1e2)
    return out


# ---------------------------------------------------------------------------
# Corpora -- five distributions a real deployment actually sees
# ---------------------------------------------------------------------------


def _wikitext(n_chars=400_000) -> str:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("Salesforce/wikitext",
                           "wikitext-2-raw-v1/test-00000-of-00001.parquet",
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


def _code(n_chars=400_000) -> str:
    """Real Python: the standard library that ships with this interpreter."""
    import sysconfig
    root = sysconfig.get_paths()["stdlib"]
    files = sorted(glob.glob(os.path.join(root, "*.py")))
    buf, total = [], 0
    for f in files:
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if len(s) < 500:
            continue
        buf.append(s)
        total += len(s)
        if total >= n_chars:
            break
    return "\n".join(buf)


def _mmlu_frame():
    import pandas as pd
    from huggingface_hub import hf_hub_download
    path = hf_hub_download("cais/mmlu", "all/test-00000-of-00001.parquet",
                           repo_type="dataset")
    return pd.read_parquet(path)


def _exam(n_chars=300_000) -> str:
    """Multiple-choice exam text -- short, dense, question-shaped."""
    df = _mmlu_frame()
    buf, total = [], 0
    for _, r in df.iterrows():
        ch = list(r["choices"])
        buf.append(f"Question: {r['question']}\nA. {ch[0]}\nB. {ch[1]}\n"
                   f"C. {ch[2]}\nD. {ch[3]}\nAnswer: {'ABCD'[int(r['answer'])]}\n\n")
        total += len(buf[-1])
        if total >= n_chars:
            break
    return "".join(buf)


_CHAT_TASKS = [
    "Explain {t} to a beginner in three sentences.",
    "Write a short paragraph about {t}.",
    "What are two common mistakes people make with {t}?",
    "Summarise the main idea behind {t}.",
    "Give a simple example involving {t}.",
    "Compare {t} with a simpler alternative.",
    "List three practical tips about {t}.",
    "Why does {t} matter in everyday life?",
]
_CHAT_TOPICS = [
    "compound interest", "sleep hygiene", "public transport", "photosynthesis",
    "recycling glass", "learning a language", "bread baking", "tidal forces",
    "budget airlines", "vaccination", "musical scales", "wind turbines",
    "credit scores", "cast-iron pans", "monsoons", "chess openings",
    "coral reefs", "noise-cancelling headphones", "crop rotation", "insurance",
    "the water cycle", "double-entry bookkeeping", "tuning a guitar", "yeast",
    "solar panels", "the metric system", "first aid", "tax brackets",
    "container ships", "antibiotic resistance", "map projections", "espresso",
]


def chat_prompts(tok, n=128, seed=0):
    """Instruction-style prompts wrapped in the model's own chat template.

    Wrapping matters more than it looks: an instruct model's activations are
    noticeably different inside `<|im_start|>user ... <|im_start|>assistant`
    than on bare text, and calibrating on bare text then serving chat traffic
    is exactly the mismatch project 34 measures."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        t = rng.choice(_CHAT_TOPICS)
        q = rng.choice(_CHAT_TASKS).format(t=t)
        out.append(tok.apply_chat_template([{"role": "user", "content": q}],
                                           tokenize=False, add_generation_prompt=True))
    return out


def _chat(tok, n_chars=300_000) -> str:
    return "".join(chat_prompts(tok, n=400))[:n_chars]


def _json_text(n_chars=200_000, seed=0) -> str:
    """Structured output: the traffic shape Phase 8 is about."""
    rng = random.Random(seed)
    recs = []
    for i in range(4000):
        recs.append(json.dumps({
            "id": i,
            "name": rng.choice(["alpha", "beta", "gamma", "delta", "omega"]),
            "score": round(rng.random() * 100, 2),
            "tags": rng.sample(["red", "blue", "green", "fast", "slow"], 2),
            "ok": rng.random() > 0.5,
        }, indent=2))
        if sum(len(r) for r in recs) > n_chars:
            break
    return "\n".join(recs)


def corpora(tok, names=("wiki", "code", "exam", "chat", "json")) -> dict:
    fns = {"wiki": _wikitext, "code": _code, "exam": _exam,
           "chat": lambda: _chat(tok), "json": _json_text}
    return {n: fns[n]() for n in names}


def token_chunks(tok, text: str, chunk: int = 512, n: int = 16, skip: int = 0):
    """Cut a corpus into `n` non-overlapping windows of `chunk` tokens."""
    ids = tok(text, return_tensors="pt").input_ids[0]
    need = (skip + n) * chunk
    if ids.numel() < need:
        reps = math.ceil(need / max(ids.numel(), 1))
        ids = ids.repeat(reps)
    ids = ids[skip * chunk: (skip + n) * chunk]
    return ids.reshape(n, chunk)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def eval_chunks(model, chunks, want_argmax=False):
    """One pass over the eval windows; returns per-token NLL and (optionally)
    the greedy prediction at every position.

    Both come out of the same forward pass because that is the whole trick
    behind project 35: a *paired* comparison against the baseline's own
    predictions is far more sensitive than comparing two aggregate scores, and
    it costs nothing extra."""
    nll_sum, ntok = 0.0, 0
    argmax = [] if want_argmax else None
    for ch in chunks:
        logits = model(ch.unsqueeze(0)).logits[0].float()
        lp = logits[:-1]
        tgt = ch[1:]
        nll_sum += F.cross_entropy(lp, tgt, reduction="sum").item()
        ntok += tgt.numel()
        if want_argmax:
            argmax.append(lp.argmax(-1))
    out = {"nll": nll_sum / ntok, "ppl": math.exp(nll_sum / ntok), "ntok": ntok}
    if want_argmax:
        out["argmax"] = torch.cat(argmax)
    return out


def perplexity(model, chunks) -> float:
    return eval_chunks(model, chunks)["ppl"]


def agreement(pred_a, pred_b) -> float:
    """Fraction of positions where two models make the same greedy choice."""
    return (pred_a == pred_b).float().mean().item()


def mmlu_items(n=200, seed=0):
    """MMLU questions, scored by reading the logits of ' A'..' D'.

    One forward pass per question -- no generation, no parsing. The whole
    measurement is `argmax` over four token ids, which is why 200 questions
    cost about a minute here instead of an hour."""
    df = _mmlu_frame()
    idx = list(range(len(df)))
    random.Random(seed).shuffle(idx)
    items = []
    for i in idx[:n]:
        r = df.iloc[i]
        ch = list(r["choices"])
        items.append({
            "prompt": (f"The following is a multiple choice question.\n\n"
                       f"{r['question']}\nA. {ch[0]}\nB. {ch[1]}\nC. {ch[2]}\n"
                       f"D. {ch[3]}\nAnswer:"),
            "answer": int(r["answer"]),
        })
    return items


@torch.inference_mode()
def mmlu_eval(model, tok, items, want_choices=False):
    letters = [tok(" " + c).input_ids[-1] for c in "ABCD"]
    correct, choices = 0, []
    for it in items:
        ids = tok(it["prompt"], return_tensors="pt").input_ids
        logits = model(ids).logits[0, -1]
        c = int(torch.tensor([logits[t] for t in letters]).argmax())
        choices.append(c)
        correct += int(c == it["answer"])
    acc = correct / len(items)
    out = {"acc": acc, "n": len(items),
           "stderr": math.sqrt(max(acc * (1 - acc), 1e-9) / len(items))}
    if want_choices:
        out["choices"] = choices
    return out


@torch.inference_mode()
def greedy_generate(model, tok, prompts, max_new=32):
    """Free-running greedy decode -- what a user actually receives.

    Teacher-forced agreement (above) never lets an error compound; this does,
    which is why a quantized model can look 98% identical per token and still
    produce visibly different answers."""
    outs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        got = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        outs.append(got[0, ids.shape[1]:].tolist())
    return outs


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


# ---------------------------------------------------------------------------
# Byte accounting -- the half of the story fake quantization cannot show
# ---------------------------------------------------------------------------

# Real config numbers. 0.5B and 1.5B are loaded and run in this phase; 7B and
# 70B are read off their published configs so the arithmetic is real even
# though the weights never fit on this box.
MODEL_SHAPES = {
    "Qwen2.5-0.5B": dict(layers=24, d_model=896, d_ffn=4864, heads=14,
                         kv_heads=2, d_head=64, vocab=151936, tied=True),
    "Qwen2.5-1.5B": dict(layers=28, d_model=1536, d_ffn=8960, heads=12,
                         kv_heads=2, d_head=128, vocab=151936, tied=True),
    "Qwen2.5-7B": dict(layers=28, d_model=3584, d_ffn=18944, heads=28,
                       kv_heads=4, d_head=128, vocab=152064, tied=False),
    "Llama-3-70B": dict(layers=80, d_model=8192, d_ffn=28672, heads=64,
                        kv_heads=8, d_head=128, vocab=128256, tied=False),
}


def param_counts(sh):
    """Split a model's parameters into the three groups that get quantized
    with different rules."""
    attn = sh["layers"] * (
        sh["d_model"] * sh["heads"] * sh["d_head"]          # q_proj
        + 2 * sh["d_model"] * sh["kv_heads"] * sh["d_head"]  # k_proj, v_proj
        + sh["heads"] * sh["d_head"] * sh["d_model"])        # o_proj
    mlp = sh["layers"] * 3 * sh["d_model"] * sh["d_ffn"]
    embed = sh["vocab"] * sh["d_model"] * (1 if sh["tied"] else 2)
    return {"attn": attn, "mlp": mlp, "embed_head": embed,
            "total": attn + mlp + embed}


def size_report(shape_name, w_bits=16, head_bits=16, group=128):
    """Bytes on the wire for a given bit plan, including the scale overhead.

    The scale overhead is the part people forget: asymmetric group-128 int4
    stores a scale *and* a zero-point per 128 weights. At fp16 each that is
    32 bits per 128 weights = 0.25 bits/weight, so "int4" is really 4.25
    bits/weight -- a 6% surprise on your memory budget."""
    sh = MODEL_SHAPES[shape_name]
    pc = param_counts(sh)
    body = pc["attn"] + pc["mlp"]
    overhead = 0.0 if w_bits >= 16 else 32.0 / group     # fp16 scale + zero
    body_bits = body * (w_bits + overhead)
    head_bits_total = pc["embed_head"] * (head_bits + (0.0 if head_bits >= 16
                                                       else 32.0 / group))
    total = (body_bits + head_bits_total) / 8
    return {
        "model": shape_name, "params": pc["total"],
        "w_bits": w_bits, "head_bits": head_bits,
        "bytes": total, "gib": total / 2**30,
        "eff_bits_per_weight": 8 * total / pc["total"],
    }


def group_params(sh):
    """Parameters per weight family -- the unit project 33 allocates bits to."""
    L, d, f = sh["layers"], sh["d_model"], sh["d_ffn"]
    kv = sh["kv_heads"] * sh["d_head"]
    return {
        "q_proj": L * d * sh["heads"] * sh["d_head"],
        "k_proj": L * d * kv,
        "v_proj": L * d * kv,
        "o_proj": L * sh["heads"] * sh["d_head"] * d,
        "gate_proj": L * d * f,
        "up_proj": L * d * f,
        "down_proj": L * f * d,
        "embed_head": sh["vocab"] * d * (1 if sh["tied"] else 2),
    }


def plan_bytes(shape_name, bits_by_group: dict, default_bits=4, group=128,
               add_scale_overhead=True):
    """Bytes for an arbitrary per-family bit assignment.

    `add_scale_overhead=False` when the bit numbers you pass already include
    their scales -- MXFP4's "4.25 bits" does, project 36's does."""
    gp = group_params(MODEL_SHAPES[shape_name])
    total = 0.0
    for name, n in gp.items():
        b = bits_by_group.get(name, default_bits)
        overhead = 0.0 if (b >= 16 or not add_scale_overhead) else 32.0 / group
        total += n * (b + overhead) / 8
    return {"bytes": total, "gib": total / 2**30,
            "params": sum(gp.values()),
            "eff_bits": 8 * total / sum(gp.values())}


def kv_bytes_per_token(shape_name, bits=16):
    sh = MODEL_SHAPES[shape_name]
    return 2 * sh["layers"] * sh["kv_heads"] * sh["d_head"] * bits / 8


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def interleaved(fns: dict, rounds: int = 3, warmup: int = 1) -> dict:
    """Time callables round-robin and keep the minimum. This box is shared with
    another agent; running A to completion and then B charges any background
    spike entirely to whichever happened to be running at the time."""
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


def save_findings(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"wrote {path}")


def load_findings(path):
    with open(path) as f:
        return json.load(f)


def add_quantlib_to_path():
    """Called by projects 32-36: `sys.path` entry for this directory."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
