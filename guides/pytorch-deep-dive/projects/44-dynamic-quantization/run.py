"""Project 44 - dynamic int8 quantization of a real small language model.

Model: HuggingFaceTB/SmolLM2-135M-Instruct (134.5M parameters, float32).
Text:  wikitext-2 validation split (real Wikipedia prose).

Sections:
  1. what `quantize_dynamic` actually replaced, and what it left alone
  2. the win: size and speed
  3. the bill: perplexity, next-token agreement, and the same prompt generated twice
  4. whose fault is it - the weights or the activations?
  5. the activation outliers, measured
  6. why the activation scale is recomputed on every call ("dynamic")
  7. a cheaper deal: quantize only the attention projections
  8. quantizing the embedding table as well

Run:  python3 run.py        (~5 minutes)
"""

from __future__ import annotations

import copy
import gc
import io
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(6)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "42-export-to-onnx"))
sys.path.insert(0, os.path.join(HERE, "..", "01-stride-explorer"))

import deploy_lib as D  # noqa: E402
from plot_style import SERIES, style_axes  # noqa: E402

OUT = os.path.join(HERE, "outputs")
DATA = os.path.join(HERE, "data")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
WIKITEXT = ("https://raw.githubusercontent.com/pytorch/examples/main/"
            "word_language_model/data/wikitext-2/valid.txt")
CTX = 512
N_CHUNKS = 20          # for the headline perplexity
N_SMALL = 8            # for the ablations

FINDINGS: list[tuple] = []


def note(section, name, value):
    FINDINGS.append((section, name, value))
    print(f"    {name:<50} {value}")


def get_text() -> str:
    path = os.path.join(DATA, "wikitext2_valid.txt")
    if not os.path.exists(path):
        import urllib.request

        urllib.request.urlretrieve(WIKITEXT, path)
        print(f"downloaded {path}")
    return open(path, encoding="utf-8").read()


def state_dict_mb(module) -> float:
    buf = io.BytesIO()
    torch.save(module.state_dict(), buf)
    return buf.tell() / 1e6


def n_quantized(module) -> int:
    return sum(1 for m in module.modules()
               if type(m).__module__.startswith("torch.ao.nn.quantized")
               and type(m).__name__ in ("Linear", "Embedding"))


@torch.no_grad()
def perplexity(model, chunks) -> tuple[float, np.ndarray]:
    """exp(mean negative log-likelihood) plus the top-1 prediction at every position."""
    total_nll, total_tok, preds = 0.0, 0, []
    for ids in chunks:
        logits = model(ids).logits.float()[:, :-1]
        targets = ids[:, 1:]
        total_nll += nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            reduction="sum").item()
        total_tok += targets.numel()
        preds.append(logits.argmax(-1).reshape(-1).numpy())
    return float(np.exp(total_nll / total_tok)), np.concatenate(preds)


@torch.no_grad()
def generate(model, tok, prompt, n_new=40):
    ids = tok(prompt, return_tensors="pt").input_ids
    out = model.generate(ids, max_new_tokens=n_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def decode_tokens(model, ids, n_new=24):
    """One token at a time with a KV cache - the shape of real chat serving."""
    out = model(ids, use_cache=True)
    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    for _ in range(n_new):
        out = model(nxt, past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)


def fake_quant_weights(model, per_channel: bool):
    """int8 weights, float32 everything else - a control that isolates the weights.

    Round each weight to one of 255 int8 levels and immediately scale it back to
    float. The stored numbers are exactly what int8 would hold; the arithmetic
    stays in float32. So any quality change comes from the *weights* alone.
    """
    out = copy.deepcopy(model)
    for mod in out.modules():
        if isinstance(mod, nn.Linear):
            w = mod.weight.data
            scale = (w.abs().amax(dim=1, keepdim=True) if per_channel
                     else w.abs().max()) / 127.0
            scale = torch.clamp(scale, min=1e-12)
            mod.weight.data = torch.round(w / scale).clamp(-127, 127) * scale
    return out


# ==========================================================================
def main():
    t_start = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"torch {torch.__version__} | quantized backend "
          f"{torch.backends.quantized.engine}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    fp32 = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).eval()

    ids_all = tok(get_text()[:400_000], return_tensors="pt").input_ids[0]
    chunks = [ids_all[i * CTX:(i + 1) * CTX].unsqueeze(0) for i in range(N_CHUNKS)]
    small = chunks[:N_SMALL]
    print(f"model {sum(p.numel() for p in fp32.parameters()) / 1e6:.1f}M params | "
          f"{N_CHUNKS} x {CTX} = {N_CHUNKS * CTX} tokens of wikitext-2")

    # ------------------------------------------------------------------ [1]
    print("\n[1] what quantize_dynamic replaced")
    t0 = time.perf_counter()
    int8 = torch.ao.quantization.quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)
    t_quant = time.perf_counter() - t0

    n_linear = sum(1 for m in fp32.modules() if isinstance(m, nn.Linear))
    emb = fp32.model.embed_tokens
    emb_ptr = emb.weight.data_ptr()
    # lm_head *shares* the embedding tensor (tie_word_embeddings=True), so counting
    # it again would double-count 28.3M values.
    lin_params = sum(m.weight.numel() for m in fp32.modules()
                     if isinstance(m, nn.Linear) and m.weight.data_ptr() != emb_ptr)
    all_params = sum(p.numel() for p in fp32.parameters())
    mb32, mb8 = state_dict_mb(fp32), state_dict_mb(int8)

    note(1, "quantization time (s)", f"{t_quant:.2f}")
    note(1, "nn.Linear modules in the model", n_linear)
    note(1, "quantized modules after the call", n_quantized(int8))
    note(1, "embedding table", f"{tuple(emb.weight.shape)} = "
                               f"{emb.weight.numel() / 1e6:.1f}M values, left in float32")
    note(1, "weights inside Linear layers (lm_head excluded, it is tied)",
         f"{lin_params / 1e6:.1f}M of {all_params / 1e6:.1f}M "
         f"({100 * lin_params / all_params:.0f}%)")
    note(1, "lm_head shares the embedding tensor",
         f"{fp32.lm_head.weight.data_ptr() == emb_ptr} - quantizing it makes an "
         f"int8 copy, so both live in the file")

    # ------------------------------------------------------------------ [2]
    print("\n[2] the win: size and speed")
    prefill_ids = chunks[0]
    short = ids_all[:32].unsqueeze(0)

    # Build every variant this project compares, then time them all in ONE
    # rotation. Every speed number below therefore comes from the same minute of
    # this machine's life, which is the only way the ratios mean anything here.
    from torch.ao.quantization import per_channel_dynamic_qconfig

    qcfg = torch.ao.quantization.default_dynamic_qconfig
    lin_names = [n for n, m in fp32.named_modules() if isinstance(m, nn.Linear)]
    models = {
        "fp32": fp32,
        "int8 all Linear": int8,
        "int8 per-channel": torch.ao.quantization.quantize_dynamic(
            fp32, {nn.Linear: per_channel_dynamic_qconfig}, dtype=torch.qint8),
        "int8 attention only": torch.ao.quantization.quantize_dynamic(
            fp32, qconfig_spec={n: qcfg for n in lin_names if "self_attn" in n},
            dtype=torch.qint8),
        "int8 MLP only": torch.ao.quantization.quantize_dynamic(
            fp32, qconfig_spec={n: qcfg for n in lin_names if ".mlp." in n},
            dtype=torch.qint8),
    }
    pre = D.interleaved({k: (lambda m=m: m(prefill_ids)) for k, m in models.items()},
                        rounds=11, calls=1, warmup=1)
    dec = D.interleaved({"fp32": lambda: decode_tokens(fp32, short, 24),
                         "int8": lambda: decode_tokens(int8, short, 24)},
                        rounds=11, calls=1, warmup=1)
    # Report the FASTEST round of each variant, not the median. Contention can only
    # ever make a run slower, so the minimum is the closest thing to the true cost;
    # the medians of these five models overlap and rank differently every run.
    prefill_ms = {k: v["min_ms"] for k, v in pre.items()}
    p32, p8 = prefill_ms["fp32"], prefill_ms["int8 all Linear"]
    d32, d8 = dec["fp32"]["min_ms"] / 24, dec["int8"]["min_ms"] / 24

    note(2, "state_dict on disk: float32", f"{mb32:.1f} MB")
    note(2, "state_dict on disk: int8", f"{mb8:.1f} MB   ({mb32 / mb8:.2f}x smaller)")
    note(2, "if every weight were int8", f"{mb32 / 4:.1f} MB (the gap is the embedding)")
    note(2, f"prefill {CTX} tokens: fp32 / int8 (ms)",
         f"{p32:.0f} / {p8:.0f}   {p32 / p8:.2f}x")
    note(2, "  prefill tokens/s", f"{CTX * 1000 / p32:.0f} / {CTX * 1000 / p8:.0f}")
    note(2, "decode 1 token: fp32 / int8 (ms)", f"{d32:.1f} / {d8:.1f}   {d32 / d8:.2f}x")
    note(2, "  decode tokens/s", f"{1000 / d32:.1f} / {1000 / d8:.1f}")
    note(2, "  decode spread over 11 rounds (fp32 / int8, ms)",
         f"[{dec['fp32']['min_ms'] / 24:.1f}-{dec['fp32']['max_ms'] / 24:.1f}] / "
         f"[{dec['int8']['min_ms'] / 24:.1f}-{dec['int8']['max_ms'] / 24:.1f}]")
    note(2, "  prefill spread over 11 rounds (fp32 / int8, ms)",
         f"[{pre['fp32']['min_ms']:.0f}-{pre['fp32']['max_ms']:.0f}] / "
         f"[{pre['int8 all Linear']['min_ms']:.0f}-"
         f"{pre['int8 all Linear']['max_ms']:.0f}]")

    # ------------------------------------------------------------------ [3]
    print("\n[3] the bill: quality")
    ppl32, pred32 = perplexity(fp32, chunks)
    ppl8, pred8 = perplexity(int8, chunks)
    note(3, "perplexity, float32", f"{ppl32:.4f}")
    note(3, "perplexity, int8 dynamic",
         f"{ppl8:.4f}   ({100 * (ppl8 / ppl32 - 1):+.1f}%)")
    note(3, "next-token top-1 agreement",
         f"{100 * float((pred32 == pred8).mean()):.2f}%  of {len(pred32)} positions")

    prompt = "The history of the printing press begins"
    g32, g8 = generate(fp32, tok, prompt), generate(int8, tok, prompt)
    note(3, "greedy continuations identical", str(g32 == g8))
    with open(os.path.join(OUT, "generations.txt"), "w") as fh:
        fh.write(f"PROMPT: {prompt}\n\n--- float32 ---\n{g32}\n\n--- int8 ---\n{g8}\n")
    print(f"    float32: {g32[len(prompt):][:100]!r}")
    print(f"    int8   : {g8[len(prompt):][:100]!r}")

    # ------------------------------------------------------------------ [4]
    print("\n[4] whose fault: the weights or the activations?")
    ppl32_s, _ = perplexity(fp32, small)
    ppl8_s, _ = perplexity(int8, small)
    note(4, f"float32 (reference, {N_SMALL} chunks)", f"{ppl32_s:.3f}")
    note(4, "int8 weights AND activations, per-tensor", f"{ppl8_s:.3f}")

    wq = fake_quant_weights(fp32, per_channel=False)
    ppl_wt, _ = perplexity(wq, small)
    del wq
    gc.collect()
    note(4, "int8 weights only, per-tensor (float32 math)", f"{ppl_wt:.3f}")

    wq = fake_quant_weights(fp32, per_channel=True)
    ppl_wc, _ = perplexity(wq, small)
    del wq
    gc.collect()
    note(4, "int8 weights only, per-channel (float32 math)", f"{ppl_wc:.3f}")

    ppl_pc, _ = perplexity(models["int8 per-channel"], small)
    note(4, "int8 both, per-channel weights",
         f"{ppl_pc:.3f}   (prefill "
         f"{p32 / prefill_ms['int8 per-channel']:.2f}x)")

    # ------------------------------------------------------------------ [5]
    print("\n[5] the activation outliers")
    PROBE_LAYER = 11          # the worst offender, found by the model-wide scan below
    CALM_LAYER = 15           # an ordinary layer, used by section 6
    chan_max, per_call_max, samples = [], [], []
    calm_max, calm_samples = [], []

    def hook(_mod, inp, _out):
        a = inp[0].detach()[0]                     # (tokens, channels)
        chan_max.append(a.abs().amax(0).numpy())
        per_call_max.append(float(a.abs().max()))
        samples.append(a.abs().flatten()[::97].numpy())

    def calm_hook(_mod, inp, _out):
        a = inp[0].detach()[0]
        calm_max.append(float(a.abs().max()))
        calm_samples.append(a.abs().flatten()[::97].numpy())

    handles2 = [
        fp32.model.layers[PROBE_LAYER].mlp.down_proj.register_forward_hook(hook),
        fp32.model.layers[CALM_LAYER].mlp.down_proj.register_forward_hook(calm_hook),
    ]
    with torch.no_grad():
        for ids in chunks[:12]:
            fp32(ids)
    for h in handles2:
        h.remove()

    cmax = np.stack(chan_max).max(0)               # worst value seen per channel
    med = float(np.median(cmax))
    worst = int(cmax.argmax())
    note(5, f"input channels at layer {PROBE_LAYER} down_proj", len(cmax))
    note(5, "typical channel's largest value (median)", f"{med:.3f}")
    note(5, "loudest channel", f"#{worst} at {cmax[worst]:.3f}  "
                               f"= {cmax[worst] / med:.1f}x the typical channel")
    n_out = int((cmax > 5 * med).sum())
    note(5, "channels above 5x the typical maximum",
         f"{n_out} of {len(cmax)} ({100 * n_out / len(cmax):.2f}%)")
    note(5, "int8 levels left for a typical channel",
         f"{255 * med / cmax[worst]:.1f} of 255")

    # the same measurement for every Linear input in the model, in one pass
    stats: dict[str, list] = {}

    def make_hook(name):
        def h(_mod, inp, _out):
            a = inp[0].detach().reshape(-1, inp[0].shape[-1])
            per_chan = a.abs().amax(0)
            stats.setdefault(name, []).append(
                (float(per_chan.max()), float(per_chan.median())))
        return h

    handles = [m.register_forward_hook(make_hook(n))
               for n, m in fp32.named_modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        fp32(chunks[0])
        fp32(chunks[1])
    for h in handles:
        h.remove()
    ratios = {n: max(a / b for a, b in v) for n, v in stats.items()}
    ranked = sorted(ratios.items(), key=lambda kv: -kv[1])
    note(5, "worst outlier ratio in the whole model",
         f"{ranked[0][0]}  {ranked[0][1]:.1f}x")
    for name, r in ranked[1:4]:
        note(5, f"  next: {name}", f"{r:.1f}x")
    note(5, "median outlier ratio over all 211 Linear inputs",
         f"{np.median(list(ratios.values())):.1f}x")
    note(5, "  effective int8 levels there",
         f"{255 / ranked[0][1]:.1f} for a typical channel of the worst layer")

    # ------------------------------------------------------------------ [6]
    print("\n[6] why the scale is recomputed on every call")
    maxes = np.array(calm_max)                     # ordinary layer, section 6
    outlier_maxes = np.array(per_call_max)         # layer 11, for the contrast
    note(6, f"layer {CALM_LAYER}: max |activation| per chunk (first 8)",
         "  ".join(f"{v:.1f}" for v in maxes[:8]))
    note(6, "  spread over 12 chunks",
         f"min {maxes.min():.1f}  max {maxes.max():.1f}  "
         f"ratio {maxes.max() / maxes.min():.2f}x")
    narrow = maxes.min()
    clipped = max(float((v > narrow).mean()) for v in calm_samples)
    note(6, "  scale frozen at the narrowest chunk: worst-case clipping",
         f"{100 * clipped:.4f}% of values, overshoot up to "
         f"{maxes.max() / narrow:.2f}x")
    lost_bits = np.log2(maxes.max() / narrow)
    note(6, "  scale frozen at the widest chunk: resolution lost",
         f"{lost_bits:.2f} bits on the narrowest chunk (8 -> "
         f"{8 - lost_bits:.2f} effective)")
    note(6, f"layer {PROBE_LAYER} (the outlier layer) spread, for contrast",
         f"{outlier_maxes.max() / outlier_maxes.min():.2f}x - pinned by its "
         f"constant loud channel")

    # ------------------------------------------------------------------ [7]
    print("\n[7] a cheaper deal: quantize only some layers")
    rows = [("all Linear", n_linear, ppl8_s, p32 / p8, mb8)]
    for label, keep in [("attention only", "self_attn"), ("MLP only", ".mlp.")]:
        key = f"int8 {label}"
        variant = models[key]
        n_layers = sum(1 for n in lin_names if keep in n)
        ppl_v, _ = perplexity(variant, small)
        sp = p32 / prefill_ms[key]
        mb_v = state_dict_mb(variant)
        rows.append((label, n_layers, ppl_v, sp, mb_v))
        note(7, f"{label}: layers / ppl / prefill / size",
             f"{n_layers} / {ppl_v:.3f} ({100 * (ppl_v / ppl32_s - 1):+.1f}%) / "
             f"{sp:.2f}x / {mb_v:.0f} MB")

    # ------------------------------------------------------------------ [8]
    print("\n[8] quantizing the embedding table as well")
    from torch.ao.quantization import float_qparams_weight_only_qconfig

    for key in ("int8 per-channel", "int8 attention only", "int8 MLP only"):
        del models[key]
    gc.collect()
    both = torch.ao.quantization.quantize_dynamic(
        fp32, {nn.Linear: qcfg, nn.Embedding: float_qparams_weight_only_qconfig},
        dtype=torch.qint8)
    mb_both = state_dict_mb(both)
    ppl_both, pred_both = perplexity(both, chunks)
    note(8, "state_dict on disk",
         f"{mb_both:.1f} MB   ({mb32 / mb_both:.2f}x smaller than float32)")
    note(8, "perplexity", f"{ppl_both:.4f}   "
                          f"({100 * (ppl_both / ppl8 - 1):+.2f}% vs int8 Linear only)")
    note(8, "top-1 agreement with float32",
         f"{100 * float((pred_both == pred32).mean()):.2f}%")

    # --------------------------------------------------------------- output
    summary = {
        "ppl": {"fp32": ppl32, "int8": ppl8, "int8_emb": ppl_both},
        "ppl_small": {"fp32": ppl32_s, "int8": ppl8_s, "w_only_tensor": ppl_wt,
                      "w_only_channel": ppl_wc, "per_channel": ppl_pc},
        "mb": {"fp32": mb32, "int8": mb8, "int8_emb": mb_both},
        "prefill_ms": {"fp32": p32, "int8": p8},
        "decode_ms": {"fp32": d32, "int8": d8},
        "act_max_per_chunk": maxes.tolist(),
        "channel_max": {"median": med, "worst": float(cmax[worst]),
                        "n_outlier": n_out, "n_channels": int(len(cmax))},
        "variants": [{"name": r[0], "layers": r[1], "ppl": r[2], "speedup": r[3],
                      "mb": r[4]} for r in rows],
    }
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    np.save(os.path.join(OUT, "channel_max.npy"), cmax)
    D.write_csv(os.path.join(OUT, "findings.csv"), FINDINGS, ["section", "name", "value"])
    figure(summary, maxes, cmax)
    print(f"\ntotal {time.time() - t_start:.0f}s")


def figure(summary, maxes, cmax):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.8), dpi=110)
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        style_axes(ax)
        ax.grid(True, color="#e1e0d9", linewidth=0.8)

    ax = axes[0]
    labels = ["float32", "int8\nLinear", "int8\nLinear+Emb"]
    vals = [summary["mb"]["fp32"], summary["mb"]["int8"], summary["mb"]["int8_emb"]]
    ax.bar(range(3), vals, color=[SERIES[0], SERIES[1], SERIES[3]])
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_ylabel("state_dict (MB)")
    ax.set_title("size: the win", loc="left", fontsize=11)

    ax = axes[1]
    keys = ["fp32", "w_only_tensor", "w_only_channel", "per_channel", "int8"]
    names = ["float32", "int8 weights\nper-tensor", "int8 weights\nper-channel",
             "int8 both\nper-channel", "int8 both\nper-tensor"]
    vals = [summary["ppl_small"][k] for k in keys]
    ax.bar(range(5), vals, color=[SERIES[0]] + [SERIES[1]] * 2 + [SERIES[3], SERIES[2]])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(5)); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylim(0, max(vals) * 1.16)
    ax.set_ylabel("perplexity (lower is better)")
    ax.set_title("the bill, and where it comes from", loc="left", fontsize=11)

    ax = axes[2]
    order = np.sort(cmax)[::-1]
    ax.plot(order, color=SERIES[2])
    ax.axhline(np.median(cmax), color=SERIES[0], linestyle="--", linewidth=1.2,
               label=f"median channel ({np.median(cmax):.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("channel, sorted")
    ax.set_ylabel("largest |value| seen (log)")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("activation outliers, layer 11 down_proj", loc="left", fontsize=11)

    ax = axes[3]
    ax.plot(range(1, len(maxes) + 1), maxes, "o-", color=SERIES[4])
    ax.axhline(maxes.min(), color=SERIES[2], linestyle="--", linewidth=1.2,
               label="narrowest chunk")
    ax.axhline(maxes.max(), color=SERIES[1], linestyle="--", linewidth=1.2,
               label="widest chunk")
    ax.set_xlabel("wikitext chunk")
    ax.set_ylabel("max |activation|")
    ax.legend(frameon=False, fontsize=8.5)
    ax.set_title("why the scale is recomputed (layer 15)", loc="left", fontsize=11)

    fig.tight_layout()
    path = os.path.join(OUT, "dynamic_quant.png")
    fig.savefig(path, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
