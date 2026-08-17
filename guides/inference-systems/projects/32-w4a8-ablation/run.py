"""Project 32 -- W4-only vs W4A8: which bits actually buy you anything?

"W4A8" names a plan: 4-bit weights, 8-bit activations. Quantizing weights is
about *memory*; quantizing activations is about *math* -- it is the only way to
reach the integer Tensor Cores, because both operands of a matmul have to be
integers before an integer multiplier can be used. So the two decisions have
different payoffs and very different risks, and this project separates them.

  A. Profile the activations. Find the outlier channels that make A8 hard.
  B. The real thing: torch's own int8 dynamic-quantization path, which
     produces a genuinely faster model on this CPU. Measure speed and quality.
  C. Split the damage. Weights-only int8, activations-only int8, both.
  D. Activation granularity: per-tensor vs per-token. One of them is free on
     hardware and one of them is not, and it is not the one you would guess.
  E. SmoothQuant: move the outliers out of the activations and into the
     weights, sweep the migration strength.
  F. The ablation itself: W16A16 / W8A8 / W4A16 / W4A8, quality and cost.

    python3 run.py           # ~7 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "30-quantize-a-7b-model-end-to-end"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import quantlib as Q  # noqa: E402

EVAL_N = 10
WIN = 512
F = {}


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run(model, ev, ref, *ctxs):
    """Evaluate with a stack of context managers applied at once."""
    import contextlib
    with contextlib.ExitStack() as st:
        for c in ctxs:
            if c is not None:
                st.enter_context(c)
        r = Q.eval_chunks(model, ev, want_argmax=True)
    return {"ppl": r["ppl"], "agree": Q.agreement(ref, r["argmax"])}


# ---------------------------------------------------------------------------
# A. the activation outliers
# ---------------------------------------------------------------------------


def section_a(model, calib):
    print("\n=== A. activation outlier profile ===")
    stats = Q.act_stats(model, calib)
    per_proj = {}
    rows = []
    for name, st in stats.items():
        amax = st["absmax"]
        ratio = float(amax.max() / amax.median())
        rows.append({"linear": name, "absmax": float(amax.max()),
                     "median_channel_absmax": float(amax.median()),
                     "outlier_ratio": ratio})
        per_proj.setdefault(Q.group_of(name), []).append(ratio)
    summary = {k: {"mean_ratio": sum(v) / len(v), "max_ratio": max(v)}
               for k, v in sorted(per_proj.items())}
    worst = max(rows, key=lambda r: r["outlier_ratio"])
    for k, v in summary.items():
        print(f"  {k:11s} outlier ratio mean {v['mean_ratio']:7.1f}x  "
              f"worst {v['max_ratio']:8.1f}x")
    print(f"  worst single linear: {worst['linear']} at "
          f"{worst['outlier_ratio']:.0f}x, absmax {worst['absmax']:.1f}")
    F["A"] = {"per_proj": summary, "worst": worst, "rows": rows}
    return stats


# ---------------------------------------------------------------------------
# B. the real int8 path
# ---------------------------------------------------------------------------


def section_b(tok, model, ev, ref, base_ppl):
    print("\n=== B. torch's real int8 dynamic quantization ===")
    import torch.ao.quantization as tq

    t0 = time.time()
    qm = tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    convert_s = time.time() - t0
    r = Q.eval_chunks(qm, ev, want_argmax=True)

    ids = ev[0].unsqueeze(0)
    one = torch.tensor([[1000]])

    def make(m):
        with torch.inference_mode():
            pkv = m(ids, use_cache=True).past_key_values
        return lambda: m(one, past_key_values=pkv, use_cache=True)

    with torch.inference_mode():
        t = Q.interleaved({"fp32 decode": make(model), "int8 decode": make(qm)},
                          rounds=4)
        tp = Q.interleaved({"fp32 prefill": lambda: model(ids),
                            "int8 prefill": lambda: qm(ids)}, rounds=2)

    def nbytes(m):
        tot = 0
        for mod in m.modules():
            w = getattr(mod, "weight", None)
            if callable(w):                      # quantized linear
                w = w()
            if torch.is_tensor(w):
                tot += w.numel() * w.element_size()
        return tot

    F["B"] = {
        "convert_s": convert_s,
        "ppl": r["ppl"], "base_ppl": base_ppl,
        "ppl_ratio": r["ppl"] / base_ppl,
        "agree": Q.agreement(ref, r["argmax"]),
        "decode_ms_fp32": t["fp32 decode"] * 1000,
        "decode_ms_int8": t["int8 decode"] * 1000,
        "decode_speedup": t["fp32 decode"] / t["int8 decode"],
        "prefill_ms_fp32": tp["fp32 prefill"] * 1000,
        "prefill_ms_int8": tp["int8 prefill"] * 1000,
        "prefill_speedup": tp["fp32 prefill"] / tp["int8 prefill"],
        "weight_bytes_fp32": nbytes(model), "weight_bytes_int8": nbytes(qm),
        "weight_qscheme": str(qm.model.layers[0].mlp.down_proj.weight().qscheme()),
    }
    print(f"  decode  {F['B']['decode_ms_fp32']:.1f} -> "
          f"{F['B']['decode_ms_int8']:.1f} ms  ({F['B']['decode_speedup']:.2f}x)")
    print(f"  prefill {F['B']['prefill_ms_fp32']:.0f} -> "
          f"{F['B']['prefill_ms_int8']:.0f} ms ({F['B']['prefill_speedup']:.2f}x)")
    print(f"  perplexity {base_ppl:.3f} -> {r['ppl']:.3f} "
          f"(x{F['B']['ppl_ratio']:.2f}), agreement "
          f"{F['B']['agree']*100:.1f}%")
    del qm


# ---------------------------------------------------------------------------
# C + D. which half hurts, and how much granularity fixes
# ---------------------------------------------------------------------------


def section_cd(model, ev, ref, base_ppl):
    print("\n=== C/D. splitting the damage ===")
    rows = []

    def add(label, *ctxs):
        r = run(model, ev, ref, *ctxs)
        r.update({"label": label, "ppl_ratio": r["ppl"] / base_ppl})
        rows.append(r)
        print(f"  {label:38s} ppl {r['ppl']:9.3f} (x{r['ppl_ratio']:6.2f})  "
              f"agree {r['agree']*100:5.1f}%")

    add("W8 per-tensor only", Q.Quantized(model, bits=8, per_tensor=True))
    add("W8 per-channel only", Q.Quantized(model, bits=8, group=0))
    add("A8 per-tensor only", Q.ActQuant(model, 8, "per-tensor"))
    add("A8 per-token only", Q.ActQuant(model, 8, "per-token"))
    add("A8 per-channel only (not implementable)",
        Q.ActQuant(model, 8, "per-channel"))
    add("W8 per-channel + A8 per-tensor", Q.Quantized(model, bits=8, group=0),
        Q.ActQuant(model, 8, "per-tensor"))
    add("W8 per-channel + A8 per-token", Q.Quantized(model, bits=8, group=0),
        Q.ActQuant(model, 8, "per-token"))
    F["C"] = {"rows": rows, "base_ppl": base_ppl}


# ---------------------------------------------------------------------------
# E. SmoothQuant
# ---------------------------------------------------------------------------


def section_e(model, stats, ev, ref, base_ppl):
    print("\n=== E. SmoothQuant migration strength ===")
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 0.9):
        # alpha = 0 means "migrate nothing", i.e. plain W8A8 per-tensor.
        s = Q.smooth_scales(model, stats, alpha) if alpha > 0 else None
        # The two halves of SmoothQuant: the activation is divided by s (inside
        # ActQuant) and the weight column is multiplied by s (Quantized takes
        # the weight-side factor through the same argument AWQ uses, because it
        # is exactly the same operation -- rescale a column, then quantize).
        r = run(model, ev, ref,
                Q.Quantized(model, bits=8, group=0, awq_scales=s),
                Q.ActQuant(model, 8, "per-tensor", smooth=s))
        rows.append({"alpha": alpha, "ppl": r["ppl"], "agree": r["agree"],
                     "ppl_ratio": r["ppl"] / base_ppl})
        print(f"  alpha {alpha:.2f}: ppl {r['ppl']:9.3f} "
              f"(x{rows[-1]['ppl_ratio']:5.2f})  agree {r['agree']*100:5.1f}%")
    F["E"] = {"rows": rows, "base_ppl": base_ppl}


# ---------------------------------------------------------------------------
# F. the ablation
# ---------------------------------------------------------------------------


def section_f(model, stats, ev, ref, base_ppl):
    print("\n=== F. W16A16 / W8A8 / W4A16 / W4A8 ===")
    awq4, _ = Q.awq_scales(model, stats, 4, 128)
    smooth = Q.smooth_scales(model, stats, 0.5)
    rows = [{"plan": "W16A16 (baseline)", "ppl": base_ppl, "agree": 1.0,
             "w_bits": 16, "a_bits": 16}]

    def add(plan, w_bits, a_bits, *ctxs):
        r = run(model, ev, ref, *ctxs)
        rows.append({"plan": plan, "ppl": r["ppl"], "agree": r["agree"],
                     "w_bits": w_bits, "a_bits": a_bits,
                     "ppl_ratio": r["ppl"] / base_ppl})
        print(f"  {plan:34s} ppl {r['ppl']:9.3f} "
              f"(x{rows[-1]['ppl_ratio']:6.2f})  agree {r['agree']*100:5.1f}%")

    add("W8A8 per-tensor acts", 8, 8, Q.Quantized(model, bits=8, group=0),
        Q.ActQuant(model, 8, "per-tensor"))
    add("W8A8 + SmoothQuant", 8, 8,
        Q.Quantized(model, bits=8, group=0, awq_scales=smooth),
        Q.ActQuant(model, 8, "per-tensor", smooth=smooth))
    add("W8A8 per-token acts", 8, 8, Q.Quantized(model, bits=8, group=0),
        Q.ActQuant(model, 8, "per-token"))
    add("W4A16 (AWQ, g128)", 4, 16,
        Q.Quantized(model, bits=4, group=128, awq_scales=awq4))
    add("W4A8 per-token acts", 4, 8,
        Q.Quantized(model, bits=4, group=128, awq_scales=awq4),
        Q.ActQuant(model, 8, "per-token"))
    add("W4A8 + SmoothQuant, per-tensor acts", 4, 8,
        Q.Quantized(model, bits=4, group=128, awq_scales=smooth),
        Q.ActQuant(model, 8, "per-tensor", smooth=smooth))

    # Cost side. Weight bytes are exact; the compute side is the ratio of
    # Tensor-Core throughputs an H100 publishes for each operand pair.
    h100_tflops = {(16, 16): 989.0, (8, 8): 1979.0, (4, 16): 989.0, (4, 8): 1979.0}
    for r in rows:
        sz = Q.size_report("Qwen2.5-7B", min(r["w_bits"], 16), 16)
        r["gib_7b"] = sz["gib"]
        r["h100_dense_tflops"] = h100_tflops.get((r["w_bits"], r["a_bits"]),
                                                 h100_tflops[(16, 16)])
    F["F"] = {"rows": rows, "base_ppl": base_ppl}


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

    pp = f["A"]["per_proj"]
    ks = list(pp)
    ax[0].bar(range(len(ks)), [pp[k]["mean_ratio"] for k in ks], color="#2471a3",
              label="mean over layers")
    ax[0].bar(range(len(ks)), [pp[k]["max_ratio"] for k in ks], 0.35,
              color="#c0392b", label="worst layer")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(ks)))
    ax[0].set_xticklabels(ks, rotation=40, ha="right", fontsize=7)
    ax[0].set_ylabel("max channel |x| / median channel |x|")
    ax[0].set_title("A. where the activation outliers live")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis="y")

    rows = f["C"]["rows"]
    ax[1].barh(range(len(rows)), [r["ppl_ratio"] for r in rows],
               color=["#27ae60" if r["ppl_ratio"] < 1.1 else "#e67e22"
                      if r["ppl_ratio"] < 2 else "#c0392b" for r in rows])
    ax[1].set_yticks(range(len(rows)))
    ax[1].set_yticklabels([r["label"] for r in rows], fontsize=6.5)
    ax[1].axvline(1.0, color="k", lw=1)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("perplexity / baseline")
    ax[1].set_title("C/D. the damage is in the activations")
    ax[1].grid(alpha=0.3, axis="x")
    ax[1].invert_yaxis()

    e = f["E"]["rows"]
    ax[2].plot([r["alpha"] for r in e], [r["ppl_ratio"] for r in e], "o-",
               color="#8e44ad")
    ax[2].axhline(1.0, color="k", ls="--", lw=1, label="fp32 baseline")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("SmoothQuant alpha (0 = do nothing)")
    ax[2].set_ylabel("perplexity / baseline")
    ax[2].set_title("E. migrating the outliers")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fr = f["F"]["rows"]
    ax[3].scatter([r["gib_7b"] for r in fr], [r["ppl"] for r in fr], s=70,
                  c=["#34495e" if r["w_bits"] == 16 else "#2471a3"
                     if r["w_bits"] == 8 else "#27ae60" for r in fr])
    for r in fr:
        ax[3].annotate(r["plan"].replace(" per-tensor acts", "")
                       .replace(" per-token acts", "/pt"),
                       (r["gib_7b"], r["ppl"]), fontsize=6,
                       xytext=(3, 3), textcoords="offset points")
    ax[3].set_yscale("log")
    ax[3].set_xlabel("7B weights (GiB)")
    ax[3].set_ylabel("perplexity")
    ax[3].set_title("F. what each plan costs and buys")
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "w4a8_ablation.png"), dpi=110)
    print("wrote outputs/w4a8_ablation.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    tok, model = Q.load()
    text = Q._wikitext(500_000)
    ev = Q.token_chunks(tok, text, WIN, EVAL_N)
    calib = Q.token_chunks(tok, text, WIN, 8, skip=EVAL_N + 4)
    base = Q.eval_chunks(model, ev, want_argmax=True)
    ref = base["argmax"]
    print(f"fp32 baseline ppl {base['ppl']:.4f}")
    F["setup"] = {"model": Q.MODEL_ID, "threads": Q.N_THREADS,
                  "eval_windows": EVAL_N, "window": WIN,
                  "base_ppl": base["ppl"]}

    stats = section_a(model, calib)
    section_cd(model, ev, ref, base["ppl"])
    section_e(model, stats, ev, ref, base["ppl"])
    section_f(model, stats, ev, ref, base["ppl"])
    section_b(tok, model, ev, ref, base["ppl"])   # last: it clones the model

    F["wall_s"] = round(time.time() - t0, 1)
    Q.save_findings(os.path.join(OUT, "findings.json"), F)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
