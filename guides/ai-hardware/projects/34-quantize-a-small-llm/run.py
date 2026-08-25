"""Project 34 -- Quantize a small LLM.

Takes Qwen2.5-0.5B-Instruct down to INT8 / INT4 / INT3 with round-to-nearest and
with GPTQ, and measures the damage three ways: WikiText-2 perplexity, agreement
with the fp32 model's own predictions, and accuracy on a subset of MMLU.

Runs in about 10 minutes on 12 CPU threads.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import quantlib as ql
from gptq import gptq_model

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

EVAL_SEQ, CALIB_SEQ, SEQLEN, MMLU_N = 8, 8, 512, 150

# MMLU costs one forward pass per question, so we only pay for it on the four
# configurations the comparison actually hinges on.
MMLU_ROWS = {"fp32 (baseline)", "RTN INT4, group 128", "RTN INT3, group 128",
             "GPTQ INT4, group 128"}
results = {}


def log(*a):
    print(*a, flush=True)


def size_report(model, bits, group_size, per_tensor=False, sym=True):
    """Bytes on disk if this quantization were stored for real."""
    quant_bits = 0
    quant_params = 0
    for _, mod in ql.quantizable_linears(model).items():
        out_f, in_f = mod.weight.shape
        bpw = ql.bits_per_weight(in_f, bits, group_size, per_tensor, sym=sym)
        quant_bits += bpw * out_f * in_f
        quant_params += out_f * in_f
    other = sum(p.numel() for p in model.parameters()) - quant_params
    total_bytes = quant_bits / 8 + other * 2          # rest kept in fp16
    return {"quantized_params": quant_params, "other_params": other,
            "bits_per_weight": quant_bits / quant_params,
            "total_MB": total_bytes / 1e6}


def make_plots(res):
    """Rebuild both figures from findings.json (so `run.py --plot` is instant)."""
    rows = res["table"]
    ppl_fp32 = rows[0]["ppl"]
    color_of = {"-": "#444444", "RTN": "#d62728", "GPTQ": "#2ca02c"}

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), constrained_layout=True)
    colors = [color_of[r["method"]] for r in rows]
    axes[0].barh([r["name"] for r in rows], [r["ppl"] for r in rows],
                 color=colors)
    axes[0].axvline(ppl_fp32, ls="--", color="k", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("WikiText-2 perplexity, log scale (lower is better)")
    axes[0].invert_yaxis()
    axes[0].set_title("Quality")

    mrows = [r for r in rows if r["mmlu"] == r["mmlu"]]
    se = 100 * (0.25 * 0.75 / MMLU_N) ** 0.5
    axes[1].barh([r["name"] for r in mrows], [r["mmlu"] * 100 for r in mrows],
                 xerr=se, color=[color_of[r["method"]] for r in mrows],
                 error_kw={"ecolor": "k", "capsize": 3})
    axes[1].axvline(25, ls=":", color="k", lw=1)
    axes[1].set_xlabel(f"MMLU accuracy % on {MMLU_N} questions "
                       f"(25% = random, error bar +/-1 s.e.)")
    axes[1].invert_yaxis()
    axes[1].set_title("Knowledge")

    axes[2].scatter([r["MB"] for r in rows], [r["ppl"] for r in rows],
                    c=colors, s=70, zorder=3)
    offsets = [(6, 4), (6, 4), (8, -16), (-42, 12), (6, 4), (14, 4), (8, -16)]
    for r, off in zip(rows, offsets):
        axes[2].annotate(r["name"].replace(", ", "\n"), (r["MB"], r["ppl"]),
                         fontsize=7, xytext=off, textcoords="offset points")
    axes[2].set_yscale("log")
    axes[2].set_xlim(350, 1150)
    axes[2].set_xlabel("model size (MB)")
    axes[2].set_ylabel("perplexity, log scale")
    axes[2].set_title("The actual trade-off")
    axes[2].grid(alpha=0.3)
    fig.suptitle("Quantizing " + ql.MODEL +
                 " -- grey fp32, red round-to-nearest, green GPTQ")
    fig.savefig(f"{OUT}/quantize_llm.png", dpi=120)
    plt.close(fig)

    per_layer = res["per_layer"]
    kinds = sorted({r["kind"] for r in per_layer})
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for kind in kinds:
        pts = sorted([(r["block"], r["rel_mse"]) for r in per_layer
                      if r["kind"] == kind])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                markersize=3, label=kind)
    ax.set_xlabel("transformer block")
    ax.set_ylabel("relative INT4 weight error")
    ax.set_title("Not all weight matrices quantize equally well")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/per_layer_error.png", dpi=120)
    plt.close(fig)
    log(f"  wrote {OUT}/quantize_llm.png and {OUT}/per_layer_error.png")


def main():
    ql.setup()
    t_start = time.time()

    log("=== A. What are we quantizing? ===")
    tok, model = ql.load(ql.MODEL)
    linears = ql.quantizable_linears(model)
    n_lin = sum(m.weight.numel() for m in linears.values())
    n_all = sum(p.numel() for p in model.parameters())
    log(f"  model: {ql.MODEL}")
    log(f"  total parameters      {n_all / 1e6:8.1f} M")
    log(f"  in transformer Linear {n_lin / 1e6:8.1f} M  "
        f"({100 * n_lin / n_all:.1f}%)  <- these get quantized")
    log(f"  embeddings + lm_head  {(n_all - n_lin) / 1e6:8.1f} M  "
        f"({100 * (n_all - n_lin) / n_all:.1f}%)  <- left in fp16")
    log(f"  Linear layers: {len(linears)}")
    results["census"] = {"total": n_all, "linear": n_lin,
                         "n_linear_layers": len(linears),
                         "fp32_MB": n_all * 4 / 1e6, "fp16_MB": n_all * 2 / 1e6}

    text = ql.wikitext_text()
    ev = ql.token_batches(tok, text, EVAL_SEQ, SEQLEN)
    calib = ql.token_batches(tok, text, CALIB_SEQ, SEQLEN, skip=EVAL_SEQ)
    mmlu = ql.mmlu_items(MMLU_N)
    log(f"  eval: {EVAL_SEQ * SEQLEN} WikiText-2 tokens; "
        f"calibration: {CALIB_SEQ * SEQLEN} tokens (disjoint); MMLU: {MMLU_N} questions")

    original = {n: m.weight.data.clone() for n, m in linears.items()}

    log("\n=== B. Baseline (fp32) ===")
    with ql.Timer() as t:
        ppl_fp32, pred_fp32 = ql.perplexity(model, ev, return_logits=True)
    acc_fp32 = ql.mmlu_accuracy(model, tok, mmlu)
    log(f"  perplexity {ppl_fp32:.3f}   MMLU {acc_fp32 * 100:.1f}%   ({t.dt:.1f}s)")

    rows = [{"name": "fp32 (baseline)", "method": "-", "bits": 16,
             "group": "-", "ppl": ppl_fp32, "agree": 1.0, "mmlu": acc_fp32,
             "MB": n_all * 2 / 1e6, "bpw": 16.0}]

    log("\n=== C. Round-to-nearest (RTN): quantize each weight on its own ===")
    rtn_specs = [
        ("RTN INT8, per-channel", 8, None),
        ("RTN INT4, per-channel", 4, None),
        ("RTN INT4, group 128", 4, 128),
        ("RTN INT3, group 128", 3, 128),
    ]
    for name, bits, group in rtn_specs:
        with ql.QuantizedWeights(model, lambda n, w: ql.fake_quant(w, bits, group)):
            ppl, pred = ql.perplexity(model, ev, return_logits=True)
            acc = (ql.mmlu_accuracy(model, tok, mmlu)
                   if name in MMLU_ROWS else float("nan"))
        sz = size_report(model, bits, group)
        rows.append({"name": name, "method": "RTN", "bits": bits,
                     "group": group or "per-channel", "ppl": ppl,
                     "agree": ql.agreement(pred, pred_fp32), "mmlu": acc,
                     "MB": sz["total_MB"], "bpw": sz["bits_per_weight"]})
        log(f"  {name:24s} ppl {ppl:8.3f}  agree {rows[-1]['agree'] * 100:5.1f}%  "
            f"MMLU {acc * 100:5.1f}%  {sz['total_MB']:6.0f} MB")

    log("\n=== D. GPTQ: quantize column by column, compensating as you go ===")
    for name, bits, group in [("GPTQ INT4, per-channel", 4, None),
                              ("GPTQ INT4, group 128", 4, 128)]:
        for n, m in linears.items():
            m.weight.data = original[n].clone()
        with ql.Timer() as t:
            gptq_model(model, calib, bits=bits, group_size=group,
                       verbose=False, quantlib=ql)
        ppl, pred = ql.perplexity(model, ev, return_logits=True)
        acc = (ql.mmlu_accuracy(model, tok, mmlu)
               if name in MMLU_ROWS else float("nan"))
        sz = size_report(model, bits, group)
        rows.append({"name": name, "method": "GPTQ", "bits": bits,
                     "group": group or "per-channel", "ppl": ppl,
                     "agree": ql.agreement(pred, pred_fp32), "mmlu": acc,
                     "MB": sz["total_MB"], "bpw": sz["bits_per_weight"],
                     "quantize_seconds": t.dt})
        log(f"  {name:24s} ppl {ppl:8.3f}  agree {rows[-1]['agree'] * 100:5.1f}%  "
            f"MMLU {acc * 100:5.1f}%  {sz['total_MB']:6.0f} MB  "
            f"(quantized in {t.dt:.0f}s)")

    for n, m in linears.items():
        m.weight.data = original[n]
    results["table"] = rows

    se = (0.25 * 0.75 / MMLU_N) ** 0.5
    log(f"\n  MMLU on {MMLU_N} questions has a standard error of about "
        f"{se * 100:.1f} percentage points --")
    log(f"  differences smaller than ~{2 * se * 100:.0f} points are noise, "
        f"which is why perplexity is the metric that ranks these.")
    results["mmlu_stderr"] = se

    log("\n=== E. Which layers hurt the most? ===")
    per_layer = []
    for name, mod in linears.items():
        W = mod.weight.data
        err = float(((W - ql.fake_quant(W, 4)) ** 2).mean() / (W ** 2).mean())
        per_layer.append({"layer": name, "rel_mse": err,
                          "kind": name.split(".")[-1],
                          "block": int(name.split(".")[1])})
    by_kind = {}
    for r in per_layer:
        by_kind.setdefault(r["kind"], []).append(r["rel_mse"])
    for kind, vals in sorted(by_kind.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        log(f"  {kind:12s} mean relative INT4 error {sum(vals) / len(vals):.4f}")
    results["per_layer"] = per_layer

    make_plots(results)
    results["total_seconds"] = time.time() - t_start
    ql.save_json(f"{OUT}/findings.json", results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("name,method,bits,group,ppl,agree,mmlu,MB,bits_per_weight\n")
        for r in rows:
            f.write(f"\"{r['name']}\",{r['method']},{r['bits']},{r['group']},"
                    f"{r['ppl']:.4f},{r['agree']:.4f},{r['mmlu']:.4f},"
                    f"{r['MB']:.1f},{r['bpw']:.3f}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    import sys
    if "--plot" in sys.argv:          # redraw from the committed findings.json
        make_plots(json.load(open(f"{OUT}/findings.json")))
    else:
        main()
