"""Project 36 -- Calibration data study.

GPTQ needs example inputs to decide how to round each weight. This project asks
two questions with real runs: how MANY examples, and examples of WHAT.

Uses SmolLM2-135M so that eight full GPTQ passes fit in one 9-minute run.
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "34-quantize-a-small-llm"))
import quantlib as ql          # noqa: E402
from gptq import gptq_model    # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

MODEL = ql.TINY
BITS, GROUP = 4, 64            # 576 and 1536 are both multiples of 64
SEQLEN, EVAL_SEQ = 512, 8
results = {}


def log(*a):
    print(*a, flush=True)


def make_plots(res):
    """Rebuild the figure from findings.json (so `run.py --plot` is instant)."""
    count_rows = res["count_sweep"]
    draw_rows = res["draw_sweep"]
    source_rows = res["source_sweep"]
    ppl_fp32 = res["baselines"]["fp32"]
    ppl_rtn = res["baselines"]["rtn"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    ns = [r["n"] for r in count_rows]
    axes[0].plot(ns, [r["ppl"] for r in count_rows], marker="o", color="#2ca02c",
                 label="GPTQ")
    axes[0].axhline(ppl_rtn, ls="--", color="#d62728",
                    label="RTN (no calibration at all)")
    axes[0].axhline(ppl_fp32, ls=":", color="k", label="fp32 ceiling")
    axes[0].scatter([4] * len(draw_rows), [r["ppl"] for r in draw_rows],
                    color="#2ca02c", marker="x", s=45, zorder=4,
                    label="other 4-sequence draws")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(ns)
    axes[0].set_xticklabels([str(n) for n in ns])
    axes[0].set_xlabel("calibration sequences (512 tokens each)")
    axes[0].set_ylabel("WikiText-2 perplexity")
    axes[0].set_title("More calibration data, up to a point")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    labels = [r.get("source", "wikitext (matched)") for r in source_rows]
    vals = [r["ppl"] for r in source_rows]
    colors = ["#2ca02c" if v < ppl_rtn else "#d62728" for v in vals]
    axes[1].barh(labels, vals, color=colors)
    axes[1].axvline(ppl_rtn, ls="--", color="#d62728")
    axes[1].axvline(ppl_fp32, ls=":", color="k")
    axes[1].set_xlim(min(ppl_fp32, min(vals)) - 1.5, max(vals) + 1.5)
    axes[1].set_xlabel("WikiText-2 perplexity  "
                       "(dotted = fp32 ceiling, dashed red = RTN floor)")
    axes[1].set_title("16 sequences each, three different sources")
    axes[1].invert_yaxis()
    axes[1].grid(alpha=0.3, axis="x")
    for label, v in zip(labels, vals):
        axes[1].annotate(f"{v:.2f}", (v, label), fontsize=8,
                         xytext=(-38, -3), textcoords="offset points",
                         color="white")
    fig.suptitle("What calibration data does for GPTQ "
                 "-- red means worse than not calibrating")
    fig.savefig(f"{OUT}/calibration.png", dpi=120)
    plt.close(fig)
    log(f"  wrote {OUT}/calibration.png")


def main():
    ql.setup()
    t_start = time.time()
    tok, model = ql.load(MODEL)
    linears = ql.quantizable_linears(model)
    original = {n: m.weight.data.clone() for n, m in linears.items()}
    log(f"  model {MODEL}; quantizing {len(linears)} Linear layers to INT{BITS}, "
        f"group {GROUP}")

    wiki = ql.wikitext_text()
    ev = ql.token_batches(tok, wiki, EVAL_SEQ, SEQLEN)
    code = ql.code_text()

    def restore():
        for n, m in linears.items():
            m.weight.data = original[n].clone()

    def run(label, calib, note=""):
        restore()
        with ql.Timer() as t:
            gptq_model(model, calib, bits=BITS, group_size=GROUP,
                       verbose=False, quantlib=ql)
        ppl = ql.perplexity(model, ev)
        log(f"  {label:34s} ppl {ppl:8.3f}   ({t.dt:.0f}s{note})")
        return {"label": label, "ppl": ppl, "seconds": t.dt,
                "calib_tokens": int(calib.numel())}

    # ------------------------------------------------------------ baselines
    log("\n=== A. Two baselines that need no calibration at all ===")
    ppl_fp32 = ql.perplexity(model, ev)
    log(f"  {'fp32 (no quantization)':34s} ppl {ppl_fp32:8.3f}")
    restore()
    with ql.QuantizedWeights(model, lambda n, w: ql.fake_quant(w, BITS, GROUP)):
        ppl_rtn = ql.perplexity(model, ev)
    log(f"  {'RTN INT4 (ignores the data)':34s} ppl {ppl_rtn:8.3f}")
    results["baselines"] = {"fp32": ppl_fp32, "rtn": ppl_rtn}

    # ------------------------------------------------ B. how many samples?
    log("\n=== B. How many calibration sequences? ===")
    count_rows = []
    for n in [1, 4, 16, 64]:
        calib = ql.token_batches(tok, wiki, n, SEQLEN, skip=EVAL_SEQ)
        row = run(f"GPTQ, {n:2d} x {SEQLEN} wikitext tokens", calib)
        row["n"] = n
        count_rows.append(row)
    results["count_sweep"] = count_rows

    # ---------------------------------------------- C. how much does luck matter?
    log("\n=== C. Same size, different draw -- is the difference above noise? ===")
    draw_rows = [r for r in count_rows if r["n"] == 4]
    for skip in [EVAL_SEQ + 200, EVAL_SEQ + 400]:
        calib = ql.token_batches(tok, wiki, 4, SEQLEN, skip=skip)
        row = run(f"GPTQ,  4 x {SEQLEN}, draw @{skip}", calib)
        row["n"] = 4
        draw_rows.append(row)
    spread = max(r["ppl"] for r in draw_rows) - min(r["ppl"] for r in draw_rows)
    log(f"  spread across three 4-sequence draws: {spread:.3f} perplexity")
    results["draw_sweep"] = draw_rows
    results["draw_spread"] = spread

    # ------------------------------------------------- D. what KIND of data?
    log("\n=== D. Calibrating on the wrong kind of text ===")
    source_rows = [dict(r, source="wikitext (matched)")
                   for r in count_rows if r["n"] == 16]
    gen = torch.Generator().manual_seed(0)
    sources = [
        ("python source code", ql.token_batches(tok, code, 16, SEQLEN)),
        ("uniform random tokens",
         torch.randint(0, tok.vocab_size, (16, SEQLEN), generator=gen)),
    ]
    for source, calib in sources:
        row = run(f"GPTQ, 16 x {SEQLEN} {source}", calib)
        row["n"] = 16
        row["source"] = source
        source_rows.append(row)
    results["source_sweep"] = source_rows

    restore()

    make_plots(results)
    results["total_seconds"] = time.time() - t_start
    ql.save_json(f"{OUT}/findings.json", results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("label,n,calib_tokens,ppl,seconds\n")
        for r in count_rows + draw_rows[1:] + source_rows[1:]:
            f.write(f"\"{r['label']}\",{r['n']},{r['calib_tokens']},"
                    f"{r['ppl']:.4f},{r['seconds']:.1f}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:      # redraw from the committed findings.json
        make_plots(json.load(open(f"{OUT}/findings.json")))
    else:
        main()
