"""Project 36 -- FP4 on Blackwell, without a Blackwell.

There is no Blackwell GPU on this machine, and there is no fp4 dtype in torch.
That removes exactly one thing from this project -- the wall-clock speedup --
and nothing else. The *numerics* of MXFP4 and NVFP4 are fully specified by
their standards, so a from-scratch implementation reproduces bit-for-bit the
weights a B200 would hold, and every quality number below is measured rather
than estimated. Throughput is the one place we do arithmetic instead, and it is
labelled as such wherever it appears.

  A. Build E2M1 from its definition and check its value grid.
  B. Split the MXFP4-vs-NVFP4 difference into its two design axes -- block size
     and scale format -- and find out which one is carrying the result.
  C. The named formats head to head, with and without AWQ.
  D. [arithmetic] Memory and throughput on B200 vs H100 for a 70B.
  E. The gotchas: what "4-bit" really costs in bits, the second-level scale,
     the output head, and the block-outlier statistics that explain B.
  F. The verdict, through the same gate project 35 built.

    python3 run.py           # ~8 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "30-quantize-a-7b-model-end-to-end"))

import torch  # noqa: E402

import quantlib as Q  # noqa: E402
from fp4 import (E2M1_LEVELS, E2M1_MAX, FP4Weights, block_fp4,  # noqa: E402
                 quantize_e2m1)

GATE = {"ppl": 1.05, "mmlu": 8.4, "shadow": 0.95}
WIN = 512
EVAL_N = 7
MMLU_N = 80
F = {}


def run(model, ev, ref, *ctxs):
    with contextlib.ExitStack() as st:
        for c in ctxs:
            if c is not None:
                st.enter_context(c)
        r = Q.eval_chunks(model, ev, want_argmax=True)
    return {"ppl": r["ppl"], "agree": Q.agreement(ref, r["argmax"])}


# ---------------------------------------------------------------------------
# A. the format itself
# ---------------------------------------------------------------------------


def section_a():
    print("\n=== A. E2M1, the value grid ===")
    lv = [round(float(x), 3) for x in E2M1_LEVELS]
    pos = [x for x in lv if x > 0]
    gaps = [round(pos[i + 1] - pos[i], 3) for i in range(len(pos) - 1)]
    probe = {str(x): float(quantize_e2m1(torch.tensor([x])))
             for x in (0.24, 0.26, 0.9, 1.3, 2.6, 5.0, 7.0, -3.4)}
    F["A"] = {"levels": lv, "positive": pos, "gaps": gaps, "max": E2M1_MAX,
              "dynamic_range": pos[-1] / pos[0], "probe": probe,
              "int4_levels": 16, "int4_gap": "uniform"}
    print(f"  levels: {lv}")
    print(f"  gaps between consecutive positives: {gaps}")
    print(f"  dynamic range {pos[-1]}/{pos[0]} = {pos[-1]/pos[0]:.0f}x "
          f"(int4's 16 levels are evenly spaced instead)")
    print(f"  rounding probe: {probe}")


# ---------------------------------------------------------------------------
# B. the two design axes
# ---------------------------------------------------------------------------


def section_b(model, ev, ref, base_ppl):
    print("\n=== B. block size x scale format ===")
    rows = []
    for block in (16, 32, 128):
        for fmt in (("e8m0", "e8m0-up", "e4m3", "fp32") if block == 32
                    else ("e8m0", "e8m0-up", "e4m3")):
            r = run(model, ev, ref, FP4Weights(model, block=block, scale_fmt=fmt))
            rows.append({"block": block, "scale_fmt": fmt, "ppl": r["ppl"],
                         "agree": r["agree"], "ppl_ratio": r["ppl"] / base_ppl,
                         "eff_bits": 4 + 8 / block})
            print(f"  block {block:3d}  scale {fmt:5s}: ppl {r['ppl']:8.3f} "
                  f"(x{rows[-1]['ppl_ratio']:.3f})  agree {r['agree']*100:5.1f}%"
                  f"  {rows[-1]['eff_bits']:.2f} bits/w")
    # the same bit budget spent on plain int4 instead
    int4 = []
    for g in (32, 128):
        r = run(model, ev, ref, Q.Quantized(model, bits=4, group=g))
        int4.append({"group": g, "ppl": r["ppl"], "agree": r["agree"],
                     "ppl_ratio": r["ppl"] / base_ppl, "eff_bits": 4 + 32 / g})
        print(f"  int4 group {g:3d} (asymmetric): ppl {r['ppl']:8.3f} "
              f"(x{int4[-1]['ppl_ratio']:.3f})  {int4[-1]['eff_bits']:.2f} bits/w")
    F["B"] = {"rows": rows, "int4": int4, "base_ppl": base_ppl}
    return rows


# ---------------------------------------------------------------------------
# C. the named formats
# ---------------------------------------------------------------------------


def section_c(model, ev, ref, base_ppl, stats):
    print("\n=== C. MXFP4 vs NVFP4 vs the incumbents ===")
    awq4, _ = Q.awq_scales(model, stats, 4, 128)
    plans = [
        ("BF16 (baseline)", None, 16.0),
        ("FP8 e4m3 weights", "fp8", 8.0),
        ("INT8 per-channel", Q.Quantized(model, bits=8, group=0), 8.0),
        ("MXFP4 (e8m0 round-to-nearest)", FP4Weights(model, 32, "e8m0"), 4.25),
        ("MXFP4 (e8m0 round-up)", FP4Weights(model, 32, "e8m0-up"), 4.25),
        ("NVFP4 (block 16, e4m3, 2-level)",
         FP4Weights(model, 16, "e4m3", two_level=True), 4.5),
        ("INT4 g128 (RTN)", Q.Quantized(model, bits=4, group=128), 4.25),
        ("INT4 g128 + AWQ", Q.Quantized(model, bits=4, group=128,
                                        awq_scales=awq4), 4.25),
        ("MXFP4 round-up + AWQ",
         FP4Weights(model, 32, "e8m0-up", awq_scales=awq4), 4.25),
        ("NVFP4 + AWQ", FP4Weights(model, 16, "e4m3", two_level=True,
                                   awq_scales=awq4), 4.5),
    ]
    rows = []
    for label, ctx, bits in plans:
        if ctx == "fp8":
            ctx = _FP8Weights(model)
        r = run(model, ev, ref, ctx) if ctx is not None else {"ppl": base_ppl,
                                                              "agree": 1.0}
        sz = Q.plan_bytes("Llama-3-70B", {}, bits, add_scale_overhead=False)
        rows.append({"plan": label, "ppl": r["ppl"], "agree": r["agree"],
                     "ppl_ratio": r["ppl"] / base_ppl, "bits": bits,
                     "gib_70b": sz["gib"]})
        print(f"  {label:34s} ppl {r['ppl']:8.3f} (x{rows[-1]['ppl_ratio']:.3f})"
              f"  agree {r['agree']*100:5.1f}%  70B = {sz['gib']:6.1f} GiB")
    F["C"] = {"rows": rows, "base_ppl": base_ppl}
    return awq4


class _FP8Weights:
    """fp8 e4m3 weights with one fp32 scale per output channel -- the server
    default this project is trying to beat."""

    def __init__(self, model):
        self.lins = Q.block_linears(model)
        self._saved = {}

    def __enter__(self):
        with torch.no_grad():
            for name, mod in self.lins.items():
                W = mod.weight.data
                self._saved[name] = W.clone()
                s = (W.abs().amax(1, keepdim=True) / 448.0).clamp_min(1e-12)
                mod.weight.data.copy_(Q.fake_quant_fp8(W, "e4m3", s))
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for name, W in self._saved.items():
                self.lins[name].weight.data.copy_(W)
        return False


# ---------------------------------------------------------------------------
# D. arithmetic
# ---------------------------------------------------------------------------


def section_d():
    print("\n=== D. [arithmetic] B200 vs H100 ===")
    # Published dense (non-sparse) Tensor-Core throughput and HBM figures.
    hw = {
        "H100-SXM": {"hbm_gb": 80, "bw_TBs": 3.35, "bf16_tflops": 989,
                     "fp8_tflops": 1979, "fp4_tflops": None},
        "B200": {"hbm_gb": 180, "bw_TBs": 8.0, "bf16_tflops": 2250,
                 "fp8_tflops": 4500, "fp4_tflops": 9000},
    }
    rows = []
    for name, spec in hw.items():
        for plan, bits, tf in (("bf16", 16, "bf16_tflops"),
                               ("fp8", 8, "fp8_tflops"),
                               ("fp4", 4.25, "fp4_tflops")):
            gib = Q.plan_bytes("Llama-3-70B", {}, bits,
                               add_scale_overhead=False)["gib"]
            cards = max(1, -(-int(gib * 2**30) // int(0.90 * spec["hbm_gb"] * 1e9)))
            tflops = spec[tf]
            rows.append({"gpu": name, "plan": plan, "gib_70b": gib,
                         "cards": cards, "tflops": tflops,
                         "decode_ms_1user_2k": None if tflops is None else
                         1000 * (gib * 2**30 / cards) / (spec["bw_TBs"] * 1e12)})
            print(f"  {name:9s} {plan:4s}: 70B = {gib:6.1f} GiB -> {cards} card(s), "
                  f"dense {tflops if tflops else 'n/a'} TFLOP/s, "
                  f"weight-read per decode step "
                  f"{rows[-1]['decode_ms_1user_2k'] if tflops else float('nan'):.2f} ms")
    F["D"] = {"hardware": hw, "rows": rows,
              "note": "arithmetic from published specs; no Blackwell on this box"}


# ---------------------------------------------------------------------------
# E. the gotchas
# ---------------------------------------------------------------------------


def section_e(model, ev, ref, base_ppl, awq4):
    print("\n=== E. gotchas ===")
    # E1: what "4-bit" actually costs
    bits = {"MXFP4 (block 32, 8-bit scale)": 4 + 8 / 32,
            "NVFP4 (block 16, 8-bit scale)": 4 + 8 / 16,
            "INT4 group-128 (fp16 scale+zero)": 4 + 32 / 128,
            "INT4 group-32 (fp16 scale+zero)": 4 + 32 / 32}
    for k, v in bits.items():
        print(f"  effective bits: {k:36s} {v:.2f}  "
              f"(+{100*(v/4-1):.0f}% over the nominal 4)")

    # E2: the second-level scale
    e2 = []
    for two in (False, True):
        r = run(model, ev, ref, FP4Weights(model, 16, "e4m3", two_level=two))
        e2.append({"two_level": two, "ppl": r["ppl"], "agree": r["agree"]})
        print(f"  NVFP4 two-level fp32 scale {'ON ' if two else 'OFF'}: "
              f"ppl {r['ppl']:8.3f}  agree {r['agree']*100:5.1f}%")

    # E3: the output head
    e3 = []
    for label, inc in (("head left in fp32", False), ("head in FP4 too", True)):
        r = run(model, ev, ref, FP4Weights(model, 16, "e4m3", two_level=True,
                                           awq_scales=awq4, include_head=inc))
        e3.append({"plan": label, "ppl": r["ppl"], "agree": r["agree"]})
        print(f"  NVFP4+AWQ, {label:20s}: ppl {r['ppl']:8.3f}  "
              f"agree {r['agree']*100:5.1f}%")

    # E4: why block size matters -- the outlier statistics inside a block
    lins = Q.block_linears(model)
    W = torch.cat([lins[n].weight.data.flatten()[:2_000_000]
                   for n in list(lins)[:8]]).reshape(1, -1)
    e4 = []
    for block in (16, 32, 128):
        n = (W.numel() // block) * block
        g = W[0, :n].reshape(-1, block)
        ratio = (g.abs().amax(1) / g.abs().median(1).values.clamp_min(1e-9))
        rec = block_fp4(g.reshape(1, -1), block, "e4m3")
        err = float((rec.reshape(-1) - W[0, :n]).pow(2).mean()
                    / W[0, :n].pow(2).mean())
        e4.append({"block": block, "mean_outlier_ratio": float(ratio.mean()),
                   "p99_outlier_ratio": float(ratio.quantile(0.99)),
                   "rel_mse": err})
        print(f"  block {block:3d}: within-block max/median = "
              f"{float(ratio.mean()):5.2f} (p99 {float(ratio.quantile(0.99)):6.2f})"
              f"  relative reconstruction MSE {err:.5f}")

    F["E"] = {"effective_bits": bits, "two_level": e2, "head": e3,
              "block_stats": e4}


# ---------------------------------------------------------------------------
# F. the verdict, through project 35's gate
# ---------------------------------------------------------------------------


def section_f(tok, model, ev, ref, base_ppl, awq4):
    print("\n=== F. the gate ===")
    items = Q.mmlu_items(MMLU_N, seed=17)
    base_mmlu = Q.mmlu_eval(model, tok, items)
    plans = [
        ("FP8 e4m3 weights", _FP8Weights(model)),
        ("MXFP4 round-up + AWQ", FP4Weights(model, 32, "e8m0-up",
                                            awq_scales=awq4)),
        ("NVFP4 + AWQ", FP4Weights(model, 16, "e4m3", two_level=True,
                                   awq_scales=awq4)),
    ]
    rows = []
    for label, ctx in plans:
        with ctx:
            e = Q.eval_chunks(model, ev, want_argmax=True)
            m = Q.mmlu_eval(model, tok, items)
        checks = {"ppl_ratio": e["ppl"] / base_ppl,
                  "mmlu_drop_pts": 100 * (base_mmlu["acc"] - m["acc"]),
                  "shadow_agree": Q.agreement(ref, e["argmax"])}
        # Thresholds from project 35's *calibrated* gate, not from intuition:
        # 1.05x perplexity, an MMLU drop within what the eval can resolve, and
        # a shadow-agreement bar set below what a known-harmless int8 scores.
        fails = [k for k, ok in (("perplexity", checks["ppl_ratio"] <= 1.05),
                                 ("mmlu", checks["mmlu_drop_pts"] <= GATE["mmlu"]),
                                 ("shadow", checks["shadow_agree"] >= 0.95))
                 if not ok]
        rows.append({"plan": label, "mmlu": m["acc"], **checks,
                     "fails": fails, "verdict": "BLOCK" if fails else "PASS"})
        print(f"  {label:22s} ppl x{checks['ppl_ratio']:.3f}  MMLU "
              f"{m['acc']*100:5.1f}% ({-checks['mmlu_drop_pts']:+.1f})  agree "
              f"{checks['shadow_agree']*100:5.1f}%  -> {rows[-1]['verdict']}")
    F["F"] = {"baseline_mmlu": base_mmlu, "rows": rows, "mmlu_n": MMLU_N,
              "gate": GATE}


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

    lv = f["A"]["levels"]
    ax[0].vlines(lv, 0, 1, color="#2471a3", lw=1.6, label="E2M1 (FP4)")
    lin = [-6 + i * 12 / 15 for i in range(16)]
    ax[0].vlines(lin, -1, 0, color="#c0392b", lw=1.2,
                 label="int4 (16 uniform levels)")
    ax[0].set_ylim(-1.4, 1.4)
    ax[0].set_yticks([])
    ax[0].set_xlabel("value")
    ax[0].set_title("A. E2M1: 15 values, unevenly spaced")
    ax[0].legend(fontsize=7)

    rows = f["B"]["rows"]
    for fmt, c, m in (("e8m0", "#c0392b", "o"), ("e8m0-up", "#e67e22", "v"),
                      ("e4m3", "#27ae60", "s"), ("fp32", "#34495e", "^")):
        pts = [r for r in rows if r["scale_fmt"] == fmt]
        ax[1].plot([r["block"] for r in pts], [r["ppl_ratio"] for r in pts],
                   m + "-", color=c, label=f"scale = {fmt}")
    pts = f["B"]["int4"]
    ax[1].plot([r["group"] for r in pts], [r["ppl_ratio"] for r in pts], "d--",
               color="#e67e22", label="int4 (uniform grid)")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("block size (weights per shared scale)")
    ax[1].set_ylabel("perplexity / bf16")
    ax[1].set_title("B. the scale format, not the block size")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)

    c = f["C"]["rows"]
    ax[2].scatter([r["bits"] for r in c], [r["ppl_ratio"] for r in c], s=60,
                  color="#8e44ad")
    for r in c:
        ax[2].annotate(r["plan"].split(" (")[0], (r["bits"], r["ppl_ratio"]),
                       fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax[2].axhline(1.0, color="k", ls="--", lw=1)
    ax[2].set_xlabel("effective bits per weight")
    ax[2].set_ylabel("perplexity / bf16")
    ax[2].set_title("C. named formats")
    ax[2].grid(alpha=0.3)

    e = f["E"]["block_stats"]
    ax[3].plot([r["block"] for r in e], [r["rel_mse"] for r in e], "o-",
               color="#2471a3", label="relative reconstruction MSE")
    ax3b = ax[3].twinx()
    ax3b.plot([r["block"] for r in e], [r["p99_outlier_ratio"] for r in e],
              "s--", color="#c0392b", label="p99 max/median in a block")
    ax[3].set_xscale("log", base=2)
    ax[3].set_xlabel("block size")
    ax[3].set_ylabel("relative MSE")
    ax3b.set_ylabel("p99 within-block outlier ratio")
    ax[3].set_title("E. block size barely moves the error")
    ax[3].grid(alpha=0.3)
    ax[3].legend(fontsize=6, loc="upper left")
    ax3b.legend(fontsize=6, loc="lower right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fp4_deployment.png"), dpi=110)
    print("wrote outputs/fp4_deployment.png")


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
    print(f"bf16-equivalent baseline ppl {base['ppl']:.4f}")
    F["setup"] = {"model": Q.MODEL_ID, "threads": Q.N_THREADS, "window": WIN,
                  "eval_windows": EVAL_N, "base_ppl": base["ppl"],
                  "mmlu_n": MMLU_N}

    section_a()
    section_b(model, ev, ref, base["ppl"])
    stats = Q.act_stats(model, calib, sample_rows=96)
    awq4 = section_c(model, ev, ref, base["ppl"], stats)
    section_d()
    section_e(model, ev, ref, base["ppl"], awq4)
    section_f(tok, model, ev, ref, base["ppl"], awq4)

    F["wall_s"] = round(time.time() - t0, 1)
    Q.save_findings(os.path.join(OUT, "findings.json"), F)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
