#!/usr/bin/env python3
"""Project 71 — FP4 inference: the activation half.

  A. Why activations are the hard half — measured outlier ratios per linear.
  B. The ladder: FP8 W8A8, FP4 weights alone, then 4-bit activations at four
     granularities.  All perplexity, all measured.
  C. Rotation (QuaRot / SpinQuant): spread the outlier before quantising it.
  D. Which layer breaks: 4-bit activations enabled one group at a time.
  E. What FP4 buys on Blackwell — memory measured, throughput arithmetic.
  F. The quality gate from project 35, applied.

  python3 run.py           # ~5 minutes
  python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import fp4act as FA                                            # noqa: E402
import quantlib as Q                                           # noqa: E402
import fp4 as F36                                              # noqa: E402

OUT = os.path.join(HERE, "outputs")
FINDINGS = os.path.join(OUT, "findings.json")
N_CHUNKS, CHUNK = 6, 512
GATE_PPL = 1.05           # project 35's gate: no more than 5% perplexity loss


def ppl(model, chunks, label, t0=None):
    t = time.time()
    v = Q.perplexity(model, chunks)
    print(f"  {label:<34} ppl {v:8.3f}   ({time.time() - t:.1f}s)", flush=True)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        with open(FINDINGS) as f:
            plot(json.load(f))
        return

    print("loading Qwen2.5-0.5B-Instruct ...", flush=True)
    tok, model = Q.load()
    torch.set_num_threads(6)
    text = Q.corpora(tok, names=("wiki",))["wiki"]
    chunks = Q.token_chunks(tok, text, chunk=CHUNK, n=N_CHUNKS)
    calib = Q.token_chunks(tok, text, chunk=CHUNK, n=4, skip=N_CHUNKS)

    base = ppl(model, chunks, "fp32 baseline")

    # ---------------------------------------------------------------- A ----
    print("\nA. activation statistics ...", flush=True)
    stats = Q.act_stats(model, calib)
    A = {"per_linear": {}}
    for name, st in stats.items():
        mx = st["absmax"]
        med = mx.median().item()
        A["per_linear"][name] = dict(
            group=Q.group_of(name),
            absmax=round(mx.max().item(), 3),
            median_channel=round(med, 5),
            outlier_ratio=round(mx.max().item() / max(med, 1e-9), 1))
    worst = max(A["per_linear"].items(), key=lambda kv: kv[1]["outlier_ratio"])
    by_group = {}
    for name, d in A["per_linear"].items():
        by_group.setdefault(d["group"], []).append(d["outlier_ratio"])
    A["by_group"] = {g: round(sum(v) / len(v), 1) for g, v in by_group.items()}
    A["worst"] = dict(name=worst[0], **worst[1])
    print(f"   worst channel ratio: {worst[0]} at {worst[1]['outlier_ratio']}x",
          flush=True)

    # ---------------------------------------------------------------- B ----
    print("\nB. the ladder (weights x activations) ...", flush=True)
    B = {"fp32": dict(ppl=round(base, 3), x=1.0, w_bits=16, a_bits=16)}

    def add(label, value, w_bits, a_bits, note=""):
        B[label] = dict(ppl=round(value, 3), x=round(value / base, 4),
                        w_bits=w_bits, a_bits=a_bits, note=note)

    # W8A8: fp8 weights + fp8 per-token activations
    saved = {n: m.weight.data.clone() for n, m in Q.block_linears(model).items()}
    with torch.no_grad():
        for n, m in Q.block_linears(model).items():
            s = m.weight.data.abs().amax() / 448.0
            m.weight.data.copy_(Q.fake_quant_fp8(m.weight.data, "e4m3", s))
    with Q.ActQuant(model, bits=8, mode="per-token"):
        add("W8A8 (fp8 w, int8 per-token a)", ppl(model, chunks, "W8A8"), 8, 8)
    with torch.no_grad():
        for n, m in Q.block_linears(model).items():
            m.weight.data.copy_(saved[n])

    # FP4 weights alone, then with activations at 8 and 4 bits
    with F36.FP4Weights(model, block=32, scale_fmt="e8m0"):
        add("W4A16 (MXFP4 weights)", ppl(model, chunks, "W4A16 mxfp4"), 4, 16)
        with Q.ActQuant(model, bits=8, mode="per-token"):
            add("W4A8 (int8 per-token a)", ppl(model, chunks, "W4A8"), 4, 8)
        for label, kw in (
                ("W4A4 per-tensor", dict(mode="per-tensor")),
                ("W4A4 per-token", dict(mode="per-token")),
                ("W4A4 MX block-32", dict(mode="block-fp4", block=32,
                                          scale_fmt="e8m0")),
                ("W4A4 NVFP4 block-16", dict(mode="block-fp4", block=16,
                                             scale_fmt="e4m3"))):
            with FA.ActFP4(model, bits=4, **kw):
                add(label, ppl(model, chunks, label), 4, 4)

        # ------------------------------------------------------------ C ----
        print("\nC. rotation ...", flush=True)
        C = {}
        for label, kw in (
                ("W4A4 per-token + rotation", dict(mode="per-token")),
                ("W4A4 MX block-32 + rotation", dict(mode="block-fp4", block=32,
                                                     scale_fmt="e8m0")),
                ("W4A4 NVFP4 block-16 + rotation",
                 dict(mode="block-fp4", block=16, scale_fmt="e4m3"))):
            with FA.ActFP4(model, bits=4, rotate=True, **kw):
                v = ppl(model, chunks, label)
            add(label, v, 4, 4, note="rotated")
            plain = label.replace(" + rotation", "")
            C[label] = dict(ppl=round(v, 3), x=round(v / base, 4),
                            without=B[plain]["x"],
                            gain=round(B[plain]["x"] / (v / base), 3))

        # ------------------------------------------------------------ D ----
        print("\nD. which linear breaks first ...", flush=True)
        D = {}
        groups = sorted({Q.group_of(n) for n in Q.block_linears(model)})
        for g in groups:
            names = [n for n in Q.block_linears(model) if Q.group_of(n) == g]
            # deliberately the *cheap* granularity here: per-token 4-bit is bad
            # enough that each linear's contribution is visible.  With block
            # scales the differences would be inside the noise.
            with FA.ActFP4(model, bits=4, mode="per-token", names=names):
                v = ppl(model, chunks, f"A4 only on {g}")
            D[g] = dict(ppl=round(v, 3), x=round(v / base, 4),
                        outlier_ratio=A["by_group"].get(g))
        # and the reverse: everything except the worst group
        worst_g = max(D, key=lambda g: D[g]["x"])
        names = [n for n in Q.block_linears(model) if Q.group_of(n) != worst_g]
        with FA.ActFP4(model, bits=4, mode="per-token", names=names):
            v = ppl(model, chunks, f"A4 everywhere except {worst_g}")
        D["_all_but_worst"] = dict(group=worst_g, ppl=round(v, 3),
                                   x=round(v / base, 4))

    # ---------------------------------------------------------------- E ----
    sh = "Qwen2.5-0.5B"
    E = dict(
        sizes={b: Q.size_report(sh, w_bits=b, head_bits=16, group=32)
               for b in (16, 8, 4)},
        kv_bytes={b: Q.kv_bytes_per_token(sh, bits=b) for b in (16, 8, 4)},
        blackwell=dict(
            note="arithmetic from published specs, not measured here",
            b200_fp8_pflops=4.5, b200_fp4_pflops=9.0,
            h100_fp8_pflops=1.98, hbm_tb_s=8.0,
            fp4_over_fp8_flops=2.0),
    )

    # ---------------------------------------------------------------- F ----
    F_ = {k: dict(v, passes=bool(v["x"] <= GATE_PPL)) for k, v in B.items()}

    out = dict(A=A, B=B, C=C, D=D, E=E, F=F_, base_ppl=round(base, 3),
               gate=GATE_PPL, n_tokens=N_CHUNKS * CHUNK)
    os.makedirs(OUT, exist_ok=True)
    with open(FINDINGS, "w") as f:
        json.dump(out, f, indent=1)

    print("\n--- B. the ladder -------------------------------------------------")
    for k, v in B.items():
        print(f"{k:<34} W{v['w_bits']}A{v['a_bits']:<3} ppl {v['ppl']:8.3f}  "
              f"x{v['x']:.3f}  {'PASS' if v['x'] <= GATE_PPL else 'FAIL'}")
    print("\n--- C. rotation ---------------------------------------------------")
    for k, v in C.items():
        print(f"{k:<34} x{v['x']:.3f} vs x{v['without']:.3f} without "
              f"({v['gain']}x better)")
    print("\n--- D. per-group blame --------------------------------------------")
    for g, v in D.items():
        if g.startswith("_"):
            continue
        print(f"  A4 on {g:<12} x{v['x']:.3f}   (outlier ratio "
              f"{v['outlier_ratio']}x)")
    print(f"  everything except {D['_all_but_worst']['group']}: "
          f"x{D['_all_but_worst']['x']:.3f}")
    plot(out)


def plot(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B, C, D = F["A"], F["B"], F["C"], F["D"]
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
    fig.suptitle("FP4 inference: 4-bit weights are the easy half — "
                 "4-bit activations are the frontier", fontsize=13)

    p = ax[0]
    groups = list(A["by_group"].keys())
    vals = [A["by_group"][g] for g in groups]
    p.barh(range(len(groups)), vals, color="#c0504d")
    p.set_yticks(range(len(groups)))
    p.set_yticklabels(groups, fontsize=8)
    p.set_xscale("log")
    p.set_xlabel("worst input channel / median channel")
    p.set_title(f"A. Activation outliers per linear\nworst: "
                f"{A['worst']['name'].split('.')[-1]} at "
                f"{A['worst']['outlier_ratio']}x")

    p = ax[1]
    ks = [k for k in B if k != "fp32"]
    xs = [B[k]["x"] for k in ks]
    cols = ["#4c9f70" if x <= F["gate"] else
            "#e0a458" if x < 1.5 else "#c0504d" for x in xs]
    p.barh(range(len(ks)), xs, color=cols)
    p.axvline(1.0, color="k", lw=1)
    p.axvline(F["gate"], color="#4c9f70", ls="--", lw=1,
              label=f"quality gate x{F['gate']}")
    for i, (k, x) in enumerate(zip(ks, xs)):
        p.text(x, i, f"  x{x:.3f}", va="center", fontsize=7)
    p.set_yticks(range(len(ks)))
    p.set_yticklabels(ks, fontsize=7)
    p.set_xscale("log")
    p.set_xlabel("perplexity relative to fp32 (lower is better)")
    p.set_title("B/C. The ladder, and what rotation fixes")
    p.legend(fontsize=8)

    p = ax[2]
    gs = [g for g in D if not g.startswith("_")]
    xs = [D[g]["x"] for g in gs]
    rs = [D[g]["outlier_ratio"] or 0 for g in gs]
    p.scatter(rs, xs, s=60, color="#4a6fa5")
    for g, r, x in zip(gs, rs, xs):
        p.annotate(g, (r, x), textcoords="offset points", xytext=(5, 4),
                   fontsize=8)
    p.set_xscale("log")
    p.set_yscale("log")
    p.set_xlabel("activation outlier ratio of that linear")
    p.set_ylabel("perplexity cost when only it runs A4")
    p.set_title("D. The outlier ratio predicts the damage")
    p.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path = os.path.join(OUT, "fp4_activations.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


if __name__ == "__main__":
    main()
