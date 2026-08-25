"""Project 37 -- Roofline plot for your engine.

Sweep batch size and prompt length through the Triton engine in enginelib.py,
measure the card's two ceilings, and place every operating point on a roofline.

  python3 run.py          # full run, ~4 minutes
  python3 run.py --plot   # redraw the figure from the committed findings.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
import triton  # noqa: E402

import enginelib as E  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
CTX = 1024
PREFILLS = [128, 256, 512, 1024, 2048, 4096]


# ---------------------------------------------------------------- ceilings
def measure_ceilings() -> dict:
    """The two numbers every roofline needs: how fast this card can move bytes,
    and how fast it can multiply floats.  Both measured, not read off a spec."""
    n = 1 << 24                      # 64 MiB in, 64 MiB out -- far beyond L2
    a = torch.empty(n, device=E.DEV)
    b = torch.empty(n, device=E.DEV)
    a.copy_(torch.randn(n))
    grid = (triton.cdiv(n, 1024),)
    t = E.gpu_time(lambda: E.k_copy[grid](a, b, n, BLOCK=1024), reps=50)
    bw = 2 * n * 4 / t / 1e6         # GB/s

    m = 2048
    x = torch.empty(m * m, device=E.DEV)
    w = torch.empty(m * m, device=E.DEV)
    y = torch.empty(m * m, device=E.DEV)
    x.copy_(torch.randn(m * m) * 0.1)
    w.copy_(torch.randn(m * m) * 0.1)
    best = 0.0
    for BM, BN, BK, nw in ((128, 128, 32, 8), (128, 64, 32, 4), (64, 64, 32, 4)):
        g = (triton.cdiv(m, BM) * triton.cdiv(m, BN),)
        tt = E.gpu_time(lambda: E.k_matmul[g](x, w, y, m, m, m, BM=BM, BN=BN,
                                              BK=BK, GROUP_M=8, num_warps=nw), reps=20)
        best = max(best, 2 * m ** 3 / tt / 1e6)
    info = E.device_info()
    return {
        "device": info,
        "copy_gbs": bw,
        "gemm_gflops": best,
        "ridge_flops_per_byte": best / bw,
        "pct_spec_bw": 100 * bw / info["spec_bw_gbs"],
        "pct_peak_flops": 100 * best / info["peak_fp32_gflops"],
    }


# ---------------------------------------------------------------- sweeps
def sweep_decode(ceil: dict) -> list:
    cfg = E.Config()
    eng = E.Engine(cfg, max_batch=max(BATCHES), max_seq=CTX, max_tokens=max(BATCHES))
    rows = []
    for B in BATCHES:
        eng.set_len(CTX - 1)
        g = E.Graph(lambda: eng.decode_step(B, advance=False))
        ms = E.gpu_time(g.replay, reps=30)
        eng.set_len(CTX - 1)
        eager = E.wall_time(lambda: eng.decode_step(B, advance=False), reps=20)
        by = eng.decode_bytes(B, CTX)
        fl = eng.decode_flops(B, CTX)
        rows.append({
            "phase": "decode", "batch": B, "ctx": CTX, "ms": ms, "eager_ms": eager,
            "tok_s": B / ms * 1e3, "bytes": by, "flops": fl,
            "ai": fl / by, "gflops": fl / ms / 1e6, "gbs": by / ms / 1e6,
            "pct_bw_ceiling": 100 * (by / ms / 1e6) / ceil["copy_gbs"],
            "pct_flop_ceiling": 100 * (fl / ms / 1e6) / ceil["gemm_gflops"],
        })
        print(f"  decode B={B:3d}  {ms:7.3f} ms  {B/ms*1e3:8.0f} tok/s  "
              f"AI={fl/by:6.2f}  {by/ms/1e6:6.0f} GB/s")
        del g
    del eng
    torch.cuda.empty_cache()
    return rows


def sweep_prefill(ceil: dict) -> list:
    cfg = E.Config()
    eng = E.Engine(cfg, max_batch=1, max_seq=max(PREFILLS), max_tokens=max(PREFILLS))
    rows = []
    for T in PREFILLS:
        ms = E.gpu_time(lambda: eng.prefill(1, T), reps=5)
        by = eng.prefill_bytes(1, T)
        fl = eng.prefill_flops(1, T)
        rows.append({
            "phase": "prefill", "batch": 1, "ctx": T, "ms": ms, "eager_ms": ms,
            "tok_s": T / ms * 1e3, "bytes": by, "flops": fl,
            "ai": fl / by, "gflops": fl / ms / 1e6, "gbs": by / ms / 1e6,
            "pct_bw_ceiling": 100 * (by / ms / 1e6) / ceil["copy_gbs"],
            "pct_flop_ceiling": 100 * (fl / ms / 1e6) / ceil["gemm_gflops"],
        })
        print(f"  prefill T={T:5d} {ms:8.2f} ms  {T/ms*1e3:8.0f} tok/s  "
              f"AI={fl/by:7.1f}  {fl/ms/1e6:6.0f} GFLOP/s")
    del eng
    torch.cuda.empty_cache()
    return rows


def sweep_context() -> list:
    """Same batch, growing KV cache: how much of decode is the cache?"""
    cfg = E.Config()
    B = 8
    rows = []
    eng = E.Engine(cfg, max_batch=B, max_seq=4096, max_tokens=B)
    for ctx in (128, 512, 1024, 2048, 4096):
        eng.set_len(ctx - 1)
        g = E.Graph(lambda: eng.decode_step(B, advance=False))
        ms = E.gpu_time(g.replay, reps=30)
        wb = cfg.weight_bytes()
        kvb = B * ctx * cfg.kv_bytes_per_token()
        rows.append({"ctx": ctx, "batch": B, "ms": ms, "tok_s": B / ms * 1e3,
                     "weight_bytes": wb, "kv_bytes": kvb,
                     "kv_share": kvb / (wb + kvb)})
        print(f"  ctx={ctx:5d}  {ms:7.3f} ms  KV is {100*kvb/(wb+kvb):4.1f}% of bytes")
        del g
    del eng
    torch.cuda.empty_cache()
    return rows


# ------------------------------------------------------- the buying decision
REAL_GPUS = {
    # name: (HBM GB/s, dense BF16 TFLOP/s, HBM GB)
    "A100 80GB": (2039, 312, 80),
    "H100 SXM": (3350, 989, 80),
    "H200 SXM": (4800, 989, 141),
    "B200": (8000, 2250, 192),
}
REAL_MODELS = {"Llama-3-8B": 8.03e9, "Llama-3-70B": 70.6e9, "Qwen2.5-72B": 72.7e9}


def buying_table() -> list:
    """The single-stream decode formula, applied to hardware we do not have.

    tok/s = HBM bandwidth / (bytes per parameter x parameters).  It is nothing
    more than "one token needs every weight read once", which is exactly what
    section B verifies on the card we do have.
    """
    rows = []
    for m, p in REAL_MODELS.items():
        for g, (bw, tf, cap) in REAL_GPUS.items():
            fits = p * 2 / 1e9 < 0.9 * cap
            rows.append({"model": m, "gpu": g, "params": p, "bw_gbs": bw,
                         "bf16_tflops": tf, "fits_bf16": fits,
                         "tok_s_bf16": bw * 1e9 / (2 * p),
                         "tok_s_fp8": bw * 1e9 / p,
                         "ridge": tf * 1e12 / (bw * 1e9)})
    return rows


# ---------------------------------------------------------------- plotting
def plot(f: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ceil = f["ceilings"]
    bw, pk = ceil["copy_gbs"], ceil["gemm_gflops"]
    ridge = ceil["ridge_flops_per_byte"]
    dec = [r for r in f["points"] if r["phase"] == "decode"]
    pre = [r for r in f["points"] if r["phase"] == "prefill"]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # -- panel 1: the roofline
    a = ax[0]
    xs = [2 ** (i / 4) for i in range(-16, 41)]
    a.plot(xs, [min(bw * x, pk) for x in xs], color="0.25", lw=2, label="roofline")
    a.axvline(ridge, color="0.7", ls=":", lw=1)
    a.annotate(f"ridge point\n{ridge:.0f} FLOP/byte", (ridge, pk * 0.055),
               fontsize=8, ha="center", color="0.4")
    a.scatter([r["ai"] for r in dec], [r["gflops"] for r in dec], s=46,
              color="#c0392b", zorder=5, label="decode (batch 1-128)")
    for r in dec:
        if r["batch"] in (1, 8, 128):
            a.annotate(f"B={r['batch']}", (r["ai"], r["gflops"]),
                       textcoords="offset points", xytext=(6, -9), fontsize=8,
                       color="#c0392b")
    a.scatter([r["ai"] for r in pre], [r["gflops"] for r in pre], s=46, marker="s",
              color="#1f6f8b", zorder=5, label="prefill (128-4096 tokens)")
    for r in pre:
        if r["ctx"] in (128, 4096):
            a.annotate(f"T={r['ctx']}", (r["ai"], r["gflops"]),
                       textcoords="offset points", xytext=(-10, -16), fontsize=8,
                       color="#1f6f8b")
    a.set_xscale("log", base=2)
    a.set_yscale("log", base=2)
    a.set_xlabel("arithmetic intensity (FLOP per byte of HBM traffic)")
    a.set_ylabel("achieved GFLOP/s")
    a.set_title(f"Roofline, measured: {bw:.0f} GB/s and {pk/1000:.1f} TFLOP/s")
    a.legend(loc="upper left", fontsize=8)
    a.grid(alpha=0.25, which="both", lw=0.4)

    # -- panel 2: what batching actually buys
    a = ax[1]
    b = [r["batch"] for r in dec]
    a.plot(b, [r["tok_s"] for r in dec], "o-", color="#c0392b", label="measured tok/s")
    a.plot(b, [dec[0]["tok_s"] * x for x in b], "--", color="0.6",
           label="perfect scaling")
    a.set_xscale("log", base=2)
    a.set_yscale("log", base=2)
    a.set_xlabel("batch size")
    a.set_ylabel("decode throughput (tokens/s)")
    a.set_title("Batching is nearly free until the cache is the traffic")
    for r in dec:
        a.annotate(f"{r['pct_bw_ceiling']:.0f}%", (r["batch"], r["tok_s"]),
                   textcoords="offset points", xytext=(4, -12), fontsize=7, color="0.35")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both", lw=0.4)

    # -- panel 3: where the bytes come from
    a = ax[2]
    ctxs = [r["ctx"] for r in f["context"]]
    kv = [100 * r["kv_share"] for r in f["context"]]
    ms = [r["ms"] for r in f["context"]]
    a.bar([str(c) for c in ctxs], kv, color="#e8b04b", label="KV cache share of bytes (%)")
    a2 = a.twinx()
    a2.plot([str(c) for c in ctxs], ms, "o-", color="#c0392b", label="ms / decode step")
    a2.set_ylabel("ms per decode step")
    a2.set_ylim(0, max(ms) * 1.35)
    a.set_ylim(0, 100)
    a.set_xlabel("context length (batch 8)")
    a.set_ylabel("KV cache share of HBM traffic (%)")
    a.set_title("The KV cache overtakes the weights")
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8)
    a.grid(alpha=0.2, axis="y", lw=0.4)

    fig.suptitle("Project 37 - roofline of a Triton inference engine on a GTX 1070 Ti",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "roofline.png"), dpi=125)
    print("wrote", os.path.join(OUT, "roofline.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(OUT, "findings.json")
    if args.plot:
        plot(json.load(open(path)))
        return

    print("A. ceilings")
    ceil = measure_ceilings()
    print(f"   copy      {ceil['copy_gbs']:.1f} GB/s  ({ceil['pct_spec_bw']:.0f}% of spec)")
    print(f"   gemm      {ceil['gemm_gflops']/1000:.2f} TFLOP/s "
          f"({ceil['pct_peak_flops']:.0f}% of peak)")
    print(f"   ridge     {ceil['ridge_flops_per_byte']:.1f} FLOP/byte")

    cfg = E.Config()
    print("B. decode sweep")
    pts = sweep_decode(ceil)
    print("C. prefill sweep")
    pts += sweep_prefill(ceil)
    print("D. context sweep")
    ctx = sweep_context()

    f = {
        "model": {"params": cfg.n_params(), "weight_bytes": cfg.weight_bytes(),
                  "kv_bytes_per_token": cfg.kv_bytes_per_token(),
                  "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                  "n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
                  "d_ff": cfg.d_ff, "vocab": cfg.vocab},
        "ceilings": ceil,
        "points": pts,
        "context": ctx,
        "hardware": buying_table(),
        "formula_check": {
            "predicted_tok_s": ceil["copy_gbs"] * 1e9 / cfg.weight_bytes(),
            "measured_tok_s": pts[0]["tok_s"],
            # As batch grows the KV bytes grow with it, so AI stops climbing:
            # limit of 2*B*P / (4*P + B*ctx*kv_per_token) as B -> infinity.
            "ai_ceiling_at_ctx": 2 * cfg.n_params() / (CTX * cfg.kv_bytes_per_token()),
        },
    }
    json.dump(f, open(path, "w"), indent=1)
    print("wrote", path)
    plot(f)


if __name__ == "__main__":
    main()
