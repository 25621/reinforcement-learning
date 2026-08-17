"""Project 39 -- FlashDecoding ablation.

vLLM exposes FlashDecoding behind a flag; this machine's GPU cannot run vLLM
(see the README), so the flag is ours: enginelib ships both decode-attention
kernels and we switch between them.

  python3 run.py          # full run, ~6 minutes
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
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "37-roofline-plot-for-your-engine"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import enginelib as E  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

BATCHES = [1, 2, 4, 8, 16, 32]
CTXS = [512, 1024, 2048, 4096]
SPLITS = [1, 2, 4, 8, 16, 32]


# ------------------------------------------------------- the naive baseline
@triton.jit
def k_attn_scores(
    q_ptr, kc_ptr, s_ptr, len_ptr,
    S, H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, scale,
    QS: tl.constexpr, BN: tl.constexpr,
):
    """Pass 1 of textbook attention: compute every score and WRITE IT OUT."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    kvh = h // (H // KVH)
    L = tl.load(len_ptr) + 1
    d = tl.arange(0, HD)
    rn = blk * BN + tl.arange(0, BN)
    q = tl.load(q_ptr + b * QS + h * HD + d)
    kbase = (b * KVH + kvh) * S * HD
    k = tl.load(kc_ptr + kbase + rn[:, None] * HD + d[None, :],
                mask=(rn < L)[:, None], other=0.0)
    tl.store(s_ptr + bh * S + rn, tl.sum(q[None, :] * k, 1) * scale, mask=rn < L)


@triton.jit
def k_attn_softmax_av(
    s_ptr, vc_ptr, o_ptr, len_ptr,
    S, H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr,
    BN: tl.constexpr,
):
    """Pass 2: read the scores back, softmax them, weight the values."""
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    kvh = h // (H // KVH)
    L = tl.load(len_ptr) + 1
    d = tl.arange(0, HD)
    kbase = (b * KVH + kvh) * S * HD
    m = -float("inf")
    for s0 in range(0, L, BN):
        rn = s0 + tl.arange(0, BN)
        s = tl.load(s_ptr + bh * S + rn, mask=rn < L, other=-float("inf"))
        m = tl.maximum(m, tl.max(s, 0))
    acc = tl.zeros((HD,), tl.float32)
    l = 0.0
    for s0 in range(0, L, BN):
        rn = s0 + tl.arange(0, BN)
        s = tl.load(s_ptr + bh * S + rn, mask=rn < L, other=-float("inf"))
        p = tl.exp(s - m)
        p = tl.where(rn < L, p, 0.0)
        v = tl.load(vc_ptr + kbase + rn[:, None] * HD + d[None, :],
                    mask=(rn < L)[:, None], other=0.0)
        acc += tl.sum(p[:, None] * v, 0)
        l += tl.sum(p, 0)
    tl.store(o_ptr + b * (H * HD) + h * HD + d, acc / l)


# ------------------------------------------------------------------ helpers
def attn_variants(eng, B: int, ctx: int, li: int = 0):
    """Three ways to compute the same decode attention, as callables."""
    c = eng.cfg
    hd, H, KVH = c.head_dim, c.n_heads, c.n_kv_heads
    scale = 1.0 / math.sqrt(hd)
    kc, vc = eng.kc[li], eng.vc[li]
    S = eng.S
    scores = eng.scores

    def plain():
        E.k_attn_decode[(B * H,)](eng.qkv, kc, vc, eng.ao, eng.seqlen,
                                  S, H, KVH, hd, scale, QS=c.qkv_out, BN=32)

    def split(ns):
        def f():
            E.k_attn_decode_split[(B * H, ns)](
                eng.qkv, kc, vc, eng.part, eng.ml, eng.seqlen,
                S, H, KVH, hd, scale, QS=c.qkv_out, NSPLIT=ns, BN=32)
            E.k_attn_combine[(B * H,)](eng.part, eng.ml, eng.ao, H, hd, NSPLIT=ns)
        return f

    def materialised():
        k_attn_scores[(B * H, triton.cdiv(ctx, 128))](
            eng.qkv, kc, scores, eng.seqlen, S, H, KVH, hd, scale,
            QS=c.qkv_out, BN=128)
        k_attn_softmax_av[(B * H,)](scores, vc, eng.ao, eng.seqlen,
                                    S, H, KVH, hd, BN=128)

    return plain, split, materialised


def kv_bytes(eng, B: int, ctx: int) -> int:
    c = eng.cfg
    return 2 * B * c.n_kv_heads * ctx * c.head_dim * 4


def time_fn(fn, inner: int = 20, reps: int = 8) -> float:
    g = E.Graph(lambda: [fn() for _ in range(inner)], warmup=2)
    try:
        return E.gpu_time(g.replay, reps=reps) / inner
    finally:
        g.close()


# ---------------------------------------------------------------- sections
def check(eng) -> dict:
    """All three variants must agree, or none of the timings mean anything."""
    B, ctx = 2, 512
    eng.set_len(ctx - 1)
    plain, split, mat = attn_variants(eng, B, ctx)
    out = {}
    ref = None
    for name, fn in (("plain", plain), ("split8", split(8)), ("materialised", mat)):
        fn()
        torch.cuda.synchronize()
        v = eng.ao[:B * eng.cfg.n_heads * eng.cfg.head_dim].cpu().clone()
        if ref is None:
            ref = v
            out[name] = 0.0
        else:
            out[name] = float((v - ref).abs().max() / ref.abs().max())
    return out


def kernel_grid(eng) -> list:
    rows = []
    for ctx in CTXS:
        for B in BATCHES:
            eng.set_len(ctx - 1)
            plain, split, mat = attn_variants(eng, B, ctx)
            by = kv_bytes(eng, B, ctx)
            t_plain = time_fn(plain)
            t_split = time_fn(split(8))
            t_mat = time_fn(mat)
            sc = B * eng.cfg.n_heads * ctx * 4
            rows.append({
                "ctx": ctx, "batch": B, "kv_bytes": by,
                "plain_us": t_plain * 1e3, "split_us": t_split * 1e3,
                "mat_us": t_mat * 1e3,
                "plain_gbs": by / t_plain / 1e6, "split_gbs": by / t_split / 1e6,
                "mat_gbs": (by + 3 * sc) / t_mat / 1e6,
                "score_bytes": sc,
                "programs_plain": B * eng.cfg.n_heads,
                "programs_split": B * eng.cfg.n_heads * 8,
                "speedup": t_plain / t_split,
                "mat_penalty": t_mat / t_split,
            })
            print(f"  ctx={ctx:5d} B={B:3d}  plain {t_plain*1e3:7.1f} us  "
                  f"split8 {t_split*1e3:7.1f} us  mat {t_mat*1e3:7.1f} us  "
                  f"speedup {t_plain/t_split:5.2f}x")
    return rows


def split_sweep(eng) -> list:
    rows = []
    for ctx in (1024, 4096):
        for B in (1, 8, 32):
            eng.set_len(ctx - 1)
            _, split, _ = attn_variants(eng, B, ctx)
            by = kv_bytes(eng, B, ctx)
            for ns in SPLITS:
                t = time_fn(split(ns))
                rows.append({"ctx": ctx, "batch": B, "nsplit": ns,
                             "us": t * 1e3, "gbs": by / t / 1e6,
                             "programs": B * eng.cfg.n_heads * ns})
            best = min((r for r in rows if r["ctx"] == ctx and r["batch"] == B),
                       key=lambda r: r["us"])
            print(f"  ctx={ctx:5d} B={B:3d}  best NSPLIT={best['nsplit']:3d} "
                  f"({best['us']:.1f} us, {best['programs']} programs)")
    return rows


def end_to_end(eng) -> list:
    rows = []
    for ctx in (1024, 4096):
        for B in (1, 8, 32):
            eng.set_len(ctx - 1)
            gs = E.Graph(lambda: eng.decode_step(B, split=True, advance=False))
            gp = E.Graph(lambda: eng.decode_step(B, split=False, advance=False))
            res = E.interleaved({"split": gs.replay, "plain": gp.replay},
                                reps=5, inner=10)
            gs.close()
            gp.close()
            rows.append({"ctx": ctx, "batch": B,
                         "split_ms": res["split"], "plain_ms": res["plain"],
                         "split_tok_s": B / res["split"] * 1e3,
                         "plain_tok_s": B / res["plain"] * 1e3,
                         "speedup": res["plain"] / res["split"]})
            print(f"  ctx={ctx:5d} B={B:3d}  step {res['plain']:6.3f} -> "
                  f"{res['split']:6.3f} ms  ({res['plain']/res['split']:.3f}x)")
    return rows


# ---------------------------------------------------------------- plotting
def plot(f: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    cols = {512: "#e8b04b", 1024: "#c0392b", 2048: "#7d3c98", 4096: "#1f6f8b"}

    a = ax[0]
    for ctx in CTXS:
        rs = [r for r in f["grid"] if r["ctx"] == ctx]
        a.plot([r["batch"] for r in rs], [r["speedup"] for r in rs], "o-",
               color=cols[ctx], label=f"context {ctx}")
    a.axhline(1.0, color="0.4", ls="--", lw=1)
    a.set_xscale("log", base=2)
    a.set_xlabel("batch size")
    a.set_ylabel("speedup of FlashDecoding over one-program-per-head")
    a.set_title("The split pays where there are too few programs")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both", lw=0.4)

    a = ax[1]
    styles = {1: "-", 8: "--", 32: ":"}
    for ctx in (1024, 4096):
        for B in (1, 8, 32):
            rs = [r for r in f["splits"] if r["ctx"] == ctx and r["batch"] == B]
            a.plot([r["nsplit"] for r in rs], [r["gbs"] for r in rs],
                   styles[B], marker="o", ms=3.5, color=cols[ctx],
                   label=f"ctx {ctx}, B={B}")
    a.set_xscale("log", base=2)
    a.set_xlabel("number of splits along the KV length")
    a.set_ylabel("achieved GB/s on the KV cache")
    a.set_title("How many pieces? Until the SMs are full")
    a.legend(fontsize=7, ncol=2)
    a.grid(alpha=0.25, which="both", lw=0.4)

    a = ax[2]
    rs = [r for r in f["grid"] if r["ctx"] == 4096]
    x = range(len(rs))
    w = 0.27
    a.bar([i - w for i in x], [r["plain_us"] for r in rs], w, label="one program per head",
          color="#e8b04b")
    a.bar(list(x), [r["split_us"] for r in rs], w, label="FlashDecoding (8 splits)",
          color="#c0392b")
    a.bar([i + w for i in x], [r["mat_us"] for r in rs], w,
          label="materialised scores", color="#1f6f8b")
    a.set_xticks(list(x))
    a.set_xticklabels([f"B={r['batch']}" for r in rs])
    a.set_yscale("log")
    a.set_ylabel("microseconds per attention call")
    a.set_title("Three kernels, context 4096")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, axis="y", which="both", lw=0.4)

    fig.suptitle("Project 39 - FlashDecoding: splitting the KV length across programs",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "flashdecoding.png"), dpi=125)
    print("wrote", os.path.join(OUT, "flashdecoding.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(OUT, "findings.json")
    if args.plot:
        plot(json.load(open(path)))
        return

    cfg = E.Config()
    eng = E.Engine(cfg, max_batch=max(BATCHES), max_seq=max(CTXS),
                   max_tokens=max(BATCHES))
    eng.nsplit = 8
    # scratch for the materialised-scores baseline: one score per (seq, head, pos)
    eng.scores = torch.empty(max(BATCHES) * cfg.n_heads * max(CTXS), device=E.DEV)
    eng.part = torch.empty(max(BATCHES) * cfg.n_heads * 32 * cfg.head_dim, device=E.DEV)
    eng.ml = torch.empty(max(BATCHES) * cfg.n_heads * 32 * 2, device=E.DEV)
    # Fill layer 0's cache and the query buffer with real numbers.  Fresh
    # cudaMalloc'd memory is whatever was there before, and attention on
    # garbage can produce inf - inf = NaN, which would make section A
    # meaningless (and did, the first time this was run).
    g = torch.Generator().manual_seed(7)
    eng.kc[0].copy_(torch.randn(eng.kc[0].numel(), generator=g) * 0.3)
    eng.vc[0].copy_(torch.randn(eng.vc[0].numel(), generator=g) * 0.3)
    eng.qkv.copy_(torch.randn(eng.qkv.numel(), generator=g) * 0.3)
    eng.ao.copy_(torch.zeros(eng.ao.numel()))

    print("A. all three variants agree?")
    agree = check(eng)
    print("  ", agree)

    print("B. attention kernel, isolated")
    grid = kernel_grid(eng)

    print("C. how many splits?")
    splits = split_sweep(eng)

    print("D. end to end")
    e2e = end_to_end(eng)

    f = {"device": E.device_info(), "agreement": agree, "grid": grid,
         "splits": splits, "end_to_end": e2e,
         "model": {"n_heads": cfg.n_heads, "n_kv_heads": cfg.n_kv_heads,
                   "head_dim": cfg.head_dim, "n_layers": cfg.n_layers}}
    json.dump(f, open(path, "w"), indent=1)
    print("wrote", path)
    plot(f)


if __name__ == "__main__":
    main()
