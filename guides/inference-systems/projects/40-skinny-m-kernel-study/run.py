"""Project 40 -- Skinny-M kernel study.

The decode GEMM has M = batch (1-128) and K = N = thousands.  Compare cuBLAS,
a generic Triton GEMM, a decode-tuned Triton GEMM, a split-K variant, and an
int4-weight kernel (the Marlin idea), on exactly that shape.

  python3 run.py          # full run, ~6 minutes (compiles skinny.cu first)
  python3 run.py --plot   # redraw the figure from the committed findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "37-roofline-plot-for-your-engine"))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import triton  # noqa: E402

import enginelib as E  # noqa: E402
import kernels40 as K4  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

MS = [1, 2, 4, 8, 16, 32, 64, 128]
SHAPES = [(8192, 8192), (1024, 1024)]
GROUP = 128   # group-128 int4, the standard serving recipe


# The decode kernels keep BM = 16 always: that is what "the decode path" means,
# and above M = 16 they simply re-read the weights once per 16 rows.  Watching
# that happen is the point of section D.
BM_DECODE = 16


def time_fn(fn, inner: int = 5, reps: int = 8) -> float:
    """Graph-replay timing: no Python in the measured loop."""
    g = E.Graph(lambda: [fn() for _ in range(inner)], warmup=2)
    try:
        return E.gpu_time(g.replay, reps=reps) / inner
    finally:
        g.close()


# ---------------------------------------------------------------- cuBLAS
def run_cublas() -> list:
    exe = os.path.join(OUT, "skinny")
    src = os.path.join(HERE, "skinny.cu")
    if not os.path.exists(exe) or os.path.getmtime(src) > os.path.getmtime(exe):
        print("   compiling skinny.cu ...")
        subprocess.run(["nvcc", "-O3", "-arch=sm_61", src, "-o", exe, "-lcublas"],
                       check=True)
    out = subprocess.run([exe], check=True, capture_output=True, text=True).stdout
    rows = []
    for line in out.strip().splitlines()[1:]:
        k, m, kk, n, ms, gf, gb = line.split(",")
        rows.append({"kernel": k, "m": int(m), "k": int(kk), "n": int(n),
                     "ms": float(ms), "gflops": float(gf), "gbs": float(gb)})
    return rows


INT4_CONFIGS = [(64, 4), (128, 4), (256, 2), (256, 4), (512, 2), (512, 4)]


def tune_int4(x, wq, sc, y, Kd, N, nbytes, kind: str, M: int = 1) -> tuple:
    """Pick (BN, num_warps) by measuring, not by guessing.

    Every real quantised kernel ships an autotuner; this is the smallest
    honest version of one, and the spread it finds is the point of section E.
    The GEMV and the GEMM are tuned separately -- they do not want the same
    shape, and using one's winner for the other cost 9x here.
    """
    best, table = None, []
    for BN, nw in INT4_CONFIGS:
        if BN > N:
            continue

        def f(BN=BN, nw=nw):
            if kind == "gemv":
                K4.k_gemv_int4[(triton.cdiv(N, BN),)](x, wq, sc, y, N, Kd,
                                                      BN=BN, BK=GROUP, num_warps=nw)
            else:
                K4.k_gemm_int4[(triton.cdiv(M, 16) * triton.cdiv(N, BN),)](
                    x, wq, sc, y, M, N, Kd, BM=16, BN=BN, BK=GROUP, num_warps=nw)
        try:
            ms = time_fn(f)
        except Exception:
            continue
        table.append({"n": N, "kind": kind, "BN": BN, "num_warps": nw, "ms": ms,
                      "gbs": nbytes / ms / 1e6})
        if best is None or ms < best[2]:
            best = (BN, nw, ms)
    return best, table


# ---------------------------------------------------------------- Triton
def triton_rows(shape, tuning: list) -> list:
    Kd, N = shape
    g = torch.Generator().manual_seed(3)
    hw = torch.randn(Kd, N, generator=g) * 0.05
    w = torch.empty(Kd * N, device=E.DEV)
    w.copy_(hw.reshape(-1))
    x = torch.empty(128 * Kd, device=E.DEV)
    x.copy_(torch.randn(128 * Kd, generator=g) * 0.5)
    y = torch.empty(128 * N, device=E.DEV)
    part = torch.empty(4 * 128 * N, device=E.DEV)   # split-K partial products

    packed, scale = K4.pack_int4(hw, group=GROUP)
    wq = torch.empty(packed.numel(), device=E.DEV, dtype=torch.int32)
    wq.copy_(packed.reshape(-1))
    sc = torch.empty(scale.numel(), device=E.DEV)
    sc.copy_(scale.reshape(-1))

    fp32_bytes = Kd * N * 4
    int4_bytes = Kd * N // 2 + (Kd // GROUP) * N * 4
    (v_bn, v_nw, _), tv = tune_int4(x, wq, sc, y, Kd, N, int4_bytes, "gemv")
    (g_bn, g_nw, _), tg = tune_int4(x, wq, sc, y, Kd, N, int4_bytes, "gemm", M=16)
    tuning.extend(tv + tg)
    for kind, bn, nw, tt in (("gemv", v_bn, v_nw, tv), ("gemm", g_bn, g_nw, tg)):
        print(f"   int4 {kind} tuned to BN={bn}, num_warps={nw} "
              f"(spread {max(t['ms'] for t in tt)/min(t['ms'] for t in tt):.2f}x "
              f"across {len(tt)} configs)")
    rows = []
    for M in MS:
        BM = BM_DECODE
        _, BN, BK = E.decode_tile(N)
        acts = (M * Kd + M * N) * 4

        def f_tiled():
            grid = (triton.cdiv(M, 64) * triton.cdiv(N, 64),)
            E.k_matmul[grid](x, w, y, M, N, Kd, BM=64, BN=64, BK=32, GROUP_M=8)

        def f_skinny():
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
            E.k_matmul[grid](x, w, y, M, N, Kd, BM=BM, BN=BN, BK=BK, GROUP_M=8)

        def f_splitk(sk):
            def f():
                grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN), sk)
                K4.k_gemm_splitk[grid](x, w, part, M, N, Kd, SK=sk,
                                       BM=BM, BN=BN, BK=BK)
                K4.k_reduce_splitk[(triton.cdiv(M * N, 1024),)](
                    part, y, M * N, SK=sk, BLOCK=1024)
            return f

        def f_int4():
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, g_bn),)
            K4.k_gemm_int4[grid](x, wq, sc, y, M, N, Kd, BM=BM, BN=g_bn,
                                 BK=GROUP, num_warps=g_nw)

        def f_int4_gemv():
            K4.k_gemv_int4[(triton.cdiv(N, v_bn),)](x, wq, sc, y, N, Kd,
                                                    BN=v_bn, BK=GROUP,
                                                    num_warps=v_nw)

        cases = {"triton_tiled": (f_tiled, fp32_bytes + acts),
                 "triton_skinny": (f_skinny, fp32_bytes + acts),
                 "triton_splitk4": (f_splitk(4), fp32_bytes + acts),
                 "triton_int4": (f_int4, int4_bytes + acts)}
        if M == 1:
            cases["triton_int4_gemv"] = (f_int4_gemv, int4_bytes + acts)
        for name, (fn, nbytes) in cases.items():
            ms = time_fn(fn)
            rows.append({"kernel": name, "m": M, "k": Kd, "n": N, "ms": ms,
                         "gflops": 2 * M * Kd * N / ms / 1e6,
                         "gbs": nbytes / ms / 1e6, "bytes": nbytes,
                         "programs": triton.cdiv(M, BM) * triton.cdiv(N, BN),
                         "BM": BM, "BN": BN, "BK": BK})
            print(f"   {name:16s} M={M:4d} K=N={N:5d}  {ms*1e3:9.1f} us  "
                  f"{2*M*Kd*N/ms/1e9:7.2f} TFLOP/s  {nbytes/ms/1e6:6.1f} GB/s")
    return rows


def check(shape=(1024, 1024)) -> dict:
    """Every kernel must reproduce the same product."""
    Kd, N = shape
    M = 8
    g = torch.Generator().manual_seed(11)
    hw = torch.randn(Kd, N, generator=g) * 0.05
    hx = torch.randn(M, Kd, generator=g) * 0.5
    w = torch.empty(Kd * N, device=E.DEV)
    w.copy_(hw.reshape(-1))
    x = torch.empty(M * Kd, device=E.DEV)
    x.copy_(hx.reshape(-1))
    y = torch.empty(M * N, device=E.DEV)
    ref = hx @ hw
    out = {}

    _, BN, BK = E.decode_tile(N)
    E.k_matmul[(triton.cdiv(M, 16) * triton.cdiv(N, BN),)](
        x, w, y, M, N, Kd, BM=16, BN=BN, BK=BK, GROUP_M=8)
    torch.cuda.synchronize()
    got = y[:M * N].cpu().reshape(M, N)
    out["triton_skinny"] = float((got - ref).abs().max() / ref.abs().max())

    part = torch.empty(4 * M * N, device=E.DEV)
    K4.k_gemm_splitk[(triton.cdiv(M, 16) * triton.cdiv(N, BN), 4)](
        x, w, part, M, N, Kd, SK=4, BM=16, BN=BN, BK=BK)
    K4.k_reduce_splitk[(triton.cdiv(M * N, 1024),)](part, y, M * N, SK=4, BLOCK=1024)
    torch.cuda.synchronize()
    got = y[:M * N].cpu().reshape(M, N)
    out["triton_splitk4"] = float((got - ref).abs().max() / ref.abs().max())

    packed, scale = K4.pack_int4(hw, group=GROUP)
    wq = torch.empty(packed.numel(), device=E.DEV, dtype=torch.int32)
    wq.copy_(packed.reshape(-1))
    sc = torch.empty(scale.numel(), device=E.DEV)
    sc.copy_(scale.reshape(-1))
    K4.k_gemm_int4[(triton.cdiv(M, 16) * triton.cdiv(N, BN),)](
        x, wq, sc, y, M, N, Kd, BM=16, BN=BN, BK=GROUP)
    torch.cuda.synchronize()
    got = y[:M * N].cpu().reshape(M, N)
    deq = K4.dequantised(packed, scale, Kd, N, GROUP)
    out["triton_int4_vs_fp32"] = float((got - ref).abs().max() / ref.abs().max())
    out["triton_int4_vs_dequant"] = float((got - hx @ deq).abs().max() / ref.abs().max())
    return out


# ---------------------------------------------------------------- plotting
def plot(f: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = f["rows"]
    big = [r for r in rows if r["n"] == 8192]
    small = [r for r in rows if r["n"] == 1024]
    style = {"cublas_gemm": ("#111111", "o", "cuBLAS"),
             "triton_int4_gemv": ("#2e8b57", "*", "Triton, int4 GEMV (M=1)"),
             "triton_tiled": ("#7d3c98", "s", "Triton, prefill tiles (BM=64)"),
             "triton_skinny": ("#c0392b", "o", "Triton, decode tiles (BM=16)"),
             "triton_splitk4": ("#1f6f8b", "^", "Triton, split-K x4"),
             "triton_int4": ("#e8b04b", "D", "Triton, int4 weights")}

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    for a, data, title in ((ax[0], big, "K = N = 8192 (a 7B-scale layer)"),
                           (ax[1], small, "K = N = 1024 (this model's o_proj)")):
        for kern, (c, mk, lab) in style.items():
            rs = sorted([r for r in data if r["kernel"] == kern], key=lambda r: r["m"])
            if not rs:
                continue
            a.plot([r["m"] for r in rs], [r["ms"] * 1e3 for r in rs], marker=mk,
                   ms=4, color=c, label=lab)
        a.set_xscale("log", base=2)
        a.set_yscale("log", base=2)
        a.set_xlabel("M (= batch size)")
        a.set_ylabel("microseconds per GEMM")
        a.set_title(title)
        a.legend(fontsize=7.5)
        a.grid(alpha=0.25, which="both", lw=0.4)

    a = ax[2]
    rs = sorted([r for r in big if r["kernel"] == "triton_skinny"], key=lambda r: r["m"])
    ms = [r["m"] for r in rs]
    a.plot(ms, [100 * r["gflops"] / f["ceilings"]["gemm_gflops"] for r in rs], "o-",
           color="#c0392b", label="% of the compute ceiling")
    a.plot(ms, [100 * r["gbs"] / f["ceilings"]["copy_gbs"] for r in rs], "s-",
           color="#1f6f8b", label="% of the memory ceiling")
    a.set_xscale("log", base=2)
    a.set_xlabel("M (= batch size)")
    a.set_ylabel("percent of ceiling")
    a.set_title("Same kernel, two verdicts")
    a.set_ylim(0, 115)
    a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both", lw=0.4)

    fig.suptitle("Project 40 - the decode GEMM: M is tiny, K and N are not",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "skinny_m.png"), dpi=125)
    print("wrote", os.path.join(OUT, "skinny_m.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(OUT, "findings.json")
    if args.plot:
        plot(json.load(open(path)))
        return

    print("A. correctness")
    agree = check()
    print("  ", {k: f"{v:.2e}" for k, v in agree.items()})

    print("B. cuBLAS")
    rows = run_cublas()
    for r in rows[:4]:
        print(f"   {r['kernel']:16s} M={r['m']:4d} K=N={r['n']:5d}  "
              f"{r['ms']*1e3:9.1f} us  {r['gflops']/1000:7.2f} TFLOP/s  {r['gbs']:6.1f} GB/s")

    print("C. Triton variants")
    tuning = []
    for shape in SHAPES:
        rows += triton_rows(shape, tuning)

    n = 1 << 24
    a = torch.empty(n, device=E.DEV)
    b = torch.empty(n, device=E.DEV)
    t = E.gpu_time(lambda: E.k_copy[(triton.cdiv(n, 1024),)](a, b, n, BLOCK=1024), reps=50)
    ceilings = {"copy_gbs": 2 * n * 4 / t / 1e6, "gemm_gflops": 5697.8}

    f = {"device": E.device_info(), "agreement": agree, "rows": rows,
         "int4_tuning": tuning, "ceilings": ceilings, "group": GROUP}
    json.dump(f, open(path, "w"), indent=1)
    print("wrote", path)
    plot(f)


if __name__ == "__main__":
    main()
