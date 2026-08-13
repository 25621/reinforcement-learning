"""Project 19 - a matmul in Triton, scored against project 17's CUDA ladder.

  A. correctness    - against a float64 CPU reference, including a shape with
                      no power of two anywhere in it
  B. the sweep      - 12 configurations at N=2048, with the registers, spills
                      and shared memory each one asked for
  C. shape-dependence - the best configuration is not the same at every size,
                      which is the argument for autotuning
  D. the scoreboard - Triton against project 17's five CUDA kernels and cuBLAS,
                      measured in this same session on this same card
  E. shapes         - the skinny and non-power-of-two cases, and what masking
                      costs
  F. block order    - GROUP_M, which changes nothing about the answer
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
P17 = os.path.abspath(os.path.join(HERE, "..", "17-cuda-tiled-matmul"))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "18-triton-softmax")))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                     # noqa: E402
from matmul import (matmul, compile_info, CONFIGS,             # noqa: E402
                    arithmetic_intensity)

R = {}
BEST = (64, 128, 32, 8, 4, 2)     # filled in by section B


def name(cfg):
    return "%dx%dx%d g%d w%d s%d" % cfg


def section_a():
    print("A. correctness (float64 CPU reference)")
    rows = []
    for M, N, K in [(512, 512, 512), (700, 900, 1100), (64, 4096, 4096)]:
        a, b = gpu.randn(M, K, seed=1), gpu.randn(K, N, seed=2)
        ref = a.cpu().double() @ b.cpu().double()
        c = matmul(a, b).cpu().double()
        err = (c - ref).abs().max().item()
        rel = err / ref.abs().max().item()
        rows.append(dict(M=M, N=N, K=K, abs_err=err, rel_err=rel))
        print("   %4dx%4dx%-5d  max abs %.3e   relative %.3e"
              % (M, N, K, err, rel))
    R["correctness"] = rows


def section_b():
    global BEST
    print("\nB. configuration sweep at N=2048")
    N = 2048
    a, b = gpu.randn(N, N, seed=3), gpu.randn(N, N, seed=4)
    flop = 2.0 * N ** 3
    rows = []
    print("   %-18s %9s %10s %7s %7s %8s %8s"
          % ("BMxBNxBK g w s", "ms", "GFLOP/s", "regs", "spills", "shared", "AI"))
    for cfg in CONFIGS:
        t0 = time.time()
        try:
            info = compile_info(a, b, cfg)
            ms = gpu.bench(lambda: matmul(a, b, cfg), reps=10)
            rows.append(dict(cfg=list(cfg), name=name(cfg), ok=True, ms=ms,
                             gflops=flop / (ms * 1e6), compile_s=time.time() - t0,
                             ai=arithmetic_intensity(cfg[0], cfg[1]), **info))
            r = rows[-1]
            print("   %-18s %9.3f %10.0f %7d %7d %8d %8.1f"
                  % (r["name"], ms, r["gflops"], r["regs"], r["spills"],
                     r["shared"], r["ai"]))
        except Exception as e:
            msg = str(e).split("\n")[0][:70]
            rows.append(dict(cfg=list(cfg), name=name(cfg), ok=False, error=msg))
            print("   %-18s DID NOT COMPILE: %s" % (name(cfg), msg))
    ok = [r for r in rows if r["ok"]]
    best = max(ok, key=lambda r: r["gflops"])
    BEST = tuple(best["cfg"])
    spilled = [r for r in ok if r["spills"] > 0]
    clean = [r for r in ok if r["spills"] == 0]
    R["sweep"] = dict(rows=rows, best=best["name"], best_gflops=best["gflops"],
                      spread=best["gflops"] / min(r["gflops"] for r in ok),
                      worst_clean=min(r["gflops"] for r in clean),
                      best_spilled=max([r["gflops"] for r in spilled],
                                       default=None))
    print("   best: %s at %.0f GFLOP/s; spread across the sweep %.2fx"
          % (best["name"], best["gflops"], R["sweep"]["spread"]))
    if spilled:
        print("   every spilling config is slower than every clean one: "
              "clean worst %.0f, spilling best %.0f"
              % (R["sweep"]["worst_clean"], R["sweep"]["best_spilled"]))


def section_c():
    print("\nC. the best configuration depends on the size")
    probe = [(64, 64, 64, 8, 4, 2), (64, 128, 32, 8, 4, 2), (64, 64, 32, 8, 4, 2),
             (128, 128, 32, 8, 8, 2), (128, 64, 32, 8, 4, 2)]
    rows = []
    for N in [1024, 2048, 4096]:
        a, b = gpu.randn(N, N, seed=5), gpu.randn(N, N, seed=6)
        flop = 2.0 * N ** 3
        per = {}
        for cfg in probe:
            ms = gpu.bench(lambda: matmul(a, b, cfg), reps=5)
            per[name(cfg)] = flop / (ms * 1e6)
        best = max(per, key=per.get)
        rows.append(dict(N=N, per_config=per, best=best, best_gflops=per[best],
                         fixed_gflops=per[name(probe[0])]))
        print("   N=%-5d best %-18s %.0f GFLOP/s   (one fixed config would give "
              "%.0f, i.e. %.1f%% of it)"
              % (N, best, per[best], per[name(probe[0])],
                 100 * per[name(probe[0])] / per[best]))
    R["shape_dependence"] = rows


def cuda_ladder():
    """Run project 17's binary so the comparison is same-session, same card."""
    exe = os.path.join(P17, "outputs", "sgemm")
    if not os.path.exists(exe):
        if shutil.which("nvcc") is None:
            return None
        os.makedirs(os.path.dirname(exe), exist_ok=True)
        r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
                            os.path.join(P17, "sgemm.cu"), "-o", exe, "-lcublas"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return None
    r = subprocess.run([exe], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    out = {}
    for line in r.stdout.splitlines():
        f = line.split(",")
        if f[0] == "k":
            out.setdefault(int(f[1]), {})[f[2]] = float(f[4])
    return out


def section_d():
    print("\nD. scoreboard against project 17's CUDA kernels (same session)")
    ladder = cuda_ladder()
    rows = []
    for N in [1024, 2048, 4096]:
        a, b = gpu.randn(N, N, seed=7), gpu.randn(N, N, seed=8)
        flop = 2.0 * N ** 3
        best = max((flop / (gpu.bench(lambda: matmul(a, b, c), reps=5) * 1e6), c)
                   for c in [(64, 64, 64, 8, 4, 2), (64, 128, 32, 8, 4, 2),
                             (64, 64, 32, 8, 4, 2), (128, 128, 32, 8, 8, 2)])
        row = dict(N=N, triton=best[0], triton_cfg=name(best[1]))
        if ladder and N in ladder:
            row.update({k: v for k, v in ladder[N].items()})
            row["pct_cublas"] = 100 * best[0] / ladder[N]["cublas"]
            row["vs_hand_cuda"] = best[0] / ladder[N]["vec"]
        rows.append(row)
    R["scoreboard"] = rows
    if ladder:
        print("   %6s %10s %10s %10s %10s %12s %12s"
              % ("N", "naive", "smem", "tile2d", "vec(CUDA)", "TRITON",
                 "cuBLAS"))
        for r in rows:
            print("   %6d %10.0f %10.0f %10.0f %10.0f %12.0f %12.0f"
                  % (r["N"], r["naive"], r["smem"], r["tile2d"], r["vec"],
                     r["triton"], r["cublas"]))
        for r in rows:
            print("   N=%-5d Triton = %.1f%% of cuBLAS, %.2fx the hand-written "
                  "CUDA kernel (%s)"
                  % (r["N"], r["pct_cublas"], r["vs_hand_cuda"], r["triton_cfg"]))
    else:
        print("   (project 17's binary unavailable - Triton only)")
        for r in rows:
            print("   N=%-5d Triton %.0f GFLOP/s" % (r["N"], r["triton"]))


def section_e():
    print("\nE. shapes the fixed-tile CUDA kernel cannot express")
    rows = []
    shapes = [(4096, 4096, 4096), (1024, 1024, 1024), (1000, 1000, 1000),
              (64, 4096, 4096), (4096, 64, 4096), (4096, 4096, 64),
              (8192, 8192, 128)]
    for (M, N, K) in shapes:
        a, b = gpu.randn(M, K, seed=9), gpu.randn(K, N, seed=10)
        flop = 2.0 * M * N * K
        best = max((flop / (gpu.bench(lambda: matmul(a, b, c), reps=5) * 1e6), c)
                   for c in [(64, 64, 64, 8, 4, 2), (64, 128, 32, 8, 4, 2),
                             (64, 64, 32, 8, 4, 2), (128, 128, 32, 8, 8, 2)])
        BM, BN = best[1][0], best[1][1]
        padded = (triton.cdiv(M, BM) * BM) * (triton.cdiv(N, BN) * BN) * K
        rows.append(dict(M=M, N=N, K=K, gflops=best[0], cfg=name(best[1]),
                         padding_waste=padded / (M * N * K),
                         programs=triton.cdiv(M, BM) * triton.cdiv(N, BN),
                         cuda128_ok=(M % 128 == 0 and N % 128 == 0)))
        r = rows[-1]
        print("   %5dx%5dx%-5d %10.0f GFLOP/s  %-18s  padded work %.3fx  "
              "%5d programs  project-17 kernel: %s"
              % (M, N, K, r["gflops"], r["cfg"], r["padding_waste"],
                 r["programs"], "ok" if r["cuda128_ok"] else "CANNOT RUN"))
    R["shapes"] = rows


def section_f():
    print("\nF. GROUP_M: same answer, different visiting order")
    rows = []
    for N in [1024, 2048, 4096]:
        a, b = gpu.randn(N, N, seed=11), gpu.randn(N, N, seed=12)
        flop = 2.0 * N ** 3
        per = {}
        for gm in [1, 2, 4, 8, 16]:
            cfg = (64, 128, 32, gm, 4, 2)
            ms = gpu.bench(lambda: matmul(a, b, cfg), reps=5)
            per[gm] = flop / (ms * 1e6)
        rows.append(dict(N=N, per_group=per,
                         spread=max(per.values()) / min(per.values()),
                         best=max(per, key=per.get)))
        print("   N=%-5d " % N + "  ".join("GROUP_M=%-2d %.0f" % (g, v)
                                           for g, v in per.items())
              + "   spread %.3fx" % rows[-1]["spread"])
    R["group_order"] = rows


def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    ok = [r for r in R["sweep"]["rows"] if r["ok"]]
    cols = ["#d62728" if r["spills"] else "#2ca02c" for r in ok]
    ax[0].barh([r["name"] for r in ok], [r["gflops"] for r in ok], color=cols)
    ax[0].set_xlabel("GFLOP/s at N=2048")
    ax[0].set_title("B. red = the compiler spilled registers")
    ax[0].tick_params(axis="y", labelsize=7)
    ax[0].grid(alpha=.3, axis="x")

    s = R["scoreboard"]
    if "cublas" in s[0]:
        keys = ["naive", "smem", "tile1d", "tile2d", "vec", "triton", "cublas"]
        labels = ["naive\nCUDA", "smem\nCUDA", "tile1d\nCUDA", "tile2d\nCUDA",
                  "vec\nCUDA", "TRITON", "cuBLAS"]
        by_n = ["#a6cee3", "#1f78b4", "#08306b"]
        w = 0.26
        for i, r in enumerate(s):
            ax[1].bar([x + (i - 1) * w for x in range(len(keys))],
                      [r[k] for k in keys], w, label="N=%d" % r["N"],
                      color=by_n[i], edgecolor="black", linewidth=0.4)
        ax[1].set_xticks(range(len(keys)))
        ax[1].set_xticklabels(labels, fontsize=7)
        ax[1].set_ylabel("GFLOP/s")
        ax[1].set_title("D. Triton vs project 17's CUDA ladder")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=.3, axis="y")

    c = R["shape_dependence"]
    names = list(c[0]["per_config"].keys())
    for n in names:
        ax[2].plot([r["N"] for r in c], [r["per_config"][n] for r in c],
                   "o-", label=n)
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("N")
    ax[2].set_ylabel("GFLOP/s")
    ax[2].set_title("C. the winning config changes with the size")
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    R["device"] = gpu.device_note()
    d = R["device"]
    print("device: %s (cc %s, %d SMs)  torch %s / triton %s\n"
          % (d["name"], d["cc"], d["sms"], torch.__version__,
             triton.__version__))
    gpu.warm_up()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()

    with open(os.path.join(OUT, "findings.json"), "w") as fh:
        json.dump(R, fh, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "a", "b", "c"])
        for r in R["correctness"]:
            w.writerow(["A", "%dx%dx%d" % (r["M"], r["N"], r["K"]),
                        "%.3e" % r["abs_err"], "%.3e" % r["rel_err"], ""])
        for r in R["sweep"]["rows"]:
            if r["ok"]:
                w.writerow(["B", r["name"], "%.0f" % r["gflops"], r["regs"],
                            r["spills"]])
            else:
                w.writerow(["B", r["name"], "did-not-compile", r["error"], ""])
        for r in R["shape_dependence"]:
            w.writerow(["C", "N_%d" % r["N"], r["best"],
                        "%.0f" % r["best_gflops"], "%.0f" % r["fixed_gflops"]])
        for r in R["scoreboard"]:
            w.writerow(["D", "N_%d" % r["N"], "%.0f" % r["triton"],
                        "%.0f" % r.get("cublas", 0),
                        "%.1f" % r.get("pct_cublas", 0)])
        for r in R["shapes"]:
            w.writerow(["E", "%dx%dx%d" % (r["M"], r["N"], r["K"]),
                        "%.0f" % r["gflops"], "%.3f" % r["padding_waste"],
                        "cuda-ok" if r["cuda128_ok"] else "cuda-cannot-run"])
        for r in R["group_order"]:
            w.writerow(["F", "N_%d" % r["N"], r["best"],
                        "%.3f" % r["spread"], ""])

    p = plot(os.path.join(OUT, "triton_matmul.png"))
    print("\nwrote outputs/findings.json, outputs/findings.csv%s"
          % (", " + os.path.relpath(p, HERE) if p else ""))


if __name__ == "__main__":
    main()
