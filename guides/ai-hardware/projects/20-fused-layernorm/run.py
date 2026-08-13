"""Project 20 - fusing LayerNorm into the linear layer that follows it.

  A. correctness   - all four paths against a float64 CPU reference
  B. numerics      - the one-pass variance formula, and where it dies
  C. LayerNorm alone - four kernels vs one
  D. the fusion    - split vs fused across output widths, against a traffic
                     model that predicts the answer in advance
  E. the budget    - the shared memory fusion costs, and the K it caps out at
  F. grid shape    - normalise once and loop, or normalise per column block
"""

import csv
import json
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "18-triton-softmax")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "19-triton-matmul")))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                      # noqa: E402
from matmul import matmul                                       # noqa: E402
from layernorm import (layernorm, layernorm_chain,              # noqa: E402
                       layernorm_naive_var, ln_linear_fused,
                       fused_compile_info, bytes_split, bytes_fused,
                       bytes_chain)

M, K = 8192, 128            # rows, model width
BM, BN = 32, 64             # the fused kernel's tile
EPS = 1e-5
R = {}


def cpu_ln(x, g, b, eps=EPS):
    xd = x.double()
    mu = xd.mean(1, keepdim=True)
    var = ((xd - mu) ** 2).mean(1, keepdim=True)
    return (xd - mu) / (var + eps).sqrt() * g.double() + b.double()


def make(M_, K_, N_, seed=0):
    return (gpu.randn(M_, K_, seed=seed + 1), gpu.randn(K_, N_, seed=seed + 2),
            gpu.randn(N_, seed=seed + 3), gpu.randn(K_, seed=seed + 4),
            gpu.randn(K_, seed=seed + 5))


def split(x, w, bias, g, beta, tmp, out):
    """What eager PyTorch runs: a LayerNorm kernel, then a matmul kernel."""
    layernorm(x, g, beta, EPS, out=tmp)
    matmul(tmp, w, c=out)
    return out


def section_a():
    print("A. correctness (float64 CPU reference)")
    x, w, bias, g, beta = make(512, K, 256, seed=0)
    ln_ref = cpu_ln(x.cpu(), g.cpu(), beta.cpu())
    lin_ref = ln_ref @ w.cpu().double() + bias.cpu().double()
    rows = []
    for nm, y, ref in [
            ("layernorm", layernorm(x, g, beta, EPS).cpu().double(), ln_ref),
            ("layernorm_chain", layernorm_chain(x, g, beta, EPS).cpu().double(), ln_ref),
            ("ln+linear fused 1D",
             ln_linear_fused(x, w, bias, g, beta, EPS, BM, BN).cpu().double(), lin_ref),
            ("ln+linear fused 2D",
             ln_linear_fused(x, w, bias, g, beta, EPS, BM, BN,
                             two_d=True).cpu().double(), lin_ref)]:
        rel = ((y - ref).abs().max() / ref.abs().max()).item()
        rows.append(dict(path=nm, rel_err=rel))
        print("   %-20s relative error %.3e" % (nm, rel))
    R["correctness"] = rows


def section_b():
    print("\nB. numerics: variance as E[x^2]-E[x]^2 vs the centred sum")
    base = gpu.randn(256, K, seed=11)
    g = gpu.ones(K)          # a plain `+ 1.0` here would be a PyTorch
    b = gpu.zeros(K)         # kernel, which this card cannot run
    rows = []
    for shift in [0.0, 1e2, 1e3, 1e4, 1e5]:
        x = (base.cpu() + shift).cuda()
        ref = cpu_ln(x.cpu(), g.cpu(), b.cpu())
        good = layernorm(x, g, b, EPS).cpu().double()
        bad = layernorm_naive_var(x, g, b, EPS).cpu().double()
        scale = ref.abs().max().item()
        rows.append(dict(shift=shift,
                         centred=((good - ref).abs().max() / scale).item(),
                         one_pass=((bad - ref).abs().max() / scale).item(),
                         one_pass_nonfinite=int((~torch.isfinite(bad)).sum())))
        r = rows[-1]
        print("   mean %8.0e | centred %.3e | one-pass %.3e%s"
              % (shift, r["centred"], r["one_pass"],
                 "  (%d non-finite)" % r["one_pass_nonfinite"]
                 if r["one_pass_nonfinite"] else ""))
    R["numerics"] = rows


def section_c():
    print("\nC. LayerNorm alone: four kernels vs one (M=%d, K=%d)" % (M, K))
    x = gpu.randn(M, K, seed=21)
    g, b = gpu.randn(K, seed=22), gpu.randn(K, seed=23)
    y = torch.empty_like(x)
    buf = (torch.empty_like(x), torch.empty_like(x), gpu.empty(M), gpu.empty(M))
    ms1 = gpu.bench(lambda: layernorm(x, g, b, EPS, out=y), reps=50)
    ms4 = gpu.bench(lambda: layernorm_chain(x, g, b, EPS, buf=buf), reps=50)
    one, four = 2 * M * K * 4, 6 * M * K * 4
    R["layernorm_alone"] = dict(one_kernel_ms=ms1, chain_ms=ms4,
                                one_gbs=gpu.gbs(one, ms1),
                                chain_gbs=gpu.gbs(four, ms4),
                                speedup=ms4 / ms1, traffic_ratio=four / one)
    print("   one kernel  %.4f ms  (%.1f GB/s over %d passes)"
          % (ms1, gpu.gbs(one, ms1), 2))
    print("   four kernels %.4f ms  (%.1f GB/s over %d passes)"
          % (ms4, gpu.gbs(four, ms4), 6))
    print("   %.2fx faster, %.0fx less traffic" % (ms4 / ms1, four / one))


def section_d():
    print("\nD. LayerNorm + Linear: split vs fused (M=%d, K=%d)" % (M, K))
    rows = []
    print("   %6s %8s %8s %9s %9s %8s %9s %9s %9s"
          % ("N", "LN ms", "mm ms", "split ms", "fused ms", "speedup",
             "predicted", "mm GF/s", "fused GF/s"))
    for N in [64, 128, 256, 512, 1024, 2048, 4096]:
        x, w, bias, g, beta = make(M, K, N, seed=30)
        tmp = torch.empty_like(x)
        out = gpu.empty(M, N)
        ms_ln = gpu.bench(lambda: layernorm(x, g, beta, EPS, out=tmp), reps=20)
        ms_mm = gpu.bench(lambda: matmul(tmp, w, c=out), reps=20)
        ms_s = gpu.bench(lambda: split(x, w, bias, g, beta, tmp, out), reps=20)
        ms_f = gpu.bench(lambda: ln_linear_fused(x, w, bias, g, beta, EPS,
                                                 BM, BN, out=out), reps=20)
        bs, bf = bytes_split(M, N, K), bytes_fused(M, N, K)
        flop = 2.0 * M * K * N
        rows.append(dict(N=N, ln_ms=ms_ln, matmul_ms=ms_mm, split_ms=ms_s,
                         fused_ms=ms_f, speedup=ms_s / ms_f,
                         split_mb=bs / 1e6, fused_mb=bf / 1e6,
                         predicted=bs / bf, flop=flop,
                         matmul_gflops=flop / (ms_mm * 1e6),
                         fused_gflops=flop / (ms_f * 1e6)))
        r = rows[-1]
        print("   %6d %8.4f %8.4f %9.4f %9.4f %7.2fx %8.2fx %9.0f %9.0f"
              % (N, ms_ln, ms_mm, ms_s, ms_f, r["speedup"], r["predicted"],
                 r["matmul_gflops"], r["fused_gflops"]))
    R["fusion"] = rows
    best, worst = max(rows, key=lambda r: r["speedup"]), min(rows, key=lambda r: r["speedup"])
    print("   fusion is worth %.2fx at N=%d and %.2fx at N=%d"
          % (best["speedup"], best["N"], worst["speedup"], worst["N"]))


def section_e():
    print("\nE. what fusion costs in shared memory")
    rows = []
    for (Kx, bm, bn) in [(64, 32, 64), (128, 16, 32), (128, 32, 64),
                         (256, 16, 32), (256, 32, 32), (512, 16, 16)]:
        x, w, bias, g, beta = make(1024, Kx, 1024, seed=40)
        need = (bm * Kx + Kx * bn) * 4
        try:
            info = fused_compile_info(x, w, bias, g, beta, bm, bn)
            rows.append(dict(K=Kx, BM=bm, BN=bn, ok=True, need_bytes=need, **info))
            print("   K=%-4d BM=%-3d BN=%-3d  needs %5.1f KB  compiler used "
                  "%5.1f KB, %3d regs, %3d spilled bytes%s"
                  % (Kx, bm, bn, need / 1024, info["shared"] / 1024,
                     info["regs"], info["spills"],
                     "   <- SPILLING" if info["spills"] else ""))
        except Exception as e:
            msg = str(e).split("\n")[0][:60]
            rows.append(dict(K=Kx, BM=bm, BN=bn, ok=False, need_bytes=need,
                             error=msg))
            print("   K=%-4d BM=%-3d BN=%-3d  needs %5.1f KB  DID NOT COMPILE: %s"
                  % (Kx, bm, bn, need / 1024, msg))
    R["budget"] = rows


def section_f():
    print("\nF. one program per row block (loop over N) vs one per tile")
    rows = []
    x0, w0, b0, g0, be0 = make(M, K, 1024, seed=50)
    info1 = fused_compile_info(x0, w0, b0, g0, be0, BM, BN)
    info2 = fused_compile_info(x0, w0, b0, g0, be0, BM, BN, two_d=True)
    R["grid_compile"] = dict(one_d=info1, two_d=info2)
    print("   compiled: 1D %d regs / %d spilled bytes;  2D %d regs / %d spilled"
          % (info1["regs"], info1["spills"], info2["regs"], info2["spills"]))
    for N in [256, 1024, 4096]:
        x, w, bias, g, beta = make(M, K, N, seed=50)
        out = gpu.empty(M, N)
        ms1 = gpu.bench(lambda: ln_linear_fused(x, w, bias, g, beta, EPS, BM,
                                                BN, out=out), reps=20)
        ms2 = gpu.bench(lambda: ln_linear_fused(x, w, bias, g, beta, EPS, BM,
                                                BN, out=out, two_d=True), reps=20)
        rows.append(dict(N=N, one_d_ms=ms1, two_d_ms=ms2, ratio=ms2 / ms1,
                         programs_1d=triton.cdiv(M, BM),
                         programs_2d=triton.cdiv(M, BM) * triton.cdiv(N, BN),
                         ln_recomputes=triton.cdiv(N, BN)))
        r = rows[-1]
        print("   N=%-5d 1D %.4f ms (%4d programs)   2D %.4f ms (%5d programs, "
              "LayerNorm done %d times)   ratio %.2fx"
              % (N, ms1, r["programs_1d"], ms2, r["programs_2d"],
                 r["ln_recomputes"], r["ratio"]))
    R["grid_shape"] = rows


def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    f = R["fusion"]
    ax[0].plot([r["N"] for r in f], [r["speedup"] for r in f], "o-",
               color="#2ca02c", label="measured")
    ax[0].plot([r["N"] for r in f], [r["predicted"] for r in f], "s--",
               color="#888888", label="traffic ratio (prediction)")
    ax[0].axhline(1.0, color="black", lw=0.8)
    ax[0].set_xscale("log", base=2)
    ax[0].set_xlabel("output width N")
    ax[0].set_ylabel("fused / split speedup")
    ax[0].set_title("D. fusion pays where the intermediate\nis a big share of the traffic")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].plot([r["N"] for r in f], [r["split_mb"] for r in f], "s-",
               color="#d62728", label="split: LayerNorm + matmul")
    ax[1].plot([r["N"] for r in f], [r["fused_mb"] for r in f], "o-",
               color="#2ca02c", label="fused: one kernel")
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log", base=2)
    ax[1].set_xlabel("output width N")
    ax[1].set_ylabel("MB moved")
    ax[1].set_title("the saving is a constant 2*M*K bytes,\nagainst a total that grows with N")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    b = R["budget"]
    lbl = ["K=%d\n%dx%d" % (r["K"], r["BM"], r["BN"]) for r in b]
    cols = ["#2ca02c" if (r["ok"] and not r.get("spills")) else
            ("#ff7f0e" if r["ok"] else "#d62728") for r in b]
    ax[2].bar(lbl, [r["need_bytes"] / 1024 for r in b], color=cols)
    ax[2].axhline(48, color="black", ls="--", label="48 KB shared per block")
    ax[2].set_ylabel("shared memory the fusion needs (KB)")
    ax[2].set_title("E. green = fits, orange = spills,\nred = will not compile")
    ax[2].tick_params(axis="x", labelsize=7)
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    R["device"] = gpu.device_note()
    R["config"] = dict(M=M, K=K, BLOCK_M=BM, BLOCK_N=BN)
    d = R["device"]
    print("device: %s (cc %s, %d SMs)  M=%d K=%d, fused tile %dx%d\n"
          % (d["name"], d["cc"], d["sms"], M, K, BM, BN))
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
            w.writerow(["A", r["path"], "%.3e" % r["rel_err"], "", ""])
        for r in R["numerics"]:
            w.writerow(["B", "shift_%g" % r["shift"], "%.3e" % r["centred"],
                        "%.3e" % r["one_pass"], r["one_pass_nonfinite"]])
        a = R["layernorm_alone"]
        w.writerow(["C", "one_kernel", "%.4f" % a["one_kernel_ms"],
                    "%.1f" % a["one_gbs"], ""])
        w.writerow(["C", "four_kernels", "%.4f" % a["chain_ms"],
                    "%.1f" % a["chain_gbs"], "%.2f" % a["speedup"]])
        for r in R["fusion"]:
            w.writerow(["D", "N_%d" % r["N"], "%.4f" % r["split_ms"],
                        "%.4f" % r["fused_ms"], "%.3f" % r["speedup"]])
            w.writerow(["D", "N_%d_parts" % r["N"], "%.4f" % r["ln_ms"],
                        "%.4f" % r["matmul_ms"],
                        "mm %.0f GF/s vs fused %.0f GF/s"
                        % (r["matmul_gflops"], r["fused_gflops"])])
        for r in R["budget"]:
            w.writerow(["E", "K%d_%dx%d" % (r["K"], r["BM"], r["BN"]),
                        r["need_bytes"], "ok" if r["ok"] else "no-compile",
                        r.get("spills", "")])
        for r in R["grid_shape"]:
            w.writerow(["F", "N_%d" % r["N"], "%.4f" % r["one_d_ms"],
                        "%.4f" % r["two_d_ms"], "%.3f" % r["ratio"]])

    p = plot(os.path.join(OUT, "fused_layernorm.png"))
    print("\nwrote outputs/findings.json, outputs/findings.csv%s"
          % (", " + os.path.relpath(p, HERE) if p else ""))


if __name__ == "__main__":
    main()
