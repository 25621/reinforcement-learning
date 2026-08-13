"""Project 18 - softmax in Triton.

  A. environment  - PyTorch's CUDA kernels do not run on this card; Triton's do
  B. correctness  - against a float64 CPU reference
  C. numerics     - the same mathematics, with and without the max subtraction
  D. fusion       - one kernel versus the four you get from tensor operations
  E. bandwidth    - achieved GB/s against the measured memory anchors
  F. tuning       - num_warps (and section E finds the row length at which the
                    fused kernel runs out of registers)
"""

import csv
import json
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                    # noqa: E402
from softmax import (softmax_fused, softmax_unsafe, softmax_online,   # noqa: E402
                     softmax_multipass, bytes_fused, bytes_online,
                     bytes_multipass)

R = {}


def cpu_softmax(x):
    """Reference in float64 on the CPU. The GPU never touches this."""
    xd = x.double()
    xd = xd - xd.max(dim=-1, keepdim=True).values
    e = xd.exp()
    return e / e.sum(dim=-1, keepdim=True)


def section_a():
    R["device"] = gpu.device_note()
    eager_ok, eager_msg = gpu.torch_eager_works()
    cublas_ok, cublas_msg = gpu.torch_cublas_works()
    x = gpu.randn(4, 8, seed=0)
    try:
        y = softmax_fused(x)
        err = (y.cpu() - cpu_softmax(x.cpu())).abs().max().item()
        triton_ok, triton_msg = True, "max err %.2e" % err
    except Exception as e:                                    # pragma: no cover
        triton_ok, triton_msg = False, str(e).split("\n")[0]
    R["environment"] = dict(torch_version=torch.__version__,
                            triton_version=triton.__version__,
                            torch_eager=eager_ok, torch_eager_msg=eager_msg,
                            torch_cublas=cublas_ok, torch_cublas_msg=cublas_msg,
                            triton=triton_ok, triton_msg=triton_msg)
    d = R["device"]
    print("A. environment")
    print("   %s (cc %s, %d SMs, %.1f GB)"
          % (d["name"], d["cc"], d["sms"], d["mem_gb"]))
    print("   torch %s / triton %s" % (torch.__version__, triton.__version__))
    print("   torch eager on GPU : %-5s  %s" % (eager_ok, eager_msg[:70]))
    print("   torch cuBLAS       : %-5s  %s" % (cublas_ok, cublas_msg[:70]))
    print("   triton kernel      : %-5s  %s" % (triton_ok, triton_msg[:70]))


def section_b():
    print("\nB. correctness (float64 CPU reference)")
    rows = []
    for M, N in [(4, 7), (128, 129), (1024, 512), (512, 4096)]:
        x = gpu.randn(M, N, seed=M + N)
        ref = cpu_softmax(x.cpu())
        f = (softmax_fused(x).cpu() - ref).abs().max().item()
        o = (softmax_online(x).cpu() - ref).abs().max().item()
        m = (softmax_multipass(x).cpu() - ref).abs().max().item()
        s = (softmax_fused(x).cpu().double().sum(-1) - 1).abs().max().item()
        rows.append(dict(M=M, N=N, fused=f, online=o, multipass=m, sum_err=s))
        print("   %5dx%-5d fused %.2e  online %.2e  multipass %.2e  "
              "|rowsum-1| %.2e" % (M, N, f, o, m, s))
    R["correctness"] = rows


def section_c():
    print("\nC. numerics: softmax(x) == softmax(x + c) in mathematics only")
    M, N = 256, 1024
    base = gpu.randn(M, N, seed=7)
    ref = cpu_softmax(base.cpu())
    rows = []
    for c in [0.0, 20.0, 60.0, 88.0, 100.0, 200.0, -100.0, -200.0]:
        x = (base.cpu() + c).cuda()
        u = softmax_unsafe(x).cpu()
        s = softmax_fused(x).cpu()
        rows.append(dict(
            shift=c,
            unsafe_bad=int((~torch.isfinite(u)).sum().item()),
            unsafe_err=float((u - ref).abs().max().item()),
            safe_bad=int((~torch.isfinite(s)).sum().item()),
            safe_err=float((s - ref).abs().max().item())))
        print("   shift %+7.1f | unsafe: %7d non-finite, err %9.3e "
              "| safe: %d non-finite, err %.3e"
              % (c, rows[-1]["unsafe_bad"], rows[-1]["unsafe_err"],
                 rows[-1]["safe_bad"], rows[-1]["safe_err"]))
    R["numerics"] = rows


def section_d():
    print("\nD. fusion: one kernel vs the four a tensor-op chain would launch")
    rows = []
    for M, N in [(8192, 512), (4096, 1024), (2048, 2048), (1024, 4096)]:
        x = gpu.randn(M, N, seed=1)
        y = torch.empty_like(x)
        t = torch.empty_like(x)
        mx = gpu.empty(M)
        sm = gpu.empty(M)
        ms_f = gpu.bench(lambda: softmax_fused(x), reps=30)
        ms_o = gpu.bench(lambda: softmax_online(x), reps=30)
        ms_m = gpu.bench(lambda: softmax_multipass(x, t, y, mx, sm), reps=30)
        rows.append(dict(
            M=M, N=N,
            fused_ms=ms_f, online_ms=ms_o, multipass_ms=ms_m,
            fused_gbs=gpu.gbs(bytes_fused(M, N), ms_f),
            online_gbs=gpu.gbs(bytes_online(M, N), ms_o),
            multipass_gbs=gpu.gbs(bytes_multipass(M, N), ms_m),
            speedup=ms_m / ms_f,
            traffic_ratio=bytes_multipass(M, N) / bytes_fused(M, N)))
        r = rows[-1]
        print("   %5dx%-5d fused %.3f ms (%6.1f GB/s) | multipass %.3f ms "
              "(%6.1f GB/s) | %.2fx faster, %.0fx less traffic"
              % (M, N, ms_f, r["fused_gbs"], ms_m, r["multipass_gbs"],
                 r["speedup"], r["traffic_ratio"]))
    R["fusion"] = rows


def fused_regs(x):
    """Registers per thread and spilled bytes, straight from the compiler."""
    from softmax import _fused_kernel
    M, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    nw = max(1, min(16, BLOCK // 256))
    k = _fused_kernel[(M,)](x, y, x.stride(0), y.stride(0), N, BLOCK=BLOCK,
                            num_warps=nw)
    return k.n_regs, k.n_spills


def section_e():
    print("\nE. bandwidth vs row length (~8M elements each time)")
    rows = []
    for N in [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
        M = max(64, (1 << 23) // N)
        x = gpu.randn(M, N, seed=2)
        try:
            ms = gpu.bench(lambda: softmax_fused(x), reps=20)
            g = gpu.gbs(bytes_fused(M, N), ms)
            ok = True
        except Exception as e:
            ms, g, ok = float("nan"), float("nan"), False
            print("   N=%-6d fused kernel FAILED: %s"
                  % (N, str(e).split("\n")[0][:60]))
        ms_o = gpu.bench(lambda: softmax_online(x), reps=20)
        g_o = gpu.gbs(bytes_online(M, N), ms_o)
        regs, spills = fused_regs(x)
        rows.append(dict(N=N, M=M, fused_ok=ok, fused_ms=ms, fused_gbs=g,
                         online_ms=ms_o, online_gbs=g_o, regs=regs,
                         spill_bytes=spills, block=triton.next_power_of_2(N)))
        if ok:
            print("   N=%-6d M=%-7d fused %7.3f ms %6.1f GB/s (%3d regs, "
                  "%3d spill B)   online %7.3f ms %6.1f GB/s"
                  % (N, M, ms, g, regs, spills, ms_o, g_o))
    R["bandwidth"] = rows


def section_f():
    print("\nF. num_warps for one row shape (M=4096, N=1024)")
    M, N = 4096, 1024
    x = gpu.randn(M, N, seed=3)
    rows = []
    for nw in [1, 2, 4, 8, 16]:
        try:
            ms = gpu.bench(lambda: softmax_fused(x, num_warps=nw), reps=30)
            rows.append(dict(num_warps=nw, ms=ms,
                             gbs=gpu.gbs(bytes_fused(M, N), ms)))
            print("   num_warps=%-3d %7.3f ms  %6.1f GB/s"
                  % (nw, ms, rows[-1]["gbs"]))
        except Exception as e:
            print("   num_warps=%-3d failed: %s" % (nw, str(e).split("\n")[0][:60]))
    best = min(rows, key=lambda r: r["ms"])
    worst = max(rows, key=lambda r: r["ms"])
    R["num_warps"] = dict(rows=rows, best=best["num_warps"],
                          spread=worst["ms"] / best["ms"])
    print("   best num_warps=%d, spread across the sweep %.2fx"
          % (best["num_warps"], worst["ms"] / best["ms"]))


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
    labels = ["%dx%d" % (r["M"], r["N"]) for r in f]
    xs = range(len(f))
    ax[0].bar([x - 0.2 for x in xs], [r["fused_ms"] for r in f], 0.4,
              label="fused (1 kernel)", color="#2ca02c")
    ax[0].bar([x + 0.2 for x in xs], [r["multipass_ms"] for r in f], 0.4,
              label="tensor-op chain (4 kernels)", color="#d62728")
    ax[0].set_xticks(list(xs))
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("ms")
    ax[0].set_title("D. fusion: %.2fx from moving 3x fewer bytes"
                    % (sum(r["speedup"] for r in f) / len(f)))
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y")

    b = [r for r in R["bandwidth"] if r["fused_ok"]]
    ax[1].plot([r["N"] for r in b], [r["fused_gbs"] for r in b], "o-",
               color="#2ca02c", label="fused")
    ax[1].plot([r["N"] for r in R["bandwidth"]],
               [r["online_gbs"] for r in R["bandwidth"]], "s--",
               color="#1f77b4", label="online (chunked)")
    ax[1].axhline(gpu.BW_COPY, color="grey", ls=":",
                  label="measured copy limit %.0f GB/s" % gpu.BW_COPY)
    ax[1].axhline(gpu.PEAK_BW, color="grey", ls="--",
                  label="DRAM spec peak %.0f GB/s" % gpu.PEAK_BW)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("row length N")
    ax[1].set_ylabel("GB/s (bytes the algorithm must move / time)")
    ax[1].set_ylim(0, 320)
    ax[1].set_title("E. softmax is a memory-bound operation")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    n = R["numerics"]
    pos = list(range(len(n)))
    ax[2].bar([p - 0.2 for p in pos], [max(r["unsafe_bad"], 0.5) for r in n],
              0.4, color="#d62728", label="without max subtraction")
    ax[2].bar([p + 0.2 for p in pos], [max(r["safe_bad"], 0.5) for r in n],
              0.4, color="#2ca02c", label="with max subtraction")
    ax[2].set_xticks(pos)
    ax[2].set_xticklabels([str(int(r["shift"])) for r in n])
    ax[2].set_yscale("log")
    ax[2].set_xlabel("constant added to every element")
    ax[2].set_ylabel("non-finite outputs (0.5 = none)")
    ax[2].set_title("C. same mathematics, different arithmetic")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
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
            w.writerow(["B", "%dx%d" % (r["M"], r["N"]), "%.3e" % r["fused"],
                        "%.3e" % r["online"], "%.3e" % r["multipass"]])
        for r in R["numerics"]:
            w.writerow(["C", "shift_%+g" % r["shift"], r["unsafe_bad"],
                        "%.3e" % r["unsafe_err"], r["safe_bad"]])
        for r in R["fusion"]:
            w.writerow(["D", "%dx%d" % (r["M"], r["N"]), "%.4f" % r["fused_ms"],
                        "%.4f" % r["multipass_ms"], "%.2f" % r["speedup"]])
        for r in R["bandwidth"]:
            w.writerow(["E", "N_%d" % r["N"], "%.1f" % r["fused_gbs"],
                        "%.1f" % r["online_gbs"],
                        "%d regs %d spill" % (r["regs"], r["spill_bytes"])])
        for r in R["num_warps"]["rows"]:
            w.writerow(["F", "warps_%d" % r["num_warps"], "%.4f" % r["ms"],
                        "%.1f" % r["gbs"], ""])

    p = plot(os.path.join(OUT, "triton_softmax.png"))
    print("\nwrote outputs/findings.json, outputs/findings.csv%s"
          % (", " + os.path.relpath(p, HERE) if p else ""))


if __name__ == "__main__":
    main()
