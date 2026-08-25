"""Project 21 - attention without the attention matrix.

  A. correctness  - flash and materialised against a float64 CPU reference,
                    with and without causal masking
  B. memory       - what the materialised path has to hold, and the sequence
                    length at which it stops fitting on this card
  C. speed        - both paths across sequence length
  D. traffic      - the bytes each path moves, and why the gap widens
  E. causal       - flash can skip the half it does not need; the materialised
                    path computes it and then throws it away
  F. tiles        - block sizes, registers and shared memory for the kernel
"""

import csv
import json
import math
import os
import sys

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in ("18-triton-softmax", "19-triton-matmul"):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", _p)))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                       # noqa: E402
from matmul import matmul                                        # noqa: E402
from softmax import softmax_fused                                # noqa: E402
from flash import (attention_flash, attention_materialised,      # noqa: E402
                   flash_compile_info, flops, bytes_flash,
                   bytes_materialised, peak_extra_bytes_materialised)

H, D = 8, 64                 # heads, head dimension
BM, BN, NW = 64, 32, 8       # the flash kernel's tile
CARD_BYTES = 8 * 2 ** 30
R = {}


def make(H_, S_, D_, seed=0):
    """q is pre-scaled by 1/sqrt(D) on the CPU, so both paths can use the
    same tensors without needing a scaling kernel."""
    g = torch.Generator().manual_seed(seed)
    scale = 1.0 / math.sqrt(D_)
    qc = torch.randn(H_, S_, D_, generator=g) * scale
    kc = torch.randn(H_, S_, D_, generator=g)
    vc = torch.randn(H_, S_, D_, generator=g)
    return (qc, kc, vc), (qc.cuda(), kc.cuda(), vc.cuda())


def cpu_attention(qc, kc, vc, causal):
    s = qc.double() @ kc.double().transpose(1, 2)
    if causal:
        keep = torch.tril(torch.ones(s.shape[-2], s.shape[-1], dtype=torch.bool))
        s = s.masked_fill(~keep, float("-inf"))
    p = torch.softmax(s, dim=-1)
    return p @ vc.double()


def materialised(q, k, v, causal, bufs):
    sc, pr, o = bufs
    return attention_materialised(
        q, k, v, causal=causal, scores=sc, probs=pr, out=o,
        matmul=matmul,
        softmax=lambda x, out: softmax_fused(x, out=out))


def section_a():
    print("A. correctness (float64 CPU reference)")
    rows = []
    for S in (128, 256):
        for causal in (False, True):
            (qc, kc, vc), (q, k, v) = make(2, S, D, seed=S)
            ref = cpu_attention(qc, kc, vc, causal)
            scale = ref.abs().max().item()
            f = attention_flash(q, k, v, causal, BM, BN,
                                num_warps=NW).cpu().double()
            bufs = (gpu.empty(2, S, S), gpu.empty(2, S, S), gpu.empty(2, S, D))
            m = materialised(q, k, v, causal, bufs).cpu().double()
            rows.append(dict(S=S, causal=causal,
                             flash=((f - ref).abs().max() / scale).item(),
                             materialised=((m - ref).abs().max() / scale).item()))
            print("   S=%-4d causal=%-5s  flash %.3e   materialised %.3e"
                  % (S, causal, rows[-1]["flash"], rows[-1]["materialised"]))
    R["correctness"] = rows


def section_b():
    print("\nB. what the materialised path has to hold (%d heads, D=%d)"
          % (H, D))
    rows = []
    for S in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
        extra = peak_extra_bytes_materialised(H, S)
        qkvo = 4 * H * S * D * 4
        rows.append(dict(S=S, scores_bytes=extra, qkvo_bytes=qkvo,
                         ratio=extra / qkvo,
                         fits=(extra + qkvo) < 0.9 * CARD_BYTES))
        r = rows[-1]
        print("   S=%-6d  Q,K,V,O %7.1f MB   score matrices %9.1f MB "
              "(%6.1fx)   %s"
              % (S, qkvo / 1e6, extra / 1e6, r["ratio"],
                 "fits in 8 GB" if r["fits"] else "DOES NOT FIT"))
    R["memory"] = rows

    # Not a prediction: actually try it.
    S = 16384
    (_, _, _), (q, k, v) = make(H, S, D, seed=99)
    try:
        sc = gpu.empty(H, S, S)
        del sc
        torch.cuda.empty_cache()
        mat_msg = "allocated"
    except Exception as e:
        mat_msg = type(e).__name__ + ": " + str(e).split("\n")[0][:70]
    o = gpu.empty(H, S, D)
    ms = gpu.bench(lambda: attention_flash(q, k, v, False, BM, BN,
                                           num_warps=NW, out=o), reps=1,
                   warmup=1)
    R["oom_test"] = dict(S=S, materialised=mat_msg, flash_ms=ms,
                         flash_gflops=flops(H, S, D) / (ms * 1e6))
    print("   at S=%d: materialised score matrix -> %s" % (S, mat_msg))
    print("   at S=%d: flash ran in %.1f ms (%.0f GFLOP/s), using %.1f MB"
          % (S, ms, R["oom_test"]["flash_gflops"], 4 * H * S * D * 4 / 1e6))
    del q, k, v, o
    torch.cuda.empty_cache()


def section_cd():
    print("\nC+D. speed and traffic across sequence length")
    rows = []
    print("   %6s %11s %10s %9s %10s %10s %9s"
          % ("S", "material ms", "flash ms", "speedup", "mat MB", "flash MB",
             "flash GF/s"))
    for S in [256, 512, 1024, 2048, 4096]:
        (_, _, _), (q, k, v) = make(H, S, D, seed=S)
        o = gpu.empty(H, S, D)
        ms_f = gpu.bench(lambda: attention_flash(q, k, v, False, BM, BN,
                                                 num_warps=NW, out=o), reps=10)
        need = peak_extra_bytes_materialised(H, S)
        if need < 3 * 2 ** 30:
            sc = gpu.empty(H, S, S)
            pr = gpu.empty(H, S, S)
            ms_m = gpu.bench(lambda: materialised(q, k, v, False, (sc, pr, o)),
                             reps=5)
            del sc, pr
            torch.cuda.empty_cache()
        else:
            ms_m = float("nan")
        bf, bm_ = bytes_flash(H, S, D, BM), bytes_materialised(H, S, D)
        fl = flops(H, S, D)
        rows.append(dict(S=S, materialised_ms=ms_m, flash_ms=ms_f,
                         speedup=ms_m / ms_f, flash_mb=bf / 1e6,
                         materialised_mb=bm_ / 1e6, traffic_ratio=bm_ / bf,
                         flash_gflops=fl / (ms_f * 1e6),
                         flash_gbs=gpu.gbs(bf, ms_f)))
        r = rows[-1]
        print("   %6d %11.4f %10.4f %8.2fx %10.1f %10.1f %9.0f"
              % (S, ms_m, ms_f, r["speedup"], r["materialised_mb"],
                 r["flash_mb"], r["flash_gflops"]))
    R["speed"] = rows


def section_e():
    print("\nE. causal masking: skipped work vs discarded work")
    rows = []
    for S in [1024, 2048, 4096]:
        (_, _, _), (q, k, v) = make(H, S, D, seed=S)
        o = gpu.empty(H, S, D)
        full = gpu.bench(lambda: attention_flash(q, k, v, False, BM, BN,
                                                 num_warps=NW, out=o), reps=10)
        caus = gpu.bench(lambda: attention_flash(q, k, v, True, BM, BN,
                                                 num_warps=NW, out=o), reps=10)
        sc, pr = gpu.empty(H, S, S), gpu.empty(H, S, S)
        m_full = gpu.bench(lambda: materialised(q, k, v, False, (sc, pr, o)),
                           reps=5)
        m_caus = gpu.bench(lambda: materialised(q, k, v, True, (sc, pr, o)),
                           reps=5)
        del sc, pr
        torch.cuda.empty_cache()
        rows.append(dict(S=S, flash_full=full, flash_causal=caus,
                         flash_ratio=full / caus,
                         mat_full=m_full, mat_causal=m_caus,
                         mat_ratio=m_full / m_caus))
        r = rows[-1]
        print("   S=%-5d flash %.4f -> %.4f ms (%.2fx faster)   "
              "materialised %.4f -> %.4f ms (%.2fx)"
              % (S, full, caus, r["flash_ratio"], m_full, m_caus,
                 r["mat_ratio"]))
    R["causal"] = rows


def section_f():
    print("\nF. tile shapes for the flash kernel (S=2048)")
    S = 2048
    (_, _, _), (q, k, v) = make(H, S, D, seed=1)
    o = gpu.empty(H, S, D)
    rows = []
    for (bm, bn, nw) in [(32, 32, 4), (64, 32, 4), (64, 32, 8), (64, 64, 4),
                         (64, 64, 8), (128, 32, 8)]:
        try:
            info = flash_compile_info(q, k, v, False, bm, bn, nw)
            ms = gpu.bench(lambda: attention_flash(q, k, v, False, bm, bn,
                                                   num_warps=nw, out=o), reps=10)
            rows.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, ok=True,
                             ms=ms, gflops=flops(H, S, D) / (ms * 1e6), **info))
            r = rows[-1]
            print("   BM=%-4d BN=%-3d warps=%-2d  %8.4f ms %8.0f GFLOP/s  "
                  "%3d regs  %3d spilled  %5.1f KB shared%s"
                  % (bm, bn, nw, ms, r["gflops"], r["regs"], r["spills"],
                     r["shared"] / 1024,
                     "   <- SPILLING" if r["spills"] else ""))
        except Exception as e:
            msg = str(e).split("\n")[0][:60]
            rows.append(dict(BLOCK_M=bm, BLOCK_N=bn, num_warps=nw, ok=False,
                             error=msg))
            print("   BM=%-4d BN=%-3d warps=%-2d  DID NOT COMPILE: %s"
                  % (bm, bn, nw, msg))
    R["tiles"] = rows


def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    m = R["memory"]
    ax[0].plot([r["S"] for r in m], [r["scores_bytes"] / 1e6 for r in m], "s-",
               color="#d62728", label="score matrices (materialised)")
    ax[0].plot([r["S"] for r in m], [r["qkvo_bytes"] / 1e6 for r in m], "o-",
               color="#2ca02c", label="Q,K,V,O (all either path needs)")
    ax[0].axhline(CARD_BYTES / 1e6, color="black", ls="--", label="8 GB card")
    ax[0].set_xscale("log", base=2)
    ax[0].set_yscale("log", base=10)
    ax[0].set_xlabel("sequence length S")
    ax[0].set_ylabel("MB")
    ax[0].set_title("B. quadratic vs linear")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, which="both")

    s = R["speed"]
    ax[1].plot([r["S"] for r in s], [r["materialised_ms"] for r in s], "s-",
               color="#d62728", label="materialised (3 kernels)")
    ax[1].plot([r["S"] for r in s], [r["flash_ms"] for r in s], "o-",
               color="#2ca02c", label="flash (1 kernel)")
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log", base=10)
    ax[1].set_xlabel("sequence length S")
    ax[1].set_ylabel("ms")
    ax[1].set_title("C. same FLOPs, %.2fx apart at S=%d"
                    % (s[-1]["speedup"], s[-1]["S"]))
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")

    c = R["causal"]
    xs = range(len(c))
    ax[2].bar([x - 0.2 for x in xs], [r["flash_ratio"] for r in c], 0.4,
              color="#2ca02c", label="flash: skips the masked blocks")
    ax[2].bar([x + 0.2 for x in xs], [r["mat_ratio"] for r in c], 0.4,
              color="#d62728", label="materialised: computes then masks")
    ax[2].axhline(1.0, color="black", lw=0.8)
    ax[2].set_xticks(list(xs))
    ax[2].set_xticklabels(["S=%d" % r["S"] for r in c])
    ax[2].set_ylabel("speedup from causal masking")
    ax[2].set_title("E. half the work, if you can avoid doing it")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    R["device"] = gpu.device_note()
    R["config"] = dict(H=H, D=D, BLOCK_M=BM, BLOCK_N=BN, num_warps=NW)
    d = R["device"]
    print("device: %s (cc %s, %d SMs, %.1f GB)   %d heads, D=%d, flash tile "
          "%dx%d\n" % (d["name"], d["cc"], d["sms"], d["mem_gb"], H, D, BM, BN))
    gpu.warm_up()
    section_a()
    section_b()
    section_cd()
    section_e()
    section_f()

    with open(os.path.join(OUT, "findings.json"), "w") as fh:
        json.dump(R, fh, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "a", "b", "c"])
        for r in R["correctness"]:
            w.writerow(["A", "S%d_causal%d" % (r["S"], r["causal"]),
                        "%.3e" % r["flash"], "%.3e" % r["materialised"], ""])
        for r in R["memory"]:
            w.writerow(["B", "S_%d" % r["S"], r["scores_bytes"],
                        r["qkvo_bytes"], "fits" if r["fits"] else "no-fit"])
        for r in R["speed"]:
            w.writerow(["C", "S_%d" % r["S"], "%.4f" % r["materialised_ms"],
                        "%.4f" % r["flash_ms"], "%.3f" % r["speedup"]])
            w.writerow(["D", "S_%d" % r["S"], "%.1f" % r["materialised_mb"],
                        "%.1f" % r["flash_mb"], "%.2f" % r["traffic_ratio"]])
        for r in R["causal"]:
            w.writerow(["E", "S_%d" % r["S"], "%.3f" % r["flash_ratio"],
                        "%.3f" % r["mat_ratio"], ""])
        for r in R["tiles"]:
            if r["ok"]:
                w.writerow(["F", "%dx%d_w%d" % (r["BLOCK_M"], r["BLOCK_N"],
                                                r["num_warps"]),
                            "%.0f" % r["gflops"], r["regs"], r["spills"]])
            else:
                w.writerow(["F", "%dx%d_w%d" % (r["BLOCK_M"], r["BLOCK_N"],
                                                r["num_warps"]),
                            "did-not-compile", r["error"], ""])

    p = plot(os.path.join(OUT, "mini_flashattention.png"))
    print("\nwrote outputs/findings.json, outputs/findings.csv%s"
          % (", " + os.path.relpath(p, HERE) if p else ""))


if __name__ == "__main__":
    main()
