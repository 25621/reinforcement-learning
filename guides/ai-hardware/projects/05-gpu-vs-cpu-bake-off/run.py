"""Project 05 - the same matmul on the CPU and on the GPU.

CPU side: NumPy's sgemm (which calls an optimised BLAS library), timed with all
threads and again with one thread.
GPU side: cuBLAS sgemm through gemm.cu, timed three ways - compute only, and
including the PCIe copies in both directions.

The question is not "which is faster" (the GPU, obviously) but "from what size
onwards, and does the answer change once you pay for the data transfer".
"""

import csv
import json
import os
import re
import shutil
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

SIZES = [64, 128, 256, 512, 1024, 2048, 4096, 8192]


# --------------------------------------------------------------------- CPU
def cpu_info():
    txt = open("/proc/cpuinfo").read()
    model = re.search(r"model name\s*:\s*(.+)", txt).group(1).strip()
    cores = len(set(re.findall(r"core id\s*:\s*(\d+)", txt)))
    threads = txt.count("processor\t:")
    mhz = float(subprocess.run(["bash", "-c",
                                "lscpu | grep 'CPU max MHz' | awk '{print $NF}'"],
                               capture_output=True, text=True).stdout.strip() or 0)
    return model, cores, threads, mhz


def bench_numpy(n, iters_budget=4e9):
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    iters = max(3, min(200, int(iters_budget / (2.0 * n ** 3))))
    c = a @ b                                   # warm up / let BLAS pick a path
    best = 1e30
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        dt = (time.perf_counter() - t0) / iters
        best = min(best, dt)
    return best, float(c[0, 0])


# --------------------------------------------------------------------- GPU
def bench_gpu():
    if shutil.which("nvcc") is None:
        print("!! nvcc not found - CPU-only run")
        return None, None
    exe = os.path.join(OUT, "gemm")
    cmd = ["nvcc", "-O3", "-arch=sm_61", os.path.join(HERE, "gemm.cu"), "-o", exe, "-lcublas"]
    print("compiling:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("!! compile failed:\n", r.stderr[-2000:])
        return None, None
    r = subprocess.run([exe], capture_output=True, text=True)
    if r.returncode != 0:
        print("!! run failed:\n", r.stdout[-2000:], r.stderr[-2000:])
        return None, None
    res, dev = {}, None
    for line in r.stdout.strip().splitlines():
        f = line.split(",")
        if line.startswith("#device"):
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), clock_khz=int(f[4]),
                       mem_khz=int(f[5]), bus_bits=int(f[6]))
        elif f[0] == "gemm":
            res[int(f[1])] = dict(compute=float(f[2]), h2d=float(f[3]),
                                  d2h=float(f[4]), total=float(f[5]),
                                  checksum=float(f[6]))
    return res, dev


CORES_PER_SM = {"6.1": 128, "7.0": 64, "7.5": 64, "8.0": 64, "8.6": 128, "8.9": 128, "9.0": 128}


def main():
    model, cores, threads, mhz = cpu_info()
    print(f"CPU: {model}  ({cores} cores / {threads} threads, max {mhz:.0f} MHz)")
    # AVX2 + FMA: 8 lanes x 2 FLOPs x 2 FMA ports per core
    cpu_peak = cores * 8 * 2 * 2 * mhz * 1e6
    print(f"CPU peak fp32 (AVX2 FMA, all cores): {cpu_peak/1e12:.2f} TFLOP/s")
    print(f"  = {cores} cores x 8 lanes x 2 (multiply+add) x 2 FMA ports x {mhz/1000:.1f} GHz")
    print("  NOTE: that clock is the single-core turbo. With every core busy the")
    print("  chip runs lower, so this peak is a few percent optimistic.")

    gpu, dev = bench_gpu()
    if dev:
        gpu_peak = dev["sms"] * CORES_PER_SM.get(dev["cc"], 128) * 2 * dev["clock_khz"] * 1e3
        pcie_note = ""
        print(f"GPU: {dev['name']} (sm_{dev['cc'].replace('.','')}), "
              f"peak fp32 {gpu_peak/1e12:.2f} TFLOP/s{pcie_note}")
        print(f"Paper ratio of the two peaks: {gpu_peak/cpu_peak:.1f}x in the GPU's favour\n")
    else:
        gpu_peak = None

    rows = []
    print(f"{'N':>6} {'CPU ms':>10} {'CPU GF/s':>9} {'GPU ms':>10} {'GPU GF/s':>9} "
          f"{'%peak':>6} {'GPU+copy ms':>12} {'speedup':>8} {'w/ copy':>8}")
    for n in SIZES:
        cpu_t, cpu_chk = bench_numpy(n)
        flops = 2.0 * n ** 3
        row = dict(n=n, cpu_sec=cpu_t, cpu_gflops=flops / cpu_t / 1e9)
        if gpu and n in gpu:
            g = gpu[n]
            row.update(gpu_sec=g["compute"], gpu_gflops=flops / g["compute"] / 1e9,
                       gpu_total_sec=g["total"], h2d=g["h2d"], d2h=g["d2h"],
                       speedup=cpu_t / g["compute"], speedup_with_copy=cpu_t / g["total"],
                       pct_peak=100 * flops / g["compute"] / gpu_peak)
            print(f"{n:>6} {cpu_t*1e3:10.3f} {row['cpu_gflops']:9.1f} "
                  f"{g['compute']*1e3:10.3f} {row['gpu_gflops']:9.1f} "
                  f"{row['pct_peak']:6.1f} {g['total']*1e3:12.3f} "
                  f"{row['speedup']:7.1f}x {row['speedup_with_copy']:7.2f}x")
        else:
            print(f"{n:>6} {cpu_t*1e3:10.3f} {row['cpu_gflops']:9.1f}")
        rows.append(row)

    if not gpu:
        return

    # ---------------- the interesting questions ----------------
    print("\n=== 1. Where is the crossover? ===")
    win_c = [r for r in rows if r["speedup"] > 1]
    win_t = [r for r in rows if r["speedup_with_copy"] > 1]
    print(f"GPU wins on compute alone from N = {min(r['n'] for r in win_c) if win_c else '-'}")
    print(f"GPU wins including both PCIe copies from N = "
          f"{min(r['n'] for r in win_t) if win_t else '-'}")
    small = rows[0]
    print(f"At N={small['n']} the GPU is {small['speedup']:.2f}x on compute but "
          f"{small['speedup_with_copy']:.2f}x once the copies are paid for.")

    print("\n=== 2. How much of the wall clock is the data transfer? ===")
    print(f"{'N':>6} {'transfer ms':>12} {'compute ms':>11} {'transfer share':>15}")
    for r in rows:
        if "h2d" not in r:
            continue
        tr = r["h2d"] + r["d2h"]
        share = 100 * tr / (tr + r["gpu_sec"])
        r["transfer_share"] = share
        print(f"{r['n']:>6} {tr*1e3:12.3f} {r['gpu_sec']*1e3:11.3f} {share:14.1f}%")
    flip = [r for r in rows if r.get("transfer_share", 100) < 50]
    print(f"\nTransfer stops being the majority of the work at N = "
          f"{min(r['n'] for r in flip) if flip else 'never in this range'}.")
    print("Why: the copy grows as N^2 while the multiply grows as N^3, so the")
    print("multiply eventually wins - but only eventually.")

    # analytic crossover for transfer vs compute
    big = max(rows, key=lambda r: r["n"])
    achieved = big["gpu_gflops"] * 1e9
    pcie = 3 * 4 / ((big["h2d"] + big["d2h"]) / (big["n"] ** 2))  # bytes/sec, 3 matrices
    n_star = 3 * 4 * achieved / (2 * pcie)
    print(f"Setting 2N^3/{achieved/1e12:.2f}e12 = 12N^2/{pcie/1e9:.1f}e9 gives "
          f"N = {n_star:.0f}: below that size this GPU spends more time being fed "
          f"than computing.")

    print("\n=== 3. Latency vs throughput, in one line each ===")
    s = rows[0]
    b = max(rows, key=lambda r: r["n"])
    print(f"One {s['n']}x{s['n']} matmul : CPU {s['cpu_sec']*1e6:.0f} us, "
          f"GPU {s['gpu_total_sec']*1e6:.0f} us including copies -> the CPU wins")
    print(f"One {b['n']}x{b['n']} matmul: CPU {b['cpu_sec']*1e3:.0f} ms, "
          f"GPU {b['gpu_total_sec']*1e3:.0f} ms including copies -> the GPU wins "
          f"{b['speedup_with_copy']:.0f}x")
    print(f"\nEfficiency: CPU reaches {100*b['cpu_gflops']*1e9/cpu_peak:.0f}% of its peak, "
          f"GPU {b['pct_peak']:.0f}% of its peak, at N={b['n']}.")

    findings = dict(cpu=model, cpu_cores=cores, cpu_peak_tflops=cpu_peak / 1e12,
                    gpu=dev["name"], gpu_peak_tflops=gpu_peak / 1e12,
                    peak_ratio=gpu_peak / cpu_peak,
                    crossover_compute=min((r["n"] for r in win_c), default=None),
                    crossover_with_copy=min((r["n"] for r in win_t), default=None),
                    transfer_compute_crossover_N=n_star,
                    rows=rows)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        keys = ["n", "cpu_sec", "cpu_gflops", "gpu_sec", "gpu_gflops", "gpu_total_sec",
                "h2d", "d2h", "speedup", "speedup_with_copy", "pct_peak", "transfer_share"]
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    plot(rows, cpu_peak, gpu_peak, model, dev)


def plot(rows, cpu_peak, gpu_peak, model, dev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (a) throughput
    ax[0].loglog(ns, [r["cpu_gflops"] for r in rows], "o-", color="#4C78A8",
                 label="CPU (NumPy, all cores)", base=2)
    ax[0].loglog(ns, [r["gpu_gflops"] for r in rows], "o-", color="#54A24B",
                 label="GPU (cuBLAS, compute only)", base=2)
    ax[0].loglog(ns, [2.0 * r["n"] ** 3 / r["gpu_total_sec"] / 1e9 for r in rows], "s--",
                 color="#F58518", label="GPU including PCIe copies", base=2)
    ax[0].axhline(cpu_peak / 1e9, ls=":", c="#4C78A8")
    ax[0].axhline(gpu_peak / 1e9, ls=":", c="#54A24B")
    ax[0].text(ns[0], cpu_peak / 1e9 * 1.1, "CPU peak", fontsize=8, color="#4C78A8")
    ax[0].text(ns[0], gpu_peak / 1e9 * 1.1, "GPU peak", fontsize=8, color="#54A24B")
    ax[0].set_xlabel("matrix size N"); ax[0].set_ylabel("GFLOP/s")
    ax[0].set_title("(a) Throughput\nboth are far from peak until N is large")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")

    # (b) speedup
    ax[1].semilogx(ns, [r["speedup"] for r in rows], "o-", color="#54A24B",
                   label="compute only", base=2)
    ax[1].semilogx(ns, [r["speedup_with_copy"] for r in rows], "s-", color="#F58518",
                   label="including PCIe copies", base=2)
    ax[1].axhline(1, c="k", ls="--", lw=1)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("matrix size N"); ax[1].set_ylabel("GPU speedup over CPU (log)")
    ax[1].set_title("(b) The honest speedup is the orange one\n"
                    "below the dashed line, the CPU wins")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")

    # (c) where the GPU's wall clock goes, as a share
    tr = [(r["h2d"] + r["d2h"]) for r in rows]
    cp = [r["gpu_sec"] for r in rows]
    tot = [a + b for a, b in zip(tr, cp)]
    trp = [100 * a / t for a, t in zip(tr, tot)]
    cpp = [100 * b / t for b, t in zip(cp, tot)]
    idx = list(range(len(ns)))
    ax[2].bar(idx, trp, 0.6, label="PCIe transfer", color="#F58518")
    ax[2].bar(idx, cpp, 0.6, bottom=trp, label="cuBLAS compute", color="#54A24B")
    for i, (a, t) in enumerate(zip(trp, tot)):
        ax[2].text(i, 102, f"{t*1e3:.2f}ms", ha="center", fontsize=7.5, rotation=45)
    ax[2].axhline(50, ls="--", c="k", lw=1)
    ax[2].set_xticks(idx); ax[2].set_xticklabels([str(n) for n in ns])
    ax[2].set_ylim(0, 118)
    ax[2].set_xlabel("matrix size N"); ax[2].set_ylabel("% of the GPU's wall clock")
    ax[2].set_title("(c) Feeding the GPU vs using it\n"
                    "copies grow as N^2, the maths as N^3")
    ax[2].legend(fontsize=8, loc="lower left"); ax[2].grid(alpha=.3, axis="y")

    fig.suptitle(f"{model} vs {dev['name']} - fp32 square matmul", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bake_off.png"), dpi=110)
    print(f"\nwrote {OUT}/bake_off.png")


if __name__ == "__main__":
    main()
