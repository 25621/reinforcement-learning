"""Project 16 - the anatomy of a CUDA kernel, measured.

Compiles vecadd.cu, runs it twice (the normal run, then the out-of-bounds run
in a fresh process because an illegal access kills the CUDA context), then
tabulates and plots:

  A. correctness           - the GPU answer against a CPU answer
  B. the async trap        - what a host-side stopwatch reports without a sync
  C. block size            - 32 .. 1024 threads per block
  D. grid-stride loop      - fewer threads, each doing more
  E. the whole trip        - PCIe in, compute, PCIe out, versus just using
                             the CPU
  F. the reuse crossover   - how many kernel calls repay one round trip
  G. missing bounds check  - one case is silent, one kills the context
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "vecadd")


def build():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    cmd = ["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
           "-Xcompiler", "-fopenmp", os.path.join(HERE, "vecadd.cu"),
           "-o", EXE]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])


def run(mode):
    r = subprocess.run([EXE, mode], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def parse(text, oob_text):
    d = {"block": [], "stride": [], "reuse": [], "oob": []}
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            d["device"] = dict(name=f[1], cc=f[2], sms=int(f[3]),
                               l2=int(f[4]))
        elif f[0] == "#n":
            d["n"] = int(f[1])
        elif f[0] == "check":
            d["max_error"] = float(f[1])
        elif f[0] == "async":
            d["async"] = dict(no_sync_ms=float(f[1]), with_sync_ms=float(f[2]))
        elif f[0] == "block":
            d["block"].append(dict(threads=int(f[1]), blocks=int(f[2]),
                                   ms=float(f[3]), gbs=float(f[4])))
        elif f[0] == "stride":
            d["stride"].append(dict(waves=int(f[1]), blocks=int(f[2]),
                                    ms=float(f[3]), gbs=float(f[4])))
        elif f[0] == "e2e":
            d["e2e"] = dict(h2d_ms=float(f[1]), kernel_ms=float(f[2]),
                            d2h_ms=float(f[3]), cpu1_ms=float(f[4]),
                            cpu_omp_ms=float(f[5]), cpu_threads=int(f[6]))
        elif f[0] == "reuse":
            d["reuse"].append(dict(k=int(f[1]), gpu_ms=float(f[2]),
                                   cpu_ms=float(f[3])))
    for line in oob_text.strip().splitlines():
        f = line.split(",")
        if f[0] == "oob":
            d["oob"].append(dict(case=f[1], n=int(f[2]), threads=int(f[3]),
                                 past_end=int(f[4]), at_launch=f[5],
                                 at_sync=f[6]))
    return d


def derive(d):
    n = d["n"]
    bytes_moved = 3.0 * n * 4.0
    e = d["e2e"]
    d["derived"] = {
        "bytes_per_pass_mb": bytes_moved / 1e6,
        "kernel_gbs": bytes_moved / (e["kernel_ms"] * 1e6),
        "cpu_gbs": bytes_moved / (e["cpu_omp_ms"] * 1e6),
        "cpu1_gbs": bytes_moved / (e["cpu1_ms"] * 1e6),
        "pcie_h2d_gbs": 2.0 * n * 4.0 / (e["h2d_ms"] * 1e6),
        "pcie_d2h_gbs": 1.0 * n * 4.0 / (e["d2h_ms"] * 1e6),
        "trip_ms": e["h2d_ms"] + e["kernel_ms"] + e["d2h_ms"],
        "kernel_share_of_trip":
            e["kernel_ms"] / (e["h2d_ms"] + e["kernel_ms"] + e["d2h_ms"]),
        "gpu_vs_cpu_end_to_end":
            (e["h2d_ms"] + e["kernel_ms"] + e["d2h_ms"]) / e["cpu_omp_ms"],
        "cpu_thread_speedup": e["cpu1_ms"] / e["cpu_omp_ms"],
        "async_lie_factor": d["async"]["with_sync_ms"] / d["async"]["no_sync_ms"],
        "block_spread": (max(b["ms"] for b in d["block"])
                         / min(b["ms"] for b in d["block"])),
        "crossover_k": (e["h2d_ms"] + e["d2h_ms"])
                       / max(1e-9, e["cpu_omp_ms"] - e["kernel_ms"]),
    }
    return d


def plot(d, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # (1) block size and grid-stride, both as achieved bandwidth
    b = d["block"]
    ax[0].plot([x["threads"] for x in b], [x["gbs"] for x in b],
               "o-", color="#1f77b4", label="one element per thread")
    s = d["stride"]
    ax[0].plot([x["blocks"] for x in s], [x["gbs"] for x in s],
               "s--", color="#d62728", label="grid-stride loop (256 thr/blk)")
    ax[0].set_xscale("log", base=2)
    ax[0].axhline(256.3, color="grey", ls=":", label="spec peak 256 GB/s")
    ax[0].set_xlabel("threads per block  /  blocks in grid")
    ax[0].set_ylabel("GB/s")
    ax[0].set_ylim(0, 280)
    ax[0].set_title("C+D. launch shape barely matters\n(once the grid is big enough)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)

    # (2) where the time goes
    e, dv = d["e2e"], d["derived"]
    ax[1].bar(["GPU\n(whole trip)"], [e["h2d_ms"]], color="#ff7f0e",
              label="host to device")
    ax[1].bar(["GPU\n(whole trip)"], [e["kernel_ms"]], bottom=[e["h2d_ms"]],
              color="#2ca02c", label="kernel")
    ax[1].bar(["GPU\n(whole trip)"], [e["d2h_ms"]],
              bottom=[e["h2d_ms"] + e["kernel_ms"]], color="#ff7f0e")
    ax[1].bar(["CPU\n(%d threads)" % e["cpu_threads"]], [e["cpu_omp_ms"]],
              color="#1f77b4", label="CPU add")
    ax[1].set_ylabel("ms")
    ax[1].set_title("E. the kernel is %.1f%% of the trip -\nand the trip loses to the CPU by %.2fx"
                    % (100 * dv["kernel_share_of_trip"],
                       dv["gpu_vs_cpu_end_to_end"]))
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, axis="y")

    # (3) the reuse crossover
    r = d["reuse"]
    ax[2].plot([x["k"] for x in r], [x["gpu_ms"] for x in r], "o-",
               color="#2ca02c", label="GPU: 1 trip + k kernels")
    ax[2].plot([x["k"] for x in r], [x["cpu_ms"] for x in r], "s-",
               color="#1f77b4", label="CPU: k adds")
    ax[2].axvline(dv["crossover_k"], color="grey", ls=":",
                  label="crossover k = %.2f" % dv["crossover_k"])
    ax[2].set_xscale("log", base=2)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("k = kernel calls on data already on the GPU")
    ax[2].set_ylabel("total ms")
    ax[2].set_title("F. one round trip is repaid\nafter %.2f passes" % dv["crossover_k"])
    ax[2].legend(fontsize=7)
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    build()
    d = derive(parse(run("main"), run("oob")))

    dv, e = d["derived"], d["e2e"]
    dev = d["device"]
    print("device: %s  cc %s  %d SMs" % (dev["name"], dev["cc"], dev["sms"]))
    print("n = %d floats, %.0f MB moved per pass\n"
          % (d["n"], dv["bytes_per_pass_mb"]))

    print("A. max |gpu - cpu| = %.3e" % d["max_error"])
    print("B. host stopwatch without a sync: %.4f ms; with a sync: %.4f ms "
          "(%.0fx)" % (d["async"]["no_sync_ms"], d["async"]["with_sync_ms"],
                       dv["async_lie_factor"]))

    print("\nC. block size sweep")
    print("   %8s %10s %8s %8s" % ("threads", "blocks", "ms", "GB/s"))
    for b in d["block"]:
        print("   %8d %10d %8.4f %8.2f"
              % (b["threads"], b["blocks"], b["ms"], b["gbs"]))
    print("   spread fastest->slowest: %.3fx" % dv["block_spread"])

    print("\nD. grid-stride loop")
    print("   %8s %8s %8s %8s" % ("waves", "blocks", "ms", "GB/s"))
    for s in d["stride"]:
        print("   %8d %8d %8.4f %8.2f"
              % (s["waves"], s["blocks"], s["ms"], s["gbs"]))

    print("\nE. the whole trip")
    print("   host->device  %8.3f ms  (%.2f GB/s over PCIe)"
          % (e["h2d_ms"], dv["pcie_h2d_gbs"]))
    print("   kernel        %8.3f ms  (%.2f GB/s in GPU memory)"
          % (e["kernel_ms"], dv["kernel_gbs"]))
    print("   device->host  %8.3f ms  (%.2f GB/s over PCIe)"
          % (e["d2h_ms"], dv["pcie_d2h_gbs"]))
    print("   total         %8.3f ms  (kernel = %.1f%% of it)"
          % (dv["trip_ms"], 100 * dv["kernel_share_of_trip"]))
    print("   CPU 1 thread  %8.3f ms  (%.2f GB/s)"
          % (e["cpu1_ms"], dv["cpu1_gbs"]))
    print("   CPU %d threads %7.3f ms  (%.2f GB/s, only %.2fx faster than 1)"
          % (e["cpu_threads"], e["cpu_omp_ms"], dv["cpu_gbs"],
             dv["cpu_thread_speedup"]))
    print("   => the GPU trip is %.2fx SLOWER than just using the CPU"
          % dv["gpu_vs_cpu_end_to_end"])

    print("\nF. reuse crossover at k = %.2f passes" % dv["crossover_k"])

    print("\nG. missing bounds check")
    for o in d["oob"]:
        print("   %-6s %d threads over %d elements (%d past the end): "
              "launch=%s sync=%s"
              % (o["case"], o["threads"], o["n"], o["past_end"],
                 o["at_launch"], o["at_sync"]))

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(d, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key", "value", "ms", "gbs"])
        w.writerow(["A", "max_error", d["max_error"], "", ""])
        w.writerow(["B", "no_sync", "", d["async"]["no_sync_ms"], ""])
        w.writerow(["B", "with_sync", "", d["async"]["with_sync_ms"], ""])
        for b in d["block"]:
            w.writerow(["C", "block_%d" % b["threads"], b["blocks"], b["ms"],
                        b["gbs"]])
        for s in d["stride"]:
            w.writerow(["D", "stride_%dwaves" % s["waves"], s["blocks"],
                        s["ms"], s["gbs"]])
        for k in ("h2d_ms", "kernel_ms", "d2h_ms", "cpu1_ms", "cpu_omp_ms"):
            w.writerow(["E", k, "", e[k], ""])
        for r in d["reuse"]:
            w.writerow(["F", "k_%d" % r["k"], "", r["gpu_ms"], r["cpu_ms"]])
        for o in d["oob"]:
            w.writerow(["G", o["case"], o["past_end"], o["at_launch"],
                        o["at_sync"]])

    p = plot(d, os.path.join(OUT, "vector_add.png"))
    print("\nwrote %s" % ", ".join(x for x in [
        "outputs/findings.json", "outputs/findings.csv",
        os.path.relpath(p, HERE) if p else None] if x))


if __name__ == "__main__":
    main()
