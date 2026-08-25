"""Project 17 - a five-rung ladder from a naive matmul to 76% of cuBLAS.

Compiles sgemm.cu, runs the ladder at three sizes, then explains the result
three ways:

  A. throughput      - GFLOP/s per rung, and the share of cuBLAS reached
  B. the roofline    - each rung's arithmetic intensity and the ceiling it
                       implies, versus what it actually got
  C. instruction mix - the same story read straight out of the SASS: how many
                       multiply-adds each rung issues per memory instruction
  D. the cost        - registers, shared memory and occupancy per rung
"""

import csv
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "sgemm")

# measured constants for this card (project 1 / project 3 / project 14)
PEAK_FLOPS = 8190.0     # GFLOP/s, fp32 FMA
PEAK_BW = 256.3         # GB/s, spec
RIDGE = PEAK_FLOPS / PEAK_BW

# block-tile shape of each rung -> arithmetic intensity 1/(2*(1/BM + 1/BN))
SHAPES = {
    "naive":  ("1x1 (no tile)", 0.25),
    "smem":   ("32x32", 32 / 4.0),
    "tile1d": ("64x64", 64 / 4.0),
    "tile2d": ("128x128", 128 / 4.0),
    "vec":    ("128x128", 128 / 4.0),
}
ORDER = ["naive", "smem", "tile1d", "tile2d", "vec", "cublas"]


def build():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
                        os.path.join(HERE, "sgemm.cu"), "-o", EXE, "-lcublas"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])


def run():
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def sass_mix():
    """Count what each kernel actually issues, from the compiled SASS."""
    if shutil.which("cuobjdump") is None:
        return {}
    r = subprocess.run(["cuobjdump", "-sass", EXE], capture_output=True,
                       text=True)
    mix, cur = {}, None
    for line in r.stdout.splitlines():
        if "Function :" in line:
            # mangled names are _Z<len><name>..., so <len> gives the exact name
            m = re.search(r"Function : _Z(\d+)(\w+)", line)
            cur = None
            if m:
                name = m.group(2)[:int(m.group(1))]
                cur = name[len("sgemm_"):] if name.startswith("sgemm_") else None
            if cur:
                mix[cur] = {"FFMA": 0, "LDS": 0, "LDG": 0, "STG": 0, "STS": 0}
            continue
        if cur is None:
            continue
        m = re.search(r"\*/\s+(?:@!?P\d+\s+)?([A-Z0-9]+)", line)
        if not m:
            continue
        op = m.group(1)
        for key in ("FFMA", "LDS", "LDG", "STG", "STS"):
            if op.startswith(key):
                mix[cur][key] += 1
    return mix


def parse(text):
    d = {"runs": [], "attrs": []}
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            d["device"] = dict(name=f[1], cc=f[2], sms=int(f[3]), l2=int(f[4]),
                               smem_per_block=int(f[5]))
        elif f[0] == "k":
            d["runs"].append(dict(n=int(f[1]), kernel=f[2], ms=float(f[3]),
                                  gflops=float(f[4]), max_err=float(f[5])))
        elif f[0] == "attr":
            d["attrs"].append(dict(kernel=f[1], threads=int(f[2]),
                                   regs=int(f[3]), smem=int(f[4]),
                                   blocks_per_sm=int(f[5]),
                                   occupancy=float(f[6])))
    return d


def plot(d, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = {"naive": "#7f7f7f", "smem": "#1f77b4", "tile1d": "#2ca02c",
              "tile2d": "#ff7f0e", "vec": "#d62728", "cublas": "#111111"}

    # (1) the ladder at each size
    sizes = sorted({r["n"] for r in d["runs"]})
    w = 0.14
    for i, k in enumerate(ORDER):
        ys = [next(r["gflops"] for r in d["runs"] if r["n"] == n
                   and r["kernel"] == k) for n in sizes]
        ax[0].bar([x + (i - 2.5) * w for x in range(len(sizes))], ys, w,
                  label=k, color=colors[k])
    ax[0].axhline(PEAK_FLOPS, color="grey", ls=":", label="fp32 peak 8190")
    ax[0].set_xticks(range(len(sizes)))
    ax[0].set_xticklabels(["N=%d" % n for n in sizes])
    ax[0].set_ylabel("GFLOP/s")
    ax[0].set_title("A. the ladder")
    ax[0].legend(fontsize=7, ncol=2)
    ax[0].grid(alpha=.3, axis="y")

    # (2) roofline
    ai = [2 ** (i / 2.0) for i in range(-8, 16)]
    roof = [min(PEAK_FLOPS, a * PEAK_BW) for a in ai]
    ax[1].plot(ai, roof, color="black", lw=1.5, label="roofline")
    for k, (_, a) in SHAPES.items():
        g = next(r["gflops"] for r in d["runs"] if r["n"] == 4096
                 and r["kernel"] == k)
        ax[1].plot([a], [g], "o", color=colors[k], ms=9, label=k)
    ax[1].axvline(RIDGE, color="grey", ls=":", label="ridge %.1f" % RIDGE)
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log", base=2)
    ax[1].set_xlabel("arithmetic intensity (FLOP/byte of DRAM traffic)")
    ax[1].set_ylabel("GFLOP/s")
    ax[1].set_title("B. why each rung stops where it does (N=4096)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, which="both")

    # (3) instruction mix
    if d.get("mix"):
        ks = [k for k in ORDER[:-1] if k in d["mix"]]
        ratios = [d["mix"][k]["FFMA"] / max(1, d["mix"][k]["LDS"] +
                                            d["mix"][k]["LDG"]) for k in ks]
        gf = [next(r["gflops"] for r in d["runs"] if r["n"] == 4096
                   and r["kernel"] == k) for k in ks]
        ax[2].plot(ratios, gf, "o-", color="#444444")
        for k, x, y in zip(ks, ratios, gf):
            ax[2].annotate(k, (x, y), textcoords="offset points",
                           xytext=(6, -3), fontsize=8, color=colors[k])
        ax[2].set_xscale("log", base=2)
        ax[2].set_xlabel("FFMA issued per memory instruction (from SASS)")
        ax[2].set_ylabel("GFLOP/s at N=4096")
        ax[2].set_title("C. the same story, in instructions")
        ax[2].grid(alpha=.3, which="both")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def main():
    build()
    d = parse(run())
    d["mix"] = sass_mix()

    dev = d["device"]
    print("device: %s  cc %s  %d SMs  %d KB shared/block"
          % (dev["name"], dev["cc"], dev["sms"], dev["smem_per_block"] // 1024))
    print("peak: %.0f GFLOP/s fp32, %.1f GB/s, ridge point %.1f FLOP/byte\n"
          % (PEAK_FLOPS, PEAK_BW, RIDGE))

    sizes = sorted({r["n"] for r in d["runs"]})
    for n in sizes:
        cub = next(r for r in d["runs"] if r["n"] == n and r["kernel"] == "cublas")
        print("N = %d" % n)
        print("   %-8s %9s %10s %8s %10s %10s"
              % ("kernel", "ms", "GFLOP/s", "% peak", "% cuBLAS", "max err"))
        for k in ORDER:
            r = next(x for x in d["runs"] if x["n"] == n and x["kernel"] == k)
            print("   %-8s %9.4f %10.2f %7.1f%% %9.1f%% %10.3e"
                  % (k, r["ms"], r["gflops"], 100 * r["gflops"] / PEAK_FLOPS,
                     100 * r["gflops"] / cub["gflops"], r["max_err"]))
        print()

    print("B. roofline accounting at N=4096")
    print("   %-8s %-12s %8s %10s %10s %8s"
          % ("kernel", "block tile", "AI", "roof", "achieved", "of roof"))
    d["roofline"] = {}
    for k in ORDER[:-1]:
        shape, a = SHAPES[k]
        roof = min(PEAK_FLOPS, a * PEAK_BW)
        g = next(r["gflops"] for r in d["runs"] if r["n"] == 4096
                 and r["kernel"] == k)
        d["roofline"][k] = dict(tile=shape, ai=a, roof=roof, achieved=g,
                                of_roof=g / roof)
        print("   %-8s %-12s %8.2f %10.0f %10.0f %7.0f%%"
              % (k, shape, a, roof, g, 100 * g / roof))
    print()

    if d["mix"]:
        print("C. instruction mix (whole kernel body, from cuobjdump -sass)")
        print("   %-8s %8s %8s %8s %8s %14s"
              % ("kernel", "FFMA", "LDS", "LDG", "STS", "FFMA per load"))
        for k in ORDER[:-1]:
            m = d["mix"].get(k)
            if not m:
                continue
            ratio = m["FFMA"] / max(1, m["LDS"] + m["LDG"])
            print("   %-8s %8d %8d %8d %8d %14.2f"
                  % (k, m["FFMA"], m["LDS"], m["LDG"], m["STS"], ratio))
        print()

    print("D. what each rung costs")
    print("   %-8s %8s %6s %8s %11s %10s"
          % ("kernel", "threads", "regs", "smem B", "blocks/SM", "occupancy"))
    for a in d["attrs"]:
        print("   %-8s %8d %6d %8d %11d %9.0f%%"
              % (a["kernel"], a["threads"], a["regs"], a["smem"],
                 a["blocks_per_sm"], 100 * a["occupancy"]))

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(d, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "kernel", "ms", "gflops", "pct_peak", "pct_cublas",
                    "max_err"])
        for n in sizes:
            cub = next(r for r in d["runs"] if r["n"] == n
                       and r["kernel"] == "cublas")
            for k in ORDER:
                r = next(x for x in d["runs"] if x["n"] == n
                         and x["kernel"] == k)
                w.writerow([n, k, "%.4f" % r["ms"], "%.2f" % r["gflops"],
                            "%.1f" % (100 * r["gflops"] / PEAK_FLOPS),
                            "%.1f" % (100 * r["gflops"] / cub["gflops"]),
                            "%.3e" % r["max_err"]])

    p = plot(d, os.path.join(OUT, "tiled_matmul.png"))
    print("\nwrote outputs/findings.json, outputs/findings.csv%s"
          % (", " + os.path.relpath(p, HERE) if p else ""))


if __name__ == "__main__":
    main()
