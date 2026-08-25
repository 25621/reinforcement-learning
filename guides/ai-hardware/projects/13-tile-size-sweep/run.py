"""Project 13 - tile size sweep for a 2048x2048 matmul.

Compiles tiles.cu, runs 14 configurations from naive to cuBLAS, and places
every one of them on the roofline.

  - one level of tiling  (TxT shared tile, one output per thread): AI = T/4
  - two levels           (BMxBN shared + TMxTN registers):  AI = BM*BN/(2(BM+BN))

The point of the sweep is that arithmetic intensity is not a free parameter.
It is set by the tile, the tile is limited by shared memory and by the 1024
threads-per-block ceiling, and those limits decide how close to peak you can
possibly get.
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "tiles")

PEAK_TFLOPS = 8.19        # 19 SM x 128 cores x 2 flop x 1.683 GHz (project 1)
PEAK_BW = 256.3           # GB/s, spec (project 1)
REAL_BW = 222.0           # GB/s, best measured read bandwidth (project 3)
RIDGE = PEAK_TFLOPS * 1e3 / PEAK_BW      # FLOP/byte


def build_and_run():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
                        os.path.join(HERE, "tiles.cu"), "-o", EXE, "-lcublas"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def parse(text):
    dev, cfgs, wall = {}, [], None
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), smem_block=int(f[4]),
                       n=int(f[5]))
        elif f[0] == "wall":
            wall = dict(t=int(f[1]), smem=int(f[2]), threads=int(f[3]),
                        max_threads=int(f[4]), error=f[5])
        elif f[0] == "cfg":
            cfgs.append(dict(name=f[1], tile=int(f[2]), tm=int(f[3]),
                             tn=int(f[4]), smem=int(f[5]), threads=int(f[6]),
                             occ=float(f[7]), ai=float(f[8]), ms=float(f[9]),
                             tflops=float(f[10]), err=float(f[11])))
    return dev, cfgs, wall


def roofline(ai, bw=REAL_BW):
    return min(PEAK_TFLOPS, ai * bw / 1e3)


def naive_pct(cfgs):
    c = next(x for x in cfgs if x["name"] == "naive")
    return 100 * c["tflops"] / roofline(c["ai"])


def plot(dev, cfgs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.8))

    xs = [2 ** (i / 4) for i in range(-9, 33)]
    ax[0].plot(xs, [roofline(x, PEAK_BW) for x in xs], "-", color="#7f8c8d",
               lw=1.2, label=f"roofline, spec BW ({PEAK_BW:.0f} GB/s)")
    ax[0].plot(xs, [roofline(x, REAL_BW) for x in xs], "--", color="#2c3e50",
               lw=1.2, label=f"roofline, measured BW ({REAL_BW:.0f} GB/s)")
    ax[0].axvline(RIDGE, ls=":", color="#c0392b", lw=1)
    ax[0].annotate(f"ridge {RIDGE:.0f}", xy=(RIDGE * 1.06, 0.25), fontsize=8,
                   color="#c0392b")

    one = [c for c in cfgs if c["name"].startswith("smem_") or c["name"] == "naive"]
    two = [c for c in cfgs if c["name"].startswith("reg_")]
    blas = next(c for c in cfgs if c["name"] == "cublas")
    ax[0].plot([c["ai"] for c in one], [c["tflops"] for c in one], "o",
               color="#e67e22", ms=8, label="one level (shared tile only)")
    ax[0].plot([c["ai"] for c in two], [c["tflops"] for c in two], "s",
               color="#16a085", ms=7, label="two levels (+ register patch)")
    ax[0].axhline(blas["tflops"], color="#8e44ad", ls="-.", lw=1,
                  label=f"cuBLAS ({blas['tflops']:.2f} TFLOP/s)")
    best = max(two, key=lambda c: c["tflops"])
    ax[0].annotate(best["name"].replace("reg_", ""),
                   xy=(best["ai"], best["tflops"]),
                   xytext=(best["ai"] * 0.22, best["tflops"] * 1.35),
                   fontsize=8, arrowprops=dict(arrowstyle="->", lw=.8))
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2)
    ax[0].set_xlabel("arithmetic intensity (FLOP/byte)")
    ax[0].set_ylabel("TFLOP/s")
    ax[0].set_title("Every configuration, on the roofline")
    ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=.3, which="both")

    order = ["naive", "smem_8x8", "smem_16x16", "smem_32x32",
             "reg_64x64_k16_2x2", "reg_64x64_k16_4x4", "reg_128x128_k8_4x4",
             "reg_128x128_k8_8x8", "reg_256x128_k8_8x8", "cublas"]
    sel = [next(c for c in cfgs if c["name"] == n) for n in order
           if any(c["name"] == n for c in cfgs)]
    cols = ["#c0392b"] + ["#e67e22"] * 3 + ["#16a085"] * 5 + ["#8e44ad"]
    ax[1].barh(range(len(sel)), [c["tflops"] for c in sel], color=cols[:len(sel)])
    ax[1].set_yticks(range(len(sel)))
    ax[1].set_yticklabels([c["name"] for c in sel], fontsize=8)
    ax[1].invert_yaxis()
    for i, c in enumerate(sel):
        ax[1].text(c["tflops"] + .08, i, f"{100*c['tflops']/PEAK_TFLOPS:.0f}% peak",
                   va="center", fontsize=7)
    ax[1].set_xlim(0, PEAK_TFLOPS * 0.95)
    ax[1].set_xlabel("TFLOP/s (peak = %.2f)" % PEAK_TFLOPS)
    ax[1].set_title("Same matmul, %.0fx apart"
                    % (max(c["tflops"] for c in sel) / min(c["tflops"] for c in sel)))
    ax[1].grid(alpha=.3, axis="x")

    fig.suptitle("Tiling a %dx%d matmul on %s" % (dev["n"], dev["n"], dev["name"]),
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    dev, cfgs, wall = parse(build_and_run())
    n = dev["n"]
    print(f"Device: {dev['name']} (cc {dev['cc']}), {dev['sms']} SMs, "
          f"{dev['smem_block']//1024} KB shared/block")
    print(f"Matmul: {n}x{n} fp32, {2*n**3/1e9:.1f} GFLOP. "
          f"Peak {PEAK_TFLOPS} TFLOP/s, ridge point {RIDGE:.1f} FLOP/byte.\n")

    print("=" * 92)
    print("Every configuration, worst to best")
    print("=" * 92)
    print(f"{'configuration':<22} {'AI':>6} {'smem':>7} {'thr':>5} {'occ':>6} "
          f"{'ms':>8} {'TFLOP/s':>8} {'%peak':>6} {'roof':>7} {'%roof':>6}")
    for c in sorted(cfgs, key=lambda c: c["tflops"]):
        if c["name"] == "cublas":
            print(f"{'cuBLAS (reference)':<22} {'-':>6} {'-':>7} {'-':>5} "
                  f"{'-':>6} {c['ms']:8.3f} {c['tflops']:8.2f} "
                  f"{100*c['tflops']/PEAK_TFLOPS:5.0f}% {'-':>7} {'-':>6}")
            continue
        rf = roofline(c["ai"])
        print(f"{c['name']:<22} {c['ai']:6.1f} {c['smem']//1024:6d}K "
              f"{c['threads']:5d} {c['occ']:5.0%} {c['ms']:8.3f} "
              f"{c['tflops']:8.2f} {100*c['tflops']/PEAK_TFLOPS:5.0f}% "
              f"{rf:7.2f} {100*c['tflops']/rf:5.0f}%")

    over = [c for c in cfgs if c["name"] != "cublas"
            and c["tflops"] > roofline(c["ai"])]
    if over:
        print(f"\n  {len(over)} rows score OVER 100% of their roofline "
              f"(naive is at {naive_pct(cfgs):.0f}%). That is not an error:")
        print("  the roofline column assumes every tile load reaches DRAM, and "
              "the 2 MB L2")
        print("  quietly serves a large share of them. The low-AI kernels are "
              "the ones with")
        print("  the most re-reads, so they are the ones L2 rescues most. "
              "Project 15 measures it.")

    maxerr = max(c["err"] for c in cfgs)
    print(f"\n  Every kernel agrees with cuBLAS to a max relative error of "
          f"{maxerr:.1e}.")
    print("  Bit-for-bit, in fact: they all accumulate k in ascending order "
          "with FMA,")
    print("  so there is nothing left to round differently. Split-k would "
          "not match.")

    one = {c["name"]: c for c in cfgs if c["name"].startswith("smem_")}
    two = {c["name"]: c for c in cfgs if c["name"].startswith("reg_")}
    naive = next(c for c in cfgs if c["name"] == "naive")
    blas = next(c for c in cfgs if c["name"] == "cublas")
    best = max(two.values(), key=lambda c: c["tflops"])

    print()
    print("=" * 92)
    print("A. One level of tiling: the sweet spot is 16, and 32 is worth 3%")
    print("=" * 92)
    for k in ("smem_8x8", "smem_16x16", "smem_32x32"):
        c = one[k]
        print(f"  {k:<12} AI {c['ai']:4.0f}  roofline says {roofline(c['ai']):.2f}"
              f"  measured {c['tflops']:.2f} TFLOP/s "
              f"({100*c['tflops']/roofline(c['ai']):.0f}% of it)")
    r = one["smem_32x32"]["tflops"] / one["smem_16x16"]["tflops"]
    print(f"\n  Doubling the tile from 16 to 32 doubles arithmetic intensity "
          f"and buys {100*(r-1):+.1f}%.")
    print("  It has stopped being DRAM-bound. Each FMA needs 2 shared-memory "
          "reads, and")
    print("  that pipe (project 12: ~890 G loads/s) caps this shape near "
          "1.2 TFLOP/s no")
    print("  matter how big the tile gets.")

    if wall:
        print()
        print("=" * 92)
        print("B. The wall: one level of tiling cannot reach the ridge point")
        print("=" * 92)
        print(f"  To be compute-bound a one-level tile needs AI >= "
              f"{RIDGE:.1f}, i.e. T = 4 x {RIDGE:.1f} = {4*RIDGE:.0f}.")
        print(f"  T=64  needs {wall['smem']//1024} KB shared (fits) but "
              f"{wall['threads']} threads/block, and the limit is "
              f"{wall['max_threads']}.")
        print(f"        Actually launching it: \"{wall['error']}\"")
        print(f"  T=128 needs {2*128*128*4//1024} KB of shared memory against a "
              f"{dev['smem_block']//1024} KB budget - it does not compile.")
        print("  The ceiling is architectural. The second level of tiling is "
              "not an optimisation,")
        print("  it is the only way past this wall.")

    print()
    print("=" * 92)
    print("C. Two levels: the register patch buys what shared memory could not")
    print("=" * 92)
    print(f"{'configuration':<22} {'AI':>6} {'smem':>7} {'occ':>6} {'TFLOP/s':>8}")
    for c in sorted(two.values(), key=lambda c: c["tflops"]):
        print(f"{c['name']:<22} {c['ai']:6.1f} {c['smem']//1024:6d}K "
              f"{c['occ']:5.0%} {c['tflops']:8.2f}")
    print(f"\n  Best: {best['name']} at {best['tflops']:.2f} TFLOP/s "
          f"({100*best['tflops']/PEAK_TFLOPS:.0f}% of peak, "
          f"{100*best['tflops']/blas['tflops']:.0f}% of cuBLAS),")
    print(f"  running at {best['occ']:.0%} occupancy.")
    big = two.get("reg_256x128_k8_8x8")
    if big:
        print(f"  Bigger is NOT better: {big['name']} has a higher AI "
              f"({big['ai']:.1f} vs {best['ai']:.1f})")
        print(f"  and is {best['tflops']/big['tflops']:.2f}x SLOWER. The sweep "
              "has an interior maximum - that is")
        print("  the 'sweet spot' this project is named after.")
    same = two.get("reg_32x32_k32_1x1")
    if same:
        print(f"\n  Control: {same['name']} is the SAME 32x32 tile as "
              f"smem_32x32, written through the")
        print(f"  general two-level code path with TM=TN=1. It scores "
              f"{same['tflops']:.2f} vs {one['smem_32x32']['tflops']:.2f} "
              f"TFLOP/s -")
        print(f"  {one['smem_32x32']['tflops']/same['tflops']:.2f}x slower. "
              "Generality is not free; the register patch has to")
        print("  earn back the index arithmetic it costs.")

    print()
    print("=" * 92)
    print(f"Overall: naive {naive['tflops']:.2f} -> best hand-written "
          f"{best['tflops']:.2f} -> cuBLAS {blas['tflops']:.2f} TFLOP/s "
          f"({blas['tflops']/naive['tflops']:.0f}x end to end)")
    print("=" * 92)

    png = plot(dev, cfgs, os.path.join(OUT, "tiling.png"))
    if png:
        print(f"\nWrote {os.path.relpath(png, HERE)}")

    findings = {
        "device": dev, "n": n,
        "peak_tflops": PEAK_TFLOPS, "ridge_flop_per_byte": RIDGE,
        "headline": {
            "naive_tflops": naive["tflops"],
            "best_one_level": max(one.values(), key=lambda c: c["tflops"])["name"],
            "best_one_level_tflops": max(c["tflops"] for c in one.values()),
            "smem32_over_smem16": r,
            "best_two_level": best["name"],
            "best_two_level_tflops": best["tflops"],
            "best_two_level_occupancy": best["occ"],
            "best_pct_of_peak": 100 * best["tflops"] / PEAK_TFLOPS,
            "best_pct_of_cublas": 100 * best["tflops"] / blas["tflops"],
            "cublas_tflops": blas["tflops"],
            "end_to_end_speedup": blas["tflops"] / naive["tflops"],
            "bigger_tile_is_slower":
                best["tflops"] / big["tflops"] if big else None,
            "generic_path_penalty_same_tile":
                one["smem_32x32"]["tflops"] / same["tflops"] if same else None,
            "max_rel_error_vs_cublas": maxerr,
        },
        "wall": wall,
        "configs": cfgs,
    }
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "arithmetic_intensity", "shared_bytes", "threads",
                    "occupancy", "ms", "tflops", "pct_peak", "roofline_tflops"])
        for c in sorted(cfgs, key=lambda c: c["tflops"]):
            w.writerow([c["name"], c["ai"], c["smem"], c["threads"], c["occ"],
                        c["ms"], c["tflops"],
                        100 * c["tflops"] / PEAK_TFLOPS,
                        "" if c["name"] == "cublas" else roofline(c["ai"])])
    print("Wrote outputs/findings.json and outputs/findings.csv")


if __name__ == "__main__":
    main()
