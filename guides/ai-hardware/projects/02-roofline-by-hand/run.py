"""Project 02 - Roofline by hand, then checked against a real GPU.

Step 1: for five operations (matmul, layernorm, softmax, gelu, transpose) count
        FLOPs and bytes by hand and compute arithmetic intensity.
Step 2: compute the ridge point of several GPUs and predict which ops are
        memory-bound on each.
Step 3: compile bench_ops.cu, run the same five ops on the GPU that is actually
        in this machine, and see how close the prediction lands.

The prediction is hardware-independent arithmetic; only step 3 needs a GPU.
If nvcc is missing, steps 1 and 2 still run and the plot is drawn without the
measured points.
"""

import csv
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Spec sheets. Peaks are DENSE (no structured-sparsity doubling) so that the
# ridge points are comparable across generations.
# --------------------------------------------------------------------------
GPUS = [
    # name,                 peak TFLOP/s, dtype,  peak GB/s, year
    ("GTX 1070 Ti",          8.19,  "fp32",   256.3, 2017),
    ("V100 SXM2",          125.0,   "fp16",   900.0, 2017),
    ("A100 80GB",          312.0,   "bf16",  2039.0, 2020),
    ("H100 SXM",           989.4,   "bf16",  3350.0, 2022),
    ("H200 SXM",           989.4,   "bf16",  4800.0, 2024),
    ("B200 SXM",          2250.0,   "bf16",  8000.0, 2025),
]

# --------------------------------------------------------------------------
# Step 1: hand counts. M rows of D columns, fp32 (4 bytes) to match bench_ops.cu
# --------------------------------------------------------------------------
M, D, N_MM, N_TR = 8192, 4096, 4096, 8192
B = 4  # bytes per element (fp32)


def hand_ops():
    n = M * D
    return [
        # name        FLOPs                     bytes                       note
        ("matmul",    2 * N_MM ** 3,            3 * N_MM ** 2 * B,
         f"{N_MM}x{N_MM} @ {N_MM}x{N_MM}: read A, read B, write C"),
        ("layernorm", 8 * n,                    2 * n * B,
         f"{M}x{D}: read x, write y; ~8 FLOPs per element"),
        ("softmax",   5 * n,                    2 * n * B,
         f"{M}x{D}: read x, write y; max+exp+sum+div ~ 5 per element"),
        ("gelu",      8 * n,                    2 * n * B,
         f"{n} elements: read, write; tanh approximation ~ 8 per element"),
        ("transpose", 0,                        2 * N_TR ** 2 * B,
         f"{N_TR}x{N_TR}: read, write, ZERO arithmetic"),
    ]


# --------------------------------------------------------------------------
# Step 3: measure
# --------------------------------------------------------------------------
def measure():
    if shutil.which("nvcc") is None:
        print("!! nvcc not found - skipping the measured half")
        return None, None
    exe = os.path.join(OUT, "bench_ops")
    cmd = ["nvcc", "-O3", "-arch=sm_61", os.path.join(HERE, "bench_ops.cu"),
           "-o", exe, "-lcublas"]
    print("compiling:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("!! compile failed:\n", r.stderr[-2000:])
        return None, None
    r = subprocess.run([exe], capture_output=True, text=True)
    if r.returncode != 0:
        print("!! run failed:\n", r.stdout[-2000:], r.stderr[-2000:])
        return None, None
    meas, dev = {}, None
    for line in r.stdout.strip().splitlines():
        if line.startswith("#device"):
            f = line.split(",")
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), clock_khz=int(f[4]),
                       mem_khz=int(f[5]), bus_bits=int(f[6]))
            continue
        if line.startswith("#"):
            continue
        f = line.split(",")
        meas[f[0]] = dict(size=f[1], sec=float(f[2]), flops=float(f[3]),
                          bytes=float(f[4]))
    return meas, dev


CORES_PER_SM = {"6.1": 128, "7.0": 64, "7.5": 64, "8.0": 64, "8.6": 128, "8.9": 128, "9.0": 128}


def main():
    ops = hand_ops()

    # ---------------- step 1 ----------------
    print("=== Step 1: arithmetic intensity, counted by hand ===")
    print(f"{'op':<10} {'GFLOP':>10} {'MB moved':>10} {'AI (FLOP/byte)':>16}")
    for name, fl, by, note in ops:
        ai = fl / by
        print(f"{name:<10} {fl/1e9:10.2f} {by/1e6:10.1f} {ai:16.3f}   {note}")

    # ---------------- step 2 ----------------
    print("\n=== Step 2: ridge points ===")
    print("The ridge point is peak FLOP/s divided by peak bytes/s. An operation")
    print("whose AI is below it can never be compute-bound on that GPU.\n")
    print(f"{'GPU':<14} {'year':>5} {'peak':>10} {'GB/s':>8} {'ridge':>8}   memory-bound ops")
    ridge_rows = []
    for name, tf, dt, gbs, year in GPUS:
        ridge = tf * 1e12 / (gbs * 1e9)
        mb = [o[0] for o in ops if (o[1] / o[2]) < ridge]
        print(f"{name:<14} {year:>5} {tf:8.1f}T {gbs:8.0f} {ridge:8.1f}   {', '.join(mb)}")
        ridge_rows.append(dict(gpu=name, year=year, peak_tflops=tf, dtype=dt,
                               bw_gbs=gbs, ridge=ridge, memory_bound=" ".join(mb)))
    r0 = ridge_rows[0]["ridge"]
    r_h100 = [r for r in ridge_rows if r["gpu"] == "H100 SXM"][0]["ridge"]
    print(f"\nRidge point grew {r_h100/r0:.1f}x from the 2017 consumer card to the H100.")
    print("Same layernorm, same AI of 1.0 FLOP/byte:")
    print(f"  on the 1070 Ti it is {r0:.0f}x below the ridge")
    print(f"  on the H100      it is {r_h100:.0f}x below the ridge")

    # ---------------- step 3 ----------------
    print("\n=== Step 3: measured on the GPU in this machine ===")
    meas, dev = measure()
    rows = []
    if dev:
        cores = CORES_PER_SM.get(dev["cc"], 128)
        peak_flops = dev["sms"] * cores * 2 * dev["clock_khz"] * 1e3
        peak_bw = dev["mem_khz"] * 1e3 * 2 * dev["bus_bits"] / 8
        ridge = peak_flops / peak_bw
        print(f"device      : {dev['name']} (sm_{dev['cc'].replace('.','')}), "
              f"{dev['sms']} SMs @ {dev['clock_khz']/1e6:.3f} GHz")
        print(f"peak fp32   : {peak_flops/1e12:.2f} TFLOP/s  "
              f"= {dev['sms']} SM x {cores} cores x 2 x clock")
        print(f"peak memory : {peak_bw/1e9:.1f} GB/s  "
              f"= {dev['mem_khz']/1e6:.3f} GHz x 2 (DDR) x {dev['bus_bits']}/8 bytes")
        print(f"ridge point : {ridge:.1f} FLOP/byte\n")

        hdr = (f"{'op':<10} {'AI':>8} {'bound by':>10} {'pred ms':>9} {'meas ms':>9} "
               f"{'% of roof':>10} {'GFLOP/s':>10} {'GB/s':>8}")
        print(hdr)
        for name, fl, by, note in ops:
            if name not in meas:
                continue
            m = meas[name]
            ai = fl / by
            t_c = fl / peak_flops
            t_m = by / peak_bw
            pred = max(t_c, t_m)
            bound = "compute" if t_c > t_m else "memory"
            frac = 100 * pred / m["sec"]
            print(f"{name:<10} {ai:8.2f} {bound:>10} {pred*1e3:9.3f} {m['sec']*1e3:9.3f} "
                  f"{frac:10.1f} {fl/m['sec']/1e9:10.1f} {by/m['sec']/1e9:8.1f}")
            rows.append(dict(op=name, ai=ai, flops=fl, bytes=by, bound=bound,
                             pred_sec=pred, meas_sec=m["sec"], pct_of_roof=frac,
                             achieved_gflops=fl / m["sec"] / 1e9,
                             achieved_gbs=by / m["sec"] / 1e9))

        best_mem = max((r for r in rows if r["bound"] == "memory"),
                       key=lambda r: r["achieved_gbs"])
        worst = min(rows, key=lambda r: r["pct_of_roof"])
        mm = [r for r in rows if r["op"] == "matmul"][0]
        print(f"\nmatmul reaches {mm['pct_of_roof']:.0f}% of the compute roof "
              f"({mm['achieved_gflops']/1e3:.2f} of {peak_flops/1e12:.2f} TFLOP/s).")
        print(f"The best memory-bound op reaches {best_mem['achieved_gbs']:.0f} GB/s "
              f"= {100*best_mem['achieved_gbs']*1e9/peak_bw:.0f}% of peak bandwidth.")
        print(f"The worst offender is {worst['op']} at {worst['pct_of_roof']:.0f}% of "
              f"its own roof - the model says WHAT limits it, not that you will hit it.")

        findings = dict(device=dev["name"], peak_tflops=peak_flops / 1e12,
                        peak_gbs=peak_bw / 1e9, ridge=ridge,
                        ops=rows, ridge_table=ridge_rows,
                        ridge_growth_1070ti_to_h100=r_h100 / r0)
        with open(os.path.join(OUT, "findings.json"), "w") as f:
            json.dump(findings, f, indent=2)
        with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(OUT, "ridge_points.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(ridge_rows[0].keys()))
            w.writeheader()
            w.writerows(ridge_rows)
        plot(rows, peak_flops, peak_bw, ridge, dev, ridge_rows, ops)
    else:
        print("(no GPU measurements)")


def plot(rows, peak_flops, peak_bw, ridge, dev, ridge_rows, ops):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # ---- (a) the roofline itself ----
    x = np.logspace(-2, 4, 400)
    roof = np.minimum(peak_flops, peak_bw * x) / 1e9
    ax[0].loglog(x, roof, "k-", lw=2, label="roofline (the ceiling)")
    ax[0].axvline(ridge, ls="--", c="gray", lw=1)
    ax[0].text(ridge * 1.15, 20, f"ridge\n{ridge:.0f} FLOP/byte", fontsize=9, color="gray")
    cols = {"matmul": "#4C78A8", "layernorm": "#F58518", "softmax": "#E45756",
            "gelu": "#54A24B", "transpose": "#B279A2"}
    offs = {"matmul": (8, -14), "layernorm": (-4, 14), "softmax": (-10, -18),
            "gelu": (10, 4), "transpose": (0, 0)}
    for r in rows:
        if r["ai"] <= 0:
            continue
        ax[0].plot(r["ai"], r["achieved_gflops"], "o", ms=10, color=cols[r["op"]],
                   label=f"{r['op']} ({r['pct_of_roof']:.0f}% of roof)")
        ax[0].annotate(r["op"], (r["ai"], r["achieved_gflops"]), color=cols[r["op"]],
                       textcoords="offset points", xytext=offs[r["op"]], fontsize=9)
    tr = [r for r in rows if r["op"] == "transpose"]
    if tr:
        ax[0].text(1.2e-2, 3, "transpose has AI = 0 exactly\n(no arithmetic at all)\n"
                              f"-> off the left edge, {tr[0]['achieved_gbs']:.0f} GB/s",
                   fontsize=8, color=cols["transpose"])
    ax[0].set_xlabel("arithmetic intensity (FLOPs / byte)")
    ax[0].set_ylabel("achieved GFLOP/s")
    ax[0].set_title(f"(a) Measured roofline, {dev['name']}\n"
                    f"peak {peak_flops/1e12:.2f} TFLOP/s, {peak_bw/1e9:.0f} GB/s")
    ax[0].set_ylim(1, peak_flops / 1e9 * 3)
    ax[0].legend(fontsize=7.5, loc="lower right")
    ax[0].grid(alpha=.3, which="both")

    # ---- (b) how much of the relevant roof each op got ----
    names = [r["op"] for r in rows]
    fr = [r["pct_of_roof"] for r in rows]
    bar_c = [cols[n] for n in names]
    ax[1].bar(names, fr, color=bar_c)
    ax[1].axhline(100, ls="--", c="k", lw=1)
    for i, v in enumerate(fr):
        ax[1].text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=9)
    ax[1].set_ylabel("% of its own roofline ceiling")
    ax[1].set_title("(b) The model is a ceiling, not a promise\n"
                    "transpose is the one that misses badly")
    ax[1].set_ylim(0, 115)
    ax[1].grid(alpha=.3, axis="y")

    # ---- (c) the ridge point over time ----
    yr = [r["year"] for r in ridge_rows]
    rd = [r["ridge"] for r in ridge_rows]
    lbl = [f"{r['gpu']} ({r['dtype']})" for r in ridge_rows]
    ax[2].plot(yr, rd, "o-", color="#4C78A8")
    for i, (xx, yy, ll) in enumerate(zip(yr, rd, lbl)):
        dy = -14 if i == 0 else 5
        ax[2].annotate(ll, (xx, yy), textcoords="offset points", xytext=(4, dy), fontsize=7.5)
    for name, fl, by, _ in ops:
        if by and fl:
            ax[2].axhline(fl / by, ls=":", lw=1, c=cols[name])
            ax[2].text(2016.6, fl / by * 1.1, name, fontsize=7.5, color=cols[name])
    ax[2].set_yscale("log")
    ax[2].set_xlabel("year")
    ax[2].set_ylabel("ridge point (FLOP/byte)")
    ax[2].set_title("(c) Everything below a line is memory-bound.\n"
                    "The line keeps rising: the wall keeps growing.")
    ax[2].grid(alpha=.3, which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "roofline.png"), dpi=110)
    print(f"\nwrote {OUT}/roofline.png")


if __name__ == "__main__":
    sys.exit(main())
