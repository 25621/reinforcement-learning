"""Project 27 - Tenstorrent's architecture, priced without a Tenstorrent card.

  A. what is here    - an honest inventory of what can and cannot be tested
  B. on-chip memory  - the central bet, measured on the nearest analogue here
  C. what fits       - how big a model each accelerator can hold on the die
  D. the NoC         - placement, routing and the bottleneck link, simulated
  E. hotspots        - the traffic pattern a mesh is worst at

The programming model itself (reader / compute / writer kernels talking
through circular buffers) is shown in the README as source. It is not run:
there is no `tt-metal` runtime on this machine and pretending otherwise would
be the one thing this guide does not do.
"""

import csv
import json
import os
import sys
import time

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "18-triton-softmax")))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                       # noqa: E402
import onchip
from onchip import read_reduce                                   # noqa: E402
import noc                                                       # noqa: E402

R = {}

# on-die SRAM and off-die memory, 2026 public figures.
# sram_mb counts memory physically on the compute die that a kernel can use
# as working storage (L2 + shared/LDS on a GPU; core-local SRAM otherwise).
ACCEL = [
    ("GTX 1070 Ti (this card)", 2 + 19 * 0.096, 8, 256),
    ("NVIDIA A100 80GB", 40 + 108 * 0.192, 80, 2039),
    ("NVIDIA H100 SXM", 50 + 132 * 0.228, 80, 3350),
    ("AMD MI300X", 256 + 304 * 0.064, 192, 5300),
    ("Tenstorrent Wormhole n150", 72 * 1.5, 12, 288),
    ("Tenstorrent Wormhole n300", 2 * 72 * 1.5, 24, 576),
    ("Tenstorrent Blackhole p150", 140 * 1.5, 32, 512),
    ("Groq LPU (LPUv1)", 230, 0, 0),
    ("Cerebras WSE-3", 44000, 0, 0),
]


# --------------------------------------------------------------- A. what is here
def section_a():
    R["A_has_tenstorrent"] = False
    R["A_tt_metal_importable"] = False
    try:
        import ttnn                                              # noqa: F401
        R["A_tt_metal_importable"] = True
    except Exception as e:
        R["A_tt_metal_error"] = type(e).__name__
    R["A_what_is_simulated"] = ["Tensix grid", "network-on-chip", "placement"]
    R["A_what_is_measured"] = ["on-die vs off-die memory bandwidth"]
    R["A_what_is_arithmetic"] = ["on-die capacity vs model size"]
    print(f"A. tt-metal importable: {R['A_tt_metal_importable']}; "
          f"simulated: {R['A_what_is_simulated']}")


# ------------------------------------------------------------ B. on-chip memory
def section_b():
    out = gpu.empty(onchip.GRID)
    sweep = []
    for kb in [512, 1024, 1536, 2048, 2560, 3072, 4096, 6144, 8192,
               16384, 32768, 65536]:
        n = kb * 1024 // 4
        assert n % onchip.STRIDE_ELEMS == 0, kb
        x = gpu.ones(n)
        reps = onchip.reps_for(n)
        ms = gpu.bench(lambda: read_reduce(x, out, reps), reps=5)
        sweep.append(dict(kb=kb, reps=reps, ms=round(ms, 5),
                          gbs=round(reps * n * 4 / (ms * 1e6), 1)))
        del x
        torch.cuda.empty_cache()
    R["B_sweep"] = sweep
    by = {d["kb"]: d["gbs"] for d in sweep}
    R["B_onchip_gbs"] = by[1024]
    R["B_offchip_gbs"] = by[65536]
    peak = max(sweep, key=lambda d: d["gbs"])
    R["B_peak_gbs"] = peak["gbs"]
    R["B_peak_at_kb"] = peak["kb"]
    R["B_peak_over_offchip"] = round(peak["gbs"] / R["B_offchip_gbs"], 2)
    R["B_onchip_over_offchip"] = round(R["B_onchip_gbs"] / R["B_offchip_gbs"], 2)
    R["B_l2_mb"] = 2
    # where does it fall off? first size at which we drop below halfway
    mid = (R["B_onchip_gbs"] + R["B_offchip_gbs"]) / 2
    R["B_cliff_kb"] = next(d["kb"] for d in sweep if d["gbs"] < mid)
    print(f"B. on-die (1 MB working set) {R['B_onchip_gbs']} GB/s vs off-die "
          f"(64 MB) {R['B_offchip_gbs']} GB/s = {R['B_onchip_over_offchip']}x; "
          f"peak {R['B_peak_gbs']} at {R['B_peak_at_kb']} KB "
          f"({R['B_peak_over_offchip']}x), cliff at {R['B_cliff_kb']} KB "
          f"against a {R['B_l2_mb']} MB L2")


# ---------------------------------------------------------------- C. what fits
MODELS = [("Llama 8B", 8e9), ("Llama 70B", 70e9), ("GPT-3 175B", 175e9)]


def section_c():
    rows = []
    for name, sram_mb, dram_gb, bw in ACCEL:
        rows.append(dict(accel=name, sram_mb=round(sram_mb, 1),
                         dram_gb=dram_gb,
                         params_int8_on_die=round(sram_mb * 1e6 / 1e6, 1),
                         chips_for_70b_int8=int(-(-70e9 // (sram_mb * 1e6)))
                         if sram_mb else None))
    R["C_rows"] = rows
    by = {r["accel"]: r for r in rows}
    R["C_h100"] = by["NVIDIA H100 SXM"]
    R["C_n300"] = by["Tenstorrent Wormhole n300"]
    R["C_groq"] = by["Groq LPU (LPUv1)"]
    R["C_wse3"] = by["Cerebras WSE-3"]
    R["C_n300_over_h100_sram"] = round(
        by["Tenstorrent Wormhole n300"]["sram_mb"]
        / by["NVIDIA H100 SXM"]["sram_mb"], 1)
    R["C_wse3_70b_int8_fits"] = 44000 * 1e6 >= 70e9
    R["C_wse3_70b_int4_fits"] = 44000 * 1e6 >= 70e9 * 0.5
    R["C_wse3_largest_int8_b"] = round(44000 * 1e6 / 1e9, 1)
    print(f"C. on-die SRAM: H100 {by['NVIDIA H100 SXM']['sram_mb']} MB, "
          f"n300 {by['Tenstorrent Wormhole n300']['sram_mb']} MB "
          f"({R['C_n300_over_h100_sram']}x), WSE-3 44,000 MB. "
          f"70B int8 on-die needs {by['Groq LPU (LPUv1)']['chips_for_70b_int8']} "
          f"Groq chips")


# -------------------------------------------------------------------- D. NoC
def section_d():
    m = noc.Mesh(rows=10, cols=8, link_bytes_per_cycle=32)
    n_stages = 32
    act = 32 * 1024                        # 32 KB of activations per hand-off

    rows = []
    for name, fn in [("snake (adjacent stages adjacent)", noc.snake),
                     ("row-major", noc.rowmajor),
                     ("column-major", noc.columnmajor),
                     ("scattered (no placement pass)", noc.scattered)]:
        pl = fn(m, n_stages)
        c = m.cost(noc.pipeline_flows(pl, act))
        rows.append(dict(placement=name, **c,
                         mean_hops=round(c["total_hops"] / (n_stages - 1), 2)))
    R["D_placements"] = rows
    best = min(rows, key=lambda d: d["cycles"])
    worst = max(rows, key=lambda d: d["cycles"])
    R["D_best"] = best["placement"]
    R["D_worst"] = worst["placement"]
    R["D_placement_spread"] = round(worst["cycles"] / best["cycles"], 2)

    # validation: the ideal placement must hit the analytic lower bound.
    # Every hand-off crosses at least one link; with all stages on distinct
    # adjacent cores no link is shared, so the bottleneck is exactly one
    # hand-off's worth of bytes.
    R["D_lower_bound_bytes"] = act
    R["D_snake_bottleneck_bytes"] = rows[0]["bottleneck_bytes"]
    R["D_snake_is_optimal"] = rows[0]["bottleneck_bytes"] == act
    R["D_snake_mean_hops"] = rows[0]["mean_hops"]

    # how the gap grows with the number of stages
    growth = []
    for n in [8, 16, 32, 64, 80]:
        a = m.cost(noc.pipeline_flows(noc.snake(m, n), act))
        b = m.cost(noc.pipeline_flows(noc.scattered(m, n), act))
        growth.append(dict(stages=n, snake_cycles=a["cycles"],
                           scattered_cycles=b["cycles"],
                           ratio=round(b["cycles"] / a["cycles"], 2)))
    R["D_growth"] = growth
    print(f"D. placement spans {R['D_placement_spread']}x "
          f"({R['D_best']} vs {R['D_worst']}); snake hits the analytic "
          f"lower bound: {R['D_snake_is_optimal']}")


# --------------------------------------------------------------- E. hotspots
def section_e():
    m = noc.Mesh(rows=10, cols=8, link_bytes_per_cycle=32)
    cores = [(r, c) for r in range(10) for c in range(8)]
    payload = 4 * 1024

    rows = []
    for name, dst in [("gather to a corner (0,0)", (0, 0)),
                      ("gather to the centre (5,4)", (5, 4))]:
        c = m.cost(noc.gather_flows(cores, dst, payload))
        rows.append(dict(pattern=name, **c))
    # the pipeline moves the same total bytes for comparison
    n = len(cores)
    pl = noc.snake(m, n)
    c = m.cost(noc.pipeline_flows(pl, payload))
    rows.append(dict(pattern="neighbour hand-off (same bytes per core)", **c))
    R["E_patterns"] = rows

    g0, gc, nb = rows[0], rows[1], rows[2]
    R["E_corner_over_centre"] = round(g0["cycles"] / gc["cycles"], 2)
    R["E_gather_over_pipeline"] = round(gc["cycles"] / nb["cycles"], 1)
    R["E_gather_bytes_vs_pipeline"] = round(
        gc["total_bytes"] / nb["total_bytes"], 2)
    print(f"E. gathering to the centre is {R['E_corner_over_centre']}x cheaper "
          f"than to a corner, and still {R['E_gather_over_pipeline']}x the "
          f"cost of neighbour hand-offs moving "
          f"{R['E_gather_bytes_vs_pipeline']}x the bytes")


# -------------------------------------------------------------------- plot
def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        print("(matplotlib missing - skipping the plot)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    s = R["B_sweep"]
    ax[0].semilogx([d["kb"] / 1024 for d in s], [d["gbs"] for d in s], "o-",
                   color="#1f77b4")
    ax[0].axvline(R["B_l2_mb"], color="#d62728", lw=1.2)
    ax[0].text(R["B_l2_mb"] * 1.1, R["B_onchip_gbs"] * 0.5,
               f"{R['B_l2_mb']} MB L2", fontsize=8, color="#d62728")
    ax[0].set_xlabel("working set (MB)")
    ax[0].set_ylabel("GB/s")
    ax[0].set_title(f"on-die vs off-die memory\n{R['B_onchip_over_offchip']}x "
                    "for staying on the chip")
    ax[0].grid(alpha=.3, which="both")

    rows = sorted(R["C_rows"], key=lambda d: d["sram_mb"])
    names = [d["accel"].replace("Tenstorrent ", "").replace("NVIDIA ", "")
             .replace(" (this card)", "").replace(" (LPUv1)", "")
             for d in rows]
    cols = ["#d62728" if "Wormhole" in d["accel"] or "Blackhole" in d["accel"]
            else "#ff7f0e" if "Groq" in d["accel"] or "Cerebras" in d["accel"]
            else "#1f77b4" for d in rows]
    ax[1].barh(names, [d["sram_mb"] for d in rows], color=cols)
    ax[1].set_xscale("log")
    for m_, lbl in [(8e3, "8B int8"), (70e3, "70B int8")]:
        ax[1].axvline(m_, color="0.4", ls=":", lw=1)
        ax[1].text(m_, -0.6, lbl, fontsize=7, ha="center", color="0.3")
    ax[1].set_xlabel("on-die SRAM (MB, log)")
    ax[1].set_title("how much model fits on the die\n(red = Tenstorrent)")
    ax[1].tick_params(labelsize=8)
    ax[1].grid(alpha=.3, axis="x", which="both")

    g = R["D_growth"]
    ax[2].plot([d["stages"] for d in g], [d["snake_cycles"] for d in g], "o-",
               color="#2ca02c", label="snake placement")
    ax[2].plot([d["stages"] for d in g], [d["scattered_cycles"] for d in g],
               "s-", color="#d62728", label="scattered placement")
    ax[2].set_xlabel("pipeline stages mapped onto an 8x10 mesh")
    ax[2].set_ylabel("cycles on the bottleneck link")
    ax[2].set_title("same work, same bytes,\nplacement is the whole difference")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    t0 = time.time()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    R["runtime_s"] = round(time.time() - t0, 1)
    R["device"] = gpu.device_note()

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in R.items():
            if not isinstance(v, (list, dict)):
                w.writerow([k, v])
        w.writerow([])
        w.writerow(["working_set_kb", "ms", "GB/s"])
        for d in R["B_sweep"]:
            w.writerow([d["kb"], d["ms"], d["gbs"]])
        w.writerow([])
        w.writerow(["accel", "sram_mb", "dram_gb", "chips_for_70b_int8"])
        for d in R["C_rows"]:
            w.writerow([d["accel"], d["sram_mb"], d["dram_gb"],
                        d["chips_for_70b_int8"]])
        w.writerow([])
        w.writerow(["placement", "bottleneck_bytes", "cycles", "links_used",
                    "mean_hops"])
        for d in R["D_placements"]:
            w.writerow([d["placement"], d["bottleneck_bytes"], d["cycles"],
                        d["links_used"], d["mean_hops"]])
    plot(os.path.join(OUT, "tenstorrent.png"))
    print(f"total {R['runtime_s']} s")


if __name__ == "__main__":
    main()
