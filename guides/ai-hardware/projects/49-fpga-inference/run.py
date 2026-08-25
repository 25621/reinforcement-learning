"""Project 49 - a CNN layer as actual hardware, and what it is worth.

Sections
  A. does it compute the right thing? - cycle-accurate simulation of the 3x3
                                        convolution unit against a Python
                                        reference, bit for bit
  B. how big is it?                   - yosys: multipliers, flip-flops, gates,
                                        and the same design at 4 image widths
  C. how many fit on a real board?    - DSP / flip-flop budgets of two popular
                                        dev boards (arithmetic)
  D. FPGA vs this GPU                 - throughput and performance per watt,
                                        against measured GPU numbers
  E. latency                          - where the FPGA actually wins
  F. the roofline of an FPGA          - why more multipliers stop helping

Simulation and synthesis are real and run here. Everything about a board we do
not own (LUT budgets, clock frequency, board power) is arithmetic and labelled.
"""

import json
import os
import subprocess
import sys
import time

from amaranth.sim import Simulator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import hdl_lib  # noqa: E402
from conv_accel import Conv3x3, golden  # noqa: E402

CLK_HZ = 100e6          # a modest, very achievable FPGA clock

# `pack` is how many int8 multiply-accumulates one DSP slice can do per clock.
# Xilinx' UltraScale+ DSP48E2 is wide enough to hold two int8 products at once
# (the "INT8 packing" trick); the older Zynq-7020 and Cyclone V get one.
BOARDS = [
    # name,                 LUTs,   FFs,    DSPs, BRAM Kb, price, MHz, W, pack
    ("Pynq-Z2 (Zynq-7020)", 53200, 106400,  220,  4900,   199, 100, 5, 1),
    ("DE10-Nano (Cyclone V)", 41910, 41910, 112,  5570,   180, 100, 5, 1),
    ("Alveo U55C (UltraScale+)", 1300000, 2600000, 9024, 270000, 5000, 300, 150, 2),
]
BOARD_KEYS = ["name", "luts", "ffs", "dsps", "bram_kb", "price", "mhz", "watts",
              "pack"]


# ------------------------------------------------------------------ A
def simulate(width=8, height=8, seed=7):
    """Feed a whole image through the design one pixel per clock and compare
    every output with the Python reference."""
    import random
    rng = random.Random(seed)
    image = [[rng.randint(-100, 100) for _ in range(width)]
             for _ in range(height)]
    weights = [rng.randint(-8, 8) for _ in range(9)]
    expect = golden(image, weights)
    flat_expect = [v for row in expect for v in row]

    dut = Conv3x3(width=width)
    got = []
    stats = {}

    async def tb(ctx):
        cycles = 0
        # load the 9 weights (weight-stationary: this happens once)
        for i, w in enumerate(weights):
            ctx.set(dut.w_addr, i); ctx.set(dut.w_data, w); ctx.set(dut.w_en, 1)
            await ctx.tick(); cycles += 1
        ctx.set(dut.w_en, 0)
        stats["weight_load_cycles"] = cycles

        first_out = None
        stream_start = cycles
        for r in range(height):
            for c in range(width):
                ctx.set(dut.pix_in, image[r][c]); ctx.set(dut.in_valid, 1)
                await ctx.tick(); cycles += 1
                if ctx.get(dut.out_valid):
                    got.append(ctx.get(dut.pix_out))
                    if first_out is None:
                        first_out = cycles - stream_start
        ctx.set(dut.in_valid, 0)
        for _ in range(4):                      # drain the pipeline
            await ctx.tick(); cycles += 1
            if ctx.get(dut.out_valid):
                got.append(ctx.get(dut.pix_out))
        stats["cycles"] = cycles
        stats["fill_latency_cycles"] = first_out

    sim = Simulator(dut)
    sim.add_clock(1 / CLK_HZ)
    sim.add_testbench(tb)
    sim.run()

    ok = got == flat_expect
    return dict(width=width, height=height, outputs=len(got),
                expected_outputs=len(flat_expect), exact_match=ok,
                first_five_got=got[:5], first_five_expected=flat_expect[:5],
                pixels=width * height, macs=9 * len(flat_expect), **stats)


# ------------------------------------------------------------------ B
def synth_widths(widths=(8, 16, 32, 64)):
    rows = []
    for w in widths:
        dut = Conv3x3(width=w)
        il = os.path.join(OUT, f"conv{w}.il")
        hdl_lib.to_rtlil(dut, f"conv{w}",
                         [dut.pix_in, dut.in_valid, dut.w_addr, dut.w_data,
                          dut.w_en, dut.pix_out, dut.out_valid], il)
        t0 = time.perf_counter()
        s = hdl_lib.synth(il, f"conv{w}")
        dt = time.perf_counter() - t0
        g = hdl_lib.gate_summary(s["gate_cells"])
        c = hdl_lib.coarse_summary(s["coarse_cells"])
        rows.append(dict(width=w, synth_s=dt, **g, **c,
                         gate_cells=s["gate_cells"],
                         coarse_cells=s["coarse_cells"]))
    return rows


# ------------------------------------------------------------------ C/D
def board_fit(unit):
    rows = []
    for b in BOARDS:
        d = dict(zip(BOARD_KEYS, b))
        by_dsp = d["dsps"] * d["pack"] // max(unit["multipliers"], 1)
        by_ff = d["ffs"] // max(unit["flops"], 1)
        n = max(min(by_dsp, by_ff), 0)
        macs = n * 9
        gops = macs * 2 * d["mhz"] * 1e6 / 1e9
        rows.append(dict(board=d["name"], price=d["price"], mhz=d["mhz"],
                         watts=d["watts"], pack=d["pack"], dsps=d["dsps"],
                         units_by_dsp=by_dsp, units_by_ff=by_ff, units=n,
                         macs=macs, gops=gops,
                         limited_by="DSP slices" if by_dsp < by_ff
                         else "flip-flops"))
    return rows


def gpu_numbers():
    """Measure this GPU's int8 (dp4a) and fp32 throughput."""
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..",
                                                    "45-2-gpu-build-plan")))
    import riglib
    exe = os.path.join(OUT, "dp4a")
    riglib.build_cu(os.path.join(HERE, "dp4a.cu"), exe)
    p = subprocess.run([exe], capture_output=True, text=True)
    out = {}
    for line in p.stdout.strip().splitlines():
        k, v = line.split(",")
        out[k] = float(v)
    return out


def main():
    findings = {"clock_hz": CLK_HZ}

    print("== A. simulate the hardware, compare against Python ==")
    sims = [simulate(width=w, height=w) for w in (8, 16)]
    findings["simulation"] = sims
    for s in sims:
        print(f"   {s['width']}x{s['height']} image: {s['outputs']} outputs, "
              f"exact match = {s['exact_match']}, {s['cycles']} cycles "
              f"(fill latency {s['fill_latency_cycles']}), "
              f"{s['macs']} MACs")
    for s in sims:
        print(f"   -> {s['width']}x{s['height']}: "
              f"{s['macs']/s['cycles']:.1f} MACs per cycle sustained "
              f"(9 is the peak; the gap is the pipeline fill and the "
              f"weight load)")

    print("== B. synthesize it ==")
    syn = synth_widths()
    findings["synthesis"] = syn
    for r in syn:
        print(f"   width {r['width']:>3}: {r['multipliers']} multipliers, "
              f"{r['flops']:>5} flip-flops, {r['gates']:>5} gates, "
              f"{r['area_um2']/1000:6.1f} k um^2 in sky130 "
              f"({r['synth_s']:.1f} s to synthesize)")
    unit = syn[0]

    print("== C. how many fit on a real board? (arithmetic) ==")
    fit = board_fit(unit)
    findings["boards"] = fit
    for r in fit:
        print(f"   {r['board']:>28}: {r['units']:>5} units "
              f"({r['macs']:>6} MACs) at {r['mhz']} MHz = {r['gops']:8.1f} "
              f"GOP/s, limited by {r['limited_by']}")

    print("== D. against the GPU in this machine (measured) ==")
    gpu = gpu_numbers()
    findings["gpu"] = gpu
    print(f"   measured GPU: {gpu['fp32']:.0f} GFLOP/s fp32, "
          f"{gpu['dp4a']:.0f} GOP/s int8 (dp4a)")
    best_fpga = max(fit, key=lambda r: r["gops"])
    small = fit[0]
    cmp_rows = []
    for r in fit:
        cmp_rows.append(dict(
            name=r["board"], gops=r["gops"], watts=r["watts"],
            gops_per_watt=r["gops"] / r["watts"],
            gops_per_dollar=r["gops"] / r["price"]))
    cmp_rows.append(dict(name="GTX 1070 Ti (measured)", gops=gpu["dp4a"],
                         watts=114.6, gops_per_watt=gpu["dp4a"] / 114.6,
                         gops_per_dollar=gpu["dp4a"] / 150))
    findings["comparison"] = cmp_rows
    for r in cmp_rows:
        print(f"   {r['name']:>28}: {r['gops']:9.1f} GOP/s  "
              f"{r['gops_per_watt']:8.2f} GOP/s per watt  "
              f"{r['gops_per_dollar']:7.2f} GOP/s per dollar")
    print(f"   -> the GPU is {gpu['dp4a']/small['gops']:.0f}x the throughput of "
          f"a full {small['board']} and "
          f"{(gpu['dp4a']/114.6)/(small['gops']/small['watts']):.1f}x its "
          f"performance per watt")
    print(f"   -> but the $5,000 datacenter FPGA reaches "
          f"{best_fpga['gops']/gpu['dp4a']:.2f}x the GPU's throughput at "
          f"{(best_fpga['gops']/best_fpga['watts'])/(gpu['dp4a']/114.6):.2f}x "
          f"its performance per watt")

    print("== E. latency: one small layer, start to finish ==")
    lat = latency_compare(sims[0], gpu)
    findings["latency"] = lat
    for k, v in lat.items():
        if k.endswith("_us"):
            print(f"   {k:>28}: {v:8.3f} us")
    print(f"   -> the FPGA finishes the whole layer in "
          f"{lat['fpga_total_us']:.2f} us; the GPU has not started work yet "
          f"({lat['gpu_launch_only_us']:.2f} us of launch overhead alone)")

    print("== F. the FPGA's own roofline ==")
    findings["roofline"] = fpga_roofline(fit[0])
    for r in findings["roofline"]:
        print(f"   {r['case']:>34}: needs {r['bytes_per_s']/1e9:7.2f} GB/s of "
              f"on-chip reads to feed {r['macs']} MACs -> "
              f"{r['verdict']}")

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=1, default=float)
    write_csv(findings)
    plot(findings)
    print("\nwrote outputs/findings.json, findings.csv, fpga.png")


def latency_compare(sim, gpu):
    """One 8x8 convolution: the FPGA's time is exactly its cycle count; the
    GPU's is a kernel launch plus a kernel that is far too small to matter."""
    fpga_cycles = sim["cycles"]
    fpga_us = fpga_cycles / CLK_HZ * 1e6
    gpu_launch_us = 1.17          # measured in project 46 on this machine
    gpu_work_us = sim["macs"] * 2 / (gpu["dp4a"] * 1e9) * 1e6
    return dict(fpga_cycles=fpga_cycles, fpga_total_us=fpga_us,
                fpga_fill_us=sim["fill_latency_cycles"] / CLK_HZ * 1e6,
                gpu_launch_only_us=gpu_launch_us,
                gpu_arithmetic_us=gpu_work_us,
                gpu_total_us=gpu_launch_us + gpu_work_us,
                fpga_advantage=(gpu_launch_us + gpu_work_us) / fpga_us)


def fpga_roofline(fit_row, bram_kb=4900, mhz=100):
    """A multiplier that is not fed is a multiplier that is not working.

    Each MAC needs 1 activation byte + 1 weight byte per cycle in the worst
    case. Block RAM can supply roughly `bram_ports x 4 bytes x clock`; the
    numbers below show when the design stops being multiplier-limited and
    starts being memory-limited - the same roofline argument as project 02,
    applied inside a chip.
    """
    rows = []
    for case, macs, reuse in [("1 unit, weights stationary", 9, 9),
                              ("all DSP units, weights stationary",
                               fit_row["macs"], 9),
                              ("all DSP units, no weight reuse",
                               fit_row["macs"], 1),
                              ("LUTs also turned into MACs (5,000)", 5000, 9),
                              ("LUTs too, no weight reuse", 5000, 1)]:
        # bytes/s = MACs x (1 activation + 1 weight / reuse) x clock
        bps = macs * (1 + 1 / reuse) * mhz * 1e6
        # Zynq-7020 BRAM: 140 x 36 Kb blocks, 2 ports x 4 bytes each
        bram_bps = 140 * 2 * 4 * mhz * 1e6
        rows.append(dict(case=case, macs=macs, reuse=reuse, bytes_per_s=bps,
                         bram_bytes_per_s=bram_bps,
                         verdict="fits in BRAM bandwidth" if bps <= bram_bps
                         else f"needs {bps/bram_bps:.1f}x the BRAM bandwidth"))
    return rows


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "unit", "kind"])
        for s in f["simulation"]:
            w.writerow(["A sim", f"{s['width']}x{s['height']} exact match",
                        s["exact_match"], "", "simulated"])
            w.writerow(["A sim", f"{s['width']}x{s['height']} cycles",
                        s["cycles"], "cycles", "simulated"])
        for r in f["synthesis"]:
            w.writerow(["B synth", f"width {r['width']} multipliers",
                        r["multipliers"], "", "synthesized"])
            w.writerow(["B synth", f"width {r['width']} flops", r["flops"],
                        "", "synthesized"])
            w.writerow(["B synth", f"width {r['width']} area",
                        round(r["area_um2"], 1), "um^2 (sky130)", "arithmetic"])
        for r in f["boards"]:
            w.writerow(["C boards", r["board"], round(r["gops"], 2), "GOP/s",
                        "arithmetic"])
        for k, v in f["gpu"].items():
            w.writerow(["D gpu", k, v, "GOP/s or GFLOP/s", "measured"])
        for r in f["comparison"]:
            w.writerow(["D compare", r["name"], round(r["gops_per_watt"], 3),
                        "GOP/s per W", "mixed"])
        for k, v in f["latency"].items():
            w.writerow(["E latency", k, round(v, 4), "", "mixed"])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))

    a = ax[0]
    syn = f["synthesis"]
    xs = [r["width"] for r in syn]
    a.plot(xs, [r["flops"] for r in syn], "o-", color="#c0392b",
           label="flip-flops")
    a.plot(xs, [r["gates"] for r in syn], "s-", color="#2980b9", label="gates")
    a.plot(xs, [r["multipliers"] for r in syn], "^-", color="#27ae60",
           label="multipliers (DSPs)")
    a.set_xscale("log", base=2); a.set_yscale("log")
    a.set_xlabel("image width the unit is built for (pixels)")
    a.set_ylabel("count (log)")
    a.set_title("B. the line buffer grows; the maths does not", fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=.3)

    a = ax[1]
    rows = f["comparison"]
    names = [r["name"].split(" (")[0] for r in rows]
    a.barh(names, [r["gops"] for r in rows], color="#8e44ad")
    for i, r in enumerate(rows):
        a.text(r["gops"], i, f"  {r['gops']:.0f} GOP/s", va="center", fontsize=7)
    a.set_xscale("log"); a.set_xlabel("int8 GOP/s (log)")
    a.set_title("D. throughput: the GPU is not close", fontsize=10)
    a.grid(alpha=.3, axis="x")

    a = ax[2]
    lat = f["latency"]
    names = ["FPGA\n(whole layer)", "GPU\n(launch only)", "GPU\n(launch + work)"]
    vals = [lat["fpga_total_us"], lat["gpu_launch_only_us"], lat["gpu_total_us"]]
    a.bar(names, vals, color=["#27ae60", "#c0392b", "#e67e22"])
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:.2f} us", ha="center", va="bottom", fontsize=9)
    a.set_ylabel("microseconds for one 8x8 layer")
    a.set_title("E. latency: where the FPGA wins", fontsize=10)
    a.grid(alpha=.3, axis="y")

    fig.suptitle("Project 49 - a convolution in hardware, simulated and sized",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fpga.png"), dpi=110)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
