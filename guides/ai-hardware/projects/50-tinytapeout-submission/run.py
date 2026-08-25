"""Project 50 - a real (tiny) chip design, sized against a real (huge) GPU.

Sections
  A. verify both designs      - cycle-accurate simulation against Python
  B. synthesize both          - gates, flip-flops, SkyWater 130 nm area
  C. does it fit in a tile?   - TinyTapeout's 1x1 tile budget
  D. what does a tile compute?- MACs per second per tile
  E. the pin roofline         - 8 input wires is the real constraint
  F. energy                   - core logic vs the I/O pads that dwarf it
  G. submission artifacts     - Verilog + info.yaml written to outputs/

Simulation, synthesis and the Verilog output are real. Area, energy and
throughput on silicon are arithmetic from published SkyWater numbers, and are
labelled that way.
"""

import json
import os
import re
import sys

from amaranth.back import verilog
from amaranth.sim import Simulator

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "49-fpga-inference")))

import hdl_lib  # noqa: E402
from tt_designs import MAC8, BinaryNeuron, binary_dot  # noqa: E402

# TinyTapeout, as of the sky130 shuttles: one tile is 161.0 um x 111.5 um of
# placeable area, the user clock is typically 50 MHz, and the interface is
# 8 dedicated inputs, 8 dedicated outputs and 8 bidirectional pins.
TILE_UM2 = 161.0 * 111.52
TILE_UTILIZATION = 0.55        # what a real place-and-route achieves
CLK_HZ = 50e6
PINS_IN = 8

# sky130 core supply and a plausible average switched capacitance per cell.
VDD = 1.8
C_CELL_F = 4e-15
ACTIVITY = 0.15                # fraction of cells toggling per clock
PAD_ENERGY_J = 20e-12          # per output-pin transition, driving a board


# ------------------------------------------------------------------ A
def sim_mac8(n=12, seed=3):
    import random
    rng = random.Random(seed)
    pairs = [(rng.randint(-128, 127), rng.randint(-128, 127)) for _ in range(n)]
    expect = sum(a * b for a, b in pairs)
    dut = MAC8()
    got = {}

    async def tb(ctx):
        ctx.set(dut.uio_in, 0b100)             # clear
        await ctx.tick()
        ctx.set(dut.uio_in, 0)
        cycles = 0
        for a, b in pairs:
            ctx.set(dut.ui_in, a & 0xFF); ctx.set(dut.uio_in, 0b001)   # load A
            await ctx.tick(); cycles += 1
            ctx.set(dut.ui_in, b & 0xFF); ctx.set(dut.uio_in, 0b010)   # MAC
            await ctx.tick(); cycles += 1
        ctx.set(dut.uio_in, 0)
        await ctx.tick()
        got["acc"] = ctx.get(dut.acc)
        got["cycles"] = cycles

    sim = Simulator(dut); sim.add_clock(1 / CLK_HZ)
    sim.add_testbench(tb); sim.run()
    return dict(design="MAC8", macs=n, cycles=got["cycles"],
                cycles_per_mac=got["cycles"] / n, got=got["acc"],
                expected=expect, exact_match=got["acc"] == expect)


def sim_binary(n=16, seed=5):
    import random
    rng = random.Random(seed)
    w = rng.randint(0, 255)
    xs = [rng.randint(0, 255) for _ in range(n)]
    expect = [binary_dot(x, w) for x in xs]
    dut = BinaryNeuron()
    got = []

    async def tb(ctx):
        ctx.set(dut.ui_in, w); ctx.set(dut.uio_in, 0b01)   # load weights
        await ctx.tick()
        ctx.set(dut.uio_in, 0)
        cycles = 0
        for x in xs:
            ctx.set(dut.ui_in, x)
            await ctx.tick(); cycles += 1
            got.append(ctx.get(dut.dot))
        got.append(cycles)

    sim = Simulator(dut); sim.add_clock(1 / CLK_HZ)
    sim.add_testbench(tb); sim.run()
    cycles = got.pop()
    # the output is combinational, so sample i is the answer for input i
    return dict(design="BinaryNeuron", macs=8 * n, cycles=cycles,
                cycles_per_mac=cycles / (8 * n), got=got[:4],
                expected=expect[:4], exact_match=got == expect)


# ------------------------------------------------------------------ B/C
def synth_design(cls, name):
    dut = cls()
    il = os.path.join(OUT, f"{name}.il")
    hdl_lib.to_rtlil(dut, name, [dut.ui_in, dut.uio_in, dut.uo_out], il)
    s = hdl_lib.synth(il, name)
    g = hdl_lib.gate_summary(s["gate_cells"])
    c = hdl_lib.coarse_summary(s["coarse_cells"])
    placed = g["area_um2"] / TILE_UTILIZATION
    return dict(name=name, **g, **c, gate_cells=s["gate_cells"],
                placed_um2=placed, tile_um2=TILE_UM2,
                tile_fraction=placed / TILE_UM2,
                copies_per_tile=int(TILE_UM2 / placed) if placed else 0)


# ------------------------------------------------------------------ D/E/F
def throughput(syn, cycles_per_mac, bits_per_operand, operands_per_mac):
    """operands_per_mac is 2 when both numbers arrive through the pins every
    time, and 1 when one of them (the weight) is already stored on chip -
    which is exactly what "weight-stationary" buys you at the pin boundary."""
    copies = max(syn["copies_per_tile"], 1)
    macs_per_s = copies * CLK_HZ / cycles_per_mac
    ops_per_s = 2 * macs_per_s
    # pins: 8 wires x 1 byte per clock is all the data that can ever arrive
    pin_bytes_per_s = PINS_IN / 8 * CLK_HZ
    operands_per_s = pin_bytes_per_s * 8 / bits_per_operand
    feedable = operands_per_s / operands_per_mac
    return dict(copies_per_tile=copies, macs_per_s=macs_per_s,
                ops_per_s=ops_per_s, gops=ops_per_s / 1e9,
                pin_bytes_per_s=pin_bytes_per_s,
                operands_per_s=operands_per_s,
                operands_per_mac=operands_per_mac,
                macs_the_pins_can_feed=feedable,
                pin_limited=feedable < macs_per_s,
                pin_shortfall=macs_per_s / feedable)


def energy(syn, ops_per_s, out_bits=8):
    cells = syn["cells"] * max(syn["copies_per_tile"], 1)
    dyn_w = cells * ACTIVITY * CLK_HZ * 0.5 * C_CELL_F * VDD ** 2
    pad_w = out_bits * 0.5 * CLK_HZ * PAD_ENERGY_J
    return dict(cells=cells, core_w=dyn_w, pad_w=pad_w, total_w=dyn_w + pad_w,
                gops_per_w_core=ops_per_s / 1e9 / dyn_w if dyn_w else 0,
                gops_per_w_total=ops_per_s / 1e9 / (dyn_w + pad_w),
                pad_over_core=pad_w / dyn_w if dyn_w else 0)


def gpu_reference():
    """The measured int8 number from project 49, so the comparison is against
    a machine that exists."""
    path = os.path.abspath(os.path.join(HERE, "..", "49-fpga-inference",
                                        "outputs", "findings.json"))
    with open(path) as f:
        d = json.load(f)
    return dict(gops=d["gpu"]["dp4a"], watts=114.6,
                gops_per_w=d["gpu"]["dp4a"] / 114.6)


# ------------------------------------------------------------------ G
INFO_YAML = """# TinyTapeout project description (outputs/info.yaml)
# Written by run.py. A real submission also needs the GDS produced by the
# TinyTapeout GitHub Action, which runs OpenLane on this Verilog.
project:
  title: "{title}"
  author: "AI hardware guide, project 50"
  description: "{desc}"
  language: "Amaranth HDL (exported to Verilog)"
  clock_hz: {clk}
  tiles: "1x1"
  top_module: "tt_um_{name}"
pinout:
  ui:
{ui}
  uo:
{uo}
  uio:
{uio}
"""


def write_artifacts(designs):
    paths = {}
    for cls, name, title, desc, ui, uo, uio in designs:
        dut = cls()
        v = verilog.convert(dut, name=f"tt_um_{name}",
                            ports=[dut.ui_in, dut.uio_in, dut.uo_out])
        # yosys stamps every wire with the absolute path of the Python line it
        # came from; shorten those to file names so the committed Verilog does
        # not depend on where this checkout lives.
        v = re.sub(r'src = "[^"]*/([^/"]+)"', r'src = "\1"', v)
        vpath = os.path.join(OUT, f"tt_um_{name}.v")
        with open(vpath, "w") as f:
            f.write(v)
        ypath = os.path.join(OUT, f"info_{name}.yaml")
        with open(ypath, "w") as f:
            f.write(INFO_YAML.format(
                title=title, desc=desc, clk=int(CLK_HZ), name=name,
                ui="\n".join(f"    - {x}" for x in ui),
                uo="\n".join(f"    - {x}" for x in uo),
                uio="\n".join(f"    - {x}" for x in uio)))
        paths[name] = dict(verilog=os.path.basename(vpath),
                           verilog_bytes=os.path.getsize(vpath),
                           info=os.path.basename(ypath))
    return paths


def main():
    findings = {"tile_um2": TILE_UM2, "clock_hz": CLK_HZ,
                "utilization": TILE_UTILIZATION}

    print("== A. simulate both designs ==")
    sims = [sim_mac8(), sim_binary()]
    findings["simulation"] = sims
    for s in sims:
        print(f"   {s['design']:>13}: exact match = {s['exact_match']}, "
              f"{s['cycles']} cycles for {s['macs']} MACs "
              f"({s['cycles_per_mac']:.3f} cycles per MAC)")

    print("== B/C. synthesize, and see if a tile can hold it ==")
    syn = [synth_design(MAC8, "mac8"), synth_design(BinaryNeuron, "binneuron")]
    findings["synthesis"] = syn
    for r in syn:
        print(f"   {r['name']:>10}: {r['gates']:>5} gates + {r['flops']:>3} "
              f"flip-flops = {r['cells']:>5} cells, {r['area_um2']:8.0f} um^2 "
              f"of cells -> {r['placed_um2']:8.0f} um^2 placed at "
              f"{100*TILE_UTILIZATION:.0f}% density")
        print(f"   {'':>10}  = {100*r['tile_fraction']:5.1f}% of a "
              f"{TILE_UM2:.0f} um^2 tile -> {r['copies_per_tile']} copies fit")

    print("== D/E. what one tile computes, and what the pins allow ==")
    thr = [throughput(syn[0], sims[0]["cycles_per_mac"], 8, operands_per_mac=2),
           throughput(syn[1], sims[1]["cycles_per_mac"], 1, operands_per_mac=1)]
    findings["throughput"] = thr
    for r, s in zip(thr, syn):
        print(f"   {s['name']:>10}: {r['copies_per_tile']} copies x "
              f"{CLK_HZ/1e6:.0f} MHz = {r['macs_per_s']/1e6:8.1f} MMAC/s "
              f"= {r['gops']:.3f} GOP/s")
        print(f"   {'':>10}  pins deliver {r['pin_bytes_per_s']/1e6:.0f} MB/s "
              f"= {r['macs_the_pins_can_feed']/1e6:.1f} MMAC/s of operands -> "
              + (f"PIN-LIMITED by {r['pin_shortfall']:.0f}x"
                 if r["pin_limited"] else "compute-limited"))

    print("== F. energy (arithmetic) ==")
    en = [energy(syn[0], thr[0]["ops_per_s"]), energy(syn[1], thr[1]["ops_per_s"])]
    findings["energy"] = en
    gpu = gpu_reference()
    findings["gpu"] = gpu
    for r, s in zip(en, syn):
        print(f"   {s['name']:>10}: core {r['core_w']*1e6:7.2f} uW, "
              f"pads {r['pad_w']*1e3:6.2f} mW "
              f"({r['pad_over_core']:.0f}x the core)")
        print(f"   {'':>10}  {r['gops_per_w_core']:9.1f} GOP/s per watt "
              f"counting logic only, {r['gops_per_w_total']:7.3f} counting the "
              f"pads")
    print(f"   measured GPU for comparison: {gpu['gops']:.0f} GOP/s at "
          f"{gpu['watts']} W = {gpu['gops_per_w']:.1f} GOP/s per watt")
    findings["ratios"] = dict(
        gpu_over_tile_throughput=gpu["gops"] / thr[0]["gops"],
        tile_over_gpu_efficiency_core=en[0]["gops_per_w_core"] / gpu["gops_per_w"],
        tile_over_gpu_efficiency_total=en[0]["gops_per_w_total"] / gpu["gops_per_w"],
        binary_over_mac_throughput=thr[1]["gops"] / thr[0]["gops"])
    r = findings["ratios"]
    print(f"   -> the GPU does {r['gpu_over_tile_throughput']:,.0f}x the work "
          f"of one tile; the tile is "
          f"{r['tile_over_gpu_efficiency_core']:.1f}x more efficient per watt "
          f"if you ignore its pads and "
          f"{r['tile_over_gpu_efficiency_total']:.3f}x if you do not")
    print(f"   -> spending the same tile on 1-bit arithmetic instead of 8-bit "
          f"multiplies gives {r['binary_over_mac_throughput']:.1f}x the "
          f"operations")

    print("== G. submission artifacts ==")
    findings["artifacts"] = write_artifacts([
        (MAC8, "mac8", "8-bit MAC", "Signed 8x8 multiply-accumulate, one byte "
         "per clock", ["ui_in[7:0]: operand byte (A then B)"],
         ["uo_out[7:0]: selected accumulator byte"],
         ["uio_in[0]: load A", "uio_in[1]: accumulate", "uio_in[2]: clear",
          "uio_in[5:4]: output byte select"]),
        (BinaryNeuron, "binneuron", "Binary neuron",
         "8 x 1-bit XNOR-popcount neuron with a threshold",
         ["ui_in[7:0]: 8 binary activations"],
         ["uo_out[6:0]: dot product", "uo_out[7]: fired"],
         ["uio_in[0]: load weights", "uio_in[1]: load threshold"]),
    ])
    for name, p in findings["artifacts"].items():
        print(f"   wrote outputs/{p['verilog']} ({p['verilog_bytes']} bytes) "
              f"and outputs/{p['info']}")

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=1, default=float)
    write_csv(findings)
    plot(findings)
    print("\nwrote outputs/findings.json, findings.csv, tinytapeout.png")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["section", "key", "value", "unit", "kind"])
        for s in f["simulation"]:
            w.writerow(["A sim", f"{s['design']} exact match", s["exact_match"],
                        "", "simulated"])
            w.writerow(["A sim", f"{s['design']} cycles per MAC",
                        round(s["cycles_per_mac"], 4), "cycles", "simulated"])
        for r in f["synthesis"]:
            for k in ("gates", "flops", "cells"):
                w.writerow(["B synth", f"{r['name']} {k}", r[k], "", "synthesized"])
            w.writerow(["C tile", f"{r['name']} area", round(r["area_um2"], 1),
                        "um^2", "arithmetic"])
            w.writerow(["C tile", f"{r['name']} copies per tile",
                        r["copies_per_tile"], "", "arithmetic"])
        for r, s in zip(f["throughput"], f["synthesis"]):
            w.writerow(["D throughput", s["name"], round(r["gops"], 4), "GOP/s",
                        "arithmetic"])
            w.writerow(["E pins", s["name"] + " pin-limited", r["pin_limited"],
                        "", "arithmetic"])
        for r, s in zip(f["energy"], f["synthesis"]):
            w.writerow(["F energy", s["name"] + " core", round(r["core_w"] * 1e6, 3),
                        "uW", "arithmetic"])
            w.writerow(["F energy", s["name"] + " pads", round(r["pad_w"] * 1e3, 3),
                        "mW", "arithmetic"])
        for k, v in f["ratios"].items():
            w.writerow(["F ratios", k, round(v, 4), "x", "mixed"])


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.8))

    # 1. area against the tile
    a = ax[0]
    names = [r["name"] for r in f["synthesis"]]
    placed = [r["placed_um2"] for r in f["synthesis"]]
    a.bar(names, placed, color=["#c0392b", "#27ae60"])
    a.axhline(f["tile_um2"], ls="--", color="#7f8c8d")
    a.text(1.4, f["tile_um2"], f"one 1x1 tile = {f['tile_um2']:.0f} um^2",
           fontsize=7, ha="right", va="bottom")
    for i, (r, v) in enumerate(zip(f["synthesis"], placed)):
        a.text(i, v, f"{v:.0f} um^2\n{r['copies_per_tile']} per tile",
               ha="center", va="bottom", fontsize=8)
    a.set_ylabel("placed area (um^2)")
    a.set_ylim(0, max(max(placed), f["tile_um2"]) * 1.35)
    a.set_title("C. what fits in one TinyTapeout tile", fontsize=10)
    a.grid(alpha=.3, axis="y")

    # 2. compute vs pins
    a = ax[1]
    x = range(len(names))
    a.bar([i - 0.2 for i in x], [r["macs_per_s"] / 1e6 for r in f["throughput"]],
          width=0.4, label="MACs the logic can do", color="#2980b9")
    a.bar([i + 0.2 for i in x],
          [r["macs_the_pins_can_feed"] / 1e6 for r in f["throughput"]],
          width=0.4, label="MACs the 8 input pins can feed", color="#e67e22")
    a.set_xticks(list(x)); a.set_xticklabels(names)
    a.set_yscale("log"); a.set_ylabel("MMAC/s (log)")
    a.set_title("E. eight wires is the real budget", fontsize=10)
    a.legend(fontsize=7); a.grid(alpha=.3, axis="y")

    # 3. efficiency
    a = ax[2]
    en = f["energy"]
    labels = ["tile\n(logic only)", "tile\n(with I/O pads)", "GTX 1070 Ti\n(measured)"]
    vals = [en[0]["gops_per_w_core"], en[0]["gops_per_w_total"],
            f["gpu"]["gops_per_w"]]
    a.bar(labels, vals, color=["#27ae60", "#c0392b", "#34495e"])
    for i, v in enumerate(vals):
        a.text(i, v, f"{v:,.1f}", ha="center", va="bottom", fontsize=8)
    a.set_yscale("log"); a.set_ylabel("GOP/s per watt (log)")
    a.set_title("F. the pads cost more than the chip", fontsize=10)
    a.grid(alpha=.3, axis="y")

    fig.suptitle("Project 50 - one square of silicon, honestly measured",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tinytapeout.png"), dpi=110)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
