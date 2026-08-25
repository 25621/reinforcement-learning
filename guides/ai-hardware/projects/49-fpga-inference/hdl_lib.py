"""hdl_lib.py - the hardware toolchain for projects 49 and 50.

Three jobs:
  1. turn an Amaranth design into RTLIL (the netlist format yosys reads)
  2. run yosys on it twice, to get the two counts that decide whether a design
     fits real silicon:
        coarse  - multipliers, adders, memories, flip-flops (what an FPGA
                  spends DSP slices and block RAM on)
        gates   - AND/OR/XOR/NOT/DFF (what an ASIC spends standard cells on)
  3. convert gate counts into an area in square micrometres, using published
     SkyWater 130 nm standard-cell sizes

Why yosys and not a vendor tool: Vivado and Quartus are tens of gigabytes and
need a licence and a board. yosys is a 16 MB pip install (`yowasp-yosys`, a
WebAssembly build) and produces the same *counts* - which is all the arithmetic
below needs. Place-and-route, timing closure and bitstream generation are the
parts we genuinely cannot do here, and the READMEs say so where it matters.
"""

import os
import re
import subprocess
import sys
import tempfile

from amaranth.back import rtlil

YOSYS = "yowasp-yosys"

# SkyWater 130 nm high-density standard cells. Every cell is a whole number of
# 0.46 um wide "sites" on a 2.72 um tall row, so areas come in multiples of
# 1.2512 um^2. These are the published sky130_fd_sc_hd areas.
SITE_UM2 = 0.46 * 2.72
SKY130_AREA = {          # um^2 per instance
    "$_AND_": 4 * SITE_UM2,      # and2_1
    "$_NAND_": 3 * SITE_UM2,     # nand2_1
    "$_OR_": 4 * SITE_UM2,       # or2_1
    "$_NOR_": 3 * SITE_UM2,      # nor2_1
    "$_XOR_": 7 * SITE_UM2,      # xor2_1
    "$_XNOR_": 7 * SITE_UM2,     # xnor2_1
    "$_NOT_": 3 * SITE_UM2,      # inv_1
    "$_MUX_": 7 * SITE_UM2,      # mux2_1
    "$_ANDNOT_": 4 * SITE_UM2,
    "$_ORNOT_": 4 * SITE_UM2,
    "$_AOI3_": 5 * SITE_UM2,
    "$_OAI3_": 5 * SITE_UM2,
    "$_AOI4_": 6 * SITE_UM2,
    "$_OAI4_": 6 * SITE_UM2,
}
DFF_AREA = 16 * SITE_UM2         # dfxtp_1, the plain D flip-flop
DEFAULT_CELL_AREA = 5 * SITE_UM2  # anything unlisted: charged an average cell


def to_rtlil(elaboratable, name, ports, path):
    """Amaranth design -> RTLIL text file that yosys can read."""
    il = rtlil.convert(elaboratable, name=name, ports=ports)
    with open(path, "w") as f:
        f.write(il)
    return path


def _yosys(script, cwd):
    with tempfile.NamedTemporaryFile("w", suffix=".ys", dir=cwd,
                                     delete=False) as f:
        f.write(script)
        ys = f.name
    try:
        r = subprocess.run([YOSYS, "-s", os.path.basename(ys)], cwd=cwd,
                           capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("yowasp-yosys not found: pip install yowasp-yosys")
    finally:
        os.unlink(ys)
    if r.returncode != 0:
        raise SystemExit("yosys failed:\n" + r.stdout[-3000:] + r.stderr[-2000:])
    return r.stdout


def _parse_stat(log):
    """Pull the cell histogram out of a yosys `stat` report."""
    cells, wires, capture = {}, 0, False
    for line in log.splitlines():
        if line.strip().startswith("=== ") and "===" in line[4:]:
            capture = True
            cells, wires = {}, 0
            continue
        if not capture:
            continue
        m = re.match(r"\s+(\d+)\s+wire bits", line)
        if m:
            wires = int(m.group(1))
        m = re.match(r"\s+(\d+)\s+(\$\S+|\w+)\s*$", line)
        if m and m.group(2).startswith("$"):
            cells[m.group(2)] = cells.get(m.group(2), 0) + int(m.group(1))
    return cells, wires


def synth(il_path, top):
    """Two syntheses of the same design.

    `coarse` stops before the word-level operators are broken up, so $mul and
    $add cells are still visible - those are the FPGA's DSP slices.
    `gates` finishes the job, leaving only 2-input logic gates and flip-flops -
    those are the ASIC's standard cells.

    Note both stop before yosys' ABC pass. ABC is a separate binary that this
    WebAssembly build cannot launch, so the gate counts here are yosys'
    straightforward mapping rather than an optimised technology mapping. A real
    flow would be 20-40% smaller; every conclusion below is therefore
    *conservative* about area, which is the safe direction.
    """
    cwd = os.path.dirname(os.path.abspath(il_path))
    base = os.path.basename(il_path)
    # `-run :coarse` stops before yosys' `alumacc` pass, which merges a
    # multiply and the adds that follow it into a single `$macc_v2` macro cell.
    # That macro is good for synthesis and useless for counting: it would
    # report "1 cell" where the hardware has nine multipliers.
    coarse = _parse_stat(_yosys(
        f"read_rtlil {base}\nsynth -top {top} -flatten -run :coarse\n"
        f"opt -fast\nstat\n", cwd))
    gates = _parse_stat(_yosys(
        f"read_rtlil {base}\nsynth -top {top} -flatten -run :fine\n"
        f"techmap\nsimplemap\nopt -full\nstat\n", cwd))
    return dict(coarse_cells=coarse[0], coarse_wire_bits=coarse[1],
                gate_cells=gates[0], gate_wire_bits=gates[1])


def gate_summary(gate_cells):
    """Total gates, flip-flops, and SkyWater 130 nm area."""
    n_gates = n_ff = 0
    area = 0.0
    for cell, count in gate_cells.items():
        if "DFF" in cell or "SDFF" in cell or "ADFF" in cell or "DLATCH" in cell:
            n_ff += count
            area += count * DFF_AREA
        else:
            n_gates += count
            area += count * SKY130_AREA.get(cell, DEFAULT_CELL_AREA)
    return dict(gates=n_gates, flops=n_ff, cells=n_gates + n_ff,
                area_um2=area)


def coarse_summary(coarse_cells):
    """The FPGA-relevant counts: multipliers (DSP slices) and memory bits."""
    mults = sum(v for k, v in coarse_cells.items() if k.startswith("$mul"))
    adds = sum(v for k, v in coarse_cells.items()
               if k.startswith("$add") or k.startswith("$sub"))
    mems = sum(v for k, v in coarse_cells.items() if k.startswith("$mem"))
    return dict(multipliers=mults, adders=adds, memories=mems)
