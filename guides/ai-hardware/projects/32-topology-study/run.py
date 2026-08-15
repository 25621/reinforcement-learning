"""Project 32 - read the topology, then predict what the network will do.

Sections
  A  measured: this machine's real topology (nvidia-smi, sysfs, NUMA)
  B  measured: PCIe host<->device bandwidth, pinned vs pageable, vs size
  C  measured: does CPU affinity move the number? (a control, and a null result)
  D  arithmetic: ring bottlenecks for four topologies, best ring vs worst ring
  E  the lookup table for reading `nvidia-smi topo -m` on someone else's box

Runtime: ~3.4 s.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from topo import (LINK_GBS, Topology, dgx1_cube_mesh,  # noqa: E402
                  parse_nvidia_smi_topo, uniform)

findings: dict = {}

PCIE_GEN_GBS_PER_LANE = {"2.5 GT/s PCIe": 0.25, "5.0 GT/s PCIe": 0.5,
                         "8.0 GT/s PCIe": 0.985, "16.0 GT/s PCIe": 1.969,
                         "32.0 GT/s PCIe": 3.938, "64.0 GT/s PCIe": 7.877}


# ------------------------------------------------------------------ A

def _sysfs(dev, name):
    try:
        return Path(f"/sys/bus/pci/devices/{dev}/{name}").read_text().strip()
    except OSError:
        return None


def _nvidia_bdfs():
    out = []
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
        d = Path(dev).name
        cls, vendor = _sysfs(d, "class"), _sysfs(d, "vendor")
        if cls and cls.startswith("0x030") and vendor == "0x10de":
            out.append(d)
    return out


# Read the link speed before anything in this process talks to the driver:
# `nvidia-smi` alone is enough to wake the link up and change the answer.
COLD_LINK = {d: _sysfs(d, "current_link_speed") for d in _nvidia_bdfs()}


def section_a():
    topo_txt = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True,
                              text=True).stdout
    (OUT / "nvidia_smi_topo.txt").write_text(topo_txt)
    labels, matrix = parse_nvidia_smi_topo(topo_txt)

    gpus = []
    for dev in sorted(glob.glob("/sys/bus/pci/devices/*")):
        d = Path(dev).name
        cls = _sysfs(d, "class")
        vendor = _sysfs(d, "vendor")
        # 0x030000 is "VGA controller", which on this box also matches the CPU's
        # built-in graphics; 0x10de is NVIDIA, which is the card we mean.
        if cls is None or not cls.startswith("0x030") or vendor != "0x10de":
            continue
        speed = _sysfs(d, "current_link_speed")
        width = _sysfs(d, "current_link_width")
        maxs = _sysfs(d, "max_link_speed")
        maxw = _sysfs(d, "max_link_width")
        numa = _sysfs(d, "numa_node")
        per_lane = PCIE_GEN_GBS_PER_LANE.get(speed or "", None)
        max_lane = PCIE_GEN_GBS_PER_LANE.get(maxs or "", None)
        gpus.append(dict(bdf=d, cold_link_speed=COLD_LINK.get(d), idle_link_speed=speed, link_width=width,
                         max_link_speed=maxs, max_link_width=maxw, numa_node=numa,
                         idle_GBs=(per_lane * int(width)) if per_lane and width else None,
                         theoretical_GBs=(max_lane * int(maxw)) if max_lane and maxw else None,
                         local_cpus=_sysfs(d, "local_cpulist")))

    a = dict(topo_matrix_labels=labels, topo_matrix=matrix, gpus=gpus,
             torch_device_count=torch.cuda.device_count(),
             numa_nodes=len(glob.glob("/sys/devices/system/node/node*")),
             cpu_count=os.cpu_count())
    findings["A_topology"] = a
    g = gpus[0] if gpus else {}
    a["numa_reported"] = g.get("numa_node")
    print(f"A: {len(gpus)} GPU(s); cold link {g.get('cold_link_speed')}; "
          f"after nvidia-smi {g.get('idle_link_speed')} x{g.get('link_width')} "
          f"= {g.get('idle_GBs')} GB/s, max {g.get('max_link_speed')} x{g.get('max_link_width')} "
          f"= {g.get('theoretical_GBs'):.2f} GB/s one way; "
          f"NUMA node {g.get('numa_node')} of {a['numa_nodes']}, {a['cpu_count']} CPUs")


# ------------------------------------------------------------------ B

def _copy_bw(nbytes, pinned, direction, reps=20):
    n = nbytes // 4
    host = torch.empty(n)
    if pinned:
        host = host.pin_memory()
    dev = torch.empty(n, device="cuda")
    for _ in range(3):
        (dev.copy_(host) if direction == "h2d" else host.copy_(dev))
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            if direction == "h2d":
                dev.copy_(host)
            else:
                host.copy_(dev)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) / reps)
    return nbytes / best / 1e9, best


def link_speed_under_load(bdf, seconds=0.4):
    """Re-read the PCIe link speed *while* the link is busy.

    A PCIe link drops to its slowest gear when nothing is using it, so the
    number sysfs reports on an idle machine is a power state, not a
    capability. This is the same reason a car's rev counter reads low when
    parked. Measuring it under load is the only way to see the real gear."""
    import threading
    host = torch.empty(1 << 24).pin_memory()
    dev = torch.empty(1 << 24, device="cuda")
    stop = threading.Event()

    def churn():
        while not stop.is_set():
            dev.copy_(host)
            torch.cuda.synchronize()

    th = threading.Thread(target=churn)
    th.start()
    time.sleep(seconds)
    speed = _sysfs(bdf, "current_link_speed")
    stop.set()
    th.join()
    return speed


def section_b():
    sizes = [1 << 12, 1 << 16, 1 << 20, 1 << 22, 1 << 24, 1 << 26]
    rows = []
    for nbytes in sizes:
        reps = 50 if nbytes <= (1 << 20) else 10
        for pinned in [True, False]:
            for direction in ["h2d", "d2h"]:
                gbs, sec = _copy_bw(nbytes, pinned, direction, reps)
                rows.append(dict(bytes=nbytes, pinned=pinned, dir=direction,
                                 GBs=gbs, us=sec * 1e6))
    findings["B_pcie"] = rows
    big = {(r["pinned"], r["dir"]): r for r in rows if r["bytes"] == (1 << 26)}
    gpu0 = findings["A_topology"]["gpus"][0]
    theo = gpu0["theoretical_GBs"]
    loaded = link_speed_under_load(gpu0["bdf"])
    gpu0["loaded_link_speed"] = loaded
    print(f"B: link speed cold {gpu0['cold_link_speed']} -> under load {loaded}")
    for k, r in big.items():
        print(f"B: 64 MiB {'pinned  ' if k[0] else 'pageable'} {k[1]}: "
              f"{r['GBs']:6.2f} GB/s ({r['GBs']/theo*100:5.1f}% of the {theo:.1f} GB/s link)")
    small = [r for r in rows if r["bytes"] == (1 << 12) and r["pinned"] and r["dir"] == "h2d"][0]
    findings["B_summary"] = dict(
        latency_us=small["us"],
        pinned_h2d_GBs=big[(True, "h2d")]["GBs"],
        pageable_h2d_GBs=big[(False, "h2d")]["GBs"],
        pinned_speedup=big[(True, "h2d")]["GBs"] / big[(False, "h2d")]["GBs"],
        theoretical_GBs=theo,
        cold_link_speed=gpu0["cold_link_speed"],
        idle_link_speed=gpu0["idle_link_speed"],
        loaded_link_speed=loaded,
        efficiency=big[(True, "h2d")]["GBs"] / theo)
    print(f"B: pinned is {findings['B_summary']['pinned_speedup']:.2f}x pageable; "
          f"4 KiB copy takes {small['us']:.1f} us (all fixed cost)")


# ------------------------------------------------------------------ C

def section_c():
    """Bind the process to different CPU sets and re-measure. On a one-socket
    machine this should change nothing -- which is the point: the method has to
    be shown to produce a null result where a null result is correct, before
    anyone trusts it where the answer is unknown."""
    all_cpus = sorted(os.sched_getaffinity(0))
    local = findings["A_topology"]["gpus"][0]["local_cpus"]
    half = len(all_cpus) // 2
    sets = {"all": all_cpus, "first_half": all_cpus[:half], "second_half": all_cpus[half:]}
    res = {}
    try:
        for name, cpus in sets.items():
            os.sched_setaffinity(0, set(cpus))
            gbs, _ = _copy_bw(1 << 25, True, "h2d", reps=20)
            res[name] = dict(cpus=cpus, GBs=gbs)
            print(f"C: affinity {name:12s} ({len(cpus)} cpus) -> {gbs:6.2f} GB/s")
    finally:
        os.sched_setaffinity(0, set(all_cpus))
    spread = max(r["GBs"] for r in res.values()) / min(r["GBs"] for r in res.values())
    res["spread"] = spread
    res["gpu_local_cpulist"] = local
    findings["C_affinity"] = res
    print(f"C: spread across affinity sets {spread:.3f}x "
          f"(one NUMA node, so the expected answer is 1.00x)")


# ------------------------------------------------------------------ D

def section_d():
    systems = [
        uniform(4, "PHB", "4-GPU PCIe workstation", "all four behind one host bridge"),
        dgx1_cube_mesh(),
        uniform(8, "NV12", "DGX A100 (NVSwitch)", "every pair full NVLink bandwidth"),
        uniform(8, "NV18", "DGX H100 (NVSwitch)", "every pair full NVLink bandwidth"),
    ]
    grad_bytes = 14e9        # 7B parameters in bf16
    rows = []
    for t in systems:
        best_order, best_bw = t.best_ring()
        worst_order, worst_bw = t.worst_ring()
        rows.append(dict(
            system=t.name, note=t.note, n=t.n,
            best_ring=best_order, best_bottleneck_GBs=best_bw,
            worst_ring=worst_order, worst_bottleneck_GBs=worst_bw,
            ring_choice_matters=best_bw / worst_bw,
            predicted_busbw_GBs=t.predicted_busbw(),
            allreduce_14GB_ms=t.predicted_allreduce_s(grad_bytes) * 1e3))
        print(f"D: {t.name:34s} best ring {best_bw:6.1f} GB/s, worst {worst_bw:6.1f} GB/s "
              f"({rows[-1]['ring_choice_matters']:5.2f}x), 14 GB all-reduce "
              f"{rows[-1]['allreduce_14GB_ms']:8.1f} ms")
    # and this machine, which has exactly one GPU and therefore no ring at all
    findings["D_rings"] = dict(rows=rows, this_machine_gpus=torch.cuda.device_count(),
                               grad_bytes=grad_bytes)


# ------------------------------------------------------------------ E

def section_e():
    guide = {
        "NV#": "# NVLink lanes between the pair. Highest bandwidth; what you want.",
        "PIX": "Same PCIe switch. Peer-to-peer DMA works, no host involvement.",
        "PXB": "Multiple PCIe switches, still below the host bridge.",
        "PHB": "Up to the host bridge (the CPU's PCIe root). Slower, host-adjacent.",
        "NODE": "Across PCIe host bridges inside one NUMA node.",
        "SYS": "Across the CPU-CPU link (UPI/QPI) or between NUMA nodes. Worst case.",
        "X": "Self.",
    }
    findings["E_legend"] = dict(meaning=guide, assumed_GBs=LINK_GBS)
    print("E: legend and assumed bandwidths written to findings.json")


# ------------------------------------------------------------------ plot

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    rows = findings["B_pcie"]
    for pinned in [True, False]:
        for d in ["h2d", "d2h"]:
            pts = [r for r in rows if r["pinned"] == pinned and r["dir"] == d]
            ax[0][0].plot([p["bytes"] for p in pts], [p["GBs"] for p in pts],
                          "o-" if pinned else "s--",
                          label=f"{'pinned' if pinned else 'pageable'} {d}")
    ax[0][0].axhline(findings["B_summary"]["theoretical_GBs"], color="red", ls=":",
                     lw=1, label="link rate")
    ax[0][0].set_xscale("log", base=2)
    ax[0][0].set_xlabel("transfer bytes")
    ax[0][0].set_ylabel("GB/s")
    ax[0][0].set_title("A. measured PCIe, this machine")
    ax[0][0].legend(fontsize=6)
    ax[0][0].grid(alpha=.3)

    c = {k: v for k, v in findings["C_affinity"].items() if isinstance(v, dict)}
    ax[0][1].bar(list(c), [v["GBs"] for v in c.values()])
    ax[0][1].set_ylabel("GB/s (32 MiB pinned H2D)")
    ax[0][1].set_title("B. CPU affinity: a control that should do nothing")
    ax[0][1].grid(alpha=.3)

    d = findings["D_rings"]["rows"]
    ys = range(len(d))
    ax[1][0].barh([y - .2 for y in ys], [r["best_bottleneck_GBs"] for r in d],
                  height=.4, label="best ring")
    ax[1][0].barh([y + .2 for y in ys], [r["worst_bottleneck_GBs"] for r in d],
                  height=.4, label="worst ring")
    ax[1][0].set_yticks(list(ys))
    ax[1][0].set_yticklabels([r["system"] for r in d], fontsize=6)
    ax[1][0].set_xscale("log")
    ax[1][0].set_xlabel("ring bottleneck (GB/s)")
    ax[1][0].set_title("C. the ring you pick is worth this much")
    ax[1][0].legend(fontsize=7)
    ax[1][0].grid(alpha=.3)

    ax[1][1].barh(list(ys), [r["allreduce_14GB_ms"] for r in d])
    ax[1][1].set_yticks(list(ys))
    ax[1][1].set_yticklabels([r["system"] for r in d], fontsize=6)
    ax[1][1].set_xscale("log")
    ax[1][1].set_xlabel("predicted ms for one 14 GB all-reduce")
    ax[1][1].set_title("D. same model, four fabrics")
    ax[1][1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(OUT / "topology.png", dpi=120)


def main():
    t0 = time.perf_counter()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    plot()
    findings["runtime_s"] = time.perf_counter() - t0
    print(f"total runtime {findings['runtime_s']:.1f} s")
    (OUT / "findings.json").write_text(json.dumps(findings, indent=1))
    with open(OUT / "findings.csv", "w") as f:
        f.write("section,key,bytes,value\n")
        for r in findings["B_pcie"]:
            f.write(f"B,{'pinned' if r['pinned'] else 'pageable'}_{r['dir']},"
                    f"{r['bytes']},{r['GBs']:.4f}\n")
        for r in findings["D_rings"]["rows"]:
            f.write(f"D,{r['system']},,{r['predicted_busbw_GBs']:.2f}\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
