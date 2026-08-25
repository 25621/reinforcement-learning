"""Project 03 - measure the real memory bandwidth of this GPU.

Compiles bandwidth.cu, runs it, and turns the raw timings into GB/s, a
comparison against the spec sheet, and three plots.

The spec number is not looked up on the internet - it is computed from the
device properties the driver reports, so the same script gives the right answer
on any card.
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)


def run_bench():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    exe = os.path.join(OUT, "bandwidth")
    cmd = ["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
           os.path.join(HERE, "bandwidth.cu"), "-o", exe]
    print("compiling:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([exe], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-3000:] + r.stderr[-3000:])
    return r.stdout


def parse(text):
    dev, kernels, sweep, pcie, blocks = None, [], [], [], []
    for line in text.strip().splitlines():
        f = line.split(",")
        if line.startswith("#device"):
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), clock_khz=int(f[4]),
                       mem_khz=int(f[5]), bus_bits=int(f[6]), l2=int(f[7]),
                       total_mem=int(f[8]))
        elif line.startswith("#"):
            print(line)
        elif f[0] == "kernel":
            kernels.append(dict(name=f[1], bytes=float(f[2]), sec=float(f[3])))
        elif f[0] == "blocksize":
            blocks.append(dict(tpb=int(f[1]), bytes=float(f[2]), sec=float(f[3])))
        elif f[0] == "sweep":
            sweep.append(dict(buf=int(f[2]), bytes=float(f[3]), sec=float(f[4])))
        elif f[0] == "pcie":
            pcie.append(dict(name=f[1], bytes=float(f[2]), sec=float(f[3])))
    return dev, kernels, blocks, sweep, pcie


def main():
    dev, kernels, blocks, sweep, pcie = parse(run_bench())

    # Spec peak, computed not googled:
    #   memory clock x 2 (DDR transfers twice per clock) x bus width in bytes
    peak = dev["mem_khz"] * 1e3 * 2 * dev["bus_bits"] / 8
    print(f"\ndevice     : {dev['name']}  (sm_{dev['cc'].replace('.','')}), "
          f"{dev['total_mem']/1e9:.1f} GB")
    print(f"L2 cache   : {dev['l2']/1e6:.1f} MB")
    print(f"spec peak  : {peak/1e9:.1f} GB/s "
          f"= {dev['mem_khz']/1e6:.3f} GHz x 2 x {dev['bus_bits']} bits")

    print(f"\n=== Kernels on a 256 MB buffer ===")
    print(f"{'kernel':<18} {'GB/s':>9} {'% of peak':>10}")
    rows = []
    for k in kernels:
        if k["bytes"] == 0:
            continue
        gbs = k["bytes"] / k["sec"] / 1e9
        print(f"{k['name']:<18} {gbs:9.1f} {100*gbs*1e9/peak:10.1f}")
        rows.append(dict(section="kernel", name=k["name"], gbs=gbs,
                         pct_peak=100 * gbs * 1e9 / peak, sec=k["sec"]))
    empty = [k for k in kernels if k["name"] == "empty_launch"][0]["sec"]
    print(f"\nempty kernel launch: {empty*1e6:.2f} us  "
          f"(every measurement below this is meaningless)")

    scalar = [r for r in rows if r["name"] == "copy_scalar"][0]
    vec = [r for r in rows if r["name"] == "copy_float4"][0]
    memcpy = [r for r in rows if r["name"] == "cudaMemcpy_D2D"][0]
    print(f"float4 vs scalar copy: {vec['gbs']/scalar['gbs']:.3f}x")
    print(f"our copy vs cudaMemcpy: {vec['gbs']/memcpy['gbs']:.3f}x")

    print(f"\n=== Block size, same copy ===")
    for b in blocks:
        gbs = b["bytes"] / b["sec"] / 1e9
        print(f"  {b['tpb']:>5} threads/block: {gbs:7.1f} GB/s")
        rows.append(dict(section="blocksize", name=str(b["tpb"]), gbs=gbs,
                         pct_peak=100 * gbs * 1e9 / peak, sec=b["sec"]))
    bg = [b["bytes"] / b["sec"] / 1e9 for b in blocks]
    print(f"  spread across 32..1024 threads/block: {max(bg)/min(bg):.3f}x")

    print(f"\n=== Size sweep (copy, float4) ===")
    print("'working set' = bytes read + bytes written. It is what has to fit in L2.")
    print("'overhead' = the fixed launch cost as a share of the measured time.\n")
    print(f"{'buffer':>10} {'work set':>10} {'time':>10} {'GB/s':>9} {'overhead':>9}"
          f"  fits in L2?")
    for s in sweep:
        gbs = s["bytes"] / s["sec"] / 1e9
        ws = s["bytes"]  # read buffer + write buffer
        oh = 100 * empty / s["sec"]
        where = "yes" if ws <= dev["l2"] else "no"
        flag = "  <- number is mostly launch cost" if oh > 50 else ""
        print(f"{s['buf']/1024:9.0f}K {ws/1024:9.0f}K {s['sec']*1e6:9.2f}us "
              f"{gbs:9.1f} {oh:8.0f}%  {where}{flag}")
        rows.append(dict(section="sweep", name=str(s["buf"]), gbs=gbs,
                         pct_peak=100 * gbs * 1e9 / peak, sec=s["sec"]))

    # Trustworthy L2 point: fits in L2 and is not dominated by launch overhead.
    in_l2 = [s for s in sweep if s["bytes"] <= dev["l2"] and empty / s["sec"] < 0.5]
    out_l2 = [s for s in sweep if s["bytes"] > 4 * dev["l2"]]
    best_l2 = max(in_l2, key=lambda s: s["bytes"] / s["sec"])
    best_dram = max(out_l2, key=lambda s: s["bytes"] / s["sec"])
    g_l2 = best_l2["bytes"] / best_l2["sec"] / 1e9
    g_l2_corr = best_l2["bytes"] / (best_l2["sec"] - empty) / 1e9
    g_dram = best_dram["bytes"] / best_dram["sec"] / 1e9
    first_dram = min(out_l2, key=lambda s: s["buf"])
    print(f"\nbest inside L2 ({best_l2['buf']/1024:.0f}K buffer): {g_l2:.0f} GB/s "
          f"= {g_l2*1e9/peak:.2f}x the DRAM spec peak")
    print(f"  with the {empty*1e6:.2f}us launch cost subtracted: {g_l2_corr:.0f} GB/s")
    print(f"best outside L2                    : {g_dram:.0f} GB/s")
    print(f"L2 is {g_l2/g_dram:.2f}x faster than DRAM on this chip "
          f"({g_l2_corr/g_dram:.2f}x with launch cost removed).")
    prev = [s for s in sweep if s["buf"] == best_l2["buf"]][0]
    nxt = [s for s in sweep if s["buf"] == prev["buf"] * 2]
    if nxt:
        d = nxt[0]["bytes"] / nxt[0]["sec"] / 1e9
        print(f"Doubling the buffer from {prev['buf']/1024:.0f}K to "
              f"{nxt[0]['buf']/1024:.0f}K - one step past the {dev['l2']/1e6:.0f} MB L2 - "
              f"drops throughput {g_l2/d:.1f}x ({g_l2:.0f} -> {d:.0f} GB/s) "
              f"with no change to the code.")

    print(f"\n=== PCIe (host <-> device) ===")
    for p in pcie:
        gbs = p["bytes"] / p["sec"] / 1e9
        print(f"  {p['name']:<14} {gbs:7.2f} GB/s")
        rows.append(dict(section="pcie", name=p["name"], gbs=gbs,
                         pct_peak=100 * gbs * 1e9 / peak, sec=p["sec"]))
    pg = {p["name"]: p["bytes"] / p["sec"] / 1e9 for p in pcie}
    print(f"  pinned vs pageable, H2D: {pg['H2D_pinned']/pg['H2D_pageable']:.2f}x")
    print(f"  pinned vs pageable, D2H: {pg['D2H_pinned']/pg['D2H_pageable']:.2f}x")
    print(f"  GPU memory is {g_dram/pg['H2D_pinned']:.1f}x faster than the PCIe link.")

    findings = dict(
        device=dev["name"], l2_bytes=dev["l2"], spec_peak_gbs=peak / 1e9,
        copy_scalar_gbs=scalar["gbs"], copy_float4_gbs=vec["gbs"],
        float4_speedup=vec["gbs"] / scalar["gbs"],
        vs_cudamemcpy=vec["gbs"] / memcpy["gbs"],
        best_pct_peak=max(r["pct_peak"] for r in rows if r["section"] == "kernel"),
        blocksize_spread=max(bg) / min(bg),
        empty_launch_us=empty * 1e6,
        l2_gbs=g_l2, dram_gbs=g_dram, l2_over_dram=g_l2 / g_dram,
        pcie=pg, dram_over_pcie=g_dram / pg["H2D_pinned"])
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "gbs", "pct_peak", "sec"])
        w.writeheader()
        w.writerows(rows)
    plot(dev, kernels, blocks, sweep, pcie, peak, empty)


def plot(dev, kernels, blocks, sweep, pcie, peak, empty):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # ---- (a) size sweep ----
    xs = [s["buf"] / 1024 for s in sweep]
    ys = [s["bytes"] / s["sec"] / 1e9 for s in sweep]
    ax[0].semilogx(xs, ys, "o-", color="#4C78A8", base=2)
    ax[0].axhline(peak / 1e9, ls="--", c="k", lw=1)
    ax[0].text(6, peak / 1e9 * 1.06, f"DRAM spec peak {peak/1e9:.0f} GB/s", fontsize=8)
    l2_buf = dev["l2"] / 2 / 1024  # buffer size at which read+write fills L2
    ax[0].axvline(l2_buf, ls="--", c="crimson", lw=1)
    ax[0].text(l2_buf * 1.15, max(ys) * 0.55,
               f"working set = L2\n({dev['l2']/1e6:.0f} MB)", fontsize=8, color="crimson")
    small = [x for x, s in zip(xs, sweep) if empty / s["sec"] > 0.5]
    if small:
        ax[0].axvspan(min(xs) / 1.5, max(small) * 1.4, color="gray", alpha=.15)
        ax[0].text(min(xs) * 1.1, max(ys) * 0.8,
                   f"launch overhead\n({empty*1e6:.1f} us) dominates", fontsize=8, color="gray")
    ax[0].set_xlabel("buffer size (KiB, log scale)")
    ax[0].set_ylabel("apparent bandwidth (GB/s)")
    ax[0].set_title("(a) The same kernel, three different answers\n"
                    "overhead-bound / cache-bound / DRAM-bound")
    ax[0].grid(alpha=.3, which="both")

    # ---- (b) kernel variants vs peak ----
    ks = [k for k in kernels if k["bytes"] > 0]
    names = [k["name"] for k in ks]
    gs = [k["bytes"] / k["sec"] / 1e9 for k in ks]
    ax[1].barh(names, gs, color="#54A24B")
    ax[1].axvline(peak / 1e9, ls="--", c="k", lw=1)
    ax[1].text(peak / 1e9 * 0.99, -0.6, f"spec {peak/1e9:.0f}", fontsize=8, ha="right")
    for i, g in enumerate(gs):
        ax[1].text(g + 3, i, f"{g:.0f} ({100*g*1e9/peak:.0f}%)", va="center", fontsize=8.5)
    ax[1].set_xlabel("GB/s on a 256 MB buffer")
    ax[1].set_xlim(0, peak / 1e9 * 1.25)
    ax[1].set_title("(b) Five ways to move bytes\nnone of them reach the sticker number")
    ax[1].invert_yaxis()
    ax[1].grid(alpha=.3, axis="x")

    # ---- (c) the hierarchy, log scale ----
    g_l2 = max(s["bytes"] / s["sec"] / 1e9 for s in sweep
               if s["bytes"] <= dev["l2"] and empty / s["sec"] < 0.5)
    g_dram = max(s["bytes"] / s["sec"] / 1e9 for s in sweep if s["bytes"] > 4 * dev["l2"])
    pg = {p["name"]: p["bytes"] / p["sec"] / 1e9 for p in pcie}
    labels = ["L2 cache\n(on chip)", "GDDR5\n(on board)", "PCIe pinned\n(to host RAM)",
              "PCIe pageable\n(to host RAM)"]
    vals = [g_l2, g_dram, pg["H2D_pinned"], pg["H2D_pageable"]]
    cs = ["#B279A2", "#4C78A8", "#F58518", "#E45756"]
    ax[2].bar(labels, vals, color=cs)
    ax[2].set_yscale("log")
    for i, v in enumerate(vals):
        ax[2].text(i, v * 1.1, f"{v:.0f} GB/s", ha="center", fontsize=9)
    ax[2].set_ylabel("GB/s (log scale)")
    ax[2].set_ylim(1, g_l2 * 4)
    ax[2].set_title(f"(c) One step down the hierarchy\ncosts {g_dram/pg['H2D_pinned']:.0f}x "
                    f"at the PCIe boundary")
    ax[2].grid(alpha=.3, axis="y", which="both")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bandwidth.png"), dpi=110)
    print(f"\nwrote {OUT}/bandwidth.png")


if __name__ == "__main__":
    main()
