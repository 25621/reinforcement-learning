"""Project 14 - saturating off-chip memory bandwidth.

Compiles hbm.cu and answers three questions:

  A. How much tuning does a vector add need to pass 80% of peak?
  B. What does a byte cost - read, write, or half-written sector?
  C. Where are the caches, seen only through bandwidth?
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "hbm")

VARIANT = {"scalar_1elem": "1 element per thread (textbook)",
           "float2": "float2 (8 B per instruction)",
           "float4": "float4 (16 B per instruction)",
           "gridstride_scalar": "grid-stride loop, scalar",
           "gridstride_float4": "grid-stride loop, float4"}

COST = {"read2": ("read 2 arrays, no write", 2, 0),
        "copy_1r1w": ("copy: 1 read + 1 write", 1, 1),
        "write_only": ("write only", 0, 1),
        "add_2r1w": ("vector add: 2 reads + 1 write", 2, 1),
        "write_half_dense": ("write half the floats, contiguous [control]", 0, .5),
        "write_half_strided": ("write half the floats, every other one", 0, .5)}


def build_and_run():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
                        os.path.join(HERE, "hbm.cu"), "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def parse(text):
    dev, s = {}, {"variant": [], "block": [], "grid": [], "cost": [],
                  "ratio": [], "size": []}
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), peak=float(f[4]),
                       l2=int(f[5]), bus=int(f[6]))
        elif f[0] == "variant":
            s["variant"].append(dict(kind=f[1], tpb=int(f[2]), grid=int(f[3]),
                                     ms=float(f[4]), gbs=float(f[5])))
        elif f[0] == "block":
            s["block"].append(dict(tpb=int(f[1]), grid=int(f[2]),
                                   ms=float(f[3]), gbs=float(f[4])))
        elif f[0] == "grid":
            s["grid"].append(dict(bps=int(f[1]), grid=int(f[2]),
                                  ms=float(f[3]), gbs=float(f[4])))
        elif f[0] == "cost":
            s["cost"].append(dict(kind=f[1], ms=float(f[2]), gbs=float(f[3])))
        elif f[0] == "ratio":
            s["ratio"].append(dict(reads=int(f[1]), ms=float(f[2]),
                                   gbs=float(f[3])))
        elif f[0] == "size":
            s["size"].append(dict(bytes=int(f[1]), ms=float(f[2]),
                                  gbs=float(f[3])))
    return dev, s


def plot(dev, s, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    peak = dev["peak"]

    rows = ([("block %d" % r["tpb"], r["gbs"]) for r in s["block"]] +
            [(VARIANT[r["kind"]].split(" (")[0], r["gbs"]) for r in s["variant"]] +
            [("grid %dx SMs" % r["bps"], r["gbs"]) for r in s["grid"]])
    ax[0].barh(range(len(rows)), [v for _, v in rows], color="#2980b9")
    ax[0].set_yticks(range(len(rows)))
    ax[0].set_yticklabels([n for n, _ in rows], fontsize=6.5)
    ax[0].invert_yaxis()
    ax[0].axvline(peak, color="#c0392b", lw=1.2, label=f"spec peak {peak:.0f}")
    ax[0].axvline(.8 * peak, color="#27ae60", ls="--", lw=1.2,
                  label=f"80% of peak = {.8*peak:.0f}")
    ax[0].set_xlim(0, peak * 1.05)
    ax[0].set_xlabel("GB/s (2 reads + 1 write counted)")
    ax[0].set_title("A. 19 configurations, %.3fx apart"
                    % (max(v for _, v in rows) / min(v for _, v in rows)))
    ax[0].legend(fontsize=7, loc="lower right"); ax[0].grid(alpha=.3, axis="x")

    ck = ["read2", "write_only", "add_2r1w", "copy_1r1w",
          "write_half_dense", "write_half_strided"]
    cd = {r["kind"]: r for r in s["cost"]}
    vals = [cd[k]["gbs"] for k in ck if k in cd]
    names = [COST[k][0].replace(" [", "\n[").replace(": ", ":\n") for k in ck if k in cd]
    cols = ["#16a085", "#16a085", "#2980b9", "#e67e22", "#7f8c8d", "#c0392b"]
    ax[1].bar(range(len(vals)), vals, color=cols[:len(vals)])
    ax[1].set_xticks(range(len(vals)))
    ax[1].set_xticklabels(names, rotation=30, ha="right", fontsize=6.5)
    ax[1].axhline(peak, color="#c0392b", lw=1)
    ax[1].set_ylabel("GB/s of USEFUL bytes")
    ax[1].set_title("B. Reads, writes, and a half-written sector")
    ax[1].grid(alpha=.3, axis="y")

    sz = s["size"]
    ax[2].plot([r["bytes"] / 1024 for r in sz], [r["gbs"] for r in sz], "o-",
               color="#8e44ad")
    ax[2].axvline(dev["l2"] / 2 / 1024, ls="--", color="#c0392b", lw=1)
    ax[2].annotate("working set = L2 (%.0f MB)" % (dev["l2"] / 1e6),
                   xy=(dev["l2"] / 2 / 1024 * 1.15, 40), fontsize=7,
                   color="#c0392b", rotation=90)
    ax[2].axhline(peak, color="#7f8c8d", ls=":", lw=1)
    ax[2].set_xscale("log", base=2); ax[2].set_yscale("log")
    ax[2].set_xlabel("array size (KiB)   [working set = 2x this]")
    ax[2].set_ylabel("copy GB/s")
    ax[2].set_title("C. The whole hierarchy, from one kernel")
    ax[2].grid(alpha=.3, which="both")

    fig.suptitle("Memory bandwidth on %s (%d-bit bus, %.1f GB/s spec)"
                 % (dev["name"], dev["bus"], peak), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    dev, s = parse(build_and_run())
    peak = dev["peak"]
    print(f"Device: {dev['name']} (cc {dev['cc']}), {dev['sms']} SMs, "
          f"{dev['bus']}-bit bus, {dev['l2']//1024//1024} MB L2")
    print(f"Spec peak bandwidth: {peak:.2f} GB/s. Target for this project: "
          f"{0.8*peak:.1f} GB/s (80%).\n")

    print("=" * 80)
    print("A. Tuning a vector add (256 MB per array, 768 MB moved)")
    print("=" * 80)
    base = next(r for r in s["variant"] if r["kind"] == "scalar_1elem")
    print(f"{'variant':<36} {'GB/s':>8} {'% peak':>7} {'vs textbook':>12}")
    for r in s["variant"]:
        print(f"{VARIANT[r['kind']]:<36} {r['gbs']:8.2f} "
              f"{100*r['gbs']/peak:6.1f}% {r['gbs']/base['gbs']:11.3f}x")
    print(f"\n{'block size':<36} {'GB/s':>8} {'% peak':>7}")
    for r in s["block"]:
        print(f"{'  %d threads/block' % r['tpb']:<36} {r['gbs']:8.2f} "
              f"{100*r['gbs']/peak:6.1f}%")
    print(f"\n{'grid size (grid-stride float4)':<36} {'GB/s':>8} {'% peak':>7}")
    for r in s["grid"]:
        print(f"{'  %d blocks/SM (%d blocks)' % (r['bps'], r['grid']):<36} "
              f"{r['gbs']:8.2f} {100*r['gbs']/peak:6.1f}%")

    allr = ([r["gbs"] for r in s["variant"]] + [r["gbs"] for r in s["block"]] +
            [r["gbs"] for r in s["grid"]])
    best = max(allr)
    print(f"\n  The textbook one-liner already reaches "
          f"{100*base['gbs']/peak:.1f}% of peak.")
    print(f"  The best of all {len(allr)} configurations reaches "
          f"{100*best/peak:.1f}%.")
    print(f"  Total spread across every knob in this project: "
          f"{best/min(allr):.3f}x.")
    print("  There is nothing to tune. On a kernel with no reuse to find, the "
          "bus is")
    print("  saturated the moment the accesses are coalesced, and every other "
          "knob is noise.")

    print()
    print("=" * 80)
    print("B. What does a byte cost?")
    print("=" * 80)
    cd = {r["kind"]: r for r in s["cost"]}
    print(f"{'kernel':<44} {'ms':>8} {'useful GB/s':>12}")
    for k in ("read2", "write_only", "add_2r1w", "copy_1r1w",
              "write_half_dense", "write_half_strided"):
        if k in cd:
            print(f"{COST[k][0]:<44} {cd[k]['ms']:8.3f} {cd[k]['gbs']:12.2f}")

    rd, wr = cd["read2"]["gbs"], cd["write_only"]["gbs"]
    cp, ad = cd["copy_1r1w"]["gbs"], cd["add_2r1w"]["gbs"]
    print(f"\n  Writes are as cheap as reads: {wr:.1f} vs {rd:.1f} GB/s "
          f"({100*wr/rd:.1f}%).")
    print(f"  But MIXING them costs: 1 read + 1 write = {cp:.1f} GB/s, "
          f"{100*(1-cp/rd):.0f}% below read-only.")
    print("  That is DRAM bus turnaround - the data bus is shared, and "
          "reversing its")
    print("  direction wastes cycles. Reads and writes are cheap; alternating "
          "is not.")
    print(f"\n{'reads per write':<20} {'GB/s':>8} {'% peak':>7}")
    for r in s["ratio"]:
        lab = "read-only (no write)" if r["reads"] == 99 else f"{r['reads']} : 1"
        print(f"{lab:<20} {r['gbs']:8.2f} {100*r['gbs']/peak:6.1f}%")
    r1 = next(r["gbs"] for r in s["ratio"] if r["reads"] == 1)
    r4 = next(r["gbs"] for r in s["ratio"] if r["reads"] == 4)
    rinf = next(r["gbs"] for r in s["ratio"] if r["reads"] == 99)
    print(f"\n  1:1 costs {100*(1-r1/rinf):.0f}% against pure reads; 4:1 costs "
          f"{100*(1-r4/rinf):.0f}%. The penalty")
    print("  saturates almost immediately - one write in the mix costs "
          "nearly all of it.")

    dense, strided = cd["write_half_dense"], cd["write_half_strided"]
    slow = strided["ms"] / dense["ms"]
    n_bytes = 256 * 1024 * 1024                      # one array
    rmw = 2.0 * n_bytes / (strided["ms"] * 1e-3) / 1e9
    masked = 1.0 * n_bytes / (strided["ms"] * 1e-3) / 1e9
    print(f"\n  The half-written sector: writing every other float takes "
          f"{slow:.2f}x as long as")
    print(f"  writing the same COUNT of floats contiguously "
          f"({strided['ms']:.3f} vs {dense['ms']:.3f} ms).")
    print("  Two hypotheses, and the arithmetic eliminates one:")
    print(f"    read-modify-write (2x traffic) would need "
          f"{rmw:.0f} GB/s - ABOVE the {peak:.0f} GB/s bus. Impossible.")
    print(f"    full-sector write with byte enables (1x traffic) needs "
          f"{masked:.0f} GB/s - {100*masked/wr:.0f}% of the")
    print(f"    write-only rate. Plausible.")
    print("  So this GPU does not fetch a sector to partially overwrite it - "
          "but it does")
    print("  move the whole sector, so half your write bandwidth is spent on "
          "bytes you")
    print("  never touched. Same lesson as project 11, on the store side.")

    print()
    print("=" * 80)
    print("C. Size sweep: the memory hierarchy, drawn by one copy kernel")
    print("=" * 80)
    print(f"{'array size':>12} {'working set':>12} {'ms':>10} {'GB/s':>9}")
    for r in s["size"]:
        ws = 2 * r["bytes"]
        mark = "  <- L2 capacity" if ws == dev["l2"] else ""
        print(f"{r['bytes']//1024:11d}K {ws//1024:11d}K {r['ms']:10.4f} "
              f"{r['gbs']:9.2f}{mark}")
    peak_row = max(s["size"], key=lambda r: r["gbs"])
    after = [r for r in s["size"] if 2 * r["bytes"] > dev["l2"]]
    cliff = min(after, key=lambda r: r["bytes"])
    print(f"\n  Best: {peak_row['gbs']:.0f} GB/s at a "
          f"{2*peak_row['bytes']//1024} KB working set - "
          f"{peak_row['gbs']/cliff['gbs']:.1f}x the DRAM rate,")
    print(f"  and {peak_row['gbs']/peak:.1f}x the DRAM chip's spec peak, "
          "because it never reached the DRAM chips.")
    print(f"  One doubling later ({2*cliff['bytes']//1024} KB working set, just "
          f"past the {dev['l2']//1024//1024} MB L2) it is "
          f"{cliff['gbs']:.0f} GB/s.")
    print(f"  The cliff is {peak_row['gbs']/cliff['gbs']:.1f}x for one "
          "doubling of the array.")
    small = s["size"][0]
    print(f"\n  At the other end, {small['bytes']//1024} KB scores only "
          f"{small['gbs']:.1f} GB/s - that is not memory,")
    print(f"  that is the {small['ms']*1000:.2f} us kernel-launch floor "
          "(project 3) being measured instead.")

    png = plot(dev, s, os.path.join(OUT, "bandwidth.png"))
    if png:
        print(f"\nWrote {os.path.relpath(png, HERE)}")

    findings = {
        "device": dev,
        "headline": {
            "spec_peak_gbs": peak,
            "textbook_vecadd_gbs": base["gbs"],
            "textbook_pct_of_peak": 100 * base["gbs"] / peak,
            "best_of_all_configs_gbs": best,
            "best_pct_of_peak": 100 * best / peak,
            "total_tuning_spread": best / min(allr),
            "read_only_gbs": rd,
            "write_only_gbs": wr,
            "copy_1r1w_gbs": cp,
            "vecadd_2r1w_gbs": ad,
            "mixing_penalty_pct_1to1": 100 * (1 - cp / rd),
            "mixing_penalty_pct_4to1": 100 * (1 - r4 / rinf),
            "half_sector_write_slowdown": slow,
            "rmw_would_need_gbs": rmw,
            "byte_enable_needs_gbs": masked,
            "l2_peak_gbs": peak_row["gbs"],
            "l2_cliff_ratio": peak_row["gbs"] / cliff["gbs"],
        },
        "raw": s,
    }
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "setting", "ms", "gbs", "pct_of_peak"])
        for r in s["variant"]:
            w.writerow(["variant", VARIANT[r["kind"]], r["ms"], r["gbs"],
                        100 * r["gbs"] / peak])
        for r in s["block"]:
            w.writerow(["block", r["tpb"], r["ms"], r["gbs"],
                        100 * r["gbs"] / peak])
        for r in s["grid"]:
            w.writerow(["grid", f"{r['bps']} blocks/SM", r["ms"], r["gbs"],
                        100 * r["gbs"] / peak])
        for r in s["cost"]:
            w.writerow(["cost", COST[r["kind"]][0], r["ms"], r["gbs"],
                        100 * r["gbs"] / peak])
        for r in s["ratio"]:
            w.writerow(["ratio", "read-only" if r["reads"] == 99
                        else f"{r['reads']} reads : 1 write", r["ms"], r["gbs"],
                        100 * r["gbs"] / peak])
        for r in s["size"]:
            w.writerow(["size", f"{r['bytes']} B array", r["ms"], r["gbs"],
                        100 * r["gbs"] / peak])
    print("Wrote outputs/findings.json and outputs/findings.csv")


if __name__ == "__main__":
    main()
