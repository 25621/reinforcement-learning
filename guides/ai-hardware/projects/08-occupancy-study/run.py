"""Project 08 - occupancy, measured four different ways.

Compiles occ.cu, runs four sweeps, and reports theoretical vs achieved
occupancy next to the thing everyone actually cares about: throughput.

  A. block size   - the knob every tutorial starts with
  B. grid size    - theoretical occupancy cannot see it; achieved occupancy is
                    almost entirely determined by it
  C. shared memory- a pure occupancy throttle with the arithmetic held fixed
  D. ILP x occupancy - two independent ways to hide the same latency
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "occ")


def build_and_run():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61",
                        os.path.join(HERE, "occ.cu"), "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def parse(text):
    dev, s = {}, {"blocksize": [], "gridsize": [], "sharedmem": [], "ilp": []}
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), max_warps=int(f[4]),
                       max_threads=int(f[5]), regs_per_sm=int(f[6]),
                       smem_per_sm=int(f[7]))
        elif f[0].startswith("#"):
            continue
        elif f[0] == "ilp":
            s["ilp"].append(dict(chains=int(f[1]), blocks_per_sm=int(f[2]),
                                 theo=float(f[3]), theo_limit=float(f[4]),
                                 achieved=float(f[5]), peak_blocks=int(f[6]),
                                 ms=float(f[7]), tflops=float(f[8]),
                                 regs=int(f[9])))
        elif f[0] in s:
            s[f[0]].append(dict(x=int(f[1]), tpb=int(f[2]), theo=float(f[3]),
                                achieved=float(f[4]), peak_blocks=int(f[5]),
                                ms=float(f[6]), gloads=float(f[7])))
    return dev, s


def bar(v, width=22):
    n = int(round(v * width))
    return "#" * n + "." * (width - n)


def main():
    dev, s = parse(build_and_run())
    print(f"Device: {dev['name']} (cc {dev['cc']}), {dev['sms']} SMs, "
          f"max {dev['max_warps']} warps/SM ({dev['max_threads']} threads), "
          f"{dev['regs_per_sm']//1024}K registers/SM, "
          f"{dev['smem_per_sm']//1024} KB shared/SM")
    print("\nThroughput below is billions of DEPENDENT random loads per second.")
    print("Each load's address is the previous load's value, so a thread can")
    print("never overlap two of its own loads. Only more warps can help.\n")

    # ---------------- A ----------------
    print("=" * 78)
    print("A. Block size (grid fixed at 32 blocks/SM, so the GPU is always full)")
    print("=" * 78)
    print(f"{'threads/blk':>11} {'theoretical':>12} {'achieved':>9}  "
          f"{'occupancy':<24} {'Gloads/s':>9}")
    for r in s["blocksize"]:
        print(f"{r['x']:11d} {r['theo']:11.0%} {r['achieved']:9.1%}  "
              f"{bar(r['achieved']):<24} {r['gloads']:9.2f}")
    g = [r["gloads"] for r in s["blocksize"]]
    lo = min(s["blocksize"], key=lambda r: r["theo"])
    print(f"\n  Spread in throughput across a 32x range of block size: "
          f"{max(g)/min(g):.3f}x")
    print(f"  At {lo['x']} threads/block theoretical occupancy is only "
          f"{lo['theo']:.0%} (a hard cap of 32 blocks per SM x "
          f"{lo['x']//32} warp each),")
    print(f"  yet throughput is {100*lo['gloads']/max(g):.0f}% of the best. "
          "Half the occupancy, same speed.")

    # ---------------- B ----------------
    print()
    print("=" * 78)
    print("B. Grid size (block size fixed at 256 - theoretical occupancy CANNOT move)")
    print("=" * 78)
    print(f"{'blocks':>7} {'blocks/SM':>10} {'theoretical':>12} {'achieved':>9}  "
          f"{'occupancy':<24} {'Gloads/s':>9}")
    for r in s["gridsize"]:
        print(f"{r['x']:7d} {r['x']/dev['sms']:10.2f} {r['theo']:11.0%} "
              f"{r['achieved']:9.1%}  {bar(r['achieved']):<24} {r['gloads']:9.2f}")
    first, last = s["gridsize"][0], s["gridsize"][-1]
    knee = next((r for r in s["gridsize"]
                 if r["gloads"] > 0.97 * max(x["gloads"] for x in s["gridsize"])),
                last)
    print(f"\n  Theoretical occupancy said {first['theo']:.0%} for every row.")
    print(f"  Achieved occupancy went {first['achieved']:.1%} -> "
          f"{last['achieved']:.1%}, and throughput went "
          f"{first['gloads']:.2f} -> {last['gloads']:.2f} Gloads/s "
          f"({last['gloads']/first['gloads']:.0f}x).")
    print(f"  Throughput is already within 3% of its best at {knee['x']} blocks "
          f"= {knee['achieved']:.0%} achieved occupancy.")
    print("  Everything above that buys nothing. The useful range of occupancy")
    print(f"  for this kernel ends at about {knee['achieved']:.0%}.")

    # ---------------- C ----------------
    print()
    print("=" * 78)
    print("C. Shared memory per block - occupancy throttled, arithmetic untouched")
    print("=" * 78)
    print(f"{'KB/block':>9} {'theoretical':>12} {'achieved':>9} "
          f"{'blocks/SM':>10}  {'occupancy':<24} {'Gloads/s':>9}")
    for r in s["sharedmem"]:
        print(f"{r['x']:9d} {r['theo']:11.0%} {r['achieved']:9.1%} "
              f"{r['peak_blocks']:10d}  {bar(r['achieved']):<24} {r['gloads']:9.2f}")
    hi = s["sharedmem"][0]
    lowest = min(s["sharedmem"], key=lambda r: r["theo"])
    print(f"\n  Occupancy fell {hi['theo']/lowest['theo']:.0f}x "
          f"({hi['theo']:.0%} -> {lowest['theo']:.0%}) and throughput changed by "
          f"{100*(lowest['gloads']/hi['gloads']-1):+.1f}%.")
    print("  Shared memory is not free, but on THIS kernel it was free.")

    # ---------------- D ----------------
    print()
    print("=" * 78)
    print("D. Two ways to hide the same latency: more warps, or more ILP")
    print("=" * 78)
    print("Compute-bound kernel: each thread runs CHAINS independent FMA chains.")
    print("An FMA takes ~6 cycles; the SM needs ~6 independent FMAs in flight per")
    print("core to stay busy. Those can come from 6 warps or from 6 chains.\n")
    chains = sorted({r["chains"] for r in s["ilp"]})
    bps_list = sorted({r["blocks_per_sm"] for r in s["ilp"]})
    grid = {(r["chains"], r["blocks_per_sm"]): r for r in s["ilp"]}
    hdr = " | ".join(f"{c:2d} chains" for c in chains)
    print(f"{'warps/SM asked':>15} | {hdr}")
    print(f"{'for':>15} | " + " | ".join("  TFLOP/s" for _ in chains))
    print("-" * (17 + 12 * len(chains)))
    for b in bps_list:
        cells = []
        for c in chains:
            r = grid.get((c, b))
            cells.append(f"{r['tflops']:9.2f}" if r else " " * 9)
        print(f"{b*2:15d} | " + " | ".join(cells))
    print(f"{'achieved occ.':>15} | " + " | ".join(
        f"{grid[(c, bps_list[-1])]['achieved']:9.1%}" for c in chains)
        + "   <- at the bottom row")
    print(f"{'registers/thread':>15} | " + " | ".join(
        f"{grid[(c, bps_list[0])]['regs']:9d}" for c in chains))
    print(f"{'occupancy ceiling':>15} | " + " | ".join(
        f"{grid[(c, bps_list[-1])]['theo_limit']:9.0%}" for c in chains)
        + "   <- set by those registers")

    best = max(s["ilp"], key=lambda r: r["tflops"])
    c1 = [r for r in s["ilp"] if r["chains"] == 1]
    best_c1 = max(c1, key=lambda r: r["tflops"])
    lowocc = min((r for r in s["ilp"] if r["tflops"] >= best_c1["tflops"]),
                 key=lambda r: r["achieved"])
    print(f"\n  Best throughput anywhere in the table: {best['tflops']:.2f} TFLOP/s "
          f"at {best['achieved']:.1%} achieved occupancy "
          f"({best['chains']} chains, {best['blocks_per_sm']} blocks/SM).")
    print(f"  Best throughput with only 1 chain: {best_c1['tflops']:.2f} TFLOP/s "
          f"at {best_c1['achieved']:.1%} occupancy - it never gets there,")
    print("  because one chain per thread cannot fill the pipeline no matter how")
    print("  many warps you add.")
    print(f"  {lowocc['chains']} chains at {lowocc['achieved']:.1%} occupancy already "
          f"matches the best 1-chain result at {best_c1['achieved']:.1%} occupancy - "
          f"{best_c1['achieved']/lowocc['achieved']:.1f}x less occupancy, same speed.")

    rows = []
    for sec, lst in s.items():
        for r in lst:
            rows.append(dict(section=sec, **r))
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    findings = dict(
        device=dev,
        blocksize_spread_x=max(g) / min(g),
        blocksize_min_theo=lo["theo"], blocksize_min_theo_pct_of_best=lo["gloads"] / max(g),
        gridsize_speedup_x=last["gloads"] / first["gloads"],
        gridsize_achieved_lo=first["achieved"], gridsize_achieved_hi=last["achieved"],
        gridsize_knee_occupancy=knee["achieved"],
        smem_occupancy_drop_x=hi["theo"] / lowest["theo"],
        smem_throughput_change_pct=100 * (lowest["gloads"] / hi["gloads"] - 1),
        ilp_best=best, ilp_best_1chain=best_c1, ilp_match_at_low_occ=lowocc,
        sections=s)
    with open(os.path.join(OUT, "findings.json"), "w") as fh:
        json.dump(findings, fh, indent=2)
    print(f"\nwrote {OUT}/findings.json and findings.csv")
    plot(dev, s, knee, best, best_c1)


def plot(dev, s, knee, best, best_c1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # (a) grid size: theoretical flat, achieved climbs, throughput follows
    xs = [r["x"] for r in s["gridsize"]]
    ax[0].semilogx(xs, [100 * r["theo"] for r in s["gridsize"]], "s--",
                   color="#B0B0B0", base=2, label="theoretical occupancy")
    ax[0].semilogx(xs, [100 * r["achieved"] for r in s["gridsize"]], "o-",
                   color="#4C78A8", base=2, label="achieved occupancy")
    ax[0].set_xlabel(f"blocks launched ({dev['sms']} SMs)")
    ax[0].set_ylabel("occupancy (%)")
    ax[0].set_ylim(0, 112)
    a2 = ax[0].twinx()
    a2.semilogx(xs, [r["gloads"] for r in s["gridsize"]], "^-", color="#54A24B",
                base=2, label="throughput")
    a2.set_ylabel("G dependent loads/s", color="#54A24B")
    a2.set_ylim(0, max(r["gloads"] for r in s["gridsize"]) * 1.2)
    a2.axvline(knee["x"], ls=":", c="#54A24B")
    a2.text(knee["x"] * 1.15, 0.35, f"throughput flat\nfrom here\n"
            f"({knee['achieved']:.0%} occupancy)", fontsize=8, color="#54A24B")
    ax[0].legend(loc="upper left", fontsize=8)
    ax[0].set_title("(a) Theoretical occupancy is blind to grid size\n"
                    "the same kernel, 0.7% to 93% achieved", fontsize=10.5)
    ax[0].grid(alpha=.3, which="both")

    # (b) shared memory throttle
    xs = [r["x"] for r in s["sharedmem"]]
    ax[1].plot(xs, [100 * r["theo"] for r in s["sharedmem"]], "s--",
               color="#B0B0B0", label="theoretical occupancy")
    ax[1].plot(xs, [100 * r["achieved"] for r in s["sharedmem"]], "o-",
               color="#4C78A8", label="achieved occupancy")
    ax[1].set_ylim(0, 112)
    ax[1].set_xlabel("dynamic shared memory per block (KB)")
    ax[1].set_ylabel("occupancy (%)")
    b2 = ax[1].twinx()
    b2.plot(xs, [r["gloads"] for r in s["sharedmem"]], "^-", color="#54A24B")
    b2.set_ylim(0, max(r["gloads"] for r in s["sharedmem"]) * 1.25)
    b2.set_ylabel("G dependent loads/s", color="#54A24B")
    ax[1].legend(loc="lower left", fontsize=8)
    ax[1].set_title("(b) Occupancy cut 4x, throughput unchanged\n"
                    "the green line is what you were trying to optimise",
                    fontsize=10.5)
    ax[1].grid(alpha=.3)

    # (c) ILP vs occupancy
    chains = sorted({r["chains"] for r in s["ilp"]})
    cols = {1: "#E45756", 4: "#F58518", 16: "#4C78A8", 64: "#B279A2"}
    for c in chains:
        pts = sorted([r for r in s["ilp"] if r["chains"] == c],
                     key=lambda r: r["achieved"])
        ax[2].plot([100 * r["achieved"] for r in pts], [r["tflops"] for r in pts],
                   "o-", color=cols.get(c, "#888"), label=f"{c} chains/thread")
    ax[2].axhline(best_c1["tflops"], ls=":", c="#E45756", lw=1)
    ax[2].annotate(f"best possible with 1 chain\n({best_c1['tflops']:.1f} TFLOP/s, "
                   f"at {best_c1['achieved']:.0%} occupancy)",
                   (12, best_c1["tflops"] - 1.4), fontsize=8, color="#E45756")
    ax[2].plot([100 * best["achieved"]], [best["tflops"]], "*", ms=17,
               color="#54A24B", zorder=5)
    ax[2].annotate(f"best overall: {best['tflops']:.2f} TFLOP/s\n"
                   f"at {best['achieved']:.0%} occupancy",
                   (100 * best["achieved"], best["tflops"]),
                   textcoords="offset points", xytext=(8, -28), fontsize=8.5,
                   color="#2c7a2c")
    ax[2].set_xlabel("achieved occupancy (%)")
    ax[2].set_ylabel("TFLOP/s")
    ax[2].set_xlim(0, 100)
    ax[2].legend(fontsize=8, loc="lower right")
    ax[2].set_title("(c) Occupancy is one of two ways to hide latency\n"
                    "more chains per thread beats more warps", fontsize=10.5)
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "occupancy.png"), dpi=110)
    print(f"wrote {OUT}/occupancy.png")


if __name__ == "__main__":
    main()
