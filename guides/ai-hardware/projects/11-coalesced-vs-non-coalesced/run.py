"""Project 11 - coalesced vs non-coalesced global memory.

Compiles coalesce.cu, runs four experiments, prints them as tables, and saves
findings.json / findings.csv / coalescing.png.

  A. stride sweep      - the collapse, and where it stops
  B. alignment sweep   - what a misaligned base pointer really costs
  C. lane permutation  - does it matter which lane reads which address?
  D. AoS vs SoA        - the same physics, as a data-layout decision
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "coalesce")

PERM_NAME = {0: "identity (lane i -> i)",
             1: "reversed (lane i -> 31-i)",
             2: "XOR 21 (lane i -> i^21)",
             3: "random permutation, from a table",
             4: "identity, from a table  [control]"}

LAYOUT_NAME = {"aos1": "AoS, read 1 of 4 fields",
               "soa1": "SoA, read 1 of 4 arrays",
               "aos4": "AoS, read all 4 fields",
               "soa4": "SoA, read all 4 arrays"}


def build_and_run():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
                        os.path.join(HERE, "coalesce.cu"), "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def parse(text):
    dev, s = {}, {"stride": [], "offset": [], "permute": [], "layout": []}
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), l2=int(f[4]))
        elif f[0] in ("stride", "random"):
            s["stride"].append(dict(stride=int(f[1]), reads=int(f[2]),
                                    sectors=int(f[3]), ms=float(f[4]),
                                    useful=float(f[5]), moved=float(f[6]),
                                    random=(f[0] == "random")))
        elif f[0] == "offset":
            s["offset"].append(dict(off=int(f[1]), ms=float(f[3]),
                                    useful=float(f[4])))
        elif f[0] == "permute":
            s["permute"].append(dict(mode=int(f[1]), ms=float(f[3]),
                                     useful=float(f[4])))
        elif f[0] == "layout":
            s["layout"].append(dict(kind=f[1], n=int(f[2]), ms=float(f[3]),
                                    useful=float(f[4]), moved=float(f[5])))
    return dev, s


def bar(v, vmax, width=20):
    n = int(round(width * v / vmax)) if vmax else 0
    return "#" * n + "." * (width - n)


def plot(dev, s, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    st = [r for r in s["stride"] if not r["random"]]
    rnd = next((r for r in s["stride"] if r["random"]), None)
    x = [r["stride"] for r in st]
    ax[0].plot(x, [r["useful"] for r in st], "o-", color="#c0392b",
               label="useful GB/s (bytes you asked for)")
    ax[0].plot(x, [r["moved"] for r in st], "s-", color="#2980b9",
               label="moved GB/s (32-B sectors fetched)")
    if rnd:
        ax[0].axhline(rnd["useful"], ls=":", color="#7f8c8d",
                      label=f"random gather: {rnd['useful']:.1f} GB/s useful")
    ax[0].axvline(8, ls="--", color="#27ae60", lw=1)
    ax[0].annotate("stride 8:\n1 sector per lane,\ncollapse stops",
                   xy=(8, 60), fontsize=8, color="#27ae60")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("stride (floats between neighbouring threads)")
    ax[0].set_ylabel("GB/s")
    ax[0].set_title("A. The bus stays full; your share of it does not")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3, which="both")

    labels = ["offset\n%d" % r["off"] for r in s["offset"]]
    vals = [r["useful"] for r in s["offset"]]
    ax[1].bar(labels, vals, color="#8e44ad")
    ax[1].set_ylim(0, max(vals) * 1.25)
    ax[1].axhline(vals[0], ls="--", color="#2c3e50", lw=1)
    ax[1].set_ylabel("useful GB/s")
    ax[1].set_title("B. Misaligned base pointer: %.1f%% worst case"
                    % (100 * (1 - min(vals) / vals[0])))
    ax[1].grid(alpha=.3, axis="y")

    short = {0: "identity", 1: "reversed", 2: "XOR 21",
             3: "random,\nfrom table", 4: "identity,\nfrom table\n(control)"}
    names = [short[r["mode"]] for r in s["permute"]]
    vals = [r["useful"] for r in s["permute"]]
    cols = ["#16a085"] * 3 + ["#d35400", "#7f8c8d"]
    ax[2].bar(range(len(vals)), vals, color=cols[:len(vals)])
    ax[2].set_xticks(range(len(vals)))
    ax[2].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax[2].set_ylabel("useful GB/s")
    ax[2].set_title("C. Shuffling lanes is free\n(grey control = the table lookup's cost)")
    ax[2].grid(alpha=.3, axis="y")

    fig.suptitle("Coalescing on %s (%d SMs, %.0f MB L2)"
                 % (dev["name"], dev["sms"], dev["l2"] / 1e6), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    dev, s = parse(build_and_run())
    print(f"Device: {dev['name']} (cc {dev['cc']}), {dev['sms']} SMs, "
          f"{dev['l2']/1024/1024:.0f} MB L2")
    print("Buffer is 256 MB - 128x the L2 - so every access is a real DRAM access.\n")

    st = [r for r in s["stride"] if not r["random"]]
    rnd = next((r for r in s["stride"] if r["random"]), None)
    base = st[0]["useful"]

    print("=" * 78)
    print("A. Stride sweep: thread i reads element i*stride")
    print("=" * 78)
    print(f"{'stride':>6} {'sectors/warp':>13} {'useful GB/s':>12} "
          f"{'vs stride 1':>11}  {'':20} {'moved GB/s':>10}")
    for r in st:
        print(f"{r['stride']:6d} {r['sectors']:13d} {r['useful']:12.2f} "
              f"{base/r['useful']:10.2f}x  {bar(r['useful'], base):20} "
              f"{r['moved']:10.2f}")
    if rnd:
        print(f"{'random':>6} {rnd['sectors']:13d} {rnd['useful']:12.2f} "
              f"{base/rnd['useful']:10.2f}x  {bar(rnd['useful'], base):20} "
              f"{rnd['moved']:10.2f}")
    plateau = [r for r in st if r["stride"] >= 8 and r["stride"] <= 64]
    print(f"\n  Useful bandwidth falls 1:1 with stride up to 8, then FLATTENS: "
          f"strides 8-64 are")
    print(f"  {min(r['useful'] for r in plateau):.2f}-"
          f"{max(r['useful'] for r in plateau):.2f} GB/s, a spread of "
          f"{max(r['useful'] for r in plateau)/min(r['useful'] for r in plateau):.3f}x.")
    print(f"  Meanwhile 'moved' sits at {min(r['moved'] for r in plateau):.0f}-"
          f"{max(r['moved'] for r in plateau):.0f} GB/s throughout: the bus never")
    print("  slowed down. You just stopped using 7 bytes out of every 8.")

    print()
    print("=" * 78)
    print("B. Alignment: same contiguous read, base pointer shifted")
    print("=" * 78)
    b0 = s["offset"][0]["useful"]
    for r in s["offset"]:
        print(f"  +{r['off']:2d} floats  {r['useful']:8.2f} GB/s   "
              f"{100*(r['useful']/b0-1):+6.2f}%  {bar(r['useful'], b0)}")
    worst = min(s["offset"], key=lambda r: r["useful"])
    print(f"\n  Worst case: {100*(1-worst['useful']/b0):.1f}% at offset "
          f"{worst['off']}. Not the 2x the textbooks imply.")

    print()
    print("=" * 78)
    print("C. Lane permutation: same 128-byte window, different lane -> address")
    print("=" * 78)
    p0 = s["permute"][0]["useful"]
    for r in s["permute"]:
        print(f"  {PERM_NAME[r['mode']]:<36} {r['useful']:8.2f} GB/s   "
              f"{100*(r['useful']/p0-1):+6.2f}%")
    m3 = next(r for r in s["permute"] if r["mode"] == 3)
    m4 = next((r for r in s["permute"] if r["mode"] == 4), None)
    if m4:
        print(f"\n  Mode 3 looks {100*(1-m3['useful']/p0):.0f}% slow - but its "
              f"control (mode 4) is {100*(1-m4['useful']/p0):.0f}% slow too.")
        print(f"  Reordering the lanes therefore costs "
              f"{100*(m4['useful']/m3['useful']-1):.1f}%. The rest is the extra load.")

    print()
    print("=" * 78)
    print("D. Array-of-structs vs struct-of-arrays (16-byte particle)")
    print("=" * 78)
    lay = {r["kind"]: r for r in s["layout"]}
    print(f"{'layout':<28} {'useful GB/s':>12} {'moved GB/s':>11} {'ms':>8}")
    for k in ("aos1", "soa1", "aos4", "soa4"):
        r = lay[k]
        print(f"{LAYOUT_NAME[k]:<28} {r['useful']:12.2f} {r['moved']:11.2f} "
              f"{r['ms']:8.3f}")
    print(f"\n  Reading ONE field: SoA is "
          f"{lay['soa1']['useful']/lay['aos1']['useful']:.2f}x faster.")
    print(f"  Reading ALL FOUR:  SoA is "
          f"{lay['soa4']['useful']/lay['aos4']['useful']:.2f}x faster - i.e. a tie.")
    print("  AoS is not slow. Reading a fraction of each struct is slow.")

    png = plot(dev, s, os.path.join(OUT, "coalescing.png"))
    if png:
        print(f"\nWrote {os.path.relpath(png, HERE)}")

    findings = {
        "device": dev,
        "headline": {
            "stride1_useful_gbs": base,
            "stride8_useful_gbs": next(r["useful"] for r in st if r["stride"] == 8),
            "stride1_over_stride8": base / next(r["useful"] for r in st
                                                if r["stride"] == 8),
            "plateau_spread_strides_8_to_64":
                max(r["useful"] for r in plateau) / min(r["useful"] for r in plateau),
            "moved_gbs_is_flat_over_plateau":
                [min(r["moved"] for r in plateau), max(r["moved"] for r in plateau)],
            "random_gather_useful_gbs": rnd["useful"] if rnd else None,
            "random_vs_coalesced": base / rnd["useful"] if rnd else None,
            "worst_alignment_penalty_pct": 100 * (1 - worst["useful"] / b0),
            "lane_shuffle_cost_pct":
                100 * (1 - m3["useful"] / m4["useful"]) if m4 else None,
            "table_lookup_cost_pct": 100 * (1 - m4["useful"] / p0) if m4 else None,
            "soa_over_aos_one_field":
                lay["soa1"]["useful"] / lay["aos1"]["useful"],
            "soa_over_aos_all_fields":
                lay["soa4"]["useful"] / lay["aos4"]["useful"],
        },
        "raw": s,
    }
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "setting", "ms", "useful_gbs", "moved_gbs",
                    "sectors_per_warp"])
        for r in s["stride"]:
            w.writerow(["stride", "random" if r["random"] else r["stride"],
                        r["ms"], r["useful"], r["moved"], r["sectors"]])
        for r in s["offset"]:
            w.writerow(["offset", r["off"], r["ms"], r["useful"], "", 4])
        for r in s["permute"]:
            w.writerow(["permute", PERM_NAME[r["mode"]], r["ms"], r["useful"],
                        "", 4])
        for r in s["layout"]:
            w.writerow(["layout", LAYOUT_NAME[r["kind"]], r["ms"], r["useful"],
                        r["moved"], ""])
    print(f"Wrote outputs/findings.json and outputs/findings.csv")


if __name__ == "__main__":
    main()
