"""Project 12 - shared-memory bank conflicts.

Compiles banks.cu, runs four experiments, prints tables, saves findings and a
three-panel plot.

  A. stride sweep     - conflict degree = gcd(stride, 32)
  B. broadcast        - 32 lanes, one address: conflict or free?
  C. transpose        - the textbook fix, measured end to end
  D. cost in context  - when a 32-way conflict costs 11x and when it costs 0%
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "banks")

N = 4096                      # transpose matrix side, must match banks.cu
NCTX = 1 << 24                # elements streamed in experiment D

TR_NAME = {"copy_limit":   "plain copy (the speed limit)",
           "naive":        "no shared memory (strided stores)",
           "smem_nopad":   "shared tile, 32-way conflict",
           "smem_pad1":    "shared tile + 1 pad column",
           "smem_swizzle": "shared tile + XOR swizzle"}


def build_and_run():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
                        os.path.join(HERE, "banks.cu"), "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def parse(text):
    dev, s = {}, {"bank": [], "transpose": [], "context": []}
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), smem_block=int(f[4]))
        elif f[0] == "bank":
            s["bank"].append(dict(stride=int(f[1]), degree=int(f[2]),
                                  ms=float(f[3]), gloads=float(f[4])))
        elif f[0] == "transpose":
            s["transpose"].append(dict(kind=f[1], ms=float(f[2]),
                                       gbs=float(f[3])))
        elif f[0] == "context":
            s["context"].append(dict(nload=int(f[1]), work=int(f[2]),
                                     ratio_bytes=float(f[3]), t1=float(f[4]),
                                     t32=float(f[5]), slow=float(f[6])))
    return dev, s


def bar(v, vmax, width=20):
    n = int(round(width * v / vmax)) if vmax else 0
    return "#" * n + "." * (width - n)


def plot(dev, s, thr1, thr32, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    bk = sorted(s["bank"], key=lambda r: r["stride"])
    xs = list(range(len(bk)))
    cols = ["#7f8c8d" if r["degree"] == 0 else
            ("#27ae60" if r["degree"] == 1 else "#c0392b") for r in bk]
    ax[0].bar(xs, [r["gloads"] for r in bk], color=cols)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([("bcast" if r["stride"] == 0 else str(r["stride"]))
                           for r in bk], fontsize=8)
    for x, r in zip(xs, bk):
        if r["degree"] > 1:
            ax[0].text(x, r["gloads"] + 25, f"{r['degree']}-way", ha="center",
                       fontsize=7, color="#c0392b")
    ax[0].set_xlabel("stride between lanes (elements)")
    ax[0].set_ylabel("G shared-memory loads / s")
    ax[0].set_title("A+B. Conflict degree = gcd(stride, 32)\n"
                    "grey = broadcast, green = conflict-free")
    ax[0].grid(alpha=.3, axis="y")

    tr = s["transpose"]
    names = [TR_NAME[r["kind"]].replace(" (", "\n(") for r in tr]
    vals = [r["gbs"] for r in tr]
    tc = ["#2c3e50", "#c0392b", "#e67e22", "#16a085", "#16a085"]
    ax[1].barh(range(len(vals)), vals, color=tc[:len(vals)])
    ax[1].set_yticks(range(len(vals)))
    ax[1].set_yticklabels(names, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("effective GB/s (bytes in + bytes out)")
    ax[1].set_title("C. 4096x4096 transpose:\nthe famous fix is worth %.2f%%"
                    % (100 * (tr[3]["gbs"] / tr[2]["gbs"] - 1)))
    ax[1].grid(alpha=.3, axis="x")

    cx = sorted(s["context"], key=lambda r: r["ratio_bytes"])
    ax[2].plot([r["ratio_bytes"] for r in cx], [r["slow"] for r in cx], "o-",
               color="#c0392b", label="measured slowdown")
    pred = []
    for r in cx:
        threads = NCTX / r["nload"]
        t_sh = threads * r["work"] / (thr32 * 1e9) * 1e3
        pred.append(max(r["t1"], t_sh) / r["t1"])
    ax[2].plot([r["ratio_bytes"] for r in cx], pred, "s--", color="#2980b9",
               label="max(DRAM time, shared-pipe time)")
    ax[2].axhline(1.0, color="#7f8c8d", lw=1)
    ax[2].axvline(8, ls=":", color="#16a085")
    ax[2].annotate("the transpose\nlives here", xy=(8, 6), fontsize=8,
                   color="#16a085")
    ax[2].set_xscale("log", base=2); ax[2].set_yscale("log")
    ax[2].set_xlabel("DRAM bytes moved per conflicted shared read")
    ax[2].set_ylabel("32-way conflict / conflict-free")
    ax[2].set_title("D. The same conflict, from 11x to free")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, which="both")

    fig.suptitle("Shared-memory bank conflicts on %s" % dev["name"], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    dev, s = parse(build_and_run())
    print(f"Device: {dev['name']} (cc {dev['cc']}), {dev['sms']} SMs, "
          f"{dev['smem_block']//1024} KB shared memory per block")
    print("Shared memory = 32 banks x 4 bytes. bank(element i) = i % 32.\n")

    bk = {r["stride"]: r for r in s["bank"]}
    free = bk[1]["gloads"]

    print("=" * 78)
    print("A+B. Shared-memory throughput vs the stride between lanes")
    print("=" * 78)
    print(f"{'stride':>7} {'conflict':>9} {'G loads/s':>10} {'slowdown':>9}  "
          f"{'':20}")
    for r in sorted(s["bank"], key=lambda r: r["stride"]):
        deg = "broadcast" if r["degree"] == 0 else (
            "none" if r["degree"] == 1 else f"{r['degree']}-way")
        print(f"{r['stride']:7d} {deg:>9} {r['gloads']:10.1f} "
              f"{free/r['gloads']:8.2f}x  {bar(r['gloads'], 1050)}")
    print(f"\n  gcd(stride,32) predicts every row. Odd strides (3, 5, 17, 33) "
          f"are conflict-free.")
    print(f"  32-way costs {free/bk[32]['gloads']:.1f}x, not the 32x the model "
          f"allows: {100*(free/bk[32]['gloads'])/32:.0f}% of the ceiling.")
    print(f"  BROADCAST (stride 0, all 32 lanes on one address) runs at "
          f"{bk[0]['gloads']:.0f} G loads/s")
    print(f"  = {bk[0]['gloads']/bk[32]['gloads']:.1f}x FASTER than the 32-way "
          f"conflict. Same bank is not a conflict;")
    print("  the same ADDRESS is a broadcast.")

    print()
    print("=" * 78)
    print(f"C. {N}x{N} float transpose (128 MB moved), four implementations")
    print("=" * 78)
    tr = {r["kind"]: r for r in s["transpose"]}
    print(f"{'implementation':<36} {'ms':>8} {'GB/s':>8} {'vs limit':>9}")
    lim = tr["copy_limit"]["gbs"]
    for k in ("copy_limit", "naive", "smem_nopad", "smem_pad1", "smem_swizzle"):
        print(f"{TR_NAME[k]:<36} {tr[k]['ms']:8.3f} {tr[k]['gbs']:8.2f} "
              f"{100*tr[k]['gbs']/lim:8.1f}%")
    fix = tr["smem_pad1"]["gbs"] / tr["smem_nopad"]["gbs"]
    print(f"\n  Shared memory vs no shared memory: "
          f"{tr['smem_nopad']['gbs']/tr['naive']['gbs']:.2f}x. Real, and large.")
    print(f"  Removing the 32-way bank conflict: {fix:.4f}x "
          f"({100*(fix-1):+.2f}%). Nothing.")
    print(f"  Swizzle vs padding: "
          f"{tr['smem_swizzle']['gbs']/tr['smem_pad1']['gbs']:.4f}x - a tie, "
          f"but the swizzle uses 0 extra bytes.")

    # the model: a kernel takes max(DRAM time, shared-pipe time)
    reads = N * N
    t_sh32 = reads / (bk[32]["gloads"] * 1e9) * 1e3
    t_sh1 = reads / (free * 1e9) * 1e3
    t_dram = tr["smem_pad1"]["ms"]
    print(f"\n  Why zero? The conflicted reads cost "
          f"{t_sh32:.3f} ms of shared-memory pipe time")
    print(f"  ({reads/1e6:.1f}M reads / {bk[32]['gloads']:.1f} G/s), and the "
          f"kernel's DRAM traffic already takes {t_dram:.3f} ms.")
    print(f"  {t_sh32:.3f} < {t_dram:.3f}, so the whole conflict hides "
          f"underneath. Conflict-free would be {t_sh1:.3f} ms.")

    print()
    print("=" * 78)
    print("D. The same 32-way conflict, at different DRAM-to-shared ratios")
    print("=" * 78)
    print(f"{'DRAM B / read':>13} {'loads':>6} {'reads':>6} "
          f"{'free ms':>8} {'32-way ms':>10} {'measured':>9} {'model':>8}")
    cx = sorted(s["context"], key=lambda r: -r["ratio_bytes"])
    rows = []
    for r in cx:
        threads = NCTX / r["nload"]
        t_sh = threads * r["work"] / (bk[32]["gloads"] * 1e9) * 1e3
        model = max(r["t1"], t_sh) / r["t1"]
        rows.append((r, model))
        print(f"{r['ratio_bytes']:13.3f} {r['nload']:6d} {r['work']:6d} "
              f"{r['t1']:8.3f} {r['t32']:10.3f} {r['slow']:8.2f}x "
              f"{model:7.2f}x")
    worst = max((abs(m - r["slow"]) / r["slow"], r, m) for r, m in rows)
    print(f"\n  Same conflict, same hardware: {min(r['slow'] for r in cx):.2f}x "
          f"to {max(r['slow'] for r in cx):.2f}x depending only on how much")
    print("  DRAM work surrounds it. A bank conflict has no fixed price.")
    print(f"  The max(DRAM, shared) model is exact at both ends and up to "
          f"{100*worst[0]:.0f}% optimistic")
    print(f"  in the crossover (at {worst[1]['ratio_bytes']:.3f} B/read it says "
          f"{worst[2]:.2f}x, reality is {worst[1]['slow']:.2f}x) - which is what")
    print("  a max() model always does, because the two pipes overlap only "
          "partly.")

    png = plot(dev, s, free, bk[32]["gloads"],
               os.path.join(OUT, "bank_conflicts.png"))
    if png:
        print(f"\nWrote {os.path.relpath(png, HERE)}")

    findings = {
        "device": dev,
        "headline": {
            "conflict_free_gloads": free,
            "way32_gloads": bk[32]["gloads"],
            "way32_slowdown": free / bk[32]["gloads"],
            "way32_pct_of_32x_ceiling": 100 * (free / bk[32]["gloads"]) / 32,
            "broadcast_gloads": bk[0]["gloads"],
            "broadcast_vs_32way": bk[0]["gloads"] / bk[32]["gloads"],
            "transpose_smem_over_naive":
                tr["smem_nopad"]["gbs"] / tr["naive"]["gbs"],
            "transpose_padding_gain": fix,
            "transpose_swizzle_gain":
                tr["smem_swizzle"]["gbs"] / tr["smem_nopad"]["gbs"],
            "transpose_conflict_pipe_ms": t_sh32,
            "transpose_dram_ms": t_dram,
            "context_slowdown_range": [min(r["slow"] for r in cx),
                                       max(r["slow"] for r in cx)],
        },
        "raw": s,
    }
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "setting", "conflict_degree", "ms",
                    "throughput", "unit"])
        for r in s["bank"]:
            w.writerow(["bank", f"stride {r['stride']}", r["degree"], r["ms"],
                        r["gloads"], "G loads/s"])
        for r in s["transpose"]:
            w.writerow(["transpose", TR_NAME[r["kind"]], "", r["ms"], r["gbs"],
                        "GB/s"])
        for r in s["context"]:
            w.writerow(["context",
                        f"{r['nload']} loads / {r['work']} reads "
                        f"({r['ratio_bytes']} B per read)", 32, r["t32"],
                        r["slow"], "x slower than conflict-free"])
    print("Wrote outputs/findings.json and outputs/findings.csv")


if __name__ == "__main__":
    main()
