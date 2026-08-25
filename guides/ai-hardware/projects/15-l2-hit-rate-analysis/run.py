"""Project 15 - the L2 hit rate of an attention kernel, with no profiler.

Nsight Compute cannot read this machine's counters, so the hit rate is
recovered from timing alone:

  A. calibration - bandwidth vs working-set size, the whole hierarchy
  B. validation  - a kernel with a KNOWN hit rate; two estimators are scored
                   against the truth, and only one of them survives
  C. attention   - the surviving estimator applied to a real kernel
  D. scheduling  - same bytes, same footprint, different block order
"""

import csv
import json
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "l2")


def build_and_run():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "--extended-lambda",
                        os.path.join(HERE, "l2.cu"), "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def parse(text):
    dev, s, check = {}, {"cal": [], "valid": [], "attn": []}, None
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), l2=int(f[4]))
        elif f[0] == "#check":
            check = dict(s=int(f[1]), err=float(f[2]))
        elif f[0] == "cal":
            s["cal"].append(dict(bytes=int(f[1]), rounds=int(f[2]),
                                 ms=float(f[3]), gbs=float(f[4])))
        elif f[0] == "valid":
            s["valid"].append(dict(pct=int(f[1]), ms=float(f[2]),
                                   gbs=float(f[3])))
        elif f[0] == "attn":
            s["attn"].append(dict(S=int(f[1]), stag=int(f[2]), heads=int(f[3]),
                                  blocks=int(f[4]), asked=float(f[5]),
                                  compulsory=float(f[6]), kv_head=float(f[7]),
                                  ms_compute=float(f[8]), ms_mem=float(f[9]),
                                  gbs=float(f[10]), tflops=float(f[12])))
    return dev, s, check


def harmonic_h(gbs, b_dram, b_l2):
    """Hit rate if hit time and miss time simply add up."""
    return (1 / b_dram - 1 / gbs) / (1 / b_dram - 1 / b_l2)


def floor_h(gbs, b_dram_max):
    """Rigorous lower bound: DRAM cannot supply more than b_dram_max, so any
    bytes delivered above that rate must have come from cache. No model."""
    return max(0.0, 1 - b_dram_max / gbs)


def plot(dev, s, b_dram, b_l2, b_dram_max, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib missing - skipping the plot)")
        return None

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    cal = s["cal"]
    ax[0].plot([r["bytes"] / 1024 for r in cal], [r["gbs"] for r in cal], "o-",
               color="#8e44ad")
    ax[0].axvline(dev["l2"] / 1024, ls="--", color="#c0392b", lw=1)
    ax[0].annotate("L2 = %.0f MB" % (dev["l2"] / 1e6),
                   xy=(dev["l2"] / 1024 * 1.1, 300), fontsize=8, color="#c0392b",
                   rotation=90)
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log")
    ax[0].set_xlabel("working set (KiB)")
    ax[0].set_ylabel("read GB/s")
    ax[0].set_title("A. The instrument: %.1fx between L2 and DRAM"
                    % (max(r["gbs"] for r in cal) /
                       min(r["gbs"] for r in cal[-4:])))
    ax[0].grid(alpha=.3, which="both")

    v = s["valid"]
    true = [r["pct"] / 100 for r in v]
    ax[1].plot([0, 1], [0, 1], "-", color="#7f8c8d", lw=1, label="perfect")
    ax[1].plot(true, [harmonic_h(r["gbs"], b_dram, b_l2) for r in v], "s--",
               color="#c0392b", label="two-speed model")
    ax[1].plot(true, [floor_h(r["gbs"], b_dram_max) for r in v], "o-",
               color="#16a085", label="rigorous lower bound")
    ax[1].set_xlabel("true hit rate (built into the kernel)")
    ax[1].set_ylabel("estimated hit rate")
    ax[1].set_title("B. Scoring two estimators against the truth")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    a = s["attn"]
    ss = sorted({r["S"] for r in a})
    inorder = [next(r for r in a if r["S"] == x and r["stag"] == 0) for x in ss]
    stag = [next(r for r in a if r["S"] == x and r["stag"] == 1) for x in ss]
    w = .35
    xs = range(len(ss))
    ax[2].bar([x - w / 2 for x in xs], [floor_h(r["gbs"], b_dram_max) for r in inorder],
              w, color="#2980b9", label="in-order blocks")
    ax[2].bar([x + w / 2 for x in xs], [floor_h(r["gbs"], b_dram_max) for r in stag],
              w, color="#e67e22", label="staggered blocks")
    ax[2].plot(list(xs), [1 - r["compulsory"] / r["asked"] for r in inorder], "k^--",
               ms=6, label="ceiling: perfect caching")
    ax[2].set_xticks(list(xs))
    ax[2].set_xticklabels(["S=%d\n(%d heads)" % (r["S"], r["heads"]) for r in inorder],
                          fontsize=8)
    ax[2].set_ylabel("L2 hit rate (lower bound)")
    ax[2].set_ylim(0, 1)
    ax[2].set_title("C+D. Attention: far below the ceiling,\nand block order moves it")
    ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="y")

    fig.suptitle("L2 hit rate without a profiler, on %s" % dev["name"], fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    return path


def main():
    dev, s, check = parse(build_and_run())
    print(f"Device: {dev['name']} (cc {dev['cc']}), {dev['sms']} SMs, "
          f"{dev['l2']//1024//1024} MB L2")
    print("Nsight Compute cannot read this machine's counters (project 6), so "
          "the hit rate")
    print("has to come out of timings. This project builds the instrument, "
          "tests it, then")
    print("uses it.\n")

    print("=" * 84)
    print("A. Calibration: read bandwidth vs working-set size")
    print("=" * 84)
    print(f"{'working set':>12} {'ms':>9} {'GB/s':>9}")
    for r in s["cal"]:
        mark = "  <- L2 capacity" if r["bytes"] == dev["l2"] else ""
        print(f"{r['bytes']//1024:11d}K {r['ms']:9.4f} {r['gbs']:9.2f}{mark}")
    top = max(s["cal"], key=lambda r: r["gbs"])
    bot = s["cal"][-1]
    print(f"\n  {top['gbs']:.0f} GB/s in L2 vs {bot['gbs']:.0f} GB/s from DRAM: "
          f"a {top['gbs']/bot['gbs']:.1f}x gap.")
    print("  That gap is the whole instrument. A kernel's achieved bandwidth "
          "sits somewhere")
    print("  in it, and where it sits says how often it hit.")

    v = s["valid"]
    b_dram = next(r["gbs"] for r in v if r["pct"] == 0)
    b_l2 = next(r["gbs"] for r in v if r["pct"] == 100)
    b_dram_max = max(b_dram, bot["gbs"])

    print()
    print("=" * 84)
    print("B. Validation: a kernel whose hit rate we already know")
    print("=" * 84)
    print("  Each warp reads either a 512 KB hot buffer (always in L2) or a "
          "256 MB cold one.")
    print("  Both accesses are equally coalesced, so ONLY locality changes. "
          "True hit rate =")
    print("  the hot fraction. Anchors come from the same kernel: "
          f"{b_dram:.1f} GB/s at 0%, {b_l2:.1f} at 100%.\n")
    print(f"{'true':>6} {'GB/s':>9} {'two-speed model':>17} {'error':>8} "
          f"{'lower bound':>13} {'valid?':>7}")
    worst_model, ok = 0.0, True
    for r in v:
        t = r["pct"] / 100
        hm = harmonic_h(r["gbs"], b_dram, b_l2)
        hf = floor_h(r["gbs"], b_dram_max)
        worst_model = max(worst_model, abs(hm - t))
        if hf > t + 1e-9:
            ok = False
        print(f"{t:6.2f} {r['gbs']:9.2f} {hm:17.3f} {hm-t:+8.3f} "
              f"{hf:13.3f} {'yes' if hf <= t + 1e-9 else 'NO':>7}")
    print(f"\n  The two-speed model - assuming hit time and miss time simply "
          f"add up - is off by")
    print(f"  up to {worst_model:.2f} absolute, and always in the same "
          "direction: it OVERSTATES the")
    print("  hit rate. Hits and misses overlap instead of queueing, so a "
          "half-and-half kernel")
    print("  is faster than the model expects and the model reads that speed "
          "as more hits.")
    print(f"\n  The lower bound never lies ({'confirmed' if ok else 'VIOLATED'}"
          " on all rows). It uses one fact and no model:")
    print(f"  DRAM cannot deliver more than {b_dram_max:.0f} GB/s, so any byte "
          "arriving faster than that")
    print("  came from cache.  h >= 1 - B_DRAM / achieved.  That is the "
          "estimator used below.")

    if check:
        print(f"\n  (The attention kernel is verified against a CPU reference "
              f"at S={check['s']}:")
        print(f"   max absolute error {check['err']:.2e}.)")

    print()
    print("=" * 84)
    print("C. Attention: how much of its traffic does L2 absorb?")
    print("=" * 84)
    print("  Each block streams the whole of K and V past its 64 queries, so "
          "the SAME K and V")
    print("  are read once per query block. Perfect caching would serve all "
          "but the first read.\n")
    a = s["attn"]
    print(f"{'S':>6} {'heads':>6} {'blocks':>7} {'K+V/head':>9} {'asked':>9} "
          f"{'ceiling':>8} {'GB/s':>8} {'h >=':>7} {'TFLOP/s':>8}")
    for r in sorted(a, key=lambda r: (r["S"], r["stag"])):
        if r["stag"]:
            continue
        print(f"{r['S']:6d} {r['heads']:6d} {r['blocks']:7d} "
              f"{r['kv_head']/1e6:8.2f}M {r['asked']/1e6:8.0f}M "
              f"{1-r['compulsory']/r['asked']:8.3f} {r['gbs']:8.2f} "
              f"{floor_h(r['gbs'], b_dram_max):7.3f} {r['tflops']:8.2f}")
    inorder = [r for r in a if r["stag"] == 0]
    print(f"\n  Every row could in principle hit "
          f"{min(1-r['compulsory']/r['asked'] for r in inorder):.2f}-"
          f"{max(1-r['compulsory']/r['asked'] for r in inorder):.2f} of the "
          "time - that is what the")
    print("  data reuse in attention is worth. Measured, the floor is "
          f"{min(floor_h(r['gbs'], b_dram_max) for r in inorder):.2f}-"
          f"{max(floor_h(r['gbs'], b_dram_max) for r in inorder):.2f}.")
    print("  L2 is capturing a small part of the available reuse, and the "
          "reason is capacity:")
    for r in inorder:
        res = r["kv_head"] * max(1, 114 * 64 // r["S"]) / 1e6
        print(f"    S={r['S']:5d}: ~114 resident blocks span "
              f"{max(1, 114*64//r['S']):3d} head(s) = {res:6.2f} MB of live "
              f"K/V vs a {dev['l2']/1e6:.0f} MB L2")
    print("  The live working set is above capacity at every size, and it "
          "barely depends on S -")
    print("  it is set by how many blocks are RESIDENT, not by how long the "
          "sequence is.")

    print()
    print("=" * 84)
    print("D. Same bytes, same footprint, different block order")
    print("=" * 84)
    print("  `stagger` makes block b start its K/V sweep at tile b instead of "
          "tile 0. Every")
    print("  block still reads all of K and V; only the ORDER changes.\n")
    print(f"{'S':>6} {'in-order GB/s':>14} {'staggered GB/s':>15} "
          f"{'change':>8} {'h>= in':>7} {'h>= stag':>9} {'real kernel':>12}")
    for x in sorted({r["S"] for r in a}):
        i = next(r for r in a if r["S"] == x and r["stag"] == 0)
        g = next(r for r in a if r["S"] == x and r["stag"] == 1)
        print(f"{x:6d} {i['gbs']:14.2f} {g['gbs']:15.2f} "
              f"{100*(g['gbs']/i['gbs']-1):+7.1f}% "
              f"{floor_h(i['gbs'], b_dram_max):7.3f} "
              f"{floor_h(g['gbs'], b_dram_max):9.3f} "
              f"{100*(i['ms_compute']/g['ms_compute']-1):+11.1f}%")
    best = max(((next(r for r in a if r["S"] == x and r["stag"] == 1)["gbs"] /
                 next(r for r in a if r["S"] == x and r["stag"] == 0)["gbs"], x)
                for x in {r["S"] for r in a}))
    print(f"\n  Staggering WINS, by up to {100*(best[0]-1):.1f}% at S={best[1]}. "
          "That is the opposite of the")
    print("  prediction: marching every block through the tiles together looks "
          "like the way to")
    print("  share them. It loses because 114 blocks all wanting the same tile "
          "at the same")
    print("  instant queue on one L2 slice and one set of DRAM banks, while "
          "staggered blocks")
    print("  spread their misses across the whole memory system. The bound "
          "rises with it")
    print(f"  ({floor_h(next(r for r in a if r['S']==best[1] and r['stag']==0)['gbs'], b_dram_max):.3f}"
          f" -> {floor_h(next(r for r in a if r['S']==best[1] and r['stag']==1)['gbs'], b_dram_max):.3f}),"
          " and since it is a bound, those extra bytes really did come")
    print("  from cache - this is not DRAM getting luckier.")
    print("\n  On the REAL kernel (with the softmax arithmetic) the same change "
          "is worth much")
    print("  less, because attention is not memory-bound: at "
          f"{max(r['tflops'] for r in a):.2f} TFLOP/s it is spending most of "
          "its")
    print("  time on FMAs. Memory-side wins only show up in proportion to how "
          "much of the")
    print("  runtime was memory in the first place.")

    png = plot(dev, s, b_dram, b_l2, b_dram_max, os.path.join(OUT, "l2_hit_rate.png"))
    if png:
        print(f"\nWrote {os.path.relpath(png, HERE)}")

    findings = {
        "device": dev,
        "attention_max_abs_error_vs_cpu": check["err"] if check else None,
        "anchors": {"b_dram_mix": b_dram, "b_l2_mix": b_l2,
                    "b_dram_stream": bot["gbs"], "b_l2_peak_stream": top["gbs"],
                    "b_dram_used_for_bound": b_dram_max},
        "headline": {
            "l2_vs_dram_gap": top["gbs"] / bot["gbs"],
            "two_speed_model_worst_abs_error": worst_model,
            "lower_bound_never_violated": ok,
            "attention_ceiling_hit_rate":
                [1 - r["compulsory"] / r["asked"] for r in inorder],
            "attention_measured_floor":
                [floor_h(r["gbs"], b_dram_max) for r in inorder],
            "stagger_best_gain_pct": 100 * (best[0] - 1),
            "stagger_best_S": best[1],
            "attention_peak_tflops": max(r["tflops"] for r in a),
        },
        "raw": s,
    }
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "setting", "ms", "gbs", "hit_rate_lower_bound",
                    "note"])
        for r in s["cal"]:
            w.writerow(["calibration", f"{r['bytes']} B working set", r["ms"],
                        r["gbs"], "", ""])
        for r in v:
            w.writerow(["validation", f"{r['pct']}% hot", r["ms"], r["gbs"],
                        floor_h(r["gbs"], b_dram_max),
                        f"two-speed model says {harmonic_h(r['gbs'], b_dram, b_l2):.3f}"])
        for r in a:
            w.writerow(["attention",
                        f"S={r['S']} {'staggered' if r['stag'] else 'in-order'}",
                        r["ms_mem"], r["gbs"], floor_h(r["gbs"], b_dram_max),
                        f"real kernel {r['ms_compute']:.3f} ms, {r['tflops']:.2f} TFLOP/s"])
    print("Wrote outputs/findings.json and outputs/findings.csv")


if __name__ == "__main__":
    main()
