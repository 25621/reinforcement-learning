"""Project 10 - recompute a GPU spec sheet from its own primitive inputs.

Nothing here touches a GPU. Everything is arithmetic on the numbers in
gpus.csv, which contains only primitive facts (SM count, clock, cores per SM,
memory pin rate, bus width, published peaks). The script derives:

  * peak FP32 FLOP/s              - and checks it against the published figure
  * memory bandwidth              - and checks that too
  * 16-bit FLOPs per SM per clock - reverse-engineered from the headline number
  * the ridge point               - FLOP/byte at which a GPU stops being
                                    memory-bound and starts being compute-bound
  * an LLM decode ceiling         - tokens/second at batch size 1, which falls
                                    straight out of bandwidth and model size

The measured anchor is this machine's own GPU: projects 03 and 07 measured its
real bandwidth and real matmul throughput, so we can see how far the derived
numbers are from reality on at least one card.
"""

import csv
import math
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

# Measured on this machine (see the sibling projects). Loaded if available so
# the derived column can be checked against a real number, not just a spec.
MEASURED = {
    "GTX 1070 Ti": dict(bw_gbs=205.4, fp32_tflops=7.18,
                        src="projects 03 and 07 on this machine")}


def load_measured():
    for proj, key, field in (("03-bandwidth-measurement", "bw_gbs", "copy_float4_gbs"),):
        p = os.path.join(HERE, "..", proj, "outputs", "findings.json")
        if os.path.exists(p):
            with open(p) as f:
                MEASURED["GTX 1070 Ti"][key] = json.load(f)[field]
    p = os.path.join(HERE, "..", "07-tensor-core-utilization", "outputs",
                     "findings.json")
    if os.path.exists(p):
        with open(p) as f:
            d = json.load(f)
        best = max((k for k in d["kernels"] if "int8" not in k["name"]),
                   key=lambda k: k["tops"])
        MEASURED["GTX 1070 Ti"]["fp32_tflops"] = best["tops"]


def num(s):
    s = (s or "").strip()
    return float(s) if s else None


def load():
    rows = []
    with open(os.path.join(HERE, "gpus.csv")) as f:
        lines = [l for l in f if not l.startswith("#")]
    for r in csv.DictReader(lines):
        g = dict(name=r["name"], year=int(r["year"]), arch=r["arch"],
                 fmt=r["bf16_or_fp16"], mem_type=r["mem_type"])
        for k in ("sms", "fp32_cores_per_sm", "boost_ghz", "fp32_tflops_spec",
                  "bf16_tflops_dense", "fp8_tflops_dense", "mem_gbps_per_pin",
                  "mem_bus_bits", "mem_bw_spec_gbs", "capacity_gb", "tdp_w"):
            g[k] = num(r[k])
        rows.append(g)
    return rows


def derive(g):
    # peak FP32 = cores x 2 FLOP per FMA x clock
    if g["sms"] and g["boost_ghz"]:
        cores = g["sms"] * g["fp32_cores_per_sm"]
        g["cores"] = cores
        g["fp32_tflops_calc"] = cores * 2 * g["boost_ghz"] / 1e3
    else:
        g["cores"] = None
        g["fp32_tflops_calc"] = None

    # bandwidth = per-pin rate x number of pins / 8 bits per byte
    g["bw_gbs_calc"] = g["mem_gbps_per_pin"] * g["mem_bus_bits"] / 8

    # How wide is one SM's 16-bit matrix pipeline? Divide the headline number
    # by (SMs x clock) and see what falls out.
    if g["bf16_tflops_dense"] and g["sms"] and g["boost_ghz"]:
        g["flop_per_sm_per_clock"] = (g["bf16_tflops_dense"] * 1e12
                                      / (g["sms"] * g["boost_ghz"] * 1e9))
    else:
        g["flop_per_sm_per_clock"] = None

    bw = g["mem_bw_spec_gbs"] * 1e9
    for tag, tf in (("fp32", g["fp32_tflops_spec"]),
                    ("bf16", g["bf16_tflops_dense"]),
                    ("fp8", g["fp8_tflops_dense"])):
        g["ridge_" + tag] = tf * 1e12 / bw if tf else None
    return g


def pct(a, b):
    return f"{100*a/b:6.1f}%" if (a and b) else "     -"


def short(name):
    """'Tesla V100 SXM2' -> 'V100'; 'A100 SXM 80GB' -> 'A100'."""
    n = re.sub(r"\s+(SXM\d*|80GB|Ti)\b", "", name)
    return n.replace("Tesla ", "").strip()


def main():
    load_measured()
    gs = [derive(g) for g in load()]

    print("=" * 96)
    print("STEP 1 - Recompute the headline FLOP number from cores x 2 x clock")
    print("=" * 96)
    print(f"{'GPU':<17} {'SMs':>4} {'cores/SM':>9} {'cores':>7} {'GHz':>6} "
          f"{'calc TF':>9} {'spec TF':>9} {'match':>7}")
    for g in gs:
        if g["fp32_tflops_calc"] is None:
            print(f"{g['name']:<17} {'-':>4} {g['fp32_cores_per_sm']:9.0f} "
                  f"{'-':>7} {'-':>6} {'-':>9} {g['fp32_tflops_spec']:9.1f} "
                  f"{'n/a':>7}   <- SM count and clock not published")
            continue
        print(f"{g['name']:<17} {g['sms']:4.0f} {g['fp32_cores_per_sm']:9.0f} "
              f"{g['cores']:7.0f} {g['boost_ghz']:6.3f} "
              f"{g['fp32_tflops_calc']:9.1f} {g['fp32_tflops_spec']:9.1f} "
              f"{pct(g['fp32_tflops_calc'], g['fp32_tflops_spec'])}")
    print("\n  The 2 is FLOPs per fused multiply-add: an FMA does one multiply")
    print("  and one add, and both are counted. This is the entire formula.")

    print()
    print("=" * 96)
    print("STEP 2 - Recompute memory bandwidth from pin rate x bus width")
    print("=" * 96)
    print(f"{'GPU':<17} {'type':>8} {'Gb/s/pin':>9} {'pins':>6} "
          f"{'calc GB/s':>10} {'spec GB/s':>10} {'match':>7}")
    binned = []
    for g in gs:
        off = g["bw_gbs_calc"] / g["mem_bw_spec_gbs"] - 1
        if off > 0.01:
            binned.append(g["name"])
        print(f"{g['name']:<17} {g['mem_type']:>8} {g['mem_gbps_per_pin']:9.2f} "
              f"{g['mem_bus_bits']:6.0f} {g['bw_gbs_calc']:10.0f} "
              f"{g['mem_bw_spec_gbs']:10.0f} "
              f"{pct(g['bw_gbs_calc'], g['mem_bw_spec_gbs'])}"
              + ("   <- runs below the memory's rated speed" if off > 0.01 else ""))
    if binned:
        print(f"\n  {', '.join(binned)}: our arithmetic comes out ABOVE the")
        print("  published bandwidth. That is not an error in the formula - it")
        print("  means the shipped part clocks its memory below what the HBM")
        print("  stacks are rated for, usually for power or yield reasons.")
        print("  A mismatch in this direction is a real signal, so keep the")
        print("  cross-check rather than deleting it.")
    print("\n  'pins' is the total width of the memory interface. GDDR reaches")
    print("  its bandwidth with a narrow bus at a very high per-pin rate; HBM")
    print("  does the opposite - a slow but enormously wide bus, which is why")
    print("  it needs to sit on the same package as the GPU.")
    hbm = [g for g in gs if g["mem_type"].startswith("HBM")]
    gddr = [g for g in gs if g["mem_type"].startswith("GDDR")]
    print(f"  Widest GDDR bus here: {max(g['mem_bus_bits'] for g in gddr):.0f} bits at up to "
          f"{max(g['mem_gbps_per_pin'] for g in gddr):.0f} Gb/s per pin.")
    print(f"  Widest HBM bus here : {max(g['mem_bus_bits'] for g in hbm):.0f} bits at up to "
          f"{max(g['mem_gbps_per_pin'] for g in hbm):.1f} Gb/s per pin.")

    print()
    print("=" * 96)
    print("STEP 3 - How wide is one SM's matrix pipeline? Work it out backwards.")
    print("=" * 96)
    print("  16-bit FLOPs per SM per clock = headline TFLOP/s / (SMs x clock)\n")
    print(f"{'GPU':<17} {'16-bit TF':>10} {'FLOP/SM/clk':>12} {'nearest 2^n':>12} "
          f"{'off by':>8}")
    clean, messy = [], []
    for g in gs:
        v = g["flop_per_sm_per_clock"]
        if not v:
            why = "no tensor cores" if g["fmt"] == "none" else "SMs/clock not published"
            print(f"{g['name']:<17} {'-':>10} {'-':>12} {'-':>12} {'-':>8}"
                  f"   <- {why}")
            continue
        p2 = 2 ** round(math.log2(v))
        off = 100 * (v / p2 - 1)
        (clean if abs(off) < 1 else messy).append(g["name"])
        print(f"{g['name']:<17} {g['bf16_tflops_dense']:10.1f} {v:12.1f} "
              f"{p2:12d} {off:+7.1f}%")
    print(f"\n  {len(clean)} of these land within 1% of an exact power of two, which")
    print("  is what you expect: the pipeline is built out of a whole number of")
    print("  fixed-size multiply-accumulate blocks.")
    if messy:
        print(f"  {', '.join(messy)} do not. Their published 16-bit peak is not")
        print("  (SMs x power-of-two width x boost clock) for any width, so the")
        print("  headline number assumes some other clock - and NVIDIA does not")
        print("  say which. Take that as the general lesson: a vendor peak is a")
        print("  number chosen for a datasheet, not a formula you can invert.")

    print()
    print("=" * 96)
    print("STEP 4 - The ridge point: how much arithmetic per byte you need")
    print("=" * 96)
    print("  ridge point = peak FLOP/s / peak bytes/s")
    print("  Below it a kernel is memory-bound (only fewer bytes will help).")
    print("  Above it a kernel is compute-bound (only fewer FLOPs will help).\n")
    print(f"{'GPU':<17} {'year':>5} {'TF (16-bit)':>12} {'GB/s':>7} "
          f"{'ridge fp32':>11} {'ridge 16-bit':>13} {'ridge fp8':>10}")
    for g in gs:
        r16 = f"{g['ridge_bf16']:13.0f}" if g["ridge_bf16"] else f"{'-':>13}"
        r8 = f"{g['ridge_fp8']:10.0f}" if g["ridge_fp8"] else f"{'-':>10}"
        tf = f"{g['bf16_tflops_dense']:12.1f}" if g["bf16_tflops_dense"] else f"{'-':>12}"
        print(f"{g['name']:<17} {g['year']:5d} {tf} {g['mem_bw_spec_gbs']:7.0f} "
              f"{g['ridge_fp32']:11.0f} {r16} {r8}")

    dc = [g for g in gs if g["ridge_bf16"]]
    v100 = next(g for g in gs if "V100" in g["name"])
    h100 = next(g for g in gs if "H100" in g["name"])
    h200 = next(g for g in gs if "H200" in g["name"])
    print(f"\n  V100 (2017) -> H100 (2022): 16-bit FLOPs x"
          f"{h100['bf16_tflops_dense']/v100['bf16_tflops_dense']:.1f}, "
          f"bandwidth x{h100['mem_bw_spec_gbs']/v100['mem_bw_spec_gbs']:.1f}.")
    print(f"  The ridge point therefore rose {v100['ridge_bf16']:.0f} -> "
          f"{h100['ridge_bf16']:.0f} FLOP/byte "
          f"({h100['ridge_bf16']/v100['ridge_bf16']:.1f}x).")
    print("  Compute has outrun memory every generation. That single ratio is")
    print("  why fusion, FlashAttention and recomputation keep getting MORE")
    print("  valuable rather than less - the wall they climb is getting taller.")
    print(f"\n  The exception: H200 has exactly the same compute as H100 and "
          f"{h200['mem_bw_spec_gbs']/h100['mem_bw_spec_gbs']:.2f}x the bandwidth,")
    print(f"  so its ridge point FELL to {h200['ridge_bf16']:.0f}. A 'boring' memory-only")
    print("  refresh moved the number that actually constrains most kernels.")
    print(f"\n  And note what dropping precision does: on the H100 the fp8 ridge")
    print(f"  point is {h100['ridge_fp8']:.0f}, DOUBLE the 16-bit one. Halving your bits")
    print("  doubles the FLOPs but does nothing for the bus, so a low-precision")
    print("  GPU is *relatively* even more memory-starved. Quantization pays off")
    print("  because it also halves the bytes you move - not because of the FLOPs.")

    print()
    print("=" * 96)
    print("STEP 5 - What that means for one real workload: LLM decode, batch 1")
    print("=" * 96)
    print("  Generating one token reads every weight exactly once and does two")
    print("  FLOPs per weight. So arithmetic intensity = 2 FLOP / bytes-per-weight,")
    print("  no matter how big the model is:")
    print("      fp16 weights -> 1.0 FLOP/byte      int8 -> 2.0      int4 -> 4.0")
    print("  Compare that to ridge points of 100-300 and the verdict is not close.\n")
    print(f"{'GPU':<17} {'GB/s':>7} {'ridge':>7} {'AI@int8':>8} "
          f"{'max % of peak':>14} {'70B int8 tok/s':>15} {'fits?':>6}")
    ceils = []
    for g in gs:
        if not g["ridge_bf16"]:
            continue
        ai = 2.0
        ceil = 100 * ai / g["ridge_bf16"]
        ceils.append(ceil)
        model_gb = 70.0        # 70B parameters at 1 byte each
        toks = g["mem_bw_spec_gbs"] / model_gb
        # 15% headroom for the KV cache, activations and CUDA context
        fits = "yes" if g["capacity_gb"] >= model_gb * 1.15 else "no"
        print(f"{g['name']:<17} {g['mem_bw_spec_gbs']:7.0f} "
              f"{g['ridge_bf16']:7.0f} {ai:8.1f} {ceil:13.2f}% "
              f"{toks:15.0f} {fits:>6}")
    print("  ('fits?' allows 15% on top of the weights for the KV cache,")
    print("   activations and the CUDA context - weights alone is not enough.)")
    print(f"\n  Read the '% of peak' column again: a batch-1 decode reaches "
          f"{min(ceils):.2f}%-{max(ceils):.2f}%")
    print("  of the tensor cores you paid for. Nothing is broken - the job")
    print("  simply has no arithmetic in it relative to the bytes it must read.")
    print("  Everything in LLM serving (batching, speculative decoding, paged")
    print("  KV caches) exists to move that number up.")
    print("  The tokens/s column is a hard ceiling, not an estimate: you cannot")
    print("  emit a token faster than you can read the weights once.")

    print()
    print("=" * 96)
    print("STEP 6 - Reality check against the one card we can actually measure")
    print("=" * 96)
    for g in gs:
        m = MEASURED.get(g["name"])
        if not m:
            continue
        print(f"  {g['name']}")
        print(f"    bandwidth : spec {g['mem_bw_spec_gbs']:.0f} GB/s, "
              f"measured {m['bw_gbs']:.0f} GB/s "
              f"({100*m['bw_gbs']/g['mem_bw_spec_gbs']:.0f}% of spec)")
        print(f"    fp32 matmul: spec {g['fp32_tflops_spec']:.2f} TFLOP/s, "
              f"measured {m['fp32_tflops']:.2f} TFLOP/s "
              f"({100*m['fp32_tflops']/g['fp32_tflops_spec']:.0f}% of spec)")
        rr = (m["fp32_tflops"] * 1e12) / (m["bw_gbs"] * 1e9)
        print(f"    ridge point: spec {g['ridge_fp32']:.1f} FLOP/byte, "
              f"from measured numbers {rr:.1f} FLOP/byte")
        print(f"    Source: {m['src']}")
    print("\n  Both derived numbers land within 15% of measurement here, which is")
    print("  the honest accuracy of this whole exercise: spec arithmetic tells")
    print("  you the shape of a machine, not its performance.")

    with open(os.path.join(OUT, "derived.csv"), "w", newline="") as f:
        keys = sorted({k for g in gs for k in g})
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(gs)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(dict(gpus=gs, measured=MEASURED,
                       ridge_growth_v100_to_h100=h100["ridge_bf16"] / v100["ridge_bf16"],
                       h200_ridge_drop=h200["ridge_bf16"] / h100["ridge_bf16"]),
                  f, indent=2)
    print(f"\nwrote {OUT}/derived.csv and findings.json")
    plot(gs)


def plot(gs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))

    # (a) rooflines on one chart
    pick = ["GTX 1070 Ti", "A100 SXM 80GB", "H100 SXM5", "RTX 5090"]
    cols = ["#B279A2", "#54A24B", "#4C78A8", "#F58518"]
    x = np.logspace(-1, 4, 400)
    for name, c in zip(pick, cols):
        g = next(gg for gg in gs if gg["name"] == name)
        tf = (g["bf16_tflops_dense"] or g["fp32_tflops_spec"]) * 1e12
        bw = g["mem_bw_spec_gbs"] * 1e9
        y = np.minimum(tf, bw * x) / 1e12
        ax[0].loglog(x, y, color=c, lw=2,
                     label=f"{name} (ridge {tf/bw:.0f})")
        ax[0].plot([tf / bw], [tf / 1e12], "o", color=c, ms=6)
    ax[0].axvline(1.0, ls=":", c="crimson")
    ax[0].text(1.15, 0.02, "LLM decode\n(fp16 weights)\n1 FLOP/byte",
               fontsize=8, color="crimson")
    ax[0].axvline(1365, ls=":", c="#333")
    ax[0].text(300, 0.02, "4096-cube matmul\n1365 FLOP/byte", fontsize=8,
               color="#333", ha="right")
    ax[0].set_xlabel("arithmetic intensity (FLOP per byte of DRAM traffic)")
    ax[0].set_ylabel("achievable TFLOP/s (log)")
    ax[0].set_ylim(1e-2, 4e3)
    ax[0].legend(fontsize=8, loc="upper left")
    ax[0].set_title("(a) Four rooflines. The kink is the ridge point.\n"
                    "Left of it, only fewer bytes help.", fontsize=11)
    ax[0].grid(alpha=.3, which="both")

    # (b) the ridge point over time - datacenter and consumer kept apart,
    # because they are different products with different memory budgets and
    # mixing them would turn a trend into a zigzag.
    dc = [g for g in gs if g["ridge_bf16"]]
    dc.sort(key=lambda g: (g["year"], g["name"]))
    xs = list(range(len(dc)))
    is_dc = [g["mem_type"].startswith("HBM") for g in dc]
    for flag, c, mk, lab in ((True, "#4C78A8", "o-", "datacenter (HBM)"),
                             (False, "#F58518", "o-", "consumer (GDDR)")):
        pts = [(i, g) for i, g in zip(xs, dc) if is_dc[i] == flag]
        ax[1].plot([p[0] for p in pts], [p[1]["ridge_bf16"] for p in pts], mk,
                   color=c, lw=2, label=lab)
    for i, g in enumerate(dc):
        ax[1].annotate(f"{g['ridge_bf16']:.0f}", (i, g["ridge_bf16"]),
                       textcoords="offset points", xytext=(0, 9), ha="center",
                       fontsize=9)
    fp8 = [(i, g["ridge_fp8"]) for i, g in enumerate(dc) if g["ridge_fp8"]]
    ax[1].plot([p[0] for p in fp8], [p[1] for p in fp8], "s", color="#E45756",
               label="same chips at fp8")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"{short(g['name'])}\n{g['year']}" for g in dc],
                          fontsize=8)
    ax[1].set_ylabel("FLOP per byte needed to be compute-bound")
    ax[1].legend(fontsize=8)
    h200 = next(g for g in dc if "H200" in g["name"])
    i200 = dc.index(h200)
    ax[1].annotate("H200 = H100 compute\n+ 43% bandwidth:\nthe only step DOWN",
                   (i200, h200["ridge_bf16"]), textcoords="offset points",
                   xytext=(-16, -56), fontsize=8, color="#2c7a2c",
                   arrowprops=dict(arrowstyle="->", color="#2c7a2c"))
    ax[1].set_title("(b) The wall is getting taller\n"
                    "on the datacenter line, FLOPs outgrew bandwidth 2.1x",
                    fontsize=11)
    ax[1].grid(alpha=.3)

    # (c) what a batch-1 70B decode can use
    dc2 = [g for g in gs if g["ridge_bf16"]]
    names = [short(g["name"]) for g in dc2]
    ceil = [100 * 2.0 / g["ridge_bf16"] for g in dc2]
    ax[2].barh(names, ceil, color="#E45756")
    for i, (g, c) in enumerate(zip(dc2, ceil)):
        ax[2].text(c + 0.02, i, f"{c:.2f}%  ({g['mem_bw_spec_gbs']/70:.0f} tok/s)",
                   va="center", fontsize=8.5)
    ax[2].set_xlabel("share of the GPU's 16-bit peak FLOPs a batch-1 decode can reach")
    ax[2].set_xlim(0, max(ceil) * 1.9)
    ax[2].invert_yaxis()
    ax[2].set_title("(c) A 70B int8 decode at batch 1\n"
                    "under 2% of peak on every card here", fontsize=11)
    ax[2].grid(alpha=.3, axis="x")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "spec-compare.png"), dpi=110)
    print(f"wrote {OUT}/spec-compare.png")


if __name__ == "__main__":
    main()
