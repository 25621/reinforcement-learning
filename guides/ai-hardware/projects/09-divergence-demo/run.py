"""Project 09 - measure what thread divergence actually costs.

Compiles diverge.cu, runs four experiments, and inspects the generated machine
code (SASS) to show *why* each number came out the way it did.

The controls matter more than the headline here. A "divergence demo" that only
shows one slow kernel proves nothing: the slow kernel might be slow for any
reason. Every measurement below is paired with a version that has the same
instruction count and the same branch count, and differs only in whether the
32 threads of a warp agree on which way to go.
"""

import csv
import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "diverge")


def build_and_run():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    r = subprocess.run(["nvcc", "-O3", "-arch=sm_61", "-lineinfo",
                        os.path.join(HERE, "diverge.cu"), "-o", EXE],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("run failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    return r.stdout


def sass_stats():
    """Count branches and predicated instructions in each compiled kernel."""
    r = subprocess.run(["cuobjdump", "-sass", EXE], capture_output=True, text=True)
    if r.returncode != 0:
        return {}
    cur, stats = None, {}
    for line in r.stdout.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            cur = m.group(1)
            stats[cur] = dict(bra=0, pred=0, total=0)
            continue
        if cur is None or "/*" not in line:
            continue
        stats[cur]["total"] += 1
        if re.search(r"\bBRA\b", line):
            stats[cur]["bra"] += 1
        if re.search(r"@!?P\d\s+\w", line):
            stats[cur]["pred"] += 1
    return stats


def parse(text):
    dev, ways, tiny, ragged = {}, [], [], []
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]))
        elif f[0] == "ways":
            ways.append(dict(ways=int(f[1]), mode=int(f[2]), ms=float(f[3]),
                             tflops=float(f[4])))
        elif f[0] == "tiny":
            tiny.append(dict(mode=int(f[1]), ms=float(f[2]), tflops=float(f[3])))
        elif f[0] == "ragged":
            ragged.append(dict(order=f[1], ms=float(f[2]), total_iters=int(f[3]),
                               warp_iters=int(f[4])))
    return dev, ways, tiny, ragged


def main():
    dev, ways, tiny, ragged = parse(build_and_run())
    sass = sass_stats()
    div = {w["ways"]: w for w in ways if w["mode"] == 0}
    uni = {w["ways"]: w for w in ways if w["mode"] == 1}
    ns = sorted(div)

    print(f"Device: {dev['name']} (compute capability {dev['cc']}), "
          f"{dev['sms']} SMs, 32 threads per warp\n")

    print("=" * 78)
    print("1 + 2. The same code, branching per-LANE vs per-WARP")
    print("=" * 78)
    print("Every row runs the identical kernel with the identical number of")
    print("possible paths. Only the thing the branch depends on changes:")
    print("  divergent : sel = lane % ways   -> one warp splits `ways` ways")
    print("  aligned   : sel = warp % ways   -> all 32 lanes agree, every time")
    print("Both layouts send the same FRACTION of all threads down each path.\n")
    print(f"{'paths':>6} {'divergent':>10} {'aligned':>9} {'ratio':>7} "
          f"{'useful TFLOP/s':>15} {'':>3} {'ideal':>6}")
    for n in ns:
        print(f"{n:6d} {div[n]['ms']:9.3f}m {uni[n]['ms']:8.3f}m "
              f"{div[n]['ms']/uni[n]['ms']:6.1f}x "
              f"{div[n]['tflops']:9.2f} vs {uni[n]['tflops']:5.2f} "
              f"{float(n):6.0f}x")
    slow = div[ns[-1]]["ms"] / div[ns[0]]["ms"]
    gap = div[ns[-1]]["ms"] / uni[ns[-1]]["ms"]
    print(f"\n  Going from 1 to {ns[-1]} divergent paths cost {slow:.1f}x "
          f"(the ceiling is {ns[-1]}x - one pass per path).")
    print(f"  Going from 1 to {ns[-1]} WARP-ALIGNED paths cost "
          f"{uni[ns[-1]]['ms']/uni[ns[0]]['ms']:.2f}x.")
    print(f"  Same source, same {ns[-1]} branch targets, same work per thread: "
          f"{gap:.1f}x apart.")
    print("  Divergence is not caused by having branches. It is caused by")
    print("  threads INSIDE ONE WARP disagreeing about which branch to take.")

    d32 = next((k for k in sass if "divergeILi32" in k), None)
    d1 = next((k for k in sass if "divergeILi1E" in k), None)
    if d32 and d1:
        print(f"\n  In the machine code: the {ns[-1]}-path kernel contains "
              f"{sass[d32]['bra']} BRA (branch) instructions, the 1-path kernel "
              f"{sass[d1]['bra']}.")
        print("  Those branches exist in BOTH the divergent and the aligned run -")
        print("  it is the same binary. Only the data decides what they cost.")

    print()
    print("=" * 78)
    print("3. The branch that costs nothing, because it is not a branch")
    print("=" * 78)
    t_uni = next(t for t in tiny if t["mode"] == 0)
    t_div = next(t for t in tiny if t["mode"] == 1)
    tb = next((k for k in sass if "tiny_branch" in k), None)
    print("Body of each arm: a single FMA. Condition: `lane < 16`, which splits")
    print("every warp exactly in half - maximally divergent by any definition.\n")
    print(f"  warp-uniform condition : {t_uni['ms']:.4f} ms")
    print(f"  half-warp condition    : {t_div['ms']:.4f} ms   "
          f"-> {t_div['ms']/t_uni['ms']:.3f}x")
    if tb:
        print(f"\n  SASS for that kernel: {sass[tb]['pred']} predicated instructions "
              f"(`@P0 FFMA`, `@!P0 FFMA`), {sass[tb]['bra']} branches.")
    print("  The compiler did not branch at all. It emitted BOTH arms and tagged")
    print("  each with a predicate register - a per-thread on/off switch. Every")
    print("  thread issues both instructions; the wrong one writes nothing.")
    print("  Cost = 2 instructions instead of 1, not 2 passes over the warp.")
    print("\n  Consequence for your own benchmarks: an `if/else` with short arms")
    print("  may show NO divergence penalty, and you will conclude divergence is")
    print("  a myth. It isn't - the compiler removed the branch. Check the SASS")
    print("  before you conclude anything from a null result here.")

    print()
    print("=" * 78)
    print("4. The version you will actually meet: ragged loop counts")
    print("=" * 78)
    rnd = next(r for r in ragged if r["order"] == "random")
    srt = next(r for r in ragged if r["order"] == "sorted")
    print("Each thread loops a random number of times between 8 and 2007. No")
    print("`if` anywhere - but a loop is a branch, so a warp keeps going until")
    print("its SLOWEST thread is done, with the finished lanes idling.\n")
    print(f"{'layout':>10} {'time':>10} {'iterations asked for':>22} "
          f"{'iterations the warps ran':>26}")
    for r in (rnd, srt):
        print(f"{r['order']:>10} {r['ms']:9.4f}m {r['total_iters']:22,d} "
              f"{r['warp_iters']*32:26,d}")
    print(f"\n  Identical total work. Sorting the trip counts so that similar")
    print(f"  threads share a warp was worth {rnd['ms']/srt['ms']:.2f}x.")
    print(f"  Predicted from the warp maxima: "
          f"{rnd['warp_iters']/srt['warp_iters']:.2f}x. Measured comes out lower")
    print("  because the sorted layout ends with a few very long warps that run")
    print("  on while the rest of the machine has already gone idle.")
    print(f"\n  Wasted lane-cycles in the random layout: "
          f"{100*(1 - rnd['total_iters']/(rnd['warp_iters']*32)):.0f}% - that share")
    print("  of every warp-cycle went to lanes that were switched off.")
    print("  This is exactly why LLM serving sorts requests into length buckets")
    print("  and why Mixture-of-Experts implementations group tokens by expert")
    print("  before the matmul: same work, arranged so warps agree.")

    rows = ([dict(section="ways", **w) for w in ways]
            + [dict(section="tiny", **t) for t in tiny]
            + [dict(section="ragged", **r) for r in ragged])
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(rows)

    findings = dict(
        device=dev,
        divergent_slowdown_x=slow, aligned_slowdown_x=uni[ns[-1]]["ms"] / uni[ns[0]]["ms"],
        divergent_vs_aligned_at_max_x=gap, max_paths=ns[-1],
        predicated_branch_cost_x=t_div["ms"] / t_uni["ms"],
        ragged_sort_speedup_x=rnd["ms"] / srt["ms"],
        ragged_predicted_x=rnd["warp_iters"] / srt["warp_iters"],
        ragged_wasted_pct=100 * (1 - rnd["total_iters"] / (rnd["warp_iters"] * 32)),
        sass={k: v for k, v in sass.items()},
        ways=ways, tiny=tiny, ragged=ragged)
    with open(os.path.join(OUT, "findings.json"), "w") as fh:
        json.dump(findings, fh, indent=2)
    print(f"\nwrote {OUT}/findings.json and findings.csv")
    plot(ns, div, uni, t_uni, t_div, rnd, srt)


def plot(ns, div, uni, t_uni, t_div, rnd, srt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    base = div[ns[0]]["ms"]
    ax[0].plot(ns, ns, "--", color="#B0B0B0", label="perfect serialization (N x)")
    ax[0].plot(ns, [div[n]["ms"] / base for n in ns], "o-", color="#E45756",
               label="divergent  (sel = lane % N)")
    ax[0].plot(ns, [uni[n]["ms"] / base for n in ns], "s-", color="#54A24B",
               label="warp-aligned (sel = warp % N)")
    ax[0].set_xscale("log", base=2); ax[0].set_yscale("log", base=2)
    ax[0].set_xticks(ns); ax[0].set_xticklabels(ns)
    ax[0].set_xlabel("distinct code paths per kernel")
    ax[0].set_ylabel("time, relative to 1 path")
    ax[0].legend(fontsize=8.5, loc="upper left")
    ax[0].annotate(f"{div[ns[-1]]['ms']/uni[ns[-1]]['ms']:.0f}x apart\nsame binary,\n"
                   "same branch count",
                   (ns[-1], (div[ns[-1]]["ms"] / base + uni[ns[-1]]["ms"] / base) / 4),
                   ha="right", fontsize=9)
    ax[0].set_title("(a) Branches are free. Disagreement is not.", fontsize=11)
    ax[0].grid(alpha=.3, which="both")

    # Each bar is a split-vs-uniform RATIO, so the two experiments are directly
    # comparable even though their absolute times differ.
    labels = ["each arm = a 4000-iteration loop\n(compiler must branch)",
              "each arm = one FMA\n(compiler predicates instead)"]
    vals = [div[2]["ms"] / uni[2]["ms"], t_div["ms"] / t_uni["ms"]]
    detail = [f"{uni[2]['ms']:.3f} -> {div[2]['ms']:.3f} ms",
              f"{t_uni['ms']:.3f} -> {t_div['ms']:.3f} ms"]
    ax[1].bar(labels, vals, color=["#E45756", "#4C78A8"], width=.55)
    ax[1].axhline(1.0, ls="--", c="k", lw=1)
    ax[1].text(1.42, 1.03, "no penalty", fontsize=8, ha="right")
    for i, v in enumerate(vals):
        ax[1].text(i, v + 0.04, f"{v:.2f}x", ha="center", fontsize=13)
        ax[1].text(i, 0.12, detail[i], ha="center", fontsize=8.5, color="white")
    ax[1].set_ylabel("cost of splitting a warp in half (x)")
    ax[1].set_ylim(0, max(vals) * 1.3)
    ax[1].tick_params(axis="x", labelsize=8.5)
    ax[1].set_title("(b) Identical condition (`lane < 16`), identical split.\n"
                    "The arm length decides whether it costs anything.",
                    fontsize=11)
    ax[1].grid(alpha=.3, axis="y")

    ax[2].bar(["random\norder", "sorted by\ntrip count"],
              [rnd["ms"], srt["ms"]], color=["#E45756", "#54A24B"], width=.5)
    for i, r in enumerate((rnd, srt)):
        ax[2].text(i, r["ms"] * 1.02, f"{r['ms']:.4f} ms", ha="center", fontsize=9.5)
        waste = 100 * (1 - r["total_iters"] / (r["warp_iters"] * 32))
        ax[2].text(i, r["ms"] * 0.5, f"{waste:.0f}% of lane-cycles\nwasted on\n"
                   "switched-off lanes", ha="center", fontsize=8.5, color="white")
    ax[2].set_ylabel("time (ms)")
    ax[2].set_ylim(0, rnd["ms"] * 1.2)
    ax[2].set_title(f"(c) Ragged loop counts: sorting is worth "
                    f"{rnd['ms']/srt['ms']:.2f}x\nzero change to the total work",
                    fontsize=11)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "divergence.png"), dpi=110)
    print(f"wrote {OUT}/divergence.png")


if __name__ == "__main__":
    main()
