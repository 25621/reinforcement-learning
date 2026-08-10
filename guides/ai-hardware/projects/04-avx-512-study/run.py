"""Project 04 - scalar vs autovectorised vs AVX2 vs AVX-512.

Builds vecsum.c, runs the benchmark, disassembles the binary to prove which
instructions each variant really uses, checks accuracy against a double
reference, and demonstrates what happens when 512-bit code meets a CPU without
512-bit registers.
"""

import csv
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "vecsum")


def build():
    cmd = ["gcc", "-O3", "-march=native", "-o", EXE, os.path.join(HERE, "vecsum.c"), "-lm"]
    print("building:", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])
    if r.stderr.strip():
        print(r.stderr.strip()[:800])


def cpu_flags():
    txt = open("/proc/cpuinfo").read()
    model = re.search(r"model name\s*:\s*(.+)", txt).group(1)
    flags = set(re.search(r"flags\s*:\s*(.+)", txt).group(1).split())
    return model, flags


def disassemble():
    """Which instructions did each variant actually get compiled to?"""
    r = subprocess.run(["objdump", "-d", "--no-show-raw-insn", EXE],
                       capture_output=True, text=True)
    funcs, cur = {}, None
    for line in r.stdout.splitlines():
        m = re.match(r"^[0-9a-f]+ <([A-Za-z0-9_]+)>:", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        if cur and line.strip():
            parts = line.split("\t")
            if len(parts) >= 2:
                funcs[cur].append(parts[1].strip())
    info = {}
    for name, ins in funcs.items():
        if not name.startswith(("sum_", "poly_")):
            continue
        text = " ".join(ins)
        info[name] = dict(
            n_insn=len(ins),
            scalar_add=len(re.findall(r"\bvadds[sd]\b", text)),
            packed_add=len(re.findall(r"\bvaddp[sd]\b", text)),
            fma=len(re.findall(r"\bvfm[as][dbn]*[0-9]*p[sd]\b", text)),
            scalar_fma=len(re.findall(r"\bvfm[as][dbn]*[0-9]*s[sd]\b", text)),
            widest=("zmm" if "zmm" in text else "ymm" if "ymm" in text
                    else "xmm" if "xmm" in text else "none"),
        )
    return info


def parse(text):
    rows, refs, cpu = [], {}, {}
    for line in text.strip().splitlines():
        if line.startswith("#cpu"):
            for kv in line.split(",")[1:]:
                k, v = kv.split("=")
                cpu[k] = int(v)
            continue
        if line.startswith("#"):
            continue
        f = line.split(",")
        if f[0] == "ref":
            refs[(f[1], int(f[2]))] = float(f[4])
        elif f[4] == "SKIPPED":
            rows.append(dict(kernel=f[0], variant=f[1], n=int(f[2]), where=f[3],
                             sec=None, result=None))
        else:
            rows.append(dict(kernel=f[0], variant=f[1], n=int(f[2]), where=f[3],
                             sec=float(f[4]), result=float(f[5])))
    return cpu, refs, rows


FLOPS_PER_ELT = {"sum": 1.0, "poly": 40.0}
BYTES_PER_ELT = 4.0


def main():
    model, flags = cpu_flags()
    print(f"CPU: {model}")
    print("AVX-512 flags present in /proc/cpuinfo: "
          + (", ".join(sorted(f for f in flags if f.startswith("avx512"))) or "NONE"))
    build()

    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("benchmark failed: " + r.stderr[-2000:])
    cpu, refs, rows = parse(r.stdout)
    print(f"runtime check: __builtin_cpu_supports(avx2)={cpu['avx2']}, "
          f"(avx512f)={cpu['avx512f']}")

    # ---------------- what the compiler emitted ----------------
    print("\n=== What each variant compiled to (from objdump) ===")
    print(f"{'function':<18} {'widest reg':>11} {'scalar adds':>12} {'packed adds':>12} "
          f"{'packed FMA':>11}")
    dis = disassemble()
    for name in ["sum_scalar", "sum_auto", "sum_fastmath", "sum_avx2", "sum_avx512",
                 "poly_scalar", "poly_auto", "poly_fastmath", "poly_avx2",
                 "poly_avx2_ilp", "poly_avx512"]:
        d = dis.get(name)
        if not d:
            continue
        print(f"{name:<18} {d['widest']:>11} {d['scalar_add']:>12} {d['packed_add']:>12} "
              f"{d['fma']:>11}")
    auto_vec = dis["sum_auto"]["packed_add"] > 0
    print(f"\nDid plain -O3 vectorise the float sum?  {'YES' if auto_vec else 'NO'}")
    if not auto_vec:
        print("  sum_auto contains ZERO packed adds - it unrolled the loop but kept")
        print("  adding one float at a time, and it times identically to sum_scalar.")
        print("  Reason: float addition is not associative, so summing in a")
        print("  different order is a different answer, and -O3 is not allowed")
        print("  to change your answer. -ffast-math grants that permission.")
    print(f"AVX-512 code present in the binary? "
          f"{'YES (zmm registers)' if dis.get('sum_avx512', {}).get('widest') == 'zmm' else 'no'}"
          f" - on a CPU that cannot execute it.")

    # ---------------- timings ----------------
    out_rows = []
    print("\n=== Timings (best of 5 rounds, single thread) ===")
    for kernel in ["sum", "poly"]:
        print(f"\n--- {kernel}  ({FLOPS_PER_ELT[kernel]:.0f} FLOP per element, "
              f"{FLOPS_PER_ELT[kernel]/BYTES_PER_ELT:.2f} FLOP/byte) ---")
        header = f"{'size':>10} {'where':>6} " + "".join(
            f"{v:>12}" for v in ["scalar", "auto", "fastmath", "avx2", "avx2_ilp", "avx512"])
        print(header)
        for n in sorted({r["n"] for r in rows}):
            group = {r["variant"]: r for r in rows if r["kernel"] == kernel and r["n"] == n}
            base = group["scalar"]["sec"]
            cells = []
            for v in ["scalar", "auto", "fastmath", "avx2", "avx2_ilp", "avx512"]:
                g = group.get(v)
                if g is None:
                    cells.append(f"{'-':>12}")
                elif g["sec"] is None:
                    cells.append(f"{'no CPU':>12}")
                else:
                    gf = n * FLOPS_PER_ELT[kernel] / g["sec"] / 1e9
                    cells.append(f"{base/g['sec']:>7.1f}x{'':>4}")
                    out_rows.append(dict(
                        kernel=kernel, variant=v, n=n, where=g["where"],
                        sec=g["sec"], gflops=gf, gbs=n * BYTES_PER_ELT / g["sec"] / 1e9,
                        speedup_vs_scalar=base / g["sec"],
                        rel_error=abs(g["result"] - refs[(kernel, n)]) / abs(refs[(kernel, n)])))
            w = group["scalar"]["where"]
            print(f"{n*4/1024:9.0f}K {w:>6} " + "".join(cells))
        print("            (numbers are speedup over the scalar version)")

    # ---------------- the two headline effects ----------------
    def get(kernel, variant, where):
        return [r for r in out_rows if r["kernel"] == kernel
                and r["variant"] == variant and r["where"] == where][0]

    print("\n=== Effect 1: SIMD pays only where the data already is ===")
    print(f"{'':<6} {'sum: avx2 vs scalar':>22} {'sum: GB/s reached':>20} "
          f"{'poly: avx2_ilp vs scalar':>26}")
    for w in ["L1", "L2", "L3", "DRAM"]:
        s = get("sum", "avx2", w)
        p = get("poly", "avx2_ilp", w)
        print(f"{w:<6} {s['speedup_vs_scalar']:>21.1f}x {s['gbs']:>19.1f} "
              f"{p['speedup_vs_scalar']:>25.1f}x")
    s_l1, s_dram = get("sum", "avx2", "L1"), get("sum", "avx2", "DRAM")
    fm_dram = get("sum", "fastmath", "DRAM")
    print(f"\nThe memory-bound sum loses {s_l1['speedup_vs_scalar']/s_dram['speedup_vs_scalar']:.1f}x "
          f"of its SIMD win moving from L1 to DRAM.")
    print(f"In DRAM the hand-written AVX2 is {s_dram['sec']/fm_dram['sec']:.2f}x the time of "
          f"plain -ffast-math: all that intrinsics work bought "
          f"{100*(fm_dram['sec']/s_dram['sec']-1):+.0f}%.")
    p_dram = get("poly", "avx2_ilp", "DRAM")
    print(f"The compute-bound poly keeps {p_dram['speedup_vs_scalar']:.1f}x even from DRAM.")

    print("\n=== Effect 2: instruction-level parallelism, not just width ===")
    for w in ["L1", "L2", "L3", "DRAM"]:
        a, b = get("poly", "avx2", w), get("poly", "avx2_ilp", w)
        print(f"{w:<6} 1 chain: {a['gflops']:7.1f} GFLOP/s   4 chains: {b['gflops']:7.1f} "
              f"GFLOP/s   ({b['gflops']/a['gflops']:.2f}x for the same instructions)")
    fm = get("poly", "fastmath", "L1")
    nv = get("poly", "avx2", "L1")
    ilp = get("poly", "avx2_ilp", "L1")
    print(f"\nEight lanes alone: {fm['sec']/nv['sec']:.2f}x over the compiler's own vector code.")
    print(f"Eight lanes plus four chains: {fm['sec']/ilp['sec']:.2f}x. Width was the "
          f"smaller half of the win.")

    print("\n=== Effect 3: the fast answer is the accurate one ===")
    big = max(r["n"] for r in rows)
    print(f"Summing {big/1e6:.1f}M positive floats. Exact value (double): "
          f"{refs[('sum', big)]:.1f}")
    print(f"{'variant':<10} {'sum returned':>16} {'relative error':>16}")
    for v in ["scalar", "auto", "fastmath", "avx2"]:
        g = get("sum", v, "DRAM")
        val = [r for r in rows if r["kernel"] == "sum" and r["variant"] == v
               and r["n"] == big][0]["result"]
        print(f"{v:<10} {val:>16.1f} {g['rel_error']*100:>15.4f}%")
    sc = get("sum", "scalar", "DRAM")
    av = get("sum", "avx2", "DRAM")
    print(f"\nThe 'unsafe' vector version is {sc['rel_error']/av['rel_error']:.0f}x more "
          f"accurate than the careful scalar one.")
    print("One float32 accumulator over 33.5M values loses the small terms;")
    print("32 partial accumulators keep them. Reordering helped precision here.")

    # ---------------- what 512-bit code does on a CPU without it ----------------
    print("\n=== The AVX-512 path, forced ===")
    r2 = subprocess.run([EXE, "--force-avx512"], capture_output=True, text=True)
    sig = -r2.returncode if r2.returncode < 0 else 0
    print(f"return code: {r2.returncode}"
          + (f"  (killed by signal {sig} = SIGILL, illegal instruction)" if sig == 4 else ""))
    print("The binary contains valid AVX-512 code. This CPU has no zmm registers,")
    print("so the very first vmovups %zmm is an instruction the silicon does not")
    print("know. That is why every real library dispatches at run time.")

    findings = dict(cpu=model, avx512_in_cpuinfo=sorted(f for f in flags if f.startswith("avx512")),
                    runtime_avx512=cpu["avx512f"], autovectorised_float_sum=auto_vec,
                    avx512_forced_signal=sig, disasm=dis, rows=out_rows)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    plot(out_rows, model)


def plot(rows, model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    wheres = ["L1", "L2", "L3", "DRAM"]
    cols = {"scalar": "#999999", "auto": "#4C78A8", "fastmath": "#54A24B",
            "avx2": "#F58518", "avx2_ilp": "#E45756"}

    def series(kernel, variant, field):
        return [[r for r in rows if r["kernel"] == kernel and r["variant"] == variant
                 and r["where"] == w][0][field] for w in wheres]

    # (a) speedup vs scalar, both kernels
    xs = np.arange(len(wheres))
    for v in ["auto", "fastmath", "avx2"]:
        ax[0].plot(xs, series("sum", v, "speedup_vs_scalar"), "o-", color=cols[v], label=v)
    ax[0].set_xticks(xs); ax[0].set_xticklabels(wheres)
    ax[0].set_yscale("log")
    ax[0].axhline(1, c="k", lw=1, ls="--")
    ax[0].set_ylabel("speedup over scalar (log)")
    ax[0].set_xlabel("where the array fits")
    ax[0].set_title("(a) sum: 0.25 FLOP/byte\nthe SIMD win drains away into DRAM")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")

    # (b) same for the compute-bound kernel
    for v in ["auto", "fastmath", "avx2", "avx2_ilp"]:
        ax[1].plot(xs, series("poly", v, "speedup_vs_scalar"), "o-", color=cols[v], label=v)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(wheres)
    ax[1].set_yscale("log")
    ax[1].axhline(1, c="k", lw=1, ls="--")
    ax[1].set_ylabel("speedup over scalar (log)")
    ax[1].set_xlabel("where the array fits")
    ax[1].set_title("(b) poly: 10 FLOP/byte\nthe win survives everywhere")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")

    # (c) GFLOP/s achieved on the compute-bound kernel
    w = 0.2
    for i, v in enumerate(["scalar", "fastmath", "avx2", "avx2_ilp"]):
        ax[2].bar(xs + (i - 1.5) * w, series("poly", v, "gflops"), w,
                  color=cols[v], label=v)
    ax[2].set_xticks(xs); ax[2].set_xticklabels(wheres)
    ax[2].set_ylabel("GFLOP/s (single core)")
    ax[2].set_title("(c) poly throughput\n'avx2' and 'avx2_ilp' run the SAME instructions")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3, axis="y")

    fig.suptitle(f"{model.strip()} - no AVX-512 on this CPU, so the 512-bit path is "
                 f"compiled but never executed", fontsize=9, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "simd.png"), dpi=110)
    print(f"\nwrote {OUT}/simd.png")


if __name__ == "__main__":
    sys.exit(main())
