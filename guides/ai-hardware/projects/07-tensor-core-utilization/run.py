"""Project 07 - what fraction of the GPU's math pipeline a matmul really uses.

The guide's brief is "run nsys and ncu on a matmul, observe % of time tensor
cores are active". Two things get in the way on this machine, and both are
worth knowing about rather than hiding:

  * `ncu` cannot read performance counters here (ERR_NVGPUCTRPERM - a
    kernel-module setting only root can change),
  * `nsys` is installed without its trace importer, so it records but cannot
    produce a report.

So this script measures the same quantity directly:  achieved ops/sec divided
by peak ops/sec, which is the definition behind Nsight's
`pct_of_peak_sustained` counters. It also attempts both profilers and prints
exactly how they fail, so you can tell a broken setup from a broken kernel.
"""

import json
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
EXE = os.path.join(OUT, "bench")
N = 2048


def sh(cmd, timeout=300):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timed out"
    except FileNotFoundError:
        return -2, "", "not installed"


def build():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    rc, _, err = sh(["nvcc", "-O3", "-arch=sm_61", os.path.join(HERE, "bench.cu"),
                     "-o", EXE, "-lcublas"])
    if rc != 0:
        raise SystemExit("compile failed:\n" + err[-3000:])


def parse(text):
    dev, mm, tl, notes = {}, [], [], {}
    for line in text.strip().splitlines():
        f = line.split(",")
        if f[0] == "#device":
            dev = dict(name=f[1], cc=f[2], sms=int(f[3]), clock_khz=int(f[4]))
        elif f[0] == "#tensor_cores":
            dev["tensor_cores"] = bool(int(f[1]))
        elif f[0].startswith("#"):
            notes.setdefault(f[0][1:], []).append(f[1:])
        elif f[0] == "mm":
            mm.append(dict(name=f[1], n=int(f[2]), ms=float(f[3])))
        elif f[0] == "timeline":
            tl.append(dict(n=int(f[1]), h2d=float(f[2]), comp=float(f[3]),
                           d2h=float(f[4])))
    return dev, mm, tl, notes


def main():
    build()
    rc, out, err = sh([EXE, str(N)])
    if rc != 0:
        raise SystemExit("bench failed:\n" + out[-2000:] + err[-2000:])
    dev, mm, tl, notes = parse(out)

    ghz = dev["clock_khz"] / 1e6
    cores = dev["sms"] * 128                       # 128 CUDA cores/SM on sm_61
    peak_fp32 = cores * 2 * ghz * 1e9              # 1 FMA/core/clock = 2 FLOP
    # dp4a runs on the same issue slot but does 4 mul + 4 add per instruction
    peak_int8 = cores * 8 * ghz * 1e9

    print("=" * 76)
    print(f"Device: {dev['name']}  compute capability {dev['cc']}  "
          f"{dev['sms']} SMs @ {ghz*1e3:.0f} MHz")
    print(f"Tensor Cores present: {dev['tensor_cores']}"
          "   <- introduced in compute capability 7.0 (Volta)")
    print(f"peak fp32 = {cores} cores x 2 FLOP x {ghz:.3f} GHz "
          f"= {peak_fp32/1e12:.2f} TFLOP/s")
    print(f"peak int8 = {cores} cores x 8 OP   x {ghz:.3f} GHz "
          f"= {peak_int8/1e12:.2f} TOP/s   (dp4a = 4 MACs per instruction)")
    print("=" * 76)

    print("\n--- Can this GPU even compile a Tensor Core kernel? ---")
    probe = os.path.join(HERE, "wmma_probe.cu")
    gate = {}
    for arch in ("sm_61", "sm_70"):
        rc2, _, err2 = sh(["nvcc", f"-arch={arch}", probe, "-o",
                           os.path.join(OUT, f"wmma_{arch}")])
        first = next((l for l in err2.splitlines() if "error" in l), "").strip()
        first = re.sub(r"^.*\.cu\(\d+\): ", "", first)
        gate[arch] = dict(ok=rc2 == 0, first_error=first)
        print(f"  nvcc -arch={arch:<6} -> "
              + ("compiles" if rc2 == 0 else f"FAILS: {first}"))
    print("  The failure is at compile time, not run time: below cc 7.0 the")
    print("  `nvcuda` namespace that `mma.h` defines does not exist, so there")
    print("  is nothing to fall back to at runtime.")

    print("\n--- The profilers the guide asks for ---")
    prof = {}
    rc3, o3, e3 = sh(["ncu", "--metrics",
                      "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                      EXE, "256"], timeout=180)
    msg = next((l.strip() for l in (o3 + e3).splitlines() if "ERROR" in l), "(ran)")
    prof["ncu"] = dict(rc=rc3, msg=msg[:200])
    print(f"  ncu : {msg[:150]}")
    rc4, o4, e4 = sh(["nsys", "profile", "--force-overwrite=true", "-o",
                      os.path.join(OUT, "trace"), "--trace=cuda", EXE, "256"],
                     timeout=180)
    msg4 = next((l.strip() for l in (o4 + e4).splitlines()
                 if "importer" in l.lower()), "(ran)")
    prof["nsys"] = dict(rc=rc4, msg=msg4[:200])
    for ext in (".qdstrm", ".nsys-rep", ".qdrep"):   # multi-MB, not worth keeping
        leftover = os.path.join(OUT, "trace" + ext)
        if os.path.exists(leftover):
            os.remove(leftover)
    print(f"  nsys: {msg4[:150]}")
    print("  Neither can be fixed from user space here, so everything below is")
    print("  measured with CUDA events instead.")

    # ---------------- the actual measurement ----------------
    print(f"\n{'='*76}\nFive matmuls, N = {N} (2*N^3 = {2*N**3/1e9:.1f} G ops each)"
          f"\n{'='*76}")
    print(f"{'kernel':<14} {'ms':>8} {'Tops/s':>9} {'peak':>9} "
          f"{'pipe util':>10}  what it uses")
    rows = []
    what = {"naive_fp32": "FMA, all operands from global memory",
            "tiled_fp32": "FMA, operands staged in shared memory",
            "cublas_fp32": "FMA, NVIDIA's own SGEMM",
            "dp4a_int8": "dp4a, our tiled kernel",
            "cublas_int8": "dp4a, NVIDIA's own IGEMM"}
    for m in mm:
        ops = 2.0 * N ** 3
        rate = ops / (m["ms"] / 1e3)
        peak = peak_int8 if "int8" in m["name"] else peak_fp32
        print(f"{m['name']:<14} {m['ms']:8.2f} {rate/1e12:9.2f} "
              f"{peak/1e12:9.2f} {100*rate/peak:9.1f}%  {what[m['name']]}")
        rows.append(dict(name=m["name"], ms=m["ms"], tops=rate / 1e12,
                         peak_tops=peak / 1e12, pct_peak=100 * rate / peak))
    by = {r["name"]: r for r in rows}

    print(f"\n  tiled beats naive by            {by['naive_fp32']['ms']/by['tiled_fp32']['ms']:.2f}x"
          "   (shared memory, same instructions)")
    print(f"  cuBLAS beats our best fp32 by   {by['tiled_fp32']['ms']/by['cublas_fp32']['ms']:.2f}x"
          "   (same instructions again)")
    if "cublas_int8" in by:
        print(f"  cuBLAS int8 beats cuBLAS fp32   "
              f"{by['cublas_fp32']['ms']/by['cublas_int8']['ms']:.2f}x"
              "   <- THIS is the Tensor Core effect, in miniature")
        print(f"  cuBLAS int8 beats OUR dp4a by   "
              f"{by['dp4a_int8']['ms']/by['cublas_int8']['ms']:.2f}x"
              "   (having the instruction is not the same as feeding it)")
    print(f"\n  verification: dp4a mismatches in an 8x8 corner = "
          f"{notes['dp4a_mismatches_in_8x8'][0][0]}")
    for v in notes.get("maxabsdiff_vs_naive", []):
        print(f"                {v[0]:<12} vs naive, max abs diff = {v[1]}")

    # SASS proof that the instruction we think we asked for is really there
    rc5, o5, _ = sh(["cuobjdump", "-sass", EXE])
    idp = len(re.findall(r"\bIDP\.4A", o5)) if rc5 == 0 else -1
    ffma = len(re.findall(r"\bFFMA\b", o5)) if rc5 == 0 else -1
    print(f"\n  SASS check (cuobjdump): {idp} IDP.4A instructions, "
          f"{ffma} FFMA instructions in the binary.")
    print("  IDP.4A is dp4a's real machine name - 'Integer Dot Product, 4 elements, "
          "type A'.")

    # ---------------- timeline ----------------
    print(f"\n{'='*76}\nWhere the wall clock goes (cuBLAS SGEMM + PCIe copies)"
          f"\n{'='*76}")
    print(f"{'N':>6} {'copy in':>9} {'compute':>9} {'copy out':>9} "
          f"{'total':>9} {'compute share':>14}")
    for t in tl:
        tot = t["h2d"] + t["comp"] + t["d2h"]
        print(f"{t['n']:6d} {t['h2d']:8.3f}m {t['comp']:8.3f}m {t['d2h']:8.3f}m "
              f"{tot:8.3f}m {100*t['comp']/tot:13.1f}%")
    small, big = tl[0], tl[-1]
    print(f"\n  At N={small['n']} the matmul is "
          f"{100*small['comp']/(small['h2d']+small['comp']+small['d2h']):.0f}% of "
          f"the timeline; at N={big['n']} it is "
          f"{100*big['comp']/(big['h2d']+big['comp']+big['d2h']):.0f}%.")
    print("  A kernel at 100% pipe utilization that is 15% of your timeline")
    print("  still leaves 85% of the wall clock untouched. Profile the whole")
    print("  timeline before you optimise the kernel - that is what nsys is for.")

    findings = dict(
        device=dev, peak_fp32_tflops=peak_fp32 / 1e12,
        peak_int8_tops=peak_int8 / 1e12, N=N, kernels=rows,
        wmma_compile_gate=gate, profilers=prof,
        sass_idp4a=idp, sass_ffma=ffma,
        dp4a_mismatches=int(notes["dp4a_mismatches_in_8x8"][0][0]),
        maxabsdiff_vs_naive={v[0]: v[1] for v in notes.get("maxabsdiff_vs_naive", [])},
        timeline=tl,
        int8_over_fp32=(by["cublas_fp32"]["ms"] / by["cublas_int8"]["ms"]
                        if "cublas_int8" in by else None),
        cublas_over_ours_fp32=by["tiled_fp32"]["ms"] / by["cublas_fp32"]["ms"],
        cublas_over_ours_int8=(by["dp4a_int8"]["ms"] / by["cublas_int8"]["ms"]
                               if "cublas_int8" in by else None))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    print(f"\nwrote {OUT}/findings.json")
    plot(rows, tl, peak_fp32, peak_int8)


def plot(rows, tl, peak_fp32, peak_int8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.8))

    # (a) throughput vs the two peaks
    names = [r["name"].replace("_", "\n") for r in rows]
    vals = [r["tops"] for r in rows]
    cols = ["#4C78A8" if "int8" not in r["name"] else "#F58518" for r in rows]
    ax[0].bar(names, vals, color=cols)
    ax[0].axhline(peak_fp32 / 1e12, ls="--", c="#4C78A8", lw=1.2)
    ax[0].text(-0.45, peak_fp32 / 1e12 * 1.03, f"fp32 peak {peak_fp32/1e12:.1f}",
               fontsize=8, color="#4C78A8")
    ax[0].axhline(peak_int8 / 1e12, ls="--", c="#F58518", lw=1.2)
    ax[0].text(-0.45, peak_int8 / 1e12 * 1.02, f"int8 peak {peak_int8/1e12:.1f}",
               fontsize=8, color="#F58518")
    for i, r in enumerate(rows):
        ax[0].text(i, r["tops"] + 0.9, f"{r['tops']:.1f}\n{r['pct_peak']:.0f}%",
                   ha="center", fontsize=8.5)
    ax[0].set_ylabel("Tera-ops/second")
    ax[0].set_ylim(0, peak_int8 / 1e12 * 1.18)
    ax[0].tick_params(axis="x", labelsize=8)
    ax[0].set_title("(a) Same matmul, five ways\n"
                    "the % under each bar is the 'pipe utilization'\n"
                    "Nsight would report", fontsize=10.5)
    ax[0].grid(alpha=.3, axis="y")

    # (b) what each step bought
    labels = ["naive\nfp32", "+ shared-mem\ntiling", "+ NVIDIA's\nSGEMM",
              "+ dp4a\n(matrix instr.)"]
    seq = ["naive_fp32", "tiled_fp32", "cublas_fp32", "cublas_int8"]
    by = {r["name"]: r for r in rows}
    seq = [s for s in seq if s in by]
    base = by[seq[0]]["tops"]
    sp = [by[s]["tops"] / base for s in seq]
    ax[1].plot(range(len(sp)), sp, "o-", color="#54A24B", lw=2, ms=8)
    for i, v in enumerate(sp):
        ax[1].annotate(f"{v:.1f}x", (i, v), textcoords="offset points",
                       xytext=(0, 11), ha="center", fontsize=10)
    ax[1].set_xticks(range(len(sp)))
    ax[1].set_xticklabels(labels[:len(sp)], fontsize=8.5)
    ax[1].set_yscale("log")
    ax[1].set_ylim(0.7, max(sp) * 2.2)
    ax[1].set_ylabel("speedup over the naive kernel (log)")
    ax[1].set_title("(b) The last step is the Tensor Core idea\n"
                    "one instruction that does a whole dot product", fontsize=10.5)
    ax[1].grid(alpha=.3, which="both")

    # (c) timeline share
    ns = [t["n"] for t in tl]
    tot = [t["h2d"] + t["comp"] + t["d2h"] for t in tl]
    bot = [0.0] * len(tl)
    for key, c, lab in (("h2d", "#E45756", "PCIe host->device"),
                        ("comp", "#54A24B", "SGEMM kernel"),
                        ("d2h", "#F58518", "PCIe device->host")):
        share = [100 * t[key] / s for t, s in zip(tl, tot)]
        ax[2].bar([str(n) for n in ns], share, bottom=bot, color=c, label=lab)
        bot = [b + s for b, s in zip(bot, share)]
    for i, t in enumerate(tl):
        ax[2].text(i, 101, f"{tot[i]:.2f} ms", ha="center", fontsize=7.5)
    ax[2].set_ylim(0, 112)
    ax[2].set_ylabel("share of wall clock (%)")
    ax[2].set_xlabel("matrix size N")
    ax[2].legend(fontsize=8, loc="lower left")
    ax[2].set_title("(c) The kernel is not the timeline\n"
                    "a perfect matmul cannot fix the copies around it",
                    fontsize=10.5)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipe-utilization.png"), dpi=110)
    print(f"wrote {OUT}/pipe-utilization.png")


if __name__ == "__main__":
    main()
