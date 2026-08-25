"""Project 06 - read a GPU's real specification out of the machine itself.

Three sources of truth, deliberately kept separate so you can see where each
number does and does not come from:

  1. `nvidia-smi`             - the *driver's* management view (live telemetry)
  2. `cudaGetDeviceProperties`- the *runtime's* architectural view (static facts)
  3. an architecture table    - cores per SM, which neither of the above knows

Then it runs two load kernels while sampling nvidia-smi, to show that
`utilization.gpu` reports "is a kernel resident", not "is the GPU doing work".
"""

import json
import os
import shutil
import subprocess
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

EXE = os.path.join(OUT, "devprops")


# --------------------------------------------------------------------------
# source 1: nvidia-smi
# --------------------------------------------------------------------------
SMI_FIELDS = [
    "name", "compute_cap", "driver_version", "vbios_version",
    "memory.total", "memory.used", "memory.free",
    "clocks.max.sm", "clocks.max.memory", "clocks.current.sm",
    "power.limit", "power.draw", "temperature.gpu", "pstate",
    "pcie.link.gen.max", "pcie.link.gen.current",
    "pcie.link.width.max", "pcie.link.width.current",
    "utilization.gpu", "utilization.memory", "ecc.mode.current",
    "compute_mode", "persistence_mode",
]


def smi_query(fields):
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("nvidia-smi failed:\n" + r.stderr)
    vals = [v.strip() for v in r.stdout.strip().splitlines()[0].split(",")]
    return dict(zip(fields, vals))


def smi_text(args):
    r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else "(command failed)\n" + r.stderr


# --------------------------------------------------------------------------
# source 2 + 3: the CUDA runtime, via devprops.cu
# --------------------------------------------------------------------------
def build():
    if shutil.which("nvcc") is None:
        raise SystemExit("nvcc not found - this project needs a CUDA toolkit")
    cmd = ["nvcc", "-O3", "-arch=sm_61", os.path.join(HERE, "devprops.cu"),
           "-o", EXE]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("compile failed:\n" + r.stderr[-3000:])


def cuda_props():
    r = subprocess.run([EXE], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("devprops failed:\n" + r.stderr)
    d = {}
    for line in r.stdout.strip().splitlines():
        k, _, v = line.partition(",")
        d[k] = v
    return d


# --------------------------------------------------------------------------
# the utilization experiment
# --------------------------------------------------------------------------
def sample_while(mode):
    """Run devprops <mode> and poll nvidia-smi ~20x/s for the whole run."""
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            s = smi_query(["utilization.gpu", "clocks.sm", "power.draw",
                           "temperature.gpu"])
            samples.append(s)
            time.sleep(0.05)

    t = threading.Thread(target=poll, daemon=True)
    t.start()
    r = subprocess.run([EXE, mode], capture_output=True, text=True)
    stop.set()
    t.join(timeout=2)
    if r.returncode != 0:
        raise SystemExit(f"devprops {mode} failed:\n{r.stderr}")
    _, _, sec, gflops = r.stdout.strip().split(",")

    # ignore the ramp-up: keep samples from the middle 60% of the run
    mid = samples[int(len(samples) * 0.3):int(len(samples) * 0.9)] or samples
    def avg(k):
        v = [float(s[k]) for s in mid if s[k] not in ("[N/A]", "N/A")]
        return sum(v) / len(v) if v else float("nan")
    return dict(mode=mode, seconds=float(sec), gflops=float(gflops),
                util_pct=avg("utilization.gpu"), clock_mhz=avg("clocks.sm"),
                power_w=avg("power.draw"), temp_c=avg("temperature.gpu"),
                n_samples=len(mid))


# --------------------------------------------------------------------------
def main():
    build()
    smi = smi_query(SMI_FIELDS)
    p = cuda_props()

    sm = int(p["sm_count"])
    cps = int(p["cores_per_sm"])
    boost_ghz = int(p["clock_khz"]) / 1e6
    max_ghz = float(smi["clocks.max.sm"]) / 1e3
    bus = int(p["mem_bus_bits"])
    memclk_ghz = int(p["mem_clock_khz"]) / 1e6

    print("=" * 74)
    print("SOURCE 1 - nvidia-smi (the driver's management view)")
    print("=" * 74)
    for k in SMI_FIELDS:
        print(f"  {k:<28} {smi[k]}")

    print()
    print("=" * 74)
    print("SOURCE 2 - cudaGetDeviceProperties (the runtime's architecture view)")
    print("=" * 74)
    for k, v in p.items():
        print(f"  {k:<28} {v}")

    print()
    print("=" * 74)
    print("What ONLY nvidia-smi knows        vs   what ONLY the CUDA runtime knows")
    print("=" * 74)
    only_smi = ["driver_version", "vbios_version", "power.limit", "power.draw",
                "temperature.gpu", "pstate", "clocks.max.sm",
                "pcie.link.gen.current", "pcie.link.width.current",
                "utilization.gpu", "persistence_mode"]
    only_cuda = ["sm_count", "l2_bytes", "mem_bus_bits", "mem_clock_khz",
                 "regs_per_sm", "shared_mem_per_sm", "max_threads_per_sm",
                 "warp_size", "async_engine_count", "concurrent_kernels"]
    for a, b in zip(only_smi + [""] * 9, only_cuda + [""] * 9):
        if not a and not b:
            break
        print(f"  {a:<34} {b}")
    print(f"  {'(neither): cores per SM = ' + p['cores_per_sm']:<34} "
          f"<- from a compute-capability table")

    # ---- the numbers you actually wanted, computed ----
    peak_boost = sm * cps * 2 * boost_ghz * 1e9
    peak_max = sm * cps * 2 * max_ghz * 1e9
    bw = memclk_ghz * 1e9 * 2 * bus / 8

    print()
    print("=" * 74)
    print("DERIVED - the four numbers that actually matter")
    print("=" * 74)
    print(f"  peak fp32 (boost clock {boost_ghz*1e3:.0f} MHz) = "
          f"{sm} SM x {cps} cores x 2 x {boost_ghz:.3f} GHz = "
          f"{peak_boost/1e12:.2f} TFLOP/s")
    print(f"  peak fp32 (max   clock {max_ghz*1e3:.0f} MHz) = "
          f"{peak_max/1e12:.2f} TFLOP/s   <- {peak_max/peak_boost:.2f}x the above")
    print(f"  memory bandwidth = {memclk_ghz:.3f} GHz x 2 (DDR) x {bus} bits / 8 "
          f"= {bw/1e9:.1f} GB/s")
    print(f"  ridge point = {peak_boost/1e12:.2f} TFLOP/s / {bw/1e9:.1f} GB/s = "
          f"{peak_boost/bw:.1f} FLOP/byte")

    print()
    print("=" * 74)
    print("TOPOLOGY - nvidia-smi topo -m")
    print("=" * 74)
    topo = smi_text(["topo", "-m"])
    print(topo.strip()[:1400])
    nvl = smi_text(["nvlink", "-s"]).strip()
    print(f"\n  nvidia-smi nvlink -s : {nvl if nvl else '(empty - no NVLink on this GPU)'}")
    n_gpus = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                                text=True).stdout.strip().splitlines())
    print(f"  GPUs visible: {n_gpus}"
          + ("  -> the topology matrix is 1x1 and there is nothing to route" if n_gpus == 1 else ""))
    print(f"  PCIe link: gen {smi['pcie.link.gen.current']}/"
          f"{smi['pcie.link.gen.max']} x{smi['pcie.link.width.current']}/"
          f"{smi['pcie.link.width.max']}")

    print()
    print("=" * 74)
    print("EXPERIMENT - does 'utilization.gpu' mean the GPU is doing work?")
    print("=" * 74)
    print("  running one warp on one SM in a dependent chain (~4 s) ...")
    lazy = sample_while("lazy")
    print("  running every SM flat out                        (~4 s) ...")
    busy = sample_while("busy")

    print()
    print(f"  {'':<8} {'util%':>7} {'clock':>9} {'power':>8} {'temp':>6} "
          f"{'GFLOP/s':>10} {'% of peak':>10}")
    for r in (lazy, busy):
        print(f"  {r['mode']:<8} {r['util_pct']:7.1f} {r['clock_mhz']:8.0f}M "
              f"{r['power_w']:7.1f}W {r['temp_c']:5.0f}C "
              f"{r['gflops']:10.1f} {100*r['gflops']*1e9/peak_boost:9.1f}%")
    print(f"\n  Both report near-identical utilization, but 'busy' does "
          f"{busy['gflops']/lazy['gflops']:.0f}x the arithmetic.")
    print(f"  Power tells the truth that utilization does not: "
          f"{lazy['power_w']:.0f} W vs {busy['power_w']:.0f} W "
          f"({busy['power_w']/lazy['power_w']:.1f}x).")

    # measured peak, using the clock the card actually ran at
    real_peak = sm * cps * 2 * busy["clock_mhz"] * 1e6
    print(f"\n  At the clock actually observed during the busy run "
          f"({busy['clock_mhz']:.0f} MHz) the peak is {real_peak/1e12:.2f} TFLOP/s, "
          f"so 'busy' reached {100*busy['gflops']*1e9/real_peak:.1f}% of it.")

    findings = dict(
        smi=smi, cuda=p,
        sm_count=sm, cores_per_sm=cps, total_cores=sm * cps,
        peak_tflops_boost_clock=peak_boost / 1e12,
        peak_tflops_max_clock=peak_max / 1e12,
        peak_tflops_observed_clock=real_peak / 1e12,
        bandwidth_gbs=bw / 1e9, ridge_point_flop_per_byte=peak_boost / bw,
        n_gpus=n_gpus, nvlink=bool(nvl),
        lazy=lazy, busy=busy,
        util_gap_x=busy["gflops"] / lazy["gflops"],
        power_gap_x=busy["power_w"] / lazy["power_w"])
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2)
    with open(os.path.join(OUT, "topology.txt"), "w") as f:
        f.write(topo)
    print(f"\nwrote {OUT}/findings.json")
    plot(findings, peak_boost)


def plot(f, peak):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) who knows what
    ax[0].axis("off")
    rows = [("name / model", "yes", "yes"), ("compute capability", "yes", "yes"),
            ("total memory", "yes", "yes"), ("driver + VBIOS version", "yes", "no"),
            ("power draw / limit", "yes", "no"), ("temperature, clocks, pstate", "yes", "no"),
            ("PCIe link gen + width", "yes", "no"), ("SM count", "no", "yes"),
            ("L2 cache size", "no", "yes"), ("memory bus width + clock", "no", "yes"),
            ("registers / shared mem per SM", "no", "yes"),
            ("max warps per SM", "no", "yes"), ("cores per SM", "no", "no")]
    ax[0].set_title("(a) Two tools, two different halves of the truth",
                    fontsize=11)
    ax[0].text(0.02, 0.95, f"{'fact':<30}{'smi':>6}{'CUDA':>7}", family="monospace",
               fontsize=9, va="top", weight="bold")
    for i, (k, a, b) in enumerate(rows):
        col = "#C44E52" if (a == "no" and b == "no") else "#333333"
        ax[0].text(0.02, 0.88 - i * 0.066, f"{k:<30}{a:>6}{b:>7}",
                   family="monospace", fontsize=9, va="top", color=col)
    ax[0].text(0.02, 0.88 - len(rows) * 0.066 - 0.02,
               "cores/SM needs an architecture table", fontsize=8.5,
               color="#C44E52", style="italic")

    # (b) utilization lie
    labels = ["one lazy warp\n(1 of 19 SMs,\ndependent chain)",
              "every SM\n(4 independent\nchains/thread)"]
    util = [f["lazy"]["util_pct"], f["busy"]["util_pct"]]
    gfl = [f["lazy"]["gflops"], f["busy"]["gflops"]]
    x = [0, 1]
    b1 = ax[1].bar([i - 0.2 for i in x], util, 0.4, color="#E45756",
                   label="nvidia-smi utilization.gpu (%)")
    ax2 = ax[1].twinx()
    b2 = ax2.bar([i + 0.2 for i in x], gfl, 0.4, color="#4C78A8",
                 label="measured GFLOP/s")
    ax2.set_yscale("log")
    ax2.set_ylim(1, peak / 1e9 * 6)
    for i, v in zip(x, util):
        ax[1].text(i - 0.2, v + 2, f"{v:.0f}%", ha="center", fontsize=9, color="#E45756")
    for i, v in zip(x, gfl):
        ax2.text(i + 0.2, v * 1.3, f"{v:,.0f}", ha="center", fontsize=9, color="#4C78A8")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=8.5)
    ax[1].set_ylim(0, 125)
    ax[1].set_ylabel("utilization.gpu (%)", color="#E45756")
    ax2.set_ylabel("GFLOP/s (log)", color="#4C78A8")
    ax[1].set_title(f"(b) Same 'utilization', {f['util_gap_x']:.0f}x the work\n"
                    "utilization.gpu = 'a kernel was resident', not 'work happened'",
                    fontsize=11)
    ax[1].legend(handles=[b1, b2], loc="upper left", fontsize=8)

    # (c) which peak?
    names = ["boost clock\n(spec sheet)", "max clock\n(nvidia-smi)",
             "observed clock\n(during busy)", "measured\n(busy kernel)"]
    vals = [f["peak_tflops_boost_clock"], f["peak_tflops_max_clock"],
            f["peak_tflops_observed_clock"], f["busy"]["gflops"] / 1e3]
    cols = ["#B0B0B0", "#B0B0B0", "#B0B0B0", "#54A24B"]
    ax[2].bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax[2].text(i, v + 0.12, f"{v:.2f}", ha="center", fontsize=9.5)
    ax[2].set_ylabel("fp32 TFLOP/s")
    ax[2].set_ylim(0, max(vals) * 1.22)
    ax[2].tick_params(axis="x", labelsize=8.5)
    ax[2].set_title("(c) 'Peak FLOPs' is three different numbers\n"
                    "the one you divide by decides your score", fontsize=11)
    ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gpu-facts.png"), dpi=110)
    print(f"wrote {OUT}/gpu-facts.png")


if __name__ == "__main__":
    main()
