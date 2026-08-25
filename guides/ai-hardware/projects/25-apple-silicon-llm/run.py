"""Project 25 - unified memory, measured on a machine that does not have it.

  A. the pools     - the three bandwidths this machine has, measured
  B. the crossover - resident vs streamed vs CPU, on a real 2 GB working set
  C. the cliff     - how much model fits on the GPU before the copy starts
  D. isolation     - a discrete GPU is immune to CPU memory traffic; a unified
                     one is not, and the CPU side of that is measured here
  E. what fits     - model sizes against the pools that exist to buy
"""

import csv
import json
import os
import sys
import threading
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "18-triton-softmax")))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                       # noqa: E402
from unified import (gpu_matvec, path_resident, path_streamed,   # noqa: E402
                     path_cpu)

R = {}
K = 4096                       # weight matrices are K x K
LAYER_BYTES = K * K * 2        # fp16
WORKING_SET = 2 << 30          # 2 GB: far past every cache on this machine
NLAYER = WORKING_SET // LAYER_BYTES


# ----------------------------------------------------------------- A. pools
def section_a():
    torch.set_num_threads(os.cpu_count())
    n = 1 << 26                                  # 64 Mi elements = 256 MB fp32

    # CPU DRAM: a matrix-vector product reads the matrix once and nothing else.
    # Sweep the thread count too, because one memory pool shared by 12 threads
    # does not go 12x faster - and the best count is needed by every later
    # CPU measurement, so it is found here rather than assumed.
    Wc = torch.randn(8192, 8192)
    xc = torch.randn(8192)
    scale = []
    for nt in [1, 2, 4, 6, 12]:
        torch.set_num_threads(nt)
        for _ in range(2):
            torch.mv(Wc, xc)
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(6):
                torch.mv(Wc, xc)
            runs.append((time.perf_counter() - t0) / 6)
        scale.append(dict(threads=nt,
                          gbs=round(8192 * 8192 * 4 / min(runs) / 1e9, 1)))
    R["A_cpu_scaling"] = scale
    top = max(scale, key=lambda d: d["gbs"])
    R["A_cpu_dram_gbs"] = top["gbs"]
    R["A_cpu_best_threads"] = top["threads"]
    R["A_cpu_1_to_best"] = round(top["gbs"] / scale[0]["gbs"], 2)
    R["A_cpu_ht_penalty"] = round(top["gbs"] / scale[-1]["gbs"], 2)
    torch.set_num_threads(top["threads"])

    # PCIe, both ways, pageable and pinned
    for pin in (False, True):
        h = torch.empty(n, dtype=torch.float16, pin_memory=pin)
        d = gpu.empty(n, dtype=torch.float16)
        for _ in range(3):
            d.copy_(h)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            d.copy_(h)
        torch.cuda.synchronize()
        R[f"A_pcie_h2d_{'pinned' if pin else 'pageable'}_gbs"] = round(
            n * 2 * 10 / (time.perf_counter() - t0) / 1e9, 1)
        del h, d

    # GPU DRAM: the resident path on a 2 GB working set
    layers = [gpu.empty(K * K, dtype=torch.float16) for _ in range(8)]
    x = gpu.empty(K, dtype=torch.float16)
    y = gpu.empty(K)
    t = path_resident(layers, x, y, K, reps=5)
    R["A_gpu_dram_gbs"] = round(len(layers) * LAYER_BYTES / t / 1e9, 1)
    del layers

    R["A_gpu_over_pcie"] = round(R["A_gpu_dram_gbs"]
                                 / R["A_pcie_h2d_pinned_gbs"], 1)
    R["A_cpu_over_pcie"] = round(R["A_cpu_dram_gbs"]
                                 / R["A_pcie_h2d_pinned_gbs"], 2)
    print(f"A. GPU DRAM {R['A_gpu_dram_gbs']} GB/s | CPU DRAM "
          f"{R['A_cpu_dram_gbs']} GB/s | PCIe {R['A_pcie_h2d_pinned_gbs']} "
          f"pinned / {R['A_pcie_h2d_pageable_gbs']} pageable GB/s")


# ------------------------------------------------------------ B. the paths
def section_b():
    g = torch.Generator().manual_seed(0)
    xh = torch.randn(K, generator=g).half()
    x = xh.cuda()
    y = gpu.empty(K)

    host = [torch.empty(K * K, dtype=torch.float16, pin_memory=True)
            for _ in range(NLAYER)]
    for h in host:
        h.normal_(0, 0.02, generator=g)
    stage = [gpu.empty(K * K, dtype=torch.float16) for _ in range(2)]

    res_layers = [gpu.empty(K * K, dtype=torch.float16) for _ in range(NLAYER)]
    for d, h in zip(res_layers, host):
        d.copy_(h)

    rows = []
    t = path_resident(res_layers, x, y, K, reps=3)
    rows.append(("GPU, weights resident in VRAM", t))
    del res_layers
    torch.cuda.empty_cache()

    t = path_streamed(host, stage, x, y, K, reps=3, overlap=True)
    rows.append(("GPU, weights streamed over PCIe (overlapped)", t))
    t = path_streamed(host, stage, x, y, K, reps=3, overlap=False)
    rows.append(("GPU, weights streamed over PCIe (serial)", t))

    cpu_layers = [h.view(K, K).float() for h in host[:NLAYER // 2]]
    xc = xh.float()
    t = path_cpu(cpu_layers, xc, reps=3)
    rows.append(("CPU, weights in RAM (no accelerator)",
                 t * 2 * (2.0 / 4.0)))     # half the layers, fp32 -> fp16 bytes
    del cpu_layers

    total = NLAYER * LAYER_BYTES
    out = []
    for name, t in rows:
        out.append(dict(path=name, ms=round(t * 1e3, 2),
                        gbs=round(total / t / 1e9, 1)))
    R["B_paths"] = out
    by = {d["path"]: d["gbs"] for d in out}
    R["B_resident"] = by["GPU, weights resident in VRAM"]
    R["B_streamed"] = by["GPU, weights streamed over PCIe (overlapped)"]
    R["B_streamed_serial"] = by["GPU, weights streamed over PCIe (serial)"]
    R["B_cpu"] = by["CPU, weights in RAM (no accelerator)"]
    R["B_resident_over_streamed"] = round(R["B_resident"] / R["B_streamed"], 1)
    R["B_cpu_over_streamed"] = round(R["B_cpu"] / R["B_streamed"], 2)
    R["B_overlap_gain"] = round(R["B_streamed"] / R["B_streamed_serial"], 2)
    R["B_working_set_gb"] = round(total / 2 ** 30, 2)
    del host, stage
    torch.cuda.empty_cache()
    print(f"B. resident {R['B_resident']} | streamed {R['B_streamed']} | "
          f"CPU {R['B_cpu']} GB/s -> the CPU beats the streaming GPU by "
          f"{R['B_cpu_over_streamed']}x")


# ------------------------------------------------------------- C. the cliff
def section_c():
    free, total = torch.cuda.mem_get_info()
    R["C_vram_total_gb"] = round(total / 2 ** 30, 2)
    R["C_vram_free_gb"] = round(free / 2 ** 30, 2)

    held, sizes = [], []
    ok_gb = 0.0
    try:
        while True:
            held.append(gpu.empty(K * K, dtype=torch.float16))
            ok_gb = len(held) * LAYER_BYTES / 2 ** 30
            sizes.append(round(ok_gb, 2))
            if ok_gb > 12:
                break
    except Exception as e:
        R["C_oom_error"] = type(e).__name__
        R["C_oom_msg"] = str(e).split("\n")[0][:160]
    R["C_max_resident_gb"] = round(ok_gb, 2)
    R["C_layers_held"] = len(held)
    R["C_usable_frac"] = round(ok_gb / R["C_vram_total_gb"], 3)
    del held
    torch.cuda.empty_cache()

    # tokens/s for a model of a given size, from the rates measured in B
    curve = []
    for gb in [0.5, 1, 2, 4, 6, 7, 8, 10, 16, 32, 70, 140]:
        fits = gb <= R["C_max_resident_gb"]
        curve.append(dict(
            model_gb=gb,
            resident=round(R["B_resident"] / gb, 2) if fits else None,
            streamed=round(R["B_streamed"] / gb, 2),
            cpu=round(R["B_cpu"] / gb, 2)))
    R["C_curve"] = curve
    print(f"C. {R['C_max_resident_gb']} GB of weights fit in "
          f"{R['C_vram_total_gb']} GB of VRAM ({R['C_usable_frac'] * 100:.0f}%); "
          f"past that the resident path stops existing")


# --------------------------------------------------------- D. shared or not
def hammer(stop, nthreads):
    """Keep `nthreads` CPU threads streaming memory until told to stop."""
    def work():
        a = torch.empty(1 << 24, dtype=torch.float32)     # 64 MB each
        b = torch.empty_like(a)
        while not stop.is_set():
            b.copy_(a)
    ts = [threading.Thread(target=work, daemon=True) for _ in range(nthreads)]
    for t in ts:
        t.start()
    return ts


def section_d():
    # (i) CPU memory bandwidth vs thread count, measured in section A: one
    #     pool shared by every core does not go N times faster
    scale = R["A_cpu_scaling"]

    # (ii) the control: does CPU traffic slow the discrete GPU down?
    #      Two clocks are needed, because two different things can degrade:
    #      CUDA events measure the GPU alone; the wall clock also contains the
    #      host thread that issues the launches.
    #      One 2 GB kernel rather than many small ones: with 24 us of launch
    #      overhead per kernel, a chain of small launches would measure the
    #      host thread's health, not the GPU's memory system.
    ROWS = (2 << 30) // (K * 2)
    W = gpu.empty(ROWS * K, dtype=torch.float16)
    x = gpu.empty(K, dtype=torch.float16)
    y = gpu.empty(ROWS)
    nbytes = ROWS * K * 2

    def measure():
        gpu.warm_up(rounds=10)
        for _ in range(2):
            gpu_matvec(W, x, y, K)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        t0 = time.perf_counter()
        a.record()
        for _ in range(5):
            gpu_matvec(W, x, y, K)
        b.record()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t0) / 5
        dev = a.elapsed_time(b) / 5 / 1e3
        return nbytes / dev / 1e9, nbytes / wall / 1e9

    # (iii) the same test with the work split into 64 small kernels instead of
    #       one big one. Same bytes, same GPU, 64x the launches.
    small = [gpu.empty(K * K, dtype=torch.float16) for _ in range(24)]
    ys = gpu.empty(K)
    sbytes = len(small) * LAYER_BYTES

    def measure_small():
        gpu.warm_up(rounds=10)
        for _ in range(2):
            for Wi in small:
                gpu_matvec(Wi, x, ys, K)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(5):
            for Wi in small:
                gpu_matvec(Wi, x, ys, K)
        b.record()
        torch.cuda.synchronize()
        return sbytes / (a.elapsed_time(b) / 5 / 1e3) / 1e9

    q_dev, q_wall = measure()
    q_small = measure_small()
    stop = threading.Event()
    ts = hammer(stop, 11)
    time.sleep(0.3)
    b_dev, b_wall = measure()
    b_small = measure_small()
    stop.set()
    for t in ts:
        t.join(timeout=2)
    del W, small
    torch.cuda.empty_cache()
    R["D_small_quiet_gbs"] = round(q_small, 1)
    R["D_small_busy_gbs"] = round(b_small, 1)
    R["D_small_interference"] = round(q_small / b_small, 2)
    R["D_gpu_quiet_dev_gbs"] = round(q_dev, 1)
    R["D_gpu_busy_dev_gbs"] = round(b_dev, 1)
    R["D_gpu_quiet_wall_gbs"] = round(q_wall, 1)
    R["D_gpu_busy_wall_gbs"] = round(b_wall, 1)
    R["D_device_interference"] = round(q_dev / b_dev, 3)
    R["D_wall_interference"] = round(q_wall / b_wall, 2)
    print(f"D. CPU 1->{R['A_cpu_best_threads']} threads = "
          f"{R['A_cpu_1_to_best']}x (one shared pool); "
          f"GPU under 11 busy CPU threads: device clock "
          f"{R['D_device_interference']}x, wall clock "
          f"{R['D_wall_interference']}x, 24 small kernels {R['D_small_interference']}x")


# --------------------------------------------------------------- E. what fits
POOLS = [
    ("this GTX 1070 Ti (VRAM)", 8, 256.3, "discrete"),
    ("this machine (system RAM)", 31, 24.0, "cpu"),
    ("RTX 4090", 24, 1008, "discrete"),
    ("RTX 5090", 32, 1792, "discrete"),
    ("Apple M4 Pro (48 GB)", 36, 273, "unified"),
    ("Apple M4 Max (128 GB)", 96, 546, "unified"),
    ("Apple M3 Ultra (512 GB)", 384, 819, "unified"),
    ("NVIDIA H100", 80, 3350, "discrete"),
]
MODELS = [("Llama 8B", 8e9), ("Llama 70B", 70e9), ("Llama 405B", 405e9)]


def section_e():
    rows = []
    for pname, cap, bw, kind in POOLS:
        for mname, params in MODELS:
            for fmt, bpp in (("fp16", 2.0), ("int4", 0.56)):
                need = params * bpp / 1e9
                rows.append(dict(pool=pname, kind=kind, cap_gb=cap,
                                 bw_gbs=bw, model=mname, fmt=fmt,
                                 need_gb=round(need, 1), fits=need < cap * 0.9,
                                 tok_s=round(0.85 * bw / need, 1)))
    R["E_rows"] = rows
    R["E_apple_wired_limit_note"] = (
        "macOS caps the GPU's share of unified memory (iogpu.wired_limit_mb); "
        "the default leaves roughly 75% of RAM available to the GPU")
    m4 = [r for r in rows if r["pool"].startswith("Apple M4 Max")
          and r["model"] == "Llama 70B" and r["fmt"] == "int4"][0]
    r4090 = [r for r in rows if r["pool"] == "RTX 4090"
             and r["model"] == "Llama 70B" and r["fmt"] == "int4"][0]
    R["E_m4max_70b_int4"] = m4
    R["E_4090_70b_int4"] = r4090
    R["E_4090_bw_over_m4max"] = round(1008 / 546, 2)
    R["E_m4max_cap_over_4090"] = round(96 / 24, 2)
    print(f"E. 70B int4 needs {m4['need_gb']} GB: fits on an M4 Max "
          f"({m4['tok_s']} tok/s), does not fit on a 4090 "
          f"(which is {R['E_4090_bw_over_m4max']}x faster per byte)")


# -------------------------------------------------------------------- plot
def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib missing - skipping the plot)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    p = R["B_paths"]
    names = [d["path"].replace("GPU, weights ", "GPU ")
             .replace("CPU, weights in RAM (no accelerator)", "CPU from RAM")
             .replace("resident in VRAM", "resident")
             .replace("streamed over PCIe ", "streamed ") for d in p]
    vals = [d["gbs"] for d in p]
    cols = ["#2ca02c", "#ff7f0e", "#d62728", "#1f77b4"]
    bars = ax[0].barh(names[::-1], vals[::-1], color=cols[::-1])
    for bar, v in zip(bars, vals[::-1]):
        ax[0].text(v, bar.get_y() + bar.get_height() / 2, f" {v:.0f}",
                   va="center", fontsize=9)
    ax[0].set_xlabel("GB/s of weights consumed")
    ax[0].set_title(f"same {R['B_working_set_gb']} GB of weights,\n"
                    "four ways to reach them")
    ax[0].tick_params(labelsize=8)
    ax[0].grid(alpha=.3, axis="x")

    c = R["C_curve"]
    gbs = [d["model_gb"] for d in c]
    res = [(d["model_gb"], d["resident"]) for d in c if d["resident"]]
    ax[1].loglog([a for a, _ in res], [b for _, b in res], "o-",
                 color="#2ca02c", label="GPU resident")
    ax[1].loglog(gbs, [d["streamed"] for d in c], "s-", color="#ff7f0e",
                 label="GPU streamed (PCIe)")
    ax[1].loglog(gbs, [d["cpu"] for d in c], "^-", color="#1f77b4",
                 label="CPU from RAM")
    ax[1].axvline(R["C_max_resident_gb"], color="#d62728", lw=1.2)
    ax[1].text(R["C_max_resident_gb"] * 1.05, 2,
               f"{R['C_max_resident_gb']} GB\nVRAM cliff", fontsize=8,
               color="#d62728")
    ax[1].set_xlabel("model size (GB of weights)")
    ax[1].set_ylabel("tokens / s (batch 1)")
    ax[1].set_title("the cliff is capacity, not speed")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")

    sel = [r for r in R["E_rows"] if r["model"] == "Llama 70B"
           and r["fmt"] == "int4"]
    names = [r["pool"].replace("Apple ", "").replace("this ", "")
             .replace(" (VRAM)", "").replace(" (system RAM)", " RAM")
             for r in sel]
    vals = [r["tok_s"] if r["fits"] else 0 for r in sel]
    cols = ["#1f77b4" if r["kind"] == "unified" else
            "#7f7f7f" if r["kind"] == "cpu" else "#2ca02c" for r in sel]
    hatch = ["" if r["fits"] else "//" for r in sel]
    order = sorted(range(len(sel)), key=lambda i: vals[i])
    b = ax[2].barh([names[i] for i in order], [max(vals[i], 0.5) for i in order],
                   color=[cols[i] for i in order])
    for bar, i in zip(b, order):
        bar.set_hatch(hatch[i])
        if not sel[i]["fits"]:
            ax[2].text(0.6, bar.get_y() + bar.get_height() / 2,
                       " does not fit", va="center", fontsize=8, color="white")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("tokens/s, 70B int4 (39 GB), batch 1")
    ax[2].set_title("blue = unified memory\nhatched = model does not fit")
    ax[2].tick_params(labelsize=8)
    ax[2].grid(alpha=.3, axis="x")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"wrote {path}")


def main():
    t0 = time.time()
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    R["runtime_s"] = round(time.time() - t0, 1)
    R["device"] = gpu.device_note()

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in R.items():
            if not isinstance(v, (list, dict)):
                w.writerow([k, v])
        w.writerow([])
        w.writerow(["path", "ms", "GB/s"])
        for d in R["B_paths"]:
            w.writerow([d["path"], d["ms"], d["gbs"]])
        w.writerow([])
        w.writerow(["model_gb", "resident_tok_s", "streamed_tok_s", "cpu_tok_s"])
        for d in R["C_curve"]:
            w.writerow([d["model_gb"], d["resident"], d["streamed"], d["cpu"]])
        w.writerow([])
        w.writerow(["cpu_threads", "GB/s"])
        for d in R["A_cpu_scaling"]:
            w.writerow([d["threads"], d["gbs"]])
        w.writerow([])
        w.writerow(["pool", "kind", "cap_gb", "bw_gbs", "model", "fmt",
                    "need_gb", "fits", "tok_s"])
        for d in R["E_rows"]:
            w.writerow([d[k] for k in ("pool", "kind", "cap_gb", "bw_gbs",
                                       "model", "fmt", "need_gb", "fits",
                                       "tok_s")])
    plot(os.path.join(OUT, "unified_memory.png"))
    print(f"total {R['runtime_s']} s")


if __name__ == "__main__":
    main()
