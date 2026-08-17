"""Project 43 -- Hardware comparison.

Run the same model, with the same weights, on the two processors this machine
has: a GTX 1070 Ti and an i7-8700K.  Then explain the gap from the spec sheets.

  python3 run.py          # full run, ~8 minutes
  python3 run.py --plot   # redraw the figure from the committed findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "37-roofline-plot-for-your-engine"))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import triton  # noqa: E402

import enginelib as E  # noqa: E402
from cpu_engine import CPUEngine  # noqa: E402

OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

CTX = 512
BATCHES = [1, 4, 16]
PREFILL = 256

# Spec-sheet numbers, for the "explain the gap" half of the project.
SPECS = {
    "GTX 1070 Ti": {
        "peak_fp32_gflops": 8186,      # 2432 cores x 2 flops x 1.683 GHz boost
        "spec_bw_gbs": 256.0,          # 256-bit GDDR5 @ 8 Gbps
        "tdp_w": 180,
        "launched": 2017,
    },
    "i7-8700K": {
        # 6 cores x 8 AVX2 lanes x 2 (FMA) x 2 FMA ports x 4.3 GHz all-core
        "peak_fp32_gflops": 825,
        "spec_bw_gbs": 41.6,           # dual-channel DDR4-2600
        "tdp_w": 95,
        "launched": 2017,
    },
}


# --------------------------------------------------------------- ceilings
def gpu_ceilings() -> dict:
    n = 1 << 24
    a = torch.empty(n, device=E.DEV)
    b = torch.empty(n, device=E.DEV)
    t = E.gpu_time(lambda: E.k_copy[(triton.cdiv(n, 1024),)](a, b, n, BLOCK=1024), reps=50)
    bw = 2 * n * 4 / t / 1e6
    m = 2048
    x = torch.empty(m * m, device=E.DEV)
    w = torch.empty(m * m, device=E.DEV)
    y = torch.empty(m * m, device=E.DEV)
    best = 0.0
    for BM, BN, BK, nw in ((128, 128, 32, 8), (128, 64, 32, 4), (64, 64, 32, 4)):
        g = (triton.cdiv(m, BM) * triton.cdiv(m, BN),)
        tt = E.gpu_time(lambda: E.k_matmul[g](x, w, y, m, m, m, BM=BM, BN=BN,
                                              BK=BK, GROUP_M=8, num_warps=nw), reps=20)
        best = max(best, 2 * m ** 3 / tt / 1e6)
    return {"copy_gbs": bw, "gemm_gflops": best}


def cpu_ceilings(threads: int) -> dict:
    torch.set_num_threads(threads)
    n = 1 << 24
    a = torch.randn(n)
    b = torch.empty(n)

    def t(fn, reps):
        for _ in range(3):
            fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    tc = t(lambda: b.copy_(a), 20)
    bw = 2 * n * 4 / tc / 1e9
    m = 2048
    x = torch.randn(m, m)
    w = torch.randn(m, m)
    tm = t(lambda: torch.mm(x, w), 10)
    return {"threads": threads, "copy_gbs": bw, "gemm_gflops": 2 * m ** 3 / tm / 1e9}


# ------------------------------------------------------------ same model?
def agreement(cfg) -> dict:
    """Same weights, same input, both machines.  Do the logits match?"""
    B, T = 1, 32
    gpu = E.Engine(cfg, max_batch=B, max_seq=CTX, max_tokens=B * T)
    gpu.prefill(B, T)
    torch.cuda.synchronize()
    gl = gpu.logits[:B * cfg.vocab].cpu().reshape(B, cfg.vocab).clone()
    del gpu
    torch.cuda.empty_cache()

    cpu = CPUEngine(cfg, max_batch=B, max_seq=CTX, max_tokens=B * T)
    cl = cpu.prefill(B, T)
    rel = float((gl - cl).abs().max() / cl.abs().max())
    same_top = int(gl.argmax(-1)[0]) == int(cl.argmax(-1)[0])
    del cpu
    return {"max_rel_diff": rel, "same_argmax": same_top,
            "gpu_top": int(gl.argmax(-1)[0]), "cpu_top": int(cl.argmax(-1)[0])}


# --------------------------------------------------------------- the runs
def gpu_runs(cfg) -> list:
    eng = E.Engine(cfg, max_batch=max(BATCHES), max_seq=CTX,
                   max_tokens=max(max(BATCHES), PREFILL))
    eng.ctx_hint = CTX
    rows = []
    for B in BATCHES:
        eng.set_len(CTX - 1)
        g = E.Graph(lambda: eng.decode_step(B, advance=False))
        ms = E.gpu_time(g.replay, reps=30)
        g.close()
        rows.append({"machine": "gpu", "phase": "decode", "batch": B, "ms": ms,
                     "tok_s": B / ms * 1e3})
        print(f"   gpu decode  B={B:3d}  {ms:8.3f} ms  {B/ms*1e3:8.1f} tok/s")
    ms = E.gpu_time(lambda: eng.prefill(1, PREFILL), reps=10)
    rows.append({"machine": "gpu", "phase": "prefill", "batch": 1, "ms": ms,
                 "tok_s": PREFILL / ms * 1e3})
    print(f"   gpu prefill T={PREFILL}  {ms:8.3f} ms  {PREFILL/ms*1e3:8.1f} tok/s")
    del eng
    torch.cuda.empty_cache()
    return rows


def cpu_runs(cfg, threads: int) -> list:
    torch.set_num_threads(threads)
    eng = CPUEngine(cfg, max_batch=max(BATCHES), max_seq=CTX,
                    max_tokens=max(max(BATCHES), PREFILL))
    rows = []
    for B in BATCHES:
        eng.decode_step(B, CTX)
        t0 = time.perf_counter()
        reps = 5
        for _ in range(reps):
            eng.decode_step(B, CTX)
        ms = (time.perf_counter() - t0) / reps * 1e3
        rows.append({"machine": "cpu", "phase": "decode", "batch": B, "ms": ms,
                     "tok_s": B / ms * 1e3, "threads": threads})
        print(f"   cpu decode  B={B:3d}  {ms:8.3f} ms  {B/ms*1e3:8.1f} tok/s")
    eng.prefill(1, PREFILL)
    t0 = time.perf_counter()
    for _ in range(3):
        eng.prefill(1, PREFILL)
    ms = (time.perf_counter() - t0) / 3 * 1e3
    rows.append({"machine": "cpu", "phase": "prefill", "batch": 1, "ms": ms,
                 "tok_s": PREFILL / ms * 1e3, "threads": threads})
    print(f"   cpu prefill T={PREFILL}  {ms:8.3f} ms  {PREFILL/ms*1e3:8.1f} tok/s")
    del eng
    return rows


def thread_sweep(cfg) -> list:
    rows = []
    for th in (1, 2, 4, 6, 12):
        torch.set_num_threads(th)
        eng = CPUEngine(cfg, max_batch=1, max_seq=CTX, max_tokens=1)
        eng.decode_step(1, CTX)
        t0 = time.perf_counter()
        for _ in range(4):
            eng.decode_step(1, CTX)
        ms = (time.perf_counter() - t0) / 4 * 1e3
        rows.append({"threads": th, "ms": ms, "tok_s": 1 / ms * 1e3})
        print(f"   {th:3d} threads  {ms:8.2f} ms  {1/ms*1e3:6.2f} tok/s")
        del eng
    return rows


# ----------------------------------------------------------------- PCIe
def pcie(cfg) -> dict:
    nbytes = cfg.weight_bytes()
    host = torch.empty(nbytes // 4)
    host_pin = torch.empty(nbytes // 4, pin_memory=True)
    dev = torch.empty(nbytes // 4, device=E.DEV)

    def t(fn, reps=5):
        fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / reps

    up = t(lambda: dev.copy_(host))
    up_pin = t(lambda: dev.copy_(host_pin))
    return {"weight_bytes": nbytes,
            "h2d_pageable_gbs": nbytes / up / 1e9,
            "h2d_pinned_gbs": nbytes / up_pin / 1e9,
            "load_ms": up_pin * 1e3,
            "streamed_tok_s": 1.0 / up_pin}


# ---------------------------------------------------------------- power
def gpu_power(cfg) -> dict:
    """Sample nvidia-smi while the GPU is decoding flat out."""
    eng = E.Engine(cfg, max_batch=1, max_seq=CTX, max_tokens=1)
    eng.ctx_hint = CTX
    eng.set_len(CTX - 1)
    g = E.Graph(lambda: eng.decode_step(1, advance=False))
    idle = float(subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().split("\n")[0])
    samples = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 6.0:
        for _ in range(50):
            g.replay()
        torch.cuda.synchronize()
        samples.append(float(subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.strip().split("\n")[0]))
    ms = E.gpu_time(g.replay, reps=30)
    g.close()
    del eng
    torch.cuda.empty_cache()
    busy = sorted(samples)[len(samples) // 2]
    return {"idle_w": idle, "busy_w": busy, "samples": len(samples),
            "ms_per_token": ms, "tokens_per_joule": (1000 / ms) / busy}


# ---------------------------------------------------------------- plotting
def plot(f: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    a = ax[0]
    rows = f["runs"]
    dec = {m: [r for r in rows if r["machine"] == m and r["phase"] == "decode"]
           for m in ("gpu", "cpu")}
    a.plot([r["batch"] for r in dec["gpu"]], [r["tok_s"] for r in dec["gpu"]], "o-",
           color="#c0392b", label="GTX 1070 Ti")
    a.plot([r["batch"] for r in dec["cpu"]], [r["tok_s"] for r in dec["cpu"]], "s-",
           color="#1f6f8b", label=f"i7-8700K ({f['best_threads']} threads)")
    for g, c in zip(dec["gpu"], dec["cpu"]):
        a.annotate(f"{g['tok_s']/c['tok_s']:.1f}x", (g["batch"], g["tok_s"]),
                   textcoords="offset points", xytext=(-30, 7), fontsize=8, color="0.3")
    a.set_xscale("log", base=2)
    a.set_yscale("log", base=2)
    a.set_xlabel("batch size")
    a.set_ylabel("decode tokens/s")
    a.set_title("Decode: the gap is the bandwidth ratio")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, which="both", lw=0.4)

    a = ax[1]
    labels = ["memory\nbandwidth", "fp32\nmatmul", "decode\n(batch 1)", "prefill\n(256 tok)"]
    pred = [f["ceilings"]["gpu"]["copy_gbs"] / f["ceilings"]["cpu"]["copy_gbs"],
            f["ceilings"]["gpu"]["gemm_gflops"] / f["ceilings"]["cpu"]["gemm_gflops"],
            0, 0]
    meas = [0, 0,
            dec["gpu"][0]["tok_s"] / dec["cpu"][0]["tok_s"],
            [r for r in rows if r["machine"] == "gpu" and r["phase"] == "prefill"][0]["tok_s"] /
            [r for r in rows if r["machine"] == "cpu" and r["phase"] == "prefill"][0]["tok_s"]]
    colors = ["0.5", "0.5", "#c0392b", "#7d3c98"]
    vals = [pred[0], pred[1], meas[2], meas[3]]
    a.bar(labels, vals, color=colors)
    for i, v in enumerate(vals):
        a.text(i, v + 0.3, f"{v:.1f}x", ha="center", fontsize=10)
    a.axhline(pred[0], color="#c0392b", ls="--", lw=1,
              label=f"bandwidth ratio {pred[0]:.1f}x")
    a.axhline(pred[1], color="#7d3c98", ls=":", lw=1,
              label=f"matmul ratio {pred[1]:.1f}x")
    a.set_ylabel("GPU / CPU")
    a.set_title("Each phase follows its own ceiling")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, axis="y", lw=0.4)

    a = ax[2]
    rows = f["threads"]
    a.plot([r["threads"] for r in rows], [r["tok_s"] for r in rows], "o-",
           color="#1f6f8b", label="measured")
    base = rows[0]["tok_s"]
    a.plot([r["threads"] for r in rows], [base * r["threads"] for r in rows], "--",
           color="0.6", label="perfect scaling")
    a.set_xlabel("CPU threads")
    a.set_ylabel("decode tokens/s (batch 1)")
    a.set_title("The CPU stops scaling at its memory wall")
    a.legend(fontsize=8)
    a.grid(alpha=0.25, lw=0.4)

    fig.suptitle("Project 43 - the same 152M model on a 2017 GPU and a 2017 CPU",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "hardware_comparison.png"), dpi=125)
    print("wrote", os.path.join(OUT, "hardware_comparison.png"))


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    path = os.path.join(OUT, "findings.json")
    if args.plot:
        plot(json.load(open(path)))
        return

    cfg = E.Config()
    idle_w = float(subprocess.run(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True).stdout.strip().split("\n")[0])
    print(f"A. measured ceilings (GPU idle draw {idle_w:.1f} W)")
    gc = gpu_ceilings()
    cc12 = cpu_ceilings(12)
    cc6 = cpu_ceilings(6)
    cc = cc12 if cc12["gemm_gflops"] > cc6["gemm_gflops"] else cc6
    print(f"   gpu  copy {gc['copy_gbs']:7.1f} GB/s  gemm {gc['gemm_gflops']/1000:6.2f} TFLOP/s")
    print(f"   cpu  copy {cc['copy_gbs']:7.1f} GB/s  gemm {cc['gemm_gflops']/1000:6.2f} TFLOP/s"
          f"  ({cc['threads']} threads)")

    print("B. is it the same model?")
    agree = agreement(cfg)
    print("  ", agree)

    print("C. CPU thread sweep")
    threads = thread_sweep(cfg)
    best_threads = min(threads, key=lambda r: r["ms"])["threads"]
    print(f"   -> using {best_threads} threads for the comparison")

    print("D. the runs")
    runs = gpu_runs(cfg) + cpu_runs(cfg, best_threads)

    print("E. the PCIe tax")
    pc = pcie(cfg)
    print(f"   {pc['h2d_pageable_gbs']:.2f} GB/s pageable, {pc['h2d_pinned_gbs']:.2f} pinned; "
          f"loading the model takes {pc['load_ms']:.0f} ms")

    print("F. power")
    pw = gpu_power(cfg)
    pw["idle_w"] = idle_w
    print(f"   idle {pw['idle_w']:.1f} W, busy {pw['busy_w']:.1f} W, "
          f"{pw['tokens_per_joule']:.2f} tokens/joule")

    f = {"specs": SPECS, "ceilings": {"gpu": gc, "cpu": cc, "cpu12": cc12, "cpu6": cc6},
         "agreement": agree, "runs": runs, "threads": threads, "pcie": pc,
         "power": pw, "best_threads": best_threads, "model": {"params": cfg.n_params(),
                                "weight_bytes": cfg.weight_bytes()},
         "ctx": CTX, "prefill_tokens": PREFILL}
    json.dump(f, open(path, "w"), indent=1)
    print("wrote", path)
    plot(f)


if __name__ == "__main__":
    main()
