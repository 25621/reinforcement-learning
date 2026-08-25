"""Project 24 - AMD's accelerator, from a machine that has none.

  A. the port      - hipify every CUDA file in this guide; count what changes
  B. the proof     - compile the ported source back through a HIP->CUDA shim
                     and require byte-identical output
  C. the landmines - what a rename cannot fix, with the warp-size one measured
  D. decode        - the operation MI300X is bought for, measured locally
  E. projection    - the same model applied to hardware we cannot touch
"""

import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import torch
import triton

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "18-triton-softmax")))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import gpu                                                       # noqa: E402
from hipify import hipify_text, landmines, line_stats            # noqa: E402
from decode import decode_step, bytes_moved, bytes_dram, flops               # noqa: E402

R = {}
PROJ = os.path.abspath(os.path.join(HERE, ".."))

# spec sheets, 2026 (see the guide's Phase 0 table). bw = GB/s, cap = GB,
# tf = bf16/fp16 dense TFLOP/s, usd = representative cloud $/hour.
ACCEL = [
    ("NVIDIA A100 80GB", 2039, 80, 312, 1.50),
    ("NVIDIA H100 SXM", 3350, 80, 990, 3.00),
    ("NVIDIA H200 SXM", 4800, 141, 990, 5.00),
    ("NVIDIA B200 SXM", 8000, 192, 2250, 8.00),
    ("AMD MI300X", 5300, 192, 1300, 3.00),
    ("AMD MI325X", 6000, 256, 1300, 4.00),
    ("Google TPU v5p", 2765, 95, 459, 4.20),
    ("Apple M4 Max (128GB)", 546, 96, 17, 0.00),
    ("NVIDIA RTX 4090", 1008, 24, 165, 2.00),
]


# ------------------------------------------------------------------ A. port
def section_a():
    files = sorted(glob.glob(os.path.join(PROJ, "*", "*.cu")))
    rows, total_lines, total_changed, total_subs = [], 0, 0, 0
    for f in files:
        src = open(f).read()
        port, counts = hipify_text(src)
        st = line_stats(src, port)
        subs = sum(counts.values())
        rows.append(dict(file=os.path.relpath(f, PROJ), lines=st["lines"],
                         changed=st["changed"], subs=subs,
                         unchanged_pct=st["unchanged_pct"]))
        total_lines += st["lines"]
        total_changed += st["changed"]
        total_subs += subs
    R["A_files"] = rows
    R["A_n_files"] = len(files)
    R["A_total_lines"] = total_lines
    R["A_total_changed"] = total_changed
    R["A_total_subs"] = total_subs
    R["A_unchanged_pct"] = round(100.0 * (total_lines - total_changed) / total_lines, 1)
    print(f"A. hipified {len(files)} CUDA files, {total_lines} lines: "
          f"{total_subs} token substitutions on {total_changed} lines "
          f"({R['A_unchanged_pct']}% of lines untouched)")


# ----------------------------------------------------------------- B. proof
def nvcc(src, exe, extra=()):
    cmd = ["nvcc", "-O3", "-arch=sm_61", *extra, "-o", exe, src]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip()[:400]


def section_b():
    if shutil.which("nvcc") is None:
        R["B_skipped"] = "nvcc not found"
        print("B. skipped (no nvcc)")
        return
    orig = os.path.join(HERE, "warpsize.cu")
    ported = os.path.join(OUT, "warpsize.hip.cpp")
    src = open(orig).read()
    port, counts = hipify_text(src)
    open(ported, "w").write(port)
    R["B_subs"] = sum(counts.values())
    R["B_uses_hip_header"] = "hip/hip_runtime.h" in port
    R["B_still_has_cuda"] = "cuda" in port.lower().replace("cuda error", "")

    ok1, err1 = nvcc(orig, os.path.join(OUT, "ws_cuda"))
    ok2, err2 = nvcc(ported, os.path.join(OUT, "ws_hip"),
                     extra=("-x", "cu", "-I", os.path.join(HERE, "hipshim")))
    R["B_cuda_compiles"] = ok1
    R["B_hip_compiles"] = ok2
    R["B_hip_error"] = err2 if not ok2 else ""
    if not (ok1 and ok2):
        print(f"B. compile failed: {err1 or err2}")
        return

    a = subprocess.run([os.path.join(OUT, "ws_cuda")], capture_output=True,
                       text=True).stdout
    b = subprocess.run([os.path.join(OUT, "ws_hip")], capture_output=True,
                       text=True).stdout
    # timings differ run to run; the structural output must not
    key = lambda t: [ln.split(",")[:2] for ln in t.strip().splitlines()]
    R["B_same_structure"] = key(a) == key(b)
    R["B_same_props"] = ([ln for ln in a.splitlines() if ln.startswith("prop")]
                         == [ln for ln in b.splitlines() if ln.startswith("prop")])
    R["B_cuda_out"] = a
    print(f"B. ported file: {R['B_subs']} substitutions, compiles through the "
          f"shim: {ok2}, identical output structure: {R['B_same_structure']}")


# ------------------------------------------------------------ C. landmines
def section_c():
    files = sorted(glob.glob(os.path.join(PROJ, "*", "*.cu")))
    agg = {}
    for f in files:
        for name, d in landmines(open(f).read()).items():
            e = agg.setdefault(name, dict(count=0, files=[], why=d["why"]))
            e["count"] += d["count"]
            e["files"].append(os.path.relpath(f, PROJ))
    R["C_landmines"] = agg
    R["C_landmine_total"] = sum(d["count"] for d in agg.values())

    # the measured one: divergence granularity vs warp size
    exe = os.path.join(OUT, "ws_cuda")
    if os.path.exists(exe):
        out = subprocess.run([exe], capture_output=True, text=True).stdout
        div, pure, props = {}, None, {}
        for ln in out.strip().splitlines():
            p = ln.split(",")
            if p[0] == "div":
                div[int(p[1])] = float(p[2])
            elif p[0] == "pure":
                pure = float(p[2])
            elif p[0] == "prop":
                props[p[1]] = p[2]
        R["C_warpSize"] = int(props.get("warpSize", 32))
        R["C_pure_ms"] = pure
        R["C_div"] = {str(k): v for k, v in sorted(div.items())}
        R["C_div_ratio"] = {str(k): round(v / div[256], 3) for k, v in sorted(div.items())}
        R["C_cost_below_warp"] = round(div[16] / div[32], 2)
        print(f"C. {R['C_landmine_total']} landmines across {len(files)} files; "
              f"divergence at G=16 costs {R['C_cost_below_warp']}x vs G=32 "
              f"(warpSize={R['C_warpSize']})")
    else:
        print(f"C. {R['C_landmine_total']} landmines (no binary to time)")


# --------------------------------------------------------------- D. decode
CFGS = [(128, 32, 16), (128, 32, 64), (128, 64, 16), (64, 64, 16)]


def best_time(W, X, Y):
    best = None
    for bn, bk, bb in CFGS:
        try:
            ms = gpu.bench(lambda: decode_step(W, X, Y, BLOCK_N=bn,
                                               BLOCK_K=bk, BLOCK_B=bb), reps=20)
        except Exception:
            continue
        if best is None or ms < best[0]:
            best = (ms, (bn, bk, bb))
    return best


def section_d():
    N = K = 4096
    g = torch.Generator().manual_seed(0)
    Wc = (torch.randn(N, K, generator=g) * 0.02).half()
    W = Wc.cuda()

    # correctness first
    Xc = torch.randn(8, K, generator=g).half()
    X, Y = Xc.cuda(), gpu.empty(8, N)
    decode_step(W, X, Y)
    torch.cuda.synchronize()
    ref = Xc.double() @ Wc.double().T
    R["D_rel_err"] = float((Y.cpu().double() - ref).abs().max() / ref.abs().max())

    sweep = []
    for B in [1, 16, 32, 64, 128, 256, 512]:
        Xc = torch.randn(B, K, generator=g).half()
        X, Y = Xc.cuda(), gpu.empty(B, N)
        ms, cfg = best_time(W, X, Y)
        bb = max(16, min(cfg[2], triton.next_power_of_2(B)))
        sweep.append(dict(B=B, ms=round(ms, 4), cfg=list(cfg),
                          weight_passes=-(-B // bb),
                          gbs_ideal=round(bytes_moved(B, N, K) / (ms * 1e6), 1),
                          gbs_kernel=round(bytes_dram(B, N, K, bb) / (ms * 1e6), 1),
                          gflops=round(flops(B, N, K) / (ms * 1e6), 0),
                          tok_s=round(B / ms * 1e3, 0),
                          ai=round(flops(B, N, K) / bytes_moved(B, N, K), 2)))
    R["D_sweep"] = sweep
    b1 = sweep[0]
    R["D_b1_gbs"] = b1["gbs_kernel"]
    R["D_b1_frac_spec"] = round(b1["gbs_kernel"] / gpu.PEAK_BW, 3)
    R["D_b1_frac_measured_read"] = round(b1["gbs_kernel"] / gpu.BW_READ, 3)
    R["D_eff"] = R["D_b1_frac_spec"]          # used for every projection in E

    # the roofline prediction vs what the kernel does
    ridge = gpu.PEAK_FLOPS_FP32 / gpu.PEAK_BW
    R["D_ridge_flop_per_byte"] = round(ridge, 1)
    R["D_batch_star"] = round(ridge * 2 / 2, 1)    # fp16 weights: B* = ridge
    pred_free = [s for s in sweep if s["B"] <= R["D_batch_star"]]
    R["D_pred_tok_s_at_B32"] = round(32 / b1["ms"] * 1e3, 0)
    got32 = [s for s in sweep if s["B"] == 32][0]
    R["D_got_tok_s_at_B32"] = got32["tok_s"]
    R["D_batching_shortfall"] = round(R["D_pred_tok_s_at_B32"] / got32["tok_s"], 2)
    R["D_peak_gflops_seen"] = max(s["gflops"] for s in sweep)
    R["D_frac_of_peak_flops"] = round(R["D_peak_gflops_seen"] / gpu.PEAK_FLOPS_FP32, 3)

    # what continuous batching is worth: 64 separate decodes vs one batched one
    Xc = torch.randn(1, K, generator=g).half()
    X1, Y1 = Xc.cuda(), gpu.empty(1, N)
    ms1 = gpu.bench(lambda: decode_step(W, X1, Y1, BLOCK_N=128, BLOCK_K=64),
                    reps=20)
    got64 = [s for s in sweep if s["B"] == 64][0]
    R["D_serial_64_ms"] = round(ms1 * 64, 3)
    R["D_batched_64_ms"] = got64["ms"]
    R["D_batching_speedup"] = round(ms1 * 64 / got64["ms"], 2)
    print(f"D. B=1 reaches {b1['gbs_kernel']} GB/s = {R['D_b1_frac_spec'] * 100:.1f}% of "
          f"spec peak (decode is bandwidth); batching 64 is worth "
          f"{R['D_batching_speedup']}x, but the roofline promised "
          f"{R['D_batching_shortfall']}x more at B=32")


# ------------------------------------------------------------ E. projection
def section_e():
    eff = R.get("D_eff", 0.85)
    P = 70e9                       # Llama-70B class
    rows = []
    for name, bw, cap, tf, usd in ACCEL:
        for fmt, bpp in (("fp16", 2.0), ("int8", 1.0), ("int4", 0.56)):
            need = P * bpp / 1e9 + 8      # + ~8 GB of KV cache and activations
            n = max(1, -(-int(need) // cap))
            agg_bw = bw * n
            tok = eff * agg_bw * 1e9 / (P * bpp)
            bstar = (tf * 1e12 / (bw * 1e9)) * bpp / 2
            rows.append(dict(accel=name, fmt=fmt, need_gb=round(need, 1),
                             gpus=n, tok_s=round(tok, 1),
                             tok_s_per_dollar=round(tok / (usd * n), 1) if usd else None,
                             batch_star=round(bstar, 0),
                             tflops=tf, bw=bw, cap=cap))
    R["E_rows"] = rows

    def pick(a, f):
        return [r for r in rows if r["accel"] == a and r["fmt"] == f][0]

    mi, h1 = pick("AMD MI300X", "fp16"), pick("NVIDIA H100 SXM", "fp16")
    R["E_mi300x_fp16"] = mi
    R["E_h100_fp16"] = h1
    R["E_h100_gpus_needed"] = h1["gpus"]
    R["E_h100_over_mi300x_speed"] = round(h1["tok_s"] / mi["tok_s"], 2)
    R["E_mi300x_over_h100_value"] = round(
        mi["tok_s_per_dollar"] / h1["tok_s_per_dollar"], 2)
    R["E_flops_ratio"] = round(1300 / 990, 2)
    R["E_bw_ratio"] = round(5300 / 3350, 2)
    print(f"E. 70B fp16: MI300X {mi['tok_s']} tok/s on {mi['gpus']} GPU; "
          f"H100 {h1['tok_s']} tok/s on {h1['gpus']} "
          f"({R['E_h100_over_mi300x_speed']}x faster, "
          f"{R['E_mi300x_over_h100_value']}x worse per dollar)")


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

    s = R["D_sweep"]
    Bs = [d["B"] for d in s]
    ax[0].loglog(Bs, [d["tok_s"] for d in s], "o-", color="#1f77b4",
                 label="measured")
    ideal = [s[0]["tok_s"] * b for b in Bs]
    ax[0].loglog(Bs, ideal, "--", color="0.6", label="if batching were free")
    ax[0].axvline(R["D_batch_star"], color="#d62728", lw=1,
                  label=f"roofline B*={R['D_batch_star']:.0f}")
    ax[0].set_xlabel("batch size B")
    ax[0].set_ylabel("tokens / s")
    ax[0].set_title("decode throughput vs batch\n(4096x4096 fp16 weights)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, which="both")

    ax[1].semilogx(Bs, [d["gbs_kernel"] for d in s], "o-", color="#2ca02c",
                   label="GB/s")
    ax[1].axhline(gpu.PEAK_BW, ls=":", color="0.5")
    ax[1].text(Bs[1], gpu.PEAK_BW * 1.02, "DRAM spec peak", fontsize=8)
    a2 = ax[1].twinx()
    a2.semilogx(Bs, [d["gflops"] for d in s], "s--", color="#ff7f0e")
    a2.set_ylabel("GFLOP/s", color="#ff7f0e")
    ax[1].set_xlabel("batch size B")
    ax[1].set_ylabel("GB/s", color="#2ca02c")
    ax[1].set_title("B=1 is a bandwidth test\nbig B is a FLOPs test")
    ax[1].grid(alpha=.3)

    names, toks, cols = [], [], []
    for r in R["E_rows"]:
        if r["fmt"] != "fp16":
            continue
        names.append(r["accel"].replace("NVIDIA ", "").replace("AMD ", "")
                     .replace(" SXM", "").replace(" (128GB)", "")
                     + (f" x{r['gpus']}" if r["gpus"] > 1 else ""))
        toks.append(r["tok_s"])
        cols.append("#d62728" if "MI" in r["accel"] else
                    "#7f7f7f" if "M4" in r["accel"] or "4090" in r["accel"]
                    else "#1f77b4")
    order = sorted(range(len(toks)), key=lambda i: toks[i])
    ax[2].barh([names[i] for i in order], [toks[i] for i in order],
               color=[cols[i] for i in order])
    ax[2].set_xlabel("projected tokens/s, 70B fp16, batch 1")
    ax[2].set_title(f"projection at {R['D_eff'] * 100:.0f}% of spec bandwidth\n"
                    "(red = AMD, blue = NVIDIA/TPU, grey = consumer)")
    ax[2].grid(alpha=.3, axis="x")
    ax[2].tick_params(labelsize=8)

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
        w.writerow(["B", "ms", "weight_passes", "GB/s (kernel)", "GB/s (ideal)", "GFLOP/s", "tok/s", "AI"])
        for d in R["D_sweep"]:
            w.writerow([d["B"], d["ms"], d["weight_passes"], d["gbs_kernel"], d["gbs_ideal"], d["gflops"], d["tok_s"], d["ai"]])
        w.writerow([])
        w.writerow(["accel", "fmt", "need_gb", "gpus", "tok_s",
                    "tok_s_per_dollar", "batch_star"])
        for d in R["E_rows"]:
            w.writerow([d["accel"], d["fmt"], d["need_gb"], d["gpus"],
                        d["tok_s"], d["tok_s_per_dollar"], d["batch_star"]])
    plot(os.path.join(OUT, "amd_mi300.png"))
    print(f"total {R['runtime_s']} s")


if __name__ == "__main__":
    main()
