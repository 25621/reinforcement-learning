"""Project 26 - the same four workloads on three real backends.

  A. the devices  - each one's own roofline, measured rather than quoted
  B. the bake-off - 4 workloads x 3 backends, same maths, same answers
  C. the ranking  - who wins depends entirely on arithmetic intensity
  D. honesty      - what the GPU column looks like once PCIe is included
  E. the zoo      - ridge points across the accelerators you might rent

Backends:
  numpy  - OpenBLAS on 6 cores. No compiler, one op at a time.
  XLA    - jax.jit on the CPU. Same silicon as numpy, different compiler.
  Triton - the GTX 1070 Ti. Different silicon entirely.
"""

import csv
import json
import os
import sys
import time

import numpy as np
import torch
import triton

os.environ.setdefault("JAX_PLATFORMS", "cpu")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
for _p in ("18-triton-softmax", "19-triton-matmul"):
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", _p)))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

import gpu                                                       # noqa: E402
from matmul import matmul                                        # noqa: E402
import kernels                                                   # noqa: E402

R = {}
N_ELEM = 1 << 24               # 16.8M floats = 67 MB per array
SM_ROWS, SM_COLS = 4096, 1024
MM_N = 2048
CPU_THREADS = 6                # measured best in project 25


def cpu_bench(fn, reps=5, rounds=3):
    fn()
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        out.append((time.perf_counter() - t0) / reps)
    return min(out) * 1e3


def jax_bench(fn, reps=5, rounds=3):
    jax.block_until_ready(fn())
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(fn())
        out.append((time.perf_counter() - t0) / reps)
    return min(out) * 1e3


# ------------------------------------------------------------- the workloads
# name -> (FLOPs, bytes moved) for one call
def work_specs():
    n = N_ELEM
    return {
        "axpy  d=a*b+c": (2.0 * n, 4.0 * n * 4),
        "chain 6 ops": (8.0 * n, 4.0 * n * 4),
        "softmax rows": (5.0 * SM_ROWS * SM_COLS, 2.0 * SM_ROWS * SM_COLS * 4),
        f"matmul {MM_N}^3": (2.0 * MM_N ** 3, 3.0 * MM_N ** 2 * 4),
    }


# ------------------------------------------------------------------ A. peaks
def section_a():
    torch.set_num_threads(CPU_THREADS)

    # CPU DRAM bandwidth: a big streaming copy
    a = np.ones(1 << 26, dtype=np.float32)
    b = np.empty_like(a)
    ms = cpu_bench(lambda: np.copyto(b, a), reps=5)
    R["A_cpu_bw_gbs"] = round(2 * a.nbytes / (ms * 1e6), 1)

    # CPU peak FLOPs: the best square matmul either CPU backend can do.
    # Both are tried, because the roofline is a property of the silicon and
    # taking the slower library's number would flatter every later efficiency.
    x = np.random.default_rng(0).standard_normal((2048, 2048), dtype=np.float32)
    ms_np = cpu_bench(lambda: x @ x, reps=3)
    jx = jnp.asarray(x)
    f = jax.jit(lambda v: jnp.dot(v, v))
    ms_xla = jax_bench(lambda: f(jx), reps=3)
    R["A_cpu_gflops_numpy"] = round(2 * 2048 ** 3 / (ms_np * 1e6), 0)
    R["A_cpu_gflops_xla"] = round(2 * 2048 ** 3 / (ms_xla * 1e6), 0)
    R["A_cpu_gflops"] = max(R["A_cpu_gflops_numpy"], R["A_cpu_gflops_xla"])

    R["A_gpu_bw_gbs"] = gpu.BW_READ
    R["A_gpu_bw_spec"] = gpu.PEAK_BW
    R["A_gpu_gflops"] = gpu.PEAK_FLOPS_FP32
    R["A_cpu_ridge"] = round(R["A_cpu_gflops"] / R["A_cpu_bw_gbs"], 1)
    R["A_gpu_ridge"] = round(R["A_gpu_gflops"] / R["A_gpu_bw_spec"], 1)
    R["A_gpu_over_cpu_flops"] = round(R["A_gpu_gflops"] / R["A_cpu_gflops"], 1)
    R["A_gpu_over_cpu_bw"] = round(R["A_gpu_bw_spec"] / R["A_cpu_bw_gbs"], 1)
    print(f"A. CPU {R['A_cpu_gflops']:.0f} GFLOP/s / {R['A_cpu_bw_gbs']} GB/s "
          f"(ridge {R['A_cpu_ridge']}) | GPU {R['A_gpu_gflops']:.0f} / "
          f"{R['A_gpu_bw_spec']} (ridge {R['A_gpu_ridge']})")


# ---------------------------------------------------------------- B. bake-off
def section_b():
    rng = np.random.default_rng(0)
    n = N_ELEM
    a = rng.standard_normal(n, dtype=np.float32)
    b = rng.standard_normal(n, dtype=np.float32)
    c = rng.standard_normal(n, dtype=np.float32)
    sm = rng.standard_normal((SM_ROWS, SM_COLS), dtype=np.float32)
    mm = rng.standard_normal((MM_N, MM_N), dtype=np.float32)

    def make_chain(xp):
        """The same six operations, written once, for whichever array library.

        numpy and jax.numpy expose the same names, so this is literally the
        same source running on both - which is what makes the comparison a
        comparison of backends rather than of two different programs.
        """
        def chain(a, b, c):
            t = a * b
            t = t + c
            t = xp.exp(-t * t)
            t = t * 0.5
            t = t - c
            return t * t + a
        return chain

    np_chain = make_chain(np)
    jnp_chain = make_chain(jnp)

    def np_softmax(x):
        e = np.exp(x - x.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    # --- numpy -------------------------------------------------------
    res = {}
    res[("numpy", "axpy  d=a*b+c")] = cpu_bench(lambda: a * b + c)
    res[("numpy", "chain 6 ops")] = cpu_bench(lambda: np_chain(a, b, c))
    res[("numpy", "softmax rows")] = cpu_bench(lambda: np_softmax(sm))
    res[("numpy", f"matmul {MM_N}^3")] = cpu_bench(lambda: mm @ mm, reps=3)
    ref = dict(axpy=a * b + c, chain=np_chain(a, b, c),
               softmax=np_softmax(sm), matmul=mm @ mm)

    # --- XLA on the CPU ----------------------------------------------
    ja, jb, jc = jnp.asarray(a), jnp.asarray(b), jnp.asarray(c)
    jsm, jmm = jnp.asarray(sm), jnp.asarray(mm)
    f_axpy = jax.jit(lambda a, b, c: a * b + c)
    f_chain = jax.jit(jnp_chain)
    f_softmax = jax.jit(lambda x: jax.nn.softmax(x, axis=-1))
    f_matmul = jax.jit(lambda x: jnp.dot(x, x))
    res[("XLA", "axpy  d=a*b+c")] = jax_bench(lambda: f_axpy(ja, jb, jc))
    res[("XLA", "chain 6 ops")] = jax_bench(lambda: f_chain(ja, jb, jc))
    res[("XLA", "softmax rows")] = jax_bench(lambda: f_softmax(jsm))
    res[("XLA", f"matmul {MM_N}^3")] = jax_bench(lambda: f_matmul(jmm), reps=3)
    err = {
        "axpy": float(np.abs(np.asarray(f_axpy(ja, jb, jc)) - ref["axpy"]).max()),
        "chain": float(np.abs(np.asarray(f_chain(ja, jb, jc)) - ref["chain"]).max()),
        "softmax": float(np.abs(np.asarray(f_softmax(jsm)) - ref["softmax"]).max()),
        "matmul": float(np.abs(np.asarray(f_matmul(jmm)) - ref["matmul"]).max()),
    }
    R["B_xla_max_abs_err"] = {k: float(f"{v:.2e}") for k, v in err.items()}

    # --- Triton on the GPU -------------------------------------------
    ga, gb_, gc = [torch.from_numpy(v).cuda() for v in (a, b, c)]
    gd = gpu.empty(n)
    gsm = torch.from_numpy(sm).cuda()
    gsmo = gpu.empty(SM_ROWS, SM_COLS)
    gmm = torch.from_numpy(mm).cuda()
    res[("Triton", "axpy  d=a*b+c")] = gpu.bench(
        lambda: kernels.axpy(ga, gb_, gc, gd), reps=20)
    res[("Triton", "chain 6 ops")] = gpu.bench(
        lambda: kernels.chain(ga, gb_, gc, gd), reps=20)
    res[("Triton", "softmax rows")] = gpu.bench(
        lambda: kernels.softmax(gsm, gsmo), reps=20)
    gmo = gpu.empty(MM_N, MM_N)
    res[("Triton", f"matmul {MM_N}^3")] = gpu.bench(
        lambda: matmul(gmm, gmm, c=gmo), reps=10)

    kernels.axpy(ga, gb_, gc, gd)
    kernels.chain(ga, gb_, gc, gd)
    kernels.softmax(gsm, gsmo)
    torch.cuda.synchronize()
    R["B_triton_max_abs_err"] = {
        "chain": float(np.abs(gd.cpu().numpy() - ref["chain"]).max()),
        "softmax": float(np.abs(gsmo.cpu().numpy() - ref["softmax"]).max()),
        "matmul": float(np.abs(gmo.cpu().numpy() - ref["matmul"]).max()
                        / np.abs(ref["matmul"]).max()),
    }

    # PCIe cost for the element-wise case: 3 arrays in, 1 out
    h = torch.from_numpy(a).pin_memory()
    dv = gpu.empty(n)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        dv.copy_(h)
    torch.cuda.synchronize()
    R["B_pcie_ms_per_array"] = round((time.perf_counter() - t0) / 10 * 1e3, 3)

    specs = work_specs()
    rows = []
    for (backend, wl), ms in sorted(res.items(), key=lambda kv: kv[0][1]):
        fl, by = specs[wl]
        rows.append(dict(backend=backend, workload=wl, ms=round(ms, 4),
                         gflops=round(fl / (ms * 1e6), 1),
                         gbs=round(by / (ms * 1e6), 1),
                         ai=round(fl / by, 3)))
    R["B_rows"] = rows
    R["B_specs"] = {k: dict(flops=v[0], bytes=v[1], ai=round(v[0] / v[1], 3))
                    for k, v in specs.items()}
    print("B. " + " | ".join(
        f"{w.split()[0]}: " + "/".join(
            f"{d['backend'][0]}{d['ms']:.2f}" for d in rows if d["workload"] == w)
        for w in specs))


# ---------------------------------------------------------------- C. ranking
def section_c():
    rows = R["B_rows"]
    per = {}
    for w in R["B_specs"]:
        sub = {d["backend"]: d["ms"] for d in rows if d["workload"] == w}
        winner = min(sub, key=sub.get)
        per[w] = dict(ai=R["B_specs"][w]["ai"], winner=winner,
                      speedup_over_numpy=round(sub["numpy"] / sub[winner], 2),
                      gpu_over_numpy=round(sub["numpy"] / sub["Triton"], 2),
                      xla_over_numpy=round(sub["numpy"] / sub["XLA"], 2))
    R["C_per_workload"] = per
    R["C_gpu_advantage_range"] = [
        round(min(d["gpu_over_numpy"] for d in per.values()), 2),
        round(max(d["gpu_over_numpy"] for d in per.values()), 2)]

    # efficiency: what fraction of each device's own roofline did it reach?
    eff = []
    for d in rows:
        ai = d["ai"]
        if d["backend"] == "Triton":
            roof = min(R["A_gpu_gflops"], R["A_gpu_bw_spec"] * ai)
        else:
            roof = min(R["A_cpu_gflops"], R["A_cpu_bw_gbs"] * ai)
        eff.append(dict(backend=d["backend"], workload=d["workload"],
                        gflops=d["gflops"], roofline=round(roof, 1),
                        pct=round(100 * d["gflops"] / roof, 1)))
    R["C_efficiency"] = eff
    print("C. " + "; ".join(
        f"{w.split()[0]} (AI {v['ai']}): {v['winner']} by {v['speedup_over_numpy']}x"
        for w, v in per.items()))


# ---------------------------------------------------------------- D. honesty
def section_d():
    rows = {(d["backend"], d["workload"]): d for d in R["B_rows"]}
    pcie = R["B_pcie_ms_per_array"]
    out = []
    for w, nbuf in [("axpy  d=a*b+c", 4), ("chain 6 ops", 4),
                    ("softmax rows", 2), (f"matmul {MM_N}^3", 3)]:
        # element-wise buffers are N_ELEM floats; the others are their own size
        scale = 1.0
        if w == "softmax rows":
            scale = SM_ROWS * SM_COLS / N_ELEM
        elif w.startswith("matmul"):
            scale = MM_N * MM_N / N_ELEM
        transfer = pcie * nbuf * scale
        k = rows[("Triton", w)]["ms"]
        cpu = min(rows[("numpy", w)]["ms"], rows[("XLA", w)]["ms"])
        out.append(dict(workload=w, kernel_ms=round(k, 4),
                        transfer_ms=round(transfer, 4),
                        end_to_end_ms=round(k + transfer, 4),
                        kernel_frac=round(k / (k + transfer), 3),
                        best_cpu_ms=round(cpu, 4),
                        gpu_wins_kernel_only=k < cpu,
                        gpu_wins_end_to_end=(k + transfer) < cpu,
                        speedup_kernel=round(cpu / k, 2),
                        speedup_end_to_end=round(cpu / (k + transfer), 2)))
    R["D_rows"] = out
    R["D_flips"] = [d["workload"] for d in out
                    if d["gpu_wins_kernel_only"] and not d["gpu_wins_end_to_end"]]
    print(f"D. once PCIe is counted the GPU loses "
          f"{len(R['D_flips'])} of 4 workloads it won on kernel time: "
          f"{R['D_flips']}")


# ------------------------------------------------------------------- E. zoo
ZOO = [
    ("this i7-8700K (6c)", None, None),
    ("this GTX 1070 Ti", 8190, 256.3),
    ("NVIDIA A100 80GB", 312000, 2039),
    ("NVIDIA H100 SXM", 990000, 3350),
    ("NVIDIA B200 SXM", 2250000, 8000),
    ("AMD MI300X", 1300000, 5300),
    ("Google TPU v5p", 459000, 2765),
    ("Apple M4 Max", 17000, 546),
]


def section_e():
    rows = []
    for name, fl, bw in ZOO:
        if fl is None:
            fl, bw = R["A_cpu_gflops"], R["A_cpu_bw_gbs"]
        rows.append(dict(device=name, gflops=fl, gbs=bw,
                         ridge=round(fl / bw, 1)))
    R["E_zoo"] = rows
    lo = min(rows, key=lambda d: d["ridge"])
    hi = max(rows, key=lambda d: d["ridge"])
    R["E_ridge_lo"] = lo
    R["E_ridge_hi"] = hi
    R["E_ridge_spread"] = round(hi["ridge"] / lo["ridge"], 1)

    # which side of each device's ridge does each workload land on?
    # The four bake-off workloads sit at the extremes, so the interesting
    # middle is filled in with the operation from project 24: an fp16 decode
    # step, whose arithmetic intensity is exactly the batch size.
    probes = [(w, s["ai"]) for w, s in R["B_specs"].items()]
    probes += [(f"fp16 decode, batch {b}", float(b))
               for b in (16, 32, 64, 128, 256)]
    grid = []
    for w, ai in probes:
        row = dict(workload=w, ai=ai)
        for d in rows:
            row[d["device"]] = "compute" if ai >= d["ridge"] else "memory"
        grid.append(row)
    R["E_grid"] = grid
    flips = [g["workload"] for g in grid
             if len({v for k, v in g.items() if k not in ("workload", "ai")}) > 1]
    R["E_workloads_that_flip"] = flips
    print(f"E. ridge points span {R['E_ridge_spread']}x "
          f"({lo['device']} {lo['ridge']} to {hi['device']} {hi['ridge']}); "
          f"{len(flips)}/{len(grid)} workloads change side: {flips}")


# -------------------------------------------------------------------- plot
def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib missing - skipping the plot)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))

    wls = list(R["B_specs"].keys())
    backends = ["numpy", "XLA", "Triton"]
    cols = {"numpy": "#7f7f7f", "XLA": "#1f77b4", "Triton": "#2ca02c"}
    w = 0.26
    xs = np.arange(len(wls))
    lut = {(d["backend"], d["workload"]): d for d in R["B_rows"]}
    for i, bk in enumerate(backends):
        vals = [lut[(bk, wl)]["gflops"] for wl in wls]
        ax[0].bar(xs + (i - 1) * w, vals, w, label=bk, color=cols[bk])
    ax[0].set_yscale("log")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([x.split()[0] for x in wls], fontsize=9)
    ax[0].set_ylabel("GFLOP/s (log)")
    ax[0].set_title("same maths, three backends")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, axis="y", which="both")

    ais = np.logspace(-2, 3, 200)
    for name, fl, bw, col in [
            ("CPU (6 cores)", R["A_cpu_gflops"], R["A_cpu_bw_gbs"], "#7f7f7f"),
            ("GTX 1070 Ti", R["A_gpu_gflops"], R["A_gpu_bw_spec"], "#2ca02c")]:
        ax[1].loglog(ais, np.minimum(fl, bw * ais), color=col, label=name)
    for wl in wls:
        a = R["B_specs"][wl]["ai"]
        ax[1].axvline(a, color="0.85", lw=0.8, zorder=0)
        ax[1].text(a, 2, wl.split()[0], rotation=90, fontsize=7, color="0.4")
    for bk, mk in [("numpy", "o"), ("Triton", "s")]:
        pts = [(lut[(bk, wl)]["ai"], lut[(bk, wl)]["gflops"]) for wl in wls]
        ax[1].loglog([p[0] for p in pts], [p[1] for p in pts], mk,
                     color=cols[bk], ms=6,
                     label=f"{bk} measured")
    ax[1].set_xlabel("arithmetic intensity (FLOP/byte)")
    ax[1].set_ylabel("GFLOP/s")
    ax[1].set_title(f"two rooflines, ridge {R['A_cpu_ridge']} vs "
                    f"{R['A_gpu_ridge']}")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=.3, which="both")

    d = R["D_rows"]
    names = [x["workload"].split()[0] for x in d]
    xs = np.arange(len(d))
    ax[2].bar(xs, [x["speedup_kernel"] for x in d], 0.38,
              label="GPU kernel only", color="#2ca02c")
    ax[2].bar(xs + 0.38, [x["speedup_end_to_end"] for x in d], 0.38,
              label="GPU incl. PCIe", color="#d62728")
    ax[2].axhline(1.0, color="k", lw=1)
    ax[2].set_yscale("log")
    ax[2].set_xticks(xs + 0.19)
    ax[2].set_xticklabels(names, fontsize=9)
    ax[2].set_ylabel("speed-up over the best CPU backend (log)")
    ax[2].set_title("above 1.0 = the GPU is worth it")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, axis="y", which="both")

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
        w.writerow(["backend", "workload", "ms", "GFLOP/s", "GB/s", "AI"])
        for d in R["B_rows"]:
            w.writerow([d["backend"], d["workload"], d["ms"], d["gflops"],
                        d["gbs"], d["ai"]])
        w.writerow([])
        w.writerow(["workload", "kernel_ms", "transfer_ms", "end_to_end_ms",
                    "best_cpu_ms", "speedup_kernel", "speedup_end_to_end"])
        for d in R["D_rows"]:
            w.writerow([d[k] for k in ("workload", "kernel_ms", "transfer_ms",
                                       "end_to_end_ms", "best_cpu_ms",
                                       "speedup_kernel", "speedup_end_to_end")])
        w.writerow([])
        w.writerow(["device", "GFLOP/s", "GB/s", "ridge"])
        for d in R["E_zoo"]:
            w.writerow([d["device"], d["gflops"], d["gbs"], d["ridge"]])
    plot(os.path.join(OUT, "compare_accelerators.png"))
    print(f"total {R['runtime_s']} s")


if __name__ == "__main__":
    main()
