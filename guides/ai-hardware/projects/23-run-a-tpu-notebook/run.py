"""Project 23 - the TPU programming model, on a machine with no TPU.

  A. what is here   - XLA, the TPU's compiler, runs on the CPU; the MXU does not
  B. tracing        - jit runs your Python once, on fake values, and never again
  C. shapes         - a new shape is a new compile; what that costs, and buckets
  D. fusion         - what XLA does to a chain of element-wise ops, measured
  E. the MXU        - a cycle-by-cycle systolic array, verified bit-exact
  F. dtypes         - the silent float64 -> float32 downgrade, and bf16 error
"""

import csv
import json
import os
import re
import time

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax                                                       # noqa: E402
import jax.numpy as jnp                                          # noqa: E402

from systolic import SystolicArray                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
R = {}


def bench(fn, reps=20, warmup=3):
    """Median-of-3. JAX dispatch is asynchronous: without block_until_ready
    this times the queueing, not the work (same trap as CUDA)."""
    for _ in range(warmup):
        jax.block_until_ready(fn())
    out = []
    for _ in range(3):
        t0 = time.perf_counter()
        for _ in range(reps):
            jax.block_until_ready(fn())
        out.append((time.perf_counter() - t0) / reps)
    out.sort()
    return out[1] * 1e3                                   # ms


# ---------------------------------------------------------------- A. machine
def section_a():
    devs = jax.devices()
    R["A_devices"] = [str(d) for d in devs]
    R["A_platform"] = devs[0].platform
    R["A_jax_version"] = jax.__version__
    R["A_device_kind"] = devs[0].device_kind
    # the one thing a real TPU has that this does not:
    R["A_has_mxu"] = False
    print(f"A. jax {jax.__version__} on {R['A_platform']} "
          f"({R['A_device_kind']}), devices={R['A_devices']}")


# ---------------------------------------------------------------- B. tracing
def section_b():
    calls = {"n": 0}

    @jax.jit
    def f(x):
        calls["n"] += 1                       # Python side effect
        return x * 2 + 1

    x = jnp.arange(8.0)
    for _ in range(5):
        f(x)
    R["B_python_body_runs"] = calls["n"]
    R["B_jit_calls"] = 5

    # what a traced value actually is
    kind = {}

    @jax.jit
    def g(x):
        kind["type"] = type(x).__name__
        kind["shape"] = tuple(x.shape)
        kind["has_value"] = hasattr(x, "_value")
        return x

    g(x)
    R["B_traced_type"] = kind["type"]
    R["B_traced_shape"] = list(kind["shape"])

    # the error every beginner hits
    @jax.jit
    def branchy(x):
        if x.sum() > 0:                       # a Python `if` on a traced value
            return x
        return -x

    try:
        branchy(x)
        R["B_python_if_error"] = None
    except Exception as e:
        R["B_python_if_error"] = type(e).__name__

    @jax.jit
    def fixed(x):
        return jax.lax.cond(x.sum() > 0, lambda v: v, lambda v: -v, x)

    R["B_lax_cond_ok"] = bool(np.allclose(np.asarray(fixed(x)), np.asarray(x)))
    print(f"B. 5 jit calls -> Python body ran {calls['n']}x; "
          f"traced value is a {R['B_traced_type']}; "
          f"`if` raises {R['B_python_if_error']}")


# ---------------------------------------------------------------- C. shapes
def make_work():
    """A FRESH function object each time.

    jit's compilation cache is keyed on the function it wraps, so
    `jax.jit(work)` twice would share one cache and the second experiment
    would inherit the first one's entries.
    """
    def work(x):
        for _ in range(6):
            x = jnp.tanh(x * 1.01 + 0.01)
        return x.sum()
    return work


def section_c():
    # (i) compile cost vs run cost, at one shape
    n = 1 << 18
    x = jnp.ones(n, dtype=jnp.float32)
    jf = jax.jit(make_work())
    t0 = time.perf_counter()
    compiled = jf.lower(x).compile()
    t_compile = (time.perf_counter() - t0) * 1e3
    t_run = bench(lambda: compiled(x), reps=20)
    R["C_compile_ms"] = round(t_compile, 1)
    R["C_run_ms"] = round(t_run, 4)
    R["C_runs_to_amortize"] = int(round(t_compile / t_run))

    # (ii) every distinct shape is a separate compile
    jf2 = jax.jit(make_work())
    ragged = [1000 + 37 * i for i in range(8)]
    t0 = time.perf_counter()
    for m in ragged:
        jax.block_until_ready(jf2(jnp.ones(m, dtype=jnp.float32)))
    t_ragged = (time.perf_counter() - t0) * 1e3
    R["C_ragged_shapes"] = len(ragged)
    R["C_ragged_compiles"] = int(jf2._cache_size())
    R["C_ragged_ms"] = round(t_ragged, 1)

    # (iii) pad every shape up to the next bucket -> one compile
    jf3 = jax.jit(make_work())
    bucket = 1 << 11
    t0 = time.perf_counter()
    for m in ragged:
        padded = -(-m // bucket) * bucket
        jax.block_until_ready(jf3(jnp.ones(padded, dtype=jnp.float32)))
    t_bucket = (time.perf_counter() - t0) * 1e3
    R["C_bucket_compiles"] = int(jf3._cache_size())
    R["C_bucket_ms"] = round(t_bucket, 1)
    R["C_bucket_speedup"] = round(t_ragged / t_bucket, 2)
    R["C_bucket_wasted_frac"] = round(
        1 - sum(ragged) / sum(-(-m // bucket) * bucket for m in ragged), 4)
    print(f"C. compile {t_compile:.0f} ms vs run {t_run:.3f} ms "
          f"({R['C_runs_to_amortize']} runs to pay it back); "
          f"8 ragged shapes = {R['C_ragged_compiles']} compiles "
          f"({t_ragged:.0f} ms) -> bucketed = {R['C_bucket_compiles']} "
          f"({t_bucket:.0f} ms, {R['C_bucket_speedup']}x)")


# ---------------------------------------------------------------- D. fusion
def chain(a, b, c):
    t = a * b
    t = t + c
    t = jnp.tanh(t)
    t = t * 0.5
    t = t - c
    t = jnp.exp(-t * t)
    return t


def count_stablehlo(text):
    """Ops in the StableHLO that jit hands to the compiler (before it runs)."""
    return re.findall(r"stablehlo\.([a-z_]+)", text)


def count_entry_hlo(text):
    """Instructions XLA actually schedules.

    Only the ENTRY computation counts. Fused ops still appear inside a
    `%fused_computation` block, but that block is not scheduled on its own --
    it is the *body* of one kernel, so counting it would count the very ops
    fusion just removed.
    """
    m = re.search(r"^ENTRY .*?\{\n(.*?)^\}", text, flags=re.M | re.S)
    body = m.group(1) if m else ""
    ops = re.findall(r"^\s+(?:ROOT\s+)?%[\w.\-]+ = \S+ ([a-z\-_]+)\(", body,
                     flags=re.M)
    return [o for o in ops if o != "parameter"]


def section_d():
    n = 1 << 21
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    a, b, c = [jax.random.normal(k, (n,), dtype=jnp.float32) for k in keys]

    lowered = jax.jit(chain).lower(a, b, c)
    pre = lowered.as_text()
    post = lowered.compile().as_text()
    ops_pre = count_stablehlo(pre)
    ops_post = count_entry_hlo(post)
    R["D_hlo_ops_before"] = len(ops_pre)
    R["D_hlo_ops_after"] = len(ops_post)
    R["D_fusions_after"] = sum(1 for o in ops_post if o == "fusion")
    R["D_pre_ops"] = sorted(set(ops_pre))
    R["D_post_ops"] = sorted(set(ops_post))
    m = re.search(r"ROOT %(\S+) = ", post.split("ENTRY")[-1])
    R["D_fusion_name"] = m.group(1) if m else "?"

    jitted = jax.jit(chain)
    t_jit = bench(lambda: jitted(a, b, c), reps=20)
    t_eager = bench(lambda: chain(a, b, c), reps=20)
    an, bn, cn = [np.asarray(v) for v in (a, b, c)]

    def np_chain():
        t = an * bn
        t = t + cn
        t = np.tanh(t)
        t = t * 0.5
        t = t - cn
        return np.exp(-t * t)

    t_np = bench(lambda: np_chain(), reps=20)

    R["D_n"] = n
    R["D_ms_jit"] = round(t_jit, 3)
    R["D_ms_jax_eager"] = round(t_eager, 3)
    R["D_ms_numpy"] = round(t_np, 3)
    R["D_speedup_vs_eager"] = round(t_eager / t_jit, 2)
    R["D_speedup_vs_numpy"] = round(t_np / t_jit, 2)
    # bytes: fused = 3 reads + 1 write; unfused = one read+write per op
    R["D_gbs_jit"] = round(n * 4 * 4 / (t_jit * 1e6), 1)
    R["D_ops_in_chain"] = 6
    print(f"D. HLO {len(ops_pre)} ops -> {len(ops_post)} after optimisation "
          f"({R['D_fusions_after']} fusion); jit {t_jit:.2f} ms, "
          f"jax eager {t_eager:.2f} ms ({R['D_speedup_vs_eager']}x), "
          f"numpy {t_np:.2f} ms ({R['D_speedup_vs_numpy']}x)")


# ---------------------------------------------------------------- E. the MXU
def section_e():
    rng = np.random.default_rng(0)

    # (i) the simulator is a real matmul, not a cost model
    checks = []
    for (M, K, N, Rr, Cc) in [(4, 3, 2, 4, 4), (8, 8, 8, 8, 8),
                              (5, 7, 3, 8, 4), (48, 16, 16, 16, 16)]:
        X = rng.integers(-7, 7, (M, K)).astype(np.float64)
        W = rng.integers(-7, 7, (K, N)).astype(np.float64)
        Y, st = SystolicArray(Rr, Cc).one_pass(X, W)
        checks.append(dict(M=M, K=K, N=N, array=f"{Rr}x{Cc}",
                           exact=bool(np.array_equal(Y, X @ W)),
                           cycles=st["cycles"],
                           util=round(st["useful_macs"] / st["cell_cycles"], 4)))
    R["E_checks"] = checks
    R["E_all_exact"] = all(c["exact"] for c in checks)

    # (ii) the 128 rule: utilisation vs N, on a 128x128 array
    mxu = SystolicArray(128, 128)
    sweep = []
    for N in sorted(set(list(range(1, 33)) + list(range(32, 545, 8))
                        + [128, 129, 136, 256, 257, 384, 512])):
        s512 = mxu.matmul_cost(512, 128, N)
        s4096 = mxu.matmul_cost(4096, 128, N)
        sweep.append(dict(N=N, cycles=s512["cycles"], tiles=s512["tiles"],
                          util=round(s512["utilization"], 4),
                          util_M4096=round(s4096["utilization"], 4)))
    R["E_sweep_N"] = sweep
    by_n = {s["N"]: s for s in sweep}
    R["E_util_128_M4096"] = by_n[128]["util_M4096"]
    R["E_util_129_M4096"] = by_n[129]["util_M4096"]
    R["E_cliff_ratio_M4096"] = round(
        by_n[128]["util_M4096"] / by_n[129]["util_M4096"], 2)
    R["E_util_128"] = by_n[128]["util"]
    R["E_util_136"] = by_n[136]["util"]
    R["E_cliff_ratio"] = round(by_n[128]["util"] / by_n[136]["util"], 2)
    R["E_util_256"] = by_n[256]["util"]
    R["E_util_8"] = by_n[8]["util"]

    # (iii) batch size: a small M cannot hide the pipeline fill or weight load
    batch = []
    for M in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]:
        ov = mxu.matmul_cost(M, 128, 128, weight_load_overlapped=True)
        no = mxu.matmul_cost(M, 128, 128, weight_load_overlapped=False)
        batch.append(dict(M=M, util_overlapped=round(ov["utilization"], 4),
                          util_serial=round(no["utilization"], 4),
                          cycles=ov["cycles"]))
    R["E_batch"] = batch
    R["E_util_M1"] = batch[0]["util_overlapped"]
    R["E_util_M512"] = [b for b in batch if b["M"] == 512][0]["util_overlapped"]

    # (iv) a real transformer shape, padded vs not
    real = []
    for name, (M, K, N) in [
            ("GPT-2 small qkv (768 -> 2304)", (512, 768, 2304)),
            ("Llama-ish MLP (4096 -> 11008)", (512, 4096, 11008)),
            ("vocab head (768 -> 50257)", (512, 768, 50257)),
            ("vocab head padded (768 -> 50304)", (512, 768, 50304)),
            ("one attention head (K=64, N=64)", (512, 64, 64)),
            ("8 heads merged (K=512, N=512)", (512, 512, 512))]:
        s = mxu.matmul_cost(M, K, N)
        real.append(dict(name=name, M=M, K=K, N=N, tiles=s["tiles"],
                         util=round(s["utilization"], 4)))
    R["E_real"] = real
    by_name = {d["name"]: d["util"] for d in real}
    R["E_pad_vocab_gain"] = round(
        by_name["vocab head padded (768 -> 50304)"]
        / by_name["vocab head (768 -> 50257)"], 4)
    R["E_head_merge_gain"] = round(
        by_name["8 heads merged (K=512, N=512)"]
        / by_name["one attention head (K=64, N=64)"], 2)
    print(f"E. simulator bit-exact: {R['E_all_exact']}; "
          f"util N=128 {R['E_util_128']:.3f} vs N=136 {R['E_util_136']:.3f} "
          f"({R['E_cliff_ratio']}x); M=1 {R['E_util_M1']:.4f}")


# ---------------------------------------------------------------- F. dtypes
def section_f():
    x64 = np.array([1.0 / 3.0], dtype=np.float64)
    xj = jnp.asarray(x64)
    R["F_numpy_dtype"] = str(x64.dtype)
    R["F_jax_dtype"] = str(xj.dtype)
    R["F_silent_downgrade"] = str(x64.dtype) != str(xj.dtype)

    rng = np.random.default_rng(0)
    a = rng.standard_normal((256, 256)).astype(np.float32)
    b = rng.standard_normal((256, 256)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)

    def relerr(y):
        return float(np.abs(np.asarray(y, dtype=np.float64) - ref).max()
                     / np.abs(ref).max())

    ja, jb = jnp.asarray(a), jnp.asarray(b)
    rows = [("float32", relerr(jnp.dot(ja, jb))),
            ("bfloat16 inputs",
             relerr(jnp.dot(ja.astype(jnp.bfloat16),
                            jb.astype(jnp.bfloat16)).astype(jnp.float32))),
            ("float16 inputs",
             relerr(jnp.dot(ja.astype(jnp.float16),
                            jb.astype(jnp.float16)).astype(jnp.float32)))]
    R["F_matmul_relerr"] = [dict(dtype=d, rel_err=float(f"{e:.3e}")) for d, e in rows]
    R["F_bf16_over_f32"] = round(rows[1][1] / rows[0][1], 1)
    print(f"F. numpy float64 -> jax {R['F_jax_dtype']}; "
          f"bf16 matmul error {R['F_bf16_over_f32']}x float32's")


# ---------------------------------------------------------------- plot
def plot(path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib missing - skipping the plot)")
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    s = R["E_sweep_N"]
    ax[0].plot([d["N"] for d in s], [d["util_M4096"] * 100 for d in s], lw=1.6,
               color="#1f77b4", label="M=4096")
    ax[0].plot([d["N"] for d in s], [d["util"] * 100 for d in s], lw=1.4,
               color="#ff7f0e", label="M=512")
    for k in (128, 256, 384, 512):
        ax[0].axvline(k, color="0.8", lw=0.8, zorder=0)
    ax[0].legend(fontsize=8)
    ax[0].set_xlabel("N (output width)")
    ax[0].set_ylabel("MXU utilization (%)")
    ax[0].set_title("128x128 MXU, K=128\none column past 128 halves it")
    ax[0].set_ylim(0, 100)
    ax[0].grid(alpha=.3)

    b = R["E_batch"]
    ax[1].semilogx([d["M"] for d in b], [d["util_overlapped"] * 100 for d in b],
                   "o-", label="weight load overlapped")
    ax[1].semilogx([d["M"] for d in b], [d["util_serial"] * 100 for d in b],
                   "s--", label="weight load serial")
    ax[1].set_xlabel("batch rows M")
    ax[1].set_ylabel("MXU utilization (%)")
    ax[1].set_title("small batches cannot fill the pipe")
    ax[1].set_ylim(0, 100)
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    names = ["jit (XLA)", "jax eager", "numpy"]
    vals = [R["D_ms_jit"], R["D_ms_jax_eager"], R["D_ms_numpy"]]
    bars = ax[2].bar(names, vals, color=["#2ca02c", "#d62728", "#7f7f7f"])
    for bar, v in zip(bars, vals):
        ax[2].text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}",
                   ha="center", va="bottom", fontsize=9)
    ax[2].set_ylabel("ms")
    ax[2].set_title(f"6 element-wise ops on {R['D_n'] / 1e6:.1f}M floats\n"
                    f"XLA leaves {R['D_hlo_ops_after']} HLO ops of "
                    f"{R['D_hlo_ops_before']}")
    ax[2].grid(alpha=.3, axis="y")

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
    section_f()
    R["runtime_s"] = round(time.time() - t0, 1)

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(R, f, indent=1)
    with open(os.path.join(OUT, "findings.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        for k, v in R.items():
            if not isinstance(v, (list, dict)):
                w.writerow([k, v])
        w.writerow([])
        w.writerow(["mxu_N", "cycles", "tiles", "util_M512", "util_M4096"])
        for d in R["E_sweep_N"]:
            w.writerow([d["N"], d["cycles"], d["tiles"], d["util"], d["util_M4096"]])
        w.writerow([])
        w.writerow(["mxu_M", "util_overlapped", "util_serial"])
        for d in R["E_batch"]:
            w.writerow([d["M"], d["util_overlapped"], d["util_serial"]])
    plot(os.path.join(OUT, "tpu_xla.png"))
    print(f"total {R['runtime_s']} s")


if __name__ == "__main__":
    main()
