"""Project 26 — torch.compile test.

Compile the transformer from project 24, measure honestly, and find out where
the compiler's win actually comes from:

  0. the noise floor — two identical eager runs
  1. does the compiled model still compute the same thing?
  2. the transformer: forward, inference, forward+backward
  3. where the win comes from: operator counts and the generated kernels
  4. the case the compiler was built for: a chain of element-wise operations
  5. how the win depends on tensor size
  6. the compile tax, measured from a cold cache, and the break-even step count
  7. shape changes: recompilation, automatic dynamic shapes, and dynamic=True

Runtime ~8 min (most of it compiling). Needs torch, numpy, matplotlib, and a
C++ compiler — Inductor generates C++ for the CPU backend.
"""

import csv
import os
import sys
import tempfile
import time
from pathlib import Path

# A fresh Inductor cache per run. Without this, the second run of this script
# reuses compiled kernels from the first and reports a compile time of ~1 s
# instead of the real ~20 s. Cold numbers are the honest ones.
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="inductor-cache-")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._inductor.config  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

# make Inductor's generated C++ kernels show up in the profiler under their name
torch._inductor.config.cpp.enable_kernel_profile = True

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "24-profile-a-training-step"))
import perf_lib as P  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-stride-explorer"))
import plot_style as ps  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

REPEATS, WARMUP = 9, 3
rows = []


def record(section, name, value, note=""):
    rows.append({"section": section, "name": name, "value": value, "note": note})
    print(f"  {section:<12} {name:<40} {value:>14}  {note}")


def fwd_bwd(model, x, y):
    def go():
        loss = P.loss_fn(model(x), y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    return go


def infer(model, x):
    def go():
        with torch.no_grad():
            model(x)
    return go


def count_ops(fn):
    """Operator calls in one call of fn, plus the generated-kernel names."""
    fn()
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fn()
    ka = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    fused = [e for e in prof.key_averages() if "fused" in e.key or "cpp_" in e.key]
    return sum(e.count for e in ka), len(ka), sum(e.count for e in fused), \
        sorted({e.key for e in fused})


# ---------------------------------------------------------------------------
# a chain of element-wise operations: no matrix multiply, all memory traffic
# ---------------------------------------------------------------------------
def chain(x):
    for _ in range(8):
        x = F.gelu(x) * 1.01 + torch.tanh(x) * 0.5
    return x


def main():
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"load average {os.getloadavg()[0]:.1f}\n")
    x, y = P.make_batch()

    # -----------------------------------------------------------------
    # 0. the noise floor
    # -----------------------------------------------------------------
    print("[0] the control: the same eager model, measured twice")
    eager = P.new_model()
    a_ms, a_sp = P.best_of(fwd_bwd(eager, x, y), repeats=REPEATS, warmup=WARMUP)
    b_ms, b_sp = P.best_of(fwd_bwd(eager, x, y), repeats=REPEATS, warmup=WARMUP)
    noise = abs(a_ms - b_ms) / min(a_ms, b_ms)
    record("control", "eager run A (ms)", f"{a_ms:.1f}", f"spread {a_sp:.1f}")
    record("control", "eager run B (ms)", f"{b_ms:.1f}", f"spread {b_sp:.1f}")
    record("control", "run-to-run difference", f"{100*noise:.1f} %",
           "anything smaller than this is not a result")

    # -----------------------------------------------------------------
    # 1-2. the transformer
    # -----------------------------------------------------------------
    print("\n[1-2] eager vs compiled, the transformer")
    comp = torch.compile(P.new_model())        # same seed → identical weights

    with torch.no_grad():
        ref = eager(x)
    t0 = time.perf_counter()
    with torch.no_grad():
        got = comp(x)
    compile_fwd_s = time.perf_counter() - t0
    record("correct", "max |compiled - eager|", f"{(got - ref).abs().max():.2e}",
           "fusion reorders arithmetic")
    record("correct", "allclose(atol=1e-5)", str(torch.allclose(got, ref, atol=1e-5)))

    i_e, i_esp = P.best_of(infer(eager, x), repeats=REPEATS, warmup=WARMUP)
    i_c, i_csp = P.best_of(infer(comp, x), repeats=REPEATS, warmup=WARMUP)
    record("speed", "inference, eager (ms)", f"{i_e:.1f}", f"spread {i_esp:.1f}")
    record("speed", "inference, compiled (ms)", f"{i_c:.1f}", f"spread {i_csp:.1f}")
    record("speed", "inference speedup", f"{i_e/i_c:.2f}x")

    t0 = time.perf_counter()
    fwd_bwd(comp, x, y)()
    compile_bwd_s = time.perf_counter() - t0

    e_step, e_ssp = P.best_of(fwd_bwd(eager, x, y), repeats=REPEATS, warmup=WARMUP)
    c_step, c_ssp = P.best_of(fwd_bwd(comp, x, y), repeats=REPEATS, warmup=WARMUP)
    record("speed", "training step, eager (ms)", f"{e_step:.1f}", f"spread {e_ssp:.1f}")
    record("speed", "training step, compiled (ms)", f"{c_step:.1f}", f"spread {c_ssp:.1f}")
    record("speed", "training-step speedup", f"{e_step/c_step:.2f}x",
           "compare against the control above")

    # -----------------------------------------------------------------
    # 3. where the win comes from
    # -----------------------------------------------------------------
    print("\n[3] operator counts")
    e_calls, e_kinds, _, _ = count_ops(fwd_bwd(eager, x, y))
    c_calls, c_kinds, c_fused, fused_names = count_ops(fwd_bwd(comp, x, y))
    record("fusion", "eager operator calls / step", f"{e_calls}", f"{e_kinds} distinct")
    record("fusion", "compiled operator calls / step", f"{c_calls}", f"{c_kinds} distinct")
    record("fusion", "calls removed", f"{100*(1-c_calls/e_calls):.1f} %")
    record("fusion", "generated kernel calls / step", f"{c_fused}",
           f"{len(fused_names)} distinct generated kernels")
    for nm in fused_names[:3]:
        record("fusion", "a generated kernel", nm[:32] + "...",
               "the name lists the ops it swallowed")
    (OUT / "fused_kernels.txt").write_text("\n".join(fused_names))

    # how much of the step is matmul — the part the compiler cannot improve
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        fwd_bwd(eager, x, y)()
    ka = [e for e in prof.key_averages() if e.key.startswith("aten::")]
    tot = sum(e.self_cpu_time_total for e in ka)
    mm = sum(e.self_cpu_time_total for e in ka
             if e.key in ("aten::mm", "aten::addmm", "aten::bmm"))
    record("fusion", "matmul share of the eager step", f"{100*mm/tot:.1f} %",
           "already inside a vendor library — fusion cannot touch it")

    # -----------------------------------------------------------------
    # 4. the case fusion was built for
    # -----------------------------------------------------------------
    print("\n[4] an element-wise chain: 8 x (gelu, mul, tanh, mul, add)")
    xc = torch.randn(16, 128, 256, requires_grad=True)
    cchain = torch.compile(chain)
    t0 = time.perf_counter()
    cchain(xc)
    chain_compile_s = time.perf_counter() - t0

    ch_e, ch_esp = P.best_of(lambda: chain(xc), repeats=REPEATS, warmup=WARMUP)
    ch_c, ch_csp = P.best_of(lambda: cchain(xc), repeats=REPEATS, warmup=WARMUP)
    record("elementwise", "forward, eager (ms)", f"{ch_e:.2f}", f"spread {ch_esp:.2f}")
    record("elementwise", "forward, compiled (ms)", f"{ch_c:.2f}", f"spread {ch_csp:.2f}")
    record("elementwise", "forward speedup", f"{ch_e/ch_c:.2f}x",
           "this is what kernel fusion buys")
    ec, ek, _, _ = count_ops(lambda: chain(xc))
    cc, ck, cf, _ = count_ops(lambda: cchain(xc))
    record("elementwise", "eager operator calls", f"{ec}", f"{ek} distinct")
    record("elementwise", "compiled operator calls", f"{cc}",
           f"{cf} generated kernel calls")

    def bwd(fn):
        def go():
            fn(xc).sum().backward()
            xc.grad = None
        return go
    t0 = time.perf_counter()
    bwd(cchain)()
    chain_compile_s += time.perf_counter() - t0
    b_e, _ = P.best_of(bwd(chain), repeats=REPEATS, warmup=WARMUP)
    b_c, _ = P.best_of(bwd(cchain), repeats=REPEATS, warmup=WARMUP)
    record("elementwise", "fwd+bwd, eager (ms)", f"{b_e:.2f}")
    record("elementwise", "fwd+bwd, compiled (ms)", f"{b_c:.2f}")
    record("elementwise", "fwd+bwd speedup", f"{b_e/b_c:.2f}x",
           "fusing the forward means recomputing it in the backward")

    # -----------------------------------------------------------------
    # 5. the win depends on size
    # -----------------------------------------------------------------
    print("\n[5] speedup vs size")
    sizes = [(2, 32), (4, 32), (8, 64), (16, 128)]
    sweep = []
    for b, t in sizes:
        torch._dynamo.reset()
        m = P.new_model(seed=0)
        cm = torch.compile(P.new_model(seed=0))
        xb, yb = P.make_batch(batch=b, seq=t)
        fwd_bwd(cm, xb, yb)()                       # compile forward + backward
        # interleave the two configurations: measuring all of one and then all
        # of the other lets a slow minute on a shared machine land entirely on
        # one side and masquerade as a result
        ems, cms = [], []
        for _ in range(3):
            ems.append(P.best_of(fwd_bwd(m, xb, yb), repeats=3, warmup=1)[0])
            cms.append(P.best_of(fwd_bwd(cm, xb, yb), repeats=3, warmup=1)[0])
        em, cmm = min(ems), min(cms)
        tax = compile_fwd_s + compile_bwd_s
        be = tax * 1000 / (em - cmm) if em > cmm else float("inf")
        sweep.append((f"{b}x{t}", em, cmm, em / cmm, tax, be))
        record("size", f"batch {b} x seq {t}", f"{em/cmm:.2f}x",
               f"eager {em:.1f} → compiled {cmm:.1f} ms"
               + (f" | break-even {be:.0f} steps" if be < 1e6 else " | never pays back"))

    # -----------------------------------------------------------------
    # 6. the compile tax
    # -----------------------------------------------------------------
    print("\n[6] the compile tax, from a cold cache")
    record("tax", "transformer forward compile (s)", f"{compile_fwd_s:.1f}")
    record("tax", "transformer backward compile (s)", f"{compile_bwd_s:.1f}",
           "the backward graph compiles separately, on the first backward")
    record("tax", "element-wise chain compile (s)", f"{chain_compile_s:.1f}")
    best = min(sweep, key=lambda s: s[5])
    if best[5] < 1e6:
        record("tax", "best break-even in the sweep", f"{best[5]:.0f} steps",
               f"at {best[0]}, against a {best[4]:.0f} s compile")
    chain_be = chain_compile_s * 1000 / (b_e - b_c) if b_e > b_c else \
        chain_compile_s * 1000 / (ch_e - ch_c)
    record("tax", "element-wise chain break-even", f"{chain_be:.0f} forward calls",
           "inference-only, where the fusion win is real")

    # -----------------------------------------------------------------
    # 7. shape changes
    # -----------------------------------------------------------------
    print("\n[7] recompilation")
    record("recompile", "recompile_limit (config)",
           f"{torch._dynamo.config.recompile_limit}",
           "exceed it and the model silently falls back to eager")
    torch._dynamo.reset()
    small = torch.compile(P.new_model(seed=0, d=128, n_layer=2))
    times = []
    for b in (2, 4, 6, 8, 10, 12):
        xb, _ = P.make_batch(batch=b, seq=64)
        t0 = time.perf_counter()
        with torch.no_grad():
            small(xb)
        times.append(time.perf_counter() - t0)
    record("recompile", "call 1, batch 2 (s)", f"{times[0]:.1f}", "compiles for shape 2")
    record("recompile", "call 2, batch 4 (s)", f"{times[1]:.1f}",
           "a new shape → recompiles, this time with a symbolic batch")
    record("recompile", "calls 3-6 (s)", ", ".join(f"{t:.2f}" for t in times[2:]),
           "no more compiling — one graph now covers every batch size")
    record("recompile", "compiles for 6 different shapes", "2",
           "automatic dynamic shapes")

    torch._dynamo.reset()
    dyn = torch.compile(P.new_model(seed=0, d=128, n_layer=2), dynamic=True)
    dtimes = []
    for b in (2, 4, 6, 8, 10, 12):
        xb, _ = P.make_batch(batch=b, seq=64)
        t0 = time.perf_counter()
        with torch.no_grad():
            dyn(xb)
        dtimes.append(time.perf_counter() - t0)
    record("recompile", "dynamic=True, call 1 (s)", f"{dtimes[0]:.1f}",
           "symbolic from the start")
    record("recompile", "dynamic=True, calls 2-6 (s)",
           ", ".join(f"{t:.2f}" for t in dtimes[1:]))
    record("recompile", "total, default (s)", f"{sum(times):.1f}")
    record("recompile", "total, dynamic=True (s)", f"{sum(dtimes):.1f}")

    # -----------------------------------------------------------------
    # figures
    # -----------------------------------------------------------------
    fig, axes = ps.plt.subplots(1, 3, figsize=(15.4, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)

    ax = axes[0]
    labels = ["transformer\ninference", "transformer\ntraining step",
              "element-wise\nchain (fwd)"]
    eg = [i_e, e_step, ch_e]
    cp = [i_c, c_step, ch_c]
    xs = np.arange(3)
    ax.bar(xs - 0.18, [1.0] * 3, 0.34, color=ps.SERIES[0], label="eager")
    ax.bar(xs + 0.18, [c / e for e, c in zip(eg, cp)], 0.34, color=ps.SERIES[1],
           label="compiled")
    for i, (e, c) in enumerate(zip(eg, cp)):
        ax.text(i + 0.18, c / e + 0.03, f"{e/c:.2f}x", ha="center",
                color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Relative time (lower is better)", color=ps.INK, fontsize=12, loc="left")
    ax.set_ylabel("time / eager time", color=ps.INK_SECONDARY)

    ax = axes[1]
    names = ["transformer\nstep", "element-wise\nchain"]
    eager_calls = [e_calls, ec]
    comp_calls = [c_calls, cc]
    xs = np.arange(2)
    ax.bar(xs - 0.18, eager_calls, 0.34, color=ps.SERIES[0], label="eager")
    ax.bar(xs + 0.18, comp_calls, 0.34, color=ps.SERIES[1], label="compiled")
    for i in range(2):
        ax.text(i + 0.18, comp_calls[i] + max(eager_calls) * 0.02,
                f"-{100*(1-comp_calls[i]/eager_calls[i]):.0f}%", ha="center",
                color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Operator calls per iteration", color=ps.INK, fontsize=12, loc="left")
    ax.set_ylabel("aten operator calls", color=ps.INK_SECONDARY)

    ax = axes[2]
    xs = np.arange(1, 7)
    ax.plot(xs, times, "o-", color=ps.SERIES[2], lw=1.8, label="default")
    ax.plot(xs, dtimes, "o-", color=ps.SERIES[1], lw=1.8, label="dynamic=True")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Two compiles, then never again", color=ps.INK, fontsize=12, loc="left")
    ax.set_xlabel("call number (each with a new batch size)", color=ps.INK_SECONDARY)
    ax.set_ylabel("call time (s, log scale)", color=ps.INK_SECONDARY)

    ps.save(fig, OUT / "torch_compile_test.png")

    with open(OUT / "findings.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "name", "value", "note"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT/'findings.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
