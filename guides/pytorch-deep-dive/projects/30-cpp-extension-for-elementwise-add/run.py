"""Project 30 -- a C++ extension for elementwise add.

Sections
  1. compile a C++ `add` at runtime and call it from Python
  2. is it right? four inputs that a raw pointer loop gets wrong
  3. is it fast? hand loop vs at::parallel_for vs torch.add
  4. the fix: TensorIterator, which is how torch.add itself is written
  5. the dispatcher: TORCH_LIBRARY, torch.ops, and what happens with autograd
  6. what the compile actually costs
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import kernels_lib as K  # noqa: E402

OUT = K.outputs_dir(__file__)
ROWS = []          # findings.csv
N = 8_000_000      # 8M floats = 32 MB per tensor, well past this CPU's L3


# ---------------------------------------------------------------------------
# The C++ source. Everything in this project is in this one string.
# ---------------------------------------------------------------------------
CPP = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/TensorIterator.h>

// ---- version 1: the obvious loop -----------------------------------------
// Takes the first float of each tensor and walks forward numel() times.
// This is the kernel almost everyone writes first. It is also wrong for any
// input whose elements are not laid out one after another in memory.
torch::Tensor add_naive(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  float* po = out.data_ptr<float>();
  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
  return out;
}

// ---- version 2: same loop, but say out loud what it assumes --------------
// TORCH_CHECK is PyTorch's assert: it raises a Python RuntimeError with this
// message instead of reading memory that does not belong to us.
torch::Tensor add_checked(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch: ", a.sizes(), " vs ", b.sizes());
  TORCH_CHECK(a.scalar_type() == torch::kFloat && b.scalar_type() == torch::kFloat,
              "add_checked only handles float32, got ", a.scalar_type(), " and ", b.scalar_type());
  TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "CPU only");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "add_checked needs contiguous inputs");
  auto out = torch::empty_like(a);
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  float* po = out.data_ptr<float>();
  const int64_t n = a.numel();
  for (int64_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
  return out;
}

// ---- version 3: use all the cores ----------------------------------------
// at::parallel_for hands each of PyTorch's threads a slice [s, e) of the
// index range. The 32768 is the grain size: chunks smaller than this are run
// on one thread, because waking up a thread costs more than the work saved.
// This is the CPU twin of a CUDA grid: same body, many workers, disjoint
// index ranges, no communication between them.
torch::Tensor add_parallel(torch::Tensor a, torch::Tensor b) {
  auto out = torch::empty_like(a);
  const float* pa = a.data_ptr<float>();
  const float* pb = b.data_ptr<float>();
  float* po = out.data_ptr<float>();
  at::parallel_for(0, a.numel(), 32768, [&](int64_t s, int64_t e) {
    for (int64_t i = s; i < e; ++i) po[i] = pa[i] + pb[i];
  });
  return out;
}

// ---- version 4: how torch.add is really written --------------------------
// TensorIterator is ATen's loop planner. You describe the inputs and the
// output; it works out broadcasting, the common dtype, the memory layout, and
// how to split the work over threads. Then it calls your inner loop with
// pointers *and strides*, so the loop is correct on any layout.
// An undefined output tensor means "you allocate it for me".
torch::Tensor add_iter(torch::Tensor a, torch::Tensor b) {
  at::Tensor undefined_output;
  auto iter = at::TensorIteratorConfig()
      .add_output(undefined_output)
      .add_input(a)
      .add_input(b)
      .build();
  iter.for_each([](char** data, const int64_t* strides, int64_t n) {
    char* o = data[0];
    const char* x = data[1];
    const char* y = data[2];
    for (int64_t i = 0; i < n; ++i) {
      *reinterpret_cast<float*>(o + i * strides[0]) =
          *reinterpret_cast<const float*>(x + i * strides[1]) +
          *reinterpret_cast<const float*>(y + i * strides[2]);
    }
  });
  return iter.output();
}

// ---- registering the op with the dispatcher ------------------------------
// pybind11 (what `functions=[...]` uses) gives you a plain Python function.
// TORCH_LIBRARY gives you something stronger: a real operator with a schema,
// visible as torch.ops.p30.add, reachable from TorchScript and from C++, and
// routable per device/dtype by the dispatcher.
TORCH_LIBRARY(p30, m) {
  m.def("add(Tensor a, Tensor b) -> Tensor", &add_iter);
}
"""


def build_ext():
    K.banner("[1] compiling the extension")
    mod, secs = K.build(
        "p30_add", CPP,
        functions=["add_naive", "add_checked", "add_parallel", "add_iter"],
    )
    how = "compiled" if secs > 1 else "loaded from the cache built by an earlier run"
    print(f"    {how} in {secs:.2f} s")
    ROWS.append(["compile", "load_inline (this run)", f"{secs:.2f}", how])
    a = torch.randn(5)
    b = torch.randn(5)
    print(f"    a + b (torch) = {(a + b).tolist()}")
    print(f"    add_naive(a,b) = {mod.add_naive(a, b).tolist()}")
    return mod, secs


# ---------------------------------------------------------------------------
def correctness(mod):
    K.banner("[2] is it right? the same kernel on four inputs")
    n = 1024
    base_a = torch.randn(n, n)
    # The last case is marked run_naive=False on purpose. `expand` makes a
    # 1024x1024 *view* of 1024 real floats (stride 0 down the rows), so
    # `numel()` says 1,048,576 while the buffer holds 1,024. The naive loop
    # would read a megabyte past the end of an allocation -- undefined
    # behaviour, which on a bad day is a crash and on a worse day is silent
    # garbage. We describe it instead of running it.
    cases = {
        "contiguous": (base_a, torch.randn(n, n), True),
        "b transposed": (base_a, torch.randn(n, n).t(), True),
        "b is every other column": (base_a, torch.randn(n, 2 * n)[:, ::2], True),
        "b broadcast from one row": (base_a, torch.randn(1, n).expand(n, n), False),
    }
    rows = []
    for label, (a, b, run_naive) in cases.items():
        ref = a + b
        if not run_naive:
            naive = "reads out of bounds (not run)"
        else:
            try:
                got = mod.add_naive(a, b)
                err = K.max_abs_diff(got, ref)
                ok = err < 1e-6
                naive = "correct" if ok else f"WRONG (max err {err:.3f})"
            except RuntimeError as e:
                naive = f"raised: {str(e).splitlines()[0][:40]}"
        try:
            iter_err = K.max_abs_diff(mod.add_iter(a, b), ref)
            it = "correct" if iter_err < 1e-6 else f"WRONG ({iter_err:.3f})"
        except RuntimeError as e:
            it = f"raised: {str(e).splitlines()[0][:40]}"
        try:
            mod.add_checked(a, b)
            chk = "correct"
        except RuntimeError as e:
            chk = "raised: " + str(e).split("\n")[0][:46]
        print(f"    {label:<26} naive={naive:<24} checked={chk:<52} iter={it}")
        rows.append([label, naive, chk, it])
        ROWS.append(["correctness", label, naive, it])

    # the loud failures too
    K.banner("[2b] wrong dtype and wrong shape")
    # b is *larger* than a here, so the naive loop's a.numel() elements are all
    # inside b's buffer: it cannot crash, it can only be quietly wrong. (With b
    # smaller it would read past the end -- undefined behaviour again.)
    for label, (a, b) in {
        "float64 input": (base_a.double(), torch.randn(n, n).double()),
        "shape mismatch": (base_a, torch.randn(n, 2 * n)),
    }.items():
        try:
            mod.add_naive(a, b)
            msg = "no error (!)"
        except RuntimeError as e:
            msg = str(e).split("\n")[0][:70]
        try:
            mod.add_checked(a, b)
            cmsg = "no error (!)"
        except RuntimeError as e:
            cmsg = str(e).split("\n")[0][:70]
        print(f"    {label:<16} naive   -> {msg}")
        print(f"    {'':<16} checked -> {cmsg}")
        ROWS.append(["error handling", label, msg, cmsg])
    return rows


# ---------------------------------------------------------------------------
def speed(mod):
    K.banner("[3] is it fast? 8M floats, three kernels plus torch.add")
    a = torch.randn(N)
    b = torch.randn(N)
    jitter, med = K.noise_floor(lambda: a + b)
    print(f"    noise floor on this machine: {jitter:.1f}% spread (median {med:.2f} ms)")
    ROWS.append(["noise", "torch.add spread", f"{jitter:.1f}", "%"])

    fns = {
        "add_naive (1 thread)": lambda: mod.add_naive(a, b),
        "add_parallel (6 threads)": lambda: mod.add_parallel(a, b),
        "add_iter (TensorIterator)": lambda: mod.add_iter(a, b),
        "torch.add": lambda: a + b,
    }
    res = K.interleaved(fns, rounds=9)
    # 3 arrays x 4 bytes: read a, read b, write out
    total_bytes = 3 * N * 4
    print(f"    {'kernel':<28}{'best ms':>10}{'spread':>9}{'GB/s':>9}{'vs torch':>10}")
    base = res["torch.add"][0]
    rows = []
    for lb, (ms, spread) in res.items():
        bw = K.gbps(total_bytes, ms)
        print(f"    {lb:<28}{ms:>10.2f}{spread:>9.2f}{bw:>9.1f}{base / ms:>9.2f}x")
        rows.append([lb, f"{ms:.2f}", f"{bw:.1f}", f"{base / ms:.2f}"])
        ROWS.append(["speed 8M", lb, f"{ms:.3f} ms", f"{bw:.1f} GB/s"])
    ai = K.arithmetic_intensity(N, total_bytes)
    print(f"    arithmetic intensity: {ai:.3f} FLOP/byte -- one add per 12 bytes moved")
    ROWS.append(["speed 8M", "arithmetic intensity", f"{ai:.4f}", "FLOP/byte"])
    return rows, base


# ---------------------------------------------------------------------------
def size_sweep(mod):
    K.banner("[4] where does threading start to pay?")
    print("    L1d 32 KB/core, L2 256 KB/core, L3 12 MB shared (i7-8700K).")
    print("    'empty' is torch.empty(n) on its own -- part of every kernel's time,")
    print("    because each call allocates its own output.\n")
    print(f"    {'elements':>12}{'MB':>7}{'naive ms':>11}{'parallel ms':>13}"
          f"{'speedup':>9}{'empty ms':>10}")
    rows = []
    for n in (10_000, 100_000, 1_000_000, 8_000_000, 32_000_000):
        a = torch.randn(n)
        b = torch.randn(n)
        r = K.interleaved(
            {"naive": lambda: mod.add_naive(a, b),
             "par": lambda: mod.add_parallel(a, b),
             "empty": lambda: torch.empty(n)},
            rounds=7,
        )
        sp = r["naive"][0] / r["par"][0]
        print(f"    {n:>12,}{n * 4 / 1e6:>7.1f}{r['naive'][0]:>11.3f}"
              f"{r['par'][0]:>13.3f}{sp:>8.2f}x{r['empty'][0]:>10.3f}")
        rows.append([n, f"{r['naive'][0]:.3f}", f"{r['par'][0]:.3f}", f"{sp:.2f}",
                     f"{r['empty'][0]:.3f}"])
        ROWS.append(["thread scaling", f"n={n}", f"{sp:.2f}x",
                     f"{n * 4 / 1e6:.1f} MB, empty {r['empty'][0]:.3f} ms"])
    return rows


# ---------------------------------------------------------------------------
def dispatcher(mod):
    K.banner("[5] the dispatcher: torch.ops.p30.add")
    a = torch.randn(4)
    b = torch.randn(4)
    print(f"    torch.ops.p30.add exists: {torch.ops.p30.add}")
    print(f"    schema: {torch.ops.p30.add.default._schema}")
    ok = torch.allclose(torch.ops.p30.add(a, b), a + b)
    print(f"    result matches torch.add: {ok}")
    ROWS.append(["dispatcher", "torch.ops.p30.add matches", str(ok), ""])

    print("\n    -- autograd --")
    x = torch.randn(4, requires_grad=True)
    y = torch.randn(4)
    z_ref = x + y
    print(f"    torch.add   : requires_grad={z_ref.requires_grad}, grad_fn={z_ref.grad_fn}")
    z = mod.add_iter(x, y)
    print(f"    our add_iter: requires_grad={z.requires_grad}, grad_fn={z.grad_fn}")
    try:
        z.sum().backward()
        msg = f"backward ran, x.grad = {x.grad.tolist()}"
    except RuntimeError as e:
        msg = "backward FAILED: " + str(e).split("\n")[0][:60]
    print(f"    {msg}")
    ROWS.append(["autograd", "our op keeps grad_fn", str(z.grad_fn is not None), msg[:60]])

    print("\n    -- dtype --")
    for dt in (torch.float32, torch.float64, torch.int32):
        try:
            r = mod.add_iter(torch.ones(4, dtype=dt), torch.ones(4, dtype=dt))
            m = f"ok, out dtype {r.dtype}"
        except RuntimeError as e:
            m = "raised: " + str(e).split("\n")[0][:50]
        print(f"    add_iter on {str(dt):<16} -> {m}")
        ROWS.append(["dtype", str(dt), m, ""])


# ---------------------------------------------------------------------------
TINY = r"""
#include <torch/extension.h>
torch::Tensor scale(torch::Tensor a, double s) { return a * s; }
"""


def compile_cost():
    K.banner("[6] what the compile costs")
    # A second, tiny extension, so this section is not entangled with the
    # TORCH_LIBRARY one above (loading that .so twice aborts the process --
    # see outputs/double_load.txt).
    _, cold = K.build("p30_tiny", TINY, functions=["scale"], force=True)
    _, raw = K.build("p30_tiny", TINY, functions=["scale"], force=True)
    _, cached = K.build("p30_tiny", TINY, functions=["scale"])
    print(f"    a one-line extension, first build        : {cold:6.1f} s")
    print(f"    load_inline again, identical source      : {raw:6.1f} s   <- NOT free")
    print(f"    kernels_lib.build, identical source      : {cached * 1e3:6.3f} ms  <- imports the .so")
    print(f"    what our own cache is worth              : {raw / max(cached, 1e-9):6.0f}x")
    print("\n    load_inline compiles with -DTORCH_EXTENSION_NAME=<name>_v<version>")
    print("    and bumps <version> on every call it has not already seen in this")
    print("    process. A new define is a new compile command, so ninja rebuilds")
    print("    even though not one character of C++ changed.")
    ROWS.append(["compile", "tiny extension cold", f"{cold:.2f}", "s"])
    ROWS.append(["compile", "load_inline, unchanged source", f"{raw:.2f}", "s (no cache)"])
    ROWS.append(["compile", "kernels_lib cache hit", f"{cached:.4f}", "s"])
    ROWS.append(["compile", "cached rebuild", f"{cached:.2f}", "s"])
    # capture the abort, in a child process so it cannot take this one down
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    child = (
        "import sys; sys.path.insert(0, %r); import run, kernels_lib as K;"
        "K.build('p30_add', run.CPP, functions=['add_iter'], force=True);"
        "K.build('p30_add', run.CPP, functions=['add_iter'], force=True);"
        "print('no abort')" % here
    )
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    text = (proc.stdout + proc.stderr).strip()
    (OUT / "double_load.txt").write_text(text + "\n")
    first_line = next((ln for ln in text.splitlines() if "TORCH_LIBRARY" in ln), text[:80])
    print(f"\n    loading the same TORCH_LIBRARY twice, in a child process:")
    print(f"      exit code {proc.returncode}: {first_line[:100]}")
    ROWS.append(["compile", "double load exit code", str(proc.returncode), "aborts"])

    a = torch.randn(N)
    b = torch.randn(N)
    per_call = K.best_of(lambda: a + b)[0]
    calls = cold * 1e3 / max(per_call, 1e-9)
    print(f"\n    an 8M-element add takes {per_call:.2f} ms, so a {cold:.0f} s compile")
    print(f"    pays for itself only after ~{calls:,.0f} calls -- if it were free per call.")
    ROWS.append(["compile", "break-even calls", f"{calls:.0f}", "calls"])


# ---------------------------------------------------------------------------
def figure(speed_rows, sweep_rows, base_ms):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    labels = [r[0].split(" (")[0] for r in speed_rows]
    ms = [float(r[1]) for r in speed_rows]
    colors = ["#c44", "#e8a33d", "#4a7", "#468"]
    ax.barh(range(len(ms)), ms, color=colors[: len(ms)])
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("ms for 8M-element add (lower is better)")
    ax.set_title("Four ways to add two tensors")
    for i, v in enumerate(ms):
        ax.text(v, i, f" {v:.1f}", va="center", fontsize=9)

    ax = axes[1]
    bw = [float(r[2]) for r in speed_rows]
    ax.barh(range(len(bw)), bw, color=colors[: len(bw)])
    ax.set_yticks(range(len(bw)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("effective DRAM bandwidth (GB/s)")
    ax.set_title("The real limit is memory, not adds")
    for i, v in enumerate(bw):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=9)

    ax = axes[2]
    ns = [r[0] for r in sweep_rows]
    sp = [float(r[3]) for r in sweep_rows]
    ax.semilogx(ns, sp, "o-", color="#468")
    ax.axhline(1.0, color="#999", ls="--", lw=1)
    ax.set_xlabel("number of elements")
    ax.set_ylabel("parallel / naive speedup")
    ax.set_title("Threads only pay off past the cache")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = OUT / "cpp_extension.png"
    fig.savefig(p, dpi=110)
    print(f"\n    wrote {p}")


# ---------------------------------------------------------------------------
def main():
    t_start = time.perf_counter()
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"cuda visible: {torch.cuda.is_available()}")
    mod, secs = build_ext()
    correctness(mod)
    speed_rows, base = speed(mod)
    sweep_rows = size_sweep(mod)
    dispatcher(mod)
    compile_cost()
    figure(speed_rows, sweep_rows, base)
    K.write_csv(OUT / "findings.csv", ROWS, ["section", "what", "value", "note"])
    print(f"\n    wrote {OUT / 'findings.csv'}")
    print(f"\ntotal wall time: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
