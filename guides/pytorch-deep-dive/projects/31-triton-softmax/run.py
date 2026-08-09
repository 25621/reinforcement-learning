"""Project 31 -- softmax as a hand-written kernel.

The guide asks for this in Triton. Triton needs compute capability sm_70 and
this machine's GPU is sm_61, so the kernel is written in C++ instead -- same
structure (one program per row, the row held in fast memory), different
hardware. The Triton version is printed and saved next to the C++ so the
mapping is visible.

Sections
  1. how many times does softmax read its input?
  2. one C++ program per row: three passes, all inside the cache
  3. online softmax: two passes instead of three, more exp calls
  4. why the max subtraction is not optional
  5. why the hand kernel loses to F.softmax, and what closes the gap
  6. threads over rows
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "30-cpp-extension-for-elementwise-add"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import kernels_lib as K  # noqa: E402

OUT = K.outputs_dir(__file__)
ROWS = []
R, C = 4096, 4096          # 4096 rows of 4096 -- the width the guide asks for
BYTES = 2 * R * C * 4      # read the input once, write the output once


CPP = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <cmath>

// ---------------------------------------------------------------------------
// One "program" per row. In Triton you would write
//     row = tl.program_id(0)
// and get one instance of the kernel per row, each holding its row in on-chip
// SRAM. Here at::parallel_for hands each thread a range of rows, and the row
// (16 KB) stays in that core's L1/L2 cache while we make our passes over it.
// The important part is the same in both worlds: the row is loaded from slow
// memory ONCE and every pass after that hits fast memory.
// ---------------------------------------------------------------------------

// Three passes: max, then exp-and-sum, then divide.
torch::Tensor softmax_3pass(torch::Tensor x) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kFloat);
  auto out = torch::empty_like(x);
  const int64_t rows = x.size(0), cols = x.size(1);
  const float* px = x.data_ptr<float>();
  float* po = out.data_ptr<float>();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* xr = px + r * cols;
      float* orow = po + r * cols;
      float m = -INFINITY;
      for (int64_t i = 0; i < cols; ++i) m = xr[i] > m ? xr[i] : m;   // pass 1
      float s = 0.f;
      for (int64_t i = 0; i < cols; ++i) { float e = std::exp(xr[i] - m); orow[i] = e; s += e; }  // pass 2
      const float inv = 1.f / s;
      for (int64_t i = 0; i < cols; ++i) orow[i] *= inv;              // pass 3
    }
  });
  return out;
}

// Two passes: keep a running max AND a running sum together.
// When a bigger value shows up, every exp we already added used the old max,
// so the whole running sum is off by a constant factor exp(m_old - m_new).
// One multiply repairs it. This is the trick FlashAttention is built on.
torch::Tensor softmax_online(torch::Tensor x) {
  TORCH_CHECK(x.dim() == 2 && x.is_contiguous() && x.scalar_type() == torch::kFloat);
  auto out = torch::empty_like(x);
  const int64_t rows = x.size(0), cols = x.size(1);
  const float* px = x.data_ptr<float>();
  float* po = out.data_ptr<float>();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* xr = px + r * cols;
      float* orow = po + r * cols;
      float m = -INFINITY, s = 0.f;
      for (int64_t i = 0; i < cols; ++i) {                            // pass 1
        float v = xr[i];
        if (v > m) { s *= std::exp(m - v); m = v; }   // rescale what we have
        s += std::exp(v - m);
      }
      const float inv = 1.f / s;
      for (int64_t i = 0; i < cols; ++i) orow[i] = std::exp(xr[i] - m) * inv;  // pass 2
    }
  });
  return out;
}

// The same three passes with the max subtraction removed. Mathematically
// identical -- exp(x)/sum(exp(x)) -- and unusable in float32.
torch::Tensor softmax_unstable(torch::Tensor x) {
  auto out = torch::empty_like(x);
  const int64_t rows = x.size(0), cols = x.size(1);
  const float* px = x.data_ptr<float>();
  float* po = out.data_ptr<float>();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* xr = px + r * cols;
      float* orow = po + r * cols;
      float s = 0.f;
      for (int64_t i = 0; i < cols; ++i) { float e = std::exp(xr[i]); orow[i] = e; s += e; }
      for (int64_t i = 0; i < cols; ++i) orow[i] /= s;
    }
  });
  return out;
}
"""

# Identical algorithm, compiled with -ffast-math so gcc may reassociate the
# reduction and call the vector exp. Kept as a separate module so both versions
# exist in the same process and can be timed against each other.
CPP_FAST = CPP.replace("softmax_3pass", "softmax_3pass_fast") \
              .replace("softmax_online", "softmax_online_fast") \
              .replace("softmax_unstable", "softmax_unstable_fast")


TRITON_REFERENCE = '''\
# What this kernel looks like in Triton (cannot run here: needs sm_70+, this
# GPU is sm_61). Compare it line by line with softmax_3pass above.

import triton
import triton.language as tl

@triton.jit
def softmax_kernel(x_ptr, o_ptr, stride, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                      # <- one program per row
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)                   # <- the same max subtraction
    e = tl.exp(x)
    tl.store(o_ptr + row * stride + cols, e / tl.sum(e, axis=0), mask=mask)

def softmax(x):
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)        # the whole row in one block
    softmax_kernel[(rows,)](x, out, x.stride(0), cols, BLOCK=BLOCK, num_warps=8)
    return out

# The three passes are invisible because `x` lives in SRAM the whole time:
# tl.max, tl.exp and tl.sum all read the same on-chip copy. That is exactly
# what the C++ version gets from the L1 cache. The kernel is one program per
# row in both cases, and both are limited by how fast the row arrives from
# main memory, not by the arithmetic.
'''


def naive_ops(x):
    """Softmax written as separate PyTorch ops -- five kernels, five passes."""
    m = x.max(dim=-1, keepdim=True).values
    z = x - m
    e = torch.exp(z)
    s = e.sum(dim=-1, keepdim=True)
    return e / s


def memory_traffic():
    K.banner("[1] how many times does softmax touch memory?")
    x = torch.randn(R, C)
    per_tensor = R * C * 4 / 1e6
    print(f"    input is {R}x{C} float32 = {per_tensor:.0f} MB\n")
    plan = [
        ("max(dim=-1)", "read x", 1, 0),
        ("x - m", "read x, write z", 1, 1),
        ("exp(z)", "read z, write e", 1, 1),
        ("e.sum(-1)", "read e", 1, 0),
        ("e / s", "read e, write out", 1, 1),
    ]
    tot_r = sum(p[2] for p in plan)
    tot_w = sum(p[3] for p in plan)
    for name, what, r, w in plan:
        print(f"      {name:<14} {what}")
    print(f"\n    op-by-op total : {tot_r} full reads + {tot_w} full writes "
          f"= {(tot_r + tot_w) * per_tensor:.0f} MB")
    print(f"    one fused pass : 1 read + 1 write = {2 * per_tensor:.0f} MB")
    print(f"    ideal saving   : {(tot_r + tot_w) / 2:.1f}x less memory traffic")
    ROWS.append(["traffic", "unfused MB", f"{(tot_r + tot_w) * per_tensor:.0f}", "5 reads + 3 writes"])
    ROWS.append(["traffic", "fused MB", f"{2 * per_tensor:.0f}", "1 read + 1 write"])
    del x


def build_all():
    K.banner("[2] compiling two versions of the same three kernels")
    mod, s1 = K.build("p31_softmax", CPP,
                      functions=["softmax_3pass", "softmax_online", "softmax_unstable"])
    print(f"    default flags   : {s1:.1f} s")
    fast, s2 = K.build("p31_softmax_fast", CPP_FAST,
                       functions=["softmax_3pass_fast", "softmax_online_fast",
                                  "softmax_unstable_fast"],
                       extra_cflags=["-ffast-math"])
    print(f"    with -ffast-math: {s2:.1f} s")
    ROWS.append(["compile", "default", f"{s1:.1f}", "s"])
    ROWS.append(["compile", "-ffast-math", f"{s2:.1f}", "s"])
    return mod, fast


def correctness(mod, fast):
    K.banner("[3] do they agree with F.softmax?")
    x = torch.randn(512, C) * 3
    ref = F.softmax(x, dim=-1)
    cands = {
        "softmax_3pass": mod.softmax_3pass(x),
        "softmax_online": mod.softmax_online(x),
        "3pass -ffast-math": fast.softmax_3pass_fast(x),
        "naive_ops (python)": naive_ops(x),
    }
    for lb, got in cands.items():
        err = K.max_abs_diff(got, ref)
        rowsum = got.sum(-1)
        print(f"    {lb:<20} max abs diff {err:.2e}   row sums in "
              f"[{rowsum.min():.6f}, {rowsum.max():.6f}]")
        ROWS.append(["correctness", lb, f"{err:.2e}", "vs F.softmax"])


def stability(mod):
    K.banner("[4] why the max subtraction is not optional")
    print(f"    float32 holds up to ~3.4e38, and exp overflows it at x = "
          f"{torch.log(torch.tensor(3.4e38)).item():.1f}\n")
    print(f"    {'input scale':>12}{'stable ok':>12}{'unstable ok':>14}"
          f"{'unstable nan/inf':>18}")
    rows = []
    for scale in (1, 10, 50, 88, 100, 1000):
        x = torch.randn(64, 1024) * scale
        ref = F.softmax(x, dim=-1)
        good = mod.softmax_3pass(x)
        bad = mod.softmax_unstable(x)
        g_ok = torch.allclose(good, ref, atol=1e-6)
        n_bad = int((~torch.isfinite(bad)).sum())
        b_ok = torch.isfinite(bad).all() and torch.allclose(bad, ref, atol=1e-6)
        print(f"    {scale:>12}{str(bool(g_ok)):>12}{str(bool(b_ok)):>14}{n_bad:>18,}")
        rows.append([scale, bool(g_ok), bool(b_ok), n_bad])
        ROWS.append(["stability", f"scale={scale}", f"stable={bool(g_ok)}",
                     f"unstable nan/inf={n_bad}"])
    return rows


def speed(mod, fast):
    K.banner("[5] speed: 4096 x 4096, best of 13 interleaved rounds")
    x = torch.randn(R, C)
    jitter, med = K.noise_floor(lambda: F.softmax(x, dim=-1), rounds=9)
    print(f"    noise floor: {jitter:.1f}% (median {med:.1f} ms)\n")
    ROWS.append(["noise", "F.softmax spread", f"{jitter:.1f}", "%"])

    fns = {
        "naive_ops (5 torch ops)": lambda: naive_ops(x),
        "F.softmax": lambda: F.softmax(x, dim=-1),
        "cpp 3-pass": lambda: mod.softmax_3pass(x),
        "cpp online (2-pass)": lambda: mod.softmax_online(x),
        "cpp 3-pass -ffast-math": lambda: fast.softmax_3pass_fast(x),
        "cpp online -ffast-math": lambda: fast.softmax_online_fast(x),
    }
    res = K.interleaved(fns, rounds=13, warmup=3)
    base = res["F.softmax"][0]
    print(f"    {'kernel':<26}{'best ms':>10}{'spread':>9}{'GB/s':>9}{'vs F.softmax':>14}")
    rows = []
    for lb, (ms, sp) in res.items():
        traffic = BYTES if "naive_ops" not in lb else 4 * R * C * 4
        print(f"    {lb:<26}{ms:>10.1f}{sp:>9.1f}{K.gbps(traffic, ms):>9.1f}"
              f"{base / ms:>13.2f}x")
        rows.append([lb, f"{ms:.1f}", f"{K.gbps(traffic, ms):.1f}", f"{base / ms:.2f}"])
        ROWS.append(["speed", lb, f"{ms:.1f} ms", f"{base / ms:.2f}x vs F.softmax"])
    return rows


def thread_scaling(mod):
    K.banner("[6] how the row-parallel kernel scales with threads")
    x = torch.randn(2048, C)

    # Each candidate sets the thread count itself, so the four configurations
    # can be interleaved. Timing all of 1-thread and then all of 6-thread would
    # charge whatever else the machine was doing to whichever went second.
    def run_with(t):
        torch.set_num_threads(t)
        return mod.softmax_3pass(x)

    counts = (1, 2, 4, 6)
    res = K.interleaved({t: (lambda t=t: run_with(t)) for t in counts},
                        rounds=7, warmup=2)
    torch.set_num_threads(K.THREADS)
    print(f"    {'threads':>8}{'ms':>10}{'speedup':>10}{'efficiency':>12}")
    rows = []
    one = res[1][0]
    for t in counts:
        ms = res[t][0]
        print(f"    {t:>8}{ms:>10.1f}{one / ms:>9.2f}x{one / ms / t * 100:>11.0f}%")
        rows.append([t, f"{ms:.1f}", f"{one / ms:.2f}"])
        ROWS.append(["threads", f"{t} threads", f"{ms:.1f} ms", f"{one / ms:.2f}x"])
    return rows


def figure(speed_rows, stab_rows, thread_rows):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    labels = [r[0] for r in speed_rows]
    ms = [float(r[1]) for r in speed_rows]
    cols = ["#c44", "#468", "#e8a33d", "#e8a33d", "#4a7", "#4a7"]
    ax.barh(range(len(ms)), ms, color=cols[: len(ms)])
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("ms, 4096x4096 softmax")
    ax.set_title("Hand kernel vs the library")
    for i, v in enumerate(ms):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)

    ax = axes[1]
    scales = [r[0] for r in stab_rows]
    nan = [r[3] for r in stab_rows]
    ax.semilogx(scales, nan, "o-", color="#c44")
    ax.set_xlabel("input scale (std of x)")
    ax.set_ylabel("non-finite outputs")
    ax.set_title("Softmax without the max subtraction")
    ax.grid(alpha=0.3)

    ax = axes[2]
    t = [r[0] for r in thread_rows]
    sp = [float(r[2]) for r in thread_rows]
    ax.plot(t, sp, "o-", color="#468", label="measured")
    ax.plot(t, t, "--", color="#999", label="perfect")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup")
    ax.set_title("Rows are independent, memory is not")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "softmax.png", dpi=110)
    print(f"\n    wrote {OUT / 'softmax.png'}")


def main():
    t0 = time.perf_counter()
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"cuda usable: {torch.cuda.is_available()}")
    (OUT / "triton_softmax.py").write_text(TRITON_REFERENCE)
    memory_traffic()
    mod, fast = build_all()
    correctness(mod, fast)
    stab = stability(mod)
    sp = speed(mod, fast)
    th = thread_scaling(mod)
    figure(sp, stab, th)
    K.write_csv(OUT / "findings.csv", ROWS, ["section", "what", "value", "note"])
    print(f"    wrote {OUT / 'findings.csv'} and {OUT / 'triton_softmax.py'}")
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
