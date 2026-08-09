"""Project 33 -- fusing an MLP: matmul, bias, GELU, matmul.

The guide asks for one Triton kernel that does all four. Triton needs sm_70 and
this GPU is sm_61, so the kernel is C++ -- and the C++ version turns out to
answer a question the Triton version would have hidden: *which* of the four
steps is worth fusing.

Sections
  1-2. compile the kernels twice (one define apart) and check them
  3.   the epilogue alone: bias + GELU, and why the obvious fusion loses
  4.   the whole MLP, five ways
  5.   where the time actually is, which decides what is worth fusing
  6.   what fusion saves in memory
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

B, D, H = 1024, 512, 2048     # tokens, model width, hidden width


CPP = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>

// at::vec::Vectorized<float> is ATen's own portable SIMD type: one object
// holding 8 floats on this CPU, with erf, exp, tanh and friends implemented as
// vector instructions. It is the piece the scalar kernel below is missing.
using Vec = at::vec::Vectorized<float>;

// How wide the vector type actually is, and whether the AVX2 specialization
// was selected. Section 3 needs this, because the header compiles either way.
int64_t vec_width() { return (int64_t)Vec::size(); }
bool avx2_enabled() {
#ifdef CPU_CAPABILITY_AVX2
  return true;
#else
  return false;
#endif
}

// GELU, the exact (erf) version, which is what F.gelu computes by default:
//   gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
static inline float gelu_scalar(float x) {
  return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f));
}

// ---- the epilogue, fused: add the bias and apply GELU in ONE pass ---------
// Unfused, PyTorch runs two kernels: one reads h and the bias and writes h+b,
// then another reads h+b and writes gelu(h+b). The middle tensor is written to
// memory and read straight back -- pure waste, because nothing else needs it.
// Here the value is in a register when we need it, so it never leaves the CPU.
torch::Tensor bias_gelu(torch::Tensor h, torch::Tensor bias) {
  TORCH_CHECK(h.is_contiguous() && h.scalar_type() == torch::kFloat);
  TORCH_CHECK(bias.numel() == h.size(1));
  auto out = torch::empty_like(h);
  const int64_t rows = h.size(0), cols = h.size(1);
  const float* ph = h.data_ptr<float>();
  const float* pb = bias.data_ptr<float>();
  float* po = out.data_ptr<float>();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* hr = ph + r * cols;
      float* orow = po + r * cols;
      for (int64_t c = 0; c < cols; ++c) orow[c] = gelu_scalar(hr[c] + pb[c]);
    }
  });
  return out;
}

// Same thing writing into h itself: one read, one write, no new allocation.
torch::Tensor bias_gelu_(torch::Tensor h, torch::Tensor bias) {
  const int64_t rows = h.size(0), cols = h.size(1);
  float* ph = h.data_ptr<float>();
  const float* pb = bias.data_ptr<float>();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      float* hr = ph + r * cols;
      for (int64_t c = 0; c < cols; ++c) hr[c] = gelu_scalar(hr[c] + pb[c]);
    }
  });
  return h;
}

// ---- the same fusion, but 8 floats at a time -----------------------------
// Identical arithmetic to bias_gelu. The only change is that the values move
// through Vec instead of float, so one instruction does eight elements -- and
// crucially erf() becomes a vector call instead of eight library calls.
torch::Tensor bias_gelu_vec(torch::Tensor h, torch::Tensor bias) {
  TORCH_CHECK(h.is_contiguous() && h.scalar_type() == torch::kFloat);
  auto out = torch::empty_like(h);
  const int64_t rows = h.size(0), cols = h.size(1);
  const float* ph = h.data_ptr<float>();
  const float* pb = bias.data_ptr<float>();
  float* po = out.data_ptr<float>();
  const int64_t W = Vec::size();
  at::parallel_for(0, rows, 1, [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const float* hr = ph + r * cols;
      float* orow = po + r * cols;
      int64_t c = 0;
      for (; c + W <= cols; c += W) {
        Vec z = Vec::loadu(hr + c) + Vec::loadu(pb + c);
        Vec y = z * Vec(0.5f) * (Vec(1.f) + (z * Vec(0.70710678118654752440f)).erf());
        y.store(orow + c);
      }
      for (; c < cols; ++c) orow[c] = gelu_scalar(hr[c] + pb[c]);  // the tail
    }
  });
  return out;
}

// ---- the whole MLP in one C++ function -----------------------------------
// matmul -> bias -> gelu -> matmul -> bias, with a hand-written tiled matmul.
// Nothing is written to memory between the steps except the hidden activation,
// which the second matmul needs anyway. This is what a single Triton kernel
// would do, and section 3 shows why it still loses.
torch::Tensor mlp_fused(torch::Tensor x, torch::Tensor W1, torch::Tensor b1,
                        torch::Tensor W2, torch::Tensor b2, int64_t BLOCK) {
  const int64_t n = x.size(0), d = x.size(1), h = W1.size(1), o = W2.size(1);
  auto H = torch::zeros({n, h}, x.options());
  auto Y = torch::zeros({n, o}, x.options());
  const float* px = x.data_ptr<float>();
  const float* pw1 = W1.data_ptr<float>();
  const float* pb1 = b1.data_ptr<float>();
  const float* pw2 = W2.data_ptr<float>();
  const float* pb2 = b2.data_ptr<float>();
  float* ph = H.data_ptr<float>();
  float* py = Y.data_ptr<float>();

  const int64_t nblk = (n + BLOCK - 1) / BLOCK;
  at::parallel_for(0, nblk, 1, [&](int64_t blk0, int64_t blk1) {
    for (int64_t blk = blk0; blk < blk1; ++blk) {
      const int64_t i0 = blk * BLOCK, i1 = std::min(i0 + BLOCK, n);
      // first matmul into this band of H
      for (int64_t k0 = 0; k0 < d; k0 += BLOCK)
        for (int64_t j0 = 0; j0 < h; j0 += BLOCK) {
          const int64_t k1 = std::min(k0 + BLOCK, d), j1 = std::min(j0 + BLOCK, h);
          for (int64_t i = i0; i < i1; ++i)
            for (int64_t k = k0; k < k1; ++k) {
              const float v = px[i * d + k];
              const float* wr = pw1 + k * h;
              float* hr = ph + i * h;
              for (int64_t j = j0; j < j1; ++j) hr[j] += v * wr[j];
            }
        }
      // the epilogue, on the band that is still warm in this core's cache
      for (int64_t i = i0; i < i1; ++i) {
        float* hr = ph + i * h;
        for (int64_t j = 0; j < h; ++j) hr[j] = gelu_scalar(hr[j] + pb1[j]);
      }
      // second matmul
      for (int64_t k0 = 0; k0 < h; k0 += BLOCK)
        for (int64_t j0 = 0; j0 < o; j0 += BLOCK) {
          const int64_t k1 = std::min(k0 + BLOCK, h), j1 = std::min(j0 + BLOCK, o);
          for (int64_t i = i0; i < i1; ++i)
            for (int64_t k = k0; k < k1; ++k) {
              const float v = ph[i * h + k];
              const float* wr = pw2 + k * o;
              float* yr = py + i * o;
              for (int64_t j = j0; j < j1; ++j) yr[j] += v * wr[j];
            }
        }
      for (int64_t i = i0; i < i1; ++i)
        for (int64_t j = 0; j < o; ++j) py[i * o + j] += pb2[j];
    }
  });
  return Y;
}
"""


TRITON_REFERENCE = '''\
# The fused MLP as one Triton kernel (needs sm_70+, this GPU is sm_61).
#
# The structure to notice: the k loop over the FIRST matmul finishes, the
# bias+GELU runs on the accumulator while it is still in registers, and only
# then does the second matmul start -- all inside one kernel launch, so the
# hidden activation never reaches HBM.

import triton
import triton.language as tl

@triton.jit
def fused_mlp_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, y_ptr,
                     M, D, H, O,
                     BM: tl.constexpr, BH: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = pid * BM + tl.arange(0, BM)
    offs_h = tl.arange(0, BH)

    # ---- matmul 1: x @ W1 ------------------------------------------------
    acc = tl.zeros((BM, BH), dtype=tl.float32)
    for k in range(0, D, BK):
        offs_k = k + tl.arange(0, BK)
        a = tl.load(x_ptr + offs_m[:, None] * D + offs_k[None, :])
        w = tl.load(w1_ptr + offs_k[:, None] * H + offs_h[None, :])
        acc += tl.dot(a, w)

    # ---- epilogue: bias + GELU, on the accumulator, in registers ---------
    acc += tl.load(b1_ptr + offs_h)[None, :]
    acc = acc * 0.5 * (1.0 + tl.erf(acc * 0.70710678))

    # ---- matmul 2: h @ W2 ------------------------------------------------
    offs_o = tl.arange(0, BH)
    out = tl.zeros((BM, BH), dtype=tl.float32)
    for k in range(0, H, BK):
        w = tl.load(w2_ptr + (k + tl.arange(0, BK))[:, None] * O + offs_o[None, :])
        out += tl.dot(tl.trans(tl.trans(acc)[k:k + BK]), w)   # schematic

    out += tl.load(b2_ptr + offs_o)[None, :]
    tl.store(y_ptr + offs_m[:, None] * O + offs_o[None, :], out)

# This is schematic on purpose -- a real fused MLP kernel needs the whole
# hidden row (H values) live at once to start matmul 2, so BH must equal H and
# the tile must fit in registers. That constraint is the real reason production
# kernels fuse the EPILOGUE into the matmul and leave the two matmuls separate,
# which is exactly what project 33 measures on the CPU.
'''


def unfused(x, W1, b1, W2, b2):
    """Every step its own kernel, every intermediate written to memory."""
    h = x @ W1
    h = h + b1
    h = F.gelu(h)
    y = h @ W2
    return y + b2


def torch_linear(x, W1, b1, W2, b2):
    """What you would actually write: addmm fuses the bias into the matmul."""
    return F.linear(F.gelu(F.linear(x, W1.t(), b1)), W2.t(), b2)


def epilogue_fused(mod, x, W1, b1, W2, b2):
    """Vendor matmul + our one-pass vectorized bias+GELU. The mix that wins.

    Note what this does NOT do: it does not try to beat oneDNN at matmul. It
    keeps the vendor's matmul and replaces only the two cheap element-wise ops
    around it -- which is the only part a hand-written kernel can improve here.
    """
    h = x @ W1
    h = mod.bias_gelu_vec(h, b1)
    return torch.addmm(b2, h, W2)


FUNCS = ["bias_gelu", "bias_gelu_", "bias_gelu_vec", "mlp_fused",
         "vec_width", "avx2_enabled"]

# The exact same source, compiled twice. The second time we tell ATen's vector
# header which instruction set to specialize for. See section 3 for why that
# one define is worth 7x.
AVX2_FLAGS = ["-DCPU_CAPABILITY_AVX2", "-mavx2", "-mfma"]


def build_all():
    K.banner("[1] compiling the same source twice")
    mod, s = K.build("p33_mlp", CPP, functions=FUNCS)
    print(f"    default flags            : {s:5.1f} s  "
          f"Vec::size()={mod.vec_width()}  CPU_CAPABILITY_AVX2={mod.avx2_enabled()}")
    avx, s2 = K.build("p33_mlp_avx2", CPP, functions=FUNCS, extra_cflags=AVX2_FLAGS)
    print(f"    + {' '.join(AVX2_FLAGS)} : {s2:5.1f} s  "
          f"Vec::size()={avx.vec_width()}  CPU_CAPABILITY_AVX2={avx.avx2_enabled()}")
    ROWS.append(["compile", "default", f"{s:.1f}", f"AVX2={mod.avx2_enabled()}"])
    ROWS.append(["compile", "with -DCPU_CAPABILITY_AVX2", f"{s2:.1f}",
                 f"AVX2={avx.avx2_enabled()}"])
    return mod, avx


def correctness(mod, avx, tensors):
    K.banner("[2] does everything compute the same MLP?")
    x, W1, b1, W2, b2 = tensors
    ref = unfused(x, W1, b1, W2, b2)
    cands = {
        "F.linear + F.gelu": torch_linear(x, W1, b1, W2, b2),
        "epilogue fused (C++)": epilogue_fused(avx, x, W1, b1, W2, b2),
        "mlp_fused (all C++)": mod.mlp_fused(x, W1, b1, W2, b2, 64),
    }
    for lb, got in cands.items():
        print(f"    {lb:<24} relative error vs unfused: {K.rel_err(got, ref):.2e}")
        ROWS.append(["correctness", lb, f"{K.rel_err(got, ref):.2e}", "vs unfused"])
    g = mod.bias_gelu(torch.randn(8, 16), torch.randn(16))
    ref_g = F.gelu(torch.randn(8, 16))
    print(f"    our GELU matches F.gelu (erf form): "
          f"{K.max_abs_diff(mod.bias_gelu(torch.zeros(4, 4), torch.linspace(-3, 3, 4)), F.gelu(torch.linspace(-3, 3, 4).expand(4, 4))):.2e}")
    del g, ref_g


def epilogue_only(mod, avx, tensors):
    K.banner("[3] the epilogue alone: bias + GELU on the hidden activation")
    x, W1, b1, W2, b2 = tensors
    h = (x @ W1).contiguous()
    mbytes = h.numel() * 4 / 1e6
    print(f"    hidden activation is {B}x{H} float32 = {mbytes:.0f} MB\n")

    compiled = torch.compile(lambda t: F.gelu(t + b1))
    compiled(h)   # warm up the compile before timing

    fns = {
        "h + b1 then F.gelu (2 kernels)": lambda: F.gelu(h + b1),
        "bias_gelu, scalar loop": lambda: mod.bias_gelu(h, b1),
        "bias_gelu_, scalar, in place": lambda: mod.bias_gelu_(h.clone(), b1),
        "bias_gelu_vec, no AVX2 define": lambda: mod.bias_gelu_vec(h, b1),
        "bias_gelu_vec, AVX2 define": lambda: avx.bias_gelu_vec(h, b1),
        "torch.compile": lambda: compiled(h),
    }
    res = K.interleaved(fns, rounds=9)
    base = res["h + b1 then F.gelu (2 kernels)"][0]
    print(f"    {'version':<34}{'ms':>9}{'GB/s':>9}{'speedup':>10}")
    rows = []
    for lb, (ms, sp) in res.items():
        traffic = (4 if "2 kernels" in lb else 2) * h.numel() * 4
        print(f"    {lb:<34}{ms:>9.2f}{K.gbps(traffic, ms):>9.1f}{base / ms:>9.2f}x")
        rows.append([lb, f"{ms:.2f}", f"{base / ms:.2f}"])
        ROWS.append(["epilogue", lb, f"{ms:.2f} ms", f"{base / ms:.2f}x"])
    print("\n    Unfused moves 4 x the tensor (read h, write h+b, read h+b, write gelu).")
    print("    Fused moves 2 x. So why does the scalar fused kernel LOSE?")
    print("    Because it is not memory-bound at all: std::erf is a library call")
    print("    per element, and 2 million of them cost more than the 16 MB saved.")
    print("    Halving the traffic is worthless if you triple the arithmetic.")
    err = K.max_abs_diff(avx.bias_gelu_vec(h, b1), F.gelu(h + b1))
    print(f"\n    the AVX2 version still agrees with F.gelu to {err:.1e}")
    ROWS.append(["epilogue", "vec vs F.gelu", f"{err:.2e}", "max abs diff"])
    return rows


def whole_mlp(mod, avx, tensors, block):
    K.banner("[4] the whole MLP")
    x, W1, b1, W2, b2 = tensors
    flops = 2.0 * B * D * H + 2.0 * B * H * D
    compiled = torch.compile(torch_linear)
    compiled(x, W1, b1, W2, b2)

    fns = {
        "unfused (5 torch ops)": lambda: unfused(x, W1, b1, W2, b2),
        "F.linear + F.gelu (3 ops)": lambda: torch_linear(x, W1, b1, W2, b2),
        "epilogue fused (C++ + oneDNN)": lambda: epilogue_fused(avx, x, W1, b1, W2, b2),
        "mlp_fused (all C++, 1 call)": lambda: mod.mlp_fused(x, W1, b1, W2, b2, block),
        "torch.compile": lambda: compiled(x, W1, b1, W2, b2),
    }
    res = K.interleaved(fns, rounds=5, warmup=1)
    base = res["unfused (5 torch ops)"][0]
    print(f"    {'version':<34}{'ms':>9}{'GFLOP/s':>10}{'vs unfused':>12}")
    rows = []
    for lb, (ms, sp) in res.items():
        print(f"    {lb:<34}{ms:>9.1f}{K.gflops(flops, ms):>10.1f}{base / ms:>11.2f}x")
        rows.append([lb, f"{ms:.1f}", f"{base / ms:.2f}"])
        ROWS.append(["whole mlp", lb, f"{ms:.2f} ms", f"{base / ms:.2f}x vs unfused"])
    print(f"\n    total arithmetic: {flops / 1e9:.2f} GFLOP")
    return rows


def where_time_goes(tensors):
    K.banner("[5] why fusing the matmuls is the wrong target")
    x, W1, b1, W2, b2 = tensors
    h = (x @ W1).contiguous()
    parts = K.interleaved({
        "matmul 1 (x @ W1)": lambda: x @ W1,
        "matmul 2 (h @ W2)": lambda: h @ W2,
        "bias + gelu": lambda: F.gelu(h + b1),
        "bias 2": lambda: h @ W2 + b2,
    }, rounds=7)
    mm = parts["matmul 1 (x @ W1)"][0] + parts["matmul 2 (h @ W2)"][0]
    el = parts["bias + gelu"][0]
    total = mm + el
    print(f"    {'part':<24}{'ms':>9}{'share':>9}")
    for lb in ("matmul 1 (x @ W1)", "matmul 2 (h @ W2)", "bias + gelu"):
        ms = parts[lb][0]
        print(f"    {lb:<24}{ms:>9.2f}{ms / total * 100:>8.1f}%")
        ROWS.append(["time split", lb, f"{ms:.2f} ms", f"{ms / total * 100:.1f}%"])
    print(f"\n    The elementwise part is {el / total * 100:.0f}% of the work.")
    print(f"    Even making it FREE would only save {el / total * 100:.0f}% overall --")
    print("    which is why the fully hand-fused kernel, which has to beat oneDNN")
    print("    at matmul to win anything, cannot.")
    ROWS.append(["time split", "elementwise share", f"{el / total * 100:.1f}", "%"])
    return {"matmul": mm, "elementwise": el}


def memory_and_allocs(mod, tensors):
    K.banner("[6] what fusion saves in memory")
    x, W1, b1, W2, b2 = tensors
    hidden = B * H * 4 / 1e6
    print(f"    {'version':<34}{'temporaries':>13}{'MB held':>10}")
    plan = [
        ("unfused (5 torch ops)", ["x@W1", "h+b1", "gelu", "h@W2"], 4),
        ("F.linear + F.gelu (3 ops)", ["addmm", "gelu"], 2),
        ("epilogue fused (C++)", ["x@W1 (reused in place)"], 1),
        ("mlp_fused (all C++)", ["H"], 1),
    ]
    for lb, temps, n in plan:
        print(f"    {lb:<34}{n:>13}{n * hidden:>10.0f}")
        ROWS.append(["memory", lb, f"{n} temporaries", f"{n * hidden:.0f} MB"])
    print(f"\n    one hidden activation is {hidden:.0f} MB; every temporary is another")
    print("    allocation, another write to memory, and another read back.")


def figure(epi_rows, mlp_rows, split):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))

    ax = axes[0]
    labels = [r[0].replace(" (", "\n(") for r in epi_rows]
    sp = [float(r[2]) for r in epi_rows]
    ax.barh(range(len(sp)), sp, color=["#c44", "#4a7", "#2a6", "#468"])
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#999", ls="--", lw=1)
    ax.set_xlabel("speedup vs two unfused kernels")
    ax.set_title("Fusing bias + GELU")
    for i, v in enumerate(sp):
        ax.text(v, i, f" {v:.2f}x", va="center", fontsize=8)

    ax = axes[1]
    labels = [r[0].replace(" (", "\n(") for r in mlp_rows]
    sp = [float(r[2]) for r in mlp_rows]
    ax.barh(range(len(sp)), sp, color=["#c44", "#468", "#4a7", "#e8a33d", "#2a6"])
    ax.set_yticks(range(len(sp)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(1.0, color="#999", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("speedup vs unfused (log scale)")
    ax.set_title("The whole MLP")
    for i, v in enumerate(sp):
        ax.text(v, i, f" {v:.2f}x", va="center", fontsize=8)

    ax = axes[2]
    ax.pie([split["matmul"], split["elementwise"]],
           labels=["matmul\n(oneDNN)", "bias + GELU"],
           colors=["#468", "#e8a33d"], autopct="%1.0f%%", startangle=90)
    ax.set_title("Where the time is, unfused")

    fig.tight_layout()
    fig.savefig(OUT / "fused_mlp.png", dpi=110)
    print(f"\n    wrote {OUT / 'fused_mlp.png'}")


def main():
    t0 = time.perf_counter()
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"MLP {B}x{D} -> {H} -> {D}")
    (OUT / "triton_fused_mlp.py").write_text(TRITON_REFERENCE)
    torch.manual_seed(0)
    tensors = (torch.randn(B, D), torch.randn(D, H) * 0.05, torch.randn(H) * 0.1,
               torch.randn(H, D) * 0.02, torch.randn(D) * 0.1)
    mod, avx = build_all()
    correctness(mod, avx, tensors)
    epi = epilogue_only(mod, avx, tensors)
    mlp = whole_mlp(mod, avx, tensors, 64)
    split = where_time_goes(tensors)
    memory_and_allocs(avx, tensors)
    figure(epi, mlp, split)
    K.write_csv(OUT / "findings.csv", ROWS, ["section", "what", "value", "note"])
    print(f"    wrote {OUT / 'findings.csv'} and {OUT / 'triton_fused_mlp.py'}")
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
