"""Project 34 -- mini FlashAttention: tiled attention with an online softmax.

The guide asks for a Triton kernel. This GPU is sm_61 and Triton needs sm_70,
so it is C++ -- which costs nothing here, because FlashAttention's whole idea is
about *memory*, and a CPU has the same memory hierarchy shape as a GPU: a small
fast level (registers / L1) and a big slow one (DRAM / HBM).

Sections
  1. what eager attention allocates, and how fast that grows
  2. the tiled kernel, and whether it gives the same numbers
  3. peak memory, measured in separate processes
  4. speed against eager and against F.scaled_dot_product_attention
  5. causal masking: blocks that can be skipped entirely
"""

import json
import os
import subprocess
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

NH, DH = 4, 64        # heads, head dimension
BR, BC = 64, 64       # query-block and key-block sizes
FUNCS = ["flash_attention", "flash_attention_vec", "blocks_visited"]
AVX2_FLAGS = ["-DCPU_CAPABILITY_AVX2", "-mavx2", "-mfma"]


CPP = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <cmath>
#include <vector>
#include <algorithm>

// FlashAttention, small and readable. Shapes: q, k, v are [H, T, D].
//
// The eager algorithm is
//     S = q @ k^T * scale        <- an entire T x T matrix, written to memory
//     P = softmax(S, dim=-1)     <- another T x T matrix
//     out = P @ v
// and the two T x T matrices are the whole problem: at T = 4096 they are
// 64 MB each PER HEAD, and they exist only to be thrown away.
//
// The tiled version never has more than BR x BC of them alive at once. To do
// that it must finish the softmax without ever seeing the whole row, which is
// what the running (m, l) pair below is for:
//
//   m = the largest logit seen in this row so far
//   l = the sum of exp(logit - m) over the logits seen so far
//   acc = the sum of exp(logit - m) * v over the logits seen so far
//
// When a new block contains a bigger logit, m changes, and everything computed
// against the old m is too large by exactly exp(m_old - m_new). One multiply
// on l and on acc fixes it. That single rescaling line is the whole trick.
torch::Tensor flash_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              int64_t BR, int64_t BC, bool causal) {
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous());
  TORCH_CHECK(q.scalar_type() == torch::kFloat);
  const int64_t H = q.size(0), T = q.size(1), D = q.size(2);
  auto out = torch::zeros_like(q);
  const float scale = 1.0f / std::sqrt((float)D);

  const int64_t n_qblk = (T + BR - 1) / BR;
  const int64_t total = H * n_qblk;

  at::parallel_for(0, total, 1, [&](int64_t begin, int64_t end) {
    // per-thread scratch: the only memory this kernel needs beyond q,k,v,out
    std::vector<float> s(BR * BC), acc(BR * D), m(BR), l(BR), mnew(BR), lsum(BR);
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t h = idx / n_qblk, qb = idx % n_qblk;
      const int64_t i0 = qb * BR, i1 = std::min(i0 + BR, T);
      const int64_t rows = i1 - i0;
      const float* qh = q.data_ptr<float>() + h * T * D;
      const float* kh = k.data_ptr<float>() + h * T * D;
      const float* vh = v.data_ptr<float>() + h * T * D;
      float* oh = out.data_ptr<float>() + h * T * D;

      std::fill(acc.begin(), acc.end(), 0.f);
      std::fill(l.begin(), l.begin() + rows, 0.f);
      std::fill(m.begin(), m.begin() + rows, -INFINITY);

      const int64_t last_key = causal ? i1 : T;   // causal: no key past this query
      for (int64_t j0 = 0; j0 < last_key; j0 += BC) {
        const int64_t j1 = std::min(j0 + BC, last_key);
        const int64_t cols = j1 - j0;

        // --- the small score tile: BR x BC, never bigger -------------------
        for (int64_t i = 0; i < rows; ++i) {
          const float* qrow = qh + (i0 + i) * D;
          for (int64_t j = 0; j < cols; ++j) {
            const float* krow = kh + (j0 + j) * D;
            float dot = 0.f;
            for (int64_t d = 0; d < D; ++d) dot += qrow[d] * krow[d];
            float val = dot * scale;
            if (causal && (j0 + j) > (i0 + i)) val = -INFINITY;  // inside-block mask
            s[i * BC + j] = val;
          }
        }

        // --- online softmax update ---------------------------------------
        for (int64_t i = 0; i < rows; ++i) {
          float rowmax = -INFINITY;
          for (int64_t j = 0; j < cols; ++j) rowmax = std::max(rowmax, s[i * BC + j]);
          const float mn = std::max(m[i], rowmax);
          float rowsum = 0.f;
          for (int64_t j = 0; j < cols; ++j) {
            const float e = std::isinf(s[i * BC + j]) && s[i * BC + j] < 0
                            ? 0.f : std::exp(s[i * BC + j] - mn);
            s[i * BC + j] = e;
            rowsum += e;
          }
          // the correction factor: everything already accumulated used m[i]
          const float corr = (m[i] == -INFINITY) ? 0.f : std::exp(m[i] - mn);
          l[i] = l[i] * corr + rowsum;
          float* a = &acc[i * D];
          for (int64_t d = 0; d < D; ++d) a[d] *= corr;
          m[i] = mn;

          // --- accumulate this tile's contribution to the output ----------
          for (int64_t j = 0; j < cols; ++j) {
            const float p = s[i * BC + j];
            if (p == 0.f) continue;
            const float* vrow = vh + (j0 + j) * D;
            for (int64_t d = 0; d < D; ++d) a[d] += p * vrow[d];
          }
        }
      }

      for (int64_t i = 0; i < rows; ++i) {
        const float inv = 1.0f / l[i];
        float* orow = oh + (i0 + i) * D;
        for (int64_t d = 0; d < D; ++d) orow[d] = acc[i * D + d] * inv;
      }
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// The same algorithm with the two inner loops vectorized.
//
// Nothing about the tiling or the online softmax changes. The only difference
// is that the q.k dot product and the p*v accumulation move 8 floats per
// instruction through at::vec::Vectorized<float> instead of one. Project 33
// found the same thing about GELU: the algorithm is half the work, and the
// instruction width is the other half.
// ---------------------------------------------------------------------------
using Vec = at::vec::Vectorized<float>;

static inline float dot_vec(const float* a, const float* b, int64_t n) {
  const int64_t W = Vec::size();
  Vec acc(0.f);
  int64_t i = 0;
  for (; i + W <= n; i += W) acc = acc + Vec::loadu(a + i) * Vec::loadu(b + i);
  float buf[32];
  acc.store(buf);
  float s = 0.f;
  for (int64_t j = 0; j < W; ++j) s += buf[j];
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

torch::Tensor flash_attention_vec(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                  int64_t BR, int64_t BC, bool causal) {
  const int64_t H = q.size(0), T = q.size(1), D = q.size(2);
  auto out = torch::zeros_like(q);
  const float scale = 1.0f / std::sqrt((float)D);
  const int64_t n_qblk = (T + BR - 1) / BR;
  const int64_t W = Vec::size();

  at::parallel_for(0, H * n_qblk, 1, [&](int64_t begin, int64_t end) {
    std::vector<float> s(BR * BC), acc(BR * D), m(BR), l(BR);
    for (int64_t idx = begin; idx < end; ++idx) {
      const int64_t h = idx / n_qblk, qb = idx % n_qblk;
      const int64_t i0 = qb * BR, i1 = std::min(i0 + BR, T), rows = i1 - i0;
      const float* qh = q.data_ptr<float>() + h * T * D;
      const float* kh = k.data_ptr<float>() + h * T * D;
      const float* vh = v.data_ptr<float>() + h * T * D;
      float* oh = out.data_ptr<float>() + h * T * D;

      std::fill(acc.begin(), acc.end(), 0.f);
      std::fill(l.begin(), l.begin() + rows, 0.f);
      std::fill(m.begin(), m.begin() + rows, -INFINITY);

      const int64_t last_key = causal ? i1 : T;
      for (int64_t j0 = 0; j0 < last_key; j0 += BC) {
        const int64_t j1 = std::min(j0 + BC, last_key), cols = j1 - j0;
        for (int64_t i = 0; i < rows; ++i) {
          const float* qrow = qh + (i0 + i) * D;
          for (int64_t j = 0; j < cols; ++j) {
            float val = dot_vec(qrow, kh + (j0 + j) * D, D) * scale;
            if (causal && (j0 + j) > (i0 + i)) val = -INFINITY;
            s[i * BC + j] = val;
          }
        }
        for (int64_t i = 0; i < rows; ++i) {
          float rowmax = -INFINITY;
          for (int64_t j = 0; j < cols; ++j) rowmax = std::max(rowmax, s[i * BC + j]);
          const float mn = std::max(m[i], rowmax);
          float rowsum = 0.f;
          for (int64_t j = 0; j < cols; ++j) {
            const float e = (std::isinf(s[i * BC + j]) && s[i * BC + j] < 0)
                            ? 0.f : std::exp(s[i * BC + j] - mn);
            s[i * BC + j] = e;
            rowsum += e;
          }
          const float corr = (m[i] == -INFINITY) ? 0.f : std::exp(m[i] - mn);
          l[i] = l[i] * corr + rowsum;
          float* a = &acc[i * D];
          const Vec vcorr(corr);
          int64_t d = 0;
          for (; d + W <= D; d += W) (Vec::loadu(a + d) * vcorr).store(a + d);
          for (; d < D; ++d) a[d] *= corr;
          m[i] = mn;

          for (int64_t j = 0; j < cols; ++j) {
            const float p = s[i * BC + j];
            if (p == 0.f) continue;
            const float* vrow = vh + (j0 + j) * D;
            const Vec vp(p);
            int64_t dd = 0;
            for (; dd + W <= D; dd += W)
              (Vec::loadu(a + dd) + vp * Vec::loadu(vrow + dd)).store(a + dd);
            for (; dd < D; ++dd) a[dd] += p * vrow[dd];
          }
        }
      }
      for (int64_t i = 0; i < rows; ++i) {
        const float inv = 1.0f / l[i];
        float* orow = oh + (i0 + i) * D;
        const Vec vinv(inv);
        int64_t d = 0;
        for (; d + W <= D; d += W) (Vec::loadu(&acc[i * D] + d) * vinv).store(orow + d);
        for (; d < D; ++d) orow[d] = acc[i * D + d] * inv;
      }
    }
  });
  return out;
}

// How many BR x BC blocks the causal version actually visits, so the README
// can quote a number instead of a guess.
int64_t blocks_visited(int64_t T, int64_t BR, int64_t BC, bool causal) {
  int64_t n = 0;
  for (int64_t i0 = 0; i0 < T; i0 += BR) {
    const int64_t last = causal ? std::min(i0 + BR, T) : T;
    for (int64_t j0 = 0; j0 < last; j0 += BC) ++n;
  }
  return n;
}
"""


TRITON_REFERENCE = '''\
# FlashAttention in Triton (needs sm_70+, this GPU is sm_61). The loop
# structure is identical to flash_attention in run.py -- compare them.

import triton
import triton.language as tl

@triton.jit
def flash_kernel(q_ptr, k_ptr, v_ptr, o_ptr, scale, T, D: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr):
    start = tl.program_id(0) * BR            # this program owns BR query rows
    offs_m = start + tl.arange(0, BR)
    offs_d = tl.arange(0, D)
    q = tl.load(q_ptr + offs_m[:, None] * D + offs_d[None, :])   # stays in SRAM

    m_i = tl.full((BR,), -float("inf"), tl.float32)   # running max
    l_i = tl.zeros((BR,), tl.float32)                 # running denominator
    acc = tl.zeros((BR, D), tl.float32)               # running output

    for j in range(0, T, BC):
        offs_n = j + tl.arange(0, BC)
        k = tl.load(k_ptr + offs_n[:, None] * D + offs_d[None, :])
        v = tl.load(v_ptr + offs_n[:, None] * D + offs_d[None, :])

        s = tl.dot(q, tl.trans(k)) * scale            # the BR x BC tile
        m_new = tl.maximum(m_i, tl.max(s, 1))
        corr = tl.exp(m_i - m_new)                    # <- the rescaling
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * corr + tl.sum(p, 1)
        acc = acc * corr[:, None] + tl.dot(p, v)
        m_i = m_new

    tl.store(o_ptr + offs_m[:, None] * D + offs_d[None, :], acc / l_i[:, None])

# The GPU version has one advantage the CPU version cannot have: q, k, v tiles
# are moved into SRAM by an explicit tl.load, and acc lives in registers for the
# whole loop. On the CPU we rely on the cache to keep the same values close.
# The algorithm -- and the memory it saves -- is the same.
'''


# ---------------------------------------------------------------------------
# a child process that measures peak RSS for one implementation and one T
# ---------------------------------------------------------------------------
CHILD = r'''
import os, sys, json, resource, importlib.util
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, sys.argv[5])       # project 30, for kernels_lib
import torch, torch.nn.functional as F
import kernels_lib as K

# `import run` would be ambiguous: several projects in this guide have a
# run.py, and kernels_lib itself puts project 24's directory on sys.path to
# reuse perf_lib. Loading THIS project's file by its full path removes the
# guesswork -- an `import` that depends on sys.path order is a bug waiting.
_spec = importlib.util.spec_from_file_location(
    "p34_run", os.path.join(sys.argv[4], "run.py"))
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

impl, T = sys.argv[1], int(sys.argv[2])
H, D = P.NH, P.DH
torch.manual_seed(0)
q = torch.randn(H, T, D); k = torch.randn(H, T, D); v = torch.randn(H, T, D)
base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

if impl == "eager":
    P.eager_attention(q, k, v)
elif impl == "sdpa":
    F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
else:
    mod, _ = K.build("p34_flash", P.CPP, functions=P.FUNCS, extra_cflags=P.AVX2_FLAGS)
    mod.flash_attention_vec(q, k, v, P.BR, P.BC, False)

peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(json.dumps({"impl": impl, "T": T, "base_kb": base, "peak_kb": peak}))
'''


def eager_attention(q, k, v, causal=False):
    """Textbook attention. The T x T matrix is unavoidable here."""
    scale = 1.0 / (q.size(-1) ** 0.5)
    s = (q @ k.transpose(-1, -2)) * scale
    if causal:
        t = q.size(-2)
        mask = torch.ones(t, t, dtype=torch.bool).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def build_all():
    K.banner("[1] compiling")
    mod, s = K.build("p34_flash", CPP, functions=FUNCS, extra_cflags=AVX2_FLAGS)
    print(f"    {s:.1f} s (with {' '.join(AVX2_FLAGS)} -- project 33 explains why)")
    ROWS.append(["compile", "p34_flash", f"{s:.1f}", "s"])
    return mod


def the_problem():
    K.banner("[2] what eager attention has to allocate")
    print(f"    {NH} heads, head dim {DH}, float32\n")
    print(f"    {'T':>6}{'q,k,v (MB)':>13}{'scores T x T (MB)':>20}{'ratio':>9}")
    for t in (256, 512, 1024, 2048, 4096, 8192):
        qkv = 3 * NH * t * DH * 4 / 1e6
        sc = NH * t * t * 4 / 1e6
        print(f"    {t:>6}{qkv:>13.1f}{sc:>20.1f}{sc / qkv:>8.1f}x")
        ROWS.append(["problem", f"T={t}", f"{sc:.1f} MB of scores", f"{sc / qkv:.1f}x q,k,v"])
    print("\n    q, k and v grow with T. The score matrix grows with T squared.")
    print("    That is the entire reason FlashAttention exists.")


def correctness(mod):
    K.banner("[3] does the tiled version give the same answer?")
    torch.manual_seed(0)
    for t in (128, 256, 500):        # 500 is not a multiple of the block size
        q, k, v = (torch.randn(NH, t, DH) for _ in range(3))
        ref = eager_attention(q, k, v)
        got = mod.flash_attention(q, k, v, BR, BC, False)
        vec = mod.flash_attention_vec(q, k, v, BR, BC, False)
        sdpa = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0),
                                              v.unsqueeze(0)).squeeze(0)
        print(f"    T={t:<5} flash vs eager: {K.rel_err(got, ref):.2e}   "
              f"vectorized vs eager: {K.rel_err(vec, ref):.2e}   "
              f"eager vs SDPA: {K.rel_err(ref, sdpa):.2e}")
        ROWS.append(["correctness", f"T={t}", f"{K.rel_err(got, ref):.2e}", "flash vs eager"])

    # causal
    t = 256
    q, k, v = (torch.randn(NH, t, DH) for _ in range(3))
    ref_c = eager_attention(q, k, v, causal=True)
    got_c = mod.flash_attention_vec(q, k, v, BR, BC, True)
    print(f"    causal T={t}: {K.rel_err(got_c, ref_c):.2e}")
    ROWS.append(["correctness", "causal T=256", f"{K.rel_err(got_c, ref_c):.2e}", "flash vs eager"])

    # extreme logits: the running max keeps exp() in range
    q, k, v = (torch.randn(NH, 256, DH) * 8 for _ in range(3))
    ref = eager_attention(q, k, v)
    got = mod.flash_attention_vec(q, k, v, BR, BC, False)
    print(f"\n    with logits ~{float((q @ k.transpose(-1, -2)).abs().max()) / DH ** 0.5:.0f} "
          f"(far past exp's float32 limit of 88):")
    print(f"      finite output: {bool(torch.isfinite(got).all())}, "
          f"relative error {K.rel_err(got, ref):.2e}")
    ROWS.append(["stability", "large logits", f"{K.rel_err(got, ref):.2e}",
                 f"finite={bool(torch.isfinite(got).all())}"])


def memory(here):
    K.banner("[4] peak memory, measured (one child process per measurement)")
    child_path = OUT / "_mem_child.py"
    child_path.write_text(CHILD)
    lib = os.path.join(os.path.dirname(here), "30-cpp-extension-for-elementwise-add")
    print(f"    {'T':>6}{'eager MB':>11}{'flash MB':>11}{'SDPA MB':>10}"
          f"{'MB saved':>13}{'predicted':>13}")
    rows = []
    for t in (512, 1024, 2048, 4096):
        got = {}
        for impl in ("eager", "flash", "sdpa"):
            p = subprocess.run([sys.executable, str(child_path), impl, str(t), "x",
                                here, lib],
                               capture_output=True, text=True)
            line = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
            if not line:
                got[impl] = float("nan")
                continue
            d = json.loads(line[-1])
            got[impl] = (d["peak_kb"] - d["base_kb"]) / 1024
        save = got["eager"] - got["flash"]
        predicted = NH * t * t * 4 / 1e6
        print(f"    {t:>6}{got['eager']:>11.0f}{got['flash']:>11.0f}"
              f"{got['sdpa']:>10.0f}{save:>13.0f}{predicted:>13.0f}")
        rows.append([t, f"{got['eager']:.0f}", f"{got['flash']:.0f}",
                     f"{got['sdpa']:.0f}", f"{save:.0f}"])
        ROWS.append(["memory", f"T={t}", f"eager {got['eager']:.0f} MB",
                     f"flash {got['flash']:.0f} MB, saves {save:.0f} MB"])
    print("\n    Measured as peak resident memory above the q,k,v baseline, in a")
    print("    fresh process each time -- PyTorch's CPU allocator reuses freed")
    print("    blocks, so measuring twice in one process would hide the second.")
    print("    The last column is ONE T x T float32 score matrix per head. The")
    print("    measurement comes out about twice that, and the factor of two is")
    print("    real: `softmax(s)` does not overwrite s, it allocates a second")
    print("    matrix of the same size. Two T x T matrices are alive at once.")
    print("    Flash and SDPA read 0 MB because what they add -- a few tiles per")
    print("    thread -- is below the 1 MB resolution of this measurement.")
    return rows


def speed(mod):
    K.banner("[5] speed")
    print(f"    {'T':>6}{'eager ms':>11}{'scalar ms':>11}{'vector ms':>11}"
          f"{'SDPA ms':>10}{'vector/eager':>14}{'GFLOP':>8}")
    rows = []
    for t in (256, 512, 1024, 2048):
        torch.manual_seed(0)
        q, k, v = (torch.randn(NH, t, DH) for _ in range(3))
        q4, k4, v4 = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        r = K.interleaved({
            "eager": lambda: eager_attention(q, k, v),
            "flash": lambda: mod.flash_attention(q, k, v, BR, BC, False),
            "vec": lambda: mod.flash_attention_vec(q, k, v, BR, BC, False),
            "sdpa": lambda: F.scaled_dot_product_attention(q4, k4, v4),
        }, rounds=4, warmup=1)
        flops = 4.0 * NH * t * t * DH
        print(f"    {t:>6}{r['eager'][0]:>11.1f}{r['flash'][0]:>11.1f}"
              f"{r['vec'][0]:>11.1f}{r['sdpa'][0]:>10.1f}"
              f"{r['eager'][0] / r['vec'][0]:>13.2f}x{flops / 1e9:>8.2f}")
        rows.append([t, f"{r['eager'][0]:.1f}", f"{r['vec'][0]:.1f}",
                     f"{r['sdpa'][0]:.1f}", f"{r['eager'][0] / r['vec'][0]:.2f}",
                     f"{r['flash'][0]:.1f}"])
        ROWS.append(["speed", f"T={t}", f"vectorized {r['vec'][0]:.1f} ms",
                     f"{r['eager'][0] / r['vec'][0]:.2f}x vs eager, "
                     f"scalar was {r['flash'][0]:.1f} ms"])
    return rows


def block_sweep(mod):
    K.banner("[6] block size")
    t = 1024
    torch.manual_seed(0)
    q, k, v = (torch.randn(NH, t, DH) for _ in range(3))
    sizes = (16, 32, 64, 128, 256)
    res = K.interleaved(
        {b: (lambda b=b: mod.flash_attention_vec(q, k, v, b, b, False)) for b in sizes},
        rounds=4, warmup=1)
    print(f"    {'BR=BC':>7}{'score tile (KB)':>18}{'ms':>9}")
    rows = []
    for b in sizes:
        print(f"    {b:>7}{b * b * 4 / 1024:>18.1f}{res[b][0]:>9.1f}")
        rows.append([b, f"{res[b][0]:.1f}"])
        ROWS.append(["block sweep", f"BR=BC={b}", f"{res[b][0]:.2f} ms", "T=1024"])
    return rows


def causal_saving(mod):
    K.banner("[7] causal masking: work that the tiled version can skip")
    print("    Eager builds every score, then sets half of them to -inf.")
    print("    The tiled loop simply does not visit a block that is entirely")
    print("    in the future.\n")
    print(f"    {'T':>6}{'blocks full':>13}{'blocks causal':>15}{'skipped':>10}"
          f"{'causal ms':>11}{'full ms':>10}")
    rows = []
    for t in (512, 1024, 2048):
        torch.manual_seed(0)
        q, k, v = (torch.randn(NH, t, DH) for _ in range(3))
        nf = mod.blocks_visited(t, BR, BC, False)
        nc = mod.blocks_visited(t, BR, BC, True)
        r = K.interleaved({
            "causal": lambda: mod.flash_attention_vec(q, k, v, BR, BC, True),
            "full": lambda: mod.flash_attention_vec(q, k, v, BR, BC, False),
        }, rounds=4, warmup=1)
        print(f"    {t:>6}{nf:>13}{nc:>15}{(1 - nc / nf) * 100:>9.0f}%"
              f"{r['causal'][0]:>11.1f}{r['full'][0]:>10.1f}")
        rows.append([t, nf, nc, f"{r['causal'][0]:.1f}", f"{r['full'][0]:.1f}"])
        ROWS.append(["causal", f"T={t}", f"{(1 - nc / nf) * 100:.0f}% blocks skipped",
                     f"{r['full'][0] / r['causal'][0]:.2f}x faster"])
    return rows


def figure(mem_rows, speed_rows, blk_rows, causal_rows):
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.3))

    ax = axes[0]
    ts = [r[0] for r in mem_rows]
    ax.plot(ts, [float(r[1]) for r in mem_rows], "o-", color="#c44", label="eager")
    ax.plot(ts, [float(r[2]) for r in mem_rows], "s-", color="#4a7", label="flash (C++)")
    ax.plot(ts, [float(r[3]) for r in mem_rows], "^-", color="#468", label="SDPA")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("peak memory above q,k,v (MB)")
    ax.set_title("Memory: quadratic vs flat")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ts = [r[0] for r in speed_rows]
    ax.loglog(ts, [float(r[1]) for r in speed_rows], "o-", color="#c44", label="eager")
    ax.loglog(ts, [float(r[2]) for r in speed_rows], "s-", color="#4a7", label="flash (vectorized)")
    ax.loglog(ts, [float(r[5]) for r in speed_rows], "d:", color="#e8a33d", label="flash (scalar)")
    ax.loglog(ts, [float(r[3]) for r in speed_rows], "^-", color="#468", label="SDPA")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("ms")
    ax.set_title("Speed (log-log)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    b = [r[0] for r in blk_rows]
    ax.semilogx([int(x) for x in b], [float(r[1]) for r in blk_rows], "o-",
                base=2, color="#a4c")
    ax.set_xlabel("block size BR = BC")
    ax.set_ylabel("ms, T=1024")
    ax.set_title("Block size trade-off")
    ax.grid(alpha=0.3)

    ax = axes[3]
    ts = [r[0] for r in causal_rows]
    w = 0.38
    xs = range(len(ts))
    ax.bar([x - w / 2 for x in xs], [r[1] for r in causal_rows], w,
           label="all blocks", color="#c44")
    ax.bar([x + w / 2 for x in xs], [r[2] for r in causal_rows], w,
           label="causal only", color="#4a7")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"T={t}" for t in ts])
    ax.set_ylabel("score blocks computed")
    ax.set_title("Causal masking skips blocks")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT / "flashattention.png", dpi=110)
    print(f"\n    wrote {OUT / 'flashattention.png'}")


def main():
    t0 = time.perf_counter()
    here = os.path.dirname(os.path.abspath(__file__))
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()} | "
          f"heads {NH}, head dim {DH}, blocks {BR}x{BC}")
    (OUT / "triton_flashattention.py").write_text(TRITON_REFERENCE)
    mod = build_all()
    the_problem()
    correctness(mod)
    mem = memory(here)
    sp = speed(mod)
    blk = block_sweep(mod)
    ca = causal_saving(mod)
    figure(mem, sp, blk, ca)
    K.write_csv(OUT / "findings.csv", ROWS, ["section", "what", "value", "note"])
    print(f"    wrote {OUT / 'findings.csv'} and {OUT / 'triton_flashattention.py'}")
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
