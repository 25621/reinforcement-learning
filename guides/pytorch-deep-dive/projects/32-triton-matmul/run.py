"""Project 32 -- a tiled matmul, and how far a hand-written one gets.

The guide asks for this in Triton against cuBLAS. This machine's GPU is sm_61
and Triton needs sm_70, so the kernel is C++ against oneDNN/MKL instead. Same
question, same tiling, different silicon: how close does a hand-written kernel
get to the vendor's?

Sections
  1-2. compile and check four versions against torch.mm
  3.   the loop order that costs nothing and buys 18x
  4.   the traffic model: why tiling should help
  5.   a tile-size sweep -- the manual version of triton.autotune
  6.   when tiling actually helps (and when it makes things worse)
  7.   threads, and the limit your own decomposition puts on them
  8.   the vendor gap, and what is inside it
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

import kernels_lib as K  # noqa: E402

OUT = K.outputs_dir(__file__)
ROWS = []
N = 512                      # 512x512x512: 268 MFLOP, small enough for the naive loop


CPP = r"""
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>

// C = A @ B, all row-major, all float32, all square-ish.
// A is MxK, B is KxN, C is MxN.

// ---- 1. the textbook order: i, j, k ---------------------------------------
// The inner loop walks DOWN a column of B. Consecutive iterations touch
// addresses N*4 bytes apart, so every single one is a new cache line: the CPU
// fetches 64 bytes to use 4 of them.
torch::Tensor mm_ijk(torch::Tensor A, torch::Tensor B) {
  const int64_t M = A.size(0), Kd = A.size(1), Nn = B.size(1);
  auto C = torch::zeros({M, Nn}, A.options());
  const float* a = A.data_ptr<float>();
  const float* b = B.data_ptr<float>();
  float* c = C.data_ptr<float>();
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < Nn; ++j) {
      float s = 0.f;
      for (int64_t k = 0; k < Kd; ++k) s += a[i * Kd + k] * b[k * Nn + j];
      c[i * Nn + j] = s;
    }
  return C;
}

// ---- 2. swap two lines: i, k, j -------------------------------------------
// Now the inner loop walks ALONG a row of B and a row of C, both contiguous.
// Same arithmetic, same result, but every cache line that arrives is fully
// used -- and gcc can turn the loop into AVX2 instructions, 8 floats at a time.
torch::Tensor mm_ikj(torch::Tensor A, torch::Tensor B) {
  const int64_t M = A.size(0), Kd = A.size(1), Nn = B.size(1);
  auto C = torch::zeros({M, Nn}, A.options());
  const float* a = A.data_ptr<float>();
  const float* b = B.data_ptr<float>();
  float* c = C.data_ptr<float>();
  for (int64_t i = 0; i < M; ++i)
    for (int64_t k = 0; k < Kd; ++k) {
      const float aik = a[i * Kd + k];
      const float* brow = b + k * Nn;
      float* crow = c + i * Nn;
      for (int64_t j = 0; j < Nn; ++j) crow[j] += aik * brow[j];
    }
  return C;
}

// ---- 3. tiled ------------------------------------------------------------
// Cut the output into BLOCK x BLOCK tiles and the K dimension into BLOCK-long
// strips. For one output tile we load one strip of A and one strip of B, and
// use each loaded value BLOCK times before dropping it.
//
// This is exactly what a Triton matmul does with tl.load into SRAM; the only
// difference is that here "fast memory" is the L1/L2 cache and the hardware
// decides what stays, instead of us writing tl.load.
torch::Tensor mm_tiled(torch::Tensor A, torch::Tensor B, int64_t BLOCK) {
  const int64_t M = A.size(0), Kd = A.size(1), Nn = B.size(1);
  auto C = torch::zeros({M, Nn}, A.options());
  const float* a = A.data_ptr<float>();
  const float* b = B.data_ptr<float>();
  float* c = C.data_ptr<float>();
  for (int64_t i0 = 0; i0 < M; i0 += BLOCK)
    for (int64_t k0 = 0; k0 < Kd; k0 += BLOCK)
      for (int64_t j0 = 0; j0 < Nn; j0 += BLOCK) {
        const int64_t i1 = std::min(i0 + BLOCK, M);
        const int64_t k1 = std::min(k0 + BLOCK, Kd);
        const int64_t j1 = std::min(j0 + BLOCK, Nn);
        for (int64_t i = i0; i < i1; ++i)
          for (int64_t k = k0; k < k1; ++k) {
            const float aik = a[i * Kd + k];
            const float* brow = b + k * Nn;
            float* crow = c + i * Nn;
            for (int64_t j = j0; j < j1; ++j) crow[j] += aik * brow[j];
          }
      }
  return C;
}

// ---- 4. tiled + threads --------------------------------------------------
// One thread per band of output rows. Output tiles never overlap, so no thread
// ever writes a value another thread reads: no locks, no atomics. This is the
// same reason a GPU can run thousands of matmul programs at once.
torch::Tensor mm_tiled_par(torch::Tensor A, torch::Tensor B, int64_t BLOCK) {
  const int64_t M = A.size(0), Kd = A.size(1), Nn = B.size(1);
  auto C = torch::zeros({M, Nn}, A.options());
  const float* a = A.data_ptr<float>();
  const float* b = B.data_ptr<float>();
  float* c = C.data_ptr<float>();
  const int64_t nblk = (M + BLOCK - 1) / BLOCK;
  at::parallel_for(0, nblk, 1, [&](int64_t blk0, int64_t blk1) {
    for (int64_t blk = blk0; blk < blk1; ++blk) {
      const int64_t i0 = blk * BLOCK, i1 = std::min(i0 + BLOCK, M);
      for (int64_t k0 = 0; k0 < Kd; k0 += BLOCK)
        for (int64_t j0 = 0; j0 < Nn; j0 += BLOCK) {
          const int64_t k1 = std::min(k0 + BLOCK, Kd);
          const int64_t j1 = std::min(j0 + BLOCK, Nn);
          for (int64_t i = i0; i < i1; ++i)
            for (int64_t k = k0; k < k1; ++k) {
              const float aik = a[i * Kd + k];
              const float* brow = b + k * Nn;
              float* crow = c + i * Nn;
              for (int64_t j = j0; j < j1; ++j) crow[j] += aik * brow[j];
            }
        }
    }
  });
  return C;
}
"""


TRITON_REFERENCE = '''\
# The same tiled matmul in Triton (needs sm_70+, this GPU is sm_61).
# Every line has a twin in mm_tiled above.

import triton
import triton.language as tl

@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  sam, sak, sbk, sbn, scm, scn,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_m = tl.program_id(0)          # which row-block of C this program owns
    pid_n = tl.program_id(1)          # which column-block
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a_ptrs = a_ptr + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = b_ptr + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BM, BN), dtype=tl.float32)     # the C tile, in registers
    for k in range(0, K, BK):                      # the k0 loop of mm_tiled
        a = tl.load(a_ptrs)                        # <- an explicit copy into SRAM
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)                        # tensor-core matmul on the tile
        a_ptrs += BK * sak
        b_ptrs += BK * sbk

    c_ptrs = c_ptr + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# Two differences worth naming:
#
# 1. tl.load is an explicit copy into on-chip memory. On the CPU there is no
#    such instruction -- the cache loads whatever you touch. Tiling on a CPU is
#    therefore about the ORDER you touch memory in; on a GPU it is about what
#    you copy. Same goal: read each value from slow memory once, use it many
#    times.
#
# 2. `acc` lives in registers for the whole k loop, so the C tile is written to
#    memory once at the end. mm_tiled writes C back on every k-block because C
#    is a plain array in memory -- one of the reasons it is slower.
'''


def build_all():
    K.banner("[1] compiling")
    mod, s = K.build("p32_matmul", CPP,
                     functions=["mm_ijk", "mm_ikj", "mm_tiled", "mm_tiled_par"])
    print(f"    {s:.1f} s")
    ROWS.append(["compile", "p32_matmul", f"{s:.1f}", "s"])
    return mod


def correctness(mod):
    K.banner("[2] do they all compute the same matrix?")
    a = torch.randn(129, 97)     # deliberately not a multiple of any tile size
    b = torch.randn(97, 131)
    ref = a @ b
    for lb, got in {
        "mm_ijk": mod.mm_ijk(a, b),
        "mm_ikj": mod.mm_ikj(a, b),
        "mm_tiled(64)": mod.mm_tiled(a, b, 64),
        "mm_tiled_par(64)": mod.mm_tiled_par(a, b, 64),
    }.items():
        err = K.rel_err(got, ref)
        print(f"    {lb:<20} relative error vs torch.mm: {err:.2e}")
        ROWS.append(["correctness", lb, f"{err:.2e}", "129x97 @ 97x131"])
    print("\n    Not bit-identical, and that is expected: every version adds the")
    print("    same K products in a different ORDER, and float addition is not")
    print("    associative -- (a+b)+c can differ from a+(b+c) in the last bits.")


def loop_order(mod):
    K.banner(f"[3] loop order and tiling, {N}x{N}x{N}")
    a = torch.randn(N, N)
    b = torch.randn(N, N)
    flops = 2.0 * N * N * N
    fns = {
        "mm_ijk (textbook)": lambda: mod.mm_ijk(a, b),
        "mm_ikj (two lines swapped)": lambda: mod.mm_ikj(a, b),
        "mm_tiled (BLOCK=64)": lambda: mod.mm_tiled(a, b, 64),
        "mm_tiled_par (6 threads)": lambda: mod.mm_tiled_par(a, b, 64),
        "torch.mm (oneDNN)": lambda: a @ b,
    }
    res = K.interleaved(fns, rounds=5, warmup=1)
    best = res["torch.mm (oneDNN)"][0]
    print(f"    {'kernel':<30}{'ms':>10}{'GFLOP/s':>10}{'vs torch.mm':>13}")
    rows = []
    for lb, (ms, sp) in res.items():
        g = K.gflops(flops, ms)
        print(f"    {lb:<30}{ms:>10.1f}{g:>10.1f}{best / ms * 100:>12.1f}%")
        rows.append([lb, f"{ms:.1f}", f"{g:.1f}", f"{best / ms * 100:.1f}"])
        ROWS.append(["speed", lb, f"{ms:.2f} ms", f"{g:.1f} GFLOP/s"])
    return rows


def traffic_model():
    K.banner("[4] why tiling helps: how often is each value re-read?")
    print("    A is NxN, B is NxN, C is NxN, all float32, N =", N)
    print("    The CPU can only hold a few MB close by, so 'read' below means")
    print("    'read again from far away'.\n")
    per = N * N * 4 / 1e6
    naive = (N * N * N + N * N * N) * 4 / 1e6      # A and B both read N times
    tiled = {}
    for blk in (16, 32, 64, 128):
        # each output tile reads (N/BLOCK) strips of A and of B
        t = 2 * (N / blk) * N * N * 4 / 1e6
        tiled[blk] = t
    print(f"    {'scheme':<22}{'DRAM reads (MB)':>18}{'vs perfect':>13}")
    perfect = 2 * per
    print(f"    {'untiled (ikj)':<22}{naive:>18.0f}{naive / perfect:>12.0f}x")
    for blk, t in tiled.items():
        print(f"    {'tiled BLOCK=' + str(blk):<22}{t:>18.0f}{t / perfect:>12.0f}x")
    print(f"    {'perfect (read once)':<22}{perfect:>18.0f}{1:>12.0f}x")
    print("\n    A BLOCK x BLOCK tile of C reuses each loaded A and B value BLOCK")
    print("    times, so traffic falls by a factor of BLOCK -- until the three")
    print("    tiles stop fitting in cache, which is what section 5 finds.")
    for blk, t in tiled.items():
        ROWS.append(["traffic model", f"tiled BLOCK={blk}", f"{t:.0f}", "MB read"])
    ROWS.append(["traffic model", "untiled", f"{naive:.0f}", "MB read"])


def tile_sweep(mod):
    K.banner("[5] the tile-size sweep (what triton.autotune does for you)")
    a = torch.randn(N, N)
    b = torch.randn(N, N)
    flops = 2.0 * N * N * N
    blocks = (8, 16, 32, 64, 128, 256)
    res = K.interleaved(
        {blk: (lambda blk=blk: mod.mm_tiled_par(a, b, blk)) for blk in blocks},
        rounds=6, warmup=1,
    )
    print(f"    {'BLOCK':>7}{'3 tiles (KB)':>15}{'ms':>10}{'GFLOP/s':>10}")
    rows = []
    for blk in blocks:
        kb = 3 * blk * blk * 4 / 1024
        ms = res[blk][0]
        print(f"    {blk:>7}{kb:>15.0f}{ms:>10.1f}{K.gflops(flops, ms):>10.1f}")
        rows.append([blk, f"{kb:.0f}", f"{ms:.1f}", f"{K.gflops(flops, ms):.1f}"])
        ROWS.append(["tile sweep", f"BLOCK={blk}", f"{ms:.2f} ms",
                     f"{K.gflops(flops, ms):.1f} GFLOP/s"])
    best = min(rows, key=lambda r: float(r[2]))
    print(f"\n    best BLOCK here: {best[0]} ({best[3]} GFLOP/s). L1d is 32 KB per")
    print("    core, L2 is 256 KB per core -- the winner is the largest tile whose")
    print("    working set still fits comfortably in L2.")
    ROWS.append(["tile sweep", "best BLOCK", str(best[0]), f"{best[3]} GFLOP/s"])
    return rows


def tiling_crossover(mod, block):
    K.banner("[6] when does tiling actually beat plain ikj?")
    print("    Both single-threaded, so this is a pure memory-layout question.")
    print("    B is NxN floats; while it still fits in the 12 MB L3 cache, the")
    print("    untiled loop is already re-reading from cache, and tiling only")
    print("    adds loop overhead.\n")
    torch.set_num_threads(1)
    print(f"    {'N':>6}{'B size (MB)':>13}{'ikj ms':>10}{'tiled ms':>11}{'tiled wins by':>15}")
    rows = []
    for n in (256, 512, 1024, 2048):
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        r = K.interleaved({"ikj": lambda: mod.mm_ikj(a, b),
                           "tiled": lambda: mod.mm_tiled(a, b, block)},
                          rounds=3, warmup=1)
        gain = r["ikj"][0] / r["tiled"][0]
        print(f"    {n:>6}{n * n * 4 / 1e6:>13.1f}{r['ikj'][0]:>10.1f}"
              f"{r['tiled'][0]:>11.1f}{gain:>14.2f}x")
        rows.append([n, f"{n * n * 4 / 1e6:.1f}", f"{r['ikj'][0]:.1f}",
                     f"{r['tiled'][0]:.1f}", f"{gain:.2f}"])
        ROWS.append(["tiling crossover", f"N={n}", f"{gain:.2f}x",
                     f"B is {n * n * 4 / 1e6:.1f} MB"])
    torch.set_num_threads(K.THREADS)
    return rows


def thread_sweep(mod, block):
    K.banner("[7] threads")
    a = torch.randn(N, N)
    b = torch.randn(N, N)
    flops = 2.0 * N * N * N
    nblk = -(-N // block)
    print(f"    mm_tiled_par splits the output into row-bands of {block} rows,")
    print(f"    so at N={N} there are exactly {nblk} independent pieces of work --")
    print(f"    and a thread count that does not divide {nblk} leaves someone idle")
    print("    while the rest wait. That granularity, plus 6 threads competing")
    print("    with the operating system on a 6-core chip, is why the last step")
    print("    is not free.\n")
    ROWS.append(["threads", "row bands available", str(nblk), f"BLOCK={block}, N={N}"])
    # interleaved, with each candidate setting its own thread count -- see
    # project 31 for why timing them in separate blocks misleads
    def run_with(t):
        torch.set_num_threads(t)
        return mod.mm_tiled_par(a, b, block)

    counts = (1, 2, 4, 6)
    res = K.interleaved({t: (lambda t=t: run_with(t)) for t in counts},
                        rounds=5, warmup=1)
    torch.set_num_threads(K.THREADS)
    print(f"    {'threads':>8}{'ms':>10}{'GFLOP/s':>10}{'speedup':>10}")
    rows = []
    one = res[1][0]
    for t in counts:
        ms = res[t][0]
        print(f"    {t:>8}{ms:>10.1f}{K.gflops(flops, ms):>10.1f}{one / ms:>9.2f}x")
        rows.append([t, f"{ms:.1f}", f"{one / ms:.2f}"])
        ROWS.append(["threads", f"{t} threads", f"{ms:.2f} ms", f"{one / ms:.2f}x"])
    return rows


def size_scaling(mod, block):
    K.banner("[8] does the gap close on bigger matrices?")
    print(f"    {'N':>6}{'GFLOP':>9}{'mine ms':>10}{'torch ms':>10}"
          f"{'mine GF/s':>11}{'torch GF/s':>12}{'% of torch':>12}")
    rows = []
    for n in (128, 256, 512, 1024):
        a = torch.randn(n, n)
        b = torch.randn(n, n)
        flops = 2.0 * n * n * n
        r = K.interleaved({"mine": lambda: mod.mm_tiled_par(a, b, block),
                           "torch": lambda: a @ b}, rounds=4, warmup=1)
        pct = r["torch"][0] / r["mine"][0] * 100
        print(f"    {n:>6}{flops / 1e9:>9.2f}{r['mine'][0]:>10.1f}{r['torch'][0]:>10.1f}"
              f"{K.gflops(flops, r['mine'][0]):>11.1f}"
              f"{K.gflops(flops, r['torch'][0]):>12.1f}{pct:>11.1f}%")
        rows.append([n, f"{K.gflops(flops, r['mine'][0]):.1f}",
                     f"{K.gflops(flops, r['torch'][0]):.1f}", f"{pct:.1f}"])
        ROWS.append(["size scaling", f"N={n}", f"{pct:.1f}% of torch.mm",
                     f"{K.gflops(flops, r['mine'][0]):.1f} GFLOP/s"])
    # 6 cores, 3.7 GHz, two AVX2 FMA units per core, 8 floats per vector,
    # 2 FLOPs per FMA (one multiply + one add) = 32 FLOPs per cycle per core.
    peak = 3.7e9 * 6 * 2 * 8 * 2 / 1e9
    print(f"\n    this CPU's theoretical peak: ~{peak:.0f} GFLOP/s "
          "(6 cores x 2 AVX2 FMA units x 8 floats x 2 FLOP x 3.7 GHz)")
    torch_best = max(float(r[2]) for r in rows)
    mine_best = max(float(r[1]) for r in rows)
    print(f"    torch.mm reaches {torch_best / peak * 100:.0f}% of it, "
          f"this kernel {mine_best / peak * 100:.0f}%.")
    ROWS.append(["reference", "theoretical peak", f"{peak:.0f}", "GFLOP/s"])
    return rows


def figure(order_rows, tile_rows, thread_rows, size_rows, cross_rows):
    fig, axes = plt.subplots(1, 5, figsize=(23, 4.2))

    ax = axes[0]
    labels = [r[0].split(" (")[0] for r in order_rows]
    g = [float(r[2]) for r in order_rows]
    ax.barh(range(len(g)), g, color=["#c44", "#e8a33d", "#4a7", "#2a6", "#468"])
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("GFLOP/s (log scale)")
    ax.set_title(f"{N}x{N}x{N} matmul")
    for i, v in enumerate(g):
        ax.text(v, i, f" {v:.0f}", va="center", fontsize=8)

    ax = axes[1]
    blks = [r[0] for r in tile_rows]
    gf = [float(r[3]) for r in tile_rows]
    ax.semilogx(blks, gf, "o-", base=2, color="#4a7")
    ax.set_xlabel("BLOCK (tile side)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Tile size has an interior optimum")
    ax.grid(alpha=0.3)

    ax = axes[2]
    t = [r[0] for r in thread_rows]
    sp = [float(r[2]) for r in thread_rows]
    ax.plot(t, sp, "o-", color="#468", label="measured")
    ax.plot(t, t, "--", color="#999", label="perfect")
    ax.set_xlabel("threads")
    ax.set_ylabel("speedup")
    ax.set_title("Thread scaling")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[3]
    ns = [r[0] for r in size_rows]
    ax.plot(ns, [float(r[1]) for r in size_rows], "o-", color="#4a7", label="mine")
    ax.plot(ns, [float(r[2]) for r in size_rows], "s-", color="#468", label="torch.mm")
    ax.set_xlabel("N")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("The vendor gap does not close")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[4]
    ns = [r[0] for r in cross_rows]
    gain = [float(r[4]) for r in cross_rows]
    ax.semilogx(ns, gain, "o-", base=2, color="#a4c")
    ax.axhline(1.0, color="#999", ls="--", lw=1)
    ax.axvline(1732, color="#c44", ls=":", lw=1)
    ax.text(1732, min(gain), " B outgrows L3", fontsize=7, color="#c44")
    ax.set_xlabel("N")
    ax.set_ylabel("tiled / untiled speedup")
    ax.set_title("Tiling pays once B leaves cache")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "matmul.png", dpi=110)
    print(f"\n    wrote {OUT / 'matmul.png'}")


def main():
    t0 = time.perf_counter()
    print(f"torch {torch.__version__} | threads {torch.get_num_threads()}")
    (OUT / "triton_matmul.py").write_text(TRITON_REFERENCE)
    mod = build_all()
    correctness(mod)
    order = loop_order(mod)
    traffic_model()
    tiles = tile_sweep(mod)
    best_block = int(min(tiles, key=lambda r: float(r[2]))[0])
    cross = tiling_crossover(mod, best_block)
    threads = thread_sweep(mod, best_block)
    sizes = size_scaling(mod, best_block)
    figure(order, tiles, threads, sizes, cross)
    K.write_csv(OUT / "findings.csv", ROWS, ["section", "what", "value", "note"])
    print(f"    wrote {OUT / 'findings.csv'} and {OUT / 'triton_matmul.py'}")
    print(f"\ntotal wall time: {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
