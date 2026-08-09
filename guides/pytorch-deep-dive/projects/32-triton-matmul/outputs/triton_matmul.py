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
