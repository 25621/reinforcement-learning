"""A tiled matmul in Triton.

The same two-level tiling as project 17's CUDA kernel, but the second level -
the per-thread register tile - is never written down. `tl.dot` takes a
BLOCK_M x BLOCK_K block and a BLOCK_K x BLOCK_N block and produces a
BLOCK_M x BLOCK_N block; how that is split across threads and registers is the
compiler's problem.

GROUP_M controls the order blocks are visited in, which changes nothing about
the answer and everything about which blocks are resident together (see
project 15, where block order alone was worth 12.4%).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mm_kernel(A, B, C, M, N, K,
               sam, sak, sbk, sbn, scm, scn,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
               BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Visit blocks in GROUP_M-row stripes rather than straight along a row.
    # Blocks that run at the same time then share more of A and B, so the L2
    # cache is asked for fewer distinct tiles. GROUP_M=1 is plain row order.
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * sam + offs_k[None, :] * sak
    b_ptrs = B + offs_k[:, None] * sbk + offs_n[None, :] * sbn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_left),
                    other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_left) & (offs_n[None, :] < N),
                    other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        a_ptrs += BLOCK_K * sak
        b_ptrs += BLOCK_K * sbk

    c_ptrs = C + offs_m[:, None] * scm + offs_n[None, :] * scn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
CONFIGS = [
    (64, 64, 32, 8, 4, 2),
    (64, 128, 32, 8, 4, 2),
    (128, 64, 32, 8, 4, 2),
    (128, 128, 16, 8, 8, 2),
    (128, 128, 32, 8, 8, 2),
    (128, 128, 32, 8, 4, 2),
    (128, 128, 32, 1, 8, 2),      # same tile, plain row order
    (128, 128, 32, 8, 8, 3),      # deeper pipeline
    (128, 128, 64, 8, 8, 2),
    (128, 256, 32, 8, 8, 2),
    (256, 128, 32, 8, 8, 2),
    (64, 64, 64, 8, 4, 2),
]


def matmul(a, b, cfg=(128, 128, 32, 8, 8, 2), c=None):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    BM, BN, BK, GM, nw, ns = cfg
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    _mm_kernel[grid](a, b, c, M, N, K,
                     a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                     c.stride(0), c.stride(1),
                     BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                     num_warps=nw, num_stages=ns)
    return c


def compile_info(a, b, cfg):
    """Registers, spills and shared memory the compiler used for one config."""
    M, K = a.shape
    _, N = b.shape
    BM, BN, BK, GM, nw, ns = cfg
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    k = _mm_kernel[grid](a, b, c, M, N, K,
                         a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                         c.stride(0), c.stride(1),
                         BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                         num_warps=nw, num_stages=ns)
    return dict(regs=k.n_regs, spills=k.n_spills, shared=k.metadata.shared)


def arithmetic_intensity(BM, BN):
    """FLOP per byte of DRAM traffic for a BM x BN block tile."""
    return 1.0 / (2.0 * (1.0 / BM + 1.0 / BN))
