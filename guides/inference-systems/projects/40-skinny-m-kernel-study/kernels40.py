"""Kernels for project 40: the shapes a decode step actually asks for.

Three ideas beyond the generic GEMM in enginelib:

  * split-K      -- when N is small there are not enough output tiles to fill
                    the GPU, so split the reduction dimension instead
  * int4 weights -- the Marlin idea: the kernel is memory-bound, so store the
                    weights in 4 bits and unpack them inside the kernel
  * packing      -- the CPU-side layout the int4 kernel expects
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def k_zero(y_ptr, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(y_ptr + off, tl.zeros((BLOCK,), tl.float32), mask=off < n)


@triton.jit
def k_gemm_splitk(
    x_ptr, w_ptr, part_ptr, M, N, K,
    SK: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """part[sk] = x[M, chunk_sk] @ w[chunk_sk, N], one program per (tile, chunk).

    Each chunk of the reduction dimension gets its own program, so the number
    of programs is multiplied by SK without changing the bytes read -- the same
    trick FlashDecoding plays on attention in project 39.

    The partials are summed by `k_reduce_splitk` rather than by atomics:
    Triton's `tl.atomic_add` emits PTX memory-ordering qualifiers (`.acq_rel`,
    and even `.relaxed`) that ptxas refuses below sm_70, so on this card a
    separate reduction pass is the only option.  It is also what CUTLASS does.
    """
    pid = tl.program_id(0)
    sk = tl.program_id(1)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    chunk = K // SK
    k_lo = sk * chunk
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(k_lo, k_lo + chunk, BK):
        a = tl.load(x_ptr + rm[:, None] * K + (k0 + rk)[None, :],
                    mask=rm[:, None] < M, other=0.0)
        b = tl.load(w_ptr + (k0 + rk)[:, None] * N + rn[None, :],
                    mask=rn[None, :] < N, other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
    tl.store(part_ptr + sk * M * N + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def k_reduce_splitk(part_ptr, y_ptr, n, SK: tl.constexpr, BLOCK: tl.constexpr):
    """Sum the SK partial products into the final output."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    acc = tl.zeros((BLOCK,), tl.float32)
    for sk in range(SK):
        acc += tl.load(part_ptr + sk * n + off, mask=mask, other=0.0)
    tl.store(y_ptr + off, acc, mask=mask)


@triton.jit
def k_gemm_int4(
    x_ptr, wq_ptr, sc_ptr, y_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """y[M,N] = x[M,K] @ dequantise(wq)[K,N], weights stored 4 bits each.

    `wq` is int32, eight 4-bit weights per word, packed along K: word r of
    column n holds the weights for k = 8r+0 .. 8r+7 at nibbles 0..7.
    `sc` holds one fp32 scale per BK consecutive k values per column (group-BK
    quantisation, the same layout AWQ and GPTQ produce).

    The inner loop loads each packed word ONCE and then walks its eight
    nibbles, accumulating an 8-way sum of thin `tl.dot`s.  The obvious
    alternative -- build one wide [BK, BN] dequantised tile -- re-reads every
    packed word eight times, which puts the fp32 byte count straight back into
    the cache hierarchy and cost 2.3x here.
    """
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BN)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    KP: tl.constexpr = BK // 8          # packed words per block
    rp = tl.arange(0, KP)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        p = tl.load(wq_ptr + (k0 // 8 + rp[:, None]) * N + rn[None, :],
                    mask=rn[None, :] < N, other=0)
        s = tl.load(sc_ptr + (k0 // BK) * N + rn, mask=rn < N, other=0.0)
        for j in tl.static_range(8):
            q = ((p >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            a = tl.load(x_ptr + rm[:, None] * K + (k0 + rp * 8 + j)[None, :],
                        mask=rm[:, None] < M, other=0.0)
            acc += tl.dot(a, q * s[None, :], input_precision="ieee")
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def k_gemv_int4(
    x_ptr, wq_ptr, sc_ptr, y_ptr, N, K,
    BN: tl.constexpr, BK: tl.constexpr,
):
    """The M = 1 case written as a true matrix-VECTOR product.

    `tl.dot` cannot take fewer than 16 rows, so the GEMM kernel above pads a
    single row to 16 and performs 16x the arithmetic it needs.  In fp32 that
    is free -- the kernel is waiting on memory anyway.  With 4-bit weights it
    is not free, because there are 7.5x fewer bytes to wait for.  This version
    drops `tl.dot` entirely and multiplies-and-adds elementwise instead.
    """
    pid_n = tl.program_id(0)
    rn = pid_n * BN + tl.arange(0, BN)
    KP: tl.constexpr = BK // 8
    rp = tl.arange(0, KP)
    acc = tl.zeros((BN,), tl.float32)
    for k0 in range(0, K, BK):
        p = tl.load(wq_ptr + (k0 // 8 + rp[:, None]) * N + rn[None, :],
                    mask=rn[None, :] < N, other=0)
        s = tl.load(sc_ptr + (k0 // BK) * N + rn, mask=rn < N, other=0.0)
        grp = tl.zeros((BN,), tl.float32)
        for j in tl.static_range(8):
            q = ((p >> (4 * j)) & 0xF).to(tl.float32) - 8.0
            xj = tl.load(x_ptr + k0 + rp * 8 + j)
            grp += tl.sum(q * xj[:, None], 0)
        acc += grp * s
    tl.store(y_ptr + rn, acc, mask=rn < N)


def pack_int4(w: torch.Tensor, group: int = 64):
    """CPU-side quantise-and-pack.  `w` is [K, N] fp32 on the host.

    Returns (packed int32 [K//8, N], scales fp32 [K//group, N]).
    Symmetric 4-bit: levels -8..7, stored biased by +8 so they fit a nibble.
    """
    K, N = w.shape
    wg = w.reshape(K // group, group, N)
    scale = wg.abs().amax(dim=1) / 7.0
    scale = torch.clamp(scale, min=1e-8)
    q = torch.round(wg / scale[:, None, :]).clamp(-8, 7).to(torch.int32) + 8
    q = q.reshape(K, N)
    packed = torch.zeros(K // 8, N, dtype=torch.int32)
    for jj in range(8):
        packed |= (q[jj::8].reshape(K // 8, N) & 0xF) << (4 * jj)
    return packed, scale.contiguous()


def dequantised(packed: torch.Tensor, scale: torch.Tensor, K: int, N: int,
                group: int = 64) -> torch.Tensor:
    """The reference the kernel must reproduce, computed on the CPU."""
    q = torch.zeros(K, N, dtype=torch.float32)
    for jj in range(8):
        q[jj::8] = ((packed >> (4 * jj)) & 0xF).to(torch.float32) - 8.0
    return q * scale.repeat_interleave(group, dim=0)
