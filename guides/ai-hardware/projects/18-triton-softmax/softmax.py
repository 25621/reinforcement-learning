"""Softmax in Triton, four ways.

  softmax_fused    one program per row, whole row in registers, one pass in
                   and one pass out. The kernel you want.
  softmax_unsafe   the same, minus the "subtract the row max" step. Exists to
                   be broken.
  softmax_online   chunked, with a running maximum that is corrected as it
                   goes. Works for rows too long to hold at once - and it is
                   the same trick FlashAttention uses (project 21).
  softmax_multipass  four separate kernels, the way softmax looks if you write
                   it as a chain of tensor operations instead of one kernel.

Every kernel is safe to import; nothing runs at import time.
"""

import torch
import triton
import triton.language as tl

NEG_INF = float("-inf")
_NEG_INF = tl.constexpr(NEG_INF)   # Triton kernels may only read constexpr globals


# ------------------------------------------------------------------ fused

@triton.jit
def _fused_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    # `other=-inf` so the padding lanes lose the max and contribute exp(-inf)=0
    x = tl.load(X + row * stride_x + cols, mask=mask, other=_NEG_INF)
    x = x - tl.max(x, axis=0)          # the one line that makes it safe
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(Y + row * stride_y + cols, y, mask=mask)


@triton.jit
def _unsafe_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=_NEG_INF)
    e = tl.exp(x)                      # no max subtraction
    y = e / tl.sum(e, axis=0)
    tl.store(Y + row * stride_y + cols, y, mask=mask)


def _launch_row_kernel(kernel, x, num_warps=None, out=None):
    M, N = x.shape
    y = torch.empty_like(x) if out is None else out
    BLOCK = triton.next_power_of_2(N)
    if num_warps is None:
        num_warps = max(1, min(16, BLOCK // 256))
    kernel[(M,)](x, y, x.stride(0), y.stride(0), N, BLOCK=BLOCK,
                 num_warps=num_warps)
    return y


def softmax_fused(x, num_warps=None, out=None):
    return _launch_row_kernel(_fused_kernel, x, num_warps, out)


def softmax_unsafe(x, num_warps=None, out=None):
    return _launch_row_kernel(_unsafe_kernel, x, num_warps, out)


# ------------------------------------------------------------------ online

@triton.jit
def _online_kernel(X, Y, stride_x, stride_y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    xp = X + row * stride_x
    m = _NEG_INF        # running maximum
    l = 0.0            # running sum of exp(x - m), always rescaled to match m
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(xp + cols, mask=mask, other=_NEG_INF)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # rescale the old sum to the new maximum, then add this chunk
        l = l * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0),
                                           axis=0)
        m = m_new
    for start in range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(xp + cols, mask=mask, other=_NEG_INF)
        tl.store(Y + row * stride_y + cols, tl.exp(x - m) / l, mask=mask)


def softmax_online(x, BLOCK=1024, num_warps=4):
    M, N = x.shape
    y = torch.empty_like(x)
    _online_kernel[(M,)](x, y, x.stride(0), y.stride(0), N, BLOCK=BLOCK,
                         num_warps=num_warps)
    return y


# --------------------------------------------------------------- multipass
# What softmax costs when it is written as a chain of tensor operations:
# every intermediate is a full-size array that goes out to memory and comes
# back. Four kernels, six passes over the data.

@triton.jit
def _rowmax_kernel(X, M_, stride_x, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X + row * stride_x + cols, mask=cols < N, other=_NEG_INF)
    tl.store(M_ + row, tl.max(x, axis=0))


@triton.jit
def _subexp_kernel(X, M_, T, stride_x, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    m = tl.load(M_ + row)
    tl.store(T + row * stride_x + cols, tl.exp(x - m), mask=mask)


@triton.jit
def _rowsum_kernel(T, S, stride_x, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    t = tl.load(T + row * stride_x + cols, mask=cols < N, other=0.0)
    tl.store(S + row, tl.sum(t, axis=0))


@triton.jit
def _div_kernel(T, S, Y, stride_x, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    t = tl.load(T + row * stride_x + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride_x + cols, t / tl.load(S + row), mask=mask)


def softmax_multipass(x, tmp=None, out=None, rowmax=None, rowsum=None):
    M, N = x.shape
    BLOCK = triton.next_power_of_2(N)
    nw = max(1, min(16, BLOCK // 256))
    t = torch.empty_like(x) if tmp is None else tmp
    y = torch.empty_like(x) if out is None else out
    mx = torch.empty(M, device=x.device) if rowmax is None else rowmax
    sm = torch.empty(M, device=x.device) if rowsum is None else rowsum
    _rowmax_kernel[(M,)](x, mx, x.stride(0), N, BLOCK=BLOCK, num_warps=nw)
    _subexp_kernel[(M,)](x, mx, t, x.stride(0), N, BLOCK=BLOCK, num_warps=nw)
    _rowsum_kernel[(M,)](t, sm, x.stride(0), N, BLOCK=BLOCK, num_warps=nw)
    _div_kernel[(M,)](t, sm, y, x.stride(0), N, BLOCK=BLOCK, num_warps=nw)
    return y


# ------------------------------------------------------------------ traffic
# Bytes each variant must move, counted by hand. fp32 = 4 bytes.

def bytes_fused(M, N):
    return 2 * M * N * 4                     # read x, write y


def bytes_online(M, N):
    return 3 * M * N * 4                     # read x twice, write y


def bytes_multipass(M, N):
    # max: read x. subexp: read x, write t. sum: read t. div: read t, write y.
    return 6 * M * N * 4
