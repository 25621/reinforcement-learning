"""The GPU half of the bake-off: three Triton kernels, one per workload class.

They are deliberately plain. The point of this project is not to win a kernel
competition, it is to compare *devices* on the same work, so every backend gets
a straightforward implementation of the same maths and none gets hand-tuned
beyond a block size.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _axpy(A, B, C, D, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    a = tl.load(A + off, mask=m)
    b = tl.load(B + off, mask=m)
    c = tl.load(C + off, mask=m)
    tl.store(D + off, a * b + c, mask=m)


@triton.jit
def _chain(A, B, C, D, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < n
    a = tl.load(A + off, mask=m)
    b = tl.load(B + off, mask=m)
    c = tl.load(C + off, mask=m)
    t = a * b
    t = t + c
    t = tl.exp(-t * t)
    t = t * 0.5
    t = t - c
    t = t * t + a
    tl.store(D + off, t, mask=m)


@triton.jit
def _softmax(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=float("-inf"))
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    tl.store(Y + row * N + cols, e / tl.sum(e, axis=0), mask=mask)


def axpy(a, b, c, d, BLOCK=1024):
    n = a.numel()
    _axpy[(triton.cdiv(n, BLOCK),)](a, b, c, d, n, BLOCK=BLOCK, num_warps=4)


def chain(a, b, c, d, BLOCK=1024):
    n = a.numel()
    _chain[(triton.cdiv(n, BLOCK),)](a, b, c, d, n, BLOCK=BLOCK, num_warps=4)


def softmax(x, y):
    M, N = x.shape
    _softmax[(M,)](x, y, N, BLOCK=triton.next_power_of_2(N), num_warps=8)
