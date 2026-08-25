"""The one operation that decides LLM serving cost: `y = W @ x`, batched.

During generation a transformer reads every weight of the model to produce
*one* token per sequence. With a batch of B sequences it reads the same
weights once and uses them B times. So:

    bytes moved  ~ (number of weights) x (bytes per weight)      -- fixed
    FLOPs        ~ 2 x (number of weights) x B                   -- grows with B

which makes the arithmetic intensity 2B/bytes_per_weight and means the batch
size is literally the knob that moves this operation along the roofline. This
file measures that curve on the local card so the projections onto hardware we
cannot touch rest on something real.

Weights are stored in float16 (what a served model actually uses) and
converted to float32 inside the kernel, because Pascal has no fast float16
arithmetic. Only the bytes matter for the memory-bound half.
"""

import torch
import triton
import triton.language as tl


# ------------------------------------------------------- B = 1: a matrix-vector
@triton.jit
def _matvec(W, X, Y, N, K, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        m = (rn[:, None] < N) & (rk[None, :] < K)
        w = tl.load(W + rn[:, None] * K + rk[None, :], mask=m, other=0.0)
        x = tl.load(X + rk, mask=rk < K, other=0.0)
        acc += tl.sum(w.to(tl.float32) * x.to(tl.float32)[None, :], axis=1)
    tl.store(Y + rn, acc, mask=rn < N)


# ------------------------------------------------------- B >= 16: a real matmul
@triton.jit
def _matmul(W, X, Y, B, N, K,
            BLOCK_B: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    rb = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_B, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rb[:, None] * K + rk[None, :],
                    mask=(rb[:, None] < B) & (rk[None, :] < K), other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :],
                    mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
        acc += tl.dot(x.to(tl.float32), tl.trans(w.to(tl.float32)))
    tl.store(Y + rb[:, None] * N + rn[None, :], acc,
             mask=(rb[:, None] < B) & (rn[None, :] < N))


def decode_step(W, X, Y, BLOCK_N=64, BLOCK_K=64, BLOCK_B=16):
    """One `y = x @ W.T` for a batch of B token positions."""
    B, K = X.shape
    N = W.shape[0]
    if B == 1:
        return _matvec[(triton.cdiv(N, BLOCK_N),)](
            W, X, Y, N, K, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=4)
    bb = max(16, min(BLOCK_B, triton.next_power_of_2(B)))
    return _matmul[(triton.cdiv(B, bb), triton.cdiv(N, BLOCK_N))](
        W, X, Y, B, N, K, BLOCK_B=bb, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4)


def bytes_moved(B, N, K, wbytes=2):
    """What the ALGORITHM requires: every weight read once, whatever B is."""
    return N * K * wbytes + (B * K + B * N) * 4


def bytes_dram(B, N, K, BLOCK_B, wbytes=2):
    """What the KERNEL actually reads.

    A tiled kernel walks the whole weight matrix once per batch tile, so a
    batch of 512 processed 64 rows at a time reads the weights 8 times. This
    is invisible in the FLOP count and is the reason a serving engine wants
    its batch tile to be at least as large as its batch.
    """
    tiles = -(-B // BLOCK_B)
    return tiles * N * K * wbytes + (B * K + B * N) * 4


def flops(B, N, K):
    return 2.0 * B * N * K
