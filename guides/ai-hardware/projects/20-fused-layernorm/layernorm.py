"""LayerNorm, and LayerNorm fused into the linear layer that follows it.

Four ways to compute the same thing:

  layernorm_chain   LayerNorm written as separate tensor operations - mean,
                    centre, variance, normalise. Four kernels, four trips to
                    memory. This is what you get without a library kernel.
  layernorm         one kernel per row. This is what `F.layer_norm` is.
  ln_linear_split   layernorm() then a separate matmul. This is what eager
                    PyTorch runs: two good kernels with a full-size
                    intermediate written to memory between them.
  ln_linear_fused   one kernel. The normalised rows never leave the chip.

Plus `layernorm_naive_var`, which computes the variance as E[x^2] - E[x]^2 and
exists to be broken.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------- LayerNorm

@triton.jit
def _ln_kernel(X, Y, G, B, stride, K, eps, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / K
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / K
    xn = xc * tl.rsqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride + cols, xn * g + b, mask=mask)


@triton.jit
def _ln_naive_var_kernel(X, Y, G, B, stride, K, eps, BLOCK_K: tl.constexpr):
    """Variance as E[x^2] - E[x]^2: one pass, and one catastrophic subtraction."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    mean = tl.sum(x, axis=0) / K
    mean_sq = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / K
    var = mean_sq - mean * mean          # <- two big numbers, small difference
    xn = (x - mean) * tl.rsqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride + cols, xn * g + b, mask=mask)


def layernorm(x, g, b, eps=1e-5, out=None, kernel=None):
    M, K = x.shape
    y = torch.empty_like(x) if out is None else out
    BLOCK_K = triton.next_power_of_2(K)
    (kernel or _ln_kernel)[(M,)](x, y, g, b, x.stride(0), K, eps,
                                 BLOCK_K=BLOCK_K,
                                 num_warps=max(1, min(16, BLOCK_K // 256)))
    return y


def layernorm_naive_var(x, g, b, eps=1e-5, out=None):
    return layernorm(x, g, b, eps, out, kernel=_ln_naive_var_kernel)


# ---------------------------------------------- LayerNorm as a tensor chain

@triton.jit
def _mean_kernel(X, MU, stride, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    x = tl.load(X + row * stride + cols, mask=cols < K, other=0.0)
    tl.store(MU + row, tl.sum(x, axis=0) / K)


@triton.jit
def _centre_kernel(X, MU, T, stride, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    tl.store(T + row * stride + cols, x - tl.load(MU + row), mask=mask)


@triton.jit
def _var_kernel(T, VR, stride, K, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    t = tl.load(T + row * stride + cols, mask=cols < K, other=0.0)
    tl.store(VR + row, tl.sum(t * t, axis=0) / K)


@triton.jit
def _scale_kernel(T, VR, G, B, Y, stride, K, eps, BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_K)
    mask = cols < K
    t = tl.load(T + row * stride + cols, mask=mask, other=0.0)
    r = tl.rsqrt(tl.load(VR + row) + eps)
    g = tl.load(G + cols, mask=mask, other=1.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    tl.store(Y + row * stride + cols, t * r * g + b, mask=mask)


def layernorm_chain(x, g, b, eps=1e-5, buf=None):
    M, K = x.shape
    BLOCK_K = triton.next_power_of_2(K)
    nw = max(1, min(16, BLOCK_K // 256))
    if buf is None:
        buf = (torch.empty_like(x), torch.empty_like(x),
               torch.empty(M, device=x.device), torch.empty(M, device=x.device))
    t, y, mu, vr = buf
    _mean_kernel[(M,)](x, mu, x.stride(0), K, BLOCK_K=BLOCK_K, num_warps=nw)
    _centre_kernel[(M,)](x, mu, t, x.stride(0), K, BLOCK_K=BLOCK_K, num_warps=nw)
    _var_kernel[(M,)](t, vr, x.stride(0), K, BLOCK_K=BLOCK_K, num_warps=nw)
    _scale_kernel[(M,)](t, vr, g, b, y, x.stride(0), K, eps, BLOCK_K=BLOCK_K,
                        num_warps=nw)
    return y


# ------------------------------------------------ LayerNorm + linear, fused

@triton.jit
def _ln_linear_fused_kernel(X, W, Bias, G, Beta, Y,
                            M, N, K, sx, swk, swn, sy, eps,
                            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                            BLOCK_K: tl.constexpr):
    """One program owns BLOCK_M rows. It normalises them once, then walks the
    whole width of the output using the normalised rows straight from
    registers - they are never written to memory."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_k = tl.arange(0, BLOCK_K)
    rmask = rows < M
    kmask = cols_k < K

    x = tl.load(X + rows[:, None] * sx + cols_k[None, :],
                mask=rmask[:, None] & kmask[None, :], other=0.0)
    mean = tl.sum(x, axis=1) / K
    xc = tl.where(kmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / K
    g = tl.load(G + cols_k, mask=kmask, other=1.0)
    beta = tl.load(Beta + cols_k, mask=kmask, other=0.0)
    xn = xc * tl.rsqrt(var + eps)[:, None] * g[None, :] + beta[None, :]
    xn = tl.where(kmask[None, :], xn, 0.0)

    for n0 in range(0, N, BLOCK_N):
        cols_n = n0 + tl.arange(0, BLOCK_N)
        nmask = cols_n < N
        w = tl.load(W + cols_k[:, None] * swk + cols_n[None, :] * swn,
                    mask=kmask[:, None] & nmask[None, :], other=0.0)
        acc = tl.dot(xn, w, input_precision="ieee")
        acc += tl.load(Bias + cols_n, mask=nmask, other=0.0)[None, :]
        tl.store(Y + rows[:, None] * sy + cols_n[None, :], acc,
                 mask=rmask[:, None] & nmask[None, :])


@triton.jit
def _ln_linear_2d_kernel(X, W, Bias, G, Beta, Y,
                         M, N, K, sx, swk, swn, sy, eps,
                         BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                         BLOCK_K: tl.constexpr):
    """Same fusion, but with a 2D grid: one program per (row block, column
    block). More programs - better for filling the machine - at the cost of
    normalising the same rows once per column block."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols_k = tl.arange(0, BLOCK_K)
    cols_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rmask, kmask, nmask = rows < M, cols_k < K, cols_n < N

    x = tl.load(X + rows[:, None] * sx + cols_k[None, :],
                mask=rmask[:, None] & kmask[None, :], other=0.0)
    mean = tl.sum(x, axis=1) / K
    xc = tl.where(kmask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / K
    g = tl.load(G + cols_k, mask=kmask, other=1.0)
    beta = tl.load(Beta + cols_k, mask=kmask, other=0.0)
    xn = xc * tl.rsqrt(var + eps)[:, None] * g[None, :] + beta[None, :]
    xn = tl.where(kmask[None, :], xn, 0.0)

    w = tl.load(W + cols_k[:, None] * swk + cols_n[None, :] * swn,
                mask=kmask[:, None] & nmask[None, :], other=0.0)
    acc = tl.dot(xn, w, input_precision="ieee")
    acc += tl.load(Bias + cols_n, mask=nmask, other=0.0)[None, :]
    tl.store(Y + rows[:, None] * sy + cols_n[None, :], acc,
             mask=rmask[:, None] & nmask[None, :])


def ln_linear_fused(x, w, bias, g, beta, eps=1e-5, BLOCK_M=32, BLOCK_N=64,
                    num_warps=8, out=None, two_d=False):
    M, K = x.shape
    _, N = w.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype) if out is None else out
    BLOCK_K = triton.next_power_of_2(K)
    args = (x, w, bias, g, beta, y, M, N, K, x.stride(0), w.stride(0),
            w.stride(1), y.stride(0), eps)
    kw = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
              num_warps=num_warps)
    if two_d:
        _ln_linear_2d_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
            *args, **kw)
    else:
        _ln_linear_fused_kernel[(triton.cdiv(M, BLOCK_M),)](*args, **kw)
    return y


def fused_compile_info(x, w, bias, g, beta, BLOCK_M=32, BLOCK_N=64,
                       num_warps=8, two_d=False):
    M, K = x.shape
    _, N = w.shape
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    BLOCK_K = triton.next_power_of_2(K)
    args = (x, w, bias, g, beta, y, M, N, K, x.stride(0), w.stride(0),
            w.stride(1), y.stride(0), 1e-5)
    kw = dict(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
              num_warps=num_warps)
    if two_d:
        k = _ln_linear_2d_kernel[(triton.cdiv(M, BLOCK_M),
                                  triton.cdiv(N, BLOCK_N))](*args, **kw)
    else:
        k = _ln_linear_fused_kernel[(triton.cdiv(M, BLOCK_M),)](*args, **kw)
    return dict(regs=k.n_regs, spills=k.n_spills, shared=k.metadata.shared)


# ------------------------------------------------------------------ traffic

def bytes_split(M, N, K):
    """LayerNorm kernel (read x, write xn) then matmul (read xn, read W,
    write y)."""
    return (2 * M * K + M * K + K * N + M * N) * 4


def bytes_fused(M, N, K):
    """One kernel: read x, read W, write y."""
    return (M * K + K * N + M * N) * 4


def bytes_chain(M, N, K):
    """Tensor-op LayerNorm (6 passes over MK) plus the matmul."""
    return (6 * M * K + M * K + K * N + M * N) * 4
