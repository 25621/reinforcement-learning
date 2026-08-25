"""Attention two ways: materialised, and FlashAttention-style.

  attention_materialised   the textbook three steps. Compute the whole S x S
                           score matrix, softmax it, multiply by V. Every
                           intermediate is a real array in GPU memory.
  attention_flash          one kernel. Queries are processed in blocks; for
                           each query block the keys and values are streamed
                           past in blocks, and a running maximum and running
                           sum keep the softmax correct without ever holding
                           a whole row of scores.

The second is the online softmax from project 18, with one addition: the
running rescale is applied not only to the running sum but to a running
weighted average of value vectors.
"""

import torch
import triton
import triton.language as tl

NEG_INF = tl.constexpr(float("-inf"))


# ---------------------------------------------------------------- flash

@triton.jit
def _flash_kernel(Q, K, V, O,
                  sqh, sqm, skh, skn, svh, svn, soh, som,
                  S, D: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  CAUSAL: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)
    mmask = offs_m < S

    # The query block is loaded once and stays put for the whole kernel.
    q = tl.load(Q + pid_h * sqh + offs_m[:, None] * sqm + offs_d[None, :],
                mask=mmask[:, None], other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)   # running row maximum
    l_i = tl.zeros([BLOCK_M], tl.float32)                 # running row sum
    acc = tl.zeros([BLOCK_M, D], tl.float32)              # running weighted sum

    # A causal block only needs the key blocks up to its own last query.
    hi = (pid_m + 1) * BLOCK_M if CAUSAL else S

    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        nmask = offs_n < S
        k = tl.load(K + pid_h * skh + offs_n[:, None] * skn + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)
        v = tl.load(V + pid_h * svh + offs_n[:, None] * svn + offs_d[None, :],
                    mask=nmask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k), input_precision="ieee")
        s = tl.where(nmask[None, :], s, float("-inf"))
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= offs_n[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)          # how much the old total shrinks
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        # the same rescale applied to the running output, not just the sum
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v,
                                            input_precision="ieee")
        m_i = m_new

    o = acc / l_i[:, None]
    tl.store(O + pid_h * soh + offs_m[:, None] * som + offs_d[None, :], o,
             mask=mmask[:, None])


def attention_flash(q, k, v, causal=False, BLOCK_M=64, BLOCK_N=32,
                    num_warps=4, num_stages=2, out=None):
    """q, k, v are (H, S, D), already scaled by 1/sqrt(D) on q."""
    H, S, D = q.shape
    o = torch.empty_like(q) if out is None else out
    grid = (triton.cdiv(S, BLOCK_M), H)
    _flash_kernel[grid](q, k, v, o,
                        q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                        v.stride(0), v.stride(1), o.stride(0), o.stride(1),
                        S, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                        CAUSAL=causal, num_warps=num_warps,
                        num_stages=num_stages)
    return o


def flash_compile_info(q, k, v, causal=False, BLOCK_M=64, BLOCK_N=32,
                       num_warps=4, num_stages=2):
    H, S, D = q.shape
    o = torch.empty_like(q)
    grid = (triton.cdiv(S, BLOCK_M), H)
    kern = _flash_kernel[grid](q, k, v, o,
                               q.stride(0), q.stride(1), k.stride(0),
                               k.stride(1), v.stride(0), v.stride(1),
                               o.stride(0), o.stride(1),
                               S, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                               CAUSAL=causal, num_warps=num_warps,
                               num_stages=num_stages)
    return dict(regs=kern.n_regs, spills=kern.n_spills,
                shared=kern.metadata.shared)


# ------------------------------------------------------------ materialised

@triton.jit
def _causal_mask_kernel(Sm, stride, S, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, BLOCK)
    ptr = Sm + head * stride * S + row * stride + cols
    x = tl.load(ptr, mask=cols < S, other=0.0)
    x = tl.where(cols <= row, x, float("-inf"))
    tl.store(ptr, x, mask=cols < S)


def attention_materialised(q, k, v, causal=False, scores=None, probs=None,
                           out=None, matmul=None, softmax=None):
    """Three steps, each a separate kernel, each writing a full array.

    `matmul` and `softmax` are injected so this uses exactly the kernels from
    projects 19 and 18 rather than a second implementation of them.
    """
    H, S, D = q.shape
    sc = torch.empty((H, S, S), device=q.device) if scores is None else scores
    pr = torch.empty((H, S, S), device=q.device) if probs is None else probs
    o = torch.empty_like(q) if out is None else out
    for h in range(H):
        matmul(q[h], k[h].transpose(0, 1), c=sc[h])       # S x S scores
    if causal:
        _causal_mask_kernel[(S, H)](sc, sc.stride(1), S,
                                    BLOCK=triton.next_power_of_2(S))
    softmax(sc.view(H * S, S), out=pr.view(H * S, S))     # S x S probabilities
    for h in range(H):
        matmul(pr[h], v[h], c=o[h])                       # S x D output
    return o


# ------------------------------------------------------------------ counts

def flops(H, S, D, causal=False):
    """Two matmuls: scores (S x D x S) and output (S x S x D)."""
    f = 4.0 * H * S * S * D
    return f / 2 if causal else f


def bytes_flash(H, S, D, BLOCK_M):
    """Q and O once; K and V once per query block."""
    blocks = (S + BLOCK_M - 1) // BLOCK_M
    return (2 * H * S * D + 2 * blocks * H * S * D) * 4


def bytes_materialised(H, S, D):
    """QK: read Q,K write scores. softmax: read+write scores. PV: read
    probs and V, write O."""
    return (2 * H * S * D + H * S * S            # scores kernel
            + 2 * H * S * S                      # softmax
            + H * S * S + H * S * D + H * S * D  # output matmul
            ) * 4


def peak_extra_bytes_materialised(H, S):
    """The intermediates the materialised path must hold at once."""
    return 2 * H * S * S * 4
