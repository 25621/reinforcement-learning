"""enginelib.py -- a tiny transformer inference engine written entirely in Triton.

Shared by inference-systems Phase 6, projects 37-43.

WHY THIS EXISTS
---------------
Phase 6 is about the kernels underneath an inference engine.  You cannot study
those with a library that hides them, so this file *is* the engine: every
operation in a forward pass is a Triton kernel we launch by name, time by name,
and can swap out.

WHY NOT PyTorch OPS
-------------------
The GPU on this machine is a GTX 1070 Ti (sm_61, Pascal).  The installed
PyTorch ships prebuilt kernels for sm_70+ only, and cuBLAS 13 dropped Pascal,
so `torch.matmul(a, b)` on a CUDA tensor raises.  Triton, however, JIT-compiles
for whatever card it finds, so Triton kernels run for real.  The rules we
therefore follow everywhere:

  * allocate with ``torch.empty(..., device='cuda')``   (a plain cudaMalloc)
  * fill with ``.copy_(cpu_tensor)``                    (a plain memcpy)
  * compute ONLY inside Triton kernels
  * check results on the CPU

Everything measured by projects 37-43 is real hardware, not a simulation.
"""

from __future__ import annotations

import ctypes
import math
import statistics
import time
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------
# device facts
# --------------------------------------------------------------------------

DEV = "cuda"


def device_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    # fp32 peak = SMs * 128 fp32 lanes/SM (Pascal GP104) * 2 flops/FMA * clock
    clock_hz = 1911e6  # max SM clock reported by nvidia-smi
    lanes = 128
    return {
        "name": p.name,
        "cc": f"sm_{p.major}{p.minor}",
        "sms": p.multi_processor_count,
        "mem_gb": p.total_memory / 1e9,
        "peak_fp32_gflops": p.multi_processor_count * lanes * 2 * clock_hz / 1e9,
        "spec_bw_gbs": 256.0,  # GTX 1070 Ti: 256-bit GDDR5 @ 8 Gbps
    }


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@triton.jit
def k_spin(out_ptr, iters):
    """Heat the card back to its steady clock.  Stores unconditionally so the
    compiler cannot delete the loop (a masked-off store made an earlier version
    of this kernel a no-op)."""
    acc = 0.0
    for i in range(iters):
        acc = acc * 1.0000001 + 1.0
    tl.store(out_ptr + tl.program_id(0), acc)


@triton.jit
def k_copy(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    """b = a.  The simplest kernel there is, and therefore the honest way to
    ask "how many bytes per second can this card actually move?"."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    tl.store(b_ptr + off, tl.load(a_ptr + off, mask=mask, other=0.0), mask=mask)


@triton.jit
def k_rmsnorm(x_ptr, w_ptr, y_ptr, D: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    rms = tl.sqrt(tl.sum(x * x) / D + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * D + cols, x / rms * w, mask=mask)


@triton.jit
def k_add(a_ptr, b_ptr, n, BLOCK: tl.constexpr):
    """a += b, elementwise (the residual connection)."""
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = off < n
    a = tl.load(a_ptr + off, mask=mask, other=0.0)
    b = tl.load(b_ptr + off, mask=mask, other=0.0)
    tl.store(a_ptr + off, a + b, mask=mask)


@triton.jit
def k_swiglu(gu_ptr, y_ptr, rows, FF: tl.constexpr, BLOCK: tl.constexpr):
    """y = silu(gate) * up, where gu is [rows, 2*FF] = [gate | up]."""
    r = tl.program_id(0)
    c0 = tl.program_id(1) * BLOCK
    cols = c0 + tl.arange(0, BLOCK)
    mask = cols < FF
    g = tl.load(gu_ptr + r * 2 * FF + cols, mask=mask, other=0.0)
    u = tl.load(gu_ptr + r * 2 * FF + FF + cols, mask=mask, other=0.0)
    tl.store(y_ptr + r * FF + cols, g * tl.sigmoid(g) * u, mask=mask)


@triton.jit
def k_matmul(
    x_ptr, w_ptr, y_ptr, M, N, K,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    """y[M,N] = x[M,K] @ w[K,N].  All row-major, all fp32.

    This is the generic GEMM every projection uses.  Prefill calls it with
    M = batch*seq_len (thousands); decode calls it with M = batch (1-128), and
    the difference between those two shapes is most of Phase 6.
    """
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BM)
    grid_n = tl.cdiv(N, BN)
    width = GROUP_M * grid_n
    group = pid // width
    m0 = group * GROUP_M
    gsize = min(grid_m - m0, GROUP_M)
    pid_m = m0 + ((pid % width) % gsize)
    pid_n = (pid % width) // gsize

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    x_p = x_ptr + rm[:, None] * K + rk[None, :]
    w_p = w_ptr + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a = tl.load(x_p, mask=(rm[:, None] < M) & (rk[None, :] + k0 < K), other=0.0)
        b = tl.load(w_p, mask=(rk[:, None] + k0 < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b, input_precision="ieee")
        x_p += BK
        w_p += BK * N
    tl.store(y_ptr + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


@triton.jit
def k_rope_q(q_ptr, len_ptr, T, HHD: tl.constexpr, HD: tl.constexpr,
             BASE: tl.constexpr, HALF: tl.constexpr):
    """Rotary position embedding, applied in place to the query rows.

    Row layout is [tokens, H*HD].  `len_ptr` holds the position of the FIRST
    token of this step, on the device, so a captured CUDA graph stays correct
    as the sequence grows (project 41 depends on this)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    base = t * HHD + h * HD
    pos = tl.load(len_ptr) + t % T
    i = tl.arange(0, HALF)
    theta = pos.to(tl.float32) / tl.exp(i.to(tl.float32) * (math.log(BASE) / HALF))
    c = tl.cos(theta)
    s = tl.sin(theta)
    a = tl.load(q_ptr + base + i)
    b = tl.load(q_ptr + base + HALF + i)
    tl.store(q_ptr + base + i, a * c - b * s)
    tl.store(q_ptr + base + HALF + i, a * s + b * c)


@triton.jit
def k_write_kv(
    kv_ptr, kc_ptr, vc_ptr, len_ptr,
    T, B, KVH: tl.constexpr, HD: tl.constexpr, S, QOFF: tl.constexpr,
    ROWW: tl.constexpr, BASE: tl.constexpr, HALF: tl.constexpr,
):
    """RoPE the keys and scatter K and V into the paged-by-sequence cache.

    Fusing the rotation with the cache write is the real engines' trick: the
    keys are already in registers, so rotating them there costs one pass
    instead of two (guide Phase 6, "QKV projection + RoPE + KV-cache write
    fused into one").
    """
    t = tl.program_id(0)          # token within this step
    b = tl.program_id(1)          # sequence
    h = tl.program_id(2)          # kv head
    pos = tl.load(len_ptr) + t
    src = (b * T + t) * ROWW + QOFF + h * HD
    i = tl.arange(0, HALF)
    theta = pos.to(tl.float32) / tl.exp(i.to(tl.float32) * (math.log(BASE) / HALF))
    c = tl.cos(theta)
    s = tl.sin(theta)
    ka = tl.load(kv_ptr + src + i)
    kb = tl.load(kv_ptr + src + HALF + i)
    dst = ((b * KVH + h) * S + pos) * HD
    tl.store(kc_ptr + dst + i, ka * c - kb * s)
    tl.store(kc_ptr + dst + HALF + i, ka * s + kb * c)
    j = tl.arange(0, HD)
    v = tl.load(kv_ptr + src + KVH * HD + j)
    tl.store(vc_ptr + dst + j, v)


@triton.jit
def k_attn_prefill(
    q_ptr, kc_ptr, vc_ptr, o_ptr,
    T, S, H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, scale,
    QS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
):
    """Causal FlashAttention over the whole prompt.  Scores never leave SRAM."""
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    kvh = h // (H // KVH)
    rm = pid_m * BM + tl.arange(0, BM)
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + (b * T + rm[:, None]) * QS + h * HD + d[None, :],
                mask=rm[:, None] < T, other=0.0)
    kbase = (b * KVH + kvh) * S * HD
    m_i = tl.full((BM,), -float("inf"), tl.float32)
    l_i = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, HD), tl.float32)
    hi = (pid_m + 1) * BM
    for s0 in range(0, hi, BN):
        rn = s0 + tl.arange(0, BN)
        kmask = rn[:, None] < T
        k = tl.load(kc_ptr + kbase + rn[:, None] * HD + d[None, :], mask=kmask, other=0.0)
        v = tl.load(vc_ptr + kbase + rn[:, None] * HD + d[None, :], mask=kmask, other=0.0)
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        qk = tl.where(rn[None, :] <= rm[:, None], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        l_i = l_i * alpha + tl.sum(p, 1)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(o_ptr + (b * T + rm[:, None]) * (H * HD) + h * HD + d[None, :], acc,
             mask=rm[:, None] < T)


@triton.jit
def k_attn_decode(
    q_ptr, kc_ptr, vc_ptr, o_ptr, len_ptr,
    S, H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, scale,
    QS: tl.constexpr, BN: tl.constexpr,
):
    """Decode attention, one program per (sequence, head).

    M = 1: there is a single query row, so there is no matmul here at all --
    it is a dot product against every cached key.  Project 39 ablates this
    against the split version below.
    """
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    kvh = h // (H // KVH)
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + b * QS + h * HD + d)
    L = tl.load(len_ptr) + 1          # this token is already in the cache
    kbase = (b * KVH + kvh) * S * HD
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HD,), tl.float32)
    for s0 in range(0, L, BN):
        rn = s0 + tl.arange(0, BN)
        kmask = rn < L
        k = tl.load(kc_ptr + kbase + rn[:, None] * HD + d[None, :],
                    mask=kmask[:, None], other=0.0)
        v = tl.load(vc_ptr + kbase + rn[:, None] * HD + d[None, :],
                    mask=kmask[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k, 1) * scale
        qk = tl.where(kmask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new
    tl.store(o_ptr + b * (H * HD) + h * HD + d, acc / l_i)


@triton.jit
def k_attn_decode_split(
    q_ptr, kc_ptr, vc_ptr, part_ptr, ml_ptr, len_ptr,
    S, H: tl.constexpr, KVH: tl.constexpr, HD: tl.constexpr, scale,
    QS: tl.constexpr, NSPLIT: tl.constexpr, BN: tl.constexpr,
):
    """FlashDecoding: split the cached sequence across NSPLIT programs.

    Each program handles a contiguous slice of the KV length and writes a
    partial output plus its running max and sum, which k_attn_combine merges.
    """
    bh = tl.program_id(0)
    sp = tl.program_id(1)
    b = bh // H
    h = bh % H
    kvh = h // (H // KVH)
    d = tl.arange(0, HD)
    q = tl.load(q_ptr + b * QS + h * HD + d)
    L = tl.load(len_ptr) + 1
    chunk = tl.cdiv(L, NSPLIT)
    lo = sp * chunk
    hi = min(lo + chunk, L)
    kbase = (b * KVH + kvh) * S * HD
    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HD,), tl.float32)
    for s0 in range(lo, hi, BN):
        rn = s0 + tl.arange(0, BN)
        kmask = rn < hi
        k = tl.load(kc_ptr + kbase + rn[:, None] * HD + d[None, :],
                    mask=kmask[:, None], other=0.0)
        v = tl.load(vc_ptr + kbase + rn[:, None] * HD + d[None, :],
                    mask=kmask[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k, 1) * scale
        qk = tl.where(kmask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        acc = acc * alpha + tl.sum(p[:, None] * v, 0)
        l_i = l_i * alpha + tl.sum(p, 0)
        m_i = m_new
    base = (bh * NSPLIT + sp)
    tl.store(part_ptr + base * HD + d, acc)
    tl.store(ml_ptr + base * 2 + 0, m_i)
    tl.store(ml_ptr + base * 2 + 1, l_i)


@triton.jit
def k_attn_combine(
    part_ptr, ml_ptr, o_ptr,
    H: tl.constexpr, HD: tl.constexpr, NSPLIT: tl.constexpr,
):
    """Merge the NSPLIT partial softmaxes into one, by rescaling each to the
    global maximum.  This is the whole reason splitting is legal: softmax is
    associative once you carry (max, sum) alongside the partial output."""
    bh = tl.program_id(0)
    b = bh // H
    h = bh % H
    d = tl.arange(0, HD)
    sp = tl.arange(0, NSPLIT)
    m = tl.load(ml_ptr + (bh * NSPLIT + sp) * 2 + 0)
    l = tl.load(ml_ptr + (bh * NSPLIT + sp) * 2 + 1)
    m_g = tl.max(m, 0)
    w = tl.exp(m - m_g)
    l_g = tl.sum(l * w, 0)
    part = tl.load(part_ptr + (bh * NSPLIT + sp)[:, None] * HD + d[None, :])
    out = tl.sum(part * w[:, None], 0) / l_g
    tl.store(o_ptr + b * (H * HD) + h * HD + d, out)


@triton.jit
def k_gather_last(x_ptr, y_ptr, T, D: tl.constexpr, BLOCK: tl.constexpr):
    """Copy the LAST token of each sequence into a compact [B, D] buffer.

    Only the final position's hidden state feeds the output head, and in
    prefill those rows are T apart, not contiguous -- so they have to be
    gathered before the head's GEMM can see them as a matrix.
    """
    b = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    v = tl.load(x_ptr + ((b + 1) * T - 1) * D + cols, mask=mask, other=0.0)
    tl.store(y_ptr + b * D + cols, v, mask=mask)


@triton.jit
def k_incr(len_ptr, n):
    tl.store(len_ptr, tl.load(len_ptr) + n)


@triton.jit
def k_argmax(logits_ptr, out_ptr, V, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    best = -float("inf")
    idx = 0
    for c0 in range(0, V, BLOCK):
        cols = c0 + tl.arange(0, BLOCK)
        x = tl.load(logits_ptr + r * V + cols, mask=cols < V, other=-float("inf"))
        m = tl.max(x, 0)
        i = tl.argmax(x, 0) + c0
        idx = tl.where(m > best, i, idx)
        best = tl.maximum(best, m)
    tl.store(out_ptr + r, idx)


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

_spin_out = None


def warm(rounds: int = 20) -> None:
    """Re-heat the GPU before every measurement.

    The card drops its clock while the CPU builds the next case; without this,
    readings drift by ~7% and produce results that look like real effects.
    """
    global _spin_out
    if _spin_out is None:
        _spin_out = torch.empty(256, device=DEV)
    for _ in range(rounds):
        k_spin[(256,)](_spin_out, 20000)
    torch.cuda.synchronize()


def gpu_time(fn, reps: int = 20, warmup: int = 3, rewarm: bool = True) -> float:
    """Milliseconds per call, measured with CUDA events on the GPU timeline."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if rewarm:
        warm()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(reps):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / reps


def wall_time(fn, reps: int = 20, warmup: int = 3) -> float:
    """Milliseconds per call as the *host* sees it, including launch overhead."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    warm()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def interleaved(variants: dict, reps: int = 5, inner: int = 10) -> dict:
    """Round-robin A/B timing.  This machine is shared, so measuring A fully
    and then B fully lets a background job land on one of them; alternating
    spreads any drift across all variants equally."""
    samples = {k: [] for k in variants}
    for _ in range(reps):
        for k, fn in variants.items():
            samples[k].append(gpu_time(fn, reps=inner))
    return {k: statistics.median(v) for k, v in samples.items()}


# --------------------------------------------------------------------------
# CUDA graphs (via ctypes -- torch's own graph API needs kernels this card
# cannot run, so we call the runtime directly)
# --------------------------------------------------------------------------

_rt = None


def _runtime():
    global _rt
    if _rt is None:
        torch.cuda.init()
        _rt = ctypes.CDLL("libcudart.so.12", mode=ctypes.RTLD_GLOBAL)
        _rt.cudaStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _rt.cudaStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        _rt.cudaGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_ulonglong]
        _rt.cudaGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        _rt.cudaGraphGetNodes.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        _rt.cudaGraphDestroy.argtypes = [ctypes.c_void_p]
        _rt.cudaGraphExecDestroy.argtypes = [ctypes.c_void_p]
    return _rt


class Graph:
    """Capture a sequence of kernel launches once, replay it with one call.

    Every launch inside `fn` must already be compiled (Triton JITs on first
    call and that would be captured as work, or fail outright), so `fn` is run
    a few times before capture.
    """

    def __init__(self, fn, warmup: int = 3):
        rt = _runtime()
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        self._graph = ctypes.c_void_p()
        self._exec = ctypes.c_void_p()
        with torch.cuda.stream(stream):
            sp = ctypes.c_void_p(stream.cuda_stream)
            _chk(rt.cudaStreamBeginCapture(sp, 0), "BeginCapture")
            fn()
            _chk(rt.cudaStreamEndCapture(sp, ctypes.byref(self._graph)), "EndCapture")
        _chk(rt.cudaGraphInstantiate(ctypes.byref(self._exec), self._graph, 0), "Instantiate")
        # Capture must happen on a side stream, but replay goes back on the
        # default stream so CUDA events recorded there still bracket the work.
        self._stream = torch.cuda.current_stream()
        self.n_nodes = self._count_nodes()

    def _count_nodes(self) -> int:
        rt = _runtime()
        n = ctypes.c_size_t(0)
        rt.cudaGraphGetNodes(self._graph, None, ctypes.byref(n))
        return n.value

    def replay(self) -> None:
        _runtime().cudaGraphLaunch(self._exec, ctypes.c_void_p(self._stream.cuda_stream))

    def close(self) -> None:
        """Free the graph and its executable.  Building thousands of graphs in
        one process without this exhausts device memory."""
        if self._exec:
            torch.cuda.synchronize()
            _runtime().cudaGraphExecDestroy(self._exec)
            _runtime().cudaGraphDestroy(self._graph)
            self._exec = ctypes.c_void_p()
            self._graph = ctypes.c_void_p()

    def __call__(self) -> None:
        self.replay()


def _chk(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"CUDA runtime error {code} in {what}")


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


@dataclass
class Config:
    d_model: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 4          # grouped-query attention: 4 KV heads feed 16 Q heads
    n_layers: int = 12
    d_ff: int = 2816
    vocab: int = 16384
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def qkv_out(self) -> int:
        return (self.n_heads + 2 * self.n_kv_heads) * self.head_dim

    def n_params(self) -> int:
        d, hd = self.d_model, self.head_dim
        per_layer = d * self.qkv_out + self.n_heads * hd * d + 2 * d * self.d_ff + self.d_ff * d
        return self.n_layers * per_layer + d * self.vocab

    def weight_bytes(self) -> int:
        return self.n_params() * 4

    def kv_bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * 4


class Engine:
    """A GPU-resident decoder-only transformer.  Random weights: this project
    measures *time*, and time does not depend on what the numbers are."""

    def __init__(self, cfg: Config, max_batch: int, max_seq: int, max_tokens: int | None = None):
        self.cfg = cfg
        self.B = max_batch
        self.S = max_seq
        self.max_tokens = max_tokens or max_batch * max_seq
        c = cfg
        g = torch.Generator().manual_seed(0)

        def w(*shape, scale=0.02):
            t = torch.empty(*shape, device=DEV)
            t.copy_((torch.randn(*shape, generator=g) * scale))
            return t

        def ones(n):
            t = torch.empty(n, device=DEV)
            t.copy_(torch.ones(n))
            return t

        self.w_qkv = [w(c.d_model, c.qkv_out) for _ in range(c.n_layers)]
        self.w_o = [w(c.n_heads * c.head_dim, c.d_model) for _ in range(c.n_layers)]
        self.w_gu = [w(c.d_model, 2 * c.d_ff) for _ in range(c.n_layers)]
        self.w_dn = [w(c.d_ff, c.d_model) for _ in range(c.n_layers)]
        self.n1 = [ones(c.d_model) for _ in range(c.n_layers)]
        self.n2 = [ones(c.d_model) for _ in range(c.n_layers)]
        self.nf = ones(c.d_model)
        self.w_lm = w(c.d_model, c.vocab)

        T = self.max_tokens
        self.x = torch.empty(T * c.d_model, device=DEV)
        self.xn = torch.empty(T * c.d_model, device=DEV)
        self.qkv = torch.empty(T * c.qkv_out, device=DEV)
        self.ao = torch.empty(T * c.n_heads * c.head_dim, device=DEV)
        self.pr = torch.empty(T * c.d_model, device=DEV)
        self.gu = torch.empty(T * 2 * c.d_ff, device=DEV)
        self.act = torch.empty(T * c.d_ff, device=DEV)
        self.logits = torch.empty(max_batch * c.vocab, device=DEV)
        self.tok = torch.empty(max_batch, device=DEV, dtype=torch.int32)
        self.kc = torch.empty(c.n_layers, max_batch * c.n_kv_heads * max_seq * c.head_dim, device=DEV)
        self.vc = torch.empty(c.n_layers, max_batch * c.n_kv_heads * max_seq * c.head_dim, device=DEV)
        self.seqlen = torch.empty(1, device=DEV, dtype=torch.int32)
        self.nsplit = 8
        self.ctx_hint = max_seq   # used only to label traced KV-read sizes
        self.part = torch.empty(max_batch * c.n_heads * 32 * c.head_dim, device=DEV)
        self.ml = torch.empty(max_batch * c.n_heads * 32 * 2, device=DEV)
        self._init_x()
        self.set_len(0)

    # -- helpers ---------------------------------------------------------
    def _init_x(self):
        g = torch.Generator().manual_seed(1)
        self.x.copy_(torch.randn(self.x.numel(), generator=g) * 0.5)

    def set_len(self, n: int) -> None:
        self.seqlen.copy_(torch.tensor([n], dtype=torch.int32))

    def bytes_weights(self) -> int:
        return self.cfg.weight_bytes()

    # -- one transformer layer ------------------------------------------
    def _layer(self, li: int, T: int, B: int, decode: bool, split: bool = True,
               nsplit: int | None = None):
        c = self.cfg
        rows = B * T
        hd, H, KVH = c.head_dim, c.n_heads, c.n_kv_heads
        half = hd // 2
        nb = rows * c.d_model * 4          # bytes of one activation buffer
        L = _launch
        L("rmsnorm", 3 * nb, k_rmsnorm, (rows,),
          self.x, self.n1[li], self.xn, c.d_model, 1e-5, BLOCK=c.d_model)
        _mm(self.xn, self.w_qkv[li], self.qkv, rows, c.qkv_out, c.d_model, decode,
            name="gemm_qkv")
        L("rope_q", 2 * rows * H * hd * 4, k_rope_q, (rows, H),
          self.qkv, self.seqlen, T, c.qkv_out, hd, BASE=c.rope_base, HALF=half)
        L("write_kv", 2 * rows * 2 * KVH * hd * 4, k_write_kv, (T, B, KVH),
          self.qkv, self.kc[li], self.vc[li], self.seqlen,
          T, B, KVH, hd, self.S, H * hd, c.qkv_out, c.rope_base, half)
        scale = 1.0 / math.sqrt(hd)
        if decode:
            ctx = self.ctx_hint
            kvb = B * KVH * ctx * hd * 4 * 2
            if split:
                ns = nsplit or self.nsplit
                L("attn_decode_split", kvb, k_attn_decode_split, (B * H, ns),
                  self.qkv, self.kc[li], self.vc[li], self.part, self.ml, self.seqlen,
                  self.S, H, KVH, hd, scale, QS=c.qkv_out, NSPLIT=ns, BN=32)
                L("attn_combine", B * H * ns * hd * 4, k_attn_combine, (B * H,),
                  self.part, self.ml, self.ao, H, hd, NSPLIT=ns)
            else:
                L("attn_decode", kvb, k_attn_decode, (B * H,),
                  self.qkv, self.kc[li], self.vc[li], self.ao, self.seqlen,
                  self.S, H, KVH, hd, scale, QS=c.qkv_out, BN=32)
        else:
            L("attn_prefill", 2 * B * KVH * T * hd * 4, k_attn_prefill,
              (triton.cdiv(T, 64), B * H),
              self.qkv, self.kc[li], self.vc[li], self.ao,
              T, self.S, H, KVH, hd, scale, QS=c.qkv_out, BM=64, BN=32)
        _mm(self.ao, self.w_o[li], self.pr, rows, c.d_model, H * hd, decode,
            name="gemm_o")
        n = rows * c.d_model
        L("residual", 3 * nb, k_add, (triton.cdiv(n, 1024),), self.x, self.pr, n, BLOCK=1024)
        L("rmsnorm", 3 * nb, k_rmsnorm, (rows,),
          self.x, self.n2[li], self.xn, c.d_model, 1e-5, BLOCK=c.d_model)
        _mm(self.xn, self.w_gu[li], self.gu, rows, 2 * c.d_ff, c.d_model, decode,
            name="gemm_gate_up")
        L("swiglu", 3 * rows * c.d_ff * 4, k_swiglu, (rows, triton.cdiv(c.d_ff, 512)),
          self.gu, self.act, rows, c.d_ff, BLOCK=512)
        _mm(self.act, self.w_dn[li], self.pr, rows, c.d_model, c.d_ff, decode,
            name="gemm_down")
        L("residual", 3 * nb, k_add, (triton.cdiv(n, 1024),), self.x, self.pr, n, BLOCK=1024)

    def prefill(self, B: int, T: int, head: bool = True) -> None:
        c = self.cfg
        self.set_len(0)
        for li in range(c.n_layers):
            self._layer(li, T, B, decode=False)
        if head:
            _launch("gather_last", 2 * B * c.d_model * 4, k_gather_last, (B,),
                    self.x, self.pr, T, c.d_model, BLOCK=c.d_model)
            _launch("rmsnorm_final", 3 * B * c.d_model * 4, k_rmsnorm, (B,),
                    self.pr, self.nf, self.xn, c.d_model, 1e-5, BLOCK=c.d_model)
            _mm(self.xn, self.w_lm, self.logits, B, c.vocab, c.d_model, False,
                name="gemm_lm_head")

    def decode_step(self, B: int, split: bool = True, nsplit: int | None = None,
                    head: bool = True, advance: bool = True) -> None:
        c = self.cfg
        for li in range(c.n_layers):
            self._layer(li, 1, B, decode=True, split=split, nsplit=nsplit)
        if head:
            _launch("rmsnorm_final", 3 * B * c.d_model * 4, k_rmsnorm, (B,),
                    self.x, self.nf, self.xn, c.d_model, 1e-5, BLOCK=c.d_model)
            _mm(self.xn, self.w_lm, self.logits, B, c.vocab, c.d_model, True,
                name="gemm_lm_head")
            _launch("argmax", B * c.vocab * 4, k_argmax, (B,),
                    self.logits, self.tok, c.vocab, BLOCK=1024)
        if advance:
            _launch("advance_pos", 8, k_incr, (1,), self.seqlen, 1)

    # -- accounting ------------------------------------------------------
    def decode_bytes(self, B: int, ctx: int) -> int:
        """Bytes that MUST cross HBM for one decode step: every weight once,
        plus every cached key and value once."""
        return self.cfg.weight_bytes() + B * ctx * self.cfg.kv_bytes_per_token()

    def decode_flops(self, B: int, ctx: int) -> int:
        c = self.cfg
        mm = 2 * B * c.n_params()
        attn = 2 * 2 * B * c.n_heads * c.head_dim * ctx * c.n_layers
        return mm + attn

    def prefill_flops(self, B: int, T: int) -> int:
        c = self.cfg
        mm = 2 * B * T * c.n_params()
        attn = 2 * 2 * B * c.n_heads * c.head_dim * T * T // 2 * c.n_layers
        return mm + attn

    def prefill_bytes(self, B: int, T: int) -> int:
        c = self.cfg
        acts = B * T * c.d_model * 4 * 10
        return c.weight_bytes() + acts + 2 * B * T * c.kv_bytes_per_token()

    def n_kernels_per_decode(self, split: bool = True, head: bool = True,
                             advance: bool = True) -> int:
        """rmsnorm, qkv-GEMM, rope-Q, write-KV, attention (+combine if split),
        o-GEMM, residual, rmsnorm, gate/up-GEMM, SwiGLU, down-GEMM, residual."""
        per_layer = 13 if split else 12
        return self.cfg.n_layers * per_layer + (3 if head else 0) + (1 if advance else 0)


# --------------------------------------------------------------------------
# per-kernel tracing (project 38's instrument -- there is no usable Nsight on
# this machine, so the engine times itself)
# --------------------------------------------------------------------------

_TRACE = None
_RECORD = None


def trace_start() -> None:
    """Start bracketing every kernel launch with a CUDA event pair.

    This is the in-situ measurement: it sees the kernels in the order and the
    cache state the real step produces.  It is also NOT free -- project 38
    measures how much the instrument distorts what it observes.
    """
    global _TRACE
    torch.cuda.synchronize()
    _TRACE = []


def trace_stop() -> list:
    """Return [(name, bytes, ms), ...] in launch order."""
    global _TRACE
    torch.cuda.synchronize()
    out = [(n, b, e0.elapsed_time(e1)) for n, b, e0, e1 in _TRACE]
    _TRACE = None
    return out


def record_start() -> None:
    """Capture the launch ARGUMENTS of every kernel, adding no timing at all.

    Replaying one recorded entry in a tight loop gives that kernel's cost with
    no observer overhead -- the other half of project 38's profiler.
    """
    global _RECORD
    _RECORD = []


def record_stop() -> list:
    """Return [(name, bytes, kernel, grid, args, kwargs), ...]."""
    global _RECORD
    out = _RECORD
    _RECORD = None
    return out


def _launch(name: str, nbytes: int, kernel, grid, *args, **kw):
    """Every kernel in the engine goes through here.

    With both hooks off this is two `is None` tests per launch.
    """
    if _RECORD is not None:
        _RECORD.append((name, nbytes, kernel, grid, args, kw))
    if _TRACE is None:
        return kernel[grid](*args, **kw)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    r = kernel[grid](*args, **kw)
    e1.record()
    _TRACE.append((name, nbytes, e0, e1))
    return r


def replay_launch(entry, inner: int = 20, reps: int = 8) -> float:
    """Milliseconds for one launch of a recorded kernel, timed in isolation.

    The `inner` repeats are captured into a CUDA graph and replayed, so the
    measurement contains no Python and no launch API at all -- otherwise a
    2 us kernel reads as 15 us, which is the cost of *issuing* it (project 38,
    section D).
    """
    _, _, kernel, grid, args, kw = entry
    g = Graph(lambda: [kernel[grid](*args, **kw) for _ in range(inner)], warmup=2)
    try:
        return gpu_time(g.replay, reps=reps) / inner
    finally:
        g.close()


def decode_tile(N: int, sms: int = 19):
    """Pick the output-tile width for a decode-shaped GEMM.

    With M = 1 the grid is one program per BN columns of output, so BN alone
    decides how many programs exist.  Too wide and there are fewer programs
    than the GPU has SMs, and most of the card sits idle -- measured at 55 GB/s
    for `o_proj` with 8 programs against 122 GB/s with 32.  The rule below
    keeps at least ~2 programs per SM, which is what an engine's autotuner
    converges to on its own.
    """
    for bn, bk in ((128, 32), (64, 64), (32, 128)):
        if N // bn >= 2 * sms:
            return 16, bn, bk
    return 16, 32, 128


def _mm(x, w, y, M, N, K, decode: bool, name: str = "gemm"):
    """Launch the GEMM with a tile shape suited to the phase.

    Decode has M as small as 1.  tl.dot cannot go below a 16-row tile, so the
    decode path pads M to 16 and throws away up to 94% of its arithmetic --
    which costs nothing, because at that shape the kernel is waiting on memory
    anyway.  Project 40 measures exactly this.
    """
    if decode or M <= 16:
        BM, BN, BK = decode_tile(N)
    else:
        BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    nbytes = (K * N + M * K + M * N) * 4
    _launch(name, nbytes, k_matmul, grid, x, w, y, M, N, K,
            BM=BM, BN=BN, BK=BK, GROUP_M=8)
