"""Shared GPU helpers for the Triton projects (18, 19, 20, 21, 22).

This machine has an unusual split that shapes every Triton project in the
phase:

  * PyTorch's own CUDA kernels DO NOT RUN here. The installed build targets
    sm_75 and newer; this card is sm_61 (Pascal), so `a + b` on two CUDA
    tensors raises `no kernel image is available for execution on the device`,
    and PyTorch's bundled cuBLAS 13 refuses the card outright with
    `CUBLAS_STATUS_ARCH_MISMATCH`.
  * TRITON DOES RUN. Triton compiles its own PTX for whatever architecture it
    finds, and sm_61 is still a supported target for the pieces we need,
    including `tl.dot`.

So the rules for this phase are:

  * allocate with `torch.empty(..., device="cuda")` (a plain `cudaMalloc`,
    no kernel involved),
  * fill by copying from the CPU with `.cuda()` (a `cudaMemcpy`),
  * compute with Triton,
  * check answers on the CPU with `.cpu()`.

`randn`, `empty`, `zeros` below wrap exactly that. Everything else is timing.
"""

import torch
import triton
import triton.language as tl


# --------------------------------------------------------------- allocation

def randn(*shape, seed=None, scale=1.0):
    """Normal data made on the CPU and copied over (no GPU kernel involved)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    return (torch.randn(*shape, generator=g) * scale).cuda()


def empty(*shape, dtype=torch.float32):
    return torch.empty(*shape, dtype=dtype, device="cuda")


def zeros(*shape, dtype=torch.float32):
    return torch.zeros(*shape, dtype=dtype).cuda()


def ones(*shape, dtype=torch.float32):
    return torch.ones(*shape, dtype=dtype).cuda()


def device_note():
    p = torch.cuda.get_device_properties(0)
    return dict(name=p.name, cc="%d.%d" % (p.major, p.minor),
                sms=p.multi_processor_count,
                mem_gb=round(p.total_memory / 2**30, 2))


def torch_eager_works():
    """Does plain PyTorch arithmetic run on this GPU? (Here: no.)"""
    try:
        a = torch.ones(8).cuda()
        (a + a).cpu()
        return True, ""
    except Exception as e:
        return False, str(e).split("\n")[0]


def torch_cublas_works():
    try:
        a = torch.ones(8, 8).cuda()
        (a @ a).cpu()
        return True, ""
    except Exception as e:
        return False, str(e).split("\n")[0]


# --------------------------------------------------------------- warm-up

@triton.jit
def _spin_kernel(SINK, ITERS):
    off = tl.arange(0, 128)
    x = off.to(tl.float32) * 1e-6
    for _ in range(ITERS):
        x = x * 1.0000001 + 1e-7
    # store unconditionally: a masked-off store lets the compiler delete the
    # whole loop, and then the "warm-up" heats nothing at all
    tl.store(SINK + off, x)


_sink = None


def warm_up(rounds=60, iters=200000):
    """Raise the clocks before timing anything.

    An idle GPU sits at ~164 MHz and boosts to ~1.7 GHz only under load, so a
    cold measurement reads ~35% slow. One short kernel is not enough heat, and
    the card cools again during any stretch of CPU-side work (building a
    reference answer, compiling the next kernel), so this is called before
    every measurement rather than once at start-up.
    """
    global _sink
    if _sink is None:
        _sink = empty(128)
    for _ in range(rounds):
        _spin_kernel[(152,)](_sink, iters, num_warps=4)
    torch.cuda.synchronize()


# --------------------------------------------------------------- timing

def bench(fn, reps=50, warmup=10):
    """Median-of-3 timing of `fn` with CUDA events, in milliseconds.

    Events are timestamps recorded inside the GPU's own work queue, so they
    measure the GPU and not the host's queueing. A launch is asynchronous:
    timing it with a host clock and no synchronise reports the launch, which
    is ~200x too fast (see project 16).
    """
    warm_up(rounds=20)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(3):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b) / reps)
    times.sort()
    return times[1]


def gbs(nbytes, ms):
    return nbytes / (ms * 1e6)


def tflops(nflop, ms):
    return nflop / (ms * 1e9)


# --------------------------------------------------------------- card facts
# Measured in phase 1-3, reused here instead of being re-derived.
PEAK_FLOPS_FP32 = 8190.0     # GFLOP/s
PEAK_BW = 256.3              # GB/s, spec
BW_READ = 222.0              # GB/s, measured read-only (project 3)
BW_COPY = 201.0              # GB/s, measured 1-read-1-write (project 3)
