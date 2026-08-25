"""Three ways to feed weights to a decode step, and what each one costs.

Apple Silicon's headline feature is *unified memory*: the CPU and the GPU
address one physical pool, so a tensor never has to be copied from one to the
other. This machine is the opposite arrangement -- a discrete GPU with its own
8 GB behind a PCIe 3.0 link -- which makes it the right place to measure what
unified memory removes, because here the copy is still there and can be timed.

The three paths:

  resident : weights already in GPU memory.        Limited by GPU DRAM.
  streamed : weights in host RAM, copied per use.  Limited by PCIe.
  cpu      : weights in host RAM, computed by CPU. Limited by CPU DRAM.

A unified-memory machine is a fourth: weights in one pool, computed by the
GPU, no copy -- so it is limited by that pool's bandwidth and nothing else.
"""

import time

import torch
import triton
import triton.language as tl


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


def gpu_matvec(W, x, y, K):
    N = W.numel() // K
    _matvec[(triton.cdiv(N, 128),)](W, x, y, N, K, BLOCK_N=128, BLOCK_K=64,
                                    num_warps=4)


# --------------------------------------------------------------- the paths
def path_resident(layers, x, y, K, reps=3):
    """Every layer already on the GPU. This is what fits-in-VRAM looks like."""
    for _ in range(2):
        for W in layers:
            gpu_matvec(W, x, y, K)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        for W in layers:
            gpu_matvec(W, x, y, K)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def path_streamed(host_layers, stage, x, y, K, reps=3, overlap=True):
    """Weights live in pinned host RAM and cross PCIe once per use.

    With `overlap`, the copy of layer i+1 runs on a second stream while layer
    i is being computed, which is the best a discrete GPU can do. Without it,
    copy and compute are serialised, which is what naive offloading does.
    """
    cp = torch.cuda.Stream()
    ev = [torch.cuda.Event() for _ in stage]

    def once():
        if not overlap:
            for i, hW in enumerate(host_layers):
                buf = stage[i % len(stage)]
                buf.copy_(hW, non_blocking=False)
                gpu_matvec(buf, x, y, K)
            return
        # prime the pipeline with the first layer
        with torch.cuda.stream(cp):
            stage[0].copy_(host_layers[0], non_blocking=True)
            ev[0].record(cp)
        for i in range(len(host_layers)):
            s = i % len(stage)
            if i + 1 < len(host_layers):
                n = (i + 1) % len(stage)
                with torch.cuda.stream(cp):
                    stage[n].copy_(host_layers[i + 1], non_blocking=True)
                    ev[n].record(cp)
            torch.cuda.current_stream().wait_event(ev[s])
            gpu_matvec(stage[s], x, y, K)

    once()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        once()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def path_cpu(host_layers, xc, reps=3, rounds=3):
    """No accelerator at all: read the weights from RAM, multiply on the CPU.

    Best-of-`rounds`, because this is the one path that competes with every
    other process on the machine for the same DRAM.
    """
    for W in host_layers[:2]:
        torch.mv(W, xc)
    out = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(reps):
            for W in host_layers:
                torch.mv(W, xc)
        out.append((time.perf_counter() - t0) / reps)
    return min(out)
