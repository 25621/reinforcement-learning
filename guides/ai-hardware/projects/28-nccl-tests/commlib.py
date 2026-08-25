"""Shared multi-process communication harness for Phase 6 (projects 28-32).

Everything here runs on `gloo` over TCP loopback, because this machine has one
GPU that PyTorch cannot launch kernels on (sm_61) and therefore no usable NCCL.
The *methodology* is exactly nccl-tests': time a collective, convert to
algorithm bandwidth, then to bus bandwidth with the collective's correction
factor.

Rules learned the hard way:
  * only JSON-able objects go through mp.Queue (tensors raise EOFError),
  * every worker pins itself to one thread so N ranks do not fight over the
    same 12 cores in a way that hides the communication we are measuring,
  * every timing is preceded by a barrier and reported as the max over ranks,
    which is what a collective actually costs.
"""

from __future__ import annotations

import json
import os
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# ---------------------------------------------------------------- launching

def _free_port() -> int:
    """Ask the kernel for an unused port instead of hard-coding one. A crashed
    previous run can leave its rendezvous port in TIME_WAIT for a minute, and a
    fixed port would then refuse to start with EADDRINUSE."""
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank: int, world: int, port: int, threads: int, fn, args, q):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["GLOO_SOCKET_IFNAME"] = os.environ.get("GLOO_SOCKET_IFNAME", "lo")
    torch.set_num_threads(threads)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        out = fn(rank, world, *args)
        if rank == 0:
            q.put(json.dumps(out))
    finally:
        dist.destroy_process_group()


def run_ranks(fn, world: int, *args, threads: int = 1, timeout: float = 900.0):
    """Spawn `world` processes running fn(rank, world, *args); return rank 0's result."""
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_entry, args=(r, world, port, threads, fn, args, q))
        for r in range(world)
    ]
    for p in procs:
        p.start()
    payload = q.get(timeout=timeout)
    for p in procs:
        p.join(timeout=timeout)
    for p in procs:
        if p.exitcode not in (0, None):
            raise RuntimeError(f"rank died with exitcode {p.exitcode}")
    return json.loads(payload)


# ---------------------------------------------------------------- timing

def _median(v):
    v = sorted(v)
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


def timed(op, reps: int, warmup: int = 2, rounds: int = 5) -> float:
    """Seconds per call: median over `rounds` independent batches of `reps`
    calls, then the max over ranks (a collective is finished only when its
    slowest participant is finished).

    The median matters because this box is shared with a desktop session; a
    single mean is one stray process away from being wrong.
    """
    return timed_many({"op": op}, reps, warmup=warmup, rounds=rounds)["op"]


def timed_many(ops: dict, reps: int, warmup: int = 2, rounds: int = 5) -> dict:
    """Time several alternatives *interleaved*: one round times each of them
    once, and we take per-alternative medians across rounds. Timing them one
    after another instead would let a slow minute land entirely on one
    alternative and be reported as its cost."""
    for op in ops.values():
        for _ in range(warmup):
            op()
    samples = {k: [] for k in ops}
    for _ in range(rounds):
        for k, op in ops.items():
            dist.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                op()
            samples[k].append((time.perf_counter() - t0) / reps)
    out = {}
    for k in ops:
        buf = torch.tensor([_median(samples[k])])
        dist.all_reduce(buf, op=dist.ReduceOp.MAX)
        out[k] = float(buf.item())
    return out


# ------------------------------------------------- nccl-tests bandwidth math
#
# algbw  = message_bytes / time          "how fast did my data move"
# busbw  = algbw * factor(n)             "how hard did the slowest link work"
#
# The factor exists because different collectives push a different number of
# bytes over the wire per byte of user data. See the project README, section E.

def bus_factor(collective: str, n: int) -> float:
    if collective == "all_reduce":
        return 2.0 * (n - 1) / n
    if collective in ("all_gather", "reduce_scatter"):
        return (n - 1) / n
    if collective in ("broadcast", "reduce"):
        return 1.0
    raise ValueError(collective)


def bandwidths(collective: str, count_bytes: int, seconds: float, n: int):
    algbw = count_bytes / seconds / 1e9
    return algbw, algbw * bus_factor(collective, n)


# ---------------------------------------------------------------- misc

def reps_for(nbytes: int) -> int:
    """Fewer repetitions for big messages so a full sweep stays inside a minute."""
    if nbytes <= 64 * 1024:
        return 30
    if nbytes <= 1024 * 1024:
        return 12
    if nbytes <= 8 * 1024 * 1024:
        return 6
    return 3


def alpha_beta(sizes_bytes, seconds):
    """The classic  T(n) = alpha + n/BW  model, read straight off the data
    instead of least-squares-fitted:

      alpha = the time of the smallest message (all fixed cost, no payload)
      BW    = the *incremental* bandwidth of the two largest messages,
              (n2-n1)/(t2-t1), which cancels the fixed cost by construction.

    A least-squares fit over a log-spaced sweep is dominated by the biggest
    point and can return a negative alpha, which is not a thing. This reading
    cannot.
    """
    pairs = sorted(zip(sizes_bytes, seconds))
    alpha = pairs[0][1]
    (n1, t1), (n2, t2) = pairs[-2], pairs[-1]
    bw = (n2 - n1) / (t2 - t1)
    return alpha, bw
