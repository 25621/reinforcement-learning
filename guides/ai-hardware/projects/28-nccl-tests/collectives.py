"""Three all-reduce algorithms written by hand on top of point-to-point sends.

They compute exactly the same answer and differ only in *how the bytes move*:

  ring               2(n-1) steps, each carrying N/n bytes  -> bandwidth-optimal
  recursive_doubling log2(n) steps, each carrying N bytes    -> latency-optimal
  flat               2 steps, but rank 0 carries (n-1)N      -> bottlenecked

Comparing them against gloo's built-in all_reduce is the point of section D.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def _wait(reqs):
    for r in reqs:
        r.wait()


def ring_all_reduce(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Reduce-scatter around the ring, then all-gather around the ring."""
    if world == 1:
        return x
    chunks = list(x.chunk(world))
    nxt, prv = (rank + 1) % world, (rank - 1) % world

    # Phase 1: reduce-scatter. After n-1 steps rank r owns the finished chunk r.
    for step in range(world - 1):
        s = (rank - step) % world
        r = (rank - step - 1) % world
        recv = torch.empty_like(chunks[r])
        _wait([dist.isend(chunks[s].contiguous(), nxt), dist.irecv(recv, prv)])
        chunks[r] += recv

    # Phase 2: all-gather. Circulate the finished chunks once around.
    for step in range(world - 1):
        s = (rank - step + 1) % world
        r = (rank - step) % world
        recv = torch.empty_like(chunks[r])
        _wait([dist.isend(chunks[s].contiguous(), nxt), dist.irecv(recv, prv)])
        chunks[r].copy_(recv)
    return x


def recursive_doubling_all_reduce(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Pair up with a partner at distance 1, 2, 4 ... and swap the whole buffer."""
    assert world & (world - 1) == 0, "recursive doubling needs a power-of-two world"
    step = 1
    while step < world:
        partner = rank ^ step
        recv = torch.empty_like(x)
        _wait([dist.isend(x.contiguous(), partner), dist.irecv(recv, partner)])
        x += recv
        step <<= 1
    return x


def flat_all_reduce(x: torch.Tensor, rank: int, world: int) -> torch.Tensor:
    """Everyone -> rank 0, rank 0 sums, rank 0 -> everyone. Two hops, one victim."""
    if world == 1:
        return x
    if rank == 0:
        bufs = [torch.empty_like(x) for _ in range(world - 1)]
        _wait([dist.irecv(bufs[p - 1], p) for p in range(1, world)])
        for b in bufs:
            x += b
        _wait([dist.isend(x, p) for p in range(1, world)])
    else:
        _wait([dist.isend(x.contiguous(), 0)])
        _wait([dist.irecv(x, 0)])
    return x


ALGORITHMS = {
    "ring": ring_all_reduce,
    "recursive_doubling": recursive_doubling_all_reduce,
    "flat": flat_all_reduce,
}


def steps_and_bytes(alg: str, world: int, nbytes: int):
    """Analytic cost of each algorithm: (latency steps, bytes each rank sends)."""
    if alg == "ring":
        return 2 * (world - 1), 2 * (world - 1) / world * nbytes
    if alg == "recursive_doubling":
        k = world.bit_length() - 1
        return k, k * nbytes
    if alg == "flat":
        return 2, nbytes if world > 1 else 0
    raise ValueError(alg)
