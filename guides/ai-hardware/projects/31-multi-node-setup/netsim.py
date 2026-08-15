"""Two all-reduce layouts for a 2-node cluster, plus a deliberately slow
inter-node link.

The link is emulated: before every message that would cross between nodes, the
sender sleeps for `latency + bytes/bandwidth`. Everything else -- the ranks, the
messages, the reduction, the wall clock -- is real. That is enough to answer the
question this project is about, which is not "how fast is my cable" but "does
the *shape* of the algorithm change when one link is slower than the others".

  flat_ring     one ring over all n ranks; 2 of its n links cross between nodes
                and every one of the 2(n-1) steps uses them.
  hierarchical  reduce-scatter inside each node -> all-reduce between the node
                leaders on 1/g of the data -> all-gather inside each node.
                Only the middle stage crosses, and it carries 1/g of the bytes.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist


class Link:
    """Charges for a message if it crosses a node boundary."""

    def __init__(self, ranks_per_node: int, latency_s: float, bw_GBs: float):
        self.rpn = ranks_per_node
        self.latency = latency_s
        self.bw = bw_GBs * 1e9
        self.crossings = 0
        self.crossed_bytes = 0

    def node_of(self, rank: int) -> int:
        return rank // self.rpn

    def charge(self, me: int, peer: int, nbytes: int):
        if self.node_of(me) == self.node_of(peer) or self.bw <= 0:
            return
        self.crossings += 1
        self.crossed_bytes += nbytes
        time.sleep(self.latency + nbytes / self.bw)


def _xchg(send_t, recv_t, send_peer, recv_peer, link, me):
    """Send to one neighbour while receiving from another. In a ring those are
    two *different* ranks; using the same rank for both deadlocks, because the
    neighbour you are waiting on is busy sending to somebody else."""
    link.charge(me, send_peer, send_t.numel() * send_t.element_size())
    reqs = [dist.isend(send_t.contiguous(), send_peer), dist.irecv(recv_t, recv_peer)]
    for r in reqs:
        r.wait()


def flat_ring(x: torch.Tensor, rank: int, world: int, link: Link) -> torch.Tensor:
    chunks = list(x.chunk(world))
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    for step in range(world - 1):
        s, r = (rank - step) % world, (rank - step - 1) % world
        recv = torch.empty_like(chunks[r])
        _xchg(chunks[s], recv, nxt, prv, link, rank)
        chunks[r] += recv
    for step in range(world - 1):
        s, r = (rank - step + 1) % world, (rank - step) % world
        recv = torch.empty_like(chunks[r])
        _xchg(chunks[s], recv, nxt, prv, link, rank)
        chunks[r].copy_(recv)
    return x


def hierarchical(x: torch.Tensor, rank: int, world: int, link: Link) -> torch.Tensor:
    g = link.rpn                      # ranks per node
    node = rank // g
    local = rank % g
    nodes = world // g

    # 1. reduce-scatter inside the node: each local rank ends up owning 1/g
    #    of the buffer, fully summed over this node's ranks.
    chunks = list(x.chunk(g))
    base = node * g
    nxt, prv = base + (local + 1) % g, base + (local - 1) % g
    for step in range(g - 1):
        s, r = (local - step) % g, (local - step - 1) % g
        recv = torch.empty_like(chunks[r])
        _xchg(chunks[s], recv, nxt, prv, link, rank)
        chunks[r] += recv
    # After g-1 reduce-scatter steps the chunk this rank has finished summing is
    # (local - (g-1)) mod g == (local + 1) mod g -- NOT chunk `local`. Getting
    # this index wrong sends the wrong slice across the node boundary and the
    # result is silently a partial sum (it cost this project one debug cycle).
    own = (local + 1) % g
    mine = chunks[own]

    # 2. all-reduce that piece across nodes -- the only stage that crosses, and
    #    it carries 1/g of the data.
    if nodes > 1:
        orig = mine.clone()      # everyone must send its pre-exchange value,
        for peer_node in range(nodes):   # or a third node would be added twice
            if peer_node == node:
                continue
            peer = peer_node * g + local
            recv = torch.empty_like(mine)
            _xchg(orig, recv, peer, peer, link, rank)
            mine += recv

    # 3. all-gather inside the node
    for step in range(g - 1):
        s, r = (local - step + 1) % g, (local - step) % g
        recv = torch.empty_like(chunks[r])
        _xchg(chunks[s], recv, nxt, prv, link, rank)
        chunks[r].copy_(recv)
    return x


def model_time(alg: str, world: int, g: int, nbytes: int,
               intra_lat, intra_bw_GBs, inter_lat, inter_bw_GBs) -> float:
    """The alpha-beta prediction, for comparison with the stopwatch."""
    a_in, b_in = intra_lat, 1.0 / (intra_bw_GBs * 1e9)
    a_out, b_out = inter_lat, 1.0 / (inter_bw_GBs * 1e9)
    if alg == "flat_ring":
        steps = 2 * (world - 1)
        per = nbytes / world
        cross = 2 / world                      # fraction of ring links that cross
        return steps * ((1 - cross) * (a_in + per * b_in) + cross * (a_out + per * b_out))
    if alg == "hierarchical":
        nodes = world // g
        intra = 2 * (g - 1) * (a_in + nbytes / g * b_in)
        inter = (nodes - 1) * (a_out + nbytes / g * b_out)
        return intra + inter
    raise ValueError(alg)
