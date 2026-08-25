"""Topology as a graph, and the one number NCCL cares about.

`nvidia-smi topo -m` prints a matrix of link *types*. To predict performance you
need link *bandwidths*, and then the answer to one question: what is the
slowest link on the ring NCCL will build?

A ring visits every GPU exactly once and comes back to the start (a Hamiltonian
cycle). Its bandwidth is the bandwidth of its slowest hop, because a ring runs
in lockstep -- every rank sends one chunk per step, so the step takes as long as
the worst link in it. Finding the best ring is therefore "maximise the minimum
edge over all cycles", which for 8 GPUs is 2520 candidate cycles: brute force.
"""

from __future__ import annotations

from itertools import permutations

# Link-type -> one-direction bandwidth in GB/s. The NV# entries are per NVLink
# lane count; the rest are the fabric the traffic falls back to.
LINK_GBS = {
    "X": float("inf"),      # self
    "NV1": 25.0,            # one NVLink 3 lane group
    "NV2": 50.0,
    "NV4": 100.0,
    "NV12": 300.0,          # A100 to NVSwitch
    "NV18": 450.0,          # H100 to NVSwitch
    "PIX": 16.0,            # same PCIe switch, Gen4 x16 one direction ~ 25 GB/s peak
    "PXB": 14.0,            # several PCIe switches
    "PHB": 12.0,            # up through the host bridge
    "NODE": 10.0,           # across host bridges inside one NUMA node
    "SYS": 6.0,             # across the CPU-CPU interconnect
}


class Topology:
    def __init__(self, name: str, matrix: list[list[str]], note: str = ""):
        self.name = name
        self.matrix = matrix
        self.n = len(matrix)
        self.note = note

    def bw(self, i: int, j: int) -> float:
        return LINK_GBS[self.matrix[i][j]]

    def best_ring(self):
        """Return (order, bottleneck_GBs) for the ring with the fastest slowest
        hop. Ranks are relabelled so rank 0 is always first -- a cycle has no
        preferred starting point, so fixing it removes n duplicate answers."""
        best, best_bw = None, -1.0
        for perm in permutations(range(1, self.n)):
            order = (0,) + perm
            bottleneck = min(self.bw(order[k], order[(k + 1) % self.n])
                             for k in range(self.n))
            if bottleneck > best_bw:
                best, best_bw = order, bottleneck
        return list(best), best_bw

    def worst_ring(self):
        best, worst_bw = None, float("inf")
        for perm in permutations(range(1, self.n)):
            order = (0,) + perm
            bottleneck = min(self.bw(order[k], order[(k + 1) % self.n])
                             for k in range(self.n))
            if bottleneck < worst_bw:
                best, worst_bw = order, bottleneck
        return list(best), worst_bw

    def predicted_busbw(self) -> float:
        """A ring all-reduce's bus bandwidth is the ring's bottleneck: every
        rank sends 2(n-1)/n * N bytes through its outgoing link, and busbw is
        defined exactly so that it equals that link's rate."""
        return self.best_ring()[1]

    def predicted_allreduce_s(self, nbytes: float) -> float:
        n = self.n
        return (2 * (n - 1) / n) * nbytes / (self.predicted_busbw() * 1e9)


def uniform(n: int, kind: str, name: str, note: str = "") -> Topology:
    m = [[("X" if i == j else kind) for j in range(n)] for i in range(n)]
    return Topology(name, m, note)


def dgx1_cube_mesh() -> Topology:
    """DGX-1 with V100: each GPU has 6 NVLink lanes, wired as two 4-GPU rings
    (0-1-2-3 and 4-5-6-7) plus cross links. Pairs with no NVLink at all must go
    over PCIe, which is the interesting part."""
    nv2 = [(0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7),
           (4, 5), (4, 7), (5, 6), (6, 7)]
    nv1 = [(0, 2), (1, 3), (4, 6), (5, 7)]
    m = [["SYS" if (i // 4) != (j // 4) else "PHB" for j in range(8)] for i in range(8)]
    for i in range(8):
        m[i][i] = "X"
    for a, b in nv2:
        m[a][b] = m[b][a] = "NV2"
    for a, b in nv1:
        m[a][b] = m[b][a] = "NV1"
    return Topology("DGX-1 (8xV100, hybrid cube mesh)", m,
                    "NVLink where it exists, PCIe/QPI where it does not")


def parse_nvidia_smi_topo(text: str):
    """Turn the real `nvidia-smi topo -m` output into (labels, matrix)."""
    rows, labels = [], []
    for line in text.splitlines():
        parts = line.split()
        if not parts or not parts[0].startswith("GPU"):
            continue
        if not any(p in LINK_GBS for p in parts[1:]):
            continue
        labels.append(parts[0])
        rows.append([p for p in parts[1:] if p in LINK_GBS])
    n = len(labels)
    return labels, [r[:n] for r in rows]
