"""A network-on-chip simulator for a 2D mesh of cores.

Tenstorrent's Tensix grid has no shared cache and no crossbar. Every core has
its own local SRAM, and data gets from core A to core B by being *routed*
across a 2D mesh of links, hop by hop, like packets on a small network. That
is what "network-on-chip" means literally, and it has one consequence that
does not exist on a GPU: **where you place a piece of work changes how fast it
runs**, because placement changes how far its data has to travel and which
links it shares with everyone else.

This file models the part that decides throughput: per-link load. Routing is
dimension-order XY (go along the row first, then along the column), which is
what almost every real mesh uses because it is deadlock-free and needs no
routing tables.

The bottleneck link -- the one carrying the most bytes -- sets the time for
the whole step, because every link runs at the same speed and the step is not
finished until the slowest one is.
"""


class Mesh:
    def __init__(self, rows=10, cols=8, link_bytes_per_cycle=32):
        self.rows = rows
        self.cols = cols
        self.bpc = link_bytes_per_cycle

    def route(self, src, dst):
        """The links an XY route uses, as a list of (from_core, to_core)."""
        (r0, c0), (r1, c1) = src, dst
        path, cur = [], (r0, c0)
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1, step):                 # X first
            nxt = (r0, c + step)
            path.append((cur, nxt))
            cur = nxt
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1, step):                 # then Y
            nxt = (r + step, c1)
            path.append((cur, nxt))
            cur = nxt
        return path

    def load(self, flows):
        """flows: [(src, dst, bytes)] -> per-link byte counts."""
        link = {}
        for src, dst, nbytes in flows:
            for e in self.route(src, dst):
                link[e] = link.get(e, 0) + nbytes
        return link

    def cost(self, flows):
        link = self.load(flows)
        if not link:
            return dict(bottleneck_bytes=0, cycles=0, links_used=0,
                        total_hops=0, total_bytes=0)
        worst = max(link.values())
        hops = sum(len(self.route(s, d)) for s, d, _ in flows)
        return dict(bottleneck_bytes=worst,
                    cycles=int(-(-worst // self.bpc)),
                    links_used=len(link),
                    total_hops=hops,
                    total_bytes=sum(b for _, _, b in flows))


# ------------------------------------------------------------- placements
def snake(mesh, n):
    """Consecutive stages on adjacent cores: row 0 left-to-right, row 1
    right-to-left, and so on. Every hand-off is then exactly one hop."""
    out = []
    for i in range(n):
        r = i // mesh.cols
        c = i % mesh.cols
        if r % 2:
            c = mesh.cols - 1 - c
        out.append((r % mesh.rows, c))
    return out


def rowmajor(mesh, n):
    """Row-major without the reversal: the jump back to column 0 at the end
    of each row costs a full row's width."""
    return [((i // mesh.cols) % mesh.rows, i % mesh.cols) for i in range(n)]


def columnmajor(mesh, n):
    return [(i % mesh.rows, (i // mesh.rows) % mesh.cols) for i in range(n)]


def scattered(mesh, n, seed=0):
    """A deterministic pseudo-random placement, as a control: it is what you
    get if the compiler does not think about placement at all."""
    s = seed
    cores = [(r, c) for r in range(mesh.rows) for c in range(mesh.cols)]
    out = []
    for _ in range(n):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(cores[s % len(cores)])
    return out


# ---------------------------------------------------------------- patterns
def pipeline_flows(placement, act_bytes):
    """Stage i sends its activations to stage i+1."""
    return [(placement[i], placement[i + 1], act_bytes)
            for i in range(len(placement) - 1)]


def gather_flows(placement, dst, nbytes):
    """Every core sends to one core: an all-reduce's final step, or a
    softmax denominator being collected. The classic NoC hotspot."""
    return [(p, dst, nbytes) for p in placement if p != dst]
