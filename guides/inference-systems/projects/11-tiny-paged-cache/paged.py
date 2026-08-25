"""A block-paginated KV cache -- the vLLM PagedAttention data structure, small.

The idea is borrowed from operating systems. A process does not get one
contiguous slab of RAM; it gets fixed-size *pages* scattered anywhere in
physical memory, plus a *page table* mapping "page 3 of my address space" to
"physical frame 917". Here a sequence does not get one contiguous KV slab; it
gets fixed-size *blocks* (16 tokens each) scattered anywhere in the pool, plus
a *block table* mapping "block 3 of my sequence" to "physical block 917".

Written for project 11; extended with reference counting so project 12 can
make two sequences share the blocks of a common prefix.
"""

from __future__ import annotations

import torch


class BlockPool:
    """The physical memory. One big tensor per layer, cut into fixed blocks.

    Layout is (n_blocks, block_size, n_kv_heads, d_head) so that the first two
    axes flatten into a single "slot" index -- slot = block_id * block_size +
    offset. That flat view is what makes writes and gathers one vectorised
    call instead of a Python loop over blocks.
    """

    def __init__(self, n_blocks, block_size, n_layers, n_kv_heads, d_head,
                 dtype=torch.float32):
        self.n_blocks, self.block_size = n_blocks, block_size
        self.n_layers, self.n_kv_heads, self.d_head = n_layers, n_kv_heads, d_head
        self.dtype = dtype
        shape = (n_layers, n_blocks, block_size, n_kv_heads, d_head)
        self.k = torch.zeros(shape, dtype=dtype)
        self.v = torch.zeros(shape, dtype=dtype)
        self.kf = self.k.view(n_layers, n_blocks * block_size, n_kv_heads, d_head)
        self.vf = self.v.view(n_layers, n_blocks * block_size, n_kv_heads, d_head)
        self.free = list(range(n_blocks - 1, -1, -1))   # pop() from the end
        self.ref = [0] * n_blocks
        self.peak_used = 0
        self.n_allocs = 0

    def alloc(self) -> int:
        if not self.free:
            raise MemoryError("KV block pool exhausted")
        b = self.free.pop()
        self.ref[b] = 1
        self.n_allocs += 1
        self.peak_used = max(self.peak_used, self.n_blocks - len(self.free))
        return b

    def incref(self, b: int):
        """Another sequence now points at this block (prefix sharing)."""
        self.ref[b] += 1

    def decref(self, b: int):
        self.ref[b] -= 1
        if self.ref[b] == 0:
            self.free.append(b)

    @property
    def used(self) -> int:
        return self.n_blocks - len(self.free)

    def nbytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class PagedCache:
    """One sequence's view of the pool. Batch size 1, which is all our
    single-request engine needs (real engines carry a batch of block tables
    and hand all of them to one kernel).
    """

    def __init__(self, pool: BlockPool, n_layers: int):
        self.pool = pool
        self.n_layers = n_layers
        self.block_table: list[int] = []
        self.length = 0
        self.shared_prefix = 0       # tokens inherited from another sequence
        self._slots = torch.zeros(0, dtype=torch.long)

    # -- block management ----------------------------------------------------

    def _ensure_capacity(self, n_tokens: int):
        bs = self.pool.block_size
        need = (n_tokens + bs - 1) // bs
        while len(self.block_table) < need:
            self.block_table.append(self.pool.alloc())
        want = len(self.block_table) * bs
        if self._slots.numel() < want:
            base = torch.tensor(self.block_table, dtype=torch.long) * bs
            self._slots = (base.view(-1, 1)
                           + torch.arange(bs, dtype=torch.long).view(1, -1)).reshape(-1)

    def free(self):
        for b in self.block_table:
            self.pool.decref(b)
        self.block_table, self.length = [], 0
        self._slots = torch.zeros(0, dtype=torch.long)

    # -- the KVCache interface ----------------------------------------------

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        # k, v: (1, n_kv_heads, t, d_head)
        t = k.shape[2]
        if layer == 0:
            self._ensure_capacity(self.length + t)
        start = self.length
        slots = self._slots[start:start + t]
        self.pool.kf[layer].index_copy_(0, slots, k[0].transpose(0, 1))
        self.pool.vf[layer].index_copy_(0, slots, v[0].transpose(0, 1))
        end = start + t
        got = self._slots[:end]
        k_all = self.pool.kf[layer][got].transpose(0, 1).unsqueeze(0)
        v_all = self.pool.vf[layer][got].transpose(0, 1).unsqueeze(0)
        if layer == self.n_layers - 1:
            self.length = end
        return k_all, v_all

    def positions(self, layer: int):
        return None

    def reset(self):
        self.free()

    # -- accounting ----------------------------------------------------------

    def blocks_used(self) -> int:
        return len(self.block_table)

    def internal_waste_tokens(self) -> int:
        """Slots reserved but unused inside the last (partial) block.

        Called *internal* fragmentation because the waste is inside an
        allocation, as opposed to *external* fragmentation, which is the
        unusable gaps between allocations."""
        return len(self.block_table) * self.pool.block_size - self.length


def fork(pool: BlockPool, parent: PagedCache, n_shared_tokens: int) -> PagedCache:
    """A child sequence that shares the parent's first `n_shared_tokens`.

    Only whole blocks can be shared -- a block that is half prefix and half
    child-specific would be written by both. So the sharing is rounded *down*
    to a block boundary, which is the reason production engines pick small
    block sizes for prefix-heavy workloads.
    """
    bs = pool.block_size
    n_full = n_shared_tokens // bs
    child = PagedCache(pool, parent.n_layers)
    child.block_table = parent.block_table[:n_full]
    for b in child.block_table:
        pool.incref(b)
    child.length = n_full * bs
    child.shared_prefix = child.length
    child._ensure_capacity(child.length)
    return child
