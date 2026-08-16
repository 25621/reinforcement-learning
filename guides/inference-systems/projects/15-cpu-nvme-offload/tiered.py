"""A two-tier KV cache: fast tier + slow tier, with real I/O.

On a GPU box the tiers are HBM -> host RAM -> NVMe. This machine has no usable
GPU, so the same structure is measured one level down: process RAM is the fast
tier and the NVMe SSD is the slow tier. The *ratios* between tiers are what
the design depends on, and they are measured here rather than assumed.

Vocabulary:

* **offload** -- move data out of the expensive tier to a cheaper one, with
  the intention of bringing it back. Different from eviction (project 14),
  which throws data away and accepts the quality loss.
* **cold** -- not touched recently. A cold block is the safe one to move.
* **page cache** -- the operating system keeps recently-read file data in RAM.
  Read a file you just wrote and you are measuring RAM, not the disk. Every
  disk number here drops the page cache first (`posix_fadvise(DONTNEED)`), so
  the numbers are the real thing.
"""

from __future__ import annotations

import os
import tempfile
import time

import torch


def drop_page_cache(fd: int, length: int):
    """Tell the kernel to forget this file's cached pages, so the next read
    actually goes to the device."""
    try:
        os.posix_fadvise(fd, 0, length, os.POSIX_FADV_DONTNEED)
        return True
    except (AttributeError, OSError):
        return False


class Tier2:
    """The slow tier. `backend` is "ram" (a dict of tensors, standing in for
    pinned host memory) or "nvme" (one file per block)."""

    def __init__(self, backend="nvme", root=None):
        self.backend = backend
        self.store = {}
        self.dir = root or tempfile.mkdtemp(prefix="kvoffload-")
        self.write_s = 0.0
        self.read_s = 0.0
        self.bytes_out = 0
        self.bytes_in = 0

    def put(self, key, k, v):
        t0 = time.perf_counter()
        nb = (k.numel() + v.numel()) * k.element_size()
        if self.backend == "ram":
            self.store[key] = (k.clone(), v.clone())
        else:
            path = os.path.join(self.dir, f"{key}.bin")
            buf = torch.stack([k, v]).contiguous()
            with open(path, "wb") as fh:
                fh.write(buf.numpy().tobytes())
                fh.flush()
                os.fsync(fh.fileno())
            self.store[key] = (path, tuple(k.shape), k.dtype)
        self.write_s += time.perf_counter() - t0
        self.bytes_out += nb
        return nb

    def get(self, key, cold=True):
        t0 = time.perf_counter()
        if self.backend == "ram":
            k, v = self.store[key]
            k, v = k.clone(), v.clone()
        else:
            path, shape, dtype = self.store[key]
            n = os.path.getsize(path)
            fd = os.open(path, os.O_RDONLY)
            try:
                if cold:
                    drop_page_cache(fd, n)
                raw = os.read(fd, n)
            finally:
                os.close(fd)
            buf = torch.frombuffer(bytearray(raw), dtype=dtype).view(2, *shape)
            k, v = buf[0].clone(), buf[1].clone()
        dt = time.perf_counter() - t0
        self.read_s += dt
        self.bytes_in += (k.numel() + v.numel()) * k.element_size()
        return k, v, dt

    def cleanup(self):
        if self.backend == "nvme":
            for val in self.store.values():
                try:
                    os.remove(val[0])
                except OSError:
                    pass


class TieredCache:
    """Contiguous per-layer cache split into fixed token blocks, with the
    coldest blocks pushed to tier 2 when the resident budget is exceeded.

    Implements project 09's `KVCache` interface, so the same runner produces
    real text through it.
    """

    def __init__(self, n_layers, block=64, resident_blocks=8, backend="nvme",
                 cold_reads=True):
        self.n_layers, self.block = n_layers, block
        self.resident_blocks = resident_blocks
        self.cold_reads = cold_reads
        self.t2 = Tier2(backend)
        self.reset()

    def reset(self):
        L = self.n_layers
        self.blocks = [[] for _ in range(L)]     # list of (k, v) or None
        self.tail_k = [None] * L                 # partially filled last block
        self.tail_v = [None] * L
        self.lru = [[] for _ in range(L)]        # block indices, oldest first
        self.fetches = 0
        self.fetch_s = 0.0
        self.evictions = 0
        # A KV block never changes after the token that produced it is
        # finished -- attention only ever reads it. So a block that has been
        # written to tier 2 once is still valid the next time it is pushed
        # out, and the write can be skipped entirely. Only the reads are
        # unavoidable.
        self.on_disk = set()
        self.writes_skipped = 0

    # -- KVCache interface ---------------------------------------------------

    def append(self, layer, k, v):
        tk, tv = self.tail_k[layer], self.tail_v[layer]
        tk = k if tk is None else torch.cat([tk, k], dim=2)
        tv = v if tv is None else torch.cat([tv, v], dim=2)
        while tk.shape[2] >= self.block:
            self.blocks[layer].append((tk[:, :, :self.block].contiguous(),
                                       tv[:, :, :self.block].contiguous()))
            self.lru[layer].append(len(self.blocks[layer]) - 1)
            tk, tv = tk[:, :, self.block:], tv[:, :, self.block:]
            self._maybe_offload(layer)
        self.tail_k[layer], self.tail_v[layer] = tk, tv
        return self._gather(layer)

    def positions(self, layer):
        return None

    # -- tiering -------------------------------------------------------------

    def _maybe_offload(self, layer):
        """Push out the least-recently-used blocks until we are inside the
        budget. LRU because the alternative -- evicting the newest -- would
        immediately have to fetch it back."""
        resident = [i for i in self.lru[layer]]
        while len(resident) > self.resident_blocks:
            idx = resident.pop(0)
            k, v = self.blocks[layer][idx]
            if (layer, idx) in self.on_disk:
                self.writes_skipped += 1
            else:
                self.t2.put((layer, idx), k, v)
                self.on_disk.add((layer, idx))
            self.blocks[layer][idx] = None
            self.lru[layer].remove(idx)
            self.evictions += 1

    def _gather(self, layer):
        """Attention needs every token, so any offloaded block has to come
        back before this layer can run. That is the whole problem with
        offloading an *active* sequence, and section B measures it."""
        ks, vs = [], []
        for idx, blk in enumerate(self.blocks[layer]):
            if blk is None:
                k, v, dt = self.t2.get((layer, idx), cold=self.cold_reads)
                self.fetches += 1
                self.fetch_s += dt
                self.blocks[layer][idx] = (k, v)
                self.lru[layer].append(idx)
                blk = (k, v)
            ks.append(blk[0])
            vs.append(blk[1])
        if self.tail_k[layer] is not None and self.tail_k[layer].shape[2]:
            ks.append(self.tail_k[layer])
            vs.append(self.tail_v[layer])
        # Re-apply the budget after the fetches, or the cache grows without
        # bound the moment anything is read back.
        self._maybe_offload(layer)
        return torch.cat(ks, dim=2), torch.cat(vs, dim=2)

    # -- accounting ----------------------------------------------------------

    def resident_bytes(self):
        tot = 0
        for layer in range(self.n_layers):
            for blk in self.blocks[layer]:
                if blk is not None:
                    tot += sum(t.numel() * t.element_size() for t in blk)
            if self.tail_k[layer] is not None:
                tot += sum(t.numel() * t.element_size()
                           for t in (self.tail_k[layer], self.tail_v[layer]))
        return tot
