"""Automatic prefix caching on top of project 11's block pool.

The trick is a *chained hash*. Cut the prompt into block-sized chunks and give
chunk i the hash `H(hash of chunks 0..i-1, tokens of chunk i)`. Two requests
land on the same hash only if every token before that point matched too, so a
hash hit is proof of a shared prefix -- no tree walk, one dictionary lookup
per block.

(SGLang's RadixAttention keeps an actual radix tree instead. The tree gives
better eviction decisions; the chained hash is easier to read and gives the
same hits, which is why vLLM's automatic prefix caching uses it.)
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "11-tiny-paged-cache"))

from paged import BlockPool, PagedCache  # noqa: E402


class PrefixCache:
    """hash -> physical block id, plus the reference that keeps it alive."""

    def __init__(self, pool: BlockPool):
        self.pool = pool
        self.index: dict[int, int] = {}
        self.hits = 0
        self.misses = 0
        self.hit_tokens = 0
        self.total_prompt_tokens = 0

    @staticmethod
    def chain(parent_hash: int, chunk: tuple) -> int:
        return hash((parent_hash, chunk))

    def match(self, token_ids):
        """Longest cached prefix, in whole blocks. Returns (block_ids, hashes).

        Only *full* blocks can be matched: a half-written block would keep
        being appended to, so its contents are not final and two requests
        cannot safely point at it.
        """
        bs = self.pool.block_size
        blocks, hashes, h = [], [], 0
        for i in range(len(token_ids) // bs):
            h = self.chain(h, tuple(token_ids[i * bs:(i + 1) * bs]))
            hashes.append(h)
            if h in self.index:
                blocks.append(self.index[h])
            else:
                break
        return blocks, hashes

    def acquire(self, token_ids, n_layers, enabled=True):
        """Give a request a cache, pre-loaded with whatever prefix already
        exists. Returns (cache, n_reused_tokens)."""
        self.total_prompt_tokens += len(token_ids)
        cache = PagedCache(self.pool, n_layers)
        if not enabled:
            self.misses += 1
            return cache, 0
        blocks, _ = self.match(token_ids)
        if not blocks:
            self.misses += 1
            return cache, 0
        for b in blocks:
            self.pool.incref(b)
        cache.block_table = list(blocks)
        cache.length = len(blocks) * self.pool.block_size
        cache.shared_prefix = cache.length
        cache._ensure_capacity(cache.length)
        self.hits += 1
        self.hit_tokens += cache.length
        return cache, cache.length

    def publish(self, token_ids, cache: PagedCache):
        """Register this request's full blocks so the next request can reuse
        them. Each published block gets one extra reference -- held by the
        index itself, not by any request -- so it survives the request."""
        _, hashes = self.match(token_ids)
        bs = self.pool.block_size
        h = 0
        for i in range(len(token_ids) // bs):
            if i >= len(cache.block_table):
                break
            h = self.chain(h, tuple(token_ids[i * bs:(i + 1) * bs]))
            if h not in self.index:
                self.index[h] = cache.block_table[i]
                self.pool.incref(cache.block_table[i])

    @property
    def hit_rate(self):
        n = self.hits + self.misses
        return self.hits / n if n else 0.0

    @property
    def token_hit_rate(self):
        return (self.hit_tokens / self.total_prompt_tokens
                if self.total_prompt_tokens else 0.0)
