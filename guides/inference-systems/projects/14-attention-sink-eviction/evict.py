"""KV caches that throw tokens away to stay inside a memory budget.

Every policy here keeps at most `budget` tokens per layer. They differ only in
*which* tokens they keep:

  full            keep everything (the control -- no budget)
  recent          keep the last `budget` tokens (a sliding window)
  sink_recent     keep the first `n_sink` tokens plus the last (budget-n_sink)
                  -- this is StreamingLLM
  h2o             keep the highest accumulated-attention tokens plus a recent
                  window -- this is H2O
  h2o_sink        H2O, but the first `n_sink` tokens are never evictable
  random          keep a random subset (the control that stops us from
                  congratulating ourselves for merely keeping *some* tokens)

Vocabulary:

* **eviction** -- the word operating systems use for removing something from a
  cache to make room. Nothing is "deleted" in the sense of being wrong; it is
  removed because it is judged less useful than what needs the space.
* **attention sink** -- the first few tokens of a sequence. Softmax forces
  attention weights to sum to 1, so when a head has nothing it particularly
  wants to look at it still has to put that 1.0 *somewhere*; trained models
  learn to dump it on the first tokens. Those tokens therefore carry almost no
  information but absorb a large share of the weight, and removing them forces
  the leftover mass onto tokens that do carry information -- which corrupts
  the output far more than their contents would suggest.
* **heavy hitter** (the H2O paper's term, and the source of the "H2O" name:
  *Heavy-Hitter Oracle*) -- a token that has attracted a lot of attention so
  far, and is therefore predicted to keep doing so.
"""

from __future__ import annotations

import torch


class EvictingCache:
    """Per-layer eviction. Each layer keeps its own set of tokens, because
    each layer attends to a different mix -- an early layer tends to look
    locally, a late layer at a few distant tokens."""

    def __init__(self, n_layers, budget=256, policy="full", n_sink=4,
                 recent_frac=0.5, seed=0):
        self.n_layers, self.budget, self.policy = n_layers, budget, policy
        self.n_sink = n_sink
        self.n_recent = max(1, int(budget * recent_frac))
        self.gen = torch.Generator().manual_seed(seed)
        self.evicted = 0
        self.reset()

    def reset(self):
        L = self.n_layers
        self.k = [None] * L
        self.v = [None] * L
        self.pos = [None] * L      # absolute position of each stored row
        self.score = [None] * L    # accumulated attention, one per stored row
        self.evicted = 0

    # -- the seam the runner calls ------------------------------------------

    def append(self, layer, k, v):
        n_new = k.shape[2]
        start = 0 if self.pos[layer] is None else int(self.pos[layer][-1]) + 1
        new_pos = torch.arange(start, start + n_new)
        new_score = torch.zeros(n_new)
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
            self.pos[layer], self.score[layer] = new_pos, new_score
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
            self.pos[layer] = torch.cat([self.pos[layer], new_pos])
            self.score[layer] = torch.cat([self.score[layer], new_score])
        return self.k[layer], self.v[layer]

    def positions(self, layer):
        return self.pos[layer]

    def observe(self, layer, w):
        """Called by the runner with this layer's attention weights, right
        *after* attention has been computed. Eviction happens here, not in
        `append`, for two reasons:

        1. H2O needs to see the attention before it can know which tokens were
           heavy hitters. Evicting on the way in would be evicting blind.
        2. Evicting in the middle of a prompt's single forward pass would hide
           keys from queries that legitimately precede them -- an early query
           would find every remaining key masked out by causality and softmax
           over an all-masked row is NaN. Trimming after the pass avoids that
           entirely.

        w is (batch, n_heads, n_queries, n_keys). Summing over heads and
        queries gives "how much attention did each stored token receive during
        this pass", which is exactly the running total H2O keeps.
        """
        if self.policy == "full":
            return
        if self.policy in ("h2o", "h2o_sink"):
            s = w.detach().sum(dim=(0, 1, 2)).float()
            if (self.score[layer] is not None
                    and s.numel() == self.score[layer].numel()):
                self.score[layer] += s
        if self.k[layer].shape[2] > self.budget:
            self._evict(layer)

    # -- the policies --------------------------------------------------------

    def _evict(self, layer):
        n = self.k[layer].shape[2]
        keep = self._choose(layer, n)
        self.evicted += n - keep.numel()
        self.k[layer] = self.k[layer][:, :, keep]
        self.v[layer] = self.v[layer][:, :, keep]
        self.pos[layer] = self.pos[layer][keep]
        self.score[layer] = self.score[layer][keep]

    def _choose(self, layer, n) -> torch.Tensor:
        b, p = self.budget, self.policy
        if p == "recent":
            return torch.arange(n - b, n)
        if p == "sink_recent":
            return torch.cat([torch.arange(self.n_sink),
                              torch.arange(n - (b - self.n_sink), n)])
        if p == "random":
            perm = torch.randperm(n, generator=self.gen)[:b]
            return perm.sort().values
        if p in ("h2o", "h2o_sink"):
            recent = torch.arange(n - self.n_recent, n)
            forced = (torch.arange(self.n_sink) if p == "h2o_sink"
                      else torch.arange(0))
            protected = torch.cat([forced, recent])
            mask = torch.ones(n, dtype=torch.bool)
            mask[protected] = False
            room = b - protected.numel()
            cand = torch.nonzero(mask, as_tuple=False).flatten()
            if room > 0 and cand.numel() > room:
                top = self.score[layer][cand].topk(room).indices
                cand = cand[top]
            elif room <= 0:
                cand = torch.arange(0, dtype=torch.long)
            return torch.cat([protected, cand]).sort().values
        raise ValueError(p)

    # -- accounting ----------------------------------------------------------

    def n_tokens(self):
        return 0 if self.k[0] is None else self.k[0].shape[2]

    def nbytes(self):
        return sum(t.numel() * t.element_size()
                   for t in self.k + self.v if t is not None)
