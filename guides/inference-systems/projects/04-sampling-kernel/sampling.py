"""Sampling as logit transforms: temperature, repetition penalty, top-k,
top-p, min-p — plus two ways to draw the token.

Every function takes and returns a (batch, vocab) float tensor of logits, so
they compose in whatever order the server decides. Order matters and is a
policy choice; the order used here matches vLLM and HF:

    repetition penalty -> temperature -> top-k -> top-p -> min-p -> draw
"""

from __future__ import annotations

import torch

NEG_INF = float("-inf")


def apply_repetition_penalty(logits, prev_ids, penalty: float):
    """Divide the logit of already-seen tokens by `penalty` (if positive).

    Negative logits are *multiplied* instead, so the penalty always pushes a
    score down rather than accidentally raising it. That asymmetry is the
    original CTRL-paper definition, and it is why the parameter is not simply
    a subtraction.
    """
    if penalty == 1.0:
        return logits
    out = logits.clone()
    for b in range(logits.shape[0]):
        ids = prev_ids[b]
        sel = out[b, ids]
        out[b, ids] = torch.where(sel > 0, sel / penalty, sel * penalty)
    return out


def apply_temperature(logits, temperature: float):
    """Divide every logit by T. T<1 sharpens, T>1 flattens, T=0 means greedy."""
    if temperature == 1.0:
        return logits
    if temperature <= 0.0:
        return logits           # caller takes argmax; nothing to scale
    return logits / temperature


def top_k_filter(logits, k: int):
    """Keep the k largest logits per row, set the rest to -inf."""
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < kth, NEG_INF)


def top_p_filter_sort(logits, p: float):
    """Nucleus sampling by sorting the WHOLE vocabulary. The obvious version."""
    if p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cum = probs.cumsum(dim=-1)
    # Keep tokens up to and including the one that crosses p.
    remove = cum - probs > p
    remove[:, 0] = False
    mask = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
    return logits.masked_fill(mask, NEG_INF)


def top_p_filter_prefilter(logits, p: float, k: int = 1024):
    """Same result as the full sort, but only k candidates are ranked.

    The saving is the sort: `topk` over 151,936 values is far cheaper than
    `sort`. The softmax is still taken over the WHOLE row -- see
    top_p_filter_prefilter_renorm for what happens if you skip that.

    Exact whenever the nucleus is smaller than k; `nucleus_size` measures how
    often that holds, and it depends entirely on the temperature.
    """
    if p >= 1.0:
        return logits
    k = min(k, logits.shape[-1])
    probs = torch.softmax(logits, dim=-1)          # O(V), no sort
    pv, idx = torch.topk(probs, k, dim=-1)
    cum = pv.cumsum(dim=-1)
    remove = cum - pv > p
    remove[:, 0] = False
    out = torch.full_like(logits, NEG_INF)
    kept = torch.gather(logits, 1, idx).masked_fill(remove, NEG_INF)
    out.scatter_(1, idx, kept)
    return out


def top_p_filter_prefilter_renorm(logits, p: float, k: int = 1024):
    """The tempting shortcut: softmax over the top k only. Subtly wrong.

    Taking the softmax of just the k best logits renormalises them to sum to
    1, so every probability comes out slightly too big and the cumulative sum
    reaches p slightly too early -- the nucleus is cut one or two tokens
    short. The error is tiny and never raises anything; it just quietly
    changes which tokens can be sampled.
    """
    if p >= 1.0:
        return logits
    k = min(k, logits.shape[-1])
    vals, idx = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(vals, dim=-1)            # <- renormalised over k only
    cum = probs.cumsum(dim=-1)
    remove = cum - probs > p
    remove[:, 0] = False
    out = torch.full_like(logits, NEG_INF)
    out.scatter_(1, idx, vals.masked_fill(remove, NEG_INF))
    return out


def min_p_filter(logits, min_p: float):
    """Keep tokens whose probability is at least `min_p` x the best token's.

    Unlike top-p it needs no sort: it is a single comparison against the max,
    which is why it is cheap. The shortlist widens automatically when the
    model is unsure (the top probability is low) and collapses when it is
    confident -- the same goal as top-p, reached without cumulative sums.
    """
    if min_p <= 0.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    thresh = probs.max(dim=-1, keepdim=True).values * min_p
    return logits.masked_fill(probs < thresh, NEG_INF)


def nucleus_size(logits, p: float):
    """How many tokens survive top-p, per row. Diagnostic, not part of sampling."""
    probs = torch.softmax(logits, dim=-1)
    s, _ = torch.sort(probs, descending=True, dim=-1)
    cum = s.cumsum(dim=-1)
    return (cum - s <= p).sum(dim=-1)


def draw_multinomial(logits, generator=None):
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator)


def draw_gumbel(logits, generator=None):
    """Gumbel-max trick: argmax(logits + Gumbel noise) ~ softmax(logits).

    Named after Emil Julius Gumbel, who studied the distribution of extreme
    values (maxima). The trick needs no softmax, no cumulative sum and no
    sort -- one elementwise add and one argmax -- which is why fused sampling
    kernels use it.
    """
    u = torch.rand(logits.shape, generator=generator, device=logits.device)
    # Clamp AWAY FROM BOTH ENDS. log(0) is -inf and log(1) is 0, and a naive
    # clamp_min on the already-negative inner log silently returns 1e-20 for
    # every value -- which produces a plausible-looking but completely wrong
    # distribution (measured: 0.377 total-variation distance, section B).
    u = u.clamp(1e-20, 1.0 - 1e-7)
    g = -torch.log(-torch.log(u))
    return (logits + g).argmax(dim=-1, keepdim=True)


def sample(logits, *, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
           prev_ids=None, repetition_penalty=1.0, prefilter=True,
           generator=None):
    """The whole pipeline, in the order a server applies it."""
    if prev_ids is not None:
        logits = apply_repetition_penalty(logits, prev_ids, repetition_penalty)
    if temperature <= 0.0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = apply_temperature(logits, temperature)
    logits = top_k_filter(logits, top_k)
    logits = (top_p_filter_prefilter if prefilter else top_p_filter_sort)(
        logits, top_p)
    logits = min_p_filter(logits, min_p)
    return draw_multinomial(logits, generator=generator)
