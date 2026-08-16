"""Accounting for padding waste in a static-batching server.

A *pad token* is a token position the accelerator computes even though no user
asked for it. There are two entirely different sources, and lumping them
together is the usual reason a padding audit comes out wrong:

  **Prompt padding** (prefill). A batch of prompts of lengths [12, 40, 190] is
  stored as a rectangle 190 wide, so 3x190 - 242 = 328 positions are filler.
  This waste is paid *once* per request.

  **Generation padding** (decode). A static batch keeps stepping until its
  longest member is finished. A request that wanted 5 tokens sitting next to
  one that wants 70 rides along for 65 extra steps. This waste is paid *every
  step*, which is why it usually dominates even though it looks smaller in the
  length histogram.

Both are counted here in two currencies:

  * **slots** -- raw token-positions. Easy to explain, but misleading: a pad
    slot late in a long sequence attends to far more keys than an early one.
  * **FLOPs** -- slots weighted by the arithmetic they actually trigger, using
    the same `flops_per_token(ctx)` model the engine uses.

Reporting only slots understates the damage; reporting only FLOPs hides how
many requests were affected. So this module reports both.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Waste:
    real_slots: int = 0
    pad_slots: int = 0
    real_flops: float = 0.0
    pad_flops: float = 0.0

    def __add__(self, o):
        return Waste(self.real_slots + o.real_slots, self.pad_slots + o.pad_slots,
                     self.real_flops + o.real_flops, self.pad_flops + o.pad_flops)

    @property
    def slot_frac(self):
        t = self.real_slots + self.pad_slots
        return self.pad_slots / t if t else 0.0

    @property
    def flop_frac(self):
        t = self.real_flops + self.pad_flops
        return self.pad_flops / t if t else 0.0


def audit_group(group, fpt):
    """Account one static batch. `group` is a list of (prompt_len, out_len);
    `fpt(ctx)` returns the FLOPs of one token position at context `ctx`."""
    pmax = max(p for p, _ in group)
    omax = max(o for _, o in group)
    pre, dec = Waste(), Waste()

    # -- prefill: the rectangle is len(group) x pmax --------------------------
    for p, _ in group:
        for j in range(pmax):
            f = fpt(j + 1)
            if j < p:
                pre.real_slots += 1
                pre.real_flops += f
            else:
                pre.pad_slots += 1
                pre.pad_flops += f

    # -- decode: omax-1 steps, every row present in every one -----------------
    for p, o in group:
        for s in range(omax - 1):
            ctx = p + s + 1
            f = fpt(ctx)
            if s < o - 1:
                dec.real_slots += 1
                dec.real_flops += f
            else:
                dec.pad_slots += 1
                dec.pad_flops += f
    return pre, dec


SORT_KEYS = {
    "arrival": None,
    "prompt": lambda x: x[0],          # known at admission -- implementable
    "output": lambda x: x[1],          # NOT known: an oracle
    "both": lambda x: x[0] + x[1],     # also an oracle
}


def audit_trace(lens, batch_size, fpt, sort="arrival", window=None):
    """Chop a trace into static batches and total the waste.

    Sorting first is *length bucketing*: put requests of similar size in the
    same batch so the rectangle is tighter. Two honesty rules are built into
    the parameters, because a bucketing result is easy to overstate:

      `sort="output"` sorts by a number the server **cannot know** -- how long
      the answer will be. It is an oracle, reported only as an upper bound.
      `sort="prompt"` sorts by a number it *can* know, so that row is the one
      you could actually ship.

      `window` limits how far ahead the sorter may look. `None` means "sort
      the entire trace", which assumes every request of the day has already
      arrived. A real server sorts inside a small window of what is queued
      right now, and pays a wait to fill even that.
    """
    key = SORT_KEYS[sort]
    items = list(lens)
    if key is not None:
        if window is None:
            items.sort(key=key)
        else:
            items = [x for i in range(0, len(items), window)
                     for x in sorted(items[i:i + window], key=key)]
    pre, dec = Waste(), Waste()
    for i in range(0, len(items), batch_size):
        g = items[i:i + batch_size]
        a, b = audit_group(g, fpt)
        pre, dec = pre + a, dec + b
    return pre, dec
