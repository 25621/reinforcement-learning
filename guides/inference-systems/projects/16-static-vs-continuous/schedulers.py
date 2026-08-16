"""The two schedulers project 16 compares, plus the priority variants used by
project 20.

Both run against the *same* `BatchedRunner` and the *same* traffic trace, so
every difference in the numbers comes from the scheduling decision and nothing
else. Both also run in **virtual time**: the clock advances by the measured
duration of each forward pass rather than by `time.time()`, so a request that
arrives at t=3.2 s is admitted when the engine's own work has taken 3.2 s.
Real wall-clock would have made the result depend on how loaded this shared
machine happened to be during the run.
"""

from __future__ import annotations

import torch

from batchlib import SlotKV


class Clock:
    """Virtual time. `now` only moves when the model does work."""

    def __init__(self):
        self.now = 0.0

    def advance(self, dt):
        self.now += dt


def _new_pool(runner, n_slots, max_len):
    return SlotKV(runner.n_layers, n_slots, runner.n_kv_heads,
                  runner.d_head, max_len)


def _greedy(logits):
    return logits.argmax(-1).tolist()


def run_static(runner, reqs, batch_size=8, max_len=320, stop_eos=False):
    """Static batching: fill a batch, pad it, run it to completion, repeat.

    Three costs are baked into this shape and all three are measurable:
      1. **Arrival wait** -- request 1 of a batch waits for request N to show
         up before anything starts.
      2. **Prompt padding** -- every prompt is padded to the longest in the
         batch, and the GPU computes those pad tokens for real.
      3. **Generation padding** -- the batch keeps stepping until its longest
         output finishes; rows that hit their limit early keep riding along.
    """
    clock = Clock()
    pool = _new_pool(runner, batch_size, max_len)
    runner.counters.__init__()
    pending = sorted(reqs, key=lambda r: r.arrive)
    i = 0
    while i < len(pending):
        group = pending[i:i + batch_size]
        i += len(group)
        # 1. the batch cannot start before its last member arrives
        clock.now = max(clock.now, group[-1].arrive)
        for j, r in enumerate(group):
            r.slot = j
            r.admit_t = clock.now

        # 2. padded prefill
        pmax = max(r.prompt_len for r in group)
        ids = torch.zeros(len(group), pmax, dtype=torch.long)
        for j, r in enumerate(group):
            ids[j, :r.prompt_len] = torch.tensor(r.prompt_ids)
        lens = [r.prompt_len for r in group]
        logits, dt = runner.prefill(pool, list(range(len(group))), ids, lens)
        clock.advance(dt)
        nxt = _greedy(logits)
        for j, r in enumerate(group):
            r.tokens.append(nxt[j])
            r.first_tok_t = clock.now
            r.token_times.append(clock.now)

        # 3. decode until the LONGEST row is done; short rows keep stepping
        cur_len = [r.prompt_len for r in group]
        steps = max(r.max_new for r in group) - 1
        for s in range(steps):
            live = [len(r.tokens) < r.max_new for r in group]
            if not any(live):
                break
            logits, dt = runner.decode_step(
                pool, list(range(len(group))), nxt, cur_len, live=live)
            clock.advance(dt)
            nxt = _greedy(logits)
            for j, r in enumerate(group):
                cur_len[j] += 1
                if live[j]:
                    r.tokens.append(nxt[j])
                    r.token_times.append(clock.now)
        # 4. everyone leaves together -- that is the definition of static
        for r in group:
            r.end_t = clock.now
        pool.free = list(range(batch_size))
    return clock.now


def run_continuous(runner, reqs, n_slots=8, max_len=320, priority=False,
                   aging=0.0, chunk=None):
    """Continuous (iteration-level) batching.

    Every iteration the scheduler looks at the world again: it admits whatever
    has arrived and fits, prefills at most one new request, decodes everyone
    else, and frees the slot of anyone who finished. Nobody waits for anybody.

    `priority` / `aging` are project 20's knobs; `chunk` is project 18's.
    """
    clock = Clock()
    pool = _new_pool(runner, n_slots, max_len)
    runner.counters.__init__()
    incoming = sorted(reqs, key=lambda r: r.arrive)
    queue, running = [], []
    nxt_idx = 0

    def admit_key(r):
        if not priority:
            return (r.arrive,)
        # A pure priority order starves the low class forever. `aging` lets a
        # waiting request gain effective priority the longer it waits, which is
        # the standard cure -- and project 20 measures what the cure costs.
        eff = r.priority - aging * max(0.0, clock.now - r.arrive)
        return (eff, r.arrive)

    while nxt_idx < len(incoming) or queue or running:
        # bring in everything that has arrived by now
        while nxt_idx < len(incoming) and incoming[nxt_idx].arrive <= clock.now:
            queue.append(incoming[nxt_idx])
            nxt_idx += 1
        if not queue and not running:
            clock.now = incoming[nxt_idx].arrive
            continue

        # --- prefill at most one waiting request, if a slot is free ---------
        if queue and pool.n_free() > 0:
            queue.sort(key=admit_key)
            r = queue.pop(0)
            r.slot = pool.acquire()
            r.admit_t = clock.now
            ids = torch.tensor(r.prompt_ids).view(1, -1)
            if chunk is None or r.prompt_len <= chunk:
                logits, dt = runner.prefill(pool, [r.slot], ids, [r.prompt_len])
                clock.advance(dt)
            else:
                # chunked prefill: several iterations, so decode is not starved
                for s in range(0, r.prompt_len, chunk):
                    part = ids[:, s:s + chunk]
                    logits, dt = runner.prefill(
                        pool, [r.slot], part, [r.prompt_len], start=s)
                    clock.advance(dt)
            r.tokens.append(_greedy(logits)[0])
            r.first_tok_t = clock.now
            r.token_times.append(clock.now)
            r.cur_len = r.prompt_len
            running.append(r)
            continue                       # a prefill iteration does no decode

        if not running:
            if nxt_idx < len(incoming):
                clock.now = max(clock.now, incoming[nxt_idx].arrive)
                continue
            break

        # --- decode every running request, one token each -------------------
        slots = [r.slot for r in running]
        toks = [r.tokens[-1] for r in running]
        lens = [r.cur_len for r in running]
        logits, dt = runner.decode_step(pool, slots, toks, lens)
        clock.advance(dt)
        nxt = _greedy(logits)
        finished = []
        for j, r in enumerate(running):
            r.cur_len += 1
            r.tokens.append(nxt[j])
            r.token_times.append(clock.now)
            if len(r.tokens) >= r.max_new:
                finished.append(r)
        # --- free finished slots IMMEDIATELY: this is the whole point -------
        for r in finished:
            r.end_t = clock.now
            pool.release(r.slot)
            running.remove(r)
    return clock.now
