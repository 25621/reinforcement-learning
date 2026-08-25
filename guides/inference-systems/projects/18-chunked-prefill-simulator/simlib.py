"""A discrete-event simulator of an iteration-level inference scheduler.

Why simulate at all, when project 16 already runs a real model? Because the
questions in projects 18, 20, 21 and 22 need *thousands* of requests with
32,000-token prompts and hour-long tails, and this machine can serve about
twelve tokens a second. Simulation is the only way to reach the regime where
scheduling decisions separate.

The trade is stated plainly so you can judge the results: the simulator gets
its **timing** from real measurements (`CostModel.fit` fits its coefficients to
timings taken from project 16's engine) and its **logic** from real engines,
but it does not model kernel launch, memory allocation, or anything the GPU
does that is not a linear function of the tokens in the batch.

One iteration = one forward pass. Its contents are chosen by the scheduler:

    iteration = [ decode: one token for each running request ]
              + [ prefill: a chunk of tokens for one waiting request ]

Shared by projects 18, 20, 21 and 22.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# cost model
# ---------------------------------------------------------------------------


@dataclass
class CostModel:
    """Seconds for one forward pass.

        t = base + per_decode * n_decode_rows + per_prefill * n_prefill_tokens
            + per_ctx_token * total_attended_context

    The first three terms are the weight traffic and the projections, which
    scale with *rows*. The fourth is attention, which scales with how much KV
    each row has to read -- the term that makes a long-context batch slower
    than a short-context batch of the same width.
    """

    base: float = 0.080
    per_decode: float = 0.008
    per_prefill: float = 0.0018
    per_key_read: float = 1.5e-6

    def iter_time(self, n_decode, n_prefill_tokens, key_reads):
        """`key_reads` is how many (query, key) pairs attention touches in this
        pass: for a decode row, its context length; for a prefill chunk of `c`
        tokens starting at offset `s`, `c * (s + c/2)`. Attention is the only
        part of a forward pass that is not linear in the number of tokens, so
        it needs its own term -- without it the model says a 16k-token prefill
        costs 16x a 1k one, and the measurement says 25x."""
        return (self.base + self.per_decode * n_decode
                + self.per_prefill * n_prefill_tokens
                + self.per_key_read * key_reads)

    @staticmethod
    def fit(decode_pts, prefill_pts, decode_ctx):
        """Fit from (batch, seconds) at fixed context and (tokens, seconds).

        Done in three steps rather than one joint solve, because at a fixed
        context the decode batch size and the key-read count are the *same
        number in disguise* (rows x ctx) and a joint least-squares would have
        no way to tell the two coefficients apart.

          1. prefill points give the quadratic: t = i + b*T + c*T(T+1)/2
          2. decode points give t = i_d + a*B at ctx = `decode_ctx`,
             so per_decode = a - c*decode_ctx
          3. base = i_d
        """
        def solve(A, y):
            n = len(A[0])
            M = [[sum(A[r][i] * A[r][j] for r in range(len(A))) for j in range(n)]
                 + [sum(A[r][i] * y[r] for r in range(len(A)))] for i in range(n)]
            for col in range(n):                        # Gaussian elimination
                piv = max(range(col, n), key=lambda r: abs(M[r][col]))
                M[col], M[piv] = M[piv], M[col]
                d = M[col][col]
                M[col] = [v / d for v in M[col]]
                for r in range(n):
                    if r != col and M[r][col]:
                        f = M[r][col]
                        M[r] = [v - f * w for v, w in zip(M[r], M[col])]
            return [M[i][n] for i in range(n)]

        A = [[1.0, float(t), t * (t + 1) / 2.0] for t, _ in prefill_pts]
        ip, b, c = solve(A, [s for _, s in prefill_pts])
        c = max(c, 0.0)
        A = [[1.0, float(bs)] for bs, _ in decode_pts]
        i_d, a = solve(A, [s for _, s in decode_pts])
        return CostModel(base=max(i_d, 0.0),
                         per_decode=max(a - c * decode_ctx, 1e-6),
                         per_prefill=max(b, 1e-6), per_key_read=c)

    def prefill_keys(self, start, n):
        """Key-reads for a prefill chunk of `n` tokens starting at `start`."""
        return n * (start + (n + 1) / 2.0)

    def request_work(self, prompt_len, out_len):
        """Roughly how many seconds of engine time one request costs, used to
        pick an arrival rate that hits a chosen load."""
        return (self.per_prefill * prompt_len
                + self.per_key_read * self.prefill_keys(0, prompt_len)
                + out_len * (self.per_decode
                             + self.per_key_read * (prompt_len + out_len / 2)))


# ---------------------------------------------------------------------------
# requests
# ---------------------------------------------------------------------------


@dataclass
class SimRequest:
    rid: int
    arrive: float
    prompt_len: int
    out_len: int
    priority: int = 0            # lower = more urgent (project 20)
    deadline: float = 0.0        # absolute time (project 22)

    # runtime
    prefilled: int = 0
    generated: int = 0
    admit_t: float | None = None
    first_t: float | None = None
    end_t: float | None = None
    rejected: bool = False
    preemptions: int = 0
    _last_tok_t: float | None = None
    itls: list = field(default_factory=list)

    @property
    def ttft(self):
        return None if self.first_t is None else self.first_t - self.arrive

    @property
    def e2e(self):
        return None if self.end_t is None else self.end_t - self.arrive

    @property
    def ctx(self):
        return self.prefilled + self.generated


def make_trace(n=2000, rate=2.0, seed=0, p_med=600, p_sigma=1.1, p_max=32768,
               o_med=180, o_sigma=0.8, o_max=2048):
    """Poisson arrivals, lognormal prompt and output lengths.

    Defaults chosen to look like a RAG/chat mix: median prompt 600 tokens with
    a fat tail of pasted documents reaching the model's 32k limit. The tail is
    the entire subject of project 18 -- with a symmetric length distribution
    chunked prefill would have nothing to fix.
    """
    rng = random.Random(seed)
    out, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        p = int(min(p_max, max(16, rng.lognormvariate(math.log(p_med), p_sigma))))
        o = int(min(o_max, max(8, rng.lognormvariate(math.log(o_med), o_sigma))))
        out.append(SimRequest(rid=i, arrive=t, prompt_len=p, out_len=o))
    return out


# ---------------------------------------------------------------------------
# the scheduler loop
# ---------------------------------------------------------------------------


def simulate(reqs, cost: CostModel, *, chunk=None, token_budget=2048,
             max_running=256, kv_capacity=None, order="fcfs", aging=0.0,
             admission=None, preempt=True, preempt_for_priority=False,
             overcommit=False, reserve=0, crash_on_overflow=False,
             max_iters=4_000_000):
    """Run one scheduling policy over one trace. Mutates the requests.

    Parameters that matter per project:
      `chunk`       -- project 18. None means "a prefill takes the whole
                       iteration to itself", the pre-Sarathi behaviour.
      `order`       -- 'fcfs' | 'priority' | 'edf' | 'sjf' | 'least_slack'
      `aging`       -- priority credit per second waited (project 20)
      `kv_capacity` -- KV budget in tokens; None = unlimited (project 21)
      `admission`   -- callable(req, now, kv_used, kv_capacity) returning
                       "admit" | "wait" | "reject" (projects 21 and 22)
      `preempt`     -- when the cache overflows, evict the newest running
                       request back to the queue (vLLM's recompute preemption)
    """
    incoming = sorted(reqs, key=lambda r: r.arrive)
    nxt = 0
    queue, prefilling, running = [], [], []
    now = 0.0
    kv_used = 0
    stats = {"iters": 0, "prefill_tokens": 0, "decode_tokens": 0,
             "preemptions": 0, "rejected": 0, "busy_s": 0.0,
             "wasted_decode_tokens": 0, "wasted_prefill_tokens": 0,
             "peak_kv": 0, "overflow_iters": 0, "oom": False, "oom_at_s": None,
             "oom_kv": 0, "oom_inflight": 0}

    def key(r):
        if order == "priority":
            return (r.priority - aging * max(0.0, now - r.arrive), r.arrive)
        if order == "edf":
            return (r.deadline, r.arrive)
        if order == "sjf":
            return (r.prompt_len + r.out_len, r.arrive)
        if order == "least_slack":
            # slack = time left before the deadline minus the work still owed
            return (r.deadline - now - r.out_len * cost.per_decode, r.arrive)
        return (r.arrive,)

    while stats["iters"] < max_iters:
        while nxt < len(incoming) and incoming[nxt].arrive <= now:
            queue.append(incoming[nxt])
            nxt += 1
        if not queue and not prefilling and not running:
            if nxt >= len(incoming):
                break
            now = incoming[nxt].arrive
            continue

        # ---- priority preemption -------------------------------------------
        # Reordering the *queue* only helps a request that is still waiting.
        # Once bronze requests fill every slot, a gold request waits behind
        # them no matter how the queue is sorted -- so a scheduler that really
        # wants to protect gold has to throw a bronze request out. That costs
        # the victim everything it has generated so far, which is why engines
        # do this reluctantly and why project 20 measures the bill.
        if preempt_for_priority and queue and running and \
                len(running) + len(prefilling) >= max_running:
            queue.sort(key=key)
            worst = max(running, key=lambda r: r.priority)
            if queue[0].priority < worst.priority:
                running.remove(worst)
                kv_used -= worst.ctx
                worst.preemptions += 1
                stats["wasted_decode_tokens"] += worst.generated
                stats["wasted_prefill_tokens"] += worst.prefilled
                worst.prefilled = worst.generated = 0
                worst.first_t = worst._last_tok_t = None
                worst.itls.clear()
                queue.append(worst)
                stats["preemptions"] += 1

        # ---- admission -----------------------------------------------------
        while queue and len(running) + len(prefilling) < max_running:
            queue.sort(key=key)
            r = queue[0]
            if admission is not None:
                verdict = admission(r, now, kv_used, kv_capacity)
                if verdict == "reject":
                    # Say no *now*. A request you cannot serve is cheaper to
                    # refuse at the door than to accept and fail late: the
                    # client can retry elsewhere, and the capacity goes to
                    # someone who can still be served.
                    queue.pop(0)
                    r.rejected = True
                    stats["rejected"] += 1
                    continue
                if verdict == "wait":
                    break
            # `reserve` is how much *future* growth the admission check books
            # on top of the prompt. 0 = book only what exists today (and hope);
            # a positive number = book room for the answer as well.
            if (kv_capacity is not None and not overcommit
                    and kv_used + r.prompt_len + reserve > kv_capacity):
                break
            queue.pop(0)
            r.admit_t = now
            kv_used += r.prompt_len
            prefilling.append(r)
            break                    # at most one new prefill per iteration

        # ---- build the iteration -------------------------------------------
        n_dec = len(running)
        pre_tok = 0
        pre_req = None
        if prefilling:
            pre_req = prefilling[0]
            left = pre_req.prompt_len - pre_req.prefilled
            if chunk is None:
                # The pre-2024 default: a prefill owns the whole forward pass,
                # and every streaming answer stops until it is finished.
                pre_tok = left
                n_dec = 0
            else:
                pre_tok = min(chunk, left, max(1, token_budget - n_dec))
        if n_dec == 0 and pre_tok == 0:
            if nxt < len(incoming):
                now = max(now, incoming[nxt].arrive)
                continue
            break

        keys = sum(r.ctx for r in running[:n_dec])
        if pre_req is not None and pre_tok:
            keys += cost.prefill_keys(pre_req.prefilled, pre_tok)
        dt = cost.iter_time(n_dec, pre_tok, keys)
        now += dt
        stats["iters"] += 1
        stats["busy_s"] += dt
        stats["prefill_tokens"] += pre_tok

        # ---- apply the results ---------------------------------------------
        if pre_req is not None and pre_tok:
            pre_req.prefilled += pre_tok
            if pre_req.prefilled >= pre_req.prompt_len:
                prefilling.remove(pre_req)
                pre_req.first_t = now
                pre_req._last_tok_t = now
                pre_req.generated = 1
                kv_used += 1
                running.append(pre_req)

        done = []
        for r in running[:n_dec]:
            r.generated += 1
            kv_used += 1
            if r._last_tok_t is not None:
                r.itls.append(now - r._last_tok_t)
            r._last_tok_t = now
            stats["decode_tokens"] += 1
            if r.generated >= r.out_len:
                done.append(r)
        for r in done:
            r.end_t = now
            kv_used -= r.ctx
            running.remove(r)

        # ---- memory pressure -----------------------------------------------
        stats["peak_kv"] = max(stats["peak_kv"], kv_used)
        if kv_capacity is not None and kv_used > kv_capacity and crash_on_overflow:
            # A server with no admission check does not gracefully degrade: it
            # allocates one block too many and the process dies, taking every
            # in-flight request with it. Modelling that as "stop here" is the
            # honest version -- there is no throughput number after an OOM.
            stats["oom"] = True
            stats["oom_at_s"] = now
            stats["oom_kv"] = kv_used
            stats["oom_inflight"] = len(running) + len(prefilling)
            break
        if kv_capacity is not None and preempt:
            if kv_used > kv_capacity:
                stats["overflow_iters"] += 1
            while kv_used > kv_capacity and running:
                # vLLM's "recompute" preemption: the victim's cache is dropped
                # and it goes back to the queue as if it had just arrived. Every
                # token it had generated has to be produced again, which is why
                # the wasted-token counter exists.
                victim = running[-1]      # newest first: it has lost the least
                running.remove(victim)
                kv_used -= victim.ctx
                victim.preemptions += 1
                stats["wasted_decode_tokens"] += victim.generated
                stats["wasted_prefill_tokens"] += victim.prefilled
                victim.prefilled = 0
                victim.generated = 0
                victim.first_t = None
                victim._last_tok_t = None
                victim.itls.clear()
                queue.append(victim)
                stats["preemptions"] += 1

    stats["makespan_s"] = now
    stats["util"] = stats["busy_s"] / now if now else 0.0
    return stats


def report(reqs, stats, label=""):
    done = [r for r in reqs if r.end_t is not None]
    ttfts = [r.ttft for r in done]
    itls = [x for r in done for x in r.itls]
    return {
        "label": label,
        "completed": len(done),
        "rejected": sum(1 for r in reqs if r.rejected),
        "preemptions": stats["preemptions"],
        "makespan_s": round(stats["makespan_s"], 1),
        "iters": stats["iters"],
        "output_tok_s": round(stats["decode_tokens"] / stats["makespan_s"], 1),
        "ttft_p50": round(pct(ttfts, 50), 3),
        "ttft_p99": round(pct(ttfts, 99), 3),
        "itl_p50": round(pct(itls, 50), 4),
        "itl_p99": round(pct(itls, 99), 4),
        "itl_max": round(max(itls) if itls else float("nan"), 3),
        "e2e_p99": round(pct([r.e2e for r in done], 99), 2),
    }


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[i]
