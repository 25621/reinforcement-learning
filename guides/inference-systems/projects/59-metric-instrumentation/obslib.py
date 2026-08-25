"""Phase 9's shared observability stack: a Prometheus-compatible metrics
library, a real HTTP `/metrics` endpoint, and an instrumented engine.

Why write this instead of `pip install prometheus_client`? Because the whole
point of Phase 9 is that a dashboard is only as trustworthy as the arithmetic
underneath it, and the two mistakes this phase measures -- reading a quantile
out of a histogram, and averaging quantiles across replicas -- are *invisible*
if the library is a black box. Every bucket boundary here is one you can see.

The exposition format, the bucket semantics (`le` = "less than or equal",
cumulative) and `histogram_quantile`'s interpolation rule are all copied from
Prometheus, so the numbers this file produces are the numbers a real
Prometheus would produce from the same observations.

Three layers:

  1. **Metric types** -- `Counter`, `Gauge`, `Histogram`, held in a `Registry`
     that renders the Prometheus text exposition format.
  2. **Transport** -- `MetricsServer` serves `/metrics` over real HTTP;
     `scrape()` fetches and parses it, the way a Prometheus server would.
  3. **The instrumented engine** -- `run_engine` is project 16's continuous
     batching loop with every metric in the Phase 9 dashboard wired into it,
     plus hooks for admission control (project 65) and fault injection
     (project 66).

Shared by projects 59, 60, 63, 65 and 66.
"""

from __future__ import annotations

import http.server
import math
import os
import socket
import threading
import time
import urllib.request
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Metric types
# ---------------------------------------------------------------------------
#
# A metric is a name, a help string, a type, and a map from a *label tuple* to
# a value. "Label" is Prometheus' word for a key=value pair attached to a
# measurement (`model="qwen"`, `phase="decode"`). Each distinct combination of
# label values is its own independent time series -- which is why section E of
# project 59 can blow the library up by labelling with a request id.


def _fmt(v: float) -> str:
    """Prometheus wants `+Inf`, `-Inf`, `NaN` spelled that way, and prefers a
    plain decimal for everything else."""
    if v == math.inf:
        return "+Inf"
    if v == -math.inf:
        return "-Inf"
    if v != v:
        return "NaN"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _labels_str(names, values) -> str:
    if not names:
        return ""
    inner = ",".join(f'{n}="{v}"' for n, v in zip(names, values))
    return "{" + inner + "}"


class _Metric:
    def __init__(self, name, help_, labelnames=()):
        self.name = name
        self.help = help_
        self.labelnames = tuple(labelnames)
        self.values = {}

    def _key(self, labels):
        if set(labels) != set(self.labelnames):
            raise KeyError(f"{self.name} expects labels {self.labelnames}, "
                           f"got {tuple(labels)}")
        return tuple(str(labels[n]) for n in self.labelnames)

    @property
    def n_series(self) -> int:
        return max(1, len(self.values))


class Counter(_Metric):
    """A number that only ever goes up (requests served, tokens generated).

    Counters are never read directly on a dashboard -- you read their *rate*.
    That is deliberate: a counter survives a process restart being reset to
    zero, and `rate()` can detect and skip the reset. A gauge cannot."""

    typ = "counter"

    def inc(self, amount=1.0, **labels):
        k = self._key(labels)
        self.values[k] = self.values.get(k, 0.0) + amount

    def get(self, **labels):
        return self.values.get(self._key(labels), 0.0)

    def samples(self):
        if not self.values and not self.labelnames:
            return [(self.name, (), 0.0)]
        return [(self.name, k, v) for k, v in self.values.items()]


class Gauge(_Metric):
    """A number that goes up and down (queue depth, KV bytes in use).

    A gauge is only true at the instant it is scraped. Everything between two
    scrapes is invisible -- a 200 ms queue spike between a 15 s scrape pair
    never happened as far as the dashboard is concerned. That blind spot is
    why latency is a histogram and not a gauge."""

    typ = "gauge"

    def set(self, value, **labels):
        self.values[self._key(labels)] = float(value)

    def inc(self, amount=1.0, **labels):
        k = self._key(labels)
        self.values[k] = self.values.get(k, 0.0) + amount

    def dec(self, amount=1.0, **labels):
        self.inc(-amount, **labels)

    def get(self, **labels):
        return self.values.get(self._key(labels), 0.0)

    def samples(self):
        if not self.values and not self.labelnames:
            return [(self.name, (), 0.0)]
        return [(self.name, k, v) for k, v in self.values.items()]


# Prometheus' own default buckets, in seconds. Designed for web handlers that
# answer in milliseconds -- note the top finite bucket is 10 s.
DEFAULT_BUCKETS = (.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0,
                   2.5, 5.0, 7.5, 10.0, math.inf)

# What vLLM ships for time-to-first-token, in seconds. Much wider, because a
# 30k-token prefill genuinely can take a minute.
VLLM_TTFT_BUCKETS = (.001, .005, .01, .02, .04, .06, .08, .1, .25, .5,
                     .75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0,
                     160.0, 640.0, 2560.0, math.inf)


class Histogram(_Metric):
    """Counts of observations falling at or below each bucket boundary.

    Prometheus histograms are *cumulative*: the bucket labelled `le="0.5"`
    holds everything <= 0.5 s, including everything already counted in
    `le="0.1"`. That is what makes them addable across replicas -- add the
    bucket counts elementwise and you have the histogram of the union. A
    quantile is NOT addable, which is the subject of section D.

    What is stored is only the counts. The individual observations are gone,
    so any percentile you read back is an *estimate* interpolated inside
    whichever bucket the percentile lands in. Section C measures the error."""

    typ = "histogram"

    def __init__(self, name, help_, buckets=DEFAULT_BUCKETS, labelnames=()):
        super().__init__(name, help_, labelnames)
        self.buckets = tuple(sorted(buckets))
        if self.buckets[-1] != math.inf:
            self.buckets = self.buckets + (math.inf,)
        self.counts = {}     # label tuple -> list of per-bucket cumulative counts
        self.sums = {}
        self.totals = {}
        self.raw = {}        # kept ONLY so this project can score the estimate

    def observe(self, value, keep_raw=False, **labels):
        k = self._key(labels)
        if k not in self.counts:
            self.counts[k] = [0] * len(self.buckets)
            self.sums[k] = 0.0
            self.totals[k] = 0
            self.raw[k] = []
        c = self.counts[k]
        for i, b in enumerate(self.buckets):
            if value <= b:
                c[i] += 1
        self.sums[k] += value
        self.totals[k] += 1
        if keep_raw:
            self.raw[k].append(value)

    @property
    def n_series(self):
        # one series per bucket, plus _sum and _count -- this is why a
        # histogram with a high-cardinality label is so much worse than a
        # counter with one.
        return max(1, len(self.counts)) * (len(self.buckets) + 2)

    def samples(self):
        out = []
        keys = list(self.counts) or [()]
        for k in keys:
            c = self.counts.get(k, [0] * len(self.buckets))
            for b, n in zip(self.buckets, c):
                out.append((self.name + "_bucket", k + (_fmt(b),), float(n)))
            out.append((self.name + "_sum", k, self.sums.get(k, 0.0)))
            out.append((self.name + "_count", k, float(self.totals.get(k, 0))))
        return out

    def bucket_labelnames(self):
        return self.labelnames + ("le",)


class Registry:
    """Holds metrics and renders the exposition format Prometheus scrapes."""

    def __init__(self):
        self.metrics = {}

    def register(self, m):
        self.metrics[m.name] = m
        return m

    def counter(self, name, help_, labelnames=()):
        return self.register(Counter(name, help_, labelnames))

    def gauge(self, name, help_, labelnames=()):
        return self.register(Gauge(name, help_, labelnames))

    def histogram(self, name, help_, buckets=DEFAULT_BUCKETS, labelnames=()):
        return self.register(Histogram(name, help_, buckets, labelnames))

    @property
    def n_series(self):
        return sum(m.n_series for m in self.metrics.values())

    def render(self) -> str:
        """The Prometheus text exposition format, version 0.0.4.

        Three lines of overhead per metric family (`# HELP`, `# TYPE`) and one
        line per series. A scrape is literally this string over HTTP -- which
        is why the response size, and the scrape latency, grow linearly with
        the number of series."""
        parts = []
        for m in self.metrics.values():
            parts.append(f"# HELP {m.name} {m.help}")
            parts.append(f"# TYPE {m.name} {m.typ}")
            names = (m.bucket_labelnames() if isinstance(m, Histogram)
                     else m.labelnames)
            for sname, key, val in m.samples():
                if sname.endswith("_bucket"):
                    parts.append(f"{sname}{_labels_str(names, key)} {_fmt(val)}")
                else:
                    parts.append(
                        f"{sname}{_labels_str(m.labelnames, key)} {_fmt(val)}")
        return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def exact_quantile(xs, q):
    """The truth: the q-th percentile of the observations themselves."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round((q / 100.0) * (len(xs) - 1)))))
    return xs[i]


def histogram_quantile(buckets, counts, q):
    """Prometheus' `histogram_quantile`, reimplemented exactly.

    Find the bucket in which the q-th observation falls, then **interpolate
    linearly between that bucket's lower and upper boundary**. The
    interpolation assumes the observations are spread evenly inside the
    bucket, which they are not -- latency piles up near the bottom of a wide
    bucket -- so the estimate is biased high whenever the bucket is wide.

    Two rules matter and both bite in practice:
      * if the answer falls in the `+Inf` bucket, Prometheus returns the
        largest FINITE boundary. Your p99 is then "at least this", reported
        as if it were exact.
      * the lower boundary of the first bucket is taken to be 0.
    """
    total = counts[-1] if counts else 0
    if total == 0:
        return float("nan")
    rank = (q / 100.0) * total
    for i, (b, c) in enumerate(zip(buckets, counts)):
        if c >= rank:
            if b == math.inf:
                # everything past the last finite bucket is unmeasurable
                return buckets[i - 1] if i else float("nan")
            lo = buckets[i - 1] if i else 0.0
            clo = counts[i - 1] if i else 0
            if c == clo:
                return b
            return lo + (b - lo) * (rank - clo) / (c - clo)
    return buckets[-2] if len(buckets) > 1 else float("nan")


def merge_counts(list_of_counts):
    """Add histograms elementwise -- the operation quantiles do not support."""
    out = [0] * len(list_of_counts[0])
    for c in list_of_counts:
        for i, v in enumerate(c):
            out[i] += v
    return out


# ---------------------------------------------------------------------------
# 2. Transport: a real /metrics endpoint and a real scraper
# ---------------------------------------------------------------------------


class MetricsServer:
    """Serves `registry.render()` at `/metrics` over HTTP, in a thread.

    This is the entire "integration" with Prometheus. There is no push, no
    agent and no SDK: the server exposes a text page, and Prometheus fetches
    it on a timer. Anything that can print text can be monitored."""

    def __init__(self, registry, host="127.0.0.1", port=0):
        self.registry = registry
        reg = registry

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):                      # noqa: N802
                if self.path.split("?")[0] != "/metrics":
                    self.send_response(404)
                    self.end_headers()
                    return
                body = reg.render().encode()
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):             # keep the console quiet
                pass

        self.httpd = http.server.ThreadingHTTPServer((host, port), Handler)
        self.port = self.httpd.server_address[1]
        self.url = f"http://{host}:{self.port}/metrics"
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def scrape(url, timeout=30):
    """Fetch and parse one scrape. Returns (samples, seconds, bytes).

    `samples` maps `name{label="v",...}` to a float, which is how a Prometheus
    server stores it too."""
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=timeout) as r:
        text = r.read().decode()
    dt = time.perf_counter() - t0
    return parse_exposition(text), dt, len(text)


def parse_exposition(text):
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        key, _, val = line.rpartition(" ")
        try:
            out[key] = float(val)
        except ValueError:
            continue
    return out


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ---------------------------------------------------------------------------
# 3. The instrumented engine
# ---------------------------------------------------------------------------


class EngineMetrics:
    """Every metric in the Phase 9 dashboard figure, on one registry.

    The names follow vLLM's, so a dashboard written against this works against
    a real engine with the prefix changed."""

    def __init__(self, registry=None, ttft_buckets=VLLM_TTFT_BUCKETS,
                 itl_buckets=DEFAULT_BUCKETS, keep_raw=True):
        self.reg = registry or Registry()
        r = self.reg
        self.keep_raw = keep_raw
        self.requests = r.counter(
            "llm_requests_total", "Requests finished, by outcome.", ("outcome",))
        self.prompt_tokens = r.counter(
            "llm_prompt_tokens_total", "Prompt tokens prefilled.")
        self.gen_tokens = r.counter(
            "llm_generation_tokens_total", "Tokens generated.")
        self.iters = r.counter(
            "llm_iterations_total", "Forward passes, by kind.", ("kind",))
        self.ttft = r.histogram(
            "llm_time_to_first_token_seconds", "TTFT.", ttft_buckets)
        self.itl = r.histogram(
            "llm_inter_token_latency_seconds", "Time per output token.",
            itl_buckets)
        self.e2e = r.histogram(
            "llm_e2e_request_latency_seconds", "End-to-end latency.",
            VLLM_TTFT_BUCKETS)
        self.queue_time = r.histogram(
            "llm_request_queue_time_seconds", "Admitted-but-not-started time.",
            DEFAULT_BUCKETS)
        self.running = r.gauge("llm_num_requests_running", "Requests decoding.")
        self.waiting = r.gauge("llm_num_requests_waiting", "Requests queued.")
        self.kv_usage = r.gauge(
            "llm_kv_cache_usage_perc", "KV slots in use / slots available.")
        self.gpu_busy = r.gauge(
            "llm_engine_busy_perc", "Fraction of wall time inside a forward pass.")

    def observe_request(self, r, now):
        self.requests.inc(outcome="ok")
        self.prompt_tokens.inc(r.prompt_len)
        self.gen_tokens.inc(len(r.tokens))
        if r.ttft is not None:
            self.ttft.observe(r.ttft, keep_raw=self.keep_raw)
        if r.admit_t is not None:
            self.queue_time.observe(max(0.0, r.admit_t - r.arrive),
                                    keep_raw=self.keep_raw)
        for x in r.itls():
            self.itl.observe(x, keep_raw=self.keep_raw)
        if r.e2e is not None:
            self.e2e.observe(r.e2e, keep_raw=self.keep_raw)


@dataclass
class TimeSeries:
    """Samples the gauges on a fixed cadence, exactly as a scrape would.

    `window` is the scrape interval. Everything that happens between two
    samples is invisible -- projects 60 and 66 both depend on that being true,
    because it is true of every real dashboard."""

    window: float = 2.0
    rows: list = field(default_factory=list)
    _next: float = 0.0

    def maybe(self, now, snap):
        while now >= self._next:
            self.rows.append(dict(t=self._next, **snap()))
            self._next += self.window


@dataclass
class EngineState:
    """What the fault-injection hooks in project 66 are allowed to change."""
    n_slots: int
    max_batch: int
    slow_factor: float = 1.0     # multiplies every measured forward-pass time
    admitting: bool = True


def run_engine(runner, reqs, *, n_slots=8, max_len=320, m=None, ts=None,
               admit=None, order=None, faults=None, state=None,
               max_batch=None, concurrency=None):
    """Continuous batching in virtual time, fully instrumented.

    This is project 16's `run_continuous` with four additions Phase 9 needs:
    a metrics object, a periodic gauge sampler, a pluggable admission rule
    (project 65), and a fault list that can change the engine mid-run
    (project 66).

    `concurrency=C` switches the driver from **open loop** (requests arrive on
    the timeline they were generated with, whatever the engine is doing) to
    **closed loop** (exactly C requests are in the system, and the next one
    arrives the instant one finishes). Project 60 measures how much that one
    switch changes the answer.

    *Virtual time* means the clock advances by the **measured** duration of
    each forward pass rather than by `time.time()`. Every forward pass is a
    real model call on real tensors -- only the arrival timeline is
    synthetic. That keeps results reproducible on a machine whose load average
    this project does not control, while still charging real model cost.
    """
    from batchlib import SlotKV

    st = state or EngineState(n_slots=n_slots, max_batch=max_batch or n_slots)
    pool = SlotKV(runner.n_layers, st.n_slots, runner.n_kv_heads,
                  runner.d_head, max_len)
    faults = sorted(faults or [], key=lambda f: f[0])
    fi = 0

    incoming = sorted(reqs, key=lambda r: r.arrive)
    queue, running = [], []
    nxt, now, busy = 0, 0.0, 0.0
    wall0 = time.perf_counter()

    def snap():
        return dict(running=len(running), waiting=len(queue),
                    kv=len(running) / max(1, st.n_slots),
                    busy=busy / now if now else 0.0,
                    done=sum(1 for r in reqs if r.end_t is not None))

    def advance(dt):
        nonlocal now, busy
        dt *= st.slow_factor
        now += dt
        busy += dt
        if m is not None:
            m.gpu_busy.set(busy / now if now else 0.0)
        if ts is not None:
            ts.maybe(now, snap)

    while nxt < len(incoming) or queue or running:
        while fi < len(faults) and faults[fi][0] <= now:
            faults[fi][1](st)
            fi += 1
        while nxt < len(incoming) and (
                len(queue) + len(running) < concurrency if concurrency
                else incoming[nxt].arrive <= now):
            r = incoming[nxt]
            nxt += 1
            if concurrency:
                r.arrive = now
            verdict = "admit" if admit is None else admit(r, now, st, queue,
                                                          running)
            if verdict == "reject":
                r.rejected = True
                r.end_t = None
                if m is not None:
                    m.requests.inc(outcome="shed")
            else:
                queue.append(r)
        if m is not None:
            m.running.set(len(running))
            m.waiting.set(len(queue))
            m.kv_usage.set(len(running) / max(1, st.n_slots))
        if ts is not None:
            ts.maybe(now, snap)

        if not queue and not running:
            if nxt >= len(incoming):
                break
            now = max(now, incoming[nxt].arrive)
            continue

        n_free = st.n_slots - len(running)
        if queue and n_free > 0 and len(running) < st.max_batch:
            if order is not None:
                queue.sort(key=lambda r: order(r, now))
            else:
                queue.sort(key=lambda r: r.arrive)
            r = queue.pop(0)
            r.slot = pool.acquire()
            r.admit_t = now
            import torch
            ids = torch.tensor(r.prompt_ids).view(1, -1)
            logits, dt = runner.prefill(pool, [r.slot], ids, [r.prompt_len])
            advance(dt)
            if m is not None:
                m.iters.inc(kind="prefill")
            r.tokens.append(int(logits.argmax(-1)[0]))
            r.first_tok_t = now
            r.token_times.append(now)
            r.cur_len = r.prompt_len
            running.append(r)
            continue

        if not running:
            if nxt < len(incoming):
                now = max(now, incoming[nxt].arrive)
                continue
            break

        slots = [r.slot for r in running]
        toks = [r.tokens[-1] for r in running]
        lens = [r.cur_len for r in running]
        logits, dt = runner.decode_step(pool, slots, toks, lens)
        advance(dt)
        if m is not None:
            m.iters.inc(kind="decode")
        nxt_tok = logits.argmax(-1).tolist()
        finished = []
        for j, r in enumerate(running):
            r.cur_len += 1
            r.tokens.append(nxt_tok[j])
            r.token_times.append(now)
            if len(r.tokens) >= r.max_new:
                finished.append(r)
        for r in finished:
            r.end_t = now
            pool.release(r.slot)
            running.remove(r)
            if m is not None:
                m.observe_request(r, now)

    if ts is not None:
        ts.maybe(now, snap)
    return dict(virtual_s=now, wall_s=time.perf_counter() - wall0,
                busy_s=busy, util=busy / now if now else 0.0)


# ---------------------------------------------------------------------------
# Small helpers reused by several projects
# ---------------------------------------------------------------------------


def pct(xs, p):
    return exact_quantile(xs, p)


def bootstrap_ci(xs, stat, n=400, alpha=0.05, seed=0):
    """A confidence interval without assuming a distribution.

    Resample the observations *with replacement* n times, recompute the
    statistic on each resample, and read off the middle 95% of the results.
    Named "bootstrap" after "pulling yourself up by your bootstraps": the only
    information used is the sample itself."""
    import random
    rng = random.Random(seed)
    if not xs:
        return (float("nan"), float("nan"))
    vals = []
    k = len(xs)
    for _ in range(n):
        vals.append(stat([xs[rng.randrange(k)] for _ in range(k)]))
    vals.sort()
    lo = vals[int(alpha / 2 * (n - 1))]
    hi = vals[int((1 - alpha / 2) * (n - 1))]
    return (lo, hi)


def add_batchlib_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    projects = os.path.dirname(here)
    for d in ("16-static-vs-continuous", "18-chunked-prefill-simulator"):
        p = os.path.join(projects, d)
        if p not in os.sys.path:
            os.sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# The shared cost model (projects 61, 62, 63, 65)
# ---------------------------------------------------------------------------


COST_JSON = "costmodel.json"


def fit_cost_model(runner, ctx=256, batches=(1, 2, 4, 8, 16),
                   toks=(64, 128, 256, 512, 1024, 2048), save_to=None):
    """Time real forward passes and fit project 18's four-coefficient model.

    Refitting rather than hard-coding matters because every number Phase 9
    reports in seconds is downstream of these four coefficients. A simulator
    with a borrowed cost model is a simulator of somebody else's machine.
    """
    import json as _json
    import torch
    from batchlib import SlotKV
    from simlib import CostModel

    dec, pre = [], []
    for b in batches:
        pool = SlotKV(runner.n_layers, b, runner.n_kv_heads, runner.d_head, 512)
        ids = torch.randint(1000, 12000, (b, ctx))
        runner.prefill(pool, list(range(b)), ids, [ctx] * b, count=False)
        best = min(runner.decode_step(pool, list(range(b)), [11] * b,
                                      [ctx] * b, count=False)[1]
                   for _ in range(3))
        dec.append((b, best))
    for t in toks:
        pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head,
                      max(t + 4, 512))
        ids = torch.randint(1000, 12000, (1, t))
        best = min(runner.prefill(pool, [0], ids, [t], count=False)[1]
                   for _ in range(2 if t <= 512 else 1))
        pre.append((t, best))
    cost = CostModel.fit(dec, pre, decode_ctx=ctx)
    err_d = sum(abs(cost.iter_time(b, 0, b * ctx) - s) / s
                for b, s in dec) / len(dec)
    err_p = sum(abs(cost.iter_time(0, t, cost.prefill_keys(0, t)) - s) / s
                for t, s in pre) / len(pre)
    info = dict(base=cost.base, per_decode=cost.per_decode,
                per_prefill=cost.per_prefill, per_key_read=cost.per_key_read,
                decode_points=dec, prefill_points=pre,
                fit_err_decode=err_d, fit_err_prefill=err_p)
    if save_to:
        with open(save_to, "w") as f:
            _json.dump(info, f, indent=1)
    return cost, info


def load_cost_model():
    """The cost model project 61 fitted, or project 18's committed fit.

    Projects 62, 63 and 65 do not refit: a fit taken while this shared machine
    happened to be busy would silently move every downstream number, and the
    point of those projects is the policy, not the silicon."""
    import json as _json
    from simlib import CostModel

    here = os.path.dirname(os.path.abspath(__file__))
    projects = os.path.dirname(here)
    p61 = os.path.join(projects, "61-slo-simulation", "outputs", COST_JSON)
    if os.path.exists(p61):
        with open(p61) as f:
            d = _json.load(f)
        return CostModel(base=d["base"], per_decode=d["per_decode"],
                         per_prefill=d["per_prefill"],
                         per_key_read=d["per_key_read"]), "project 61 fit"
    p18 = os.path.join(projects, "18-chunked-prefill-simulator", "outputs",
                       "findings.json")
    with open(p18) as f:
        a = _json.load(f)["A"]
    return CostModel(base=a["base_s"], per_decode=a["per_decode_s"],
                     per_prefill=a["per_prefill_s"],
                     per_key_read=a["per_key_read_s"]), "project 18 fit"
