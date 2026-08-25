"""Project 59 -- Instrumenting a serving engine, and reading the gauges right.

Five sections:

  A. Wire every metric in the Phase 9 dashboard into a real continuous-batching
     engine, expose them at a real HTTP `/metrics` endpoint, and scrape it.
  B. What the average hides: mean TTFT against p50/p95/p99 on the same run.
  C. A percentile read out of a histogram is an ESTIMATE. Three bucket layouts,
     scored against the exact percentile of the same observations.
  D. You cannot average percentiles. Four windows of one run, avg(p99) against
     the true p99 -- and the merged histogram that gets it right.
  E. What instrumentation costs: nanoseconds per observation, and what one
     high-cardinality label does to the scrape.

Usage:
    python3 run.py            # ~4 minutes
    python3 run.py --plot
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "16-static-vs-continuous"))

import obslib as O            # noqa: E402
import batchlib as B          # noqa: E402

OUT = os.path.join(HERE, "outputs")
N_REQ = 44
RATE = 0.9
N_SLOTS = 8


# ---------------------------------------------------------------------------
# A. Instrument a real engine
# ---------------------------------------------------------------------------


def section_a():
    runner, tok = B.load_runner()
    reqs = B.make_workload(tok, n=N_REQ, rate=RATE, seed=0)
    m = O.EngineMetrics()
    ts = O.TimeSeries(window=2.0)
    srv = O.MetricsServer(m.reg).start()

    # one scrape before any traffic: an empty engine still exposes every
    # series, which is what lets a dashboard show "0" instead of "no data".
    cold, cold_dt, cold_bytes = O.scrape(srv.url)

    res = O.run_engine(runner, reqs, n_slots=N_SLOTS, max_len=352, m=m, ts=ts)

    hot, hot_dt, hot_bytes = O.scrape(srv.url)
    exposition = m.reg.render()
    srv.stop()

    done = [r for r in reqs if r.end_t is not None]
    ttfts = [r.ttft for r in done]
    itls = [x for r in done for x in r.itls()]
    e2es = [r.e2e for r in done]

    with open(os.path.join(OUT, "metrics.txt"), "w") as f:
        f.write(exposition)

    return dict(
        engine=res,
        n_requests=len(done),
        prompt_tokens=int(hot["llm_prompt_tokens_total"]),
        gen_tokens=int(hot["llm_generation_tokens_total"]),
        prefill_iters=int(hot['llm_iterations_total{kind="prefill"}']),
        decode_iters=int(hot['llm_iterations_total{kind="decode"}']),
        scrape_cold_s=round(cold_dt, 5), scrape_hot_s=round(hot_dt, 5),
        exposition_bytes=hot_bytes, cold_series=len(cold), hot_series=len(hot),
        n_metric_families=len(m.reg.metrics),
        url_port=srv.port,
        ts_rows=ts.rows,
        ttfts=ttfts, itls=itls, e2es=e2es,
        completion_times=[r.end_t for r in done],
        first_lines=exposition.splitlines()[:14],
    ), m, reqs


# ---------------------------------------------------------------------------
# B. What the average hides
# ---------------------------------------------------------------------------


def section_b(a):
    def summ(xs, name):
        xs = sorted(xs)
        mean = sum(xs) / len(xs)
        return dict(metric=name, n=len(xs), mean=mean,
                    p50=O.exact_quantile(xs, 50), p95=O.exact_quantile(xs, 95),
                    p99=O.exact_quantile(xs, 99), maximum=xs[-1],
                    frac_above_mean=sum(1 for x in xs if x > mean) / len(xs),
                    p99_over_mean=O.exact_quantile(xs, 99) / mean)
    return [summ(a["ttfts"], "TTFT"), summ(a["itls"], "ITL"),
            summ(a["e2es"], "E2E")]


# ---------------------------------------------------------------------------
# C. A histogram percentile is an estimate
# ---------------------------------------------------------------------------


TIGHT = tuple([round(0.5 * 1.35 ** i, 4) for i in range(24)]) + (math.inf,)


def section_c(a):
    layouts = {
        "prometheus default": O.DEFAULT_BUCKETS,
        "vLLM TTFT": O.VLLM_TTFT_BUCKETS,
        "tuned to this workload": TIGHT,
    }
    rows = []
    for name, buckets in layouts.items():
        h = O.Histogram("t", "", buckets)
        for x in a["ttfts"]:
            h.observe(x)
        c = h.counts[()]
        r = dict(layout=name, n_buckets=len(buckets),
                 top_finite=buckets[-2], overflow=c[-1] - c[-2])
        for q in (50, 95, 99):
            est = O.histogram_quantile(buckets, c, q)
            true = O.exact_quantile(a["ttfts"], q)
            r[f"p{q}_est"] = est
            r[f"p{q}_true"] = true
            r[f"p{q}_err_pct"] = 100.0 * (est - true) / true
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# D. Percentiles do not average
# ---------------------------------------------------------------------------


def section_d(a, n_shards=4):
    """Split the run into consecutive time windows -- the same arithmetic as
    splitting traffic across replicas, and the same mistake either way."""
    pairs = sorted(zip(a["completion_times"], a["ttfts"]))
    k = len(pairs)
    shards = [pairs[i * k // n_shards:(i + 1) * k // n_shards]
              for i in range(n_shards)]
    buckets = O.VLLM_TTFT_BUCKETS
    per, counts = [], []
    for i, s in enumerate(shards):
        xs = [x for _, x in s]
        h = O.Histogram("t", "", buckets)
        for x in xs:
            h.observe(x)
        counts.append(h.counts[()])
        per.append(dict(shard=i, n=len(xs),
                        p99=O.exact_quantile(xs, 99),
                        p95=O.exact_quantile(xs, 95),
                        p50=O.exact_quantile(xs, 50)))
    merged = O.merge_counts(counts)
    out = dict(per_shard=per, n_shards=n_shards)
    for q in (50, 95, 99):
        avg_of = sum(s[f"p{q}"] for s in per) / n_shards
        true = O.exact_quantile(a["ttfts"], q)
        out[f"p{q}"] = dict(
            avg_of_shard_quantiles=avg_of, true=true,
            merged_histogram=O.histogram_quantile(buckets, merged, q),
            avg_err_pct=100.0 * (avg_of - true) / true,
            merged_err_pct=100.0 * (O.histogram_quantile(buckets, merged, q)
                                    - true) / true)
    return out


# ---------------------------------------------------------------------------
# E. What instrumentation costs
# ---------------------------------------------------------------------------


def section_e():
    reps = 200_000
    rng = random.Random(0)
    vals = [rng.random() for _ in range(1000)]

    def timeit(fn):
        t0 = time.perf_counter()
        for i in range(reps):
            fn(vals[i & 1023 if i & 1023 < 1000 else 0])
        return (time.perf_counter() - t0) / reps * 1e9

    r = O.Registry()
    c = r.counter("c", "")
    g = r.gauge("g", "")
    h_small = O.Histogram("hs", "", (0.1, 0.5, 1.0, math.inf))
    h_big = O.Histogram("hb", "", O.VLLM_TTFT_BUCKETS)
    cost = dict(
        counter_inc_ns=timeit(lambda v: c.inc()),
        gauge_set_ns=timeit(lambda v: g.set(v)),
        hist_4_buckets_ns=timeit(lambda v: h_small.observe(v)),
        hist_23_buckets_ns=timeit(lambda v: h_big.observe(v)),
        noop_ns=timeit(lambda v: None),
    )
    del g

    # cardinality: the same observations, labelled at different granularity
    card = []
    for n_labels in (1, 10, 100, 1000, 10000):
        reg = O.Registry()
        hh = reg.histogram("llm_ttft_seconds", "TTFT.", O.VLLM_TTFT_BUCKETS,
                           ("route",))
        cc = reg.counter("llm_requests_total", "Reqs.", ("route",))
        for i in range(10_000):
            lab = f"r{i % n_labels}"
            hh.observe(vals[i % 1000], route=lab)
            cc.inc(route=lab)
        srv = O.MetricsServer(reg).start()
        samples, dt, nbytes = O.scrape(srv.url)
        dt2 = min(O.scrape(srv.url)[1] for _ in range(3))
        srv.stop()
        card.append(dict(cardinality=n_labels, series=reg.n_series,
                         scrape_s=dt2, bytes=nbytes,
                         parsed_samples=len(samples)))
    return dict(per_observation_ns=cost, cardinality=card)


# ---------------------------------------------------------------------------


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    a, b, c, d, e = (res["A"], res["B"], res["C"], res["D"], res["E"])
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    # A -- the dashboard
    p = ax[0][0]
    rows = a["ts_rows"]
    t = [r["t"] for r in rows]
    p.plot(t, [r["running"] for r in rows], label="running (= KV slots used)",
           color="#27ae60", lw=2)
    p.plot(t, [r["waiting"] for r in rows], label="queued", color="#c0392b")
    p.set_xlabel("seconds"), p.set_ylabel("requests")
    p.set_ylim(-0.3, N_SLOTS + 0.5)
    p2 = p.twinx()
    p2.plot(t, [r["busy"] * 100 for r in rows], color="#8e44ad", ls=":", lw=2,
            label="engine busy %")
    p2.set_ylabel("engine busy (%)", color="#8e44ad")
    p2.set_ylim(0, 105)
    h1, l1 = p.get_legend_handles_labels()
    h2, l2 = p2.get_legend_handles_labels()
    p.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    p.set_title(f"A. The scraped dashboard\n{a['n_requests']} requests, "
                f"{a['prefill_iters']} prefill + {a['decode_iters']} decode "
                f"passes, {a['engine']['util']*100:.0f}% busy")

    # B -- mean vs tail
    p = ax[0][1]
    tt = sorted(a["ttfts"])
    p.hist(tt, bins=24, color="#bdc3c7", edgecolor="white")
    row = b[0]
    for k, col in (("mean", "#2c3e50"), ("p50", "#27ae60"),
                   ("p95", "#e67e22"), ("p99", "#c0392b")):
        p.axvline(row[k], color=col, lw=2,
                  label=f"{k} = {row[k]:.2f} s")
    p.set_xlabel("TTFT (s)"), p.set_ylabel("requests")
    p.legend(fontsize=8)
    p.set_title(f"B. The average hides the tail\n"
                f"p99 is {row['p99_over_mean']:.1f}x the mean; "
                f"{row['frac_above_mean']*100:.0f}% of requests are above it")

    # C -- bucket layouts
    p = ax[0][2]
    names = [r["layout"] for r in c]
    x = np.arange(len(names))
    for i, (q, col) in enumerate(((50, "#27ae60"), (95, "#e67e22"),
                                  (99, "#c0392b"))):
        p.bar(x + (i - 1) * 0.26, [r[f"p{q}_err_pct"] for r in c], 0.26,
              label=f"p{q}", color=col)
    p.axhline(0, color="k", lw=.8)
    p.set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=8)
    p.set_ylabel("estimate error vs the exact percentile (%)")
    p.legend(fontsize=8)
    p.set_title("C. A histogram percentile is an estimate\n"
                "the bucket layout is the accuracy")

    # D -- averaging percentiles
    p = ax[1][0]
    labels, vals, cols = [], [], []
    for q in (50, 95, 99):
        labels += [f"p{q}\navg of 4", f"p{q}\nmerged", f"p{q}\ntruth"]
        vals += [d[f"p{q}"]["avg_of_shard_quantiles"],
                 d[f"p{q}"]["merged_histogram"], d[f"p{q}"]["true"]]
        cols += ["#c0392b", "#2980b9", "#27ae60"]
    bars = p.bar(range(len(vals)), vals, color=cols)
    p.set_xticks(range(len(vals)), labels, fontsize=7)
    p.legend([bars[0], bars[1], bars[2]],
             ["average of the 4 windows", "merged histogram", "the truth"],
             fontsize=8)
    p.set_ylabel("TTFT (s)")
    p.set_title(f"D. Percentiles do not average\n"
                f"avg of 4 window p99s is "
                f"{d['p99']['avg_err_pct']:+.0f}% off; merged histogram "
                f"{d['p99']['merged_err_pct']:+.0f}%")

    # E -- cardinality
    p = ax[1][1]
    card = e["cardinality"]
    xs = [r["cardinality"] for r in card]
    p.plot(xs, [r["series"] for r in card], "o-", color="#c0392b",
           label="time series")
    p.set_xscale("log"), p.set_yscale("log")
    p.set_xlabel("distinct values of one label")
    p.set_ylabel("series", color="#c0392b")
    p2 = p.twinx()
    p2.plot(xs, [r["scrape_s"] * 1000 for r in card], "s--", color="#2980b9")
    p2.set_ylabel("scrape time (ms)", color="#2980b9")
    p2.set_yscale("log")
    p.set_title(f"E. One label with a request id in it\n"
                f"{card[0]['series']} series and "
                f"{card[0]['scrape_s']*1000:.1f} ms -> "
                f"{card[-1]['series']:,} and "
                f"{card[-1]['scrape_s']*1000:.0f} ms")

    # F -- per-observation cost
    p = ax[1][2]
    cost = e["per_observation_ns"]
    ks = ["noop_ns", "counter_inc_ns", "gauge_set_ns",
          "hist_4_buckets_ns", "hist_23_buckets_ns"]
    lab = ["loop only", "counter", "gauge", "histogram\n4 buckets",
           "histogram\n23 buckets"]
    p.bar(lab, [cost[k] for k in ks],
          color=["#95a5a6", "#27ae60", "#27ae60", "#e67e22", "#c0392b"])
    for i, k in enumerate(ks):
        p.text(i, cost[k], f"{cost[k]:.0f}", ha="center", va="bottom",
               fontsize=8)
    p.set_ylabel("nanoseconds per observation")
    p.set_title("F. What one measurement costs\n"
                f"a decode step here is "
                f"{a['engine']['busy_s']/max(1,a['decode_iters'])*1e9/1e6:.0f} "
                f"ms -- six orders of magnitude more")

    fig.suptitle("Metric instrumentation: the gauges, and the four ways to "
                 "misread them", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "metrics.png"), dpi=118)
    print("wrote", os.path.join(OUT, "metrics.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    a, m, reqs = section_a()
    print(f"[A] done {time.time()-t0:.0f}s  util={a['engine']['util']:.2f}")
    b = section_b(a)
    print(f"[B] TTFT mean {b[0]['mean']:.2f}s  p99 {b[0]['p99']:.2f}s")
    c = section_c(a)
    for r in c:
        print(f"[C] {r['layout']:24s} p99 est {r['p99_est']:.2f} "
              f"true {r['p99_true']:.2f} ({r['p99_err_pct']:+.1f}%)")
    d = section_d(a)
    print(f"[D] avg-of-p99 {d['p99']['avg_of_shard_quantiles']:.2f} vs true "
          f"{d['p99']['true']:.2f}")
    e = section_e()
    print(f"[E] {e['cardinality'][-1]['series']:,} series -> "
          f"{e['cardinality'][-1]['scrape_s']*1000:.0f} ms scrape")

    res = dict(A=a, B=b, C=c, D=d, E=e,
               config=dict(n_requests=N_REQ, rate=RATE, n_slots=N_SLOTS,
                           model=B.MODEL_ID),
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    if ap.parse_args().plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main()
