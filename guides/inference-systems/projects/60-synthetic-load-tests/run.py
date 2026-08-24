"""Project 60 -- Synthetic load tests: how the harness decides the answer.

A load test is a measurement instrument, and like any instrument it has
systematic errors. This project builds one against the real engine of project
59 and then measures the instrument itself.

  A. The length distribution IS the benchmark. Constant-length traffic against
     lognormal traffic with the *same mean* prompt and output length.
  B. Coordinated omission. A closed-loop driver (N workers, each sends the
     next request when the last one returns) against an open-loop driver
     (Poisson arrivals) at the SAME achieved throughput.
  C. The concurrency sweep the guide asks for: 1x / 2x / 5x, plus Little's law
     as a sanity check on every row.
  D. How long must a load test run? The p99 estimated from the first 25%,
     50%, 75% and 100% of the same run.

Usage:
    python3 run.py            # ~7 minutes
    python3 run.py --plot
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "16-static-vs-continuous"))
sys.path.insert(0, os.path.join(PROJ, "59-metric-instrumentation"))

import obslib as O            # noqa: E402
import batchlib as B          # noqa: E402

OUT = os.path.join(HERE, "outputs")
N_REQ = 44
N_SLOTS = 8
P_MED, O_MED = 64, 20


def fresh(reqs):
    """A trace can only be run once -- `run_engine` writes timings into it."""
    return copy.deepcopy(reqs)


def lognormal_trace(tok, seed=0, rate=1.0):
    return B.make_workload(tok, n=N_REQ, rate=rate, seed=seed,
                           p_med=P_MED, p_sigma=0.75, o_med=O_MED, o_sigma=0.7,
                           p_max=192, o_max=48)


def constant_trace(tok, like, seed=0, rate=1.0):
    """Same arrival process, same MEAN prompt and output length, no spread."""
    rng = random.Random(seed + 99)
    pm = int(round(sum(r.prompt_len for r in like) / len(like)))
    om = int(round(sum(r.max_new for r in like) / len(like)))
    out = []
    for r in like:
        ids = [rng.choice(range(1000, 12000)) for _ in range(pm)]
        out.append(B.Request(rid=r.rid, arrive=r.arrive, prompt_ids=ids,
                             max_new=om))
    return out, pm, om


def measure(reqs, res, label):
    done = [r for r in reqs if r.end_t is not None]
    ttfts = [r.ttft for r in done]
    e2es = [r.e2e for r in done]
    itls = [x for r in done for x in r.itls()]
    gen = sum(len(r.tokens) for r in done)
    T = res["virtual_s"]
    mean_e2e = sum(e2es) / len(e2es)
    return dict(
        label=label, n=len(done), virtual_s=round(T, 2),
        req_s=len(done) / T, out_tok_s=gen / T,
        prompt_tokens=sum(r.prompt_len for r in done), gen_tokens=gen,
        util=res["util"],
        ttft_p50=O.exact_quantile(ttfts, 50), ttft_p95=O.exact_quantile(ttfts, 95),
        ttft_p99=O.exact_quantile(ttfts, 99),
        itl_p50=O.exact_quantile(itls, 50), itl_p99=O.exact_quantile(itls, 99),
        e2e_p50=O.exact_quantile(e2es, 50), e2e_p95=O.exact_quantile(e2es, 95),
        e2e_p99=O.exact_quantile(e2es, 99), e2e_mean=mean_e2e,
        # Little's law: the average number of requests in the system equals the
        # arrival rate times the average time each one spends there.
        little_L=len(done) / T * mean_e2e,
        _ttfts=ttfts, _e2es=e2es, _ends=[r.end_t for r in done],
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    runner, tok = B.load_runner()
    base = lognormal_trace(tok)
    const, pm, om = constant_trace(tok, base)
    plens = [r.prompt_len for r in base]
    olens = [r.max_new for r in base]

    res = {"config": dict(n=N_REQ, n_slots=N_SLOTS, model=B.MODEL_ID,
                          const_prompt=pm, const_out=om,
                          prompt_mean=sum(plens) / len(plens),
                          prompt_p99=O.exact_quantile(plens, 99),
                          prompt_max=max(plens),
                          out_mean=sum(olens) / len(olens),
                          out_max=max(olens))}

    def run(trace, label, **kw):
        t = fresh(trace)
        r = O.run_engine(runner, t, n_slots=N_SLOTS, max_len=352, **kw)
        row = measure(t, r, label)
        print(f"  {label:26s} {row['req_s']:.3f} req/s  "
              f"ttft p50 {row['ttft_p50']:.2f} p99 {row['ttft_p99']:.2f}  "
              f"e2e p99 {row['e2e_p99']:.2f}  [{time.time()-t0:.0f}s]")
        return row

    # --- C first: the closed-loop sweep, because B needs its throughput ----
    print("[C] closed-loop concurrency sweep")
    sweep = {}
    for c in (2, 4, 10):
        sweep[c] = run(base, f"closed loop C={c}", concurrency=c)

    # --- B: open loop at the SAME achieved throughput ----------------------
    print("[B] open loop matched to closed C=4")
    lam = sweep[4]["req_s"]
    matched = lognormal_trace(tok, rate=lam)
    open_matched = run(matched, f"open loop {lam:.3f} req/s")

    # --- A: distribution shape, both open loop at the same rate ------------
    print("[A] length distribution")
    a_rate = lam * 0.85
    a_log = lognormal_trace(tok, rate=a_rate)
    a_const, _, _ = constant_trace(tok, a_log)
    a_lognormal = run(a_log, "lognormal lengths")
    a_constant = run(a_const, "constant lengths")

    # --- D: how long must the test run? ------------------------------------
    src = sorted(zip(open_matched["_ends"], open_matched["_e2es"]))
    dur = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        k = max(2, int(len(src) * frac))
        xs = [e for _, e in src[:k]]
        dur.append(dict(frac=frac, n=k, p50=O.exact_quantile(xs, 50),
                        p95=O.exact_quantile(xs, 95),
                        p99=O.exact_quantile(xs, 99), maximum=max(xs)))
    warm = dict(
        with_warmup=O.exact_quantile(open_matched["_e2es"], 50),
        without_first3=O.exact_quantile(
            [e for _, e in sorted(zip(open_matched["_ends"],
                                      open_matched["_e2es"]))][3:], 50))

    res.update(A=dict(lognormal=a_lognormal, constant=a_constant,
                      rate=a_rate),
               B=dict(closed=sweep[4], open=open_matched, rate=lam),
               C={str(k): v for k, v in sweep.items()},
               D=dict(duration=dur, warmup=warm))
    res["wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print(f"total {time.time()-t0:.0f}s")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    A, Bs, C, D = res["A"], res["B"], res["C"], res["D"]

    # A -- lengths
    p = ax[0][0]
    names = ["constant\nlengths", "lognormal\nlengths"]
    rows = [A["constant"], A["lognormal"]]
    x = np.arange(2)
    p.bar(x - .2, [r["e2e_p50"] for r in rows], .4, label="E2E p50",
          color="#27ae60")
    p.bar(x + .2, [r["e2e_p99"] for r in rows], .4, label="E2E p99",
          color="#c0392b")
    for i, r in enumerate(rows):
        p.text(i - .2, r["e2e_p50"], f"{r['e2e_p50']:.1f}", ha="center",
               va="bottom", fontsize=8)
        p.text(i + .2, r["e2e_p99"], f"{r['e2e_p99']:.1f}", ha="center",
               va="bottom", fontsize=8)
    p.set_xticks(x, names)
    p.set_ylabel("seconds")
    p.legend(fontsize=8)
    p.set_title(f"A. Same mean work, different spread\n"
                f"p99 {A['lognormal']['e2e_p99']/A['constant']['e2e_p99']:.2f}x "
                f"worse; throughput "
                f"{A['lognormal']['out_tok_s']/A['constant']['out_tok_s']:.2f}x")

    # B -- coordinated omission
    p = ax[0][1]
    rows = [Bs["closed"], Bs["open"]]
    labels = [f"closed loop\nC=4\n{rows[0]['req_s']:.2f} req/s",
              f"open loop\n{rows[1]['req_s']:.2f} req/s"]
    x = np.arange(2)
    p.bar(x - .2, [r["e2e_p50"] for r in rows], .4, label="E2E p50",
          color="#27ae60")
    p.bar(x + .2, [r["e2e_p99"] for r in rows], .4, label="E2E p99",
          color="#c0392b")
    p.set_xticks(x, labels, fontsize=8)
    p.set_ylabel("seconds")
    p.legend(fontsize=8)
    p.set_title(f"B. Coordinated omission\n"
                f"closed loop reports "
                f"{rows[0]['req_s']/rows[1]['req_s']:.2f}x the throughput AND "
                f"{rows[0]['e2e_p99']/rows[1]['e2e_p99']:.2f}x the p99")

    # C -- concurrency sweep
    p = ax[0][2]
    cs = sorted(int(k) for k in C)
    p.plot(cs, [C[str(c)]["out_tok_s"] for c in cs], "o-", color="#2980b9",
           label="output tok/s")
    p.set_xlabel("closed-loop concurrency"), p.set_ylabel("output tok/s",
                                                          color="#2980b9")
    p2 = p.twinx()
    p2.plot(cs, [C[str(c)]["e2e_p50"] for c in cs], "s--", color="#c0392b",
            label="E2E p50 (s)")
    p2.set_ylabel("E2E p50 (s)", color="#c0392b")
    base_t = C[str(cs[0])]["out_tok_s"]
    p.set_title(f"C. 1x / 2x / 5x concurrency\n"
                f"5x the users buys "
                f"{C[str(cs[-1])]['out_tok_s']/base_t:.2f}x the tokens and "
                f"{C[str(cs[-1])]['e2e_p50']/C[str(cs[0])]['e2e_p50']:.2f}x "
                f"the latency")

    # D -- Little's law check
    p = ax[1][0]
    allrows = [("closed C=2", C["2"]), ("closed C=4", C["4"]),
               ("closed C=10", C["10"]), ("open matched", Bs["open"]),
               ("open lognormal", A["lognormal"]),
               ("open constant", A["constant"])]
    xs = np.arange(len(allrows))
    p.bar(xs, [r["little_L"] for _, r in allrows], color="#8e44ad")
    for i, (n, r) in enumerate(allrows):
        if n.startswith("closed"):
            p.plot([i - .4, i + .4], [int(n.split("=")[1])] * 2, color="k",
                   lw=2)
    p.set_xticks(xs, [n.replace(" ", "\n") for n, _ in allrows], fontsize=7)
    p.set_ylabel("throughput x mean latency  (Little's L)")
    p.set_title("D1. Little's law as a harness self-check\n"
                "black bar = the concurrency the driver was told to hold")

    # E -- test duration
    p = ax[1][1]
    dd = D["duration"]
    p.plot([d["frac"] * 100 for d in dd], [d["p50"] for d in dd], "o-",
           label="p50", color="#27ae60")
    p.plot([d["frac"] * 100 for d in dd], [d["p95"] for d in dd], "s-",
           label="p95", color="#e67e22")
    p.plot([d["frac"] * 100 for d in dd], [d["p99"] for d in dd], "^-",
           label="p99", color="#c0392b")
    p.set_xlabel("% of the run used"), p.set_ylabel("E2E latency (s)")
    p.legend(fontsize=8)
    p.set_title(f"D2. A short test under-reports the tail\n"
                f"p99 from the first quarter is "
                f"{dd[0]['p99']/dd[-1]['p99']:.2f}x the full-run p99")

    # F -- latency vs arrival index (the shape closed loop cannot produce)
    p = ax[1][2]
    for name, row, col in (("open loop", Bs["open"], "#c0392b"),
                           ("closed loop C=4", Bs["closed"], "#27ae60")):
        xs = sorted(row["_e2es"])
        ys = [100.0 * (i + 1) / len(xs) for i in range(len(xs))]
        p.plot(xs, ys, label=name, color=col)
    p.set_xlabel("E2E latency (s)"), p.set_ylabel("percentile")
    p.axhline(99, color="k", lw=.6, ls=":")
    p.legend(fontsize=8)
    p.set_title("F. The whole distribution, not two numbers\n"
                "closed loop is missing the right-hand tail")

    fig.suptitle("Synthetic load tests: the harness is part of the "
                 "measurement", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "loadtest.png"), dpi=118)
    print("wrote", os.path.join(OUT, "loadtest.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    if ap.parse_args().plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main()
