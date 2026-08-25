"""Project 62 -- An error budget, computed daily, spent by real incidents.

An SLO of 99.9% is also a permission slip: it says you may be bad 0.1% of the
time. That 0.1% is the **error budget**, and tracking what you spend it on is
the only way an SLO changes behaviour instead of decorating a slide.

This project simulates a month of traffic against the engine cost model
fitted in project 61, injects three incidents, and then computes the budget
four different ways -- because the definition you pick decides whether you
shipped on Friday.

  A. A month of traffic: a diurnal shape, a weekend, three injected incidents.
  B. Two SLIs over the same month: request-based and time-based. They
     disagree, and the size of the disagreement is the point.
  C. "p95 < T" is not something you can track over a month. The month's p95
     against the fraction of requests that were actually fast.
  D. Burn-rate alerting (Google's multi-window rule) against the naive
     threshold alert: pages fired, false pages, and time to detect.
  E. The budget over time, and the day the team must stop shipping.

Usage:
    python3 run.py            # ~3 minutes, no model needed
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
PROJ = os.path.dirname(HERE)
for d in ("16-static-vs-continuous", "18-chunked-prefill-simulator",
          "59-metric-instrumentation"):
    sys.path.insert(0, os.path.join(PROJ, d))

import obslib as O                                   # noqa: E402
from simlib import SimRequest, simulate, pct         # noqa: E402

OUT = os.path.join(HERE, "outputs")

DAYS = 30
HOURS = 24
WINDOW_S = 300.0          # simulated seconds standing in for one hour
SLOTS = 24
BASE_RATE = 0.30           # req/s at the daily peak on a weekday
TIMEOUT_S = 90.0           # a request slower than this is an ERROR, not slow
LAT_TARGET_S = 4.0         # the latency SLI threshold
AVAIL_SLO = 0.999          # availability objective
LAT_SLO = 0.99             # 99% of requests must beat LAT_TARGET_S


# Three incidents, each a different failure mode, on purpose at different
# times of day so the traffic-weighting question in section B has teeth.
INCIDENTS = [
    dict(name="replica loss (peak)", day=8, hours=(13, 17),
         effect=dict(max_running=6)),
    dict(name="traffic spike", day=17, hours=(19, 22),
         effect=dict(rate_mult=2.4)),
    dict(name="slow degradation (night)", day=23, hours=(1, 7),
         effect=dict(slow=1.9)),
]


def diurnal(day, hour):
    """A weekday peak in the afternoon, a quiet night, a lighter weekend."""
    weekday = (day % 7) not in (5, 6)
    shape = 0.25 + 0.75 * max(0.0, math.sin(math.pi * (hour - 5) / 16.0))
    return BASE_RATE * shape * (1.0 if weekday else 0.55)


def window_trace(rate, seed, dur=WINDOW_S):
    rng = random.Random(seed)
    reqs, t = [], 0.0
    i = 0
    while True:
        t += rng.expovariate(max(rate, 1e-3))
        if t > dur:
            break
        p = int(min(1024, max(16, rng.lognormvariate(math.log(200), 0.70))))
        o = int(min(300, max(8, rng.lognormvariate(math.log(90), 0.70))))
        reqs.append(SimRequest(rid=i, arrive=t, prompt_len=p, out_len=o))
        i += 1
    return reqs


def simulate_month(cost):
    rows = []
    for day in range(DAYS):
        for hour in range(HOURS):
            rate = diurnal(day, hour)
            mr, slow = SLOTS, 1.0
            incident = None
            for inc in INCIDENTS:
                if inc["day"] == day and inc["hours"][0] <= hour < inc["hours"][1]:
                    incident = inc["name"]
                    rate *= inc["effect"].get("rate_mult", 1.0)
                    mr = inc["effect"].get("max_running", mr)
                    slow = inc["effect"].get("slow", 1.0)
            c = cost
            if slow != 1.0:
                from simlib import CostModel
                c = CostModel(base=cost.base * slow,
                              per_decode=cost.per_decode * slow,
                              per_prefill=cost.per_prefill * slow,
                              per_key_read=cost.per_key_read * slow)
            reqs = window_trace(rate, seed=day * 100 + hour)
            simulate(reqs, c, max_running=mr, token_budget=4096)
            done = [r for r in reqs if r.end_t is not None]
            e2es = [r.e2e for r in done]
            ttfts = [r.ttft for r in done]
            errors = sum(1 for e in e2es if e > TIMEOUT_S)
            slow_n = sum(1 for t in ttfts if t > LAT_TARGET_S)
            rows.append(dict(
                day=day, hour=hour, incident=incident, rate=rate,
                n=len(done), errors=errors, slow=slow_n,
                ttft_p95=pct(ttfts, 95) if ttfts else 0.0,
                ttft_p50=pct(ttfts, 50) if ttfts else 0.0,
                e2e_p99=pct(e2es, 99) if e2es else 0.0))
    return rows


# ---------------------------------------------------------------------------
# SLI arithmetic
# ---------------------------------------------------------------------------


def request_based(rows, field, slo):
    """good events / all events, the definition Google recommends."""
    tot = sum(r["n"] for r in rows)
    bad = sum(r[field] for r in rows)
    sli = 1.0 - bad / tot
    return dict(kind="request-based", total=tot, bad=bad, sli=sli,
                budget_total=(1 - slo) * tot,
                budget_used=bad / ((1 - slo) * tot))


def time_based(rows, field, slo, bad_window_ratio=0.01):
    """A window is bad if MORE THAN `bad_window_ratio` of its requests were
    bad. Then count bad windows. This is what "minutes of downtime" means."""
    bad_w = sum(1 for r in rows
                if r["n"] and r[field] / r["n"] > bad_window_ratio)
    sli = 1.0 - bad_w / len(rows)
    return dict(kind="time-based", total=len(rows), bad=bad_w, sli=sli,
                budget_total=(1 - slo) * len(rows),
                budget_used=bad_w / ((1 - slo) * len(rows)))


def burn_rate(rows, i, n_windows, field, slo):
    """How many times faster than 'even' the budget is being spent.

    A burn rate of 1 spends exactly the whole month's budget in exactly a
    month. A burn rate of 14.4 spends 2% of it in one hour."""
    w = rows[max(0, i - n_windows + 1):i + 1]
    tot = sum(r["n"] for r in w)
    if not tot:
        return 0.0
    return (sum(r[field] for r in w) / tot) / (1 - slo)


def alerting(rows, field, slo):
    """Two alert designs over the same month."""
    naive, multi = [], []
    for i, r in enumerate(rows):
        if r["n"] and r[field] / r["n"] > (1 - slo):
            naive.append(i)
        # Google SRE's multi-window multi-burn-rate rule, mapped onto hours:
        #   page if 1h burn >= 14.4 AND 6h burn >= 6
        # The long window says "this is real", the short one says "this is now".
        if burn_rate(rows, i, 1, field, slo) >= 14.4 and \
                burn_rate(rows, i, 6, field, slo) >= 6.0:
            multi.append(i)
    return naive, multi


def incident_windows(rows):
    spans = []
    for inc in INCIDENTS:
        idx = [i for i, r in enumerate(rows) if r["incident"] == inc["name"]]
        spans.append((inc["name"], min(idx), max(idx)))
    return spans


def score_alerts(fired, spans, grace=2):
    """Time to detect each incident, and how many pages were noise."""
    out, matched = [], set()
    for name, lo, hi in spans:
        hits = [i for i in fired if lo <= i <= hi + grace]
        matched.update(hits)
        out.append(dict(incident=name, detected=bool(hits),
                        detect_hours=(min(hits) - lo) if hits else None,
                        pages=len(hits)))
    return dict(per_incident=out, total_pages=len(fired),
                false_pages=len([i for i in fired if i not in matched]))


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cost, src = O.load_cost_model()
    print(f"[cost model] {src}")
    rows = simulate_month(cost)
    print(f"[A] {len(rows)} hourly windows, "
          f"{sum(r['n'] for r in rows):,} requests [{time.time()-t0:.0f}s]")

    avail_req = request_based(rows, "errors", AVAIL_SLO)
    avail_time = time_based(rows, "errors", AVAIL_SLO)
    lat_req = request_based(rows, "slow", LAT_SLO)
    lat_time = time_based(rows, "slow", LAT_SLO, bad_window_ratio=1 - LAT_SLO)
    print(f"[B] availability SLI: request {avail_req['sli']*100:.3f}% "
          f"({avail_req['budget_used']*100:.0f}% of budget) vs time "
          f"{avail_time['sli']*100:.3f}% ({avail_time['budget_used']*100:.0f}%)")

    # --- C. the percentile that cannot be tracked --------------------------
    all_p95 = pct([r["ttft_p95"] for r in rows], 95)
    month_p95_of_windows = sum(r["ttft_p95"] * r["n"] for r in rows) / \
        sum(r["n"] for r in rows)
    worst_day = max(range(DAYS), key=lambda d: sum(
        r["slow"] for r in rows if r["day"] == d))
    c = dict(
        window_p95_median=pct([r["ttft_p95"] for r in rows], 50),
        window_p95_p95=all_p95,
        traffic_weighted_mean_of_window_p95=month_p95_of_windows,
        frac_requests_under_target=lat_req["sli"],
        latency_slo=LAT_SLO, target_s=LAT_TARGET_S,
        windows_meeting_p95_target=sum(1 for r in rows
                                       if r["ttft_p95"] <= LAT_TARGET_S) / len(rows),
        worst_day=worst_day,
    )
    print(f"[C] median hourly p95 = {c['window_p95_median']:.2f}s "
          f"(target {LAT_TARGET_S}s), yet only "
          f"{c['frac_requests_under_target']*100:.2f}% of requests beat it "
          f"(SLO {LAT_SLO*100:.0f}%)")

    # --- D. alerting -------------------------------------------------------
    spans = incident_windows(rows)
    naive, multi = alerting(rows, "slow", LAT_SLO)
    d = dict(naive=score_alerts(naive, spans),
             multiwindow=score_alerts(multi, spans),
             naive_fired=naive, multi_fired=multi,
             spans=[dict(name=n, lo=lo, hi=hi) for n, lo, hi in spans])
    print(f"[D] naive: {d['naive']['total_pages']} pages "
          f"({d['naive']['false_pages']} false); multi-window: "
          f"{d['multiwindow']['total_pages']} pages "
          f"({d['multiwindow']['false_pages']} false)")

    # --- E. the budget over time -------------------------------------------
    budget = (1 - LAT_SLO) * sum(r["n"] for r in rows)
    left, curve = budget, []
    for r in rows:
        left -= r["slow"]
        curve.append(left / budget)
    exhausted = next((i for i, v in enumerate(curve) if v < 0), None)
    e = dict(budget_events=budget, curve=curve,
             exhausted_window=exhausted,
             exhausted_day=exhausted // HOURS if exhausted is not None else None,
             final_remaining=curve[-1],
             minutes_of_a_real_month=(1 - AVAIL_SLO) * 30 * 24 * 60)
    print(f"[E] latency budget exhausted on day "
          f"{e['exhausted_day']}, ending at {curve[-1]*100:.0f}%")

    res = dict(config=dict(days=DAYS, hours=HOURS, window_s=WINDOW_S,
                           slots=SLOTS, base_rate=BASE_RATE,
                           timeout_s=TIMEOUT_S, lat_target_s=LAT_TARGET_S,
                           avail_slo=AVAIL_SLO, lat_slo=LAT_SLO,
                           cost_model=src, incidents=INCIDENTS),
               rows=rows,
               B=dict(avail_request=avail_req, avail_time=avail_time,
                      lat_request=lat_req, lat_time=lat_time),
               C=c, D=d, E=e, wall_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print(f"total {time.time()-t0:.0f}s")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows, B, C, D, E = res["rows"], res["B"], res["C"], res["D"], res["E"]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    x = np.arange(len(rows)) / HOURS

    p = ax[0][0]
    p.plot(x, [r["rate"] for r in rows], lw=.8, color="#2980b9")
    for s in D["spans"]:
        p.axvspan(s["lo"] / HOURS, s["hi"] / HOURS, color="#c0392b", alpha=.25)
    p.set_xlabel("day"), p.set_ylabel("offered rate (req/s)")
    p.set_title("A. A month of traffic\n"
                "diurnal peak, lighter weekends, three shaded incidents")

    p = ax[0][1]
    p.plot(x, [r["ttft_p95"] for r in rows], lw=.8, color="#c0392b")
    p.axhline(res["config"]["lat_target_s"], color="k", ls=":",
              label=f"SLI threshold {res['config']['lat_target_s']} s")
    for s in D["spans"]:
        p.axvspan(s["lo"] / HOURS, s["hi"] / HOURS, color="#c0392b", alpha=.2)
    p.set_yscale("log")
    p.set_xlabel("day"), p.set_ylabel("hourly TTFT p95 (s, log)")
    p.legend(fontsize=8)
    p.set_title(f"C. 'p95 under target' is almost always true\n"
                f"{C['windows_meeting_p95_target']*100:.1f}% of hours pass it, "
                f"while the request SLI lands on the SLO line")

    p = ax[0][2]
    labels = ["availability\nrequest-based", "availability\ntime-based",
              "latency\nrequest-based", "latency\ntime-based"]
    vals = [B["avail_request"]["budget_used"] * 100,
            B["avail_time"]["budget_used"] * 100,
            B["lat_request"]["budget_used"] * 100,
            B["lat_time"]["budget_used"] * 100]
    cols = ["#27ae60" if v <= 100 else "#c0392b" for v in vals]
    p.bar(labels, vals, color=cols)
    p.axhline(100, color="k", ls="--", label="budget exhausted")
    for i, v in enumerate(vals):
        p.text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
    p.set_ylabel("% of the month's error budget spent")
    p.set_yscale("log")
    p.legend(fontsize=8)
    p.set_title("B. Four ways to score the same month\n"
                "the definition decides whether you shipped")

    p = ax[1][0]
    p.plot(x, [v * 100 for v in E["curve"]], color="#8e44ad", lw=1.2)
    p.axhline(0, color="#c0392b", ls="--")
    if E["exhausted_window"] is not None:
        p.axvline(E["exhausted_window"] / HOURS, color="#c0392b", ls=":",
                  label=f"exhausted on day {E['exhausted_day']}")
        p.legend(fontsize=8)
    p.set_xlabel("day"), p.set_ylabel("latency budget remaining (%)")
    p.set_title(f"E. Spending the budget\n"
                f"ends the month at {E['final_remaining']*100:.0f}%")

    p = ax[1][1]
    naive, multi = D["naive_fired"], D["multi_fired"]
    p.eventplot([np.array(naive) / HOURS], colors=["#c0392b"], lineoffsets=1,
                linelengths=.7)
    p.eventplot([np.array(multi) / HOURS], colors=["#27ae60"], lineoffsets=0,
                linelengths=.7)
    for s in D["spans"]:
        p.axvspan(s["lo"] / HOURS, s["hi"] / HOURS, color="#95a5a6", alpha=.4)
    p.set_yticks([0, 1], ["multi-window\nburn rate", "naive\nthreshold"],
                 fontsize=8)
    p.set_xlabel("day")
    p.set_title(f"D. Pages over the month\n"
                f"naive {D['naive']['total_pages']} "
                f"({D['naive']['false_pages']} false) vs burn-rate "
                f"{D['multiwindow']['total_pages']} "
                f"({D['multiwindow']['false_pages']} false)")

    p = ax[1][2]
    inc = [i["incident"] for i in D["naive"]["per_incident"]]
    xs = np.arange(len(inc))
    nd = [i["pages"] for i in D["naive"]["per_incident"]]
    md = [i["pages"] for i in D["multiwindow"]["per_incident"]]
    p.bar(xs - .2, nd, .4, label="naive threshold", color="#c0392b")
    p.bar(xs + .2, md, .4, label="multi-window burn rate", color="#27ae60")
    for i, (a_, b_) in enumerate(zip(nd, md)):
        if a_ == 0:
            p.text(i - .2, .1, "MISSED", ha="center", va="bottom",
                   rotation=90, fontsize=8, color="#c0392b")
        if b_ == 0:
            p.text(i + .2, .1, "MISSED", ha="center", va="bottom",
                   rotation=90, fontsize=8, color="#27ae60")
    p.set_xticks(xs, [i.replace(" (", "\n(") for i in inc], fontsize=7)
    p.set_ylabel("pages raised during the incident")
    p.legend(fontsize=8)
    p.set_title("D2. What the quiet alert costs\n"
                "the burn-rate rule misses the 3 a.m. one entirely")

    fig.suptitle("Error budgets: the same month, scored four ways",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "budget.png"), dpi=118)
    print("wrote", os.path.join(OUT, "budget.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    if ap.parse_args().plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main()
