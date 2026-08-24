"""Project 65 -- Load shedding: saying no, on purpose, to the right people.

Under 2x overload every policy that tries to serve everybody fails everybody.
The only thing that keeps a promise to *anyone* is refusing somebody, and this
project measures which way of refusing works.

Two classes of traffic -- **gold** (30%, promised p95 TTFT under 3.54 s, the
SLO project 61 located) and **bronze** (70%, best effort) -- offered at twice
the rate the engine can serve inside that promise.

  A. Do nothing. What 2x overload does to both classes.
  B. Reorder the queue by priority. Enough?
  C. Preempt: throw a running bronze request out for a waiting gold one.
  D. Shed: refuse bronze at the door once the backlog is long.
  E. Shed by predicted wait, per class -- refuse anyone you already know you
     will fail.
  F. Two controls: shedding without priority, and a plain rate limiter.
  G. Goodput and wasted work: what the engine spent on answers nobody could
     use.

Usage:
    python3 run.py            # ~2 minutes, no model needed
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
for d in ("16-static-vs-continuous", "18-chunked-prefill-simulator",
          "59-metric-instrumentation"):
    sys.path.insert(0, os.path.join(PROJ, d))

import obslib as O                                     # noqa: E402
from simlib import SimRequest, simulate, pct           # noqa: E402

OUT = os.path.join(HERE, "outputs")

N = 700
SLOTS = 24
P_MED, P_SIGMA, P_MAX = 200, 0.70, 1024
O_MED, O_SIGMA, O_MAX = 90, 0.70, 300

SAFE_RATE = 0.35            # project 61: the SLO breaks just past here
OVERLOAD = 2.0
RATE = SAFE_RATE * OVERLOAD

GOLD_FRAC = 0.30
GOLD_SLO = 3.54             # project 61's medium TTFT p95 target
BRONZE_SLO = 20.0           # best effort, but not unbounded


def trace(seed=0, rate=RATE, n=N):
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        p = int(min(P_MAX, max(16, rng.lognormvariate(math.log(P_MED), P_SIGMA))))
        o = int(min(O_MAX, max(8, rng.lognormvariate(math.log(O_MED), O_SIGMA))))
        gold = rng.random() < GOLD_FRAC
        reqs.append(SimRequest(rid=i, arrive=t, prompt_len=p, out_len=o,
                               priority=0 if gold else 1,
                               deadline=t + (GOLD_SLO if gold else BRONZE_SLO)))
    return reqs


class Shedder:
    """An admission rule that can see the backlog, the way a real one can.

    `simulate` hands the rule only the request and the clock, so the backlog
    is recovered from the trace itself: everything that has arrived and has
    neither started nor been refused. A real server reads this straight off
    its own queue -- it is the `llm_num_requests_waiting` gauge from project
    59.
    """

    def __init__(self, reqs, cost, policy, backlog_max=8, service_s=2.0,
                 rate_limit=None, burst=8.0):
        self.reqs = sorted(reqs, key=lambda r: r.arrive)
        self.i = 0
        self.policy = policy
        self.backlog_max = backlog_max
        self.service_s = service_s
        self.rate_limit = rate_limit
        self.burst = burst
        self.tokens = burst                      # token bucket, starts full
        self.last = 0.0
        self.log = []

    def backlog(self, now):
        while self.i < len(self.reqs) and self.reqs[self.i].arrive <= now:
            self.i += 1
        return sum(1 for r in self.reqs[:self.i]
                   if r.admit_t is None and not r.rejected)

    def __call__(self, r, now, kv_used, kv_cap):
        p = self.policy
        if p == "none":
            return "admit"
        if p == "rate_limit":
            # A token bucket: refill at theservice rate, spend one token per
            # admission. Blind to how long the queue actually is.
            self.tokens = min(self.burst,
                              self.tokens + (now - self.last) * self.rate_limit)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return "admit"
            return "reject"
        b = self.backlog(now)
        if p == "shed_bronze":
            return "reject" if (r.priority > 0 and b > self.backlog_max) \
                else "admit"
        if p == "shed_uniform":
            return "reject" if b > self.backlog_max else "admit"
        if p == "shed_wait_bronze":
            if r.priority == 0:
                return "admit"
            return "reject" if b * self.service_s > BRONZE_SLO else "admit"
        if p == "shed_wait":
            # Refuse anyone whose predicted wait already exceeds their own
            # deadline. Gold gets a tight budget, bronze a loose one.
            wait = b * self.service_s
            budget = GOLD_SLO if r.priority == 0 else BRONZE_SLO
            return "reject" if wait > budget else "admit"
        return "admit"


ARMS = [
    ("A. no shedding (FCFS)", dict(policy="none", order="fcfs")),
    ("B. priority queue", dict(policy="none", order="priority")),
    ("C. priority + preemption", dict(policy="none", order="priority",
                                      preempt_for_priority=True)),
    ("D. shed bronze on backlog", dict(policy="shed_bronze", order="priority")),
    ("E. shed bronze on predicted wait", dict(policy="shed_wait_bronze",
                                              order="priority")),
    ("F. shed bronze + preemption", dict(policy="shed_bronze",
                                         order="priority",
                                         preempt_for_priority=True)),
    ("G1. shed everyone (control)", dict(policy="shed_uniform", order="fcfs")),
    ("G2. rate limit at capacity", dict(policy="rate_limit", order="priority")),
]


def rate_limit_ingress(reqs, rate, burst=8.0):
    """A token bucket at the DOOR, applied before anything reaches the queue.

    This has to happen at arrival time, not at admission time. A limiter
    consulted when a slot frees is consulted at exactly the rate the engine
    can already serve, so its bucket never empties and it refuses nobody --
    a rate limiter behind the queue cannot shed anything.
    """
    tokens, last, n = burst, 0.0, 0
    for r in sorted(reqs, key=lambda r: r.arrive):
        tokens = min(burst, tokens + (r.arrive - last) * rate)
        last = r.arrive
        if tokens >= 1.0:
            tokens -= 1.0
        else:
            r.rejected = True
            n += 1
    return n


def measure(reqs, stats, label, service_s):
    out = dict(label=label, makespan=stats["makespan_s"],
               preemptions=stats["preemptions"],
               wasted_decode_tokens=stats["wasted_decode_tokens"],
               util=stats["util"])
    for cls, pri in (("gold", 0), ("bronze", 1)):
        grp = [r for r in reqs if r.priority == pri]
        done = [r for r in grp if r.end_t is not None]
        rej = [r for r in grp if r.rejected]
        slo = GOLD_SLO if pri == 0 else BRONZE_SLO
        met = [r for r in done if r.ttft is not None and r.ttft <= slo]
        out[cls] = dict(
            offered=len(grp), completed=len(done), rejected=len(rej),
            reject_rate=len(rej) / len(grp),
            ttft_p50=pct([r.ttft for r in done], 50) if done else float("nan"),
            ttft_p95=pct([r.ttft for r in done], 95) if done else float("nan"),
            met_slo=len(met),
            # the honest denominator: everyone who asked, including refusals
            attainment=len(met) / len(grp),
            attainment_of_served=len(met) / max(1, len(done)))
    served_in_slo = out["gold"]["met_slo"] + out["bronze"]["met_slo"]
    out["goodput_req_s"] = served_in_slo / stats["makespan_s"]
    out["throughput_req_s"] = (out["gold"]["completed"] +
                               out["bronze"]["completed"]) / stats["makespan_s"]
    # engine seconds spent on answers that arrived too late to be useful
    late = [r for r in reqs if r.end_t is not None
            and r.ttft is not None
            and r.ttft > (GOLD_SLO if r.priority == 0 else BRONZE_SLO)]
    out["late_completed"] = len(late)
    out["wasted_share"] = len(late) / max(1, out["gold"]["completed"] +
                                          out["bronze"]["completed"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cost, src = O.load_cost_model()
    print(f"[cost model] {src}")

    base = trace()
    # what one request costs the engine on average -- the shedder's estimator
    service_s = sum(cost.request_work(r.prompt_len, r.out_len)
                    for r in base) / len(base)
    capacity = 1.0 / service_s
    print(f"[setup] mean service {service_s:.2f} s -> capacity "
          f"{capacity:.2f} req/s; offering {RATE:.2f} req/s "
          f"({RATE/capacity:.2f}x)")

    rows = []
    for label, cfg in ARMS:
        reqs = copy.deepcopy(base)
        sh = Shedder(reqs, cost, cfg["policy"], backlog_max=8,
                     service_s=service_s, rate_limit=capacity)
        if cfg["policy"] == "rate_limit":
            rate_limit_ingress(reqs, capacity)
            reqs = [r for r in reqs if not r.rejected] + \
                   [r for r in reqs if r.rejected]
        st = simulate([r for r in reqs if not r.rejected], cost,
                      max_running=SLOTS, token_budget=4096,
                      order=cfg.get("order", "fcfs"),
                      admission=None if cfg["policy"] in ("none",
                                                           "rate_limit")
                      else sh,
                      preempt_for_priority=cfg.get("preempt_for_priority",
                                                   False))
        r = measure(reqs, st, label, service_s)
        rows.append(r)
        print(f"  {label:34s} gold p95 {r['gold']['ttft_p95']:7.2f}s "
              f"attain {r['gold']['attainment']*100:5.1f}%  "
              f"bronze rej {r['bronze']['reject_rate']*100:5.1f}%  "
              f"goodput {r['goodput_req_s']:.3f}/s")

    # --- how hard should we shed? sweep the backlog threshold --------------
    sweep = []
    for bmax in (2, 4, 6, 8, 12, 16, 24, 40, 1000):
        reqs = copy.deepcopy(base)
        sh = Shedder(reqs, cost, "shed_bronze", backlog_max=bmax,
                     service_s=service_s)
        st = simulate(reqs, cost, max_running=SLOTS, token_budget=4096,
                      order="priority", admission=sh,
                      preempt_for_priority=True)
        r = measure(reqs, st, f"backlog_max={bmax}", service_s)
        r["backlog_max"] = bmax
        sweep.append(r)
    print("[sweep] " + "  ".join(
        f"{s['backlog_max']}:{s['gold']['attainment']*100:.0f}%/"
        f"{s['bronze']['reject_rate']*100:.0f}%" for s in sweep))

    # --- how much overload can priority shedding absorb? -------------------
    ladder = []
    for mult in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        for pol, order in (("none", "fcfs"), ("shed_bronze", "priority")):
            reqs = trace(seed=1, rate=SAFE_RATE * mult)
            sh = Shedder(reqs, cost, pol, backlog_max=8, service_s=service_s)
            st = simulate(reqs, cost, max_running=SLOTS, token_budget=4096,
                          order=order,
                          admission=None if pol == "none" else sh,
                          preempt_for_priority=(pol != "none"))
            r = measure(reqs, st, f"{mult}x {pol}", service_s)
            r["mult"], r["policy"] = mult, pol
            ladder.append(r)
    print("[ladder] " + "  ".join(
        f"{l['mult']}x/{l['policy'][:4]}:{l['gold']['attainment']*100:.0f}%"
        for l in ladder))

    res = dict(config=dict(n=N, slots=SLOTS, rate=RATE, safe_rate=SAFE_RATE,
                           overload=OVERLOAD, gold_frac=GOLD_FRAC,
                           gold_slo=GOLD_SLO, bronze_slo=BRONZE_SLO,
                           service_s=service_s, capacity=capacity,
                           cost_model=src),
               arms=rows, sweep=sweep, ladder=ladder,
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print(f"total {time.time()-t0:.0f}s")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms, sweep, ladder = res["arms"], res["sweep"], res["ladder"]
    short = [a["label"].split(". ", 1)[1].replace(" ", "\n") for a in arms]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    p = ax[0][0]
    x = np.arange(len(arms))
    p.bar(x - .2, [a["gold"]["attainment"] * 100 for a in arms], .4,
          label="gold", color="#d4ac0d")
    p.bar(x + .2, [a["bronze"]["attainment"] * 100 for a in arms], .4,
          label="bronze", color="#7f8c8d")
    p.axhline(95, color="#c0392b", ls=":", label="95% target")
    p.set_xticks(x, short, fontsize=6)
    p.set_ylabel("% of OFFERED requests served inside their SLO")
    p.legend(fontsize=8)
    p.set_title(f"A-F. {res['config']['overload']:.0f}x overload: who gets "
                f"served?\ndenominator includes refusals")

    p = ax[0][1]
    p.bar(x, [a["gold"]["ttft_p95"] for a in arms], color="#d4ac0d")
    p.axhline(res["config"]["gold_slo"], color="#c0392b", ls="--",
              label=f"gold SLO {res['config']['gold_slo']} s")
    p.set_yscale("log")
    p.set_xticks(x, short, fontsize=6)
    p.set_ylabel("gold TTFT p95 (s, log)")
    p.legend(fontsize=8)
    p.set_title("The promise, kept or not\n"
                "reordering the queue is not enough; refusing is")

    p = ax[0][2]
    p.bar(x, [a["goodput_req_s"] for a in arms], color="#27ae60")
    for i, a in enumerate(arms):
        p.text(i, a["goodput_req_s"], f"{a['goodput_req_s']:.2f}",
               ha="center", va="bottom", fontsize=7)
    p.set_xticks(x, short, fontsize=6)
    p.set_ylabel("goodput: requests/s served inside their SLO")
    p.set_title("G. Refusing work raises useful output\n"
                f"best {max(a['goodput_req_s'] for a in arms):.2f}/s vs "
                f"{arms[0]['goodput_req_s']:.2f}/s doing nothing")

    p = ax[1][0]
    b = [s["backlog_max"] for s in sweep[:-1]] + [sweep[-1]["backlog_max"]]
    p.plot(b, [s["gold"]["attainment"] * 100 for s in sweep], "o-",
           color="#d4ac0d", label="gold served in SLO")
    p.plot(b, [s["bronze"]["reject_rate"] * 100 for s in sweep], "s-",
           color="#c0392b", label="bronze refused")
    p.plot(b, [s["bronze"]["attainment"] * 100 for s in sweep], "^-",
           color="#7f8c8d", label="bronze served in SLO")
    p.set_xscale("log")
    p.set_xlabel("backlog threshold at which bronze is refused")
    p.set_ylabel("%")
    p.legend(fontsize=8)
    p.set_title("D2. How hard to shed\n"
                "the knob trades gold's promise against bronze's service")

    p = ax[1][1]
    for pol, col, lab in (("none", "#c0392b", "no shedding"),
                          ("shed_bronze", "#27ae60",
                           "shed bronze + priority + preemption")):
        rows = [l for l in ladder if l["policy"] == pol]
        p.plot([l["mult"] for l in rows],
               [l["gold"]["attainment"] * 100 for l in rows], "o-",
               color=col, label=lab)
    p.axhline(95, color="k", ls=":")
    p.set_xlabel("offered load, as a multiple of the SLO-safe rate")
    p.set_ylabel("gold served inside its SLO (%)")
    p.legend(fontsize=8)
    p.set_title("How much overload can shedding absorb?\n"
                "the answer is finite, and worth knowing before the incident")

    p = ax[1][2]
    p.bar(x, [a["wasted_share"] * 100 for a in arms], color="#8e44ad")
    for i, a in enumerate(arms):
        p.text(i, a["wasted_share"] * 100, f"{a['late_completed']}",
               ha="center", va="bottom", fontsize=7)
    p.set_xticks(x, short, fontsize=6)
    p.set_ylabel("% of completed answers that were already too late")
    p.set_title("G2. Work the engine did for nobody\n"
                "label = number of such requests")

    fig.suptitle("Load shedding: under overload the only way to keep a "
                 "promise is to break a different one", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "shedding.png"), dpi=118)
    print("wrote", os.path.join(OUT, "shedding.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    if ap.parse_args().plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main()
