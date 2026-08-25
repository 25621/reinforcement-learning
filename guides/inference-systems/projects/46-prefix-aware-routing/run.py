"""Project 46 -- route on the prompt's prefix, so the cache is already there.

  A. Eight tenants, four replicas: round-robin vs prefix-hash routing.
     Cache hit rate, TTFT, and how many copies of each prefix the fleet holds.
  B. The failure mode: one hot tenant. Pure hashing piles its whole load on
     one replica; the guide's affinity-with-load-fallback router fixes it.

    python3 run.py           # ~6 minutes; starts real server processes
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "45-vllm-multi-replica"))

from fleetlib import (Fleet, HashRouter, RoundRobin, run_load,  # noqa: E402
                      summarize)

F = {}
N_TENANTS = 8
SYS_LEN = 192      # every tenant's system prompt, in tokens
USER_LEN = 24      # what actually differs per request
MAX_NEW = 16
PREFIX_CAP = 2     # prefixes ONE replica can hold; see the note below

# Why the cache is deliberately small (2 entries against 8 tenants):
#
# A prefix cache lives in GPU memory, next to the weights and the running
# requests' KV. It is always far too small to hold every tenant's prompt --
# that is the normal condition of a multi-tenant server, not an edge case.
# With a cache big enough for all 8 tenants, EVERY policy eventually reaches a
# high hit rate just by copying every prefix onto every replica, and routing
# stops mattering. Capacity is what makes the routing decision real: scatter
# the tenants and each replica must somehow hold all 8; concentrate them and
# each replica holds 2. The ratio of those two numbers is the whole argument
# for prefix-aware routing, and the eviction counters below measure it.


def tenants(seed=3):
    rng = random.Random(seed)
    return [[rng.randrange(1000, 12000) for _ in range(SYS_LEN)]
            for _ in range(N_TENANTS)]


def workload(sys_prompts, n, tenant_of, seed=4):
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        t = tenant_of(i, rng)
        ids = sys_prompts[t] + [rng.randrange(1000, 12000)
                                for _ in range(USER_LEN)]
        reqs.append({"rid": i, "tenant": t, "ids": ids,
                     "prefix_len": SYS_LEN, "max_new": MAX_NEW})
    return reqs


class AffinityWithLoad(HashRouter):
    """The guide's PrefixRouter sketch, made runnable: honor the remembered
    affinity only while that replica is not much busier than the least-busy
    one; otherwise spill to the least-loaded replica and remember IT as the
    new home for this prefix. Affinity as a hint, not a law."""

    def __init__(self, n, key_fn, slack=2):
        super().__init__(n, key_fn)
        self.affinity = {}
        self.slack = slack

    def pick(self, req):
        key = self.key_fn(req)
        r = self.affinity.get(key)
        if r is not None and self.outstanding[r] < min(self.outstanding) + self.slack:
            self.outstanding[r] += 1
            return r
        r = min(range(self.n), key=lambda i: (self.outstanding[i], i))
        self.affinity[key] = r
        self.outstanding[r] += 1
        return r


def prefix_key(req):
    return tuple(req["ids"][: req["prefix_len"]])


def tenants_seen(records):
    """Which distinct tenants did each replica have to serve? This is the
    number a replica's cache has to cover, and the one routing changes."""
    d = {}
    for r in records:
        if r.get("replica") is not None and r.get("tenant") is not None:
            d.setdefault(r["replica"], set()).add(r["tenant"])
    return d


def run_policy(fleet, reqs, router, label, conc=6):
    recs, wall, _ = asyncio.run(run_load(fleet.urls, reqs, router,
                                         concurrency=conc))
    by_rid = {r["rid"]: r for r in reqs}
    for rec in recs:
        rec["tenant"] = by_rid[rec["rid"]]["tenant"]
    s = summarize(recs, wall, label=label)
    hits = sum(1 for r in recs if r.get("reused", 0) > 0)
    s["hit_rate"] = round(hits / len(recs), 3)
    s["prefill_ms_mean"] = round(
        sum(r.get("prefill_ms", 0) for r in recs) / len(recs), 1)
    s["queue_ms_p99"] = round(sorted(r.get("queue_ms", 0) for r in recs)[-1], 1)
    stats = fleet.stats()
    s["fleet_prefix_entries"] = sum(x["prefix_entries"] for x in stats if x)
    s["fleet_evictions"] = sum(x["prefix_evictions"] for x in stats if x)
    s["tenants_per_replica"] = round(
        sum(len(t) for t in tenants_seen(recs).values()) / len(fleet.urls), 2)
    s["fleet_cache_mb"] = round(
        sum(x["cache_bytes"] for x in stats if x) / 1e6, 1)
    s["per_replica"] = {x["name"]: x["done"] for x in stats if x}
    print(f"[{label}] hit {s['hit_rate']:.0%}  ttft p50 {s['ttft_p50_s']} "
          f"p99 {s['ttft_p99_s']}  evictions {s['fleet_evictions']}  "
          f"tenants/replica {s['tenants_per_replica']}", flush=True)
    return s


def main():
    sys_prompts = tenants()
    fleet = Fleet(4, threads=2, log_dir=OUT, prefix_cap=PREFIX_CAP)
    try:
        fleet.wait_ready()

        # ---- A. uniform tenants -----------------------------------------
        # Tenants arrive in RANDOM order. With a round-robin router and a
        # tenant index of `i % 8`, tenant t would land on replica t % 4 every
        # single time -- round-robin would silently BE prefix routing, and the
        # two arms would tie for a reason that has nothing to do with routing.
        wl = workload(sys_prompts, 48, lambda i, rng: rng.randrange(N_TENANTS))
        res = {}
        for label, mk in (("round_robin", lambda: RoundRobin(4)),
                          ("prefix_hash", lambda: HashRouter(4, prefix_key)),
                          ("affinity_load",
                           lambda: AffinityWithLoad(4, prefix_key))):
            fleet.reset()          # cold caches for every policy
            res[label] = run_policy(fleet, wl, mk(), label)
        F["uniform"] = res

        # ---- B. one hot tenant --------------------------------------------
        def hot(i, rng):
            return 0 if rng.random() < 0.6 else rng.randrange(1, N_TENANTS)

        wlh = workload(sys_prompts, 48, hot, seed=5)
        res = {}
        for label, mk in (("round_robin", lambda: RoundRobin(4)),
                          ("prefix_hash", lambda: HashRouter(4, prefix_key)),
                          ("affinity_load",
                           lambda: AffinityWithLoad(4, prefix_key))):
            fleet.reset()
            res[label] = run_policy(fleet, wlh, mk(), "hot_" + label)
        F["hot_tenant"] = res
    finally:
        fleet.stop()

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(F, f, indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    u = f["uniform"]
    names = ["round_robin", "prefix_hash", "affinity_load"]
    short = ["round-robin", "prefix-hash", "affinity\n+load"]
    cols = ["#7f8c8d", "#c0392b", "#2471a3"]
    ax[0].bar(range(3), [100 * u[n]["hit_rate"] for n in names], color=cols)
    for i, n in enumerate(names):
        ax[0].text(i, 100 * u[n]["hit_rate"], f"{u[n]['hit_rate']:.0%}",
                   ha="center", va="bottom")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(short, fontsize=8)
    ax[0].set_ylim(0, 100)
    ax[0].set_ylabel("prefix-cache hit rate, %")
    ax[0].set_title("A. hit rate")

    w = .35
    for j, key, lab, col in ((0, "ttft_p50_s", "TTFT p50", "#2471a3"),
                             (1, "ttft_p99_s", "TTFT p99", "#c0392b")):
        ax[1].bar([i + (j - .5) * w for i in range(3)],
                  [u[n][key] for n in names], w, color=col, label=lab)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(short, fontsize=8)
    ax[1].set_ylabel("seconds")
    ax[1].set_title("A. hits are not the goal; latency is")
    ax[1].legend(fontsize=8)

    ax[2].bar(range(3), [u[n]["fleet_evictions"] for n in names], color=cols)
    for i, n in enumerate(names):
        ax[2].text(i, u[n]["fleet_evictions"],
                   f"{u[n]['fleet_evictions']} evicted\n"
                   f"{u[n]['tenants_per_replica']} tenants/replica",
                   ha="center", va="bottom", fontsize=8)
    ax[2].set_xticks(range(3))
    ax[2].set_xticklabels(short, fontsize=8)
    ax[2].set_ylim(0, max(u[n]["fleet_evictions"] for n in names) * 1.45)
    ax[2].set_ylabel(f"prefix evictions (cache holds {PREFIX_CAP})")
    ax[2].set_title("A. why: cache thrash")

    h = f["hot_tenant"]
    hn = ["round_robin", "prefix_hash", "affinity_load"]
    labels = ["round-robin", "prefix-hash", "affinity+load"]
    vals = [h[n]["ttft_p99_s"] for n in hn]
    ax[3].bar(range(3), vals, color=["#7f8c8d", "#c0392b", "#2471a3"])
    for i, n in enumerate(hn):
        ax[3].text(i, vals[i], f"hit {h[n]['hit_rate']:.0%}", ha="center",
                   va="bottom", fontsize=8)
    ax[3].set_xticks(range(3))
    ax[3].set_xticklabels(labels, fontsize=8)
    ax[3].set_ylim(0, max(vals) * 1.2)
    ax[3].set_ylabel("TTFT p99, s")
    ax[3].set_title("B. one hot tenant (60% of traffic)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "prefix_routing.png"), dpi=110)
    print("wrote outputs/prefix_routing.png")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        main()
    plot()
