"""Project 45 -- N complete model copies behind a load balancer.

  A. Scale-out: the same workload against 1, 2 and 4 replicas, plus the
     scale-UP control (one replica given all 8 threads).
  B. Round-robin vs least-outstanding on heterogeneous requests.
  C. Who actually served what: per-replica balance under each policy.

    python3 run.py           # ~9 minutes; starts real server processes
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

from fleetlib import (Fleet, LeastOutstanding, RoundRobin,  # noqa: E402
                      run_load, summarize)

F = {}
CONC = 8


def uniform_workload(n, plen=64, max_new=24, seed=0):
    rng = random.Random(seed)
    return [{"rid": i, "ids": [rng.randrange(1000, 12000) for _ in range(plen)],
             "max_new": max_new} for i in range(n)]


def hetero_workload(n, seed=1):
    """Lognormal prompt AND output lengths -- the traffic mix where a router
    that watches the fleet should beat one that counts to four."""
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        plen = int(min(192, max(16, rng.lognormvariate(math.log(48), 0.8))))
        onew = int(min(64, max(6, rng.lognormvariate(math.log(20), 0.8))))
        reqs.append({"rid": i, "ids": [rng.randrange(1000, 12000)
                                       for _ in range(plen)], "max_new": onew})
    return reqs


def by_replica(records):
    d = {}
    for r in records:
        if r.get("replica"):
            d[r["replica"]] = d.get(r["replica"], 0) + 1
    return d


def main():
    wl = uniform_workload(24)

    # ---- fleet of 4, reused for the 1/2/4 subsets and for section B --------
    fleet = Fleet(4, threads=2, log_dir=OUT)
    try:
        fleet.wait_ready()
        rows = []
        for k in (1, 2, 4):
            recs, wall, _ = asyncio.run(run_load(
                fleet.urls[:k], wl, RoundRobin(k), concurrency=CONC))
            s = summarize(recs, wall, label=f"{k} replica(s) x 2 threads")
            s["replicas"] = k
            s["by_replica"] = by_replica(recs)
            rows.append(s)
            print(f"[A] {k} replicas: {s['throughput_tok_s']} tok/s  "
                  f"ttft p50 {s['ttft_p50_s']} s  p99 {s['ttft_p99_s']} s",
                  flush=True)
        F["scaling"] = rows

        # ---- B. routing policy on heterogeneous traffic --------------------
        wlh = hetero_workload(32)
        pol = {}
        for name, mk in (("round_robin", lambda: RoundRobin(4)),
                         ("least_outstanding", lambda: LeastOutstanding(4))):
            recs, wall, _ = asyncio.run(run_load(
                fleet.urls, wlh, mk(), concurrency=CONC))
            s = summarize(recs, wall, label=name)
            s["by_replica"] = by_replica(recs)
            s["queue_ms_p99"] = max(r.get("queue_ms", 0) for r in recs)
            pol[name] = s
            print(f"[B] {name}: ttft p99 {s['ttft_p99_s']} s  "
                  f"e2e p99 {s['e2e_p99_s']} s  share {s['by_replica']}",
                  flush=True)
        F["policy"] = pol
    finally:
        fleet.stop()

    # ---- A'. the scale-UP control: one replica, all 8 threads --------------
    # The fleet above is gone by now (Fleet.stop blocks until it is), so this
    # replica has the whole machine -- which is the point of the control.
    fleet1 = Fleet(1, threads=8, base_port=8730)
    try:
        fleet1.wait_ready()
        recs, wall, _ = asyncio.run(run_load(
            fleet1.urls, wl, RoundRobin(1), concurrency=CONC))
        s = summarize(recs, wall, label="1 replica x 8 threads")
        F["scale_up"] = s
        print(f"[A'] 1x8 threads: {s['throughput_tok_s']} tok/s  "
              f"ttft p50 {s['ttft_p50_s']} s", flush=True)
    finally:
        fleet1.stop()

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(F, f, indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    rows = f["scaling"]
    xs = [r["replicas"] for r in rows]
    tps = [r["throughput_tok_s"] for r in rows]
    ax[0].bar([str(x) for x in xs], tps, color="#2471a3")
    ax[0].bar(["1 (8 thr)"], [f["scale_up"]["throughput_tok_s"]], color="#f39c12")
    ideal = [tps[0] * x for x in xs]
    ax[0].plot(range(len(xs)), ideal, "k--", lw=1, label="linear from 1 replica")
    for i, t in enumerate(tps):
        ax[0].text(i, t, f"{t:.1f}", ha="center", va="bottom", fontsize=8)
    ax[0].text(3, f["scale_up"]["throughput_tok_s"],
               f"{f['scale_up']['throughput_tok_s']:.1f}", ha="center",
               va="bottom", fontsize=8)
    ax[0].set_xlabel("replicas")
    ax[0].set_ylabel("output tokens / s")
    ax[0].set_title("A. scale out vs scale up")
    ax[0].legend(fontsize=8)

    w = 0.35
    for j, (key, col, lab) in enumerate((("ttft_p50_s", "#2471a3", "TTFT p50"),
                                         ("ttft_p99_s", "#c0392b", "TTFT p99"))):
        ax[1].bar([i + (j - .5) * w for i in range(len(rows))],
                  [r[key] for r in rows], w, color=col, label=lab)
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels([r["replicas"] for r in rows])
    ax[1].set_xlabel("replicas")
    ax[1].set_ylabel("seconds")
    ax[1].set_title("B. waiting is what replicas buy")
    ax[1].legend(fontsize=8)

    pol = f["policy"]
    names = ["round_robin", "least_outstanding"]
    for j, key in enumerate(("ttft_p99_s", "e2e_p99_s")):
        ax[2].bar([i + (j - .5) * w for i in range(2)],
                  [pol[n][key] for n in names], w,
                  color=["#c0392b", "#2471a3"][j],
                  label={"ttft_p99_s": "TTFT p99", "e2e_p99_s": "E2E p99"}[key])
    ax[2].set_xticks(range(2))
    ax[2].set_xticklabels(["round-robin", "least-outstanding"])
    ax[2].set_ylabel("seconds")
    ax[2].set_title("C. heterogeneous traffic: the policy gap")
    ax[2].legend(fontsize=8)

    for j, n in enumerate(names):
        shares = pol[n]["by_replica"]
        ks = sorted(shares)
        ax[3].bar([i + (j - .5) * w for i in range(len(ks))],
                  [shares[k] for k in ks], w,
                  color=["#c0392b", "#2471a3"][j], label=n)
    ax[3].set_xticks(range(4))
    ax[3].set_xticklabels([f"r{i}" for i in range(4)])
    ax[3].set_ylabel("requests served")
    ax[3].set_title("C'. who served what")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "multireplica.png"), dpi=110)
    print("wrote outputs/multireplica.png")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        main()
    plot()
