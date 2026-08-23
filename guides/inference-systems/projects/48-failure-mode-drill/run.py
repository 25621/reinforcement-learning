"""Project 48 -- kill a replica under load and measure what users felt.

  A. The drill: `kill -9` one of three replicas 12 s into an open-loop load
     test running at ~60% of fleet capacity, under three routing layers --
     naive, retry-only, health+retry.
  B. The gray failure: a replica that answers every health check instantly
     and generates tokens three times too slowly. Liveness cannot see it;
     least-outstanding routing walks around it without being told.

    python3 run.py           # ~8 minutes; starts real server processes
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "45-vllm-multi-replica"))

from fleetlib import (Fleet, LeastOutstanding, RoundRobin, pct,  # noqa: E402
                      run_load, summarize)

F = {}
N_REP = 3
KILL_AT = 12.0        # seconds into the run
RATE = 2.4            # requests per second (open loop)
N_REQ = 72
PLEN, MAX_NEW = 48, 14

# Choosing the arrival rate is the whole design of this drill.
#
# One replica serves roughly 1.35 requests/s here, so three of them cap out
# near 4.0/s. At 2.4/s the fleet runs at about 60% of capacity BEFORE the kill
# and about 90% after it -- busy enough that losing a third of the fleet
# actually hurts the survivors, and not so busy that the queue was already
# exploding beforehand.
#
# A gentler rate makes the drill lie by omission. At 1.1/s (27% utilisation,
# the first version of this project) two replicas absorb the traffic without
# a ripple: the only visible damage is the handful of requests that were on
# the dead replica, and the report reads "failover was free". That conclusion
# is an artifact of testing an idle fleet. Real fleets are sized to be busy,
# which is exactly when losing capacity compounds.


def workload(n=N_REQ, rate=RATE, seed=11):
    """Poisson arrivals in OPEN loop: request i arrives at its own time no
    matter how the fleet is doing. A closed-loop test would quietly throttle
    itself when replicas die -- exactly the wrong instrument for a drill,
    because real users keep arriving during an outage."""
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        reqs.append({"rid": i, "arrive": t, "max_new": MAX_NEW,
                     "ids": [rng.randrange(1000, 12000) for _ in range(PLEN)]})
    return reqs


async def drill(urls, reqs, router, retries, health_interval, kill_fn):
    """Run the load and fire the kill at KILL_AT seconds."""
    alive = set(range(len(urls)))
    t0 = time.perf_counter()

    async def killer():
        await asyncio.sleep(KILL_AT)
        kill_fn()
        return time.perf_counter() - t0

    task = asyncio.create_task(killer())
    recs, wall, hlog = await run_load(
        urls, reqs, router, retries=retries, alive=alive, open_loop=True,
        health_interval=health_interval, health_timeout=1.0, timeout=90.0)
    t_kill = await task
    for h in hlog:
        h["t"] = h["t"] - t0
    return recs, wall, hlog, t_kill


def analyse(recs, wall, hlog, t_kill, label):
    s = summarize(recs, wall, label=label)
    failed = [r for r in recs if not r["ok"]]
    # A request "spans the kill" if it was sent before and ended after it.
    inflight = [r for r in recs
                if r["t_send"] < t_kill < r.get("t_end", 0)]
    s.update({
        "t_kill_s": round(t_kill, 2),
        "failed_rids": sorted(r["rid"] for r in failed),
        "error_kinds": sorted({r.get("error") for r in failed if r.get("error")}),
        "inflight_at_kill": len(inflight),
        "inflight_lost": sum(1 for r in inflight if not r["ok"]),
        "retried": sum(1 for r in recs if r.get("attempt", 0) > 0),
        "detect_s": round(min((h["t"] for h in hlog
                               if not h["alive"]), default=float("nan")) - t_kill, 2)
        if hlog else None,
    })
    # user-visible blast radius: last failure minus the kill
    last_fail = max((r["t_end"] for r in failed), default=None)
    s["error_window_s"] = round(last_fail - t_kill, 2) if last_fail else 0.0
    # latency before vs after the kill (successful requests only)
    before = [r["ttft_s"] for r in recs
              if r["ok"] and r["ttft_s"] and r["t_send"] < t_kill]
    after = [r["ttft_s"] for r in recs
             if r["ok"] and r["ttft_s"] and r["t_send"] >= t_kill]
    s["ttft_p50_before"] = round(pct(before, 50), 3)
    s["ttft_p50_after"] = round(pct(after, 50), 3)
    s["ttft_p99_after"] = round(pct(after, 99), 3)
    s["timeline"] = [{"t": round(r.get("t_end", 0), 2), "ok": r["ok"],
                      "ttft": round(r["ttft_s"], 3) if r["ttft_s"] else None,
                      "target": r.get("target")} for r in recs]
    print(f"[A] {label:14s} errors {s['errors']:2d}/{s['requests']}  "
          f"lost-in-flight {s['inflight_lost']}  detect {s['detect_s']}  "
          f"window {s['error_window_s']} s  ttft p50 "
          f"{s['ttft_p50_before']}->{s['ttft_p50_after']} s", flush=True)
    return s


def section_a():
    reqs = workload()
    arms = [
        ("naive", 0, None),           # no retry, no health check
        ("retry", 1, None),           # retry once, still routes to the corpse
        ("health_retry", 1, 1.0),     # probe every second, drop the dead
    ]
    out = {}
    for label, retries, hi in arms:
        fleet = Fleet(N_REP, threads=2, log_dir=OUT)
        try:
            fleet.wait_ready()
            recs, wall, hlog, t_kill = asyncio.run(drill(
                fleet.urls, reqs, RoundRobin(N_REP), retries, hi,
                lambda: fleet.kill(1)))
            out[label] = analyse(recs, wall, hlog, t_kill, label)
        finally:
            fleet.stop()
    return out


def section_b():
    """One replica with a third of the threads: alive, healthy, and slow."""
    reqs = workload(n=40, rate=2.0, seed=12)
    fleet = Fleet(N_REP, threads=[2, 1, 2], log_dir=OUT)
    out = {}
    try:
        fleet.wait_ready()
        # What a liveness probe sees. Probe each replica several times and
        # keep the best: the first request to any server also pays for opening
        # a connection, and that one-off cost is bigger than the difference we
        # are looking for -- it would make a healthy replica look like the
        # slow one.
        import httpx

        probes = []
        for u in fleet.urls:
            best = 1e9
            for _ in range(4):
                t1 = time.perf_counter()
                httpx.get(u, timeout=5.0)
                best = min(best, time.perf_counter() - t1)
            probes.append(round(best * 1e3, 2))
        out["probe_ms"] = probes
        for label, mk in (("round_robin", lambda: RoundRobin(N_REP)),
                          ("least_outstanding",
                           lambda: LeastOutstanding(N_REP))):
            fleet.reset()
            recs, wall, _ = asyncio.run(run_load(
                fleet.urls, reqs, mk(), open_loop=True, timeout=120.0))
            s = summarize(recs, wall, label=label)
            per = {}
            for r in recs:
                if r.get("replica"):
                    per.setdefault(r["replica"], []).append(r["e2e_s"])
            s["by_replica"] = {k: len(v) for k, v in sorted(per.items())}
            s["e2e_p50_by_replica"] = {k: round(pct(v, 50), 2)
                                       for k, v in sorted(per.items())}
            out[label] = s
            print(f"[B] {label:18s} e2e p50 {s['e2e_p50_s']} p99 "
                  f"{s['e2e_p99_s']}  share {s['by_replica']}", flush=True)
    finally:
        fleet.stop()
    print(f"[B] liveness probe replies: {out['probe_ms']} ms "
          f"(the slow replica is indistinguishable)", flush=True)
    return out


def main():
    F["drill"] = section_a()
    F["gray"] = section_b()
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(F, f, indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    d = f["drill"]
    arms = ["naive", "retry", "health_retry"]
    labels = ["naive", "retry only", "health + retry"]

    ax[0].bar(range(3), [d[a]["errors"] for a in arms],
              color=["#c0392b", "#e67e22", "#2471a3"])
    for i, a in enumerate(arms):
        ax[0].text(i, d[a]["errors"], str(d[a]["errors"]), ha="center",
                   va="bottom")
    ax[0].set_xticks(range(3))
    ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel(f"failed requests (of {d['naive']['requests']})")
    ax[0].set_title("A. what the user lost")

    ax[1].bar(range(3), [d[a]["error_window_s"] for a in arms],
              color=["#c0392b", "#e67e22", "#2471a3"])
    for i, a in enumerate(arms):
        ax[1].text(i, d[a]["error_window_s"], f"{d[a]['error_window_s']:.1f}s",
                   ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel("seconds of errors after the kill")
    ax[1].set_title("A. how long it hurt")

    # timeline of the naive arm vs the health arm
    for j, (a, col) in enumerate((("naive", "#c0392b"),
                                  ("health_retry", "#2471a3"))):
        tl = d[a]["timeline"]
        ok = [(x["t"], x["ttft"]) for x in tl if x["ok"] and x["ttft"]]
        bad = [x["t"] for x in tl if not x["ok"]]
        ax[2].plot([p[0] for p in ok], [p[1] for p in ok], "o", ms=4,
                   color=col, alpha=.7, label=f"{a} ok")
        ax[2].plot(bad, [0.02] * len(bad), "x", ms=8, color=col,
                   label=f"{a} failed")
    ax[2].axvline(d["naive"]["t_kill_s"], color="k", ls="--", lw=1,
                  label="kill -9")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("seconds into the run")
    ax[2].set_ylabel("TTFT, s (log)")
    ax[2].set_title("A. the drill, request by request")
    ax[2].legend(fontsize=6)

    g = f["gray"]
    names = ["round_robin", "least_outstanding"]
    w = .35
    for j, key, lab, col in ((0, "e2e_p50_s", "E2E p50", "#2471a3"),
                             (1, "e2e_p99_s", "E2E p99", "#c0392b")):
        ax[3].bar([i + (j - .5) * w for i in range(2)],
                  [g[n][key] for n in names], w, color=col, label=lab)
    for i, n in enumerate(names):
        share = g[n]["by_replica"]
        ax[3].text(i, 0, f"r1 got {share.get('r1', 0)}", ha="center",
                   va="bottom", fontsize=7)
    ax[3].set_xticks(range(2))
    ax[3].set_xticklabels(["round-robin", "least-outstanding"], fontsize=8)
    ax[3].set_ylabel("seconds")
    ax[3].set_title("B. gray failure (r1 is slow, not dead)")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "failure_drill.png"), dpi=110)
    print("wrote outputs/failure_drill.png")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        main()
    plot()
