"""Project 50 -- what geography costs, and what it does not.

  A. TTFT against the same replica seen from four distances.
  B. The decomposition: how many round trips a request actually pays, and
     which of them regional routing removes.
  C. Connection reuse -- the cheapest cross-region win there is.
  D. Streaming vs buffered: distance charges the first token once, but a
     buffered response pays it after the LAST token instead.

    python3 run.py           # ~5 minutes; starts one replica + proxies
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "45-vllm-multi-replica"))

from fleetlib import Fleet, pct  # noqa: E402
from wan import OWD, WanProxy  # noqa: E402

F = {}
PLEN, MAX_NEW = 64, 12
N_REQ = 12
PROXY_BASE = 8790


async def timed_request(client, url, ids, max_new, reuse_client=True):
    """One streaming request, timed from the client's point of view."""
    t0 = time.perf_counter()
    ttft = None
    arrivals = []
    async with client.stream("POST", url + "/generate",
                             json={"ids": ids, "max_new": max_new},
                             timeout=120.0) as r:
        async for line in r.aiter_lines():
            if not line.strip():
                continue
            obj = json.loads(line)
            now = time.perf_counter()
            if "t" in obj:
                if ttft is None:
                    ttft = now - t0
                arrivals.append(now)
    itls = [b - a for a, b in zip(arrivals, arrivals[1:])]
    return {"ttft_s": ttft, "e2e_s": time.perf_counter() - t0, "itls": itls}


async def measure(url, ids, n=N_REQ, fresh_connection=False):
    import httpx

    recs = []
    if fresh_connection:
        # A brand-new client per request = a brand-new TCP connection, the
        # cold-start path a user hits when nothing is pooled.
        for _ in range(n):
            async with httpx.AsyncClient() as c:
                recs.append(await timed_request(c, url, ids, MAX_NEW))
    else:
        async with httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=4)) as c:
            await timed_request(c, url, ids, 2)      # warm the connection
            for _ in range(n):
                recs.append(await timed_request(c, url, ids, MAX_NEW))
    return recs


def summarise(recs, label, owd):
    ttfts = [r["ttft_s"] for r in recs]
    itls = [x for r in recs for x in r["itls"]]
    return {
        "label": label, "owd_ms": owd * 1e3, "rtt_ms": owd * 2e3,
        "ttft_p50_s": round(pct(ttfts, 50), 4),
        "ttft_min_s": round(min(ttfts), 4),
        "e2e_p50_s": round(pct([r["e2e_s"] for r in recs], 50), 4),
        "itl_p50_ms": round(pct(itls, 50) * 1e3, 2),
        "n": len(recs),
    }


def main():
    import httpx

    fleet = Fleet(1, threads=6, base_port=8780, log_dir=OUT)
    proxies = []
    try:
        fleet.wait_ready()
        ids = list(range(2000, 2000 + PLEN))
        home_port = fleet.ports[0]

        # one proxy per "region"
        regions = list(OWD.items())
        for i, (name, owd) in enumerate(regions):
            p = WanProxy(PROXY_BASE + i, home_port, owd)
            p.start()
            p.wait_ready()
            proxies.append(p)

        # ---- A. distance vs the clocks -----------------------------------
        rows = []
        for i, (name, owd) in enumerate(regions):
            url = f"http://127.0.0.1:{PROXY_BASE + i}"
            recs = asyncio.run(measure(url, ids))
            s = summarise(recs, name, owd)
            rows.append(s)
            print(f"[A] {name:16s} rtt {s['rtt_ms']:5.1f} ms -> ttft "
                  f"{s['ttft_min_s']*1e3:7.1f} ms  itl {s['itl_p50_ms']:6.2f} ms",
                  flush=True)
        F["regions"] = rows

        # ---- B. how many round trips does a request pay? -------------------
        # Compare a warm (pooled) connection with a cold one at each distance:
        # the difference is the handshake, in units of RTT.
        cold = []
        for i, (name, owd) in enumerate(regions):
            url = f"http://127.0.0.1:{PROXY_BASE + i}"
            recs = asyncio.run(measure(url, ids, n=10, fresh_connection=True))
            s = summarise(recs, name + "_cold", owd)
            warm = next(r for r in rows if r["label"] == name)
            s["warm_ttft_s"] = warm["ttft_min_s"]
            s["handshake_ms"] = round(
                (s["ttft_min_s"] - warm["ttft_min_s"]) * 1e3, 2)
            s["handshake_rtts"] = round(
                s["handshake_ms"] / s["rtt_ms"], 2) if s["rtt_ms"] else None
            cold.append(s)
            print(f"[C] {name:16s} cold {s['ttft_min_s']*1e3:7.1f} ms  "
                  f"warm {s['warm_ttft_s']*1e3:7.1f} ms  handshake "
                  f"{s['handshake_ms']:6.1f} ms = {s['handshake_rtts']} RTT",
                  flush=True)
        F["cold_vs_warm"] = cold

        # ---- D. streaming vs buffered at distance --------------------------
        # A buffered client cannot show the first token early; measure what
        # each pays by comparing TTFT with E2E.
        stream_rows = []
        for i, (name, owd) in enumerate(regions):
            r = next(x for x in rows if x["label"] == name)
            stream_rows.append({
                "region": name, "rtt_ms": r["rtt_ms"],
                "streaming_first_token_s": r["ttft_min_s"],
                "buffered_first_token_s": r["e2e_p50_s"],
            })
        F["streaming"] = stream_rows

        # ---- E. the routing decision ---------------------------------------
        # A user in Europe, a replica in each region: what does routing them
        # home cost or save, relative to always using us-east?
        home = next(r for r in rows if r["label"] == "same_region")
        far = next(r for r in rows if r["label"] == "trans_atlantic")
        F["decision"] = {
            "local_ttft_s": home["ttft_min_s"],
            "remote_ttft_s": far["ttft_min_s"],
            "delta_ms": round((far["ttft_min_s"] - home["ttft_min_s"]) * 1e3, 1),
            "compute_ms": round(home["ttft_min_s"] * 1e3, 1),
            "note": "the replica is the same process in both cases; the whole "
                    "difference is distance",
        }
    finally:
        for p in proxies:
            p.stop()
        fleet.stop()

    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(F, f, indent=1)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    rows = f["regions"]
    names = [r["label"].replace("_", "\n") for r in rows]
    rtts = [r["rtt_ms"] for r in rows]
    ttfts = [r["ttft_min_s"] * 1e3 for r in rows]

    ax[0].bar(range(len(rows)), ttfts, color="#2471a3", label="TTFT")
    ax[0].bar(range(len(rows)), rtts, color="#c0392b", alpha=.75,
              label="network RTT")
    ax[0].set_xticks(range(len(rows)))
    ax[0].set_xticklabels(names, fontsize=7)
    ax[0].set_ylabel("ms")
    ax[0].set_title("A. TTFT is compute + one round trip")
    ax[0].legend(fontsize=8)

    ax[1].plot(rtts, ttfts, "o-", color="#2471a3", label="measured TTFT")
    base = ttfts[0] - rtts[0]
    ax[1].plot(rtts, [base + r for r in rtts], "k--", lw=1,
               label=f"compute ({base:.0f} ms) + RTT")
    ax[1].set_xlabel("network RTT, ms")
    ax[1].set_ylabel("TTFT, ms")
    ax[1].set_title("B. distance is charged at least once")
    ax[1].legend(fontsize=8)

    cold = f["cold_vs_warm"]
    w = .35
    ax[2].bar([i - w / 2 for i in range(len(cold))],
              [c["warm_ttft_s"] * 1e3 for c in cold], w, color="#2471a3",
              label="pooled connection")
    ax[2].bar([i + w / 2 for i in range(len(cold))],
              [c["ttft_min_s"] * 1e3 for c in cold], w, color="#c0392b",
              label="fresh connection")
    for i, c in enumerate(cold):
        # A "+12 RTT" label on a 0.5 ms round trip is 6 ms over a near-zero
        # denominator, not a result -- show the raw millisecond cost there.
        lab = (f"+{c['handshake_rtts']}\nRTT" if c["rtt_ms"] > 5
               else f"+{c['handshake_ms']:.0f}\nms")
        ax[2].text(i + w / 2, c["ttft_min_s"] * 1e3, lab, ha="center",
                   va="bottom", fontsize=7)
    ax[2].set_xticks(range(len(cold)))
    ax[2].set_xticklabels(names, fontsize=7)
    ax[2].set_ylabel("TTFT, ms")
    ax[2].set_title("C. the handshake is the hidden cost")
    ax[2].legend(fontsize=8)

    st = f["streaming"]
    ax[3].bar([i - w / 2 for i in range(len(st))],
              [s["streaming_first_token_s"] * 1e3 for s in st], w,
              color="#2471a3", label="streaming: first token")
    ax[3].bar([i + w / 2 for i in range(len(st))],
              [s["buffered_first_token_s"] * 1e3 for s in st], w,
              color="#7f8c8d", label="buffered: whole answer")
    ax[3].set_xticks(range(len(st)))
    ax[3].set_xticklabels(names, fontsize=7)
    ax[3].set_ylabel("ms until the user sees anything")
    ax[3].set_title("D. streaming hides distance")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cross_region.png"), dpi=110)
    print("wrote outputs/cross_region.png")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        main()
    plot()
