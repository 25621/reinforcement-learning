"""Project 49 -- sticky routing, so a conversation keeps its cache.

  A. Multi-turn chats, 4 replicas: round-robin vs session-hash routing.
     Session-cache hit rate, tokens re-prefilled, TTFT per turn.
  B. The cost of stickiness: when a sticky replica dies, its sessions lose
     their cache and must re-prefill the whole history. Measured, not
     asserted -- and it grows with the turn number.

    python3 run.py           # ~7 minutes; starts real server processes
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

from fleetlib import (Fleet, HashRouter, RoundRobin, pct,  # noqa: E402
                      request_once, summarize)

F = {}
N_REP = 4
N_SESSIONS = 12
N_TURNS = 4
FIRST_LEN = 64        # tokens in turn 1's prompt
USER_LEN = 16         # tokens the user adds each later turn
MAX_NEW = 12


def session_key(req):
    return req["session_id"]


async def run_conversations(urls, router, n_sessions=N_SESSIONS,
                            n_turns=N_TURNS, seed=21, alive=None):
    """Run `n_sessions` conversations concurrently, `n_turns` each.

    A conversation is SEQUENTIAL by nature -- turn 2's prompt contains turn
    1's answer, so it cannot be built until turn 1 comes back. That is why
    this cannot reuse the generic `run_load`: it is a different traffic
    shape, and it is the shape session affinity exists for.
    """
    import httpx

    rng = random.Random(seed)
    records = []

    async def conversation(sid):
        history = [rng.randrange(1000, 12000) for _ in range(FIRST_LEN)]
        async with httpx.AsyncClient() as client:
            for turn in range(n_turns):
                req = {"rid": f"{sid}:{turn}", "ids": list(history),
                       "max_new": MAX_NEW, "session_id": sid}
                rec = None
                # Two attempts: the sticky home, then a live replica. Without
                # the retry a killed replica simply ends its conversations,
                # and section B would measure "sessions that stopped" instead
                # of "sessions that failed over and lost their cache" -- which
                # is the thing that actually costs a user time.
                for attempt in range(2):
                    cand = alive if alive is not None else set(range(router.n))
                    if not cand:
                        break
                    r = router.pick(req)
                    if r not in cand:
                        router.release(r)
                        r = min(cand)
                        router.outstanding[r] += 1
                    rec = await request_once(client, urls[r], req)
                    router.release(r)
                    rec.update({"session": sid, "turn": turn, "target": r,
                                "history_len": len(history), "attempt": attempt})
                    if rec["ok"]:
                        break
                if rec is None:
                    break
                records.append(rec)
                if not rec["ok"]:
                    break
                # the next turn's prompt = everything so far + a new user line
                history = history + rec["toks"] + [
                    rng.randrange(1000, 12000) for _ in range(USER_LEN)]

    await asyncio.gather(*[conversation(f"s{i}") for i in range(n_sessions)])
    return records


def analyse(records, label):
    later = [r for r in records if r["turn"] > 0 and r["ok"]]
    hits = [r for r in later if r.get("reused", 0) > 0]
    reused = sum(r.get("reused", 0) for r in later)
    total_prompt = sum(r["history_len"] for r in later)
    s = {
        "label": label,
        "turns_total": len(records),
        "later_turns": len(later),
        "hit_rate": round(len(hits) / len(later), 3) if later else 0.0,
        "tokens_reused": reused,
        "tokens_in_later_prompts": total_prompt,
        "prefill_saved_frac": round(reused / total_prompt, 3) if total_prompt else 0,
        "ttft_p50_s": round(pct([r["ttft_s"] for r in later], 50), 3),
        "ttft_p99_s": round(pct([r["ttft_s"] for r in later], 99), 3),
        "prefill_ms_mean": round(
            sum(r.get("prefill_ms", 0) for r in later) / len(later), 1)
        if later else 0,
        "by_turn": {},
    }
    for t in range(N_TURNS):
        rows = [r for r in records if r["turn"] == t and r["ok"]]
        if rows:
            s["by_turn"][t] = {
                "ttft_s": round(pct([r["ttft_s"] for r in rows], 50), 3),
                "prefill_ms": round(
                    sum(r.get("prefill_ms", 0) for r in rows) / len(rows), 1),
                "hit_rate": round(
                    sum(1 for r in rows if r.get("reused", 0) > 0) / len(rows), 2),
                "prompt_len": int(sum(r["history_len"] for r in rows) / len(rows)),
            }
    # how many replicas did each session touch? (1 = perfectly sticky)
    seen = {}
    for r in records:
        seen.setdefault(r["session"], set()).add(r["target"])
    s["replicas_per_session_mean"] = round(
        sum(len(v) for v in seen.values()) / len(seen), 2)
    print(f"[{label}] hit {s['hit_rate']:.0%}  prefill saved "
          f"{s['prefill_saved_frac']:.0%}  ttft p50 {s['ttft_p50_s']} s  "
          f"replicas/session {s['replicas_per_session_mean']}", flush=True)
    return s


def main():
    fleet = Fleet(N_REP, threads=2, log_dir=OUT)
    try:
        fleet.wait_ready()

        # ---- A. sticky vs not ---------------------------------------------
        res = {}
        for label, mk in (("round_robin", lambda: RoundRobin(N_REP)),
                          ("session_hash",
                           lambda: HashRouter(N_REP, session_key))):
            fleet.reset()
            recs = asyncio.run(run_conversations(fleet.urls, mk()))
            res[label] = analyse(recs, label)
        F["affinity"] = res

        # ---- B. the sticky replica dies ------------------------------------
        fleet.reset()
        alive = set(range(N_REP))
        router = HashRouter(N_REP, session_key)

        async def kill_midway():
            # let the first two turns build caches, then take r0 away
            await asyncio.sleep(14.0)
            fleet.kill(0)
            alive.discard(0)

        async def both():
            task = asyncio.create_task(kill_midway())
            recs = await run_conversations(fleet.urls, router, alive=alive,
                                           seed=22)
            await task
            return recs

        recs = asyncio.run(both())
        s = analyse(recs, "sticky_with_failure")
        homeless = [r for r in recs
                    if r["ok"] and r["turn"] > 0 and r.get("reused", 0) == 0]
        s["cold_after_failover"] = len(homeless)
        s["cold_prefill_ms_mean"] = round(
            sum(r.get("prefill_ms", 0) for r in homeless) / len(homeless), 1) \
            if homeless else 0.0
        warm = [r for r in recs
                if r["ok"] and r["turn"] > 0 and r.get("reused", 0) > 0]
        s["warm_prefill_ms_mean"] = round(
            sum(r.get("prefill_ms", 0) for r in warm) / len(warm), 1) \
            if warm else 0.0
        s["failed"] = sum(1 for r in recs if not r["ok"])
        F["failover"] = s
        print(f"[B] after the kill: {s['cold_after_failover']} cold turns, "
              f"re-prefill {s['cold_prefill_ms_mean']} ms vs warm "
              f"{s['warm_prefill_ms_mean']} ms", flush=True)
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

    a = f["affinity"]
    names = ["round_robin", "session_hash"]
    labels = ["round-robin", "session-hash"]

    ax[0].bar(range(2), [100 * a[n]["hit_rate"] for n in names],
              color=["#c0392b", "#2471a3"])
    for i, n in enumerate(names):
        ax[0].text(i, 100 * a[n]["hit_rate"], f"{a[n]['hit_rate']:.0%}",
                   ha="center", va="bottom")
    ax[0].set_xticks(range(2))
    ax[0].set_xticklabels(labels, fontsize=9)
    ax[0].set_ylabel("session-cache hit rate (turns 2+), %")
    ax[0].set_title("A. does the cache survive the turn?")

    for n, col, lab in ((("round_robin"), "#c0392b", "round-robin"),
                        (("session_hash"), "#2471a3", "session-hash")):
        bt = a[n]["by_turn"]
        ks = sorted(int(k) for k in bt)
        ax[1].plot(ks, [bt[str(k)]["prefill_ms"] for k in ks], "o-",
                   color=col, label=lab)
    ax[1].set_xlabel("turn")
    ax[1].set_ylabel("prefill, ms")
    ax[1].set_title("B. the work each turn repeats")
    ax[1].set_xticks(range(4))
    ax[1].legend(fontsize=8)

    for n, col, lab in ((("round_robin"), "#c0392b", "round-robin"),
                        (("session_hash"), "#2471a3", "session-hash")):
        bt = a[n]["by_turn"]
        ks = sorted(int(k) for k in bt)
        ax[2].plot(ks, [bt[str(k)]["ttft_s"] for k in ks], "o-",
                   color=col, label=lab)
    ax[2].set_xlabel("turn")
    ax[2].set_ylabel("TTFT p50, s")
    ax[2].set_title("C. what the user feels")
    ax[2].set_xticks(range(4))
    ax[2].legend(fontsize=8)

    fo = f["failover"]
    ax[3].bar([0, 1], [fo["warm_prefill_ms_mean"], fo["cold_prefill_ms_mean"]],
              color=["#2471a3", "#c0392b"])
    for i, v in enumerate([fo["warm_prefill_ms_mean"],
                           fo["cold_prefill_ms_mean"]]):
        ax[3].text(i, v, f"{v:.0f} ms", ha="center", va="bottom")
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(["cache hit\n(home replica)",
                           "after failover\n(cold replica)"], fontsize=8)
    ax[3].set_ylabel("prefill, ms")
    ax[3].set_title(f"D. stickiness has a price\n"
                    f"{fo['cold_after_failover']} turns went cold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "session_affinity.png"), dpi=110)
    print("wrote outputs/session_affinity.png")


if __name__ == "__main__":
    if "--plot" not in sys.argv:
        main()
    plot()
