"""Project 66 -- An incident, start to finish, on a real engine.

Three faults are injected into one continuous run of the instrumented engine
from project 59, each chosen to produce a *different* story on the dashboard:

  1. **Capacity loss** -- most of the concurrency disappears, as if a replica
     died and the survivors have fewer usable slots.
  2. **Gray degradation** -- every forward pass becomes 2.2x slower. Nothing
     errors, nothing crashes, no alert that watches for failures fires.
  3. **Traffic surge** -- arrivals triple for a while. The system is healthy;
     the world is not.

Then the part that is actually the drill:

  A. The timeline the dashboard saw.
  B. Detection: which signal moved first, and by how much.
  C. Blind diagnosis -- a written decision procedure reads only the metrics
     and names the cause. Scored.
  D. The same procedure with a LATENCY-ONLY dashboard. Scored again.
  E. The error-budget bill for the incident, and the postmortem.

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
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "16-static-vs-continuous"))
sys.path.insert(0, os.path.join(PROJ, "59-metric-instrumentation"))

import obslib as O            # noqa: E402
import batchlib as B          # noqa: E402

OUT = os.path.join(HERE, "outputs")

N_SLOTS = 8
WINDOW = 6.0                  # the scrape interval, in seconds
RATE = 1.05
TTFT_SLO = 4.0                # the latency SLI threshold, as in project 62

# Each phase is a stretch of virtual time. Faults switch on and off at the
# boundaries, with a quiet stretch between so the metrics return to baseline.
PHASES = [
    ("baseline", 0.0, 36.0, None),
    ("capacity loss", 36.0, 72.0, "capacity"),
    ("recovery 1", 72.0, 108.0, None),
    ("gray degradation", 108.0, 144.0, "slow"),
    ("recovery 2", 144.0, 180.0, None),
    ("traffic surge", 180.0, 216.0, "surge"),
    ("recovery 3", 216.0, 258.0, None),
]
SURGE_MULT = 3.0


def phase_at(t):
    for name, a, b, kind in PHASES:
        if a <= t < b:
            return name, kind
    return PHASES[-1][0], None


def make_trace(tok, seed=0):
    """Poisson arrivals whose rate triples during the surge phase."""
    rng = random.Random(seed)
    vocab = list(range(1000, 12000))
    reqs, t, i = [], 0.0, 0
    end = PHASES[-1][2]
    while t < end:
        _, kind = phase_at(t)
        r = RATE * (SURGE_MULT if kind == "surge" else 1.0)
        t += rng.expovariate(r)
        if t >= end:
            break
        plen = int(min(160, max(16, rng.lognormvariate(math.log(52), 0.6))))
        olen = int(min(32, max(6, rng.lognormvariate(math.log(15), 0.5))))
        reqs.append(B.Request(rid=i, arrive=t,
                              prompt_ids=[rng.choice(vocab) for _ in range(plen)],
                              max_new=olen))
        i += 1
    return reqs


def build_faults():
    """(virtual time, mutation) pairs handed to `run_engine`."""
    def cap_on(st):
        st.max_batch = 2                     # 8 -> 2 usable slots
    def cap_off(st):
        st.max_batch = N_SLOTS
    def slow_on(st):
        st.slow_factor = 2.2
    def slow_off(st):
        st.slow_factor = 1.0
    return [(36.0, cap_on), (72.0, cap_off),
            (108.0, slow_on), (144.0, slow_off)]


# ---------------------------------------------------------------------------
# Turning raw requests into the per-window dashboard a human would read
# ---------------------------------------------------------------------------


def windows(reqs, ts_rows, end):
    n = int(end / WINDOW) + 1
    rows = []
    gauge = {int(r["t"] / WINDOW): r for r in ts_rows}
    for w in range(n):
        t0, t1 = w * WINDOW, (w + 1) * WINDOW
        arrived = [r for r in reqs if t0 <= r.arrive < t1]
        first = [r for r in reqs if r.first_tok_t is not None
                 and t0 <= r.first_tok_t < t1]
        itls = [x for r in reqs for a, x in
                zip(r.token_times[1:], r.itls()) if t0 <= a < t1]
        g = gauge.get(w, {})
        name, kind = phase_at(t0)
        rows.append(dict(
            w=w, t=t0, phase=name, fault=kind,
            arrivals=len(arrived), arrival_rate=len(arrived) / WINDOW,
            ttft_p95=O.exact_quantile([r.ttft for r in first], 95)
            if first else float("nan"),
            ttft_p50=O.exact_quantile([r.ttft for r in first], 50)
            if first else float("nan"),
            itl_p95=O.exact_quantile(itls, 95) if itls else float("nan"),
            itl_p50=O.exact_quantile(itls, 50) if itls else float("nan"),
            running=g.get("running", 0), waiting=g.get("waiting", 0),
            busy=g.get("busy", 0.0),
            slow_requests=sum(1 for r in first if r.ttft > TTFT_SLO),
            n_first=len(first)))
    return rows


def baseline(rows):
    b = [r for r in rows if r["phase"] == "baseline" and r["n_first"]]
    def med(k):
        xs = sorted(r[k] for r in b if r[k] == r[k])
        return xs[len(xs) // 2] if xs else float("nan")
    return dict(ttft_p95=med("ttft_p95"), itl_p50=med("itl_p50"),
                arrival_rate=med("arrival_rate"), running=med("running"),
                waiting=med("waiting"))


# ---------------------------------------------------------------------------
# C/D. The diagnosis, written down as code so it can be scored
# ---------------------------------------------------------------------------


def diagnose(row, base, latency_only=False):
    """Name the cause from one window of dashboard, or say 'unclear'.

    The tree is the one an on-call engineer would walk, in order:

      1. Is the *inter-token* latency up? Then the engine itself got slower.
         Neither of the other two faults does this -- a capacity loss makes
         ITL *better* (fewer rows share each forward pass) and a demand surge
         leaves it alone (the batch is already full).
      2. Otherwise the queue is what grew. Did demand grow with it? Then it
         is a surge -- the system is fine, the world changed.
      3. Queue grew, demand did not: capacity went away.

    `latency_only=True` hides everything except the two latency histograms,
    which is what most LLM dashboards actually show. Step 1 still works;
    steps 2 and 3 both become "something is queueing".
    """
    hot_ttft = row["ttft_p95"] == row["ttft_p95"] and \
        row["ttft_p95"] > 2.0 * base["ttft_p95"]
    hot_itl = row["itl_p50"] == row["itl_p50"] and \
        row["itl_p50"] > 1.4 * base["itl_p50"]
    if not hot_ttft and not hot_itl:
        return "healthy"
    if hot_itl:
        return "slow"
    if latency_only:
        return "unclear (queueing: capacity or demand?)"
    if row["arrival_rate"] > 1.6 * base["arrival_rate"]:
        return "surge"
    if row["waiting"] > base["waiting"] or row["ttft_p95"] > \
            2.0 * base["ttft_p95"]:
        return "capacity"
    return "unclear (queueing: capacity or demand?)"


def score(rows, base, latency_only):
    """Per fault: did the procedure ever name it, and how long did it take?"""
    out = {}
    for name, a, b, kind in PHASES:
        if kind is None:
            continue
        w = [r for r in rows if r["phase"] == name]
        calls = [diagnose(r, base, latency_only) for r in w]
        hits = [i for i, c in enumerate(calls) if c == kind]
        wrong = [c for c in calls if c not in (kind, "healthy")
                 and not c.startswith("unclear")]
        out[name] = dict(truth=kind, calls=calls,
                         detected=bool(hits),
                         detect_s=hits[0] * WINDOW if hits else None,
                         misdiagnosed=len(wrong),
                         windows=len(w))
    # A window in the first 12 s after a fault clears is still draining the
    # backlog the fault created. Counting those as false alarms would blame
    # the alert for correctly noticing a real queue.
    fault_ends = [b for _, _, b, k in PHASES if k]
    def draining(r):
        return any(0 <= r["t"] - e < 12.0 for e in fault_ends)
    healthy = [r for r in rows if r["fault"] is None and not draining(r)]
    drain = [r for r in rows if r["fault"] is None and draining(r)]
    out["_false_alarms"] = sum(1 for r in healthy
                               if diagnose(r, base, latency_only) != "healthy")
    out["_healthy_windows"] = len(healthy)
    out["_draining_alarms"] = sum(1 for r in drain
                                  if diagnose(r, base, latency_only) != "healthy")
    out["_draining_windows"] = len(drain)
    out["_correct"] = sum(1 for k, v in out.items()
                          if not k.startswith("_") and v["detected"])
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    runner, tok = B.load_runner()
    reqs = make_trace(tok)
    print(f"[trace] {len(reqs)} requests over {PHASES[-1][2]:.0f} s of "
          f"virtual time")

    m = O.EngineMetrics()
    ts = O.TimeSeries(window=WINDOW)
    st = O.EngineState(n_slots=N_SLOTS, max_batch=N_SLOTS)
    res = O.run_engine(runner, reqs, n_slots=N_SLOTS, max_len=248, m=m, ts=ts,
                       state=st, faults=build_faults())
    print(f"[engine] {res['virtual_s']:.0f} s virtual / "
          f"{res['wall_s']:.0f} s wall, util {res['util']:.2f} "
          f"[{time.time()-t0:.0f}s]")

    rows = windows(reqs, ts.rows, res["virtual_s"])
    base = baseline(rows)
    print(f"[baseline] TTFT p95 {base['ttft_p95']:.2f}s  ITL p50 "
          f"{base['itl_p50']*1000:.0f}ms  running {base['running']:.1f}  "
          f"arrivals {base['arrival_rate']:.2f}/s")

    # --- B. what each fault did to each signal ------------------------------
    signals = {}
    for name, a, b, kind in PHASES:
        w = [r for r in rows if r["phase"] == name and r["n_first"]]
        if not w:
            continue
        def med(k):
            xs = sorted(r[k] for r in w if r[k] == r[k])
            return xs[len(xs) // 2] if xs else float("nan")
        signals[name] = dict(
            fault=kind,
            ttft_p95=med("ttft_p95"), itl_p50=med("itl_p50"),
            running=med("running"), waiting=med("waiting"),
            arrival_rate=med("arrival_rate"),
            ttft_x=med("ttft_p95") / base["ttft_p95"],
            itl_x=med("itl_p50") / base["itl_p50"],
            running_x=med("running") / max(base["running"], 1e-9),
            arr_x=med("arrival_rate") / max(base["arrival_rate"], 1e-9))
        s = signals[name]
        print(f"  {name:18s} TTFT {s['ttft_x']:5.2f}x  ITL {s['itl_x']:5.2f}x "
              f" running {s['running_x']:5.2f}x  arrivals {s['arr_x']:5.2f}x")

    # --- C/D. diagnosis with the full dashboard and with latency only -------
    full = score(rows, base, latency_only=False)
    lat = score(rows, base, latency_only=True)
    print(f"[C] full dashboard: {full['_correct']}/3 faults named, "
          f"{full['_false_alarms']}/{full['_healthy_windows']} false alarms")
    print(f"[D] latency only:   {lat['_correct']}/3 faults named, "
          f"{lat['_false_alarms']}/{lat['_healthy_windows']} false alarms")

    # --- E. the bill --------------------------------------------------------
    served = [r for r in reqs if r.first_tok_t is not None]
    slow = [r for r in served if r.ttft > TTFT_SLO]
    per_phase = {}
    for name, a, b, kind in PHASES:
        grp = [r for r in served if a <= r.first_tok_t < b]
        bad = [r for r in grp if r.ttft > TTFT_SLO]
        per_phase[name] = dict(fault=kind, served=len(grp), slow=len(bad),
                               share_of_all_slow=len(bad) / max(1, len(slow)))
    # Burn rate (project 62): bad-request ratio divided by the budget the SLO
    # allows. Scale-free, so a 36-second drill and a 30-day month use the same
    # number and the same alert threshold.
    SLO_RATIO = 0.99
    for k, v in per_phase.items():
        v["bad_ratio"] = v["slow"] / max(1, v["served"])
        v["burn_rate"] = v["bad_ratio"] / (1 - SLO_RATIO)
        v["pages_at_14_4"] = v["burn_rate"] >= 14.4
    incident_cost = {k: v["burn_rate"] for k, v in per_phase.items()
                     if v["fault"]}
    E = dict(served=len(served), slow=len(slow),
             slow_share=len(slow) / len(served), slo_ratio=SLO_RATIO,
             per_phase=per_phase, incident_burn_rate=incident_cost,
             minutes_of_budget_per_hour_at_burn={
                 k: v * (1 - SLO_RATIO) * 60 for k, v in incident_cost.items()})
    print("[E] " + "  ".join(f"{k}: burn {v:.0f}x" for k, v in
                             incident_cost.items()))

    out = dict(config=dict(slots=N_SLOTS, window=WINDOW, rate=RATE,
                           surge=SURGE_MULT, ttft_slo=TTFT_SLO,
                           phases=[(n, a, b, k) for n, a, b, k in PHASES],
                           model=B.MODEL_ID, n_requests=len(reqs)),
               engine=res, rows=rows, baseline=base, signals=signals,
               full=full, latency_only=lat, E=E,
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT, "postmortem.md"), "w") as f:
        f.write(postmortem_md(out))
    plot(out)
    print(f"total {time.time()-t0:.0f}s")


def postmortem_md(res):
    """The deliverable: a blameless postmortem, generated from the data."""
    s, E, base = res["signals"], res["E"], res["baseline"]
    L = ["# Postmortem -- three incidents in one drill\n",
         "_Generated by `66-postmortem-drill/run.py`. Blameless: the subject "
         "is the system and the instruments, never the people._\n",
         "## Summary\n",
         f"Over {res['engine']['virtual_s']:.0f} s of serving, three faults "
         f"were injected. {E['slow']} of {E['served']} requests "
         f"({E['slow_share']*100:.1f}%) missed the "
         f"{res['config']['ttft_slo']} s TTFT threshold.\n",
         "## Timeline\n",
         "| phase | TTFT p95 | ITL p50 | running | arrivals/s | slow reqs |",
         "|---|---|---|---|---|---|"]
    for name, a, b, kind in res["config"]["phases"]:
        if name not in s:
            continue
        r = s[name]
        L.append(f"| {name} ({a:.0f}-{b:.0f} s) | {r['ttft_p95']:.2f} s "
                 f"({r['ttft_x']:.1f}x) | {r['itl_p50']*1000:.0f} ms "
                 f"({r['itl_x']:.1f}x) | {r['running']:.1f} | "
                 f"{r['arrival_rate']:.2f} | "
                 f"{E['per_phase'][name]['slow']} |")
    L.append("\n## Diagnosis\n")
    L.append(f"- With the full dashboard the written procedure named "
             f"**{res['full']['_correct']} of 3** faults correctly, with "
             f"{res['full']['_false_alarms']} false alarms across "
             f"{res['full']['_healthy_windows']} healthy windows.")
    L.append(f"- With a latency-only dashboard it named "
             f"**{res['latency_only']['_correct']} of 3**.")
    for k, v in res["full"].items():
        if k.startswith("_"):
            continue
        d = v["detect_s"]
        L.append(f"- `{k}`: {'detected in ' + str(d) + ' s' if v['detected'] else 'NEVER named'}"
                 f" (truth: {v['truth']})")
    L.append("\n## Error-budget impact\n")
    L.append(f"Against a {E['slo_ratio']*100:.0f}% latency SLO "
             f"(requests under {res['config']['ttft_slo']} s):\n")
    L.append("| incident | served | slow | burn rate | pages at 14.4x? |")
    L.append("|---|---|---|---|---|")
    for k, v in E["incident_burn_rate"].items():
        pp = E["per_phase"][k]
        L.append(f"| {k} | {pp['served']} | {pp['slow']} | **{v:.0f}x** | "
                 f"{'yes' if pp['pages_at_14_4'] else 'no'} |")
    L.append("\n## Action items\n")
    L.append("1. Add `llm_num_requests_running` and the arrival rate to the "
             "main dashboard. Without them, capacity loss and a demand surge "
             "are the same picture.")
    L.append("2. Alert on queue wait, not on engine-busy: the busy gauge is "
             "saturated in all three incidents and in the healthy windows.")
    L.append("3. The gray degradation raised no errors. Any alert that "
             "watches only for failures would have slept through it.")
    return "\n".join(L)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows, base, s = res["rows"], res["baseline"], res["signals"]
    t = [r["t"] for r in rows]
    shade = [(a, b, n) for n, a, b, k in res["config"]["phases"] if k]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    def bands(p):
        for a, b, n in shade:
            p.axvspan(a, b, color="#c0392b", alpha=.13)
            p.text((a + b) / 2, p.get_ylim()[1], n, ha="center", va="top",
                   fontsize=7, color="#7b241c")

    p = ax[0][0]
    p.plot(t, [r["ttft_p95"] for r in rows], color="#c0392b", label="TTFT p95")
    p.axhline(res["config"]["ttft_slo"], color="k", ls=":",
              label=f"{res['config']['ttft_slo']} s threshold")
    p.set_yscale("log"), p.set_ylabel("TTFT p95 (s, log)")
    p.set_xlabel("seconds"), p.legend(fontsize=8)
    bands(p)
    p.set_title("A. What the latency dashboard showed\n"
                "three different faults, one shape")

    p = ax[0][1]
    p.plot(t, [r["itl_p50"] * 1000 for r in rows], color="#2980b9",
           label="ITL p50")
    p.set_ylabel("ITL p50 (ms)"), p.set_xlabel("seconds"), p.legend(fontsize=8)
    bands(p)
    p.set_title("The other tail\n"
                "only one of the three faults moves it")

    p = ax[0][2]
    p.plot(t, [r["running"] for r in rows], color="#27ae60", label="running")
    p.plot(t, [r["waiting"] for r in rows], color="#c0392b", label="queued")
    p.plot(t, [r["arrival_rate"] for r in rows], color="#8e44ad",
           label="arrivals/s")
    p.set_xlabel("seconds"), p.set_ylabel("requests"), p.legend(fontsize=8)
    bands(p)
    p.set_title("B. The three signals that tell them apart\n"
                "running, queued, arrival rate")

    p = ax[1][0]
    names = [n for n in s if s[n]["fault"]]
    x = np.arange(len(names))
    for i, (k, lab, col) in enumerate((("ttft_x", "TTFT p95", "#c0392b"),
                                       ("itl_x", "ITL p50", "#2980b9"),
                                       ("running_x", "running", "#27ae60"),
                                       ("arr_x", "arrivals/s", "#8e44ad"))):
        p.bar(x + (i - 1.5) * .2, [s[n][k] for n in names], .2, label=lab,
              color=col)
    p.axhline(1.0, color="k", lw=.8)
    p.set_yscale("log")
    p.set_xticks(x, [n.replace(" ", "\n") for n in names], fontsize=8)
    p.set_ylabel("x baseline (log)"), p.legend(fontsize=7)
    p.set_title("B2. The fingerprint of each fault\n"
                "TTFT alone cannot separate rows 1 and 3")

    p = ax[1][1]
    labels = ["full dashboard", "latency only"]
    vals = [res["full"]["_correct"], res["latency_only"]["_correct"]]
    fa = [res["full"]["_false_alarms"], res["latency_only"]["_false_alarms"]]
    p.bar(labels, vals, color=["#27ae60", "#c0392b"], width=.5)
    for i, v in enumerate(vals):
        p.text(i, v, f"{v}/3 named\n{fa[i]} false alarms", ha="center",
               va="bottom", fontsize=9)
    p.set_ylim(0, 3.8), p.set_ylabel("faults correctly named")
    p.set_title("C/D. The same procedure, two dashboards\n"
                "the missing metrics are the diagnosis")

    p = ax[1][2]
    ks = list(res["E"]["incident_burn_rate"])
    p.bar([k.replace(" ", "\n") for k in ks],
          [res["E"]["incident_burn_rate"][k] for k in ks], color="#8e44ad")
    p.axhline(14.4, color="#c0392b", ls="--", label="pages at 14.4x")
    for i, k in enumerate(ks):
        p.text(i, res["E"]["incident_burn_rate"][k],
               f"{res['E']['per_phase'][k]['slow']}/"
               f"{res['E']['per_phase'][k]['served']} slow", ha="center",
               va="bottom", fontsize=8)
    p.set_ylabel("error-budget burn rate (x)")
    p.legend(fontsize=8)
    p.set_title("E. What each incident cost\n"
                "the same minutes, very different bills")

    fig.suptitle("Postmortem drill: three faults, one dashboard, and the "
                 "metric that decides the diagnosis", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "incident.png"), dpi=118)
    print("wrote", os.path.join(OUT, "incident.png"))


def reanalyse():
    """Recompute sections C, D and E from the committed windows."""
    path = os.path.join(OUT, "findings.json")
    with open(path) as f:
        res = json.load(f)
    rows, base = res["rows"], res["baseline"]
    res["full"] = score(rows, base, latency_only=False)
    res["latency_only"] = score(rows, base, latency_only=True)
    print(f"[C] full dashboard: {res['full']['_correct']}/3, "
          f"{res['full']['_false_alarms']}/{res['full']['_healthy_windows']} "
          f"false alarms")
    print(f"[D] latency only:   {res['latency_only']['_correct']}/3, "
          f"{res['latency_only']['_false_alarms']}/"
          f"{res['latency_only']['_healthy_windows']} false alarms")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "postmortem.md"), "w") as f:
        f.write(postmortem_md(res))
    plot(res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--reanalyse", action="store_true",
                    help="redo the diagnosis from the committed windows")
    a = ap.parse_args()
    if a.reanalyse:
        reanalyse()
    elif a.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main()
