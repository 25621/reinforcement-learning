"""Project 22 -- an SLO-aware scheduler.

Every request arrives with a deadline. The scheduler's job stops being "go fast
on average" and becomes "finish as many as possible *before their deadline*".
Those are different objectives, and the schedule that wins one can lose the
other badly.

  A. Calibrate the experiment so it can actually say something: measure the
     server's saturation point, then set deadlines that a baseline meets some
     but not all of.
  B. Four orderings as load rises: first-come-first-served, shortest-job-first,
     earliest-deadline-first, least-slack-first.
  C. Deadline-aware *dropping*: refuse requests that can no longer make it and
     spend the time on ones that can.
  D. Who each policy protects -- short requests, long ones, tight deadlines,
     loose ones.

    python3 run.py           # ~3 minutes, no model needed
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import copy
import csv
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "18-chunked-prefill-simulator"))

from simlib import CostModel, make_trace, pct, simulate  # noqa: E402

F = {}
# Fitted on this machine in project 18 section A.
COST = CostModel(base=0.0797, per_decode=0.0093, per_prefill=0.00236,
                 per_key_read=1.1e-6)
MAX_RUNNING = 8          # the batch the KV cache can hold at once
CHUNK = 512
TRACE_KW = dict(p_med=500, p_sigma=0.9, o_med=200, o_sigma=0.9,
                p_max=4096, o_max=1500)
ORDERS = ["fcfs", "sjf", "edf", "least_slack"]
PRETTY = {"fcfs": "FCFS", "sjf": "shortest-job-first",
          "edf": "earliest-deadline-first", "least_slack": "least-slack-first"}
LOADS = [0.5, 0.7, 0.9, 1.1]


# ---------------------------------------------------------------------------
# A. calibration
# ---------------------------------------------------------------------------


def capacity():
    """Requests per second the server sustains when it never runs out of work.

    Measured, not derived. A closed-form estimate would have to guess the
    average batch size, and the per-iteration fixed cost means the answer
    depends on it strongly -- so the honest way to find capacity is to drown
    the server and count what comes out.
    """
    sat = make_trace(n=300, rate=1e6, seed=99, **TRACE_KW)
    st = simulate(sat, COST, chunk=CHUNK, max_running=MAX_RUNNING)
    done = [r for r in sat if r.end_t is not None]
    return len(done) / st["makespan_s"], st["decode_tokens"] / st["makespan_s"]


def build(cap, util, n=700, seed=6, slack_lo=5.0, slack_hi=20.0):
    """A trace at a chosen fraction of capacity, with deadlines attached.

    The deadline is `arrive + slack x (time this request would take on an idle
    server)`. Two decisions in there are worth defending:

      *Why scale with the request's own size?* A flat deadline ("everyone gets
      5 seconds") quietly turns the experiment into shortest-job-first: the
      shortest request is always the most likely to meet a fixed budget, so
      every policy that favours short requests wins for the wrong reason. Real
      users also expect a long answer to take longer.

      *Why 5x-20x and not 1x?* Because on a loaded server nothing runs at idle
      speed. The measured slowdown here is about 9x at half load, so a 1x
      deadline would be missed by everyone and the comparison would return
      zeros. 5x-20x brackets the real slowdown, which is the only range where
      the policies disagree.
    """
    tr = make_trace(n=n, rate=util * cap, seed=seed, **TRACE_KW)
    rng = random.Random(0)
    for r in tr:
        r.solo_s = COST.request_work(r.prompt_len, r.out_len)
        r.slack_mult = rng.uniform(slack_lo, slack_hi)
        r.deadline = r.arrive + r.slack_mult * r.solo_s
    return tr


def score(reqs, st, label):
    done = [r for r in reqs if r.end_t is not None]
    on_time = [r for r in done if r.end_t <= r.deadline]
    late = [r for r in done if r.end_t > r.deadline]
    return {
        "policy": label,
        "on_time": len(on_time),
        "on_time_pct": round(100 * len(on_time) / len(reqs), 1),
        "completed": len(done),
        "dropped": st["rejected"],
        "late_p50_s": round(pct([r.end_t - r.deadline for r in late], 50), 1),
        "lateness_sum_s": round(sum(r.end_t - r.deadline for r in late)),
        "e2e_p50": round(pct([r.e2e for r in done], 50), 1),
        "e2e_p99": round(pct([r.e2e for r in done], 99), 1),
        "output_tok_s": round(st["decode_tokens"] / st["makespan_s"], 1),
    }


def run_one(trace, order, admission=None):
    reqs = copy.deepcopy(trace)
    st = simulate(reqs, COST, chunk=CHUNK, max_running=MAX_RUNNING,
                  order=order, admission=admission)
    return reqs, st


def section_a(cap, tok_s):
    print("\n=== A. calibration ===")
    print(f"  saturated capacity: {cap:.4f} req/s ({tok_s:.1f} output tok/s) "
          f"at batch <= {MAX_RUNNING}")
    rows = []
    for util in LOADS:
        tr = build(cap, util)
        reqs, st = run_one(tr, "fcfs")
        done = [r for r in reqs if r.end_t is not None]
        ratios = [r.e2e / r.solo_s for r in done]
        rows.append({"util": util, "rate_req_s": round(util * cap, 4),
                     "achieved_util": round(st["util"], 3),
                     "e2e_p50": round(pct([r.e2e for r in done], 50), 1),
                     "slowdown_p50": round(pct(ratios, 50), 1),
                     "slowdown_p90": round(pct(ratios, 90), 1)})
        print(f"  load {util:4.1f} -> {rows[-1]['rate_req_s']:.4f} req/s, "
              f"engine busy {rows[-1]['achieved_util']:.0%}, e2e p50 "
              f"{rows[-1]['e2e_p50']:6.1f}s = {rows[-1]['slowdown_p50']:5.1f}x "
              f"the idle-server time")
    F["A"] = {"capacity_req_s": round(cap, 4), "capacity_tok_s": round(tok_s, 1),
              "max_running": MAX_RUNNING, "chunk": CHUNK,
              "slack_range": [5.0, 20.0], "loads": rows}


# ---------------------------------------------------------------------------
# B. orderings
# ---------------------------------------------------------------------------


def section_b(cap):
    print("\n=== B. four orderings, rising load ===")
    rows = []
    for util in LOADS:
        tr = build(cap, util)
        for order in ORDERS:
            reqs, st = run_one(tr, order)
            r = score(reqs, st, PRETTY[order])
            r["util"] = util
            rows.append(r)
        got = {r["policy"]: r for r in rows if r["util"] == util}
        print(f"  load {util:4.1f}: " + "  ".join(
            f"{PRETTY[o][:5]:5s}={got[PRETTY[o]]['on_time_pct']:5.1f}%" for o in ORDERS))
    F["B"] = rows
    summary = {}
    for util in LOADS:
        got = {r["policy"]: r for r in rows if r["util"] == util}
        best = max(got.values(), key=lambda r: r["on_time_pct"])
        worst = min(got.values(), key=lambda r: r["on_time_pct"])
        summary[str(util)] = {
            "best": best["policy"], "best_pct": best["on_time_pct"],
            "worst": worst["policy"], "worst_pct": worst["on_time_pct"],
            "spread_pts": round(best["on_time_pct"] - worst["on_time_pct"], 1),
            "fcfs_pct": got["FCFS"]["on_time_pct"],
            "edf_pct": got["earliest-deadline-first"]["on_time_pct"],
            "sjf_pct": got["shortest-job-first"]["on_time_pct"],
        }
        print(f"    load {util}: best {best['policy']} {best['on_time_pct']}%, "
              f"worst {worst['policy']} {worst['on_time_pct']}%  "
              f"(spread {summary[str(util)]['spread_pts']} points)")
    F["B_summary"] = summary


# ---------------------------------------------------------------------------
# C. dropping
# ---------------------------------------------------------------------------


def make_dropper(factor):
    """Refuse a request whose deadline it can no longer reach.

    `factor` is what the test assumes about how busy the server is:
      1.0            -- optimistic: reject only if it could not make it even
                        with the whole machine to itself.
      MAX_RUNNING/2  -- realistic: assume it will share the batch, so its work
                        stretches by about the average batch size.

    The optimistic test almost never fires and lets doomed work in; the
    realistic one fires early and frees the capacity for requests that can
    still be saved. That difference is the entire result of section C.
    """
    def admit(r, now, kv_used, kv_capacity):
        return "admit" if (r.deadline - now) >= r.solo_s * factor else "reject"
    return admit


def section_c(cap):
    print("\n=== C. dropping what can no longer make it ===")
    rows = []
    for util in [0.9, 1.1]:
        tr = build(cap, util)
        for order in ORDERS:
            for factor, tag in [(None, ""),
                                (1.0, " + drop (idle-speed test)"),
                                (MAX_RUNNING / 2, " + drop (loaded test)")]:
                adm = None if factor is None else make_dropper(factor)
                reqs, st = run_one(tr, order, admission=adm)
                r = score(reqs, st, PRETTY[order] + tag)
                r["util"] = util
                r["order"] = PRETTY[order]
                r["drop"] = tag.strip(" +") or "none"
                rows.append(r)
        for r in [x for x in rows if x["util"] == util]:
            print(f"  load {util}  {r['policy']:48s} on-time {r['on_time_pct']:5.1f}%"
                  f"  dropped {r['dropped']:4d}  lateness sum {r['lateness_sum_s']:9d}s")
    F["C"] = rows
    hi = [r for r in rows if r["util"] == 1.1]
    nodrop = {r["order"]: r for r in hi if r["drop"] == "none"}
    loaded = {r["order"]: r for r in hi if r["drop"].startswith("drop (loaded")}
    F["C_summary"] = {
        "at_load_1.1": {
            o: {"no_drop": nodrop[o]["on_time_pct"],
                "with_drop": loaded[o]["on_time_pct"],
                "gain_x": round(loaded[o]["on_time_pct"] / nodrop[o]["on_time_pct"], 2),
                "dropped": loaded[o]["dropped"]}
            for o in nodrop},
        "spread_without_drop": round(
            max(r["on_time_pct"] for r in nodrop.values())
            - min(r["on_time_pct"] for r in nodrop.values()), 1),
        "spread_with_drop": round(
            max(r["on_time_pct"] for r in loaded.values())
            - min(r["on_time_pct"] for r in loaded.values()), 1),
    }
    print(f"  --> at load 1.1 the ordering spread is "
          f"{F['C_summary']['spread_without_drop']} points without dropping and "
          f"{F['C_summary']['spread_with_drop']} points with it")


# ---------------------------------------------------------------------------
# D. who gets protected
# ---------------------------------------------------------------------------


def section_d(cap):
    print("\n=== D. who makes the deadline? ===")
    tr = build(cap, 1.1)
    out = {}
    for order in ORDERS:
        reqs, st = run_one(tr, order)
        done = [r for r in reqs if r.end_t is not None]
        med_work = sorted(r.solo_s for r in done)[len(done) // 2]
        med_slack = sorted(r.slack_mult for r in done)[len(done) // 2]
        buckets = {}
        for name, sel in [
                ("short requests", lambda r: r.solo_s <= med_work),
                ("long requests", lambda r: r.solo_s > med_work),
                ("tight deadlines", lambda r: r.slack_mult <= med_slack),
                ("loose deadlines", lambda r: r.slack_mult > med_slack)]:
            grp = [r for r in done if sel(r)]
            buckets[name] = round(100 * sum(1 for r in grp
                                            if r.end_t <= r.deadline) / len(grp), 1)
        out[PRETTY[order]] = buckets
        print(f"  {PRETTY[order]:26s} " + "  ".join(
            f"{k}={v:5.1f}%" for k, v in buckets.items()))
    F["D"] = out


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    cols = {"FCFS": "#7f8c8d", "shortest-job-first": "#e67e22",
            "earliest-deadline-first": "#c0392b", "least-slack-first": "#2471a3"}

    rows = f["B"]
    utils = sorted({r["util"] for r in rows})
    for p, c in cols.items():
        ax[0].plot(utils, [next(r["on_time_pct"] for r in rows
                                if r["util"] == u and r["policy"] == p)
                           for u in utils], "o-", color=c, label=p)
    ax[0].axvline(1.0, color="k", ls=":", lw=1)
    ax[0].set_xlabel("offered load (1.0 = measured capacity)")
    ax[0].set_ylabel("% of requests on time")
    ax[0].set_title("B. below 0.9 the ordering does nothing")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)

    for p, c in cols.items():
        ax[1].plot(utils, [next(r["lateness_sum_s"] for r in rows
                                if r["util"] == u and r["policy"] == p)
                           for u in utils], "o-", color=c)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("offered load")
    ax[1].set_ylabel("total lateness (s, log)")
    ax[1].set_title("B. how late the late ones are")
    ax[1].grid(alpha=.3)

    hi = [r for r in f["C"] if r["util"] == 1.1]
    drops = ["none", "drop (idle-speed test)", "drop (loaded test)"]
    w = .8 / len(drops)
    orders = list(cols)
    for j, d in enumerate(drops):
        ax[2].bar([i + (j - 1) * w for i in range(len(orders))],
                  [next(r["on_time_pct"] for r in hi
                        if r["order"] == o and r["drop"] == d) for o in orders],
                  w, label=d, color=["#95a5a6", "#f39c12", "#27ae60"][j])
    ax[2].set_xticks(range(len(orders)))
    ax[2].set_xticklabels([o.replace("-", "-\n") for o in orders], fontsize=6)
    ax[2].set_ylabel("% on time")
    ax[2].set_title("C. dropping beats ordering (load 1.1)")
    ax[2].legend(fontsize=6)

    d = f["D"]
    keys = list(next(iter(d.values())))
    w = .8 / len(d)
    for j, (p, b) in enumerate(d.items()):
        ax[3].bar([i + (j - 1.5) * w for i in range(len(keys))],
                  [b[k] for k in keys], w, color=cols[p], label=p)
    ax[3].set_xticks(range(len(keys)))
    ax[3].set_xticklabels([k.replace(" ", "\n") for k in keys], fontsize=7)
    ax[3].set_ylabel("% on time")
    ax[3].set_title("D. who each policy protects")
    ax[3].legend(fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "slo_scheduler.png"), dpi=110)
    print("wrote outputs/slo_scheduler.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    cap, tok_s = capacity()
    section_a(cap, tok_s)
    section_b(cap)
    section_c(cap)
    section_d(cap)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    with open(os.path.join(OUT, "load_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(F["B"][0]), lineterminator="\n")
        w.writeheader()
        w.writerows(F["B"])
    plot()


if __name__ == "__main__":
    main()
