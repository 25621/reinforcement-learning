"""Project 20 -- a priority queue for an inference scheduler.

The guide's version of this project says "add a priority class to vLLM". vLLM
does not run on this machine (its kernels need compute capability 7.0; the GPU
here is 6.1), so the priority class goes into the scheduler this phase already
owns -- project 18's simulator for the load, project 16's real engine for the
confirmation. The mechanism is the same one vLLM's `priority` field drives.

  A. The baseline. Two tenants, one server, first-come-first-served.
  B. Strict priority: what gold gains, what bronze pays.
  C. Starvation, and the standard cure (aging). Sweep it.
  D. The catch: reordering a queue only helps a request that is *in* the
     queue. Measure how little strict priority does for a long answer, and
     what preemption adds.
  E. Confirm on the real model that the ordering does what the simulator says.

    python3 run.py           # ~4 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import copy
import csv
import json
import math  # noqa: F401
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "16-static-vs-continuous"))
sys.path.insert(0, os.path.join(HERE, "..", "18-chunked-prefill-simulator"))

from simlib import CostModel, make_trace, pct, simulate  # noqa: E402

F = {}
GOLD_SHARE = 0.2
MAX_RUNNING = 8
CHUNK = 512
TRACE_KW = dict(p_med=500, p_sigma=1.0, o_med=160, o_sigma=0.8,
                p_max=4096, o_max=1500)
LOADS = [0.95, 1.25]

# Fitted on this machine in project 18 section A. Copied rather than re-measured
# so this project does not need to load the model twice.
COST = CostModel(base=0.0797, per_decode=0.0093, per_prefill=0.00236,
                 per_key_read=1.1e-6)


def tag_tenants(trace, seed=0):
    """Mark 20% of the trace 'gold' (priority 0) and the rest 'bronze' (1).

    Gold requests are also given short prompts and short answers, because that
    is what an interactive tenant looks like: a chat turn, not a 30k-token
    document summary. Making them identical to bronze would hide the most
    interesting effect -- that a short request is the one most damaged by
    waiting behind a long one.
    """
    import random
    rng = random.Random(seed)
    for r in trace:
        if rng.random() < GOLD_SHARE:
            r.priority = 0
            r.prompt_len = min(r.prompt_len, 300)
            r.out_len = min(r.out_len, 120)
        else:
            r.priority = 1
    return trace


def split(reqs):
    g = [r for r in reqs if r.priority == 0 and r.end_t is not None]
    b = [r for r in reqs if r.priority == 1 and r.end_t is not None]
    return g, b


def measure(reqs, stats, label):
    g, b = split(reqs)
    done = g + b
    useful = sum(r.out_len for r in done)
    total = useful + stats["wasted_decode_tokens"]
    row = {"policy": label,
           "gold_n": len(g), "bronze_n": len(b),
           "goodput_tok_s": round(useful / stats["makespan_s"], 1),
           "wasted_frac": round(stats["wasted_decode_tokens"] / total, 4) if total else 0.0,
           "gold_ttft_p50": round(pct([r.ttft for r in g], 50), 2),
           "gold_ttft_p99": round(pct([r.ttft for r in g], 99), 2),
           "bronze_ttft_p50": round(pct([r.ttft for r in b], 50), 2),
           "bronze_ttft_p99": round(pct([r.ttft for r in b], 99), 2),
           "bronze_ttft_max": round(max((r.ttft for r in b), default=0), 2),
           "gold_e2e_p99": round(pct([r.e2e for r in g], 99), 2),
           "bronze_e2e_p99": round(pct([r.e2e for r in b], 99), 2),
           "preemptions": stats["preemptions"],
           "output_tok_s": round(stats["decode_tokens"] / stats["makespan_s"], 1)}
    return row


def run(trace, **kw):
    reqs = copy.deepcopy(trace)
    st = simulate(reqs, COST, chunk=CHUNK, max_running=MAX_RUNNING, **kw)
    return reqs, st


def capacity():
    """Requests per second this server sustains when it never runs out of work.

    Measured by drowning the simulated server, not derived: the per-iteration
    fixed cost means the answer depends on the average batch size, which a
    closed form would have to guess. Every load below is quoted as a fraction
    of this number, so "load 0.9" means the same thing in every run.
    """
    sat = make_trace(n=300, rate=1e6, seed=99, **TRACE_KW)
    st = simulate(sat, COST, chunk=CHUNK, max_running=MAX_RUNNING)
    return len([r for r in sat if r.end_t is not None]) / st["makespan_s"]


def build_trace(cap, n=900, util=0.95, seed=3):
    tr = make_trace(n=n, rate=util * cap, seed=seed, **TRACE_KW)
    return tag_tenants(tr), util * cap


# ---------------------------------------------------------------------------


POLICIES = [
    ("FCFS (no priority)", dict(order="fcfs")),
    ("strict priority", dict(order="priority")),
    ("priority + preemption", dict(order="priority", preempt_for_priority=True)),
]


def section_abd(cap):
    print("\n=== A/B/D. FCFS vs strict priority vs preemptive priority ===")
    rows = []
    for load in LOADS:
        trace, _ = build_trace(cap, util=load)
        for label, kw in POLICIES:
            reqs, st = run(trace, **kw)
            row = measure(reqs, st, label)
            row["load"] = load
            rows.append(row)
            print(f"  load {load:4.2f}  {label:22s} gold TTFT p50 "
                  f"{row['gold_ttft_p50']:7.2f} p99 {row['gold_ttft_p99']:8.2f} | "
                  f"bronze p99 {row['bronze_ttft_p99']:9.2f} max "
                  f"{row['bronze_ttft_max']:9.2f} | preempt {row['preemptions']:5d} "
                  f"| goodput {row['goodput_tok_s']:6.1f} wasted {row['wasted_frac']:5.1%}")
    F["ABD"] = rows
    F["summary"] = {}
    for load in LOADS:
        got = [r for r in rows if r["load"] == load]
        f, p, pp = got
        F["summary"][str(load)] = {
            "gold_ttft_p99_gain": round(f["gold_ttft_p99"] / p["gold_ttft_p99"], 2),
            "gold_e2e_p99_gain": round(f["gold_e2e_p99"] / p["gold_e2e_p99"], 2),
            "bronze_ttft_p99_cost": round(p["bronze_ttft_p99"] / f["bronze_ttft_p99"], 2),
            "bronze_ttft_max_cost": round(p["bronze_ttft_max"] / f["bronze_ttft_max"], 2),
            "goodput_change": round(p["goodput_tok_s"] / f["goodput_tok_s"], 3),
            "preempt_gold_ttft_p99_gain": round(f["gold_ttft_p99"] / pp["gold_ttft_p99"], 2),
            "preempt_extra_over_strict": round(p["gold_ttft_p99"] / pp["gold_ttft_p99"], 3),
            "preempt_bronze_max_cost": round(pp["bronze_ttft_max"] / f["bronze_ttft_max"], 1),
            "preempt_wasted_frac": pp["wasted_frac"],
        }
        s = F["summary"][str(load)]
        print(f"    load {load}: strict priority -> gold TTFT p99 {s['gold_ttft_p99_gain']}x "
              f"better but gold END-TO-END p99 only {s['gold_e2e_p99_gain']}x; "
              f"preemption adds {s['preempt_extra_over_strict']}x for "
              f"{s['preempt_bronze_max_cost']}x bronze worst-case")


def section_c(cap):
    print("\n=== C. aging: buying starvation protection back ===")
    rows = []
    for load in LOADS:
        trace, _ = build_trace(cap, util=load)
        for aging in [0.0, 0.01, 0.05, 0.2, 1.0, 5.0]:
            reqs, st = run(trace, order="priority", aging=aging)
            row = measure(reqs, st, f"aging={aging}")
            row["aging"] = aging
            row["load"] = load
            rows.append(row)
            print(f"  load {load:4.2f}  aging {aging:5.2f}/s: gold TTFT p99 "
                  f"{row['gold_ttft_p99']:9.2f}   bronze TTFT max "
                  f"{row['bronze_ttft_max']:9.2f}")
    F["C"] = rows


def section_e():
    print("\n=== E. the same ordering on the real model ===")
    import batchlib
    from batchlib import make_workload, pct as bpct
    from schedulers import run_continuous

    runner, tok = batchlib.load_runner()
    base = make_workload(tok, n=32, rate=6.0, seed=2)
    import random
    rng = random.Random(1)
    for r in base:
        r.priority = 0 if rng.random() < 0.25 else 1
    rows = []
    for label, kw in [("FCFS", dict(priority=False)),
                      ("priority", dict(priority=True))]:
        reqs = copy.deepcopy(base)
        wall = run_continuous(runner, reqs, n_slots=6, **kw)
        g = [r for r in reqs if r.priority == 0]
        b = [r for r in reqs if r.priority == 1]
        row = {"policy": label, "wall_s": round(wall, 2),
               "gold_n": len(g),
               "gold_ttft_p50": round(bpct([r.ttft for r in g], 50), 2),
               "gold_ttft_p99": round(bpct([r.ttft for r in g], 99), 2),
               "bronze_ttft_p50": round(bpct([r.ttft for r in b], 50), 2),
               "bronze_ttft_max": round(max(r.ttft for r in b), 2)}
        rows.append(row)
        print(f"  {label:9s} wall {row['wall_s']:6.2f}s  gold TTFT p50 "
              f"{row['gold_ttft_p50']:6.2f} p99 {row['gold_ttft_p99']:6.2f}  "
              f"bronze p50 {row['bronze_ttft_p50']:6.2f} max {row['bronze_ttft_max']:6.2f}")
    F["E"] = rows
    F["E_gain"] = round(rows[0]["gold_ttft_p50"] / rows[1]["gold_ttft_p50"], 2)
    print(f"  --> gold TTFT p50 improves {F['E_gain']}x on the real engine too")


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))

    loads = f["trace"]["loads"]
    hi = loads[-1]
    rows = [r for r in f["ABD"] if r["load"] == hi]
    x = range(len(rows))
    w = .35
    ax[0].bar([i - w / 2 for i in x], [r["gold_ttft_p99"] for r in rows], w,
              color="#d4ac0d", label="gold")
    ax[0].bar([i + w / 2 for i in x], [r["bronze_ttft_max"] for r in rows], w,
              color="#7f8c8d", label="bronze (worst case)")
    ax[0].set_yscale("log")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([r["policy"].replace(" ", "\n") for r in rows], fontsize=7)
    ax[0].set_ylabel("TTFT (s, log)")
    ax[0].set_title(f"A/B. who waits (load {hi})")
    ax[0].legend(fontsize=8)

    for j, load in enumerate(loads):
        rr = [r for r in f["ABD"] if r["load"] == load]
        ax[1].bar([i + (j - .5) * w for i in range(len(rr))],
                  [rr[0]["gold_ttft_p99"] / r["gold_ttft_p99"] for r in rr], w,
                  color=["#d4ac0d", "#b9770e"][j], label=f"TTFT p99, load {load}")
        ax[1].plot(range(len(rr)),
                   [rr[0]["gold_e2e_p99"] / r["gold_e2e_p99"] for r in rr],
                   "o--", color=["#2471a3", "#1a5276"][j],
                   label=f"end-to-end p99, load {load}")
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].set_xticks(range(len(rows)))
    ax[1].set_xticklabels([r["policy"].replace(" ", "\n") for r in rows], fontsize=7)
    ax[1].set_ylabel("gold improvement vs FCFS (x)")
    ax[1].set_title("D. TTFT moves, end-to-end barely does")
    ax[1].legend(fontsize=6)

    for j, load in enumerate(loads):
        c = [r for r in f["C"] if r["load"] == load]
        ax[2].plot([r["aging"] for r in c], [r["gold_ttft_p99"] for r in c], "o-",
                   color=["#d4ac0d", "#b9770e"][j], label=f"gold p99, load {load}")
        ax[2].plot([r["aging"] for r in c], [r["bronze_ttft_max"] for r in c], "s--",
                   color=["#c0392b", "#7b241c"][j], label=f"bronze worst, load {load}")
    ax[2].set_xscale("symlog", linthresh=0.01)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("aging credit (priority levels per second waited)")
    ax[2].set_ylabel("seconds")
    ax[2].set_title("C. aging trades one for the other")
    ax[2].legend(fontsize=6)
    ax[2].grid(alpha=.3)

    e = f["E"]
    ax[3].bar([i - w / 2 for i in range(2)], [r["gold_ttft_p50"] for r in e], w,
              color="#d4ac0d", label="gold p50")
    ax[3].bar([i + w / 2 for i in range(2)], [r["bronze_ttft_p50"] for r in e], w,
              color="#7f8c8d", label="bronze p50")
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels([r["policy"] for r in e])
    ax[3].set_ylabel("TTFT p50 (s)")
    ax[3].set_title("E. real model, real weights")
    ax[3].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "priority_queue.png"), dpi=110)
    print("wrote outputs/priority_queue.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    cap = capacity()
    trace, rate = build_trace(cap)
    gold = [r for r in trace if r.priority == 0]
    F["trace"] = {"n": len(trace), "capacity_req_s": round(cap, 4),
                  "loads": LOADS, "rate_at_load_0.95": round(rate, 4),
                  "gold_n": len(gold), "gold_share": round(len(gold) / len(trace), 3),
                  "max_running": MAX_RUNNING, "chunk": CHUNK,
                  "cost_model": {"base_s": COST.base, "per_decode_s": COST.per_decode,
                                 "per_prefill_s": COST.per_prefill,
                                 "per_key_read_s": COST.per_key_read}}
    print("  trace:", F["trace"])
    section_abd(cap)
    section_c(cap)
    section_e()
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    with open(os.path.join(OUT, "policies.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(F["ABD"][0]), lineterminator="\n")
        w.writeheader()
        w.writerows(F["ABD"])
    plot()


if __name__ == "__main__":
    main()
