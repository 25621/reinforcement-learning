"""Project 21 -- cache-aware admission control.

A serving engine can always accept one more request. Whether it can *finish*
it is a different question, and the difference is the KV cache: every admitted
request keeps growing its cache until it stops generating. Admit too many and
the cache overflows, at which point the engine either crashes or throws work
away.

  A. Show the crash. Run the real engine with no admission check and let it
     fail, then show the check preventing exactly that failure.
  B. Three admission policies under load: none (overcommit), reserve the
     prompt, reserve the prompt plus room for the answer.
  C. Sweep the cache size. Where does each policy break?
  D. The bill for guessing wrong: wasted tokens, preemption storms, and the
     one number a capacity plan should be written against.

    python3 run.py           # ~2 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "16-static-vs-continuous"))
sys.path.insert(0, os.path.join(HERE, "..", "18-chunked-prefill-simulator"))

from simlib import CostModel, make_trace, pct, simulate  # noqa: E402

F = {}
COST = CostModel(base=0.0797, per_decode=0.0093, per_prefill=0.00236,
                 per_key_read=1.1e-6)
MAX_MODEL_LEN = 8192


# ---------------------------------------------------------------------------
# A. the real failure
# ---------------------------------------------------------------------------


def section_a():
    print("\n=== A. what 'the cache is full' actually looks like ===")
    import torch

    import batchlib
    from batchlib import SlotKV

    runner, _ = batchlib.load_runner()
    n_slots, max_len = 4, 128
    pool = SlotKV(runner.n_layers, n_slots, runner.n_kv_heads, runner.d_head,
                  max_len)
    bpt = 2 * runner.n_layers * runner.n_kv_heads * runner.d_head * 4
    print(f"  pool: {n_slots} lanes x {max_len} tokens = "
          f"{pool.nbytes()/1e6:.0f} MB at {bpt/1024:.0f} KB/token")

    # 1. slots run out
    slots = [pool.acquire() for _ in range(n_slots)]
    fifth = pool.acquire()
    print(f"  5th request on a 4-lane pool -> acquire() returns {fifth}")
    for s in slots:
        pool.release(s)

    # 2. a request longer than a lane: the write goes out of bounds
    slot = pool.acquire()
    ids = torch.randint(1000, 12000, (1, max_len + 8))
    crashed = ""
    try:
        runner.prefill(pool, [slot], ids, [ids.shape[1]], count=False)
    except Exception as e:                       # noqa: BLE001
        crashed = f"{type(e).__name__}: {str(e).splitlines()[0][:90]}"
    print(f"  {max_len+8}-token prompt into a {max_len}-token lane -> {crashed or 'NO ERROR'}")

    # 3. the check that prevents it
    def admit(prompt_len, max_new):
        return pool.n_free() > 0 and prompt_len + max_new <= max_len

    ok = admit(max_len + 8, 32)
    print(f"  cache-aware admit({max_len+8} prompt, 32 new) -> {ok}  (rejected before it can fail)")
    F["A"] = {"n_slots": n_slots, "lane_tokens": max_len,
              "pool_MB": round(pool.nbytes() / 1e6, 1),
              "kv_bytes_per_token": bpt,
              "fifth_acquire": fifth,
              "overlong_prefill_error": crashed,
              "admission_rejects_it": not ok}


# ---------------------------------------------------------------------------
# B / C / D. policies under load
# ---------------------------------------------------------------------------


def build_trace(n=800, util=0.9, seed=5):
    p_med, p_sigma, o_med, o_sigma = 700, 1.0, 250, 0.9
    probe = make_trace(n=3000, rate=1.0, seed=seed + 77, p_med=p_med,
                       p_sigma=p_sigma, o_med=o_med, o_sigma=o_sigma,
                       p_max=MAX_MODEL_LEN, o_max=2048)
    work = sum(COST.request_work(r.prompt_len, r.out_len) for r in probe) / len(probe)
    rate = util / work
    tr = make_trace(n=n, rate=rate, seed=seed, p_med=p_med, p_sigma=p_sigma,
                    o_med=o_med, o_sigma=o_sigma, p_max=MAX_MODEL_LEN, o_max=2048)
    mean_out = sum(r.out_len for r in tr) / len(tr)
    return tr, rate, mean_out


POLICIES = [
    # No check at all. The engine admits whatever arrives and dies the first
    # time the cache does not fit -- which is what an inference server without
    # admission control actually does.
    ("no check (crashes)", dict(overcommit=True, crash_on_overflow=True)),
    # Book only what exists right now. Never OOMs at admission time, but a
    # request that keeps generating can still push the cache over, so this
    # policy needs preemption to survive.
    ("reserve prompt only", dict(reserve=0)),
    # Book the prompt plus a guess at the answer, so growth is pre-paid.
    ("reserve prompt + mean answer", dict(reserve=None)),      # filled in
    # Book the prompt plus the worst case the model allows -- project 11's
    # `reserve_max`, seen from the scheduler instead of the allocator.
    ("reserve prompt + max_model_len", dict(reserve=MAX_MODEL_LEN)),
]


def measure(reqs, st, label, kv_capacity):
    done = [r for r in reqs if r.end_t is not None]
    useful = sum(r.out_len for r in done)
    total = useful + st["wasted_decode_tokens"]
    return {
        "policy": label,
        "kv_capacity": kv_capacity,
        "crashed": bool(st["oom"]),
        "crashed_at_s": round(st["oom_at_s"], 1) if st["oom_at_s"] else None,
        "inflight_lost_at_crash": st["oom_inflight"],
        "completed": len(done),
        "completed_pct": round(100 * len(done) / len(reqs), 1),
        "rejected": st["rejected"],
        "preemptions": st["preemptions"],
        "peak_kv_over_capacity": round(st["peak_kv"] / kv_capacity, 3),
        "wasted_decode_tokens": st["wasted_decode_tokens"],
        "wasted_frac": round(st["wasted_decode_tokens"] / total, 4) if total else 0.0,
        "goodput_tok_s": round(useful / st["makespan_s"], 1),
        "ttft_p50": round(pct([r.ttft for r in done], 50), 2),
        "ttft_p99": round(pct([r.ttft for r in done], 99), 2),
        "e2e_p99": round(pct([r.e2e for r in done], 99), 2),
        "makespan_s": round(st["makespan_s"], 1),
    }


def run_policies(trace, kv_capacity, mean_out):
    rows = []
    for label, kw in POLICIES:
        kw = dict(kw)
        if kw.get("reserve", 0) is None:
            kw["reserve"] = int(mean_out)
        reqs = copy.deepcopy(trace)
        st = simulate(reqs, COST, chunk=512, kv_capacity=kv_capacity,
                      max_running=512, **kw)
        rows.append(measure(reqs, st, label, kv_capacity))
    return rows


def section_bd(trace, kv_capacity, mean_out):
    print(f"\n=== B/D. four admission policies, {kv_capacity} tokens of cache ===")
    rows = run_policies(trace, kv_capacity, mean_out)
    for r in rows:
        tag = f"CRASHED at {r['crashed_at_s']}s ({r['inflight_lost_at_crash']} lost)" \
            if r["crashed"] else "survived"
        print(f"  {r['policy']:31s} {tag:34s} completed {r['completed_pct']:5.1f}%  "
              f"goodput {r['goodput_tok_s']:6.1f} tok/s  preempt {r['preemptions']:5d}  "
              f"wasted {r['wasted_frac']:6.1%}  TTFT p99 {r['ttft_p99']:7.1f}")
    F["B"] = rows
    base = rows[0]
    survived = [r for r in rows if not r["crashed"]]
    best = max(survived, key=lambda r: r["goodput_tok_s"])
    F["B_summary"] = {
        "no_check_crashed": base["crashed"],
        "no_check_crashed_at_s": base["crashed_at_s"],
        "no_check_completed_pct": base["completed_pct"],
        "no_check_inflight_lost": base["inflight_lost_at_crash"],
        "best_policy": best["policy"],
        "best_goodput": best["goodput_tok_s"],
        "maxlen_goodput_ratio": round(rows[3]["goodput_tok_s"] / best["goodput_tok_s"], 3),
        "maxlen_rejected": rows[3]["rejected"],
        "prompt_only_preemptions": rows[1]["preemptions"],
        "prompt_only_wasted_frac": rows[1]["wasted_frac"],
        "prompt_only_vs_best": round(rows[1]["goodput_tok_s"] / best["goodput_tok_s"], 3),
    }
    print(f"  --> best surviving policy is '{best['policy']}' at "
          f"{best['goodput_tok_s']} tok/s; reserving max_model_len gets "
          f"{F['B_summary']['maxlen_goodput_ratio']}x that")


def section_c(trace, mean_out):
    print("\n=== C. sweeping the cache size ===")
    rows = []
    for cap in [4_000, 8_000, 16_000, 32_000, 64_000, 128_000]:
        for r in run_policies(trace, cap, mean_out):
            rows.append(r)
        got = [r for r in rows if r["kv_capacity"] == cap]
        print(f"  cap {cap:7d}: " + "  ".join(
            f"{r['policy'].split()[0][:6]}={r['goodput_tok_s']:6.1f}"
            + ("*" if r["crashed"] else " ") for r in got) + "   (* = crashed)")
    F["C"] = rows


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
    short = {p[0]: p[0].split("(")[0].replace("reserve ", "").strip()
             for p in [(r["policy"],) for r in f["B"]]}

    b = f["B"]
    x = range(len(b))
    ax[0].bar(x, [r["completed_pct"] for r in b],
              color=["#c0392b", "#27ae60", "#2471a3", "#95a5a6"])
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels([short[r["policy"]].replace(" ", "\n") for r in b],
                          fontsize=7)
    ax[0].set_ylabel("% of requests completed")
    ax[0].set_title("B. did the server survive the trace?")
    for i, r in enumerate(b):
        ax[0].text(i, r["completed_pct"],
                   ("CRASH\n" if r["crashed"] else "") + f"{r['completed_pct']:.0f}%",
                   ha="center", va="bottom", fontsize=7)

    ax[1].bar(x, [r["wasted_frac"] * 100 for r in b],
              color=["#c0392b", "#27ae60", "#2471a3", "#95a5a6"])
    ax[1].set_xticks(list(x))
    ax[1].set_xticklabels([short[r["policy"]].replace(" ", "\n") for r in b],
                          fontsize=7)
    ax[1].set_ylabel("% of decode tokens thrown away")
    ax[1].set_title("D. work redone after preemption")
    for i, r in enumerate(b):
        ax[1].text(i, r["wasted_frac"] * 100, f"{r['wasted_frac']:.1%}",
                   ha="center", va="bottom", fontsize=8)

    ax[2].bar(x, [r["goodput_tok_s"] for r in b],
              color=["#c0392b", "#27ae60", "#2471a3", "#95a5a6"])
    ax[2].set_xticks(list(x))
    ax[2].set_xticklabels([short[r["policy"]].replace(" ", "\n") for r in b],
                          fontsize=7)
    ax[2].set_ylabel("goodput (useful tok/s)")
    ax[2].set_title("B. what the survivors deliver")
    for i, r in enumerate(b):
        ax[2].text(i, r["goodput_tok_s"], f"{r['goodput_tok_s']:.0f}",
                   ha="center", va="bottom", fontsize=8)

    c = f["C"]
    pols = list(dict.fromkeys(r["policy"] for r in c))
    for p, col in zip(pols, ["#c0392b", "#27ae60", "#2471a3", "#95a5a6"]):
        rr = [r for r in c if r["policy"] == p]
        ax[3].plot([r["kv_capacity"] for r in rr],
                   [r["goodput_tok_s"] for r in rr], "o-", color=col,
                   label=short[p])
    ax[3].set_xscale("log", base=2)
    ax[3].set_xlabel("KV cache capacity (tokens)")
    ax[3].set_ylabel("goodput (useful tok/s)")
    ax[3].set_title("C. how much cache do you need?")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "admission.png"), dpi=110)
    print("wrote outputs/admission.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    section_a()
    trace, rate, mean_out = build_trace()
    plens = sorted(r.prompt_len for r in trace)
    F["trace"] = {"n": len(trace), "rate_req_s": round(rate, 3),
                  "prompt_median": plens[len(plens) // 2],
                  "prompt_max": max(plens),
                  "out_mean": round(mean_out, 1),
                  "out_max": max(r.out_len for r in trace),
                  "max_model_len": MAX_MODEL_LEN}
    print("  trace:", F["trace"])
    section_bd(trace, 32_000, mean_out)
    section_c(trace, mean_out)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    with open(os.path.join(OUT, "capacity_sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(F["C"][0]), lineterminator="\n")
        w.writeheader()
        w.writerows(F["C"])
    plot()


if __name__ == "__main__":
    main()
