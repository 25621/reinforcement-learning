"""Project 61 -- Find the load at which the promise breaks.

An SLO is a number you commit to. This project takes one, raises the arrival
rate until the system can no longer keep it, and then asks the only question
that matters afterwards: *which part gave out?*

  A. Fit the cost model on real forward passes (shared with 62, 63 and 65).
  B. Where does the promise come from? The unloaded latency is a floor -- an
     SLO below it is broken at zero traffic. Three candidate targets.
  C. Sweep the arrival rate; find the breaking point for each target; report
     the engine utilisation there.
  D. Which part broke? TTFT split into queue wait and prefill, at every rate.
  E. The two tails fail differently. Sweep the running-batch cap at a fixed
     rate: it fixes TTFT and breaks TPOT.
  F. Capacity planning: the same mean rate, smooth against bursty, and the
     safety factor the burst actually costs.

Usage:
    python3 run.py            # ~5 minutes
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
for d in ("16-static-vs-continuous", "18-chunked-prefill-simulator",
          "59-metric-instrumentation"):
    sys.path.insert(0, os.path.join(PROJ, d))

import obslib as O                                        # noqa: E402
from simlib import SimRequest, simulate, pct              # noqa: E402

OUT = os.path.join(HERE, "outputs")
N = 700
P_MED, P_SIGMA, P_MAX = 200, 0.70, 1024
O_MED, O_SIGMA, O_MAX = 90, 0.70, 300
SLOTS = 24


def trace(rate, seed=0, n=N, burst=None):
    """Poisson arrivals unless `burst` is given.

    `burst=(frac, mult)` keeps the same *mean* rate but delivers `frac` of the
    hour at `mult` times the rate and the rest proportionally slower. Real
    traffic looks like this and a Poisson benchmark does not.
    """
    rng = random.Random(seed)
    reqs, t = [], 0.0
    if burst:
        frac, mult = burst
        # rate_low chosen so the time-average rate is still `rate`
        rate_hi = rate * mult
        rate_lo = rate * (1 - frac * mult) / (1 - frac)
        period = 900.0
    for i in range(n):
        if burst:
            phase = (t % period) / period
            r = rate_hi if phase < frac else max(rate_lo, 1e-3)
        else:
            r = rate
        t += rng.expovariate(r)
        p = int(min(P_MAX, max(16, rng.lognormvariate(math.log(P_MED), P_SIGMA))))
        o = int(min(O_MAX, max(8, rng.lognormvariate(math.log(O_MED), O_SIGMA))))
        reqs.append(SimRequest(rid=i, arrive=t, prompt_len=p, out_len=o))
    return reqs


def run_one(cost, rate, *, seed=0, max_running=SLOTS, burst=None, n=N,
            chunk=None):
    reqs = trace(rate, seed=seed, n=n, burst=burst)
    stats = simulate(reqs, cost, max_running=max_running, token_budget=4096,
                     chunk=chunk)
    done = [r for r in reqs if r.end_t is not None]
    ttfts = [r.ttft for r in done]
    queue = [r.admit_t - r.arrive for r in done if r.admit_t is not None]
    prefill = [r.first_t - r.admit_t for r in done if r.admit_t is not None]
    itls = [x for r in done for x in r.itls]
    return dict(
        rate=rate, n=len(done), util=stats["util"],
        makespan=stats["makespan_s"],
        offered_req_s=n / (reqs[-1].arrive or 1.0),
        ttft_p50=pct(ttfts, 50), ttft_p95=pct(ttfts, 95),
        ttft_p99=pct(ttfts, 99),
        queue_p95=pct(queue, 95), prefill_p95=pct(prefill, 95),
        queue_mean=sum(queue) / len(queue), prefill_mean=sum(prefill) / len(prefill),
        itl_p50=pct(itls, 50), itl_p95=pct(itls, 95), itl_p99=pct(itls, 99),
        e2e_p95=pct([r.e2e for r in done], 95),
        out_tok_s=stats["decode_tokens"] / stats["makespan_s"],
    )


def crossing(rows, key, target):
    """Linear interpolation on the rate axis for where `key` crosses `target`."""
    prev = None
    for r in sorted(rows, key=lambda r: r["rate"]):
        if r[key] > target:
            if prev is None:
                return r["rate"], r
            f = (target - prev[key]) / (r[key] - prev[key])
            return prev["rate"] + f * (r["rate"] - prev["rate"]), prev
        prev = r
    return None, prev


def main(reuse=False):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()

    # --- A. calibrate ------------------------------------------------------
    cm_path = os.path.join(OUT, O.COST_JSON)
    if reuse and os.path.exists(cm_path):
        cost, src = O.load_cost_model()
        info = json.load(open(cm_path))
        print(f"[A] reusing {src}")
    else:
        import batchlib as B
        runner, tok = B.load_runner()
        cost, info = O.fit_cost_model(runner, save_to=cm_path)
        del runner
    print(f"[A] base={cost.base*1000:.1f} ms  per_decode={cost.per_decode*1000:.2f} ms "
          f"per_prefill={cost.per_prefill*1000:.3f} ms  "
          f"fit err {info['fit_err_decode']:.1%}/{info['fit_err_prefill']:.1%} "
          f"[{time.time()-t0:.0f}s]")

    # --- B. what can we even promise? --------------------------------------
    idle = run_one(cost, 0.02, n=200)
    targets = {
        "tight  (1.25x idle p95)": 1.25 * idle["ttft_p95"],
        "medium (2x idle p95)": 2.0 * idle["ttft_p95"],
        "loose  (4x idle p95)": 4.0 * idle["ttft_p95"],
    }
    tpot_target = 2.0 * idle["itl_p95"]
    print(f"[B] unloaded TTFT p95 = {idle['ttft_p95']:.2f} s, "
          f"ITL p95 = {idle['itl_p95']*1000:.0f} ms")

    # --- C/D. the sweep ----------------------------------------------------
    rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.34, 0.38, 0.42, 0.46,
             0.50, 0.55, 0.60, 0.70]
    sweep = []
    for r in rates:
        row = run_one(cost, r)
        sweep.append(row)
        print(f"  rate {r:4.1f}  util {row['util']:.2f}  "
              f"ttft p95 {row['ttft_p95']:7.2f}  itl p99 {row['itl_p99']*1000:6.0f} ms "
              f"queue/ttft {row['queue_mean']/(row['queue_mean']+row['prefill_mean']):.0%}")
    breaks = {}
    for name, tgt in targets.items():
        rate_x, last_ok = crossing(sweep, "ttft_p95", tgt)
        breaks[name] = dict(target_s=tgt, break_rate=rate_x,
                            util_at_break=last_ok["util"] if last_ok else None,
                            queue_share=(last_ok["queue_mean"] /
                                         (last_ok["queue_mean"] +
                                          last_ok["prefill_mean"]))
                            if last_ok else None)
        print(f"[C] {name}: breaks at {rate_x if rate_x else float('nan'):.2f} req/s, "
              f"engine {breaks[name]['util_at_break']:.0%} busy")
    tpot_break, _ = crossing(sweep, "itl_p99", tpot_target)

    # --- E. the two tails, and the knob that trades them -------------------
    fixed_rate = 0.38
    caps = [2, 4, 8, 12, 16, 24, 32, 48, 64]
    cap_rows = []
    for c in caps:
        row = run_one(cost, fixed_rate, max_running=c)
        row["max_running"] = c
        cap_rows.append(row)
        print(f"[E] cap {c:3d}  ttft p95 {row['ttft_p95']:7.2f}  "
              f"itl p99 {row['itl_p99']*1000:6.0f} ms")

    # --- F. bursty traffic and the safety factor ---------------------------
    tgt = targets["medium (2x idle p95)"]
    burst_rows = []
    for r in rates:
        row = run_one(cost, r, burst=(0.20, 3.0))
        row["kind"] = "bursty"
        burst_rows.append(row)
    burst_break, _ = crossing(burst_rows, "ttft_p95", tgt)
    smooth_break = breaks["medium (2x idle p95)"]["break_rate"]
    safety = smooth_break / burst_break if burst_break else float("nan")
    print(f"[F] smooth breaks at {smooth_break:.2f}, bursty at "
          f"{burst_break:.2f} req/s -> safety factor {safety:.2f}x")

    res = dict(
        config=dict(n=N, p_med=P_MED, o_med=O_MED, slots=SLOTS,
                    cost=dict(base=cost.base, per_decode=cost.per_decode,
                              per_prefill=cost.per_prefill,
                              per_key_read=cost.per_key_read),
                    fit_err_decode=info["fit_err_decode"],
                    fit_err_prefill=info["fit_err_prefill"]),
        idle=idle, targets=targets, tpot_target=tpot_target,
        sweep=sweep, breaks=breaks, tpot_break=tpot_break,
        caps=cap_rows, fixed_rate=fixed_rate,
        burst=dict(rows=burst_rows, break_rate=burst_break,
                   smooth_break=smooth_break, safety_factor=safety,
                   shape=[0.20, 3.0]),
        wall_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print(f"total {time.time()-t0:.0f}s")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sw = sorted(res["sweep"], key=lambda r: r["rate"])
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    p = ax[0][0]
    p.plot([r["rate"] for r in sw], [r["ttft_p95"] for r in sw], "o-",
           color="#c0392b", label="TTFT p95")
    p.plot([r["rate"] for r in sw], [r["ttft_p50"] for r in sw], "s-",
           color="#27ae60", label="TTFT p50")
    cols = ["#16a085", "#e67e22", "#8e44ad"]
    for (name, b), c in zip(res["breaks"].items(), cols):
        p.axhline(b["target_s"], color=c, ls=":", lw=1.4)
        if b["break_rate"]:
            p.axvline(b["break_rate"], color=c, ls="--", lw=1.4,
                      label=f"{name.split()[0]} breaks @ {b['break_rate']:.2f}/s")
    p.set_yscale("log")
    p.set_xlabel("offered rate (req/s)"), p.set_ylabel("TTFT (s, log)")
    p.legend(fontsize=7)
    p.set_title("C. Sweep until the promise breaks\n"
                "the curve is a hockey stick, not a slope")

    p = ax[0][1]
    p.plot([r["util"] * 100 for r in sw], [r["ttft_p95"] for r in sw], "o-",
           color="#c0392b")
    for (name, b), c in zip(res["breaks"].items(), cols):
        if b["util_at_break"]:
            p.axvline(b["util_at_break"] * 100, color=c, ls="--", lw=1.4,
                      label=f"{name.split()[0]}: {b['util_at_break']*100:.0f}% busy")
    p.set_yscale("log")
    p.set_xlabel("engine busy (%)"), p.set_ylabel("TTFT p95 (s, log)")
    p.legend(fontsize=7)
    p.set_title("C2. Utilisation goes blind exactly where it matters\n"
                "the last few % of busy cover a 100x range of latency")

    p = ax[0][2]
    rr = [r["rate"] for r in sw]
    tot = [r["prefill_mean"] + r["queue_mean"] for r in sw]
    p.stackplot(rr, [100 * r["prefill_mean"] / t for r, t in zip(sw, tot)],
                [100 * r["queue_mean"] / t for r, t in zip(sw, tot)],
                labels=["prefill (the model)", "queue wait (the scheduler)"],
                colors=["#2980b9", "#c0392b"])
    p.set_ylim(0, 100)
    p.set_xlabel("offered rate (req/s)")
    p.set_ylabel("share of mean TTFT (%)")
    p2 = p.twinx()
    p2.plot(rr, tot, "k-", lw=1.4, label="mean TTFT (s)")
    p2.set_yscale("log"), p2.set_ylabel("mean TTFT (s, log)")
    p.legend(fontsize=8, loc="upper left")
    p.set_title("D. Which part broke?\n"
                f"at {sw[-1]['rate']:.1f} req/s the queue is "
                f"{sw[-1]['queue_mean']/(sw[-1]['queue_mean']+sw[-1]['prefill_mean']):.0%} "
                f"of TTFT")

    p = ax[1][0]
    cr = res["caps"]
    p.plot([r["max_running"] for r in cr], [r["ttft_p95"] for r in cr], "o-",
           color="#c0392b", label="TTFT p95 (s)")
    p.set_xlabel("max concurrent requests in the batch")
    p.set_ylabel("TTFT p95 (s, log)", color="#c0392b")
    p.set_xscale("log", base=2), p.set_yscale("log")
    p2 = p.twinx()
    p2.plot([r["max_running"] for r in cr], [r["itl_p99"] * 1000 for r in cr],
            "s--", color="#2980b9")
    p2.axhline(res["tpot_target"] * 1000, color="#2980b9", ls=":", lw=1.2)
    p2.set_ylabel("ITL p99 (ms)", color="#2980b9")
    p.set_title(f"E. The two tails pull opposite ways\n"
                f"at {res['fixed_rate']} req/s: bigger batch fixes TTFT, "
                f"breaks TPOT")

    p = ax[1][1]
    bs = sorted(res["burst"]["rows"], key=lambda r: r["rate"])
    p.plot([r["rate"] for r in sw], [r["ttft_p95"] for r in sw], "o-",
           color="#27ae60", label="smooth (Poisson)")
    p.plot([r["rate"] for r in bs], [r["ttft_p95"] for r in bs], "s-",
           color="#c0392b", label="bursty (same mean rate)")
    p.axhline(res["targets"]["medium (2x idle p95)"], color="k", ls=":",
              label="the SLO")
    p.set_yscale("log")
    p.set_xlabel("MEAN offered rate (req/s)"), p.set_ylabel("TTFT p95 (s, log)")
    p.legend(fontsize=8)
    p.set_title(f"F. Sizing on the mean rate\n"
                f"the burst costs a "
                f"{res['burst']['safety_factor']:.2f}x safety factor")

    p = ax[1][2]
    p.plot([r["rate"] for r in sw], [r["out_tok_s"] for r in sw], "o-",
           color="#8e44ad")
    if res["breaks"]["medium (2x idle p95)"]["break_rate"]:
        p.axvline(res["breaks"]["medium (2x idle p95)"]["break_rate"],
                  color="#c0392b", ls="--", label="SLO break")
    p.set_xlabel("offered rate (req/s)"), p.set_ylabel("output tokens/s")
    p.legend(fontsize=8)
    p.set_title("The throughput you may actually sell\n"
                "everything right of the dashed line is unsellable")

    fig.suptitle("SLO simulation: where the promise breaks, and which part "
                 "gave out", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "slo.png"), dpi=118)
    print("wrote", os.path.join(OUT, "slo.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the committed cost-model fit instead of "
                         "re-timing the model (for iterating on the sweep)")
    a = ap.parse_args()
    if a.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main(reuse=a.reuse)
