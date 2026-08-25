"""Project 63 -- A defensible dollars-per-million-output-tokens number.

Cost per million tokens is a fraction. The numerator is easy and public (what
the hardware costs per hour). The denominator is the hard part, because
"tokens per hour" depends on the arrival rate, the length distribution, the
batch size and the latency you are willing to promise -- and picking the
flattering value of each is how a serving cost gets understated by 3x.

  A. The denominator, three ways: peak, SLO-safe, and delivered.
  B. The arithmetic, line by line, for this machine and for the guide's H100.
  C. A confidence interval, by bootstrapping over 40 independent traffic days.
  D. Sensitivity: which line item actually moves the number.
  E. Input tokens are not free -- the measured prefill share of engine time,
     and the input:output price ratio it implies.
  F. What it would take to halve the cost.

Usage:
    python3 run.py            # ~1 minute, no model needed
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

import obslib as O                                  # noqa: E402
from simlib import SimRequest, simulate, pct        # noqa: E402

OUT = os.path.join(HERE, "outputs")

# --- the workload, identical to project 61's so the numbers are comparable --
P_MED, P_SIGMA, P_MAX = 200, 0.70, 1024
O_MED, O_SIGMA, O_MAX = 90, 0.70, 300
SLOTS = 24
N = 700

# --- the price list. ILLUSTRATIVE, and every downstream number says so. -----
# This machine is a 12-core desktop CPU; the closest rentable equivalent is a
# general-purpose 16-vCPU cloud VM. The H100 row is the guide's own example,
# reproduced here so the two can be compared with the same arithmetic.
PRICES = {
    "this box (16-vCPU VM equivalent)": 0.55,
    "H100 SXM node, 8 GPU (guide's example)": 32.00,
}
OVERHEAD = 0.25          # gateways, observability, control plane, on-call
DUTY = 0.50              # sustained utilisation: you rent the hour, you use
                         # half of it. The guide's example assumes the same.
SLO_TTFT_P95 = 3.54      # project 61's medium SLO, in seconds
SAFE_RATE = 0.30         # project 61: SLO breaks at 0.353 req/s
PEAK_RATE = 0.70         # deep in overload; the engine's flat-out throughput


def trace(rate, seed=0, n=N):
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        p = int(min(P_MAX, max(16, rng.lognormvariate(math.log(P_MED), P_SIGMA))))
        o = int(min(O_MAX, max(8, rng.lognormvariate(math.log(O_MED), O_SIGMA))))
        reqs.append(SimRequest(rid=i, arrive=t, prompt_len=p, out_len=o))
    return reqs


def run_point(cost, rate, seed=0, max_running=SLOTS, n=N):
    reqs = trace(rate, seed=seed, n=n)
    st = simulate(reqs, cost, max_running=max_running, token_budget=4096)
    done = [r for r in reqs if r.end_t is not None]
    # Split engine time into prefill and decode. With chunk=None every prefill
    # is one whole iteration, so its cost is exactly the model's prefill term
    # plus one `base`; everything left over is decode.
    pre_s = sum(cost.iter_time(0, r.prompt_len, cost.prefill_keys(0, r.prompt_len))
                for r in done)
    return dict(
        rate=rate, n=len(done), makespan=st["makespan_s"], util=st["util"],
        busy_s=st["busy_s"], prefill_s=pre_s, decode_s=st["busy_s"] - pre_s,
        out_tokens=st["decode_tokens"], in_tokens=st["prefill_tokens"],
        out_tok_s=st["decode_tokens"] / st["makespan_s"],
        in_tok_s=st["prefill_tokens"] / st["makespan_s"],
        out_tok_s_busy=st["decode_tokens"] / st["busy_s"],
        ttft_p95=pct([r.ttft for r in done], 95),
    )


def dollars_per_m(price_hr, out_tok_s, overhead=OVERHEAD, duty=DUTY):
    """The whole formula, in one place, so section D can differentiate it.

        $/M = price_per_hour x (1 + overhead)
              -----------------------------------------
              output_tokens_per_hour x sustained_duty / 1e6

    `duty` is the fraction of the rented hour during which the replica is
    actually serving at the measured rate. You pay for the other half."""
    tokens_per_hour = out_tok_s * 3600.0 * duty
    return price_hr * (1 + overhead) / (tokens_per_hour / 1e6)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cost, src = O.load_cost_model()
    print(f"[cost model] {src}")

    # --- A. three denominators --------------------------------------------
    peak = run_point(cost, PEAK_RATE)
    safe = run_point(cost, SAFE_RATE)
    light = run_point(cost, SAFE_RATE * 0.5)
    A = dict(peak=peak, slo_safe=safe, light=light)
    print(f"[A] peak {peak['out_tok_s']:.1f} tok/s | SLO-safe "
          f"{safe['out_tok_s']:.1f} | half-load {light['out_tok_s']:.1f}")

    # --- B. the bill -------------------------------------------------------
    price = PRICES["this box (16-vCPU VM equivalent)"]
    B = {}
    for name, row in A.items():
        B[name] = dict(
            out_tok_s=row["out_tok_s"],
            tokens_per_hour=row["out_tok_s"] * 3600,
            raw=dollars_per_m(price, row["out_tok_s"], overhead=0.0, duty=1.0),
            at_duty=dollars_per_m(price, row["out_tok_s"], overhead=0.0),
            all_in=dollars_per_m(price, row["out_tok_s"]))
    # the guide's H100 example, same formula
    h100 = dict(price_hr=PRICES["H100 SXM node, 8 GPU (guide's example)"],
                replicas=4, tok_s_per_replica=1800)
    h100["out_tok_s"] = h100["replicas"] * h100["tok_s_per_replica"]
    h100["raw"] = dollars_per_m(h100["price_hr"], h100["out_tok_s"], 0.0, 1.0)
    h100["at_duty"] = dollars_per_m(h100["price_hr"], h100["out_tok_s"], 0.0)
    h100["all_in"] = dollars_per_m(h100["price_hr"], h100["out_tok_s"])
    B["h100_reference"] = h100
    print(f"[B] this box, SLO-safe, all-in: "
          f"${B['slo_safe']['all_in']:.2f}/M output tokens   "
          f"(H100 example: ${h100['all_in']:.2f})")

    # --- C. a confidence interval ------------------------------------------
    days = [run_point(cost, SAFE_RATE, seed=100 + s) for s in range(40)]
    tps = [d["out_tok_s"] for d in days]
    costs = [dollars_per_m(price, t) for t in tps]
    mean_cost = sum(costs) / len(costs)
    lo, hi = O.bootstrap_ci(costs, lambda xs: sum(xs) / len(xs), n=2000)
    mean_tok_s = sum(tps) / len(tps)
    C = dict(n_days=len(days), tok_s=tps, costs=costs, mean=mean_cost,
             mean_tok_s=mean_tok_s, single_day_cost=dollars_per_m(
                 price, safe["out_tok_s"]),
             single_day_tok_s=safe["out_tok_s"],
             ci_lo=lo, ci_hi=hi,
             spread_pct=100 * (max(costs) - min(costs)) / mean_cost,
             day_min=min(costs), day_max=max(costs))
    print(f"[C] ${mean_cost:.2f}/M  95% CI [{lo:.2f}, {hi:.2f}]  "
          f"day-to-day spread {C['spread_pct']:.0f}%")

    # --- D. sensitivity ----------------------------------------------------
    base_cost = dollars_per_m(price, safe["out_tok_s"])
    D = {}
    # (i) average utilisation: you pay for the hardware 24/7, you use it less
    D["duty"] = [dict(x=u, cost=dollars_per_m(price, safe["out_tok_s"],
                                              duty=u))
                 for u in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)]
    # (ii) hardware price
    D["price"] = [dict(x=p, cost=dollars_per_m(p, safe["out_tok_s"]))
                  for p in (0.30, 0.40, 0.55, 0.70, 0.90, 1.20)]
    # (iii) batch cap -- a real engine knob, re-simulated each time
    D["batch"] = []
    for cap in (4, 8, 16, 24, 32, 48, 64):
        r = run_point(cost, SAFE_RATE, max_running=cap)
        D["batch"].append(dict(x=cap, cost=dollars_per_m(price, r["out_tok_s"]),
                               out_tok_s=r["out_tok_s"],
                               ttft_p95=r["ttft_p95"],
                               slo_ok=r["ttft_p95"] <= SLO_TTFT_P95))
    # (iv) platform overhead
    D["overhead"] = [dict(x=o, cost=dollars_per_m(price, safe["out_tok_s"], o))
                     for o in (0.0, 0.1, 0.25, 0.4, 0.6)]
    # (v) arrival rate, which moves the denominator AND the SLO
    D["rate"] = []
    for r in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70):
        row = run_point(cost, r)
        D["rate"].append(dict(x=r, cost=dollars_per_m(price, row["out_tok_s"]),
                              out_tok_s=row["out_tok_s"],
                              ttft_p95=row["ttft_p95"],
                              slo_ok=row["ttft_p95"] <= SLO_TTFT_P95))
    # elasticity: the % change in cost for a 2x change in each lever
    def span(rows):
        rows = sorted(rows, key=lambda r: r["x"])
        cheap = min(rows, key=lambda r: r["cost"])
        dear = max(rows, key=lambda r: r["cost"])
        return dict(from_x=rows[0]["x"], to_x=rows[-1]["x"],
                    cheapest_at=cheap["x"], cheapest=cheap["cost"],
                    dearest_at=dear["x"], dearest=dear["cost"],
                    ratio=dear["cost"] / cheap["cost"])
    D["levers"] = {k: span(v) for k, v in D.items() if k != "levers"}
    print("[D] " + "  ".join(f"{k}:{v['ratio']:.2f}x"
                             for k, v in D["levers"].items()))

    # --- E. input tokens are not free --------------------------------------
    pre_share = safe["prefill_s"] / safe["busy_s"]
    in_out_tokens = safe["in_tokens"] / safe["out_tokens"]
    # cost-per-token if you charge in proportion to engine seconds consumed
    cost_per_in = (safe["prefill_s"] / safe["in_tokens"])
    cost_per_out = (safe["decode_s"] / safe["out_tokens"])
    E = dict(prefill_share=pre_share, decode_share=1 - pre_share,
             in_tokens=safe["in_tokens"], out_tokens=safe["out_tokens"],
             in_out_token_ratio=in_out_tokens,
             engine_s_per_in_token=cost_per_in,
             engine_s_per_out_token=cost_per_out,
             fair_out_over_in=cost_per_out / cost_per_in,
             typical_published_ratio=3.0,
             blended_all_in=dollars_per_m(price, safe["out_tok_s"]),
             output_only_error=(cost_per_out / cost_per_in) / 3.0)
    print(f"[E] prefill is {pre_share*100:.0f}% of engine time; a fair "
          f"output:input price ratio is {E['fair_out_over_in']:.1f}:1 "
          f"(published schemes use ~3:1)")

    # --- F. how to halve it -------------------------------------------------
    # For each lever: what value of it would halve the bill, and is that value
    # reachable? A lever whose whole swept range cannot get there is not a
    # lever, it is a footnote.
    pretty = {"duty": "sustained utilisation", "price": "hardware $/hr",
              "batch": "max batch", "overhead": "platform overhead",
              "rate": "arrival rate (how full the box is)"}
    F = []
    for k, rows in D.items():
        if k == "levers":
            continue
        best = min(rows, key=lambda r: r["cost"])
        F.append(dict(lever=pretty[k], key=k, span=D["levers"][k]["ratio"],
                      best_x=best["x"], best_cost=best["cost"],
                      halves=best["cost"] <= base_cost / 2,
                      vs_base=base_cost / best["cost"]))
    F.sort(key=lambda f: -f["span"])
    print("[F] " + "  ".join(f"{f['lever']}={f['span']:.2f}x" for f in F))

    res = dict(config=dict(prices=PRICES, overhead=OVERHEAD, duty=DUTY, slots=SLOTS,
                           p_med=P_MED, o_med=O_MED, cost_model=src,
                           safe_rate=SAFE_RATE, peak_rate=PEAK_RATE,
                           slo_ttft_p95=SLO_TTFT_P95),
               A=A, B=B, C=C, D=D, E=E, F=F, base_cost=base_cost,
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(OUT, "cost_report.md"), "w") as f:
        f.write(report_md(res))
    plot(res)
    print(f"total {time.time()-t0:.0f}s")


def report_md(res):
    """The deliverable itself: a one-page cost report you could send."""
    B, C, E = res["B"], res["C"], res["E"]
    p = res["config"]["prices"]["this box (16-vCPU VM equivalent)"]
    L = []
    L.append("# Cost report -- Qwen2.5-0.5B on one 16-vCPU replica\n")
    L.append(f"_Generated by `63-cost-report/run.py`. Hardware price "
             f"(${p:.2f}/hr) is illustrative; everything else is measured or "
             f"simulated from a cost model fitted to this machine._\n")
    L.append("## Headline\n")
    L.append(f"**${C['mean']:.2f} per million output tokens** "
             f"(95% CI ${C['ci_lo']:.2f}-${C['ci_hi']:.2f}), all-in, at an "
             f"arrival rate that keeps the p95 TTFT SLO.\n")
    ov, du = res["config"]["overhead"], res["config"]["duty"]
    t = C["mean_tok_s"]
    raw = p / (t * 3600 / 1e6)
    at_duty = raw / du
    allin = at_duty * (1 + ov)
    L.append("## Line items (mean of 40 simulated days)\n")
    L.append("| item | value |")
    L.append("|---|---|")
    L.append(f"| hardware | ${p:.2f}/hr |")
    L.append(f"| delivered output tokens while serving | "
             f"{t * 3600 / 1e6:.3f} M/hr |")
    L.append(f"| compute at 100% duty | ${raw:.2f}/M |")
    L.append(f"| idle capacity ({du*100:.0f}% sustained duty) | "
             f"+${at_duty - raw:.2f}/M |")
    L.append(f"| platform overhead ({ov*100:.0f}%) | "
             f"+${allin - at_duty:.2f}/M |")
    L.append(f"| **all-in** | **${allin:.2f}/M** |\n")
    L.append("## The three levers that move it\n")
    for f in res["F"][:3]:
        L.append(f"- **{f['lever']}** -- {f['span']:.2f}x across the range "
                 f"tested; cheapest at {f['best_x']} "
                 f"(${f['best_cost']:.2f}/M)")
    L.append("")
    L.append("## Input vs output\n")
    L.append(f"Prefill consumes **{E['prefill_share']*100:.0f}%** of engine "
             f"time for **{E['in_out_token_ratio']:.2f}** input tokens per "
             f"output token. Charging by engine seconds implies an "
             f"output:input price ratio of **{E['fair_out_over_in']:.1f}:1**.\n")
    return "\n".join(L)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    A, B, C, D, E = res["A"], res["B"], res["C"], res["D"], res["E"]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    p = ax[0][0]
    names = ["half load", "SLO-safe", "flat out\n(SLO broken)"]
    keys = ["light", "slo_safe", "peak"]
    vals = [B[k]["all_in"] for k in keys]
    cols = ["#c0392b", "#27ae60", "#8e44ad"]
    p.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        p.text(i, v, f"${v:.2f}\n{A[keys[i]]['out_tok_s']:.0f} tok/s",
               ha="center", va="bottom", fontsize=9)
    p.set_ylabel("$ per million output tokens (all-in)")
    p.set_title(f"A. Three denominators, one machine\n"
                f"quoting the flat-out number understates by "
                f"{B['slo_safe']['all_in']/B['peak']['all_in']:.2f}x")

    p = ax[0][1]
    p.hist(C["costs"], bins=14, color="#2980b9", edgecolor="white")
    p.axvline(C["mean"], color="#c0392b", lw=2, label=f"mean ${C['mean']:.2f}")
    p.axvspan(C["ci_lo"], C["ci_hi"], color="#c0392b", alpha=.15,
              label=f"95% CI ${C['ci_lo']:.2f}-${C['ci_hi']:.2f}")
    p.set_xlabel("$ per million output tokens"), p.set_ylabel("days")
    p.legend(fontsize=8)
    p.set_title(f"C. 40 identical days\n"
                f"day-to-day spread is {C['spread_pct']:.0f}% of the mean")

    p = ax[0][2]
    for name, col in (("duty", "#c0392b"), ("price", "#2980b9"),
                      ("overhead", "#e67e22")):
        rows = D[name]
        xs = [r["x"] for r in rows]
        p.plot([x / xs[len(xs) // 2] for x in xs], [r["cost"] for r in rows],
               "o-", color=col, label=name)
    p.set_xscale("log")
    p.set_xlabel("lever, relative to its baseline value")
    p.set_ylabel("$ per million output tokens")
    p.legend(fontsize=8)
    p.set_title("D. Which lever moves the number\n"
                f"duty {D['levers']['duty']['ratio']:.1f}x, "
                f"price {D['levers']['price']['ratio']:.1f}x, "
                f"overhead {D['levers']['overhead']['ratio']:.1f}x")

    p = ax[1][0]
    rows = D["batch"]
    okc = ["#27ae60" if r["slo_ok"] else "#c0392b" for r in rows]
    p.bar([str(r["x"]) for r in rows], [r["cost"] for r in rows], color=okc)
    for i, r in enumerate(rows):
        p.text(i, r["cost"], f"{r['ttft_p95']:.1f}s", ha="center",
               va="bottom", fontsize=7)
    p.set_xlabel("max batch"), p.set_ylabel("$ per million output tokens")
    p.set_title("D2. The batch knob, priced\n"
                "green = the SLO still holds; label = TTFT p95")

    p = ax[1][1]
    rows = D["rate"]
    xs = [r["x"] for r in rows]
    p.plot(xs, [r["cost"] for r in rows], "o-", color="#8e44ad")
    for r in rows:
        if not r["slo_ok"]:
            p.plot(r["x"], r["cost"], "x", color="#c0392b", ms=11, mew=2)
    p.set_xlabel("arrival rate (req/s)")
    p.set_ylabel("$ per million output tokens")
    p.set_yscale("log")
    p.set_title("E1. Cheap and broken, or dearer and kept\n"
                "red x = the SLO is violated at that rate")

    p = ax[1][2]
    p.bar(["prefill\n(input)", "decode\n(output)"],
          [E["prefill_share"] * 100, E["decode_share"] * 100],
          color=["#2980b9", "#27ae60"])
    p.set_ylabel("% of engine seconds")
    p.text(0, E["prefill_share"] * 100,
           f"{E['in_tokens']:,} tokens", ha="center", va="bottom", fontsize=8)
    p.text(1, E["decode_share"] * 100,
           f"{E['out_tokens']:,} tokens", ha="center", va="bottom", fontsize=8)
    p.set_title(f"E2. Input is not free\n"
                f"a fair output:input price ratio is "
                f"{E['fair_out_over_in']:.1f}:1, not 3:1")

    fig.suptitle("Cost report: the numerator is public, the denominator is "
                 "the whole argument", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "cost.png"), dpi=118)
    print("wrote", os.path.join(OUT, "cost.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    if ap.parse_args().plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main()
