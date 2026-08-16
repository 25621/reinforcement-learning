"""Project 18 -- chunked prefill simulator.

  A. Calibrate. Measure this machine's real decode and prefill timings and fit
     the simulator's cost model to them, so the sweep is not built on
     invented constants.
  B. The disease. Without chunking, one long prompt takes an entire forward
     pass to itself and every streaming user's tokens stop arriving.
  C. The cure, swept. Chunk size from 128 to "no chunking"; TTFT, ITL and
     throughput for each.
  D. The trade, stated as a curve: what does the smoothest ITL cost in
     throughput and in TTFT?
  E. Does the conclusion survive a machine 100x faster? Re-run the sweep on
     an H100-shaped cost model.

    python3 run.py           # ~4 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "16-static-vs-continuous"))

import torch  # noqa: E402

import batchlib  # noqa: E402
from batchlib import SlotKV  # noqa: E402
from simlib import CostModel, make_trace, report, simulate  # noqa: E402

F = {}
CHUNKS = [32, 64, 128, 256, 512, 1024, 2048, None]


# ---------------------------------------------------------------------------
# A. calibration
# ---------------------------------------------------------------------------


def section_a(runner, tok):
    print("\n=== A. calibrating the cost model on real timings ===")
    ctx = 256
    dec_pts, pre_pts = [], []

    for b in [1, 2, 4, 8, 16]:
        pool = SlotKV(runner.n_layers, b, runner.n_kv_heads, runner.d_head, 512)
        ids = torch.randint(1000, 12000, (b, ctx))
        runner.prefill(pool, list(range(b)), ids, [ctx] * b, count=False)
        best = float("inf")
        for _ in range(3):
            _, dt = runner.decode_step(pool, list(range(b)), [11] * b,
                                       [ctx] * b, count=False)
            best = min(best, dt)
        dec_pts.append((b, best))
        print(f"  decode  batch {b:3d}: {best*1000:7.1f} ms")

    # Calibrate up to 2048 tokens because the trace below is capped at 4096:
    # fitting on 1024 and simulating 32k prompts would be a 30x extrapolation
    # of a quadratic, which is how simulators produce confident nonsense.
    for t in [64, 128, 256, 512, 1024, 2048]:
        pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head, 2100)
        ids = torch.randint(1000, 12000, (1, t))
        best = float("inf")
        for _ in range(2 if t <= 512 else 1):
            _, dt = runner.prefill(pool, [0], ids, [t], count=False)
            best = min(best, dt)
        pre_pts.append((t, best))
        print(f"  prefill {t:5d} tok: {best*1000:7.1f} ms  ({t/best:6.0f} tok/s)")

    cost = CostModel.fit(dec_pts, pre_pts, decode_ctx=ctx)
    err_d = [abs(cost.iter_time(b, 0, b * ctx) - s) / s for b, s in dec_pts]
    err_p = [abs(cost.iter_time(0, t, cost.prefill_keys(0, t)) - s) / s
             for t, s in pre_pts]
    print(f"  fitted: base={cost.base*1000:.1f} ms  per_decode_row="
          f"{cost.per_decode*1000:.2f} ms  per_prefill_token="
          f"{cost.per_prefill*1000:.3f} ms  per_key_read="
          f"{cost.per_key_read*1e6:.3f} us")
    print(f"  fit error: decode {statistics.mean(err_d):.1%}, "
          f"prefill {statistics.mean(err_p):.1%}")
    F["A"] = {
        "decode_ctx": ctx,
        "decode_points": [[b, round(s, 4)] for b, s in dec_pts],
        "prefill_points": [[t, round(s, 4)] for t, s in pre_pts],
        "base_s": round(cost.base, 5),
        "per_decode_s": round(cost.per_decode, 5),
        "per_prefill_s": round(cost.per_prefill, 6),
        "per_key_read_s": cost.per_key_read,
        "prefill_tok_s_at_1k": round(1024 / (cost.per_prefill * 1024 +
                                             cost.per_key_read *
                                             cost.prefill_keys(0, 1024)), 0),
        "mean_fit_error_decode": round(statistics.mean(err_d), 4),
        "mean_fit_error_prefill": round(statistics.mean(err_p), 4),
    }
    return cost


def build_trace(cost, n, seed, util=0.55):
    """Pick the arrival rate that loads the server to `util`, then draw a trace.

    The rate is *derived*, not chosen: a latency comparison between two
    policies only means something if both see the same load, and the load a
    given rate produces depends entirely on the cost model. Measuring the mean
    per-request work on a sample of the actual trace is more honest than a
    closed-form mean, because the lognormal's tail is clipped at 32k.
    """
    p_med, p_sigma, o_med, o_sigma, p_max = 600, 1.1, 180, 0.8, 4096
    kw = dict(p_med=p_med, p_sigma=p_sigma, o_med=o_med, o_sigma=o_sigma,
              p_max=p_max)
    probe = make_trace(n=3000, rate=1.0, seed=seed + 999, **kw)
    work = sum(cost.request_work(r.prompt_len, r.out_len) for r in probe) / len(probe)
    rate = util / work
    return make_trace(n=n, rate=rate, seed=seed, **kw), rate


# ---------------------------------------------------------------------------


def sweep(cost, trace, label):
    rows = []
    for ch in CHUNKS:
        reqs = copy.deepcopy(trace)
        # token_budget is set out of the way so that `chunk` alone is the
        # knob being swept; in a real engine the two are the same dial.
        st = simulate(reqs, cost, chunk=ch, token_budget=10 ** 9)
        r = report(reqs, st, label=("none" if ch is None else str(ch)))
        r["chunk"] = ch if ch is not None else 0
        r["util"] = round(st["util"], 3)
        rows.append(r)
        print(f"  chunk {str(ch):>5s}: tok/s {r['output_tok_s']:7.1f}  "
              f"TTFT p50 {r['ttft_p50']:7.2f} p99 {r['ttft_p99']:8.2f}  "
              f"ITL p50 {r['itl_p50']:.3f} p99 {r['itl_p99']:7.3f} "
              f"max {r['itl_max']:8.2f}")
    F[label] = rows
    return rows


def section_b(cost):
    print("\n=== B/C. chunk-size sweep (measured cost model) ===")
    trace, rate = build_trace(cost, n=1200, seed=4)
    plens = sorted(r.prompt_len for r in trace)
    F["trace"] = {
        "n": len(trace), "rate_req_s": round(rate, 4),
        "prompt_median": plens[len(plens) // 2],
        "prompt_p99": plens[int(len(plens) * .99)],
        "prompt_max": max(plens),
        "out_median": sorted(r.out_len for r in trace)[len(trace) // 2],
        "target_util": 0.55,
    }
    print("  trace:", F["trace"])
    rows = sweep(cost, trace, "C")

    none = [r for r in rows if r["chunk"] == 0][0]
    best_itl = min(rows, key=lambda r: r["itl_p99"])
    best_tp = max(rows, key=lambda r: r["output_tok_s"])
    F["B"] = {
        "no_chunk_itl_p99": none["itl_p99"],
        "no_chunk_itl_p50": none["itl_p50"],
        "no_chunk_itl_max": none["itl_max"],
        "no_chunk_itl_ratio": round(none["itl_p99"] / none["itl_p50"], 1),
        "best_itl_chunk": best_itl["chunk"],
        "best_itl_p99": best_itl["itl_p99"],
        "best_itl_max": best_itl["itl_max"],
        "itl_p99_improvement": round(none["itl_p99"] / best_itl["itl_p99"], 2),
        "itl_max_improvement": round(none["itl_max"] / best_itl["itl_max"], 2),
        "throughput_cost": round(1 - best_itl["output_tok_s"] / none["output_tok_s"], 4),
        "ttft_p50_change": round(best_itl["ttft_p50"] / none["ttft_p50"], 3),
        "ttft_p99_change": round(best_itl["ttft_p99"] / none["ttft_p99"], 3),
        "best_throughput_chunk": best_tp["chunk"],
    }
    print(f"  --> no chunking: ITL p99 is {F['B']['no_chunk_itl_ratio']}x its own p50, "
          f"worst stall {none['itl_max']} s")
    print(f"  --> best ITL p99 at chunk {best_itl['chunk']}: "
          f"{F['B']['itl_p99_improvement']}x better (worst stall "
          f"{F['B']['itl_max_improvement']}x better), "
          f"throughput {-F['B']['throughput_cost']:+.1%}, "
          f"TTFT p99 x{F['B']['ttft_p99_change']}")


def section_e(cost):
    print("\n=== E. sensitivity: the same sweep on an H100-shaped machine ===")
    # Published order-of-magnitude figures for a 7B-class model on one H100:
    # ~10,000 prefill tokens/s and ~25 ms per decode iteration at batch 32.
    fast = CostModel(base=0.008, per_decode=0.0005, per_prefill=0.0001,
                     per_key_read=2.0e-9)
    trace, rate = build_trace(fast, n=1200, seed=4)
    F["E_rate_req_s"] = round(rate, 3)
    print(f"  same trace shape, rate {rate:.2f} req/s to hit the same load")
    rows = sweep(fast, trace, "E")
    none = [r for r in rows if r["chunk"] == 0][0]
    best_itl = min(rows, key=lambda r: r["itl_p99"])
    F["E_summary"] = {
        "no_chunk_itl_p99": none["itl_p99"],
        "no_chunk_itl_max": none["itl_max"],
        "best_itl_chunk": best_itl["chunk"],
        "itl_p99_improvement": round(none["itl_p99"] / best_itl["itl_p99"], 2),
        "itl_max_improvement": round(none["itl_max"] / best_itl["itl_max"], 2),
        "throughput_cost": round(1 - best_itl["output_tok_s"] / none["output_tok_s"], 4),
        "ttft_p99_change": round(best_itl["ttft_p99"] / none["ttft_p99"], 3),
    }
    print(f"  --> {F['E_summary']['itl_p99_improvement']}x ITL p99 at chunk "
          f"{best_itl['chunk']}, throughput "
          f"{-F['E_summary']['throughput_cost']:+.1%}")


# ---------------------------------------------------------------------------


def write_csv():
    for key, name in [("C", "sweep_measured.csv"), ("E", "sweep_h100.csv")]:
        with open(os.path.join(OUT, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(F[key][0]), lineterminator="\n")
            w.writeheader()
            w.writerows(F[key])


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    a = f["A"]
    xs = [p[0] for p in a["decode_points"]]
    ys = [p[1] * 1000 for p in a["decode_points"]]
    ax[0].plot(xs, ys, "o", color="#c0392b", label="measured decode")
    ctx = a["decode_ctx"]
    fit = [(a["base_s"] + (a["per_decode_s"] + a["per_key_read_s"] * ctx) * x)
           * 1000 for x in xs]
    ax[0].plot(xs, fit, "-", color="#c0392b", alpha=.5, label="fit")
    ax2 = ax[0].twinx()
    xp = [p[0] for p in a["prefill_points"]]
    yp = [p[1] * 1000 for p in a["prefill_points"]]
    ax2.plot(xp, yp, "s", color="#2471a3", label="measured prefill")
    ax2.plot(xp, [(a["base_s"] + a["per_prefill_s"] * x
                   + a["per_key_read_s"] * x * (x + 1) / 2) * 1000 for x in xp],
             "-", color="#2471a3", alpha=.5)
    ax[0].set_xlabel("decode rows  /  prefill tokens")
    ax[0].set_ylabel("decode ms", color="#c0392b")
    ax2.set_ylabel("prefill ms", color="#2471a3")
    ax[0].set_title("A. cost model, fitted to this box")
    ax[0].legend(fontsize=7, loc="upper left")

    def lab(r):
        return "none" if r["chunk"] == 0 else str(r["chunk"])

    for i, (key, title) in enumerate([("C", "C. measured cost model"),
                                      ("E", "E. H100-shaped model")]):
        rows = f[key]
        a_ = ax[1 + i]
        x = range(len(rows))
        a_.plot(list(x), [r["itl_p99"] * 1000 for r in rows], "o-",
                color="#c0392b", label="ITL p99")
        a_.plot(list(x), [r["itl_p50"] * 1000 for r in rows], "s--",
                color="#e67e22", label="ITL p50")
        a_.set_yscale("log")
        a_.set_xticks(list(x))
        a_.set_xticklabels([lab(r) for r in rows], rotation=45, fontsize=8)
        a_.set_xlabel("chunk size (tokens)")
        a_.set_ylabel("inter-token latency (ms)")
        a_.set_title(title)
        a_.grid(alpha=.3)
        a_.legend(fontsize=8)

    rows = f["C"]
    a_ = ax[3]
    a_.plot([r["output_tok_s"] for r in rows], [r["itl_p99"] * 1000 for r in rows],
            "o-", color="#8e44ad")
    for r in rows:
        a_.annotate(lab(r), (r["output_tok_s"], r["itl_p99"] * 1000),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    a_.set_yscale("log")
    a_.set_xlabel("throughput (output tok/s)")
    a_.set_ylabel("ITL p99 (ms)")
    a_.set_title("D. the trade-off curve")
    a_.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "chunked_prefill.png"), dpi=110)
    print("wrote outputs/chunked_prefill.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    runner, tok = batchlib.load_runner()
    F["model"] = {"id": batchlib.MODEL_ID, "threads": batchlib.N_THREADS}
    cost = section_a(runner, tok)
    section_b(cost)
    section_e(cost)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv()
    plot()


if __name__ == "__main__":
    main()
