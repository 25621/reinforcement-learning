"""Project 40 -- Latency vs throughput.

Sweeps batch size on the from-scratch engine of project 39 and measures both
sides of the trade: server tokens/sec, and what each individual user feels
(TTFT, TPOT, end-to-end latency).  Then it picks the operating point three ways
-- the geometric "knee", a latency SLA, and goodput -- and shows that they do
not agree.

Runs in about 70 seconds on 12 CPU threads.
"""

import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "39-deploy-with-vllm"))
import servelib as S  # noqa: E402

OUT = S.outdir(__file__)
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
CONTEXTS = [128, 512, 2048]
PROMPT_LEN, NEW_TOKENS = 32, 16
TPOT_SLA_MS = 200.0        # "at least 5 tokens/second for every user"
TTFT_SLA_S = 3.0

results = {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------- the sweep
def sweep(w):
    """End-to-end: prefill B prompts, then decode NEW_TOKENS tokens."""
    prompt = S.prompt_ids(w.tok, "A serving engine has two customers: the user "
                                 "waiting for a reply, and the accountant paying "
                                 "for the hardware. ", PROMPT_LEN)
    rows = []
    for B in BATCHES:
        eng = S.Engine(w, num_blocks=B * ((PROMPT_LEN + NEW_TOKENS) // 16 + 2))
        seqs = [S.Sequence(i, prompt, max_new=NEW_TOKENS) for i in range(B)]
        t0 = time.perf_counter()
        lg = eng.forward(seqs, [s.prompt_ids for s in seqs])
        ttft = time.perf_counter() - t0
        for s, g in zip(seqs, lg):
            s.out_ids.append(S.greedy(g))
        steps = []
        for _ in range(NEW_TOKENS - 1):
            t1 = time.perf_counter()
            lg = eng.decode_step(seqs)
            steps.append(time.perf_counter() - t1)
            for s, g in zip(seqs, lg):
                s.out_ids.append(S.greedy(g))
        tpot = sorted(steps)[len(steps) // 2]
        total = ttft + sum(steps)
        rows.append(dict(
            batch=B,
            ttft_s=ttft,
            tpot_ms=tpot * 1e3,
            latency_s=total,
            throughput_tok_s=B * NEW_TOKENS / total,
            decode_tok_s=B / tpot,
            requests_per_s=B / total,
            user_tok_s=1.0 / tpot,
            meets_sla=bool(tpot * 1e3 <= TPOT_SLA_MS and ttft <= TTFT_SLA_S),
        ))
        r = rows[-1]
        log(f"   batch {B:3d}: TTFT {r['ttft_s']:5.2f} s | TPOT {r['tpot_ms']:6.1f} ms "
            f"| user {r['user_tok_s']:5.2f} tok/s | server {r['decode_tok_s']:7.2f} tok/s "
            f"| {'OK ' if r['meets_sla'] else 'SLA violated'}")
    return rows


# ------------------------------------------------------- knee of the curve
def knee(xs, ys):
    """Kneedle: the point furthest from the straight line joining the ends.

    Named after the 2011 paper 'Finding a Kneedle in a Haystack'.  Both axes are
    rescaled to [0, 1] first, otherwise 'furthest' would just mean 'largest unit'.
    """
    x0, x1, y0, y1 = xs[0], xs[-1], ys[0], ys[-1]
    nx = [(x - x0) / (x1 - x0) for x in xs]
    ny = [(y - y0) / (y1 - y0) for y in ys]
    d = [abs(b - a) for a, b in zip(nx, ny)]      # distance to the diagonal
    return max(range(len(d)), key=lambda i: d[i]), d


# ---------------------------------------------- how context length moves it
def context_grid(w):
    grid = {}
    for ctx in CONTEXTS:
        per_seq_blocks = ctx // 16 + 2
        row = {}
        for B in [1, 4, 16, 32, 64]:
            eng = S.Engine(w, num_blocks=B * per_seq_blocks, gather_stats=True)
            seqs = S.synthetic_seqs(eng, B, ctx)
            step = S.time_decode(eng, seqs, rounds=3)
            row[B] = dict(step_ms=step * 1e3, tok_s=B / step,
                          kv_MB=B * ctx * eng.pool.bytes_per_token() / 1e6,
                          gather_ms=eng.gather_time / 4 * 1e3)
            del eng, seqs
        grid[ctx] = row
        log("   context %4d: " % ctx + "  ".join(
            f"B{b}={row[b]['tok_s']:6.1f} tok/s" for b in row))
    return grid


# -------------------------------------------------------------------- figures
def make_plots(res):
    rows = res["sweep"]
    B = [r["batch"] for r in rows]
    thr = [r["decode_tok_s"] for r in rows]
    lat = [r["tpot_ms"] for r in rows]
    ki = res["knee"]["index"]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), constrained_layout=True)

    ax[0].plot(thr, lat, "o-", color="#1f77b4")
    ax[0].plot([thr[0], thr[-1]], [lat[0], lat[-1]], "--", color="#999", lw=1,
               label="chord")
    ax[0].plot(thr[ki], lat[ki], "*", ms=18, color="#d62728",
               label=f"knee (batch {B[ki]})")
    ax[0].axhline(res["tpot_sla_ms"], color="#2ca02c", ls=":",
                  label=f"SLA {res['tpot_sla_ms']:.0f} ms")
    for r in rows:
        ax[0].annotate(str(r["batch"]), (r["decode_tok_s"], r["tpot_ms"]),
                       fontsize=7, xytext=(3, -9), textcoords="offset points")
    ax[0].set_xlabel("server throughput (decode tokens/s)")
    ax[0].set_ylabel("TPOT: time per output token (ms)")
    ax[0].set_title("The trade, drawn once")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    ax[1].plot(B, [r["decode_tok_s"] for r in rows], "o-", color="#1f77b4",
               label="server tok/s")
    ax[1].plot(B, [r["user_tok_s"] * 10 for r in rows], "s-", color="#d62728",
               label="per-user tok/s (x10)")
    ax[1].plot(B, [r["ttft_s"] * 10 for r in rows], "^-", color="#ff7f0e",
               label="TTFT s (x10)")
    ax[1].set_xscale("log", base=2)
    ax[1].set_yscale("log")
    ax[1].set_xlabel("batch size")
    ax[1].set_title("Same run, three points of view")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3)

    grid = res["context_grid"]
    for ctx, color in zip(grid, ["#1f77b4", "#ff7f0e", "#d62728"]):
        row = {int(b): v for b, v in grid[ctx].items()}   # JSON keys come back as str
        bs = sorted(row)
        ax[2].plot(bs, [row[b]["tok_s"] for b in bs], "o-",
                   color=color, label=f"context {ctx}")
    ax[2].set_xscale("log", base=2)
    ax[2].set_xlabel("batch size")
    ax[2].set_ylabel("decode tokens/s")
    ax[2].set_title("Long contexts flatten the curve")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)
    fig.savefig(f"{OUT}/latency_throughput.png", dpi=130)
    log(f"   wrote {OUT}/latency_throughput.png")


def main():
    S.setup()
    t0 = time.time()
    w = S.Weights(S.SMALL)
    log("A. batch-size sweep, prompt %d tokens, %d new tokens" % (PROMPT_LEN, NEW_TOKENS))
    rows = sweep(w)
    results["sweep"] = rows
    results["tpot_sla_ms"] = TPOT_SLA_MS
    results["ttft_sla_s"] = TTFT_SLA_S

    ki, dists = knee([r["decode_tok_s"] for r in rows], [r["tpot_ms"] for r in rows])
    results["knee"] = dict(index=ki, batch=rows[ki]["batch"], distances=dists)
    log(f"\nB. knee of the throughput/latency curve: batch {rows[ki]['batch']} "
        f"({rows[ki]['decode_tok_s']:.0f} tok/s at {rows[ki]['tpot_ms']:.0f} ms)")

    ok = [r for r in rows if r["meets_sla"]]
    best_sla = max(ok, key=lambda r: r["decode_tok_s"]) if ok else None
    results["sla_pick"] = best_sla
    if best_sla:
        log(f"   largest batch inside the SLA ({TPOT_SLA_MS:.0f} ms TPOT, "
            f"{TTFT_SLA_S:.0f} s TTFT): batch {best_sla['batch']} at "
            f"{best_sla['decode_tok_s']:.0f} tok/s")
    goodput = [(r["batch"], r["decode_tok_s"] if r["meets_sla"] else 0.0) for r in rows]
    results["goodput"] = goodput
    log("   goodput (throughput that actually meets the SLA): " +
        ", ".join(f"B{b}={g:.0f}" for b, g in goodput))

    # The knee is one number; a product has a different one for every promise
    # it makes.  Re-pick the batch size under a range of TPOT budgets.
    table = []
    for sla in [110, 130, 150, 200, 300, 500, 1e9]:
        ok2 = [r for r in rows if r["tpot_ms"] <= sla]
        pick = max(ok2, key=lambda r: r["decode_tok_s"]) if ok2 else None
        table.append(dict(sla_ms=sla, batch=pick["batch"] if pick else None,
                          tok_s=pick["decode_tok_s"] if pick else 0.0,
                          user_tok_s=pick["user_tok_s"] if pick else 0.0))
        log(f"   TPOT budget {sla:>6.0f} ms -> batch {table[-1]['batch']}, "
            f"{table[-1]['tok_s']:.0f} server tok/s, "
            f"{table[-1]['user_tok_s']:.1f} tok/s per user")
    results["sla_table"] = table

    log("\nC. does the knee move with context length?")
    results["context_grid"] = context_grid(w)

    make_plots(results)
    results["total_seconds"] = time.time() - t0
    S.save_findings(__file__, results)
    with open(f"{OUT}/findings.csv", "w") as f:
        f.write("batch,ttft_s,tpot_ms,user_tok_s,server_tok_s,latency_s,meets_sla\n")
        for r in rows:
            f.write(f"{r['batch']},{r['ttft_s']:.3f},{r['tpot_ms']:.1f},"
                    f"{r['user_tok_s']:.3f},{r['decode_tok_s']:.2f},"
                    f"{r['latency_s']:.2f},{int(r['meets_sla'])}\n")
    log(f"\ntotal {results['total_seconds']:.0f}s")


if __name__ == "__main__":
    if "--plot" in sys.argv:
        make_plots(S.load_findings(__file__))
    else:
        main()
