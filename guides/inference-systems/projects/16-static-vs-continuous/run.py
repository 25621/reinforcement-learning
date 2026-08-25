"""Project 16 -- static vs. continuous batching.

Four questions:

  A. Does batching change what the model says? (It must not -- and a
     heterogeneous batch is exactly where a position bug would hide.)
  B. On one traffic trace, what do the two schedulers actually deliver:
     throughput, TTFT, end-to-end latency?
  C. Is the gap really about *variation*? Run the same test where every
     request is the same size and see whether static catches up.
  D. Where does static batching's time go? Split the waste into its three
     named sources.

    python3 run.py           # ~6 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import torch  # noqa: E402

import batchlib  # noqa: E402
from batchlib import SlotKV, make_workload, summarize  # noqa: E402
from schedulers import run_continuous, run_static  # noqa: E402

F = {}


# ---------------------------------------------------------------------------
# A. correctness
# ---------------------------------------------------------------------------


def section_a(runner, tok):
    print("\n=== A. batching must not change the output ===")
    prompts = ["The capital of France is",
               "Once upon a time there was a small",
               "In 1969 the first human"]
    n_new = 10

    def solo(p):
        pool = SlotKV(runner.n_layers, 1, runner.n_kv_heads, runner.d_head, 128)
        ids = tok(p, return_tensors="pt").input_ids
        lg, _ = runner.prefill(pool, [0], ids, [ids.shape[1]], count=False)
        out, cur = [int(lg.argmax(-1))], ids.shape[1]
        for _ in range(n_new - 1):
            lg, _ = runner.decode_step(pool, [0], [out[-1]], [cur], count=False)
            cur += 1
            out.append(int(lg.argmax(-1)))
        return out

    ref = [solo(p) for p in prompts]

    # (1) heterogeneous batch: three rows, three different positions
    pool = SlotKV(runner.n_layers, 3, runner.n_kv_heads, runner.d_head, 128)
    idl = [tok(p, return_tensors="pt").input_ids for p in prompts]
    outs, cur = [[], [], []], []
    for j, ids in enumerate(idl):
        lg, _ = runner.prefill(pool, [j], ids, [ids.shape[1]], count=False)
        outs[j].append(int(lg.argmax(-1)))
        cur.append(ids.shape[1])
    for _ in range(n_new - 1):
        lg, _ = runner.decode_step(pool, [0, 1, 2], [o[-1] for o in outs], cur,
                                   count=False)
        nx = lg.argmax(-1).tolist()
        for j in range(3):
            outs[j].append(nx[j])
            cur[j] += 1
    het_match = all(ref[j] == outs[j] for j in range(3))

    # (2) right-padded prefill: does the padding leak into a real row?
    pool = SlotKV(runner.n_layers, 3, runner.n_kv_heads, runner.d_head, 128)
    pmax = max(i.shape[1] for i in idl)
    ids = torch.zeros(3, pmax, dtype=torch.long)
    for j, i in enumerate(idl):
        ids[j, :i.shape[1]] = i[0]
    lens = [int(i.shape[1]) for i in idl]
    lg, _ = runner.prefill(pool, [0, 1, 2], ids, lens, count=False)
    pad_first = lg.argmax(-1).tolist()
    pad_match = all(pad_first[j] == ref[j][0] for j in range(3))

    print(f"  heterogeneous batch == solo : {het_match}")
    print(f"  padded prefill    == solo   : {pad_match}")
    F["A"] = {
        "prompt_lens": lens,
        "heterogeneous_batch_identical": het_match,
        "padded_prefill_identical": pad_match,
        "texts": [tok.decode(o) for o in outs],
    }


# ---------------------------------------------------------------------------
# B / C. head to head
# ---------------------------------------------------------------------------


def one_run(runner, reqs, kind, **kw):
    reqs = copy.deepcopy(reqs)
    wall = run_static(runner, reqs, **kw) if kind == "static" \
        else run_continuous(runner, reqs, **kw)
    s = summarize(reqs, wall, label=kind)
    c = runner.counters
    s.update({
        "forward_passes": c.forward_passes,
        "model_s": round(c.model_s, 2),
        "pad_slot_frac": round(c.pad_slot_frac, 4),
        "pad_flop_frac": round(c.pad_flop_frac, 4),
        "useful_tflops": round(c.useful_flops / 1e12, 2),
        "pad_tflops": round(c.pad_flops / 1e12, 2),
        "idle_frac": round(1.0 - c.model_s / wall, 4),
    })
    return s


def section_b(runner, tok):
    print("\n=== B. one trace, two schedulers ===")
    reqs = make_workload(tok, n=40, rate=5.0, seed=7)
    F["workload"] = {
        "n": 40, "rate_per_s": 5.0,
        "prompt_len_median": sorted(r.prompt_len for r in reqs)[20],
        "prompt_len_max": max(r.prompt_len for r in reqs),
        "out_len_median": sorted(r.max_new for r in reqs)[20],
        "out_len_max": max(r.max_new for r in reqs),
    }
    print("  workload:", F["workload"])
    rows = []
    for kind, kw in [("static", dict(batch_size=8)),
                     ("continuous", dict(n_slots=8))]:
        s = one_run(runner, reqs, kind, **kw)
        rows.append(s)
        print(f"  {kind:11s} wall={s['wall_s']:7.2f}s  tok/s={s['throughput_tok_s']:6.2f}"
              f"  ttft50={s['ttft_p50']:6.2f} ttft99={s['ttft_p99']:6.2f}"
              f"  e2e50={s['e2e_p50']:6.2f}  padflops={s['pad_flop_frac']:.1%}")
    F["B"] = rows


def section_c(runner, tok):
    print("\n=== C. the control: same trace shape, NO length variation ===")
    reqs = make_workload(tok, n=40, rate=5.0, seed=7)
    med_p = sorted(r.prompt_len for r in reqs)[20]
    med_o = sorted(r.max_new for r in reqs)[20]
    flat = copy.deepcopy(reqs)
    for r in flat:
        r.prompt_ids = r.prompt_ids[:med_p] + [11] * max(0, med_p - r.prompt_len)
        r.max_new = med_o
    print(f"  every request: prompt={med_p}, output={med_o}")
    rows = []
    for kind, kw in [("static", dict(batch_size=8)),
                     ("continuous", dict(n_slots=8))]:
        s = one_run(runner, flat, kind, **kw)
        rows.append(s)
        print(f"  {kind:11s} wall={s['wall_s']:7.2f}s  tok/s={s['throughput_tok_s']:6.2f}"
              f"  ttft50={s['ttft_p50']:6.2f}  padflops={s['pad_flop_frac']:.1%}")
    F["C"] = {"prompt_len": med_p, "out_len": med_o, "rows": rows}


# ---------------------------------------------------------------------------
# D. where does static lose?
# ---------------------------------------------------------------------------


def section_d(runner, tok):
    print("\n=== D. decomposing static batching's loss ===")
    reqs = make_workload(tok, n=40, rate=5.0, seed=7)
    groups = []
    srt = sorted(reqs, key=lambda r: r.arrive)
    for i in range(0, len(srt), 8):
        g = srt[i:i + 8]
        groups.append({
            "arrival_spread_s": round(g[-1].arrive - g[0].arrive, 3),
            "prompt_tokens_real": sum(r.prompt_len for r in g),
            "prompt_tokens_padded": len(g) * max(r.prompt_len for r in g),
            "decode_slots_real": sum(r.max_new - 1 for r in g),
            "decode_slots_padded": len(g) * (max(r.max_new for r in g) - 1),
        })
    tot = {k: sum(g[k] for g in groups) for k in groups[0]}
    d = {
        "groups": groups,
        "arrival_wait_total_s": round(tot["arrival_spread_s"], 2),
        "prompt_pad_frac": round(1 - tot["prompt_tokens_real"] / tot["prompt_tokens_padded"], 4),
        "decode_pad_frac": round(1 - tot["decode_slots_real"] / tot["decode_slots_padded"], 4),
    }
    print(f"  head-of-batch arrival wait, summed : {d['arrival_wait_total_s']} s")
    print(f"  prompt tokens that are padding     : {d['prompt_pad_frac']:.1%}")
    print(f"  decode slots that are padding      : {d['decode_pad_frac']:.1%}")
    F["D"] = d


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))
    cols = {"static": "#c0392b", "continuous": "#2471a3"}

    b = {r["label"]: r for r in f["B"]}
    c = {r["label"]: r for r in f["C"]["rows"]}

    names = ["static", "continuous"]
    x = range(2)
    ax[0].bar(x, [b[n]["throughput_tok_s"] for n in names],
              color=[cols[n] for n in names])
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(names)
    ax[0].set_ylabel("output tokens / s")
    ax[0].set_title("B. throughput (mixed lengths)")
    for i, n in enumerate(names):
        ax[0].text(i, b[n]["throughput_tok_s"], f"{b[n]['throughput_tok_s']:.1f}",
                   ha="center", va="bottom")

    w = 0.35
    for i, n in enumerate(names):
        ax[1].bar([0 + (i - .5) * w, 1 + (i - .5) * w],
                  [b[n]["ttft_p50"], b[n]["ttft_p99"]], w, color=cols[n], label=n)
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["TTFT p50", "TTFT p99"])
    ax[1].set_ylabel("seconds")
    ax[1].set_title("B. time to first token")
    ax[1].legend(fontsize=8)

    d = f["D"]
    ax[2].bar([0, 1, 2],
              [d["prompt_pad_frac"] * 100, d["decode_pad_frac"] * 100,
               b["static"]["pad_flop_frac"] * 100],
              color=["#2471a3", "#e67e22", "#c0392b"])
    ax[2].set_xticks([0, 1, 2])
    ax[2].set_xticklabels(["prompt\npadding", "generation\npadding",
                           "all FLOPs\n(measured)"], fontsize=8)
    ax[2].set_ylabel("% wasted")
    ax[2].set_title("D. where static's work goes")
    for i, v in enumerate([d["prompt_pad_frac"], d["decode_pad_frac"],
                           b["static"]["pad_flop_frac"]]):
        ax[2].text(i, v * 100, f"{v:.1%}", ha="center", va="bottom")

    r = [b[n]["throughput_tok_s"] for n in names]
    rc = [c[n]["throughput_tok_s"] for n in names]
    ax[3].bar([0, 1], [r[1] / r[0], rc[1] / rc[0]], color=["#2471a3", "#7f8c8d"])
    ax[3].axhline(1.0, color="k", lw=1, ls="--")
    ax[3].set_xticks([0, 1])
    ax[3].set_xticklabels(["mixed lengths", "uniform (control)"])
    ax[3].set_ylabel("continuous / static throughput")
    ax[3].set_title("C. the gap is variation")
    for i, val in enumerate([r[1] / r[0], rc[1] / rc[0]]):
        ax[3].text(i, val, f"{val:.2f}x", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "static_vs_continuous.png"), dpi=110)
    print("wrote outputs/static_vs_continuous.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    runner, tok = batchlib.load_runner()
    F["model"] = {"id": batchlib.MODEL_ID, "layers": runner.n_layers,
                  "kv_heads": runner.n_kv_heads, "d_head": runner.d_head,
                  "threads": batchlib.N_THREADS}
    section_a(runner, tok)
    section_b(runner, tok)
    section_c(runner, tok)
    section_d(runner, tok)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()


if __name__ == "__main__":
    main()
