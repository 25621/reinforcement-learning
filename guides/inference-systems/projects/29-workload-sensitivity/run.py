"""Project 29 -- workload sensitivity.

Six routes, three decoding configurations, one question: why does the same
speculative setup produce a 3x speedup on one endpoint and nothing on another?

  A. The table. Six workloads x {no speculation, 0.5B draft, n-gram lookup}.
  B. Two predictors, both measured from a single teacher-forced pass over
     text the model already produced: how *predictable* the text is (entropy),
     and how much the draft *agrees* with the target (top-1 agreement, TV).
  C. Do the predictors work? Agreement should predict the model drafter's
     acceptance almost exactly; copy rate should predict the n-gram drafter's.
  D. Per-route drafter selection vs one fleet-wide setting -- what the choice
     is worth.
  E. What you can and cannot promise a capacity planner.

    python3 run.py           # ~6 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "23-greedy-speculative-decoding"))

import torch  # noqa: E402

import speclib as S  # noqa: E402

K = 4
MAX_NEW = 48
F = {}

PASSAGE = (
    "The Antikythera mechanism is an ancient Greek hand-powered device that "
    "has been described as the oldest known analogue computer. It was used to "
    "predict astronomical positions and eclipses decades in advance, and to "
    "track the four-year cycle of the ancient Olympic Games. The artefact was "
    "recovered in 1901 from a shipwreck off the coast of the Greek island of "
    "Antikythera, and its complexity was not matched by any known device for "
    "well over a thousand years."
)

WORKLOADS = {
    "chat": (
        "Explain in a short paragraph why the sky looks blue during the day."
    ),
    "reasoning": (
        "A train leaves at 09:15 and arrives at 13:40. It stops twice for 12 "
        "minutes each. How long was it actually moving? Think step by step."
    ),
    "summarize": (
        "Summarise the following passage in two sentences.\n\n" + PASSAGE
    ),
    "code": (
        "Complete this Python function. Repeat the whole function in your "
        "answer.\n\n"
        "def normalise_scores(scores):\n"
        "    total = sum(scores)\n"
        "    if total == 0:\n"
        "        return [0.0 for s in scores]\n"
        "    return [\n"
    ),
    "json": (
        "Return a JSON object with the keys \"name\", \"city\", \"country\" "
        "and \"population\" for the city of Lyon, France. Output JSON only."
    ),
    "copy_edit": (
        "Repeat the following paragraph exactly, changing only the word "
        "\"Greek\" to \"Hellenic\" everywhere it appears. Output the "
        "paragraph and nothing else.\n\n" + PASSAGE
    ),
}


# ---------------------------------------------------------------------------
# predictors, all from one teacher-forced pass
# ---------------------------------------------------------------------------


def probe(target, draft, tc, dc, prompt_ids, out_tokens):
    """Feed prompt+answer through both models once and read three numbers.

    This is the cheap, offline version of everything projects 23-26 measured
    the expensive way. It needs no speculative loop, only text the model has
    already produced -- so it can be run over yesterday's logs.
    """
    full = torch.cat([prompt_ids, torch.tensor([out_tokens])], dim=1)
    p0 = prompt_ids.shape[1]
    tc.reset()
    dc.reset()
    tl = target.forward(full, tc, start_pos=0)[0]
    dl = draft.forward(full, dc, start_pos=0)[0]
    # Position j predicts token j+1, so the answer tokens are predicted from
    # positions p0-1 .. len(full)-2.
    tl = tl[p0 - 1:-1]
    dl = dl[p0 - 1:-1]
    tp = torch.softmax(tl.float(), dim=-1)
    dp = torch.softmax(dl.float(), dim=-1)
    ent = float((-(tp * torch.log(tp.clamp(min=1e-12))).sum(-1)).mean())
    top1 = float((tl.argmax(-1) == dl.argmax(-1)).float().mean())
    tv = float((0.5 * (tp - dp).abs().sum(-1)).mean())
    conf = float(tp.max(-1).values.mean())
    return {"entropy_nats": round(ent, 4),
            "entropy_bits": round(ent / math.log(2), 4),
            "target_top1_prob": round(conf, 4),
            "draft_agreement": round(top1, 4),
            "mean_tv": round(tv, 4)}


def copy_rate(prompt_ids, out_tokens, n=3):
    seq = list(prompt_ids)
    hits = 0
    for t in out_tokens:
        pat = seq[-n:]
        pred = None
        for i in range(len(seq) - n - 1, -1, -1):
            if seq[i:i + n] == pat:
                pred = seq[i + n]
                break
        if pred == t:
            hits += 1
        seq.append(t)
    return hits / len(out_tokens) if out_tokens else 0.0


# ---------------------------------------------------------------------------
# A + B. the table
# ---------------------------------------------------------------------------


def section_ab(target, draft, tok):
    print("\n== A+B. six workloads ==")
    tc = S.make_cache(target, max_len=1024)
    dc = S.make_cache(draft, max_len=1024)
    md = S.ModelDrafter(draft, max_len=1024)
    rows = []
    for name, prompt in WORKLOADS.items():
        ids = S.chat_ids(tok, prompt)
        base = S.greedy_decode(target, tc, ids, max_new=MAX_NEW)
        base_tok_s = MAX_NEW / base["decode_s"]
        m = S.speculative_greedy(target, md, tc, ids, k=K, max_new=MAX_NEW)
        g = S.speculative_greedy(target, S.NgramDrafter(max_n=4, min_n=2), tc,
                                 ids, k=K, max_new=MAX_NEW)
        pr = probe(target, draft, tc, dc, ids, base["tokens"])
        row = {
            "workload": name,
            "prompt_tokens": int(ids.shape[1]),
            "text": tok.decode(base["tokens"]),
            "baseline_tok_s": round(base_tok_s, 3),
            "model_identical": base["tokens"] == m["tokens"],
            "ngram_identical": base["tokens"] == g["tokens"],
            "model_accept": round(m["conditional_acceptance"], 4),
            "model_alpha": round(m["tokens_per_iter"], 3),
            "model_speedup": round(m["produced"] / m["decode_s"] / base_tok_s, 3),
            "ngram_accept": round(g["conditional_acceptance"], 4),
            "ngram_alpha": round(g["tokens_per_iter"], 3),
            "ngram_speedup": round(g["produced"] / g["decode_s"] / base_tok_s, 3),
            "copy_rate": round(copy_rate(ids[0].tolist(), base["tokens"], 3), 4),
        }
        row.update(pr)
        rows.append(row)
        print(f"  {name:10s} model {row['model_speedup']:.2f}x "
              f"(acc {row['model_accept']:.2f})  n-gram "
              f"{row['ngram_speedup']:.2f}x (acc {row['ngram_accept']:.2f})  "
              f"entropy {row['entropy_bits']:.2f} bits  agree "
              f"{row['draft_agreement']:.2f}  copy {row['copy_rate']:.2f}")
    F["A"] = {"k": K, "max_new": MAX_NEW, "rows": rows}


# ---------------------------------------------------------------------------
# C. do the predictors work?
# ---------------------------------------------------------------------------


def corr(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    dx = math.sqrt(sum((a - mx) ** 2 for a in xs))
    dy = math.sqrt(sum((b - my) ** 2 for b in ys))
    return num / (dx * dy) if dx and dy else 0.0


def section_c():
    print("\n== C. predictors ==")
    rows = F["A"]["rows"]
    tests = [
        ("draft_agreement -> model acceptance", "draft_agreement",
         "model_accept"),
        ("1 - mean_tv -> model acceptance", None, "model_accept"),
        ("entropy (bits) -> model acceptance", "entropy_bits", "model_accept"),
        ("copy_rate -> n-gram alpha", "copy_rate", "ngram_alpha"),
        ("entropy (bits) -> n-gram alpha", "entropy_bits", "ngram_alpha"),
    ]
    out = []
    for label, xk, yk in tests:
        xs = ([1 - r["mean_tv"] for r in rows] if xk is None
              else [r[xk] for r in rows])
        ys = [r[yk] for r in rows]
        out.append({"test": label, "r": round(corr(xs, ys), 4)})
        print(f"  {label:38s} r = {out[-1]['r']:+.3f}")
    # how close is agreement to acceptance, in absolute terms?
    gaps = [r["model_accept"] - r["draft_agreement"] for r in rows]
    F["C"] = {"correlations": out,
              "mean_abs_gap_agreement_vs_acceptance":
                  round(sum(abs(g) for g in gaps) / len(gaps), 4),
              "gaps": [round(g, 4) for g in gaps]}
    print(f"  |acceptance - agreement| averages "
          f"{F['C']['mean_abs_gap_agreement_vs_acceptance']:.3f}")


# ---------------------------------------------------------------------------
# D. routing
# ---------------------------------------------------------------------------


def section_d():
    print("\n== D. one setting for the fleet, or one per route? ==")
    rows = F["A"]["rows"]
    n = len(rows)
    # Geometric mean: speedups are ratios, so averaging them arithmetically
    # over-weights the big wins.
    def gmean(vals):
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    strategies = {
        "no speculation": [1.0] * n,
        "0.5B draft everywhere": [r["model_speedup"] for r in rows],
        "n-gram everywhere": [r["ngram_speedup"] for r in rows],
        "best per route": [max(r["model_speedup"], r["ngram_speedup"])
                           for r in rows],
        "route by copy rate > 0.4": [
            r["ngram_speedup"] if r["copy_rate"] > 0.4 else r["model_speedup"]
            for r in rows],
    }
    out = []
    for label, vals in strategies.items():
        out.append({"strategy": label, "gmean": round(gmean(vals), 4),
                    "worst": round(min(vals), 3),
                    "per_workload": [round(v, 3) for v in vals]})
        print(f"  {label:26s} gmean {out[-1]['gmean']:.3f}x  worst "
              f"{out[-1]['worst']:.2f}x")

    # Where should the routing threshold sit? Pure arithmetic on the table
    # above -- no model runs -- which is exactly how you would tune it from
    # logs.
    sweep = []
    for thr in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.01):
        vals = [r["ngram_speedup"] if r["copy_rate"] > thr
                else r["model_speedup"] for r in rows]
        sweep.append({"threshold": thr, "gmean": round(gmean(vals), 4),
                      "worst": round(min(vals), 3)})
    best = max(sweep, key=lambda s: s["gmean"])
    F["D"] = {"workloads": [r["workload"] for r in rows], "strategies": out,
              "threshold_sweep": sweep, "best_threshold": best}
    print("  copy-rate routing threshold sweep:")
    for s in sweep:
        mark = "  <-- best" if s is best else ""
        print(f"    > {s['threshold']:.2f}  gmean {s['gmean']:.3f}x  worst "
              f"{s['worst']:.2f}x{mark}")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))

    rows = f["A"]["rows"]
    names = [r["workload"] for r in rows]
    xs = range(len(rows))
    ax[0].bar([i - 0.2 for i in xs], [r["model_speedup"] for r in rows], 0.4,
              color="#2471a3", label="0.5B draft model")
    ax[0].bar([i + 0.2 for i in xs], [r["ngram_speedup"] for r in rows], 0.4,
              color="#27ae60", label="n-gram lookup")
    ax[0].axhline(1.0, color="k", lw=1, ls="--")
    ax[0].set_xticks(list(xs))
    ax[0].set_xticklabels(names, fontsize=8, rotation=20)
    ax[0].set_ylabel("speedup vs plain decoding")
    ax[0].set_title("A. same setup, six routes")
    ax[0].legend(fontsize=8)

    ax[1].scatter([r["draft_agreement"] for r in rows],
                  [r["model_accept"] for r in rows], s=70, color="#2471a3")
    lo = min([r["draft_agreement"] for r in rows]
             + [r["model_accept"] for r in rows]) - 0.05
    ax[1].plot([lo, 1.0], [lo, 1.0], "k--", lw=1, label="y = x")
    for r in rows:
        ax[1].annotate(r["workload"], (r["draft_agreement"], r["model_accept"]),
                       textcoords="offset points", xytext=(6, 3), fontsize=7)
    ax[1].set_xlabel("top-1 agreement, measured offline")
    ax[1].set_ylabel("acceptance in the speculative loop")
    ax[1].set_title("C. agreement predicts acceptance")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    ax[2].scatter([r["copy_rate"] for r in rows],
                  [r["ngram_alpha"] for r in rows], s=70, color="#27ae60")
    for r in rows:
        ax[2].annotate(r["workload"], (r["copy_rate"], r["ngram_alpha"]),
                       textcoords="offset points", xytext=(6, 3), fontsize=7)
    ax[2].axhline(1.0, color="k", lw=1, ls="--")
    ax[2].set_xlabel("copy rate, measured offline")
    ax[2].set_ylabel("n-gram alpha (tokens per target pass)")
    ax[2].set_title("C. copy rate predicts lookup")
    ax[2].grid(alpha=0.3)

    st = f["D"]["strategies"]
    ax[3].barh(range(len(st)), [s["gmean"] for s in st],
               color=["#7f8c8d", "#2471a3", "#27ae60", "#8e44ad", "#e67e22"])
    ax[3].set_yticks(range(len(st)))
    ax[3].set_yticklabels([s["strategy"] for s in st], fontsize=7)
    ax[3].axvline(1.0, color="k", lw=1, ls="--")
    ax[3].set_xlabel("geometric-mean speedup over 6 routes")
    ax[3].set_xlim(0, max(s["gmean"] for s in st) * 1.3)
    ax[3].set_title("D. what routing is worth")
    for i, s in enumerate(st):
        ax[3].text(s["gmean"], i, f" {s['gmean']:.2f}x", va="center",
                   fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "workload_sensitivity.png"), dpi=110)
    print("wrote outputs/workload_sensitivity.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    target, draft, tok, _ = S.load_pair()
    F["setup"] = {"target": S.TARGET_ID, "draft": S.DRAFT_ID, "k": K,
                  "max_new": MAX_NEW, "threads": S.N_THREADS}
    section_ab(target, draft, tok)
    section_c()
    section_d()
    F["wall_s"] = round(time.time() - t0, 1)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
