"""Project 25 -- n-gram (prompt-lookup) decoding.

The drafter with no model in it: to guess the next few tokens, look at the last
few tokens you just wrote, find where that same short phrase appeared earlier
in the prompt, and propose whatever followed it there. Cost: a list scan.

  A. Does it work at all, and does it change the output? (It must not.)
  B. Four workloads, three drafters: none, n-gram, and the 0.5B model from
     project 23. Where does a free drafter beat a real one?
  C. The matching parameters -- how long an n-gram to match on -- and the trap
     at n = 1.
  D. A pre-deployment predictor: measure the "copy rate" of a workload from
     logs you already have, and see whether it forecasts the speedup.
  E. The negative case, stated plainly.

    python3 run.py           # ~7 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "23-greedy-speculative-decoding"))

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

SNIPPET = (
    "def load_config(path):\n"
    "    with open(path) as f:\n"
    "        raw = f.read()\n"
    "    cfg = json.loads(raw)\n"
    "    cfg['timeout'] = int(cfg.get('timeout', 30))\n"
    "    return cfg\n"
)

WORKLOADS = {
    # Copy-heavy on purpose: the answer is the input with one word changed.
    "copy_edit": (
        "Repeat the following paragraph exactly, changing only the word "
        "\"Greek\" to \"Hellenic\" everywhere it appears. Output the "
        "paragraph and nothing else.\n\n" + PASSAGE
    ),
    # Code edit: same shape, different vocabulary.
    "code_edit": (
        "Rewrite this function so the default timeout is 60 instead of 30. "
        "Output the whole function and nothing else.\n\n"
        "```python\n" + SNIPPET + "```"
    ),
    # The answer is grounded in the passage but has to be re-worded.
    "summarize": (
        "Summarise the following passage in two sentences.\n\n" + PASSAGE
    ),
    # No source text at all -- nothing to copy from.
    "chat": "Explain in a short paragraph why the sky looks blue during the day.",
}


def copy_rate(prompt_ids, out_tokens, n=3):
    """Share of generated tokens an n-gram lookup could have predicted.

    Computed *after the fact* from a plain generation, using only text that
    was already available when each token was produced. That makes it a
    pre-deployment estimate: you can run it over yesterday's request logs
    without touching the serving path.
    """
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
# A + B. four workloads, three drafters
# ---------------------------------------------------------------------------


def section_ab(target, draft, tok):
    print("\n== A+B. four workloads, three drafters ==")
    tc = S.make_cache(target, max_len=1024)
    model_drafter = S.ModelDrafter(draft, max_len=1024)
    rows = []
    for name, prompt in WORKLOADS.items():
        ids = S.chat_ids(tok, prompt)
        base = S.greedy_decode(target, tc, ids, max_new=MAX_NEW)
        ng = S.speculative_greedy(target, S.NgramDrafter(max_n=4, min_n=2), tc,
                                  ids, k=K, max_new=MAX_NEW)
        md = S.speculative_greedy(target, model_drafter, tc, ids, k=K,
                                  max_new=MAX_NEW)
        rows.append({
            "workload": name,
            "prompt_tokens": int(ids.shape[1]),
            "text": tok.decode(base["tokens"]),
            "ngram_identical": base["tokens"] == ng["tokens"],
            "model_identical": base["tokens"] == md["tokens"],
            "baseline_decode_s": round(base["decode_s"], 3),
            "ngram_decode_s": round(ng["decode_s"], 3),
            "model_decode_s": round(md["decode_s"], 3),
            "ngram_speedup": round(base["decode_s"] / ng["decode_s"], 3),
            "model_speedup": round(base["decode_s"] / md["decode_s"], 3),
            "ngram_alpha": round(ng["tokens_per_iter"], 3),
            "model_alpha": round(md["tokens_per_iter"], 3),
            "ngram_accept": round(ng["conditional_acceptance"], 4),
            "model_accept": round(md["conditional_acceptance"], 4),
            "ngram_draft_s": round(ng["draft_s"], 4),
            "ngram_draft_share": round(ng["draft_s"] / ng["decode_s"], 5),
            "model_draft_share": round(md["draft_s"] / md["decode_s"], 4),
            "copy_rate_n3": round(copy_rate(ids[0].tolist(), base["tokens"], 3), 4),
            "copy_rate_n2": round(copy_rate(ids[0].tolist(), base["tokens"], 2), 4),
        })
        r = rows[-1]
        print(f"  {name:11s} n-gram {r['ngram_speedup']:.2f}x (alpha "
              f"{r['ngram_alpha']:.2f})   model {r['model_speedup']:.2f}x "
              f"(alpha {r['model_alpha']:.2f})   copy@3 {r['copy_rate_n3']:.2f}"
              f"   identical={r['ngram_identical']}")
    F["AB"] = {"k": K, "max_new": MAX_NEW, "rows": rows}


# ---------------------------------------------------------------------------
# C. how long an n-gram to match on
# ---------------------------------------------------------------------------


def section_c(target, tok):
    print("\n== C. match length ==")
    ids = S.chat_ids(tok, WORKLOADS["copy_edit"])
    tc = S.make_cache(target, max_len=1024)
    base = S.greedy_decode(target, tc, ids, max_new=MAX_NEW)
    rows = []
    for n in (1, 2, 3, 4, 6):
        d = S.NgramDrafter(max_n=n, min_n=n)
        r = S.speculative_greedy(target, d, tc, ids, k=K, max_new=MAX_NEW)
        rows.append({
            "n": n,
            "identical": base["tokens"] == r["tokens"],
            "hit_rate": round(d.hits / (d.hits + d.misses), 4),
            "acceptance": round(r["conditional_acceptance"], 4),
            "alpha": round(r["tokens_per_iter"], 3),
            "decode_s": round(r["decode_s"], 3),
            "speedup": round(base["decode_s"] / r["decode_s"], 3),
        })
        print(f"  n={n}  hit {rows[-1]['hit_rate']:.2f}  accept "
              f"{rows[-1]['acceptance']:.2f}  alpha {rows[-1]['alpha']:.2f}"
              f"  {rows[-1]['speedup']:.2f}x")
    F["C"] = {"baseline_decode_s": round(base["decode_s"], 3), "rows": rows}


# ---------------------------------------------------------------------------
# D. copy rate as a predictor
# ---------------------------------------------------------------------------


def section_d():
    print("\n== D. copy rate predicts the speedup ==")
    rows = F["AB"]["rows"]
    pairs = [(r["copy_rate_n3"], r["ngram_speedup"], r["workload"])
             for r in rows]
    # Rank correlation over 4 points -- reported as the ordering, because a
    # correlation coefficient on 4 samples means very little.
    by_copy = [w for _, _, w in sorted(pairs, reverse=True)]
    by_speed = [w for _, w in sorted(
        [(r["ngram_speedup"], r["workload"]) for r in rows], reverse=True)]
    F["D"] = {"order_by_copy_rate": by_copy, "order_by_speedup": by_speed,
              "same_order": by_copy == by_speed,
              "points": [{"workload": w, "copy_rate": c, "speedup": s}
                         for c, s, w in pairs]}
    print(f"  ranked by copy rate : {by_copy}")
    print(f"  ranked by speedup   : {by_speed}")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    rows = f["AB"]["rows"]
    names = [r["workload"] for r in rows]
    xs = range(len(rows))
    ax[0].bar([i - 0.2 for i in xs], [r["ngram_speedup"] for r in rows], 0.4,
              color="#27ae60", label="n-gram (free)")
    ax[0].bar([i + 0.2 for i in xs], [r["model_speedup"] for r in rows], 0.4,
              color="#2471a3", label="0.5B draft model")
    ax[0].axhline(1.0, color="k", lw=1, ls="--")
    ax[0].set_xticks(list(xs))
    ax[0].set_xticklabels(names, fontsize=8, rotation=15)
    ax[0].set_ylabel("speedup vs plain decoding")
    ax[0].set_title("B. free drafter vs a real one")
    ax[0].legend(fontsize=8)

    ax[1].bar([i - 0.2 for i in xs], [r["ngram_alpha"] for r in rows], 0.4,
              color="#27ae60", label="n-gram")
    ax[1].bar([i + 0.2 for i in xs], [r["model_alpha"] for r in rows], 0.4,
              color="#2471a3", label="0.5B draft")
    ax[1].axhline(1.0, color="k", lw=1, ls="--")
    ax[1].set_xticks(list(xs))
    ax[1].set_xticklabels(names, fontsize=8, rotation=15)
    ax[1].set_ylabel("tokens per target forward pass")
    ax[1].set_title("B. alpha (hardware-independent)")
    ax[1].legend(fontsize=8)

    c = f["C"]["rows"]
    ns = [r["n"] for r in c]
    ax[2].plot(ns, [r["hit_rate"] for r in c], "o-", color="#7f8c8d",
               label="match found")
    ax[2].plot(ns, [r["acceptance"] for r in c], "s-", color="#e67e22",
               label="accepted when tested")
    ax[2].plot(ns, [r["speedup"] for r in c], "^-", color="#27ae60",
               label="speedup")
    ax[2].axhline(1.0, color="k", lw=1, ls="--")
    ax[2].set_xlabel("n-gram length matched on")
    ax[2].set_title("C. match length: quantity vs quality")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    pts = f["D"]["points"]
    ax[3].scatter([p["copy_rate"] for p in pts], [p["speedup"] for p in pts],
                  s=70, color="#c0392b")
    for p in pts:
        ax[3].annotate(p["workload"], (p["copy_rate"], p["speedup"]),
                       textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax[3].axhline(1.0, color="k", lw=1, ls="--")
    ax[3].set_xlabel("copy rate measured from logs (3-gram)")
    ax[3].set_ylabel("measured n-gram speedup")
    ax[3].set_title("D. predict it before you deploy it")
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ngram_lookup.png"), dpi=110)
    print("wrote outputs/ngram_lookup.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    target, draft, tok, _ = S.load_pair()
    F["setup"] = {"target": S.TARGET_ID, "draft": S.DRAFT_ID, "k": K,
                  "max_new": MAX_NEW, "threads": S.N_THREADS}
    section_ab(target, draft, tok)
    section_c(target, tok)
    section_d()
    F["wall_s"] = round(time.time() - t0, 1)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
