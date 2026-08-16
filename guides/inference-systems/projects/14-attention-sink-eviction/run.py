"""Project 14 -- attention-sink eviction.

Caps the KV cache at a fixed number of tokens and compares six ways of
choosing which tokens survive. Two measurements:

  A. Language-modelling quality (teacher-forced perplexity) at long context.
  B. A retrieval probe: a fact is planted at the very start of a long prompt,
     and we check whether the model can still recall it after eviction. This
     is where the policies separate, because perplexity is dominated by local
     structure and barely notices a missing distant token.

    python3 run.py           # ~8 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))

import torch  # noqa: E402
import kvlib  # noqa: E402
from evict import EvictingCache  # noqa: E402

PREFILL = 1024
EVAL_STEPS = 128
BUDGET = 256

POLICIES = ["full", "recent", "sink_recent", "h2o", "h2o_sink", "random"]
PRETTY = {"full": "full cache (no budget)", "recent": "sliding window",
          "sink_recent": "sink + window (StreamingLLM)", "h2o": "H2O",
          "h2o_sink": "H2O + sink", "random": "random keep (control)"}


def make_cache(runner, policy, budget=BUDGET):
    c = EvictingCache(runner.n_layers, budget=budget, policy=policy)
    # `observe` is where eviction happens for *every* policy, not just the
    # attention-based ones, so the recorder is always attached.
    runner.attn_recorder = None if policy == "full" else c.observe
    return c


@torch.inference_mode()
def perplexity(runner, cache, ids, baseline=None, steps=EVAL_STEPS):
    nll, argmaxes = [], []
    t0 = time.perf_counter()
    logits = runner.forward(ids[:, :PREFILL], cache, start_pos=0)
    prefill_s = time.perf_counter() - t0
    for i in range(steps):
        target = int(ids[0, PREFILL + i])
        lp = torch.log_softmax(logits[0, -1].float(), dim=-1)
        nll.append(-float(lp[target]))
        argmaxes.append(int(logits[0, -1].argmax()))
        logits = runner.forward(ids[:, PREFILL + i:PREFILL + i + 1], cache,
                                start_pos=PREFILL + i)
    out = {"ppl": math.exp(statistics.mean(nll)), "prefill_s": prefill_s,
           "tokens_held": cache.n_tokens(), "bytes": cache.nbytes(),
           "evicted": cache.evicted}
    if baseline is not None:
        out["top1_agreement"] = sum(a == b for a, b in zip(argmaxes, baseline)) / len(argmaxes)
    out["_argmax"] = argmaxes
    return out


# ---------------------------------------------------------------------------
# B. the retrieval probe
# ---------------------------------------------------------------------------

FACTS = [("Zurich", " Zurich"), ("Lisbon", " Lisbon"), ("Osaka", " Osaka"),
         ("Nairobi", " Nairobi")]


def probe_prompt(tok, fact, filler_tokens):
    """A fact at position ~0, a long stretch of unrelated text, then a question
    whose answer is that fact. If eviction dropped the fact's tokens, the model
    physically cannot answer -- the information is gone from the cache."""
    head = f"Note: the project meeting will be held in {fact}. Remember this.\n\n"
    filler = ("The quarterly report describes routine maintenance of the "
              "logistics network and lists the standard delivery windows. ") * 200
    tail = f"\n\nQuestion: In which city will the project meeting be held?\nAnswer: The meeting will be held in"
    h = tok(head, return_tensors="pt").input_ids[0]
    fl = tok(filler, return_tensors="pt").input_ids[0][:filler_tokens]
    t = tok(tail, return_tensors="pt").input_ids[0]
    return torch.cat([h, fl, t]).unsqueeze(0)


@torch.inference_mode()
def probe(runner, policy, tok, budget, filler_tokens=1200, chunk=128):
    """Feed the prompt in chunks so eviction actually happens *before* the
    question is asked.

    This matters: if the whole prompt goes through in one forward pass, the
    answer is predicted from the full cache and eviction has not run yet -- the
    probe would report a perfect score for every policy and measure nothing.
    Chunking mirrors how a real engine handles a long prompt (chunked prefill,
    Phase 3) or a long-running session.
    """
    hits = 0
    for fact, answer in FACTS:
        ids = probe_prompt(tok, fact, filler_tokens)
        cache = make_cache(runner, policy, budget)
        n = ids.shape[1]
        logits = None
        for s in range(0, n, chunk):
            logits = runner.forward(ids[:, s:s + chunk], cache, start_pos=s)
        nxt = int(logits[0, -1].argmax())
        want = tok(answer, add_special_tokens=False).input_ids[0]
        hits += (nxt == want)
    return hits / len(FACTS)


def main():
    f = {"config": {"prefill": PREFILL, "eval_steps": EVAL_STEPS,
                    "budget": BUDGET}}
    runner, tok, _ = kvlib.load_runner()
    text = kvlib.wikitext_lines(120_000)
    ids = tok(text, return_tensors="pt").input_ids[:, :PREFILL + EVAL_STEPS + 1]
    assert ids.shape[1] >= PREFILL + EVAL_STEPS + 1
    print(f"context {PREFILL} + {EVAL_STEPS} scored tokens, budget {BUDGET} "
          f"({100*BUDGET/(PREFILL+EVAL_STEPS):.0f}% of the cache kept)")

    # ------------------------------------------------------------------ A
    print("A. perplexity under each policy")
    rows, base = [], None
    for p in POLICIES:
        cache = make_cache(runner, p)
        r = perplexity(runner, cache, ids, base)
        if base is None:
            base = r["_argmax"]
            r["top1_agreement"] = 1.0
        r.pop("_argmax")
        r["policy"] = p
        rows.append(r)
        print(f"   {PRETTY[p]:>30}: ppl {r['ppl']:8.3f}  held {r['tokens_held']:5d} "
              f"tokens  cache {r['bytes']/1e6:6.2f} MB  agree "
              f"{r['top1_agreement']*100:5.1f}%", flush=True)
    f["A_perplexity"] = rows

    # ------------------------------------------------------------------ B
    print("B. retrieval probe (fact planted at the start)")
    prb = []
    for p in POLICIES:
        acc = probe(runner, p, tok, BUDGET)
        prb.append({"policy": p, "accuracy": acc})
        print(f"   {PRETTY[p]:>30}: {acc*100:5.1f}% recalled", flush=True)
    f["B_probe"] = prb

    # ------------------------------------------------------------------ C
    print("C. budget sweep")
    # The sweep scores a shorter window than section A, so it needs its own
    # full-cache reference -- perplexities from different scoring windows are
    # not comparable numbers.
    cache = make_cache(runner, "full")
    ref = perplexity(runner, cache, ids, steps=64)
    f["C_full_ppl"] = ref["ppl"]
    print(f"   reference: full cache over the same 64 tokens: ppl {ref['ppl']:.3f}")
    sweep = []
    for budget in (64, 256, 512):
        for p in ("recent", "sink_recent", "h2o_sink"):
            cache = make_cache(runner, p, budget)
            r = perplexity(runner, cache, ids, steps=64)
            r.pop("_argmax")
            sweep.append({"policy": p, "budget": budget, "ppl": r["ppl"],
                          "bytes": r["bytes"]})
            print(f"   budget {budget:>4} {PRETTY[p]:>30}: ppl {r['ppl']:8.3f}",
                  flush=True)
    f["C_sweep"] = sweep

    runner.attn_recorder = None
    json.dump(f, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    write_csv(f)
    plot(f)
    print("wrote outputs/")


def write_csv(f):
    import csv
    with open(os.path.join(OUT, "policies.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["policy", "ppl", "tokens_held", "bytes", "top1_agreement",
                    "probe_accuracy"])
        probe_by = {r["policy"]: r["accuracy"] for r in f["B_probe"]}
        for r in f["A_perplexity"]:
            w.writerow([r["policy"], round(r["ppl"], 4), r["tokens_held"],
                        r["bytes"], round(r["top1_agreement"], 4),
                        probe_by[r["policy"]]])
    with open(os.path.join(OUT, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, lineterminator="\n", fieldnames=list(f["C_sweep"][0].keys()))
        w.writeheader()
        for r in f["C_sweep"]:
            w.writerow(r)


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    rows = f["A_perplexity"]
    labels = [PRETTY[r["policy"]] for r in rows]
    base = rows[0]["ppl"]

    cols = ["tab:gray"] + ["tab:blue"] * (len(rows) - 1)
    ax[0].barh(labels, [r["ppl"] for r in rows], color=cols)
    ax[0].axvline(base, color="k", ls="--", lw=1)
    ax[0].set_xscale("log")
    for i, r in enumerate(rows):
        ax[0].text(r["ppl"] * 1.1, i, f"{r['ppl']:.1f}", va="center", fontsize=8)
    ax[0].set_xlabel("perplexity, log scale (lower is better)")
    ax[0].set_title(f"A. quality at budget {f['config']['budget']}")
    ax[0].tick_params(labelsize=8)
    ax[0].grid(alpha=.3, axis="x")

    prb = f["B_probe"]
    ax[1].barh([PRETTY[r["policy"]] for r in prb],
               [r["accuracy"] * 100 for r in prb],
               color=["tab:gray"] + ["tab:green"] * (len(prb) - 1))
    ax[1].set_xlabel("% of planted facts recalled")
    ax[1].set_xlim(0, 105)
    ax[1].set_title("B. can it still see the start of the prompt?")
    ax[1].tick_params(labelsize=8)
    ax[1].grid(alpha=.3, axis="x")

    for p in ("recent", "sink_recent", "h2o_sink"):
        rs = [r for r in f["C_sweep"] if r["policy"] == p]
        ax[2].plot([r["budget"] for r in rs], [r["ppl"] for r in rs], "o-",
                   label=PRETTY[p])
    ax[2].axhline(f.get("C_full_ppl", base), color="k", ls="--", lw=1,
                  label="full cache")
    ax[2].set_xscale("log", base=2)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("cache budget (tokens)")
    ax[2].set_ylabel("perplexity (log)")
    ax[2].set_title("C. how much budget do you actually need?")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eviction.png"), dpi=120)


if __name__ == "__main__":
    if "--plot" in sys.argv:
        plot(json.load(open(os.path.join(OUT, "findings.json"))))
    else:
        main()
