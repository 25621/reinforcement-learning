"""Project 35 -- build the gate, then audit the gate.

Writing a deploy gate takes twenty lines. Knowing whether it works takes the
rest of this project. The question that matters is not "does the gate run in
CI?" but "how often does it block a healthy model, and how often does it wave a
damaged one through?" -- and you cannot answer that without keeping the
*per-item* eval record and resampling it.

  A. Score five candidates on four evals, keeping every per-item result.
  B. The noise floor. Bootstrap each metric on the baseline: how much does it
     move when nothing has changed?
  C. Separation: how far each metric moves per unit of its own noise, and how
     much compute that costs.
  D. Ranking: do all four evals order the five candidates the same way?
  E. The audit. False-block rate on a harmless candidate, false-pass rate on a
     damaged one, over 400 resampled eval sets -- with and without the paired
     shadow check.
  F. The gate this project recommends, and what it costs to run.

    python3 run.py           # ~8 minutes on 6 CPU threads
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "30-quantize-a-7b-model-end-to-end"))

import torch  # noqa: E402
import torch.nn.functional as TF  # noqa: E402

import quantlib as Q  # noqa: E402
from gate import (CALIBRATED_CHECKS, NAIVE_CHECKS, EvalResult, Gate,  # noqa: E402
                  bootstrap_spread, separation, split_half_verdicts)

WIN = 512
EVAL_N = 10
MMLU_N = 140
N_GEN = 8
GEN_TOK = 24
F = {}


# ---------------------------------------------------------------------------
# A. score everything, keep every item
# ---------------------------------------------------------------------------


@torch.inference_mode()
def evaluate(name, model, tok, ev, items, prompts, ref_tokens=None,
             ref_gens=None):
    res = EvalResult(name=name)

    t0 = time.time()
    for ch in ev:
        logits = model(ch.unsqueeze(0)).logits[0].float()
        lp, tgt = logits[:-1], ch[1:]
        res.window_nll.append(TF.cross_entropy(lp, tgt).item())
        res.window_ntok.append(int(tgt.numel()))
        pred = lp.argmax(-1)
        res.token_match.append(pred)
    preds = torch.cat(res.token_match)
    res.token_match = ([1] * preds.numel() if ref_tokens is None
                       else (preds == ref_tokens).int().tolist())
    res.seconds["perplexity+shadow"] = time.time() - t0

    t0 = time.time()
    m = Q.mmlu_eval(model, tok, items, want_choices=True)
    res.mmlu_choice = m["choices"]
    res.mmlu_correct = [int(c == it["answer"]) for c, it in zip(m["choices"], items)]
    res.seconds["mmlu"] = time.time() - t0

    t0 = time.time()
    gens = Q.greedy_generate(model, tok, prompts, max_new=GEN_TOK)
    res.gen_len = [len(g) for g in gens]
    res.gen_divergence = ([len(g) for g in gens] if ref_gens is None
                          else [Q.first_divergence(a, b)
                                for a, b in zip(ref_gens, gens)])
    res.seconds["generation"] = time.time() - t0
    return res, preds, gens


def section_a(tok, model, ev, items, prompts, stats):
    print("\n=== A. score five candidates, keep every item ===")
    t0 = time.time()
    base, ref_tokens, ref_gens = evaluate("fp32 baseline", model, tok, ev,
                                          items, prompts)
    print(f"  fp32 baseline: ppl {base.ppl():.3f}  MMLU {base.mmlu()*100:.1f}%  "
          f"({time.time()-t0:.0f}s)")

    awq, _ = Q.awq_scales(model, stats, 4, 128)
    cands_spec = [
        ("INT8 per-channel", lambda: Q.Quantized(model, bits=8, group=0)),
        ("AWQ INT4 g128", lambda: Q.Quantized(model, bits=4, group=128,
                                              awq_scales=awq)),
        ("RTN INT4 g128", lambda: Q.Quantized(model, bits=4, group=128)),
        ("RTN INT4 per-channel", lambda: Q.Quantized(model, bits=4, group=0)),
        ("RTN INT3 g128", lambda: Q.Quantized(model, bits=3, group=128)),
    ]
    cands = []
    for name, mk in cands_spec:
        with mk():
            r, _, _ = evaluate(name, model, tok, ev, items, prompts,
                               ref_tokens, ref_gens)
        cands.append(r)
        print(f"  {name:22s} ppl {r.ppl():8.3f}  MMLU {r.mmlu()*100:5.1f}%  "
              f"agree {r.agreement()*100:6.2f}%  identical gens "
              f"{r.gen_identical()*100:.0f}%")

    F["A"] = {
        "baseline": {"ppl": base.ppl(), "mmlu": base.mmlu(),
                     "seconds": base.seconds},
        "rows": [{"name": c.name, "ppl": c.ppl(), "ppl_ratio": c.ppl() / base.ppl(),
                  "mmlu": c.mmlu(), "mmlu_drop_pts": 100 * (base.mmlu() - c.mmlu()),
                  "agree": c.agreement(), "gen_identical": c.gen_identical(),
                  "mean_first_divergence": sum(c.gen_divergence) / len(c.gen_divergence)}
                 for c in cands],
        "mmlu_n": MMLU_N, "windows": EVAL_N, "gens": N_GEN}
    return base, cands


# ---------------------------------------------------------------------------
# B. the noise floor
# ---------------------------------------------------------------------------


def section_b(base, cands):
    print("\n=== B. what each metric cannot see ===")
    mm = bootstrap_spread(base.mmlu_correct)
    nl = bootstrap_spread(base.window_nll)
    rows = [
        {"metric": f"MMLU accuracy (n={MMLU_N})", "unit": "points",
         "sd": mm["sd"] * 100, "ci95": (mm["p97.5"] - mm["p2.5"]) * 100},
        {"metric": f"perplexity ({EVAL_N} windows)", "unit": "x",
         "sd": math.exp(nl["sd"]) - 1,
         "ci95": math.exp(nl["p97.5"]) / math.exp(nl["p2.5"]) - 1},
        {"metric": "shadow agreement", "unit": "fraction", "sd": 0.0, "ci95": 0.0},
    ]
    for r in rows:
        print(f"  {r['metric']:32s} 1 s.d. = {r['sd']:.4f} {r['unit']}, "
              f"95% band {r['ci95']:.4f}")
    thr = NAIVE_CHECKS["mmlu"]["max"]
    F["B"] = {"rows": rows, "mmlu_threshold_pts": thr,
              "mmlu_threshold_in_sd": thr / max(mm["sd"] * 100, 1e-9),
              "mmlu_n_for_1pt_resolution": int(
                  math.ceil(base.mmlu() * (1 - base.mmlu()) / (1.0 / 100 / 2) ** 2))}
    print(f"  the MMLU gate threshold of {thr} points is "
          f"{F['B']['mmlu_threshold_in_sd']:.2f} standard deviations of its own noise")
    print(f"  resolving a 1-point drop at 2 s.d. would need "
          f"{F['B']['mmlu_n_for_1pt_resolution']:,} questions")


# ---------------------------------------------------------------------------
# C. separation per second
# ---------------------------------------------------------------------------


def section_c(base, cands):
    print("\n=== C. signal per unit of noise, and per second ===")
    cost = base.seconds
    rows = []
    for metric, sec_key in (("mmlu", "mmlu"), ("perplexity", "perplexity+shadow"),
                            ("shadow_agreement", "perplexity+shadow")):
        s = separation(base, cands, metric)
        # The mildest candidate is the hard case; report that one.
        mild = min(range(len(cands)), key=lambda i: abs(s["shifts"][i]))
        rows.append({"metric": metric, "noise_sd": s["noise_sd"],
                     "seconds": cost[sec_key],
                     "separation_per_cand": s["separation"],
                     "mildest_candidate": cands[mild].name,
                     "mildest_separation": s["separation"][mild]})
        print(f"  {metric:18s} noise sd {s['noise_sd']:.5f}  cost "
              f"{cost[sec_key]:5.1f}s  separation on the mildest candidate "
              f"({cands[mild].name}): {s['separation'][mild]:.1f}")
    F["C"] = {"rows": rows, "candidates": [c.name for c in cands]}


# ---------------------------------------------------------------------------
# D. does every eval agree on the ranking?
# ---------------------------------------------------------------------------


def section_d(base, cands):
    print("\n=== D. do the evals agree on the order? ===")
    metrics = {
        "perplexity": [c.ppl() for c in cands],
        "mmlu (lower acc = worse)": [-c.mmlu() for c in cands],
        "shadow disagreement": [1 - c.agreement() for c in cands],
        "generation divergence": [-sum(c.gen_divergence) / len(c.gen_divergence)
                                  for c in cands],
    }
    names = [c.name for c in cands]
    ref = sorted(range(len(cands)), key=lambda i: metrics["perplexity"][i])
    rows = []
    for m, vals in metrics.items():
        order = sorted(range(len(cands)), key=lambda i: vals[i])
        inv = sum(1 for a in range(len(order)) for b in range(a + 1, len(order))
                  if ref.index(order[a]) > ref.index(order[b]))
        rows.append({"metric": m, "order": [names[i] for i in order],
                     "inversions_vs_perplexity": inv})
        print(f"  {m:26s} {' < '.join(names[i] for i in order)}"
              f"   ({inv} inversions)")
    F["D"] = {"rows": rows, "reference": [names[i] for i in ref]}


# ---------------------------------------------------------------------------
# E. how often is the gate wrong?
# ---------------------------------------------------------------------------


def section_e(base, cands):
    print("\n=== E. auditing the gate over 400 resampled eval sets ===")
    by = {c.name: c for c in cands}
    healthy = by["INT8 per-channel"]        # true damage ~ nil
    damaged = by["RTN INT4 g128"]           # real, moderate damage

    variants = {
        "naive full gate": NAIVE_CHECKS,
        "naive, scores only": {k: v for k, v in NAIVE_CHECKS.items()
                               if k in ("perplexity", "mmlu")},
        "naive, MMLU only": {"mmlu": NAIVE_CHECKS["mmlu"]},
        "calibrated full gate": CALIBRATED_CHECKS,
        "calibrated, scores only": {k: v for k, v in CALIBRATED_CHECKS.items()
                                    if k in ("perplexity", "mmlu")},
        "calibrated, MMLU only": {"mmlu": CALIBRATED_CHECKS["mmlu"]},
        "calibrated, shadow only": {
            "shadow_agreement": CALIBRATED_CHECKS["shadow_agreement"]},
    }
    rows = []
    for label, checks in variants.items():
        g = Gate(checks)
        fb = split_half_verdicts(g, base, healthy)
        fp = split_half_verdicts(g, base, damaged)
        rows.append({"gate": label, "false_block_rate": fb["block_rate"],
                     "false_pass_rate": 1 - fp["block_rate"],
                     "false_block_by_check": fb["per_check_block_rate"],
                     "catch_by_check": fp["per_check_block_rate"]})
        print(f"  {label:26s} false-block {fb['block_rate']*100:5.1f}%   "
              f"false-pass {100*(1-fp['block_rate']):5.1f}%")
    F["E"] = {"rows": rows, "healthy": healthy.name, "damaged": damaged.name,
              "n_trials": 400}


# ---------------------------------------------------------------------------
# F. the recommendation
# ---------------------------------------------------------------------------


def section_f(base, cands):
    print("\n=== F. the two gates side by side ===")
    out = {}
    for label, checks in (("naive", NAIVE_CHECKS),
                          ("calibrated", CALIBRATED_CHECKS)):
        g = Gate(checks)
        rows = []
        for c in cands:
            ok, detail, fails = g.verdict(base, c)
            rows.append({"candidate": c.name,
                         "verdict": "PASS" if ok else "BLOCK", "fails": fails,
                         "detail": {k: {"value": v["value"], "pass": v["pass"]}
                                    for k, v in detail.items()}})
            print(f"  [{label:10s}] {c.name:22s} {'PASS' if ok else 'BLOCK':5s} "
                  + (f"[{', '.join(fails)}]" if fails else ""))
        out[label] = {"checks": checks, "rows": rows}
    total_s = sum(base.seconds.values())
    F["F"] = {"gates": out, "seconds_per_model": total_s,
              "seconds_breakdown": base.seconds}
    print(f"  cost: {total_s:.0f}s per model on this box "
          f"({', '.join(f'{k} {v:.0f}s' for k, v in base.seconds.items())})")


# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.4))

    rows = f["A"]["rows"]
    names = [r["name"] for r in rows]
    x = range(len(rows))
    ax[0].plot(x, [r["ppl_ratio"] for r in rows], "o-", color="#c0392b",
               label="perplexity / baseline")
    ax0b = ax[0].twinx()
    ax0b.plot(x, [r["agree"] * 100 for r in rows], "s-", color="#27ae60",
              label="shadow agreement (%)")
    ax[0].set_xticks(list(x))
    ax[0].set_xticklabels(names, rotation=30, ha="right", fontsize=6.5)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("perplexity ratio")
    ax0b.set_ylabel("agreement (%)")
    ax[0].set_title("A. five candidates, two views")
    ax[0].grid(alpha=0.3)

    b = f["B"]["rows"]
    ax[1].bar(range(len(b)), [r["ci95"] for r in b], color="#2471a3")
    ax[1].set_xticks(range(len(b)))
    ax[1].set_xticklabels([r["metric"].split(" (")[0] for r in b],
                          rotation=25, ha="right", fontsize=7)
    ax[1].set_ylabel("95% band when nothing changed")
    ax[1].set_title("B. the noise floor")
    ax[1].grid(alpha=0.3, axis="y")

    c = f["C"]["rows"]
    cn = f["C"]["candidates"]
    for r in c:
        ax[2].plot(range(len(cn)), [min(s, 1e4) for s in r["separation_per_cand"]],
                   "o-", label=r["metric"])
    ax[2].axhline(2.0, color="k", ls="--", lw=1, label="2 s.d. (barely resolvable)")
    ax[2].set_yscale("log")
    ax[2].set_xticks(range(len(cn)))
    ax[2].set_xticklabels(cn, rotation=30, ha="right", fontsize=6.5)
    ax[2].set_ylabel("shift / own noise")
    ax[2].set_title("C. separation")
    ax[2].legend(fontsize=6)
    ax[2].grid(alpha=0.3)

    e = f["E"]["rows"]
    w = 0.38
    ax[3].bar([i - w / 2 for i in range(len(e))],
              [r["false_block_rate"] * 100 for r in e], w,
              label=f"false BLOCK on {f['E']['healthy']}", color="#e67e22")
    ax[3].bar([i + w / 2 for i in range(len(e))],
              [r["false_pass_rate"] * 100 for r in e], w,
              label=f"false PASS on {f['E']['damaged']}", color="#c0392b")
    ax[3].set_xticks(range(len(e)))
    ax[3].set_xticklabels([r["gate"].split(" gate")[0] for r in e],
                          rotation=25, ha="right", fontsize=6.5)
    ax[3].set_ylabel("% of 400 resampled eval sets")
    ax[3].set_title("E. how often the gate is wrong")
    ax[3].legend(fontsize=6.5)
    ax[3].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "eval_gate.png"), dpi=110)
    print("wrote outputs/eval_gate.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    tok, model = Q.load()
    text = Q._wikitext(500_000)
    ev = Q.token_chunks(tok, text, WIN, EVAL_N)
    calib = Q.token_chunks(tok, text, WIN, 8, skip=EVAL_N + 4)
    items = Q.mmlu_items(MMLU_N, seed=11)
    prompts = Q.chat_prompts(tok, N_GEN, seed=5)
    stats = Q.act_stats(model, calib, sample_rows=96)
    F["setup"] = {"model": Q.MODEL_ID, "threads": Q.N_THREADS, "window": WIN,
                  "eval_windows": EVAL_N, "mmlu_n": MMLU_N, "gens": N_GEN,
                  "gen_tokens": GEN_TOK}

    base, cands = section_a(tok, model, ev, items, prompts, stats)
    section_b(base, cands)
    section_c(base, cands)
    section_d(base, cands)
    section_e(base, cands)
    section_f(base, cands)

    F["wall_s"] = round(time.time() - t0, 1)
    Q.save_findings(os.path.join(OUT, "findings.json"), F)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
