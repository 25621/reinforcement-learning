"""Project 26 -- tune k.

`k` is the number of tokens the drafter proposes per round. Bigger k means
more tokens can come out of one target forward pass, and also more drafting
work and a fatter verification pass. There is an optimum. This project finds
it three ways and checks they agree.

  A. Measure the verification cost as a function of width. Is a 11-wide pass
     really "almost free"?
  B. Sweep k with the 0.5B draft model. Speedup, alpha, and the knee.
  C. Sweep k with the free n-gram drafter. Does removing the draft cost move
     the optimum, and how far?
  D. The closed-form model: predict speedup(k) from three measured numbers,
     and compare its argmax to the measured one.
  E. Tail latency. Bigger k improves the mean and hurts the worst case --
     put a number on both.

    python3 run.py           # ~6 minutes on 6 CPU threads
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

import torch  # noqa: E402

import speclib as S  # noqa: E402

KS = [1, 2, 3, 4, 5, 7, 10]
MAX_NEW = 48
F = {}

COPY_PROMPT = (
    "Repeat the following paragraph exactly, changing only the word "
    "\"Greek\" to \"Hellenic\" everywhere it appears. Output the paragraph "
    "and nothing else.\n\n"
    "The Antikythera mechanism is an ancient Greek hand-powered device that "
    "has been described as the oldest known analogue computer. It was used to "
    "predict astronomical positions and eclipses decades in advance, and to "
    "track the four-year cycle of the ancient Olympic Games. The artefact was "
    "recovered in 1901 from a shipwreck off the coast of the Greek island of "
    "Antikythera, and its complexity was not matched by any known device for "
    "well over a thousand years."
)


# ---------------------------------------------------------------------------
# A. what a wide verification pass really costs
# ---------------------------------------------------------------------------


def section_a(target, draft, tok):
    print("\n== A. verification cost vs width ==")
    ids = S.chat_ids(tok, S.WORKLOADS["chat"])
    n = int(ids.shape[1])
    tc, dc = S.make_cache(target), S.make_cache(draft)
    for r, c in ((target, tc), (draft, dc)):
        c.reset()
        r.forward(ids, c, start_pos=0)

    def pass_fn(runner, cache, width):
        blk = torch.full((1, width), 9707, dtype=torch.long)

        def f():
            cache.truncate(n)
            runner.forward(blk, cache, start_pos=n)
        return f

    fns = {f"target_{w}": pass_fn(target, tc, w) for w in
           [1] + [k + 1 for k in KS]}
    fns["draft_1"] = pass_fn(draft, dc, 1)
    t = S.interleaved(fns, rounds=3, warmup=1)
    base = t["target_1"]
    rows = [{"width": w, "ms": round(t[f"target_{w}"] * 1000, 2),
             "vs_1_token": round(t[f"target_{w}"] / base, 4)}
            for w in sorted({1} | {k + 1 for k in KS})]
    F["A"] = {
        "target_1tok_ms": round(base * 1000, 2),
        "draft_1tok_ms": round(t["draft_1"] * 1000, 2),
        "cost_ratio": round(t["draft_1"] / base, 4),
        "rows": rows,
    }
    for r in rows:
        print(f"  width {r['width']:2d}: {r['ms']:7.1f} ms  = "
              f"{r['vs_1_token']:.3f}x a 1-token pass")
    print(f"  draft pass {F['A']['draft_1tok_ms']:.1f} ms -> cost_ratio "
          f"{F['A']['cost_ratio']:.3f}")


# ---------------------------------------------------------------------------
# B + C. the two sweeps
# ---------------------------------------------------------------------------


def sweep(target, tok, prompt, make_drafter, label, baseline_tok_s=None):
    """Sweep k on one (prompt, drafter) pair.

    Speed is reported as *tokens per second*, not as time for a fixed 48
    tokens. A round emits up to k+1 tokens, so a run that stops at "48 or
    more" overshoots by up to k tokens -- at k=10 that is a fifth of the
    generation, and it would show up as a fake penalty for large k. Real
    requests generate hundreds of tokens, so steady-state throughput is the
    honest measure.
    """
    ids = S.chat_ids(tok, prompt)
    tc = S.make_cache(target, max_len=1024)
    if baseline_tok_s is None:
        b = S.greedy_decode(target, tc, ids, max_new=MAX_NEW)
        baseline_tok_s = MAX_NEW / b["decode_s"]
    rows = []
    for k in KS:
        r = S.speculative_greedy(target, make_drafter(), tc, ids, k=k,
                                 max_new=MAX_NEW)
        itl = r["itl"]
        tok_s = r["produced"] / r["decode_s"]
        rows.append({
            "k": k,
            "alpha": round(r["tokens_per_iter"], 3),
            "acceptance": round(r["conditional_acceptance"], 4),
            "iters": r["iters"],
            "produced": r["produced"],
            "decode_s": round(r["decode_s"], 3),
            "tok_s": round(tok_s, 3),
            "speedup": round(tok_s / baseline_tok_s, 3),
            "draft_share": round(r["draft_s"] / r["decode_s"], 4),
            "itl_mean": round(sum(itl) / len(itl), 4),
            "itl_p50": round(S.pct(itl, 50), 4),
            "itl_p99": round(S.pct(itl, 99), 4),
            "iter_max_s": round(max(r["iter_s"]), 4),
        })
        print(f"  [{label}] k={k:2d}  alpha {rows[-1]['alpha']:.2f}  "
              f"accept {rows[-1]['acceptance']:.2f}  "
              f"{rows[-1]['speedup']:.2f}x  itl p99 "
              f"{rows[-1]['itl_p99']*1000:.0f} ms")
    best = max(rows, key=lambda r: r["speedup"])
    return {"label": label, "baseline_tok_s": round(baseline_tok_s, 3),
            "rows": rows, "best_k": best["k"], "best_speedup": best["speedup"]}


def section_bc(target, draft, tok):
    md = S.ModelDrafter(draft, max_len=1024)
    print("\n== B. 0.5B draft model, chat (draft is imperfect) ==")
    F["B"] = sweep(target, tok, S.WORKLOADS["chat"], lambda: md, "model/chat")
    print(f"  best k = {F['B']['best_k']} at {F['B']['best_speedup']:.2f}x")

    print("\n== C1. 0.5B draft model, copy_edit (draft is perfect) ==")
    F["C1"] = sweep(target, tok, COPY_PROMPT, lambda: md, "model/copy")
    print(f"  best k = {F['C1']['best_k']} at {F['C1']['best_speedup']:.2f}x")

    print("\n== C2. free n-gram drafter, copy_edit ==")
    F["C2"] = sweep(target, tok, COPY_PROMPT,
                    lambda: S.NgramDrafter(max_n=4, min_n=2), "ngram/copy",
                    baseline_tok_s=F["C1"]["baseline_tok_s"])
    print(f"  best k = {F['C2']['best_k']} at {F['C2']['best_speedup']:.2f}x")


# ---------------------------------------------------------------------------
# D. the closed-form model
# ---------------------------------------------------------------------------


def section_d():
    print("\n== D. closed-form model ==")
    base_ms = F["A"]["target_1tok_ms"]
    width_ms = {r["width"]: r["ms"] for r in F["A"]["rows"]}
    cr = F["A"]["cost_ratio"]

    def predict(a, cost_ratio):
        out = []
        for k in KS:
            # Expected tokens per iteration if each position is accepted
            # independently with probability a: 1 + a + a^2 + ... + a^k.
            alpha = sum(a ** i for i in range(k + 1))
            per_iter = width_ms[k + 1] / base_ms + k * cost_ratio
            out.append({"k": k, "alpha_pred": round(alpha, 3),
                        "speedup_pred": round(alpha / per_iter, 3)})
        return out

    F["D"] = {"cost_ratio": cr, "curves": []}
    for key, cost in (("B", cr), ("C1", cr), ("C2", 0.0)):
        a = F[key]["rows"][3]["acceptance"]        # measured at k=4
        pred = predict(a, cost)
        best = max(pred, key=lambda r: r["speedup_pred"])
        F["D"]["curves"].append({
            "sweep": key, "label": F[key]["label"], "a": a,
            "cost_ratio": cost, "rows": pred,
            "argmax_pred": best["k"], "argmax_meas": F[key]["best_k"],
            "peak_pred": best["speedup_pred"],
            "peak_meas": F[key]["best_speedup"],
        })
        print(f"  {F[key]['label']:12s} a={a:.2f} cost={cost:.3f} -> "
              f"predicted best k={best['k']} ({best['speedup_pred']:.2f}x), "
              f"measured k={F[key]['best_k']} ({F[key]['best_speedup']:.2f}x)")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    a = f["A"]["rows"]
    ax[0].plot([r["width"] for r in a], [r["vs_1_token"] for r in a], "o-",
               color="#2471a3", label="measured target pass")
    ax[0].plot([r["width"] for r in a], [r["width"] for r in a], "--",
               color="#c0392b", label="if it scaled with width")
    ax[0].set_xlabel("tokens verified in one pass (k+1)")
    ax[0].set_ylabel("cost / a 1-token pass")
    ax[0].set_title("A. wide verification is nearly free")
    ax[0].set_ylim(0, 3)
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    cols = {"B": "#c0392b", "C1": "#2471a3", "C2": "#27ae60"}
    for key in ("B", "C1", "C2"):
        rows = f[key]["rows"]
        ax[1].plot([r["k"] for r in rows], [r["speedup"] for r in rows], "o-",
                   color=cols[key], label=f[key]["label"])
        best = max(rows, key=lambda r: r["speedup"])
        ax[1].scatter([best["k"]], [best["speedup"]], s=140,
                      facecolors="none", edgecolors=cols[key], linewidths=2)
    ax[1].axhline(1.0, color="k", lw=1, ls="--")
    ax[1].set_xlabel("k (tokens proposed per round)")
    ax[1].set_ylabel("speedup vs plain decoding")
    ax[1].set_title("B+C. the knee moves with acceptance and cost")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for c in f["D"]["curves"]:
        key = c["sweep"]
        rows = f[key]["rows"]
        ax[2].plot([r["k"] for r in rows], [r["speedup"] for r in rows], "o-",
                   color=cols[key], label=c["label"] + " meas.")
        ax[2].plot([r["k"] for r in c["rows"]],
                   [r["speedup_pred"] for r in c["rows"]], "--",
                   color=cols[key], alpha=0.6, label=c["label"] + " pred.")
    ax[2].set_xlabel("k")
    ax[2].set_ylabel("speedup")
    ax[2].set_title("D. closed-form vs measured")
    ax[2].legend(fontsize=6)
    ax[2].grid(alpha=0.3)

    for key in ("B", "C2"):
        rows = f[key]["rows"]
        ks = [r["k"] for r in rows]
        ax[3].plot(ks, [r["itl_p99"] * 1000 for r in rows], "s-",
                   color=cols[key], label=f[key]["label"] + " p99")
        ax[3].plot(ks, [r["itl_mean"] * 1000 for r in rows], "o--",
                   color=cols[key], alpha=0.6,
                   label=f[key]["label"] + " mean")
    ax[3].set_xlabel("k")
    ax[3].set_ylabel("inter-token latency (ms)")
    ax[3].set_title("E. a paid drafter grows the tail")
    ax[3].legend(fontsize=7)
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tune_k.png"), dpi=110)
    print("wrote outputs/tune_k.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    target, draft, tok, _ = S.load_pair()
    F["setup"] = {"target": S.TARGET_ID, "draft": S.DRAFT_ID, "ks": KS,
                  "max_new": MAX_NEW, "threads": S.N_THREADS}
    section_a(target, draft, tok)
    section_bc(target, draft, tok)
    section_d()
    F["wall_s"] = round(time.time() - t0, 1)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
