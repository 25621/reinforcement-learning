"""Project 23 -- greedy speculative decoding.

Pair Qwen2.5-0.5B (draft) with Qwen2.5-1.5B (target), implement the verify
loop, and answer four questions in order:

  A. Does it change the output?  (It must not. Greedy speculation is supposed
     to be bit-for-bit the same text as greedy decoding.)
  B. How often is a draft token accepted, and does that depend on how far
     ahead it was proposed?
  C. What does it do to the wall clock, and where does the time go?
  D. Can a three-number cost model predict the speedup -- and what does that
     model say about a real 1B-draft / 70B-target deployment?
  E. Control: what happens with a drafter that is deliberately useless?

    python3 run.py           # ~4 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

import torch  # noqa: E402

import speclib as S  # noqa: E402

K = 4
MAX_NEW = 48
F = {}


class RandomDrafter:
    """The control drafter: plausible-looking token ids, chosen by dice.

    It costs almost nothing to run, so it isolates one question -- how much of
    speculation's win comes from the *mechanism* (one target pass can emit
    several tokens) and how much from the draft model actually being right.
    """

    name = "random"

    def __init__(self, seed=0):
        import random
        self.rng = random.Random(seed)
        self.draft_s = 0.0
        self.draft_passes = 0

    def reset(self):
        self.draft_s = 0.0
        self.draft_passes = 0

    def propose(self, tokens, k):
        t0 = time.perf_counter()
        out = [self.rng.randrange(1000, 120000) for _ in range(k)]
        self.draft_s += time.perf_counter() - t0
        self.draft_passes += k
        return out

    def rollback(self, n):
        pass


# ---------------------------------------------------------------------------
# A. speculation must not change the text
# ---------------------------------------------------------------------------


def section_a(target, draft, tok):
    print("\n== A. output equivalence ==")
    t_cache = S.make_cache(target)
    drafter = S.ModelDrafter(draft)
    rows = []
    for name, prompt in S.WORKLOADS.items():
        ids = S.chat_ids(tok, prompt)
        base = S.greedy_decode(target, t_cache, ids, max_new=MAX_NEW)
        spec = S.speculative_greedy(target, drafter, t_cache, ids,
                                    k=K, max_new=MAX_NEW)
        same = base["tokens"] == spec["tokens"]
        rows.append({
            "workload": name,
            "prompt_tokens": int(ids.shape[1]),
            "identical": same,
            "text": tok.decode(base["tokens"]),
            "acceptance_rate": round(spec["acceptance_rate"], 4),
            "tokens_per_iter": round(spec["tokens_per_iter"], 3),
            "iters": spec["iters"],
            "base_decode_s": round(base["decode_s"], 3),
            "spec_decode_s": round(spec["decode_s"], 3),
            "spec_draft_s": round(spec["draft_s"], 3),
            "spec_verify_s": round(spec["verify_s"], 3),
            "per_pos_hits": spec["per_pos_hits"],
            "accept_run": spec["accept_run"],
        })
        print(f"  {name:10s} identical={same}  accept={spec['acceptance_rate']:.3f}"
              f"  tok/iter={spec['tokens_per_iter']:.2f}")
    F["A"] = {"k": K, "max_new": MAX_NEW, "rows": rows,
              "all_identical": all(r["identical"] for r in rows)}


# ---------------------------------------------------------------------------
# B. anatomy of acceptance
# ---------------------------------------------------------------------------


def section_b():
    print("\n== B. acceptance anatomy ==")
    rows = F["A"]["rows"]
    iters = sum(r["iters"] for r in rows)
    hits = [0] * K
    for r in rows:
        for i, h in enumerate(r["per_pos_hits"]):
            hits[i] += h
    # P(position i accepted | reached) -- a draft token at position i is only
    # ever looked at if every earlier one was accepted.
    cond = []
    for i in range(K):
        reached = iters if i == 0 else hits[i - 1]
        cond.append(hits[i] / reached if reached else 0.0)
    runs = [n for r in rows for n in r["accept_run"]]
    hist = [runs.count(i) / len(runs) for i in range(K + 1)]
    F["B"] = {
        "iters": iters,
        "marginal_accept": [round(h / iters, 4) for h in hits],
        "conditional_accept": [round(c, 4) for c in cond],
        "run_hist": [round(h, 4) for h in hist],
        "mean_accepted": round(statistics.mean(runs), 3),
        "mean_tokens_per_iter": round(statistics.mean(runs) + 1, 3),
    }
    print("  P(accept | reached) by position:",
          [f"{c:.2f}" for c in F["B"]["conditional_accept"]])
    print("  accepted-per-iteration histogram:",
          [f"{h:.2f}" for h in F["B"]["run_hist"]])


# ---------------------------------------------------------------------------
# C. wall clock, timed the only way this box allows
# ---------------------------------------------------------------------------


def section_c(target, draft, tok):
    print("\n== C. wall clock ==")
    t_cache = S.make_cache(target)
    drafter = S.ModelDrafter(draft)
    rows = []
    for name in ("chat", "summarize"):
        ids = S.chat_ids(tok, S.WORKLOADS[name])
        best = {"baseline": float("inf"), "spec": float("inf")}
        detail = {}
        # Interleaved: baseline, spec, baseline, spec ... and keep the minimum
        # of each. This box is shared; running one to completion and then the
        # other charges any background spike entirely to whoever ran then.
        for _ in range(2):
            r = S.greedy_decode(target, t_cache, ids, max_new=MAX_NEW)
            best["baseline"] = min(best["baseline"], r["decode_s"])
            r = S.speculative_greedy(target, drafter, t_cache, ids,
                                     k=K, max_new=MAX_NEW)
            if r["decode_s"] < best["spec"]:
                best["spec"] = r["decode_s"]
                detail = {"draft_s": r["draft_s"], "verify_s": r["verify_s"],
                          "iters": r["iters"],
                          "tokens_per_iter": r["tokens_per_iter"]}
        rows.append({
            "workload": name,
            "baseline_decode_s": round(best["baseline"], 3),
            "spec_decode_s": round(best["spec"], 3),
            "speedup": round(best["baseline"] / best["spec"], 3),
            "draft_s": round(detail["draft_s"], 3),
            "verify_s": round(detail["verify_s"], 3),
            "draft_share": round(detail["draft_s"] / best["spec"], 4),
            "iters": detail["iters"],
            "tokens_per_iter": round(detail["tokens_per_iter"], 3),
            "baseline_tok_s": round(MAX_NEW / best["baseline"], 3),
            "spec_tok_s": round(MAX_NEW / best["spec"], 3),
        })
        print(f"  {name:10s} {rows[-1]['baseline_decode_s']:.2f}s -> "
              f"{rows[-1]['spec_decode_s']:.2f}s  = {rows[-1]['speedup']:.2f}x"
              f"  (draft is {rows[-1]['draft_share']:.0%} of it)")
    F["C"] = {"rows": rows}


# ---------------------------------------------------------------------------
# D. the cost model, and what it says about real hardware
# ---------------------------------------------------------------------------


def section_d(target, draft, tok):
    print("\n== D. cost model ==")
    ids = S.chat_ids(tok, S.WORKLOADS["chat"])
    n = int(ids.shape[1])

    tc, dc = S.make_cache(target), S.make_cache(draft)
    for runner, cache in ((target, tc), (draft, dc)):
        cache.reset()
        runner.forward(ids, cache, start_pos=0)

    def pass_fn(runner, cache, width):
        """A callable that runs exactly one forward pass and nothing else.

        `interleaved` times the *whole* callable, so anything else inside --
        a prompt prefill, a cache rebuild -- lands in the number. Rewinding
        with `truncate` costs nothing, which is what makes this possible.
        """
        blk = torch.full((1, width), 9707, dtype=torch.long)

        def f():
            cache.truncate(n)
            runner.forward(blk, cache, start_pos=n)
        return f

    t = S.interleaved({
        "target_1": pass_fn(target, tc, 1),
        "target_k1": pass_fn(target, tc, K + 1),
        "draft_1": pass_fn(draft, dc, 1),
    }, rounds=3, warmup=1)
    cost_ratio = t["draft_1"] / t["target_1"]
    verify_overhead = t["target_k1"] / t["target_1"] - 1.0

    # Validate per workload: each one has its own acceptance, so each one has
    # its own prediction. Averaging the alphas first would hide the fit.
    checks = []
    for r in F["C"]["rows"]:
        p = S.speedup_model(r["tokens_per_iter"], K, cost_ratio, verify_overhead)
        checks.append({"workload": r["workload"],
                       "alpha_len": r["tokens_per_iter"],
                       "predicted": round(p, 3),
                       "measured": r["speedup"],
                       "error": round(abs(p - r["speedup"]) / r["speedup"], 4)})
        print(f"  {r['workload']:10s} alpha={r['tokens_per_iter']:.2f} "
              f"predicted {p:.2f}x vs measured {r['speedup']:.2f}x")

    alpha_len = F["B"]["mean_tokens_per_iter"]
    pred = S.speedup_model(alpha_len, K, cost_ratio, verify_overhead)
    meas = statistics.mean(r["speedup"] for r in F["C"]["rows"])

    # What the same acceptance would buy on production-shaped hardware, where
    # the draft is ~1-2% of the target instead of ~34%.
    scenarios = []
    for label, cr, vo in [
        ("this box: 0.5B draft / 1.5B target", cost_ratio, verify_overhead),
        ("1B draft / 7B target", 1.0 / 7, 0.05),
        ("1B draft / 70B target", 1.0 / 70, 0.05),
        ("zero-cost drafter (n-gram, Medusa)", 0.0, 0.05),
    ]:
        scenarios.append({"label": label, "cost_ratio": round(cr, 4),
                          "verify_overhead": round(vo, 4),
                          "speedup": round(S.speedup_model(alpha_len, K, cr, vo), 3)})

    F["D"] = {
        "target_1tok_ms": round(t["target_1"] * 1000, 2),
        "target_k1tok_ms": round(t["target_k1"] * 1000, 2),
        "draft_1tok_ms": round(t["draft_1"] * 1000, 2),
        "cost_ratio": round(cost_ratio, 4),
        "verify_overhead": round(verify_overhead, 4),
        "alpha_len": alpha_len,
        "predicted_speedup": round(pred, 3),
        "measured_speedup": round(meas, 3),
        "model_error": round(abs(pred - meas) / meas, 4),
        "per_workload": checks,
        "scenarios": scenarios,
    }
    print(f"  target 1 tok {t['target_1']*1000:.1f} ms, k+1 tok "
          f"{t['target_k1']*1000:.1f} ms, draft {t['draft_1']*1000:.1f} ms")
    print(f"  cost_ratio {cost_ratio:.3f}  verify_overhead {verify_overhead:.3f}")
    print(f"  predicted {pred:.2f}x vs measured {meas:.2f}x")
    for s in scenarios:
        print(f"    {s['label']:38s} -> {s['speedup']:.2f}x")


# ---------------------------------------------------------------------------
# E. the control: a drafter that knows nothing
# ---------------------------------------------------------------------------


def section_e(target, tok):
    print("\n== E. control: random drafter ==")
    ids = S.chat_ids(tok, S.WORKLOADS["chat"])
    t_cache = S.make_cache(target)
    base = S.greedy_decode(target, t_cache, ids, max_new=MAX_NEW)
    rnd = S.speculative_greedy(target, RandomDrafter(), t_cache, ids,
                               k=K, max_new=MAX_NEW)
    F["E"] = {
        "identical": base["tokens"] == rnd["tokens"],
        "acceptance_rate": round(rnd["acceptance_rate"], 4),
        "tokens_per_iter": round(rnd["tokens_per_iter"], 3),
        "baseline_decode_s": round(base["decode_s"], 3),
        "random_decode_s": round(rnd["decode_s"], 3),
        "speedup": round(base["decode_s"] / rnd["decode_s"], 3),
        "verify_overhead_paid": round(
            rnd["verify_s"] / base["decode_s"], 3),
    }
    print(f"  identical text: {F['E']['identical']}  "
          f"acceptance {F['E']['acceptance_rate']:.3f}  "
          f"speedup {F['E']['speedup']:.2f}x")


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    b = f["B"]
    x = list(range(1, K + 1))
    ax[0].bar([i - 0.2 for i in x], b["conditional_accept"], 0.4,
              color="#2471a3", label="P(accept | reached)")
    ax[0].bar([i + 0.2 for i in x], b["marginal_accept"], 0.4,
              color="#7f8c8d", label="P(accept) overall")
    ax[0].set_xticks(x)
    ax[0].set_xlabel("draft position (1 = next token)")
    ax[0].set_ylabel("probability")
    ax[0].set_title("B. acceptance decays with distance")
    ax[0].legend(fontsize=8)
    ax[0].set_ylim(0, 1.05)

    ax[1].bar(range(K + 1), b["run_hist"], color="#e67e22")
    ax[1].set_xticks(range(K + 1))
    ax[1].set_xlabel("draft tokens accepted in one iteration")
    ax[1].set_ylabel("fraction of iterations")
    ax[1].set_title(f"B. mean {b['mean_accepted']:.2f} accepted "
                    f"+ 1 bonus")

    rows = f["C"]["rows"]
    names = [r["workload"] for r in rows]
    xs = range(len(rows))
    ax[2].bar([i - 0.2 for i in xs], [r["baseline_decode_s"] for r in rows],
              0.4, color="#7f8c8d", label="baseline")
    ax[2].bar([i + 0.2 for i in xs], [r["verify_s"] for r in rows], 0.4,
              color="#2471a3", label="spec: target verify")
    ax[2].bar([i + 0.2 for i in xs], [r["draft_s"] for r in rows], 0.4,
              bottom=[r["verify_s"] for r in rows], color="#c0392b",
              label="spec: draft")
    ax[2].set_xticks(list(xs))
    ax[2].set_xticklabels(names)
    ax[2].set_ylabel(f"seconds for {f['A']['max_new']} tokens")
    ax[2].set_title("C. where the time goes")
    ax[2].legend(fontsize=8)
    for i, r in enumerate(rows):
        ax[2].text(i + 0.2, r["verify_s"] + r["draft_s"],
                   f"{r['speedup']:.2f}x", ha="center", va="bottom")

    sc = f["D"]["scenarios"]
    ax[3].barh(range(len(sc)), [s["speedup"] for s in sc],
               color=["#c0392b", "#e67e22", "#2471a3", "#27ae60"])
    ax[3].set_yticks(range(len(sc)))
    ax[3].set_yticklabels([s["label"].replace(": ", ":\n") for s in sc],
                          fontsize=7)
    ax[3].axvline(1.0, color="k", lw=1, ls="--")
    ax[3].set_xlabel("predicted speedup")
    ax[3].set_xlim(0, max(s["speedup"] for s in sc) * 1.25)
    ax[3].set_title("D. same acceptance, different draft cost")
    for i, s in enumerate(sc):
        ax[3].text(s["speedup"], i, f" {s['speedup']:.2f}x", va="center",
                   fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "speculative_decoding.png"), dpi=110)
    print("wrote outputs/speculative_decoding.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    target, draft, tok, _ = S.load_pair()
    F["setup"] = {
        "target": S.TARGET_ID, "draft": S.DRAFT_ID,
        "target_layers": target.n_layers, "draft_layers": draft.n_layers,
        "vocab": int(target.lm_head.shape[0]),
        "threads": S.N_THREADS, "k": K, "max_new": MAX_NEW,
    }
    section_a(target, draft, tok)
    section_b()
    section_c(target, draft, tok)
    section_d(target, draft, tok)
    section_e(target, tok)
    F["wall_s"] = round(time.time() - t0, 1)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
