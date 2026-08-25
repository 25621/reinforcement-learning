"""Project 04 — Sampling kernel.

Everything is measured on REAL logits: one forward pass over a mixed prompt
gives 62 genuine next-token distributions, and those are what the filters run
on. Random logits are not a substitute -- a `randn` vocabulary has a nucleus
of 17,000 tokens where a real one has 25, which flips every conclusion below.

  A. correctness vs transformers' own logits processors
  B. a sampler is a distribution — 400k draws vs the truth (and one broken drawer)
  C. cost of each filter, and the pre-filter that makes top-p cheap
  D. batched vs per-request loop, swept over vocabulary size
  E. sampling's share of a decode step
  F. what the knobs do to a real distribution

Run:  python3 run.py          (~1 min)
      python3 run.py --plot   (redraw from outputs/findings.json)
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
P01 = os.path.join(os.path.dirname(HERE), "01-manual-inference-loop")
sys.path.insert(0, HERE)
sys.path.insert(0, P01)

import torch  # noqa: E402

import sampling as S  # noqa: E402

torch.set_num_threads(6)
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BANK_TEXT = (
    "The capital of France is Paris. In 1969 humans first walked on the Moon. "
    "def add(a, b):\n    return a + b\n\nQ: What is 2+2?\nA: 4.\n"
    "Once upon a time there was a small robot who wanted to learn to cook.")


def build_bank():
    """62 real next-token logit vectors: prose, code, Q&A, story."""
    import loop_lib as L
    tok, model = L.load(MODEL_ID)
    ids = tok(BANK_TEXT, return_tensors="pt").input_ids
    with torch.inference_mode():
        logits = model(ids).logits[0].float()
    return tok, model, logits


def rows(bank, n):
    """n rows from the bank, repeating if n exceeds what we have."""
    idx = torch.arange(n) % bank.shape[0]
    return bank[idx].contiguous()


# ---------------------------------------------------------------------------


def section_a(f, bank):
    from transformers.generation.logits_process import (
        MinPLogitsWarper, RepetitionPenaltyLogitsProcessor, TopKLogitsWarper,
        TopPLogitsWarper)

    logits = bank[:16].clone()
    ids = torch.randint(0, logits.shape[1], (16, 16))
    out = []

    def compare(name, ours, theirs):
        ours_keep, theirs_keep = ~torch.isinf(ours), ~torch.isinf(theirs)
        same = torch.equal(ours_keep, theirs_keep)
        both = ours_keep & theirs_keep
        max_diff = (ours[both] - theirs[both]).abs().max().item()
        disputed = (ours_keep ^ theirs_keep)
        # How much probability mass do the tokens we disagree about carry?
        p = torch.softmax(logits, dim=-1)
        mass = float(p[disputed].sum()) if disputed.any() else 0.0
        out.append({"filter": name, "same_kept_set": bool(same),
                    "rows_differing": int((disputed.any(dim=1)).sum()),
                    "tokens_disputed": int(disputed.sum()),
                    "disputed_prob_mass": mass,
                    "max_value_diff": float(max_diff),
                    "kept_row0": int(ours_keep[0].sum())})
        print(f"  A: {name:26s} identical={same!s:5s} "
              f"rows differing {out[-1]['rows_differing']:2d}/16  "
              f"disputed mass {mass:.2e}")

    compare("top_k(50)", S.top_k_filter(logits.clone(), 50),
            TopKLogitsWarper(50)(ids, logits.clone()))
    compare("min_p(0.05)", S.min_p_filter(logits.clone(), 0.05),
            MinPLogitsWarper(0.05)(ids, logits.clone()))
    compare("repetition_penalty(1.1)",
            S.apply_repetition_penalty(logits.clone(), ids, 1.1),
            RepetitionPenaltyLogitsProcessor(1.1)(ids, logits.clone()))
    compare("top_p(0.9) full sort", S.top_p_filter_sort(logits.clone(), 0.9),
            TopPLogitsWarper(0.9)(ids, logits.clone()))
    f["A_correctness"] = out


def section_b(f, n_draws=400_000):
    torch.manual_seed(0)
    small = torch.tensor([[3.0, 2.0, 1.0, 0.5, -1.0, -4.0]])
    truth = torch.softmax(small, dim=-1)[0]
    big = small.expand(n_draws, -1).contiguous()

    def broken_gumbel(logits, generator=None):
        """The version with the clamp bug, kept so the failure is visible."""
        u = torch.rand(logits.shape, generator=generator)
        g = -torch.log(-torch.log(u.clamp_min(1e-20)).clamp_min(1e-20))
        return (logits + g).argmax(dim=-1, keepdim=True)

    rows_ = []
    for name, fn in (("torch.multinomial", S.draw_multinomial),
                     ("gumbel-max", S.draw_gumbel),
                     ("gumbel-max (clamp bug)", broken_gumbel)):
        g = torch.Generator().manual_seed(7)
        idx = fn(big, generator=g).flatten()
        emp = torch.bincount(idx, minlength=small.shape[1]).float()
        emp /= emp.sum()
        tvd = 0.5 * (emp - truth).abs().sum().item()
        rows_.append({"drawer": name, "tvd": round(tvd, 5),
                      "empirical": [round(x, 4) for x in emp.tolist()],
                      "truth": [round(x, 4) for x in truth.tolist()]})
        print(f"  B: {name:24s} total-variation distance {tvd:.5f}")
    f["B_distribution"] = {"n_draws": n_draws, "rows": rows_}


def section_c(f, bank, batches=(1, 8, 32), reps=3):
    out = []
    for b in batches:
        logits = rows(bank, b)
        fns = {
            "top_p full sort": lambda l=logits: S.top_p_filter_sort(l.clone(), 0.9),
            "top_p prefilter k=1024": lambda l=logits: S.top_p_filter_prefilter(
                l.clone(), 0.9, 1024),
            "top_k(50) only": lambda l=logits: S.top_k_filter(l.clone(), 50),
            "min_p(0.05) only": lambda l=logits: S.min_p_filter(l.clone(), 0.05),
            "argmax (greedy)": lambda l=logits: l.argmax(-1, keepdim=True),
        }
        best = {k: float("inf") for k in fns}
        for _ in range(reps + 1):
            for name, fn in fns.items():
                t0 = time.perf_counter()
                fn()
                best[name] = min(best[name], time.perf_counter() - t0)
        a = S.top_p_filter_sort(logits.clone(), 0.9)
        c = S.top_p_filter_prefilter(logits.clone(), 0.9, 1024)
        out.append({"batch": b,
                    **{k: round(1000 * v, 3) for k, v in best.items()},
                    "prefilter_speedup": round(
                        best["top_p full sort"] / best["top_p prefilter k=1024"], 2),
                    "identical_result": bool(torch.equal(torch.isinf(a),
                                                         torch.isinf(c)))})
        print(f"  C: b={b:2d} full sort {1000 * best['top_p full sort']:6.2f} ms | "
              f"prefilter {1000 * best['top_p prefilter k=1024']:5.2f} ms "
              f"({out[-1]['prefilter_speedup']}x, identical="
              f"{out[-1]['identical_result']}) | argmax "
              f"{1000 * best['argmax (greedy)']:5.2f} ms")
    f["C_filter_cost"] = out

    # When is the pre-filter safe? Depends entirely on temperature -- and on
    # whether you remembered to take the softmax over the whole row.
    sizes = {}
    for temp in (0.7, 1.0, 1.5, 2.0):
        scaled = bank / temp
        n = S.nucleus_size(scaled, 0.9)
        full = S.top_p_filter_sort(scaled.clone(), 0.9)
        pre = S.top_p_filter_prefilter(scaled.clone(), 0.9, 1024)
        bad = S.top_p_filter_prefilter_renorm(scaled.clone(), 0.9, 1024)
        differ = (torch.isinf(full) ^ torch.isinf(pre)).any(dim=1)
        differ_bad = (torch.isinf(full) ^ torch.isinf(bad)).any(dim=1)
        sizes[str(temp)] = {"mean_nucleus": round(n.float().mean().item(), 1),
                            "median_nucleus": int(n.median()),
                            "max_nucleus": int(n.max()),
                            "rows_over_1024": int((n > 1024).sum()),
                            "n_rows": int(bank.shape[0]),
                            "rows_where_prefilter_differs": int(differ.sum()),
                            "rows_where_renorm_differs": int(differ_bad.sum())}
        s = sizes[str(temp)]
        print(f"  C: T={temp}: nucleus median {s['median_nucleus']:6d} "
              f"max {s['max_nucleus']:6d} | rows over k=1024: "
              f"{s['rows_over_1024']:2d}/{s['n_rows']} | prefilter differs on "
              f"{s['rows_where_prefilter_differs']:2d} rows "
              f"(renormalised variant: {s['rows_where_renorm_differs']})")
    f["C_nucleus_size"] = sizes


def section_d(f, batch=32, reps=3):
    """Batched vs per-request loop, as a function of how big each row is."""
    out = []
    for vocab in (4096, 32768, 151936):
        g = torch.Generator().manual_seed(11)
        logits = torch.randn(batch, vocab, generator=g) * 3.0
        temps = [0.7 + 0.01 * i for i in range(batch)]

        def per_request():
            res = []
            for i in range(batch):
                row = logits[i:i + 1] / temps[i]
                row = S.top_k_filter(row, 50)
                res.append(S.draw_multinomial(row))
            return torch.cat(res)

        def batched():
            row = logits / torch.tensor(temps).unsqueeze(1)
            row = S.top_k_filter(row, 50)
            return S.draw_multinomial(row)

        best = {"per_request_loop": float("inf"), "batched": float("inf")}
        for _ in range(reps + 1):
            for name, fn in (("per_request_loop", per_request),
                             ("batched", batched)):
                t0 = time.perf_counter()
                fn()
                best[name] = min(best[name], time.perf_counter() - t0)
        out.append({"vocab": vocab, "batch": batch,
                    "per_request_loop_ms": round(1000 * best["per_request_loop"], 3),
                    "batched_ms": round(1000 * best["batched"], 3),
                    "speedup": round(best["per_request_loop"] / best["batched"], 2)})
        print(f"  D: vocab {vocab:6d}: loop "
              f"{1000 * best['per_request_loop']:7.2f} ms vs batched "
              f"{1000 * best['batched']:6.2f} ms ({out[-1]['speedup']}x)")
    f["D_batched_vs_loop"] = out


def section_e(f):
    p01 = os.path.join(P01, "outputs", "findings.json")
    decode = {r["batch"]: r["step_s"] for r in
              json.load(open(p01))["D_decode_batch_scaling"]} \
        if os.path.exists(p01) else {}
    out = []
    for r in f["C_filter_cost"]:
        b = r["batch"]
        step_ms = 1000 * decode.get(b, float("nan"))
        full = r["top_p full sort"] + r["top_k(50) only"]
        fast = r["top_p prefilter k=1024"] + r["top_k(50) only"]
        out.append({"batch": b, "decode_step_ms": round(step_ms, 1),
                    "greedy_argmax_ms": r["argmax (greedy)"],
                    "topk_topp_full_sort_ms": round(full, 2),
                    "topk_topp_prefilter_ms": round(fast, 2),
                    "share_greedy_pct": round(100 * r["argmax (greedy)"] / step_ms, 1),
                    "share_full_sort_pct": round(100 * full / step_ms, 1),
                    "share_prefilter_pct": round(100 * fast / step_ms, 1)})
        print(f"  E: b={b:2d} decode step {step_ms:6.1f} ms | greedy "
              f"{out[-1]['share_greedy_pct']:4.1f}% | top-k+top-p "
              f"{out[-1]['share_full_sort_pct']:5.1f}% -> "
              f"{out[-1]['share_prefilter_pct']:4.1f}% with the pre-filter")
    f["E_share_of_decode"] = out


def section_f(f, tok, bank):
    logits = bank[7:8].clone()          # the position right after "capital of France is"
    probs = torch.softmax(logits, dim=-1)[0]
    top = torch.topk(probs, 5)
    out = []
    for temp in (0.0, 0.7, 1.0, 1.5):
        l = logits.clone() if temp == 0 else S.apply_temperature(logits.clone(), temp)
        n_p = int(S.nucleus_size(l, 0.9)[0]) if temp > 0 else 1
        n_minp = int((~torch.isinf(S.min_p_filter(l.clone(), 0.05))).sum()) \
            if temp > 0 else 1
        out.append({"temperature": temp, "nucleus_size_p090": n_p,
                    "kept_by_min_p_005": n_minp,
                    "top1_prob": round(float(torch.softmax(l, -1).max()), 4)})
        print(f"  F: T={temp}: top-p(0.9) keeps {n_p:5d}, min-p(0.05) keeps "
              f"{n_minp:3d}, p(best)={out[-1]['top1_prob']:.3f}")
    f["F_real_logits"] = {
        "context": BANK_TEXT[:40],
        "top5": [{"token": tok.decode([int(i)]), "prob": round(float(p), 4)}
                 for p, i in zip(top.values, top.indices)],
        "rows": out}
    print("  F: top-5: " + ", ".join(f"{d['token']!r}={d['prob']}"
                                     for d in f["F_real_logits"]["top5"]))


# ---------------------------------------------------------------------------


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    c = f["C_filter_cost"]
    bs = [r["batch"] for r in c]
    w = .18
    xs = list(range(len(bs)))
    series = [("top_p full sort", "top-p (sort all 151,936)", "tab:red"),
              ("top_p prefilter k=1024", "top-p (top-1024 first)", "tab:green"),
              ("top_k(50) only", "top-k(50)", "tab:blue"),
              ("min_p(0.05) only", "min-p(0.05)", "tab:purple"),
              ("argmax (greedy)", "argmax (greedy)", "tab:grey")]
    for i, (key, label, col) in enumerate(series):
        ax[0].bar([x + (i - 2) * w for x in xs], [r[key] for r in c],
                  width=w, label=label, color=col)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels([f"batch {b}" for b in bs])
    ax[0].set_ylabel("milliseconds")
    ax[0].set_yscale("log")
    ax[0].set_title("C. cost of each logit filter\n(real logits, 151,936 vocab)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3, axis="y", which="both")

    e = f["E_share_of_decode"]
    xs = list(range(len(e)))
    ax[1].bar([x - .22 for x in xs], [r["share_full_sort_pct"] for r in e],
              width=.44, color="tab:red", label="top-k + top-p (full sort)")
    ax[1].bar([x + .22 for x in xs], [r["share_prefilter_pct"] for r in e],
              width=.44, color="tab:green", label="top-k + top-p (pre-filtered)")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels([f"batch {r['batch']}" for r in e])
    ax[1].set_ylabel("% of one decode step")
    ax[1].set_title("E. sampling is not free —\nand its share grows with batch")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis="y")
    for i, r in enumerate(e):
        ax[1].text(i - .22, r["share_full_sort_pct"] + 1,
                   f"{r['share_full_sort_pct']:.0f}%", ha="center", fontsize=8)
        ax[1].text(i + .22, r["share_prefilter_pct"] + 1,
                   f"{r['share_prefilter_pct']:.0f}%", ha="center", fontsize=8)

    n = f["C_nucleus_size"]
    temps = sorted(n, key=float)
    ax[2].plot(temps, [n[t]["median_nucleus"] for t in temps], "o-",
               label="median nucleus size")
    ax[2].plot(temps, [n[t]["max_nucleus"] for t in temps], "s--",
               label="largest of 62 rows")
    ax[2].axhline(1024, color="grey", ls=":", label="pre-filter k = 1024")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("temperature")
    ax[2].set_ylabel("tokens inside the top-p = 0.9 nucleus")
    ax[2].set_title("C2. the pre-filter is exact until\nthe temperature opens the tail")
    ax[2].legend(fontsize=8)
    ax[2].grid(alpha=.3, which="both")

    fig.tight_layout()
    p = os.path.join(OUT, "sampling.png")
    fig.savefig(p, dpi=110)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")
    if "--plot" in sys.argv:
        plot(json.load(open(fpath)))
        return
    t0 = time.time()
    print("building a bank of real logits ...")
    tok, model, bank = build_bank()
    print(f"  {bank.shape[0]} real next-token distributions, vocab {bank.shape[1]}")
    f = {"vocab": int(bank.shape[1]), "model": MODEL_ID, "threads": 6,
         "bank_rows": int(bank.shape[0]), "bank_text": BANK_TEXT}
    section_a(f, bank)
    section_b(f)
    section_c(f, bank)
    section_d(f)
    section_e(f)
    section_f(f, tok, bank)
    f["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(f, open(fpath, "w"), indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w") as fh:
        fh.write("section,key,value\n")
        for r in f["A_correctness"]:
            fh.write(f"A,{r['filter']}|same_kept_set,{r['same_kept_set']}\n")
            fh.write(f"A,{r['filter']}|disputed_prob_mass,{r['disputed_prob_mass']}\n")
        for r in f["B_distribution"]["rows"]:
            fh.write(f"B,tvd|{r['drawer']},{r['tvd']}\n")
        for r in f["C_filter_cost"]:
            fh.write(f"C,prefilter_speedup@b{r['batch']},{r['prefilter_speedup']}\n")
        for t, s in f["C_nucleus_size"].items():
            fh.write(f"C,median_nucleus@T{t},{s['median_nucleus']}\n")
            fh.write(f"C,prefilter_rows_differing@T{t},"
                     f"{s['rows_where_prefilter_differs']}\n")
        for r in f["D_batched_vs_loop"]:
            fh.write(f"D,batched_speedup@vocab{r['vocab']},{r['speedup']}\n")
        for r in f["E_share_of_decode"]:
            fh.write(f"E,share_full_sort_pct@b{r['batch']},{r['share_full_sort_pct']}\n")
            fh.write(f"E,share_prefilter_pct@b{r['batch']},{r['share_prefilter_pct']}\n")
    print(f"  wrote {fpath}")
    plot(f)
    print(f"done in {f['wall_clock_s']}s")


if __name__ == "__main__":
    main()
