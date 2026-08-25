"""Project 06 — Determinism audit.

Greedy decoding has no randomness in it. Run it twice and you can still get
two different answers. This project finds out where the difference comes from,
how big it is, and how often it changes a token.

  A. repeatability   — same everything, 100 forward passes
  B. batch size      — the same prompt alone vs inside a batch
  C. threads         — the same prompt on 1 / 2 / 3 / 12 CPU threads
  D. padding         — the same prompt with left padding and a mask
  E. text divergence — 20 full generations under a randomised environment
  F. decision margin — the top1-vs-top2 logit gap, over 240 decode steps
  G. bfloat16        — redo the audit at the precision production actually uses
  H. the fix         — pin the environment, re-run, count

Run:  python3 run.py          (~4 min)
      python3 run.py --plot   (redraw from outputs/findings.json)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
P01 = os.path.join(os.path.dirname(HERE), "01-manual-inference-loop")
sys.path.insert(0, HERE)
sys.path.insert(0, P01)

import loop_lib as L  # noqa: E402
import torch  # noqa: E402

PROMPT = ("Explain in two sentences why a large language model server may "
          "return different answers to the same question.")
REF_THREADS = 6


def logits_for(model, ids, *, batch=1, threads=REF_THREADS, left_pad=0,
               filler=None):
    """Last-position logits for `ids`, computed under a chosen environment."""
    torch.set_num_threads(threads)
    with torch.inference_mode():
        if left_pad:
            pad = filler[:1, :left_pad]
            x = torch.cat([pad, ids], dim=1)
            mask = torch.cat([torch.zeros(1, left_pad, dtype=torch.long),
                              torch.ones_like(ids)], dim=1)
            out = model(x, attention_mask=mask)
            return out.logits[0, -1, :].clone()
        if batch == 1:
            return model(ids).logits[0, -1, :].clone()
        # Same prompt at row 0, unrelated rows around it. Same length, so no
        # padding is involved -- only the shape of the matrix multiply changes.
        others = filler[:batch - 1, :ids.shape[1]]
        x = torch.cat([ids, others], dim=0)
        return model(x).logits[0, -1, :].clone()


def compare(ref, other):
    d = (ref - other).abs()
    return {"max_abs_diff": float(d.max()),
            "mean_abs_diff": float(d.mean()),
            "bitwise_identical": bool(torch.equal(ref, other)),
            "argmax_same": bool(ref.argmax() == other.argmax()),
            "top5_same": bool(torch.equal(ref.topk(5).indices,
                                          other.topk(5).indices))}


# ---------------------------------------------------------------------------


def section_a(model, ids, f, n=100):
    ref = logits_for(model, ids)
    diffs = 0
    t0 = time.time()
    for _ in range(n):
        if not torch.equal(ref, logits_for(model, ids)):
            diffs += 1
    f["A_repeatability"] = {"runs": n, "bitwise_different": diffs,
                            "seconds": round(time.time() - t0, 1)}
    print(f"  A: {n} identical forward passes -> {diffs} bitwise differences")
    return ref


def section_b(model, ids, ref, f, batches=(2, 4, 8, 16, 32), filler=None):
    rows = []
    for b in batches:
        r = compare(ref, logits_for(model, ids, batch=b, filler=filler))
        r["batch"] = b
        rows.append(r)
        print(f"  B: batch {b:3d} -> bitwise identical {r['bitwise_identical']!s:5s} "
              f"max |diff| {r['max_abs_diff']:.2e}  argmax same "
              f"{r['argmax_same']}  top-5 same {r['top5_same']}")
    f["B_batch_size"] = rows


def section_c(model, ids, ref, f, threads=(1, 2, 3, 12)):
    rows = []
    for t in threads:
        r = compare(ref, logits_for(model, ids, threads=t))
        r["threads"] = t
        rows.append(r)
        print(f"  C: {t:2d} threads -> bitwise identical "
              f"{r['bitwise_identical']!s:5s} max |diff| {r['max_abs_diff']:.2e} "
              f" argmax same {r['argmax_same']}")
    torch.set_num_threads(REF_THREADS)
    f["C_threads"] = rows


def section_d(model, ids, ref, f, pads=(1, 8, 64), filler=None):
    rows = []
    for p in pads:
        r = compare(ref, logits_for(model, ids, left_pad=p, filler=filler))
        r["left_pad_tokens"] = p
        rows.append(r)
        print(f"  D: left pad {p:3d} -> bitwise identical "
              f"{r['bitwise_identical']!s:5s} max |diff| {r['max_abs_diff']:.2e} "
              f" argmax same {r['argmax_same']}")
    f["D_padding"] = rows


def generate_text(model, tok, ids, *, batch=1, threads=REF_THREADS, filler=None,
                  max_new=24):
    """Greedy generation, but with row 0 of a batch of `batch` sequences."""
    torch.set_num_threads(threads)
    with torch.inference_mode():
        if batch == 1:
            x = ids
        else:
            x = torch.cat([ids, filler[:batch - 1, :ids.shape[1]]], dim=0)
        out = model(x, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        got = [int(nxt[0])]
        for _ in range(max_new - 1):
            out = model(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            got.append(int(nxt[0]))
    return got


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1


def section_e(model, tok, ids, f, n=20, filler=None, seed=0):
    """The user-visible question: does the ANSWER change?"""
    rng = random.Random(seed)
    ref = generate_text(model, tok, ids, filler=filler)
    rows, diverged = [], 0
    t0 = time.time()
    for _ in range(n):
        b = rng.choice([1, 2, 4, 8, 16])
        th = rng.choice([1, 2, 3, 6, 12])
        got = generate_text(model, tok, ids, batch=b, threads=th, filler=filler)
        fd = first_diff(ref, got)
        diverged += int(fd != -1)
        rows.append({"batch": b, "threads": th, "first_divergent_token": fd})
    torch.set_num_threads(REF_THREADS)
    fds = [r["first_divergent_token"] for r in rows if r["first_divergent_token"] >= 0]
    f["E_text_divergence"] = {
        "runs": n, "diverged": diverged,
        "divergence_rate": round(diverged / n, 3),
        "median_first_divergent_token": sorted(fds)[len(fds) // 2] if fds else None,
        "earliest_divergence": min(fds) if fds else None,
        "reference_text": tok.decode(ref),
        "rows": rows,
        "seconds": round(time.time() - t0, 1)}
    print(f"  E: {diverged}/{n} randomised runs produced different text "
          f"(median first divergence at token "
          f"{f['E_text_divergence']['median_first_divergent_token']})")


MARGIN_PROMPTS = [
    PROMPT,
    ("Repeat the following exactly: the quick brown fox jumps over the lazy "
     "dog. the quick brown fox"),
    "Count: 1, 2, 3, 4,",
    "A B A B A B A B A B",
]


def margins(model, tok, prompts, steps, dtype_name="float32"):
    """Greedy-decode each prompt and record the top1-vs-top2 logit gap."""
    torch.set_num_threads(REF_THREADS)
    gaps, ties = [], 0
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            past = out.past_key_values
            for _ in range(steps):
                lg = out.logits[0, -1, :].float()
                top2 = lg.topk(2).values
                gap = float(top2[0] - top2[1])
                gaps.append(gap)
                ties += int(gap == 0.0)
                nxt = lg.argmax().view(1, 1)
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
    return gaps, ties


def section_f(model, tok, f, steps=60):
    """Why a 1e-5 wobble does or does not flip a token: the decision margin."""
    gaps, ties = margins(model, tok, MARGIN_PROMPTS, steps)
    gaps_sorted = sorted(gaps)
    noise = max(r["max_abs_diff"] for r in f["B_batch_size"])
    f["F_decision_margin"] = {
        "n_steps": len(gaps),
        "prompts": len(MARGIN_PROMPTS),
        "gaps_sorted_head": [round(g, 5) for g in gaps_sorted[:20]],
        "min_gap": round(min(gaps), 6),
        "p10_gap": round(gaps_sorted[len(gaps) // 10], 4),
        "median_gap": round(gaps_sorted[len(gaps) // 2], 4),
        "exact_ties": ties,
        "batch_noise_max_abs_diff": noise,
        "margin_over_noise_worst_case": round(min(gaps) / noise, 1),
        "steps_within_noise": int(sum(1 for g in gaps if g < 2 * noise)),
        "fraction_within_noise": round(
            sum(1 for g in gaps if g < 2 * noise) / len(gaps), 4)}
    m = f["F_decision_margin"]
    print(f"  F: {m['n_steps']} decode steps over {m['prompts']} prompts — "
          f"median margin {m['median_gap']}, p10 {m['p10_gap']}, "
          f"min {m['min_gap']}")
    print(f"     the worst margin is {m['margin_over_noise_worst_case']}x the "
          f"batch noise ({noise:.2e}); {m['steps_within_noise']} steps are "
          f"inside it")


def section_g_precision(tok, f, steps=60):
    """Production does not serve fp32. Redo the audit in bfloat16."""
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(REF_THREADS)
    m16 = AutoModelForCausalLM.from_pretrained(L.MODEL_ID,
                                               dtype=torch.bfloat16).eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    torch.manual_seed(0)
    filler = torch.randint(1000, 20000, (32, ids.shape[1]))
    rows = []
    with torch.inference_mode():
        ref = m16(ids).logits[0, -1, :].float().clone()
        for b in (2, 8, 32):
            x = torch.cat([ids, filler[:b - 1]], dim=0)
            other = m16(x).logits[0, -1, :].float()
            r = compare(ref, other)
            r["batch"] = b
            rows.append(r)
            print(f"  G: bf16 batch {b:3d} -> bitwise identical "
                  f"{r['bitwise_identical']!s:5s} max |diff| "
                  f"{r['max_abs_diff']:.2e}")
    gaps, ties = margins(m16, tok, MARGIN_PROMPTS, steps)
    gaps_sorted = sorted(gaps)
    f["G_bfloat16"] = {
        "batch_rows": rows,
        "n_steps": len(gaps),
        "min_gap": round(min(gaps), 6),
        "median_gap": round(gaps_sorted[len(gaps) // 2], 4),
        "exact_ties": ties,
        "tie_rate": round(ties / len(gaps), 4),
        "fp32_exact_ties": f["F_decision_margin"]["exact_ties"]}
    print(f"  G: bf16 decode margins — median {f['G_bfloat16']['median_gap']}, "
          f"EXACT ties in {ties}/{len(gaps)} steps "
          f"({100 * f['G_bfloat16']['tie_rate']:.1f}%), vs "
          f"{f['F_decision_margin']['exact_ties']} in fp32")
    del m16


def section_h(model, tok, ids, f, n=20, filler=None):
    """The fix: pin the one variable we control, then re-run the same test."""
    ref = generate_text(model, tok, ids, batch=1, threads=REF_THREADS,
                        filler=filler)
    diverged = 0
    for _ in range(n):
        got = generate_text(model, tok, ids, batch=1, threads=REF_THREADS,
                            filler=filler)
        diverged += int(first_diff(ref, got) != -1)
    f["H_fix"] = {
        "runs": n, "diverged": diverged,
        "divergence_rate": round(diverged / n, 3),
        "fixed_variables": ["batch position", "thread count"],
        "still_unfixed": ["different hardware", "different engine",
                          "different library version", "a scheduler that "
                          "batches your request with other traffic"]}
    print(f"  H: after pinning batch size and thread count: {diverged}/{n} "
          f"runs diverged")


# ---------------------------------------------------------------------------


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))

    b = f["B_batch_size"]
    c = f["C_threads"]
    labels = [f"batch {r['batch']}" for r in b] + \
             [f"{r['threads']} thr" for r in c]
    vals = [r["max_abs_diff"] for r in b] + [r["max_abs_diff"] for r in c]
    cols = ["tab:blue"] * len(b) + ["tab:orange"] * len(c)
    ax[0].bar(range(len(vals)), vals, color=cols)
    ax[0].set_xticks(range(len(vals)))
    ax[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax[0].set_yscale("log")
    ax[0].set_ylabel("max |logit difference| vs the reference run")
    ax[0].set_title("B/C. the logits move when the batch shape\nor the thread "
                    "count does (fp32)")
    ax[0].grid(alpha=.3, axis="y", which="both")

    fdm = f["F_decision_margin"]
    gaps = fdm["gaps_sorted_head"]
    ax[1].plot(range(len(gaps)), gaps, "o-", label="fp32 (20 closest calls)")
    ax[1].axhline(fdm["batch_noise_max_abs_diff"], color="crimson", ls="--",
                  label=f"batch-shape noise ({fdm['batch_noise_max_abs_diff']:.1e})")
    ax[1].set_yscale("log")
    ax[1].axhline(max(f["G_bfloat16"]["min_gap"], 1e-6), color="tab:purple",
                  ls=":", label="bf16 smallest margin (exact ties at 0)")
    ax[1].set_xlabel("the 20 closest decode calls, sorted by margin")
    ax[1].set_ylabel("logit gap between the best and second-best token")
    ax[1].set_title("F. the closest call in 240 steps is still\n"
                    f"{fdm['margin_over_noise_worst_case']}x above the noise")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")

    e, h, g16 = f["E_text_divergence"], f["H_fix"], f["G_bfloat16"]
    vals = [f["F_decision_margin"]["exact_ties"], g16["exact_ties"]]
    ax[2].bar(["float32", "bfloat16"], vals, color=["tab:blue", "tab:purple"])
    ax[2].set_ylabel(f"decode steps out of {g16['n_steps']} where the top two\n"
                     "logits are EXACTLY equal")
    ax[2].set_ylim(0, max(vals) + 1.5)
    ax[2].set_title("G. lower precision does not add noise here —\nit creates ties")
    ax[2].grid(alpha=.3, axis="y")
    for i, v in enumerate(vals):
        ax[2].text(i, v + .1, str(v), ha="center")
    ax[2].text(0.5, max(vals) + 1.0,
               f"text divergence: {100 * e['divergence_rate']:.0f}% randomised, "
               f"{100 * h['divergence_rate']:.0f}% pinned",
               ha="center", fontsize=8, style="italic")

    fig.tight_layout()
    p = os.path.join(OUT, "determinism.png")
    fig.savefig(p, dpi=110)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")
    if "--plot" in sys.argv:
        plot(json.load(open(fpath)))
        return
    t0 = time.time()
    tok, model = L.load()
    ids = tok(PROMPT, return_tensors="pt").input_ids
    torch.manual_seed(0)
    filler = torch.randint(1000, 20000, (32, ids.shape[1] + 64))
    f = {"model": L.MODEL_ID, "prompt": PROMPT,
         "prompt_tokens": int(ids.shape[1]), "reference_threads": REF_THREADS}

    ref = section_a(model, ids, f)
    section_b(model, ids, ref, f, filler=filler)
    section_c(model, ids, ref, f)
    section_d(model, ids, ref, f, filler=filler)
    section_e(model, tok, ids, f, filler=filler)
    section_f(model, tok, f)
    section_g_precision(tok, f)
    section_h(model, tok, ids, f, filler=filler)

    f["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(f, open(fpath, "w"), indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w") as fh:
        fh.write("section,key,value\n")
        fh.write(f"A,bitwise_different_of_{f['A_repeatability']['runs']},"
                 f"{f['A_repeatability']['bitwise_different']}\n")
        for r in f["B_batch_size"]:
            fh.write(f"B,max_abs_diff@batch{r['batch']},{r['max_abs_diff']}\n")
        for r in f["C_threads"]:
            fh.write(f"C,max_abs_diff@threads{r['threads']},{r['max_abs_diff']}\n")
        for r in f["D_padding"]:
            fh.write(f"D,max_abs_diff@pad{r['left_pad_tokens']},{r['max_abs_diff']}\n")
        fh.write(f"E,divergence_rate,{f['E_text_divergence']['divergence_rate']}\n")
        fh.write(f"F,median_gap,{f['F_decision_margin']['median_gap']}\n")
        fh.write(f"F,min_gap,{f['F_decision_margin']['min_gap']}\n")
        fh.write(f"G,bf16_tie_rate,{f['G_bfloat16']['tie_rate']}\n")
        fh.write(f"H,divergence_rate,{f['H_fix']['divergence_rate']}\n")
    print(f"  wrote {fpath}")
    plot(f)
    print(f"done in {f['wall_clock_s']}s")


if __name__ == "__main__":
    main()
