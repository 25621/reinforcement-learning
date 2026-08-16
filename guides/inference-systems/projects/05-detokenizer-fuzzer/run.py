"""Project 05 — Detokenizer fuzzer.

  A. how much of the vocabulary is not a character at all
  B. hand-written strings that break naive streaming (emoji, scripts, ZWJ)
  C. fuzz: 5,000 random token sequences vs an oracle
  D. a REAL generation from the model that breaks it
  E. cost: a moving window vs re-decoding everything

Run:  python3 run.py          (~1 min)
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

from detok import (DecodeAllDetok, IncrementalDetok, naive_stream,  # noqa: E402
                   oracle, run)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

STRINGS = [
    "🎉 party time",
    "🫠 melting face",
    "👨‍👩‍👧‍👦 family",
    "ᚠᚢᚦᚨᚱᚲ runes",
    "ကျွန်ုပ် Burmese",
    "𓀀 hieroglyph",
    "🥹🫶🏽 skin tone",
    "café naïve",          # 2-byte characters: usually fine
    "你好世界",              # 3-byte CJK: Qwen has whole tokens for these
    "plain ascii text",
]


def section_a(tok, f):
    """A byte-level BPE vocabulary contains pieces that are not characters."""
    t0 = time.perf_counter()
    n_frag = 0
    examples = []
    for i in range(tok.vocab_size):
        s = tok.decode([i])
        if "�" in s:
            n_frag += 1
            if len(examples) < 6:
                examples.append(i)
    f["A_vocab"] = {
        "vocab_size": int(tok.vocab_size),
        "fragment_tokens": n_frag,
        "fragment_pct": round(100 * n_frag / tok.vocab_size, 2),
        "example_ids": examples,
        "scan_s": round(time.perf_counter() - t0, 1)}
    print(f"  A: {n_frag} of {tok.vocab_size} vocabulary entries "
          f"({f['A_vocab']['fragment_pct']}%) decode to a replacement "
          f"character on their own")


def section_b(tok, f):
    rows = []
    for s in STRINGS:
        ids = tok(s, add_special_tokens=False).input_ids
        truth = oracle(tok, ids)
        naive = naive_stream(tok, ids)
        inc = run(IncrementalDetok(tok), ids)
        alld = run(DecodeAllDetok(tok), ids)
        rows.append({"text": s, "n_tokens": len(ids),
                     "naive_ok": naive == truth, "naive_out": naive,
                     "incremental_ok": inc == truth,
                     "decode_all_ok": alld == truth,
                     "replacement_chars_emitted": naive.count("�")})
        print(f"  B: {s!r:32s} tokens={len(ids):2d} naive_ok="
              f"{rows[-1]['naive_ok']!s:5s} bad_chars="
              f"{rows[-1]['replacement_chars_emitted']:2d} "
              f"incremental_ok={rows[-1]['incremental_ok']}")
    f["B_strings"] = rows
    f["B_summary"] = {
        "n": len(rows),
        "naive_passed": sum(r["naive_ok"] for r in rows),
        "incremental_passed": sum(r["incremental_ok"] for r in rows),
        "decode_all_passed": sum(r["decode_all_ok"] for r in rows)}


# Unicode ranges whose characters need 3 or 4 UTF-8 bytes. A byte-level BPE
# vocabulary trained mostly on English keeps whole tokens for common CJK but
# splits rare emoji and minority scripts into byte fragments.
RANGES = [
    (0x1F300, 0x1F9FF),   # emoji
    (0x16A0, 0x16F0),     # runic
    (0x1000, 0x109F),     # Burmese
    (0x13000, 0x1342F),   # Egyptian hieroglyphs
    (0x0E00, 0x0E7F),     # Thai
    (0x4E00, 0x9FFF),     # CJK
    (0x0020, 0x007E),     # plain ASCII, so the corpus is mixed
]


def random_text(rng, n_chars):
    out = []
    for _ in range(n_chars):
        lo, hi = rng.choice(RANGES)
        out.append(chr(rng.randrange(lo, hi)))
    return "".join(out)


def fuzz(tok, f, key, sampler, n_cases, seed):
    rng = random.Random(seed)
    stats = {"n_cases": n_cases, "naive_wrong": 0, "incremental_wrong": 0,
             "decode_all_wrong": 0, "naive_bad_chars": 0,
             "max_pending_tokens": 0, "shortest_failing_case": None}
    for _ in range(n_cases):
        ids = sampler(rng)
        if len(ids) < 2:
            continue
        n = len(ids)
        truth = oracle(tok, ids)
        naive = naive_stream(tok, ids)
        d = IncrementalDetok(tok)
        inc = run(d, ids)
        alld = run(DecodeAllDetok(tok), ids)
        stats["max_pending_tokens"] = max(stats["max_pending_tokens"],
                                          d.max_pending_tokens)
        if naive != truth:
            stats["naive_wrong"] += 1
            stats["naive_bad_chars"] += naive.count("�")
            cur = stats["shortest_failing_case"]
            if cur is None or n < cur["n_tokens"]:
                stats["shortest_failing_case"] = {
                    "n_tokens": n, "ids": ids,
                    "pieces": [tok.decode([i]) for i in ids],
                    "naive": naive, "truth": truth}
        stats["incremental_wrong"] += int(inc != truth)
        stats["decode_all_wrong"] += int(alld != truth)
    stats["naive_error_rate"] = round(stats["naive_wrong"] / n_cases, 4)
    stats["incremental_accuracy"] = round(1 - stats["incremental_wrong"] / n_cases, 4)
    stats["decode_all_accuracy"] = round(1 - stats["decode_all_wrong"] / n_cases, 4)
    f[key] = stats
    print(f"  C[{key}]: {n_cases} cases — naive wrong in "
          f"{stats['naive_wrong']} ({100 * stats['naive_error_rate']:.1f}%), "
          f"incremental {100 * stats['incremental_accuracy']:.2f}% exact, "
          f"decode-all {100 * stats['decode_all_accuracy']:.2f}% exact, "
          f"max pending {stats['max_pending_tokens']} tokens")
    sc = stats["shortest_failing_case"]
    if sc:
        print(f"     shortest failure: {sc['n_tokens']} tokens "
              f"{sc['pieces']} -> naive {sc['naive']!r} vs {sc['truth']!r}")


def section_c(tok, f, n_cases=5000):
    V = tok.vocab_size

    def uniform_ids(rng):
        return [rng.randrange(V) for _ in range(rng.randint(2, 12))]

    def text_ids(rng):
        return tok(random_text(rng, rng.randint(2, 10)),
                   add_special_tokens=False).input_ids

    fuzz(tok, f, "C_fuzz_uniform_tokens", uniform_ids, n_cases, seed=0)
    fuzz(tok, f, "C_fuzz_random_text", text_ids, n_cases, seed=1)


def section_d(f, max_new=40):
    """Not a fuzz case: the model, asked for emoji, produces one by itself."""
    import loop_lib as L
    tok, model = L.load(MODEL_ID)
    prompt = "Write one sentence full of emoji about a party: 🎉"
    ids = tok(prompt, return_tensors="pt").input_ids
    res = L.generate_with_cache(model, ids, max_new_tokens=max_new,
                                eos_id=tok.eos_token_id)
    out_ids = res.token_ids
    truth = oracle(tok, out_ids)
    naive = naive_stream(tok, out_ids)
    inc = run(IncrementalDetok(tok), out_ids)
    f["D_real_generation"] = {
        "prompt": prompt,
        "n_tokens": len(out_ids),
        "truth": truth,
        "naive": naive,
        "incremental": inc,
        "naive_ok": naive == truth,
        "incremental_ok": inc == truth,
        "replacement_chars": naive.count("�"),
        "first_pieces": [tok.decode([t]) for t in out_ids[:12]]}
    print(f"  D: real generation, {len(out_ids)} tokens")
    print(f"     correct     : {truth[:70]!r}")
    print(f"     naive stream: {naive[:70]!r}")
    print(f"     naive emitted {naive.count(chr(0xFFFD))} replacement characters; "
          f"incremental_ok={inc == truth}")
    return tok


def section_e(tok, f, lengths=(250, 500, 1000, 2000, 4000), reps=2):
    rng = random.Random(1)
    rows = []
    for n in lengths:
        ids = [rng.randrange(1000, 20000) for _ in range(n)]

        def windowed():
            run(IncrementalDetok(tok), ids)

        def decode_all():
            run(DecodeAllDetok(tok), ids)

        best = {"windowed": float("inf"), "decode_all": float("inf")}
        for _ in range(reps):
            for name, fn in (("windowed", windowed), ("decode_all", decode_all)):
                t0 = time.perf_counter()
                fn()
                best[name] = min(best[name], time.perf_counter() - t0)
        rows.append({"n_tokens": n,
                     "windowed_us_per_token": round(1e6 * best["windowed"] / n, 2),
                     "decode_all_us_per_token": round(
                         1e6 * best["decode_all"] / n, 2),
                     "ratio": round(best["decode_all"] / best["windowed"], 2)})
        print(f"  E: {n:5d} tokens — windowed "
              f"{rows[-1]['windowed_us_per_token']:7.2f} us/tok, decode-all "
              f"{rows[-1]['decode_all_us_per_token']:8.2f} us/tok "
              f"({rows[-1]['ratio']}x)")
    f["E_cost"] = rows


# ---------------------------------------------------------------------------


def plot(f):
    import warnings

    import matplotlib
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")   # emoji glyphs are missing from the font
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))

    b = f["B_summary"]
    cu, ct = f["C_fuzz_uniform_tokens"], f["C_fuzz_random_text"]
    names = ["naive\n(decode each token)", "decode-all\n+ hold back",
             "moving window"]
    vals_u = [100 * cu["naive_error_rate"],
              100 * (1 - cu["decode_all_accuracy"]),
              100 * (1 - cu["incremental_accuracy"])]
    vals_t = [100 * ct["naive_error_rate"],
              100 * (1 - ct["decode_all_accuracy"]),
              100 * (1 - ct["incremental_accuracy"])]
    xs = range(3)
    ax[0].bar([x - .2 for x in xs], vals_u, width=.4, color="tab:grey",
              label="uniform random token IDs")
    ax[0].bar([x + .2 for x in xs], vals_t, width=.4, color="tab:red",
              label="random multilingual TEXT")
    ax[0].set_xticks(list(xs))
    ax[0].set_xticklabels(names, fontsize=8)
    ax[0].legend(fontsize=8)
    ax[0].set_ylabel(f"% wrong out of {cu['n_cases']} fuzz cases")
    ax[0].set_title("C. the same fuzzer finds nothing or\neverything, depending on its inputs")
    ax[0].grid(alpha=.3, axis="y")
    for i, v in enumerate(vals_t):
        ax[0].text(i + .2, v + .5, f"{v:.1f}%", ha="center", fontsize=8)
    for i, v in enumerate(vals_u):
        ax[0].text(i - .2, v + .5, f"{v:.1f}%", ha="center", fontsize=8)

    rows = f["B_strings"]
    labels = [r["text"][:14] for r in rows]
    bad = [r["replacement_chars_emitted"] for r in rows]
    ax[1].barh(range(len(rows)), bad,
               color=["tab:red" if x else "tab:green" for x in bad])
    ax[1].set_yticks(range(len(rows)))
    ax[1].set_yticklabels(labels, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("replacement characters the naive stream shows the user")
    ax[1].set_title(f"B. {b['naive_passed']}/{b['n']} strings survive naive\n"
                    f"streaming ({b['incremental_passed']}/{b['n']} with a window)")
    ax[1].grid(alpha=.3, axis="x")

    e = f["E_cost"]
    ax[2].plot([r["n_tokens"] for r in e],
               [r["windowed_us_per_token"] for r in e], "o-",
               color="tab:green", label="moving window")
    ax[2].plot([r["n_tokens"] for r in e],
               [r["decode_all_us_per_token"] for r in e], "s--",
               color="tab:red", label="re-decode everything")
    ax[2].set_xscale("log", base=2)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("tokens generated so far")
    ax[2].set_ylabel("microseconds per token")
    ax[2].set_title("E. both are correct;\nonly one stays cheap "
                    f"({e[-1]['ratio']}x at {e[-1]['n_tokens']} tokens)")
    ax[2].legend()
    ax[2].grid(alpha=.3, which="both")

    fig.tight_layout()
    p = os.path.join(OUT, "detokenizer.png")
    fig.savefig(p, dpi=110)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")
    if "--plot" in sys.argv:
        plot(json.load(open(fpath)))
        return
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    f = {"model": MODEL_ID}
    section_a(tok, f)
    section_b(tok, f)
    section_c(tok, f)
    section_d(f)
    section_e(tok, f)
    f["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(f, open(fpath, "w"), indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "findings.csv"), "w") as fh:
        fh.write("section,key,value\n")
        fh.write(f"A,fragment_tokens,{f['A_vocab']['fragment_tokens']}\n")
        fh.write(f"A,fragment_pct,{f['A_vocab']['fragment_pct']}\n")
        fh.write(f"B,naive_passed,{f['B_summary']['naive_passed']}\n")
        fh.write(f"B,incremental_passed,{f['B_summary']['incremental_passed']}\n")
        for key in ("C_fuzz_uniform_tokens", "C_fuzz_random_text"):
            c = f[key]
            for k in ("naive_error_rate", "incremental_accuracy",
                      "decode_all_accuracy", "max_pending_tokens"):
                fh.write(f"{key},{k},{c[k]}\n")
        fh.write(f"D,naive_replacement_chars,"
                 f"{f['D_real_generation']['replacement_chars']}\n")
        for r in f["E_cost"]:
            fh.write(f"E,ratio@{r['n_tokens']},{r['ratio']}\n")
    print(f"  wrote {fpath}")
    plot(f)
    print(f"done in {f['wall_clock_s']}s")


if __name__ == "__main__":
    main()
