"""Project 03 — Stop-string matcher.

  A. hand-picked pathological cases   — the ones that break naive code
  B. fuzz against an oracle           — 4000 random cases on a real tokenizer
  C. the hold-back bound              — how much text must be delayed, and why
  D. cost                             — incremental matching vs rescanning
  E. in a real generation             — decode steps saved by stopping early

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

from matcher import (StreamingStopMatcher, eager_scan_matcher,  # noqa: E402
                     naive_token_matcher, oracle, run_streaming)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


# ---------------------------------------------------------------------------
# A. Cases chosen by hand, because a fuzzer finds them slowly and a reader
#    understands them instantly.
# ---------------------------------------------------------------------------
CASES = [
    {"name": "stop inside one token",
     "pieces": ["Answer", ": 42<|end|>", " more"], "stops": ["<|end|>"]},
    {"name": "stop split over two tokens",
     "pieces": ["Answer", ": 42<|", "end|>", " more"], "stops": ["<|end|>"]},
    {"name": "stop split over four tokens",
     "pieces": ["ok", "<", "|", "end", "|>", "tail"], "stops": ["<|end|>"]},
    {"name": "double newline split across tokens",
     "pieces": ["line one", "\n", "\n", "line two"], "stops": ["\n\n"]},
    {"name": "false start that never completes",
     "pieces": ["a<", "b", "c"], "stops": ["<|end|>"]},
    {"name": "two stops, the later one is shorter",
     "pieces": ["hello ", "wor", "ld STOP tail"], "stops": ["STOP", "world"]},
    {"name": "stop is the very first thing emitted",
     "pieces": ["STO", "P", "anything"], "stops": ["STOP"]},
    {"name": "overlapping near-miss then a real hit",
     "pieces": ["aa", "ab", "aab", "x"], "stops": ["aab"]},
]


def section_a(findings):
    rows = []
    for c in CASES:
        exp_text, exp_hit = oracle(c["pieces"], c["stops"])
        n_text, n_hit = naive_token_matcher(c["pieces"], c["stops"])
        e_text, e_hit = eager_scan_matcher(c["pieces"], c["stops"])
        s_text, s_hit, s_tokens, s_hold = run_streaming(c["pieces"], c["stops"])
        rows.append({
            "name": c["name"],
            "pieces": c["pieces"],
            "stops": c["stops"],
            "expected": exp_text,
            "naive_ok": (n_text, n_hit) == (exp_text, exp_hit),
            "naive_text": n_text,
            "eager_ok": (e_text, e_hit) == (exp_text, exp_hit),
            "eager_leaked_chars": max(0, len(e_text) - len(exp_text)),
            "streaming_ok": (s_text, s_hit) == (exp_text, exp_hit),
            "streaming_max_hold": s_hold,
        })
        print(f"  A: {c['name']:38s} naive={rows[-1]['naive_ok']!s:5s} "
              f"eager={rows[-1]['eager_ok']!s:5s} "
              f"streaming={rows[-1]['streaming_ok']!s:5s}")
    findings["A_cases"] = rows
    findings["A_summary"] = {
        "n": len(rows),
        "naive_passed": sum(r["naive_ok"] for r in rows),
        "eager_passed": sum(r["eager_ok"] for r in rows),
        "streaming_passed": sum(r["streaming_ok"] for r in rows),
    }


# ---------------------------------------------------------------------------
# B / C. Fuzz on a real tokenizer's vocabulary.
# ---------------------------------------------------------------------------


def ascii_vocab(tok, limit=8000):
    """Token ids whose text is plain ASCII, so per-token concatenation is exact.

    Multi-byte characters split across tokens are a real and separate problem
    -- that is project 05's subject. Excluding them here keeps this project
    about stop strings only.
    """
    ids = []
    for i in range(min(limit, tok.vocab_size)):
        s = tok.decode([i])
        if s and s.isascii() and s.isprintable() or s in ("\n", "\n\n", " "):
            ids.append(i)
    return ids


def make_case(tok, ids, rng):
    n = rng.randint(4, 24)
    seq = [rng.choice(ids) for _ in range(n)]
    pieces = [tok.decode([i]) for i in seq]
    full = "".join(pieces)

    stops = []
    mode = rng.random()
    if mode < 0.6 and len(full) > 6:
        # A stop string carved out of the text itself, deliberately positioned
        # to straddle a token boundary where possible.
        bounds = []
        acc = 0
        for p in pieces[:-1]:
            acc += len(p)
            bounds.append(acc)
        if bounds:
            b = rng.choice(bounds)
            left = rng.randint(1, min(3, b))
            right = rng.randint(1, min(3, max(1, len(full) - b)))
            s = full[b - left:b + right]
            if s:
                stops.append(s)
    if not stops or rng.random() < 0.3:
        stops.append("".join(rng.choice("abcde \n") for _ in range(rng.randint(2, 4))))
    if rng.random() < 0.25:
        stops.append("".join(rng.choice("xyz ") for _ in range(rng.randint(2, 3))))
    return pieces, [s for s in stops if s]


def section_bc(tok, findings, n_cases=4000, seed=0):
    rng = random.Random(seed)
    ids = ascii_vocab(tok)
    stats = {"n_cases": n_cases, "vocab_sampled": len(ids),
             "cases_with_a_hit": 0,
             "naive_wrong": 0, "naive_missed_hit": 0,
             "eager_wrong": 0, "eager_leak_chars_total": 0,
             "eager_max_leak": 0,
             "streaming_wrong": 0, "streaming_max_hold": 0,
             "bound_violations": 0,
             "hits_within_one_token": 0, "hits_straddling": 0,
             "naive_missed_within": 0, "naive_missed_straddling": 0}
    examples = []
    for _ in range(n_cases):
        pieces, stops = make_case(tok, ids, rng)
        exp_text, exp_hit = oracle(pieces, stops)
        stats["cases_with_a_hit"] += int(exp_hit)

        # Does the winning stop string fit inside a single token's text?
        within = exp_hit and any(s in p for p in pieces for s in stops)
        if exp_hit:
            stats["hits_within_one_token" if within else "hits_straddling"] += 1

        n_text, n_hit = naive_token_matcher(pieces, stops)
        if (n_text, n_hit) != (exp_text, exp_hit):
            stats["naive_wrong"] += 1
            if exp_hit and not n_hit:
                stats["naive_missed_hit"] += 1
                stats["naive_missed_within" if within
                      else "naive_missed_straddling"] += 1
                if len(examples) < 3:
                    examples.append({"pieces": pieces, "stops": stops,
                                     "expected": exp_text, "naive_gave": n_text})

        e_text, e_hit = eager_scan_matcher(pieces, stops)
        if (e_text, e_hit) != (exp_text, exp_hit):
            stats["eager_wrong"] += 1
        leak = max(0, len(e_text) - len(exp_text)) if exp_hit else 0
        stats["eager_leak_chars_total"] += leak
        stats["eager_max_leak"] = max(stats["eager_max_leak"], leak)

        s_text, s_hit, _, s_hold = run_streaming(pieces, stops)
        if (s_text, s_hit) != (exp_text, exp_hit):
            stats["streaming_wrong"] += 1
        stats["streaming_max_hold"] = max(stats["streaming_max_hold"], s_hold)
        bound = max(len(s) for s in stops) - 1
        if s_hold > bound:
            stats["bound_violations"] += 1

    hits = max(1, stats["cases_with_a_hit"])
    stats["naive_detection_rate"] = round(
        1 - stats["naive_missed_hit"] / hits, 4)
    stats["naive_detection_rate_within_one_token"] = round(
        1 - stats["naive_missed_within"] / max(1, stats["hits_within_one_token"]), 4)
    stats["naive_detection_rate_straddling"] = round(
        1 - stats["naive_missed_straddling"] / max(1, stats["hits_straddling"]), 4)
    stats["eager_mean_leak_chars"] = round(
        stats["eager_leak_chars_total"] / hits, 2)
    stats["streaming_accuracy"] = round(1 - stats["streaming_wrong"] / n_cases, 4)
    findings["B_fuzz"] = stats
    findings["B_examples"] = examples
    print(f"  B: {n_cases} cases, {stats['cases_with_a_hit']} contain a stop")
    print(f"     naive      : detected {100 * stats['naive_detection_rate']:.1f}% "
          f"of real hits, wrong output in {stats['naive_wrong']} cases")
    print(f"     eager scan : always detects, leaks "
          f"{stats['eager_mean_leak_chars']} chars past the stop on average "
          f"(max {stats['eager_max_leak']})")
    print(f"     streaming  : {100 * stats['streaming_accuracy']:.2f}% exact, "
          f"max hold {stats['streaming_max_hold']} chars, "
          f"{stats['bound_violations']} bound violations")


# ---------------------------------------------------------------------------
# D. Cost per token.
# ---------------------------------------------------------------------------


def section_d(findings, lengths=(1000, 2000, 4000, 8000, 16000), reps=3):
    """Both matchers are cheap next to a decode step; only one stays cheap."""
    stops = ["<|end_of_turn|>"]
    rows = []
    for n_tokens in lengths:
        pieces = ["token%d " % i for i in range(n_tokens)]

        def incremental():
            m = StreamingStopMatcher(stops)
            for p in pieces:
                m.push(p)

        def rescan():
            buf = ""
            for p in pieces:
                buf += p
                for s in stops:
                    buf.find(s)      # searches the WHOLE text every token

        best = {"incremental": float("inf"), "rescan": float("inf")}
        for _ in range(reps):
            for name, fn in (("incremental", incremental), ("rescan", rescan)):
                t0 = time.perf_counter()
                fn()
                best[name] = min(best[name], time.perf_counter() - t0)
        rows.append({
            "n_tokens": n_tokens,
            "incremental_s": round(best["incremental"], 4),
            "rescan_s": round(best["rescan"], 4),
            "rescan_over_incremental": round(best["rescan"] / best["incremental"], 2),
            "incremental_us_per_token": round(1e6 * best["incremental"] / n_tokens, 2),
            "rescan_us_per_token": round(1e6 * best["rescan"] / n_tokens, 2)})
        print(f"  D: {n_tokens:6d} tokens — incremental "
              f"{rows[-1]['incremental_us_per_token']:6.2f} us/tok, rescan "
              f"{rows[-1]['rescan_us_per_token']:7.2f} us/tok "
              f"({rows[-1]['rescan_over_incremental']}x)")
    findings["D_cost"] = rows


# ---------------------------------------------------------------------------
# E. On a real generation.
# ---------------------------------------------------------------------------


def tokens_consumed(pieces, stops, matcher):
    """How many tokens a server using `matcher` would have decoded."""
    if matcher == "streaming":
        _, stopped, used, _ = run_streaming(pieces, stops)
        return used, stopped
    for i in range(1, len(pieces) + 1):
        text, hit = naive_token_matcher(pieces[:i], stops)
        if hit:
            return i, True
    return len(pieces), False


def section_e(findings, max_new=48, stops=("\n4.",)):
    """A real generation whose stop string really does straddle three tokens.

    The prompt starts a numbered list at 3, so the model writes
    '3', '.', ' blue', '\\n', '4', '.', ... and the user's stop string
    '\\n4.' lands across the 4th, 5th and 6th tokens.
    """
    import loop_lib as L
    tok, model = L.load(MODEL_ID)
    prompt = ("List three colours, one per line, then stop.\n"
              "1. red\n2. green\n")
    ids = tok(prompt, return_tensors="pt").input_ids

    res = L.generate_with_cache(model, ids, max_new_tokens=max_new,
                                eos_id=tok.eos_token_id)
    pieces = [tok.decode([t]) for t in res.token_ids]
    per_step = res.median_itl_s

    s_used, s_stopped = tokens_consumed(pieces, list(stops), "streaming")
    n_used, n_stopped = tokens_consumed(pieces, list(stops), "naive")
    delivered, _, _, hold = run_streaming(pieces, list(stops))

    findings["E_real_generation"] = {
        "prompt": prompt,
        "stops": list(stops),
        "max_new_tokens": max_new,
        "first_tokens": pieces[:8],
        "streaming_tokens_used": s_used,
        "streaming_stopped": s_stopped,
        "naive_tokens_used": n_used,
        "naive_stopped": n_stopped,
        "text_delivered": delivered,
        "full_text_if_never_stopped": "".join(pieces),
        "decode_steps_saved": n_used - s_used,
        "median_decode_step_s": round(per_step, 4),
        "seconds_saved": round((n_used - s_used) * per_step, 2),
    }
    print(f"  E: tokens {pieces[:8]}")
    print(f"     streaming matcher stops at token {s_used}; naive runs to "
          f"{n_used} (stopped={n_stopped}) — {n_used - s_used} decode steps "
          f"wasted, ~{(n_used - s_used) * per_step:.1f} s")
    print(f"     delivered {delivered!r}")


# ---------------------------------------------------------------------------


def plot(f):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    a = f["A_summary"]
    names = ["naive\n(per token)", "eager scan\n(emit then search)",
             "streaming\n(hold-back)"]
    vals = [a["naive_passed"], a["eager_passed"], a["streaming_passed"]]
    ax[0].bar(names, vals, color=["tab:red", "tab:orange", "tab:green"])
    ax[0].axhline(a["n"], ls="--", color="grey")
    ax[0].set_ylabel(f"hand-picked cases passed (of {a['n']})")
    ax[0].set_title("A. the cases a reader can check")
    ax[0].grid(alpha=.3, axis="y")
    for i, v in enumerate(vals):
        ax[0].text(i, v + .1, str(v), ha="center")

    b = f["B_fuzz"]
    labels = ["naive:\nmissed stops\n(% of hits)", "eager:\nleaked text\n(chars)",
              "streaming:\nwrong output\n(% of cases)"]
    vals = [100 * (1 - b["naive_detection_rate"]),
            b["eager_mean_leak_chars"],
            100 * (1 - b["streaming_accuracy"])]
    colors = ["tab:red", "tab:orange", "tab:green"]
    ax[1].bar(labels, vals, color=colors)
    ax[1].set_title(f"B. {b['n_cases']} fuzz cases on a real tokenizer\n"
                    "(left/right: % of cases; middle: mean chars)")
    ax[1].grid(alpha=.3, axis="y")
    for i, v in enumerate(vals):
        ax[1].text(i, v + .05, f"{v:.2f}", ha="center")

    d = f["D_cost"]
    ax[2].plot([r["n_tokens"] for r in d],
               [r["incremental_us_per_token"] for r in d], "o-",
               color="tab:green", label="incremental matcher")
    ax[2].plot([r["n_tokens"] for r in d],
               [r["rescan_us_per_token"] for r in d], "s--",
               color="tab:red", label="rescan the whole buffer")
    ax[2].set_xscale("log", base=2)
    ax[2].set_yscale("log")
    ax[2].set_xlabel("tokens generated so far")
    ax[2].set_ylabel("microseconds per token")
    ax[2].set_title(f"D. per-token cost grows with length\n"
                    f"only for the rescanner ({d[-1]['rescan_over_incremental']}x "
                    f"at {d[-1]['n_tokens']} tokens)")
    ax[2].legend()
    ax[2].grid(alpha=.3, which="both")
    fig.tight_layout()
    p = os.path.join(OUT, "stopstring.png")
    fig.savefig(p, dpi=110)
    print(f"  wrote {p}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, "findings.json")
    if "--plot" in sys.argv:
        plot(json.load(open(fpath)))
        return

    t0 = time.time()
    findings = {"model": MODEL_ID}
    section_a(findings)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    section_bc(tok, findings)
    section_d(findings)
    section_e(findings)

    findings["wall_clock_s"] = round(time.time() - t0, 1)
    json.dump(findings, open(fpath, "w"), indent=2)
    with open(os.path.join(OUT, "findings.csv"), "w") as fh:
        fh.write("section,key,value\n")
        a, b, d = findings["A_summary"], findings["B_fuzz"], findings["D_cost"]
        for k in ("naive_passed", "eager_passed", "streaming_passed", "n"):
            fh.write(f"A,{k},{a[k]}\n")
        for k in ("naive_detection_rate", "eager_mean_leak_chars",
                  "eager_max_leak", "streaming_accuracy", "streaming_max_hold",
                  "bound_violations"):
            fh.write(f"B,{k},{b[k]}\n")
        for r in d:
            fh.write(f"D,rescan_over_incremental@{r['n_tokens']},"
                     f"{r['rescan_over_incremental']}\n")
        e = findings["E_real_generation"]
        fh.write(f"E,streaming_tokens_used,{e['streaming_tokens_used']}\n")
        fh.write(f"E,naive_tokens_used,{e['naive_tokens_used']}\n")
        fh.write(f"E,decode_steps_saved,{e['decode_steps_saved']}\n")
    print(f"  wrote {fpath}")
    plot(findings)
    print(f"done in {findings['wall_clock_s']}s")


if __name__ == "__main__":
    main()
