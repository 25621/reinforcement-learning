#!/usr/bin/env python3
"""Project 67 — serving a reasoning model.

  A. How long does it think?  Real output-length distributions for the same 24
     problems in chat mode and in thinking mode, on a real hybrid reasoning
     model (Qwen3-0.6B).
  B. What that variance does to a queue.  The measured length distributions are
     replayed through the phase's shared engine simulator, so the difference in
     tail latency comes only from the lengths.
  C. The thinking-budget knob.  Cut the thinking off at B tokens, force the
     answer, and measure what accuracy that buys or loses.
  D. The budget as a serving control: capacity, tail latency, and cost per
     correct answer.

  python3 run.py            # a long run: ~40 minutes on this shared CPU
  python3 run.py --cap 512  # ~13 minutes, and more requests get cut off
  python3 run.py --reuse    # reuse outputs/raw.json, redo B/C/D analysis only
  python3 run.py --plot     # redraw the figure from outputs/findings.json

The whole cost is generation: ~19,000 real tokens from a real reasoning model on
a CPU.  `--cap` lowers the generation limit, which is faster and *changes the
result* — section A's length distribution is censored by exactly this number,
and section C's accuracy falls because more thoughts are interrupted.  The
committed findings use the default 1024.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECTS = os.path.dirname(HERE)
for d in ("18-chunked-prefill-simulator", "59-metric-instrumentation"):
    sys.path.insert(0, os.path.join(PROJECTS, d))
sys.path.insert(0, HERE)

import reasonlib as R                                          # noqa: E402

OUT = os.path.join(HERE, "outputs")
RAW = os.path.join(OUT, "raw.json")
FINDINGS = os.path.join(OUT, "findings.json")
BUDGETS = [0, 64, 256, 512]            # 0 = "do not think at all"
BATCH = 15


# ---------------------------------------------------------------------------
# A + C: the real generations
# ---------------------------------------------------------------------------

def measure():
    print("loading Qwen3-0.6B ...", flush=True)
    t0 = time.time()
    tok, model = R.load()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    tiers = [t for t, _, _ in R.PROBLEMS]
    qs = [q for _, q, _ in R.PROBLEMS]
    ans = [a for _, _, a in R.PROBLEMS]

    chat_p = [R.prompt_ids(tok, q, thinking=False) for q in qs]
    think_p = [R.prompt_ids(tok, q, thinking=True) for q in qs]
    brief_p = [R.prompt_ids(tok, q, thinking=True, system=R.BRIEF) for q in qs]

    print(f"arm 1/3: chat mode (thinking off), cap {R.CHAT_CAP} tokens ...",
          flush=True)
    chat_out, chat_s = R.batch_generate(model, tok, chat_p, R.CHAT_CAP,
                                        batch=BATCH)
    print(f"  {chat_s:.1f}s", flush=True)

    print(f"arm 2/3: thinking mode, cap {R.MAX_THINK} tokens ...", flush=True)
    think_out, think_s = R.batch_generate(model, tok, think_p, R.MAX_THINK,
                                          batch=BATCH)
    print(f"  {think_s:.1f}s", flush=True)

    print("arm 3/3: thinking mode, *asked* to be brief ...", flush=True)
    brief_out, brief_s = R.batch_generate(model, tok, brief_p, R.MAX_THINK,
                                          batch=BATCH)
    print(f"  {brief_s:.1f}s", flush=True)

    rows = []
    for i, (tier, q, a) in enumerate(R.PROBLEMS):
        c_txt = tok.decode(chat_out[i])
        th, ansr, closed = R.think_split(tok, think_out[i])
        bth, bans, bclosed = R.think_split(tok, brief_out[i])
        rows.append(dict(
            idx=i, tier=tier, question=q, answer=a,
            prompt_tokens=len(think_p[i]),
            chat_tokens=len(chat_out[i]), chat_ok=R.correct(c_txt, a),
            chat_text=c_txt,
            think_tokens=len(th), answer_tokens=len(ansr), closed=closed,
            total_tokens=len(think_out[i]),
            think_ok=R.correct(tok.decode(ansr), a) if closed else False,
            think_ids=th, answer_text=tok.decode(ansr),
            brief_think=len(bth), brief_answer=len(bans),
            brief_closed=bclosed, brief_total=len(brief_out[i]),
            brief_ok=R.correct(tok.decode(bans), a) if bclosed else False,
        ))

    # --- C: the same traces, cut short at each budget ----------------------
    budget_rows = {}
    for b in BUDGETS:
        prompts = [R.budget_prompt(tok, think_p[i], rows[i]["think_ids"], b)
                   for i in range(len(rows))]
        print(f"budget {b:>3} tokens: forcing the answer ...", end="", flush=True)
        outs, secs = R.batch_generate(model, tok, prompts, 96, batch=BATCH)
        got = []
        for i, o in enumerate(outs):
            txt = tok.decode(o)
            got.append(dict(idx=i, ok=R.correct(txt, ans[i]),
                            answer_tokens=len(o),
                            think_used=min(b, rows[i]["think_tokens"]),
                            text=txt[:200]))
        budget_rows[str(b)] = got
        acc = 100.0 * sum(g["ok"] for g in got) / len(got)
        print(f" {secs:5.1f}s  accuracy {acc:5.1f}%", flush=True)

    for r in rows:                       # ids were only needed for the budgets
        r.pop("think_ids")
    raw = dict(rows=rows, budgets=budget_rows, chat_seconds=chat_s,
               think_seconds=think_s, brief_seconds=brief_s, tiers=tiers)
    os.makedirs(OUT, exist_ok=True)
    with open(RAW, "w") as f:
        json.dump(raw, f, indent=1)
    return raw


# ---------------------------------------------------------------------------
# B + D: what those lengths do to a queue
# ---------------------------------------------------------------------------

def queue_study(raw):
    import obslib
    from simlib import SimRequest, simulate, report, pct

    cost, src = obslib.load_cost_model()
    rows = raw["rows"]
    prompt_len = int(statistics.mean(r["prompt_tokens"] for r in rows))

    pools = {
        "chat": [r["chat_tokens"] for r in rows],
        "reasoning": [r["total_tokens"] for r in rows],
        "asked-to-be-brief": [r["brief_total"] for r in rows],
    }
    for b in BUDGETS[1:]:
        pools[f"budget-{b}"] = [
            min(b, r["think_tokens"]) + g["answer_tokens"]
            for r, g in zip(rows, raw["budgets"][str(b)])]

    # One arrival rate for every arm: the load the *chat* workload would put on
    # the engine at 60% utilisation.  Holding the rate fixed is the point --
    # the same users asking the same questions, one model that thinks.
    mean_chat = statistics.mean(
        cost.request_work(prompt_len, o) for o in pools["chat"])
    rate = 0.60 / mean_chat

    out = {}
    for name, pool in pools.items():
        rng = random.Random(7)
        reqs, t = [], 0.0
        for i in range(400):
            t += rng.expovariate(rate)
            reqs.append(SimRequest(rid=i, arrive=t, prompt_len=prompt_len,
                                   out_len=rng.choice(pool)))
        stats = simulate(reqs, cost, chunk=256, max_running=24)
        rep = report(reqs, stats, name)
        work = statistics.mean(cost.request_work(prompt_len, o) for o in pool)
        rep["util"] = round(rate * work, 3)
        rep["mean_out"] = round(statistics.mean(pool), 1)
        rep["p99_out"] = R.pct(pool, 99)
        rep["e2e_p50"] = round(pct([r.e2e for r in reqs
                                    if r.end_t is not None], 50), 2)
        out[name] = rep
    out["_rate"] = round(rate, 4)
    out["_prompt_len"] = prompt_len
    out["_cost_src"] = src
    out["_cost"] = dict(base=cost.base, per_decode=cost.per_decode,
                        per_prefill=cost.per_prefill,
                        per_key_read=cost.per_key_read)
    return out


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------

def analyse(raw):
    rows = raw["rows"]
    n = len(rows)

    def stats_of(xs):
        return dict(mean=round(statistics.mean(xs), 1),
                    p50=R.pct(xs, 50), p95=R.pct(xs, 95), max=max(xs),
                    min=min(xs),
                    cv=round(statistics.pstdev(xs) / statistics.mean(xs), 3),
                    ratio_p95_p50=round(R.pct(xs, 95) / max(R.pct(xs, 50), 1),
                                        2))

    chat_len = [r["chat_tokens"] for r in rows]
    think_len = [r["total_tokens"] for r in rows]
    brief_len = [r["brief_total"] for r in rows]
    A = dict(
        n=n, cap=R.MAX_THINK, chat_cap=R.CHAT_CAP,
        chat=stats_of(chat_len), reasoning=stats_of(think_len),
        brief=stats_of(brief_len),
        brief_acc=round(100.0 * sum(r["brief_ok"] for r in rows) / n, 1),
        brief_unfinished=sum(1 for r in rows if not r["brief_closed"]),
        brief_saving=round(1 - statistics.mean(brief_len)
                           / statistics.mean(think_len), 3),
        capped=sum(1 for r in rows if r["total_tokens"] >= R.MAX_THINK - 1),
        chat_acc=round(100.0 * sum(r["chat_ok"] for r in rows) / n, 1),
        think_acc=round(100.0 * sum(r["think_ok"] for r in rows) / n, 1),
        unfinished=sum(1 for r in rows if not r["closed"]),
        length_x=round(statistics.mean(think_len) / statistics.mean(chat_len),
                       2),
        by_tier={}, per_problem=[
            dict(idx=r["idx"], tier=r["tier"], chat=r["chat_tokens"],
                 think=r["think_tokens"], total=r["total_tokens"],
                 closed=r["closed"], chat_ok=r["chat_ok"],
                 think_ok=r["think_ok"]) for r in rows],
    )
    for tier in ("easy", "medium", "hard"):
        sub = [r for r in rows if r["tier"] == tier]
        A["by_tier"][tier] = dict(
            n=len(sub),
            mean_think=round(statistics.mean(r["think_tokens"] for r in sub), 1),
            max_think=max(r["think_tokens"] for r in sub),
            chat_acc=round(100.0 * sum(r["chat_ok"] for r in sub) / len(sub), 1),
            think_acc=round(100.0 * sum(r["think_ok"] for r in sub) / len(sub),
                            1),
            unfinished=sum(1 for r in sub if not r["closed"]))

    # --- C: the budget curve ----------------------------------------------
    C = {}
    for b in BUDGETS:
        got = raw["budgets"][str(b)]
        used = [g["think_used"] for g in got]
        tot = [g["think_used"] + g["answer_tokens"] for g in got]
        C[str(b)] = dict(
            budget=b,
            acc=round(100.0 * sum(g["ok"] for g in got) / n, 1),
            mean_think=round(statistics.mean(used), 1),
            mean_total=round(statistics.mean(tot), 1),
            p95_total=R.pct(tot, 95),
            truncated=sum(1 for r, g in zip(rows, got)
                          if r["think_tokens"] > b),
            by_tier={t: round(100.0 * sum(g["ok"] for g, r in zip(got, rows)
                                          if r["tier"] == t)
                              / sum(1 for r in rows if r["tier"] == t), 1)
                     for t in ("easy", "medium", "hard")},
        )
    C["full"] = dict(budget=R.MAX_THINK,
                     acc=A["think_acc"],
                     mean_think=round(statistics.mean(
                         r["think_tokens"] for r in rows), 1),
                     mean_total=round(statistics.mean(think_len), 1),
                     p95_total=R.pct(think_len, 95),
                     truncated=0,
                     by_tier={t: A["by_tier"][t]["think_acc"]
                              for t in ("easy", "medium", "hard")})
    return A, C


def cost_per_correct(A, C, Q):
    """Dollars per 1000 correct answers, using project 63's formula.

    The two prompt-level arms (chat and "asked to be brief") are priced with
    the same formula so that every way of spending fewer thinking tokens can be
    compared on one axis.
    """
    # project 63: $/M tokens = price_per_hour / (tokens_per_second * 3600) * 1e6
    # with the duty cycle and overhead it committed to.
    price_hr, duty, overhead = 0.55, 0.50, 1.25
    rows = dict(C)
    rows["chat (no thinking)"] = dict(budget=-1, mean_total=A["chat"]["mean"],
                                      acc=A["chat_acc"])
    rows["asked-to-be-brief"] = dict(budget=-2, mean_total=A["brief"]["mean"],
                                     acc=A["brief_acc"])
    out = {}
    for key, row in rows.items():
        toks = row["mean_total"]
        # engine seconds per request from the shared cost model
        c = Q["_cost"]
        secs = (c["per_prefill"] * Q["_prompt_len"]
                + toks * (c["per_decode"] + c["per_key_read"]
                          * (Q["_prompt_len"] + toks / 2)))
        dollars = price_hr / 3600.0 * secs / duty * overhead
        n_correct = max(row["acc"], 1e-9) / 100.0
        out[key] = dict(tokens=toks, engine_s=round(secs, 3),
                        usd_per_1k_requests=round(dollars * 1000, 3),
                        usd_per_1k_correct=round(dollars * 1000 / n_correct, 3))
    return out


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

def plot(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, C, Q = F["A"], F["C"], F["B"]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.2))
    fig.suptitle(f"Serving a reasoning model: {A['n']} problems, one 0.6B "
                 "hybrid model, thinking on and off", fontsize=13)

    # panel 1: per-problem output length, sorted
    p = ax[0][0]
    rows = sorted(A["per_problem"], key=lambda r: r["total"])
    xs = range(len(rows))
    colors = {"easy": "#4c9f70", "medium": "#e0a458", "hard": "#c0504d"}
    p.bar(xs, [r["total"] for r in rows],
          color=[colors[r["tier"]] for r in rows], label=None)
    p.bar(xs, [r["chat"] for r in rows], color="#4a6fa5", width=0.45,
          label="chat mode (thinking off)")
    p.axhline(A["reasoning"]["mean"], color="k", ls="--", lw=1,
              label=f"reasoning mean {A['reasoning']['mean']:.0f}")
    from matplotlib.patches import Patch
    handles = [Patch(color="#4a6fa5", label="chat mode (thinking off)")]
    handles += [Patch(color=c, label=f"thinking, {t}")
                for t, c in colors.items()]
    p.set_title(f"A. Same {A['n']} questions: {A['length_x']}x the tokens, "
                f"and the cap is doing the work")
    p.axhline(A["cap"], color="#c0504d", ls=":", lw=1.2)
    p.text(0.2, A["cap"], f" cap {A['cap']}", fontsize=7, va="bottom",
           color="#c0504d")
    p.set_xlabel("problem, sorted by output length")
    p.set_ylabel("output tokens")
    p.legend(handles=handles + [
        plt.Line2D([], [], color="k", ls="--",
                   label=f"reasoning mean {A['reasoning']['mean']:.0f}")],
        fontsize=8, loc="upper left")

    # panel 2: queue effect
    p = ax[0][1]
    names = (["chat", "reasoning", "asked-to-be-brief"]
             + [f"budget-{b}" for b in BUDGETS[1:]])
    names = [n for n in names if n in Q]
    ttft = [Q[n]["ttft_p99"] for n in names]
    util = [Q[n]["util"] for n in names]
    bars = p.bar(range(len(names)), ttft,
                 color=["#4a6fa5", "#c0504d"] + ["#8d9db6"] * (len(names) - 2))
    for i, (b, u) in enumerate(zip(bars, util)):
        p.text(i, b.get_height(), f" util {u:.2f}", ha="center", va="bottom",
               fontsize=8, rotation=90)
    p.set_xticks(range(len(names)))
    p.set_xticklabels(names, rotation=20, fontsize=8)
    p.set_yscale("log")
    p.set_title(f"B. Same users, same arrival rate ({Q['_rate']:.2f} req/s):\n"
                "p99 time-to-first-token")
    p.set_ylabel("TTFT p99 (s, log scale)")

    # panel 3: budget curve
    p = ax[1][0]
    ks = [str(b) for b in BUDGETS] + ["full"]
    xs = [C[k]["mean_total"] for k in ks]
    ys = [C[k]["acc"] for k in ks]
    p.plot(xs, ys, "o-", color="#4a6fa5", label="hard cap (interrupted)")
    for k, x, y in zip(ks, xs, ys):
        p.annotate(k, (x, y), textcoords="offset points", xytext=(4, 5),
                   fontsize=8)
    p.scatter([A["brief"]["mean"]], [A["brief_acc"]], marker="*", s=220,
              color="#4c9f70", zorder=5,
              label=f"asked to be brief: {A['brief_acc']}% at "
                    f"{A['brief']['mean']:.0f} tokens")
    p.scatter([A["chat"]["mean"]], [A["chat_acc"]], marker="s", s=70,
              color="#c0504d", zorder=5,
              label=f"no thinking: {A['chat_acc']}%")
    p.set_title("C. What each thinking token buys — and who spends it")
    p.set_xlabel("mean output tokens per request")
    p.set_ylabel("accuracy (%)")
    p.legend(fontsize=8)
    p.grid(alpha=0.3)

    # panel 4: cost per correct answer
    p = ax[1][1]
    D = F["D"]
    ks2 = [k for k in ks if k in D] + ["chat (no thinking)",
                                       "asked-to-be-brief"]
    ks2 = [k for k in ks2 if k in D]
    v1 = [D[k]["usd_per_1k_requests"] for k in ks2]
    v2 = [D[k]["usd_per_1k_correct"] for k in ks2]
    w = 0.4
    p.bar([i - w / 2 for i in range(len(ks2))], v1, w, label="$ / 1k requests",
          color="#8d9db6")
    p.bar([i + w / 2 for i in range(len(ks2))], v2, w,
          label="$ / 1k correct answers", color="#4a6fa5")
    p.set_xticks(range(len(ks2)))
    p.set_xticklabels([k.replace(" (no thinking)", "\nno think")
                       .replace("asked-to-be-brief", "asked\nbrief")
                       for k in ks2], fontsize=7)
    p.set_xlabel("thinking budget (tokens), then the two prompt-level arms")
    p.set_title("D. The bill: thinking tokens are billed like any other")
    p.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = os.path.join(OUT, "reasoning.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse outputs/raw.json instead of running the model")
    ap.add_argument("--plot", action="store_true",
                    help="redraw the figure from outputs/findings.json")
    ap.add_argument("--cap", type=int, default=R.MAX_THINK,
                    help="generation limit for the thinking arms "
                         "(default %(default)s; lower is faster and censors "
                         "more of the distribution)")
    args = ap.parse_args()
    R.MAX_THINK = args.cap
    BUDGETS[:] = [b for b in BUDGETS if b < args.cap] or [0]

    if args.plot:
        with open(FINDINGS) as f:
            plot(json.load(f))
        return

    if args.reuse and os.path.exists(RAW):
        with open(RAW) as f:
            raw = json.load(f)
        print("reusing", RAW)
    else:
        raw = measure()

    A, C = analyse(raw)
    Q = queue_study(raw)
    D = cost_per_correct(A, C, Q)
    F = dict(A=A, B=Q, C=C, D=D)
    os.makedirs(OUT, exist_ok=True)
    with open(FINDINGS, "w") as f:
        json.dump(F, f, indent=1)

    print("\n--- A. output length -------------------------------------------")
    print(f"chat      mean {A['chat']['mean']:6.1f}  p50 {A['chat']['p50']:5.0f}"
          f"  p95 {A['chat']['p95']:5.0f}  max {A['chat']['max']:5.0f}"
          f"  CV {A['chat']['cv']}")
    print(f"reasoning mean {A['reasoning']['mean']:6.1f}  "
          f"p50 {A['reasoning']['p50']:5.0f}  p95 {A['reasoning']['p95']:5.0f}"
          f"  max {A['reasoning']['max']:5.0f}  CV {A['reasoning']['cv']}")
    print(f"brief     mean {A['brief']['mean']:6.1f}  "
          f"p50 {A['brief']['p50']:5.0f}  p95 {A['brief']['p95']:5.0f}  "
          f"max {A['brief']['max']:5.0f}  CV {A['brief']['cv']}")
    print(f"accuracy  chat {A['chat_acc']}%   thinking {A['think_acc']}%   "
          f"brief {A['brief_acc']}%   unfinished thoughts "
          f"{A['unfinished']}/{A['n']} (brief {A['brief_unfinished']}), "
          f"hit the {A['cap']}-token cap {A['capped']}/{A['n']}")
    print("\n--- B. the same traffic through one engine ---------------------")
    for k in (["chat", "reasoning", "asked-to-be-brief"]
              + [f"budget-{b}" for b in BUDGETS[1:]]):
        if k in Q:
            r = Q[k]
            print(f"{k:<12} util {r['util']:.2f}  ttft p50 {r['ttft_p50']:7.2f}"
                  f"  p99 {r['ttft_p99']:8.2f}  e2e p99 {r['e2e_p99']:8.2f}"
                  f"  out tok/s {r['output_tok_s']:.1f}")
    print("\n--- C. the thinking budget -------------------------------------")
    for k in [str(b) for b in BUDGETS] + ["full"]:
        r = C[k]
        print(f"budget {k:>5}: acc {r['acc']:5.1f}%  mean tokens "
              f"{r['mean_total']:6.1f}  truncated {r['truncated']:2d}/{A['n']}"
              f"  easy/med/hard {r['by_tier']['easy']:.0f}/"
              f"{r['by_tier']['medium']:.0f}/{r['by_tier']['hard']:.0f}")
    print("\n--- D. cost ----------------------------------------------------")
    for k, v in D.items():
        print(f"budget {k:>5}: ${v['usd_per_1k_requests']:7.3f}/1k requests"
              f"   ${v['usd_per_1k_correct']:8.3f}/1k correct")
    plot(F)


if __name__ == "__main__":
    main()
