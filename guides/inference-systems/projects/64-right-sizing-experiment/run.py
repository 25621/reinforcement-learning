"""Project 64 -- Right-sizing: four models, one real eval, one bill.

The guide's version of this experiment is "7B vs 13B vs 70B". None of those
fit on this machine, so the ladder is scaled down but kept honest: four real
instruct models spanning **11x in parameter count**, two families, evaluated
on a task with a checkable right answer, and priced with the same formula
project 63 used.

The fifth and sixth arms are the ones that make it a *right-sizing* study
rather than a size sweep: the two smallest models are run again with
constrained decoding (project 53's grammar), which is what "invest in the
small model instead of buying the big one" actually looks like in practice.

  A. Quality: schema validity and exact-extraction accuracy, per model.
  B. Cost: measured tokens/s, dollars per million output tokens, memory.
  C. The frontier: dollars per 1,000 *correct* answers -- the only number
     that combines the two.
  D. Where each model actually fails, field by field.

Usage:
    python3 run.py            # ~8 minutes
    python3 run.py --plot
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for d in ("51-needle-in-a-haystack", "53-json-mode-reliability"):
    sys.path.insert(0, os.path.join(PROJ, d))

import ctxlib                                                # noqa: E402
import gramlib                                               # noqa: E402
from jsontask import INSTRUCTION, PATTERN, SYSTEM, make_cases  # noqa: E402

OUT = os.path.join(HERE, "outputs")

N_CASES = 32
BATCH = 8
MAX_NEW = 44
PRICE_HR = 0.55        # same illustrative price as project 63
OVERHEAD, DUTY = 0.25, 0.50

LADDER = [
    ("SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct", True),
    ("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M-Instruct", False),
    ("Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct", True),
    ("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B-Instruct", False),
]


def grade(text, case):
    """Project 53's grader, unchanged, so the numbers are comparable."""
    r = {"parse_ok": False, "schema_ok": False, "name_ok": False,
         "age_ok": False, "skills_ok": False, "all_ok": False}
    s = text.strip()
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return r
    try:
        obj = json.loads(s[i:j + 1])
    except Exception:
        return r
    r["parse_ok"] = True
    if not (isinstance(obj, dict)
            and isinstance(obj.get("name"), str)
            and isinstance(obj.get("age"), int)
            and not isinstance(obj.get("age"), bool)
            and isinstance(obj.get("skills"), list)
            and obj["skills"]
            and all(isinstance(x, str) for x in obj["skills"])):
        return r
    r["schema_ok"] = True
    r["name_ok"] = obj["name"].strip().lower() == case["name"].lower()
    r["age_ok"] = obj["age"] == case["age"]
    got = {x.strip().lower() for x in obj["skills"]}
    r["skills_ok"] = got == set(case["skills"])
    r["all_ok"] = r["name_ok"] and r["age_ok"] and r["skills_ok"]
    return r


def prompts_for(tok, cases):
    """Left-pad to a common length: the last token must be the real one."""
    raw = [ctxlib.chat_ids(tok, f"Bio: {c['bio']}\n\n{INSTRUCTION}",
                           system=SYSTEM)[0].tolist() for c in cases]
    width = max(len(r) for r in raw)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return [[pad] * (width - len(r)) + r for r in raw], width


def dollars_per_m(out_tok_s):
    return PRICE_HR * (1 + OVERHEAD) / (out_tok_s * 3600.0 * DUTY / 1e6)


@torch.inference_mode()
def run_arm(model, tok, cases, prompts, index, label):
    rows = []
    gen_tokens, gen_s, pre_s, pre_tokens = 0, 0.0, 0.0, 0
    for b0 in range(0, len(cases), BATCH):
        chunk = cases[b0:b0 + BATCH]
        ids = torch.tensor([prompts[b0 + i] for i in range(len(chunk))])
        t0 = time.perf_counter()
        outs = gramlib.generate(model, ids, MAX_NEW, temperature=0.0,
                                index=index, eos_id=tok.eos_token_id,
                                seed=1000 + b0)
        dt = time.perf_counter() - t0
        gen_s += dt
        gen_tokens += sum(len(o) for o in outs)
        pre_tokens += ids.numel()
        for c, o in zip(chunk, outs):
            g = grade(tok.decode(o, skip_special_tokens=True), c)
            g["text"] = tok.decode(o, skip_special_tokens=True)
            g["n_tokens"] = len(o)
            rows.append(g)
        print(f"    {label}: {len(rows)}/{len(cases)}  ({dt:.0f}s)", flush=True)
    n = len(rows)
    acc = sum(r["all_ok"] for r in rows) / n
    out_tok_s = gen_tokens / gen_s
    dpm = dollars_per_m(out_tok_s)
    tok_per_case = gen_tokens / n
    # dollars per 1000 CORRECT answers -- the number that combines the two
    per_correct = (tok_per_case * dpm / 1e6) / max(acc, 1e-9) * 1000
    return dict(
        label=label, n=n,
        parse=sum(r["parse_ok"] for r in rows) / n,
        schema=sum(r["schema_ok"] for r in rows) / n,
        name=sum(r["name_ok"] for r in rows) / n,
        age=sum(r["age_ok"] for r in rows) / n,
        skills=sum(r["skills_ok"] for r in rows) / n,
        accuracy=acc, gen_tokens=gen_tokens, gen_s=gen_s,
        tok_per_case=tok_per_case, out_tok_s=out_tok_s,
        dollars_per_m=dpm, dollars_per_1k_correct=per_correct,
        samples=[r["text"] for r in rows[:3]])


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    cases = make_cases(N_CASES, seed=11)
    arms, models = [], []

    for name, mid, do_grammar in LADDER:
        print(f"[{name}] loading  [{time.time()-t0:.0f}s]", flush=True)
        tok, model = ctxlib.load(mid)
        params = sum(p.numel() for p in model.parameters())
        prompts, width = prompts_for(tok, cases)
        info = dict(model=name, model_id=mid, params=params,
                    fp32_gb=params * 4 / 1e9, fp16_gb=params * 2 / 1e9,
                    prompt_width=width)

        a = run_arm(model, tok, cases, prompts, None, f"{name}")
        a.update(info, arm="plain")
        arms.append(a)
        print(f"  {name} plain:  schema {a['schema']*100:.0f}%  "
              f"exact {a['accuracy']*100:.0f}%  {a['out_tok_s']:.1f} tok/s  "
              f"${a['dollars_per_m']:.2f}/M", flush=True)

        if do_grammar:
            vocab = model.get_output_embeddings().weight.shape[0]
            ti0 = time.perf_counter()
            index, _ = gramlib.build_index(tok, PATTERN, vocab)
            build_s = time.perf_counter() - ti0
            b = run_arm(model, tok, cases, prompts, index,
                        f"{name} + grammar")
            b.update(info, arm="grammar", index_build_s=build_s)
            arms.append(b)
            print(f"  {name} grammar: schema {b['schema']*100:.0f}%  "
                  f"exact {b['accuracy']*100:.0f}%  {b['out_tok_s']:.1f} tok/s "
                  f" ${b['dollars_per_m']:.2f}/M", flush=True)

        del model, tok
        gc.collect()

    # --- the frontier: which arms are not dominated by another? ------------
    for a in arms:
        a["dominated_by"] = next(
            (b["label"] for b in arms
             if b is not a and b["accuracy"] >= a["accuracy"]
             and b["dollars_per_1k_correct"] <= a["dollars_per_1k_correct"]
             and (b["accuracy"] > a["accuracy"]
                  or b["dollars_per_1k_correct"] < a["dollars_per_1k_correct"])),
            None)
    frontier = [a["label"] for a in arms if a["dominated_by"] is None]
    best = max(arms, key=lambda a: a["accuracy"])
    cheapest_ok = min((a for a in arms if a["accuracy"] >= 0.9 * best["accuracy"]),
                      key=lambda a: a["dollars_per_1k_correct"])
    print(f"\n[frontier] {frontier}")
    print(f"[recommend] {cheapest_ok['label']}: "
          f"{cheapest_ok['accuracy']*100:.0f}% exact at "
          f"${cheapest_ok['dollars_per_1k_correct']:.2f}/1k correct "
          f"vs best {best['label']} {best['accuracy']*100:.0f}% at "
          f"${best['dollars_per_1k_correct']:.2f}")

    res = dict(config=dict(n_cases=N_CASES, batch=BATCH, max_new=MAX_NEW,
                           price_hr=PRICE_HR, overhead=OVERHEAD, duty=DUTY,
                           ladder=[(n, m) for n, m, _ in LADDER]),
               arms=arms, frontier=frontier,
               recommend=cheapest_ok["label"], best=best["label"],
               wall_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=1)
    plot(res)
    print(f"total {time.time()-t0:.0f}s")


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arms = res["arms"]
    labels = [a["label"] for a in arms]
    short = [l.replace(" + grammar", "\n+ grammar").replace("SmolLM2-", "Smol")
             .replace("Qwen2.5-", "Qwen") for l in labels]
    cols = ["#8e44ad" if a["arm"] == "grammar" else "#2980b9" for a in arms]
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))

    p = ax[0][0]
    x = np.arange(len(arms))
    p.bar(x - .2, [a["schema"] * 100 for a in arms], .4, label="valid JSON",
          color="#27ae60")
    p.bar(x + .2, [a["accuracy"] * 100 for a in arms], .4,
          label="all three fields correct", color="#c0392b")
    p.set_xticks(x, short, fontsize=7)
    p.set_ylabel("% of cases"), p.legend(fontsize=8)
    p.set_title("A. Quality on a task with a checkable answer\n"
                f"{res['config']['n_cases']} extractions, greedy decoding")

    p = ax[0][1]
    p.bar(x, [a["out_tok_s"] for a in arms], color=cols)
    for i, a in enumerate(arms):
        p.text(i, a["out_tok_s"], f"{a['out_tok_s']:.0f}", ha="center",
               va="bottom", fontsize=8)
    p.set_xticks(x, short, fontsize=7)
    p.set_ylabel("output tokens/s (batch 8)")
    p.set_title("B. Speed, measured on this machine\n"
                "purple = the same weights with a grammar attached")

    p = ax[0][2]
    p.bar(x, [a["dollars_per_m"] for a in arms], color=cols)
    for i, a in enumerate(arms):
        p.text(i, a["dollars_per_m"], f"${a['dollars_per_m']:.1f}",
               ha="center", va="bottom", fontsize=8)
    p.set_xticks(x, short, fontsize=7)
    p.set_ylabel("$ per million output tokens (all-in)")
    p.set_title("B2. Cost per token\n"
                "the usual metric -- and the wrong one for this decision")

    p = ax[1][0]
    for a, c in zip(arms, cols):
        m = "*" if a["dominated_by"] is None else "o"
        s = 260 if a["dominated_by"] is None else 90
        p.scatter(a["dollars_per_1k_correct"], a["accuracy"] * 100, s=s,
                  marker=m, color=c, zorder=3)
        p.annotate(a["label"].replace(" + grammar", "+gram"),
                   (a["dollars_per_1k_correct"], a["accuracy"] * 100),
                   textcoords="offset points", xytext=(7, 5), fontsize=7)
    p.set_xscale("log")
    p.set_xlabel("$ per 1,000 CORRECT extractions (log)")
    p.set_ylabel("exact-extraction accuracy (%)")
    p.set_title("C. The only chart that decides anything\n"
                "stars = on the frontier (nothing beats them on both axes)")

    p = ax[1][1]
    p.scatter([a["params"] / 1e6 for a in arms],
              [a["accuracy"] * 100 for a in arms], s=90, color=cols)
    for a in arms:
        p.annotate(a["label"].replace(" + grammar", "+gram"),
                   (a["params"] / 1e6, a["accuracy"] * 100),
                   textcoords="offset points", xytext=(6, 4), fontsize=7)
    p.set_xscale("log")
    p.set_xlabel("parameters (millions, log)")
    p.set_ylabel("exact-extraction accuracy (%)")
    p.set_title("C2. Accuracy is not a function of size\n"
                "two families, and a grammar moves a model vertically")

    p = ax[1][2]
    fields = ["parse", "schema", "name", "age", "skills"]
    w = 0.8 / len(arms)
    for i, (a, c) in enumerate(zip(arms, cols)):
        p.bar(np.arange(len(fields)) + i * w - 0.4,
              [a[f] * 100 for f in fields], w, label=a["label"], color=c,
              alpha=1.0 - 0.08 * i)
    p.set_xticks(np.arange(len(fields)) + 0.4 - w, fields, fontsize=8)
    p.set_ylabel("% correct")
    p.legend(fontsize=6)
    p.set_title("D. Where each model actually fails\n"
                "format, or the facts?")

    fig.suptitle("Right-sizing: 11x of parameters, one eval, and the "
                 "cheapest thing that is good enough", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(OUT, "rightsize.png"), dpi=118)
    print("wrote", os.path.join(OUT, "rightsize.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    if ap.parse_args().plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        main()
