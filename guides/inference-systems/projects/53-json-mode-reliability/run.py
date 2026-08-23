"""Project 53 -- JSON-mode reliability.

800 generations of the same extraction task: 160 per arm, five arms.

    unconstrained  T=0.0 / 0.7 / 1.0
    constrained    T=0.7 / 1.0     (same model, same seeds, same prompts)

Measured for each arm: how often the output parses as JSON, how often it
matches the schema, and -- the part a validity dashboard cannot see -- how
often the *content* is right. Plus the price of the mask, timed per decode
step at batch 50.

Usage:
    python3 run.py            # ~8 minutes
    python3 run.py --plot     # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "51-needle-in-a-haystack"))
import ctxlib  # noqa: E402
import gramlib  # noqa: E402
from jsontask import (INSTRUCTION, PATTERN, SCHEMA, SYSTEM,  # noqa: E402
                      make_cases)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

N_PER_ARM = 160
BATCH = 80         # 2 batches per arm; bigger batches decode more
                   # efficiently, and the arms are identical either way
MAX_NEW = 56


def prompt_ids(tok, case, pad_to):
    """Build the prompt and pad it on the LEFT to a fixed length.

    Left padding, not right: the last token of the prompt has to be the real
    final token for every row in the batch, because that is the position the
    first output token is predicted from. Padding on the right would make the
    model continue from a pad token.
    """
    user = f"Bio: {case['bio']}\n\n{INSTRUCTION}"
    ids = ctxlib.chat_ids(tok, user, system=SYSTEM)
    ids = ids[0].tolist()
    pad = tok.pad_token_id or tok.eos_token_id
    return [pad] * (pad_to - len(ids)) + ids


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def grade(text, case):
    r = {"parse_ok": False, "schema_ok": False, "name_ok": False,
         "age_ok": False, "skills_ok": False, "all_ok": False}
    s = text.strip()
    # Be generous to the unconstrained arm: pull out the first {...} block
    # rather than demanding the model emit nothing else. A stricter reading
    # would flatter the constrained arm for free.
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


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_arm(model, tok, index, cases, temperature, prompts, name):
    rows, t0 = [], time.perf_counter()
    for b0 in range(0, len(cases), BATCH):
        chunk = cases[b0:b0 + BATCH]
        ids = torch.tensor([prompts[b0 + i] for i in range(len(chunk))])
        outs = gramlib.generate(model, ids, MAX_NEW, temperature=temperature,
                                index=index, eos_id=tok.eos_token_id,
                                seed=1000 + b0)
        for c, o in zip(chunk, outs):
            text = tok.decode(o, skip_special_tokens=True)
            g = grade(text, c)
            g["text"] = text
            g["n_tokens"] = len(o)
            rows.append(g)
        print(f"  {name}: {len(rows)}/{len(cases)} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return rows, time.perf_counter() - t0


def summarise(rows):
    n = len(rows)
    keys = ["parse_ok", "schema_ok", "name_ok", "age_ok", "skills_ok", "all_ok"]
    s = {k: sum(r[k] for r in rows) / n for k in keys}
    s["n"] = n
    s["mean_tokens"] = sum(r["n_tokens"] for r in rows) / n
    return s


def mask_cost(model, tok, index, prompts, steps=6):
    """How much does the mask itself cost per decode step?

    Timed round-robin and kept at the minimum: this box is shared, and
    running one arm to completion then the other charges any background
    spike entirely to whichever happened to be running.
    """
    ids = torch.tensor(prompts[:BATCH])
    t = ctxlib.interleaved({
        "plain": lambda: gramlib.generate(model, ids, steps, 0.0, None,
                                          tok.eos_token_id),
        "masked": lambda: gramlib.generate(model, ids, steps, 0.0, index,
                                           tok.eos_token_id),
    }, rounds=4, warmup=1)
    return {"plain_s_per_step": t["plain"] / steps,
            "masked_s_per_step": t["masked"] / steps,
            "steps": steps, "batch": BATCH}


def measure():
    tok, model = ctxlib.load()
    vocab_width = model.config.vocab_size
    print("== compiling the grammar ==")
    t0 = time.perf_counter()
    index, strings = gramlib.build_index(tok, PATTERN, vocab_width)
    print(f"  {index.dfa.n_states} DFA states, {index.walks} token walks, "
          f"index built in {index.build_s:.2f}s "
          f"(regex->DFA {time.perf_counter()-t0-index.build_s:.2f}s)")

    cases = make_cases(N_PER_ARM)
    raw = [ctxlib.chat_ids(tok, f"Bio: {c['bio']}\n\n{INSTRUCTION}",
                           system=SYSTEM).shape[1]
           for c in cases]
    pad_to = max(raw)
    prompts = [prompt_ids(tok, c, pad_to) for c in cases]
    print(f"  {len(cases)} cases, prompt padded to {pad_to} tokens")

    arms = [("unconstrained T=0.0", 0.0, None),
            ("unconstrained T=0.7", 0.7, None),
            ("unconstrained T=1.0", 1.0, None),
            ("constrained T=0.7", 0.7, index),
            ("constrained T=1.0", 1.0, index)]

    res = {"model": ctxlib.MODEL_ID, "pattern": PATTERN, "schema": SCHEMA,
           "n_per_arm": N_PER_ARM, "max_new": MAX_NEW,
           "dfa_states": index.dfa.n_states, "index_build_s": index.build_s,
           "index_walks": index.walks, "arms": {}, "samples": {}}
    for name, temp, idx in arms:
        print(f"\n== {name} ==")
        rows, secs = run_arm(model, tok, idx, cases, temp, prompts, name)
        s = summarise(rows)
        s["wall_s"] = secs
        res["arms"][name] = s
        res["samples"][name] = [r["text"] for r in rows[:4]]
        print(f"  parse {s['parse_ok']*100:5.1f}%  schema {s['schema_ok']*100:5.1f}%"
              f"  name {s['name_ok']*100:5.1f}%  age {s['age_ok']*100:5.1f}%"
              f"  all {s['all_ok']*100:5.1f}%")

    print("\n== mask cost ==")
    res["mask_cost"] = mask_cost(model, tok, index, prompts)
    mc = res["mask_cost"]
    print(f"  plain {mc['plain_s_per_step']*1000:.1f} ms/step   "
          f"masked {mc['masked_s_per_step']*1000:.1f} ms/step   "
          f"overhead {mc['masked_s_per_step']/mc['plain_s_per_step']-1:+.1%}")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(res["arms"])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))

    a = ax[0]
    x = np.arange(len(names))
    parse = [res["arms"][n]["parse_ok"] * 100 for n in names]
    schema = [res["arms"][n]["schema_ok"] * 100 for n in names]
    a.bar(x - .2, parse, .4, label="parses as JSON", color="#95a5a6")
    a.bar(x + .2, schema, .4, label="matches the schema", color="#27ae60")
    a.set_xticks(x, [n.replace(" ", "\n", 1) for n in names], fontsize=8)
    a.set_ylabel("% of 200 generations")
    a.set_ylim(0, 105)
    a.legend(fontsize=8)
    a.set_title("A. Structural validity")
    for i, v in enumerate(schema):
        a.text(i + .2, v + 1, f"{v:.0f}", ha="center", fontsize=8)

    a = ax[1]
    allok = [res["arms"][n]["all_ok"] * 100 for n in names]
    nameok = [res["arms"][n]["name_ok"] * 100 for n in names]
    ageok = [res["arms"][n]["age_ok"] * 100 for n in names]
    a.bar(x - .27, nameok, .27, label="name correct", color="#2980b9")
    a.bar(x, ageok, .27, label="age correct", color="#8e44ad")
    a.bar(x + .27, allok, .27, label="every field correct", color="#e67e22")
    a.set_xticks(x, [n.replace(" ", "\n", 1) for n in names], fontsize=8)
    a.set_ylabel("% of 200 generations")
    a.set_ylim(0, 105)
    a.legend(fontsize=8)
    a.set_title("B. Content accuracy — the mask does not supply facts")

    a = ax[2]
    mc = res["mask_cost"]
    a.bar(["no mask", "masked"],
          [mc["plain_s_per_step"] * 1000, mc["masked_s_per_step"] * 1000],
          color=["#95a5a6", "#c0392b"], width=.55)
    a.set_ylabel(f"ms per decode step (batch {mc['batch']})")
    for i, v in enumerate([mc["plain_s_per_step"], mc["masked_s_per_step"]]):
        a.text(i, v * 1000, f"{v*1000:.0f} ms", ha="center", va="bottom")
    ov = mc["masked_s_per_step"] / mc["plain_s_per_step"] - 1
    a.set_title(f"C. What the mask costs — {ov:+.1%} per step\n"
                f"({res['dfa_states']} DFA states, index built once in "
                f"{res['index_build_s']:.1f} s)")

    fig.suptitle("JSON mode: structural validity goes to 100%, "
                 "accuracy does not follow", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "json_mode.png"), dpi=120)
    print("wrote", os.path.join(OUT, "json_mode.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    if a.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        t0 = time.time()
        plot(measure())
        print(f"total {time.time()-t0:.0f}s")
