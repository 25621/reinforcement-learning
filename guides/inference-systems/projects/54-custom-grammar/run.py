"""Project 54 -- A custom grammar for SQL, enforced at decode time.

Three arms on the same 150 questions, same model, same seeds:

    A. free       no mask at all; the schema is described in the prompt
    B. generic    a grammar for "valid SQL shape", identifiers left free
    C. schema     the same grammar, but the column and table names are
                  baked into the automaton as literal alternatives

The point of having B *and* C is the whole lesson: B guarantees the query
parses, C guarantees it also refers to things that exist. They are different
guarantees with different costs, and only one of them stops the model
inventing a column.

Grading is not a heuristic -- every query is executed against a real SQLite
database and compared with the gold query's result.

Usage:
    python3 run.py            # ~5 minutes
    python3 run.py --plot     # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "51-needle-in-a-haystack"))
sys.path.insert(0, os.path.join(PROJ, "53-json-mode-reliability"))
import ctxlib  # noqa: E402
import gramlib  # noqa: E402

OUT = os.path.join(HERE, "outputs")

N_CASES = 150
BATCH = 50
MAX_NEW = 40

COLUMNS = ["name", "city", "age", "salary", "department"]
TABLE = "employees"
CITIES = ["lisbon", "osaka", "nairobi", "helsinki", "perth"]
DEPTS = ["research", "sales", "support", "design"]


# ---------------------------------------------------------------------------
# The database
# ---------------------------------------------------------------------------


def make_db(seed=5, n=200):
    con = sqlite3.connect(":memory:")
    con.execute(f"CREATE TABLE {TABLE} (name TEXT, city TEXT, age INTEGER, "
                f"salary INTEGER, department TEXT)")
    rnd = random.Random(seed)
    for i in range(n):
        con.execute(f"INSERT INTO {TABLE} VALUES (?,?,?,?,?)",
                    (f"person{i}", rnd.choice(CITIES), rnd.randrange(21, 65),
                     rnd.randrange(30, 200) * 1000, rnd.choice(DEPTS)))
    con.commit()
    return con


SCHEMA_TEXT = (f"Table {TABLE}(name TEXT, city TEXT, age INTEGER, "
               f"salary INTEGER, department TEXT)")


# ---------------------------------------------------------------------------
# The grammars
# ---------------------------------------------------------------------------
#
# Both are ordinary regular expressions. The only difference is what stands
# in for an identifier:
#
#   generic:  [a-z_]+            -- "some lowercase word"
#   schema:   (name|city|age|…)  -- "one of the five columns that exist"
#
# That one substitution is the difference between a query that parses and a
# query that runs.

def _alt(words):
    return "(" + "|".join(words) + ")"


GENERIC = (r"SELECT (COUNT\(\*\)|[a-z_]+) FROM [a-z_]+ WHERE [a-z_]+ "
           r"(=|>|<) ('[a-z ]+'|[0-9]+);")

SCHEMA = (r"SELECT (COUNT\(\*\)|" + _alt(COLUMNS) + r") FROM " + TABLE +
          r" WHERE " + _alt(COLUMNS) + r" (=|>|<) ('" +
          _alt(CITIES + DEPTS) + r"'|[0-9]+);")


# ---------------------------------------------------------------------------
# The questions
# ---------------------------------------------------------------------------


def make_cases(n, seed=17):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        kind = rnd.randrange(4)
        if kind == 0:
            c = rnd.choice(CITIES)
            out.append((f"How many employees live in {c}?",
                        f"SELECT COUNT(*) FROM {TABLE} WHERE city = '{c}';"))
        elif kind == 1:
            a = rnd.randrange(25, 60)
            out.append((f"How many employees are older than {a}?",
                        f"SELECT COUNT(*) FROM {TABLE} WHERE age > {a};"))
        elif kind == 2:
            d = rnd.choice(DEPTS)
            out.append((f"List the names of employees in the {d} department.",
                        f"SELECT name FROM {TABLE} WHERE department = '{d}';"))
        else:
            s = rnd.randrange(60, 180) * 1000
            out.append((f"How many employees earn more than {s}?",
                        f"SELECT COUNT(*) FROM {TABLE} WHERE salary > {s};"))
    return out


INSTRUCTION = ("Write one SQLite query that answers the question. "
               "Use only the columns listed. Output only the query, "
               "ending with a semicolon.")


def prompt_ids(tok, q, pad_to):
    user = f"{SCHEMA_TEXT}\n\nQuestion: {q}\n\n{INSTRUCTION}"
    ids = ctxlib.chat_ids(tok, user,
                          system="You write SQLite queries and nothing else.")
    ids = ids[0].tolist()
    pad = tok.pad_token_id or tok.eos_token_id
    return [pad] * (pad_to - len(ids)) + ids


# ---------------------------------------------------------------------------
# Grading against the real database
# ---------------------------------------------------------------------------


def extract_sql(text):
    s = text.strip()
    for fence in ("```sql", "```"):
        if fence in s:
            s = s.split(fence, 1)[1].split("```", 1)[0]
    i = s.lower().find("select")
    if i < 0:
        return None
    s = s[i:]
    j = s.find(";")
    return s[:j + 1] if j >= 0 else s.split("\n")[0]


def grade(con, sql, gold):
    r = {"has_sql": False, "parses": False, "runs": False,
         "bad_identifier": False, "correct": False}
    if not sql:
        return r
    r["has_sql"] = True
    try:
        con.execute("EXPLAIN " + sql)
        r["parses"] = True
    except sqlite3.Error as e:
        # SQLite reports an unknown column as a *parse* error, so separate
        # "the shape is wrong" from "the name does not exist" explicitly.
        if "no such column" in str(e) or "no such table" in str(e):
            r["bad_identifier"] = True
        return r
    try:
        got = con.execute(sql).fetchall()
        r["runs"] = True
    except sqlite3.Error as e:
        if "no such column" in str(e) or "no such table" in str(e):
            r["bad_identifier"] = True
        return r
    want = con.execute(gold).fetchall()
    r["correct"] = sorted(map(str, got)) == sorted(map(str, want))
    return r


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def run_arm(model, tok, index, cases, prompts, con, label):
    rows, t0 = [], time.perf_counter()
    for b0 in range(0, len(cases), BATCH):
        chunk = cases[b0:b0 + BATCH]
        ids = torch.tensor([prompts[b0 + i] for i in range(len(chunk))])
        outs = gramlib.generate(model, ids, MAX_NEW, temperature=0.7,
                                index=index, eos_id=tok.eos_token_id,
                                seed=500 + b0)
        for (q, gold), o in zip(chunk, outs):
            text = tok.decode(o, skip_special_tokens=True)
            sql = extract_sql(text)
            g = grade(con, sql, gold)
            g["text"], g["sql"], g["gold"] = text, sql, gold
            rows.append(g)
        print(f"  {label}: {len(rows)}/{len(cases)} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return rows, time.perf_counter() - t0


def summarise(rows):
    n = len(rows)
    keys = ["has_sql", "parses", "runs", "bad_identifier", "correct"]
    s = {k: sum(r[k] for r in rows) / n for k in keys}
    s["n"] = n
    return s


def measure():
    tok, model = ctxlib.load()
    vocab_width = model.config.vocab_size
    con = make_db()
    cases = make_cases(N_CASES)

    print("== compiling the two grammars ==")
    strings = gramlib.token_strings(tok)
    grammars = {}
    for name, pat in (("generic", GENERIC), ("schema", SCHEMA)):
        t0 = time.perf_counter()
        dfa = gramlib.compile_regex(pat)
        compile_s = time.perf_counter() - t0
        idx = gramlib.TokenIndex(dfa, strings, vocab_width, tok.eos_token_id)
        grammars[name] = idx
        print(f"  {name:8} {dfa.n_states:>4} states  "
              f"regex->DFA {compile_s*1000:6.1f} ms  "
              f"token index {idx.build_s:5.2f}s  "
              f"{idx.walks:>8} token walks")

    raw = [ctxlib.chat_ids(
        tok, f"{SCHEMA_TEXT}\n\nQuestion: {q}\n\n{INSTRUCTION}",
        system="You write SQLite queries and nothing else.").shape[1]
        for q, _ in cases]
    pad_to = max(raw)
    prompts = [prompt_ids(tok, q, pad_to) for q, _ in cases]

    res = {"model": ctxlib.MODEL_ID, "n_cases": N_CASES,
           "generic_pattern": GENERIC, "schema_pattern": SCHEMA,
           "grammars": {k: {"states": v.dfa.n_states, "walks": v.walks,
                            "build_s": v.build_s,
                            "mask_states_cached": len(v.allowed)}
                        for k, v in grammars.items()},
           "arms": {}, "samples": {}}

    for label, idx in (("free", None), ("generic", grammars["generic"]),
                       ("schema", grammars["schema"])):
        print(f"\n== {label} ==")
        rows, secs = run_arm(model, tok, idx, cases, prompts, con, label)
        s = summarise(rows)
        s["wall_s"] = secs
        res["arms"][label] = s
        res["samples"][label] = [{"sql": r["sql"], "gold": r["gold"],
                                  "correct": r["correct"]} for r in rows[:6]]
        print(f"  parses {s['parses']*100:5.1f}%  runs {s['runs']*100:5.1f}%"
              f"  bad-identifier {s['bad_identifier']*100:5.1f}%"
              f"  correct {s['correct']*100:5.1f}%")

    # How often does the schema grammar leave the model no choice at all?
    forced = {}
    for name, idx in grammars.items():
        n_forced = sum(1 for st in range(idx.dfa.n_states)
                       if idx.forced(st) is not None)
        sizes = [len(idx.allowed.get(st, {})) for st in range(idx.dfa.n_states)]
        forced[name] = {"states": idx.dfa.n_states, "forced_states": n_forced,
                        "median_choices": sorted(sizes)[len(sizes) // 2],
                        "max_choices": max(sizes)}
    res["choice"] = forced
    print("\n== how much choice the grammar leaves ==")
    for k, v in forced.items():
        print(f"  {k:8} {v['forced_states']}/{v['states']} states have exactly "
              f"one legal token; median {v['median_choices']} choices, "
              f"max {v['max_choices']}")

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
    x = np.arange(len(names))

    a = ax[0]
    for k, (key, col) in enumerate([("parses", "#95a5a6"), ("runs", "#27ae60"),
                                    ("correct", "#e67e22")]):
        a.bar(x + (k - 1) * .27, [res["arms"][n][key] * 100 for n in names],
              .27, label=key, color=col)
    a.set_xticks(x, names)
    a.set_ylim(0, 105)
    a.set_ylabel(f"% of {res['n_cases']} questions")
    a.legend(fontsize=8)
    a.set_title("A. Parses → runs → answers correctly")

    a = ax[1]
    bad = [res["arms"][n]["bad_identifier"] * 100 for n in names]
    a.bar(names, bad, color=["#c0392b", "#c0392b", "#27ae60"], width=.55)
    a.set_ylabel("% referring to a column/table that does not exist")
    for i, v in enumerate(bad):
        a.text(i, v, f"{v:.1f}%", ha="center", va="bottom")
    a.set_title("B. Inventing an identifier —\nonly the schema grammar "
                "makes it impossible")

    a = ax[2]
    gs = res["grammars"]
    ks = list(gs)
    a.bar(np.arange(len(ks)) - .2, [gs[k]["states"] for k in ks], .4,
          label="DFA states", color="#2980b9")
    a2 = a.twinx()
    a2.bar(np.arange(len(ks)) + .2, [gs[k]["build_s"] for k in ks], .4,
           label="index build (s)", color="#8e44ad")
    a.set_xticks(np.arange(len(ks)), ks)
    a.set_ylabel("DFA states", color="#2980b9")
    a2.set_ylabel("token-index build (s)", color="#8e44ad")
    a.set_title("C. What the grammar costs to compile\n(paid once, per "
                "grammar, not per request)")

    fig.suptitle("A custom grammar: syntax is free, existence costs a "
                 "bigger automaton, correctness is still the model's job",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "grammar.png"), dpi=120)
    print("wrote", os.path.join(OUT, "grammar.png"))


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
