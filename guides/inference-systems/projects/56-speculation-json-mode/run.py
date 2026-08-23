"""Project 56 -- Speculation on a JSON-mode workload.

Structured output is mostly boilerplate: the same braces, quotes and key
names on every single request. Two very different tricks exploit that, and
this project runs both against the same task and measures which one the
predictability actually pays.

    plain    constrained decode, one forward pass per token
    lookup   + prompt-lookup speculation: draft the next few tokens by
             copying from text already in the prompt, verify them in one
             forward pass, keep the prefix that was right
    jump     + grammar jump-forward: when the automaton leaves exactly one
             legal token, emit it with no forward pass at all
    both     lookup and jump together

Then the control that makes the result mean something: the same speculation
on a free-prose task, where nothing is predictable.

The unit that matters is **forward passes per output token**. Wall-clock is
reported too, but on a shared CPU it moves for reasons that have nothing to
do with the algorithm.

Usage:
    python3 run.py            # ~7 minutes
    python3 run.py --plot
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "51-needle-in-a-haystack"))
sys.path.insert(0, os.path.join(PROJ, "53-json-mode-reliability"))
import ctxlib  # noqa: E402
import gramlib  # noqa: E402
from jsontask import INSTRUCTION, PATTERN, SYSTEM, make_cases  # noqa: E402

OUT = os.path.join(HERE, "outputs")

N_CASES = 12
MAX_NEW = 60
DRAFT_K = 4          # tokens drafted per speculation attempt
NGRAM = 2            # how many trailing tokens must match to copy


# ---------------------------------------------------------------------------
# Prompt-lookup drafting
# ---------------------------------------------------------------------------


def lookup_draft(context: list[int], k: int, n: int = NGRAM):
    """Find the most recent earlier occurrence of the last n tokens and copy
    what followed it.

    No second model, no training, no extra memory: the draft comes from text
    the request already contains. That is why it is called *prompt* lookup.
    It works when the output repeats the input -- summarisation, editing,
    and, as this project measures, schema-shaped output when the prompt
    carries an example.
    """
    if len(context) <= n:
        return []
    pat = context[-n:]
    for i in range(len(context) - n - 1, -1, -1):
        if context[i:i + n] == pat:
            cand = context[i + n:i + n + k]
            if cand:
                return cand
    return []


# ---------------------------------------------------------------------------
# One decoding run
# ---------------------------------------------------------------------------


@torch.inference_mode()
def decode(model, tok, prompt_ids, index, max_new, use_lookup, use_jump,
           eos_id):
    """Greedy decode with optional speculation. Batch of one.

    Bookkeeping returned:
        forwards        model calls (the thing that costs money)
        tokens          tokens actually emitted
        jumped          tokens the grammar supplied for free
        drafted/accept  speculation attempts and how many tokens survived
    """
    ids = list(prompt_ids)
    out = []
    state = 0 if index is not None else None
    forwards = jumped = drafted = accepted = 0
    past = None
    pending = list(ids)          # emitted but not yet pushed through the model
    t0 = time.perf_counter()
    stop = False

    while len(out) < max_new and not stop:
        # --- 1. text the grammar leaves no choice about -------------------
        # Ask the automaton for the longest run of characters that is
        # completely determined from here, tokenise it, and emit it. These
        # tokens cost no forward pass: they ride along as extra *input* on
        # the next call, and a forward pass over m+1 tokens costs barely more
        # than one over 1 token because decode is memory-bound.
        #
        # Asking for forced *text* rather than a forced *token* is the whole
        # trick. Byte-pair encoding can usually spell the same characters
        # several ways, so a single next token is almost never unique even
        # when the next characters are -- see the measurement in the README.
        if use_jump and index is not None:
            while len(out) < max_new:
                f = index.forced_text(state)
                if f is None:
                    break
                text, end_state = f
                toks = tok(text, add_special_tokens=False).input_ids
                st, ok = state, True
                for t in toks:
                    nx = index.step(st, t)
                    if nx is None:
                        ok = False
                        break
                    st = nx
                if not ok or st != end_state or not toks:
                    break
                out += toks
                pending += toks
                state = end_state
                jumped += len(toks)
            if len(out) >= max_new:
                break

        # --- 2. build the speculation ------------------------------------
        draft = []
        if use_lookup:
            draft = lookup_draft(ids + out, DRAFT_K)
            if index is not None:
                # A drafted token the grammar forbids can never be accepted,
                # so drop it now instead of spending a verification slot.
                st, keep = state, []
                for t in draft:
                    nx = index.step(st, t)
                    if nx is None:
                        break
                    keep.append(t)
                    st = nx
                draft = keep
            if draft:
                drafted += 1

        # --- 3. ONE forward pass over pending + draft --------------------
        block = pending + draft
        n_logits = len(draft) + 1
        o = model(torch.tensor([block]), past_key_values=past, use_cache=True,
                  logits_to_keep=n_logits)
        past = o.past_key_values
        forwards += 1
        logits = o.logits[0].float()      # (n_logits, V)
        # logits[0] predicts the token after `pending`, i.e. it verifies
        # draft[0]; logits[j] verifies draft[j]; logits[len(draft)] is the
        # free bonus token you get when every draft was right.

        st = state
        n_accepted = 0
        rejected = False
        for j in range(n_logits):
            lg = logits[j]
            if index is not None:
                lg = lg.masked_fill(~index.mask(st), float("-inf"))
            pick = int(lg.argmax())
            if pick == eos_id:
                stop = True
                break
            nxt_state = index.step(st, pick) if index is not None else None
            if j < len(draft) and pick == draft[j]:
                out.append(pick)
                accepted += 1
                n_accepted += 1
                st = nxt_state
            else:
                # Either the draft was wrong here, or this is the bonus slot.
                out.append(pick)
                st = nxt_state
                rejected = j < len(draft)
                break
            if index is not None and st is None:
                stop = True
                break
            if len(out) >= max_new:
                break

        # --- 4. roll the cache back to what really happened --------------
        # The forward pass wrote KV rows for every drafted token. Anything
        # after the first rejection never happened, so those rows must go --
        # otherwise the next step attends to tokens the model did not emit.
        if rejected:
            surplus = len(draft) - n_accepted
            if surplus > 0:
                past.crop(past.get_seq_length() - surplus)

        if index is not None and st is None and not stop:
            stop = True             # the grammar says the output is finished
        state = st
        pending = [out[-1]] if out else [ids[-1]]

    return {"tokens": len(out), "forwards": forwards, "jumped": jumped,
            "drafted": drafted, "accepted": accepted,
            "wall_s": time.perf_counter() - t0,
            "text": tok.decode(out, skip_special_tokens=True)}


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


EXAMPLE = ('{"name": "Grace Hopper", "age": 45, '
           '"skills": ["compilers", "mathematics"]}')


def json_prompt(tok, case):
    """A one-shot prompt: the example is what prompt-lookup drafts FROM.

    Without an example in the prompt there is nothing to copy and prompt
    lookup has no material -- which is itself worth knowing, and is why the
    one-shot form is the realistic one for this trick.
    """
    user = (f"Example output:\n{EXAMPLE}\n\n"
            f"Bio: {case['bio']}\n\n{INSTRUCTION}")
    return ctxlib.chat_ids(
        tok, user, system=SYSTEM)[0].tolist()


def prose_prompt(tok, case):
    user = (f"Bio: {case['bio']}\n\n"
            "Write two sentences about this person in plain English.")
    return ctxlib.chat_ids(tok, user)[0].tolist()


def run_arms(model, tok, cases, eos, label, prompt_fn, arms):
    res = {}
    for name, lookup, jump, idx in arms:
        agg = {"tokens": 0, "forwards": 0, "jumped": 0, "drafted": 0,
               "accepted": 0, "wall_s": 0.0}
        sample = None
        for c in cases:
            r = decode(model, tok, prompt_fn(tok, c), idx, MAX_NEW,
                       lookup, jump, eos)
            for k in agg:
                agg[k] += r[k]
            sample = sample or r["text"]
        agg["forwards_per_token"] = agg["forwards"] / agg["tokens"]
        agg["ms_per_token"] = agg["wall_s"] / agg["tokens"] * 1000
        agg["free_token_pct"] = agg["jumped"] / agg["tokens"] * 100
        agg["accept_per_draft"] = (agg["accepted"] / agg["drafted"]
                                   if agg["drafted"] else 0.0)
        agg["sample"] = sample.strip()[:90]
        res[name] = agg
        print(f"  {label}/{name:7}: {agg['forwards_per_token']:.3f} fwd/token"
              f"  {agg['ms_per_token']:6.1f} ms/token"
              f"  free {agg['free_token_pct']:5.1f}%"
              f"  accept/draft {agg['accept_per_draft']:.2f}", flush=True)
    return res


def measure():
    tok, model = ctxlib.load()
    eos = tok.eos_token_id
    index, _ = gramlib.build_index(tok, PATTERN, model.config.vocab_size)
    cases = make_cases(N_CASES, seed=99)
    print(f"grammar: {index.dfa.n_states} states, index {index.build_s:.2f}s")

    print("\n== JSON workload (constrained) ==")
    json_arms = [("plain", False, False, index),
                 ("lookup", True, False, index),
                 ("jump", False, True, index),
                 ("both", True, True, index)]
    json_res = run_arms(model, tok, cases, eos, "json",
                        json_prompt, json_arms)

    print("\n== prose control (unconstrained) ==")
    prose_arms = [("plain", False, False, None),
                  ("lookup", True, False, None)]
    prose_res = run_arms(model, tok, cases, eos, "prose",
                         prose_prompt, prose_arms)

    res = {"model": ctxlib.MODEL_ID, "n_cases": N_CASES, "max_new": MAX_NEW,
           "draft_k": DRAFT_K, "ngram": NGRAM,
           "dfa_states": index.dfa.n_states,
           "json": json_res, "prose": prose_res}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 5.2))
    names = list(res["json"])

    a = ax[0]
    v = [res["json"][n]["forwards_per_token"] for n in names]
    a.bar(names, v, color=["#7f8c8d", "#2980b9", "#8e44ad", "#27ae60"],
          width=.6)
    a.axhline(1.0, ls="--", color="#c0392b", label="one forward per token")
    for i, x in enumerate(v):
        a.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=9)
    a.set_ylabel("forward passes per output token")
    a.legend(fontsize=8)
    a.set_title(f"A. JSON mode — best is "
                f"{min(v):.3f} fwd/token ({1/min(v):.2f}x fewer)")

    a = ax[1]
    j = [res["json"][n]["free_token_pct"] for n in names]
    acc = [res["json"][n]["accept_per_draft"] for n in names]
    x = np.arange(len(names))
    a.bar(x - .2, j, .4, label="% tokens the grammar forced (free)",
          color="#27ae60")
    a2 = a.twinx()
    a2.bar(x + .2, acc, .4, label="tokens accepted per draft",
           color="#2980b9")
    a.set_xticks(x, names)
    a.set_ylabel("% free tokens", color="#27ae60")
    a2.set_ylabel(f"accepted / draft (max {res['draft_k']})", color="#2980b9")
    a.set_title("B. Where the saving comes from")

    a = ax[2]
    pn = list(res["prose"])
    jv = [res["json"][n]["forwards_per_token"] for n in pn]
    pv = [res["prose"][n]["forwards_per_token"] for n in pn]
    x = np.arange(len(pn))
    a.bar(x - .2, jv, .4, label="JSON (constrained)", color="#27ae60")
    a.bar(x + .2, pv, .4, label="free prose", color="#c0392b")
    a.set_xticks(x, pn)
    a.set_ylabel("forward passes per output token")
    a.legend(fontsize=8)
    a.set_title("C. The control — the same drafter,\na workload with "
                "nothing to copy")

    fig.suptitle("Speculation on structured output: the predictable part of "
                 "the output is the cheap part", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "spec_json.png"), dpi=120)
    print("wrote", os.path.join(OUT, "spec_json.png"))


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
