"""Project 51 -- Needle in a haystack.

Three questions, in order:

  A. With the full KV cache, how far can this model actually go? Hide one fact
     at a known depth inside prompts of 512..8,192 tokens, surround it with
     seven look-alike facts, and ask for it back.
  B. What does that length cost? Prefill seconds and KV megabytes as the
     prompt grows -- the bill you pay whether or not the recall holds.
  C. Can a *serving decision* create a cliff the model does not have? Re-run
     the same grid with a bounded cache (sliding window, then sliding window
     plus attention sinks) and watch recall fall off a diagonal.

Usage:
    python3 run.py            # ~8 minutes
    python3 run.py --plot     # redraw the figure from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch

import ctxlib

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")

# The grid. Depth is *where in the haystack the needle sits*, 0.0 = very
# first token, 1.0 = very last.
LENGTHS = [512, 1024, 2048, 4096, 8192]
DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
COST_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]

# Section C: bounded caches. `sinks` = how many of the very first tokens are
# kept forever no matter how old they get.
BOUNDED_LENGTHS = [2048, 4096]
WINDOW = 1024

TARGET = "Zurich"
DISTRACTOR_CITIES = ["Lisbon", "Nairobi", "Osaka", "Bogota",
                     "Helsinki", "Dakar", "Perth"]


# ---------------------------------------------------------------------------
# Building a haystack
# ---------------------------------------------------------------------------


def make_codes(seed: int = 7) -> dict:
    rnd = random.Random(seed)
    cities = [TARGET] + DISTRACTOR_CITIES
    return {c: str(1_000_000 + rnd.randrange(8_999_999)) for c in cities}


def sentence(city: str, code: str) -> str:
    return f" The secret {city} access code is {code}. "


def build_prompt(tok, filler, codes, n_filler: int, depth: float):
    """Return (input_ids, answer_code).

    The target fact goes in at `depth`; the seven distractors are spread
    evenly through the whole haystack so that the model can never win by
    "just report the only number you saw".
    """
    hay = list(filler[:n_filler])
    inserts = []
    for j, city in enumerate(DISTRACTOR_CITIES):
        d = (j + 0.5) / len(DISTRACTOR_CITIES)
        inserts.append((d, tok(sentence(city, codes[city]),
                               add_special_tokens=False).input_ids))
    inserts.append((depth, tok(sentence(TARGET, codes[TARGET]),
                               add_special_tokens=False).input_ids))
    # Insert from the back so earlier indices stay valid.
    inserts.sort(key=lambda z: -z[0])
    for d, toks in inserts:
        k = int(d * len(hay))
        hay[k:k] = toks
    question = (f"What is the secret {TARGET} access code? "
                f"Answer with the number only.")
    user = tok.decode(hay) + "\n\n" + question
    return ctxlib.chat_ids(tok, user), codes[TARGET]


# ---------------------------------------------------------------------------
# Bounded-cache masks
# ---------------------------------------------------------------------------


def bounded_mask(T: int, window: int, sinks: int) -> torch.Tensor:
    """A (1, 1, T, T) additive mask for "the cache only held `window` tokens".

    Why a mask instead of really throwing tensors away? Because what we are
    measuring is *what the model can see*, and a mask reproduces that exactly
    while letting the whole prompt go through in one prefill pass. The memory
    saving is then reported arithmetically (window + sinks tokens x 24 KB)
    rather than measured -- said plainly so nobody mistakes it for a timing.

    Rows are query positions, columns are key positions. A query at position i
    may look at key j when:
        j <= i                      (causal -- no peeking at the future)
        and (i - j < window         (inside the sliding window)
             or j < sinks)          (or it is one of the pinned first tokens)
    """
    i = torch.arange(T).view(-1, 1)
    j = torch.arange(T).view(1, -1)
    keep = (j <= i) & (((i - j) < window) | (j < sinks))
    mask = torch.zeros(T, T)
    mask.masked_fill_(~keep, float("-inf"))
    return mask.view(1, 1, T, T)


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


@torch.inference_mode()
def trial(model, tok, ids, answer, mask=None, n_new=10):
    t0 = time.perf_counter()
    if mask is None:
        out = model(ids, use_cache=True, logits_to_keep=1)
    else:
        out = model(ids, attention_mask=mask, use_cache=True, logits_to_keep=1)
    prefill_s = time.perf_counter() - t0
    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    new = [int(nxt)]
    for _ in range(n_new - 1):
        # Decoding after a masked prefill: the new token is allowed to see the
        # whole cache. A real windowed engine would keep masking, but by this
        # point the answer is already decided by what prefill could attend to,
        # and leaving decode unmasked keeps the comparison about the prompt.
        out = model(nxt, past_key_values=past, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        new.append(int(nxt))
    text = tok.decode(new, skip_special_tokens=True)
    del past
    return (answer in text), text.strip(), prefill_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def measure():
    tok, model = ctxlib.load()
    cfg = model.config
    filler = ctxlib.filler_tokens(tok)
    codes = make_codes()
    kv_per_tok = ctxlib.kv_bytes_per_token(cfg, dtype_bytes=4)
    print(f"filler tokens available: {len(filler)}")
    print(f"KV bytes/token: {kv_per_tok}")

    res = {
        "model": ctxlib.MODEL_ID,
        "kv_bytes_per_token": kv_per_tok,
        "n_layers": cfg.num_hidden_layers,
        "window": WINDOW,
        "lengths": LENGTHS,
        "depths": DEPTHS,
        "distractors": len(DISTRACTOR_CITIES),
        "grid": [],       # section A
        "cost": [],       # section B
        "bounded": [],    # section C
    }

    # -- A: full cache ------------------------------------------------------
    print("\n== A. full cache ==")
    for T in LENGTHS:
        for d in DEPTHS:
            ids, ans = build_prompt(tok, filler, codes, T, d)
            hit, text, pre = trial(model, tok, ids, ans)
            res["grid"].append({"len": T, "depth": d, "tokens": ids.shape[1],
                                "hit": hit, "text": text, "prefill_s": pre})
            print(f"  T={T:>5} depth={d:.1f} tok={ids.shape[1]:>5} "
                  f"{pre:6.2f}s hit={hit} {text[:40]!r}", flush=True)

    # -- B: what the length costs ------------------------------------------
    # Section A already prefilled every grid length at depth 0.5, so reuse
    # those timings rather than paying for them twice; only the extra
    # 16k point has to be measured on its own.
    print("\n== B. cost of length ==")
    mid_depth = DEPTHS[len(DEPTHS) // 2]
    for T in COST_LENGTHS:
        if T in LENGTHS:
            cell = next(c for c in res["grid"]
                        if c["len"] == T and c["depth"] == mid_depth)
            n, dt = cell["tokens"], cell["prefill_s"]
        else:
            ids, _ = build_prompt(tok, filler, codes, T, mid_depth)
            n = ids.shape[1]
            with torch.inference_mode():
                t0 = time.perf_counter()
                out = model(ids, use_cache=True, logits_to_keep=1)
                dt = time.perf_counter() - t0
            del out
        res["cost"].append({"tokens": n, "prefill_s": dt,
                            "tok_per_s": n / dt,
                            "kv_mb": n * kv_per_tok / 1e6})
        print(f"  {n:>6} tok  {dt:7.2f}s  {n/dt:6.0f} tok/s  "
              f"KV {n*kv_per_tok/1e6:7.1f} MB", flush=True)

    # -- C: a cliff made by the serving policy ------------------------------
    print("\n== C. bounded cache ==")
    for sinks in (0, 4):
        for T in BOUNDED_LENGTHS:
            for d in DEPTHS:
                ids, ans = build_prompt(tok, filler, codes, T, d)
                n = ids.shape[1]
                mask = bounded_mask(n, WINDOW, sinks)
                hit, text, pre = trial(model, tok, ids, ans, mask=mask)
                del mask
                # Was the needle inside the window at all? The needle sits at
                # roughly depth*n; the last token is at n-1.
                needle_pos = int(d * n)
                visible = (n - 1 - needle_pos) < WINDOW
                res["bounded"].append({
                    "sinks": sinks, "len": T, "depth": d, "tokens": n,
                    "hit": hit, "text": text, "in_window": visible,
                    "prefill_s": pre})
                print(f"  sinks={sinks} T={T:>5} depth={d:.1f} "
                      f"in_window={visible} hit={hit} {text[:32]!r}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "findings.json"), "w") as f:
        json.dump(res, f, indent=2)
    return res


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    lengths, depths = res["lengths"], res["depths"]
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    def heat(a, rows, cells, title, key=lambda c: c["hit"]):
        grid = np.full((len(depths), len(rows)), np.nan)
        for c in cells:
            if c["len"] not in rows:
                continue
            grid[depths.index(c["depth"]), rows.index(c["len"])] = float(key(c))
        a.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto",
                 origin="lower")
        a.set_xticks(range(len(rows)), [str(r) for r in rows])
        a.set_yticks(range(len(depths)), [f"{d:.1f}" for d in depths])
        a.set_xlabel("haystack tokens")
        a.set_ylabel("needle depth")
        a.set_title(title)
        for yi in range(grid.shape[0]):
            for xi in range(grid.shape[1]):
                v = grid[yi, xi]
                if not np.isnan(v):
                    a.text(xi, yi, "hit" if v else "MISS", ha="center",
                           va="center", fontsize=8,
                           color="black" if v else "white")
        return grid

    ga = heat(ax[0][0], lengths, res["grid"],
              "A. Full KV cache — recall of 1 fact among 8")
    n_hit = int(np.nansum(ga))
    ax[0][0].set_title(f"A. Full KV cache — {n_hit}/{ga.size} recalled")

    # B: cost
    a = ax[0][1]
    n = [c["tokens"] for c in res["cost"]]
    s = [c["prefill_s"] for c in res["cost"]]
    a.plot(n, s, "o-", color="#c0392b", label="prefill seconds (measured)")
    lin = [s[0] * (x / n[0]) for x in n]
    a.plot(n, lin, "--", color="#7f8c8d", label="perfectly linear in length")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("prompt tokens"); a.set_ylabel("seconds to first token")
    a.grid(alpha=.3, which="both")
    a2 = a.twinx()
    a2.plot(n, [c["kv_mb"] for c in res["cost"]], "s-", color="#2980b9",
            label="KV cache MB")
    a2.set_ylabel("KV cache (MB)", color="#2980b9")
    a2.set_yscale("log")
    a.legend(loc="upper left", fontsize=8)
    a.set_title("B. The bill: TTFT and cache size vs. prompt length")

    b0 = [c for c in res["bounded"] if c["sinks"] == 0]
    b4 = [c for c in res["bounded"] if c["sinks"] == 4]
    g0 = heat(ax[1][0], BOUNDED_LENGTHS, b0,
              f"C1. Sliding window {res['window']}, no sinks")
    g4 = heat(ax[1][1], BOUNDED_LENGTHS, b4,
              f"C2. Sliding window {res['window']} + 4 attention sinks")
    ax[1][0].set_title(f"C1. Window {res['window']}, 0 sinks — "
                       f"{int(np.nansum(g0))}/{g0.size} recalled")
    ax[1][1].set_title(f"C2. Window {res['window']} + 4 sinks — "
                       f"{int(np.nansum(g4))}/{g4.size} recalled")

    fig.suptitle("Needle in a haystack: the model's limit, the bill, "
                 "and the cliff your cache policy invents", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT, "needle.png"), dpi=120)
    print("wrote", os.path.join(OUT, "needle.png"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.plot:
        with open(os.path.join(OUT, "findings.json")) as f:
            plot(json.load(f))
    else:
        t0 = time.time()
        plot(measure())
        print(f"total {time.time()-t0:.0f}s")
