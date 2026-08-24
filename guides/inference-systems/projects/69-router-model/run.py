#!/usr/bin/env python3
"""Project 69 — a tiny router in front of a fast path and a slow path.

  A. The label matrix.  Run all 60 questions through both paths and find out
     how many of them actually *need* the big model.  That is the routing
     headroom, and nothing a router does can exceed it.
  B. Six routers, each scored as a curve.  Every router produces a score per
     request; escalating the top x% traces a quality-versus-cost curve.
  C. The control: a random router traces the straight line between the two
     paths.  A router earns its keep only by sitting above that line.
  D. The router's own bill — measured, because a judge model that costs as much
     as the fast path cannot save anything.

  python3 run.py           # ~7 minutes
  python3 run.py --reuse   # re-analyse outputs/raw.json
  python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import routelib as RL                                          # noqa: E402

OUT = os.path.join(HERE, "outputs")
RAW = os.path.join(OUT, "raw.json")
FINDINGS = os.path.join(OUT, "findings.json")

PRICE_HR, DUTY, OVERHEAD = 0.55, 0.50, 1.25       # project 63's cost formula


def dollars_per_1k(engine_s: float) -> float:
    return PRICE_HR / 3600.0 * engine_s / DUTY * OVERHEAD * 1000


# ---------------------------------------------------------------------------
# A: run both paths
# ---------------------------------------------------------------------------

def run_path(model_id: str, label: str):
    print(f"  loading {model_id} ...", flush=True)
    tok, model = RL.load(model_id)
    ids = [RL.chat_ids(tok, q) for _, q, _, _ in RL.QUERIES]
    texts, secs, n_tok = RL.generate(model, tok, ids, max_new=32, batch=16)
    ok = [RL.graded(t, a, k) for t, (_, _, a, k) in zip(texts, RL.QUERIES)]
    print(f"  {label}: {sum(ok)}/{len(ok)} correct, {secs:.1f}s, "
          f"{n_tok} new tokens", flush=True)
    del model
    return dict(label=label, model=model_id, ok=ok, texts=texts,
                seconds=secs, new_tokens=n_tok,
                prompt_tokens=[len(x) for x in ids],
                sec_per_req=secs / len(ids))


def run_judge():
    """The prompted router: one forward pass, two candidate tokens compared.

    This is how a router is actually deployed — not by generating a sentence,
    but by reading a single next-token distribution.  The score is
    logit("HARD") - logit("EASY"): positive means "send it to the big model".
    """
    print(f"  loading judge {RL.JUDGE_ID} ...", flush=True)
    tok, model = RL.load(RL.JUDGE_ID)
    sysmsg = ("You are a routing classifier. Answer with one word: EASY if a "
              "very small language model can answer the question correctly, "
              "HARD if it needs a large model.")
    ids = [RL.chat_ids(tok, q, system=sysmsg) for _, q, _, _ in RL.QUERIES]
    easy = tok(" EASY", add_special_tokens=False).input_ids[0]
    hard = tok(" HARD", add_special_tokens=False).input_ids[0]
    easy2 = tok("EASY", add_special_tokens=False).input_ids[0]
    hard2 = tok("HARD", add_special_tokens=False).input_ids[0]
    scores, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(ids), 16):
            chunk = ids[i:i + 16]
            w = max(len(x) for x in chunk)
            pad = tok.pad_token_id
            inp = torch.tensor([[pad] * (w - len(x)) + x for x in chunk])
            att = torch.tensor([[0] * (w - len(x)) + [1] * len(x)
                                for x in chunk])
            lg = model(input_ids=inp, attention_mask=att).logits[:, -1, :]
            s = (torch.logsumexp(lg[:, [hard, hard2]], dim=-1)
                 - torch.logsumexp(lg[:, [easy, easy2]], dim=-1))
            scores += s.tolist()
    secs = time.time() - t0
    print(f"  judge: {secs:.1f}s for {len(ids)} classifications "
          f"({secs / len(ids) * 1000:.0f} ms each)", flush=True)
    del model
    return dict(scores=scores, seconds=secs, sec_per_req=secs / len(ids))


def run_embed():
    """Features for the trained router: sentence embeddings of the prompt."""
    from transformers import AutoModel, AutoTokenizer
    print(f"  loading embedder {RL.EMBED_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(RL.EMBED_ID)
    model = AutoModel.from_pretrained(RL.EMBED_ID).eval()
    qs = [q for _, q, _, _ in RL.QUERIES]
    t0 = time.time()
    with torch.no_grad():
        enc = tok(qs, return_tensors="pt", padding=True, truncation=True)
        out = model(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (out * mask).sum(1) / mask.sum(1)
        emb = torch.nn.functional.normalize(emb, dim=-1)
    secs = time.time() - t0
    print(f"  embeddings: {secs:.2f}s total "
          f"({secs / len(qs) * 1000:.1f} ms each)", flush=True)
    del model
    return emb.tolist(), secs / len(qs)


def measure():
    print("running both serving paths on 60 questions", flush=True)
    fast = run_path(RL.FAST_ID, "fast(135M)")
    slow = run_path(RL.SLOW_ID, "slow(1.5B)")
    judge = run_judge()
    emb, emb_s = run_embed()
    raw = dict(fast=fast, slow=slow, judge=judge, emb=emb, emb_sec_per_req=emb_s,
               cats=[c for c, _, _, _ in RL.QUERIES],
               questions=[q for _, q, _, _ in RL.QUERIES])
    os.makedirs(OUT, exist_ok=True)
    with open(RAW, "w") as f:
        json.dump(raw, f)
    return raw


# ---------------------------------------------------------------------------
# the trained router: logistic regression, 4-fold cross-validated
# ---------------------------------------------------------------------------

def trained_scores(emb, target, folds=4, epochs=300, lr=0.5, seed=0):
    """Predict "the fast path will get this wrong", out of fold.

    Out-of-fold means every prediction comes from a model that never saw that
    question during training.  Skipping this is the single easiest way to
    publish a router that looks brilliant and does nothing in production.
    """
    X = torch.tensor(emb, dtype=torch.float32)
    y = torch.tensor([1.0 if t else 0.0 for t in target])
    n = len(y)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    scores = torch.zeros(n)
    for f in range(folds):
        te = perm[f::folds]
        tr = torch.tensor([i for i in perm.tolist() if i not in set(te.tolist())])
        w = torch.zeros(X.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            z = X[tr] @ w + b
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                z, y[tr]) + 1e-3 * w.square().sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            scores[te] = X[te] @ w + b
    return scores.tolist()


# ---------------------------------------------------------------------------
# B/C: score every router as a curve
# ---------------------------------------------------------------------------

def curve(scores, fast_ok, slow_ok, fast_cost, slow_cost, router_cost):
    """Escalate the top-x% by score; return (fraction, accuracy, $/1k) points."""
    n = len(scores)
    order = sorted(range(n), key=lambda i: -scores[i])
    pts = []
    for k in range(n + 1):
        esc = set(order[:k])
        acc = sum((slow_ok[i] if i in esc else fast_ok[i]) for i in range(n)) / n
        frac = k / n
        cost = fast_cost * (1 - frac) + slow_cost * frac + router_cost
        pts.append(dict(frac=round(frac, 4), acc=round(100 * acc, 2),
                        usd=round(cost, 4)))
    return pts


def best_at_or_above(pts, target_acc):
    ok = [p for p in pts if p["acc"] >= target_acc - 1e-9]
    return min(ok, key=lambda p: p["usd"]) if ok else None


def analyse(raw):
    fast_ok = raw["fast"]["ok"]
    slow_ok = raw["slow"]["ok"]
    n = len(fast_ok)
    cats = raw["cats"]

    fast_cost = dollars_per_1k(raw["fast"]["sec_per_req"])
    slow_cost = dollars_per_1k(raw["slow"]["sec_per_req"])
    judge_cost = dollars_per_1k(raw["judge"]["sec_per_req"])
    emb_cost = dollars_per_1k(raw["emb_sec_per_req"])

    quad = dict(both=0, slow_only=0, fast_only=0, neither=0)
    per_cat = {}
    for i in range(n):
        k = ("both" if fast_ok[i] and slow_ok[i] else
             "slow_only" if slow_ok[i] else
             "fast_only" if fast_ok[i] else "neither")
        quad[k] += 1
        d = per_cat.setdefault(cats[i], dict(n=0, fast=0, slow=0))
        d["n"] += 1
        d["fast"] += bool(fast_ok[i])
        d["slow"] += bool(slow_ok[i])

    A = dict(n=n, fast_acc=round(100 * sum(fast_ok) / n, 1),
             slow_acc=round(100 * sum(slow_ok) / n, 1),
             quadrants=quad, per_cat=per_cat,
             fast_sec=round(raw["fast"]["sec_per_req"], 3),
             slow_sec=round(raw["slow"]["sec_per_req"], 3),
             judge_sec=round(raw["judge"]["sec_per_req"], 4),
             emb_sec=round(raw["emb_sec_per_req"], 4),
             fast_usd_1k=round(fast_cost, 3), slow_usd_1k=round(slow_cost, 3),
             judge_usd_1k=round(judge_cost, 4), emb_usd_1k=round(emb_cost, 4),
             slow_over_fast=round(slow_cost / fast_cost, 2))

    # --- the routers -------------------------------------------------------
    import random
    rng = random.Random(3)
    fast_wrong = [not o for o in fast_ok]

    def category_scores():
        """Route by task type, the feature a real API already has.

        Each question is scored by how often the *other* questions of its own
        category needed the big model — leave-one-out, so no question ever
        contributes to its own score.
        """
        out = []
        for i in range(n):
            same = [j for j in range(n) if cats[j] == cats[i] and j != i]
            need = sum(1 for j in same if slow_ok[j] and not fast_ok[j])
            out.append(need / max(len(same), 1))
        return out
    routers = {
        "oracle": (
            [(1.0 if (slow_ok[i] and not fast_ok[i]) else
              0.5 if (not slow_ok[i] and not fast_ok[i]) else 0.0)
             for i in range(n)], 0.0),
        "random": ([rng.random() for _ in range(n)], 0.0),
        "length": ([float(x) for x in raw["fast"]["prompt_tokens"]], 0.0),
        "prompted-0.5B": (raw["judge"]["scores"], judge_cost),
        "trained-embed": (trained_scores(raw["emb"], fast_wrong), emb_cost),
        # the same features and the same learner, aimed at a different target:
        # not "the small model will fail" but "the big model will fix it".
        # Those are not the same question — 13 of these 60 questions are wrong
        # on both paths, and escalating them buys nothing.
        "category (leave-one-out)": (category_scores(), 0.0),
        "trained-embed (fixable)": (
            trained_scores(raw["emb"],
                           [bool(s_ok and not f_ok)
                            for s_ok, f_ok in zip(slow_ok, fast_ok)]),
            emb_cost),
    }
    fixable = [bool(s_ok and not f_ok) for s_ok, f_ok in zip(slow_ok, fast_ok)]
    B = {}
    for name, (sc, rc) in routers.items():
        pts = curve(sc, fast_ok, slow_ok, fast_cost, slow_cost, rc)
        # how well does the score separate the two classes? (AUC)
        pos = [sc[i] for i in range(n) if fast_wrong[i]]
        neg = [sc[i] for i in range(n) if not fast_wrong[i]]
        auc = (sum(1 for a in pos for b in neg if a > b)
               + 0.5 * sum(1 for a in pos for b in neg if a == b)) \
            / max(len(pos) * len(neg), 1)
        p2 = [sc[i] for i in range(n) if fixable[i]]
        n2 = [sc[i] for i in range(n) if not fixable[i]]
        auc_fix = (sum(1 for a in p2 for b in n2 if a > b)
                   + 0.5 * sum(1 for a in p2 for b in n2 if a == b)) \
            / max(len(p2) * len(n2), 1)
        B[name] = dict(points=pts, auc=round(auc, 3),
                       auc_fixable=round(auc_fix, 3),
                       router_usd_1k=round(rc, 4))

    # --- headline comparisons ---------------------------------------------
    target = A["slow_acc"]
    C = {}
    for name, d in B.items():
        hit = best_at_or_above(d["points"], target)
        half = best_at_or_above(d["points"],
                                (A["fast_acc"] + A["slow_acc"]) / 2)
        C[name] = dict(
            auc=d["auc"], auc_fixable=d["auc_fixable"],
            match_slow=hit,
            saving_vs_slow=(round(100 * (1 - hit["usd"] / (slow_cost)), 1)
                            if hit else None),
            halfway=half,
            acc_at_25=next(p["acc"] for p in d["points"]
                           if abs(p["frac"] - 0.25) < 0.008),
            acc_at_50=next(p["acc"] for p in d["points"]
                           if abs(p["frac"] - 0.50) < 0.008),
        )
    # area above the random line, at matched escalation fraction
    rnd = {p["frac"]: p["acc"] for p in B["random"]["points"]}
    for name, d in B.items():
        gains = [p["acc"] - rnd[p["frac"]] for p in d["points"]]
        C[name]["mean_gain_over_random"] = round(sum(gains) / len(gains), 2)
        C[name]["max_gain_over_random"] = round(max(gains), 2)
    return A, B, C


# ---------------------------------------------------------------------------

def plot(F):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A, B, C = F["A"], F["B"], F["C"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    fig.suptitle("Routing 60 questions between a 135M fast path and a 1.5B "
                 "slow path", fontsize=13)

    p = ax[0]
    q = A["quadrants"]
    labels = ["both right\n(router can save)", "slow only\n(must escalate)",
              "fast only\n(escalating hurts)", "neither\n(nothing helps)"]
    vals = [q["both"], q["slow_only"], q["fast_only"], q["neither"]]
    p.bar(range(4), vals, color=["#4c9f70", "#c0504d", "#e0a458", "#8d9db6"])
    for i, v in enumerate(vals):
        p.text(i, v, f" {v}", ha="center", va="bottom")
    p.set_xticks(range(4))
    p.set_xticklabels(labels, fontsize=8)
    p.set_ylabel("questions")
    p.set_title(f"A. Routing headroom\nfast {A['fast_acc']}%  "
                f"slow {A['slow_acc']}%  of {A['n']}")

    p = ax[1]
    colors = {"oracle": "#000000", "random": "#8d9db6", "length": "#e0a458",
              "prompted-0.5B": "#c0504d", "trained-embed": "#4a6fa5",
              "trained-embed (fixable)": "#4c9f70",
              "category (leave-one-out)": "#8e5ea2"}
    for name, d in B.items():
        xs = [pt["usd"] for pt in d["points"]]
        ys = [pt["acc"] for pt in d["points"]]
        p.plot(xs, ys, "-", color=colors[name], lw=2 if name != "random" else 2,
               ls="--" if name == "random" else "-",
               label=f"{name} (AUC {d['auc']})")
    p.scatter([A["fast_usd_1k"]], [A["fast_acc"]], marker="s", color="k",
              zorder=5)
    p.annotate("all fast", (A["fast_usd_1k"], A["fast_acc"]),
               textcoords="offset points", xytext=(6, -10), fontsize=8)
    p.scatter([A["slow_usd_1k"]], [A["slow_acc"]], marker="s", color="k",
              zorder=5)
    p.annotate("all slow", (A["slow_usd_1k"], A["slow_acc"]),
               textcoords="offset points", xytext=(-52, 4), fontsize=8)
    p.set_xlabel("$ per 1,000 requests (measured seconds, project 63 formula)")
    p.set_ylabel("accuracy (%)")
    p.set_title("B/C. Every router as a curve.\nThe dashed line is random — "
                "beating it is the whole job")
    p.legend(fontsize=8, loc="lower right")
    p.grid(alpha=0.3)

    p = ax[2]
    names = [n for n in B if n != "oracle"]
    gains = [C[n]["mean_gain_over_random"] for n in names]
    p.barh(range(len(names)), gains,
           color=[colors[n] for n in names])
    p.axvline(0, color="k", lw=1)
    p.set_yticks(range(len(names)))
    p.set_yticklabels(names, fontsize=9)
    p.set_xlabel("mean accuracy points above the random router")
    p.set_title("C. Value added over flipping a coin\n"
                f"(oracle: {C['oracle']['mean_gain_over_random']} points)")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = os.path.join(OUT, "router.png")
    fig.savefig(path, dpi=110)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

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

    A, B, C = analyse(raw)
    F = dict(A=A, B=B, C=C)
    os.makedirs(OUT, exist_ok=True)
    with open(FINDINGS, "w") as f:
        json.dump(F, f, indent=1)

    print("\n--- A. the two paths ---------------------------------------------")
    print(f"fast(135M) {A['fast_acc']}%  {A['fast_sec'] * 1000:.0f} ms/req  "
          f"${A['fast_usd_1k']}/1k")
    print(f"slow(1.5B) {A['slow_acc']}%  {A['slow_sec'] * 1000:.0f} ms/req  "
          f"${A['slow_usd_1k']}/1k   ({A['slow_over_fast']}x the price)")
    print(f"quadrants {A['quadrants']}")
    for c, d in A["per_cat"].items():
        print(f"  {c:<10} n={d['n']:2d}  fast {d['fast']:2d}  slow {d['slow']:2d}")
    print("\n--- B/C. routers --------------------------------------------------")
    for name, d in C.items():
        m = d["match_slow"]
        print(f"{name:<24} AUC(fail) {d['auc']:.3f} AUC(fixable) "
              f"{d['auc_fixable']:.3f}  acc@25% {d['acc_at_25']:5.1f}  "
              f"acc@50% {d['acc_at_50']:5.1f}  "
              f"mean gain vs random {d['mean_gain_over_random']:+5.2f} pts  "
              + (f"matches slow at {m['frac'] * 100:.0f}% escalation, "
                 f"${m['usd']:.3f}/1k (saves {d['saving_vs_slow']}%)"
                 if m else "never matches the slow path"))
    print(f"\nrouter overhead: judge ${A['judge_usd_1k']}/1k "
          f"({A['judge_sec'] * 1000:.0f} ms), embedder ${A['emb_usd_1k']}/1k "
          f"({A['emb_sec'] * 1000:.1f} ms) vs fast path ${A['fast_usd_1k']}/1k")
    plot(F)


if __name__ == "__main__":
    main()
