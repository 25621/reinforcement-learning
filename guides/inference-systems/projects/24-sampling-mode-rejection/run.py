"""Project 24 -- sampling-mode rejection.

Greedy speculation asks a yes/no question ("is this the token the target would
pick?"). Random sampling has no single right answer, so verification has to
preserve a whole *distribution* instead. This project implements the
accept/reject step and then tries hard to break it.

  A. The rule, on a toy distribution where the exact answer is computable.
     Four verification rules, 400,000 draws each, measured against the target
     distribution.
  B. The same four rules on a real 151,936-entry distribution from the model.
  C. The acceptance rate is not a free parameter: it equals 1 - TV(p, q).
     Check that against the running loop.
  D. Temperature sweep: what users' sampling settings do to your speedup.
  E. The mismatch trap -- score a draft token against a distribution it was
     not drawn from and the guarantee quietly dies.

    python3 run.py           # ~4 minutes on 6 CPU threads
    python3 run.py --plot    # redraw from outputs/findings.json
"""

from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, "..", "23-greedy-speculative-decoding"))

import torch  # noqa: E402

import speclib as S  # noqa: E402

N_DRAWS = 400_000
K = 4
MAX_NEW = 32
MODES = ["rejection", "resample_p", "always", "greedy_check"]
F = {}


def exact_output(p, q, mode):
    """The distribution each rule *actually* produces, computed in closed form
    so that section A does not rest on Monte-Carlo noise."""
    m = torch.minimum(p, q)              # P(propose x and accept it)
    p_rej = float(1 - m.sum())
    if mode == "always":
        return q.clone()
    if mode == "greedy_check":
        out = torch.zeros_like(p)
        out[int(p.argmax())] = 1.0
        return out
    if mode == "resample_p":
        return m + p_rej * p
    return m + p_rej * S.residual(p, q)  # == p, exactly


def compare(p, q, label, seed=0, q_test=None):
    rows = []
    for mode in MODES:
        draws = S.draw_many(p, q, N_DRAWS, mode, seed=seed, q_test=q_test)
        emp = S.empirical(draws, p.numel())
        rows.append({
            "mode": mode,
            "tv_measured": round(S.tv_distance(emp, p), 5),
            "tv_exact": round(S.tv_distance(exact_output(p, q, mode), p), 5),
        })
    return {"label": label, "tv_pq": round(S.tv_distance(p, q), 5),
            "predicted_acceptance": round(1 - S.tv_distance(p, q), 5),
            "rows": rows}


# ---------------------------------------------------------------------------
# A. the rule on a toy
# ---------------------------------------------------------------------------


def section_a():
    print("\n== A. toy distributions ==")
    # Eight tokens. The draft is confidently wrong about token 0 and misses
    # token 5 almost entirely -- both directions of disagreement, on purpose.
    p = torch.tensor([0.05, 0.30, 0.20, 0.15, 0.10, 0.15, 0.03, 0.02])
    q = torch.tensor([0.30, 0.25, 0.20, 0.10, 0.10, 0.01, 0.03, 0.01])
    res = compare(p, q, "toy")
    res["p"] = [round(float(x), 4) for x in p]
    res["q"] = [round(float(x), 4) for x in q]
    res["residual"] = [round(float(x), 4) for x in S.residual(p, q)]
    # Empirical acceptance for the correct rule, to check 1 - TV.
    g = torch.Generator().manual_seed(1)
    x = torch.multinomial(q, N_DRAWS, replacement=True, generator=g)
    r = torch.rand(N_DRAWS, generator=g)
    acc = float((r < (p[x] / q[x]).clamp(max=1.0)).float().mean())
    res["measured_acceptance"] = round(acc, 5)
    F["A"] = res
    for row in res["rows"]:
        print(f"  {row['mode']:14s} TV(output, p) = {row['tv_measured']:.5f} "
              f"(exact {row['tv_exact']:.5f})")
    print(f"  acceptance: predicted 1-TV = {res['predicted_acceptance']:.4f}, "
          f"measured {acc:.4f}")


# ---------------------------------------------------------------------------
# B. the same rules on a real distribution
# ---------------------------------------------------------------------------


def section_b(target, draft, tok):
    print("\n== B. real model distributions ==")
    ids = S.chat_ids(tok, S.WORKLOADS["chat"])
    tc, dc = S.make_cache(target), S.make_cache(draft)
    tl = target.forward(ids, tc, start_pos=0)
    dl = draft.forward(ids, dc, start_pos=0)
    p = S.probs_from(tl[0, -1], temperature=1.0)
    q = S.probs_from(dl[0, -1], temperature=1.0)
    res = compare(p, q, "real")
    top = torch.topk(p, 5)
    res["target_top5"] = [
        {"token": tok.decode([int(i)]), "p": round(float(p[i]), 4),
         "q": round(float(q[i]), 4)}
        for i in top.indices]
    res["support_p"] = int((p > 1e-6).sum())
    res["support_q"] = int((q > 1e-6).sum())
    F["B"] = res
    for row in res["rows"]:
        print(f"  {row['mode']:14s} TV(output, p) = {row['tv_measured']:.5f}")
    print(f"  TV(p, q) = {res['tv_pq']:.4f} -> acceptance should be "
          f"{res['predicted_acceptance']:.4f}")


# ---------------------------------------------------------------------------
# C + D. the loop, across temperatures
# ---------------------------------------------------------------------------


def section_cd(target, draft, tok):
    print("\n== C+D. temperature sweep ==")
    ids = S.chat_ids(tok, S.WORKLOADS["chat"])
    tc, dc = S.make_cache(target), S.make_cache(draft)
    seeds = (7, 11, 23)
    rows = []
    for temp in (0.3, 0.7, 1.0, 1.5):
        # Average over seeds: alpha is a random variable here, and a single
        # trajectory of ~10 iterations is far too few to quote to 3 digits.
        runs = [S.speculative_sampling(target, draft, tc, dc, ids, k=K,
                                       max_new=MAX_NEW, temperature=temp,
                                       seed=s) for s in seeds]
        mean = lambda key: sum(r[key] for r in runs) / len(runs)  # noqa: E731
        pred = S.speedup_model(mean("tokens_per_iter"), K, 0.345, 0.195)
        rows.append({
            "temperature": temp,
            "acceptance_rate": round(mean("acceptance_rate"), 4),
            "conditional_acceptance": round(mean("conditional_acceptance"), 4),
            "one_minus_tv": round(mean("one_minus_tv"), 4),
            "tokens_per_iter": round(mean("tokens_per_iter"), 3),
            "iters": round(mean("iters"), 1),
            "predicted_speedup": round(pred, 3),
            "text": tok.decode(runs[0]["tokens"]),
        })
        print(f"  T={temp:<4} accept(cond) {mean('conditional_acceptance'):.3f}"
              f"  1-TV {mean('one_minus_tv'):.3f}"
              f"  alpha {mean('tokens_per_iter'):.2f}  -> {pred:.2f}x predicted")

    # Greedy, for reference: the T -> 0 limit of the same loop.
    gd = S.ModelDrafter(draft)
    gr = S.speculative_greedy(target, gd, tc, ids, k=K, max_new=MAX_NEW)
    rows.insert(0, {
        "temperature": 0.0,
        "acceptance_rate": round(gr["acceptance_rate"], 4),
        "conditional_acceptance": round(gr["conditional_acceptance"], 4),
        "one_minus_tv": None,
        "tokens_per_iter": round(gr["tokens_per_iter"], 3),
        "iters": gr["iters"],
        "predicted_speedup": round(
            S.speedup_model(gr["tokens_per_iter"], K, 0.345, 0.195), 3),
        "text": tok.decode(gr["tokens"]),
    })
    print(f"  T=0.0 (greedy) accept(cond) {gr['conditional_acceptance']:.3f}  "
          f"alpha {gr['tokens_per_iter']:.2f}")

    # Real wall clock at T=1.0. Averaged, not minimised: the spread here is
    # the algorithm's own randomness (how many tokens each iteration accepts),
    # not machine noise, and taking the minimum would flatter speculation.
    bl, sp = [], []
    for i in range(3):
        bl.append(S.sample_decode(target, tc, ids, max_new=MAX_NEW,
                                  temperature=1.0, seed=100 + i)["decode_s"])
        sp.append(S.speculative_sampling(target, draft, tc, dc, ids, k=K,
                                         max_new=MAX_NEW, temperature=1.0,
                                         seed=100 + i)["decode_s"])
    b_mean, s_mean = sum(bl) / len(bl), sum(sp) / len(sp)
    F["C"] = {"rows": rows, "seeds": list(seeds),
              "wall": {"baseline_decode_s": round(b_mean, 3),
                       "spec_decode_s": round(s_mean, 3),
                       "spec_runs_s": [round(x, 3) for x in sp],
                       "speedup": round(b_mean / s_mean, 3)}}
    print(f"  wall clock at T=1.0: {b_mean:.2f}s -> {s_mean:.2f}s "
          f"= {b_mean/s_mean:.2f}x  (spec runs: "
          f"{[round(x,2) for x in sp]})")


# ---------------------------------------------------------------------------
# E. the mismatch trap
# ---------------------------------------------------------------------------


def section_e(target, draft, tok):
    print("\n== E. mismatch trap ==")
    ids = S.chat_ids(tok, S.WORKLOADS["chat"])
    tc, dc = S.make_cache(target), S.make_cache(draft)
    tl = target.forward(ids, tc, start_pos=0)
    dl = draft.forward(ids, dc, start_pos=0)
    p = S.probs_from(tl[0, -1], temperature=1.0)
    q_filtered = S.probs_from(dl[0, -1], temperature=1.0, top_p=0.9)
    q_raw = S.probs_from(dl[0, -1], temperature=1.0)

    rows = []
    for label, q_draw, q_test in [
        ("matched (q = the distribution sampled from)", q_filtered, q_filtered),
        ("mismatched (sampled from top-p 0.9, scored against raw q)",
         q_filtered, q_raw),
        ("mismatched (sampled from raw q, scored against top-p 0.9)",
         q_raw, q_filtered),
    ]:
        draws = S.draw_many(p, q_draw, N_DRAWS, "rejection", seed=3,
                            q_test=q_test)
        emp = S.empirical(draws, p.numel())
        rows.append({"case": label,
                     "tv_to_target": round(S.tv_distance(emp, p), 5)})
        print(f"  {label:58s} TV = {rows[-1]['tv_to_target']:.5f}")
    F["E"] = {"rows": rows, "draft_top_p": 0.9,
              "q_support_raw": int((q_raw > 1e-6).sum()),
              "q_support_filtered": int((q_filtered > 0).sum())}


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    a = f["A"]
    x = range(len(a["p"]))
    ax[0].bar([i - 0.2 for i in x], a["p"], 0.4, color="#2471a3", label="target p")
    ax[0].bar([i + 0.2 for i in x], a["q"], 0.4, color="#c0392b", label="draft q")
    ax[0].plot(list(x), a["residual"], "o--", color="#27ae60",
               label="residual norm(p-q)⁺")
    ax[0].set_xlabel("toy token id")
    ax[0].set_ylabel("probability")
    ax[0].set_title("A. where the draft is wrong")
    ax[0].legend(fontsize=7)

    for i, (key, title) in enumerate([("A", "A. toy (8 tokens)"),
                                      ("B", "B. real (151,936 tokens)")]):
        rows = f[key]["rows"]
        vals = [r["tv_measured"] for r in rows]
        cols = ["#27ae60" if r["mode"] == "rejection" else "#c0392b"
                for r in rows]
        ax[1 + i].bar(range(len(rows)), vals, color=cols)
        ax[1 + i].set_xticks(range(len(rows)))
        ax[1 + i].set_xticklabels([r["mode"] for r in rows], fontsize=7,
                                  rotation=20)
        ax[1 + i].set_ylabel("TV distance from the target distribution")
        ax[1 + i].set_title(title)
        for j, v in enumerate(vals):
            ax[1 + i].text(j, v, f"{v:.3f}", ha="center", va="bottom",
                           fontsize=8)

    rows = f["C"]["rows"]
    t = [r["temperature"] for r in rows]
    ax[3].plot(t, [r["tokens_per_iter"] for r in rows], "o-", color="#2471a3",
               label="tokens per target pass")
    ax[3].plot(t, [r["conditional_acceptance"] for r in rows], "s-",
               color="#e67e22", label="acceptance (per position tested)")
    ax[3].plot(t, [r["predicted_speedup"] for r in rows], "^--",
               color="#27ae60", label="predicted speedup")
    ax[3].set_xlabel("temperature")
    ax[3].set_title("D. users' settings move your speedup")
    ax[3].legend(fontsize=8)
    ax[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sampling_rejection.png"), dpi=110)
    print("wrote outputs/sampling_rejection.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    section_a()
    target, draft, tok, _ = S.load_pair()
    F["setup"] = {"target": S.TARGET_ID, "draft": S.DRAFT_ID, "k": K,
                  "max_new": MAX_NEW, "n_draws": N_DRAWS,
                  "threads": S.N_THREADS}
    section_b(target, draft, tok)
    section_cd(target, draft, tok)
    section_e(target, draft, tok)
    F["wall_s"] = round(time.time() - t0, 1)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
