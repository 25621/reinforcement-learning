"""Project 27 -- Medusa heads.

Train three extra prediction heads on the target model itself, so the drafter
is part of the model instead of being a second model. Then ask the question
the guide asks: how does acceptance compare against an external draft?

  A. Train the heads. Two label sets from one forward pass: the corpus's next
     token, and the model's own next token (self-distillation).
  B. Held-out accuracy per head -- how much harder is "two ahead" than "one
     ahead"?
  C. Acceptance and alpha in the real loop: Medusa vs the 0.5B external draft.
  D. Cost. This is where self-speculation is supposed to win.
  E. What our four-minute training run does and does not prove.

    python3 run.py           # ~7 minutes on 6 CPU threads
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
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "23-greedy-speculative-decoding"))

import torch  # noqa: E402

import medusa as M  # noqa: E402
import speclib as S  # noqa: E402

N_HEADS = 3
K = N_HEADS          # the external draft gets the same budget
MAX_NEW = 48
STEPS = 250
N_TOKENS = 16384
PROMPTS = ("chat", "code")
F = {}


# ---------------------------------------------------------------------------
# A + B. collect, train, score
# ---------------------------------------------------------------------------


def section_ab(target, tok):
    print("\n== A. collect hidden states ==")
    with torch.no_grad():
        data = M.collect(target, tok, n_tokens=N_TOKENS)
    agree = float((data["gt"] == data["sd"]).float().mean())
    print(f"  {data['n']} positions in {data['collect_s']:.1f}s; the model's "
          f"own next token matches the corpus {agree:.1%} of the time")

    print("\n== A/B. train the heads ==")
    trained, stats = {}, {}
    for key, name in (("gt", "ground-truth labels"),
                      ("sd", "self-distilled labels")):
        t0 = time.time()
        heads, st = M.train_heads(data, key, target.d_model, target.lm_head,
                                  n_heads=N_HEADS, steps=STEPS)
        trained[key] = heads
        stats[key] = st
        print(f"  [{name}] {STEPS} steps in {time.time()-t0:.0f}s, "
              f"loss {st['curve'][0]['loss']:.2f} -> "
              f"{st['curve'][-1]['loss']:.2f}, "
              f"top-1 vs model's own tokens {st['holdout_top1']['sd']}")
    F["A"] = {
        "n_positions": data["n"], "collect_s": data["collect_s"],
        "corpus_model_agreement": round(agree, 4),
        "n_params_per_headset": stats["gt"]["n_params"],
        "d_model": target.d_model, "vocab": int(target.lm_head.shape[0]),
        "untied_head_params": int(target.d_model * target.lm_head.shape[0]
                                  * N_HEADS),
        "train": {k: stats[k] for k in stats},
    }
    return trained


# ---------------------------------------------------------------------------
# C. the loop
# ---------------------------------------------------------------------------


def head_probe(target, draft, heads, tc, dc, prompt_ids, out_tokens):
    """Score the drafters on text the target has already produced.

    One teacher-forced pass, no speculative loop. For each answer position it
    asks: would head i have named the token i+2 places ahead, and would the
    0.5B draft have named the very next one? That separates "the heads are
    weak on this text" from "the speculative loop has a bug" -- two failures
    that look identical from the acceptance rate alone.
    """
    full = torch.cat([prompt_ids, torch.tensor([out_tokens])], dim=1)
    p0 = prompt_ids.shape[1]
    tc.reset()
    dc.reset()
    with torch.no_grad():
        h = M.final_hidden(target, full, tc, start_pos=0)[0]
        dl = draft.forward(full, dc, start_pos=0)[0]
    n = full.shape[1]
    k = heads.n_heads
    hh = h[p0 - 1:n - 1 - k]                     # positions with k labels left
    per_head = []
    for i in range(k):
        with torch.no_grad():
            pred = heads.logits(hh, i).argmax(-1)
        lab = full[0, p0 + 1 + i: p0 + 1 + i + hh.shape[0]]
        per_head.append(round(float((pred == lab).float().mean()), 4))
    tl = (h[p0 - 1:-1] @ target.lm_head.T).argmax(-1)
    agree = float((tl == dl[p0 - 1:-1].argmax(-1)).float().mean())
    return {"head_top1_on_this_text": per_head,
            "draft_top1_agreement": round(agree, 4),
            "positions": int(hh.shape[0])}


def section_c(target, draft, tok, trained):
    print("\n== C. acceptance in the real loop ==")
    tc = S.make_cache(target, max_len=1024)
    dc = S.make_cache(draft, max_len=1024)
    md = S.ModelDrafter(draft, max_len=1024)
    rows = []
    for name in PROMPTS:
        ids = S.chat_ids(tok, S.WORKLOADS[name])
        base = S.greedy_decode(target, tc, ids, max_new=MAX_NEW)
        base_tok_s = MAX_NEW / base["decode_s"]
        entry = {"workload": name, "baseline_tok_s": round(base_tok_s, 3),
                 "baseline_decode_s": round(base["decode_s"], 3),
                 "methods": {}}
        runs = {
            "medusa_gt": M.medusa_decode(target, trained["gt"], tc, ids,
                                         max_new=MAX_NEW),
            "medusa_sd": M.medusa_decode(target, trained["sd"], tc, ids,
                                         max_new=MAX_NEW),
            "external_draft": S.speculative_greedy(target, md, tc, ids, k=K,
                                                   max_new=MAX_NEW),
        }
        for label, r in runs.items():
            tok_s = r["produced"] / r["decode_s"]
            entry["methods"][label] = {
                "identical": base["tokens"] == r["tokens"],
                "acceptance": round(r["conditional_acceptance"], 4),
                "alpha": round(r["tokens_per_iter"], 3),
                "per_pos_hits": r["per_pos_hits"],
                "iters": r["iters"],
                "decode_s": round(r["decode_s"], 3),
                "draft_s": round(r["draft_s"], 4),
                "draft_share": round(r["draft_s"] / r["decode_s"], 4),
                "tok_s": round(tok_s, 3),
                "speedup": round(tok_s / base_tok_s, 3),
                "itl_p99": round(S.pct(r["itl"], 99), 4),
            }
            print(f"  {name:10s} {label:15s} accept "
                  f"{entry['methods'][label]['acceptance']:.3f}  alpha "
                  f"{entry['methods'][label]['alpha']:.2f}  "
                  f"{entry['methods'][label]['speedup']:.2f}x  draft "
                  f"{entry['methods'][label]['draft_share']:.1%}")
        entry["probe"] = head_probe(target, draft, trained["sd"], tc, dc,
                                    ids, base["tokens"])
        print(f"  {name:10s} {'probe':15s} head top-1 on this text "
              f"{entry['probe']['head_top1_on_this_text']}  vs the 0.5B "
              f"draft's {entry['probe']['draft_top1_agreement']:.3f}")
        rows.append(entry)
    F["C"] = {"k": K, "max_new": MAX_NEW, "rows": rows}


# ---------------------------------------------------------------------------
# D. cost
# ---------------------------------------------------------------------------


def section_d(target, draft, tok, trained):
    print("\n== D. what a draft token costs ==")
    ids = S.chat_ids(tok, S.WORKLOADS["chat"])
    n = int(ids.shape[1])
    tc, dc = S.make_cache(target), S.make_cache(draft)
    with torch.no_grad():
        h = M.final_hidden(target, ids, tc, start_pos=0)[0, -1].clone()
    dc.reset()
    draft.forward(ids, dc, start_pos=0)

    blk = torch.full((1, 1), 9707, dtype=torch.long)

    def target_pass():
        tc.truncate(n)
        target.forward(blk, tc, start_pos=n)

    def draft_pass():
        dc.truncate(n)
        draft.forward(blk, dc, start_pos=n)

    def medusa_pass():
        with torch.no_grad():
            trained["sd"].propose(h)

    t = S.interleaved({"target": target_pass, "draft": draft_pass,
                       "medusa": medusa_pass}, rounds=5, warmup=2)
    F["D"] = {
        "target_pass_ms": round(t["target"] * 1000, 2),
        "draft_pass_ms": round(t["draft"] * 1000, 2),
        "medusa_all_heads_ms": round(t["medusa"] * 1000, 2),
        "external_cost_ratio": round(t["draft"] / t["target"], 4),
        "medusa_cost_ratio_per_token": round(
            t["medusa"] / N_HEADS / t["target"], 5),
        "cheaper_by": round(t["draft"] * N_HEADS / t["medusa"], 2),
    }
    print(f"  one target pass          {F['D']['target_pass_ms']:7.1f} ms")
    print(f"  one 0.5B draft pass      {F['D']['draft_pass_ms']:7.1f} ms  "
          f"(cost_ratio {F['D']['external_cost_ratio']:.3f})")
    print(f"  all {N_HEADS} Medusa heads    "
          f"{F['D']['medusa_all_heads_ms']:7.1f} ms  (cost_ratio/token "
          f"{F['D']['medusa_cost_ratio_per_token']:.4f})")
    print(f"  -> drafting is {F['D']['cheaper_by']:.1f}x cheaper")

    # What acceptance would the heads need to overtake the external draft?
    # Both pay the same verification overhead, so it cancels; only the draft
    # cost differs. Solve alpha_med / (vo + 3*c_med) = alpha_ext / (vo + 3*c_ext).
    vo = 1.14                          # measured in project 26, width 4
    c_med = F["D"]["medusa_cost_ratio_per_token"]
    c_ext = F["D"]["external_cost_ratio"]
    ext = F["C"]["rows"][0]["methods"]["external_draft"]
    need_alpha = ext["alpha"] * (vo + N_HEADS * c_med) / (vo + N_HEADS * c_ext)
    # invert alpha = 1 + a + a^2 + a^3 numerically
    a_lo, a_hi = 0.0, 1.0
    for _ in range(60):
        a = (a_lo + a_hi) / 2
        if sum(a ** i for i in range(N_HEADS + 1)) < need_alpha:
            a_lo = a
        else:
            a_hi = a
    F["D"]["breakeven"] = {
        "verify_overhead_used": vo,
        "external_alpha": ext["alpha"],
        "external_acceptance": ext["acceptance"],
        "medusa_alpha_needed": round(need_alpha, 3),
        "medusa_acceptance_needed": round((a_lo + a_hi) / 2, 4),
    }
    print(f"  to tie the external draft on chat, the heads need alpha "
          f"{need_alpha:.2f} (acceptance {(a_lo+a_hi)/2:.2f}); the external "
          f"draft has alpha {ext['alpha']:.2f} at acceptance "
          f"{ext['acceptance']:.2f}")

    # And the same arithmetic at production scale. A Medusa head is one
    # output projection; a 70B target forward pass is ~46x a 1.5B one, while
    # the head's cost grows only with hidden size.
    scen = []
    for label, c_m, c_e in (("this box (1.5B target)", c_med, c_ext),
                            ("7B target, 1B draft", c_med / 4.7, 1 / 7),
                            ("70B target, 1B draft", c_med / 46.7, 1 / 70)):
        scen.append({
            "label": label,
            "medusa_cost_ratio": round(c_m, 5),
            "external_cost_ratio": round(c_e, 5),
            "medusa_acceptance_needed": None,
        })
        need = ext["alpha"] * (vo + N_HEADS * c_m) / (vo + N_HEADS * c_e)
        lo, hi = 0.0, 1.0
        for _ in range(60):
            a = (lo + hi) / 2
            if sum(a ** i for i in range(N_HEADS + 1)) < need:
                lo = a
            else:
                hi = a
        scen[-1]["medusa_alpha_needed"] = round(need, 3)
        scen[-1]["medusa_acceptance_needed"] = round((lo + hi) / 2, 4)
        print(f"    {label:26s} heads need acceptance "
              f"{scen[-1]['medusa_acceptance_needed']:.3f} to tie")
    F["D"]["scenarios"] = scen


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = json.load(open(os.path.join(OUT, "findings.json")))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.0))

    cols = {"gt": "#c0392b", "sd": "#27ae60"}
    for key, lab in (("gt", "ground-truth labels"),
                     ("sd", "self-distilled labels")):
        c = f["A"]["train"][key]["curve"]
        ax[0].plot([p["step"] for p in c], [p["loss"] for p in c], "o-",
                   color=cols[key], label=lab)
    ax[0].set_xlabel("training step")
    ax[0].set_ylabel("mean cross-entropy over 3 heads")
    ax[0].set_title("A. training the heads")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    hs = range(1, len(f["A"]["train"]["sd"]["holdout_top1"]["sd"]) + 1)
    for key, lab in (("gt", "trained on corpus"), ("sd", "trained on model")):
        v = f["A"]["train"][key]["holdout_top1"]["sd"]
        ax[1].plot(list(hs), v, "o-", color=cols[key], label=lab)
    ax[1].set_xticks(list(hs))
    ax[1].set_xlabel("Medusa head (1 = two tokens ahead)")
    ax[1].set_ylabel("held-out top-1 vs the model's own token")
    ax[1].set_title("B. predicting further ahead is harder")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    rows = f["C"]["rows"]
    labels = ["medusa_gt", "medusa_sd", "external_draft"]
    lcol = {"medusa_gt": "#c0392b", "medusa_sd": "#27ae60",
            "external_draft": "#2471a3"}
    xs = range(len(rows))
    for j, lab in enumerate(labels):
        ax[2].bar([i + (j - 1) * 0.27 for i in xs],
                  [r["methods"][lab]["alpha"] for r in rows], 0.27,
                  color=lcol[lab], label=lab)
    ax[2].axhline(1.0, color="k", lw=1, ls="--")
    ax[2].set_xticks(list(xs))
    ax[2].set_xticklabels([r["workload"] for r in rows], fontsize=8)
    ax[2].set_ylabel("tokens per target forward pass")
    ax[2].set_title("C. alpha: heads vs a whole model")
    ax[2].legend(fontsize=7)

    for j, lab in enumerate(labels):
        ax[3].bar([i + (j - 1) * 0.27 for i in xs],
                  [r["methods"][lab]["speedup"] for r in rows], 0.27,
                  color=lcol[lab], label=lab)
    ax[3].axhline(1.0, color="k", lw=1, ls="--")
    ax[3].set_xticks(list(xs))
    ax[3].set_xticklabels([r["workload"] for r in rows], fontsize=8)
    ax[3].set_ylabel("speedup vs plain decoding")
    ax[3].set_title("C. wall clock: cheap loses to accurate")
    ax[3].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "medusa_heads.png"), dpi=110)
    print("wrote outputs/medusa_heads.png")


def main():
    if "--plot" in sys.argv:
        plot()
        return
    t0 = time.time()
    target, draft, tok, _ = S.load_pair()
    F["setup"] = {"target": S.TARGET_ID, "draft": S.DRAFT_ID,
                  "n_heads": N_HEADS, "k": K, "max_new": MAX_NEW,
                  "train_steps": STEPS, "n_tokens": N_TOKENS,
                  "threads": S.N_THREADS}
    trained = section_ab(target, tok)
    section_c(target, draft, tok, trained)
    section_d(target, draft, tok, trained)
    # The trained weights are deliberately not committed: 3 heads x 2
    # matrices x 1536^2 in fp32 is 56 MB per label set, which does not belong
    # in a teaching repo. `run.py` retrains them in about three minutes.
    F["wall_s"] = round(time.time() - t0, 1)
    json.dump(F, open(os.path.join(OUT, "findings.json"), "w"), indent=2)
    plot()
    print(f"\ntotal {F['wall_s']:.0f} s")


if __name__ == "__main__":
    main()
