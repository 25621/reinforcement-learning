"""Project 53 — DPO on the same preference data PPO-RLHF used.

Direct Preference Optimization: a single supervised loss on (prompt, chosen,
rejected) triples that provably optimizes the same KL-regularized objective
as PPO-RLHF — with no reward model, no rollouts, no RL loop. Two arms differ
only in how the rejected answer is built:

  arm "random":    rejected is a random wrong number  -> DPO helps
  arm "nearmiss":  rejected is off by one             -> DPO degenerates

The near-miss arm exposes DPO's classic failure mode: the loss only cares
about the GAP between chosen and rejected log-probs, so when the two texts
are nearly identical the easiest way to widen the gap is to push BOTH down —
including the correct answer.

Outputs: accuracy curves per arm, chosen/rejected log-prob trajectories.
Run:  python3 dpo.py            (~2 min on CPU)
"""

import copy
import csv
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "50-sft-a-small-base-model"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import rlhf_lib as L  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

BETA = 0.1
LR = 3e-4
EPOCHS = 3
BS = 64
N_PAIRS = 3000
EVAL_EVERY = 20


def dpo_loss(policy, ref, batch, beta=BETA):
    """The DPO loss on a batch of (a, b, chosen, rejected) pairs.

    logits = beta * [ (logp_pi(ch) - logp_ref(ch)) - (logp_pi(rej) - logp_ref(rej)) ]
    loss   = -log sigmoid(logits)
    """
    ch_seqs, ch_masks, rj_seqs, rj_masks = [], [], [], []
    for a, b, cho, rej in batch:
        s, m = L.seq_from(a, b, cho); ch_seqs.append(s); ch_masks.append(m)
        s, m = L.seq_from(a, b, rej); rj_seqs.append(s); rj_masks.append(m)
    lp_ch = L.completion_logprobs(policy, ch_seqs, ch_masks)
    lp_rj = L.completion_logprobs(policy, rj_seqs, rj_masks)
    with torch.no_grad():
        ref_ch = L.completion_logprobs(ref, ch_seqs, ch_masks)
        ref_rj = L.completion_logprobs(ref, rj_seqs, rj_masks)
    logits = beta * ((lp_ch - ref_ch) - (lp_rj - ref_rj))
    return -F.logsigmoid(logits).mean(), lp_ch.mean(), lp_rj.mean()


def reference_dpo_loss(policy, ref, batch, beta=BETA):
    """Independent, deliberately-naive reimplementation (one pair at a time)
    used only to verify dpo_loss is exactly right."""
    total = 0.0
    for a, b, cho, rej in batch:
        vals = {}
        for name, model in (("pi", policy), ("ref", ref)):
            for side, comp in (("ch", cho), ("rj", rej)):
                ids, mask = L.seq_from(a, b, comp)
                x = torch.tensor([ids])
                logp = F.log_softmax(model(x[:, :-1]), -1)
                tok = logp[0].gather(-1, x[0, 1:, None]).squeeze(-1)
                sel = torch.tensor(mask[1:], dtype=torch.bool)
                vals[f"{name}_{side}"] = tok[sel].sum()
        lg = beta * ((vals["pi_ch"] - vals["ref_ch"]) - (vals["pi_rj"] - vals["ref_rj"]))
        total = total + (-F.logsigmoid(lg))
    return total / len(batch)


def train_dpo(sft, ref, pairs, tag):
    policy = copy.deepcopy(sft)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    hist = [dict(step=0, acc=L.accuracy(policy), lp_ch=float("nan"),
                 lp_rj=float("nan"))]
    step = 0
    for ep in range(EPOCHS):
        order = list(range(len(pairs)))
        random.Random(ep).shuffle(order)
        for i in range(0, len(order), BS):
            batch = [pairs[j] for j in order[i:i + BS]]
            loss, lp_ch, lp_rj = dpo_loss(policy, ref, batch)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1
            if step % EVAL_EVERY == 0:
                hist.append(dict(step=step, acc=L.accuracy(policy),
                                 lp_ch=float(lp_ch), lp_rj=float(lp_rj)))
                print(f"[{tag}] step {step:3d}  loss {loss.item():.4f}  "
                      f"acc {hist[-1]['acc']:.3f}  "
                      f"logp chosen {lp_ch:+.2f} rejected {lp_rj:+.2f}", flush=True)
    return policy, hist


def main():
    t0 = time.time()
    print("== building partial SFT policy ==", flush=True)
    sft = L.sft_model(seed=0)
    ref = copy.deepcopy(sft)
    for p in ref.parameters():
        p.requires_grad_(False)
    acc0 = L.accuracy(sft)
    print(f"start greedy accuracy {acc0:.3f}", flush=True)

    # sanity: batched loss == naive per-pair loss, to the last bit
    probe = L.preference_pairs(8, random.Random(7))
    a1, _, _ = dpo_loss(sft, ref, probe)
    a2 = reference_dpo_loss(sft, ref, probe)
    print(f"loss check: batched {a1.item():.10f}  naive {a2.item():.10f}  "
          f"diff {abs(a1.item() - a2.item()):.2e}", flush=True)

    arms = {}
    times = {}
    for kind in ("random", "nearmiss"):
        pairs = L.preference_pairs(N_PAIRS, random.Random(0), kind=kind)
        t = time.time()
        _, hist = train_dpo(sft, ref, pairs, tag=kind)
        times[kind] = time.time() - t
        arms[kind] = hist
        print(f"[{kind}] final acc {hist[-1]['acc']:.3f} "
              f"({times[kind]:.0f}s)", flush=True)

    with open(OUT / "dpo_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "step", "greedy_accuracy", "logp_chosen", "logp_rejected"])
        for arm, hist in arms.items():
            for h in hist:
                w.writerow([arm, h["step"], f"{h['acc']:.4f}",
                            f"{h['lp_ch']:.4f}", f"{h['lp_rj']:.4f}"])

    labels = {"random": "rejected = random wrong",
              "nearmiss": "rejected = off by one"}
    colors = {"random": ps.SERIES[0], "nearmiss": ps.SERIES[2]}

    fig, ax = ps.new_axes()
    for arm, hist in arms.items():
        ax.plot([h["step"] for h in hist], [h["acc"] for h in hist],
                color=colors[arm], linewidth=2, label=labels[arm])
    ax.axhline(acc0, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(2, acc0 + 0.012, "SFT start", color=ps.INK_MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "DPO helps with distinct pairs and degenerates on near-identical ones",
              "DPO step", "greedy exact-match accuracy", OUT / "dpo_accuracy.png")

    fig, ax = ps.new_axes()
    for arm, hist in arms.items():
        st = [h["step"] for h in hist[1:]]
        ax.plot(st, [h["lp_ch"] for h in hist[1:]], color=colors[arm],
                linewidth=2, label=f"{labels[arm]} — chosen")
        ax.plot(st, [h["lp_rj"] for h in hist[1:]], color=colors[arm],
                linewidth=2, linestyle="--", label=f"{labels[arm]} — rejected")
    ax.legend(frameon=False, fontsize=8)
    ps.finish(fig, ax, "The near-miss arm pushes BOTH answers down — the DPO degeneracy",
              "DPO step", "mean completion log-prob", OUT / "dpo_logps.png")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
