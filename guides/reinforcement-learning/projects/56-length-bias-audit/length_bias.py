"""Project 56 — length-bias audit: how a tiny rater habit inflates answers.

Builds preference data the way RLHF labs do — sample two completions per
prompt, ask a rater which is better — and runs DPO twice on it:

  arm "biased":   when the two completions TIE on correctness, the rater
                  picks the LONGER one (a mild, realistic habit)
  arm "control":  tied pairs are simply dropped; only correctness decides

Both arms share every pair where correctness differs, so any behavioral gap
between them is caused purely by the tie-breaking habit. The audit tracks the
completion-length distribution EPOCH BY EPOCH: early training improves both
arms identically; keep optimizing and the biased arm drifts into a 'verbose'
answer style (a chain of thought that pointlessly restates its final answer)
while the control arm never does. The task (project 55's four-number sums)
offers three answer styles — short direct, long CoT, longer verbose CoT — so
the model has a way to BE more verbose if training rewards it.

Outputs: length/verbose-rate trajectories, length histograms, pair statistics.
Run:  python3 length_bias.py            (~8 min on CPU)
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
sys.path.insert(0, str(HERE.parent / "55-rlvr-on-math"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import cot_lib as C  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

P_COT = 0.35
P_VERBOSE = 0.20
SFT_STEPS = 1200
N_PROMPTS = 2500
BETA = 0.1
LR = 1e-4
EPOCHS = 6
BS = 64


def build_pairs(model, seed=0):
    """Sample 2 completions per prompt at temp 1.0 and label them two ways."""
    rng = random.Random(seed)
    probs = [C.problem(rng) for _ in range(N_PROMPTS)]
    two = probs + probs
    torch.manual_seed(seed)
    comps = C.sample_batch(model, two, temperature=1.0)
    biased, control = [], []
    for i, ops in enumerate(probs):
        c1, c2 = comps[i], comps[i + N_PROMPTS]
        ok1, ok2 = C.is_correct(ops, c1), C.is_correct(ops, c2)
        if ok1 != ok2:                       # correctness decides — both arms
            pair = (ops, c1, c2) if ok1 else (ops, c2, c1)
            biased.append(pair); control.append(pair)
        elif len(c1) != len(c2):             # a tie — only the biased rater votes
            pair = (ops, c1, c2) if len(c1) > len(c2) else (ops, c2, c1)
            biased.append(pair)
    return biased, control


def dpo_with_trajectory(sft, ref, pairs, tag):
    """DPO for EPOCHS epochs, evaluating after each — the drift, in motion."""
    policy = copy.deepcopy(sft)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    traj = [dict(epoch=0, **C.evaluate(policy))]
    for ep in range(EPOCHS):
        order = list(range(len(pairs)))
        random.Random(ep).shuffle(order)
        for i in range(0, len(order), BS):
            batch = [pairs[j] for j in order[i:i + BS]]
            ch_s, ch_m, rj_s, rj_m = [], [], [], []
            for ops, cho, rej in batch:
                s, m = C.seq_from(ops, cho or ";"); ch_s.append(s); ch_m.append(m)
                s, m = C.seq_from(ops, rej or ";"); rj_s.append(s); rj_m.append(m)
            lp_ch = C.completion_logprobs(policy, ch_s, ch_m)
            lp_rj = C.completion_logprobs(policy, rj_s, rj_m)
            with torch.no_grad():
                ref_ch = C.completion_logprobs(ref, ch_s, ch_m)
                ref_rj = C.completion_logprobs(ref, rj_s, rj_m)
            logits = BETA * ((lp_ch - ref_ch) - (lp_rj - ref_rj))
            loss = -F.logsigmoid(logits).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        e = C.evaluate(policy)
        traj.append(dict(epoch=ep + 1, **e))
        print(f"[{tag}] epoch {ep + 1}: acc {e['acc']:.3f}  len {e['length']:.1f}  "
              f"direct {e['direct']:.2f} cot {e['cot']:.2f} "
              f"verbose {e['verbose']:.2f}", flush=True)
    return policy, traj


@torch.no_grad()
def lengths_of(model, n=300):
    comps = C.sample_batch(model, C.eval_problems(n))
    return [len(c) for c in comps]


def main():
    t0 = time.time()
    print("== SFT with three answer styles (direct / CoT / verbose CoT) ==", flush=True)
    sft = C.new_model(seed=0)
    C.train_sft(sft, steps=SFT_STEPS, p_cot=P_COT, p_verbose=P_VERBOSE,
                seed=0, log_every=200)
    ref = copy.deepcopy(sft)
    for p in ref.parameters():
        p.requires_grad_(False)
    e = C.evaluate(sft)
    print(f"SFT: acc {e['acc']:.3f}  len {e['length']:.1f}  direct {e['direct']:.2f} "
          f"cot {e['cot']:.2f} verbose {e['verbose']:.2f}", flush=True)

    print("== building preference pairs from the model's own samples ==", flush=True)
    biased_pairs, control_pairs = build_pairs(sft)
    tie_pairs = len(biased_pairs) - len(control_pairs)
    mean_len = lambda pairs, i: sum(len(p[i]) for p in pairs) / len(pairs)
    print(f"correctness-decided pairs: {len(control_pairs)}  "
          f"tie pairs (biased arm only): {tie_pairs}", flush=True)
    for name, pairs in (("biased", biased_pairs), ("control", control_pairs)):
        print(f"{name} arm: chosen len {mean_len(pairs, 1):.1f} vs "
              f"rejected len {mean_len(pairs, 2):.1f}", flush=True)

    models, trajs = {}, {}
    models["biased"], trajs["biased"] = dpo_with_trajectory(
        sft, ref, biased_pairs, "biased")
    models["control"], trajs["control"] = dpo_with_trajectory(
        sft, ref, control_pairs, "control")

    with open(OUT / "length_bias.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "epoch", "greedy_accuracy", "mean_length",
                    "direct_rate", "cot_rate", "verbose_rate"])
        for arm, traj in trajs.items():
            for e in traj:
                w.writerow([arm, e["epoch"], f"{e['acc']:.4f}",
                            f"{e['length']:.2f}", f"{e['direct']:.4f}",
                            f"{e['cot']:.4f}", f"{e['verbose']:.4f}"])

    labels = {"biased": "biased rater (ties -> prefer longer)",
              "control": "control rater (ties dropped)"}
    colors = {"biased": ps.SERIES[2], "control": ps.SERIES[0]}

    # ---- figure 1: the drift, epoch by epoch (length + verbose + accuracy)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7.2, 8.2), dpi=110,
                                        sharex=True)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in (ax1, ax2, ax3):
        ps.style_axes(ax)
    for arm, traj in trajs.items():
        eps = [e["epoch"] for e in traj]
        ax1.plot(eps, [e["length"] for e in traj], color=colors[arm],
                 linewidth=2, label=labels[arm])
        ax2.plot(eps, [e["verbose"] for e in traj], color=colors[arm],
                 linewidth=2, label=labels[arm])
        ax3.plot(eps, [e["acc"] for e in traj], color=colors[arm],
                 linewidth=2, label=labels[arm])
    ax1.set_ylabel("mean completion length", color=ps.INK_SECONDARY, fontsize=10)
    ax1.legend(frameon=False, fontsize=9)
    ax2.set_ylabel("verbose-style rate", color=ps.INK_SECONDARY, fontsize=10)
    ax3.set_ylabel("greedy accuracy", color=ps.INK_SECONDARY, fontsize=10)
    ax3.set_xlabel("DPO epoch", color=ps.INK_SECONDARY, fontsize=10)
    ax1.set_title("Same data, one tie-breaking habit: only the biased arm inflates",
                  color=ps.INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "drift_trajectory.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'drift_trajectory.png'}")

    # ---- figure 2: final length distributions
    fig, ax = ps.new_axes()
    bins = range(0, 44, 2)
    series = [("SFT (before DPO)", sft, ps.INK_MUTED),
              ("biased rater, epoch 6", models["biased"], ps.SERIES[2]),
              ("control rater, epoch 6", models["control"], ps.SERIES[0])]
    for name, m, color in series:
        ax.hist(lengths_of(m), bins=bins, histtype="step", linewidth=2,
                color=color, label=name, density=True)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "One tie-breaking habit shifts the whole length distribution",
              "greedy completion length (chars)", "density",
              OUT / "length_histogram.png")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
