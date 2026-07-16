"""Project 57 — reward hacking, provoked on purpose.

Over-trains a policy against a learned reward model with the KL leash ON
(beta 0.5) vs OFF (beta 0), then characterizes what the unleashed policy
actually produces. The reward model was trained only on random-wrong pairs,
so plausible wrong answers are a blind spot — and the unleashed policy
migrates its probability mass onto whatever wrong answers the RM happens to
over-score, while the RM's opinion of the policy never budges.

Outputs: proxy-vs-truth curves, error-distance histograms, sample tables.
Run:  python3 reward_hacking.py            (~2.5 min on CPU)
"""

import copy
import csv
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "50-sft-a-small-base-model"))
sys.path.insert(0, str(HERE.parent / "52-ppo-style-rlhf"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import rlhf_lib as L  # noqa: E402
from ppo_rlhf import ppo_train  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

ITERS = 40    # deliberately long: we WANT to over-optimize


def error_profile(model, n=400):
    """For greedy completions: how far is each answer from the truth?
    Returns dict distance -> fraction (distance 0 = correct, 99 = not a number)."""
    probs = L.eval_problems(n)
    comps = L.sample_batch(model, probs)
    dist = {}
    for (a, b), c in zip(probs, comps):
        head = c.split(";")[0]
        try:
            d = min(abs(int(head) - (a + b)), 10)
        except ValueError:
            d = 99
        dist[d] = dist.get(d, 0) + 1 / n
    return dist, list(zip(probs, comps))


def main():
    t0 = time.time()
    print("== training the exploitable reward model ==", flush=True)
    rm = L.train_reward_model(L.preference_pairs(3000, random.Random(0)),
                              epochs=6, seed=0, log_every=2)
    for p in rm.parameters():
        p.requires_grad_(False)

    sft = L.sft_model(seed=0)
    ref = copy.deepcopy(sft)
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"start greedy accuracy {L.accuracy(sft):.3f}", flush=True)

    def rm_reward(ab, comps, seqs):
        with torch.no_grad():
            return rm.reward(seqs)

    policies, arms = {}, {}
    for tag, beta in (("leash", 0.5), ("no_leash", 0.0)):
        pol = copy.deepcopy(sft)
        arms[tag] = ppo_train(pol, ref, rm_reward, rm, iters=ITERS, beta=beta,
                              seed=1, tag=f"{tag} beta={beta}")
        policies[tag] = pol

    with open(OUT / "hacking_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "iter", "rm_reward", "kl", "greedy_accuracy"])
        for arm, hist in arms.items():
            for h in hist:
                w.writerow([arm, h["iter"], f"{h['rm_reward']:.4f}",
                            f"{h['kl']:.5f}", f"{h['acc']:.4f}"])

    labels = {"leash": "beta = 0.5 (KL leash on)", "no_leash": "beta = 0 (no leash)"}
    colors = {"leash": ps.SERIES[0], "no_leash": ps.SERIES[2]}

    # ---- figure 1: proxy vs truth, the money plot (two panels, shared x)
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.2), dpi=110,
                                   sharex=True)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in (ax1, ax2):
        ps.style_axes(ax)
    for arm, hist in arms.items():
        its = [h["iter"] for h in hist]
        ax1.plot(its, [h["acc"] for h in hist], color=colors[arm], linewidth=2,
                 label=labels[arm])
        ax2.plot(its, [h["rm_reward"] for h in hist], color=colors[arm],
                 linewidth=2, label=labels[arm])
    ax1.set_ylabel("TRUE accuracy (verifier)", color=ps.INK_SECONDARY, fontsize=10)
    ax1.legend(frameon=False, fontsize=9)
    ax2.set_ylim(1.5, 3.0)
    ax2.set_ylabel("RM score (the proxy)", color=ps.INK_SECONDARY, fontsize=10)
    ax2.set_xlabel("PPO iteration", color=ps.INK_SECONDARY, fontsize=10)
    ax1.set_title("The proxy never notices: RM score stays flat while truth collapses",
                  color=ps.INK, fontsize=12, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "proxy_vs_truth.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'proxy_vs_truth.png'}")

    # ---- figure 2: where did the answers go? error-distance histogram
    fig, ax = ps.new_axes()
    prof_sft, _ = error_profile(sft)
    prof_hack, samples = error_profile(policies["no_leash"])
    with open(OUT / "error_profile.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance", "sft_fraction", "hacked_fraction"])
        for d in sorted(set(prof_sft) | set(prof_hack)):
            w.writerow([d, f"{prof_sft.get(d, 0):.4f}", f"{prof_hack.get(d, 0):.4f}"])
    xs = list(range(0, 11))
    width = 0.4
    ax.bar([x - width / 2 for x in xs], [prof_sft.get(x, 0) for x in xs],
           width=width, color=ps.SERIES[0], label="SFT policy (before)")
    ax.bar([x + width / 2 for x in xs], [prof_hack.get(x, 0) for x in xs],
           width=width, color=ps.SERIES[2], label="hacked policy (beta = 0)")
    ax.set_xticks(xs)
    ax.set_xticklabels(["correct"] + [str(x) for x in xs[1:-1]] + ["10+"])
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "The hack: answers leave 'correct' for wherever the judge over-scores",
              "distance of greedy answer from the truth", "fraction of prompts",
              OUT / "error_histogram.png")

    with open(OUT / "hacked_samples.txt", "w") as f:
        f.write("greedy completions of the beta=0 policy after 40 iterations\n")
        f.write("(RM score in brackets; the RM was never taught near-misses are bad)\n\n")
        for (a, b), c in samples[:16]:
            ids, _ = L.seq_from(a, b, c or ";")
            with torch.no_grad():
                r = float(rm.reward([ids])[0])
            ok = "ok " if L.is_correct(a, b, c) else "WRONG"
            f.write(f"{a}+{b}= -> {c:<6} truth {a + b:<4} {ok}  [rm {r:+.2f}]\n")
    print((OUT / "hacked_samples.txt").read_text())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
