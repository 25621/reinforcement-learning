"""Project 51 — train a reward model on pairwise preferences.

Trains a Bradley-Terry reward model on (prompt, chosen, rejected) pairs where
the rejected answer is a random wrong number, then probes it on held-out pairs
— including near-miss pairs (off by one) it never saw — to expose the blind
spot that later projects (52, 57) will watch a policy exploit.

Outputs: held-out accuracy bars, reward-margin-vs-error-distance curve.
Run:  python3 rm.py            (~1 min on CPU)
"""

import csv
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "50-sft-a-small-base-model"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import rlhf_lib as L  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


@torch.no_grad()
def margin_by_distance(rm, n=300, max_d=12, seed=99):
    """Mean reward margin r(correct) - r(wrong) as the wrong answer moves
    further from the truth. Distance 1 = a near miss, distance 12 = obviously
    wrong."""
    rng = random.Random(seed)
    rows = []
    for d in range(1, max_d + 1):
        pairs = []
        for _ in range(n):
            a, b, c = L.sample_problem(rng)
            w = c + d if (c - d < 0 or rng.random() < 0.5) else c - d
            pairs.append((a, b, f"{c};", f"{w};"))
        ch, rj = L.pair_seqs(pairs)
        m = (rm.reward(ch) - rm.reward(rj))
        acc = float((m > 0).float().mean())
        rows.append((d, float(m.mean()), acc))
    return rows


def main():
    t0 = time.time()
    rng = random.Random(0)

    print("== training reward model on 3000 random-wrong pairs ==", flush=True)
    train_pairs = L.preference_pairs(3000, rng, kind="random")
    rm = L.train_reward_model(train_pairs, epochs=6, seed=0, log_every=1)

    heldout = {
        "random wrong\n(held out)": L.preference_pairs(1000, random.Random(1), kind="random"),
        "near miss\n(off by one)": L.preference_pairs(1000, random.Random(2), kind="nearmiss"),
    }
    accs = {}
    for name, pairs in heldout.items():
        accs[name] = L.rm_pair_accuracy(rm, pairs)
        print(f"held-out accuracy on {name.split(chr(10))[0]}: {accs[name]:.3f}")

    rows = margin_by_distance(rm)
    with open(OUT / "rm_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["probe", "value"])
        for k, v in accs.items():
            w.writerow([k.replace("\n", " "), f"{v:.4f}"])
        w.writerow(["distance", "mean_margin", "accuracy"])
        w.writerows((d, f"{m:.4f}", f"{a:.4f}") for d, m, a in rows)

    # ------- figure 1: held-out accuracy by probe type
    fig, ax = ps.new_axes(6.4, 4.0)
    names = list(accs)
    vals = [accs[n] for n in names]
    ax.bar(names, vals, color=[ps.SERIES[0], ps.SERIES[2]], width=0.5)
    ax.axhline(0.5, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(1.32, 0.51, "coin flip", color=ps.INK_MUTED, fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color=ps.INK, fontsize=10)
    ax.set_ylim(0, 1.0)
    ps.finish(fig, ax, "The reward model ranks pairs it was trained on — and guesses on near misses",
              "", "held-out pair accuracy", OUT / "rm_accuracy.png")

    # ------- figure 2: margin vs distance (the blind spot, continuously)
    fig, ax = ps.new_axes()
    ds = [d for d, _, _ in rows]
    ms = [m for _, m, _ in rows]
    ax.plot(ds, ms, color=ps.SERIES[0], linewidth=2, marker="o", markersize=4)
    ax.axhline(0.0, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.annotate("near misses: margin ~ 0\n(the RM can't tell)", (1, ms[0]),
                xytext=(1.4, max(ms) * 0.55), color=ps.INK_SECONDARY, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=ps.INK_MUTED))
    ps.finish(fig, ax, "Reward margin grows with how wrong the rejected answer is",
              "distance of wrong answer from the truth",
              "mean reward margin  r(correct) − r(wrong)", OUT / "rm_margin.png")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
