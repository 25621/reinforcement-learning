"""Project 50 — SFT a small base model.

Builds a 'base model' (fluent in the task's format but trained on wrong
answers, standing in for raw web pretraining), then supervised-fine-tunes it
on correct demonstrations and tracks exact-match accuracy while it learns.

Outputs: accuracy curve, base-vs-SFT sample table, per-step CSV.
Run:  python3 sft.py            (~2 min on CPU)
"""

import csv
import sys
import time
from pathlib import Path

import rlhf_lib as L

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

SFT_STEPS = 1500
EVAL_EVERY = 75


def show(model, probs):
    comps = L.sample_batch(model, probs)
    return [(f"{a}+{b}=", g, "yes" if L.is_correct(a, b, g) else "no")
            for (a, b), g in zip(probs, comps)]


def main():
    t0 = time.time()

    print("== stage 1: 'pretrained' base (150 steps on wrong-answer text) ==")
    base = L.base_model(seed=0)
    base_acc = L.accuracy(base)
    print(f"base accuracy {base_acc:.3f}")

    demo = L.eval_problems(8, seed=7)
    base_rows = show(base, demo)

    print("== stage 2: SFT on correct demonstrations ==")
    curve = [(0, base_acc)]
    model = L.base_model(seed=0)          # start SFT FROM the base model
    L.train_sft(model, steps=SFT_STEPS, seed=1, log_every=100,
                eval_every=EVAL_EVERY, curve=curve)
    sft_rows = show(model, demo)

    with open(OUT / "sft_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "greedy_accuracy"])
        w.writerows(curve)

    with open(OUT / "samples.txt", "w") as f:
        f.write("prompt        base completion   SFT completion   correct(base/sft)\n")
        for (p, gb, cb), (_, gs, cs) in zip(base_rows, sft_rows):
            f.write(f"{p:<13} {gb:<17} {gs:<16} {cb}/{cs}\n")
    print((OUT / "samples.txt").read_text())

    steps = [s for s, _ in curve]
    accs = [a for _, a in curve]
    fig, ax = ps.new_axes()
    ax.plot(steps, accs, color=ps.SERIES[0], linewidth=2)
    ax.scatter([0], [base_acc], color=ps.SERIES[2], zorder=5)
    ax.annotate("base model (0 correct)", (0, base_acc), xytext=(30, base_acc + 0.06),
                color=ps.INK_SECONDARY, fontsize=9)
    ax.set_ylim(-0.03, 1.0)
    ps.finish(fig, ax, "SFT turns a fluent-but-wrong base model into a working assistant",
              "SFT step", "greedy exact-match accuracy (400 held-out prompts)",
              OUT / "sft_accuracy.png")

    final = L.accuracy(model)
    print(f"final SFT accuracy {final:.3f}")
    # the partial policy the RL projects will start from (stopped pre-grok)
    partial = L.sft_model(seed=0)
    print(f"partial sft_model() accuracy {L.accuracy(partial):.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
