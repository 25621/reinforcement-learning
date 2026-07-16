"""Project 55 — RLVR on math: watch the chain of thought take over.

The SFT stage runs in two phases that mimic a realistic pretraining mix:
first the model masters the chain-of-thought scratchpad (easy: three
two-number additions), then a flood of direct-answer data (hard: sum four
numbers in one shot) buries that skill. The result is a model whose greedy
decoding answers directly and is usually wrong — while the scratchpad
survives as a minority mode that sampling still visits ~10% of the time.

Then RL with a verifiable reward (GRPO + the exact verifier, no reward model)
does its thing: CoT samples are right far more often, collect more reward,
and get amplified. Accuracy, CoT rate, and completion LENGTH all climb
together — the toy version of "reasoning models learn to think longer".

Outputs: accuracy/CoT-rate curve, completion-length curve, before/after samples.
Run:  python3 rlvr.py            (~5 min on CPU)
"""

import copy
import csv
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "54-grpo-from-scratch"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import cot_lib as C  # noqa: E402
from grpo import grpo_update  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

STAGE1 = 800      # SFT steps on pure chain-of-thought data (learn the skill)
STAGE2 = 250      # SFT steps on 85% direct / 15% CoT data (bury the skill)
ITERS = 18
M = 32            # prompts per iteration
G = 8             # completions per prompt
TEMP = 0.8
LR = 3e-4
BETA = 0.08
EVAL_N = 250


@torch.no_grad()
def acc_by_style(model, n=400, temperature=0.7, seed=555):
    """Sampled accuracy split by completion style — the 'latent modes' probe."""
    probs = C.eval_problems(n, seed)
    comps = C.sample_batch(model, probs, temperature=temperature)
    stats = {}
    for ops, comp in zip(probs, comps):
        s = C.style_of(comp)
        hit, tot = stats.get(s, (0, 0))
        stats[s] = (hit + C.is_correct(ops, comp), tot + 1)
    return {s: (tot / n, hit / max(tot, 1)) for s, (hit, tot) in stats.items()}


def snapshot(model, path, title):
    probs = C.eval_problems(6, seed=42)
    comps = C.sample_batch(model, probs)
    with open(path, "w") as f:
        f.write(title + "\n\n")
        for ops, comp in zip(probs, comps):
            ok = "ok " if C.is_correct(ops, comp) else "WRONG"
            f.write(f"{C.prompt_str(ops):<14} -> {comp:<32} truth {sum(ops):<3} {ok}\n")


def main():
    t0 = time.time()
    print(f"== SFT stage 1: {STAGE1} steps of pure CoT ==", flush=True)
    policy = C.new_model(seed=0)
    C.train_sft(policy, steps=STAGE1, p_cot=1.0, seed=0, log_every=200)
    e = C.evaluate(policy, n=EVAL_N)
    print(f"after stage 1: greedy acc {e['acc']:.3f}  len {e['length']:.1f}  "
          f"cot-rate {e['cot'] + e['verbose']:.3f}", flush=True)

    print(f"== SFT stage 2: {STAGE2} steps of 85% direct / 15% CoT ==", flush=True)
    C.train_sft(policy, steps=STAGE2, p_cot=0.15, seed=1, log_every=100)

    e0 = C.evaluate(policy, n=EVAL_N)
    modes = acc_by_style(policy)
    print(f"greedy: acc {e0['acc']:.3f}  len {e0['length']:.1f}  "
          f"cot-rate {e0['cot'] + e0['verbose']:.3f}", flush=True)
    for s, (rate, acc) in sorted(modes.items()):
        print(f"  temp-0.7 mode '{s}': rate {rate:.3f}, accuracy {acc:.3f}", flush=True)
    snapshot(policy, OUT / "samples_before.txt", "greedy completions BEFORE RLVR")

    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    rng = random.Random(0)
    torch.manual_seed(0)

    hist = [dict(iter=0, **e0)]
    for it in range(1, ITERS + 1):
        probs = [C.problem(rng) for _ in range(M)]
        ops_flat = [o for o in probs for _ in range(G)]
        comps = C.sample_batch(policy, ops_flat, temperature=TEMP)
        seqs, masks = zip(*[C.seq_from(o, c or ";") for o, c in zip(ops_flat, comps)])
        rewards = torch.tensor([1.0 if C.is_correct(o, c) else 0.0
                                for o, c in zip(ops_flat, comps)])
        gids = torch.tensor([i // G for i in range(len(ops_flat))])
        grpo_update(policy, ref, opt, list(seqs), list(masks), rewards, gids,
                    beta=BETA, logps_fn=C.completion_token_logps)
        e = C.evaluate(policy, n=EVAL_N)
        hist.append(dict(iter=it, **e))
        print(f"iter {it:2d}  sampled-reward {rewards.mean():.3f}  greedy acc "
              f"{e['acc']:.3f}  len {e['length']:.1f}  "
              f"cot {e['cot'] + e['verbose']:.3f}", flush=True)

    snapshot(policy, OUT / "samples_after.txt", "greedy completions AFTER RLVR")
    print((OUT / "samples_before.txt").read_text())
    print((OUT / "samples_after.txt").read_text())

    with open(OUT / "rlvr_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "greedy_accuracy", "mean_length", "direct", "cot", "verbose"])
        for h in hist:
            w.writerow([h["iter"], f"{h['acc']:.4f}", f"{h['length']:.2f}",
                        f"{h['direct']:.4f}", f"{h['cot']:.4f}", f"{h['verbose']:.4f}"])

    its = [h["iter"] for h in hist]
    fig, ax = ps.new_axes()
    ax.plot(its, [h["acc"] for h in hist], color=ps.SERIES[0], linewidth=2,
            label="greedy accuracy")
    ax.plot(its, [h["cot"] + h["verbose"] for h in hist], color=ps.SERIES[1],
            linewidth=2, label="chain-of-thought rate")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "RLVR amplifies the mode that earns reward: the scratchpad",
              "GRPO iteration", f"fraction (of {EVAL_N} eval prompts)",
              OUT / "rlvr_accuracy.png")

    fig, ax = ps.new_axes()
    ax.plot(its, [h["length"] for h in hist], color=ps.SERIES[3], linewidth=2)
    ps.finish(fig, ax, "Completions grow longer as reasoning takes over",
              "GRPO iteration", "mean completion length (chars)",
              OUT / "rlvr_length.png")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
