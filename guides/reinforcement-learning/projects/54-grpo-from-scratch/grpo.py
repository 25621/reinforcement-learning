"""Project 54 — GRPO from scratch on a verifiable math task.

Implements Group Relative Policy Optimization: for each prompt, sample a
GROUP of completions, grade each one with the exact verifier (1 = correct sum,
0 = wrong), and use each completion's advantage *relative to its own group's
average* as the policy-gradient weight — no value network anywhere. A
PPO-style clipped ratio and a k3 KL penalty to the frozen SFT reference keep
the updates stable.

`grpo_update` is generic (works on any token sequences + rewards) and is
reused by project 55 for RLVR on the chain-of-thought task.

Outputs: accuracy curve, reward/KL curves, one group's worked example.
Run:  python3 grpo.py            (~2.5 min on CPU)
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

ITERS = 20
M = 48            # prompts per iteration
G = 8             # completions per prompt (the "group")
TEMP = 0.7        # sampling temperature for rollouts
LR = 3e-4
BETA = 0.08       # KL penalty strength
CLIP = 0.2        # PPO clip range
EPOCHS = 2        # optimization epochs per batch of rollouts


def group_advantages(rewards, group_ids):
    """Advantage = (r - group mean) / (group std + eps), computed per group."""
    adv = torch.zeros_like(rewards)
    for g in group_ids.unique():
        m = group_ids == g
        r = rewards[m]
        adv[m] = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    return adv


def k3_kl(new_lp, ref_lp):
    """Per-token k3 KL estimate exp(ref-new) - (ref-new) - 1 (always >= 0).
    The difference is clamped so a token the policy has pushed very far from
    the reference cannot blow the loss up."""
    d = (ref_lp - new_lp).clamp(-8, 8)
    return d.exp() - d - 1


def grpo_update(policy, ref, opt, seqs, masks, rewards, group_ids,
                beta=BETA, clip=CLIP, epochs=EPOCHS, logps_fn=None):
    """One GRPO update on a batch of rollouts. Returns (loss, kl) means.

    logps_fn lets another task (project 55's chain-of-thought vocabulary)
    plug in its own log-prob helper; the update rule itself never changes."""
    logps_fn = logps_fn or L.completion_token_logps
    adv = group_advantages(rewards, group_ids).unsqueeze(1)        # (B,1)
    with torch.no_grad():
        old_lp, m = logps_fn(policy, seqs, masks)
        ref_lp, _ = logps_fn(ref, seqs, masks)
    kl_val = loss_val = 0.0
    for _ in range(epochs):
        new_lp, _ = logps_fn(policy, seqs, masks)
        ratio = (new_lp - old_lp).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        kl = k3_kl(new_lp, ref_lp)
        tok_loss = (pg + beta * kl) * m
        loss = tok_loss.sum() / m.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            kl_val = float((kl * m).sum() / m.sum())
            loss_val = float(loss)
    return loss_val, kl_val


def rollout_groups(policy, rng, sample_fn=None, reward_fn=None, m=M, g=G, temp=TEMP):
    """Sample G completions for each of M prompts; grade with the verifier."""
    probs = [(rng.randint(0, L.MAXN), rng.randint(0, L.MAXN)) for _ in range(m)]
    ab = [p for p in probs for _ in range(g)]
    comps = L.sample_batch(policy, ab, temperature=temp)
    seqs, masks = zip(*[L.seq_from(a, b, c or ";") for (a, b), c in zip(ab, comps)])
    rewards = torch.tensor([1.0 if L.is_correct(a, b, c) else 0.0
                            for (a, b), c in zip(ab, comps)])
    group_ids = torch.tensor([i // g for i in range(len(ab))])
    return list(seqs), list(masks), rewards, group_ids, ab, comps


def save_group_example(ab, comps, rewards, path):
    """Write one prompt's group with rewards and advantages — what GRPO 'sees'."""
    a, b = ab[0]
    r = rewards[:G]
    adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)
    with open(path, "w") as f:
        f.write(f"prompt: {a}+{b}=      (true answer {a + b})\n\n")
        f.write("completion   reward   advantage\n")
        for c, rw, ad in zip(comps[:G], r, adv):
            f.write(f"{c:<12} {rw:>6.1f}   {ad:>+9.3f}\n")
        f.write("\ngroup mean reward = baseline; above-average completions get\n")
        f.write("positive advantage, below-average ones negative.\n")


def main():
    t0 = time.time()
    print("== building partial SFT policy ==", flush=True)
    policy = L.sft_model(seed=0)
    ref = copy.deepcopy(policy)
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(policy.parameters(), lr=LR)
    rng = random.Random(0)
    torch.manual_seed(0)

    acc0 = L.accuracy(policy)
    print(f"start greedy accuracy {acc0:.3f}", flush=True)
    hist = [(0, acc0, float("nan"), float("nan"))]

    for it in range(1, ITERS + 1):
        seqs, masks, rewards, gids, ab, comps = rollout_groups(policy, rng)
        if it == 1:
            save_group_example(ab, comps, rewards, OUT / "group_example.txt")
        _, kl = grpo_update(policy, ref, opt, seqs, masks, rewards, gids)
        acc = L.accuracy(policy)
        hist.append((it, acc, float(rewards.mean()), kl))
        print(f"iter {it:2d}  sampled-reward {rewards.mean():.3f}  "
              f"greedy-acc {acc:.3f}  kl {kl:.4f}", flush=True)

    with open(OUT / "grpo_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "greedy_accuracy", "mean_sampled_reward", "kl"])
        w.writerows(hist)

    its = [h[0] for h in hist]
    fig, ax = ps.new_axes()
    ax.plot(its, [h[1] for h in hist], color=ps.SERIES[0], linewidth=2,
            label="greedy accuracy (verifier)")
    ax.plot(its[1:], [h[2] for h in hist[1:]], color=ps.SERIES[1], linewidth=2,
            label="mean sampled reward (temp 0.7)")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "GRPO on a verifiable task: no reward model, no value network",
              "GRPO iteration", "fraction correct", OUT / "grpo_accuracy.png")

    fig, ax = ps.new_axes()
    ax.plot(its[1:], [h[3] for h in hist[1:]], color=ps.SERIES[3], linewidth=2)
    ax.set_yscale("log")
    ps.finish(fig, ax, "KL to the frozen SFT reference stays small (log scale; spikes are single tokens)",
              "GRPO iteration", "mean per-token k3 KL", OUT / "grpo_kl.png")

    print(f"final greedy accuracy {L.accuracy(policy):.3f}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
