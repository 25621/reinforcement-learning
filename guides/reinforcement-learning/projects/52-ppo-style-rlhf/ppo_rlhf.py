"""Project 52 — PPO-style RLHF on a small model and a small reward model.

The InstructGPT recipe in miniature: prompt = state, completion = a sequence
of token actions, reward = a learned reward model's score on the finished
completion, algorithm = PPO with a KL penalty back to the frozen SFT
reference. Two arms share the identical PPO loop and differ only in the
reward source:

  arm "rm":        reward model score, strong KL leash (beta 0.5)
  arm "verifier":  exact correctness check, light leash (beta 0.08)

Tracking RM reward, KL, and TRUE accuracy side by side shows what RLHF really
optimizes: the proxy. `ppo_train` is reused by project 57 to over-train
against the reward model on purpose.

Outputs: reward/accuracy/KL curves for both arms, sample completions.
Run:  python3 ppo_rlhf.py            (~2.5 min on CPU)
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
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import rlhf_lib as L  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

ITERS = 25
M = 64            # prompts per iteration (one completion each)
TEMP = 0.7
LR = 1.5e-4
CLIP = 0.2
EPOCHS = 2


def k3_kl(new_lp, ref_lp):
    d = (ref_lp - new_lp).clamp(-8, 8)
    return d.exp() - d - 1


def ppo_update(policy, ref, opt, seqs, masks, rewards, beta, clip=CLIP,
               epochs=EPOCHS):
    """One PPO update. The advantage is the batch-standardized terminal reward
    broadcast to every completion token (a whitened batch-mean baseline
    instead of a learned value head — see the README for why that is enough
    at 4-token completions). Returns mean KL."""
    adv = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
    adv = adv.clamp(-3, 3).unsqueeze(1)          # RM reward std can be tiny
    with torch.no_grad():
        old_lp, m = L.completion_token_logps(policy, seqs, masks)
        ref_lp, _ = L.completion_token_logps(ref, seqs, masks)
    kl_val = 0.0
    for _ in range(epochs):
        new_lp, _ = L.completion_token_logps(policy, seqs, masks)
        ratio = (new_lp - old_lp).exp()
        pg = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)
        kl = k3_kl(new_lp, ref_lp)
        loss = ((pg + beta * kl) * m).sum() / m.sum()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        kl_val = float((kl * m).sum() / m.sum())
    return kl_val


@torch.no_grad()
def eval_policy(policy, rm, n=400):
    """Greedy accuracy AND mean RM score on one FIXED prompt set, so both the
    truth and the proxy are measured on the same yardstick every iteration."""
    probs = L.eval_problems(n)
    comps = L.sample_batch(policy, probs)
    seqs = [L.seq_from(a, b, c or ";")[0] for (a, b), c in zip(probs, comps)]
    acc = sum(L.is_correct(a, b, c) for (a, b), c in zip(probs, comps)) / n
    return acc, float(rm.reward(seqs).mean())


def ppo_train(policy, ref, reward_fn, rm, iters=ITERS, beta=0.5, lr=LR,
              m_prompts=M, temp=TEMP, seed=0, tag=""):
    """Runs PPO against `reward_fn`; logs RM reward AND true accuracy each
    iteration so proxy and truth can be compared. Returns the history."""
    opt = torch.optim.AdamW(policy.parameters(), lr=lr)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    acc, rm_r = eval_policy(policy, rm)
    hist = [dict(iter=0, rm_reward=rm_r, kl=0.0, acc=acc)]
    for it in range(1, iters + 1):
        ab = [(rng.randint(0, L.MAXN), rng.randint(0, L.MAXN))
              for _ in range(m_prompts)]
        comps = L.sample_batch(policy, ab, temperature=temp)
        comps = [c or ";" for c in comps]
        seqs, masks = zip(*[L.seq_from(a, b, c) for (a, b), c in zip(ab, comps)])
        seqs, masks = list(seqs), list(masks)
        rewards = reward_fn(ab, comps, seqs)
        kl = ppo_update(policy, ref, opt, seqs, masks, rewards, beta=beta)
        acc, rm_r = eval_policy(policy, rm)
        hist.append(dict(iter=it, rm_reward=rm_r, kl=kl, acc=acc))
        print(f"[{tag}] iter {it:2d}  rm-reward {rm_r:+.3f}  "
              f"greedy-acc {acc:.3f}  kl {kl:.4f}", flush=True)
    return hist


def main():
    t0 = time.time()
    print("== training the reward model (random-wrong pairs) ==", flush=True)
    rm = L.train_reward_model(L.preference_pairs(3000, random.Random(0)),
                              epochs=6, seed=0, log_every=2)
    for p in rm.parameters():
        p.requires_grad_(False)

    print("== building partial SFT policy ==", flush=True)
    sft = L.sft_model(seed=0)
    ref = copy.deepcopy(sft)
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"start greedy accuracy {L.accuracy(sft):.3f}", flush=True)

    def rm_reward(ab, comps, seqs):
        with torch.no_grad():
            return rm.reward(seqs)

    def verifier_reward(ab, comps, seqs):
        return torch.tensor([1.0 if L.is_correct(a, b, c) else 0.0
                             for (a, b), c in zip(ab, comps)])

    arms = {}
    arms["rm"] = ppo_train(copy.deepcopy(sft), ref, rm_reward, rm,
                           beta=0.5, seed=1, tag="rm beta=0.5")
    arms["verifier"] = ppo_train(copy.deepcopy(sft), ref, verifier_reward, rm,
                                 beta=0.08, seed=1, tag="verifier beta=0.08")

    with open(OUT / "ppo_curves.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "iter", "rm_reward", "kl", "greedy_accuracy"])
        for arm, hist in arms.items():
            for h in hist:
                w.writerow([arm, h["iter"], f"{h['rm_reward']:.4f}",
                            f"{h['kl']:.5f}", f"{h['acc']:.4f}"])

    labels = {"rm": "reward = RM score (beta 0.5)",
              "verifier": "reward = exact verifier (beta 0.08)"}
    colors = {"rm": ps.SERIES[2], "verifier": ps.SERIES[0]}

    fig, ax = ps.new_axes()
    for arm, hist in arms.items():
        ax.plot([h["iter"] for h in hist], [h["acc"] for h in hist],
                color=colors[arm], linewidth=2, label=labels[arm])
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Same PPO loop, different reward: only the trustworthy one improves truth",
              "PPO iteration", "greedy exact-match accuracy", OUT / "ppo_accuracy.png")

    fig, ax = ps.new_axes()
    for arm, hist in arms.items():
        ax.plot([h["iter"] for h in hist], [h["rm_reward"] for h in hist],
                color=colors[arm], linewidth=2, label=labels[arm])
    ax.set_ylim(1.5, 3.0)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "The judge can't tell the two policies apart: same RM score throughout",
              "PPO iteration", "mean RM score, greedy on the fixed eval set",
              OUT / "ppo_rm_reward.png")

    fig, ax = ps.new_axes()
    for arm, hist in arms.items():
        ax.plot([h["iter"] for h in hist[1:]], [h["kl"] for h in hist[1:]],
                color=colors[arm], linewidth=2, label=labels[arm])
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "KL to the SFT reference: the leash in action",
              "PPO iteration", "mean per-token k3 KL", OUT / "ppo_kl.png")

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
