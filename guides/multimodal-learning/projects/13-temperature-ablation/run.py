"""Project 13 -- Temperature ablation.

One scalar, tau, divides every similarity before the softmax. This project
sweeps it from 0.01 to 1.0, trains a tiny CLIP at each setting, and looks at
both the score (retrieval) and the shape of the space that produced it.

Stages:
    sweep     train at fixed tau = 0.01 .. 1.0, plus one run that LEARNS tau
    geometry  alignment / uniformity / gap, measured on the trained models
    figures   redraw the plots from saved results

    python3 run.py --stage all      # ~9 min
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "10-tiny-clip"))

import plot_style as ps                                           # noqa: E402
import tiny_clip as tc                                            # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"

BATCH = 128
STEPS = 550
LR = 3e-4
TAUS = [0.01, 0.05, 0.1, 0.3, 1.0]
LEARNED_INIT = 0.07


def build_pool():
    images, captions = tc.load_coco()
    vocab = tc.build_vocab(captions)
    train_ids, test_ids = tc.splits(len(images))
    return tc.Pairs(images, captions, vocab, train_ids), vocab, test_ids


def train_logged(model, pool, steps, batch, lr, seed=0):
    """tc.train, but also recording tau at every step (the learned run needs it)."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.98), eps=1e-6)
    loss_hist, tau_hist = [], []
    model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = tc.cosine_lr(step, steps, lr)
        ids = rng.choice(pool.index, size=batch, replace=False)
        px, tok = pool.batch(ids, rng)
        loss = tc.clip_loss(model(px, tok))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=np.log(100.0))
        loss_hist.append(float(loss.detach()))
        tau_hist.append(float(1.0 / model.logit_scale.exp().detach()))
        if step % max(steps // 5, 1) == 0:
            print(f"    step {step:4d}  loss {loss_hist[-1]:.4f}  "
                  f"tau {tau_hist[-1]:.4f}", flush=True)
    return loss_hist, tau_hist


def run_one(tag, pool, vocab, test_ids, tau, learn):
    CKPT.mkdir(exist_ok=True)
    path = CKPT / f"{tag}.json"
    if path.exists():
        m = json.loads(path.read_text())
        print(f"  [{tag}] cached  i2t R@1 {m['i2t_r1']:.3f}")
        return m
    torch.manual_seed(0)
    model = tc.TinyCLIP(len(vocab), temperature=tau, learn_temperature=learn)
    loss_hist, tau_hist = train_logged(model, pool, STEPS, BATCH, LR)
    m = tc.evaluate(model, pool, test_ids)
    m.update({"tag": tag, "tau_init": tau, "learned": learn,
              "tau_final": round(tau_hist[-1], 5),
              "final_train_loss": round(float(np.mean(loss_hist[-30:])), 4),
              # what the loss would be if the model were guessing
              "chance_loss": round(float(np.log(BATCH)), 4)})
    torch.save(model.state_dict(), CKPT / f"{tag}.pt")
    np.savez(CKPT / f"{tag}_hist.npz", loss=np.array(loss_hist, dtype=np.float32),
             tau=np.array(tau_hist, dtype=np.float32))
    path.write_text(json.dumps(m, indent=2))
    print(f"  [{tag}] i2t R@1 {m['i2t_r1']:.3f}  matched cos {m['matched_cos']:.3f}"
          f"  tau_final {m['tau_final']:.4f}")
    return m


def stage_sweep():
    pool, vocab, test_ids = build_pool()
    rows = [run_one(f"tau{t}", pool, vocab, test_ids, t, learn=False) for t in TAUS]
    rows.append(run_one("learned", pool, vocab, test_ids, LEARNED_INIT, learn=True))
    save_rows(rows)


def save_rows(rows):
    OUT.mkdir(exist_ok=True)
    keys = ["tag", "tau_init", "learned", "tau_final"] + \
        sorted(k for k in rows[0] if k not in ("tag", "tau_init", "learned", "tau_final"))
    with open(OUT / "temperature.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {OUT / 'temperature.csv'}")


def stage_geometry():
    """Look at the shape of the space, not just the score.

    alignment   how far a matched image and caption sit from each other
                (smaller = the two towers agree more)
    uniformity  how evenly the vectors spread over the sphere
                (more negative = better spread, less collapse)

    These two pull against each other -- a model that pulls everything together
    scores well on alignment and terribly on uniformity, and tau is the dial
    that decides the trade.
    """
    rows = [json.loads((CKPT / f"tau{t}.json").read_text()) for t in TAUS]
    rows.append(json.loads((CKPT / "learned.json").read_text()))
    for r in rows:
        print(f"  {r['tag']:8s} align {r['alignment']:.3f}  "
              f"unif_img {r['uniformity_img']:.3f}  "
              f"matched cos {r['matched_cos']:.3f}  gap {r['gap']:.3f}")
    (OUT / "geometry.json").write_text(json.dumps(rows, indent=2))


def stage_figures():
    rows = [json.loads((CKPT / f"tau{t}.json").read_text()) for t in TAUS]
    learned = json.loads((CKPT / "learned.json").read_text())

    # R@10 is the headline here rather than R@1: at 550 steps R@1 is 3-11 correct
    # out of 500, which is too few to rank anything.
    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(TAUS, [r["i2t_r10"] for r in rows], "o-", color=ps.SERIES[0],
            linewidth=2, label="image -> text  R@10")
    ax.plot(TAUS, [r["t2i_r10"] for r in rows], "o-", color=ps.SERIES[1],
            linewidth=2, label="text -> image  R@10")
    ax.plot(TAUS, [r["i2t_r1"] for r in rows], "o--", color=ps.SERIES[3],
            linewidth=1.5, alpha=0.8, label="image -> text  R@1 (noisy)")
    ax.scatter([learned["tau_final"]], [learned["i2t_r10"]], s=90, zorder=5,
               color=ps.SERIES[2], label=f"learned tau (ends at {learned['tau_final']:.3f})")
    ax.axhline(10 / tc.N_TEST, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(0.012, 10 / tc.N_TEST + 0.004, "chance R@10", fontsize=8, color=ps.INK_MUTED)
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Everything from 0.01 to 0.3 ties; tau = 1.0 falls off a cliff",
              "temperature tau (log scale)",
              f"recall over a {tc.N_TEST}-image gallery", OUT / "tau_vs_recall.png")

    fig, ax = ps.new_axes(7.2, 4.2)
    ax.plot(TAUS, [r["matched_cos"] for r in rows], "o-", color=ps.SERIES[0],
            linewidth=2, label="matched pair cosine")
    ax.plot(TAUS, [r["mismatched_cos"] for r in rows], "o-", color=ps.SERIES[2],
            linewidth=2, label="mismatched pair cosine")
    ax.plot(TAUS, [r["gap"] for r in rows], "o-", color=ps.SERIES[3],
            linewidth=2, label="modality gap (distance between cloud centres)")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Low tau collapses the space; high tau separates it -- and still loses",
              "temperature tau (log scale)", "cosine similarity / distance",
              OUT / "tau_vs_geometry.png")

    fig, ax = ps.new_axes(7.2, 4.2)
    h = np.load(CKPT / "learned_hist.npz")
    ax.plot(h["tau"], color=ps.SERIES[2], linewidth=2)
    ax.axhline(LEARNED_INIT, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(len(h["tau"]) * 0.6, LEARNED_INIT + 0.002,
            f"initialized at {LEARNED_INIT}", fontsize=8, color=ps.INK_MUTED)
    ax.set_ylim(0.06, 0.08)
    ps.finish(fig, ax, "Left to itself, the model barely moves tau at all",
              "step", "tau", OUT / "learned_tau.png")


STAGES = {"sweep": stage_sweep, "geometry": stage_geometry, "figures": stage_figures}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = ap.parse_args()
    tc.setup()
    for n in (STAGES if args.stage == "all" else [args.stage]):
        print(f"\n=== {n}", flush=True)
        STAGES[n]()
