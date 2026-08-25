"""Project 10 -- Train a small CLIP from scratch.

Stages:
    sweep      batch 32 / 128 / 512 at an EQUAL DATA budget (same pairs seen)
    steps      batch 32 / 128 / 512 at an EQUAL UPDATE budget (same steps)
    examples   qualitative top-3 retrieval from the best model
    figures    redraw every plot from the saved results (no retraining)

Two stages are deliberately NOT part of `all`, so that `all` stays under ten
minutes. Run them separately:
    steps      the equal-updates sweep     (+3.5 min)
    direction  rows-only vs symmetric      (+2.5 min)

    python3 run.py --stage all      # ~8 min once data/coco_64.npz exists
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-modality-survey"))
import plot_style as ps                                           # noqa: E402
import tiny_clip as tc                                            # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"

# Every run in the sweep sees the same number of (image, caption) pair-views, so
# the comparison is "same data budget, different batch size" -- not "same number
# of updates", which would silently hand 16x more gradient steps to batch 32.
PAIR_BUDGET = 115_200
STEP_BUDGET = 300      # for the equal-updates sweep
BATCHES = [32, 128, 512]
EVAL_GALLERY = tc.N_TEST


def build_pool():
    images, captions = tc.load_coco()
    vocab = tc.build_vocab(captions)
    train_ids, test_ids = tc.splits(len(images))
    pool = tc.Pairs(images, captions, vocab, train_ids)
    return pool, vocab, train_ids, test_ids


def run_one(tag, pool, vocab, test_ids, batch, steps, lr, seed=0, direction="both",
            sampler=None):
    """Train one configuration, save its checkpoint and metrics, return them."""
    CKPT.mkdir(exist_ok=True)
    result_path = CKPT / f"{tag}.json"
    if result_path.exists():
        print(f"  [{tag}] cached")
        return json.loads(result_path.read_text())

    torch.manual_seed(seed)
    model = tc.TinyCLIP(len(vocab))
    t0 = time.time()
    if direction == "both":
        history = tc.train(model, pool, steps, batch=batch, lr=lr, seed=seed,
                           sampler=sampler, log_every=max(steps // 5, 1))
    else:
        history = train_one_direction(model, pool, steps, batch, lr, seed, direction)
    wall = time.time() - t0

    metrics = tc.evaluate(model, pool, test_ids)
    metrics.update({"tag": tag, "batch": batch, "steps": steps, "lr": lr,
                    "direction": direction, "pair_views": batch * steps,
                    "wall_seconds": round(wall, 1),
                    "final_loss": round(float(np.mean(history[-20:])), 4),
                    "gallery": len(test_ids)})
    torch.save(model.state_dict(), CKPT / f"{tag}.pt")
    np.save(CKPT / f"{tag}_loss.npy", np.array(history, dtype=np.float32))
    result_path.write_text(json.dumps(metrics, indent=2))
    print(f"  [{tag}] i2t R@1 {metrics['i2t_r1']:.3f}  t2i R@1 {metrics['t2i_r1']:.3f}"
          f"  ({wall:.0f}s)")
    return metrics


def train_one_direction(model, pool, steps, batch, lr, seed, direction):
    """Same loop as tc.train but using only one half of the symmetric loss."""
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1,
                            betas=(0.9, 0.98), eps=1e-6)
    history = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = tc.cosine_lr(step, steps, lr)
        ids = rng.choice(pool.index, size=batch, replace=False)
        px, tok = pool.batch(ids, rng)
        logits = model(px, tok)
        labels = torch.arange(len(logits))
        loss = (F.cross_entropy(logits, labels) if direction == "rows"
                else F.cross_entropy(logits.T, labels))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            model.logit_scale.clamp_(max=np.log(100.0))
        history.append(float(loss))
        if step % max(steps // 5, 1) == 0:
            print(f"    step {step:5d}  loss {float(loss):.4f}", flush=True)
    return history


# ---------------------------------------------------------------------------
def stage_sweep():
    pool, vocab, train_ids, test_ids = build_pool()
    print(f"  {len(train_ids)} train images, {len(test_ids)} held out, "
          f"vocab {len(vocab)}")
    rows = []
    for batch in BATCHES:
        steps = PAIR_BUDGET // batch
        # Bigger batches average more examples per update, so they tolerate (and
        # need) a larger learning rate. The square-root rule is the usual default.
        lr = 3e-4 * (batch / 128) ** 0.5
        rows.append(run_one(f"bs{batch}", pool, vocab, test_ids, batch, steps, lr))
    save_rows(rows, "batch_sweep.csv")


def stage_steps():
    """The other fair comparison: same number of optimizer steps, so the large
    batch simply sees more data. This is the framing in which more negatives is
    supposed to win outright."""
    pool, vocab, train_ids, test_ids = build_pool()
    rows = []
    for batch in BATCHES:
        lr = 3e-4 * (batch / 128) ** 0.5
        rows.append(run_one(f"steps{batch}", pool, vocab, test_ids, batch,
                            STEP_BUDGET, lr))
    save_rows(rows, "equal_steps.csv")


def stage_direction():
    pool, vocab, train_ids, test_ids = build_pool()
    steps = 450
    rows = [run_one(f"dir_{d}", pool, vocab, test_ids, 128, steps, 3e-4, direction=d)
            for d in ("both", "rows")]
    save_rows(rows, "direction.csv")


def save_rows(rows, name):
    OUT.mkdir(exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    keys = ["tag", "batch", "steps", "direction"] + [k for k in keys
                                                     if k not in
                                                     ("tag", "batch", "steps", "direction")]
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {OUT / name}")


# ---------------------------------------------------------------------------
def stage_examples(tag=None):
    """What the model actually retrieves -- the sanity check no metric replaces."""
    import matplotlib.pyplot as plt
    pool, vocab, train_ids, test_ids = build_pool()
    if tag is None:   # whichever batch size won the equal-data sweep
        scored = [(json.loads((CKPT / f"bs{b}.json").read_text())["i2t_r1"], f"bs{b}")
                  for b in BATCHES if (CKPT / f"bs{b}.json").exists()]
        tag = max(scored)[1]
    print(f"  using checkpoint {tag}")
    model = tc.TinyCLIP(len(vocab))
    model.load_state_dict(torch.load(CKPT / f"{tag}.pt"))
    v, t = tc.embed_all(model, pool, test_ids)
    sim = v @ t.T

    rng = np.random.default_rng(3)
    # show a few queries the model got right and a few it got wrong
    rank = np.argmax(np.argsort(-sim, axis=1) == np.arange(len(sim))[:, None], axis=1)
    right = rng.choice(np.flatnonzero(rank == 0), 3, replace=False)
    wrong = rng.choice(np.flatnonzero(rank > 20), 3, replace=False)
    picks = list(right) + list(wrong)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    lines = []
    for ax, q in zip(axes.ravel(), picks):
        img = pool.images[test_ids[q]]
        ax.imshow(img)
        ax.axis("off")
        top = np.argsort(-sim[q])[:3]
        text = "\n".join(
            f"{'*' if j == q else ' '} {pool.captions[test_ids[j]][0][:38]}"
            for j in top)
        ax.set_title(f"true rank {rank[q] + 1}", fontsize=9, color=ps.INK, loc="left")
        ax.text(0, 1.02, "", transform=ax.transAxes)
        ax.set_xlabel(text)
        ax.xaxis.set_label_position("bottom")
        ax.xaxis.label.set_fontsize(7.5)
        ax.xaxis.label.set_color(ps.INK_SECONDARY)
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        lines.append({"query_image": int(test_ids[q]), "true_rank": int(rank[q] + 1),
                      "true_caption": pool.captions[test_ids[q]][0],
                      "top1_retrieved": pool.captions[test_ids[top[0]]][0]})
    fig.suptitle("Top-3 captions retrieved for each image  (* = the correct one)",
                 color=ps.INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "examples.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    (OUT / "examples.json").write_text(json.dumps(lines, indent=2))
    print(f"wrote {OUT / 'examples.png'}")


def stage_figures():
    rows = [json.loads((CKPT / f"bs{b}.json").read_text()) for b in BATCHES]

    fig, ax = ps.new_axes(7.2, 4.2)
    for i, r in enumerate(rows):
        hist = np.load(CKPT / f"{r['tag']}_loss.npy")
        seen = np.arange(1, len(hist) + 1) * r["batch"]
        smooth = np.convolve(hist, np.ones(21) / 21, mode="valid")
        # Each curve is divided by ln(batch), the chance level for that batch
        # size -- otherwise the small-batch curve looks better purely because it
        # is ranking fewer candidates (project 09, stage `floor`).
        ax.plot(seen[10:len(smooth) + 10], smooth / np.log(r["batch"]),
                color=ps.SERIES[i], linewidth=2, label=f"batch {r['batch']}")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Training loss as a fraction of its own chance level",
              "(image, caption) pair-views", "InfoNCE loss / ln(batch)",
              OUT / "loss_curves.png")

    fig, ax = ps.new_axes(7.2, 4.2)
    x = np.arange(len(rows))
    for k, (key, label) in enumerate([("i2t_r1", "image -> text  R@1"),
                                      ("t2i_r1", "text -> image  R@1"),
                                      ("i2t_r5", "image -> text  R@5")]):
        ax.bar(x + (k - 1) * 0.27, [r[key] for r in rows], 0.25,
               color=ps.SERIES[k], label=label)
    ax.axhline(1 / EVAL_GALLERY, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(len(rows) - 0.6, 1 / EVAL_GALLERY + 0.004, "chance", fontsize=8,
            color=ps.INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"batch {r['batch']}\n{r['steps']} steps" for r in rows])
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Same data budget, more negatives per update",
              "", f"recall over a {EVAL_GALLERY}-image gallery",
              OUT / "batch_sweep.png")


# `steps` and `direction` are deliberately not in ALL_STAGES: both are extra
# training runs, and `all` is kept under ten minutes.
STAGES = {"sweep": stage_sweep, "steps": stage_steps,
          "examples": stage_examples, "figures": stage_figures,
          "direction": stage_direction}
ALL_STAGES = ["sweep", "examples", "figures"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = ap.parse_args()
    tc.setup()
    for n in (ALL_STAGES if args.stage == "all" else [args.stage]):
        print(f"\n=== {n}", flush=True)
        STAGES[n]()
