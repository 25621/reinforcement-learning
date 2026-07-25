"""Project 14 -- Filtering a noisy image-text dataset with CLIP scores.

We take the clean COCO pairs from project 10 and deliberately break 60% of them
by giving each image somebody else's caption. That gives us a noisy web-scale
dataset in miniature -- and, unlike a real web crawl, a ground-truth answer key.

Stages:
    score     score every pair with a real frozen CLIP; measure how well that
              score separates the pairs we broke from the ones we did not
    train     train the tiny CLIP on unfiltered / filtered / oracle-clean data
    figures   redraw the plots from saved results

    python3 run.py --stage all      # ~9 min (2 min of it is CLIP scoring)
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
sys.path.insert(0, str(PROJECTS / "02-visualize-the-modality-gap"))
sys.path.insert(0, str(PROJECTS / "10-tiny-clip"))

import clip_lib                                                   # noqa: E402
import plot_style as ps                                           # noqa: E402
import tiny_clip as tc                                            # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"

CORRUPT_FRACTION = 0.6
BATCH = 128
STEPS = 550
LR = 3e-4
KEEP_FRACTIONS = [1.0, 0.7, 0.4, 0.2]


# ---------------------------------------------------------------------------
# the noisy dataset
# ---------------------------------------------------------------------------
def make_noisy(captions, train_ids, seed=0):
    """Return (caption per training image, is_broken flag).

    Broken pairs get a caption sampled from a *different* image, which is what
    unrelated alt-text looks like: a fluent, grammatical English sentence that
    simply does not describe the picture. Nothing about the text alone gives it
    away -- you have to look at the image to know."""
    rng = np.random.default_rng(seed)
    n = len(train_ids)
    broken = rng.random(n) < CORRUPT_FRACTION
    texts, donors = [], rng.permutation(n)
    for k, img in enumerate(train_ids):
        if broken[k]:
            donor = train_ids[donors[k]] if donors[k] != k else train_ids[(k + 1) % n]
            texts.append(captions[donor][0])
        else:
            texts.append(captions[img][0])
    return texts, broken


# ---------------------------------------------------------------------------
# scoring with a real, frozen CLIP
# ---------------------------------------------------------------------------
@torch.no_grad()
def clip_scores(images, train_ids, texts, batch=64):
    """Cosine similarity of every (image, caption) pair under a frozen CLIP B/32.

    The images we cached in project 10 are 72x72, and CLIP wants 224x224, so we
    upscale. That loses genuine detail -- but the filter only has to answer
    'is this caption about this picture at all', which survives a blurry image.
    Stage `score` measures exactly how much survives instead of assuming it.
    """
    from PIL import Image
    model, _ = clip_lib.get_model()
    out_i, out_t = [], []
    for i in range(0, len(train_ids), batch):
        chunk = train_ids[i:i + batch]
        arr = np.stack([
            np.asarray(Image.fromarray(images[j]).resize((224, 224), Image.BICUBIC),
                       dtype=np.float32) / 255.0 for j in chunk])
        arr = (arr - clip_lib.CLIP_MEAN) / clip_lib.CLIP_STD
        px = torch.from_numpy(arr.transpose(0, 3, 1, 2))
        out_i.append(clip_lib._pooled(model.get_image_features(pixel_values=px)).numpy())
        if (i // batch) % 10 == 0:
            print(f"    scored {i}/{len(train_ids)}", flush=True)
    img = clip_lib.l2_normalize(np.concatenate(out_i))
    txt = clip_lib.l2_normalize(clip_lib.encode_texts(list(texts), verbose=False))
    return np.sum(img * txt, axis=1).astype(np.float32)


def auc(scores, positive):
    """Area under the ROC curve, by rank -- no sklearn in this environment.

    Read it as: pick one true pair and one broken pair at random; the AUC is the
    probability the true one gets the higher CLIP score. 0.5 is a coin flip.
    """
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def stage_score():
    images, captions = tc.load_coco()
    train_ids, test_ids = tc.splits(len(images))
    texts, broken = make_noisy(captions, train_ids)

    cache = CKPT / "scores.npz"
    CKPT.mkdir(exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        scores = z["scores"]
    else:
        print("  scoring 4,500 noisy pairs with frozen CLIP B/32...")
        scores = clip_scores(images, train_ids, texts)
        np.savez(cache, scores=scores, broken=broken)

    clean = ~broken
    summary = {
        "n_pairs": int(len(scores)),
        "share_broken": round(float(broken.mean()), 3),
        "mean_score_true_pair": round(float(scores[clean].mean()), 4),
        "mean_score_broken_pair": round(float(scores[broken].mean()), 4),
        "auc": round(auc(scores, clean), 4),
    }
    # What does each keep-fraction actually keep?
    rows = []
    order = np.argsort(-scores)
    for frac in KEEP_FRACTIONS:
        keep = order[:int(len(order) * frac)]
        kept_clean = clean[keep].sum()
        rows.append({
            "keep_fraction": frac,
            "pairs_kept": len(keep),
            "precision_share_kept_that_are_true": round(float(clean[keep].mean()), 4),
            "recall_share_of_true_pairs_kept": round(float(kept_clean / clean.sum()), 4),
            "true_pairs_kept": int(kept_clean),
        })
        print(f"  keep {frac:.0%}: {len(keep)} pairs, "
              f"{clean[keep].mean():.1%} of them genuine, "
              f"covering {kept_clean / clean.sum():.1%} of all genuine pairs")

    OUT.mkdir(exist_ok=True)
    (OUT / "filter_quality.json").write_text(json.dumps(
        {"summary": summary, "thresholds": rows}, indent=2))
    print(f"  AUC {summary['auc']:.4f}   true {summary['mean_score_true_pair']:.3f}"
          f"  vs broken {summary['mean_score_broken_pair']:.3f}")

    fig, ax = ps.new_axes(7.2, 4.2)
    bins = np.linspace(scores.min(), scores.max(), 60)
    ax.hist(scores[clean], bins=bins, color=ps.SERIES[1], alpha=0.75,
            label="genuine pair")
    ax.hist(scores[broken], bins=bins, color=ps.SERIES[2], alpha=0.75,
            label="caption from another image")
    for frac in (0.4,):
        thr = np.sort(scores)[::-1][int(len(scores) * frac)]
        ax.axvline(thr, color=ps.INK, linestyle="--", linewidth=1.2)
        ax.text(thr, ax.get_ylim()[1] * 0.92, f"  keep top {frac:.0%}", fontsize=8,
                color=ps.INK)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "One number per pair, and the two populations barely touch",
              "CLIP score (cosine similarity)", "pairs", OUT / "score_histogram.png")


# ---------------------------------------------------------------------------
# downstream training
# ---------------------------------------------------------------------------
class FixedCaptionPairs(tc.Pairs):
    """Like tc.Pairs, but every image has exactly ONE caption -- the possibly
    wrong one from the noisy dataset -- instead of COCO's five clean ones."""

    def __init__(self, images, captions, vocab, index, texts_by_image):
        super().__init__(images, captions, vocab, index)
        self.texts_by_image = texts_by_image

    def batch(self, ids, rng=None):
        ids = np.asarray(ids)
        tok = torch.from_numpy(tc.tokenize([self.texts_by_image[i] for i in ids],
                                           self.vocab))
        return tc.to_pixels(self.images[ids], rng), tok


def stage_train():
    images, captions = tc.load_coco()
    vocab = tc.build_vocab(captions)
    train_ids, test_ids = tc.splits(len(images))
    texts, broken = make_noisy(captions, train_ids)
    scores = np.load(CKPT / "scores.npz")["scores"]
    texts_by_image = {int(img): t for img, t in zip(train_ids, texts)}

    order = np.argsort(-scores)
    conditions = {}
    for frac in KEEP_FRACTIONS:
        keep = order[:int(len(order) * frac)]
        conditions[f"keep{int(frac * 100)}"] = train_ids[keep]
    # the answer key: exactly the pairs that were never broken
    conditions["oracle"] = train_ids[~broken]

    # Evaluation always uses the CLEAN caption, because we are measuring the
    # model, not the noise.
    eval_pool = tc.Pairs(images, captions, vocab, test_ids)

    rows = []
    CKPT.mkdir(exist_ok=True)
    for name, ids in conditions.items():
        path = CKPT / f"{name}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            print(f"  [{name}] cached  i2t R@5 {rows[-1]['i2t_r5']:.3f}")
            continue
        pool = FixedCaptionPairs(images, captions, vocab, ids, texts_by_image)
        torch.manual_seed(0)
        model = tc.TinyCLIP(len(vocab))
        history = tc.train(model, pool, STEPS, batch=BATCH, lr=LR, seed=0,
                           log_every=STEPS // 5)
        m = tc.evaluate(model, eval_pool, test_ids)
        kept_mask = np.isin(train_ids, ids)
        m.update({
            "condition": name, "pairs_kept": int(len(ids)),
            "share_kept_that_are_true": round(float((~broken)[kept_mask].mean()), 4),
            "final_train_loss": round(float(np.mean(history[-30:])), 4),
        })
        torch.save(model.state_dict(), CKPT / f"{name}.pt")
        path.write_text(json.dumps(m, indent=2))
        rows.append(m)
        print(f"  [{name}] {len(ids)} pairs, {m['share_kept_that_are_true']:.0%} clean"
              f"  ->  i2t R@1 {m['i2t_r1']:.3f}  R@5 {m['i2t_r5']:.3f}")

    OUT.mkdir(exist_ok=True)
    keys = ["condition", "pairs_kept", "share_kept_that_are_true"] + \
        sorted(k for k in rows[0] if k not in
               ("condition", "pairs_kept", "share_kept_that_are_true"))
    with open(OUT / "downstream.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in r.items()})
    print(f"wrote {OUT / 'downstream.csv'}")


def stage_figures():
    names = [f"keep{int(f * 100)}" for f in KEEP_FRACTIONS] + ["oracle"]
    rows = [json.loads((CKPT / f"{n}.json").read_text()) for n in names]

    fig, ax = ps.new_axes(7.8, 4.2)
    x = np.arange(len(rows))
    for k, (key, label) in enumerate([("i2t_r1", "image -> text  R@1"),
                                      ("i2t_r5", "image -> text  R@5"),
                                      ("t2i_r5", "text -> image  R@5")]):
        ax.bar(x + (k - 1) * 0.27, [r[key] for r in rows], 0.25,
               color=ps.SERIES[k], label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['condition']}\n{r['pairs_kept']} pairs\n"
                        f"{r['share_kept_that_are_true']:.0%} clean" for r in rows],
                       fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Throwing data away made the model better -- up to a point",
              "", f"recall over a {tc.N_TEST}-image gallery", OUT / "downstream.png")


STAGES = {"score": stage_score, "train": stage_train, "figures": stage_figures}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = ap.parse_args()
    tc.setup()
    for n in (STAGES if args.stage == "all" else [args.stage]):
        print(f"\n=== {n}", flush=True)
        STAGES[n]()
