"""Project 12 -- Hard-negative mining.

Same model, same data, same number of updates as project 10. The only thing that
changes is *which pairs end up in a batch together*.

Stages:
    inspect   what a mined batch actually looks like, and how many of its
              "negatives" are secretly correct answers
    train     random / semi-hard / hard batches, head to head
    figures   redraw the plots from saved results

    clipmine  NOT part of `all` (+4 min): repeat the hard condition but mine
              with a real frozen CLIP, to separate "mining is a bad idea" from
              "our miner was too weak to know what hard means"

    python3 run.py --stage all      # ~7 min
"""

import argparse
import csv
import json
import re
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
STEPS = 700
LR = 3e-4
REFRESH_EVERY = 100        # how often the mining index is rebuilt

# English words that carry no visual information; ignored when we measure how
# much two captions overlap.
STOP = set("a an the of on in at to and or with is are was were for from by "
           "his her its their this that these those there here it as".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOP}


# ---------------------------------------------------------------------------
# the samplers
# ---------------------------------------------------------------------------
class MinedSampler:
    """Builds each batch out of images the model currently finds similar.

    How the mining works, and why it is cheap: every `refresh_every` steps we
    encode the whole training set once with the *current* model and store the
    image vectors. Choosing a batch is then one anchor plus its nearest
    neighbours in that stored table -- no extra forward passes during the step.
    The table goes stale between refreshes, which is fine: negatives that were
    hard 100 steps ago are still hard-ish now.

    `lo`/`hi` select the neighbour band:
        lo=1,  hi=None  the very hardest available (true hard negatives)
        lo=30, hi=400   a semi-hard band, deliberately skipping the nearest few
    """

    def __init__(self, model, pool, batch, lo=1, hi=None, refresh_every=REFRESH_EVERY):
        self.model, self.pool, self.batch = model, pool, batch
        self.lo, self.hi, self.refresh_every = lo, hi, refresh_every
        self.ids = np.asarray(pool.index)
        self.order = None
        self.mined_ranks = []

    def refresh(self):
        v, _ = tc.embed_all(self.model, self.pool, self.ids, batch=500)
        sim = v @ v.T
        np.fill_diagonal(sim, -np.inf)          # never mine the anchor itself
        self.order = np.argsort(-sim, axis=1)   # (N, N) neighbour ranking

    def __call__(self, step, rng):
        if step % self.refresh_every == 0:
            self.model.eval()
            with torch.no_grad():
                self.refresh()
            self.model.train()
        anchor = int(rng.integers(len(self.ids)))
        hi = self.hi if self.hi is not None else self.batch * 4
        band = self.order[anchor][self.lo - 1:hi]
        pick = rng.choice(len(band), size=self.batch - 1, replace=False)
        rows = np.concatenate([[anchor], band[np.sort(pick)]])
        return self.ids[rows]


class FixedIndexSampler(MinedSampler):
    """Mine against a *fixed*, externally supplied similarity ranking.

    Used to settle the question the main experiment raises: when mining fails,
    is the *idea* wrong, or was the model doing the mining simply too weak to
    know what "similar" means? Here the ranking comes from a real frozen CLIP,
    which does know -- so if mining still fails, it is not the miner's fault.
    """

    def __init__(self, order, pool, batch, lo=1, hi=None):
        self.pool, self.batch = pool, batch
        self.lo, self.hi, self.refresh_every = lo, hi, 10 ** 9
        self.ids = np.asarray(pool.index)
        self.order = order

    def __call__(self, step, rng):
        anchor = int(rng.integers(len(self.ids)))
        hi = self.hi if self.hi is not None else self.batch * 4
        band = self.order[anchor][self.lo - 1:hi]
        pick = rng.choice(len(band), size=self.batch - 1, replace=False)
        return self.ids[np.concatenate([[anchor], band[np.sort(pick)]])]


def clip_neighbour_order(pool):
    """Rank every training image's neighbours using a real frozen CLIP B/32."""
    cache = CKPT / "clip_order.npy"
    if cache.exists():
        return np.load(cache)
    sys.path.insert(0, str(PROJECTS / "02-visualize-the-modality-gap"))
    import clip_lib
    from PIL import Image
    print("  encoding 4,500 images with frozen CLIP to build the mining index...")
    model, _ = clip_lib.get_model()
    ids = np.asarray(pool.index)
    feats = []
    with torch.no_grad():
        for i in range(0, len(ids), 64):
            chunk = ids[i:i + 64]
            arr = np.stack([np.asarray(
                Image.fromarray(pool.images[j]).resize((224, 224), Image.BICUBIC),
                dtype=np.float32) / 255.0 for j in chunk])
            arr = (arr - clip_lib.CLIP_MEAN) / clip_lib.CLIP_STD
            px = torch.from_numpy(arr.transpose(0, 3, 1, 2))
            feats.append(clip_lib._pooled(model.get_image_features(pixel_values=px)).numpy())
            if (i // 64) % 10 == 0:
                print(f"    {i}/{len(ids)}", flush=True)
    v = clip_lib.l2_normalize(np.concatenate(feats))
    sim = v @ v.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1).astype(np.int32)
    CKPT.mkdir(exist_ok=True)
    np.save(cache, order)
    return order


def stage_clipmine():
    """Extra stage (~4 min): repeat the hard condition, but mine with real CLIP."""
    pool, vocab, test_ids = build_pool()
    order = clip_neighbour_order(pool)
    sampler = FixedIndexSampler(order, pool, BATCH, lo=1)

    # First the diagnostic that does not need any training: how confusable are
    # CLIP-mined batches compared with the tiny model's own mined batches?
    diag = {"clip-mined": batch_difficulty(pool, sampler, "clip-mined"),
            "random": batch_difficulty(pool, random_sampler(pool, BATCH), "random")}
    print(f"  caption overlap: CLIP-mined {diag['clip-mined']['caption_overlap']:.4f}"
          f"   random {diag['random']['caption_overlap']:.4f}")

    path = CKPT / "clip-mined.json"
    if path.exists():
        m = json.loads(path.read_text())
    else:
        torch.manual_seed(0)
        model = tc.TinyCLIP(len(vocab))
        history = tc.train(model, pool, STEPS, batch=BATCH, lr=LR, seed=0,
                           sampler=sampler, log_every=STEPS // 5)
        m = tc.evaluate(model, pool, test_ids)
        m.update({"condition": "clip-mined", "steps": STEPS, "batch": BATCH,
                  "final_train_loss": round(float(np.mean(history[-30:])), 4)})
        m.update(diag["clip-mined"])
        np.save(CKPT / "clip-mined_loss.npy", np.array(history, dtype=np.float32))
        path.write_text(json.dumps(m, indent=2))
    print(f"  [clip-mined] i2t R@1 {m['i2t_r1']:.3f}  R@10 {m['i2t_r10']:.3f}"
          f"  t2i R@1 {m['t2i_r1']:.3f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "clip_mining.json").write_text(json.dumps(
        {"batch_difficulty": diag, "result": m}, indent=2))

    # And one example batch, so the difference is visible rather than numeric.
    rng = np.random.default_rng(7)
    ids = FixedIndexSampler(order, pool, 8, lo=1)(0, rng)
    anchor = content_words(pool.captions[ids[0]][0])
    lines = ["mined with a real frozen CLIP B/32", "",
             f"ANCHOR: {pool.captions[ids[0]][0]}", ""]
    for j in ids[1:]:
        cap = pool.captions[j][0]
        share = len(anchor & content_words(cap)) / max(len(anchor), 1)
        lines.append(f"  overlap {share:4.2f}   {cap}")
    (OUT / "clip_mined_batch.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def random_sampler(pool, batch):
    ids = np.asarray(pool.index)

    def sample(step, rng):
        return rng.choice(ids, size=batch, replace=False)
    return sample


CONDITIONS = {
    "random": None,
    "semi-hard": dict(lo=30, hi=400),
    "hard": dict(lo=1, hi=None),
}


# ---------------------------------------------------------------------------
def build_pool():
    images, captions = tc.load_coco()
    vocab = tc.build_vocab(captions)
    train_ids, test_ids = tc.splits(len(images))
    return tc.Pairs(images, captions, vocab, train_ids), vocab, test_ids


def stage_train():
    pool, vocab, test_ids = build_pool()
    CKPT.mkdir(exist_ok=True)
    rows = []
    for name, cfg in CONDITIONS.items():
        path = CKPT / f"{name}.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
            print(f"  [{name}] cached  i2t R@1 {rows[-1]['i2t_r1']:.3f}")
            continue
        torch.manual_seed(0)
        model = tc.TinyCLIP(len(vocab))
        sampler = (random_sampler(pool, BATCH) if cfg is None
                   else MinedSampler(model, pool, BATCH, **cfg))
        history = tc.train(model, pool, STEPS, batch=BATCH, lr=LR, seed=0,
                           sampler=sampler, log_every=STEPS // 5)
        m = tc.evaluate(model, pool, test_ids)
        m.update({"condition": name, "steps": STEPS, "batch": BATCH,
                  "final_train_loss": round(float(np.mean(history[-30:])), 4)})
        m.update(batch_difficulty(pool, sampler, name))
        np.save(CKPT / f"{name}_loss.npy", np.array(history, dtype=np.float32))
        torch.save(model.state_dict(), CKPT / f"{name}.pt")
        path.write_text(json.dumps(m, indent=2))
        rows.append(m)
        print(f"  [{name}] i2t R@1 {m['i2t_r1']:.3f}  t2i R@1 {m['t2i_r1']:.3f}"
              f"  train loss {m['final_train_loss']:.3f}")

    OUT.mkdir(exist_ok=True)
    keys = ["condition"] + sorted(k for k in rows[0] if k != "condition")
    with open(OUT / "conditions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def batch_difficulty(pool, sampler, name, n_batches=40, seed=1):
    """How confusable are the pairs a sampler puts in one batch?

    Two measurements, both on the *captions*, so they are independent of the
    model being trained:
      caption_overlap        average share of content words a batch member's
                             caption shares with the anchor's caption
      likely_false_negatives share of batch members whose caption overlaps the
                             anchor's by more than half -- those are not really
                             wrong answers, and InfoNCE will punish the model
                             for ranking them highly anyway
    """
    rng = np.random.default_rng(seed)
    overlaps, false_neg = [], []
    for b in range(n_batches):
        ids = sampler(b * REFRESH_EVERY + 1 if isinstance(sampler, MinedSampler) else b,
                      rng)
        anchor = content_words(pool.captions[ids[0]][0])
        if not anchor:
            continue
        for j in ids[1:]:
            other = content_words(pool.captions[j][0])
            share = len(anchor & other) / max(len(anchor), 1)
            overlaps.append(share)
            false_neg.append(share > 0.5)
    return {"caption_overlap": round(float(np.mean(overlaps)), 4),
            "likely_false_negatives": round(float(np.mean(false_neg)), 4)}


# ---------------------------------------------------------------------------
def stage_inspect():
    """Print one mined batch so the reader can see what 'hard' means."""
    pool, vocab, test_ids = build_pool()
    torch.manual_seed(0)
    model = tc.TinyCLIP(len(vocab))
    # Mine with a *trained* model if one exists -- an untrained model's idea of
    # "similar" is just noise, which is itself worth knowing.
    trained = CKPT / "random.pt"
    tag = "untrained"
    if trained.exists():
        model.load_state_dict(torch.load(trained))
        tag = "trained (the random-batch model from this project)"
    sampler = MinedSampler(model, pool, 8, lo=1)
    rng = np.random.default_rng(7)
    ids = sampler(0, rng)
    lines = [f"mining model: {tag}", "", f"ANCHOR: {pool.captions[ids[0]][0]}", ""]
    anchor = content_words(pool.captions[ids[0]][0])
    for j in ids[1:]:
        cap = pool.captions[j][0]
        share = len(anchor & content_words(cap)) / max(len(anchor), 1)
        lines.append(f"  overlap {share:4.2f}   {cap}")

    rand = random_sampler(pool, 8)(0, rng)
    lines += ["", "for comparison, a RANDOM batch against the same anchor:", ""]
    for j in rand[1:]:
        cap = pool.captions[j][0]
        share = len(anchor & content_words(cap)) / max(len(anchor), 1)
        lines.append(f"  overlap {share:4.2f}   {cap}")

    OUT.mkdir(exist_ok=True)
    (OUT / "mined_batch.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------------------
def stage_figures():
    names = list(CONDITIONS)
    # include the CLIP-mined run if `--stage clipmine` has been run
    if (CKPT / "clip-mined.json").exists():
        names.append("clip-mined")
    rows = [json.loads((CKPT / f"{n}.json").read_text()) for n in names]

    fig, ax = ps.new_axes(7.2, 4.2)
    for i, r in enumerate(rows):
        h = np.load(CKPT / f"{r['condition']}_loss.npy")
        ax.plot(np.convolve(h, np.ones(21) / 21, mode="valid"),
                color=ps.SERIES[i], linewidth=2, label=r["condition"])
    ax.axhline(np.log(BATCH), color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(STEPS * 0.6, np.log(BATCH) + 0.05, "chance = ln(128)", fontsize=8,
            color=ps.INK_MUTED)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Hard batches keep the loss high -- that is the point, not a bug",
              "step", "InfoNCE training loss", OUT / "loss_curves.png")

    fig, ax = ps.new_axes(8.2, 4.2)
    x = np.arange(len(rows))
    for k, (key, label) in enumerate([("i2t_r10", "image -> text  R@10"),
                                      ("t2i_r10", "text -> image  R@10"),
                                      ("i2t_r5", "image -> text  R@5")]):
        ax.bar(x + (k - 1) * 0.27, [r[key] for r in rows], 0.25,
               color=ps.SERIES[k], label=label)
    ax.set_xticks(x)
    ax.axhline(10 / tc.N_TEST, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(len(rows) - 0.6, 10 / tc.N_TEST + 0.004, "chance R@10", fontsize=8,
            color=ps.INK_MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['condition']}\ncaption overlap {r['caption_overlap']:.3f}"
                        for r in rows], fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Harder batches, measurably -- and worse held-out retrieval",
              "", f"recall over a {tc.N_TEST}-image gallery", OUT / "conditions.png")


# `clipmine` is not in ALL_STAGES: it adds ~4 minutes (it encodes 4,500 images
# with a real CLIP to build a strong mining index).
STAGES = {"inspect": stage_inspect, "train": stage_train, "figures": stage_figures,
          "clipmine": stage_clipmine}
ALL_STAGES = ["train", "inspect", "figures"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = ap.parse_args()
    tc.setup()
    for n in (ALL_STAGES if args.stage == "all" else [args.stage]):
        print(f"\n=== {n}", flush=True)
        STAGES[n]()
