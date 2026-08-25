"""Caption ablation: same pictures, five different caption sources, one model.

Project 37 left us with a pile of images whose alt-text is real web junk. There
are two standard ways to fix that pile -- *filter* it (throw away the pairs a
CLIP model says do not match) or *recaption* it (throw the alt-text away and let
a stronger model describe the picture). This project runs both, plus the
do-nothing baseline and a human-caption ceiling, through one identical training
recipe, and scores every arm on the same held-out human captions.

Arms (identical model, identical steps, identical images -- only the text moves)
    alt        the web alt-text exactly as crawled
    filtered45 only the pairs whose CLIP score is in the top 45%
    filtered70 the same rule, looser cut-off (project 37 measured that this one
               already catches 99% of the broken pairs)
    recap      BLIP's description of every image
    blend      half alt-text, half recaption, drawn per sample
    human      the original MS-COCO human caption (the ceiling)

Stages
    data       collect the pool, recaption everything, build the vocabulary
    train      one arm  (python3 run.py --stage train --arm recap)
    plot       figures and tables

Everything is cached, so re-running a finished stage is free.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "10-tiny-clip"))
sys.path.insert(0, str(PROJECTS / "37-mini-laion-pipeline"))
import pipeline_lib as P  # noqa: E402
import plot_style as ps  # noqa: E402
import tiny_clip as TC  # noqa: E402

OUT = HERE / "outputs"
DATA = HERE / "data"
ARMS = ["alt", "filtered45", "filtered70", "recap", "blend", "human", "human5",
        "alt-dirty30", "alt-dirty60"]
N_GALLERY = 400
STEPS, BATCH, LR = 1500, 128, 3e-4
KEEP = {"filtered45": 0.45, "filtered70": 0.70}   # project 37's two candidate cut-offs
DIRTY = {"alt-dirty30": 0.30, "alt-dirty60": 0.60}  # extra caption noise, on purpose
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


def _load(name):
    return json.loads((OUT / name).read_text())


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_vocab(all_texts, min_count=4):
    """One word list shared by every arm.

    Building a separate vocabulary per arm would be a second difference between
    the runs, and then a win could be "richer captions" *or* "bigger vocabulary".
    One shared vocabulary keeps the caption text as the only moving part.
    """
    path = DATA / "vocab.json"
    if path.exists():
        return json.loads(path.read_text())
    counts = {}
    for t in all_texts:
        for w in TC._WORD.findall(t.lower()):
            counts[w] = counts.get(w, 0) + 1
    vocab = {"<pad>": TC.PAD, "<unk>": TC.UNK, "<sot>": TC.SOT, "<eot>": TC.EOT}
    for w in sorted([w for w, c in counts.items() if c >= min_count]):
        vocab[w] = len(vocab)
    DATA.mkdir(exist_ok=True)
    path.write_text(json.dumps(vocab))
    return vocab


def pool():
    """-> dict with the images and every caption source, plus the split."""
    crawl = P.build_crawl()
    alive = np.load(P.data_dir() / "alive_after_cheap.npy")   # dedup+size+text pass
    scores = np.load(P.data_dir() / "clip_scores.npy")[alive]
    caps = P.load_recaptions()
    missing = [int(i) for i in alive if str(int(i)) not in caps]
    if missing:
        raise SystemExit(f"{len(missing)} images still need recaptioning -- run "
                         f"`python3 run.py --stage data` first")
    images = crawl["images"][alive]
    # 72x72 is what the tiny-CLIP tower expects (it random-crops to 64 during
    # training, which is where the augmentation comes from).
    small = np.stack([np.asarray(Image.fromarray(a).resize((TC.STORE, TC.STORE),
                                                           Image.BICUBIC),
                                 dtype=np.uint8) for a in images])
    return {
        "ids": alive,
        "images": small,
        "alt": [crawl["alt"][i] for i in alive],
        "recap": [caps[str(int(i))] for i in alive],
        "human": [crawl["human"][i] for i in alive],
        "clip": scores,
        "defect": [crawl["defect"][i] for i in alive],
    }


def stage_data(args):
    crawl = P.build_crawl()
    alive = np.load(P.data_dir() / "alive_after_cheap.npy")
    print(f"pool after project 37's cheap filters: {len(alive)} images")
    todo = [int(i) for i in alive if str(int(i)) not in P.load_recaptions()]
    print(f"{len(todo)} still need a recaption")
    if todo:
        cache, secs = P.recaption(crawl["images"], todo[:args.limit or len(todo)])
        print(f"recaptioned in {secs:.0f}s ({secs / max(len(todo), 1):.2f}s each)")
    p = pool()
    vocab = build_vocab(p["alt"] + p["recap"] + [c for h in p["human"] for c in h])
    stats = {
        "pool": int(len(alive)),
        "gallery": N_GALLERY,
        "train": int(len(alive) - N_GALLERY),
        "vocab": len(vocab),
        "mean_words": {k: float(np.mean([len(t.split()) for t in p[k]]))
                       for k in ("alt", "recap")},
        "human_mean_words": float(np.mean([len(h[0].split()) for h in p["human"]])),
        "unique_fraction": {k: len(set(t.lower() for t in p[k])) / len(p[k])
                            for k in ("alt", "recap")},
        "mean_clip": {"alt": float(p["clip"].mean())},
        "defects_left_in_pool": {d: int(sum(1 for x in p["defect"] if x == d))
                                 for d in P.DEFECTS},
    }
    _save("data.json", stats)
    print(json.dumps(stats, indent=1))


def splits(n, seed=0):
    order = np.random.default_rng(seed).permutation(n)
    return order[N_GALLERY:], order[:N_GALLERY]


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------
def arm_captions(p, arm, train_ids, rng_seed=0):
    """-> (captions per image as list[list[str]], usable training ids).

    `filtered` is the only arm that changes *which* images are usable; every
    other arm sees every image, so a win cannot come from seeing more pictures.
    """
    caps = [[""] for _ in range(len(p["alt"]))]
    if arm == "alt":
        for i in range(len(caps)):
            caps[i] = [p["alt"][i]]
        use = train_ids
    elif arm == "recap":
        for i in range(len(caps)):
            caps[i] = [p["recap"][i]]
        use = train_ids
    elif arm == "human":
        for i in range(len(caps)):
            caps[i] = [p["human"][i][0]]
        use = train_ids
    elif arm == "human5":
        # The control for the `blend` arm: same caption *source* as `human`, but
        # five captions per image instead of one. If a blend wins simply because
        # it hands the model two different sentences per picture, this arm --
        # which hands it five, all from one source -- must win too.
        for i in range(len(caps)):
            caps[i] = list(p["human"][i]) or [""]
        use = train_ids
    elif arm == "blend":
        # Both captions live in the list; `Pairs.batch` picks one at random every
        # time the image is drawn, so the mixture is 50/50 over the whole run.
        for i in range(len(caps)):
            caps[i] = [p["alt"][i], p["recap"][i]]
        use = train_ids
    elif arm in KEEP:
        for i in range(len(caps)):
            caps[i] = [p["alt"][i]]
        cut = np.quantile(p["clip"][train_ids], 1 - KEEP[arm])
        use = np.array([i for i in train_ids if p["clip"][i] >= cut])
    elif arm in DIRTY:
        # A dirtier version of the same pool: swap this share of the alt-text
        # between images, the way project 14 did. Real web alt-text is far worse
        # than MS-COCO's, and the recaption arms above cannot show what they are
        # for until the text they replace is actually broken.
        rng = np.random.default_rng(rng_seed + 11)
        alt = list(p["alt"])
        n = len(alt)
        hit = np.flatnonzero(rng.random(n) < DIRTY[arm])
        shuffled = rng.permutation(hit)
        for a, b in zip(hit, shuffled):
            alt[int(a)] = p["alt"][int(b)]
        for i in range(len(caps)):
            caps[i] = [alt[i]]
        use = train_ids
    else:
        raise ValueError(arm)
    return caps, np.asarray(use)


@torch.no_grad()
def gallery_recall(model, images, texts, vocab, batch=200):
    """Retrieval over a fixed gallery. `texts` is one caption per image."""
    vs, ts = [], []
    for i in range(0, len(images), batch):
        px = TC.to_pixels(images[i:i + batch])
        tok = torch.from_numpy(TC.tokenize(texts[i:i + batch], vocab))
        v, t = model.encode(px, tok)
        vs.append(v.numpy())
        ts.append(t.numpy())
    v, t = np.concatenate(vs), np.concatenate(ts)
    sim = v @ t.T
    return TC.recall_at_k(sim), TC.recall_at_k(sim.T)


def evaluate_arm(model, p, gal_ids, vocab):
    """Score on human captions (the honest test) and on BLIP captions (the
    test that quietly favours whoever trained on BLIP captions)."""
    imgs = p["images"][gal_ids]
    out = {}
    # Average over the five independent human captions: five gallery draws for
    # the price of one training run, which halves the noise on R@10.
    i2t, t2i = [], []
    for k in range(5):
        texts = [p["human"][i][k % len(p["human"][i])] for i in gal_ids]
        a, b = gallery_recall(model, imgs, texts, vocab)
        i2t.append(a)
        t2i.append(b)
    for k in (1, 5, 10):
        out[f"human_i2t_r{k}"] = float(np.mean([x[k] for x in i2t]))
        out[f"human_t2i_r{k}"] = float(np.mean([x[k] for x in t2i]))
    a, b = gallery_recall(model, imgs, [p["recap"][i] for i in gal_ids], vocab)
    for k in (1, 5, 10):
        out[f"blip_i2t_r{k}"] = float(a[k])
        out[f"blip_t2i_r{k}"] = float(b[k])
    a, b = gallery_recall(model, imgs, [p["alt"][i] for i in gal_ids], vocab)
    for k in (1, 5, 10):
        out[f"alt_i2t_r{k}"] = float(a[k])
        out[f"alt_t2i_r{k}"] = float(b[k])
    return out


def stage_train(args):
    p = pool()
    vocab = build_vocab(p["alt"] + p["recap"] + [c for h in p["human"] for c in h])
    train_ids, gal_ids = splits(len(p["alt"]))
    arms = args.arm.split(",") if args.arm else ARMS
    results = _load("results.json") if (OUT / "results.json").exists() else {}
    for arm in arms:
        caps, use = arm_captions(p, arm, train_ids)
        pairs = TC.Pairs(p["images"], caps, vocab, use)
        torch.manual_seed(0)
        model = TC.TinyCLIP(len(vocab))
        t0 = time.time()
        print(f"--- {arm}: {len(use)} training images, {args.steps} steps", flush=True)
        hist = TC.train(model, pairs, args.steps, batch=BATCH, lr=LR,
                        log_every=250)
        rec = evaluate_arm(model, p, gal_ids, vocab)
        rec.update(arm=arm, train_images=int(len(use)), steps=args.steps,
                   seconds=time.time() - t0,
                   final_train_loss=float(np.mean(hist[-50:])),
                   pairs_seen=int(args.steps * BATCH))
        results[arm] = rec
        _save("results.json", results)
        print(f"    human R@10 i2t {rec['human_i2t_r10']:.3f}"
              f"  t2i {rec['human_t2i_r10']:.3f}"
              f"  ({rec['seconds']:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------
def stage_plot(args):
    res = _load("results.json")
    order = [a for a in ARMS if a in res]
    label = {"alt": "alt-text\n(as crawled)",
             "filtered45": "CLIP-filtered\n(keep top 45%)",
             "filtered70": "CLIP-filtered\n(keep top 70%)",
             "recap": "recaptioned\n(BLIP)", "blend": "50/50 blend",
             "human": "human captions\n(1 per image)",
             "human5": "human captions\n(all 5)",
             "alt-dirty30": "alt-text\n+30% swapped",
             "alt-dirty60": "alt-text\n+60% swapped"}

    fig, ax = ps.new_axes(11.0, 4.4)
    x = np.arange(len(order))
    for k, (key, name, c) in enumerate((("human_i2t_r10", "image -> text", ps.SERIES[0]),
                                        ("human_t2i_r10", "text -> image", ps.SERIES[1]))):
        vals = [res[a][key] for a in order]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.34, color=c, label=name)
        for xi, v in zip(x + (k - 0.5) * 0.36, vals):
            ax.text(xi, v + 0.004, f"{v:.3f}", ha="center", fontsize=8,
                    color=ps.INK_SECONDARY)
    ax.axhline(10 / N_GALLERY, color=ps.INK_MUTED, linestyle="--", linewidth=1.0)
    ax.text(len(order) - 0.5, 10 / N_GALLERY + 0.003, "chance", fontsize=8,
            color=ps.INK_MUTED, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([label[a] for a in order], fontsize=7.5)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Scored on held-out human captions", "",
              f"recall@10 over {N_GALLERY} images", OUT / "recall.png")

    # the home-field-advantage figure
    fig, ax = ps.new_axes(11.0, 4.4)
    for k, (key, name, c) in enumerate((("human_t2i_r10", "judged on human captions", ps.SERIES[0]),
                                        ("blip_t2i_r10", "judged on BLIP captions", ps.SERIES[3]),
                                        ("alt_t2i_r10", "judged on web alt-text", ps.SERIES[2]))):
        vals = [res[a][key] for a in order]
        ax.bar(x + (k - 1) * 0.27, vals, 0.25, color=c, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([label[a] for a in order], fontsize=7.5)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Change the test set and the winner changes", "",
              "text -> image recall@10", OUT / "eval_set.png")

    # The explanatory figure: how well each arm generalised against how
    # thoroughly it memorised its own captions.
    fig, ax = ps.new_axes(7.4, 4.4)
    n_caps = {"blend": 2, "human5": 5}
    for k, a in enumerate(order):
        c = ps.SERIES[min(n_caps.get(a, 1) - 1, len(ps.SERIES) - 1)]
        y = 0.5 * (res[a]["human_i2t_r10"] + res[a]["human_t2i_r10"])
        ax.scatter(res[a]["final_train_loss"], y, s=80, color=c, zorder=3)
        off = (8, -11) if a == "alt-dirty60" else (8, 3)
        ax.annotate(f"{a} ({n_caps.get(a, 1)})", (res[a]["final_train_loss"], y),
                    textcoords="offset points", xytext=off, fontsize=8,
                    color=ps.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlim(0.002, 4.0)
    ps.finish(fig, ax, "Memorise less, generalise more (number of captions per image in brackets)",
              "training loss at the end of the run (log scale)",
              f"recall@10 over {N_GALLERY} images", OUT / "diversity.png")

    table = [{"arm": a, "train_images": res[a]["train_images"],
              "human_i2t_r10": res[a]["human_i2t_r10"],
              "human_t2i_r10": res[a]["human_t2i_r10"],
              "human_mean_r10": 0.5 * (res[a]["human_i2t_r10"] + res[a]["human_t2i_r10"]),
              "blip_t2i_r10": res[a]["blip_t2i_r10"],
              "train_loss": res[a]["final_train_loss"]} for a in order]
    _save("table.json", table)
    for r in table:
        print(f"{r['arm']:10s} n={r['train_images']:5d}  human R@10 "
              f"{r['human_mean_r10']:.3f}   BLIP-gallery {r['blip_t2i_r10']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["data", "train", "plot"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"data": stage_data, "train": stage_train, "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
