"""Modality ratio sweep: turn one data knob and watch two loss curves move.

One transformer, one vocabulary, one next-token loss -- the Phase-7 stack from
project 33. The corpus is two kinds of row:

    text row    <bos> a smiling young woman <eos>              ~14 tokens
    image row   <bos> <boi> 391 12 508 ... 77 <eoi> <eos>       68 tokens

and the single knob is `p`, the probability that a row drawn into the training
batch is an image row. Sweeping p from 0 to 1 gives a whole family of corpora,
and the two ends double as the *reference ceilings*: p=0 is a text-only model,
p=1 is an image-only model, both trained with exactly the same compute as every
mixture in between. That is what makes the middle readable -- a loss of 4.7 on
images means nothing until you know an image-only model reaches 4.6.

Stages
    data      build the rows and the vocabulary                     (~10 s)
    sweep     nine mixtures, val loss logged per modality           (~10 min)
    remedy    loss re-weighting at a starved ratio                  (~2 min)
    plot      figures + tables                                      (~10 s)

    python3 run.py --stage data
    python3 run.py --stage sweep
    python3 run.py --stage remedy
    python3 run.py --stage plot

Requires project 32's tokenizer and project 33's token cache; both rebuild
themselves with `python3 run.py --stage data` inside those directories.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "32-discrete-image-tokens"))
sys.path.insert(0, str(PROJECTS / "33-tiny-chameleon"))
import plot_style as ps  # noqa: E402
import unified as U  # noqa: E402

OUT = HERE / "outputs"
DATA = HERE / "data"
CTX = 68                      # 1 bos + 1 boi + 64 codes + 1 eoi + 1 eos
D, LAYERS, HEADS = 192, 4, 4
STEPS, BATCH, LR = 600, 32, 3e-3
LOG_EVERY = 50
N_VAL = 400

# The sweep. p is the share of *rows* that carry an image; the share of
# *tokens* that are image tokens is a different (and much larger) number,
# which is one of the things this project measures.
RATIOS = [0.0, 0.02, 0.05, 0.12, 0.25, 0.5, 0.75, 0.95, 1.0]
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
def text_row(vocab, caption):
    return [U.BOS] + vocab.text_ids(caption)[:U.TEXT_CTX] + [U.EOS]


def image_row(vocab, codes):
    return [U.BOS, U.BOI] + vocab.image_ids(codes) + [U.EOI, U.EOS]


def build():
    """-> (text rows, image rows, val rows, vocab). Cached on disk."""
    DATA.mkdir(exist_ok=True)
    cache = DATA / "rows.npz"
    pairs = U.load_pairs()
    vocab = U.pair_vocab(pairs)
    if cache.exists():
        z = np.load(cache)
        return z["text"], z["image"], z["val"], vocab
    tr_caps, tr_codes = pairs["tr_caps"], pairs["tr_codes"]
    va_caps, va_codes = pairs["va_caps"][:N_VAL], pairs["va_codes"][:N_VAL]
    text = U.pad_batch([text_row(vocab, c) for c in tr_caps], CTX)
    image = U.pad_batch([image_row(vocab, c) for c in tr_codes], CTX)
    val = U.pad_batch([text_row(vocab, c) for c in va_caps]
                      + [image_row(vocab, c) for c in va_codes], CTX)
    np.savez_compressed(cache, text=text, image=image, val=val)
    return text, image, val, vocab


def stage_data(args):
    text, image, val, vocab = build()
    kt, ki = vocab.kind(text), vocab.kind(image)
    t_tok = int((kt == 1).sum())
    i_tok = int((ki == 2).sum())
    stats = {
        "vocab_size": vocab.size,
        "words": len(vocab.words),
        "image_codes": vocab.n_image,
        "text_rows": int(len(text)),
        "image_rows": int(len(image)),
        "val_rows": int(len(val)),
        "text_tokens_per_row": t_tok / len(text),
        "image_tokens_per_row": i_tok / len(image),
        # If you mix rows 50/50, the token mix is nowhere near 50/50 -- an image
        # row is several times longer than a caption.
        "token_share_at_50_50_rows": 64.0 / (64.0 + t_tok / len(text)),
    }
    _save("data.json", stats)
    print(json.dumps(stats, indent=1))


def token_share(p, text, image, vocab):
    """Expected share of *image* tokens in a batch, given a row ratio p."""
    t = float((vocab.kind(text) == 1).sum()) / len(text)
    i = float((vocab.kind(image) == 2).sum()) / len(image)
    return p * i / (p * i + (1 - p) * t)


# ---------------------------------------------------------------------------
# one training run
# ---------------------------------------------------------------------------
def run_arm(text, image, val, vocab, p, steps=STEPS, weights=None, seed=0):
    """Train one mixture. Rows are drawn from the text block with probability
    1-p and from the image block with probability p."""
    seqs = np.concatenate([text, image])
    n_text = len(text)

    def sampler(rng, batch):
        pick_img = rng.random(batch) < p
        out = np.where(pick_img,
                       n_text + rng.integers(0, len(image), batch),
                       rng.integers(0, n_text, batch))
        return out

    torch.manual_seed(seed)
    model = U.UnifiedLM(vocab.size, d=D, layers=LAYERS, heads=HEADS, ctx=CTX)
    t0 = time.time()
    hist = U.train_lm(model, seqs, vocab, val_seqs=val, steps=steps, batch=BATCH,
                      lr=LR, seed=seed, log_every=LOG_EVERY, sampler=sampler,
                      weights=weights, verbose=False)
    final = U.evaluate_lm(model, val, vocab)
    share = token_share(p, text, image, vocab)

    def blend(t, i):
        """The number a practitioner actually watches: one loss averaged over the
        tokens in *their* mixture. It is a weighted average of the two curves,
        and the weights are the mixture -- which is exactly why it can hide a
        modality that supplies almost none of the tokens."""
        return share * i + (1 - share) * t

    return {
        "p_rows": p,
        "token_share_image": share,
        "weights": weights,
        "seconds": time.time() - t0,
        "curve": [{"step": h["step"], "text": h["val_text"], "image": h["val_image"],
                   "mixture_blend": blend(h["val_text"], h["val_image"])}
                  for h in hist],
        "final": {"text": final["text"], "image": final["image"],
                  "mixture_blend": blend(final["text"], final["image"])},
    }


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_sweep(args):
    text, image, val, vocab = build()
    ratios = [float(x) for x in args.ratios.split(",")] if args.ratios else RATIOS
    prev = _load("sweep.json") if (OUT / "sweep.json").exists() else {"arms": []}
    done = {round(a["p_rows"], 4) for a in prev["arms"]}
    arms = list(prev["arms"])
    for p in ratios:
        if round(p, 4) in done:
            print(f"p={p} already done, skipping")
            continue
        print(f"--- p_rows={p}  (image token share "
              f"{token_share(p, text, image, vocab):.3f})", flush=True)
        arm = run_arm(text, image, val, vocab, p, steps=args.steps)
        print(f"    text {arm['final']['text']:.3f}  image {arm['final']['image']:.3f}"
              f"  blended {arm['final']['mixture_blend']:.3f}  {arm['seconds']:.0f}s", flush=True)
        arms.append(arm)
        arms.sort(key=lambda a: a["p_rows"])
        _save("sweep.json", {"steps": args.steps, "arms": arms})
    _save("sweep.json", {"steps": args.steps, "arms": arms})
    summarise(arms)


def summarise(arms):
    by_p = {round(a["p_rows"], 4): a for a in arms}
    ceil_text = by_p.get(0.0, min(arms, key=lambda a: a["final"]["text"]))
    ceil_img = by_p.get(1.0, min(arms, key=lambda a: a["final"]["image"]))
    rows = []
    for a in arms:
        rows.append({
            "p_rows": a["p_rows"],
            "image_token_share": a["token_share_image"],
            "text": a["final"]["text"],
            "image": a["final"]["image"],
            "blended": a["final"]["mixture_blend"],
            "text_gap": a["final"]["text"] - ceil_text["final"]["text"],
            "image_gap": a["final"]["image"] - ceil_img["final"]["image"],
        })
    for r in rows:
        r["max_gap"] = max(r["text_gap"], r["image_gap"])
    best = min([r for r in rows if 0 < r["p_rows"] < 1], key=lambda r: r["max_gap"])
    _save("summary.json", {
        "ceilings": {"text": ceil_text["final"]["text"],
                     "image": ceil_img["final"]["image"]},
        "rows": rows,
        "balanced_ratio": best,
    })
    print(f"{'p':>6} {'img tok%':>9} {'text':>7} {'image':>7} {'blend':>7}"
          f" {'text gap':>9} {'img gap':>8}")
    for r in rows:
        print(f"{r['p_rows']:6.2f} {100 * r['image_token_share']:9.1f}"
              f" {r['text']:7.3f} {r['image']:7.3f} {r['blended']:7.3f}"
              f" {r['text_gap']:9.3f} {r['image_gap']:8.3f}")


def stage_remedy(args):
    """At a starved ratio, does re-weighting the loss do what re-mixing the data does?"""
    text, image, val, vocab = build()
    sweep = _load("sweep.json")["arms"]
    starved_p = 0.05
    starved = next(a for a in sweep if abs(a["p_rows"] - starved_p) < 1e-6)
    share = starved["token_share_image"]
    # Inverse-share weights, normalised so the average weight stays 1: the
    # textbook fix for an unbalanced mixture.
    w_img = 0.5 / share
    w_txt = 0.5 / (1 - share)
    norm = share * w_img + (1 - share) * w_txt
    weights = {"image": w_img / norm, "text": w_txt / norm}
    print(f"starved p={starved_p}: image token share {share:.3f} -> weights {weights}")
    arm = run_arm(text, image, val, vocab, starved_p, steps=args.steps,
                  weights=weights)
    print(f"    reweighted: text {arm['final']['text']:.3f}"
          f"  image {arm['final']['image']:.3f}", flush=True)
    # The data-side fix at the same starved ratio: whichever swept ratio puts
    # half the *tokens* on images.
    matched = min(sweep, key=lambda a: abs(a["token_share_image"] - 0.5))
    _save("remedy.json", {
        "starved": starved, "reweighted": arm, "rebalanced_data": matched,
        "weights": weights,
    })


def stage_plot(args):
    sweep = _load("sweep.json")["arms"]
    summ = _load("summary.json")

    # 1. the diagnostic itself: per-modality curves for four mixtures
    show = [a for a in sweep if round(a["p_rows"], 4) in (0.02, 0.12, 0.5, 0.95)]
    fig, ax = ps.new_axes(7.6, 4.4)
    for k, a in enumerate(show):
        c = ps.SERIES[k]
        xs = [h["step"] for h in a["curve"]]
        ax.plot(xs, [h["image"] for h in a["curve"]], color=c, linewidth=1.9,
                label=f"p={a['p_rows']:.2f} image")
        ax.plot(xs, [h["text"] for h in a["curve"]], color=c, linewidth=1.4,
                linestyle="--", label=f"p={a['p_rows']:.2f} text")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ps.finish(fig, ax, "Solid = image loss, dashed = text loss",
              "training step", "validation loss (nats/token)", OUT / "curves.png")

    # 2. the trap: the blended loss looks fine while one modality is stuck
    a = next(x for x in sweep if abs(x["p_rows"] - 0.02) < 1e-6)
    b = next(x for x in sweep if abs(x["p_rows"] - 0.5) < 1e-6)
    fig, ax = ps.new_axes(7.4, 4.2)
    for arm, name, c in ((a, "starved p=0.02", ps.SERIES[2]),
                         (b, "balanced p=0.50", ps.SERIES[0])):
        xs = [h["step"] for h in arm["curve"]]
        ax.plot(xs, [h["mixture_blend"] for h in arm["curve"]], color=c,
                linewidth=2.4, label=f"{name}: blended")
        ax.plot(xs, [h["image"] for h in arm["curve"]], color=c, linewidth=1.4,
                linestyle=":", label=f"{name}: image only")
    ax.legend(frameon=False, fontsize=8.5)
    ps.finish(fig, ax, "One blended number hides a stalled modality",
              "training step", "validation loss (nats/token)", OUT / "blended.png")

    # 3. gap-to-ceiling vs the token share
    fig, ax = ps.new_axes(7.4, 4.2)
    rows = summ["rows"]
    xs = [100 * r["image_token_share"] for r in rows]
    ax.plot(xs, [r["text_gap"] for r in rows], "-o", color=ps.SERIES[0],
            label="text: loss above a text-only model")
    ax.plot(xs, [r["image_gap"] for r in rows], "-o", color=ps.SERIES[2],
            label="image: loss above an image-only model")
    # The p=0 arm never sees an image token, so its image gap is enormous (the
    # model actively learns to suppress codes it is never asked to predict).
    # Plotting it linearly would flatten every other point into one line, so the
    # axis is clipped and the outlier is called out in words instead.
    inner = [r for r in rows if 0 < r["p_rows"] < 1]
    top = max(max(r["image_gap"] for r in inner),
              max(r["text_gap"] for r in inner)) * 1.25
    ax.set_ylim(min(-0.05, min(r["text_gap"] for r in rows) * 1.3), top)
    for r in rows:
        if r["image_gap"] > top:
            ax.annotate(f"{r['image_gap']:.1f} off the top",
                        (100 * r["image_token_share"], top * 0.92),
                        fontsize=8, color=ps.SERIES[2])
    bal = summ["balanced_ratio"]
    ax.axvline(100 * bal["image_token_share"], color=ps.INK_MUTED,
               linestyle="--", linewidth=1.1)
    ax.text(100 * bal["image_token_share"], ax.get_ylim()[1] * 0.9,
            f"  best worst-case\n  p={bal['p_rows']:.2f}", fontsize=8,
            color=ps.INK_MUTED, va="top")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "What each mixture costs each modality",
              "share of training tokens that are image tokens (%)",
              "loss above that modality's own ceiling", OUT / "gaps.png")

    # 4. rows vs tokens
    fig, ax = ps.new_axes(6.6, 4.0)
    ax.plot([100 * r["p_rows"] for r in rows],
            [100 * r["image_token_share"] for r in rows], "-o", color=ps.SERIES[4])
    ax.plot([0, 100], [0, 100], color=ps.BASELINE, linewidth=1.0,
            linestyle="--")
    ax.text(52, 30, "if rows were tokens", fontsize=8, color=ps.INK_MUTED)
    ps.finish(fig, ax, "A 50/50 row mix is not a 50/50 token mix",
              "image rows (% of rows)", "image tokens (% of tokens)",
              OUT / "rows_vs_tokens.png")

    if (OUT / "remedy.json").exists():
        rem = _load("remedy.json")
        fig, ax = ps.new_axes(7.0, 4.0)
        names = ["starved\np=0.05", "starved + loss\nre-weighting",
                 f"re-mixed data\np={rem['rebalanced_data']['p_rows']:.2f}"]
        arms = [rem["starved"], rem["reweighted"], rem["rebalanced_data"]]
        x = np.arange(3)
        ax.bar(x - 0.19, [a["final"]["image"] for a in arms], 0.36,
               color=ps.SERIES[2], label="image loss")
        ax.bar(x + 0.19, [a["final"]["text"] for a in arms], 0.36,
               color=ps.SERIES[0], label="text loss")
        for i, a in enumerate(arms):
            ax.text(i - 0.19, a["final"]["image"] + 0.05,
                    f"{a['final']['image']:.2f}", ha="center", fontsize=8,
                    color=ps.INK_SECONDARY)
            ax.text(i + 0.19, a["final"]["text"] + 0.05, f"{a['final']['text']:.2f}",
                    ha="center", fontsize=8, color=ps.INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5)
        ax.legend(frameon=False, fontsize=9)
        ps.finish(fig, ax, "Two ways to un-starve a modality", "",
                  "validation loss (nats/token)", OUT / "remedy.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["data", "sweep", "remedy", "plot"])
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--ratios", default="")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    {"data": stage_data, "sweep": stage_sweep, "remedy": stage_remedy,
     "plot": stage_plot}[args.stage](args)


if __name__ == "__main__":
    main()
