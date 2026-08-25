"""Modality balancing: one transformer, three modalities, one loss going wrong.

Project 33 showed that a single next-token loss can cover text and images. This
project adds audio and then shows the failure that appears the moment the
modalities are not equally represented: whichever modality supplies most of the
tokens gets most of the gradient, and the others quietly stop improving.

The measurement problem comes first. Text loss 1.2 and image loss 4.1 does NOT
mean text is doing better -- they are drawn from different-sized alphabets with
different amounts of inherent surprise, so the two numbers are not comparable
at all. The fix used throughout: train a solo model per modality first, and
report every joint model as a *gap to its own solo reference*.

Stages
    data     tokenize faces (project 32) and spoken digits (EnCodec)  (~2 min once)
    solo     one reference model per modality -- the ceilings         (~3 min)
    joint    four mixtures: natural / starved / oversampled / reweighted (~5 min)
    plot     figures + the gap table
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
sys.path.insert(0, str(HERE))
import plot_style as ps  # noqa: E402
import tri_modal as T  # noqa: E402
import unified as U  # noqa: E402

OUT = HERE / "outputs"
DATA = HERE / "data"
CTX = T.CTX
D, LAYERS, HEADS = 192, 4, 4
STEPS, BATCH, LR = 800, 32, 3e-3
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def stage_data():
    d, vocab = T.build()
    ir = T.image_rows(vocab, d["cap_tr"], d["img_tr"])
    ar = T.audio_rows(vocab, d["acap_tr"], d["aud_tr"])
    ii, aa = U.pad_batch(ir, CTX), U.pad_batch(ar, CTX)
    ki, ka = vocab.kind(ii), vocab.kind(aa)
    tot_text = int((ki == 1).sum() + (ka == 1).sum())
    tot_img = int((ki == 2).sum())
    tot_aud = int((ka == 3).sum())
    tot = tot_text + tot_img + tot_aud
    print(f"vocabulary {vocab.size} = {U.N_SPECIAL} specials "
          f"+ {len(vocab.words)} words + {vocab.n_image} image + {vocab.n_audio} audio")
    print(f"rows: {len(ir)} image+text, {len(ar)} audio+text")
    for n, v in (("text", tot_text), ("image", tot_img), ("audio", tot_aud)):
        print(f"  {n:6s} {v:8d} tokens = {100 * v / tot:5.1f}% of the corpus")
    _save("data.json", {
        "vocab": vocab.to_json(),
        "n_image_rows": len(ir), "n_audio_rows": len(ar),
        "tokens": {"text": tot_text, "image": tot_img, "audio": tot_aud},
        "share": {"text": tot_text / tot, "image": tot_img / tot, "audio": tot_aud / tot},
        "mean_text_tokens_per_row": float((ki == 1).sum(1).mean()),
        "image_tokens_per_row": int((ki == 2).sum(1)[0]),
        "audio_tokens_per_row": int((ka == 3).sum(1)[0]),
        "vocab_block_sizes": {"words": len(vocab.words), "image": vocab.n_image,
                              "audio": vocab.n_audio},
    })


# ---------------------------------------------------------------------------
def _fit(vocab, rows, val, steps, weights=None, sampler=None, tag="", seed=0):
    torch.manual_seed(seed)
    model = U.UnifiedLM(vocab.size, d=D, layers=LAYERS, heads=HEADS, ctx=CTX)
    t0 = time.time()
    hist = U.train_lm(model, U.pad_batch(rows, CTX), vocab, val_seqs=val,
                      steps=steps, batch=BATCH, lr=LR, weights=weights,
                      sampler=sampler, log_every=max(steps // 4, 100))
    ev = U.evaluate_lm(model, val, vocab)
    print(f"  [{tag}] text {ev['text']:.3f}  image {ev['image']:.3f}  "
          f"audio {ev['audio']:.3f}  ({time.time() - t0:.0f}s)")
    return model, hist, ev


def stage_solo(steps=STEPS):
    """One model per modality: the best each can do with the whole budget."""
    d, vocab = T.build()
    val = T.mixed_val(vocab, d)
    out = {}
    arms = {
        "image_solo": T.image_rows(vocab, d["cap_tr"], d["img_tr"]),
        "audio_solo": T.audio_rows(vocab, d["acap_tr"], d["aud_tr"]),
        "text_solo": T.text_rows(vocab, list(d["cap_tr"]) + list(d["acap_tr"])),
    }
    for name, rows in arms.items():
        print(f"\n--- {name} ({len(rows)} rows)")
        _, hist, ev = _fit(vocab, rows, val, steps, tag=name)
        out[name] = {"history": hist, "eval": ev, "n_rows": len(rows)}
    _save("solo.json", {"steps": steps, "arms": out})


def stage_joint(steps=STEPS):
    """Four mixtures of the same two corpora."""
    d, vocab = T.build()
    val = T.mixed_val(vocab, d)
    ir = T.image_rows(vocab, d["cap_tr"], d["img_tr"])
    ar = T.audio_rows(vocab, d["acap_tr"], d["aud_tr"])
    rows = ir + ar
    n_img, n_aud = len(ir), len(ar)

    def ratio_sampler(p_image):
        """Draw each example from the image corpus with probability p_image.

        This is the *data* lever: nothing about the loss changes, only how often
        the optimizer sees each modality.
        """
        def s(rng, batch):
            take_img = rng.random(batch) < p_image
            idx = np.where(take_img,
                           rng.integers(0, n_img, batch),
                           n_img + rng.integers(0, n_aud, batch))
            return idx
        return s

    natural = n_img / (n_img + n_aud)
    arms = {
        "natural": dict(sampler=None,
                        note=f"draw uniformly from all rows ({natural:.0%} are faces)"),
        "starved": dict(sampler=ratio_sampler(0.99),
                        note="99% faces, 1% audio -- the failure, on purpose"),
        "oversampled": dict(sampler=ratio_sampler(0.5),
                            note="50/50 rows -- the data-side fix"),
        "reweighted": dict(sampler=None, weights=None,
                           note="natural mixture, per-modality loss weights -- the loss-side fix"),
    }
    # the reweighting factors are the inverse token share, normalised to mean 1
    ki = vocab.kind(U.pad_batch(rows, CTX))
    share = {name: float((ki == code).sum())
             for name, code in (("text", 1), ("image", 2), ("audio", 3))}
    tot = sum(share.values())
    inv = {k: tot / (3 * v) for k, v in share.items()}
    arms["reweighted"]["weights"] = inv
    print(f"token shares {['%s %.3f' % (k, v / tot) for k, v in share.items()]}")
    print(f"inverse-share weights {inv}")

    out = {}
    for name, cfg in arms.items():
        print(f"\n--- {name}: {cfg['note']}")
        _, hist, ev = _fit(vocab, rows, val, steps, weights=cfg.get("weights"),
                           sampler=cfg.get("sampler"), tag=name)
        out[name] = {"history": hist, "eval": ev, "note": cfg["note"],
                     "weights": cfg.get("weights")}
    _save("joint.json", {"steps": steps, "arms": out, "token_share":
                         {k: v / tot for k, v in share.items()},
                         "inverse_weights": inv})


# ---------------------------------------------------------------------------
def stage_plot():
    solo = json.loads((OUT / "solo.json").read_text())["arms"]
    joint = json.loads((OUT / "joint.json").read_text())
    ceil = {m: solo[f"{m}_solo"]["eval"][m] for m in ("image", "audio", "text")}

    table = []
    for name, arm in joint["arms"].items():
        row = {"arm": name}
        for m in ("text", "image", "audio"):
            row[m] = arm["eval"][m]
            row[m + "_gap"] = arm["eval"][m] - ceil[m]
        table.append(row)
    _save("gaps.json", {"solo_reference": ceil, "table": table})
    print(f"\n{'arm':14s} " + "  ".join(f"{m:>16s}" for m in ("text", "image", "audio")))
    print(f"{'solo':14s} " + "  ".join(f"{ceil[m]:16.3f}" for m in ("text", "image", "audio")))
    for r in table:
        print(f"{r['arm']:14s} " + "  ".join(
            f"{r[m]:8.3f} ({r[m + '_gap']:+.3f})" for m in ("text", "image", "audio")))

    fig, ax = ps.new_axes(7.4, 4.2)
    names = [r["arm"] for r in table]
    x = np.arange(len(names))
    for i, m in enumerate(("text", "image", "audio")):
        ax.bar(x + (i - 1) * 0.27, [r[m + "_gap"] for r in table], 0.25,
               color=ps.SERIES[i], label=m)
    ax.axhline(0, color=ps.INK_MUTED, lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "How far each modality falls behind its own solo model",
              "", "joint loss − solo loss (nats/token; 0 = no cost)",
              OUT / "gaps.png")

    fig, ax = ps.new_axes(7.2, 4.2)
    for i, (name, arm) in enumerate(joint["arms"].items()):
        h = arm["history"]
        ax.plot([r["step"] for r in h], [r["val_audio"] for r in h],
                color=ps.SERIES[i], lw=2, label=name)
    ax.axhline(ceil["audio"], color=ps.BASELINE, lw=1.6, ls="--",
               label="audio solo reference")
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Starve a modality and its loss flat-lines",
              "training step", "validation loss on AUDIO tokens", OUT / "audio_curve.png")


STAGES = {"data": stage_data, "solo": stage_solo, "joint": stage_joint,
          "plot": stage_plot}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for nm in (list(STAGES) if a.stage == "all" else [a.stage]):
        print(f"\n=== {nm} " + "=" * (60 - len(nm)))
        if a.steps and nm in ("solo", "joint"):
            STAGES[nm](steps=a.steps)
        else:
            STAGES[nm]()
