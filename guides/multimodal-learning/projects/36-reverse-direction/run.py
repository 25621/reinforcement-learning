"""Reverse direction: teach a caption-only VLM to draw.

The starting point is deliberately one-way. We train a small VLM the way LLaVA
is trained: image tokens go IN as context, and the loss is applied only to the
caption that comes out. The image positions are never prediction targets, so
the model has no reason to learn what an image looks like from the inside -- it
only learns what images are *about*.

Then we bolt an image-token output head on and try four ways of teaching it the
other direction, and we score each one twice: how well it draws, and how much
captioning it forgot in the process.

    head_only     freeze everything, train only the new head
    lora          freeze everything, train the head + rank-8 LoRA corrections
    full          unfreeze everything (the usual "just fine-tune it")
    from_scratch  the control: same size, random init, trained on drawing only

Stages
    base    train the understanding-only VLM                (~2 min)
    graft   the four arms                                   (~6 min)
    gen     samples + the referee's verdict + forgetting     (~2 min)
    plot    figures
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "32-discrete-image-tokens"))
sys.path.insert(0, str(PROJECTS / "33-tiny-chameleon"))
sys.path.insert(0, str(HERE))
import graft as G  # noqa: E402
import plot_style as ps  # noqa: E402
import unified as U  # noqa: E402
import vqvae as VQ  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
CTX = 88          # 1 + 20 caption words + 1 + 64 image codes + 2 markers
D, LAYERS, HEADS = 192, 4, 4
BASE_STEPS, GRAFT_STEPS, BATCH, LR = 1000, 800, 32, 3e-3
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


# ---------------------------------------------------------------------------
# data: the same faces, in the two directions
# ---------------------------------------------------------------------------
def bundle():
    p = U.load_pairs()
    return dict(tr_c=p["tr_codes"], tr_cap=list(p["tr_caps"]),
                va_c=p["va_codes"], va_cap=list(p["va_caps"]),
                tr_i=p["tr_imgs"], tr_a=p["tr_attrs"],
                va_i=p["va_imgs"], va_a=p["va_attrs"],
                vocab=U.pair_vocab(p))


def i2t_rows(vocab, caps, codes):
    """<bos> <boi> image <eoi> caption <eos> -- the understanding direction."""
    return [[U.BOS, U.BOI] + vocab.image_ids(ic) + [U.EOI]
            + vocab.text_ids(c)[:U.TEXT_CTX] + [U.EOS] for c, ic in zip(caps, codes)]


def t2i_rows(vocab, caps, codes):
    """<bos> caption <boi> image <eoi> <eos> -- the generation direction."""
    return [[U.BOS] + vocab.text_ids(c)[:U.TEXT_CTX] + [U.BOI]
            + vocab.image_ids(ic) + [U.EOI, U.EOS] for c, ic in zip(caps, codes)]


def masked_loss(logits, targets, kinds, keep, pad=U.PAD):
    """Cross-entropy over only the token kinds in `keep`.

    Masking the image positions out of the loss is not a detail -- it is what
    makes the base model *understanding-only*. LLaVA does exactly this.
    """
    lp = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                         reduction="none")
    m = (targets.reshape(-1) != pad)
    sel = torch.zeros_like(m)
    for k in keep:
        sel |= (kinds.reshape(-1) == k)
    m = m & sel
    return (lp * m).sum() / m.sum().clamp_min(1)


def fit(model, rows, vocab, keep, steps, val=None, val_keep=None, lr=LR,
        params=None, seed=0, tag="", log_every=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    seqs = torch.from_numpy(U.pad_batch(rows, CTX))
    kinds = torch.from_numpy(vocab.kind(seqs.numpy()))
    ps_ = [p for p in (params if params is not None else model.parameters())
           if p.requires_grad]
    opt = torch.optim.AdamW(ps_, lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    log_every = log_every or max(steps // 4, 100)
    hist, t0 = [], time.time()
    for step in range(1, steps + 1):
        p = rng.integers(0, len(rows), BATCH)
        ids = seqs[p]
        loss = masked_loss(model(ids[:, :-1]), ids[:, 1:], kinds[p][:, 1:], keep)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ps_, 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            rec = {"step": step, "train": float(loss.detach()),
                   "secs": time.time() - t0}
            if val is not None:
                rec["val"] = eval_masked(model, val, vocab, val_keep or keep)
            hist.append(rec)
            print(f"  [{tag}] step {step:5d}  train {float(loss.detach()):.3f}"
                  + (f"  val {rec['val']:.3f}" if val is not None else "")
                  + f"  {time.time() - t0:5.0f}s", flush=True)
    return hist


@torch.no_grad()
def eval_masked(model, rows, vocab, keep, batch=64):
    seqs = torch.from_numpy(U.pad_batch(rows, CTX))
    kinds = torch.from_numpy(vocab.kind(seqs.numpy()))
    tot, n = 0.0, 0
    for i in range(0, len(seqs), batch):
        ids = seqs[i:i + batch]
        lp = F.cross_entropy(model(ids[:, :-1]).reshape(-1, model_vocab(model)),
                             ids[:, 1:].reshape(-1), reduction="none")
        m = (ids[:, 1:].reshape(-1) != U.PAD)
        sel = torch.zeros_like(m)
        for k in keep:
            sel |= (kinds[i:i + batch, 1:].reshape(-1) == k)
        m = m & sel
        tot += float((lp * m).sum()); n += int(m.sum())
    return tot / max(n, 1)


def model_vocab(model):
    b = model.backbone if hasattr(model, "backbone") else model
    return b.head.out_features


# ---------------------------------------------------------------------------
def stage_base(steps=BASE_STEPS):
    """The understanding-only VLM: image in, caption out, loss on text only."""
    b = bundle()
    vocab = b["vocab"]
    CKPT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    model = U.UnifiedLM(vocab.size, d=D, layers=LAYERS, heads=HEADS, ctx=CTX)
    tr = i2t_rows(vocab, b["tr_cap"], b["tr_c"])
    va = i2t_rows(vocab, b["va_cap"], b["va_c"])
    hist = fit(model, tr, vocab, keep=(0, 1), steps=steps, val=va, tag="base")
    # What does it think about image tokens it was never asked to predict?
    # Measured on the SAME i2t rows it trained on, so nothing but the choice of
    # which positions we score has changed.
    img_loss = eval_masked(model, va, vocab, keep=(2,))
    print(f"base VLM: caption loss {hist[-1]['val']:.3f}, "
          f"image-token loss {img_loss:.3f} (never trained -- "
          f"chance is {np.log(VQ.CODEBOOK):.3f})")
    torch.save(model.state_dict(), CKPT / "base.pt")
    _save("base.json", {"steps": steps, "history": hist,
                        "caption_loss": hist[-1]["val"],
                        "image_loss_untrained": img_loss,
                        "chance_image_loss": float(np.log(VQ.CODEBOOK)),
                        "params": sum(p.numel() for p in model.parameters())})


def _fresh_backbone(vocab):
    torch.manual_seed(0)
    return U.UnifiedLM(vocab.size, d=D, layers=LAYERS, heads=HEADS, ctx=CTX)


def _load_base(vocab):
    m = _fresh_backbone(vocab)
    m.load_state_dict(torch.load(CKPT / "base.pt", map_location="cpu"))
    return m


def stage_graft(steps=GRAFT_STEPS):
    b = bundle()
    vocab = b["vocab"]
    tr = t2i_rows(vocab, b["tr_cap"], b["tr_c"])
    va = t2i_rows(vocab, b["va_cap"], b["va_c"])
    i2t_va = i2t_rows(vocab, b["va_cap"], b["va_c"])
    base_caption = json.loads((OUT / "base.json").read_text())["caption_loss"]

    def make(arm):
        if arm == "from_scratch":
            bb = _fresh_backbone(vocab)
        else:
            bb = _load_base(vocab)
        m = G.Grafted(bb, vocab.image_base, vocab.n_image, d=D,
                      use_new_head=(arm != "tied_head_only"))
        if arm == "head_only":
            for p in bb.parameters():
                p.requires_grad_(False)
        elif arm == "tied_head_only":
            # No new head at all: the only thing that may move is the tied
            # embedding/output matrix the base model already had. This is the
            # honest "you already have image codes in the vocabulary" option.
            for p in bb.parameters():
                p.requires_grad_(False)
            bb.tok.weight.requires_grad_(True)
            for p in m.image_head.parameters():
                p.requires_grad_(False)
        elif arm == "lora":
            for p in bb.parameters():
                p.requires_grad_(False)
            n = G.inject_lora(bb, r=8, alpha=16)
            print(f"    LoRA on {n} linear layers")
        return m

    arms = ["head_only", "tied_head_only", "lora", "full", "from_scratch"]
    out = {}
    for arm in arms:
        print(f"\n--- {arm}")
        m = make(arm)
        n_train = G.trainable(m)
        hist = fit(m, tr, vocab, keep=(2,), steps=steps, val=va, tag=arm)
        img = eval_masked(m, va, vocab, keep=(2,))
        cap = eval_masked(m, i2t_va, vocab, keep=(0, 1))
        out[arm] = {"history": hist, "image_loss": img, "caption_loss_after": cap,
                    "caption_loss_before": base_caption,
                    "forgetting": cap - base_caption,
                    "trainable_params": n_train,
                    "total_params": sum(p.numel() for p in m.parameters())}
        print(f"    image loss {img:.3f}   caption loss {cap:.3f} "
              f"(was {base_caption:.3f}, so {cap - base_caption:+.3f})   "
              f"{n_train/1e6:.2f}M trainable")
        torch.save(m.state_dict(), CKPT / f"{arm}.pt")
    _save("graft.json", {"steps": steps, "arms": out})


# ---------------------------------------------------------------------------
# The same minimal-pair prompts project 33 used, so the two projects' numbers
# are directly comparable. Each pair changes exactly ONE attribute word; a pair
# that changed several at once would make the referee's swing uninterpretable,
# because you could not tell which word moved it.
GEN_TESTS = [
    ("Male", "a young man", "a young woman"),
    ("Blond_Hair", "a young woman with blond hair", "a young woman with black hair"),
]
GEN_ARMS = ("head_only", "lora", "full", "from_scratch")


def stage_gen(n_gen=48):
    b = bundle()
    vocab = b["vocab"]
    tok = VQ.load_tokenizer()
    probe = U.AttrProbe()
    probe.load_state_dict(torch.load(PROJECTS / "33-tiny-chameleon" /
                                     "checkpoints" / "probe.pt", map_location="cpu"))
    probe.eval()
    n_img = int(b["va_c"].shape[1])

    # the ceiling: the same referee comparing REAL faces that have the
    # attribute against real faces that do not
    ceiling = {}
    for attr, _, _ in GEN_TESTS:
        ix = VQ.ATTRS.index(attr)
        j = U.GRADED_ATTRS.index(attr)
        with torch.no_grad():
            pos = torch.sigmoid(probe(VQ.to_tensor(
                b["va_i"][b["va_a"][:, ix] > 0][:128])))[:, j].mean()
            neg = torch.sigmoid(probe(VQ.to_tensor(
                b["va_i"][b["va_a"][:, ix] < 0][:128])))[:, j].mean()
        ceiling[attr] = float(pos - neg)

    res, panels, titles = {}, [], []
    for arm in GEN_ARMS:
        bb = _fresh_backbone(vocab)
        m = G.Grafted(bb, vocab.image_base, vocab.n_image, d=D)
        if arm == "lora":
            G.inject_lora(bb, r=8, alpha=16)
        m.load_state_dict(torch.load(CKPT / f"{arm}.pt", map_location="cpu"))
        m.eval()
        row = {}
        for attr, pos_prompt, neg_prompt in GEN_TESTS:
            j = U.GRADED_ATTRS.index(attr)
            scores = {}
            for tag, prompt in (("positive", pos_prompt), ("negative", neg_prompt)):
                ids = [[U.BOS] + vocab.text_ids(prompt) + [U.BOI]] * n_gen
                gen = U.sample(m, ids, n_img, temperature=1.0, top_k=100, seed=5,
                               allow=(vocab.image_base,
                                      vocab.image_base + vocab.n_image))
                codes = vocab.decode_image(gen[:, -n_img:].reshape(-1).tolist())
                imgs = tok.decode_indices(torch.from_numpy(
                    codes.reshape(n_gen, tok.grid, tok.grid)))
                with torch.no_grad():
                    scores[tag] = float(torch.sigmoid(probe(imgs))[:, j].mean())
                if attr == "Blond_Hair" and tag == "positive":
                    panels.append(VQ.to_uint8(imgs[:8]))
                    titles.append(arm)
            scores["swing"] = scores["positive"] - scores["negative"]
            scores["real_swing"] = ceiling[attr]
            scores["obeyed"] = (scores["swing"] / ceiling[attr]) if ceiling[attr] else 0.0
            scores["positive_prompt"] = pos_prompt
            scores["negative_prompt"] = neg_prompt
            row[attr] = scores
            print(f"  {arm:14s} {attr:11s} pos {scores['positive']:.3f} "
                  f"neg {scores['negative']:.3f}  swing {scores['swing']:+.3f} "
                  f"(real {ceiling[attr]:+.3f}, obeyed {100*scores['obeyed']:.0f}%)")
        res[arm] = row
    _panels(panels, titles, OUT / "samples.png")
    _save("gen.json", {"tests": [{"attribute": a, "positive": p, "negative": n}
                                 for a, p, n in GEN_TESTS],
                       "real_swing": ceiling, "arms": res})


def _panels(rows, titles, path):
    import matplotlib.pyplot as plt
    n = min(8, min(len(r) for r in rows))
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.1, len(rows) * 1.28), dpi=110)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(ps.SURFACE)
    for i, row in enumerate(rows):
        for j in range(n):
            axes[i, j].imshow(row[j])
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_color(ps.BASELINE)
        axes[i, 0].set_ylabel(titles[i], color=ps.INK_SECONDARY, fontsize=9,
                              rotation=0, ha="right", va="center", labelpad=6)
    fig.suptitle("'a young woman with blond hair', four ways of grafting",
                 color=ps.INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def stage_plot():
    g = json.loads((OUT / "graft.json").read_text())["arms"]
    names = list(g)
    fig, ax = ps.new_axes(7.2, 4.2)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [g[n]["image_loss"] for n in names], 0.38, color=ps.SERIES[0],
           label="image-token loss (lower = draws better)")
    ax.bar(x + 0.2, [g[n]["forgetting"] for n in names], 0.38, color=ps.SERIES[2],
           label="captioning loss increase (higher = forgot more)")
    ax.axhline(0, color=ps.INK_MUTED, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n{g[n]['trainable_params']/1e6:.2f}M trainable"
                        for n in names], fontsize=8)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Learning to draw, and what it costs the old skill",
              "", "nats/token", OUT / "arms.png")

    fig, ax = ps.new_axes(7.0, 4.2)
    for i, n in enumerate(names):
        h = g[n]["history"]
        ax.plot([r["step"] for r in h], [r["val"] for r in h],
                color=ps.SERIES[i % len(ps.SERIES)], lw=2, label=n)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Image-token loss while learning the reverse direction",
              "training step", "validation loss on image tokens", OUT / "curves.png")


STAGES = {"base": stage_base, "graft": stage_graft, "gen": stage_gen,
          "plot": stage_plot}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for nm in (list(STAGES) if a.stage == "all" else [a.stage]):
        print(f"\n=== {nm} " + "=" * (60 - len(nm)))
        if a.steps and nm in ("base", "graft"):
            STAGES[nm](steps=a.steps)
        else:
            STAGES[nm]()
