"""Tiny Chameleon: one vocabulary, one transformer, one loss, both directions.

Project 32 turned a face into 64 whole numbers. Splice those numbers into a
sentence and you can train an ordinary language model on the mixture -- no
vision tower, no projector, no fusion module. This project builds that model
and then asks the only questions that matter:

  1. Does one model really do both jobs, or does it just do one badly?
  2. Does sharing hurt? (controls: an image-only LM and a text-only LM)
  3. When you ask it for "a smiling man with glasses", does the picture it
     draws actually show a smiling man with glasses? (an independent referee
     answers, not the model itself)

Stages
    data    tokenize 8,000 faces with project 32's frozen VQ-VAE  (~30 s)
    probe   train the attribute referee on REAL faces             (~1 min)
    train   the unified model + the two single-modality controls  (~7 min)
    gen     text -> image and image -> text, both graded          (~2 min)
    plot    figures
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
sys.path.insert(0, str(PROJECTS / "01-modality-survey"))
sys.path.insert(0, str(PROJECTS / "32-discrete-image-tokens"))
sys.path.insert(0, str(HERE))
import plot_style as ps  # noqa: E402
import unified as U  # noqa: E402
import vqvae as VQ  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
DATA = HERE / "data"
CTX = 88          # 1 + 20 caption words + 1 + 64 image codes + 2 markers
D, LAYERS, HEADS = 192, 4, 4
STEPS, BATCH, LR = 1500, 32, 3e-3
torch.set_num_threads(6)


def _save(name, obj):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1))
    print(f"wrote outputs/{name}")


# ---------------------------------------------------------------------------
def build_data():
    """Tokenize every face once with the frozen tokenizer from project 32.

    The loader itself lives in `unified.py` so that projects 34/35/36 can call
    it without importing this script.
    """
    p = U.load_pairs()
    return (p["tr_codes"], list(p["tr_caps"]), p["va_codes"], list(p["va_caps"]),
            p["tr_imgs"], p["tr_attrs"], p["va_imgs"], p["va_attrs"])


def get_vocab(tr_caps, va_caps):
    return U.build_vocab(list(tr_caps) + list(va_caps), VQ.CODEBOOK)


def stage_data():
    tr_c, tr_cap, va_c, va_cap, tr_i, _, va_i, _ = build_data()
    vocab = get_vocab(tr_cap, va_cap)
    row = U.pair_sequence(vocab, tr_cap[0], tr_c[0], "t2i")
    print(f"vocabulary: {vocab.size} entries "
          f"= {U.N_SPECIAL} specials + {len(vocab.words)} words + {vocab.n_image} image codes")
    print(f"example row ({len(row)} tokens): {row[:12]} ... {row[-4:]}")
    print(f"  decoded text part: '{vocab.decode_text(row)}'")
    seqs = U.build_sequences(vocab, tr_cap, tr_c, length=CTX)
    real = seqs != U.PAD          # padding also has kind 0; exclude it
    kinds = np.where(real, vocab.kind(seqs), -1)
    n_real = int(real.sum())
    _save("data.json", {
        "vocab": vocab.to_json() | {"words_preview": vocab.words[:40]},
        "seq_len": CTX,
        "n_train": len(seqs), "n_val": len(va_c),
        "tokens_per_image": int(tr_c.shape[1]),
        "example_caption": str(tr_cap[0]),
        "example_row_head": row[:12],
        "real_tokens": n_real,
        "token_share": {
            "text": float((kinds == 1).sum() / n_real),
            "image": float((kinds == 2).sum() / n_real),
            "special": float((kinds == 0).sum() / n_real),
        },
        "mean_text_tokens": float((kinds == 1).sum(1).mean()),
    })


# ---------------------------------------------------------------------------
def stage_probe(steps=700):
    """Train the referee that grades generated faces."""
    _, _, _, _, tr_i, tr_a, va_i, va_a = build_data()
    probe, acc = U.train_probe(tr_i, tr_a, va_i, va_a, steps=steps)
    CKPT.mkdir(exist_ok=True)
    torch.save(probe.state_dict(), CKPT / "probe.pt")
    base = {}
    for n in U.GRADED_ATTRS:
        y = (va_a[:, VQ.ATTRS.index(n)] > 0).mean()
        base[n] = float(max(y, 1 - y))
    _save("probe.json", {"val_accuracy": acc, "majority_baseline": base,
                         "steps": steps})


# ---------------------------------------------------------------------------
def _make_model(vocab, mlp_factory=None, seed=0):
    torch.manual_seed(seed)
    return U.UnifiedLM(vocab.size, d=D, layers=LAYERS, heads=HEADS, ctx=CTX,
                       mlp_factory=mlp_factory)


def stage_train(steps=STEPS):
    tr_c, tr_cap, va_c, va_cap, *_ = build_data()
    vocab = get_vocab(tr_cap, va_cap)
    CKPT.mkdir(exist_ok=True)
    results = {}

    arms = {
        # the unified model: both modalities, both orders, one loss
        "unified": dict(orders=("t2i", "i2t"), keep=("text", "image")),
        # control 1: the same transformer, images only. If the unified model's
        # image loss is worse than this, sharing cost something.
        "image_only": dict(orders=("t2i",), keep=("image",)),
        # control 2: the same transformer, captions only.
        "text_only": dict(orders=("t2i",), keep=("text",)),
    }
    for name, cfg in arms.items():
        print(f"\n--- {name}")
        tr = _strip(vocab, tr_cap, tr_c, cfg["orders"], cfg["keep"], seed=1)
        va = _strip(vocab, va_cap, va_c, ("t2i", "i2t") if len(cfg["orders"]) > 1
                    else cfg["orders"], cfg["keep"], seed=2)
        model = _make_model(vocab)
        n_steps = steps if name != "text_only" else max(steps // 3, min(steps, 400))
        hist = U.train_lm(model, tr, vocab, val_seqs=va, steps=n_steps,
                          batch=BATCH, lr=LR, log_every=max(n_steps // 6, 100))
        ev_own = U.evaluate_lm(model, va, vocab)
        # every arm is ALSO scored on the full mixed validation set, so the
        # numbers are comparable across arms
        va_full = U.build_sequences(vocab, va_cap, va_c, ("t2i", "i2t"),
                                    seed=2, length=CTX)
        ev_full = U.evaluate_lm(model, va_full, vocab)
        results[name] = {"steps": n_steps, "history": hist,
                         "eval_own_format": ev_own, "eval_mixed": ev_full,
                         "params": sum(p.numel() for p in model.parameters()),
                         "train_tokens": int((tr != U.PAD).sum())}
        print(f"    own-format val: text {ev_own['text']:.3f}  image {ev_own['image']:.3f}")
        torch.save(model.state_dict(), CKPT / f"{name}.pt")
    _save("train.json", results)


def _strip(vocab, caps, codes, orders, keep, seed):
    """Build sequences containing only the kept modalities.

    Dropping a modality has to keep the *format* legal, so an image-only row is
    still <bos> <boi> codes <eoi> <eos> -- only the caption words are gone.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c, ic in zip(caps, codes):
        o = orders[rng.integers(len(orders))]
        t = vocab.text_ids(c)[:U.TEXT_CTX] if "text" in keep else []
        im = vocab.image_ids(ic) if "image" in keep else []
        if not im:
            rows.append([U.BOS] + t + [U.EOS])
        elif o == "t2i":
            rows.append([U.BOS] + t + [U.BOI] + im + [U.EOI, U.EOS])
        else:
            rows.append([U.BOS, U.BOI] + im + [U.EOI] + t + [U.EOS])
    return U.pad_batch(rows, CTX)


# ---------------------------------------------------------------------------
def _load(name, vocab):
    m = _make_model(vocab)
    m.load_state_dict(torch.load(CKPT / f"{name}.pt", map_location="cpu"))
    m.eval()
    return m


def stage_gen(n_gen=64):
    """Both directions, both graded against an independent yardstick."""
    tr_c, tr_cap, va_c, va_cap, tr_i, tr_a, va_i, va_a = build_data()
    vocab = get_vocab(tr_cap, va_cap)
    tok = VQ.load_tokenizer()
    uni = _load("unified", vocab)
    img_only = _load("image_only", vocab)
    txt_only = _load("text_only", vocab)
    probe = U.AttrProbe()
    probe.load_state_dict(torch.load(CKPT / "probe.pt", map_location="cpu"))
    probe.eval()

    n_img = int(va_c.shape[1])
    out = {}

    # ---- direction 1: text -> image ---------------------------------------
    # Paired prompts: the same sentence with ONE attribute word flipped. If the
    # model is conditioning on the words at all, the referee's score for that
    # attribute must move between the two halves of the pair.
    tests = [
        ("Male", "a young man", "a young woman"),
        ("Smiling", "a smiling young woman", "a young woman"),
        ("Eyeglasses", "a young man with glasses", "a young man"),
        ("Blond_Hair", "a young woman with blond hair", "a young woman with black hair"),
    ]
    t2i, panels, panel_titles = [], [], []
    for attr, pos, neg in tests:
        row = {"attribute": attr, "positive_prompt": pos, "negative_prompt": neg}
        for tag, prompt in (("positive", pos), ("negative", neg)):
            ids = [[U.BOS] + vocab.text_ids(prompt) + [U.BOI]] * n_gen
            gen = U.sample(uni, ids, n_img, temperature=1.0, top_k=100, seed=7,
                           allow=(vocab.image_base, vocab.image_base + vocab.n_image))
            codes = vocab.decode_image(gen[:, -n_img:].reshape(-1).tolist())
            codes = torch.from_numpy(codes.reshape(n_gen, tok.grid, tok.grid))
            imgs = tok.decode_indices(codes)
            with torch.no_grad():
                p = torch.sigmoid(probe(imgs))[:, U.GRADED_ATTRS.index(attr)]
            row[tag] = float(p.mean())
            if tag == "positive":
                panels.append(VQ.to_uint8(imgs[:8]))
                panel_titles.append(pos)
            else:
                panels.append(VQ.to_uint8(imgs[:8]))
                panel_titles.append(neg)
        row["swing"] = row["positive"] - row["negative"]
        # the same referee run on REAL faces gives the ceiling for that swing
        ix = VQ.ATTRS.index(attr)
        real_pos = va_i[va_a[:, ix] > 0][:128]
        real_neg = va_i[va_a[:, ix] < 0][:128]
        with torch.no_grad():
            rp = torch.sigmoid(probe(VQ.to_tensor(real_pos)))[:, U.GRADED_ATTRS.index(attr)].mean()
            rn = torch.sigmoid(probe(VQ.to_tensor(real_neg)))[:, U.GRADED_ATTRS.index(attr)].mean()
        row["real_swing"] = float(rp - rn)
        row["obedience"] = row["swing"] / row["real_swing"] if row["real_swing"] else 0.0
        print(f"  {attr:14s} pos {row['positive']:.3f}  neg {row['negative']:.3f}  "
              f"swing {row['swing']:+.3f}  (real faces: {row['real_swing']:+.3f})")
        t2i.append(row)
    out["text_to_image"] = t2i
    _panels(panels, panel_titles, OUT / "t2i.png")

    # unconditional samples from the image-only control, for comparison
    ids = [[U.BOS, U.BOI]] * 8
    gen = U.sample(img_only, ids, n_img, temperature=1.0, top_k=100, seed=3,
                   allow=(vocab.image_base, vocab.image_base + vocab.n_image))
    codes = torch.from_numpy(vocab.decode_image(gen[:, -n_img:].reshape(-1).tolist())
                             .reshape(8, tok.grid, tok.grid))
    _panels([VQ.to_uint8(tok.decode_indices(codes)),
             VQ.to_uint8(tok.decode_indices(torch.from_numpy(
                 va_c[:8].astype(np.int64)).view(8, tok.grid, tok.grid)))],
            ["image-only control, no prompt", "real faces through the tokenizer"],
            OUT / "uncond.png")

    # ---- direction 2: image -> text ---------------------------------------
    # A two-way forced choice: the true caption against the same caption with
    # ONE attribute flipped. Both models answer the same question, so the gap
    # between them is exactly the information the image supplied.
    i2t = {}
    for name, model, sees_image in (("unified", uni, True),
                                    ("text_only_prior", txt_only, False)):
        acc = _caption_choice(model, vocab, va_c, va_a, sees_image)
        i2t[name] = acc
        print(f"  captioning [{name}]: mean 2-way accuracy {acc['mean']:.3f} "
              f"(chance 0.500)")
    out["image_to_text"] = i2t
    _save("gen.json", out)


CHOICE_ATTRS = ["Male", "Smiling", "Young", "Eyeglasses", "Blond_Hair",
                "Wearing_Hat"]


@torch.no_grad()
def _caption_logprob(model, vocab, prefixes, captions):
    """Total log-probability the model assigns to each caption, continuing from
    its prefix. Teacher forcing -- one forward pass per batch, not per word."""
    rows, spans = [], []
    for pre, cap in zip(prefixes, captions):
        t = vocab.text_ids(cap)[:U.TEXT_CTX] + [U.EOS]
        rows.append(pre + t)
        spans.append((len(pre), len(pre) + len(t)))
    ids = torch.from_numpy(U.pad_batch(rows, CTX))
    lp = F.log_softmax(model(ids[:, :-1]), dim=-1)
    tgt = ids[:, 1:]
    got = lp.gather(-1, tgt[..., None])[..., 0]                 # (B, T-1)
    pos = torch.arange(tgt.shape[1])[None]
    lo = torch.tensor([s - 1 for s, _ in spans])[:, None]
    hi = torch.tensor([e - 1 for _, e in spans])[:, None]
    mask = (pos >= lo) & (pos < hi)
    return (got * mask).sum(1)


def _caption_choice(model, vocab, codes, attrs, sees_image, n=400, batch=64):
    """Accuracy at telling the true caption from a one-attribute-flipped one.

    Why a forced choice instead of free generation: a blind model can still
    write a fluent caption by guessing the most common one, so free generation
    would flatter it. Making both models rank two nearly identical sentences
    isolates the single word that the image is supposed to decide.
    """
    n = min(n, len(codes))
    per = {}
    for attr in CHOICE_ATTRS:
        ix = VQ.ATTRS.index(attr)
        pre, true_c, false_c = [], [], []
        for i in range(n):
            a = attrs[i]
            ct = VQ.attr_caption(a)
            b = a.copy(); b[ix] = -b[ix]
            cf = VQ.attr_caption(b)
            if ct == cf:              # flipping this attribute changed nothing
                continue
            pre.append([U.BOS, U.BOI] + vocab.image_ids(codes[i]) + [U.EOI]
                       if sees_image else [U.BOS])
            true_c.append(ct); false_c.append(cf)
        if len(pre) < 20:
            continue
        ok = []
        for i in range(0, len(pre), batch):
            s = slice(i, i + batch)
            lt = _caption_logprob(model, vocab, pre[s], true_c[s])
            lf = _caption_logprob(model, vocab, pre[s], false_c[s])
            ok.append((lt > lf).float())
        per[attr] = {"accuracy": float(torch.cat(ok).mean()), "n": len(pre)}
    return {"per_attribute": per,
            "mean": float(np.mean([v["accuracy"] for v in per.values()]))
            if per else float("nan")}


def _panels(rows, titles, path, per_row=8):
    import matplotlib.pyplot as plt
    n = min(per_row, min(len(r) for r in rows))
    fig, axes = plt.subplots(len(rows), n, figsize=(n * 1.15, len(rows) * 1.32), dpi=110)
    axes = np.atleast_2d(axes)
    fig.patch.set_facecolor(ps.SURFACE)
    for i, row in enumerate(rows):
        for j in range(n):
            axes[i, j].imshow(row[j])
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_color(ps.BASELINE)
        axes[i, 0].set_ylabel(titles[i].replace(" with ", "\nwith "),
                              color=ps.INK_SECONDARY, fontsize=7.5, rotation=0,
                              ha="right", va="center", labelpad=6)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
def stage_plot():
    tr = json.loads((OUT / "train.json").read_text())
    fig, ax = ps.new_axes(7.0, 4.2)
    for i, (name, colour) in enumerate((("unified", ps.SERIES[0]),
                                        ("image_only", ps.SERIES[1]),
                                        ("text_only", ps.SERIES[2]))):
        h = tr[name]["history"]
        for key, style in (("val_image", "-"), ("val_text", "--")):
            xs = [r["step"] for r in h if not np.isnan(r.get(key, np.nan))]
            ys = [r[key] for r in h if not np.isnan(r.get(key, np.nan))]
            if ys:
                ax.plot(xs, ys, style, color=colour, lw=2,
                        label=f"{name} · {key.replace('val_', '')}")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=ps.INK_SECONDARY, ncol=2)
    ps.finish(fig, ax, "One model, two modalities — and the two controls",
              "training step", "validation loss (nats/token)", OUT / "curves.png")

    gen = json.loads((OUT / "gen.json").read_text())
    fig, ax = ps.new_axes(6.6, 4.0)
    names = [r["attribute"] for r in gen["text_to_image"]]
    x = np.arange(len(names))
    ax.bar(x - 0.2, [r["swing"] for r in gen["text_to_image"]], 0.38,
           color=ps.SERIES[0], label="generated faces")
    ax.bar(x + 0.2, [r["real_swing"] for r in gen["text_to_image"]], 0.38,
           color=ps.BASELINE, label="real faces (ceiling)")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.axhline(0, color=ps.INK_MUTED, lw=1)
    ax.legend(frameon=False, fontsize=9, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Does flipping one word change the picture?",
              "", "referee's score, positive prompt − negative prompt",
              OUT / "obedience.png")


STAGES = {"data": stage_data, "probe": stage_probe, "train": stage_train,
          "gen": stage_gen, "plot": stage_plot}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    for nm in (list(STAGES) if a.stage == "all" else [a.stage]):
        print(f"\n=== {nm} " + "=" * (60 - len(nm)))
        if a.steps and nm in ("train", "probe"):
            STAGES[nm](steps=a.steps)
        else:
            STAGES[nm]()
