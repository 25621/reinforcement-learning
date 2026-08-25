"""Project 15 -- four ways to fuse an image and a question, on one VQA task.

Stage 1 trains a small ViT once on a pretext task and freezes it, exactly as a
real VLM borrows a frozen CLIP. Stage 2 runs four fusion modules on the SAME
cached patch tokens, so every difference between them is the fusion's doing.

  concat        one pooled image vector + one question vector -> MLP
  image-token   the same single vector, as one token in the question sequence
  projector     all 16 patch tokens, projected into the question sequence
  cross-attn    all 16 patch tokens, reached through cross-attention layers

`image-token` is the control that makes the experiment conclusive: it has the
transformer fusion machinery of the last two but still sees only one summary
vector. If it scores like `concat`, then what matters is the granularity of
access, not the fusion mechanism.

Usage:
    python3 run.py --stage vision     # pretrain + freeze the encoder (~1 min)
    python3 run.py --stage train      # the four fusion modules (~10 min)
    python3 run.py --stage plot
    python3 run.py --stage examples
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import vqa_lib as V

torch.set_num_threads(12)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CACHE = HERE / "data"
D = 128
STEPS = 2500
BATCH = 128
TRAIN_SCENES = 8000
TEST_SCENES = 1000

VARIANTS = ["concat", "image-token", "projector", "cross-attn"]


# ---------------------------------------------------------------------------
# the four fusion heads
# ---------------------------------------------------------------------------
class ConcatFusion(nn.Module):
    """Late fusion. Glue the two vectors end to end, then classify.

    The image side is compressed to a single vector by a trainable attention
    pool BEFORE the question is known. That is the whole limitation: whatever gets thrown away in that squeeze is gone, and the
    squeeze cannot depend on what is being asked.
    """

    def __init__(self, d=D):
        super().__init__()
        self.pool = V.AttentionPool(d)
        self.text = V.TextTower(d)
        self.head = nn.Sequential(
            nn.LayerNorm(2 * d), nn.Linear(2 * d, 4 * d), nn.GELU(),
            nn.Linear(4 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, len(V.ANSWERS)))

    def forward(self, feats, tok):
        img = self.pool(feats)
        txt = self.text(tok)[1]
        return self.head(torch.cat([img, txt], dim=-1))


class SequenceFusion(nn.Module):
    """One shared implementation for `image-token` and `projector`.

    Both put image information *into the question sequence* and then run plain
    self-attention over the joint sequence -- early fusion, the LLaVA pattern.
    They differ only in how many image tokens go in: 1 pooled vector, or all 16
    patch tokens.

    Why a projector at all, when the frozen encoder already outputs d=128
    vectors that would fit the sequence: the projector is a *trainable
    translator*. The vision encoder and the text encoder were never told to use
    the same coordinate system, so "patch dimension 7" and "word dimension 7"
    mean unrelated things. The linear layer learns the change of basis. In a
    real VLM the two spaces also have different widths (1024 vision vs 4096
    language), so the projector resizes as well as rotates. It stays trainable
    while the encoder is frozen, which is precisely how a frozen encoder can be
    made useful to a model it was never trained with.
    """

    def __init__(self, mode, d=D, layers=2, heads=4):
        super().__init__()
        self.mode = mode
        self.pool = V.AttentionPool(d) if mode == "image-token" else None
        self.text = V.TextTower(d)
        self.projector = nn.Linear(d, d)
        self.type_emb = nn.Parameter(torch.randn(2, d) * 0.02)   # image vs text
        self.answer_token = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers += [V.SelfAttention(d, heads), V.MLP(d)]
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(V.ANSWERS))

    def forward(self, feats, tok):
        img = self.pool(feats)[:, None] if self.pool is not None else feats
        img = self.projector(img) + self.type_emb[0]
        txt = self.text(tok)[0] + self.type_emb[1]
        ans = self.answer_token.expand(len(feats), -1, -1)
        x = torch.cat([img, txt, ans], dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x[:, -1]))


class CrossAttnFusion(nn.Module):
    """Flamingo-style. The question keeps its own stream; extra cross-attention
    layers let it *query* the patch tokens without them joining the sequence.

    The compute profile differs from the projector variant: self-attention over
    the joint sequence costs (16+12)^2 = 784 token pairs, cross-attention costs
    12x16 = 192. With long text and many images that gap is why Flamingo chose
    this shape.
    """

    def __init__(self, d=D, layers=2, heads=4):
        super().__init__()
        self.text = V.TextTower(d)
        self.projector = nn.Linear(d, d)
        self.answer_token = nn.Parameter(torch.zeros(1, 1, d))
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers += [V.SelfAttention(d, heads),
                            V.CrossAttention(d, heads, gate=True), V.MLP(d)]
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, len(V.ANSWERS))

    def forward(self, feats, tok):
        ctx = self.projector(feats)
        txt = self.text(tok)[0]
        x = torch.cat([txt, self.answer_token.expand(len(feats), -1, -1)], dim=1)
        for layer in self.layers:
            x = layer(x, ctx) if isinstance(layer, V.CrossAttention) else layer(x)
        return self.head(self.norm(x[:, -1]))


def build(variant):
    if variant == "concat":
        return ConcatFusion()
    if variant in ("image-token", "projector"):
        return SequenceFusion(variant)
    return CrossAttnFusion()


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def _paths(pretext):
    """`what-where` is the real encoder; `what-only` is the ablation that never
    learns to keep position, kept because its failure is instructive."""
    suffix = "" if pretext == "what-where" else "_whatonly"
    return CACHE if not suffix else CACHE.parent / f"data{suffix}", suffix


def stage_vision(args):
    """Pretrain the image encoder once, freeze it, cache its patch tokens."""
    cache, suffix = _paths(args.pretext)
    cache.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    print("rendering scenes ...", flush=True)
    tr_img, tr_sh, tr_col, tr_cell, tr_yx = V.make_scenes(TRAIN_SCENES, seed=0)
    te_img, te_sh, te_col, te_cell, te_yx = V.make_scenes(TEST_SCENES, seed=999)

    t0 = time.time()
    model, acc = V.pretrain_vision(tr_img, tr_col, tr_cell, tr_sh,
                                   teach_position=args.pretext == "what-where")
    print(f"  pretext accuracy on held-out scenes: colour {acc['color']:.3f}  "
          f"shape {acc['shape']:.3f}  position {acc['position']}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    for p in model.parameters():
        p.requires_grad_(False)
    np.save(cache / "train_feats.npy", V.cache_features(model.tower, tr_img))
    np.save(cache / "test_feats.npy", V.cache_features(model.tower, te_img))
    for name, (sh, col, yx) in (("train", (tr_sh, tr_col, tr_yx)),
                                ("test", (te_sh, te_col, te_yx))):
        tok, ans = V.make_questions(sh, col, yx)
        np.savez(cache / f"{name}_qa.npz", tokens=tok, answers=ans)
    torch.save(model.state_dict(), cache / "vision.pt")

    chance = {}
    tok, ans = V.make_questions(te_sh, te_col, te_yx)
    for q in range(V.N_QUESTIONS):
        chance[q] = float(np.bincount(ans[:, q], minlength=len(V.ANSWERS)).max()
                          / len(ans))
    (OUT / f"vision{suffix}.json").write_text(json.dumps(
        dict(pretext=args.pretext, pretext_color_acc=acc["color"],
             pretext_shape_acc=acc["shape"], pretext_position_acc=acc["position"],
             encoder_params=V.count_params(model.tower),
             majority_class=chance), indent=1))
    print("  cached", cache / "train_feats.npy")


def load_data(pretext="what-where"):
    cache, _ = _paths(pretext)
    tr = np.load(cache / "train_qa.npz")
    te = np.load(cache / "test_qa.npz")
    return (V.FeatureVQA(np.load(cache / "train_feats.npy"), tr["tokens"],
                         tr["answers"]),
            V.FeatureVQA(np.load(cache / "test_feats.npy"), te["tokens"],
                         te["answers"]))


def stage_train(args):
    OUT.mkdir(exist_ok=True)
    _, suffix = _paths(args.pretext)
    train_data, test_data = load_data(args.pretext)
    print(f"{len(train_data)} train QA pairs, {len(test_data)} test QA pairs",
          flush=True)

    rows, curves = [], {}
    for variant in (args.only or VARIANTS):
        print(f"\n=== {variant}", flush=True)
        torch.manual_seed(0)
        model = build(variant)
        n_params = V.count_params(model)
        t0 = time.time()
        hist = V.train(model, train_data, STEPS, batch=BATCH, lr=1e-3, seed=0)
        secs = time.time() - t0
        acc, per_kind = V.evaluate(model, test_data)
        curves[variant] = np.array(hist)
        rows.append(dict(variant=variant, fusion_params=n_params, acc=acc,
                         ms_per_step=1000 * secs / STEPS, seconds=secs,
                         **{f"kind{q}": per_kind[q] for q in per_kind}))
        print(f"  acc {acc:.3f}  fusion params {n_params/1e6:.2f}M  "
              f"{1000*secs/STEPS:.0f} ms/step", flush=True)
        print("  per question type: " +
              "  ".join(f"{q}:{V.QUESTION_KIND[q][:4]}={per_kind[q]:.2f}"
                        for q in per_kind), flush=True)

    path = OUT / f"fusion{suffix}.csv"
    old = {r["variant"]: r for r in csv.DictReader(open(path))} if path.exists() else {}
    old.update({r["variant"]: r for r in rows})
    ordered = [old[v] for v in VARIANTS if v in old]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(ordered)
    curve_path = OUT / f"curves{suffix}.npz"
    old_curves = dict(np.load(curve_path)) if curve_path.exists() else {}
    old_curves.update(curves)
    np.savez(curve_path, **old_curves)
    print("\nwrote", path)


def stage_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, suffix = _paths(args.pretext)
    rows = list(csv.DictReader(open(OUT / f"fusion{suffix}.csv")))
    meta = json.loads((OUT / f"vision{suffix}.json").read_text())
    chance = meta["majority_class"]
    curves = np.load(OUT / f"curves{suffix}.npz")
    colors = dict(zip(VARIANTS, ["#888888", "#c44e52", "#4c72b0", "#55a868"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    for r in rows:
        h = curves[r["variant"]]
        ax.plot(np.convolve(h, np.ones(50) / 50, mode="valid"),
                label=r["variant"], color=colors[r["variant"]])
    ax.set_xlabel("step")
    ax.set_ylabel("training loss (nats)")
    ax.set_title("Training loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    x = np.arange(len(rows))
    ax.bar(x, [float(r["acc"]) for r in rows],
           color=[colors[r["variant"]] for r in rows])
    for i, r in enumerate(rows):
        ax.text(i, float(r["acc"]) + 0.012, f"{float(r['acc']):.3f}", ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([r["variant"] for r in rows], rotation=20, fontsize=9)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.08)
    ax.set_title("Overall accuracy (16-way)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[2]
    width = 0.2
    kinds = np.arange(V.N_QUESTIONS)
    for i, r in enumerate(rows):
        ax.bar(kinds + (i - 1.5) * width,
               [float(r[f"kind{q}"]) for q in range(V.N_QUESTIONS)], width,
               label=r["variant"], color=colors[r["variant"]])
    ax.plot(kinds, [chance[str(q)] for q in range(V.N_QUESTIONS)], "k_",
            markersize=22, label="majority-class")
    ax.set_xticks(kinds)
    ax.set_xticklabels([f"Q{q}\n{V.QUESTION_KIND[q]}" for q in range(V.N_QUESTIONS)],
                       fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy by question type")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(OUT / f"fusion{suffix}.png", dpi=130)

    # the ablation: same four models on an encoder that never kept position
    if suffix == "" and (OUT / "fusion_whatonly.csv").exists():
        abl = {r["variant"]: r for r in
               csv.DictReader(open(OUT / "fusion_whatonly.csv"))}
        fig3, ax3 = plt.subplots(figsize=(7.4, 4.4))
        x = np.arange(V.N_QUESTIONS)
        for r in rows:
            if r["variant"] not in abl:
                continue
            ax3.plot(x, [float(r[f"kind{q}"]) for q in x], "o-",
                     color=colors[r["variant"]], label=f"{r['variant']} (what+where)")
            ax3.plot(x, [float(abl[r["variant"]][f"kind{q}"]) for q in x], "s--",
                     color=colors[r["variant"]], alpha=0.5,
                     label=f"{r['variant']} (what only)")
        ax3.plot(x, [chance[str(q)] for q in x], "k_", markersize=22,
                 label="majority-class")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"Q{q}\n{V.QUESTION_KIND[q]}" for q in x], fontsize=8)
        ax3.set_ylabel("test accuracy")
        ax3.set_title("What the frozen encoder kept decides what fusion can do")
        ax3.legend(fontsize=6, ncol=2)
        ax3.grid(alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(OUT / "pretext_ablation.png", dpi=130)

    fig2, ax = plt.subplots(figsize=(5.8, 4.3))
    for r in rows:
        ax.scatter(float(r["fusion_params"]) / 1e6, float(r["acc"]), s=110,
                   color=colors[r["variant"]])
        ax.annotate(r["variant"], (float(r["fusion_params"]) / 1e6, float(r["acc"])),
                    textcoords="offset points", xytext=(7, -3), fontsize=8)
    ax.set_xlabel("fusion-module parameters (millions)")
    ax.set_ylabel("test accuracy")
    ax.set_title("What the extra parameters buy")
    ax.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(OUT / f"params_vs_acc{suffix}.png", dpi=130)
    print("wrote", OUT / f"fusion{suffix}.png")


def stage_examples(_args):
    """A picture of the world plus its five questions, for the README."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT.mkdir(exist_ok=True)
    imgs, shapes, colors, cells, yx = V.make_scenes(3, seed=7)
    tok, ans = V.make_questions(shapes, colors, yx)
    inv = {v: k for k, v in V.VOCAB.items()}
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.6))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i])
        ax.axis("off")
        lines = []
        for q in range(V.N_QUESTIONS):
            words = " ".join(inv[t] for t in tok[i, q] if t != V.PAD)
            lines.append(f"Q{q} ({V.QUESTION_KIND[q]}): {words}?\n      -> "
                         f"{V.ANSWERS[ans[i, q]]}")
        ax.set_title("\n".join(lines), fontsize=7, loc="left", family="monospace")
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=130)
    print("wrote", OUT / "examples.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage", default="train",
                   choices=["vision", "train", "plot", "examples"])
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--pretext", default="what-where",
                   choices=["what-where", "what-only"],
                   help="what the frozen encoder was taught to keep")
    a = p.parse_args()
    {"vision": stage_vision, "train": stage_train, "plot": stage_plot,
     "examples": stage_examples}[a.stage](a)
