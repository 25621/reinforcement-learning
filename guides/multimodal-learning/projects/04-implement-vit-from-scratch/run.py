"""Build a ViT from scratch, prove each piece is what it claims to be, then train it.

Stages:
  verify   three numerical proofs about the architecture (seconds, no training)
  train    three CIFAR-10 runs at an equal budget: cls+pos, cls-no-pos, mean+pos
  figures  redraw every chart from the saved results

    python3 run.py --stage all
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "01-modality-survey"))
import plot_style as ps  # noqa: E402

from vit import (CIFAR_CLASSES, CIFAR_MEAN, CIFAR_STD, PatchEmbed, ViT,  # noqa: E402
                 cifar_data, evaluate, patchify_unfold, to_tensor, train_vit)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)

# One budget shared by all three configurations, so the comparison is fair.
STEPS = 700
CONFIGS = {
    "cls+pos":   dict(pool="cls", use_pos=True),
    "cls-nopos": dict(pool="cls", use_pos=False),
    "mean+pos":  dict(pool="mean", use_pos=True),
}
ARCH = dict(img_size=32, patch=4, dim=128, depth=4, heads=4)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------
def stage_verify():
    """Three claims the README makes, checked numerically instead of asserted."""
    torch.manual_seed(0)
    results = {}
    x = torch.randn(4, 3, 32, 32)

    # (1) The strided convolution really is "cut into squares, then project".
    pe = PatchEmbed(32, 4, 3, 128)
    with torch.no_grad():
        a = pe(x)
        b = pe.as_linear(x)
    d1 = (a - b).abs().max().item()
    results["conv_vs_unfold_maxdiff"] = d1
    print(f"(1) strided conv vs unfold+matmul : max |diff| = {d1:.2e}")

    # (2) The hand-written softmax attention matches the fused kernel.
    from vit import Attention
    att = Attention(128, heads=4, fast=True)
    t = torch.randn(4, 65, 128)
    with torch.no_grad():
        fast = att(t)
        att.fast = False
        slow = att(t)
    d2 = (fast - slow).abs().max().item()
    results["sdpa_vs_manual_maxdiff"] = d2
    print(f"(2) manual softmax vs F.sdpa     : max |diff| = {d2:.2e}")

    # (3) Without positional embeddings the model is *exactly* order-blind.
    #     Shuffle the patches and the prediction does not move at all.
    perm = torch.randperm(64)
    for use_pos in (False, True):
        torch.manual_seed(1)
        m = ViT(**ARCH, use_pos=use_pos)
        m.eval()
        with torch.no_grad():
            plain = m(x)
            # shuffle patches by shuffling 4x4-pixel blocks of the input
            blocks = patchify_unfold(x, 4)[:, perm]                 # (B, 64, 48)
            shuffled = F.fold(blocks.transpose(1, 2), output_size=(32, 32),
                              kernel_size=4, stride=4)
            after = m(shuffled)
        d = (plain - after).abs().max().item()
        key = "shuffle_maxdiff_with_pos" if use_pos else "shuffle_maxdiff_no_pos"
        results[key] = d
        tag = "with pos-embed" if use_pos else "no pos-embed "
        print(f"(3) patch shuffle, {tag}   : max |diff| = {d:.2e}")

    (OUT / "verify.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT / 'verify.json'}")
    return results


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def stage_train(only=None):
    data = cifar_data(DATA)
    hists = _load_hists()
    for name, cfg in CONFIGS.items():
        if only and name != only:
            continue
        torch.manual_seed(0)          # identical init across configs
        model = ViT(**ARCH, **cfg)
        print(f"\n=== {name}: {model.n_params() / 1e6:.2f}M params, "
              f"{model.patch_embed.n_patches} patches")
        h = train_vit(model, data, steps=STEPS, label=name)
        # Final score on the full 10k test set, not the 2k eval subset.
        xte, yte = data["test"]
        h["final_acc"] = evaluate(model, to_tensor(xte), torch.from_numpy(yte))
        h["params"] = model.n_params()
        print(f"  [{name}] full test set: {h['final_acc']:.4f}")
        hists[name] = h
        (OUT / "training.json").write_text(json.dumps(hists, indent=2))
        tag = name.replace("+", "_").replace("-", "_")
        torch.save(model.state_dict(), OUT / f"vit_{tag}.pt")
    return hists


def _load_hists():
    f = OUT / "training.json"
    return json.loads(f.read_text()) if f.exists() else {}


def shuffle_patches(x, patch, perm):
    """Scramble the image's patches into a fixed random order, pixels intact."""
    blocks = patchify_unfold(x, patch)[:, perm]
    return F.fold(blocks.transpose(1, 2), output_size=(x.shape[-1],) * 2,
                  kernel_size=patch, stride=patch)


def stage_shuffle():
    """The payoff for the pos-embed ablation, measured on *trained* models.

    Scramble every test image's 64 patches into one fixed random order. The
    model that never had positional embeddings cannot tell -- its accuracy is
    unchanged to the last digit, because the scramble is a no-op for it.
    """
    data = cifar_data(DATA)
    xte, yte = data["test"]
    x, y = to_tensor(xte), torch.from_numpy(yte)
    torch.manual_seed(7)
    perm = torch.randperm(64)
    x_shuf = shuffle_patches(x, 4, perm)

    rows = {}
    for name, cfg in CONFIGS.items():
        ckpt = OUT / f"vit_{name.replace('+', '_').replace('-', '_')}.pt"
        if not ckpt.exists():
            continue
        m = ViT(**ARCH, **cfg)
        m.load_state_dict(torch.load(ckpt))
        m.eval()
        rows[name] = {"clean": evaluate(m, x, y),
                      "shuffled": evaluate(m, x_shuf, y)}
        print(f"  {name:10s} clean {rows[name]['clean']:.4f}  "
              f"shuffled {rows[name]['shuffled']:.4f}")
    (OUT / "shuffle.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {OUT / 'shuffle.json'}")
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_curves(hists):
    fig, ax = ps.new_axes(7.4, 4.4)
    for i, name in enumerate(CONFIGS):
        if name not in hists:
            continue
        h = hists[name]
        ax.plot(h["step"], h["acc"], color=ps.SERIES[i], lw=2,
                marker="o", ms=3.5,
                label=f"{name}  →  {h['final_acc']:.3f}")
    ax.axhline(0.1, color=ps.BASELINE, ls="--", lw=1.2)
    ax.text(STEPS * 0.02, 0.115, "chance (10 classes)",
            color=ps.INK_MUTED, fontsize=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right",
              title="final full-test accuracy", title_fontsize=8)
    ps.finish(fig, ax, "What the CLS token and the positional embedding are worth",
              "training step", "test accuracy", OUT / "training_curves.png")


def fig_patches():
    """Show the picture the transformer actually receives: a list of squares."""
    data = cifar_data(DATA)
    xte, yte = data["test"]
    img = xte[7]
    fig, axes = ps.plt.subplots(1, 3, figsize=(8.4, 3.1), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    axes[0].imshow(img)
    axes[0].set_title(f"input 32x32\n({CIFAR_CLASSES[yte[7]]})",
                      fontsize=9, color=ps.INK)
    for ax, patch in zip(axes[1:], (4, 8)):
        g = 32 // patch
        axes_img = img.copy()
        ax.imshow(axes_img)
        for k in range(1, g):
            ax.axhline(k * patch - 0.5, color="w", lw=1.1)
            ax.axvline(k * patch - 0.5, color="w", lw=1.1)
        ax.set_title(f"patch {patch} → {g * g} tokens\n"
                     f"each is {patch}x{patch}x3 = {patch * patch * 3} numbers",
                     fontsize=9, color=ps.INK)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "patchification.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'patchification.png'}")


def fig_attention():
    """Where does the CLS token look? Average attention over heads, last block."""
    ckpt = OUT / "vit_cls_pos.pt"
    if not ckpt.exists():
        print("skip attention figure (train first)")
        return
    model = ViT(**ARCH, **CONFIGS["cls+pos"])
    model.load_state_dict(torch.load(ckpt))
    model.eval()
    data = cifar_data(DATA)
    xte, yte = data["test"]
    pick = [3, 6, 25, 12]
    x = to_tensor(xte[pick])
    with torch.no_grad():
        logits, attn = model(x, return_attn=True)
    # attn: (B, heads, N, N). Row 0 is the CLS token's query; drop its self-column.
    cls_attn = attn[:, :, 0, 1:].mean(1).reshape(-1, 8, 8).numpy()
    pred = logits.argmax(1).numpy()

    fig, axes = ps.plt.subplots(2, 4, figsize=(8.6, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for j in range(4):
        axes[0, j].imshow(xte[pick[j]])
        axes[0, j].set_title(f"true {CIFAR_CLASSES[yte[pick[j]]]}\n"
                             f"pred {CIFAR_CLASSES[pred[j]]}",
                             fontsize=8.5, color=ps.INK)
        axes[1, j].imshow(cls_attn[j], cmap="magma")
        axes[1, j].set_title("CLS attention (8x8 patches)", fontsize=8,
                             color=ps.INK_SECONDARY)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(OUT / "cls_attention.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'cls_attention.png'}")


def fig_pos_similarity():
    """Did the positional embedding learn that the image is a 2D grid?"""
    ckpt = OUT / "vit_cls_pos.pt"
    if not ckpt.exists():
        print("skip pos-embed figure (train first)")
        return
    sd = torch.load(ckpt)
    pos = sd["pos_embed"][0, 1:]                       # (64, dim), drop CLS slot
    pos = F.normalize(pos, dim=-1)
    sim = (pos @ pos.t()).numpy()                      # (64, 64)
    grid = sim.reshape(8, 8, 8, 8)
    tiled = np.concatenate([np.concatenate([grid[r, c] for c in range(8)], axis=1)
                            for r in range(8)], axis=0)
    fig, ax = ps.plt.subplots(figsize=(5.6, 5.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    im = ax.imshow(tiled, cmap="viridis")
    for k in range(1, 8):
        ax.axhline(k * 8 - 0.5, color=ps.SURFACE, lw=0.8)
        ax.axvline(k * 8 - 0.5, color=ps.SURFACE, lw=0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Each 8x8 tile: how one patch position's embedding\n"
                 "resembles all 64 positions (cosine)",
                 fontsize=10, color=ps.INK, loc="left", pad=10)
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    fig.savefig(OUT / "pos_embed_similarity.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'pos_embed_similarity.png'}")

    # Quantify it: is a position more similar to its 4 grid neighbours than to
    # a random other position?
    near, far = [], []
    for r in range(8):
        for c in range(8):
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < 8 and 0 <= cc < 8:
                    near.append(sim[r * 8 + c, rr * 8 + cc])
    off = ~np.eye(64, dtype=bool)
    far = sim[off]
    stats = {"neighbour_mean_cos": float(np.mean(near)),
             "all_other_mean_cos": float(np.mean(far))}
    (OUT / "pos_embed_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"pos-embed: neighbours {stats['neighbour_mean_cos']:.3f} vs "
          f"all others {stats['all_other_mean_cos']:.3f}")


def stage_figures():
    hists = _load_hists()
    fig_patches()
    if hists:
        fig_curves(hists)
    fig_attention()
    fig_pos_similarity()


# ---------------------------------------------------------------------------
def main():
    torch.set_num_threads(12)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "verify", "train", "shuffle", "figures"])
    ap.add_argument("--only", default=None, help="train just one config")
    a = ap.parse_args()
    if a.stage in ("all", "verify"):
        stage_verify()
    if a.stage in ("all", "train"):
        stage_train(a.only)
    if a.stage in ("all", "shuffle"):
        stage_shuffle()
    if a.stage in ("all", "figures"):
        stage_figures()


if __name__ == "__main__":
    main()
