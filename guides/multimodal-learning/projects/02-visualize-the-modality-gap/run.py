"""Project 02 — See the modality gap, then try (and fail) to close it.

Stages
    data      download 1,000 COCO (image, caption) pairs  (~40 s)
    encode    run the frozen CLIP over all 2,000 items    (~40 s)
    cone      encode a subset with a RANDOMLY INITIALISED CLIP  (~25 s)
    close     try five ways of pushing the two clouds together, score each (~2 s)
    figures   draw everything                              (~5 s)
    all       every stage in order                         (~2 min)

The story the stages tell, in order:

  1. CLIP is trained to put an image and its caption in the same place. It does
     not. The two modalities land in two separate blobs.
  2. The separation is not a rounding error: it is the single largest direction
     of variation among all 2,000 vectors.
  3. It is not caused by training either -- an untrained CLIP has a *bigger*
     gap. Training shrinks it and then stops.
  4. And "fixing" it makes retrieval worse. Every intervention that raises the
     pretty cosine number lowers the accuracy that actually matters.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

import clip_lib as L

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
sys.path.insert(0, str(HERE.parent / "01-modality-survey"))

N = 1000
N_CONE = 300          # the random-init run only needs enough to see the cone


# ---------------------------------------------------------------------------
def stage_data(args):
    t = time.time()
    L.fetch_coco(DATA, n=N)
    paths, caps, allc = L.load_pairs(DATA, N)
    print(f"{len(paths)} images, {len(caps)} captions in {time.time() - t:.0f}s")
    print(f"example caption: {caps[0]!r}")


def stage_encode(args):
    t = time.time()
    img, txt, paths, caps, _ = L.cached_embeddings(DATA, N)
    print(f"encoded in {time.time() - t:.0f}s -> {img.shape}, {txt.shape}")

    I, T = L.l2_normalize(img), L.l2_normalize(txt)
    eye = np.eye(N, dtype=bool)
    stats = dict(
        matched=float((I * T).sum(1).mean()),
        random_it=float((I @ T.T)[~eye].mean()),
        img_img=float((I @ I.T)[~eye].mean()),
        txt_txt=float((T @ T.T)[~eye].mean()),
        gap_norm=float(np.linalg.norm(I.mean(0) - T.mean(0))),
    )
    _, comps, evr, _ = L.pca(np.concatenate([I, T]), 2)
    g = I.mean(0) - T.mean(0)
    stats["pc1_evr"] = float(evr[0])
    stats["pc2_evr"] = float(evr[1])
    stats["pc1_vs_gap_cos"] = float(abs(comps[0] @ (g / np.linalg.norm(g))))
    (OUT / "gap_stats.json").write_text(json.dumps(stats, indent=2))

    print("\n  cosine similarity, averaged over the set")
    print(f"    an image and ITS OWN caption   {stats['matched']:.3f}")
    print(f"    an image and a random caption  {stats['random_it']:.3f}")
    print(f"    two random images              {stats['img_img']:.3f}")
    print(f"    two random captions            {stats['txt_txt']:.3f}")
    print(f"\n  distance between the two cloud centres  {stats['gap_norm']:.3f}")
    print(f"  PC1 explains {stats['pc1_evr']:.1%} of all variance; "
          f"|cos(PC1, gap direction)| = {stats['pc1_vs_gap_cos']:.4f}")


def stage_cone(args):
    """Encode with an UNTRAINED CLIP: is the gap made by training, or born with it?"""
    import torch
    from transformers import CLIPConfig, CLIPModel, CLIPTokenizerFast

    cache = DATA / f"emb_randinit_{N_CONE}.npz"
    if not cache.exists():
        torch.manual_seed(0)
        cfg = CLIPConfig.from_pretrained(L.MODEL_ID)
        L._MODEL = CLIPModel(cfg).eval()          # same architecture, random weights
        L._TOK = CLIPTokenizerFast.from_pretrained(L.MODEL_ID)
        paths, caps, _ = L.load_pairs(DATA, N_CONE)
        t = time.time()
        img = L.encode_images(paths, verbose=False)
        txt = L.encode_texts(caps, verbose=False)
        np.savez(cache, img=img, txt=txt)
        print(f"random-init encode in {time.time() - t:.0f}s")
    z = np.load(cache)
    I, T = L.l2_normalize(z["img"]), L.l2_normalize(z["txt"])

    eye = np.eye(len(I), dtype=bool)
    _, comps, evr, _ = L.pca(np.concatenate([I, T]), 2)
    g = I.mean(0) - T.mean(0)
    stats = dict(
        matched=float((I * T).sum(1).mean()),
        img_img=float((I @ I.T)[~eye].mean()),
        txt_txt=float((T @ T.T)[~eye].mean()),
        gap_norm=float(np.linalg.norm(g)),
        pc1_evr=float(evr[0]),
        pc1_vs_gap_cos=float(abs(comps[0] @ (g / np.linalg.norm(g)))),
    )
    (OUT / "cone_stats.json").write_text(json.dumps(stats, indent=2))
    print("\n  UNTRAINED CLIP, same architecture")
    print(f"    two random images   {stats['img_img']:.3f}   <- a 'cone': "
          f"everything already points the same way")
    print(f"    two random captions {stats['txt_txt']:.3f}")
    print(f"    gap between centres {stats['gap_norm']:.3f}")
    print(f"    PC1 explains {stats['pc1_evr']:.1%}")


def stage_close(args):
    """Five ways to push the clouds together. Do any of them help retrieval?"""
    img, txt, paths, caps, _ = L.cached_embeddings(DATA, N)
    I, T = L.l2_normalize(img), L.l2_normalize(txt)
    g = I.mean(0) - T.mean(0)
    u = g / np.linalg.norm(g)

    def score(name, A, B, renorm=True):
        if renorm:
            A, B = L.l2_normalize(A), L.l2_normalize(B)
        S = A @ B.T
        r_i2t = L.recall_at_k(S)
        r_t2i = L.recall_at_k(S.T)
        row = dict(method=name, renormalized=int(renorm),
                   matched_score=float((A * B).sum(1).mean()),
                   gap_norm=float(np.linalg.norm(A.mean(0) - B.mean(0))),
                   i2t_r1=r_i2t[1], i2t_r5=r_i2t[5],
                   t2i_r1=r_t2i[1], t2i_r5=r_t2i[5])
        print(f"  {name:<32} score {row['matched_score']:.3f}  "
              f"gap {row['gap_norm']:.3f}"
              f"  i2t R@1 {row['i2t_r1']:.3f}  t2i R@1 {row['t2i_r1']:.3f}")
        return row

    print("\n  method                           matched score / gap / retrieval")
    rows = [score("as-is (no fix)", I, T)]
    # slide every caption vector along the gap direction, then re-normalize
    for a in (0.5, 1.0):
        rows.append(score(f"shift text {a:g}x gap", I, T + a * g))
    # the same shift WITHOUT re-normalizing: a pure translation. Watch what a pure
    # translation does to each direction of retrieval -- the two are not symmetric.
    rows.append(score("shift text 1x gap, no renorm", I, T + g, renorm=False))
    # subtract each modality's own mean ("centering"), the standard whitening move
    rows.append(score("center each modality", I - I.mean(0), T - T.mean(0)))
    # delete the gap direction outright: 1 of 512 dimensions zeroed
    rows.append(score("delete the gap axis",
                      I - (I @ u)[:, None] * u, T - (T @ u)[:, None] * u))

    with open(OUT / "closing_the_gap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / 'closing_the_gap.csv'}")

    # how much of each modality's own spread lives on the gap axis?
    within = dict(image=float(np.std(I @ u)), text=float(np.std(T @ u)),
                  between=float(np.linalg.norm(g)))
    (OUT / "gap_axis_spread.json").write_text(json.dumps(within, indent=2))
    print(f"  spread along the gap axis: images ±{within['image']:.3f}, "
          f"captions ±{within['text']:.3f}, centres {within['between']:.3f} apart")


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib.pyplot as plt
    import plot_style as ps

    img, txt, paths, caps, _ = L.cached_embeddings(DATA, N)
    I, T = L.l2_normalize(img), L.l2_normalize(txt)
    stats = json.loads((OUT / "gap_stats.json").read_text())

    # ---- 1. the PCA scatter -------------------------------------------------
    X = np.concatenate([I, T])
    sc, comps, evr, mean = L.pca(X, 2)
    fig, ax = ps.new_axes(7.2, 5.0)
    ax.scatter(sc[:N, 0], sc[:N, 1], s=9, alpha=0.55, edgecolor="none",
               color=ps.SERIES[0], label="1,000 images")
    ax.scatter(sc[N:, 0], sc[N:, 1], s=9, alpha=0.55, edgecolor="none",
               color=ps.SERIES[3], label="their 1,000 captions")
    # draw 30 matched pairs as thin lines: every line crosses the empty middle
    for i in range(0, N, N // 30):
        ax.plot([sc[i, 0], sc[N + i, 0]], [sc[i, 1], sc[N + i, 1]],
                color=ps.INK_MUTED, linewidth=0.5, alpha=0.5, zorder=0)
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    ps.finish(fig, ax,
              f"Matched pairs, joined by a line — and none of the lines are short\n"
              f"PC1 alone is {stats['pc1_evr']:.0%} of all variance and is "
              f"{stats['pc1_vs_gap_cos']:.2f}-aligned with the gap",
              f"principal component 1 ({evr[0]:.1%} of variance)",
              f"principal component 2 ({evr[1]:.1%})",
              OUT / "pca_gap.png")

    # ---- 2. the four similarity histograms ---------------------------------
    rng = np.random.default_rng(0)
    j = rng.permutation(N)
    j[j == np.arange(N)] = (j[j == np.arange(N)] + 1) % N     # never self-pair
    series = [
        ("an image and ITS OWN caption", (I * T).sum(1), ps.SERIES[1]),
        ("an image and a random caption", (I * T[j]).sum(1), ps.SERIES[2]),
        ("two random images", (I * I[j]).sum(1), ps.SERIES[0]),
        ("two random captions", (T * T[j]).sum(1), ps.SERIES[3]),
    ]
    fig, ax = ps.new_axes(7.6, 4.4)
    for label, vals, color in series:
        ax.hist(vals, bins=60, range=(-0.1, 1.0), histtype="step", linewidth=1.8,
                color=color, label=f"{label}  (mean {vals.mean():.2f})")
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ps.finish(fig, ax,
              "A photo is LESS similar to its own caption than to an unrelated photo",
              "cosine similarity", "number of pairs",
              OUT / "similarity_histograms.png")

    # ---- 3. the cone effect -------------------------------------------------
    cone = json.loads((OUT / "cone_stats.json").read_text())
    z = np.load(DATA / f"emb_randinit_{N_CONE}.npz")
    Ir, Tr = L.l2_normalize(z["img"]), L.l2_normalize(z["txt"])
    scr, _, evrr, _ = L.pca(np.concatenate([Ir, Tr]), 2)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)
    axes[0].scatter(scr[:N_CONE, 0], scr[:N_CONE, 1], s=9, alpha=0.6,
                    edgecolor="none", color=ps.SERIES[0], label="images")
    axes[0].scatter(scr[N_CONE:, 0], scr[N_CONE:, 1], s=9, alpha=0.6,
                    edgecolor="none", color=ps.SERIES[3], label="captions")
    axes[0].set_title(f"UNTRAINED CLIP\nPC1 = {evrr[0]:.0%} of variance",
                      color=ps.INK, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].scatter(sc[:N, 0], sc[:N, 1], s=9, alpha=0.5, edgecolor="none",
                    color=ps.SERIES[0])
    axes[1].scatter(sc[N:, 0], sc[N:, 1], s=9, alpha=0.5, edgecolor="none",
                    color=ps.SERIES[3])
    axes[1].set_title(f"TRAINED CLIP\nPC1 = {evr[0]:.0%} of variance",
                      color=ps.INK, fontsize=11, loc="left")
    labels = ["gap between\ncentres", "two random\nimages", "two random\ncaptions"]
    before = [cone["gap_norm"], cone["img_img"], cone["txt_txt"]]
    after = [stats["gap_norm"], stats["img_img"], stats["txt_txt"]]
    x = np.arange(3)
    axes[2].bar(x - 0.19, before, 0.36, color=ps.INK_MUTED, label="untrained")
    axes[2].bar(x + 0.19, after, 0.36, color=ps.SERIES[0], label="trained")
    for xi, (b, a_) in enumerate(zip(before, after)):
        axes[2].text(xi - 0.19, b + 0.02, f"{b:.2f}", ha="center", fontsize=8.5,
                     color=ps.INK_SECONDARY)
        axes[2].text(xi + 0.19, a_ + 0.02, f"{a_:.2f}", ha="center", fontsize=8.5,
                     color=ps.INK_SECONDARY)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, fontsize=9)
    axes[2].set_title("Training SHRINKS the gap — it never closes it",
                      color=ps.INK, fontsize=11, loc="left")
    axes[2].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "cone_effect.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'cone_effect.png'}")

    # ---- 4. closing the gap costs accuracy ---------------------------------
    with open(OUT / "closing_the_gap.csv") as f:
        rows = list(csv.DictReader(f))
    short = {"as-is (no fix)": "as-is\n(no fix)",
             "shift text 0.5x gap": "shift text\n0.5× gap",
             "shift text 1x gap": "shift text\n1× gap",
             "shift text 1x gap, no renorm": "shift 1× gap\nno renorm",
             "center each modality": "center\neach modality",
             "delete the gap axis": "delete\ngap axis"}
    names = [short[r["method"]] for r in rows]
    cos = [float(r["matched_score"]) for r in rows]
    i2t = [float(r["i2t_r1"]) for r in rows]
    t2i = [float(r["t2i_r1"]) for r in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)

    axes[0].bar(x, cos, 0.55, color=ps.SERIES[1])
    for xi, c in enumerate(cos):
        axes[0].text(xi, c + 0.012, f"{c:.2f}", ha="center", fontsize=9,
                     color=ps.INK_SECONDARY)
    axes[0].set_title("The number that LOOKS good\nmean score of matched pairs",
                      color=ps.INK, fontsize=11, loc="left")
    axes[0].set_ylim(0, 0.8)

    axes[1].bar(x - 0.19, i2t, 0.36, color=ps.SERIES[0], label="image → text R@1")
    axes[1].bar(x + 0.19, t2i, 0.36, color=ps.SERIES[2], label="text → image R@1")
    for xi, (a_, b_) in enumerate(zip(i2t, t2i)):
        axes[1].text(xi - 0.19, a_ + 0.012, f"{a_:.2f}", ha="center", fontsize=8,
                     color=ps.INK_SECONDARY)
        axes[1].text(xi + 0.19, b_ + 0.012, f"{b_:.2f}", ha="center", fontsize=8,
                     color=ps.INK_SECONDARY)
    axes[1].axhline(i2t[0], color=ps.INK_MUTED, linewidth=0.9, linestyle="--")
    axes[1].legend(frameon=False, fontsize=9.5, loc="upper right")
    axes[1].set_title("The number that MATTERS\nretrieval accuracy, dashed = as-is",
                      color=ps.INK, fontsize=11, loc="left")
    axes[1].set_ylim(0, 0.72)

    for a in axes:
        a.set_xticks(x)
        a.set_xticklabels(names, fontsize=8)
    fig.suptitle("Whenever the good-looking score goes up, the accuracy goes down",
                 color=ps.INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(OUT / "closing_the_gap.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'closing_the_gap.png'}")

    # ---- 5. what the two clouds actually contain ---------------------------
    from PIL import Image
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 4.9), dpi=110,
                             gridspec_kw=dict(height_ratios=[3.0, 1.45]))
    fig.patch.set_facecolor(ps.SURFACE)
    order = np.argsort(sc[:N, 0])
    picks = [order[0], order[N // 3], order[2 * N // 3], order[-1]]
    for col, i in enumerate(picks):
        axes[0, col].imshow(Image.open(paths[i]))
        axes[0, col].set_title(f"image PC1 = {sc[i, 0]:+.2f}", fontsize=9,
                               color=ps.SERIES[0])
        axes[0, col].axis("off")
        axes[1, col].axis("off")
        wrapped = "\n".join(_wrap(caps[i], 28))
        axes[1, col].text(0.5, 0.95, wrapped, ha="center", va="top", fontsize=8.5,
                          color=ps.INK, transform=axes[1, col].transAxes)
        axes[1, col].text(0.5, 0.02, f"caption PC1 = {sc[N + i, 0]:+.2f}",
                          ha="center", fontsize=9, color=ps.SERIES[3],
                          transform=axes[1, col].transAxes)
    fig.suptitle("Content changes completely across the four columns.\n"
                 "PC1 does not care: every image is positive, every caption negative.",
                 color=ps.INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "pc1_examples.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'pc1_examples.png'}")


def _wrap(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
STAGES = dict(data=stage_data, encode=stage_encode, cone=stage_cone,
              close=stage_close, figures=stage_figures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        print(f"\n=== {name} ===")
        STAGES[name](args)


if __name__ == "__main__":
    main()
