"""Project 03 — A toy cross-modal search engine, and what actually moves its score.

The engine itself is about six lines: encode everything once, L2-normalize,
one matrix multiplication, argsort, take the top 5. The interesting part is
everything around it -- the knobs that change the headline number without
changing the model at all.

Stages
    search      build the index, run both search directions, time the matmul  (~5 s)
    ablate      five knobs: normalization, gallery size, which caption, ensembling, size
    qualitative top-5 result grids, including the failures                     (~10 s)
    figures     draw everything
    all         every stage in order                                     (~1 min)

Embeddings and images come from project 02's cache, so run
`python3 ../02-visualize-the-modality-gap/run.py --stage data` first (or just
let this script trigger the download itself -- it calls the same helper).
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
LIB = HERE.parent / "02-visualize-the-modality-gap"
sys.path.insert(0, str(LIB))
sys.path.insert(0, str(HERE.parent / "01-modality-survey"))

import clip_lib as L        # noqa: E402  (path must be set first)

DATA = LIB / "data"
N = 1000
N_CAPS = 5                  # COCO ships 5 human captions per image


# ---------------------------------------------------------------------------
def load_all():
    """-> (image emb, list of 5 caption-emb arrays, paths, all captions)."""
    L.fetch_coco(DATA, n=N)
    img, _, paths, _, allc = L.cached_embeddings(DATA, N, caption_index=0)
    caps = [L.l2_normalize(L.cached_embeddings(DATA, N, caption_index=c)[1])
            for c in range(N_CAPS)]
    return L.l2_normalize(img), caps, paths, allc


# ---------------------------------------------------------------------------
def stage_search(args):
    I, caps, paths, allc = load_all()
    T = caps[0]

    # THE ENTIRE SEARCH ENGINE
    t = time.time()
    S = I @ T.T                      # (1000 queries, 1000 gallery items)
    matmul_ms = (time.time() - t) * 1000
    top5 = np.argsort(-S, axis=1)[:, :5]

    i2t = L.recall_at_k(S)
    t2i = L.recall_at_k(S.T)

    # One matmul vs. a Python loop over queries. Both call the same BLAS kernel,
    # so at a small gallery the loop is only a little slower -- the advantage of
    # batching everything into one call grows with the size of the gallery.
    timing = []
    big = np.stack(caps).transpose(1, 0, 2).reshape(N * N_CAPS, -1)
    for gallery, label in ((T, f"{len(T):,}"), (big, f"{len(big):,}")):
        t = time.time()
        _ = I @ gallery.T
        one = (time.time() - t) * 1000
        t = time.time()
        for i in range(len(I)):
            _ = gallery @ I[i]
        loop = (time.time() - t) * 1000
        timing.append(dict(gallery=label, matmul_ms=one, loop_ms=loop))
        print(f"  {len(I):,} queries x {label} items:  one matmul {one:6.1f} ms   "
              f"Python loop {loop:7.1f} ms   ({loop/max(one, 1e-9):.0f}x)")

    stats = dict(i2t={str(k): v for k, v in i2t.items()},
                 t2i={str(k): v for k, v in t2i.items()},
                 timing=timing, index_floats=int(I.size + T.size))
    (OUT / "search_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"\n  gallery: {len(T)} captions, {T.shape[1]} numbers each")
    print(f"  image → text   R@1 {i2t[1]:.3f}   R@5 {i2t[5]:.3f}   R@10 {i2t[10]:.3f}")
    print(f"  text → image   R@1 {t2i[1]:.3f}   R@5 {t2i[5]:.3f}   R@10 {t2i[10]:.3f}")
    print(f"  chance-level R@1 would be {1/len(T):.3f}")
    print(f"  first row's top-5 caption ids: {top5[0].tolist()} "
          f"(correct answer is 0)")


# ---------------------------------------------------------------------------
def stage_ablate(args):
    I, caps, paths, allc = load_all()
    img_raw = np.load(DATA / f"emb_img_{N}.npz")["img"]
    txt_raw = np.load(DATA / f"emb_txt_{N}_0.npz")["txt"]

    rows = []

    def add(group, setting, S, note=""):
        r = dict(group=group, setting=setting,
                 i2t_r1=L.recall_at_k(S)[1], i2t_r5=L.recall_at_k(S)[5],
                 t2i_r1=L.recall_at_k(S.T)[1], t2i_r5=L.recall_at_k(S.T)[5],
                 note=note)
        rows.append(r)
        print(f"  {group:<16} {setting:<22} i2t R@1 {r['i2t_r1']:.3f}  "
              f"t2i R@1 {r['t2i_r1']:.3f}  {note}")
        return r

    # (a) does L2 normalization matter, or is it decoration?
    print("\n  (a) normalize the vectors, or not")
    add("normalize", "cosine (normalized)", I @ caps[0].T)
    add("normalize", "raw dot product", img_raw @ txt_raw.T,
        f"text vector lengths vary ±{np.linalg.norm(txt_raw, axis=1).std():.2f}")

    # (b) the same model, a bigger haystack
    print("\n  (b) how big is the gallery")
    size_rows = []
    for n in (100, 250, 500, 1000):
        r = add("gallery size", f"{n} captions", I[:n] @ caps[0][:n].T)
        size_rows.append((n, r["i2t_r1"]))

    # (c) COCO gives 5 captions per image -- the "right answer" is not unique
    print("\n  (c) which of the 5 human captions you call the ground truth")
    per_caption = []
    for c in range(N_CAPS):
        r = add("caption choice", f"caption #{c}", I @ caps[c].T)
        per_caption.append(r["i2t_r1"])

    # (d) average all 5 caption vectors into one gallery entry
    print("\n  (d) merge all 5 captions into one vector before indexing")
    ens = L.l2_normalize(np.stack(caps).mean(0))
    add("ensembling", "mean of 5 captions", I @ ens.T)

    # (e) put all 5,000 captions in the gallery; any of an image's own 5 counts
    print("\n  (e) index all 5,000 captions at once (a 5x bigger haystack)")
    G = np.stack(caps).transpose(1, 0, 2).reshape(N * N_CAPS, -1)
    S5 = I @ G.T
    order = np.argsort(-S5, axis=1)
    owner = order // N_CAPS          # gallery slot -> which image the caption describes
    rank = np.argmax(owner == np.arange(N)[:, None], axis=1)
    r5 = {k: float((rank < k).mean()) for k in (1, 5, 10)}
    rows.append(dict(group="big gallery", setting="5,000 captions",
                     i2t_r1=r5[1], i2t_r5=r5[5], t2i_r1="", t2i_r5="",
                     note="correct = any of that image's own 5 captions"))
    print(f"  {'big gallery':<16} {'5,000 captions':<22} i2t R@1 {r5[1]:.3f}  "
          f"R@5 {r5[5]:.3f}   (any of the image's own 5 counts)")

    with open(OUT / "ablations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (OUT / "ablation_extra.json").write_text(json.dumps(dict(
        size_rows=size_rows, per_caption=per_caption,
        caption_spread=float(max(per_caption) - min(per_caption)),
        img_norm_std=float(np.linalg.norm(img_raw, axis=1).std()),
        txt_norm_std=float(np.linalg.norm(txt_raw, axis=1).std()),
        big_gallery=r5), indent=2))
    print(f"wrote {OUT / 'ablations.csv'}")


# ---------------------------------------------------------------------------
def stage_qualitative(args):
    """Save the actual top-5 lists, right and wrong, so the score has a face."""
    I, caps, paths, allc = load_all()
    T = caps[0]
    S = I @ T.T
    order_i2t = np.argsort(-S, axis=1)
    order_t2i = np.argsort(-S.T, axis=1)
    rank_i2t = np.argmax(order_i2t == np.arange(N)[:, None], axis=1)

    out = dict(hits=[], misses=[], text_queries=[])
    hits = [i for i in range(N) if rank_i2t[i] == 0][:4]
    misses = [i for i in np.argsort(-rank_i2t)[:4]]
    for i in hits:
        out["hits"].append(dict(image=paths[i].name, truth=allc[i][0],
                                top5=[allc[j][0] for j in order_i2t[i, :5]],
                                scores=[float(S[i, j]) for j in order_i2t[i, :5]]))
    for i in misses:
        out["misses"].append(dict(image=paths[i].name, truth=allc[i][0],
                                  rank_of_truth=int(rank_i2t[i]) + 1,
                                  top5=[allc[j][0] for j in order_i2t[i, :5]],
                                  scores=[float(S[i, j]) for j in order_i2t[i, :5]]))
    for q in (0, 7, 21, 42):
        out["text_queries"].append(dict(query=allc[q][0],
                                        top5=[paths[j].name for j in order_t2i[q, :5]],
                                        correct=paths[q].name))
    (OUT / "qualitative.json").write_text(json.dumps(out, indent=2))
    print(f"  {len(hits)} hits and {len(misses)} worst misses saved")
    for m in out["misses"][:2]:
        print(f"    MISS: true caption ranked #{m['rank_of_truth']} of 1000")
        print(f"          truth : {m['truth']}")
        print(f"          top-1 : {m['top5'][0]}")


# ---------------------------------------------------------------------------
def stage_figures(args):
    import matplotlib.pyplot as plt
    from PIL import Image
    import plot_style as ps

    I, caps, paths, allc = load_all()
    T = caps[0]
    S = I @ T.T
    order_i2t = np.argsort(-S, axis=1)
    order_t2i = np.argsort(-S.T, axis=1)
    rank_i2t = np.argmax(order_i2t == np.arange(N)[:, None], axis=1)
    extra = json.loads((OUT / "ablation_extra.json").read_text())
    stats = json.loads((OUT / "search_stats.json").read_text())

    # ---- 1. text query -> top-5 images -------------------------------------
    queries = [0, 7, 21, 42]
    fig, axes = plt.subplots(len(queries), 6, figsize=(11.4, 2.1 * len(queries)),
                             dpi=110,
                             gridspec_kw=dict(width_ratios=[1.7] + [1] * 5))
    fig.patch.set_facecolor(ps.SURFACE)
    for r, q in enumerate(queries):
        axes[r, 0].axis("off")
        axes[r, 0].text(0.0, 0.5, "“" + _wrap_one(allc[q][0], 30) + "”",
                        fontsize=9, color=ps.INK, ha="left", va="center",
                        transform=axes[r, 0].transAxes)
        for c in range(5):
            j = order_t2i[q, c]
            ax = axes[r, c + 1]
            ax.imshow(Image.open(paths[j]))
            ax.set_xticks([])
            ax.set_yticks([])
            correct = j == q
            ax.set_title(f"#{c+1}  {S[j, q]:.2f}", fontsize=8.5,
                         color=ps.SERIES[1] if correct else ps.INK_MUTED,
                         fontweight="bold" if correct else "normal")
            for side in ("top", "bottom", "left", "right"):
                ax.spines[side].set_visible(correct)
                ax.spines[side].set_color(ps.SERIES[1])
                ax.spines[side].set_linewidth(3)
    fig.suptitle("Type a caption, get 5 photos — green box = the true match",
                 color=ps.INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "text_to_image.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'text_to_image.png'}")

    # ---- 2. image query -> top-5 captions, one hit and one miss ------------
    hit = next(i for i in range(N) if rank_i2t[i] == 0)
    miss = int(np.argsort(-rank_i2t)[0])
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 2.9), dpi=110,
                             gridspec_kw=dict(width_ratios=[1, 2.1, 1, 2.1]))
    fig.patch.set_facecolor(ps.SURFACE)
    for k, (i, label) in enumerate([(hit, "a hit"), (miss, "the worst miss")]):
        axes[2 * k].imshow(Image.open(paths[i]))
        axes[2 * k].axis("off")
        axes[2 * k].set_title(label, fontsize=11, color=ps.INK, loc="left")
        ax = axes[2 * k + 1]
        ax.axis("off")
        for c in range(5):
            j = order_i2t[i, c]
            ok = j == i
            ax.text(0.0, 0.92 - 0.155 * c,
                    f"{c+1}. [{S[i, j]:.2f}] " + _wrap_one(allc[j][0], 44),
                    fontsize=8.5, va="top", transform=ax.transAxes,
                    color=ps.SERIES[1] if ok else ps.INK_SECONDARY,
                    fontweight="bold" if ok else "normal")
        if rank_i2t[i] > 4:
            ax.text(0.0, 0.06, f"true caption is ranked #{rank_i2t[i]+1}:\n"
                               + _wrap_one(allc[i][0], 44),
                    fontsize=8.5, va="top", transform=ax.transAxes,
                    color=ps.SERIES[2])
    fig.suptitle("Hand in a photo, get 5 captions", color=ps.INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "image_to_text.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'image_to_text.png'}")

    # ---- 3. the four knobs --------------------------------------------------
    with open(OUT / "ablations.csv") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in axes:
        ps.style_axes(a)

    ns = [n for n, _ in extra["size_rows"]]
    r1 = [v for _, v in extra["size_rows"]]
    axes[0].plot(ns, r1, "o-", color=ps.SERIES[0], linewidth=2)
    for n, v in zip(ns, r1):
        axes[0].annotate(f"{v:.2f}", (n, v), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=9,
                         color=ps.INK_SECONDARY)
    axes[0].set_xscale("log")
    axes[0].set_xticks(ns)
    axes[0].set_xticklabels([str(n) for n in ns])
    axes[0].set_title("Same model, bigger haystack\n→ the score falls",
                      color=ps.INK, fontsize=11, loc="left")
    axes[0].set_xlabel("captions in the gallery", color=ps.INK_SECONDARY, fontsize=9.5)
    axes[0].set_ylabel("image→text R@1", color=ps.INK_SECONDARY, fontsize=9.5)
    axes[0].set_ylim(0, 1)

    norm = [r for r in rows if r["group"] == "normalize"]
    labels = ["cosine\n(L2-normalized)", "raw dot\nproduct"]
    vals = [float(r["i2t_r1"]) for r in norm]
    axes[1].bar(labels, vals, color=[ps.SERIES[1], ps.SERIES[2]], width=0.5)
    for xi, v in enumerate(vals):
        axes[1].text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=10,
                     color=ps.INK_SECONDARY)
    axes[1].set_title("One line of code (L2-normalize)\nis worth 25 points of R@1",
                      color=ps.INK, fontsize=11, loc="left")
    axes[1].set_ylim(0, 0.72)
    axes[1].set_ylabel("image→text R@1", color=ps.INK_SECONDARY, fontsize=9.5)

    pc = extra["per_caption"]
    ens = float(next(r["i2t_r1"] for r in rows if r["setting"] == "mean of 5 captions"))
    axes[2].bar(range(len(pc)), pc, color=ps.SERIES[0], width=0.6,
                label="one human caption")
    axes[2].bar([len(pc)], [ens], color=ps.SERIES[1], width=0.6,
                label="all 5, averaged")
    for xi, v in enumerate(list(pc) + [ens]):
        axes[2].text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=9,
                     color=ps.INK_SECONDARY)
    axes[2].set_xticks(range(len(pc) + 1))
    axes[2].set_xticklabels([f"#{i}" for i in range(len(pc))] + ["mean"], fontsize=9)
    axes[2].set_ylim(0, 0.78)
    axes[2].legend(frameon=False, fontsize=9)
    axes[2].set_title("Which of the 5 captions: barely matters\nAveraging all 5: +23 points",
                      color=ps.INK, fontsize=11, loc="left")
    axes[2].set_ylabel("image→text R@1", color=ps.INK_SECONDARY, fontsize=9.5)

    fig.suptitle("Three knobs that move the headline number without touching the model",
                 color=ps.INK, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "knobs.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'knobs.png'}")

    # ---- 4. where the true answer lands ------------------------------------
    fig, ax = ps.new_axes(7.4, 4.2)
    ranks = rank_i2t + 1
    ax.hist(ranks, bins=np.logspace(0, 3, 40), color=ps.SERIES[0])
    ax.set_xscale("log")
    top = ax.get_ylim()[1]
    for (k, color), y in zip(((1, ps.SERIES[1]), (5, ps.SERIES[3]),
                              (10, ps.SERIES[2])), (0.94, 0.86, 0.78)):
        frac = float((ranks <= k).mean())
        ax.axvline(k + 0.5, color=color, linewidth=1.4, linestyle="--")
        ax.text(k + 0.9, top * y, f"R@{k} = {frac:.2f}", fontsize=9, color=color)
    ps.finish(fig, ax,
              f"Where the true caption actually lands "
              f"(median rank {int(np.median(ranks))} of 1,000)",
              "rank of the correct caption (log scale)", "number of images",
              OUT / "rank_distribution.png")


def _wrap_one(s, width):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
STAGES = dict(search=stage_search, ablate=stage_ablate,
              qualitative=stage_qualitative, figures=stage_figures)


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
