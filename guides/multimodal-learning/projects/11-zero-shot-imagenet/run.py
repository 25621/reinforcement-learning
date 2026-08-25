"""Project 11 -- Zero-shot classification with a frozen CLIP.

Stages:
    prompts    sweep prompting styles on an easy and a hard 10-class task
    ensemble   normalize-then-average vs average-then-normalize
    probe      the supervised ceiling, for context
    confusion  which dog breeds CLIP mixes up, and what a better prompt fixes

    python3 run.py --stage all      # ~5 min cold (two ~95 MB downloads)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-modality-survey"))
sys.path.insert(0, str(HERE.parent / "02-visualize-the-modality-gap"))

import clip_lib                                                   # noqa: E402
import plot_style as ps                                           # noqa: E402
import zeroshot as zs                                             # noqa: E402

OUT = HERE / "outputs"
DATASETS = ["imagenette", "imagewoof"]
STYLES = ["wnid", "bare", "photo", "context", "ensemble"]
PER_CLASS = 100


def stage_prompts():
    rows = []
    for name in DATASETS:
        feat, labels = zs.cached_features(name, PER_CLASS)
        for style in STYLES:
            w = zs.classifier_weights(name, style)
            acc, _ = zs.zero_shot_accuracy(feat, labels, w)
            rows.append({"dataset": name, "prompt_style": style,
                         "accuracy": round(acc, 4), "chance": 0.1})
            print(f"  {name:11s} {style:9s} {acc:.4f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "prompts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = ps.new_axes(7.6, 4.2)
    x = np.arange(len(STYLES))
    for i, name in enumerate(DATASETS):
        vals = [r["accuracy"] for r in rows if r["dataset"] == name]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, color=ps.SERIES[i], label=name)
        for xi, v in zip(x + (i - 0.5) * 0.38, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=8,
                    color=ps.INK_SECONDARY)
    ax.axhline(0.1, color=ps.INK_MUTED, linestyle="--", linewidth=1)
    ax.text(4.3, 0.12, "chance", color=ps.INK_MUTED, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(STYLES)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "The same wording change is worth 1 point on an easy task and 20 on a hard one",
              "how the class name was written", "zero-shot accuracy", OUT / "prompts.png")


SOURCES = ["wnid", "lemma", "openai"]
FULL_STYLES = ["bare", "photo", "ensemble"]


def stage_fullspace():
    """The real task: 1,000 candidate labels, not 10."""
    rows = []
    for name in DATASETS:
        feat, labels = zs.cached_features(name, PER_CLASS)
        for source in SOURCES:
            for style in FULL_STYLES:
                w = zs.full_classifier(source, style)
                acc, _, _ = zs.full_space_accuracy(feat, labels, name, w)
                rows.append({"dataset": name, "name_source": source,
                             "prompt_style": style, "accuracy": round(acc, 4),
                             "chance": 0.001})
                print(f"  {name:11s} {source:7s} {style:9s} {acc:.4f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "fullspace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = ps.new_axes(7.8, 4.2)
    x = np.arange(len(FULL_STYLES))
    slot = 0
    for name in DATASETS:
        for source in ("lemma", "openai"):
            vals = [r["accuracy"] for r in rows
                    if r["dataset"] == name and r["name_source"] == source]
            ax.bar(x + (slot - 1.5) * 0.2, vals, 0.19, color=ps.SERIES[slot],
                   label=f"{name} / {source}")
            slot += 1
    ax.set_xticks(x)
    ax.set_xticklabels(FULL_STYLES)
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ps.finish(fig, ax, "1,000 candidate labels: now the wording is worth real points",
              "prompt style", "zero-shot accuracy (1,000-way)", OUT / "fullspace.png")


def stage_ensemble():
    """Is the ensemble better than its best member -- or just better than its
    average member? Run in the 1,000-way space, where templates actually differ."""
    names = zs.imagenet_names("openai")
    singles = []
    for k, template in enumerate(zs.TEMPLATES):
        emb = clip_lib.l2_normalize(zs.encode_texts([template.format(n) for n in names]))
        singles.append(emb.astype(np.float32))
    stack = np.stack(singles)                                # (7, 1000, 512)

    rows = []
    for name in DATASETS:
        feat, labels = zs.cached_features(name, PER_CLASS)
        accs = [zs.full_space_accuracy(feat, labels, name, w)[0] for w in singles]
        for k, a in enumerate(accs):
            rows.append({"dataset": name, "classifier": f"template {k + 1}: "
                                                        f"{zs.TEMPLATES[k]}",
                         "accuracy": round(a, 4)})
        # normalize each template first, then average, then normalize again
        w_norm = clip_lib.l2_normalize(stack.mean(0))
        acc_norm = zs.full_space_accuracy(feat, labels, name, w_norm)[0]
        rows.append({"dataset": name, "classifier": "ensemble (normalize first)",
                     "accuracy": round(acc_norm, 4)})
        rows.append({"dataset": name, "classifier": "best single template",
                     "accuracy": round(float(max(accs)), 4)})
        rows.append({"dataset": name, "classifier": "mean of single templates",
                     "accuracy": round(float(np.mean(accs)), 4)})
        rows.append({"dataset": name, "classifier": "worst single template",
                     "accuracy": round(float(min(accs)), 4)})
        print(f"  {name:11s} templates {min(accs):.4f}..{max(accs):.4f} "
              f"(mean {np.mean(accs):.4f})   ensemble {acc_norm:.4f}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "ensemble.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # How different are the 7 sentence vectors for the same class? If they were
    # identical, averaging them could not possibly change anything.
    sims = []
    for a in range(len(zs.TEMPLATES)):
        for b in range(a + 1, len(zs.TEMPLATES)):
            sims.append(float(np.mean(np.sum(stack[a] * stack[b], axis=1))))
    (OUT / "template_agreement.json").write_text(json.dumps({
        "mean_cosine_between_two_templates_of_the_same_class": round(float(np.mean(sims)), 4),
        "min": round(float(np.min(sims)), 4), "max": round(float(np.max(sims)), 4)},
        indent=2))
    print(f"  two templates for the SAME class agree at cosine "
          f"{np.mean(sims):.3f} on average")


def stage_probe():
    rows = []
    for name in DATASETS:
        feat, labels = zs.cached_features(name, PER_CLASS)
        # train split of the dataset, so the probe never sees the test images
        tr_paths, tr_labels = zs.load_split(name, per_class=60, split="train")
        cache = zs.DATA / f"feat_{name}_train60.npz"
        if cache.exists():
            tr_feat = np.load(cache)["feat"]
        else:
            print(f"  encoding {len(tr_paths)} {name} train images...")
            tr_feat = zs.encode_images(tr_paths)
            np.savez(cache, feat=tr_feat)

        best = max(
            (zs.zero_shot_accuracy(feat, labels, zs.classifier_weights(name, s))[0], s)
            for s in STYLES)
        probe = zs.linear_probe(tr_feat, tr_labels, feat, labels)
        rows.append({"dataset": name,
                     "zero_shot_best": round(best[0], 4),
                     "zero_shot_best_style": best[1],
                     "linear_probe_600_labels": round(probe, 4),
                     "gap": round(probe - best[0], 4)})
        print(f"  {name:11s} zero-shot {best[0]:.4f} ({best[1]})   probe {probe:.4f}")

    OUT.mkdir(exist_ok=True)
    (OUT / "probe.json").write_text(json.dumps(rows, indent=2))


def stage_confusion():
    name = "imagewoof"
    feat, labels = zs.cached_features(name, PER_CLASS)
    names = [n for _, n in zs.CLASSES[name]]
    fig, axes = None, None
    import matplotlib.pyplot as plt

    mats = {}
    for style in ("bare", "ensemble"):
        w = zs.classifier_weights(name, style)
        acc, scores = zs.zero_shot_accuracy(feat, labels, w)
        pred = scores.argmax(1)
        m = np.zeros((10, 10))
        for a, b in zip(labels, pred):
            m[a, b] += 1
        mats[style] = m / m.sum(1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, style in zip(axes, ("bare", "ensemble")):
        ax.imshow(mats[style], cmap="magma_r", vmin=0, vmax=1)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(names, rotation=90, fontsize=7, color=ps.INK_SECONDARY)
        ax.set_yticklabels(names, fontsize=7, color=ps.INK_SECONDARY)
        ax.set_title(f"prompt: {style}", color=ps.INK, fontsize=11, loc="left")
        ax.set_xlabel("predicted", color=ps.INK_SECONDARY, fontsize=9)
        ax.set_ylabel("true", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "confusion.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'confusion.png'}")

    # the single worst confusion under each prompt
    worst = {}
    for style, m in mats.items():
        off = m - np.diag(np.diag(m))
        i, j = np.unravel_index(off.argmax(), off.shape)
        worst[style] = {"true": names[i], "predicted_as": names[j],
                        "rate": round(float(off[i, j]), 3),
                        "recall_of_true_class": round(float(m[i, i]), 3)}
        print(f"  {style:9s} worst: {names[i]} -> {names[j]} ({off[i, j]:.2f})")
    (OUT / "confusion.json").write_text(json.dumps(worst, indent=2))

    # Where do the mistakes go once all 1,000 labels are on the table? This is
    # the part the 10-class matrix cannot show: most wrong answers are not other
    # imagewoof breeds at all, they are the 110 dog breeds we hold no images for.
    all_names = zs.imagenet_names("openai")
    w = zs.full_classifier("openai", "ensemble")
    acc, pred, true_idx = zs.full_space_accuracy(feat, labels, name, w)
    subset = set(zs.dataset_indices(name).tolist())
    wrong = pred != true_idx
    leaks = {}
    for cls in range(10):
        mask = wrong & (labels == cls)
        picked = pred[mask]
        counts = {}
        for p in picked:
            counts[all_names[p]] = counts.get(all_names[p], 0) + 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        leaks[names[cls]] = {
            "recall": round(float((~wrong & (labels == cls)).sum() / (labels == cls).sum()), 3),
            "top_wrong_labels": [{"label": k, "count": v} for k, v in top]}
    outside = float(np.mean([p not in subset for p in pred[wrong]]))
    summary = {"accuracy_1000_way": round(acc, 4),
               "share_of_errors_landing_outside_the_10_classes": round(outside, 3),
               "per_class": leaks}
    (OUT / "confusion_1000way.json").write_text(json.dumps(summary, indent=2))
    print(f"  1,000-way: {acc:.3f} correct; {outside:.0%} of the mistakes name a "
          f"class we hold no images for")


STAGES = {"prompts": stage_prompts, "fullspace": stage_fullspace, "ensemble": stage_ensemble,
          "probe": stage_probe, "confusion": stage_confusion}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    args = ap.parse_args()
    torch.set_num_threads(12)
    for n in (STAGES if args.stage == "all" else [args.stage]):
        print(f"\n=== {n}")
        STAGES[n]()
