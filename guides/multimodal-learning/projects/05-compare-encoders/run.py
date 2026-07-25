"""Freeze four image encoders, probe them on the same images, and rank them.

Stages:
  embed    run all four frozen towers over the same 1,100 Imagenette images
  probe    linear probe + k-NN probe, swept over how many labels you have
  figures  redraw the charts from the saved CSV

    python3 run.py --stage all      # ~7 min cold, ~10 s once features are cached
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "01-modality-survey"))
import plot_style as ps  # noqa: E402

from encoders import (CLASSES, WNID_TO_NAME, all_encoders, fetch_imagenette,  # noqa: E402
                      knn_probe, linear_probe, load_split)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE / "data"
OUT.mkdir(exist_ok=True)

TRAIN_PER_CLASS = 60      # 600 labelled images, the largest probe budget
TEST_PER_CLASS = 50       # 500 held-out images
SHOTS = [1, 5, 15, 60]    # labels per class, swept


def stage_embed():
    root = fetch_imagenette(DATA)
    cache = DATA / "features.npz"
    xtr, ytr = load_split(root, "train", TRAIN_PER_CLASS, seed=0)
    xte, yte = load_split(root, "val", TEST_PER_CLASS, seed=1)
    print(f"[data] {len(ytr)} train / {len(yte)} test images, "
          f"{len(CLASSES)} classes")

    store, meta = {"ytr": ytr, "yte": yte}, {}
    for enc in all_encoders():
        print(f"\n=== {enc.name} ({enc.how})")
        store[f"{enc.key}_tr"] = enc.embed(xtr)
        store[f"{enc.key}_te"] = enc.embed(xte)
        meta[enc.key] = {"name": enc.name, "how": enc.how,
                         "dim": int(store[f"{enc.key}_tr"].shape[1]),
                         "params_m": round(enc.n_params() / 1e6, 1),
                         "ms_per_image": round(enc.ms_per_image, 1)}
        enc.model = None                      # free ~350 MB before the next one
    np.savez_compressed(cache, **store)
    (OUT / "encoders.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {cache} and {OUT / 'encoders.json'}")


def stage_probe():
    d = np.load(DATA / "features.npz")
    meta = json.loads((OUT / "encoders.json").read_text())
    ytr, yte = d["ytr"], d["yte"]

    rows = []
    for key, m in meta.items():
        ftr, fte = d[f"{key}_tr"], d[f"{key}_te"]
        for shots in SHOTS:
            # Take the first `shots` examples of each class -- a fixed, shared
            # subset, so every encoder is probed on exactly the same labels.
            idx = np.concatenate([np.where(ytr == c)[0][:shots]
                                  for c in range(len(CLASSES))])
            lin = linear_probe(ftr[idx], ytr[idx], fte, yte)
            # k can never exceed the examples available per class: with 1 label
            # per class, asking for the 5 nearest neighbours guarantees 4 of them
            # come from wrong classes, and the vote becomes noise.
            knn = knn_probe(ftr[idx], ytr[idx], fte, yte, k=min(10, shots))
            rows.append(dict(encoder=key, name=m["name"], how=m["how"],
                             dim=m["dim"], params_m=m["params_m"],
                             ms_per_image=m["ms_per_image"],
                             shots=shots, linear=round(lin, 4),
                             knn=round(knn, 4)))
            print(f"  {m['name']:14s} {shots:3d} labels/class  "
                  f"linear {lin:.3f}  knn {knn:.3f}", flush=True)

    with open(OUT / "probes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT / 'probes.csv'}")
    return rows


def _read_rows():
    with open(OUT / "probes.csv") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("shots", "dim"):
            r[k] = int(r[k])
        for k in ("linear", "knn", "ms_per_image", "params_m"):
            r[k] = float(r[k])
    return rows


def fig_shots(rows):
    keys = list(dict.fromkeys(r["encoder"] for r in rows))
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, metric, title in zip(axes, ("linear", "knn"),
                                 ("Linear probe", "k-NN probe (no training)")):
        ps.style_axes(ax)
        for i, k in enumerate(keys):
            rs = sorted([r for r in rows if r["encoder"] == k],
                        key=lambda r: r["shots"])
            ax.plot([r["shots"] for r in rs], [r[metric] for r in rs],
                    color=ps.SERIES[i], lw=2, marker="o", ms=4,
                    label=rs[0]["name"])
        ax.set_xscale("log")
        ax.set_xticks(SHOTS); ax.set_xticklabels([str(s) for s in SHOTS])
        ax.axhline(0.1, color=ps.BASELINE, ls="--", lw=1.1)
        ax.set_ylim(0, 1.03)
        ax.set_title(title, color=ps.INK, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("labelled images per class", color=ps.INK_SECONDARY,
                      fontsize=10)
    axes[0].set_ylabel("test accuracy", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "shots_sweep.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'shots_sweep.png'}")


def fig_cost(rows):
    """Accuracy against what it costs you: milliseconds per image."""
    full = [r for r in rows if r["shots"] == max(SHOTS)]
    low = {r["encoder"]: r for r in rows if r["shots"] == 5}
    fig, ax = ps.new_axes(7.4, 4.4)
    for i, r in enumerate(full):
        ax.scatter(r["ms_per_image"], low[r["encoder"]]["linear"],
                   s=140, color=ps.SERIES[i], zorder=3, edgecolor="white",
                   linewidth=1.2)
        ax.annotate(f"{r['name']}\n{r['params_m']:.0f}M · {r['dim']}-d",
                    (r["ms_per_image"], low[r["encoder"]]["linear"]),
                    textcoords="offset points", xytext=(10, -14),
                    fontsize=8.5, color=ps.INK_SECONDARY)
    ax.set_xlim(0, max(r["ms_per_image"] for r in full) * 1.45)
    ps.finish(fig, ax,
              "Low-label accuracy (5 labels/class) vs CPU cost per image",
              "milliseconds per image (12 CPU threads)",
              "linear-probe test accuracy", OUT / "accuracy_vs_cost.png")


def stage_figures():
    rows = _read_rows()
    fig_shots(rows)
    fig_cost(rows)


def main():
    import torch
    torch.set_num_threads(12)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "embed", "probe", "figures"])
    a = ap.parse_args()
    if a.stage in ("all", "embed"):
        stage_embed()
    if a.stage in ("all", "probe"):
        stage_probe()
    if a.stage in ("all", "figures"):
        stage_figures()


if __name__ == "__main__":
    main()
