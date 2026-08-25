"""Train the same ViT at three patch sizes and price the accuracy it buys.

Reuses project 04's `vit.py` verbatim -- same architecture, same optimizer, same
data -- so the only thing that changes across runs is the patch size.

The trick that makes this cheap: every run gets the same *wall-clock* budget and
logs (step, seconds, accuracy) as it goes. One set of runs then answers two
different questions, because you can slice the same curves along either axis:

  equal steps -> who is ahead after the same number of updates?
  equal time  -> who is ahead after the same number of seconds?

They do not give the same answer, and that gap is the point of the project.

Stages:
  flops    count real FLOPs per image at each patch size (no training)
  train    one wall-clock-budgeted run per patch size
  figures  redraw the charts from the saved results

    python3 run.py --stage all      # ~10 min on a CPU
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P04 = HERE.parent / "04-implement-vit-from-scratch"
sys.path.insert(0, str(HERE.parent / "01-modality-survey"))
sys.path.insert(0, str(P04))
import plot_style as ps  # noqa: E402

from vit import ViT, cifar_data, evaluate, to_tensor, train_vit  # noqa: E402

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
DATA = P04 / "data"          # share project 04's CIFAR-10 cache

PATCHES = [8, 4, 2]
ARCH = dict(img_size=32, dim=128, depth=4, heads=4)
BUDGET_SECONDS = 170         # identical compute for every patch size
LR = 1e-3

# CIFAR-10 is 32x32, so patch 8 / 4 / 2 give 16 / 64 / 256 tokens. Those are the
# same token counts a 224x224 ViT gets from patch 56 / 28 / 14 -- the guide's
# "8, 16, 32" sweep, rescaled to an image 7x smaller in each direction. What a
# ViT actually feels is the *number of tokens*, not the pixel width of a patch.
EVAL_EVERY = {8: 150, 4: 45, 2: 12}


def stage_flops():
    """Count real multiply-adds per image, and split attention out from the rest.

    `FlopCounterMode` instruments the actual tensor ops, so this is a
    measurement, not an estimate from a formula.
    """
    from torch.utils.flop_counter import FlopCounterMode
    rows = []
    x = torch.randn(1, 3, 32, 32)
    for p in PATCHES:
        # fast=False on purpose. FlopCounterMode does not know how to price the
        # fused `scaled_dot_product_attention` kernel on CPU and silently counts
        # it as zero, which would hide exactly the term this project is about.
        # The unfused path spells attention out as two `bmm`s, which it does count.
        m = ViT(**ARCH, patch=p, fast=False)
        m.eval()
        with torch.no_grad():
            with FlopCounterMode(display=False) as ctr:
                m(x)
            total = ctr.get_total_flops()
            by_op = ctr.get_flop_counts()["Global"]
            # The two batched matmuls inside attention -- queries x keys, then
            # weights x values -- are the part that grows with the SQUARE of the
            # token count. Every other layer grows only linearly.
            attn = sum(v for k, v in by_op.items() if str(k).endswith("bmm"))
            assert attn > 0, "attention FLOPs not counted"
        n_tok = m.patch_embed.n_patches + 1
        # How many raw numbers one patch holds, versus the width of the vector
        # it is projected into. Above 1.0 the projection must throw something
        # away; below 1.0 it has room to spare.
        raw = p * p * 3
        rows.append(dict(patch=p, tokens=n_tok, params=m.n_params(),
                         raw_numbers_per_patch=raw,
                         compression=round(raw / ARCH["dim"], 3),
                         mflops=round(total / 1e6, 2),
                         attn_mflops=round(attn / 1e6, 2),
                         attn_share=round(attn / total, 4)))
        print(f"  patch {p}: {n_tok:3d} tokens, {total / 1e6:7.2f} MFLOPs/image, "
              f"attention is {attn / total * 100:4.1f}% of them, "
              f"{raw:3d} raw numbers -> {ARCH['dim']}-d "
              f"({raw / ARCH['dim']:.2f}x), {m.n_params() / 1e6:.3f}M params")
    _write_csv(OUT / "flops.csv", rows)
    return rows


def stage_train(only=None):
    data = cifar_data(DATA)
    hists = _load()
    for p in PATCHES:
        if only and p != only:
            continue
        torch.manual_seed(0)
        model = ViT(**ARCH, patch=p)
        n_tok = model.patch_embed.n_patches + 1
        print(f"\n=== patch {p}: {n_tok} tokens, "
              f"{model.n_params() / 1e6:.3f}M params, "
              f"{BUDGET_SECONDS}s budget")
        h = train_vit(model, data, steps=100000, lr=LR, seed=0,
                      eval_every=EVAL_EVERY[p], n_eval=1000,
                      label=f"patch{p}", max_seconds=BUDGET_SECONDS,
                      schedule="const")
        xte, yte = data["test"]
        h["final_acc"] = evaluate(model, to_tensor(xte), torch.from_numpy(yte))
        h["tokens"] = n_tok
        h["params"] = model.n_params()
        print(f"  [patch{p}] full test set: {h['final_acc']:.4f} "
              f"after {h['step'][-1]} steps / {h['sec'][-1]:.0f}s")
        hists[str(p)] = h
        (OUT / "training.json").write_text(json.dumps(hists, indent=2))
    return hists


def _load():
    f = OUT / "training.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {path}")


def _at(h, axis, value):
    """Accuracy at the last point where h[axis] <= value (None if never reached)."""
    best = None
    for a, acc in zip(h[axis], h["acc"]):
        if a <= value:
            best = acc
    return best


def stage_slices():
    """Read the same three curves along both axes."""
    hists = _load()
    common_steps = min(h["step"][-1] for h in hists.values())
    common_sec = min(h["sec"][-1] for h in hists.values())
    rows = []
    for p in PATCHES:
        h = hists[str(p)]
        rows.append(dict(
            patch=p, tokens=h["tokens"],
            steps_done=h["step"][-1], seconds=round(h["sec"][-1], 1),
            equal_step_acc=_at(h, "step", common_steps),
            equal_time_acc=_at(h, "sec", common_sec),
            budget_acc=round(h["final_acc"], 4)))
    print(f"\nequal-step slice at {common_steps} steps; "
          f"equal-time slice at {common_sec:.0f}s")
    for r in rows:
        print(f"  patch {r['patch']}: {r['steps_done']:5d} steps in "
              f"{r['seconds']:5.0f}s | equal-step {r['equal_step_acc']:.3f} | "
              f"equal-time {r['equal_time_acc']:.3f} | "
              f"full budget {r['budget_acc']:.3f}")
    _write_csv(OUT / "slices.csv", rows)
    (OUT / "slice_points.json").write_text(json.dumps(
        {"common_steps": common_steps, "common_seconds": round(common_sec, 1)},
        indent=2))
    return rows


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_two_axes():
    hists = _load()
    pts = json.loads((OUT / "slice_points.json").read_text())
    fig, axes = ps.plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, axis, xlabel, marker in (
            (axes[0], "step", "training steps", pts["common_steps"]),
            (axes[1], "sec", "seconds of CPU time", pts["common_seconds"])):
        ps.style_axes(ax)
        for i, p in enumerate(PATCHES):
            h = hists[str(p)]
            ax.plot(h[axis], h["acc"], color=ps.SERIES[i], lw=2,
                    label=f"patch {p} ({h['tokens']} tokens)")
        ax.axvline(marker, color=ps.INK_MUTED, ls="--", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel(xlabel, color=ps.INK_SECONDARY, fontsize=10)
        ax.set_ylim(0.05, 0.6)
    axes[0].set_ylabel("test accuracy", color=ps.INK_SECONDARY, fontsize=10)
    axes[0].set_title("Same number of UPDATES\n"
                      "(the three curves lie on top of each other)",
                      color=ps.INK, fontsize=11, loc="left", pad=8)
    axes[1].set_title("Same number of SECONDS\n"
                      "(the same runs, now clearly ranked)",
                      color=ps.INK, fontsize=11, loc="left", pad=8)
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "two_axes.png", facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'two_axes.png'}")


def fig_cost():
    with open(OUT / "flops.csv") as f:
        fl = {int(r["patch"]): r for r in csv.DictReader(f)}
    with open(OUT / "slices.csv") as f:
        sl = {int(r["patch"]): r for r in csv.DictReader(f)}
    fig, axes = ps.plt.subplots(1, 2, figsize=(11.2, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    ax = axes[0]; ps.style_axes(ax)
    toks = [int(fl[p]["tokens"]) for p in PATCHES]
    tot = [float(fl[p]["mflops"]) for p in PATCHES]
    att = [float(fl[p]["attn_mflops"]) for p in PATCHES]
    ax.plot(toks, tot, color=ps.SERIES[0], lw=2, marker="o", ms=6,
            label="total FLOPs")
    ax.plot(toks, att, color=ps.SERIES[2], lw=2, marker="s", ms=6,
            label="attention only (grows with tokens²)")
    for t, v, p in zip(toks, tot, PATCHES):
        ax.annotate(f"patch {p}", (t, v), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color=ps.INK_SECONDARY)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(toks); ax.set_xticklabels([str(t) for t in toks])
    ax.set_xlabel("tokens per image", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("MFLOPs per image", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_title("Cost per image", color=ps.INK, fontsize=11, loc="left", pad=8)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    ax = axes[1]; ps.style_axes(ax)
    for i, p in enumerate(PATCHES):
        ax.scatter(float(fl[p]["mflops"]), float(sl[p]["budget_acc"]),
                   s=150, color=ps.SERIES[i], zorder=3, edgecolor="white",
                   linewidth=1.2, label=f"patch {p}")
        ax.annotate(f"patch {p}\n{sl[p]['steps_done']} steps",
                    (float(fl[p]["mflops"]), float(sl[p]["budget_acc"])),
                    textcoords="offset points", xytext=(10, -6), fontsize=8.5,
                    color=ps.INK_SECONDARY)
    ax.set_xscale("log")
    ax.set_xlabel("MFLOPs per image", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel(f"accuracy after {BUDGET_SECONDS}s", color=ps.INK_SECONDARY,
                  fontsize=10)
    ax.set_title("What that cost buys, at a fixed time budget",
                 color=ps.INK, fontsize=11, loc="left", pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "flops_vs_accuracy.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {OUT / 'flops_vs_accuracy.png'}")


def main():
    torch.set_num_threads(12)
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "flops", "train", "slices", "figures"])
    ap.add_argument("--only", type=int, default=None)
    a = ap.parse_args()
    if a.stage in ("all", "flops"):
        stage_flops()
    if a.stage in ("all", "train"):
        stage_train(a.only)
    if a.stage in ("all", "slices"):
        stage_slices()
    if a.stage in ("all", "figures"):
        fig_two_axes()
        fig_cost()


if __name__ == "__main__":
    main()
