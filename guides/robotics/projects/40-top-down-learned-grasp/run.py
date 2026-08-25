"""Project 40 -- a learned top-down grasp scorer, and what it buys you.

Six experiments:

  1. the scene, the labels, and what a training patch looks like
  2. train the network; how well it separates good grasps from bad
  3. top-1 grasp success: learned vs two hand-written scorers vs random
  4. the noise sweep -- where learning starts to pay
  5. novel objects: shapes the network has never seen
  6. how much data it took, and what the network actually keyed on

Runs in about five minutes on CPU.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from topdown import (IMG, PX_PER_M, TABLE_Z, TEST_SHAPES, TRAIN_SHAPES,       # noqa: E402
                     GRIPPER_MAX, Scene, label, patch, render, sample_candidates,
                     score_analytic_observed, score_depth_heuristic,
                     score_depth_heuristic_fixed)
from net import GQCNN, score, train                                           # noqa: E402
from plot_style import COLORS, use_style, save                                # noqa: E402

import matplotlib.pyplot as plt                                               # noqa: E402
from matplotlib.patches import Polygon as MplPolygon                          # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

NOISE = 0.0015          # 1.5 mm axial noise: a good consumer depth camera
DROPOUT = 0.004
N_TRAIN = 1000
N_TEST = 220
N_CAND = 44


def record(exp, key, value):
    RESULTS.append((exp, key, value))
    print(f"    {key:<48s} {value}")


def build(n_scenes, shapes, rng, noise, dropout, n_cand=N_CAND, keep_scene=False):
    """Render scenes, sample candidates, cut patches, ask the oracle."""
    X, y, meta, scenes = [], [], [], []
    for _ in range(n_scenes):
        sc = Scene(rng, shapes, n=int(rng.integers(2, 5)))
        if not sc.polys:
            continue
        depth, ids = render(sc, rng, noise, dropout)
        cands = sample_candidates(sc, rng, n_cand)
        for g in cands:
            ok, info = label(sc, g)
            X.append(patch(depth, g))
            y.append(ok)
            meta.append(info["reason"])
        if keep_scene:
            scenes.append((sc, depth, cands))
    return (np.array(X, np.float32), np.array(y, bool), meta, scenes)


# ---------------------------------------------------------------------------
# 1. the picture
# ---------------------------------------------------------------------------

def exp1_scene():
    print("\n[1] a scene, its labels, and a training patch")
    rng = np.random.default_rng(4)
    sc = Scene(rng, TRAIN_SHAPES, n=4)
    clean, _ = render(sc, rng, 0.0, 0.0)
    noisy, _ = render(sc, rng, NOISE, DROPOUT)
    cands = sample_candidates(sc, rng, 90)
    labs = np.array([label(sc, g)[0] for g in cands])

    fig = plt.figure(figsize=(11.0, 3.6))
    ax = fig.add_subplot(1, 4, 1)
    ax.imshow(TABLE_Z - clean, cmap="magma", origin="lower")
    ax.set_title("depth (height above table)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(1, 4, 2)
    ax.imshow(TABLE_Z - noisy, cmap="magma", origin="lower")
    ax.set_title(f"+ sensor noise ({1000 * NOISE:.1f} mm, worse at edges)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)

    ax = fig.add_subplot(1, 4, 3)
    for P in sc.polys:
        ax.add_patch(MplPolygon(P * PX_PER_M + IMG / 2, closed=True,
                                fc="#d7dee6", ec="#42505e", lw=1.0))
    half = 0.5 * GRIPPER_MAX * PX_PER_M
    for g, ok in zip(cands, labs):
        u = g[0] * PX_PER_M + IMG / 2
        v = g[1] * PX_PER_M + IMG / 2
        dx, dy = np.cos(g[2]) * half, np.sin(g[2]) * half
        ax.plot([u - dx, u + dx], [v - dy, v + dy],
                color=COLORS[2] if ok else COLORS[1], lw=1.0,
                alpha=0.9 if ok else 0.35)
    ax.set_xlim(0, IMG); ax.set_ylim(0, IMG)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"candidates: {labs.sum()} hold (green), "
                 f"{(~labs).sum()} fail", fontsize=9)

    k = int(np.flatnonzero(labs)[0]) if labs.any() else 0
    ax = fig.add_subplot(1, 4, 4)
    im = ax.imshow(patch(noisy, cands[k]), cmap="magma", origin="lower")
    ax.axhline(16, color="w", lw=0.6, ls=":")
    ax.set_title("one 32x32 patch\n(rotated, centre depth removed)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, os.path.join(OUT, "scene.png"))
    record("1_scene", "positive rate in this scene",
           f"{labs.mean():.3f}")


# ---------------------------------------------------------------------------
# 2 + 3. train, and compare top-1 success
# ---------------------------------------------------------------------------

def _top1(scorer, scenes, rng):
    """Take the single best-scoring candidate per scene; did it hold?"""
    hits = tot = 0
    for sc, depth, cands in scenes:
        s = scorer(depth, cands)
        if not np.isfinite(s).any():
            continue
        k = int(np.argmax(s))
        hits += label(sc, cands[k])[0]
        tot += 1
    return hits / max(tot, 1)


def exp23_train_and_compare(state):
    print("\n[2] building the training set")
    rng = np.random.default_rng(0)
    t0 = time.time()
    Xtr, ytr, reasons, _ = build(N_TRAIN, TRAIN_SHAPES, rng, NOISE, DROPOUT)
    print(f"    {len(Xtr)} patches in {time.time() - t0:.0f}s")
    record("2_train", "training patches", len(Xtr))
    record("2_train", "positive rate", round(float(ytr.mean()), 4))
    from collections import Counter
    for why, c in Counter(reasons).most_common():
        record("2_train", f"  label reason: {why}", f"{c} ({100 * c / len(reasons):.1f}%)")

    Xva, yva, _, scenes = build(N_TEST, TRAIN_SHAPES, np.random.default_rng(1),
                                NOISE, DROPOUT, keep_scene=True)
    t0 = time.time()
    net, curve = train(Xtr, ytr, log=lambda s: print(s))
    record("2_train", "training time (s)", round(time.time() - t0, 1))

    p = score(net, Xva)
    acc = float(((p > 0.5) == yva).mean())
    # ranking quality: probability a random good grasp outranks a random bad one
    pos, neg = p[yva], p[~yva]
    auc = float((pos[:, None] > neg[None, :]).mean())
    record("2_train", "held-out accuracy", round(acc, 4))
    record("2_train", "held-out AUC (good outranks bad)", round(auc, 4))
    record("2_train", "held-out positive rate", round(float(yva.mean()), 4))

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    axes[0].plot(range(1, len(curve) + 1), curve, "o-", ms=3)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("weighted BCE loss")
    axes[0].set_title("training")
    axes[1].hist(neg, bins=40, alpha=0.6, label="grasps that fail", color=COLORS[1])
    axes[1].hist(pos, bins=40, alpha=0.6, label="grasps that hold", color=COLORS[2])
    axes[1].set_xlabel("network score"); axes[1].set_ylabel("count")
    axes[1].set_title(f"held out: AUC {auc:.3f}")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "training.png"))

    print("\n[3] top-1 grasp success on fresh scenes")
    rnd = np.random.default_rng(7)
    methods = {
        "random": lambda d, c: rnd.random(len(c)),
        "depth heuristic (as printed in the guide)": score_depth_heuristic,
        "depth heuristic, sign fixed": score_depth_heuristic_fixed,
        "analytic on observed depth": score_analytic_observed,
        "learned GQ-CNN": lambda d, c: score(net, np.stack([patch(d, g) for g in c])),
    }
    rates = {}
    for name, fn in methods.items():
        t0 = time.time()
        rates[name] = _top1(fn, scenes, rnd)
        dt = (time.time() - t0) / len(scenes) * 1000
        record("3_compare", f"{name}: top-1 success", round(rates[name], 4))
        record("3_compare", f"{name}: ms per scene", round(dt, 1))
    # the ceiling: how often ANY candidate in the set would have worked
    ceil = np.mean([any(label(sc, g)[0] for g in c) for sc, _, c in scenes])
    record("3_compare", "ceiling (some candidate holds)", round(float(ceil), 4))

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ks = list(rates)
    ax.barh(ks, [100 * rates[k] for k in ks],
            color=[COLORS[6], COLORS[4], COLORS[5], COLORS[0], COLORS[2]])
    ax.axvline(100 * ceil, color="#42505e", ls="--", lw=1.2)
    ax.text(100 * ceil - 1, -0.45, "ceiling", ha="right", fontsize=8)
    ax.set_xlabel("top-1 grasp success (%)")
    ax.set_title(f"one pick per scene, {1000 * NOISE:.1f} mm depth noise")
    save(fig, os.path.join(OUT, "compare.png"))
    state["net"] = net
    state["rates"] = rates
    state["Xtr"] = Xtr
    state["ytr"] = ytr


# ---------------------------------------------------------------------------
# 4. the noise sweep
# ---------------------------------------------------------------------------

def exp4_noise(state):
    print("\n[4] the noise sweep")
    levels = [0.0, 0.001, 0.002, 0.004, 0.008]
    curves = {"depth heuristic, sign fixed": [], "analytic on observed depth": [],
              "learned GQ-CNN (trained at 1.5 mm)": [],
              "learned GQ-CNN (retrained per level)": []}
    rnd = np.random.default_rng(11)
    base = state["net"]
    for lv in levels:
        Xtr, ytr, _, _ = build(320, TRAIN_SHAPES, np.random.default_rng(20),
                               lv, DROPOUT * (lv / NOISE if NOISE else 0))
        net_lv, _ = train(Xtr, ytr, epochs=6)
        _, _, _, scenes = build(130, TRAIN_SHAPES, np.random.default_rng(21),
                                lv, DROPOUT * (lv / NOISE if NOISE else 0),
                                keep_scene=True)
        curves["depth heuristic, sign fixed"].append(
            _top1(score_depth_heuristic_fixed, scenes, rnd))
        curves["analytic on observed depth"].append(
            _top1(score_analytic_observed, scenes, rnd))
        curves["learned GQ-CNN (trained at 1.5 mm)"].append(
            _top1(lambda d, c: score(base, np.stack([patch(d, g) for g in c])),
                  scenes, rnd))
        curves["learned GQ-CNN (retrained per level)"].append(
            _top1(lambda d, c: score(net_lv, np.stack([patch(d, g) for g in c])),
                  scenes, rnd))
        print(f"    noise {1000 * lv:.1f} mm done")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, v in curves.items():
        ax.plot([1000 * x for x in levels], [100 * y for y in v], "o-", ms=4, label=k)
    ax.set_xlabel("depth noise (mm, doubled at object edges)")
    ax.set_ylabel("top-1 grasp success (%)")
    ax.legend(fontsize=8)
    ax.set_title("computing the geometry vs learning from it")
    save(fig, os.path.join(OUT, "noise.png"))
    for k, v in curves.items():
        for lv, y in zip(levels, v):
            record("4_noise", f"{k} @ {1000 * lv:.1f} mm", round(float(y), 4))
    state["noise_curves"] = (levels, curves)


# ---------------------------------------------------------------------------
# 5. novel objects
# ---------------------------------------------------------------------------

def exp5_novel(state):
    print("\n[5] shapes the network has never seen")
    rnd = np.random.default_rng(31)
    net = state["net"]
    rows = {}
    for tag, shapes in (("training shapes", TRAIN_SHAPES),
                        ("novel shapes", TEST_SHAPES)):
        _, _, _, scenes = build(180, shapes, np.random.default_rng(32),
                                NOISE, DROPOUT, keep_scene=True)
        rows[tag] = {
            "depth heuristic, sign fixed": _top1(score_depth_heuristic_fixed, scenes, rnd),
            "analytic on observed depth": _top1(score_analytic_observed, scenes, rnd),
            "learned GQ-CNN": _top1(
                lambda d, c: score(net, np.stack([patch(d, g) for g in c])),
                scenes, rnd),
        }
        ceil = float(np.mean([any(label(sc, g)[0] for g in c) for sc, _, c in scenes]))
        rows[tag]["ceiling"] = ceil
        for k, v in rows[tag].items():
            record("5_novel", f"{tag}: {k}", round(float(v), 4))
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    keys = list(rows["training shapes"])
    x = np.arange(len(keys))
    for i, tag in enumerate(rows):
        ax.bar(x + (i - 0.5) * 0.36, [100 * rows[tag][k] for k in keys],
               width=0.34, label=tag, color=COLORS[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["depth heuristic\n(sign fixed)", "analytic on\nobserved depth",
                        "learned\nGQ-CNN", "ceiling"], fontsize=8)
    ax.set_ylabel("top-1 success (%)")
    ax.legend(fontsize=8)
    ax.set_title("does the network generalise, or memorise outlines?")
    save(fig, os.path.join(OUT, "novel.png"))


# ---------------------------------------------------------------------------
# 6. data efficiency and what the net looks at
# ---------------------------------------------------------------------------

def exp6_data(state):
    print("\n[6] how much data, and what it keyed on")
    Xtr, ytr = state["Xtr"], state["ytr"]
    _, _, _, scenes = build(130, TRAIN_SHAPES, np.random.default_rng(41),
                            NOISE, DROPOUT, keep_scene=True)
    rnd = np.random.default_rng(43)
    sizes = [1500, 6000, 20000, len(Xtr)]
    accs = []
    for n in sizes:
        net_n, _ = train(Xtr[:n], ytr[:n], epochs=6, seed=1)
        accs.append(_top1(lambda d, c: score(net_n, np.stack([patch(d, g) for g in c])),
                          scenes, rnd))
        record("6_data", f"{n} patches: top-1 success", round(float(accs[-1]), 4))

    # saliency: how much does the score move when each pixel is blanked?
    net = state["net"]
    pos = np.flatnonzero(ytr)[:200]
    P = Xtr[pos]
    base = score(net, P)
    sal = np.zeros((32, 32))
    for i in range(0, 32, 2):
        for j in range(0, 32, 2):
            Q = P.copy()
            Q[:, i:i + 2, j:j + 2] = 0.0
            sal[i:i + 2, j:j + 2] = np.abs(score(net, Q) - base).mean()

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    axes[0].plot(sizes, [100 * a for a in accs], "o-", ms=4)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training patches")
    axes[0].set_ylabel("top-1 success (%)")
    axes[0].set_title("data efficiency")
    im = axes[1].imshow(P.mean(0), cmap="magma", origin="lower")
    axes[1].set_title("mean patch of a grasp that holds", fontsize=9)
    axes[1].set_xticks([]); axes[1].set_yticks([]); axes[1].grid(False)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    im = axes[2].imshow(sal, cmap="viridis", origin="lower")
    axes[2].set_title("where blanking 2x2 pixels\nchanges the score most", fontsize=9)
    axes[2].set_xticks([]); axes[2].set_yticks([]); axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    save(fig, os.path.join(OUT, "data.png"))
    # is the important region the finger sites (left/right edges) or the centre?
    mid = sal[10:22, 10:22].mean()
    ends = 0.5 * (sal[:, :8].mean() + sal[:, 24:].mean())
    record("6_data", "saliency at the fingertip sites", round(float(ends), 5))
    record("6_data", "saliency at the patch centre", round(float(mid), 5))
    record("6_data", "fingertip / centre", round(float(ends / max(mid, 1e-9)), 2))


def main():
    use_style()
    t0 = time.time()
    state = {}
    exp1_scene()
    exp23_train_and_compare(state)
    exp4_noise(state)
    exp5_novel(state)
    exp6_data(state)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "metric", "value"])
        w.writerows(RESULTS)
    print(f"\ndone in {time.time() - t0:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
