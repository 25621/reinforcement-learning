"""Project 23 -- "Pick up the red mug": language in, gripper pose out.

Six experiments:

  1. the whole pipeline on one scene, drawn
  2. what CLIP can and cannot ground (the answer is not "objects vs not")
  3. how you crop the region matters more than the model does
  4. prompt wording
  5. centroid grasp vs antipodal search, per object shape
  6. end to end, with the failures attributed to the stage that caused them

Runs in about five minutes on a CPU, most of it CLIP.
"""

import csv
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
for _p in ("16-camera-calibration", "22-object-6-dof-pose", "01-transform-calculator"):
    sys.path.insert(0, os.path.join(_PROJ, _p))

from tabletop import (CAM, CATALOG, COLOR_NAME, GRIPPER_MAX_WIDTH, TABLE_Z,  # noqa: E402
                      deproject, random_scene, render_scene)
from grasp import (best_grasp, draw_grasp, evaluate_grasp,                   # noqa: E402
                   principal_axis_grasp, segment_by_depth)
from ground import crop_for, encode_images, encode_texts, ground             # noqa: E402
from plot_style import COLORS, use_style                                     # noqa: E402

import matplotlib.pyplot as plt                                              # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []
N_SCENES = 24


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def build_scenes(n=N_SCENES, seed=0, n_obj=4, spread=1.0):
    """Render n random tabletops and segment each one."""
    scenes = []
    rng = np.random.default_rng(seed)
    for i in range(n):
        items = random_scene(rng, n=n_obj)
        if spread != 1.0:                       # push everything toward the middle
            items = [(nm, x * spread, y * spread, a) for nm, x, y, a in items]
        img, depth, ids = render_scene(items, rng=rng)
        masks = segment_by_depth(depth, TABLE_Z)
        pts = deproject(depth)
        # which true object does each detected blob belong to?
        owner = []
        for m in masks:
            v, c = np.unique(ids[m], return_counts=True)
            owner.append(int(v[np.argmax(c)]))
        scenes.append(dict(items=items, img=img, depth=depth, ids=ids,
                           masks=masks, pts=pts, owner=owner))
    return scenes


def phrase_for(name, kind="color_noun"):
    c = COLOR_NAME[name]
    return {"color_noun": f"the {c} {name}",
            "noun": f"the {name}",
            "color": f"the {c} object"}[kind]


# --------------------------------------------------------------------------
# 1. the whole pipeline
# --------------------------------------------------------------------------

def stage_demo(scenes):
    print("\n[1] the pipeline on one scene")
    sc = scenes[0]
    names = [it[0] for it in sc["items"]]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    axes[0].imshow(sc["img"]); axes[0].set_title("RGB, top-down")
    axes[1].imshow(sc["depth"], cmap="viridis")
    axes[1].set_title(f"depth (table at {TABLE_Z:.2f} m)")
    seg = np.zeros_like(sc["img"])
    for k, m in enumerate(sc["masks"]):
        hexcol = str(COLORS[k % len(COLORS)])
        seg[m] = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)])
    axes[2].imshow(seg)
    axes[2].set_title(f"{len(sc['masks'])} blobs standing off the table")

    target = names[0]
    phrase = phrase_for(target)
    idx, scores = ground(sc["img"], sc["masks"], phrase)
    others = np.zeros(sc["masks"][0].shape, bool)
    for k, m in enumerate(sc["masks"]):
        if k != idx:
            others |= m
    g = best_grasp(sc["masks"][idx], sc["pts"], others, GRIPPER_MAX_WIDTH)
    vis = draw_grasp(sc["img"], g)
    axes[3].imshow(vis)
    axes[3].set_title(f'"{phrase}"\n' +
                      ("grasp width %.0f mm" % (g["width"] * 1000) if g else "no grasp found"))
    for a in axes:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipeline.png"))
    plt.close(fig)

    log(dict(stage="demo", objects=len(sc["items"]), blobs=len(sc["masks"]),
             phrase=phrase, grounded_correctly=bool(sc["owner"][idx] ==
                                                    names.index(target) + 1),
             margin=round(float(np.sort(scores)[-1] - np.sort(scores)[-2]), 4),
             grasp_width_mm=round(g["width"] * 1000, 1) if g else None,
             grasp_ok=bool(g["ok"]) if g else False))


# --------------------------------------------------------------------------
# 2. what CLIP can ground
# --------------------------------------------------------------------------

def stage_grounding(scenes):
    print("\n[2] what CLIP can and cannot ground")
    kinds = ["color_noun", "noun", "color"]
    rows = []
    for kind in kinds:
        hit, n, margins = 0, 0, []
        for sc in scenes:
            names = [it[0] for it in sc["items"]]
            for ti, name in enumerate(names):
                idx, s = ground(sc["img"], sc["masks"], phrase_for(name, kind))
                n += 1
                hit += (sc["owner"][idx] == ti + 1)
                margins.append(float(np.sort(s)[-1] - np.sort(s)[-2]))
        rows.append((kind, 100 * hit / n, float(np.mean(margins))))
        log(dict(stage="grounding", phrase_kind=kind, n=n,
                 accuracy_pct=round(rows[-1][1], 1),
                 mean_margin=round(rows[-1][2], 4),
                 chance_pct=round(100 / len(scenes[0]["masks"]), 1)))

    # relational phrases: the target is defined only by where it is
    hit, n = 0, 0
    for sc in scenes:
        names = [it[0] for it in sc["items"]]
        cx = [np.nonzero(m)[1].mean() for m in sc["masks"]]
        order = np.argsort(cx)
        for want, rel in ((order[0], "leftmost"), (order[-1], "rightmost")):
            other = names[sc["owner"][order[len(order) // 2]] - 1]
            phrase = f"the object on the {'left' if rel == 'leftmost' else 'right'} of the {other}"
            idx, s = ground(sc["img"], sc["masks"], phrase)
            n += 1
            hit += (idx == want)
    rows.append(("relational", 100 * hit / n, np.nan))
    log(dict(stage="grounding", phrase_kind="relational (left of / right of)",
             n=n, accuracy_pct=round(100 * hit / n, 1),
             chance_pct=round(100 / len(scenes[0]["masks"]), 1)))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(rows))
    ax.bar(x, [r[1] for r in rows], color=[COLORS[2]] * 3 + [COLORS[1]])
    ax.axhline(100 / len(scenes[0]["masks"]), color=COLORS[6], ls="--", lw=1)
    ax.text(0.05, 100 / len(scenes[0]["masks"]) + 2, "chance", color=COLORS[6], fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['"the red mug"', '"the mug"', '"the red object"',
                        '"left of the box"'], fontsize=7.5)
    ax.set_ylabel("grounding accuracy (%)")
    ax.set_title("CLIP grounds attributes, not relations")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grounding.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 3 + 4. crops and prompts
# --------------------------------------------------------------------------

def stage_crops_prompts(scenes):
    print("\n[3] how you crop the region")
    for mode in ("tight", "pad", "masked", "highlight"):
        hit, n = 0, 0
        for sc in scenes:
            names = [it[0] for it in sc["items"]]
            for ti, name in enumerate(names):
                idx, _ = ground(sc["img"], sc["masks"], phrase_for(name), mode=mode)
                n += 1
                hit += (sc["owner"][idx] == ti + 1)
        log(dict(stage="crop", mode=mode, n=n, accuracy_pct=round(100 * hit / n, 1)))

    print("\n[4] prompt wording")
    for tpl in ("{}", "a photo of {}", "a photo of {} on a table",
                "a close-up photo of {}, a single object on a plain table"):
        hit, n = 0, 0
        for sc in scenes:
            names = [it[0] for it in sc["items"]]
            for ti, name in enumerate(names):
                idx, _ = ground(sc["img"], sc["masks"], phrase_for(name), template=tpl)
                n += 1
                hit += (sc["owner"][idx] == ti + 1)
        log(dict(stage="prompt", template=tpl, n=n, accuracy_pct=round(100 * hit / n, 1)))


# --------------------------------------------------------------------------
# 5. grasp geometry
# --------------------------------------------------------------------------

def stage_grasp(scenes):
    print("\n[5] centroid grasp against antipodal search")
    per_object = {}
    for sc in scenes:
        names = [it[0] for it in sc["items"]]
        for k, m in enumerate(sc["masks"]):
            oid = sc["owner"][k]
            if oid <= 0:
                continue
            name = names[oid - 1]
            others = np.zeros(m.shape, bool)
            for j, mm in enumerate(sc["masks"]):
                if j != k:
                    others |= mm
            c, ang = principal_axis_grasp(m, sc["pts"])
            naive = evaluate_grasp(c, ang, m, sc["pts"], others, GRIPPER_MAX_WIDTH)
            search = best_grasp(m, sc["pts"], others, GRIPPER_MAX_WIDTH)
            d = per_object.setdefault(name, dict(n=0, naive=0, search=0,
                                                 naive_wide=0, naive_offobj=0,
                                                 naive_angle=0))
            d["n"] += 1
            d["naive"] += bool(naive and naive["ok"])
            d["search"] += bool(search and search["ok"])
            if naive and not naive["ok"]:
                d["naive_wide"] += naive["width"] > GRIPPER_MAX_WIDTH
                d["naive_offobj"] += naive["on_object"] <= 0.97
                d["naive_angle"] += naive["normal_angle"] > 25.0
    for name, d in sorted(per_object.items()):
        log(dict(stage="grasp", object=f"{name} ({COLOR_NAME[name]})", n=d["n"],
                 naive_ok_pct=round(100 * d["naive"] / d["n"], 1),
                 search_ok_pct=round(100 * d["search"] / d["n"], 1),
                 naive_fail_too_wide=d["naive_wide"],
                 naive_fail_off_object=d["naive_offobj"],
                 naive_fail_bad_normals=d["naive_angle"]))

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    names = sorted(per_object)
    x = np.arange(len(names))
    ax.bar(x - 0.2, [100 * per_object[n]["naive"] / per_object[n]["n"] for n in names],
           0.4, color=COLORS[1], label="centroid + narrow axis")
    ax.bar(x + 0.2, [100 * per_object[n]["search"] / per_object[n]["n"] for n in names],
           0.4, color=COLORS[2], label="antipodal search")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("legal grasps found (%)"); ax.legend(fontsize=8)
    ax.set_title("one guess against a few hundred")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grasp.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. end to end
# --------------------------------------------------------------------------

def stage_end_to_end(scenes, tight_scenes):
    print("\n[6] end to end, with the blame apportioned")
    for tag, scs in (("spread out", scenes), ("pushed together", tight_scenes)):
        n = seg_ok = ground_ok = grasp_ok = 0
        for sc in scs:
            names = [it[0] for it in sc["items"]]
            for ti, name in enumerate(names):
                n += 1
                # did segmentation give this object a blob of its own?
                own = [k for k, o in enumerate(sc["owner"]) if o == ti + 1]
                if not own:
                    continue
                mine = own[0]
                if not (sc["ids"] == ti + 1).sum() > 200:
                    continue
                iou = ((sc["masks"][mine]) & (sc["ids"] == ti + 1)).sum() / \
                    max(((sc["masks"][mine]) | (sc["ids"] == ti + 1)).sum(), 1)
                if iou < 0.6:
                    continue
                seg_ok += 1
                idx, _ = ground(sc["img"], sc["masks"], phrase_for(name))
                if idx != mine:
                    continue
                ground_ok += 1
                others = np.zeros(sc["masks"][0].shape, bool)
                for j, mm in enumerate(sc["masks"]):
                    if j != idx:
                        others |= mm
                g = best_grasp(sc["masks"][idx], sc["pts"], others, GRIPPER_MAX_WIDTH)
                grasp_ok += bool(g and g["ok"])
        log(dict(stage="end_to_end", scenes=tag, requests=n,
                 segmentation_ok_pct=round(100 * seg_ok / n, 1),
                 and_grounded_ok_pct=round(100 * ground_ok / n, 1),
                 and_grasp_ok_pct=round(100 * grasp_ok / n, 1)))


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    print("loading CLIP...", flush=True)
    from ground import load
    load()
    print(f"    {time.time() - t0:.0f} s", flush=True)

    scenes = build_scenes()
    tight = build_scenes(n=N_SCENES, seed=5, n_obj=5, spread=0.45)
    print(f"    scenes ready ({time.time() - t0:.0f} s)", flush=True)
    stage_demo(scenes)
    stage_grounding(scenes)
    stage_crops_prompts(scenes)
    stage_grasp(scenes)
    stage_end_to_end(scenes, tight)

    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
