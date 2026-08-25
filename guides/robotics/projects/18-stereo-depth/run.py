"""Project 18 -- Stereo depth: disparity, and everything that ruins it.

Seven experiments:

  1. a stereo pair -> disparity -> depth -> point cloud, scored against truth
  2. rectification: what skipping it costs, and what 0.3 deg of misalignment costs
  3. depth error vs distance -- the z-squared law, measured
  4. baseline: bigger is more accurate and sees less
  5. texture: the one thing block matching cannot work without
  6. window size: noise against edge fattening
  7. our matcher vs OpenCV's semi-global matcher

Runs in about four minutes on a CPU.
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
for _p in ("16-camera-calibration", "01-transform-calculator"):
    sys.path.insert(0, os.path.join(_PROJ, _p))

np.seterr(invalid="ignore", divide="ignore")

from camera import Camera, default_camera, rodrigues                  # noqa: E402
from render import Plane, render, speckle_texture                     # noqa: E402
from stereo import (StereoRig, block_match, build_scene, depth_to_cloud,  # noqa: E402
                    rectify_maps)
from plot_style import COLORS, use_style                              # noqa: E402

import matplotlib.pyplot as plt                                       # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def rectify(cam, img, nearest=False):
    mx, my = rectify_maps(cam)
    return cv2.remap(img, mx, my, cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def robust_std(x):
    """Spread of a noisy quantity, ignoring outliers.

    Standard deviation is useless here: a handful of grossly wrong matches
    drag it up by an order of magnitude and hide the real precision.  The
    median absolute deviation, scaled by 1.4826 so it equals the standard
    deviation for clean Gaussian noise, measures the bulk of the data.
    """
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def score(rig, disp, Z_true):
    """The three numbers that describe a depth map.

    - density: what fraction of pixels got an answer at all
    - median depth error: the typical error, immune to the wild outliers
      that a handful of bad matches always produce
    - bad-1px: the fraction of answers off by more than one pixel of
      disparity.  This is the standard stereo metric, and it is stated in
      disparity rather than metres precisely because one pixel of disparity
      means a very different number of millimetres near and far.
    """
    Z = rig.disparity_to_depth(disp)
    ok = np.isfinite(disp) & np.isfinite(Z) & np.isfinite(Z_true) & (Z_true > 0)
    if ok.sum() < 50:
        return dict(density=0.0, median_mm=None, bad1_pct=None)
    d_true = rig.depth_to_disparity(Z_true)
    err_mm = np.abs(Z[ok] - Z_true[ok]) * 1000
    bad = np.abs(disp[ok] - d_true[ok]) > 1.0
    return dict(density=float(np.isfinite(disp)[np.isfinite(Z_true)].mean()),
                median_mm=float(np.median(err_mm)),
                bad1_pct=float(100 * bad.mean()))


# --------------------------------------------------------------------------
# 1. the pipeline, end to end
# --------------------------------------------------------------------------

def stage_pipeline():
    print("\n[1] stereo pair -> disparity -> depth -> point cloud")
    rig = StereoRig(baseline=0.09)
    planes, _ = build_scene(blank_poster=True)
    t0 = time.time()
    L, R, ZL, _ = rig.render(planes)
    Lr, Rr = rectify(rig.cam, L), rectify(rig.cam, R)
    Zr = rectify(rig.cam, np.where(np.isfinite(ZL), ZL, 0).astype(np.float32), nearest=True)
    Zr[Zr <= 0] = np.nan
    disp, valid = block_match(Lr, Rr, max_disp=96, win=9, uniqueness=0.05)
    print(f"    render + match: {time.time() - t0:.1f} s")
    s = score(rig, disp, Zr)
    log(dict(stage="pipeline", baseline_cm=9, win=9, **{k: (round(v, 3) if v else v)
                                                        for k, v in s.items()}))

    Z = rig.disparity_to_depth(disp)
    pts, col = depth_to_cloud(rig.cam, Z, Lr, step=3)

    fig = plt.figure(figsize=(12, 6.4))
    ax = fig.add_subplot(2, 3, 1); ax.imshow(Lr); ax.set_title("left image (rectified)")
    ax = fig.add_subplot(2, 3, 2); ax.imshow(Rr); ax.set_title("right image (rectified)")
    ax = fig.add_subplot(2, 3, 3)
    im = ax.imshow(disp, cmap="magma", vmin=0, vmax=96)
    ax.set_title("disparity (px)"); plt.colorbar(im, ax=ax, fraction=0.035)
    ax = fig.add_subplot(2, 3, 4)
    im = ax.imshow(np.clip(Z, 0, 5), cmap="viridis")
    ax.set_title("depth (m)"); plt.colorbar(im, ax=ax, fraction=0.035)
    ax = fig.add_subplot(2, 3, 5)
    e = np.abs(Z - Zr) * 1000
    im = ax.imshow(np.clip(e, 0, 200), cmap="inferno")
    ax.set_title("|depth error| (mm, clipped at 200)")
    plt.colorbar(im, ax=ax, fraction=0.035)
    ax = fig.add_subplot(2, 3, 6, projection="3d")
    keep = pts[:, 2] < 4.6                      # drop the handful of wild outliers
    pts, col = pts[keep], col[keep]
    sub = np.random.default_rng(0).choice(len(pts), min(9000, len(pts)), replace=False)
    ax.scatter(pts[sub, 0], pts[sub, 2], -pts[sub, 1], c=col[sub] / 255.0, s=1)
    ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)"); ax.set_zlabel("-y (m)")
    ax.set_zlim(-1.6, 1.6); ax.set_xlim(-2.0, 2.0)
    ax.set_title("the point cloud"); ax.view_init(elev=22, azim=-80)
    for a in fig.axes[:5]:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pipeline.png"))
    plt.close(fig)
    return rig, planes, Lr, Rr, Zr


# --------------------------------------------------------------------------
# 2. rectification
# --------------------------------------------------------------------------

def stage_rectification(rig, planes):
    print("\n[2] rectification, and small misalignments")
    L, R, ZL, _ = rig.render(planes)
    Zr = rectify(rig.cam, np.where(np.isfinite(ZL), ZL, 0).astype(np.float32), nearest=True)
    Zr[Zr <= 0] = np.nan
    Lr, Rr = rectify(rig.cam, L), rectify(rig.cam, R)

    # (a) skipping undistortion entirely
    disp_raw, _ = block_match(L, R, max_disp=96, win=9, uniqueness=0.05)
    Zraw = np.where(np.isfinite(ZL), ZL, np.nan)
    log(dict(stage="rectify", case="raw (distorted) images",
             **{k: (round(v, 3) if v else v) for k, v in score(rig, disp_raw, Zraw).items()}))
    disp, _ = block_match(Lr, Rr, max_disp=96, win=9, uniqueness=0.05)
    log(dict(stage="rectify", case="undistorted",
             **{k: (round(v, 3) if v else v) for k, v in score(rig, disp, Zr).items()}))

    # (b) the right camera very slightly rotated
    rows = []
    for tilt in (0.0, 0.1, 0.3, 0.6, 1.2):
        R_wc = rodrigues([np.radians(tilt), 0, 0])       # pitch: moves rows
        Rimg, _, _ = render(rig.cam, planes, R_wc,
                            np.array([rig.baseline, 0, 0]), supersample=2, seed=2, noise=1.5)
        d2, _ = block_match(Lr, rectify(rig.cam, Rimg), max_disp=96, win=9, uniqueness=0.05)
        s = score(rig, d2, Zr)
        rows.append((tilt, s["density"], s["median_mm"] or np.nan, s["bad1_pct"] or np.nan))
        log(dict(stage="misalign", pitch_deg=tilt, rows_off_px=round(tilt * np.pi / 180 * rig.cam.fy, 2),
                 **{k: (round(v, 3) if v else v) for k, v in s.items()}))
    a = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    ax[0].plot(a[:, 0], 100 * a[:, 1], "o-", color=COLORS[0])
    ax[0].set_xlabel("right-camera pitch error (deg)"); ax[0].set_ylabel("pixels with an answer (%)")
    ax[0].set_title("a matcher that searches rows\ncannot find a point that moved off its row")
    ax[1].plot(a[:, 0], a[:, 3], "o-", color=COLORS[1])
    ax[1].set_xlabel("right-camera pitch error (deg)"); ax[1].set_ylabel("bad-1px (%)")
    ax[1].set_title("and the answers it does find get worse")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rectification.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. depth error vs distance
# --------------------------------------------------------------------------

def flat_scene(z, seed=7, scale=3):
    """One textured wall filling the view at distance z."""
    w = 3.0 * z
    return [Plane([-w / 2, -w / 2, z], [w, 0, 0], [0, w, 0],
                  speckle_texture(700, 700, seed=seed, scale=scale), "wall")]


def stage_distance(rig):
    print("\n[3] depth error vs distance")
    rows = []
    for z in (0.4, 0.6, 0.9, 1.3, 1.8, 2.5, 3.5, 5.0):
        L, R, _, _ = rig.render(flat_scene(z))
        Lr, Rr = rectify(rig.cam, L), rectify(rig.cam, R)
        d_true = rig.depth_to_disparity(np.array(z))
        # search far enough to contain the answer: a matcher asked for at most
        # 110 px of disparity simply cannot find a surface at 0.4 m, which
        # needs 122 px.  That is not noise, it is a range limit.
        md = int(np.ceil(float(d_true))) + 12
        disp, _ = block_match(Lr, Rr, max_disp=md, win=9, uniqueness=0.05)
        Z = rig.disparity_to_depth(disp)
        ok = np.isfinite(disp) & np.isfinite(Z)
        bias = float(np.median(disp[ok])) - float(d_true)
        rows.append((z, float(d_true), robust_std(disp[ok] - d_true),
                     float(np.median(np.abs(Z[ok] - z))) * 1000, float(ok.mean()),
                     bias, robust_std(Z[ok]) * 1000))
        log(dict(stage="distance", z_m=z, true_disp_px=round(rows[-1][1], 2),
                 true_disp_frac=round(float(d_true) % 1.0, 3),
                 disp_bias_px=round(bias, 4),
                 disp_noise_px=round(rows[-1][2], 4),
                 depth_noise_mm=round(rows[-1][6], 2),
                 median_depth_err_mm=round(rows[-1][3], 2),
                 rel_pct=round(100 * rows[-1][3] / (z * 1000), 3),
                 density=round(rows[-1][4], 3)))
    a = np.array(rows)
    # fit the exponent of  error ~ z^p  in log space, for the NOISE (which the
    # theory describes) and for the total median error (which also carries the
    # pixel-locking bias measured above)
    p_noise = np.polyfit(np.log(a[:, 0]), np.log(a[:, 6]), 1)[0]
    p_total = np.polyfit(np.log(a[:, 0]), np.log(a[:, 3]), 1)[0]
    sd = float(np.median(a[:, 2]))
    pred = a[:, 0] ** 2 * sd / (rig.cam.fx * rig.baseline) * 1000
    log(dict(stage="distance_fit", exponent_of_noise=round(float(p_noise), 3),
             exponent_of_total_error=round(float(p_total), 3),
             median_disp_noise_px=round(sd, 4),
             theory_noise_at_5m_mm=round(float(pred[-1]), 1),
             measured_noise_at_5m_mm=round(float(a[-1, 6]), 1),
             measured_total_at_5m_mm=round(float(a[-1, 3]), 1)))

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.2))
    ax[0].loglog(a[:, 0], a[:, 6], "o-", color=COLORS[0], label="measured spread")
    ax[0].loglog(a[:, 0], pred, "--", color=COLORS[2],
                 label=r"theory: $Z^2 \sigma_d / (f B)$")
    ax[0].loglog(a[:, 0], a[:, 3], "s:", color=COLORS[1], label="median |error| (spread + bias)")
    ax[0].set_xlabel("distance (m)"); ax[0].set_ylabel("depth error (mm)")
    ax[0].set_title(f"noise grows as $Z^{{{p_noise:.2f}}}$"); ax[0].legend(fontsize=7)
    ax[1].semilogx(a[:, 0], a[:, 1], "o-", color=COLORS[1])
    ax[1].set_xlabel("distance (m)"); ax[1].set_ylabel("disparity (px)")
    ax[1].set_title("what the camera actually measures")
    ax[2].plot(a[:, 1] % 1.0, a[:, 5], "o", color=COLORS[3], ms=7)
    ax[2].axhline(0, color=COLORS[6], lw=1)
    # a guide line with the SHAPE of a pull toward the nearest whole pixel,
    # scaled to the size of the effect we actually measured (drawing it at
    # full pixel amplitude would dwarf the data and hide the pattern)
    amp = float(np.max(np.abs(a[:, 5])))
    ff = np.linspace(0, 1, 200)
    ax[2].plot(ff, -amp * np.sin(2 * np.pi * ff), "--", color=COLORS[6], lw=1,
               label="shape of a pull toward\nthe nearest whole pixel")
    ax[2].set_xlabel("fractional part of the true disparity")
    ax[2].set_ylabel("disparity bias (px)")
    ax[2].set_title("pixel locking"); ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "distance.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. baseline
# --------------------------------------------------------------------------

def stage_baseline():
    print("\n[4] baseline: accuracy against how much you can see")
    planes, _ = build_scene(blank_poster=False)
    rows = []
    for b in (0.03, 0.06, 0.09, 0.15, 0.25):
        rig = StereoRig(baseline=b)
        L, R, ZL, _ = rig.render(planes)
        Lr, Rr = rectify(rig.cam, L), rectify(rig.cam, R)
        Zr = rectify(rig.cam, np.where(np.isfinite(ZL), ZL, 0).astype(np.float32), nearest=True)
        Zr[Zr <= 0] = np.nan
        md = int(np.ceil(rig.depth_to_disparity(np.array(0.5))))
        disp, _ = block_match(Lr, Rr, max_disp=min(md, 220), win=9, uniqueness=0.05)
        s = score(rig, disp, Zr)
        # far accuracy specifically: the 4 m back wall
        Z = rig.disparity_to_depth(disp)
        far = np.isfinite(Zr) & (np.abs(Zr - 4.0) < 0.01) & np.isfinite(Z)
        far_mm = float(np.median(np.abs(Z[far] - 4.0))) * 1000 if far.sum() > 100 else np.nan
        rows.append((b, s["density"], s["median_mm"], far_mm, md))
        log(dict(stage="baseline", baseline_cm=round(b * 100, 1), max_disp_searched=md,
                 density=round(s["density"], 3), median_mm=round(s["median_mm"], 2),
                 far_wall_err_mm=round(far_mm, 1)))
    a = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(a[:, 0] * 100, a[:, 3], "o-", color=COLORS[0])
    ax[0].set_xlabel("baseline (cm)"); ax[0].set_ylabel("error on the 4 m wall (mm)")
    ax[0].set_title("a wider baseline sees depth better")
    ax[1].plot(a[:, 0] * 100, 100 * a[:, 1], "o-", color=COLORS[1])
    ax[1].set_xlabel("baseline (cm)"); ax[1].set_ylabel("pixels with an answer (%)")
    ax[1].set_title("and sees less of the scene")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "baseline.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. texture
# --------------------------------------------------------------------------

def stage_texture(rig):
    print("\n[5] texture, the thing block matching cannot do without")
    rows = []
    for contrast in (0, 4, 10, 25, 60, 120, 210):
        lo, hi = 128 - contrast / 2, 128 + contrast / 2
        L, R, _, _ = rig.render([Plane([-1.5, -1.5, 1.0], [3, 0, 0], [0, 3, 0],
                                       speckle_texture(700, 700, seed=9, scale=3,
                                                       lo=lo, hi=hi), "wall")])
        Lr, Rr = rectify(rig.cam, L), rectify(rig.cam, R)
        disp, _ = block_match(Lr, Rr, max_disp=96, win=9, uniqueness=0.05)
        ok = np.isfinite(disp)
        d_true = float(rig.depth_to_disparity(np.array(1.0)))
        rows.append((contrast, float(ok.mean()),
                     float(np.median(np.abs(disp[ok] - d_true))) if ok.sum() > 50 else np.nan))
        log(dict(stage="texture", contrast_levels=contrast, density=round(rows[-1][1], 3),
                 median_disp_err_px=round(rows[-1][2], 3)))

    # what the cost curve looks like with and without texture
    L, R, _, _ = rig.render([Plane([-1.5, -1.5, 1.0], [3, 0, 0], [0, 3, 0],
                                   np.full((64, 64, 3), 170, np.uint8), "blank")])
    _, _, cost_blank = block_match(rectify(rig.cam, L), rectify(rig.cam, R),
                                   max_disp=96, win=9, return_cost=True)
    L, R, _, _ = rig.render([Plane([-1.5, -1.5, 1.0], [3, 0, 0], [0, 3, 0],
                                   speckle_texture(700, 700, seed=9, scale=3), "rich")])
    _, _, cost_rich = block_match(rectify(rig.cam, L), rectify(rig.cam, R),
                                  max_disp=96, win=9, return_cost=True)
    yx = (240, 400)
    cb = cost_blank[:, yx[0], yx[1]]
    cr = cost_rich[:, yx[0], yx[1]]
    log(dict(stage="cost_curve", blank_min_over_mean=round(float(np.nanmin(cb) /
                                                                np.nanmean(cb[np.isfinite(cb)])), 4),
             textured_min_over_mean=round(float(np.nanmin(cr) /
                                                np.nanmean(cr[np.isfinite(cr)])), 4)))

    a = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
    ax[0].plot(a[:, 0], 100 * a[:, 1], "o-", color=COLORS[0])
    ax[0].set_xlabel("texture contrast (grey levels, peak to peak)")
    ax[0].set_ylabel("pixels with an answer (%)")
    ax[0].set_title("no texture, no depth")
    ax[1].plot(np.arange(len(cr)), cr / np.nanmax(cr[np.isfinite(cr)]), color=COLORS[0],
               label="textured surface")
    ax[1].plot(np.arange(len(cb)), cb / np.nanmax(cb[np.isfinite(cb)]), color=COLORS[1],
               label="blank surface")
    ax[1].axvline(float(rig.depth_to_disparity(np.array(1.0))), color=COLORS[6], ls="--", lw=1)
    ax[1].set_xlabel("candidate disparity (px)"); ax[1].set_ylabel("match cost (normalized)")
    ax[1].set_title("the matcher's view of one pixel"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "texture.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. window size
# --------------------------------------------------------------------------

def stage_window(rig, Lr, Rr, Zr):
    print("\n[6] window size: noise against edge fattening")
    # pixels near a depth discontinuity, found from the ground-truth depth
    grad = np.abs(np.gradient(np.nan_to_num(Zr, nan=0.0))[0]) + \
        np.abs(np.gradient(np.nan_to_num(Zr, nan=0.0))[1])
    near_edge = cv2.dilate((grad > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    rows = []
    for win in (3, 5, 9, 15, 21, 31):
        disp, _ = block_match(Lr, Rr, max_disp=96, win=win, uniqueness=0.05)
        Z = rig.disparity_to_depth(disp)
        ok = np.isfinite(disp) & np.isfinite(Z) & np.isfinite(Zr)
        # Score in DISPARITY pixels, not millimetres: the scene spans 0.55 m
        # to 4 m, and a millimetre error means something completely different
        # at each end, so a depth average would just report which surface has
        # the most pixels.
        d_true = rig.depth_to_disparity(Zr)
        flat = ok & ~near_edge
        edge = ok & near_edge
        rows.append((win,
                     float(np.median(np.abs(disp[flat] - d_true[flat]))),
                     float(np.median(np.abs(disp[edge] - d_true[edge]))),
                     float(np.isfinite(disp).mean()),
                     float(100 * (np.abs(disp[edge] - d_true[edge]) > 1).mean())))
        log(dict(stage="window", win=win, flat_disp_err_px=round(rows[-1][1], 4),
                 edge_disp_err_px=round(rows[-1][2], 4),
                 edge_bad1_pct=round(rows[-1][4], 2), density=round(rows[-1][3], 3)))
    a = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(a[:, 0], a[:, 1], "o-", color=COLORS[0], label="away from depth edges")
    ax.plot(a[:, 0], a[:, 2], "s-", color=COLORS[1], label="within 9 px of a depth edge")
    ax.set_xlabel("matching window (px)"); ax.set_ylabel("median disparity error (px)")
    ax.set_yscale("log"); ax.legend(fontsize=8)
    ax.set_title("the window-size trade-off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "window.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. against OpenCV's semi-global matcher
# --------------------------------------------------------------------------

def stage_sgbm(rig, Lr, Rr, Zr):
    print("\n[7] our block matcher vs OpenCV's semi-global matcher")
    gl = cv2.cvtColor(Lr, cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(Rr, cv2.COLOR_RGB2GRAY)
    for name, disp in [
        ("ours (SAD 9x9)", block_match(Lr, Rr, max_disp=96, win=9, uniqueness=0.05)[0]),
        ("OpenCV StereoBM", None),
        ("OpenCV StereoSGBM", None),
    ]:
        if name == "OpenCV StereoBM":
            m = cv2.StereoBM.create(numDisparities=96, blockSize=9)
            disp = m.compute(gl, gr).astype(np.float32) / 16.0
            disp[disp <= 0] = np.nan
        elif name == "OpenCV StereoSGBM":
            m = cv2.StereoSGBM.create(minDisparity=0, numDisparities=96, blockSize=5,
                                      P1=8 * 25, P2=32 * 25, uniquenessRatio=5,
                                      speckleWindowSize=100, speckleRange=2)
            disp = m.compute(gl, gr).astype(np.float32) / 16.0
            disp[disp <= 0] = np.nan
        t0 = time.time()
        s = score(rig, disp, Zr)
        log(dict(stage="vs_opencv", matcher=name, density=round(s["density"], 3),
                 median_mm=round(s["median_mm"], 2), bad1_pct=round(s["bad1_pct"], 2)))


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    rig, planes, Lr, Rr, Zr = stage_pipeline()
    stage_rectification(rig, planes)
    stage_distance(rig)
    stage_baseline()
    stage_texture(rig)
    stage_window(rig, Lr, Rr, Zr)
    stage_sgbm(rig, Lr, Rr, Zr)

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
