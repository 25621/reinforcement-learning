"""Project 17 -- AprilTag pose, and the second answer that fits just as well.

Seven experiments:

  1. render tags, detect them, solve the pose, draw the axes back on
  2. how accurate is it, as a function of distance
  3. the planar pose ambiguity: two poses, one image
  4. two cures -- reject ambiguous detections, or use a plate of four tags
  5. a tag measured wrong with a ruler
  6. a camera calibrated wrong (project 16's fronto-parallel trap, reused)
  7. subpixel corner refinement, on and off

Runs in about three minutes on a CPU.
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

from camera import Camera, default_camera, rot_angle_deg, rodrigues     # noqa: E402
from render import render                                               # noqa: E402
from tags import (TAG_SIZE, tag_object_points, tag_plane, detect_tags,  # noqa: E402
                  draw_axes, tag_board, pose_from_angles)
from pnp import pnp_planar, pnp_multi                                    # noqa: E402
from plot_style import COLORS, use_style                                 # noqa: E402

import matplotlib.pyplot as plt                                          # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
CAM = default_camera()
OBJ = tag_object_points()
RESULTS = []


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def render_tag(R_ct, t_ct, tag_id=7, quiet=1.0, noise=2.0, ss=2, cam=CAM):
    """Render one tag whose pose in the CAMERA frame is (R_ct, t_ct)."""
    R_wc, t_wc = R_ct.T, -R_ct.T @ t_ct        # put the tag at the world origin
    plane = tag_plane(tag_id, np.eye(3), np.zeros(3), quiet=quiet)
    img, _, _ = render(cam, [plane], R_wc, t_wc, supersample=ss, seed=0, noise=noise)
    return img


def synth_corners(R, t, rng, noise=0.15, obj=OBJ, cam=CAM):
    return cam.project(obj @ R.T + t) + rng.normal(0, noise, (len(obj), 2))


def crop_around(img, uv, pad=70):
    """Zoom in on the tag so the figure shows something bigger than a stamp."""
    x0 = max(int(uv[:, 0].min()) - pad, 0); x1 = min(int(uv[:, 0].max()) + pad, img.shape[1])
    y0 = max(int(uv[:, 1].min()) - pad, 0); y1 = min(int(uv[:, 1].max()) + pad, img.shape[0])
    return img[y0:y1, x0:x1]


def pose_error(R, t, R_true, t_true):
    return (rot_angle_deg(R_true, R),
            float(np.linalg.norm(t - t_true)) * 1000.0,
            float(t[2] - t_true[2]) * 1000.0)


# --------------------------------------------------------------------------
# 1. detect, solve, draw
# --------------------------------------------------------------------------

def stage_detect():
    print("\n[1] detect a tag, solve its pose, project the axes back")
    poses = [pose_from_angles(0.45, 35, 20, 10), pose_from_angles(0.30, 55, -60, 0),
             pose_from_angles(0.80, 15, 120, 30), pose_from_angles(0.55, 45, 200, -20)]
    imgs, rows = [], []
    t0 = time.time()
    for i, (R, t) in enumerate(poses):
        img = render_tag(R, t, tag_id=7 + i)
        dets = detect_tags(img)
        if not dets:
            rows.append(None)
            imgs.append(img)
            continue
        uv = list(dets.values())[0]
        corner_err = float(np.mean(np.linalg.norm(uv - CAM.project(OBJ @ R.T + t), axis=1)))
        sols = pnp_planar(CAM, OBJ, uv)
        Rh, th, rms = sols[0]
        ang, terr, zerr = pose_error(Rh, th, R, t)
        # OpenCV, on the same corners, as an independent check.  IPPE is the
        # planar solver; it also returns two solutions, so we compare its
        # best-fitting one with ours.
        n, rvs, tvs, errs = cv2.solvePnPGeneric(
            OBJ.astype(np.float32), uv.astype(np.float32), CAM.K,
            CAM.dist.reshape(1, 5), flags=cv2.SOLVEPNP_IPPE)
        best = int(np.argmin(np.asarray(errs).reshape(-1)))
        d_cv = rot_angle_deg(rodrigues(rvs[best].reshape(3)), Rh)
        log(dict(stage="detect", pose=i, dist_mm=round(t[2] * 1000),
                 corner_px=round(corner_err, 3), rms_px=round(rms, 3),
                 rot_err_deg=round(ang, 3), trans_err_mm=round(terr, 2),
                 vs_opencv_deg=round(d_cv, 5)))
        imgs.append(crop_around(draw_axes(img, CAM, Rh, th), uv))
        rows.append((ang, terr))
    print(f"    {time.time() - t0:.1f} s")

    fig, axes = plt.subplots(1, 4, figsize=(12, 2.6))
    for ax, img, r in zip(axes, imgs, rows):
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if r:
            ax.set_title(f"{r[0]:.2f} deg, {r[1]:.1f} mm", fontsize=9)
    fig.suptitle("Recovered pose, drawn back onto the image "
                 "(red = tag x, green = tag y, blue = tag normal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "detect.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. accuracy vs distance
# --------------------------------------------------------------------------

def stage_distance():
    print("\n[2] accuracy vs distance, on rendered images")
    rows = []
    rng = np.random.default_rng(4)
    for d in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8):
        side, rot, orc, dep, seen = [], [], [], [], 0
        for k in range(8):                      # average over 8 azimuths: one
            az = 45 * k                         # sample per distance is far too
            R, t = pose_from_angles(d, 30, az, 15)   # noisy to read a trend from
            img = render_tag(R, t)
            dets = detect_tags(img)
            if not dets:
                continue
            seen += 1
            uv = list(dets.values())[0]
            side.append(float(np.mean([np.linalg.norm(uv[i] - uv[(i + 1) % 4])
                                       for i in range(4)])))
            sols = pnp_planar(CAM, OBJ, uv)
            Rh, th, rms = sols[0]
            ang, terr, zerr = pose_error(Rh, th, R, t)
            rot.append(ang)
            # the best of the two candidate poses -- see experiment 3.  Kept
            # separate here so the depth trend is not swamped by pose flips.
            orc.append(min(rot_angle_deg(R, x[0]) for x in sols))
            dep.append(abs(zerr))
        if not seen:
            log(dict(stage="distance", dist_m=d, detected=0))
            continue
        rows.append((d, float(np.mean(side)), float(np.mean(rot)),
                     float(np.mean(dep)), float(np.mean(orc))))
        log(dict(stage="distance", dist_m=d, detected=seen, tag_px=round(rows[-1][1], 1),
                 rot_err_deg=round(rows[-1][2], 3),
                 rot_err_best_of_two_deg=round(rows[-1][4], 3),
                 depth_err_mm=round(rows[-1][3], 2),
                 rel_depth_pct=round(100 * rows[-1][3] / (d * 1000), 3)))
    a = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.1))
    ax[0].loglog(a[:, 0], a[:, 3], "o-", color=COLORS[0], label="measured")
    ref = a[0, 3] * (a[:, 0] / a[0, 0]) ** 2
    ax[0].loglog(a[:, 0], ref, "--", color=COLORS[6], label="a $z^2$ reference line")
    ax[0].set_xlabel("distance (m)"); ax[0].set_ylabel("depth error (mm)")
    ax[0].set_title("depth error vs distance"); ax[0].legend(fontsize=8)
    ax[1].loglog(a[:, 0], a[:, 2], "o-", color=COLORS[1], label="pose actually reported")
    ax[1].loglog(a[:, 0], a[:, 4], "s--", color=COLORS[2], label="better of the two candidates")
    ax[1].set_xlabel("distance (m)"); ax[1].set_ylabel("rotation error (deg)")
    ax[1].set_title("rotation error vs distance"); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "distance.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. the ambiguity
# --------------------------------------------------------------------------

def ambiguity_trials(dist, tilt, trials, rng, noise=0.30):
    """Solve `trials` random tags at a given distance and tilt.

    Returns (pct_with_two_solutions, pct_of_those_where_the_better_fit_is_wrong,
             median_rms_ratio, mean_shipped_error, mean_oracle_error).
    "Shipped" = the pose you would actually use (lowest reprojection error).
    "Oracle" = the better of the two, which you could only pick if you already
    knew the answer.  The gap between them is what the ambiguity costs.
    """
    wrong, ratios, ship, oracle, two = 0, [], [], [], 0
    for _ in range(trials):
        R, t = pose_from_angles(dist, tilt, rng.uniform(0, 360), rng.uniform(0, 360))
        uv = synth_corners(R, t, rng, noise=noise)
        sols = pnp_planar(CAM, OBJ, uv)
        errs = [rot_angle_deg(R, s[0]) for s in sols]
        ship.append(errs[0])
        oracle.append(min(errs))
        if len(sols) > 1:
            two += 1
            ratios.append(sols[1][2] / max(sols[0][2], 1e-9))
            if errs[1] < errs[0]:
                wrong += 1
    return (100 * two / trials, 100 * wrong / max(two, 1),
            float(np.median(ratios)) if ratios else float("nan"),
            float(np.mean(ship)), float(np.mean(oracle)))


def stage_ambiguity():
    print("\n[3] the planar pose ambiguity")
    trials = 120

    # (a) tilt sweep, at a distance where a 6 cm tag spans about 27 px
    tilt_curve = []
    for tilt in (3, 6, 9, 12, 16, 20, 25, 30, 40, 50, 60):
        rng = np.random.default_rng(1000 + tilt)
        r = ambiguity_trials(1.2, tilt, trials, rng)
        tilt_curve.append((tilt,) + r)
        log(dict(stage="ambiguity_tilt", tilt_deg=tilt, dist_m=1.2,
                 two_solutions_pct=round(r[0], 1), better_fit_is_wrong_pct=round(r[1], 1),
                 median_rms_ratio=round(r[2], 3),
                 shipped_err_deg=round(r[3], 2), oracle_err_deg=round(r[4], 2)))

    # (b) distance sweep at a fixed, modest tilt -- the same ambiguity, seen
    #     as a function of how many pixels the tag covers
    dist_curve = []
    for d in (0.4, 0.6, 0.9, 1.2, 1.6, 2.0, 2.6):
        rng = np.random.default_rng(2000 + int(d * 10))
        r = ambiguity_trials(d, 15, trials, rng)
        px = float(np.mean(np.linalg.norm(
            CAM.project(OBJ @ pose_from_angles(d, 15)[0].T + pose_from_angles(d, 15)[1])
            - np.roll(CAM.project(OBJ @ pose_from_angles(d, 15)[0].T
                                  + pose_from_angles(d, 15)[1]), 1, axis=0), axis=1)))
        dist_curve.append((d, px) + r)
        log(dict(stage="ambiguity_dist", dist_m=d, tag_px=round(px, 1), tilt_deg=15,
                 two_solutions_pct=round(r[0], 1), better_fit_is_wrong_pct=round(r[1], 1),
                 median_rms_ratio=round(r[2], 3),
                 shipped_err_deg=round(r[3], 2), oracle_err_deg=round(r[4], 2)))

    # (c) a picture of both solutions on one real rendered image
    R, t = pose_from_angles(0.75, 12, 30, 0)
    img = render_tag(R, t)
    uv = list(detect_tags(img).values())[0]
    sols = pnp_planar(CAM, OBJ, uv)
    twin = img.copy()
    for s in sols:
        twin = draw_axes(twin, CAM, s[0], s[1], length=0.05, thickness=2)
    log(dict(stage="ambiguity_example", tilt_deg=12, dist_m=0.75,
             rms_best=round(sols[0][2], 4),
             rms_second=round(sols[1][2], 4) if len(sols) > 1 else None,
             err_best_deg=round(rot_angle_deg(R, sols[0][0]), 2),
             err_second_deg=round(rot_angle_deg(R, sols[1][0]), 2) if len(sols) > 1 else None,
             normals_apart_deg=round(float(np.degrees(np.arccos(np.clip(
                 float(np.dot(sols[0][0][:, 2], sols[1][0][:, 2])), -1, 1)))), 2)
             if len(sols) > 1 else None))

    a, b = np.array(tilt_curve), np.array(dist_curve)
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    ax[0].plot(a[:, 0], a[:, 2], "o-", color=COLORS[1], label="vs tilt (at 1.2 m)")
    ax[0].set_xlabel("tag tilt away from facing the camera (deg)")
    ax[0].set_ylabel("% of two-solution cases")
    ax[0].set_title("how often the BETTER-fitting pose\nis the wrong one")
    ax2 = ax[0].twiny()
    ax2.plot(b[:, 0], b[:, 3], "s--", color=COLORS[0], label="vs distance (at 15 deg)")
    ax2.set_xlabel("distance (m)", color=COLORS[0])
    ax[0].legend(loc="upper right", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax[1].plot(a[:, 0], a[:, 4], "o-", color=COLORS[1], label="what you ship")
    ax[1].plot(a[:, 0], a[:, 5], "s--", color=COLORS[2], label="if you always picked right")
    ax[1].set_xlabel("tag tilt (deg)"); ax[1].set_ylabel("rotation error (deg)")
    ax[1].legend(fontsize=8); ax[1].set_title("the cost of picking wrong")
    ax[2].imshow(crop_around(twin, uv))
    ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].grid(False)
    ax[2].set_title("both solutions drawn on one tag\n(12 deg tilt, 0.75 m)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ambiguity.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. two cures
# --------------------------------------------------------------------------

def stage_cures():
    print("\n[4] two cures: reject the ambiguous ones, or use four tags")
    rng = np.random.default_rng(11)
    trials = 250
    single, ratios, plate, picked_wrong = [], [], [], []
    objs, make_planes = tag_board(4, spacing=0.11)
    for _ in range(trials):
        tilt = rng.uniform(3, 25)                 # the regime where it bites
        R, t = pose_from_angles(1.2, tilt, rng.uniform(0, 360), rng.uniform(0, 360))
        uv = synth_corners(R, t, rng, noise=0.30)
        sols = pnp_planar(CAM, OBJ, uv)
        single.append(rot_angle_deg(R, sols[0][0]))
        ratios.append(sols[1][2] / max(sols[0][2], 1e-9) if len(sols) > 1 else np.inf)
        picked_wrong.append(len(sols) > 1 and
                            rot_angle_deg(R, sols[1][0]) < single[-1])
        dets = {i: synth_corners(R, t, rng, noise=0.30, obj=objs[i]) for i in objs}
        res = pnp_multi(CAM, objs, dets)
        plate.append(rot_angle_deg(R, res[0]))
    single, ratios = np.array(single), np.array(ratios)
    plate, picked_wrong = np.array(plate), np.array(picked_wrong)

    log(dict(stage="cure_none", n=trials, mean_deg=round(float(single.mean()), 3),
             median_deg=round(float(np.median(single)), 3),
             p90_deg=round(float(np.percentile(single, 90)), 3),
             over_10deg_pct=round(100 * float((single > 10).mean()), 1)))
    for thr in (1.1, 1.3, 1.6, 2.0, 3.0):
        keep = ratios > thr
        log(dict(stage="cure_reject", ratio_threshold=thr,
                 kept_pct=round(100 * float(keep.mean()), 1),
                 mean_deg=round(float(single[keep].mean()), 3) if keep.any() else None,
                 median_deg=round(float(np.median(single[keep])), 3) if keep.any() else None,
                 p90_deg=round(float(np.percentile(single[keep], 90)), 3) if keep.any() else None,
                 over_10deg_pct=round(100 * float((single[keep] > 10).mean()), 1)
                 if keep.any() else None))
    # does the ratio predict the error at all?  (it barely does -- see README)
    fin = np.isfinite(ratios)
    log(dict(stage="cure_reject_corr",
             corr_log_ratio_vs_log_err=round(float(np.corrcoef(
                 np.log(np.minimum(ratios[fin], 10.0)), np.log(single[fin]))[0, 1]), 3),
             wrong_pick_pct_ratio_below_1p2=round(100 * float(
                 np.mean(picked_wrong[ratios < 1.2])), 1) if (ratios < 1.2).any() else None,
             wrong_pick_pct_ratio_above_2=round(100 * float(
                 np.mean(picked_wrong[ratios > 2.0])), 1) if (ratios > 2.0).any() else None))
    log(dict(stage="cure_plate", n_tags=4, mean_deg=round(float(plate.mean()), 3),
             median_deg=round(float(np.median(plate)), 3),
             p90_deg=round(float(np.percentile(plate, 90)), 3),
             over_10deg_pct=round(100 * float((plate > 10).mean()), 1)))

    # and the plate, rendered, so you can see what it is
    R, t = pose_from_angles(1.0, 15, 30, 0)
    planes = make_planes(np.eye(3), np.zeros(3))
    img, _, _ = render(CAM, planes, R.T, -R.T @ t, supersample=2, seed=0, noise=2.0)
    dets = detect_tags(img)
    res = pnp_multi(CAM, objs, dets)
    if res is not None:
        img = draw_axes(img, CAM, res[0], res[1], length=0.08, thickness=2)
        log(dict(stage="cure_plate_rendered", tags_found=len(dets),
                 rot_err_deg=round(rot_angle_deg(R, res[0]), 3),
                 trans_err_mm=round(float(np.linalg.norm(res[1] - t)) * 1000, 2)))

    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.2))
    bins = np.logspace(-1.5, 2, 40)
    ax[0].hist(single, bins=bins, color=COLORS[1], alpha=0.75, label="one tag")
    ax[0].hist(plate, bins=bins, color=COLORS[2], alpha=0.75, label="four tags on a plate")
    ax[0].set_xscale("log"); ax[0].set_xlabel("rotation error (deg)")
    ax[0].set_ylabel("detections"); ax[0].legend(fontsize=8)
    ax[0].set_title("250 tags at 1.2 m, 3-25 deg tilt")
    ax[1].scatter(np.minimum(ratios, 4), single, s=8, color=COLORS[0], alpha=0.6)
    ax[1].axvline(1.3, color=COLORS[1], ls="--", lw=1)
    ax[1].set_yscale("log"); ax[1].set_xlabel("rms ratio (2nd solution / best)")
    ax[1].set_ylabel("rotation error (deg)")
    ax[1].set_title("the ratio flags WHICH pose is right,\nnot HOW wrong you are")
    ax[2].imshow(crop_around(img, np.concatenate(list(dets.values())), pad=40))
    ax[2].set_xticks([]); ax[2].set_yticks([]); ax[2].grid(False)
    ax[2].set_title("the four-tag plate, solved as one rigid body")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cures.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5 + 6. wrong ruler, wrong camera
# --------------------------------------------------------------------------

def stage_wrong_inputs():
    print("\n[5] a tag measured wrong, and a camera calibrated wrong")
    rng = np.random.default_rng(21)
    # Spread the tags over the whole image, not just the middle.  Distortion
    # is nearly zero at the principal point, so a test that keeps the tag
    # centred would "prove" that the distortion model does not matter.
    poses = []
    while len(poses) < 40:
        R, t = pose_from_angles(0.6, rng.uniform(15, 45), rng.uniform(0, 360),
                                rng.uniform(0, 360))
        t = t + np.array([rng.uniform(-0.22, 0.22), rng.uniform(-0.16, 0.16), 0.0])
        uv = CAM.project(OBJ @ R.T + t)
        if uv.min() > 4 and uv[:, 0].max() < 636 and uv[:, 1].max() < 476:
            poses.append((R, t))

    for err_pct in (-5, -2, 0, 2, 5):
        size = TAG_SIZE * (1 + err_pct / 100)
        obj_believed = tag_object_points(size)
        rot, dep = [], []
        for R, t in poses:
            uv = synth_corners(R, t, rng)
            Rh, th, _ = pnp_planar(CAM, obj_believed, uv)[0]
            rot.append(rot_angle_deg(R, Rh))
            dep.append(100 * (th[2] - t[2]) / t[2])
        log(dict(stage="wrong_size", size_err_pct=err_pct,
                 depth_err_pct=round(float(np.mean(dep)), 3),
                 rot_err_deg=round(float(np.mean(rot)), 4)))

    # project 16 showed that a fronto-parallel-only calibration lands ~11%
    # low on the focal length while reporting a perfect reprojection error.
    # Here is what that mistake costs downstream.
    for name, cam_used in [("correct", CAM),
                           ("fronto-parallel calib (fx 11% low)",
                            Camera(478.4, 476.6, CAM.cx, CAM.cy, CAM.dist)),
                           ("no distortion model",
                            Camera(CAM.fx, CAM.fy, CAM.cx, CAM.cy))]:
        rot, dep = [], []
        for R, t in poses:
            uv = synth_corners(R, t, rng)          # the IMAGE is always the truth
            Rh, th, _ = pnp_planar(cam_used, OBJ, uv)[0]
            rot.append(rot_angle_deg(R, Rh))
            dep.append(100 * (th[2] - t[2]) / t[2])
        log(dict(stage="wrong_camera", camera=name,
                 depth_err_pct=round(float(np.mean(dep)), 3),
                 rot_err_deg=round(float(np.mean(rot)), 3)))


# --------------------------------------------------------------------------
# 7. corner refinement
# --------------------------------------------------------------------------

def stage_refinement():
    print("\n[6] subpixel corner refinement, on and off")
    rng = np.random.default_rng(31)
    for refine in (False, True):
        rot, tr, cor = [], [], []
        for _ in range(12):
            R, t = pose_from_angles(rng.uniform(0.4, 0.9), rng.uniform(20, 50),
                                    rng.uniform(0, 360), rng.uniform(0, 360))
            img = render_tag(R, t)
            dets = detect_tags(img, refine=refine)
            if not dets:
                continue
            uv = list(dets.values())[0]
            cor.append(float(np.mean(np.linalg.norm(
                uv - CAM.project(OBJ @ R.T + t), axis=1))))
            Rh, th, _ = pnp_planar(CAM, OBJ, uv)[0]
            rot.append(rot_angle_deg(R, Rh))
            tr.append(float(np.linalg.norm(th - t)) * 1000)
        log(dict(stage="refinement", subpixel=refine, n=len(rot),
                 corner_err_px=round(float(np.mean(cor)), 3),
                 rot_err_deg=round(float(np.mean(rot)), 3),
                 trans_err_mm=round(float(np.mean(tr)), 2)))


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    stage_detect()
    stage_distance()
    stage_ambiguity()
    stage_cures()
    stage_wrong_inputs()
    stage_refinement()

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
