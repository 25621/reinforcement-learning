"""Project 20 -- Monocular visual odometry: 100 m of corridor, and the drift.

Six experiments:

  1. the front end: track, solve, accumulate; drift over 100 m
  2. where the scale comes from (it is not in the images)
  3. outlier rejection, with up to 35% of the tracks deliberately wrong
  4. keyframe spacing, and which stage it really tests
  5. why drift is a ROTATION problem, measured
  6. how many features, and how noisy a single drift number is

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

from camera import Camera, default_camera, rodrigues, rot_angle_deg     # noqa: E402
from vo import (corridor, trajectory, render_sequence, track,           # noqa: E402
                relative_motion, integrate, align_yaw_only)
from plot_style import COLORS, use_style                                # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

# A smaller sensor than the rest of the phase: 200 frames have to be rendered,
# and the geometry does not care about resolution the way calibration does.
CAM = Camera(fx=380.0, fy=380.0, cx=239.5, cy=179.5,
             dist=(-0.10, 0.03, 0, 0, 0), width=480, height=360)
N_FRAMES = 201
STEP = 0.5                                    # metres per frame


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def build_sequence():
    poses = trajectory(N_FRAMES, step=STEP)
    t0 = time.time()
    imgs = render_sequence(CAM, corridor(), poses, supersample=1, noise=2.0)
    print(f"    rendered {len(imgs)} frames in {time.time() - t0:.1f} s", flush=True)
    return imgs, poses


def run_vo(imgs, poses, stride=1, ransac=True, outlier_frac=0.0, max_corners=600,
           scale_mode="ground_truth", yaw_bias_deg=0.0, seed=0, solver="opencv"):
    """One complete odometry run.  Returns (estimated poses, diagnostics)."""
    rng = np.random.default_rng(seed)
    steps, scales, inliers, ntracks = [], [], [], []
    gt_used = []
    for i in range(0, len(imgs) - stride, stride):
        p0, p1 = track(imgs[i], imgs[i + stride], max_corners=max_corners)
        if p0 is None or len(p0) < 20:
            steps.append((np.eye(3), np.array([0.0, 0.0, 1.0])))
            scales.append(0.0)
            inliers.append(0.0)
            ntracks.append(0)
            gt_used.append((poses[i], poses[i + stride]))
            continue
        if outlier_frac > 0:
            # a fraction of the tracks land somewhere random -- exactly what a
            # repeated texture, a moving object, or a specular highlight does
            k = int(len(p1) * outlier_frac)
            sel = rng.choice(len(p1), k, replace=False)
            p1 = p1.copy()
            p1[sel] = rng.uniform([0, 0], [CAM.width, CAM.height], (k, 2))
        res = relative_motion(CAM, p0, p1, ransac=ransac, solver=solver, rng=rng)
        if res is None:
            steps.append((np.eye(3), np.array([0.0, 0.0, 1.0])))
            scales.append(0.0)
            inliers.append(0.0)
            ntracks.append(len(p0))
            continue
        R, t, inl = res
        if yaw_bias_deg:
            R = rodrigues([0, np.radians(yaw_bias_deg), 0]) @ R
        steps.append((R, t))
        inliers.append(inl)
        ntracks.append(len(p0))
        if scale_mode == "ground_truth":
            scales.append(float(np.linalg.norm(poses[i + stride][1] - poses[i][1])))
        elif scale_mode == "constant":
            scales.append(STEP * stride)
        else:
            scales.append(1.0)
    est = integrate(steps, scales)
    gt = [poses[i] for i in range(0, len(imgs), stride)][:len(est)]
    return est, gt, dict(inlier=float(np.mean(inliers)), tracks=float(np.mean(ntracks)))


def summarize(est, gt, **extra):
    err, dist = align_yaw_only(est, gt)
    n = len(err)
    rot = [rot_angle_deg(gt[i][0], est[i][0]) for i in range(n)]
    return dict(distance_m=round(float(dist[-1]), 1),
                final_err_m=round(float(err[-1]), 3),
                drift_pct=round(100 * float(err[-1]) / max(float(dist[-1]), 1e-9), 3),
                max_err_m=round(float(err.max()), 3),
                final_rot_err_deg=round(float(rot[-1]), 3), **extra)


# --------------------------------------------------------------------------
# 1. the baseline run
# --------------------------------------------------------------------------

def stage_baseline(imgs, poses):
    print("\n[1] the front end, over 100 m")
    t0 = time.time()
    est, gt, diag = run_vo(imgs, poses)
    s = summarize(est, gt, seconds=round(time.time() - t0, 1),
                  mean_tracks=round(diag["tracks"], 1),
                  mean_inlier_pct=round(100 * diag["inlier"], 1))
    log(dict(stage="baseline", **s))

    e = np.array([p for _, p in est])
    g = np.array([p for _, p in gt])
    err, dist = align_yaw_only(est, gt)

    p0, p1 = track(imgs[40], imgs[41])
    vis = cv2.cvtColor(imgs[40], cv2.COLOR_GRAY2RGB)
    for a, b in zip(p0[:250], p1[:250]):
        cv2.arrowedLine(vis, tuple(a.astype(int)), tuple(b.astype(int)),
                        (255, 120, 0), 1, cv2.LINE_AA, tipLength=0.4)

    fig = plt.figure(figsize=(12, 3.4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(vis); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    ax.set_title(f"tracked features between two frames\n({len(p0)} of them)")
    ax = fig.add_subplot(1, 3, 2)
    ax.plot(g[:, 2], g[:, 0], color=COLORS[6], lw=2, label="ground truth")
    ax.plot(e[:, 2], e[:, 0], color=COLORS[1], lw=1.4, label="visual odometry")
    ax.set_xlabel("forward z (m)"); ax.set_ylabel("lateral x (m)")
    ax.legend(fontsize=8); ax.set_title("the path from above")
    ax = fig.add_subplot(1, 3, 3)
    ax.plot(dist, err, color=COLORS[1])
    ax.set_xlabel("distance travelled (m)"); ax.set_ylabel("position error (m)")
    ax.set_title(f"drift: {s['drift_pct']:.2f}% of distance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "baseline.png"))
    plt.close(fig)
    return est, gt


# --------------------------------------------------------------------------
# 2. scale
# --------------------------------------------------------------------------

def stage_scale(imgs, poses):
    print("\n[2] where the scale comes from")
    curves = {}
    for mode, label in [("ground_truth", "per-step scale from ground truth"),
                        ("constant", "assume a constant 0.5 m per frame"),
                        ("unit", "no scale at all (unit steps)")]:
        est, gt, _ = run_vo(imgs, poses, scale_mode=mode)
        s = summarize(est, gt, scale_source=label)
        log(dict(stage="scale", **s))
        curves[label] = (np.array([p for _, p in est]), np.array([p for _, p in gt]))

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    g = list(curves.values())[0][1]
    ax[0].plot(g[:, 2], g[:, 0], color=COLORS[6], lw=2.5, label="ground truth")
    for (label, (e, _)), c in zip(curves.items(), [COLORS[0], COLORS[1], COLORS[2]]):
        ax[0].plot(e[:, 2], e[:, 0], color=c, lw=1.3, label=label)
    ax[0].set_xlabel("forward z (m)"); ax[0].set_ylabel("lateral x (m)")
    ax[0].legend(fontsize=7); ax[0].set_title("three scale sources")
    e = curves["per-step scale from ground truth"][0]
    steps_e = np.linalg.norm(np.diff(e, axis=0), axis=1)
    steps_g = np.linalg.norm(np.diff(g, axis=0), axis=1)
    ax[1].plot(steps_g[:len(steps_e)], color=COLORS[6], label="true step length")
    ax[1].plot(steps_e, color=COLORS[1], lw=0.9, label="reconstructed step length")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("step (m)"); ax[1].legend(fontsize=8)
    ax[1].set_title("with the true scale handed in, the steps match")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "scale.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. RANSAC
# --------------------------------------------------------------------------

def stage_ransac(imgs, poses):
    print("\n[3] RANSAC against bad tracks")
    rows = []
    arms = [("plain 8-point, no rejection", dict(solver="ours", ransac=False)),
            ("our RANSAC + 8-point", dict(solver="ours", ransac=True)),
            ("OpenCV RANSAC", dict(solver="opencv", ransac=True))]
    for frac in (0.0, 0.05, 0.10, 0.20, 0.35):
        for name, kw in arms:
            est, gt, diag = run_vo(imgs, poses, outlier_frac=frac, seed=4, **kw)
            s = summarize(est, gt, outlier_frac=frac, arm=name,
                          mean_inlier_pct=round(100 * diag["inlier"], 1))
            log(dict(stage="ransac", **s))
            rows.append((frac, name, s["drift_pct"]))
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for (name, _), col in zip(arms, [COLORS[1], COLORS[2], COLORS[0]]):
        xs = [r[0] * 100 for r in rows if r[1] == name]
        ys = [r[2] for r in rows if r[1] == name]
        ax.plot(xs, ys, "o-", color=col, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("fraction of tracks that are wrong (%)")
    ax.set_ylabel("drift (% of distance)"); ax.legend(fontsize=8)
    ax.set_title("outlier rejection under increasing corruption")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ransac.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. keyframe spacing
# --------------------------------------------------------------------------

def stage_stride(imgs, poses):
    print("\n[4] keyframe spacing")
    rows = []
    for stride in (1, 2, 3, 5, 8, 12):
        est, gt, diag = run_vo(imgs, poses, stride=stride)
        s = summarize(est, gt, stride=stride, baseline_m=round(stride * STEP, 2),
                      mean_tracks=round(diag["tracks"], 1),
                      mean_inlier_pct=round(100 * diag["inlier"], 1))
        log(dict(stage="stride", **s))
        rows.append((stride * STEP, s["drift_pct"], diag["tracks"], diag["inlier"]))
    a = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(a[:, 0], a[:, 1], "o-", color=COLORS[0])
    ax[0].set_xlabel("distance between the two views (m)")
    ax[0].set_ylabel("drift (% of distance)")
    ax[0].set_title("wider spacing loses the tracker\nlong before it helps the geometry")
    ax[1].plot(a[:, 0], a[:, 2], "o-", color=COLORS[1], label="features tracked")
    ax[1].set_xlabel("distance between the two views (m)"); ax[1].set_ylabel("features")
    ax[1].legend(fontsize=8); ax[1].set_title("features survive; it is the MATCHES that fail\n(watch the inlier rate in results.csv)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stride.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. drift is a rotation problem
# --------------------------------------------------------------------------

def stage_rotation(imgs, poses):
    print("\n[5] why drift is a rotation problem")
    rows = []
    for bias in (0.0, 0.01, 0.02, 0.05, 0.10):
        est, gt, _ = run_vo(imgs, poses, yaw_bias_deg=bias)
        s = summarize(est, gt, yaw_bias_deg_per_step=bias)
        log(dict(stage="yaw_bias", **s))
        err, dist = align_yaw_only(est, gt)
        rows.append((bias, err, dist, s["drift_pct"]))

    # how does the error grow with distance, for a constant heading bias?
    b, err, dist, _ = rows[-1]
    m = dist > 5
    p = np.polyfit(np.log(dist[m]), np.log(np.maximum(err[m], 1e-6)), 1)[0]
    log(dict(stage="yaw_bias_fit", bias_deg=b, growth_exponent=round(float(p), 3)))

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for (bias, err, dist, _), c in zip(rows, [COLORS[6]] + COLORS[:4]):
        ax[0].plot(dist, err, color=c, label=f"{bias:.2f} deg/step")
    ax[0].set_xlabel("distance travelled (m)"); ax[0].set_ylabel("position error (m)")
    ax[0].legend(fontsize=7); ax[0].set_title("a tiny constant heading error")
    ax[1].loglog(dist[m], err[m], color=COLORS[1], label="measured")
    ax[1].loglog(dist[m], err[m][0] * (dist[m] / dist[m][0]) ** 2, "--",
                 color=COLORS[6], label="a $d^2$ reference line")
    ax[1].set_xlabel("distance (m)"); ax[1].set_ylabel("position error (m)")
    ax[1].legend(fontsize=8); ax[1].set_title(f"growth exponent {p:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rotation.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. how many features
# --------------------------------------------------------------------------

def stage_features(imgs, poses):
    print("\n[6] how many features -- and how repeatable one drift number is")
    for n in (40, 80, 150, 300, 600, 1200):
        est, gt, diag = run_vo(imgs, poses, max_corners=n)
        log(dict(stage="features", max_corners=n,
                 **summarize(est, gt, mean_tracks=round(diag["tracks"], 1))))


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    imgs, poses = build_sequence()
    stage_baseline(imgs, poses)
    stage_scale(imgs, poses)
    stage_ransac(imgs, poses)
    stage_stride(imgs, poses)
    stage_rotation(imgs, poses)
    stage_features(imgs, poses)

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
