"""Project 16 -- Camera calibration, and how to tell a good one from a lucky one.

Seven experiments on one question: what does the number you report actually
prove?

  1. end to end on rendered images: detect corners, run Zhang, check K
  2. our solver against OpenCV's, on the same corners
  3. the fronto-parallel trap -- 0.13 px reprojection error, 16% wrong focal
  4. how many views do you need
  5. how many distortion terms: too few underfits, too many overfit
  6. corner noise in, parameter error out
  7. the lens model outside the region the board ever covered

Everything is rendered through a camera whose true parameters we know, so
every claim can be checked against ground truth -- which is the one thing a
real webcam can never give you.  Runs in about four minutes on a CPU.
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
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from camera import Camera, default_camera, rodrigues, rot_angle_deg          # noqa: E402
from render import (Checkerboard, board_pose, camera_pose_from_board,        # noqa: E402
                    render)
from calib import (calibrate, refine, homography_dlt, intrinsics_zhang,      # noqa: E402
                   extrinsics_from_H, reprojection_rms, parameter_std, solve_pose)
from plot_style import COLORS, use_style                                     # noqa: E402

import matplotlib.pyplot as plt                                              # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
BOARD = Checkerboard(cols=9, rows=6, square=0.025)
CENTER = np.array([-(BOARD.cols - 2) * BOARD.square / 2,
                   -(BOARD.rows - 2) * BOARD.square / 2])   # puts the board mid-image
RESULTS = []


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


# --------------------------------------------------------------------------
# generating views
# --------------------------------------------------------------------------

def sample_poses(kind, n, rng):
    """n board poses of a given flavour, all fully inside the image."""
    poses = []
    tries = 0
    while len(poses) < n and tries < 400:
        tries += 1
        if kind == "varied":
            R, t = board_pose(rng.uniform(0.34, 0.52),
                              rng.uniform(-35, 35), rng.uniform(-35, 35),
                              rng.uniform(-30, 30),
                              shift=CENTER + rng.uniform(-0.05, 0.05, 2))
        elif kind == "fronto":            # every board flat against the sensor
            R, t = board_pose(rng.uniform(0.34, 0.52), 0.0, 0.0,
                              rng.uniform(-30, 30),
                              shift=CENTER + rng.uniform(-0.05, 0.05, 2))
        elif kind == "one_axis":          # tilted, but always about the same axis
            R, t = board_pose(rng.uniform(0.34, 0.52),
                              rng.uniform(-35, 35), 0.0, 0.0,
                              shift=CENTER + rng.uniform(-0.04, 0.04, 2))
        elif kind == "centered":          # varied tilt but always mid-image
            R, t = board_pose(rng.uniform(0.40, 0.46),
                              rng.uniform(-30, 30), rng.uniform(-30, 30),
                              rng.uniform(-20, 20), shift=CENTER)
        else:
            raise ValueError(kind)
        uv = default_camera().project(BOARD.object_points @ R.T + t)
        if (uv[:, 0].min() < 10 or uv[:, 0].max() > 630 or
                uv[:, 1].min() < 10 or uv[:, 1].max() > 470):
            continue
        poses.append((R, t))
    if len(poses) < n:
        raise RuntimeError(f"only {len(poses)}/{n} {kind} poses fit in the image")
    return poses


def synth_obs(cam, poses, noise, rng):
    """Corner observations without rendering: project, then add pixel noise.

    Used for the sweeps.  Rendering 20 images per configuration would cost
    minutes and would add nothing -- the detector's error IS pixel noise, and
    here we get to dial it.
    """
    return [cam.project(BOARD.object_points @ R.T + t) +
            rng.normal(0.0, noise, (len(BOARD.object_points), 2))
            for R, t in poses]


def detect(img):
    """Find the inner corners, then refine them to subpixel accuracy."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ok, c = cv2.findChessboardCorners(
        g, BOARD.inner, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    c = cv2.cornerSubPix(g, c, (7, 7), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4))
    return c.reshape(-1, 2)


def param_errors(cam, truth):
    return dict(fx=cam.fx - truth.fx, fy=cam.fy - truth.fy,
                cx=cam.cx - truth.cx, cy=cam.cy - truth.cy,
                k1=cam.dist[0] - truth.dist[0])


# --------------------------------------------------------------------------
# 1. end to end on rendered images
# --------------------------------------------------------------------------

def stage_render():
    print("\n[1] rendering 15 views and detecting corners")
    truth = default_camera()
    rng = np.random.default_rng(1)
    poses = sample_poses("varied", 15, rng)
    imgs, obs, gts = [], [], []
    t0 = time.time()
    for R, t in poses:
        R_wc, t_wc = camera_pose_from_board(R, t)
        img, _, _ = render(truth, [BOARD.plane(np.eye(3), np.zeros(3))],
                           R_wc, t_wc, supersample=2, seed=0, noise=2.0)
        c = detect(img)
        if c is None:
            continue
        imgs.append(img)
        obs.append(c)
        gts.append(truth.project(BOARD.object_points @ R.T + t))
    print(f"    rendered+detected {len(obs)}/15 views in {time.time() - t0:.1f} s")

    err = np.linalg.norm(np.concatenate(obs) - np.concatenate(gts), axis=1)
    log(dict(stage="detector", n_views=len(obs), corners=len(err),
             mean_px=round(float(err.mean()), 4), max_px=round(float(err.max()), 4)))

    np.savez(os.path.join(OUT, "views.npz"),
             obs=np.array(obs), poses_R=np.array([R for R, _ in poses[:len(obs)]]),
             poses_t=np.array([t for _, t in poses[:len(obs)]]))

    fig, axes = plt.subplots(3, 5, figsize=(11, 6.2))
    for ax, img, c in zip(axes.reshape(-1), imgs, obs):
        ax.imshow(img)
        ax.plot(c[:, 0], c[:, 1], ".", color=COLORS[1], ms=2.2)
        ax.plot(c[0, 0], c[0, 1], "o", color=COLORS[2], ms=5, mfc="none")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in axes.reshape(-1)[len(imgs):]:
        ax.axis("off")
    fig.suptitle("The 15 calibration views (green circle = corner #1, "
                 "so the ordering is consistent)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "views.png"))
    plt.close(fig)
    return obs


def stage_baseline(obs):
    print("\n[2] calibrating from the DETECTED corners")
    truth = default_camera()
    obj = BOARD.object_points

    Hs = [homography_dlt(obj[:, :2], uv) for uv in obs]
    K0 = intrinsics_zhang(Hs)
    cam0 = Camera.from_K(K0)
    poses0 = [extrinsics_from_H(K0, H) for H in Hs]
    log(dict(stage="closed_form", fx=round(cam0.fx, 2), fy=round(cam0.fy, 2),
             cx=round(cam0.cx, 2), cy=round(cam0.cy, 2),
             rms_px=round(reprojection_rms(cam0, poses0, obj, obs), 3),
             fx_err=round(cam0.fx - truth.fx, 2)))

    cam, poses, rms = refine(cam0, poses0, obj, obs, n_dist=5)
    std, cond = parameter_std(cam, poses, obj, obs)
    e = param_errors(cam, truth)
    log(dict(stage="refined", fx=round(cam.fx, 2), fy=round(cam.fy, 2),
             cx=round(cam.cx, 2), cy=round(cam.cy, 2),
             k1=round(cam.dist[0], 4), rms_px=round(rms, 4),
             fx_err=round(e["fx"], 2), cx_err=round(e["cx"], 2),
             fx_std=round(float(std[0]), 2)))

    # OpenCV as an independent second opinion on the same corners
    ok, Kc, dc, rv, tv = cv2.calibrateCamera(
        [obj.astype(np.float32)] * len(obs),
        [o.astype(np.float32).reshape(-1, 1, 2) for o in obs], (640, 480), None, None)
    log(dict(stage="opencv", fx=round(float(Kc[0, 0]), 2), fy=round(float(Kc[1, 1]), 2),
             cx=round(float(Kc[0, 2]), 2), cy=round(float(Kc[1, 2]), 2),
             k1=round(float(dc.reshape(-1)[0]), 4), rms_px=round(float(ok), 4),
             fx_err=round(float(Kc[0, 0]) - truth.fx, 2)))

    # the pose error that the calibration buys you
    ang = [rot_angle_deg(R, rodrigues(rv[i].reshape(3)))
           for i, (R, _) in enumerate(poses)]
    dist_err = [np.linalg.norm(t - tv[i].reshape(3)) * 1000 for i, (_, t) in enumerate(poses)]
    log(dict(stage="ours_vs_opencv_poses", max_rot_deg=round(float(np.max(ang)), 4),
             max_trans_mm=round(float(np.max(dist_err)), 4)))
    return cam, poses


# --------------------------------------------------------------------------
# 3. the fronto-parallel trap
# --------------------------------------------------------------------------

def stage_traps():
    print("\n[3] which views you take matters more than how many")
    truth = default_camera()
    obj = BOARD.object_points
    rows = []
    for kind in ("varied", "one_axis", "centered", "fronto", "fronto_perfect_lens"):
        rng = np.random.default_rng(7)
        # "fronto_perfect_lens" repeats the fronto-parallel set with a lens
        # that has NO distortion.  That isolates the ambiguity: with a real
        # lens, distortion is the only thing pinning the focal length down.
        cam_truth = truth if kind != "fronto_perfect_lens" else Camera(
            truth.fx, truth.fy, truth.cx, truth.cy, (0, 0, 0, 0, 0))
        poses = sample_poses("fronto" if kind.startswith("fronto") else kind, 15, rng)
        obs = synth_obs(cam_truth, poses, 0.10, rng)
        Hs = [homography_dlt(obj[:, :2], uv) for uv in obs]
        try:
            K0 = intrinsics_zhang(Hs)
            closed_form = "ok"
            if not (100 < K0[0, 0] < 3000):
                raise np.linalg.LinAlgError("absurd focal length")
        except np.linalg.LinAlgError:
            # Zhang's linear step has nothing to bite on.  A real toolbox
            # falls back to the standard guess: focal = image width, principal
            # point = image centre.  It then converges happily -- to a wrong
            # answer, which is the whole point of this experiment.
            K0 = np.array([[640.0, 0, 320.0], [0, 640.0, 240.0], [0, 0, 1.0]])
            closed_form = "FAILED"
        cam0 = Camera.from_K(K0)
        poses0 = [extrinsics_from_H(K0, H) for H in Hs]
        cam, ps, rms = refine(cam0, poses0, obj, obs, n_dist=5)
        std, cond = parameter_std(cam, ps, obj, obs)
        d_true = np.mean([t[2] for _, t in poses])
        d_est = np.mean([t[2] for _, t in ps])
        rows.append(dict(views=kind, closed_form=closed_form,
                         rms_px=round(rms, 4), fx=round(cam.fx, 1),
                         fx_err_pct=round(100 * (cam.fx - truth.fx) / truth.fx, 2),
                         fx_std=round(float(std[0]), 1),
                         cond=f"{cond:.1e}",
                         depth_err_pct=round(100 * (d_est - d_true) / d_true, 2)))
        log(dict(stage="views_" + kind, **rows[-1]))

    # the sharpest form of the trap: start the SAME fronto-parallel data from
    # three different focal-length guesses and see where each one settles.
    rng = np.random.default_rng(7)
    poses = sample_poses("fronto", 15, rng)
    obs = synth_obs(truth, poses, 0.10, rng)
    for f0 in (400.0, 540.0, 700.0, 900.0):
        cam0 = Camera(f0, f0, 320.0, 240.0)
        Hs = [homography_dlt(obj[:, :2], uv) for uv in obs]
        poses0 = [extrinsics_from_H(cam0.K, H) for H in Hs]
        cam, ps, rms = refine(cam0, poses0, obj, obs, n_dist=5)
        log(dict(stage="fronto_init", fx_init=f0, fx_final=round(cam.fx, 1),
                 rms_px=round(rms, 4),
                 mean_depth_mm=round(float(np.mean([t[2] for _, t in ps])) * 1000, 1),
                 true_depth_mm=round(float(np.mean([t[2] for _, t in poses])) * 1000, 1)))

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    x = np.arange(len(rows))
    names = [r["views"] for r in rows]
    axes[0].bar(x, [r["rms_px"] for r in rows], color=COLORS[0])
    axes[0].set_title("what you report:\nreprojection error (px)")
    axes[0].axhline(0.5, color=COLORS[1], ls="--", lw=1)
    axes[0].text(0.1, 0.52, "the usual 0.5 px target", color=COLORS[1], fontsize=8)
    axes[1].bar(x, [abs(r["fx_err_pct"]) for r in rows], color=COLORS[1])
    axes[1].set_title("what you wanted:\n|focal length error| (%)")
    axes[2].bar(x, [r["fx_std"] for r in rows], color=COLORS[2])
    axes[2].set_yscale("log")
    axes[2].set_title("what would have warned you:\n1-sigma on fx (px, log)")
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    fig.suptitle("Four view sets, all with a 'good' reprojection error", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "traps.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 4. number of views
# --------------------------------------------------------------------------

def stage_views():
    print("\n[4] how many views")
    truth = default_camera()
    obj = BOARD.object_points
    counts = [3, 4, 5, 6, 8, 10, 14, 20]
    curve = []
    for n in counts:
        errs, stds, rmss = [], [], []
        for seed in range(6):
            rng = np.random.default_rng(100 + seed)
            poses = sample_poses("varied", n, rng)
            obs = synth_obs(truth, poses, 0.10, rng)
            try:
                cam, ps, rms = calibrate(obj, obs)
            except np.linalg.LinAlgError:
                continue
            std, _ = parameter_std(cam, ps, obj, obs)
            errs.append(abs(cam.fx - truth.fx))
            stds.append(float(std[0]))
            rmss.append(rms)
        curve.append((n, float(np.mean(errs)), float(np.mean(stds)), float(np.mean(rmss))))
        log(dict(stage="n_views", n=n, fx_err_px=round(curve[-1][1], 3),
                 fx_std_px=round(curve[-1][2], 3), rms_px=round(curve[-1][3], 4)))
    c = np.array(curve)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].plot(c[:, 0], c[:, 1], "o-", color=COLORS[0], label="true |fx error|")
    ax[0].plot(c[:, 0], c[:, 2], "s--", color=COLORS[2], label="predicted 1-sigma")
    ax[0].set_xlabel("views"); ax[0].set_ylabel("pixels"); ax[0].legend()
    ax[0].set_title("focal-length accuracy vs number of views")
    ax[1].plot(c[:, 0], c[:, 3], "o-", color=COLORS[1])
    ax[1].set_xlabel("views"); ax[1].set_ylabel("RMS reprojection (px)")
    ax[1].set_title("reprojection error barely moves")
    ax[1].set_ylim(0, 0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "n_views.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. how many distortion terms
# --------------------------------------------------------------------------

def stage_model():
    print("\n[5] distortion model complexity, judged on held-out views")
    truth = default_camera()
    obj = BOARD.object_points
    rng = np.random.default_rng(5)
    train_poses = sample_poses("centered", 12, rng)      # all corners mid-image
    rng2 = np.random.default_rng(6)
    test_poses = sample_poses("varied", 8, rng2)         # boards out at the edges
    train = synth_obs(truth, train_poses, 0.10, rng)
    test = synth_obs(truth, test_poses, 0.10, rng2)

    rows = []
    for n_dist, name in [(0, "none"), (1, "k1"), (2, "k1,k2"),
                         (4, "k1,k2,p1,p2"), (5, "k1,k2,p1,p2,k3")]:
        cam, ps, rms = calibrate(obj, train, n_dist=n_dist)
        # Score on the held-out views: fit ONLY their board poses, with the
        # camera frozen.  `solve_pose` exists precisely so the test cannot
        # quietly re-fit the intrinsics it is supposed to be testing.
        held = [solve_pose(cam, obj, uv)[2] for uv in test]
        rows.append(dict(model=name, n_params=n_dist, train_rms=round(rms, 4),
                         heldout_rms=round(float(np.mean(held)), 4),
                         fx_err=round(cam.fx - truth.fx, 2),
                         k1=round(cam.dist[0], 4)))
        log(dict(stage="dist_model", **rows[-1]))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["train_rms"] for r in rows], 0.4, color=COLORS[0], label="training views")
    ax.bar(x + 0.2, [r["heldout_rms"] for r in rows], 0.4, color=COLORS[1], label="held-out views")
    ax.set_xticks(x); ax.set_xticklabels([r["model"] for r in rows], rotation=15)
    ax.set_ylabel("RMS reprojection (px)"); ax.set_yscale("log"); ax.legend()
    ax.set_title("Distortion terms: trained on centre views, tested at the edges")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dist_model.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 6. noise in, error out
# --------------------------------------------------------------------------

def stage_noise():
    print("\n[6] corner noise in, parameter error out")
    truth = default_camera()
    obj = BOARD.object_points
    curve = []
    for noise in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8):
        errs, rmss = [], []
        for seed in range(5):
            rng = np.random.default_rng(200 + seed)
            poses = sample_poses("varied", 12, rng)
            obs = synth_obs(truth, poses, noise, rng)
            cam, ps, rms = calibrate(obj, obs)
            errs.append(abs(cam.fx - truth.fx))
            rmss.append(rms)
        curve.append((noise, float(np.mean(errs)), float(np.mean(rmss))))
        log(dict(stage="noise", corner_noise_px=noise,
                 fx_err_px=round(curve[-1][1], 3), rms_px=round(curve[-1][2], 4)))
    c = np.array(curve)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(c[:, 0], c[:, 1], "o-", color=COLORS[0], label="|fx error| (px)")
    ax.plot(c[:, 0], c[:, 2], "s--", color=COLORS[1], label="RMS reprojection (px)")
    ax.set_xlabel("corner-detection noise, 1 sigma (px)"); ax.legend()
    ax.set_title("Both grow linearly -- and the ratio is about 10x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "noise.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. what the distortion actually does
# --------------------------------------------------------------------------

def radial_profile(cam, r):
    """Distorted radius (in pixels) for an ideal normalized radius r."""
    k1, k2, p1, p2, k3 = cam.dist
    return cam.fx * r * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)


def stage_undistort(cam, obs):
    print("\n[7] the lens model outside the region the board ever covered")
    truth = default_camera()
    gx, gy = np.meshgrid(np.linspace(0, 639, 33), np.linspace(0, 479, 25))
    uv = np.stack([gx.reshape(-1), gy.reshape(-1)], axis=1)
    xn = truth.pixel_to_normalized(uv)
    ideal = np.stack([truth.fx * xn[:, 0] + truth.cx,
                      truth.fy * xn[:, 1] + truth.cy], axis=1)
    shift = np.linalg.norm(uv - ideal, axis=1)
    log(dict(stage="distortion_field", max_shift_px=round(float(shift.max()), 2),
             mean_shift_px=round(float(shift.mean()), 2)))

    # How far out did the calibration data actually reach?  Everything beyond
    # that radius is the polynomial guessing.
    allc = np.concatenate(obs)
    r_data = float(np.max(np.linalg.norm(truth.pixel_to_normalized(allc), axis=1)))
    r_corner = float(np.max(np.linalg.norm(truth.pixel_to_normalized(
        np.array([[0, 0], [639, 0], [0, 479], [639, 479]], float)), axis=1)))
    log(dict(stage="coverage", r_data=round(r_data, 3), r_image_corner=round(r_corner, 3),
             covered_pct=round(100 * r_data / r_corner, 1)))

    obj = BOARD.object_points
    rr = np.linspace(0, r_corner, 200)
    curves = {"true lens": radial_profile(truth, rr)}
    for n_dist, name in [(2, "k1,k2"), (5, "k1,k2,p1,p2,k3")]:
        c, _, rms = calibrate(obj, obs, n_dist=n_dist)
        curves[name] = radial_profile(c, rr)
        err_in = float(np.max(np.abs(curves[name][rr <= r_data] -
                                     curves["true lens"][rr <= r_data])))
        err_out = float(abs(curves[name][-1] - curves["true lens"][-1]))
        # is the model even invertible over the image?  the mapping must be
        # strictly increasing in radius, or two scene points land on one pixel
        mono = bool(np.all(np.diff(curves[name]) > 0))
        log(dict(stage="extrapolation", model=name, train_rms_px=round(rms, 4),
                 err_inside_data_px=round(err_in, 3),
                 err_at_image_corner_px=round(err_out, 2), invertible=mono))

    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.3))
    for (name, y), col, lw in zip(curves.items(), ["#000000", COLORS[0], COLORS[1]],
                                  [5.0, 1.8, 1.8]):
        ax[0].plot(rr, y - radial_profile(Camera(truth.fx, truth.fy, 0, 0), rr),
                   color=col, label=name, lw=lw, alpha=0.25 if lw > 3 else 1.0)
    ax[0].axvline(r_data, color=COLORS[2], ls="--", lw=1)
    ax[0].text(r_data * 0.99, ax[0].get_ylim()[0] * 0.9, " boards never went past here",
               color=COLORS[2], fontsize=8, rotation=90, va="bottom", ha="right")
    ax[0].set_xlabel("normalized radius from the image centre")
    ax[0].set_ylabel("radial shift (px)")
    ax[0].set_title("all three models agree on the data\nand disagree beyond it")
    ax[0].legend(fontsize=8)
    ax[1].quiver(uv[:, 0], uv[:, 1], (ideal - uv)[:, 0], (ideal - uv)[:, 1],
                 shift, cmap="viridis", scale=140)
    ax[1].set_title(f"where the true lens moved each pixel\n(max {shift.max():.1f} px)")
    ax[1].invert_yaxis(); ax[1].set_aspect("equal"); ax[1].grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "extrapolation.png"))
    plt.close(fig)

    rng = np.random.default_rng(3)
    R, t = sample_poses("varied", 1, rng)[0]
    R_wc, t_wc = camera_pose_from_board(R, t)
    img, _, _ = render(truth, [BOARD.plane(np.eye(3), np.zeros(3))], R_wc, t_wc,
                       supersample=2)
    mapx, mapy = cv2.initUndistortRectifyMap(
        cam.K, cam.dist.reshape(1, 5), None, cam.K, (640, 480), cv2.CV_32FC1)
    und = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3.1))
    ax[0].imshow(img); ax[0].set_title("as the camera sees it")
    ax[1].imshow(und); ax[1].set_title("undistorted with OUR recovered model")
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "undistort.png"))
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    obs = stage_render()
    cam, _ = stage_baseline(obs)
    stage_traps()
    stage_views()
    stage_model()
    stage_noise()
    stage_undistort(cam, obs)

    keys = sorted({k for r in RESULTS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stage"] + [k for k in keys if k != "stage"],
                           lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
