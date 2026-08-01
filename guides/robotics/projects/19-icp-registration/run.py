"""Project 19 -- ICP: lining up two point clouds, and knowing when it lied.

Seven experiments:

  1. two depth scans -> one aligned cloud, watched iteration by iteration
  2. point-to-point against point-to-plane
  3. the basin of convergence: how wrong may the initial guess be?
  4. the residual does NOT tell you whether the answer is right
  5. partial overlap and outliers, and what trimming buys
  6. degenerate geometry: a flat wall, and the number that warns you
  7. voxel size: accuracy against time

Runs in about four minutes on a CPU.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
for _p in ("16-camera-calibration", "01-transform-calculator"):
    sys.path.insert(0, os.path.join(_PROJ, _p))

from camera import rodrigues                                            # noqa: E402
from scene import room, bare_wall, scan, relative_pose, VIEW_A, VIEW_B  # noqa: E402
from icp import (icp, voxel_downsample, estimate_normals, nearest,      # noqa: E402
                 pose_error, kabsch)
from plot_style import COLORS, use_style                                # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []
OK_ROT, OK_TRANS = 1.0, 20.0          # what counts as a successful alignment


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def make_scans(voxel=0.04, step=4, noise_m=0.0, seed=0):
    planes = room()
    rng = np.random.default_rng(seed)
    A, Ra, ta = scan(planes, VIEW_A, step=step, noise_m=noise_m, rng=rng)
    B, Rb, tb = scan(planes, VIEW_B, step=step, noise_m=noise_m, rng=rng)
    Ad = voxel_downsample(A, voxel)
    Bd = voxel_downsample(B, voxel)
    R_true, t_true = relative_pose(Ra, ta, Rb, tb)
    return Ad, Bd, R_true, t_true


def perturb(R_true, t_true, rot_deg, trans_m, rng):
    """An initial guess that is deliberately wrong by a known amount."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    dR = rodrigues(axis * np.radians(rot_deg))
    d = rng.normal(size=3)
    d = d / np.linalg.norm(d) * trans_m
    return dR @ R_true, t_true + d


# --------------------------------------------------------------------------
# 1 + 2. the pipeline, and the two flavours
# --------------------------------------------------------------------------

def stage_align():
    print("\n[1] two scans in, one alignment out")
    t0 = time.time()
    A, B, R_true, t_true = make_scans()
    N, plan = estimate_normals(B, 16)
    print(f"    scanned and downsampled in {time.time() - t0:.1f} s "
          f"({len(A)} + {len(B)} points)")
    ang_true = np.degrees(np.arccos(np.clip((np.trace(R_true) - 1) / 2, -1, 1)))
    log(dict(stage="setup", pts_a=len(A), pts_b=len(B),
             true_rot_deg=round(float(ang_true), 3),
             true_trans_mm=round(float(np.linalg.norm(t_true)) * 1000, 1),
             start_rms_mm=round(float(np.sqrt((nearest(A, B)[1] ** 2).mean())) * 1000, 1)))

    runs = {}
    for mode in ("point", "plane"):
        t0 = time.time()
        r = icp(A, B, mode=mode, dst_normals=N if mode == "plane" else None,
                max_iter=60, trim=0.9, history=True)
        rot, tr = pose_error(r["R"], r["t"], R_true, t_true)
        runs[mode] = r
        log(dict(stage="align", mode=mode, iters=r["iters"],
                 seconds=round(time.time() - t0, 2),
                 final_rms_mm=round(r["rms"] * 1000, 2),
                 rot_err_deg=round(rot, 4), trans_err_mm=round(tr, 2)))

    fig = plt.figure(figsize=(12.5, 6.4))
    snaps = [0, 1, 3, len(runs["point"]["history"]) - 1]
    for k, it in enumerate(snaps):
        ax = fig.add_subplot(2, 4, k + 1)
        if it == 0:
            P = A
            lbl = "before (identity guess)"
        else:
            _, _, R, t = runs["point"]["history"][it]
            P = A @ R.T + t
            lbl = f"point-to-point, iteration {it}"
        ax.scatter(B[:, 0], B[:, 2], s=0.6, color=COLORS[6], label="target scan")
        ax.scatter(P[:, 0], P[:, 2], s=0.6, color=COLORS[1], label="source scan")
        ax.set_title(lbl, fontsize=9); ax.set_aspect("equal")
        ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
        if k == 0:
            ax.legend(fontsize=7, markerscale=6)
    ax = fig.add_subplot(2, 2, 3)
    for mode, col in (("point", COLORS[1]), ("plane", COLORS[2])):
        h = runs[mode]["history"]
        ax.semilogy([x[0] for x in h], [x[1] * 1000 for x in h], "o-", color=col,
                    ms=3, label=f"point-to-{mode}")
    ax.set_xlabel("iteration"); ax.set_ylabel("mean pair distance (mm)")
    ax.legend(fontsize=8); ax.set_title("convergence")
    ax = fig.add_subplot(2, 2, 4)
    r = runs["plane"]
    P = A @ r["R"].T + r["t"]
    ax.scatter(B[:, 0], B[:, 2], s=0.6, color=COLORS[6])
    ax.scatter(P[:, 0], P[:, 2], s=0.6, color=COLORS[2])
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("point-to-plane, converged")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "align.png"))
    plt.close(fig)
    return A, B, N, R_true, t_true


# --------------------------------------------------------------------------
# 3 + 4. the basin, and the residual that cannot see it
# --------------------------------------------------------------------------

def stage_basin(A, B, N, R_true, t_true):
    print("\n[3] how wrong may the initial guess be?")
    rng = np.random.default_rng(3)
    rows = []
    for rot0 in (0, 5, 10, 20, 30, 45, 60):
        for tr0 in (0.0, 0.10, 0.25, 0.50):
            succ, errs, rmss = 0, [], []
            for k in range(5):
                R0, t0 = perturb(R_true, t_true, rot0, tr0, rng)
                r = icp(A, B, R0=R0, t0=t0, mode="plane", dst_normals=N,
                        max_iter=40, trim=0.9)
                rot, tr = pose_error(r["R"], r["t"], R_true, t_true)
                ok = (rot < OK_ROT) and (tr < OK_TRANS)
                succ += ok
                errs.append(rot)
                rmss.append(r["rms"])
                rows.append((rot0, tr0, rot, tr, r["rms"], ok))
            log(dict(stage="basin", init_rot_deg=rot0, init_trans_m=tr0,
                     success_pct=round(100 * succ / 5, 1),
                     median_rot_err_deg=round(float(np.median(errs)), 3),
                     median_rms_mm=round(float(np.median(rmss)) * 1000, 2)))
    a = np.array([(r[0], r[1], r[2], r[3], r[4], float(r[5])) for r in rows])

    grid = np.zeros((7, 4))
    for i, rot0 in enumerate((0, 5, 10, 20, 30, 45, 60)):
        for j, tr0 in enumerate((0.0, 0.10, 0.25, 0.50)):
            m = (a[:, 0] == rot0) & (a[:, 1] == tr0)
            grid[i, j] = 100 * a[m, 5].mean()

    print("\n[4] does the residual know when it is wrong?")
    good = a[:, 5] > 0.5
    log(dict(stage="residual_vs_truth",
             n=len(a), converged_ok_pct=round(100 * float(good.mean()), 1),
             rms_when_right_mm=round(float(np.median(a[good, 4])) * 1000, 2),
             rms_when_wrong_mm=round(float(np.median(a[~good, 4])) * 1000, 2),
             corr_rms_vs_rot_err=round(float(np.corrcoef(
                 np.log(a[:, 4] + 1e-9), np.log(a[:, 2] + 1e-6))[0, 1]), 3)))
    # how good a detector is a residual threshold?
    for thr_mm in (16, 18, 20, 25):
        flag = a[:, 4] * 1000 > thr_mm
        tp = float((flag & ~good).sum()); fn = float((~flag & ~good).sum())
        fp = float((flag & good).sum())
        log(dict(stage="residual_threshold", threshold_mm=thr_mm,
                 caught_pct_of_failures=round(100 * tp / max(tp + fn, 1), 1),
                 false_alarm_pct_of_successes=round(100 * fp / max(good.sum(), 1), 1)))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    im = ax[0].imshow(grid, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(["0", "10", "25", "50"])
    ax[0].set_yticks(range(7)); ax[0].set_yticklabels(["0", "5", "10", "20", "30", "45", "60"])
    ax[0].set_xlabel("initial translation error (cm)")
    ax[0].set_ylabel("initial rotation error (deg)")
    ax[0].set_title("success rate (%)"); ax[0].grid(False)
    for i in range(7):
        for j in range(4):
            ax[0].text(j, i, f"{grid[i, j]:.0f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax[0], fraction=0.04)
    ax[1].scatter(a[good, 4] * 1000, a[good, 2], s=12, color=COLORS[2], label="converged correctly")
    ax[1].scatter(a[~good, 4] * 1000, a[~good, 2], s=12, color=COLORS[1], label="wrong alignment")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("final residual reported by ICP (mm)")
    ax[1].set_ylabel("true rotation error (deg)")
    ax[1].legend(fontsize=8)
    ax[1].set_title("for THIS failure the residual separates\nthem cleanly (contrast experiment 5)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "basin.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. partial overlap and outliers
# --------------------------------------------------------------------------

def stage_overlap(A, B, R_true, t_true):
    print("\n[5] partial overlap and outliers")
    rng = np.random.default_rng(5)
    for overlap in (1.0, 0.7, 0.5):
        # keep only the part of the target on one side -- a scan that simply
        # did not see the rest of the scene
        thr = np.quantile(B[:, 0], 1 - overlap)
        Bc = B[B[:, 0] >= thr]
        Nc, _ = estimate_normals(Bc, 16)
        for trim in (1.0, 0.8, 0.6):
            r = icp(A, Bc, mode="plane", dst_normals=Nc, max_iter=40, trim=trim)
            rot, tr = pose_error(r["R"], r["t"], R_true, t_true)
            log(dict(stage="overlap", overlap=overlap, trim=trim,
                     rot_err_deg=round(rot, 3), trans_err_mm=round(tr, 2),
                     rms_mm=round(r["rms"] * 1000, 2)))

    Nb, _ = estimate_normals(B, 16)
    for frac in (0.0, 0.05, 0.15, 0.30):
        n = int(len(A) * frac)
        junk = rng.uniform(A.min(0), A.max(0), (n, 3))
        Aj = np.concatenate([A, junk]) if n else A
        for trim in (1.0, 0.8, 0.6):
            r = icp(Aj, B, mode="plane", dst_normals=Nb, max_iter=40, trim=trim)
            rot, tr = pose_error(r["R"], r["t"], R_true, t_true)
            log(dict(stage="outliers", outlier_frac=frac, trim=trim,
                     rot_err_deg=round(rot, 3), trans_err_mm=round(tr, 2)))


# --------------------------------------------------------------------------
# 6. degenerate geometry
# --------------------------------------------------------------------------

def stage_degenerate():
    print("\n[6] a flat wall: perfect residual, arbitrary answer")
    planes = bare_wall()
    A, Ra, ta = scan(planes, (0.30, -0.6, 0.55), target=(0.0, 1.4, 0.5), step=4)
    B, Rb, tb = scan(planes, (0.10, -0.62, 0.60), target=(0.0, 1.4, 0.5), step=4)
    A = voxel_downsample(A, 0.04)
    B = voxel_downsample(B, 0.04)
    R_true, t_true = relative_pose(Ra, ta, Rb, tb)
    N, _ = estimate_normals(B, 16)
    r = icp(A, B, mode="plane", dst_normals=N, max_iter=40, trim=0.9)
    rot, tr = pose_error(r["R"], r["t"], R_true, t_true)
    log(dict(stage="degenerate", scene="bare wall", pts=len(A),
             rms_mm=round(r["rms"] * 1000, 3), rot_err_deg=round(rot, 3),
             trans_err_mm=round(tr, 2),
             smallest_eigenvalue=f"{r['min_eig']:.3e}",
             condition_number=f"{r['cond']:.2e}"))

    A2, B2, Rt2, tt2 = make_scans()
    N2, _ = estimate_normals(B2, 16)
    r2 = icp(A2, B2, mode="plane", dst_normals=N2, max_iter=40, trim=0.9)
    rot2, tr2 = pose_error(r2["R"], r2["t"], Rt2, tt2)
    log(dict(stage="degenerate", scene="floor + backdrop + 3 boxes", pts=len(A2),
             rms_mm=round(r2["rms"] * 1000, 3), rot_err_deg=round(rot2, 3),
             trans_err_mm=round(tr2, 2),
             smallest_eigenvalue=f"{r2['min_eig']:.3e}",
             condition_number=f"{r2['cond']:.2e}"))

    # where exactly does the wall alignment slide?  along the wall, of course
    err_vec = r["t"] - t_true
    wall_normal = N.mean(axis=0)
    wall_normal /= np.linalg.norm(wall_normal)
    along = float(np.linalg.norm(err_vec - np.dot(err_vec, wall_normal) * wall_normal))
    across = float(abs(np.dot(err_vec, wall_normal)))
    log(dict(stage="degenerate_direction", slide_along_wall_mm=round(along * 1000, 2),
             error_across_wall_mm=round(across * 1000, 2)))

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    P = A @ r["R"].T + r["t"]
    ax[0].scatter(B[:, 0], B[:, 1], s=1, color=COLORS[6], label="target")
    ax[0].scatter(P[:, 0], P[:, 1], s=1, color=COLORS[1], label="aligned source")
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)"); ax[0].legend(fontsize=8, markerscale=6)
    ax[0].set_title(f"bare wall: residual {r['rms'] * 1000:.1f} mm,\n"
                    f"translation off by {tr:.0f} mm")
    names = ["bare wall", "floor + boxes"]
    vals = [max(r["min_eig"], 1e-16), max(r2["min_eig"], 1e-16)]
    ax[1].bar([0, 1], vals, color=[COLORS[1], COLORS[2]])
    ax[1].set_yscale("log"); ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("smallest eigenvalue of the 6x6 system")
    ax[1].set_title("the warning that DOES work")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "degenerate.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 7. voxel size
# --------------------------------------------------------------------------

def stage_voxel():
    print("\n[7] voxel size: accuracy against time")
    for v in (0.015, 0.025, 0.04, 0.06, 0.10, 0.16):
        A, B, R_true, t_true = make_scans(voxel=v)
        N, _ = estimate_normals(B, min(16, max(4, len(B) // 20)))
        t0 = time.time()
        r = icp(A, B, mode="plane", dst_normals=N, max_iter=40, trim=0.9)
        rot, tr = pose_error(r["R"], r["t"], R_true, t_true)
        log(dict(stage="voxel", voxel_cm=round(v * 100, 1), pts=len(A),
                 seconds=round(time.time() - t0, 2), iters=r["iters"],
                 rot_err_deg=round(rot, 4), trans_err_mm=round(tr, 2)))


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    A, B, N, R_true, t_true = stage_align()
    stage_basin(A, B, N, R_true, t_true)
    stage_overlap(A, B, R_true, t_true)
    stage_degenerate()
    stage_voxel()

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
