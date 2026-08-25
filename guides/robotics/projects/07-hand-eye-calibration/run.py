"""Project 07 -- Hand-eye calibration.

  1. noise-free sanity check: the solver must be exact
  2. how many poses do you need?  and how does the error fall?
  3. how does camera noise turn into calibration error?
  4. the failure that looks like success: rotate about one axis only
  5. the ground-truth-free self-checks, and whether they can be trusted
  6. closed form vs Gauss-Newton refinement

Runs in about 25 seconds on a CPU.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ("01-transform-calculator", "02-urdf-visualizer", "03-forward-kinematics-from-scratch",
            "04-jacobian-from-scratch", "05-damped-least-squares-ik"):
    sys.path.insert(0, os.path.join(HERE, "..", rel))

import handeye as he  # noqa: E402
import sim  # noqa: E402
import transforms as tf  # noqa: E402
import viz  # noqa: E402
from fk import fk_all  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402
from urdf import load_urdf  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
MODELS = os.path.join(HERE, "..", "02-urdf-visualizer", "models")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []
POOL = 40
PIXEL_NOISE = 0.3  # a good corner detector on a well-lit tag


def record(name, value, unit=""):
    RESULTS.append({"quantity": name, "value": value, "unit": unit})
    print(f"    {name:<62s} {value:>12.4e} {unit}")


def solve(T_ees, obs):
    AB = he.motion_pairs(T_ees, obs)
    return he.solve_park_martin(AB), AB


# ---------------------------------------------------------------------------
# 1. Sanity
# ---------------------------------------------------------------------------
def sanity(T_ees, T_cts, rng):
    obs, _ = sim.observe(T_cts, rng, pixel_noise=0.0)
    X, AB = solve(T_ees, obs)
    rot, trans = he.pose_error(X, sim.X_TRUE)
    record("noise-free: rotation error", rot, "deg")
    record("noise-free: translation error", trans, "mm")
    record("noise-free: AX=XB rotation residual", he.residual_axxb(X, AB)[0], "deg")
    record("number of motion pairs from 40 poses", float(len(AB)))


# ---------------------------------------------------------------------------
# 2 & 3. How many poses, and how much noise?
# ---------------------------------------------------------------------------
def sweep_poses(T_ees, T_cts, trials=24, seed=1):
    ns = [3, 4, 5, 6, 8, 10, 14, 20, 30, 40]
    rot_med, tr_med, tr_lo, tr_hi = [], [], [], []
    for n in ns:
        rots, trs = [], []
        for t in range(trials):
            rng = np.random.default_rng(seed * 1000 + t)
            idx = rng.permutation(len(T_ees))[:n]
            obs, _ = sim.observe([T_cts[i] for i in idx], rng, PIXEL_NOISE)
            X, _ = solve([T_ees[i] for i in idx], obs)
            r, tt = he.pose_error(X, sim.X_TRUE)
            rots.append(r)
            trs.append(tt)
        rot_med.append(np.median(rots))
        tr_med.append(np.median(trs))
        tr_lo.append(np.percentile(trs, 25))
        tr_hi.append(np.percentile(trs, 75))
        record(f"{n} poses: median rotation error", float(np.median(rots)), "deg")
        record(f"{n} poses: median translation error", float(np.median(trs)), "mm")

    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
    axs[0].loglog(ns, rot_med, "o-", color=COLORS[0])
    ref = rot_med[0] * (np.array(ns) / ns[0]) ** -0.5
    axs[0].loglog(ns, ref, "--", color="#999999", label=r"$1/\sqrt{N}$")
    axs[0].set_xlabel("number of calibration poses")
    axs[0].set_ylabel("rotation error (deg)")
    axs[0].set_title("rotation")
    axs[0].legend(fontsize=8)
    axs[1].loglog(ns, tr_med, "o-", color=COLORS[1])
    axs[1].fill_between(ns, tr_lo, tr_hi, color=COLORS[1], alpha=0.18)
    ref = tr_med[0] * (np.array(ns) / ns[0]) ** -0.5
    axs[1].loglog(ns, ref, "--", color="#999999", label=r"$1/\sqrt{N}$")
    axs[1].set_xlabel("number of calibration poses")
    axs[1].set_ylabel("translation error (mm)")
    axs[1].set_title("translation (shaded: middle half of 24 trials)")
    axs[1].legend(fontsize=8)
    fig.suptitle(f"Averaging works, and it works slowly ({PIXEL_NOISE} px corner noise)", y=1.03)
    save(fig, f"{OUT}/error_vs_poses.png")


def sweep_noise(T_ees, T_cts, trials=16, seed=2):
    noises = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
    rot_med, tr_med = [], []
    for s in noises:
        rots, trs = [], []
        for t in range(trials):
            rng = np.random.default_rng(seed * 1000 + t)
            obs, _ = sim.observe(T_cts[:20], rng, s)
            X, _ = solve(T_ees[:20], obs)
            r, tt = he.pose_error(X, sim.X_TRUE)
            rots.append(r)
            trs.append(tt)
        rot_med.append(np.median(rots))
        tr_med.append(np.median(trs))
        record(f"{s} px corner noise: median translation error", float(np.median(trs)), "mm")

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.loglog(noises, tr_med, "o-", color=COLORS[1], label="translation (mm)")
    ax.loglog(noises, rot_med, "o-", color=COLORS[0], label="rotation (deg)")
    ax.loglog(noises, np.array(tr_med)[0] * np.array(noises) / noises[0], "--",
              color="#999999", label="proportional to noise")
    ax.set_xlabel("corner-detection noise (pixels, 1 sigma)")
    ax.set_ylabel("calibration error")
    ax.set_title("20 poses: the error is simply proportional to the pixel noise")
    ax.legend(fontsize=8)
    save(fig, f"{OUT}/error_vs_noise.png")


# ---------------------------------------------------------------------------
# 4. The failure that looks like success
# ---------------------------------------------------------------------------
def degeneracy(robot, T_ees, T_cts, seed=3):
    rng = np.random.default_rng(seed)
    _, T_ees_d, T_cts_d = sim.collect(robot, 20, rng, degenerate=True)
    record("poses collected in the degenerate (one-axis) set", float(len(T_ees_d)))

    rows = {}
    for name, (Te, Tc) in (("varied viewpoints", (T_ees[:20], T_cts[:20])),
                           ("one axis only", (T_ees_d, T_cts_d))):
        errs = []
        for t in range(16):
            r = np.random.default_rng(500 + t)
            obs, _ = sim.observe(Tc, r, PIXEL_NOISE)
            X, AB = solve(Te, obs)
            errs.append(np.abs(X[:3, 3] - sim.X_TRUE[:3, 3]) * 1e3)
        errs = np.array(errs)
        obs, _ = sim.observe(Tc, np.random.default_rng(999), PIXEL_NOISE)
        X, AB = solve(Te, obs)
        s = he.translation_conditioning(AB)
        rows[name] = {"per_axis": np.median(errs, axis=0), "sv": s,
                      "residual": he.residual_axxb(X, AB)}
        record(f"[{name}] median |dt| along x", float(np.median(errs[:, 0])), "mm")
        record(f"[{name}] median |dt| along y", float(np.median(errs[:, 1])), "mm")
        record(f"[{name}] median |dt| along z", float(np.median(errs[:, 2])), "mm")
        record(f"[{name}] smallest singular value of the translation system", float(s[-1]))
        record(f"[{name}] AX=XB translation residual", rows[name]["residual"][1], "mm")

    fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.0))
    w = 0.35
    x = np.arange(3)
    for k, (name, r) in enumerate(rows.items()):
        axs[0].bar(x + (k - 0.5) * w, r["per_axis"], width=w, color=COLORS[k], label=name)
    axs[0].set_yscale("log")
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(["x", "y", "z"])
    axs[0].set_xlabel("component of the camera offset")
    axs[0].set_ylabel("median error (mm, log)")
    axs[0].set_title("one bad component, two fine ones")
    axs[0].legend(fontsize=7)
    for k, (name, r) in enumerate(rows.items()):
        axs[1].bar(np.arange(3) + (k - 0.5) * w, r["sv"], width=w, color=COLORS[k], label=name)
    axs[1].set_yscale("log")
    axs[1].set_xticks(np.arange(3))
    axs[1].set_xticklabels(["1st", "2nd", "3rd"])
    axs[1].set_xlabel(r"singular values of the stacked $(R_A - I)$")
    axs[1].set_title("and the warning sign, visible before you look\nat any ground truth")
    axs[1].legend(fontsize=7)
    fig.suptitle("Turning about only one axis leaves the offset ALONG that axis unmeasured", y=1.04)
    save(fig, f"{OUT}/degenerate.png")


# ---------------------------------------------------------------------------
# 5. Can the self-check be trusted?  6. Does refinement help?
# ---------------------------------------------------------------------------
def selfcheck_and_refine(T_ees, T_cts, trials=40, seed=4):
    res_rot, res_tr, scat, true_rot, true_tr = [], [], [], [], []
    ref_rot, ref_tr = [], []
    for t in range(trials):
        rng = np.random.default_rng(seed * 1000 + t)
        n = int(rng.integers(4, 25))
        idx = rng.permutation(len(T_ees))[:n]
        Te = [T_ees[i] for i in idx]
        obs, _ = sim.observe([T_cts[i] for i in idx], rng, PIXEL_NOISE)
        X, AB = solve(Te, obs)
        r, tt = he.pose_error(X, sim.X_TRUE)
        rr, rt = he.residual_axxb(X, AB)
        res_rot.append(rr)
        res_tr.append(rt)
        scat.append(he.tag_scatter(X, Te, obs)[0])
        true_rot.append(r)
        true_tr.append(tt)
        Xr, _ = he.refine(X, Te, obs)
        r2, t2 = he.pose_error(Xr, sim.X_TRUE)
        ref_rot.append(r2)
        ref_tr.append(t2)

    c_res = float(np.corrcoef(res_tr, true_tr)[0, 1])
    c_scat = float(np.corrcoef(scat, true_tr)[0, 1])
    record("correlation: AX=XB translation residual vs the TRUE error", c_res)
    record("correlation: tag-prediction scatter vs the TRUE error", c_scat)
    record("closed form: median translation error", float(np.median(true_tr)), "mm")
    record("after Gauss-Newton refinement: median translation error", float(np.median(ref_tr)), "mm")
    record("closed form: median rotation error", float(np.median(true_rot)), "deg")
    record("after Gauss-Newton refinement: median rotation error", float(np.median(ref_rot)), "deg")
    record("refinement improved translation in this fraction of trials",
           float(np.mean(np.array(ref_tr) < np.array(true_tr))))

    fig, axs = plt.subplots(1, 3, figsize=(9.4, 3.0))
    axs[0].plot(res_tr, true_tr, "o", ms=4, color=COLORS[0])
    axs[0].set_xlabel("AX=XB translation residual (mm)")
    axs[0].set_ylabel("TRUE translation error (mm)")
    axs[0].set_title(f"the residual you can compute\nvs the error you cannot   (r = {c_res:.2f})",
                     fontsize=8.5)
    axs[1].plot(scat, true_tr, "o", ms=4, color=COLORS[2])
    axs[1].set_xlabel("spread of the predicted tag position (mm)")
    axs[1].set_ylabel("TRUE translation error (mm)")
    axs[1].set_title(f"the same question, asked in\nmillimetres of the table   (r = {c_scat:.2f})",
                     fontsize=8.5)
    lim = max(max(true_tr), max(ref_tr)) * 1.1
    axs[2].plot([0, lim], [0, lim], "--", color="#999999")
    axs[2].plot(true_tr, ref_tr, "o", ms=4, color=COLORS[1])
    axs[2].set_xlabel("closed form (mm)")
    axs[2].set_ylabel("after refinement (mm)")
    axs[2].set_title("below the line = refinement helped", fontsize=8.5)
    save(fig, f"{OUT}/selfcheck.png")


# ---------------------------------------------------------------------------
# figures of the rig itself
# ---------------------------------------------------------------------------
def draw_frustum(ax, T_cam, depth=0.10, color="#D55E00"):
    fx, fy, cx, cy = sim.K[0, 0], sim.K[1, 1], sim.K[0, 2], sim.K[1, 2]
    corners = []
    for u, v in ((0, 0), (sim.W, 0), (sim.W, sim.H), (0, sim.H)):
        corners.append([(u - cx) / fx * depth, (v - cy) / fy * depth, depth])
    P = np.array(corners) @ T_cam[:3, :3].T + T_cam[:3, 3]
    o = T_cam[:3, 3]
    for p in P:
        ax.plot(*zip(o, p), color=color, lw=0.9)
    ax.plot(*zip(*np.vstack([P, P[:1]])), color=color, lw=0.9)


def fig_setup(robot, qs, T_ees):
    s = sim.TAG_SIZE / 2
    quad = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0], [-s, -s, 0]])
    Q = quad @ sim.T_BASE_TAG[:3, :3].T + sim.T_BASE_TAG[:3, 3]
    fig = plt.figure(figsize=(9.0, 2.6))
    for k in range(4):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        viz.draw_ground(ax, radius=0.42)
        viz.draw_robot(ax, robot, fk_all(robot, qs[k]))
        draw_frustum(ax, T_ees[k] @ sim.X_TRUE, color=COLORS[1])
        ax.plot(Q[:, 0], Q[:, 1], Q[:, 2], color="#111111", lw=2.2)
        viz.style_3d(ax, None, radius=0.44, center=(0.24, 0, 0.30), ticks=False,
                     azim=-64 + 10 * k, elev=22)
    fig.suptitle("Four of the forty calibration poses. Orange = the camera's field of view; "
                 "black square = the tag.", y=0.99)
    fig.subplots_adjust(wspace=-0.2, top=0.92, bottom=-0.04, left=0.0, right=1.0)
    save(fig, f"{OUT}/setup.png")


def fig_tag_views(T_cts, rng):
    fig, axs = plt.subplots(1, 4, figsize=(9.0, 2.5))
    for ax, T_ct in zip(axs, T_cts[:4]):
        clean = sim.project_tag(T_ct)
        noisy = sim.project_tag(T_ct, rng, PIXEL_NOISE * 8)  # exaggerated so it is visible
        ax.plot(*np.vstack([clean, clean[:1]]).T, color=COLORS[0], lw=1.4, label="true corners")
        ax.plot(noisy[:, 0], noisy[:, 1], "x", color=COLORS[1], ms=7, label="detected")
        ax.set_xlim(0, sim.W)
        ax.set_ylim(sim.H, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axs[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle("What the wrist camera sees (corner noise exaggerated 8x so it is visible)", y=1.0)
    save(fig, f"{OUT}/tag_views.png")


def main():
    robot = load_urdf(os.path.join(MODELS, "arm7.urdf"))
    rng = np.random.default_rng(0)
    print(f"\n[0] driving the arm to {POOL} viewpoints that all see the tag")
    qs, T_ees, T_cts = sim.collect(robot, POOL, rng)
    record("calibration poses collected", float(len(T_ees)))
    record("true camera offset: distance from the tool frame",
           float(np.linalg.norm(sim.X_TRUE[:3, 3]) * 1e3), "mm")
    record("true camera offset: rotation away from the tool frame",
           float(np.degrees(tf.rot_angle(sim.X_TRUE[:3, :3]))), "deg")
    fig_setup(robot, qs, T_ees)
    fig_tag_views(T_cts, np.random.default_rng(5))

    print("\n[1] noise-free sanity check")
    sanity(T_ees, T_cts, rng)

    print("\n[2] how many poses do you need?")
    sweep_poses(T_ees, T_cts)

    print("\n[3] how much does camera noise cost?")
    sweep_noise(T_ees, T_cts)

    print("\n[4] the degenerate motion set")
    degeneracy(robot, T_ees, T_cts)

    print("\n[5] self-checks, and refinement")
    selfcheck_and_refine(T_ees, T_cts)

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit"], lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"  wrote {OUT}/results.csv")


if __name__ == "__main__":
    main()
