"""Project 03 -- Forward kinematics from scratch.

  1. verify the from-scratch sweep against MuJoCo on three robots
  2. time it
  3. inject five classic frame bugs and measure what each one costs
  4. show that a wrong robot still looks like a robot

Runs in about 6 seconds on a CPU.
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01-transform-calculator"))
sys.path.insert(0, os.path.join(HERE, "..", "02-urdf-visualizer"))

import transforms as tf  # noqa: E402
import viz  # noqa: E402
from fk import BUGS, annotate_axis_norms, fk_all, fk_all_buggy  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402
from urdf import load_urdf  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
MODELS = os.path.join(HERE, "..", "02-urdf-visualizer", "models")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []
ROBOTS = ["arm6", "arm7", "testarm"]


def record(name, value, unit=""):
    RESULTS.append({"quantity": name, "value": value, "unit": unit})
    print(f"    {name:<58s} {value:>12.4e} {unit}")


def load(name):
    path = os.path.join(MODELS, f"{name}.urdf")
    return annotate_axis_norms(load_urdf(path), path), path


# ---------------------------------------------------------------------------
# 1. Verify against MuJoCo
# ---------------------------------------------------------------------------
def verify(robot, path, n=2000, seed=0):
    """Compare every link pose against MuJoCo's C implementation."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)

    # MuJoCo welds fixed joints away, so a link like tool0 is not one of its
    # bodies.  Count and name the ones we could not compare instead of quietly
    # dropping them: an unreported skip is how a "passing" test tests nothing.
    all_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in robot.links}
    body_ids = {n: b for n, b in all_ids.items() if b >= 0}
    absent = sorted(n for n, b in all_ids.items() if b < 0)
    print(f"     comparing {len(body_ids)}/{len(robot.links)} links; "
          f"welded away by MuJoCo: {absent}")
    record(f"{robot.name}: links compared against MuJoCo", float(len(body_ids)))

    p_err, R_err = [], []
    for _ in range(n):
        q = robot.random_q(rng)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mine = fk_all(robot, q)
        for name, bid in body_ids.items():
            p_err.append(float(np.linalg.norm(d.xpos[bid] - mine[name][:3, 3])))
            R_err.append(tf.rot_geodesic(d.xmat[bid].reshape(3, 3), mine[name][:3, :3]))
    p_err, R_err = np.array(p_err), np.array(R_err)
    record(f"{robot.name}: worst position error vs MuJoCo ({n} configs)", p_err.max(), "m")
    record(f"{robot.name}: worst orientation error vs MuJoCo", R_err.max(), "rad")
    record(f"{robot.name}: median position error vs MuJoCo", float(np.median(p_err)), "m")
    return p_err, R_err


def time_fk(robot, n=3000, seed=1):
    rng = np.random.default_rng(seed)
    Q = robot.random_q(rng, n)
    t0 = time.perf_counter()
    for q in Q:
        fk_all(robot, q)
    dt = (time.perf_counter() - t0) / n
    record(f"{robot.name}: one full forward-kinematics sweep", dt * 1e6, "us")
    record(f"{robot.name}: cost per joint", dt * 1e6 / robot.n, "us")


# ---------------------------------------------------------------------------
# 2. The bug study
# ---------------------------------------------------------------------------
def bug_study(robots, n=500, seed=2):
    """For each robot and each bug: how far off is the tool, and would you notice?"""
    rows = {}
    for robot, _ in robots:
        rng = np.random.default_rng(seed)
        Q = robot.random_q(rng, n)
        good = [fk_all(robot, q)["tool0"] for q in Q]
        good_zero = fk_all(robot, np.zeros(robot.n))["tool0"]
        for bug in BUGS:
            bad = [fk_all_buggy(robot, q, bug)["tool0"] for q in Q]
            dp = np.array([np.linalg.norm(a[:3, 3] - b[:3, 3]) for a, b in zip(good, bad)])
            dR = np.array([tf.rot_geodesic(a[:3, :3], b[:3, :3]) for a, b in zip(good, bad)])
            zero = fk_all_buggy(robot, np.zeros(robot.n), bug)["tool0"]
            dp_zero = float(np.linalg.norm(good_zero[:3, 3] - zero[:3, 3]))
            rows[(robot.name, bug)] = {
                "median_mm": float(np.median(dp)) * 1e3,
                "max_mm": float(dp.max()) * 1e3,
                "median_deg": float(np.degrees(np.median(dR))),
                "home_pose_mm": dp_zero * 1e3,
                "silent": dp.max() < 1e-12,
            }
    for (rname, bug), v in rows.items():
        RESULTS.append(
            {"quantity": f"bug[{rname}/{bug}] median tool error", "value": v["median_mm"], "unit": "mm"}
        )
        RESULTS.append(
            {"quantity": f"bug[{rname}/{bug}] error at the home pose", "value": v["home_pose_mm"], "unit": "mm"}
        )
    return rows


def fig_bug_bars(rows, fname):
    bugs = list(BUGS)
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    w = 0.26
    FLOOR = 2e-2  # bottom of the log axis; a "silent" bug has NO error at all
    x = np.arange(len(bugs))
    for k, rname in enumerate(ROBOTS):
        raw = [rows[(rname, b)]["median_mm"] for b in bugs]
        vals = [v if v > 1e-9 else FLOOR for v in raw]   # stubs for the silent ones
        bars = ax.bar(x + (k - 1) * w, vals, width=w, color=COLORS[k], label=rname)
        for b, v in zip(bars, raw):
            if v <= 1e-9:
                ax.text(b.get_x() + w / 2, FLOOR * 1.4, "silent", rotation=90, fontsize=6.5,
                        ha="center", va="bottom", color=COLORS[k])
    ax.set_yscale("log")
    ax.set_ylim(FLOOR * 0.8, 4e3)
    ax.set_xticks(x)
    ax.set_xticklabels([b.replace("_", "\n") for b in bugs], fontsize=7.5)
    ax.set_ylabel("median tool position error (mm, log scale)")
    ax.axhline(1.0, color="#888888", ls="--", lw=1.0)
    ax.text(len(bugs) - 0.52, 1.25, "1 mm", fontsize=7, color="#666666")
    ax.set_title("What each classic frame bug costs -- and which robots hide it")
    ax.legend(fontsize=8, ncol=1, loc="center", bbox_to_anchor=(0.72, 0.30))
    save(fig, f"{OUT}/{fname}")


def fig_bug_overlay(robot, fname):
    """Draw the true robot and three broken ones at the SAME joint values."""
    q = robot.clamp(np.array([0.7, 0.06, -0.9, 0.8, 1.1])[: robot.n])
    good = fk_all(robot, q)
    show = ["transposed_rotation", "swapped_order", "unnormalised_axis"]
    fig = plt.figure(figsize=(8.4, 2.3))
    for k, bug in enumerate([None] + show):
        ax = fig.add_subplot(1, 4, k + 1, projection="3d")
        poses = good if bug is None else fk_all_buggy(robot, q, bug)
        if bug is not None:
            viz.draw_robot(ax, robot, good, color="#CFCFCF", alpha=0.45, bone_color="#BBBBBB")
        viz.draw_robot(ax, robot, poses, color="#7FA8C9" if bug is None else "#E4A28A",
                       show_frames=["tool0"], frame_scale=0.07)
        ax.set_title("correct" if bug is None else bug.replace("_", " "), fontsize=8)
        viz.style_3d(ax, None, radius=0.32, center=(0.06, 0.12, 0.24), ticks=False, azim=-55)
    fig.suptitle("Same joint angles, four different beliefs about where the tool is", y=1.0)
    fig.subplots_adjust(wspace=-0.14, top=0.86, bottom=-0.14, left=-0.01, right=1.01)
    save(fig, f"{OUT}/{fname}")


# ---------------------------------------------------------------------------
# 3. The inverted-transform trap
# ---------------------------------------------------------------------------
def inverted_transform_trap(robot, seed=3):
    """``T_world_camera`` vs ``T_camera_world``: the single most common frame bug.

    A camera sees an object at ``p_cam``.  To place it in the world you need
    ``T_world_camera @ p_cam``.  Using the inverse by mistake type-checks, runs,
    and returns a perfectly reasonable-looking point in the wrong place.
    """
    rng = np.random.default_rng(seed)
    q = robot.random_q(rng)
    T_world_cam = fk_all(robot, q)["camera_link"]
    p_cam = np.array([0.05, -0.02, 0.30, 1.0])  # 30 cm in front of the lens
    right = (T_world_cam @ p_cam)[:3]
    wrong = (tf.T_inv(T_world_cam) @ p_cam)[:3]
    record("inverted-transform trap: distance between right and wrong answer",
           float(np.linalg.norm(right - wrong)), "m")
    record("inverted-transform trap: both answers are inside the workspace",
           float(max(np.linalg.norm(right), np.linalg.norm(wrong))), "m")


def main():
    robots = []
    all_p_err = {}
    for name in ROBOTS:
        robot, path = load(name)
        print(f"\n[{name}] {robot.summary()}")
        robots.append((robot, path))
        p_err, _ = verify(robot, path)
        all_p_err[name] = p_err
        time_fk(robot)

    print("\n[trap] the inverted transform")
    inverted_transform_trap(robots[2][0])

    print("\n[bugs] injecting five classic frame mistakes")
    rows = bug_study(robots)
    print(f"    {'bug':<24s}" + "".join(f"{r:>22s}" for r in ROBOTS))
    for bug in BUGS:
        cells = []
        for rname in ROBOTS:
            v = rows[(rname, bug)]
            cells.append("SILENT" if v["silent"] else "%.1f mm" % v["median_mm"])
        print(f"    {bug:<24s}" + "".join(f"{c:>22s}" for c in cells))
    fig_bug_bars(rows, "bug_study.png")
    fig_bug_overlay(robots[2][0], "bug_overlay.png")

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    for k, name in enumerate(ROBOTS):
        e = np.maximum(all_p_err[name], 1e-19)
        ax.hist(np.log10(e), bins=50, alpha=0.6, color=COLORS[k], label=name)
    ax.set_xlabel("log10 of the link-position disagreement with MuJoCo (m)")
    ax.set_ylabel("link poses")
    ax.set_title("Two independent implementations, ~40,000 link poses, "
                 "agreement at the last bit of a double")
    ax.legend()
    save(fig, f"{OUT}/fk_error.png")

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit"], lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"  wrote {OUT}/results.csv")


if __name__ == "__main__":
    main()
