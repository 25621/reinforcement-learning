"""Project 02 -- URDF visualizer.

Load two robot descriptions, print their kinematic trees, draw them at random
joint angles, and check the parse against MuJoCo's independent URDF importer.

Runs in about 25 seconds on a CPU (almost all of it matplotlib drawing).
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "01-transform-calculator"))

import transforms as tf  # noqa: E402  (project 01's toolbox)
import viz  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402
from urdf import load_urdf  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
MODELS = os.path.join(HERE, "models")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []


def record(name, value, unit=""):
    RESULTS.append({"quantity": name, "value": value, "unit": unit})
    v = f"{value:>12.4e}" if isinstance(value, float) else f"{value:>12}"
    print(f"    {name:<52s} {v} {unit}")


# ---------------------------------------------------------------------------
# Forward kinematics in twelve lines.
#
# This is the whole idea: start at the root with the identity, and for every
# joint in parent-before-child order, walk from the parent frame through the
# joint's FIXED offset and then through its MOVING part.  Project 03 rebuilds
# this carefully and proves it correct; here it exists so the picture can be
# drawn.
# ---------------------------------------------------------------------------
def link_poses(robot, q):
    T = {robot.root: np.eye(4)}
    qi = 0
    for j in robot.ordered:
        T_joint = np.eye(4)
        if j.movable:
            if j.jtype == "prismatic":
                T_joint[:3, 3] = j.axis * q[qi]
            else:  # revolute or continuous
                T_joint[:3, :3] = tf.axis_angle_to_R(j.axis * q[qi])
            qi += 1
        T[j.child] = T[j.parent] @ j.T_origin @ T_joint
    return T


# ---------------------------------------------------------------------------
# 1. Parse and describe
# ---------------------------------------------------------------------------
def describe(robot, path):
    print(f"\n[parse] {os.path.basename(path)}")
    print("   ", robot.summary())
    record(f"{robot.name}: movable joints", robot.n)
    record(f"{robot.name}: links", len(robot.links))
    record(f"{robot.name}: total mass", float(sum(l.mass for l in robot.links.values())), "kg")
    span = float(robot.upper.sum() - robot.lower.sum())
    record(f"{robot.name}: total joint range", span, "rad")
    return robot.tree_str()


# ---------------------------------------------------------------------------
# 2. Cross-check the parse against MuJoCo
# ---------------------------------------------------------------------------
def check_against_mujoco(robot, path, n=200, seed=0):
    """Load the same URDF with MuJoCo and compare every link pose.

    MuJoCo has its own, completely separate URDF reader and its own kinematics
    written in C.  If two independent implementations agree to 1e-15 on 200
    random configurations, the parse is right -- a much stronger statement than
    "the picture looks like an arm".
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    rng = np.random.default_rng(seed)

    mj_joint_order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    assert mj_joint_order == robot.joint_names, (mj_joint_order, robot.joint_names)

    # MuJoCo WELDS fixed joints away on import, so links attached by a fixed
    # joint (here: tool0, and the root itself, which becomes 'world') are not
    # bodies in its model.  Naming them explicitly keeps the check honest --
    # a silent "skip if not found" would quietly test nothing at all.
    ids = {name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name in robot.links}
    absent = sorted(k for k, v in ids.items() if v < 0)
    record(f"{robot.name}: links compared against MuJoCo", len(robot.links) - len(absent))
    print(f"     (welded away by MuJoCo, checked in project 04 instead: {absent})")

    worst_p = worst_R = 0.0
    for _ in range(n):
        q = robot.random_q(rng)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mine = link_poses(robot, q)
        for name, T in mine.items():
            bid = ids[name]
            if bid < 0:
                continue
            worst_p = max(worst_p, float(np.abs(d.xpos[bid] - T[:3, 3]).max()))
            worst_R = max(worst_R, tf.rot_geodesic(d.xmat[bid].reshape(3, 3), T[:3, :3]))
    record(f"{robot.name}: worst link position error vs MuJoCo", worst_p, "m")
    record(f"{robot.name}: worst link rotation error vs MuJoCo", worst_R, "rad")
    return worst_p, worst_R


# ---------------------------------------------------------------------------
# 3. Figures
# ---------------------------------------------------------------------------
def fig_zero_pose(robot, radius, center, fname, title):
    fig = plt.figure(figsize=(7.4, 3.0))
    for k, (q, lab) in enumerate(
        [(np.zeros(robot.n), "all joints at 0 (the URDF's own zero pose)"),
         (robot.clamp(np.full(robot.n, 0.6)), "every joint at +0.6 rad")]
    ):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        poses = link_poses(robot, q)
        viz.draw_ground(ax, radius=0.55)
        viz.draw_robot(ax, robot, poses, show_frames=list(poses), frame_scale=0.075)
        viz.style_3d(ax, lab, radius=radius, center=center, ticks=False)
    fig.suptitle(title, y=0.99)
    fig.subplots_adjust(wspace=-0.06, top=0.88, bottom=-0.10, left=0.0, right=1.0)
    save(fig, f"{OUT}/{fname}")


def fig_random_poses(robot, radius, center, fname, title, seed=1, rows=2, cols=4):
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(2.0 * cols, 1.95 * rows))
    for k in range(rows * cols):
        ax = fig.add_subplot(rows, cols, k + 1, projection="3d")
        q = robot.random_q(rng)
        viz.draw_ground(ax, radius=0.5)
        viz.draw_robot(ax, robot, link_poses(robot, q), show_frames=["tool0"], frame_scale=0.10)
        viz.style_3d(ax, None, radius=radius, center=center, azim=-60 + 8 * k, ticks=False)
    fig.suptitle(title, y=0.99)
    fig.subplots_adjust(wspace=-0.25, hspace=-0.1, top=0.94, bottom=0.01, left=0.0, right=1.0)
    save(fig, f"{OUT}/{fname}")


def fig_workspace(robot, fname, title, n=40000, seed=2):
    """Sample joint space, keep the tool positions, and look at the cloud."""
    rng = np.random.default_rng(seed)
    Q_in = robot.random_q(rng, n)
    # The same sampler with the limits thrown away, to price what they cost.
    Q_free = rng.uniform(-np.pi, np.pi, size=(n, robot.n))

    def tool_cloud(Q):
        return np.array([link_poses(robot, q)["tool0"][:3, 3] for q in Q])

    P_in, P_free = tool_cloud(Q_in), tool_cloud(Q_free)

    # A voxel count is a fair SIZE estimate, but it is NOT fair for comparing
    # the two samplers: the limit-free sampler spreads the same 40,000 draws
    # over a joint space many times larger, so it hits fewer voxels even though
    # its reachable set is strictly bigger.  Extremes (furthest, lowest) do not
    # have that bias, so the comparison below uses those.
    h = 0.04
    v_in = len(set(map(tuple, np.floor(P_in / h).astype(int)))) * h**3
    record(f"{robot.name}: reachable volume (voxel estimate, 4 cm, 40k samples)", v_in, "m^3")
    for lab, P in (("within limits", P_in), ("ignoring limits", P_free)):
        record(f"{robot.name}: furthest tool reach, {lab}", float(np.linalg.norm(P, axis=1).max()), "m")
        record(f"{robot.name}: lowest tool height, {lab}", float(P[:, 2].min()), "m")

    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.9))
    r = np.hypot(P_in[:, 0], P_in[:, 1])
    r_f = np.hypot(P_free[:, 0], P_free[:, 1])
    axs[0].plot(r_f[:8000], P_free[:8000, 2], ".", ms=0.7, color="#C9C9C9", label="limits ignored")
    axs[0].plot(r[:8000], P_in[:8000, 2], ".", ms=0.7, color=COLORS[0], label="within limits")
    axs[0].set_xlabel("horizontal distance from base axis (m)")
    axs[0].set_ylabel("height z (m)")
    axs[0].set_title("side view of the workspace")
    axs[0].legend(markerscale=8, fontsize=7)
    axs[0].set_aspect("equal")

    axs[1].plot(P_in[:8000, 0], P_in[:8000, 1], ".", ms=0.7, color=COLORS[0])
    axs[1].set_xlabel("x (m)")
    axs[1].set_ylabel("y (m)")
    axs[1].set_title("top view")
    axs[1].set_aspect("equal")

    axs[2].hist(np.linalg.norm(P_in, axis=1), bins=60, color=COLORS[0], alpha=0.85)
    axs[2].set_xlabel("distance from base (m)")
    axs[2].set_ylabel("samples")
    axs[2].set_title("uniform joint angles are NOT\nuniform tool positions")
    fig.suptitle(title, y=1.10)
    fig.subplots_adjust(wspace=0.42)
    save(fig, f"{OUT}/{fname}")
    return P_in


def fig_frame_chain(robot, fname):
    """One pose, every frame labelled -- the picture the guide's pipeline describes."""
    q = robot.clamp(np.array([0.6, -0.7, 1.2, 0.5, 0.9, 0.3])[: robot.n])
    poses = link_poses(robot, q)
    fig = plt.figure(figsize=(5.4, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    viz.draw_ground(ax, radius=0.5)
    viz.draw_robot(ax, robot, poses, alpha=0.35, show_frames=list(poses), frame_scale=0.11)
    for name, T in poses.items():
        ax.text(*(T[:3, 3] + np.array([0.02, 0.02, 0.03])), name, fontsize=6.5, color="#333333")
    viz.style_3d(ax, "Every link carries its own frame\n(red = x, green = y, blue = z)", radius=0.60, center=(0, 0, 0.52))
    save(fig, f"{OUT}/{fname}")


def main():
    trees = []
    for fname, radius, center in (("arm6.urdf", 0.72, (0, 0, 0.52)), ("arm7.urdf", 0.66, (0, 0, 0.48))):
        path = os.path.join(MODELS, fname)
        robot = load_urdf(path)
        trees.append(f"===== {fname} =====\n{describe(robot, path)}\n")
        check_against_mujoco(robot, path)

        tag = robot.name
        if tag == "arm6":
            fig_zero_pose(robot, radius, center, "arm6_zero.png", "arm6: the same robot, two joint vectors")
            fig_frame_chain(robot, "frames.png")
        fig_random_poses(
            robot, radius, center, f"{tag}_poses.png",
            f"{tag}: eight joint vectors sampled uniformly inside the joint limits",
        )
        fig_workspace(robot, f"{tag}_workspace.png", f"{tag}: where can the tool frame actually go?")

    with open(f"{OUT}/tree.txt", "w") as f:
        f.write("\n".join(trees))
    print(f"  wrote {OUT}/tree.txt")

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit"], lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"  wrote {OUT}/results.csv")


if __name__ == "__main__":
    main()
