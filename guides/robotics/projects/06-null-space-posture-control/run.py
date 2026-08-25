"""Project 06 -- Null-space posture control.

  0. prove the projector: it is idempotent, and it moves the tool by ~1e-15
  1. repair a bad posture WITHOUT moving the tool at all
  2. trace the same circle six times with four different secondary tasks
  3. the headline: a closed hand path is not a closed arm path -- unless the
     null space is used to make it one
  4. an honest look at what the secondary tasks do NOT buy

Runs in about 15 seconds on a CPU.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ("01-transform-calculator", "02-urdf-visualizer", "03-forward-kinematics-from-scratch",
            "04-jacobian-from-scratch", "05-damped-least-squares-ik"):
    sys.path.insert(0, os.path.join(HERE, "..", rel))

import numpy.linalg as npl  # noqa: E402
import transforms as tf  # noqa: E402
import viz  # noqa: E402
from fk import fk_all  # noqa: E402
from ik import ik  # noqa: E402
from jacobian import jacobian_analytic  # noqa: E402
from nullspace import (  # noqa: E402
    circle_path, null_projector, secondary_limits, secondary_manipulability, secondary_posture, track,
)
from plot_style import COLORS, save, use_style  # noqa: E402
from urdf import load_urdf  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
MODELS = os.path.join(HERE, "..", "02-urdf-visualizer", "models")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

# The task: draw a circle on a table, tool pointing straight down.
CENTER = np.array([0.45, 0.0, 0.30])
RADIUS = 0.15
R_TOOL = tf.Ry(np.pi)  # flip the tool's z axis to point at the table
LAPS = 6
STEPS_PER_LAP = 260
DT = 0.008

# A posture a human would call comfortable.  Deliberately NOT the middle of
# every joint's travel, so "go home" and "stay away from the limits" really are
# two different requests rather than two names for the same one.
Q_HOME = np.array([0.0, 0.90, 0.0, 2.00, 0.0, -1.00, 0.0])


def record(name, value, unit=""):
    RESULTS.append({"quantity": name, "value": value, "unit": unit})
    print(f"    {name:<64s} {value:>12.4e} {unit}")


def limit_cost(robot, Q):
    """The quantity the joint-limit task actually minimises, averaged over a run."""
    mid = 0.5 * (robot.lower + robot.upper)
    half = 0.5 * (robot.upper - robot.lower)
    return float((((Q - mid) / half) ** 2).mean())


TASKS = {
    "no secondary task": None,
    "posture (go home)": lambda robot, q: secondary_posture(robot, q, Q_HOME, k=4.0),
    "joint-limit avoidance": lambda robot, q: secondary_limits(robot, q, k=12.0),
    "manipulability": lambda robot, q: secondary_manipulability(robot, q, k=12.0),
}


# ---------------------------------------------------------------------------
# 0. Does the projector really project?
# ---------------------------------------------------------------------------
def projector_proof(robot, seed=0):
    rng = np.random.default_rng(seed)
    worst_exact, worst_damped, idem = 0.0, 0.0, 0.0
    for _ in range(300):
        q = robot.random_q(rng)
        J = jacobian_analytic(robot, q)
        qd0 = rng.normal(size=robot.n)
        N0 = null_projector(J, 0.0)
        worst_exact = max(worst_exact, float(npl.norm(J @ (N0 @ qd0))))
        worst_damped = max(worst_damped, float(npl.norm(J @ (null_projector(J, 1e-2) @ qd0))))
        idem = max(idem, float(np.abs(N0 @ N0 - N0).max()))
        rank = npl.matrix_rank(J)
    record("worst tool twist caused by the null-space term (exact projector)", worst_exact)
    record("worst tool twist caused by the null-space term (damped projector, lam=0.01)", worst_damped)
    record("worst |NN - N| (applying a projector twice must change nothing)", idem)
    record("rank of J (6 rows, 7 joints)", float(rank))
    record("dimension of the null space = joints minus rank", float(robot.n - rank))


# ---------------------------------------------------------------------------
# 1. Repair a posture without moving the tool
# ---------------------------------------------------------------------------
def repair_posture(robot, T_target):
    """Plain IK parks joints on their limits.  Fix that without disturbing the tool."""
    q_bad = None
    for seed in range(60):
        rng = np.random.default_rng(seed)
        q, info = ik(robot, robot.random_q(rng), T_target, lam=1e-2, max_iters=400, clamp_limits=True)
        if info["ok"] and robot.limit_margin(q).min() < 0.02:
            q_bad = q
            break
    assert q_bad is not None, "no jammed solution found"
    record("jammed IK solution: smallest joint-limit margin", float(robot.limit_margin(q_bad).min()), "rad")

    hold = [T_target] * 500  # the tool is asked to stand perfectly still
    q_fixed, log = track(robot, q_bad, hold, DT, secondary=TASKS["joint-limit avoidance"],
                         lam=1e-2, clamp=False)
    record("after null-space repair: smallest joint-limit margin",
           float(robot.limit_margin(q_fixed).min()), "rad")
    tool_mm = float(log["err_pos"].max() * 1e3)
    record("worst TOOL position error during the repair", tool_mm, "mm")
    record("worst tool twist the repair itself caused (the projected term)",
           float(log["disturbance"].max()))
    record("how far the ARM moved during the repair", float(npl.norm(q_fixed - q_bad)), "rad")

    fig = plt.figure(figsize=(7.0, 3.0))
    for k, (q, lab) in enumerate([(q_bad, "straight out of IK\n(a joint is on its hard stop)"),
                                  (q_fixed, f"after 4 s of null-space repair\n"
                                            f"(tool never left by more than {tool_mm:.2f} mm)")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        viz.draw_ground(ax, radius=0.42)
        viz.draw_robot(ax, robot, fk_all(robot, q), show_frames=["tool0"], frame_scale=0.10)
        ax.set_title(lab, fontsize=8.5)
        viz.style_3d(ax, None, radius=0.56, center=(0.10, 0, 0.42), ticks=False, azim=-62)
    fig.suptitle("Same tool pose, better arm", y=1.0)
    fig.subplots_adjust(wspace=-0.05, top=0.84, bottom=-0.08, left=0.0, right=1.0)
    save(fig, f"{OUT}/repair.png")
    return q_bad, q_fixed


# ---------------------------------------------------------------------------
# 2 & 3. Six laps of a circle
# ---------------------------------------------------------------------------
def run_tasks(robot, q0, path):
    runs = {}
    for name, sec in TASKS.items():
        # clamp=False on purpose: a limit violation should show up as a
        # violation.  Clamping would silently repair it AND break the
        # projector's guarantee, hiding two things at once.
        _, log = track(robot, q0, path, DT, secondary=sec, lam=1e-2, clamp=False)
        Q = log["q"]
        log["home_dist"] = npl.norm(Q - Q_HOME, axis=1)
        log["start_dist"] = npl.norm(Q - Q[0], axis=1)
        runs[name] = log

        record(f"[{name}] mean tool position error", float(log["err_pos"].mean() * 1e3), "mm")
        record(f"[{name}] worst tool orientation error", float(np.degrees(log["err_rot"].max())), "deg")
        record(f"[{name}] worst tool twist caused by the secondary task", float(log["disturbance"].max()))
        record(f"[{name}] joint-space drift after 1 lap", float(log["start_dist"][STEPS_PER_LAP]), "rad")
        record(f"[{name}] joint-space drift after 6 laps", float(log["start_dist"][-1]), "rad")
        record(f"[{name}] mean distance from the home posture", float(log["home_dist"].mean()), "rad")
        record(f"[{name}] mean joint-limit cost (0 = mid-travel, 1 = at a limit)", limit_cost(robot, Q))
        record(f"[{name}] smallest joint-limit margin", float(log["margin"].min()), "rad")
        record(f"[{name}] mean manipulability", float(log["manip"].mean()))
    return runs


def fig_drift(runs):
    t = np.arange(len(next(iter(runs.values()))["err_pos"])) * DT
    fig, axs = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for k, (name, log) in enumerate(runs.items()):
        axs[0].plot(t, log["start_dist"], color=COLORS[k], label=name, lw=1.6)
    for lap in range(1, LAPS + 1):
        axs[0].axvline(lap * STEPS_PER_LAP * DT, color="#E4E4E4", lw=1.0, zorder=0)
    axs[0].set_xlabel("time (s)   -- grey lines: the tool is exactly back at the start")
    axs[0].set_ylabel("joint-space distance from the start (rad)")
    axs[0].set_title("A closed hand path is not a closed arm path")
    axs[0].legend(fontsize=7)

    laps = np.arange(1, LAPS + 1)
    w = 0.2
    for k, (name, log) in enumerate(runs.items()):
        vals = [log["start_dist"][min(l * STEPS_PER_LAP, len(log["start_dist"]) - 1)] for l in laps]
        axs[1].bar(laps + (k - 1.5) * w, vals, width=w, color=COLORS[k], label=name)
    axs[1].set_xticks(laps)
    axs[1].set_xlabel("after lap number")
    axs[1].set_ylabel("joint-space distance from the start (rad)")
    axs[1].set_title("posture control makes the motion REPEATABLE")
    save(fig, f"{OUT}/drift.png")


def fig_timeseries(runs):
    t = np.arange(len(next(iter(runs.values()))["err_pos"])) * DT
    fig, axs = plt.subplots(2, 2, figsize=(8.6, 4.8))
    panels = [
        ("err_pos", "tool position error (mm)", 1e3, "log",
         "the primary task is untouched\n(all four curves sit on top of each other)"),
        ("home_dist", "distance from the home posture (rad)", 1.0, "linear",
         "only the posture task holds the shape"),
        ("margin", "smallest joint-limit margin (rad)", 1.0, "linear",
         "the binding joint is set by the tool pose,\nso no secondary task can move it"),
        ("manip", "manipulability", 1.0, "linear", "distance from a singularity"),
    ]
    for ax, (key, ylab, scale, yscale, title) in zip(axs.ravel(), panels):
        for k, (name, log) in enumerate(runs.items()):
            ax.plot(t, log[key] * scale, color=COLORS[k], label=name, lw=1.4)
        ax.set_yscale(yscale)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(title, fontsize=8.5)
    axs[0, 0].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("arm7 traces the same circle six times: the tool does not notice, the arm does",
                 y=1.0)
    fig.tight_layout()
    save(fig, f"{OUT}/timeseries.png")


def fig_arms(robot, runs, path):
    """Same point of the circle, first lap versus sixth."""
    P = np.array([T[:3, 3] for T in path[:STEPS_PER_LAP]])
    phase = STEPS_PER_LAP // 4
    fig = plt.figure(figsize=(7.2, 3.2))
    for c, name in enumerate(["no secondary task", "posture (go home)"]):
        Q = runs[name]["q"]
        ax = fig.add_subplot(1, 2, c + 1, projection="3d")
        viz.draw_ground(ax, radius=0.42)
        viz.draw_robot(ax, robot, fk_all(robot, Q[phase]), color="#CFCFCF", alpha=0.55,
                       bone_color="#BBBBBB")
        viz.draw_robot(ax, robot, fk_all(robot, Q[phase + (LAPS - 1) * STEPS_PER_LAP]),
                       show_frames=["tool0"], frame_scale=0.09)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=COLORS[1], lw=1.6)
        drift = npl.norm(Q[phase + (LAPS - 1) * STEPS_PER_LAP] - Q[phase])
        ax.set_title(f"{name}\ngrey = lap 1, blue = lap 6, apart by {drift:.2f} rad", fontsize=8)
        viz.style_3d(ax, None, radius=0.54, center=(0.12, 0, 0.42), ticks=False, azim=-62)
    fig.suptitle("Identical tool pose, one lap apart", y=1.0)
    fig.subplots_adjust(wspace=-0.05, top=0.80, bottom=-0.08, left=0.0, right=1.0)
    save(fig, f"{OUT}/arms.png")


def fig_joint_traces(robot, runs):
    """Which joints actually wander?"""
    t = np.arange(len(runs["no secondary task"]["q"])) * DT
    fig, axs = plt.subplots(2, 4, figsize=(9.6, 4.0), sharex=True)
    for i, ax in enumerate(axs.ravel()):
        if i >= robot.n:
            ax.axis("off")
            continue
        for k, name in enumerate(["no secondary task", "posture (go home)"]):
            ax.plot(t, runs[name]["q"][:, i], color=COLORS[k * 1], lw=1.3, label=name)
        ax.axhline(Q_HOME[i], color="#999999", ls=":", lw=1.0)
        ax.set_title(robot.joint_names[i], fontsize=8)
        ax.tick_params(labelsize=6)
        if i >= 4:
            ax.set_xlabel("time (s)", fontsize=7)
    axs[0, 0].legend(fontsize=6.5, loc="best")
    axs[1, 3].axis("off")
    axs[1, 3].text(0.05, 0.5, "dotted grey =\nthe home value\nfor that joint",
                   fontsize=7.5, transform=axs[1, 3].transAxes, va="center")
    fig.suptitle("Per joint, six laps. Without a secondary task, joints 1, 3, 5 and 7 "
                 "ratchet along the null space.", y=1.0)
    fig.tight_layout()
    save(fig, f"{OUT}/joint_traces.png")


def fig_scoreboard(robot, runs):
    names = list(runs)
    metrics = [
        ("repeatability\n(drift after 6 laps, rad)", lambda log: log["start_dist"][-1], "min"),
        ("posture\n(mean distance from home, rad)", lambda log: log["home_dist"].mean(), "min"),
        ("joint-limit cost\n(0 = mid-travel)", lambda log: limit_cost(robot, log["q"]), "min"),
        ("manipulability\n(higher is better)", lambda log: log["manip"].mean(), "max"),
    ]
    fig, axs = plt.subplots(1, 4, figsize=(10.0, 3.0))
    for ax, (title, fn, sense) in zip(axs, metrics):
        vals = [fn(runs[n]) for n in names]
        best = int(np.argmin(vals)) if sense == "min" else int(np.argmax(vals))
        ax.bar(range(len(names)), vals, color=[COLORS[i] for i in range(len(names))])
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=6)
        ax.set_title(title, fontsize=8.5)
        ax.annotate("best", (best, vals[best]), textcoords="offset points", xytext=(0, 3),
                    ha="center", fontsize=7.5, color="#222222")
    fig.suptitle("Four scoreboards. Each task wins its own -- and the margins on three of them "
                 "are small.", y=1.03)
    save(fig, f"{OUT}/scoreboard.png")


def main():
    robot = load_urdf(os.path.join(MODELS, "arm7.urdf"))
    path = circle_path(CENTER, RADIUS, STEPS_PER_LAP * LAPS + 1, laps=LAPS,
                       R_tool=R_TOOL, plane=("x", "y"))

    print("\n[0] does the projector really project?")
    projector_proof(robot)

    print("\n[1] repairing a jammed posture without moving the tool")
    repair_posture(robot, path[0])

    print("\n[2] six laps of the circle, four secondary tasks")
    q0, info = ik(robot, Q_HOME, path[0], lam=1e-2, max_iters=500, clamp_limits=True)
    assert info["ok"], info
    record("start-of-circle IK residual", info["e_pos"], "m")
    record("start-of-circle smallest joint-limit margin", float(robot.limit_margin(q0).min()), "rad")
    runs = run_tasks(robot, q0, path)

    fig_drift(runs)
    fig_timeseries(runs)
    fig_arms(robot, runs, path)
    fig_joint_traces(robot, runs)
    fig_scoreboard(robot, runs)

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit"], lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"  wrote {OUT}/results.csv")


if __name__ == "__main__":
    main()
