"""Project 04 -- Jacobian from scratch.

  1. analytic vs MuJoCo             -> should agree to ~1e-15
  2. analytic vs finite difference  -> should agree to ~1e-6, and NOT better
  3. why 1e-6 is the ceiling: sweep the step size and watch the two error terms
  4. does J mean what it claims?  integrate q_dot and check the tool moved
  5. the branching trap: a camera bracket that the tool Jacobian must ignore
  6. the unit trap: "the condition number of J" is not a property of the robot

Runs in about 8 seconds on a CPU.
"""

import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for rel in ("01-transform-calculator", "02-urdf-visualizer", "03-forward-kinematics-from-scratch"):
    sys.path.insert(0, os.path.join(HERE, "..", rel))

import transforms as tf  # noqa: E402
from fk import fk_all  # noqa: E402
from jacobian import (  # noqa: E402
    condition_number, jacobian_analytic, jacobian_fd, manipulability, singular_values,
)
from plot_style import COLORS, save, use_style  # noqa: E402
from urdf import load_urdf  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(HERE, "outputs")
MODELS = os.path.join(HERE, "..", "02-urdf-visualizer", "models")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []


def record(name, value, unit=""):
    RESULTS.append({"quantity": name, "value": value, "unit": unit})
    print(f"    {name:<58s} {value:>12.4e} {unit}")


# ---------------------------------------------------------------------------
# 1. Analytic vs MuJoCo
# ---------------------------------------------------------------------------
def vs_mujoco(robot, path, n=500, seed=0):
    """Compare against MuJoCo -- but NOT at ``tool0``.

    MuJoCo welds fixed joints away when it imports a URDF, so ``tool0`` and
    ``camera_link`` simply do not exist as bodies in its model; asking for
    their Jacobian returns a silent block of zeros rather than an error.  We
    therefore check the deepest link that MuJoCo *does* keep, and check the
    tool separately with :func:`adjoint_check`, which is an exact identity.
    """
    import mujoco

    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    link = robot.movable[-1].child  # the last link MuJoCo still models as a body
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
    assert bid >= 0, f"{link} was welded away too"
    rng = np.random.default_rng(seed)

    worst = 0.0
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(n):
        q = robot.random_q(rng)
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_jacBody(m, d, jp, jr, bid)
        J_mj = np.vstack([jp, jr])
        worst = max(worst, float(np.abs(J_mj - jacobian_analytic(robot, q, link)).max()))
    record(f"{robot.name}: worst analytic-vs-MuJoCo entry at '{link}' ({n} configs)", worst)
    return worst


def adjoint_check(robot, n=500, seed=0):
    """The tool Jacobian must equal the wrist Jacobian plus a lever-arm term.

    A rigid body has ONE angular velocity, so the bottom three rows are shared.
    A point offset by ``p`` from the wrist picks up the extra linear velocity
    ``w x p``.  In matrix form that is exactly ``J_tool[:3] = J_wrist[:3] -
    skew(p) J_wrist[3:]``.  Verifying it to machine precision covers the part
    of the chain MuJoCo cannot see.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    wrist = robot.movable[-1].child
    for _ in range(n):
        q = robot.random_q(rng)
        poses = fk_all(robot, q)
        p = poses["tool0"][:3, 3] - poses[wrist][:3, 3]
        Jw = jacobian_analytic(robot, q, wrist)
        expect = np.vstack([Jw[:3] - tf.skew(p) @ Jw[3:], Jw[3:]])
        worst = max(worst, float(np.abs(expect - jacobian_analytic(robot, q, "tool0")).max()))
    record(f"{robot.name}: worst error in the lever-arm identity for tool0", worst)


# ---------------------------------------------------------------------------
# 2 & 3. Analytic vs finite difference, and the step-size sweep
# ---------------------------------------------------------------------------
def vs_finite_difference(robot, link="tool0", n=200, seed=1, h=1e-6):
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        q = robot.random_q(rng)
        worst = max(worst, float(np.abs(jacobian_analytic(robot, q, link) - jacobian_fd(robot, q, link, h)).max()))
    record(f"{robot.name}: worst analytic-vs-finite-difference entry (h = {h:g})", worst)
    record(f"{robot.name}: matching decimal places", -np.log10(max(worst, 1e-300)))
    return worst


def stepsize_sweep(robot, link="tool0", n=25, seed=2):
    """The classic U curve: truncation error falls with h, round-off rises."""
    rng = np.random.default_rng(seed)
    Q = robot.random_q(rng, n)
    hs = np.logspace(-12, -1, 34)
    err_c, err_f = [], []
    for h in hs:
        ec = ef = 0.0
        for q in Q:
            Ja = jacobian_analytic(robot, q, link)
            ec = max(ec, float(np.abs(Ja - jacobian_fd(robot, q, link, h, central=True)).max()))
            ef = max(ef, float(np.abs(Ja - jacobian_fd(robot, q, link, h, central=False)).max()))
        err_c.append(ec)
        err_f.append(ef)
    err_c, err_f = np.array(err_c), np.array(err_f)

    eps = np.finfo(float).eps
    record("best central-difference error", err_c.min())
    record("step size at the central-difference optimum", float(hs[err_c.argmin()]), "rad")
    record("theory: eps^(1/3) for central differences", float(eps ** (1 / 3)), "rad")
    record("best forward-difference error", err_f.min())
    record("step size at the forward-difference optimum", float(hs[err_f.argmin()]), "rad")
    record("theory: sqrt(eps) for forward differences", float(np.sqrt(eps)), "rad")

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.loglog(hs, err_c, "o-", ms=3, color=COLORS[0], label="central difference  (f(q+h) - f(q-h)) / 2h")
    ax.loglog(hs, err_f, "o-", ms=3, color=COLORS[1], label="forward difference  (f(q+h) - f(q)) / h")
    ax.axvline(eps ** (1 / 3), color=COLORS[0], ls=":", lw=1.2)
    ax.axvline(np.sqrt(eps), color=COLORS[1], ls=":", lw=1.2)
    ax.text(eps ** (1 / 3) * 1.2, 3e-3, r"$\epsilon^{1/3}$", color=COLORS[0], fontsize=8)
    ax.text(np.sqrt(eps) * 1.2, 3e-5, r"$\sqrt{\epsilon}$", color=COLORS[1], fontsize=8)
    ax.set_xlabel("finite-difference step h (rad)")
    ax.set_ylabel("worst Jacobian entry error")
    ax.set_title("Round-off on the left, truncation on the right;\n"
                 "the best you can buy is ~1e-10 and it is NOT at a small h")
    ax.legend(fontsize=8, loc="upper center")
    save(fig, f"{OUT}/fd_stepsize.png")


# ---------------------------------------------------------------------------
# 4. Does J mean what it claims?
# ---------------------------------------------------------------------------
def velocity_check(robot, link="tool0", seed=3):
    """Move the joints for a short time and see whether the tool went where J said.

    ``J`` is a DERIVATIVE, so it is exact only in the limit.  Predicting a
    finite step should leave an error proportional to dt^2.  Measuring that
    slope is a much stronger test than "the numbers look similar": it confirms
    J is the right first-order term, not merely a close-by matrix.
    """
    rng = np.random.default_rng(seed)
    q = robot.random_q(rng)
    qd = rng.normal(size=robot.n)
    qd /= np.linalg.norm(qd)
    J = jacobian_analytic(robot, q, link)
    twist = J @ qd  # predicted (v, w)

    dts = np.logspace(-6, -1, 26)
    errs = []
    T0 = fk_all(robot, q)[link]
    for dt in dts:
        T1 = fk_all(robot, q + qd * dt)[link]
        dp_true = T1[:3, 3] - T0[:3, 3]
        dw_true = tf.R_to_axis_angle(T1[:3, :3] @ T0[:3, :3].T)
        pred = twist * dt
        errs.append(float(np.linalg.norm(np.concatenate([dp_true, dw_true]) - pred)))
    errs = np.array(errs)
    # Fit a slope over the clean middle of the range.
    mid = (dts > 1e-4) & (dts < 1e-2)
    slope = float(np.polyfit(np.log10(dts[mid]), np.log10(errs[mid]), 1)[0])
    record("log-log slope of the prediction error vs dt (theory: 2)", slope)
    record("prediction error after a 1 ms step", float(errs[np.argmin(np.abs(dts - 1e-3))]), "m+rad")

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.loglog(dts, errs, "o-", ms=3, color=COLORS[0], label=f"measured (slope {slope:.2f})")
    ax.loglog(dts, errs[-1] * (dts / dts[-1]) ** 2, "--", color="#999999", label=r"reference slope 2 ($dt^2$)")
    ax.set_xlabel("integration step dt (s)")
    ax.set_ylabel("|actual tool motion - J q_dot dt|")
    ax.set_title("The Jacobian is exactly the first-order term")
    ax.legend(fontsize=8)
    save(fig, f"{OUT}/velocity_check.png")


# ---------------------------------------------------------------------------
# 5. The branching trap
# ---------------------------------------------------------------------------
def branching_trap(robot, seed=4):
    """On testarm, joints 3-5 move the tool but CANNOT move the camera bracket."""
    rng = np.random.default_rng(seed)
    q = robot.random_q(rng)
    J_tool = jacobian_analytic(robot, q, "tool0")
    J_cam = jacobian_analytic(robot, q, "camera_link")
    dead = [robot.joint_names[i] for i in range(robot.n) if np.abs(J_cam[:, i]).max() < 1e-15]
    print(f"    joints that provably cannot move camera_link: {dead}")
    record("columns of the camera Jacobian that are exactly zero", float(len(dead)))
    record("columns of the tool Jacobian that are exactly zero",
           float(sum(1 for i in range(robot.n) if np.abs(J_tool[:, i]).max() < 1e-15)))
    # Confirm it physically: move only those joints and watch the camera not move.
    qd = np.zeros(robot.n)
    for i in range(robot.n):
        if robot.joint_names[i] in dead:
            qd[i] = 1.0
    T0 = fk_all(robot, q)["camera_link"]
    T1 = fk_all(robot, q + 0.3 * qd)["camera_link"]
    record("camera motion when only those joints turn by 0.3 rad",
           float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])), "m")


# ---------------------------------------------------------------------------
# 6. The unit trap
# ---------------------------------------------------------------------------
def unit_trap(robot, seed=5):
    """Rescale lengths and the condition number changes -- the robot does not.

    Rows 0-2 of J are in metres per radian; rows 3-5 are dimensionless.  Adding
    them up inside an SVD is like adding kilograms to seconds.  Whatever number
    comes out depends on your choice of unit, so no single number of that kind
    is a property of the robot.
    """
    rng = np.random.default_rng(seed)
    Q = robot.random_q(rng, 200)
    scales = {"metres": 1.0, "centimetres": 100.0, "millimetres": 1000.0, "kilometres": 1e-3}
    for name, s in scales.items():
        cn, mp = [], []
        for q in Q:
            J = jacobian_analytic(robot, q)
            J = np.vstack([J[:3] * s, J[3:]])
            cn.append(condition_number(J))
            mp.append(manipulability(J))
        record(f"median condition number with lengths in {name}", float(np.median(cn)))
        record(f"median manipulability with lengths in {name}", float(np.median(mp)))


# ---------------------------------------------------------------------------
# figures + timing
# ---------------------------------------------------------------------------
def fig_structure(robot, seed=6):
    rng = np.random.default_rng(seed)
    q = robot.random_q(rng)
    J = jacobian_analytic(robot, q)
    s = singular_values(J)

    fig, axs = plt.subplots(1, 2, figsize=(8.0, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
    im = axs[0].imshow(J, cmap="RdBu_r", vmin=-np.abs(J).max(), vmax=np.abs(J).max(), aspect="auto")
    axs[0].set_xticks(range(robot.n))
    axs[0].set_xticklabels(robot.joint_names, rotation=40, ha="right", fontsize=7)
    axs[0].set_yticks(range(6))
    axs[0].set_yticklabels(["v_x (m/s)", "v_y (m/s)", "v_z (m/s)", "w_x (rad/s)", "w_y (rad/s)", "w_z (rad/s)"],
                           fontsize=7)
    axs[0].set_title(f"{robot.name}: J(q) at one configuration\n"
                     "top three rows and bottom three rows have DIFFERENT units", fontsize=8.5)
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], shrink=0.85)

    axs[1].bar(range(1, 7), s, color=COLORS[0])
    axs[1].set_xlabel("singular value index")
    axs[1].set_ylabel("singular value")
    axs[1].set_title(f"6 singular values, {robot.n} joints\n"
                     f"condition number {condition_number(J):.1f}", fontsize=8.5)
    save(fig, f"{OUT}/jacobian_structure.png")


def timing(robot, n=400, seed=7):
    rng = np.random.default_rng(seed)
    Q = robot.random_q(rng, n)
    t0 = time.perf_counter()
    for q in Q:
        jacobian_analytic(robot, q)
    t_a = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for q in Q:
        jacobian_fd(robot, q)
    t_f = (time.perf_counter() - t0) / n
    record(f"{robot.name}: analytic Jacobian", t_a * 1e6, "us")
    record(f"{robot.name}: finite-difference Jacobian", t_f * 1e6, "us")
    record(f"{robot.name}: finite difference is slower by", t_f / t_a, "x")


def main():
    arm7 = load_urdf(os.path.join(MODELS, "arm7.urdf"))
    arm6 = load_urdf(os.path.join(MODELS, "arm6.urdf"))
    testarm = load_urdf(os.path.join(MODELS, "testarm.urdf"))

    print("\n[1] analytic vs MuJoCo, and the lever-arm identity for the tool")
    for r, f in ((arm6, "arm6.urdf"), (arm7, "arm7.urdf"), (testarm, "testarm.urdf")):
        vs_mujoco(r, os.path.join(MODELS, f))
        adjoint_check(r)

    print("\n[2] analytic vs finite difference")
    for r in (arm6, arm7, testarm):
        vs_finite_difference(r)

    print("\n[3] step-size sweep (arm7)")
    stepsize_sweep(arm7)

    print("\n[4] does J predict real motion?")
    velocity_check(arm7)

    print("\n[5] the branching trap (testarm)")
    branching_trap(testarm)

    print("\n[6] the unit trap (arm7)")
    unit_trap(arm7)

    print("\n[timing]")
    timing(arm7)
    fig_structure(arm7)

    with open(f"{OUT}/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "value", "unit"], lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"  wrote {OUT}/results.csv")


if __name__ == "__main__":
    main()
