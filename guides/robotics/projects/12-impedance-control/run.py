"""Project 12 -- Impedance control.

Six experiments on the 6-DoF arm from Phase 1, driven by project 10's dynamics:

  1. does the arm actually have the stiffness you asked for?
  2. soft in one direction, stiff in the others -- the reason to bother
  3. what a wrong gravity model does to a soft spring
  4. the stiffness ceiling your control rate imposes
  5. a stiff wall: position control vs impedance control
  6. choosing the damping: bounce, or treacle

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
sys.path.insert(0, os.path.join(_PROJ, "10-inverse-dynamics-from-scratch"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

import matplotlib.pyplot as plt  # noqa: E402

import dynamics as dyn  # noqa: E402
import impedance as imp  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

URDF = os.path.join(_PROJ, "02-urdf-visualizer", "models", "arm6.urdf")
ARM = dyn.Model(URDF)
N = ARM.n

# A comfortable, well-conditioned posture: the arm reaching forward with the
# elbow bent, tool at (0.69, 0, 0.40).  The smallest singular value of the
# Jacobian here is 0.228 -- picked by scanning tidy postures, because the first
# choice for this project was a nearly straight-up arm whose smallest singular
# value was 0.032, and experiment 1b shows what that did.
Q_HOME = np.array([0.0, 0.50, 1.40, 0.0, 0.90, 0.0])
Q_STRETCHED = np.array([0.0, 0.70, -1.20, 0.0, 0.50, 0.0])  # nearly straight up
T_HOME = dyn.tool_pose(ARM, Q_HOME)
# Orientation stiffness and damping, held fixed except where noted.  DO was
# 6.0 in the first version of this file and that was unstable: the wrist's own
# inertia is about 0.003 kg m^2, so D*dt/I = 6*0.001/0.003 = 2 sits exactly on
# the discrete-time damping stability boundary, and the loop buzzed hard enough
# to report a sideways stiffness 400x the commanded one.  Discrete stability is
# set by the fastest thing in the loop, not by the thing you were thinking about.
KO, DO = 40.0, 2.5


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<56s} {value:>12.5f} {unit}")


def damping_for(k, m_eff=3.0, zeta=1.0):
    """Critical damping for a virtual mass-spring: D = 2 zeta sqrt(k m).

    ``m_eff`` is a stand-in for the arm's apparent mass at the tool.  There is
    no single right number -- it depends on the configuration AND on the
    direction of the push (``cartesian_mass`` below measures both), so one
    constant cannot critically damp all three axes at once.  Experiment 1
    measures what that costs and experiment 6 measures what to do about it.
    """
    return 2.0 * zeta * np.sqrt(np.asarray(k, dtype=float) * m_eff)


def cartesian_mass(q):
    """The arm's apparent mass at the tool, per direction.

    Push the tool and it resists as if it had a mass -- but not the same mass
    in every direction, because different directions of push have to accelerate
    different amounts of arm.  The exact object is the OPERATIONAL-SPACE inertia

        Lambda(q) = ( J M(q)^-1 J^T )^-1

    whose 3x3 translational block is what a hand feels.  It is the Cartesian
    counterpart of the joint-space mass matrix, and it is what a proper
    impedance controller uses to choose its damping.
    """
    J = dyn.tool_jacobian(ARM, q)[:3, :]
    M = dyn.mass_matrix(ARM, q)
    return np.linalg.inv(J @ np.linalg.solve(M, J.T))


# 1 kHz.  An earlier version of this file ran the first three experiments at
# 500 Hz on the argument that the virtual spring's own frequency is only about
# 29 rad/s, so a 500 Hz loop is seventeen times faster than anything it has to
# follow.  That argument is wrong, and experiment 4 is the reason: the binding
# constraint is not the spring, it is the DAMPER acting on the wrist's tiny
# inertia, and at 500 Hz that loop buzzed hard enough to report a stiffness
# 400x the commanded one.  Discrete-time stability is set by the fastest thing
# in the loop, not by the thing you were thinking about.
DT = 1e-3


def push(K, force, T=1.2, gravity_gain=1.0, dt=DT, zeta=1.0, ko=KO, q_home=None):
    """Hold the tool where it starts, apply a constant force, see where it ends up."""
    q_home = Q_HOME if q_home is None else q_home
    T_hold = dyn.tool_pose(ARM, q_home)
    K = np.broadcast_to(np.asarray(K, dtype=float), (3,))
    D = damping_for(K, zeta=zeta)

    def controller(t, q, qd):
        tau, _, _ = imp.impedance_torque(ARM, q, qd, T_hold, np.zeros(6), K, D,
                                         Ko=ko, Do=DO, gravity=False)
        return tau + gravity_gain * dyn.gravity_torque(ARM, q)

    def ext(t, p, v):
        # A 0.4 s ramp instead of a step, so the measurement is of the spring,
        # not of the impulse response to an impossible jump in force.
        return np.concatenate([np.asarray(force, dtype=float) * min(t / 0.4, 1.0), np.zeros(3)])

    ts, P, Q, TAU, F, ok = imp.simulate(ARM, q_home, controller, T=T, dt=dt, ext_fn=ext)
    return ts, P, Q, TAU, F, ok


# ---------------------------------------------------------------------------
# 1. Commanded vs measured stiffness
# ---------------------------------------------------------------------------
def exp1_stiffness():
    print("[1] commanded vs measured stiffness")
    p0 = T_HOME[:3, 3]
    ks = [200.0, 800.0, 2000.0]
    dirs = {"x (forward)": np.array([12.0, 0, 0]),
            "y (sideways)": np.array([0, 12.0, 0]),
            "z (down)": np.array([0, 0, -12.0])}
    table = {}
    # The ORIENTATION spring is switched off (ko=0, damping only) for this
    # measurement.  On this arm, moving the tool sideways requires turning the
    # base, which also turns the tool -- so an orientation spring resists a
    # sideways push too, and its contribution would be counted as translational
    # stiffness.  With it on, the measured sideways stiffness came out 11x the
    # commanded one, which is a true statement about the arm and a useless
    # statement about the spring we set.
    for k in ks:
        for name, f in dirs.items():
            _, P, _, _, _, ok = push(k, f, ko=0.0)
            disp = P[-1] - p0
            along = float(disp @ (f / np.linalg.norm(f)))
            k_meas = float(np.linalg.norm(f) / max(abs(along), 1e-12))
            cross = float(np.linalg.norm(disp - along * f / np.linalg.norm(f)))
            table[(k, name)] = (k_meas, 1e3 * along, 1e3 * cross)
            record("1-stiff", f"K={k:.0f} push {name}: displacement", 1e3 * along, "mm")
            record("1-stiff", f"K={k:.0f} push {name}: measured stiffness", k_meas, "N/m")
            record("1-stiff", f"K={k:.0f} push {name}: measured/commanded", k_meas / k, "x")
            record("1-stiff", f"K={k:.0f} push {name}: sideways slip", 1e3 * cross, "mm")

    errs = [abs(v[0] / k - 1) for (k, _), v in table.items()]
    record("1-stiff", "worst stiffness error across all nine tests", 100 * max(errs), "%")

    # Why the sideways axis misbehaves: the arm is not equally heavy in every
    # direction, and damping_for() assumes it is.
    Lam = cartesian_mass(Q_HOME)
    for i, nm in enumerate(("x (forward)", "y (sideways)", "z (down)")):
        record("1-stiff", f"apparent mass at the tool along {nm}", float(Lam[i, i]), "kg")
    record("1-stiff", "heaviest direction / lightest direction",
           float(np.max(np.diag(Lam)) / np.min(np.diag(Lam))), "x")
    record("1-stiff", "damping assumed by damping_for()", 3.0, "kg")
    record("1-stiff", "actual damping ratio on the sideways axis at K=2000",
           float(np.sqrt(3.0 / Lam[1, 1])), "")

    # 1b.  The same 800 N/m command, from a posture where the arm is nearly
    # straight up.  The tool is then almost ON the base's turning axis, so the
    # shoulder has almost no lever arm to push sideways with -- and the spring
    # the operator asked for is not the spring they get.
    sig_good = float(np.linalg.svd(dyn.tool_jacobian(ARM, Q_HOME), compute_uv=False)[-1])
    sig_bad = float(np.linalg.svd(dyn.tool_jacobian(ARM, Q_STRETCHED), compute_uv=False)[-1])
    record("1-stiff", "smallest singular value, working posture", sig_good, "")
    record("1-stiff", "smallest singular value, nearly-straight-up posture", sig_bad, "")
    p_bad = dyn.tool_pose(ARM, Q_STRETCHED)[:3, 3]
    _, P2, _, TAU2, _, _ = push(800.0, np.array([0, 12.0, 0]), q_home=Q_STRETCHED)
    d2 = P2[-1] - p_bad
    record("1-stiff", "straight-up posture, 12 N sideways: tool moved",
           1e3 * float(np.linalg.norm(d2)), "mm")
    record("1-stiff", "  effective stiffness there", 12.0 / max(float(np.linalg.norm(d2)), 1e-9), "N/m")
    record("1-stiff", "  joint torques saturated",
           100 * float(np.mean(np.abs(np.abs(TAU2) - ARM.tau_max) < 1e-9)), "% of ticks")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for j, name in enumerate(dirs):
        axes[0].plot(ks, [table[(k, name)][0] for k in ks], "o-", color=COLORS[j], label=name)
    axes[0].plot(ks, ks, "--", color=COLORS[6], lw=1.2, label="ideal")
    axes[0].set_xlabel("commanded stiffness (N/m)")
    axes[0].set_ylabel("measured stiffness (N/m)")
    axes[0].set_title("The arm really is as stiff as you asked")
    axes[0].legend(fontsize=8)
    for j, name in enumerate(dirs):
        axes[1].plot(ks, [abs(table[(k, name)][2]) for k in ks], "o-", color=COLORS[j], label=name)
    axes[1].set_xlabel("commanded stiffness (N/m)")
    axes[1].set_ylabel("sideways slip (mm)")
    axes[1].set_title("How much the tool slides across the push")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "stiffness.png"))


# ---------------------------------------------------------------------------
# 2. Anisotropic stiffness
# ---------------------------------------------------------------------------
def exp2_anisotropic():
    print("[2] soft in one direction, stiff in the others")
    p0 = T_HOME[:3, 3]
    K = np.array([2500.0, 2500.0, 150.0])
    out = {}
    for name, f in (("push down (soft axis)", np.array([0, 0, -12.0])),
                    ("push forward (stiff axis)", np.array([12.0, 0, 0]))):
        _, P, _, _, _, _ = push(K, f)
        d = P[-1] - p0
        out[name] = d
        record("2-aniso", f"{name}: displacement along the push",
               1e3 * float(d @ (f / np.linalg.norm(f))), "mm")
    record("2-aniso", "commanded stiffness ratio (x / z)", K[0] / K[2], "x")
    ratio = abs(out["push down (soft axis)"][2]) / abs(out["push forward (stiff axis)"][0])
    record("2-aniso", "measured displacement ratio", ratio, "x")

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    for j, (name, d) in enumerate(out.items()):
        ax.bar(j, 1e3 * np.linalg.norm(d), color=COLORS[j])
        ax.text(j, 1e3 * np.linalg.norm(d), f"{1e3 * np.linalg.norm(d):.1f} mm",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["12 N down\n(Kz = 150 N/m)", "12 N forward\n(Kx = 2500 N/m)"], fontsize=8)
    ax.set_ylabel("tool displacement (mm)")
    ax.set_title("One arm, two stiffnesses, at the same instant")
    save(fig, os.path.join(OUT, "anisotropic.png"))


# ---------------------------------------------------------------------------
# 3. Gravity model quality
# ---------------------------------------------------------------------------
def exp3_gravity():
    print("[3] what a wrong gravity model costs a soft spring")
    p0 = T_HOME[:3, 3]
    K = 1200.0
    sags = []
    gains = [0.0, 0.5, 0.9, 1.0, 1.1]
    for g in gains:
        _, P, _, _, _, _ = push(K, np.zeros(3), gravity_gain=g, T=1.5)
        sag = 1e3 * float(P[-1, 2] - p0[2])
        sags.append(sag)
        record("3-gravity", f"gravity model x{g:.2f}: vertical sag", sag, "mm")
    # The prediction: an alpha-fraction gravity model leaves (1-alpha) of the
    # weight for the spring to hold, so sag = -(1-alpha) * W / K.
    W = -(sags[0] / 1e3) * K
    record("3-gravity", "apparent weight carried at the tool", W, "N")
    record("3-gravity", "predicted sag at gravity x0.5", -0.5 * W / K * 1e3, "mm")
    record("3-gravity", "sag with a perfect model", sags[3], "mm")

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.plot(gains, sags, "o-", color=COLORS[0], label="measured")
    ax.plot(gains, [-(1 - g) * W / K * 1e3 for g in gains], "--", color=COLORS[6],
            label="-(1 - alpha) W / K")
    ax.axhline(0, color=COLORS[2], lw=1.0)
    ax.set_xlabel("gravity model quality (1.0 = perfect)")
    ax.set_ylabel("vertical sag (mm)")
    ax.set_title("A soft spring cannot tell a modelling error from a payload")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "gravity.png"))


# ---------------------------------------------------------------------------
# 4. The stiffness ceiling the control rate imposes
# ---------------------------------------------------------------------------
def exp4_ceiling():
    print("[4] the stiffness ceiling your control rate imposes")
    ks = [500.0, 2000.0, 8000.0, 25000.0, 60000.0]
    rates = [1000, 200, 100]
    grid = {}
    for f in rates:
        dt = 1.0 / f
        for k in ks:
            D = damping_for(k)
            def controller(t, q, qd, k=k, D=D):
                tau, _, _ = imp.impedance_torque(ARM, q, qd, T_HOME, np.zeros(6), k, D,
                                                 Ko=KO, Do=DO)
                return tau
            q0 = Q_HOME + np.array([0.0, 0.02, 0.0, 0.0, 0.0, 0.0])
            ts, P, Q, TAU, F, ok = imp.simulate(ARM, q0, controller, T=1.2, dt=dt)
            wobble = 1e3 * float(np.std(P[len(P) // 2:, 2]))
            stable = ok and np.isfinite(wobble) and wobble < 1.0
            grid[(f, k)] = (stable, min(wobble, 1e4))
            record("4-ceiling", f"{f} Hz K={k:.0f}: settled wobble", min(wobble, 1e4), "mm")
        highest = max([k for k in ks if grid[(f, k)][0]], default=0.0)
        record("4-ceiling", f"{f} Hz: highest stable stiffness", highest, "N/m")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for j, f in enumerate(rates):
        ys = [grid[(f, k)][1] for k in ks]
        ax.loglog(ks, ys, "o-", color=COLORS[j], label=f"{f} Hz")
        for k in ks:
            ax.plot(k, grid[(f, k)][1], "o", ms=8,
                    color=COLORS[2] if grid[(f, k)][0] else COLORS[1])
    ax.axhline(1.0, color=COLORS[6], ls="--", lw=1.0)
    ax.set_xlabel("commanded stiffness (N/m)")
    ax.set_ylabel("residual wobble (mm)")
    ax.set_title("Green = quiet, orange = buzzing.  Slower loops top out sooner")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "ceiling.png"))


# ---------------------------------------------------------------------------
# 5. Hitting a wall
# ---------------------------------------------------------------------------
def exp5_wall():
    print("[5] a stiff wall")
    p0 = T_HOME[:3, 3]
    wall_x = p0[0] + 0.01  # the wall is 1 cm in front of the tool
    # The reference is commanded 5 cm PAST the wall -- the classic mistake:
    # the geometry was measured wrong, or the part is thicker than the drawing.
    T_ref = T_HOME.copy()
    T_ref[0, 3] += 0.06

    def ext(t, p, v):
        return imp.wall_force(p, v, wall_x)

    out = {}
    # A soft spring: 150 N/m x 5 cm of commanded overshoot = 7.5 N, which is a
    # push, not a crash.  Choosing K here IS choosing the contact force.
    K = 150.0
    D = damping_for(K)

    def imp_ctrl(t, q, qd):
        tau, _, _ = imp.impedance_torque(ARM, q, qd, T_ref, np.zeros(6), K, D, Ko=KO, Do=DO)
        return tau

    # The position baseline: a joint-space PD stiff enough to track well, given
    # the same reference through inverse kinematics.  We reuse Phase 1's IK.
    sys.path.insert(0, os.path.join(_PROJ, "05-damped-least-squares-ik"))
    import ik as ik_mod  # noqa: E402
    q_ref, ik_info = ik_mod.ik(ARM.robot, Q_HOME, T_ref)
    record("5-wall", "IK residual for the position baseline's target",
           1e3 * float(np.linalg.norm(dyn.tool_pose(ARM, q_ref)[:3, 3] - T_ref[:3, 3])), "mm")
    kp_j = np.array([600.0, 600.0, 400.0, 120.0, 120.0, 80.0])
    kd_j = np.array([60.0, 60.0, 40.0, 12.0, 12.0, 8.0])

    def pos_ctrl(t, q, qd):
        return imp.joint_pd_torque(ARM, q, qd, q_ref, np.zeros(N), kp_j, kd_j)

    for label, ctrl in (("stiff joint PD", pos_ctrl), ("impedance", imp_ctrl)):
        ts, P, Q, TAU, F, ok = imp.simulate(ARM, Q_HOME, ctrl, T=1.8, dt=DT, ext_fn=ext)
        peak = float(np.max(np.abs(F[:, 0])))
        steady = float(np.mean(np.abs(F[-300:, 0])))
        out[label] = (ts, P, F)
        record("5-wall", f"{label}: peak contact force", peak, "N")
        record("5-wall", f"{label}: steady contact force", steady, "N")
        record("5-wall", f"{label}: penetration into the wall",
               1e3 * float(np.max(P[:, 0] - wall_x)), "mm")
    record("5-wall", "peak force ratio (position / impedance)",
           float(np.max(np.abs(out["stiff joint PD"][2][:, 0]))
                 / np.max(np.abs(out["impedance"][2][:, 0]))), "x")
    record("5-wall", "impedance steady force predicted K * overshoot", K * 0.05, "N")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for j, (label, (ts, P, F)) in enumerate(out.items()):
        axes[0].plot(ts, np.abs(F[:, 0]), color=COLORS[j], label=label)
        axes[1].plot(ts, 1e3 * (P[:, 0] - wall_x), color=COLORS[j], label=label)
    axes[0].set_ylabel("contact force (N)")
    axes[0].set_title("Commanded 5 cm past a wall it cannot see")
    axes[1].axhline(0, color=COLORS[6], lw=1.0)
    axes[1].set_ylabel("tool position past the wall (mm)")
    axes[1].set_title("How far the tool pushes into it")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "wall.png"))


# ---------------------------------------------------------------------------
# 6. Choosing the damping
# ---------------------------------------------------------------------------
def exp6_damping():
    print("[6] choosing the damping")
    p0 = T_HOME[:3, 3]
    K = 800.0
    out = {}
    for zeta in (0.3, 1.0, 2.5):
        ts, P, _, _, _, _ = push(K, np.array([0, 0, -15.0]), T=1.8, zeta=zeta)
        # Release the force halfway and watch the return.
        d = 1e3 * (P[:, 2] - p0[2])
        out[zeta] = (ts, d)
        settle_idx = np.nonzero(np.abs(d - d[-1]) > 0.5)[0]
        record("6-damping", f"zeta={zeta:.1f}: overshoot past the final position",
               float(np.max(np.abs(d)) - abs(d[-1])), "mm")
        record("6-damping", f"zeta={zeta:.1f}: time to settle within 0.5 mm",
               float(ts[settle_idx[-1]]) if len(settle_idx) else 0.0, "s")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for j, (zeta, (ts, d)) in enumerate(out.items()):
        ax.plot(ts, d, color=COLORS[j], label=f"zeta = {zeta}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("tool displacement (mm)")
    ax.set_title("A 15 N push, four damping choices")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "damping.png"))


def main():
    t0 = time.perf_counter()
    exp1_stiffness()
    exp2_anisotropic()
    exp3_gravity()
    exp4_ceiling()
    exp5_wall()
    exp6_damping()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
