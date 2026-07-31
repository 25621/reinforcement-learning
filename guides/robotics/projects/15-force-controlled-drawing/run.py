"""Project 15 -- Force-controlled drawing on a surface the robot cannot see.

Six experiments, all on the same task: drag a pen 20 cm along a gently curved
surface while pressing at a steady 5 N, with a planner that has the surface
wrong.

  1. position control with a 3 mm error, in both directions
  2. impedance control: soft into the surface, stiff along the path
  3. choosing the normal stiffness -- the force/tracking trade-off
  4. adding an integral on the force, and what it costs
  5. a steep surface: world-z compliance vs normal-aligned compliance
  6. drawing faster, until the pen breaks

The pen survives up to 15 N and lifts off below 0.5 N; both are marked on
every force plot.  Runs in about three minutes on a CPU.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
for _p in ("12-impedance-control", "10-inverse-dynamics-from-scratch",
           "05-damped-least-squares-ik", "01-transform-calculator"):
    sys.path.insert(0, os.path.join(_PROJ, _p))

import matplotlib.pyplot as plt  # noqa: E402

import dynamics as dyn  # noqa: E402
import impedance as imp  # noqa: E402
import transforms as tf  # noqa: E402
import ik as ik_mod  # noqa: E402
from surface import Cylinder, BelievedSurface  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

URDF = os.path.join(_PROJ, "02-urdf-visualizer", "models", "arm6.urdf")
ARM = dyn.Model(URDF)
N = ARM.n

F_BREAK = 12.0   # N -- above this the pen snaps
F_LIFT = 0.5     # N -- below this the pen is not touching the paper
F_DES = 5.0      # N -- the pressure we want

X0, X1 = 0.52, 0.68    # the stroke, in x (16 cm)
Y_DRAW = 0.0
KP_INPLANE = 2500.0    # N/m along the path: we know where we want to be
R_TOOL = tf.Ry(np.pi)  # tool pointing straight down


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<56s} {value:>12.5f} {unit}")


def stroke_x(t, T):
    """Position along the stroke: a smooth ramp with zero speed at both ends."""
    s = np.clip(t / T, 0.0, 1.0)
    s = 3 * s ** 2 - 2 * s ** 3  # smoothstep
    return X0 + (X1 - X0) * s


def start_config(surf, z_extra=0.03):
    """An arm configuration with the pen just above the start of the stroke."""
    T_goal = tf.T_from_Rp(R_TOOL, np.array([X0, Y_DRAW, float(surf.height(X0)) + z_extra]))
    q0 = np.array([0.0, 0.50, 1.40, 0.0, 0.90, 0.0])
    q, info = ik_mod.ik(ARM.robot, q0, T_goal)
    return q, T_goal


# How far INTO the believed surface the position-control plan aims.  A plan
# that aims exactly at the surface only grazes it: the arm is not infinitely
# stiff, so it stops short and the pen leaves no line.  Real CAM plans always
# command a small nominal press, and this is it -- chosen so that a PERFECT
# surface model gives about the 5 N we want.
PRESS_NOMINAL = 0.0030


def draw(surf, believed, mode, T=2.0, dt=1e-3, kz=250.0, press=0.020, f_int=0.0,
         align_normal=False, kp_joint=None):
    """One stroke.  ``mode`` is 'position' or 'impedance'.

    'position' commands the arm to follow the BELIEVED surface with a stiff
    joint-space controller -- the naive approach.
    'impedance' hangs a spring between the tool and a reference that sits
    ``press`` metres BELOW the believed surface, so at rest the spring is
    stretched by that much and pushes with ``kz * press`` newtons.  Choosing
    press = F_DES / kz is how you ask for a particular force with a spring.
    """
    q0, _ = start_config(surf)
    q_start = q0.copy()
    state = {"f_err_int": 0.0, "f_n": 0.0}

    # For the position baseline we need a joint trajectory, so solve IK once per
    # 20 ms along the believed surface and interpolate -- exactly what a real
    # "plan then execute" pipeline does.
    if mode == "position":
        ts_plan = np.arange(0.0, T + 0.02, 0.02)
        qs = []
        q_it = q_start.copy()
        for tp in ts_plan:
            x = stroke_x(tp, T)
            p = np.array([x, Y_DRAW, float(believed.height(x)) - PRESS_NOMINAL])
            q_it, _ = ik_mod.ik(ARM.robot, q_it, tf.T_from_Rp(R_TOOL, p))
            qs.append(q_it.copy())
        qs = np.array(qs)
        kpj = np.array([900.0, 900.0, 600.0, 200.0, 200.0, 120.0]) if kp_joint is None else kp_joint
        kdj = 0.1 * kpj

        def controller(t, q, qd, state):
            q_ref = np.array([np.interp(t, ts_plan, qs[:, i]) for i in range(N)])
            return imp.joint_pd_torque(ARM, q, qd, q_ref, np.zeros(N), kpj, kdj)
    else:
        K_in = KP_INPLANE

        def controller(t, q, qd, tool):
            T_now, v, J = tool
            x = stroke_x(t, T)
            n = believed.normal(x) if align_normal else np.array([0.0, 0.0, 1.0])
            p_ref = np.array([x, Y_DRAW, float(believed.height(x))]) - press * n
            # An optional integral on the FORCE error, applied along the normal.
            if f_int > 0.0:
                state["f_err_int"] += (F_DES - state["f_n"]) * dt
                p_ref = p_ref - f_int * state["f_err_int"] * n
            # Stiffness that is soft along the normal and stiff across it:
            #   K = K_in * (I - n n^T)  +  kz * n n^T
            # Written as a full 3x3 matrix this is exact for any normal; the
            # controller below uses the diagonal, which is the same thing when
            # the normal is the z axis and a good approximation while the slope
            # is small.  Experiment 5 measures where that stops being true.
            Kvec = np.array([K_in, K_in, kz]) if not align_normal else None
            if align_normal:
                P = np.eye(3) - np.outer(n, n)
                Kmat = K_in * P + kz * np.outer(n, n)
                Dmat = 2 * 0.9 * np.sqrt(3.0) * (np.sqrt(K_in) * P + np.sqrt(kz) * np.outer(n, n))
                e = tf.pose_error(T_now, tf.T_from_Rp(R_TOOL, p_ref))
                wrench = np.concatenate([Kmat @ e[:3] - Dmat @ v[:3],
                                         60.0 * e[3:] - 8.0 * v[3:]])
                return J.T @ wrench + dyn.gravity_torque(ARM, q)
            D = 2 * 0.9 * np.sqrt(Kvec * 3.0)
            e = tf.pose_error(T_now, tf.T_from_Rp(R_TOOL, p_ref))
            wrench = np.concatenate([Kvec * e[:3] - D * v[:3],
                                     60.0 * e[3:] - 8.0 * v[3:]])
            return J.T @ wrench + dyn.gravity_torque(ARM, q)

    def ext(t, p, v):
        w, f_n = surf.contact_wrench(p, v)
        state["f_n"] = f_n
        return w

    steps = int(T / dt)
    q, qd = q_start.copy(), np.zeros(N)
    P = np.zeros((steps, 3))
    F = np.zeros(steps)
    for k in range(steps):
        t = k * dt
        # One forward-kinematics + Jacobian evaluation per step, shared between
        # the contact model and the controller.  Computing it twice (the obvious
        # structure) costs about a fifth of the whole simulation.
        tool = imp.tool_state(ARM, q, qd)
        T_now, v, _ = tool
        p = T_now[:3, 3]
        wrench, f_n = surf.contact_wrench(p, v)
        state["f_n"] = f_n
        tau = np.clip(controller(t, q, qd, tool), -ARM.tau_max, ARM.tau_max)
        P[k], F[k] = p, f_n
        f_ext = None if not np.any(wrench) else {"tool0": wrench}
        qdd = dyn.forward_dynamics(ARM, q, qd, tau, f_ext=f_ext)
        qd = qd + dt * qdd
        q = q + dt * qd
        if not np.all(np.isfinite(q)):
            P[k + 1:], F[k + 1:] = P[k], F[k]
            break
    return np.arange(steps) * dt, P, F


def score(ts, P, F, surf, label, section, settle=0.4):
    """Report the numbers that decide whether the drawing worked."""
    m = ts > settle
    f = F[m]
    record(section, f"{label}: mean contact force", float(f.mean()), "N")
    record(section, f"{label}: force std", float(f.std()), "N")
    record(section, f"{label}: peak force", float(f.max()), "N")
    record(section, f"{label}: time above the {F_BREAK:.0f} N break limit",
           100 * float(np.mean(f > F_BREAK)), "%")
    record(section, f"{label}: time below the {F_LIFT:.1f} N lift-off limit",
           100 * float(np.mean(f < F_LIFT)), "%")
    return float(f.mean()), float(f.std()), float(f.max())


# ---------------------------------------------------------------------------
# 1. Position control against a 3 mm error
# ---------------------------------------------------------------------------
def exp1_position():
    print("[1] position control on a surface it has wrong")
    surf = Cylinder()
    record("1-position", "surface height change across the stroke",
           1e3 * float(surf.height(0.5 * (X0 + X1)) - surf.height(X0)), "mm")
    record("1-position", "steepest slope over the stroke", surf.slope_deg(X0), "deg")
    record("1-position", "planned press into the believed surface", 1e3 * PRESS_NOMINAL, "mm")

    # (a) The same controller, the same nominal press, three model errors.
    out = {}
    for label, off in (("model 3 mm too LOW", -0.003), ("model exact", 0.0),
                       ("model 3 mm too HIGH", +0.003)):
        ts, P, F = draw(surf, BelievedSurface("shape", surf, offset=off), "position")
        out[label] = (ts, P, F)
        score(ts, P, F, surf, label, "1-position")
    ts, P, F = draw(surf, BelievedSurface("flat", surf), "position")
    out["model FLAT (bulge ignored)"] = (ts, P, F)
    score(ts, P, F, surf, "model FLAT (bulge ignored)", "1-position")

    # (b) The stiffness you did NOT choose.  A position-controlled arm still has
    # a contact stiffness -- it is just set by the joint gains and the geometry
    # rather than by you, and it changes as the arm moves.  Measure it, then
    # raise the gains (which is what you would do to track better) and watch
    # what that does to the contact.
    base = np.array([900.0, 900.0, 600.0, 200.0, 200.0, 120.0])
    exact = BelievedSurface("shape", surf)
    stiff = {}
    for mult in (1.0, 3.0, 6.0):
        ts, P, F = draw(surf, exact, "position", kp_joint=base * mult)
        m = ts > 0.4
        # Effective contact stiffness = the SLOPE of force against penetration,
        # fitted through the origin over the samples that are actually touching.
        # Averaging the per-sample ratio F/penetration instead is meaningless:
        # near the moment of contact both are ~0 and the ratio is whatever the
        # round-off happened to be, which is how this line first reported
        # 2,000,000 N/m for a 25,000 N/m surface.
        pen = np.maximum(surf.height(P[m, 0]) - P[m, 2], 0.0)
        touching = (F[m] > F_LIFT) & (pen > 1e-5)
        k_eff = (float((pen[touching] @ F[m][touching]) / (pen[touching] @ pen[touching]))
                 if touching.sum() > 5 else np.nan)
        stiff[mult] = (ts, P, F)
        record("1-position", f"joint gains x{mult:.0f}: effective contact stiffness", k_eff, "N/m")
        record("1-position", f"joint gains x{mult:.0f}: mean force", float(F[m].mean()), "N")
        record("1-position", f"joint gains x{mult:.0f}: peak force", float(F[m].max()), "N")
        record("1-position", f"joint gains x{mult:.0f}: time above the break limit",
               100 * float(np.mean(F[m] > F_BREAK)), "%")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    for k, (label, (ts, P, F)) in enumerate(out.items()):
        axes[0].plot(ts, F, color=COLORS[k], label=label)
    axes[0].axhline(F_BREAK, color=COLORS[1], ls="--", lw=1.2)
    axes[0].axhline(F_LIFT, color=COLORS[2], ls=":", lw=1.2)
    axes[0].set_yscale("symlog", linthresh=1.0)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("contact force (N)")
    axes[0].set_title("3 mm of model error, one nominal press")
    axes[0].legend(fontsize=7)
    for k, (mult, (ts, P, F)) in enumerate(stiff.items()):
        axes[1].plot(ts, F, color=COLORS[k], label=f"joint gains x{mult:.0f}")
    axes[1].axhline(F_BREAK, color=COLORS[1], ls="--", lw=1.2)
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("contact force (N)")
    axes[1].set_title("Stiffer position control, PERFECT model")
    axes[1].legend(fontsize=7)
    save(fig, os.path.join(OUT, "position.png"))


# ---------------------------------------------------------------------------
# 2. Impedance control
# ---------------------------------------------------------------------------
def exp2_impedance():
    print("[2] impedance control on the same wrong model")
    surf = Cylinder()
    out = {}
    for label, believed in (("flat model", BelievedSurface("flat", surf)),
                            ("3 mm too low", BelievedSurface("shape", surf, offset=-0.003)),
                            ("3 mm too high", BelievedSurface("shape", surf, offset=+0.003))):
        kz = 250.0
        press = F_DES / kz
        ts, P, F = draw(surf, believed, "impedance", kz=kz, press=press)
        out[label] = (ts, P, F)
        score(ts, P, F, surf, label, "2-impedance")
    record("2-impedance", "commanded normal stiffness", 250.0, "N/m")
    record("2-impedance", "commanded press depth", 1e3 * F_DES / 250.0, "mm")
    record("2-impedance", "force this predicts", 250.0 * F_DES / 250.0, "N")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    for k, (label, (ts, P, F)) in enumerate(out.items()):
        axes[0].plot(ts, F, color=COLORS[k], label=label)
    axes[0].axhline(F_DES, color=COLORS[6], ls="--", lw=1.2)
    axes[0].set_ylim(0, 12)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("contact force (N)")
    axes[0].set_title("The same 3 mm errors, now harmless")
    axes[0].legend(fontsize=7)
    ts, P, F = out["flat model"]
    xs = np.linspace(X0, X1, 200)
    axes[1].plot(1e3 * xs, 1e3 * surf.height(xs), color=COLORS[6], lw=1.4, label="true surface")
    axes[1].plot(1e3 * xs, 1e3 * np.full_like(xs, surf.height(surf.x_c)), "--",
                 color=COLORS[1], lw=1.2, label="what the planner believed")
    axes[1].plot(1e3 * P[:, 0], 1e3 * P[:, 2], color=COLORS[0], lw=1.4, label="pen tip")
    axes[1].set_xlabel("x (mm)")
    axes[1].set_ylabel("z (mm)")
    axes[1].set_title("The pen finds the surface the plan did not have")
    axes[1].legend(fontsize=7)
    save(fig, os.path.join(OUT, "impedance.png"))


# ---------------------------------------------------------------------------
# 3. Choosing the normal stiffness
# ---------------------------------------------------------------------------
def exp3_stiffness():
    print("[3] choosing the normal stiffness")
    surf = Cylinder()
    believed = BelievedSurface("flat", surf)
    kzs = [60.0, 120.0, 250.0, 500.0, 1000.0, 2000.0]
    means, stds, peaks = [], [], []
    for kz in kzs:
        ts, P, F = draw(surf, believed, "impedance", kz=kz, press=F_DES / kz)
        mean, std, peak = score(ts, P, F, surf, f"Kz={kz:.0f} N/m", "3-stiffness")
        means.append(mean)
        stds.append(std)
        peaks.append(peak)
    record("3-stiffness", "force std at the softest setting", stds[0], "N")
    record("3-stiffness", "force std at the stiffest setting", stds[-1], "N")
    record("3-stiffness", "ratio", stds[-1] / max(stds[0], 1e-9), "x")
    record("3-stiffness", "peak force at the stiffest setting", peaks[-1], "N")
    record("3-stiffness", "press depth needed at the softest setting",
           1e3 * F_DES / kzs[0], "mm")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.errorbar(kzs, means, yerr=stds, fmt="o-", color=COLORS[0], capsize=3)
    ax.axhline(F_DES, color=COLORS[6], ls="--", lw=1.2, label="wanted")
    ax.axhline(F_BREAK, color=COLORS[1], ls="--", lw=1.0, label="pen snaps")
    ax.set_xscale("log")
    ax.set_xlabel("normal stiffness Kz (N/m)")
    ax.set_ylabel("contact force (N), bars = 1 std")
    ax.set_title("Softer along the normal means a steadier force")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "stiffness.png"))


# ---------------------------------------------------------------------------
# 4. An integral on the force
# ---------------------------------------------------------------------------
def exp4_force_integral():
    print("[4] adding an integral on the force error")
    surf = Cylinder()
    believed = BelievedSurface("flat", surf)
    out = {}
    for label, fi, kz in (("spring only, Kz = 250", 0.0, 250.0),
                          ("spring only, Kz = 1000", 0.0, 1000.0),
                          ("Kz = 1000 + force integral", 4e-4, 1000.0)):
        ts, P, F = draw(surf, believed, "impedance", kz=kz, press=F_DES / kz, f_int=fi)
        out[label] = (ts, P, F)
        mean, std, peak = score(ts, P, F, surf, label, "4-integral")
        record("4-integral", f"{label}: mean force error", abs(mean - F_DES), "N")

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    for k, (label, (ts, P, F)) in enumerate(out.items()):
        ax.plot(ts, F, color=COLORS[k], label=label)
    ax.axhline(F_DES, color=COLORS[6], ls="--", lw=1.2)
    ax.set_ylim(0, 14)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("contact force (N)")
    ax.set_title("A stiff spring plus an integrator behaves like a soft one")
    ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "integral.png"))


# ---------------------------------------------------------------------------
# 5. A steep surface
# ---------------------------------------------------------------------------
def exp5_steep():
    print("[5] a steep surface: world-z compliance vs normal-aligned")
    rows = []
    traces = {}
    for R, name in ((0.55, "gentle (R = 55 cm)"), (0.16, "steep (R = 16 cm)")):
        surf = Cylinder(R=R)
        believed = BelievedSurface("shape", surf)
        slope = max(surf.slope_deg(X0), surf.slope_deg(X1))
        record("5-steep", f"{name}: steepest slope over the stroke", slope, "deg")
        for align, label in ((False, "compliant along world z"), (True, "compliant along the normal")):
            ts, P, F = draw(surf, believed, "impedance", kz=250.0, press=F_DES / 250.0,
                            align_normal=align)
            mean, std, peak = score(ts, P, F, surf, f"{name}, {label}", "5-steep")
            rows.append((name, label, mean, std, peak))
            traces[(name, align)] = (ts, F)   # keep them; the figure reuses these
    for name in ("gentle (R = 55 cm)", "steep (R = 16 cm)"):
        a = [r for r in rows if r[0] == name and "world z" in r[1]][0]
        b = [r for r in rows if r[0] == name and "normal" in r[1]][0]
        record("5-steep", f"{name}: force-std ratio (world z / normal)",
               a[3] / max(b[3], 1e-9), "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    for ax, (R, name) in zip(axes, ((0.55, "gentle (R = 55 cm)"), (0.16, "steep (R = 16 cm)"))):
        for k, (align, label) in enumerate(((False, "world z"), (True, "surface normal"))):
            ts, F = traces[(name, align)]
            ax.plot(ts, F, color=COLORS[k], label=f"compliant along {label}")
        ax.axhline(F_DES, color=COLORS[6], ls="--", lw=1.2)
        ax.set_ylim(0, 14)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("contact force (N)")
        ax.set_title(name)
        ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "steep.png"))


# ---------------------------------------------------------------------------
# 6. Drawing faster
# ---------------------------------------------------------------------------
def exp6_speed():
    print("[6] drawing faster")
    surf = Cylinder()
    believed = BelievedSurface("flat", surf)
    Ts = [3.2, 2.0, 1.2, 0.8, 0.55]
    stds, peaks, means = [], [], []
    for T in Ts:
        ts, P, F = draw(surf, believed, "impedance", T=T, kz=250.0, press=F_DES / 250.0)
        mean, std, peak = score(ts, P, F, surf, f"stroke in {T:.1f} s", "6-speed",
                                settle=0.25 * T)
        stds.append(std)
        peaks.append(peak)
        means.append(mean)
        record("6-speed", f"stroke in {T:.1f} s: pen speed", (X1 - X0) / T, "m/s")
    broke = [T for T, pk in zip(Ts, peaks) if pk > F_BREAK]
    record("6-speed", "fastest stroke that keeps the pen intact",
           min([T for T, pk in zip(Ts, peaks) if pk <= F_BREAK], default=np.nan), "s")
    record("6-speed", "force std at the slowest stroke", stds[0], "N")
    record("6-speed", "force std at the fastest stroke", stds[-1], "N")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    speeds = [(X1 - X0) / T for T in Ts]
    ax.plot(speeds, peaks, "o-", color=COLORS[1], label="peak force")
    ax.plot(speeds, means, "s-", color=COLORS[0], label="mean force")
    ax.fill_between(speeds, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                    color=COLORS[0], alpha=0.15)
    ax.axhline(F_BREAK, color=COLORS[1], ls="--", lw=1.2)
    ax.set_xlabel("pen speed (m/s)")
    ax.set_ylabel("contact force (N)")
    ax.set_title("The soft spring has a speed limit too")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "speed.png"))


def main():
    t0 = time.perf_counter()
    exp1_position()
    exp2_impedance()
    exp3_stiffness()
    exp4_force_integral()
    exp5_steep()
    exp6_speed()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
