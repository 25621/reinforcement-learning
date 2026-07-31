"""Project 08 -- Pendulum PID.

Seven experiments on one inverted pendulum:

  1. the tuned step response, with the three terms drawn separately
  2. what each gain actually does (and the gain floor gravity imposes)
  3. integral windup against a torque limit, and the two-line cure
  4. derivative kick and derivative noise
  5. the same gains at nine control rates
  6. Ziegler-Nichols against the hand-tuned gains
  7. the basin of attraction, and why swing-up is a different problem

Runs in about 40 seconds on a CPU.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "01-transform-calculator"))

import matplotlib.pyplot as plt  # noqa: E402

from pendulum import Pendulum, simulate, G  # noqa: E402
from pid import PID, step_metrics  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

DT = 1e-3  # 1 kHz control loop
PLANT = dict(m=0.5, l=0.4, b=0.01)
MGL = PLANT["m"] * G * PLANT["l"]
INERTIA = PLANT["m"] * PLANT["l"] ** 2

# Hand-tuned gains: place the closed-loop poles at wn = 12 rad/s, zeta = 0.8.
#   Kp - m g l = I wn^2      (the spring gravity has NOT already eaten)
#   Kd         = 2 zeta wn I
WN, ZETA = 12.0, 0.8
KP = MGL + INERTIA * WN ** 2
KD = 2 * ZETA * WN * INERTIA
KI = 40.0


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<54s} {value:>12.4f} {unit}")


def new_pid(kp=KP, ki=KI, kd=KD, **kw):
    kw.setdefault("dt", DT)
    kw.setdefault("u_min", -3.0)
    kw.setdefault("u_max", 3.0)
    return PID(kp, ki, kd, **kw)


# ---------------------------------------------------------------------------
# 1. The tuned step response
# ---------------------------------------------------------------------------
def exp1_step():
    print("[1] tuned step response")
    plant = Pendulum(**PLANT, tau_max=3.0)
    step_t, target = 0.2, 0.15

    record("1-step", "Kp (hand-tuned)", KP, "N*m/rad")
    record("1-step", "Ki", KI, "N*m/(rad*s)")
    record("1-step", "Kd", KD, "N*m*s/rad")
    record("1-step", "gravity's gain floor  m*g*l", MGL, "N*m/rad")

    runs = {}
    for label, b in (("b = 1 (plain PID)", 1.0), ("b = 0 (setpoint weighting)", 0.0)):
        ctrl = new_pid(b_sp=b)
        r = simulate(plant, ctrl, theta0=0.0, T=1.8,
                     setpoint_fn=lambda t: target if t >= step_t else 0.0)
        mask = r["t"] >= step_t
        m = step_metrics(r["t"][mask] - step_t, r["theta"][mask], target)
        runs[label] = (r, m)
        record("1-step", f"{label}: rise time (10-90%)", 1e3 * m["rise"], "ms")
        record("1-step", f"{label}: overshoot", m["overshoot"], "%")
        record("1-step", f"{label}: settling time (2% band)", 1e3 * m["settle"], "ms")
        record("1-step", f"{label}: steady-state error", 1e3 * m["sse"], "mrad")

    # The same two controllers, judged on a DISTURBANCE instead of a target
    # change: a 0.2 N*m push arrives at t = 0.2 s and stays.  Setpoint
    # weighting must not change this, because a push does not move the setpoint.
    dist = {}
    for label, b in (("b = 1 (plain PID)", 1.0), ("b = 0 (setpoint weighting)", 0.0)):
        ctrl = new_pid(b_sp=b)
        r = simulate(plant, ctrl, theta0=0.0, T=2.0, load=0.2)
        dist[label] = r
        record("1-step", f"{label}: peak angle after a 0.2 N*m push",
               1e3 * float(np.max(np.abs(r["theta"]))), "mrad")

    # Re-run the b = 1 controller recording the three terms separately.
    ctrl2 = new_pid()
    plant2 = Pendulum(**PLANT, tau_max=3.0)
    theta, thetad = 0.0, 0.0
    ts, terms = [], []
    for k in range(1800):
        t = k * DT
        sp = target if t >= step_t else 0.0
        ctrl2(sp, theta)
        ts.append(t)
        terms.append(ctrl2.last_terms)
        u = float(np.clip(sum(ctrl2.last_terms), -3.0, 3.0))
        for _ in range(5):
            theta, thetad = plant2.rk4(theta, thetad, u, DT / 5)
    terms = np.array(terms)

    fig, axes = plt.subplots(3, 1, figsize=(6.8, 5.8), sharex=True)
    axes[0].plot(runs["b = 1 (plain PID)"][0]["t"], runs["b = 1 (plain PID)"][0]["sp"],
                 color=COLORS[6], ls="--", label="target")
    for k, (label, (r, m)) in enumerate(runs.items()):
        axes[0].plot(r["t"], r["theta"], color=COLORS[k],
                     label=f"{label}: {m['overshoot']:.0f}% overshoot")
    axes[0].axhspan(target * 0.98, target * 1.02, color=COLORS[2], alpha=0.15)
    axes[0].set_ylabel("angle from upright (rad)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Step response, same three gains, one extra knob")
    for k, (label, (r, _)) in enumerate(runs.items()):
        axes[1].plot(r["t"], r["u"], color=COLORS[k], label=label)
    axes[1].set_ylabel("applied torque (N*m)")
    axes[1].legend(fontsize=8)
    for k, (lab, col) in enumerate([("P", 0), ("I", 2), ("D", 3)]):
        axes[2].plot(ts, terms[:, k], color=COLORS[col], label=f"{lab} term")
    axes[2].set_ylabel("torque contribution (N*m)")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8, ncol=3)
    axes[2].set_title("Who supplies the torque, over the b = 1 run")
    save(fig, os.path.join(OUT, "step_response.png"))


# ---------------------------------------------------------------------------
# 2. What each gain does
# ---------------------------------------------------------------------------
def exp2_gains():
    print("[2] the anatomy of the three gains")
    plant = Pendulum(**PLANT, tau_max=3.0)

    # (a) Kp below and above the gravity floor, no damping term.
    kps = [0.5 * MGL, 0.9 * MGL, 1.05 * MGL, 3.0 * MGL, 7.0 * MGL]
    runs_p = []
    for kp in kps:
        r = simulate(plant, new_pid(kp=kp, ki=0, kd=0), theta0=0.10, T=3.0)
        runs_p.append((kp, r))
        final = abs(r["theta"][-1])
        record("2-gains", f"P-only Kp={kp / MGL:.2f}*mgl: |angle| at 3 s", final, "rad")

    # (b) adding Kd.  The metric is how long the arm takes to come back inside
    # a 5 mrad band after being displaced -- "settling time", the number a
    # human tuning by feel is actually watching.
    runs_d = []
    for kd in [0.0, 0.4 * KD, KD, 3.0 * KD]:
        r = simulate(plant, new_pid(ki=0, kd=kd), theta0=0.10, T=3.0)
        outside = np.nonzero(np.abs(r["theta"]) > 0.005)[0]
        settle = r["t"][outside[-1]] if len(outside) and outside[-1] + 1 < len(r["t"]) else np.nan
        runs_d.append((kd, r))
        record("2-gains", f"PD Kd={kd:.3f}: time to settle inside 5 mrad", 1e3 * settle, "ms")

    # (c) a constant load torque -- what the integral term is for.
    load = 0.25  # N*m, e.g. an off-centre payload or a motor with a bias
    r_pd = simulate(plant, new_pid(ki=0), theta0=0.0, T=4.0, load=load)
    r_pid = simulate(plant, new_pid(), theta0=0.0, T=4.0, load=load)
    off_pd = float(np.mean(r_pd["theta"][-500:]))
    off_pid = float(np.mean(r_pid["theta"][-500:]))
    # Steady state of the PD loop: (Kp - mgl) theta = -load, because near
    # theta = 0 gravity acts like a NEGATIVE spring that eats Kp.
    predicted = -load / (KP - MGL)
    record("2-gains", "PD steady offset under a 0.25 N*m load", 1e3 * off_pd, "mrad")
    record("2-gains", "  predicted -load/(Kp - mgl)", 1e3 * predicted, "mrad")
    record("2-gains", "PID steady offset under the same load", 1e3 * off_pid, "mrad")
    record("2-gains", "offset reduction from the I term", abs(off_pd / off_pid), "x")

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0))
    for k, (kp, r) in enumerate(runs_p):
        axes[0].plot(r["t"], r["theta"], color=COLORS[k], label=f"Kp = {kp / MGL:.2f} m g l")
    axes[0].set_ylim(-0.6, 0.6)
    axes[0].set_title("P only: gravity sets a gain floor")
    axes[0].legend(fontsize=7)
    for k, (kd, r) in enumerate(runs_d):
        axes[1].plot(r["t"], r["theta"], color=COLORS[k], label=f"Kd = {kd:.2f}")
    axes[1].set_title("D adds the damping")
    axes[1].legend(fontsize=7)
    axes[2].plot(r_pd["t"], 1e3 * r_pd["theta"], color=COLORS[1], label="PD")
    axes[2].plot(r_pid["t"], 1e3 * r_pid["theta"], color=COLORS[0], label="PID")
    axes[2].axhline(1e3 * predicted, color=COLORS[6], ls="--", lw=1.2, label="predicted PD offset")
    axes[2].set_ylabel("angle (mrad)")
    axes[2].set_title("I erases the offset a constant load leaves")
    axes[2].legend(fontsize=7)
    for ax in axes[:2]:
        ax.set_ylabel("angle (rad)")
    for ax in axes:
        ax.set_xlabel("time (s)")
    save(fig, os.path.join(OUT, "gains.png"))


# ---------------------------------------------------------------------------
# 3. Windup
# ---------------------------------------------------------------------------
def exp3_windup():
    print("[3] integral windup against a torque limit")
    # This is the ONE experiment that turns the pendulum the right way up.
    # Windup needs the motor to sit at its limit for a long stretch, and an
    # INVERTED pendulum falls over in a fraction of a second once it is beyond
    # what the motor can hold -- you never get the long saturated stretch, only
    # a crash.  A hanging joint is stable on its own, so it will sit there
    # saturated for as long as the move takes.  This is also the honest setting:
    # windup is a positioning-servo problem, and almost every joint on a real
    # robot arm is exactly this -- a load that gravity pulls back toward rest.
    tau_lim = 2.2  # a weak motor, but above m*g*l so every target is holdable
    plant = Pendulum(**PLANT, tau_max=tau_lim, inverted=False)
    # Kp * 1.5 rad = 20 N*m is nine times the limit, so the motor spends the
    # first part of the move flat out with the integrator counting the whole time.
    target = 1.5

    out = {}
    for label, aw in (("no anti-windup", False), ("anti-windup", True)):
        ctrl = new_pid(u_min=-tau_lim, u_max=tau_lim, anti_windup=aw)
        r = simulate(plant, ctrl, theta0=0.0, T=6.0,
                     setpoint_fn=lambda t: target if t >= 0.2 else 0.0)
        mask = r["t"] >= 0.2
        m = step_metrics(r["t"][mask] - 0.2, r["theta"][mask], target)
        out[label] = (r, m)
        record("3-windup", f"{label}: overshoot", m["overshoot"], "%")
        record("3-windup", f"{label}: settling time", 1e3 * m["settle"], "ms")
        record("3-windup", f"{label}: peak unsaturated command", float(np.max(np.abs(r["u_raw"]))), "N*m")
    record("3-windup", "overshoot ratio (no AW / AW)",
           out["no anti-windup"][1]["overshoot"] / max(out["anti-windup"][1]["overshoot"], 1e-9), "x")

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.2), sharex=True)
    for k, (label, (r, _)) in enumerate(out.items()):
        axes[0].plot(r["t"], r["theta"], color=COLORS[k], label=label)
        axes[1].plot(r["t"], r["u_raw"], color=COLORS[k], label=f"{label} (before the limit)")
    axes[0].axhline(target, color=COLORS[6], ls="--", lw=1.2)
    axes[0].set_ylabel("angle (rad)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Hanging joint, motor limited to {tau_lim} N*m: the integrator counts anyway")
    axes[1].axhspan(-tau_lim, tau_lim, color=COLORS[2], alpha=0.12)
    axes[1].set_ylabel("commanded torque (N*m)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "windup.png"))


# ---------------------------------------------------------------------------
# 4. Derivative kick and derivative noise
# ---------------------------------------------------------------------------
def exp4_derivative():
    print("[4] derivative kick and derivative noise")
    plant = Pendulum(**PLANT, tau_max=50.0)  # limit lifted so the kick is visible
    target = 0.15

    kicks = {}
    for label, dom in (("D on error", False), ("D on measurement", True)):
        ctrl = new_pid(u_min=-50, u_max=50, d_on_measurement=dom)
        r = simulate(plant, ctrl, theta0=0.0, T=1.2,
                     setpoint_fn=lambda t: target if t >= 0.2 else 0.0)
        kicks[label] = r
        record("4-derivative", f"{label}: peak commanded torque", float(np.max(np.abs(r["u_raw"]))), "N*m")
    record("4-derivative", "kick multiplier (on error / on measurement)",
           float(np.max(np.abs(kicks["D on error"]["u_raw"]))
                 / np.max(np.abs(kicks["D on measurement"]["u_raw"]))), "x")

    noise = 0.002  # 2 mrad of encoder noise, about a 12-bit encoder on one turn
    noisy = {}
    for label, fhz in (("no filter", None), ("30 Hz filter", 30.0), ("8 Hz filter", 8.0)):
        ctrl = new_pid(d_filter_hz=fhz)
        r = simulate(plant, ctrl, theta0=0.05, T=4.0, noise_std=noise, seed=3)
        noisy[label] = r
        tail = r["t"] > 1.5
        record("4-derivative", f"{label}: torque RMS with a noisy encoder",
               float(np.sqrt(np.mean(r["u"][tail] ** 2))), "N*m")
        record("4-derivative", f"{label}: angle RMS",
               1e3 * float(np.sqrt(np.mean(r["theta"][tail] ** 2))), "mrad")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for k, (label, r) in enumerate(kicks.items()):
        axes[0].plot(r["t"], r["u_raw"], color=COLORS[k], label=label)
    axes[0].set_xlim(0.15, 0.45)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("commanded torque (N*m)")
    axes[0].set_title("Derivative kick at the instant the target moves")
    axes[0].legend(fontsize=8)
    for k, (label, r) in enumerate(noisy.items()):
        axes[1].plot(r["t"], r["u"], color=COLORS[k], lw=1.0, label=label)
    axes[1].set_xlim(2.0, 2.5)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("applied torque (N*m)")
    axes[1].set_title("2 mrad of encoder noise, through Kd")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "derivative.png"))


# ---------------------------------------------------------------------------
# 5. Control rate
# ---------------------------------------------------------------------------
def exp5_rate():
    print("[5] the same gains at ten control rates")
    plant = Pendulum(**PLANT, tau_max=8.0)
    target = 0.10
    rates = [2000, 1000, 500, 250, 150, 100, 70, 50, 40, 30, 20]
    over, stable, traces = [], [], {}
    # Setpoint weighting is on (b = 0) so the baseline overshoot is near zero
    # and every percent that appears is caused by the sampling, not by the
    # closed-loop zero that experiment 1 measured.
    for f in rates:
        ctrl = PID(KP, KI, KD, dt=1.0 / f, u_min=-8, u_max=8, b_sp=0.0)
        r = simulate(plant, ctrl, theta0=0.0, T=3.0, physics_dt=1e-4,
                     setpoint_fn=lambda t: target if t >= 0.2 else 0.0)
        mask = r["t"] >= 0.2
        m = step_metrics(r["t"][mask] - 0.2, r["theta"][mask], target)
        peak = float(np.max(np.abs(r["theta"])))
        ok = np.isfinite(peak) and peak < 0.5
        os_pct = min(m["overshoot"], 500.0) if np.isfinite(m["overshoot"]) else 500.0
        over.append(os_pct)
        stable.append(ok)
        if f in (1000, 100, 40, 30):
            traces[f] = r
        record("5-rate", f"{f} Hz: overshoot", os_pct, "%")
    last_ok = min([f for f, s in zip(rates, stable) if s])
    record("5-rate", "lowest rate that still holds the pendulum", last_ok, "Hz")
    record("5-rate", "ratio to the design rate", 1000.0 / last_ok, "x")
    record("5-rate", "overshoot at 1000 Hz", over[rates.index(1000)], "%")
    record("5-rate", "overshoot at 100 Hz", over[rates.index(100)], "%")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    axes[0].plot(rates, over, "-", color=COLORS[6], lw=1.0, zorder=1)
    for f, s, v in zip(rates, stable, over):
        axes[0].plot(f, v, "o", color=COLORS[2] if s else COLORS[1], ms=8, zorder=2)
    axes[0].set_xscale("log")
    axes[0].invert_xaxis()
    axes[0].set_xlabel("control rate (Hz), slower to the right")
    axes[0].set_ylabel("step overshoot (%)")
    axes[0].set_title("One gain set: green holds, orange falls over")
    for k, (f, r) in enumerate(sorted(traces.items(), reverse=True)):
        axes[1].plot(r["t"], r["theta"], color=COLORS[k], label=f"{f} Hz")
    axes[1].axhline(target, color=COLORS[6], ls="--", lw=1.2)
    axes[1].set_ylim(-0.35, 0.45)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("angle (rad)")
    axes[1].set_title("The same step, sampled more and more slowly")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "rate.png"))


# ---------------------------------------------------------------------------
# 6. Ziegler-Nichols
# ---------------------------------------------------------------------------
def find_ultimate_gain(plant, lo=3.0, hi=200.0, iters=30):
    """Bisect for the P-only gain at which oscillation neither grows nor decays.

    The classic Ziegler-Nichols recipe starts here: turn I and D off, raise Kp
    until the loop hums at a steady amplitude, and read off that gain (Ku) and
    the period of the hum (Tu).  The oscillation exists at all only because the
    loop has one sample of dead time; a delay-free model of this nearly
    frictionless plant would happily take any gain.

    The bracket starts at Kp = 3, safely above the gravity floor m g l = 1.96.
    Below that floor the pendulum simply falls over, which ALSO looks like
    "amplitude grew" to the detector -- bisecting from zero would converge on
    the gravity floor instead of on the ultimate gain.
    """
    def growth(kp):
        ctrl = PID(kp, 0.0, 0.0, dt=DT, u_min=-1e6, u_max=1e6)
        r = simulate(plant, ctrl, theta0=0.02, T=4.0, physics_dt=1e-4)
        a_early = np.max(np.abs(r["theta"][(r["t"] > 0.5) & (r["t"] < 1.5)]))
        a_late = np.max(np.abs(r["theta"][r["t"] > 2.5]))
        if not np.isfinite(a_late) or a_late > 1e3:
            return np.inf
        return a_late / max(a_early, 1e-12)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if growth(mid) > 1.0:
            hi = mid
        else:
            lo = mid
    ku = 0.5 * (lo + hi)

    ctrl = PID(ku, 0.0, 0.0, dt=DT, u_min=-1e6, u_max=1e6)
    r = simulate(plant, ctrl, theta0=0.02, T=4.0, physics_dt=1e-4)
    th = r["theta"][r["t"] > 1.0]
    crossings = np.nonzero(np.diff(np.sign(th)) != 0)[0]
    tu = 2.0 * DT * float(np.mean(np.diff(crossings))) if len(crossings) > 2 else np.nan
    return ku, tu


def exp6_zn():
    print("[6] Ziegler-Nichols vs the hand-tuned gains")
    plant = Pendulum(**PLANT, tau_max=8.0)
    ku, tu = find_ultimate_gain(plant)
    record("6-zn", "ultimate gain Ku", ku, "N*m/rad")
    record("6-zn", "ultimate period Tu", 1e3 * tu, "ms")

    zn = dict(kp=0.6 * ku, ki=1.2 * ku / tu, kd=0.075 * ku * tu)
    record("6-zn", "ZN Kp", zn["kp"], "N*m/rad")
    record("6-zn", "ZN Ki", zn["ki"], "N*m/(rad*s)")
    record("6-zn", "ZN Kd", zn["kd"], "N*m*s/rad")
    record("6-zn", "ZN Kp / hand-tuned Kp", zn["kp"] / KP, "x")

    target = 0.15
    out = {}
    for label, gains in (("hand-tuned", dict(kp=KP, ki=KI, kd=KD)), ("Ziegler-Nichols", zn)):
        ctrl = PID(gains["kp"], gains["ki"], gains["kd"], dt=DT, u_min=-8, u_max=8)
        r = simulate(plant, ctrl, theta0=0.0, T=2.5,
                     setpoint_fn=lambda t: target if t >= 0.2 else 0.0)
        mask = r["t"] >= 0.2
        m = step_metrics(r["t"][mask] - 0.2, r["theta"][mask], target)
        out[label] = (r, m)
        record("6-zn", f"{label}: overshoot", m["overshoot"], "%")
        record("6-zn", f"{label}: settling time", 1e3 * m["settle"], "ms")
        record("6-zn", f"{label}: peak torque", float(np.max(np.abs(r["u"]))), "N*m")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for k, (label, (r, m)) in enumerate(out.items()):
        ax.plot(r["t"], r["theta"], color=COLORS[k],
                label=f"{label}: {m['overshoot']:.0f}% overshoot")
    ax.axhline(target, color=COLORS[6], ls="--", lw=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("angle (rad)")
    ax.set_title("A recipe designed for a different kind of plant")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "ziegler_nichols.png"))


# ---------------------------------------------------------------------------
# 7. Basin of attraction, and swing-up
# ---------------------------------------------------------------------------
def basin_grid(kp, ki, kd, tau_max, thetas, rates, T=4.0, dt=2e-3):
    """Vectorised PID + plant, so a whole grid of starts runs in one pass.

    The scalar simulator would need one Python loop per grid point; running the
    entire grid as NumPy arrays turns 1,271 simulations into 2,000 array
    operations.  The physics and the controller are identical to the scalar
    version -- only the bookkeeping is different.
    """
    TH, RA = np.meshgrid(thetas, rates, indexing="ij")
    th = TH.ravel().copy()
    rt = RA.ravel().copy()
    integ = np.zeros_like(th)
    prev = th.copy()
    u_prev = np.zeros_like(th)
    m, l, b = PLANT["m"], PLANT["l"], PLANT["b"]
    I, mgl = m * l * l, m * G * l
    sub, steps = 4, int(T / dt)
    h = dt / sub
    for _ in range(steps):
        e = -th
        d = -(th - prev) / dt
        prev = th.copy()
        u = kp * e + ki * integ + kd * d
        u_sat = np.clip(u, -tau_max, tau_max)
        grow = (u != u_sat) & (np.sign(e) == np.sign(u))
        integ = np.where(grow, integ, integ + e * dt)
        applied = u_prev  # one sample of loop delay, as in the scalar version
        u_prev = u_sat
        for _ in range(sub):
            def acc(a, v):
                return (applied - b * v + mgl * np.sin(a)) / I

            k1a, k1v = rt, acc(th, rt)
            k2a, k2v = rt + 0.5 * h * k1v, acc(th + 0.5 * h * k1a, rt + 0.5 * h * k1v)
            k3a, k3v = rt + 0.5 * h * k2v, acc(th + 0.5 * h * k2a, rt + 0.5 * h * k2v)
            k4a, k4v = rt + h * k3v, acc(th + h * k3a, rt + h * k3v)
            th = th + (h / 6) * (k1a + 2 * k2a + 2 * k3a + k4a)
            rt = rt + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
        th = np.clip(th, -50, 50)
    ok = (np.abs(th) < 0.05) & (np.abs(rt) < 0.3)
    return ok.reshape(TH.shape)


def swing_up(plant, k_energy=1.2, tau_max=1.2, T=10.0, dt=1e-3):
    """Energy pumping until the pendulum is near upright, then hand over to PID.

    The swing-up law has nothing to do with position.  Differentiating the
    pendulum's energy along its own motion gives

        dE/dt = thetad * tau  -  b * thetad^2

    so pushing in the direction the pendulum is ALREADY moving always adds
    energy, whatever the angle:

        tau = k (E_top - E) sign(thetad)

    With a motor too weak to lift the pendulum directly, this is the only way
    up -- several swings, each one a little higher, exactly how a child on a
    playground swing gets going.  (Getting this sign wrong is easy: the famous
    Astrom-Furuta law for a CART-pole carries an extra cos(theta) factor,
    because there the input is a horizontal cart acceleration rather than a
    torque at the pivot.  Copying that version here pumps no energy at all --
    it was the first thing this script got wrong.)

    The pendulum starts hanging with a nudge of 0.05 rad/s.  From exactly zero
    velocity ``sign(thetad)`` is zero, the law commands nothing, and a perfectly
    balanced hanging pendulum stays hanging forever.
    """
    ctrl = new_pid(u_min=-tau_max, u_max=tau_max)
    theta, thetad = np.pi, 0.05
    E_top = plant.mgl * 2.0  # energy of the upright state, measured from hanging
    ts, ths, us, modes = [], [], [], []
    catching = False
    for k in range(int(T / dt)):
        wrapped = np.arctan2(np.sin(theta), np.cos(theta))
        if not catching and abs(wrapped) < 0.40 and abs(thetad) < 2.5:
            catching = True
            ctrl.reset()
        if catching:
            u = ctrl(0.0, wrapped)
        else:
            E = plant.energy(theta, thetad)
            u = float(np.clip(k_energy * (E_top - E) * np.sign(thetad), -tau_max, tau_max))
        ts.append(k * dt)
        ths.append(wrapped)
        us.append(u)
        modes.append(1.0 if catching else 0.0)
        for _ in range(4):
            theta, thetad = plant.rk4(theta, thetad, u, dt / 4)
    return (np.array(ts), np.array(ths), np.array(us), np.array(modes),
            abs(np.arctan2(np.sin(theta), np.cos(theta))))


def exp7_basin():
    print("[7] basin of attraction, and swing-up")
    tau_max = 1.2  # weaker than m g l = 1.96, so it cannot lift from horizontal
    thetas = np.linspace(-np.pi, np.pi, 61)
    rates = np.linspace(-8, 8, 41)
    ok = basin_grid(KP, KI, KD, tau_max, thetas, rates)
    frac = float(ok.mean())
    widest = float(np.max(np.abs(thetas[ok[:, len(rates) // 2]]))) if ok[:, len(rates) // 2].any() else 0.0
    record("7-basin", "fraction of the start grid PID recovers", 100 * frac, "%")
    record("7-basin", "largest recoverable tilt at zero speed", widest, "rad")
    record("7-basin", "  same, in degrees", np.degrees(widest), "deg")
    record("7-basin", "motor torque limit", tau_max, "N*m")
    record("7-basin", "gravity torque when horizontal (m*g*l)", MGL, "N*m")

    ok_strong = basin_grid(KP, KI, KD, 4.0, thetas, rates)
    record("7-basin", "same grid with a 4 N*m motor", 100 * float(ok_strong.mean()), "%")

    plant = Pendulum(**PLANT, tau_max=tau_max)
    ts, ths, us, modes, final = swing_up(plant, tau_max=tau_max)
    caught = bool(modes.max() > 0)
    hand = float(ts[np.argmax(modes > 0)]) if caught else np.nan
    record("7-basin", "swing-up + catch: final |angle|", final, "rad")
    record("7-basin", "time of hand-over to PID", hand, "s")
    record("7-basin", "peak swing-up torque used", float(np.max(np.abs(us[modes == 0]))), "N*m")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    for ax, grid, title in ((axes[0], ok, f"tau_max = {tau_max} N*m"),
                            (axes[1], ok_strong, "tau_max = 4.0 N*m")):
        ax.imshow(grid.T, origin="lower", aspect="auto", cmap="Greens",
                  extent=[thetas[0], thetas[-1], rates[0], rates[-1]], vmin=0, vmax=1.4)
        ax.set_xlabel("starting angle from upright (rad)")
        ax.set_ylabel("starting rate (rad/s)")
        ax.set_title(f"recovers: {100 * grid.mean():.0f}%  ({title})")
        ax.grid(False)
    axes[2].plot(ts, ths, color=COLORS[0], label="angle (rad)")
    axes[2].plot(ts, us, color=COLORS[1], lw=1.0, label="torque (N*m)")
    if caught:
        axes[2].axvline(hand, color=COLORS[6], ls="--", lw=1.2)
        axes[2].text(hand + 0.1, 2.5, "PID takes over", fontsize=8, color=COLORS[6])
    axes[2].set_xlabel("time (s)")
    axes[2].set_title("Energy swing-up, then the PID catch")
    axes[2].legend(fontsize=8)
    save(fig, os.path.join(OUT, "basin.png"))


def main():
    t0 = time.perf_counter()
    exp1_step()
    exp2_gains()
    exp3_windup()
    exp4_derivative()
    exp5_rate()
    exp6_zn()
    exp7_basin()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
