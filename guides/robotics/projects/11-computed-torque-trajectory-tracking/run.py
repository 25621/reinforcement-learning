"""Project 11 -- Computed-torque trajectory tracking.

The controller and the robot both use project 10's ``dynamics.py``, but they
are not the same object: the SIMULATED ARM is one ``Model`` and the
CONTROLLER'S MODEL is another, so the two can be made to disagree on purpose.
That separation is the whole experiment -- a controller that shares the
simulator's numbers is not being tested, it is being flattered.

Six experiments:

  1. PID alone, gravity compensation, and full computed torque, side by side
  2. the same three as the trajectory speeds up
  3. how wrong the model can be before feedforward stops paying
  4. an unmodelled payload, and what the integral term can and cannot fix
  5. raising the PID gains instead: how far brute force gets you
  6. what the feedforward actually costs per control tick

Runs in about four minutes on a CPU.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJ, "10-inverse-dynamics-from-scratch"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

import matplotlib.pyplot as plt  # noqa: E402

import dynamics as dyn  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

URDF = os.path.join(_PROJ, "02-urdf-visualizer", "models", "arm6.urdf")
ARM = dyn.Model(URDF)  # the "real" robot
N = ARM.n
DT = 1e-3

# Outer-loop PD gains, in acceleration units.  Computed torque turns the arm
# into six independent unit masses, so these are chosen as a second-order
# system directly: wn = 20 rad/s, zeta = 1 (critically damped, no overshoot).
WN, ZETA = 20.0, 1.0
KP = np.full(N, WN ** 2)
KD = np.full(N, 2 * ZETA * WN)
KI = np.full(N, 60.0)

Q0 = np.array([0.0, 0.55, -1.10, 0.0, 0.60, 0.0])
AMP = np.array([0.45, 0.35, 0.50, 0.55, 0.45, 0.60])
PHASE = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])
BASE_HZ = 0.35


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<56s} {value:>12.5f} {unit}")


def reference(t, speed=1.0):
    """A smooth sinusoidal joint trajectory and its first two derivatives."""
    w = 2 * np.pi * BASE_HZ * speed
    return (Q0 + AMP * np.sin(w * t + PHASE),
            AMP * w * np.cos(w * t + PHASE),
            -AMP * w * w * np.sin(w * t + PHASE))


# Every run is 1.0 s long.  The arm starts exactly ON the reference, so there is
# no start-up transient to wait out, and at the base rate of 0.35 Hz one second
# covers about a third of a cycle -- with the six joints' phases staggered, that
# is enough of the trajectory for the RMS to be representative.  Longer runs
# changed no conclusion here and tripled the runtime.
def track(mode, speed=1.0, T=1.0, model=None, payload=0.0, kp=KP, kd=KD, ki=KI,
          tau_limit=True):
    """Run one closed loop.

    ``mode`` selects what the controller feeds forward:
      'pid'      -- nothing.  tau = PID(error) only.
      'gravity'  -- g(q) only: one term of the manipulator equation.
      'ct'       -- full computed torque: M(q) qdd_cmd + C(q,qd) qd + g(q),
                    evaluated in one RNEA call.

    ``model`` is the controller's belief about the arm; ``payload`` adds mass
    to the real arm's tool that the controller knows nothing about.
    """
    ctrl_model = model if model is not None else ARM
    plant = ARM
    if payload > 0.0:
        plant = ARM.scaled(1.0)
        tool = plant.inertials["tool0"]
        plant.inertials["tool0"] = type(tool)(tool.mass + payload, tool.com.copy(), tool.I.copy())

    q, qd, _ = reference(0.0, speed)
    q = q.copy()
    qd = qd.copy()
    integ = np.zeros(N)
    steps = int(T / DT)
    errs = np.zeros((steps, N))
    taus = np.zeros((steps, N))
    ts = np.arange(steps) * DT

    for k in range(steps):
        t = k * DT
        qr, qdr, qddr = reference(t, speed)
        e, ed = qr - q, qdr - qd
        integ += e * DT
        qdd_cmd = qddr + kp * e + kd * ed + ki * integ

        if mode == "pid":
            # No model at all: the PD/PID output IS the torque command.  The
            # gains have to be re-expressed in torque units, so they are scaled
            # by a single representative inertia -- otherwise this arm would be
            # given a hundred times too little torque and the comparison would
            # be a strawman rather than a baseline.
            tau = SCALE * (kp * e + kd * ed + ki * integ)
        elif mode == "gravity":
            tau = SCALE * (kp * e + kd * ed + ki * integ) + dyn.gravity_torque(ctrl_model, q)
        else:  # 'ct'
            tau = dyn.rnea(ctrl_model, q, qd, qdd_cmd)

        if tau_limit:
            tau = np.clip(tau, -ARM.tau_max, ARM.tau_max)
        errs[k], taus[k] = e, tau
        qdd = dyn.forward_dynamics(plant, q, qd, tau)
        qd = qd + DT * qdd
        q = q + DT * qd
        if not np.all(np.isfinite(q)) or np.abs(q - qr).max() > 5:
            errs[k + 1:] = errs[k]
            taus[k + 1:] = taus[k]
            break

    return ts, errs, taus


# The representative inertia that turns acceleration-unit gains into torque
# units for the model-free baselines.  Taken as the diagonal of M at the
# trajectory's mid-point -- the single best constant guess available without
# a model, which is exactly what a hand-tuned joint PID is.
SCALE = np.diag(dyn.mass_matrix(ARM, Q0))


def rms(errs):
    return float(np.sqrt(np.mean(errs ** 2)))


# ---------------------------------------------------------------------------
# 1. The three controllers
# ---------------------------------------------------------------------------
MODES = (("pid", "PID only"), ("gravity", "gravity comp + PID"), ("ct", "computed torque + PID"))


def exp1_three():
    print("[1] PID vs gravity compensation vs computed torque")
    runs = {}
    for mode, label in MODES:
        ts, errs, taus = track(mode, speed=1.0)
        runs[label] = (ts, errs, taus)
        record("1-three", f"{label}: joint error RMS", 1e3 * rms(errs), "mrad")
        record("1-three", f"{label}: worst joint error", 1e3 * float(np.abs(errs).max()), "mrad")
        record("1-three", f"{label}: torque RMS", float(np.sqrt(np.mean(taus ** 2))), "N*m")
    a = rms(runs["PID only"][1])
    b = rms(runs["gravity comp + PID"][1])
    c = rms(runs["computed torque + PID"][1])
    record("1-three", "gravity comp vs PID only", a / b, "x better")
    record("1-three", "computed torque vs PID only", a / c, "x better")
    record("1-three", "computed torque vs gravity comp", b / c, "x better")
    record("1-three", "share of the total gain that gravity comp alone buys",
           100 * (a - b) / (a - c), "%")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.4))
    for k, (label, (ts, errs, taus)) in enumerate(runs.items()):
        axes[0].plot(ts, 1e3 * errs[:, 1], color=COLORS[k], label=label)
        axes[1].semilogy(ts, 1e3 * np.abs(errs).max(axis=1), color=COLORS[k], label=label)
    axes[0].set_ylabel("shoulder-lift error (mrad)")
    axes[0].set_title("Tracking error, joint 2")
    axes[1].set_ylabel("worst joint error (mrad)")
    axes[1].set_title("Worst error across all six joints")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "three_controllers.png"))


# ---------------------------------------------------------------------------
# 2. Speed
# ---------------------------------------------------------------------------
def exp2_speed():
    print("[2] the same three, faster and faster")
    speeds = [0.25, 0.5, 1.0, 2.0, 3.0]
    table = {label: [] for _, label in MODES}
    for s in speeds:
        for mode, label in MODES:
            _, errs, _ = track(mode, speed=s)
            table[label].append(1e3 * rms(errs))
            record("2-speed", f"speed x{s:.2f} {label}: error RMS", table[label][-1], "mrad")
    for label in table:
        ratio = table[label][-1] / table[label][0]
        record("2-speed", f"{label}: error growth from x0.25 to x3.0", ratio, "x")
    record("2-speed", "CT advantage at x0.25", table["PID only"][0] / table["computed torque + PID"][0], "x")
    record("2-speed", "CT advantage at x3.0", table["PID only"][-1] / table["computed torque + PID"][-1], "x")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for k, (_, label) in enumerate(MODES):
        ax.loglog(speeds, table[label], "o-", color=COLORS[k], label=label)
    ax.set_xlabel("speed multiplier")
    ax.set_ylabel("joint error RMS (mrad)")
    ax.set_title("Feedback alone falls behind as the arm speeds up")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "speed.png"))


# ---------------------------------------------------------------------------
# 3. Model error
# ---------------------------------------------------------------------------
def exp3_model_error():
    print("[3] how wrong may the model be?")
    factors = [0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0]
    vals = []
    for f in factors:
        _, errs, _ = track("ct", speed=2.0, model=ARM.scaled(f))
        vals.append(1e3 * rms(errs))
        record("3-model", f"controller mass x{f:.2f}: error RMS", vals[-1], "mrad")
    _, errs_pid, _ = track("pid", speed=2.0)
    _, errs_g, _ = track("gravity", speed=2.0)
    pid_rms = 1e3 * rms(errs_pid)
    record("3-model", "PID-only baseline at the same speed", pid_rms, "mrad")
    record("3-model", "gravity-comp baseline at the same speed", 1e3 * rms(errs_g), "mrad")
    worst = max(vals)
    record("3-model", "worst computed-torque result in the whole sweep", worst, "mrad")
    record("3-model", "  still better than PID only by", pid_rms / worst, "x")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.plot(factors, vals, "o-", color=COLORS[0], label="computed torque")
    ax.axhline(pid_rms, color=COLORS[1], ls="--", label="PID only")
    ax.axhline(1e3 * rms(errs_g), color=COLORS[2], ls=":", label="gravity comp + PID")
    ax.set_xlabel("controller's mass estimate / true mass")
    ax.set_ylabel("joint error RMS (mrad)")
    ax.set_title("A model that is 50% wrong still beats no model")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "model_error.png"))


# ---------------------------------------------------------------------------
# 4. An unmodelled payload
# ---------------------------------------------------------------------------
def payload_model(payload):
    """The controller's model, updated to include a payload it was told about."""
    m = ARM.scaled(1.0)
    tool = m.inertials["tool0"]
    m.inertials["tool0"] = type(tool)(tool.mass + payload, tool.com.copy(), tool.I.copy())
    return m


def exp4_payload():
    print("[4] an unmodelled payload")
    loads = [0.0, 0.5, 1.0, 2.0, 3.0]
    unknown, known, no_i = [], [], []
    for p in loads:
        _, e_u, _ = track("ct", speed=1.0, payload=p)
        _, e_k, _ = track("ct", speed=1.0, payload=p, model=payload_model(p))
        _, e_n, _ = track("ct", speed=1.0, payload=p, ki=np.zeros(N))
        unknown.append(1e3 * rms(e_u))
        known.append(1e3 * rms(e_k))
        no_i.append(1e3 * rms(e_n))
        record("4-payload", f"payload {p:.1f} kg: unknown to the controller", unknown[-1], "mrad")
        record("4-payload", f"payload {p:.1f} kg: weighed and modelled", known[-1], "mrad")
        record("4-payload", f"payload {p:.1f} kg: unknown, and no I term", no_i[-1], "mrad")
    record("4-payload", "3 kg unknown vs no payload", unknown[-1] / unknown[0], "x")
    record("4-payload", "3 kg: what the I term recovers", no_i[-1] / unknown[-1], "x")
    record("4-payload", "3 kg: what WEIGHING the payload recovers", unknown[-1] / known[-1], "x")

    # Why the I term cannot help: the payload does not add a constant torque,
    # it changes M -- and M multiplies the feedback command.
    M_arm = dyn.mass_matrix(ARM, Q0)
    for p in (1.0, 3.0):
        M_pl = dyn.mass_matrix(payload_model(p), Q0)
        ratio = np.linalg.eigvals(np.linalg.solve(M_pl, M_pl - M_arm)).real.max()
        record("4-payload", f"payload {p:.1f} kg: worst eigenvalue of M_true^-1 (M_true - M_model)",
               float(ratio), "")
        record("4-payload", f"  so the feedback gain is multiplied by as little as",
               float(1 - ratio), "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].semilogy(loads, unknown, "s-", color=COLORS[1], label="payload unknown")
    axes[0].semilogy(loads, no_i, "^--", color=COLORS[3], label="payload unknown, no I term")
    axes[0].semilogy(loads, known, "o-", color=COLORS[0], label="payload weighed and modelled")
    axes[0].set_xlabel("payload at the tool (kg)")
    axes[0].set_ylabel("joint error RMS (mrad)")
    axes[0].set_title("The integral term does not rescue an inertia error")
    axes[0].legend(fontsize=7)
    ps = np.linspace(0, 3, 25)
    scale = []
    for p in ps:
        M_pl = dyn.mass_matrix(payload_model(p), Q0)
        scale.append(1 - np.linalg.eigvals(np.linalg.solve(M_pl, M_pl - M_arm)).real.max())
    axes[1].plot(ps, scale, color=COLORS[0])
    axes[1].axhline(0, color=COLORS[6], lw=1.0)
    axes[1].set_xlabel("payload at the tool (kg)")
    axes[1].set_ylabel("smallest surviving fraction of the feedback gain")
    axes[1].set_title("An unmodelled mass shrinks the gain you thought you set")
    save(fig, os.path.join(OUT, "payload.png"))


# ---------------------------------------------------------------------------
# 5. Brute force instead
# ---------------------------------------------------------------------------
def exp5_gains():
    print("[5] raising the PID gains instead of adding a model")
    mults = [1.0, 2.0, 4.0, 8.0, 16.0]
    pid_vals, ct_vals, sat = [], [], []
    for mlt in mults:
        _, e, tau = track("pid", speed=2.0, kp=KP * mlt, kd=KD * np.sqrt(mlt))
        pid_vals.append(1e3 * rms(e))
        sat.append(100.0 * float(np.mean(np.abs(np.abs(tau) - ARM.tau_max) < 1e-9)))
        _, e2, _ = track("ct", speed=2.0, kp=KP * mlt, kd=KD * np.sqrt(mlt))
        ct_vals.append(1e3 * rms(e2))
        record("5-gains", f"gains x{mlt:.0f}: PID-only error RMS", pid_vals[-1], "mrad")
        record("5-gains", f"gains x{mlt:.0f}: torque saturated", sat[-1], "% of ticks")
        record("5-gains", f"gains x{mlt:.0f}: computed-torque error RMS", ct_vals[-1], "mrad")
    record("5-gains", "gain multiplier PID needs to match plain computed torque",
           float(np.interp(ct_vals[0], pid_vals[::-1], mults[::-1])), "x")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.loglog(mults, pid_vals, "s-", color=COLORS[1], label="PID only")
    ax.loglog(mults, ct_vals, "o-", color=COLORS[0], label="computed torque + PID")
    ax.set_xlabel("feedback gain multiplier")
    ax.set_ylabel("joint error RMS (mrad)")
    ax.set_title("Stiffer feedback closes some of the gap, at a price")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "gains.png"))


# ---------------------------------------------------------------------------
# 6. What it costs
# ---------------------------------------------------------------------------
def exp6_cost():
    print("[6] the cost of the feedforward, per control tick")
    q, qd, qdd = reference(0.4, 1.0)
    reps = 400
    t0 = time.perf_counter()
    for _ in range(reps):
        dyn.rnea(ARM, q, qd, qdd)
    t_ct = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        dyn.gravity_torque(ARM, q)
    t_g = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps * 20):
        KP * q + KD * qd
    t_pid = (time.perf_counter() - t0) / (reps * 20)

    record("6-cost", "computed torque (one RNEA)", 1e6 * t_ct, "us")
    record("6-cost", "gravity compensation (one RNEA)", 1e6 * t_g, "us")
    record("6-cost", "the PID arithmetic itself", 1e6 * t_pid, "us")
    record("6-cost", "computed torque as a share of a 1 ms tick", 100 * t_ct / 1e-3, "%")
    record("6-cost", "how much slower than plain PID", t_ct / t_pid, "x")

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    labels = ["PID\narithmetic", "gravity\ncomp", "computed\ntorque"]
    vals = [1e6 * t_pid, 1e6 * t_g, 1e6 * t_ct]
    ax.bar(labels, vals, color=[COLORS[2], COLORS[1], COLORS[0]])
    ax.axhline(1000, color=COLORS[6], ls="--", lw=1.2)
    ax.text(2.4, 1050, "1 ms tick", fontsize=8, color=COLORS[6], ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("time per control tick (us)")
    ax.set_title("This is pure Python; a C implementation is ~100x faster")
    save(fig, os.path.join(OUT, "cost.png"))


def main():
    t0 = time.perf_counter()
    exp1_three()
    exp2_speed()
    exp3_model_error()
    exp4_payload()
    exp5_gains()
    exp6_cost()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
