"""Project 14 -- Tuning a joint that behaves like real hardware.

Six experiments on ``servo.py``'s stand-in for a hobby servo:

  1. identify the friction curve the way you would on a bench
  2. friction feedforward: what it fixes, and where it fixes it
  3. stick-slip: an integral term hunting against static friction
  4. backlash: the hysteresis loop and the motion you lose
  5. relay auto-tuning, and what Ziegler-Nichols does with the answer
  6. loop latency and encoder resolution, the two limits nobody quotes

Runs in about a minute on a CPU.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "01-transform-calculator"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "08-pendulum-pid"))

import matplotlib.pyplot as plt  # noqa: E402

from servo import Joint, GearedJoint, run, constant_velocity_torque  # noqa: E402
from pid import PID  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

DT = 2e-3  # 500 Hz -- a realistic rate for a serial-bus hobby servo


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<56s} {value:>12.5f} {unit}")


def new_joint(**kw):
    return Joint(**kw)


# ---------------------------------------------------------------------------
# 1. Identify the friction
# ---------------------------------------------------------------------------
def exp1_identify():
    print("[1] identifying the friction curve")
    j = new_joint()
    speeds = np.concatenate([-np.logspace(np.log10(0.6), np.log10(0.01), 14),
                             np.logspace(np.log10(0.01), np.log10(0.6), 14)])
    meas = constant_velocity_torque(j, speeds)

    # Fit the two terms you can identify with a straight line: Coulomb friction
    # (the offset at zero speed) and viscous friction (the slope).  Fit only on
    # the fast half, where the Stribeck dip has died away -- fitting through it
    # would bend the line and corrupt BOTH numbers.
    fast = np.abs(speeds) > 0.15
    A = np.stack([np.sign(speeds[fast]), speeds[fast]], axis=1)
    coef, *_ = np.linalg.lstsq(A, meas[fast], rcond=None)
    fc_hat, fv_hat = float(coef[0]), float(coef[1])
    record("1-identify", "true Coulomb friction", j.f_coulomb, "N*m")
    record("1-identify", "fitted Coulomb friction", fc_hat, "N*m")
    record("1-identify", "  error", 100 * abs(fc_hat / j.f_coulomb - 1), "%")
    record("1-identify", "true viscous friction", j.f_viscous, "N*m*s/rad")
    record("1-identify", "fitted viscous friction", fv_hat, "N*m*s/rad")
    record("1-identify", "  error", 100 * abs(fv_hat / j.f_viscous - 1), "%")

    # The Stribeck peak is what the straight line CANNOT see.
    slow = np.abs(speeds) < 0.03
    peak = float(np.max(np.abs(meas[slow])))
    line_at_peak = fc_hat + fv_hat * 0.01
    record("1-identify", "measured torque at 0.01 rad/s", peak, "N*m")
    record("1-identify", "what the straight-line fit predicts there", line_at_peak, "N*m")
    record("1-identify", "how much the fit under-predicts near zero speed",
           100 * (peak / line_at_peak - 1), "%")
    record("1-identify", "true static friction (the break-away torque)", j.f_static, "N*m")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    order = np.argsort(speeds)
    ax.plot(speeds[order], meas[order], "o", ms=4, color=COLORS[0], label="measured")
    grid = np.linspace(-0.65, 0.65, 400)
    ax.plot(grid, np.sign(grid) * fc_hat + fv_hat * grid, "--", color=COLORS[1],
            label="Coulomb + viscous fit")
    ax.plot(grid, j.friction(grid), color=COLORS[2], lw=1.2, label="true Stribeck curve")
    ax.set_xlabel("joint speed (rad/s)")
    ax.set_ylabel("torque needed to hold that speed (N*m)")
    ax.set_title("The dip near zero speed is the part a straight line misses")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "friction.png"))
    return fc_hat, fv_hat


# ---------------------------------------------------------------------------
# 2. Friction feedforward
# ---------------------------------------------------------------------------
def exp2_feedforward(fc_hat, fv_hat):
    print("[2] friction feedforward")
    j = new_joint()
    # A deliberately SLOW move (0.08 Hz, +-0.3 rad, peak speed 0.15 rad/s).
    # Friction is a fixed torque, so it matters most when the torque you
    # actually need is small -- at high speed the inertia term swamps it.
    amp, freq = 0.30, 0.08
    kp, kd = 12.0, 0.6

    def make(ff_kind, ki):
        ctrl = PID(kp, ki, kd, dt=DT, u_min=-j.tau_max, u_max=j.tau_max, d_filter_hz=25.0)

        def controller(t, meas):
            return ctrl(amp * np.sin(2 * np.pi * freq * t), meas)

        def ff(t):
            w_ref = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t)
            g = j.m_g_l * np.cos(amp * np.sin(2 * np.pi * freq * t))
            if ff_kind == "none":
                return 0.0
            if ff_kind == "gravity":
                return g
            # tanh instead of sign: a hard sign() would chatter between +fc and
            # -fc every time the reference speed crosses zero, which on hardware
            # is an audible buzz.  tanh smooths the switch over a small speed.
            return g + fc_hat * np.tanh(w_ref / 0.02) + fv_hat * w_ref

        return controller, ff

    out = {}
    T = 20.0
    for ki, family in ((0.0, "PD"), (25.0, "PID")):
        for kind, short in (("none", "no feedforward"), ("gravity", "gravity FF"),
                            ("full", "gravity + friction FF")):
            label = f"{family}, {short}"
            controller, ff = make(kind, ki)
            ts, TH, W, TAU = run(j, controller, T=T, dt_ctrl=DT, feedforward=ff)
            ref = amp * np.sin(2 * np.pi * freq * ts)
            err = TH - ref
            keep = ts > 0.2 * T  # drop the start-up transient
            out[label] = (ts, err, TAU, ref)
            record("2-ff", f"{label}: tracking error RMS",
                   1e3 * float(np.sqrt(np.mean(err[keep] ** 2))), "mrad")
            record("2-ff", f"{label}: worst error", 1e3 * float(np.abs(err[keep]).max()), "mrad")
    def rms(label):
        ts, err, _, _ = out[label]
        return float(np.sqrt(np.mean(err[ts > 0.2 * T] ** 2)))
    record("2-ff", "PD: friction FF improvement over no feedforward",
           rms("PD, no feedforward") / rms("PD, gravity + friction FF"), "x")
    record("2-ff", "PD: what the friction term adds on top of gravity FF",
           rms("PD, gravity FF") / rms("PD, gravity + friction FF"), "x")
    record("2-ff", "PID: friction FF improvement over no feedforward",
           rms("PID, no feedforward") / rms("PID, gravity + friction FF"), "x")
    record("2-ff", "PID: what the friction term adds on top of gravity FF",
           rms("PID, gravity FF") / rms("PID, gravity + friction FF"), "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    for ax, family in ((axes[0], "PD"), (axes[1], "PID")):
        for k, short in enumerate(("no feedforward", "gravity FF", "gravity + friction FF")):
            ts, err, _, _ = out[f"{family}, {short}"]
            ax.plot(ts, 1e3 * err, color=COLORS[k], label=short)
        ax.set_xlim(4, 20)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("tracking error (mrad)")
        ax.set_title(f"{family} feedback")
        ax.legend(fontsize=7)
    axes[0].set_ylim(-60, 60)
    axes[1].set_ylim(-20, 20)
    save(fig, os.path.join(OUT, "feedforward.png"))


# ---------------------------------------------------------------------------
# 3. Stick-slip
# ---------------------------------------------------------------------------
def exp3_stickslip(fc_hat, fv_hat):
    print("[3] stick-slip hunting")
    # A STICKIER joint than the default: 0.25 N*m to break free against
    # 0.055 N*m once moving, a 4.5:1 ratio.  Real joints drift this way as the
    # grease ages and the bearings wear, and it is the ratio -- not the absolute
    # friction -- that decides whether a joint hunts.
    j = new_joint(f_static=0.25)
    target = 0.30
    out = {}
    record("3-stickslip", "break-away torque of this joint", j.f_static, "N*m")
    record("3-stickslip", "torque once it is moving", j.f_coulomb, "N*m")
    record("3-stickslip", "static-to-moving ratio", j.f_static / j.f_coulomb, "x")

    def go(ki, ff_kind, label):
        ctrl = PID(6.0, ki, 0.35, dt=DT, u_min=-j.tau_max, u_max=j.tau_max, d_filter_hz=25.0)

        def controller(t, meas):
            return ctrl(target, meas)

        def ff(t):
            g = j.m_g_l * np.cos(target)
            if ff_kind == "gravity":
                return g
            # Friction feedforward at a STANDSTILL has no reference velocity to
            # key off, so it uses the sign of the error instead: "if I still
            # need to go up, pre-pay most of the break-away torque".  0.8 of the
            # estimate, not 1.0 -- over-paying would push the joint past the
            # target and start the hunt from the other side.
            return g + 0.8 * j.f_static * np.tanh(ctrl.last_terms[0] / 0.05)

        jj = new_joint(f_static=0.25)
        ts, TH, W, TAU = run(jj, controller, T=20.0, dt_ctrl=DT,
                             feedforward=None if ff_kind == "none" else ff)
        tail = ts > 6.0
        e = TH[tail] - target
        amp = 1e3 * 0.5 * float(e.max() - e.min())
        crossings = np.nonzero(np.diff(np.sign(e - e.mean())) != 0)[0]
        period = 2 * DT * float(np.mean(np.diff(crossings))) if len(crossings) > 3 else np.nan
        out[label] = (ts, TH, TAU)
        record("3-stickslip", f"{label}: residual amplitude", amp, "mrad")
        record("3-stickslip", f"{label}: hunting period", period, "s")
        record("3-stickslip", f"{label}: steady-state error", 1e3 * float(np.mean(e)), "mrad")
        return amp

    a_pid = go(4.0, "none", "PID, no feedforward")
    a_noi = go(0.0, "none", "PD only (Ki = 0)")
    a_g = go(4.0, "gravity", "PID + gravity feedforward")
    a_ff = go(4.0, "friction", "PID + friction feedforward")
    record("3-stickslip", "amplitude reduction from removing the I term", a_pid / max(a_noi, 1e-9), "x")
    record("3-stickslip", "amplitude reduction from GRAVITY feedforward", a_pid / max(a_g, 1e-9), "x")
    record("3-stickslip", "amplitude reduction from FRICTION feedforward", a_pid / max(a_ff, 1e-9), "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    for k, (label, (ts, TH, TAU)) in enumerate(out.items()):
        axes[0].plot(ts, 1e3 * (TH - target), color=COLORS[k], label=label)
        axes[1].plot(ts, TAU, color=COLORS[k], lw=1.0, label=label)
    axes[0].set_xlim(6, 20)
    axes[0].set_ylabel("error from the target (mrad)")
    axes[0].set_title("The integral term winds up against stiction, then lets go")
    axes[1].set_xlim(6, 20)
    axes[1].set_ylabel("applied torque (N*m)")
    axes[1].axhline(0.25, color=COLORS[6], ls="--", lw=1.0)
    axes[1].set_title("The dashed line is the break-away torque")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "stickslip.png"))


# ---------------------------------------------------------------------------
# 4. Backlash
# ---------------------------------------------------------------------------
def exp4_backlash():
    print("[4] backlash")
    dt = 2e-5
    T = 12.0
    n = int(T / dt)
    kp, kd = 30.0, 1.2

    # A back-and-forth move that DWELLS at each end.  A pure sine never stops,
    # so its tracking error is dominated by ordinary lag and the backlash hides
    # inside it.  Waiting at each end lets the lag decay to nothing, and
    # whatever offset is left is the dead band and only the dead band.
    def REF(t):
        return 0.30 * np.clip(1.8 * np.sin(2 * np.pi * 0.12 * t), -1.0, 1.0)

    def go(sense):
        """``sense`` is which encoder the loop closes on: 'motor' or 'load'.

        This is a real design choice, not a detail.  Putting the encoder on the
        motor is cheap, gives high resolution (the gear ratio multiplies it) and
        is what almost every hobby servo does.  Putting it on the load costs
        more and needs a bigger, coarser encoder.  The experiment measures what
        the cheap choice actually costs.
        """
        g = GearedJoint()
        st = np.zeros(4)
        ts = np.arange(0, n, 20) * dt
        M = np.zeros(len(ts))
        L = np.zeros(len(ts))
        idx = 0
        for k in range(n):
            t = k * dt
            ref = REF(t)
            th_fb, w_fb = (st[0], st[1]) if sense == "motor" else (st[2], st[3])
            tau = kp * (ref - th_fb) + kd * (0.0 - w_fb) + g.m_g_l * np.cos(st[2])
            st = g.step(st, tau, dt)
            if k % 20 == 0:
                M[idx], L[idx] = st[0], st[2]
                idx += 1
        return ts, M, L, g

    out = {}
    for sense in ("load", "motor"):
        ts, M, L, g = go(sense)
        ref = REF(ts)
        # Score only the DWELLS, where the reference is standing still.
        dwell = (np.abs(np.abs(ref) - 0.30) < 1e-9) & (ts > 3.0)
        out[sense] = (ts, M, L, ref)
        e_sensed = (M if sense == "motor" else L) - ref
        e_true = L - ref
        record("4-backlash", f"encoder on the {sense}: error the CONTROLLER sees at rest",
               1e3 * float(np.sqrt(np.mean(e_sensed[dwell] ** 2))), "mrad")
        record("4-backlash", f"encoder on the {sense}: error the TOOL actually has at rest",
               1e3 * float(np.sqrt(np.mean(e_true[dwell] ** 2))), "mrad")
    ts, M, L, ref = out["motor"]
    tail = (np.abs(np.abs(ref) - 0.30) < 1e-9) & (ts > 3.0)
    g = GearedJoint()
    record("4-backlash", "gear dead band (half-width)", g.backlash, "rad")
    record("4-backlash", "  in degrees", np.degrees(g.backlash), "deg")
    record("4-backlash", "worst motor-to-load discrepancy, motor-side sensing",
           float(np.max(np.abs(M[tail] - L[tail]))), "rad")
    record("4-backlash", "how much the motor encoder flatters the result",
           float(np.sqrt(np.mean((L[tail] - ref[tail]) ** 2))
                 / max(np.sqrt(np.mean((M[tail] - ref[tail]) ** 2)), 1e-12)), "x")
    TH_M, TH_L = M, L

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].plot(ts, TH_M, color=COLORS[1], lw=1.0, label="motor side (what it measures)")
    axes[0].plot(ts, TH_L, color=COLORS[0], lw=1.0, label="load side (what matters)")
    axes[0].plot(ts, ref, "--", color=COLORS[6], lw=1.0, label="reference")
    axes[0].set_xlim(4, 12)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("angle (rad)")
    axes[0].set_title("The load lags whenever the direction reverses")
    axes[0].legend(fontsize=7)
    axes[1].plot(TH_M[tail], TH_L[tail], color=COLORS[0], lw=1.0)
    axes[1].plot([-0.4, 0.4], [-0.4, 0.4], "--", color=COLORS[6], lw=1.0)
    axes[1].set_xlabel("motor angle (rad)")
    axes[1].set_ylabel("load angle (rad)")
    axes[1].set_title("The hysteresis loop backlash draws")
    save(fig, os.path.join(OUT, "backlash.png"))


# ---------------------------------------------------------------------------
# 5. Relay auto-tuning
# ---------------------------------------------------------------------------
def relay_test(j, d=0.40, T=10.0, dt_ctrl=DT, setpoint=0.0):
    """Astrom-Hagglund relay feedback: replace the controller with a switch.

    Drive +d whenever the joint is below the setpoint and -d whenever it is
    above.  The loop settles into a steady oscillation whose amplitude ``a`` and
    period ``Tu`` you can read off, and describing-function analysis gives the
    ultimate gain directly:

        Ku = 4 d / (pi a)

    The 4/pi is the size of the FUNDAMENTAL sine hiding inside a square wave --
    the relay outputs a square wave, but the joint (being an inertia) only
    really responds to its slowest component, so the effective gain of the relay
    is that component's amplitude divided by the oscillation it produced.

    This is much safer on hardware than the textbook "raise Kp until it hums":
    the relay output is bounded by ``d`` by construction, so the experiment
    cannot run away, whereas a gain ramp finds the stability edge by crossing it.
    """
    # Gravity is compensated first.  Without it the relay is not symmetric
    # about the setpoint: gravity biases every swing one way and the joint
    # simply falls instead of oscillating (this is what happened first).
    def controller(t, meas):
        return j.m_g_l * np.cos(meas) + (d if meas < setpoint else -d)

    ts, TH, W, TAU = run(j, controller, T=T, dt_ctrl=dt_ctrl, th0=setpoint, delay_steps=1)
    tail = ts > T * 0.4
    e = TH[tail] - setpoint
    a = 0.5 * float(e.max() - e.min())
    crossings = np.nonzero(np.diff(np.sign(e)) != 0)[0]
    tu = 2 * dt_ctrl * float(np.mean(np.diff(crossings))) if len(crossings) > 3 else np.nan
    ku = 4 * d / (np.pi * a)
    return ku, tu, a, ts, TH


def exp5_relay():
    print("[5] relay auto-tuning")
    j = new_joint()
    ku, tu, a, ts, TH = relay_test(j)
    record("5-relay", "relay amplitude d", 0.40, "N*m")
    record("5-relay", "limit-cycle amplitude a", 1e3 * a, "mrad")
    record("5-relay", "ultimate period Tu", 1e3 * tu, "ms")
    record("5-relay", "ultimate gain Ku = 4d/(pi a)", ku, "N*m/rad")

    zn = dict(kp=0.6 * ku, ki=1.2 * ku / tu, kd=0.075 * ku * tu)
    hand = dict(kp=6.0, ki=12.0, kd=0.35)
    record("5-relay", "ZN Kp", zn["kp"], "N*m/rad")
    record("5-relay", "ZN Ki", zn["ki"], "N*m/(rad*s)")
    record("5-relay", "ZN Kd", zn["kd"], "N*m*s/rad")

    out = {}
    for label, gains in (("hand-tuned", hand), ("Ziegler-Nichols from the relay", zn)):
        ctrl = PID(gains["kp"], gains["ki"], gains["kd"], dt=DT,
                   u_min=-j.tau_max, u_max=j.tau_max, d_filter_hz=25.0)
        jj = new_joint()

        def controller(t, meas):
            return ctrl(0.30 if t > 0.3 else 0.0, meas)

        ts2, TH2, W2, TAU2 = run(jj, controller, T=6.0, dt_ctrl=DT)
        out[label] = (ts2, TH2)
        peak = float(np.max(TH2))
        record("5-relay", f"{label}: overshoot", 100 * (peak - 0.30) / 0.30, "%")
        tail = ts2 > 4.0
        record("5-relay", f"{label}: residual wobble", 1e3 * float(np.std(TH2[tail])), "mrad")
        record("5-relay", f"{label}: steady error", 1e3 * float(np.mean(TH2[tail]) - 0.30), "mrad")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].plot(ts, 1e3 * TH, color=COLORS[0])
    axes[0].set_xlim(4, 8)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("angle (mrad)")
    axes[0].set_title(f"The relay's limit cycle: a = {1e3 * a:.1f} mrad, Tu = {1e3 * tu:.0f} ms")
    for k, (label, (ts2, TH2)) in enumerate(out.items()):
        axes[1].plot(ts2, TH2, color=COLORS[k], label=label)
    axes[1].axhline(0.30, color=COLORS[6], ls="--", lw=1.0)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("angle (rad)")
    axes[1].set_title("The step response each gain set gives")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "relay.png"))


# ---------------------------------------------------------------------------
# 6. Latency and encoder resolution
# ---------------------------------------------------------------------------
def exp6_limits():
    print("[6] latency and encoder resolution")
    # (a) latency vs the highest usable gain
    delays = [0, 1, 2, 4, 8]
    kus = []
    for d in delays:
        j = new_joint()
        lo, hi = 1.0, 400.0
        for _ in range(22):
            mid = 0.5 * (lo + hi)
            ctrl = PID(mid, 0.0, 0.0, dt=DT, u_min=-j.tau_max, u_max=j.tau_max)
            jj = new_joint()

            def controller(t, meas):
                return ctrl(0.0, meas)

            ts, TH, W, TAU = run(jj, controller, T=3.0, dt_ctrl=DT, th0=0.02, delay_steps=d)
            late = np.abs(TH[ts > 2.0])
            early = np.abs(TH[(ts > 0.5) & (ts < 1.5)])
            grows = (not np.all(np.isfinite(late))) or late.max() > early.max()
            if grows:
                hi = mid
            else:
                lo = mid
        kus.append(0.5 * (lo + hi))
        record("6-limits", f"delay {d} ticks ({1e3 * d * DT:.0f} ms): highest stable Kp",
               kus[-1], "N*m/rad")
    record("6-limits", "gain lost going from 0 to 8 ticks of delay", kus[0] / kus[-1], "x")

    # (b) encoder resolution vs torque chatter
    # Measured while the joint is MOVING slowly, not standing still: a stuck
    # joint reads the same encoder count every tick, so its derivative is
    # exactly zero and it would look beautifully quiet no matter how coarse the
    # encoder is.  Quantisation noise appears when the reading is stepping.
    counts = [512, 1024, 2048, 4096, 16384]
    chatter, chatter_f, err_f = [], [], []
    amp, freq = 0.30, 0.08
    for c in counts:
        for filt, store in ((None, chatter), (20.0, chatter_f)):
            j = new_joint(counts_per_rev=c)
            ctrl = PID(12.0, 25.0, 0.6, dt=DT, u_min=-j.tau_max, u_max=j.tau_max,
                       d_filter_hz=filt)

            def controller(t, meas):
                return ctrl(amp * np.sin(2 * np.pi * freq * t), meas)

            ts, TH, W, TAU = run(j, controller, T=12.0, dt_ctrl=DT)
            tail = ts > 4.0
            store.append(float(np.std(np.diff(TAU[tail]))))
            if filt == 20.0:
                err_f.append(1e3 * float(np.sqrt(np.mean(
                    (TH[tail] - amp * np.sin(2 * np.pi * freq * ts[tail])) ** 2))))
        record("6-limits", f"{c} counts/rev: torque chatter, raw D", chatter[-1], "N*m")
        record("6-limits", f"{c} counts/rev: torque chatter, 20 Hz filtered D", chatter_f[-1], "N*m")
        record("6-limits", f"{c} counts/rev: tracking error, filtered D", err_f[-1], "mrad")
    record("6-limits", "chatter reduction from filtering at 512 counts",
           chatter[0] / max(chatter_f[0], 1e-12), "x")
    record("6-limits", "chatter at 512 counts vs at 16384 counts, raw D",
           chatter[0] / max(chatter[-1], 1e-12), "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.2))
    axes[0].plot([1e3 * d * DT for d in delays], kus, "o-", color=COLORS[0])
    axes[0].set_xlabel("loop latency (ms)")
    axes[0].set_ylabel("highest stable proportional gain")
    axes[0].set_yscale("log")
    axes[0].set_title("Every millisecond of delay costs you gain")
    axes[1].loglog(counts, chatter, "o-", color=COLORS[1], label="raw derivative")
    axes[1].loglog(counts, chatter_f, "s-", color=COLORS[0], label="20 Hz filtered")
    axes[1].set_xlabel("encoder counts per revolution")
    axes[1].set_ylabel("torque chatter (N*m per tick)")
    axes[1].set_title("Quantisation, amplified by Kd")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "limits.png"))


def main():
    t0 = time.perf_counter()
    fc_hat, fv_hat = exp1_identify()
    exp2_feedforward(fc_hat, fv_hat)
    exp3_stickslip(fc_hat, fv_hat)
    exp4_backlash()
    exp5_relay()
    exp6_limits()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
