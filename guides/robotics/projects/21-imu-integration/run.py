"""Project 21 -- An IMU on a desk: how fast does dead reckoning fall apart?

Seven experiments:

  1. sitting perfectly still: angle in seconds, position in seconds
  2. which error source actually kills you (it is not the accelerometer)
  3. the Allan deviation, and reading a sensor's real noise off it
  4. calibrating the bias by averaging at rest: how long is long enough?
  5. how you integrate the rotation, and when it starts to matter
  6. zero-velocity updates
  7. the error-state covariance against a Monte Carlo

Runs in about two minutes on a CPU.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from imu import (G, IMUParams, simulate, simulate_static, integrate,     # noqa: E402
                 allan_deviation, angle_between, quat_to_R, quat_from_rotvec,
                 orthonormalize, orthogonality_error)
from plot_style import COLORS, use_style                                 # noqa: E402

import matplotlib.pyplot as plt                                          # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []
P = IMUParams(rate=100.0)


def log(row):
    RESULTS.append(row)
    print("   ", " | ".join(f"{k}={v}" for k, v in row.items()), flush=True)


def drift_run(params, seconds, rng, **kw):
    n = int(seconds * params.rate)
    gyro, accel, _ = simulate_static(params, n, rng)
    Rs, vs, ps = integrate(gyro, accel, params.dt, stride=int(params.rate // 10), **kw)
    t = np.arange(len(ps)) * (params.dt * int(params.rate // 10))
    ang = np.array([angle_between(np.eye(3), R) for R in Rs])
    return t, ang, np.linalg.norm(vs, axis=1), np.linalg.norm(ps, axis=1)


# --------------------------------------------------------------------------
# 1. sitting still
# --------------------------------------------------------------------------

def stage_still():
    print("\n[1] a perfectly still IMU")
    runs = []
    for s in range(24):
        runs.append(drift_run(P, 60.0, np.random.default_rng(100 + s)))
    t = runs[0][0]
    ang = np.median([r[1] for r in runs], axis=0)
    vel = np.median([r[2] for r in runs], axis=0)
    pos = np.median([r[3] for r in runs], axis=0)

    for sec in (1, 2, 5, 10, 30, 60):
        i = int(np.argmin(np.abs(t - sec)))
        log(dict(stage="still", seconds=sec, angle_deg=round(float(ang[i]), 3),
                 velocity_m_s=round(float(vel[i]), 3), position_m=round(float(pos[i]), 3)))

    m = (t > 1) & (t <= 60)
    pa = np.polyfit(np.log(t[m]), np.log(ang[m]), 1)[0]
    pv = np.polyfit(np.log(t[m]), np.log(vel[m]), 1)[0]
    pp = np.polyfit(np.log(t[m]), np.log(pos[m]), 1)[0]
    log(dict(stage="still_growth", angle_exponent=round(float(pa), 2),
             velocity_exponent=round(float(pv), 2), position_exponent=round(float(pp), 2)))

    fig, ax = plt.subplots(1, 3, figsize=(12, 3.1))
    for a, y, lbl, col in zip(ax, [ang, vel, pos],
                              ["attitude error (deg)", "velocity error (m/s)",
                               "position error (m)"], [COLORS[0], COLORS[1], COLORS[2]]):
        for r, yy in zip(runs, [1, 2, 3]):
            pass
        for r in runs:
            a.plot(r[0], r[{"attitude error (deg)": 1, "velocity error (m/s)": 2,
                            "position error (m)": 3}[lbl]], color=col, alpha=0.12, lw=0.8)
        a.plot(t, y, color=col, lw=2)
        a.set_xlabel("time (s)"); a.set_ylabel(lbl)
        a.set_xscale("log"); a.set_yscale("log")
    ax[0].set_title(f"grows as $t^{{{pa:.2f}}}$")
    ax[1].set_title(f"grows as $t^{{{pv:.2f}}}$")
    ax[2].set_title(f"grows as $t^{{{pp:.2f}}}$")
    fig.suptitle("24 runs of an IMU that never moves", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "still.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 2. which error source
# --------------------------------------------------------------------------

def stage_sources():
    print("\n[2] which error source dominates")
    base = dict(rate=100.0)
    cases = {
        "everything": {},
        "gyro white noise only": dict(gyro_bias=0, gyro_bias_walk=0, accel_noise=0,
                                      accel_bias=0, accel_bias_walk=0),
        "gyro bias only": dict(gyro_noise=0, gyro_bias_walk=0, accel_noise=0,
                               accel_bias=0, accel_bias_walk=0),
        "accel white noise only": dict(gyro_noise=0, gyro_bias=0, gyro_bias_walk=0,
                                       accel_bias=0, accel_bias_walk=0),
        "accel bias only": dict(gyro_noise=0, gyro_bias=0, gyro_bias_walk=0,
                                accel_noise=0, accel_bias_walk=0),
    }
    table = {}
    for name, kw in cases.items():
        p = IMUParams(**base, **kw)
        pos = {1: [], 10: [], 60: []}
        for s in range(16):
            t, ang, vel, ps = drift_run(p, 60.0, np.random.default_rng(300 + s))
            for sec in pos:
                pos[sec].append(ps[int(np.argmin(np.abs(t - sec)))])
        table[name] = {k: float(np.median(v)) for k, v in pos.items()}
        log(dict(stage="sources", source=name,
                 pos_1s_mm=round(table[name][1] * 1000, 2),
                 pos_10s_m=round(table[name][10], 3),
                 pos_60s_m=round(table[name][60], 2)))

    # the coupling made explicit: a pure tilt error, no sensor noise at all
    for tilt in (0.05, 0.1, 0.5, 1.0):
        p = IMUParams(rate=100.0, gyro_noise=0, gyro_bias=0, gyro_bias_walk=0,
                      accel_noise=0, accel_bias=0, accel_bias_walk=0)
        n = int(10 * p.rate)
        gyro, accel, _ = simulate_static(p, n, np.random.default_rng(0), tilt_deg=tilt)
        # the integrator is told the sensor is level; it is not
        _, _, ps = integrate(gyro, accel, p.dt, stride=10)
        log(dict(stage="tilt_leak", tilt_deg=tilt,
                 leaked_accel_m_s2=round(float(9.81 * np.sin(np.radians(tilt))), 4),
                 position_after_10s_m=round(float(np.linalg.norm(ps[-1])), 3)))

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    names = list(table)
    x = np.arange(len(names))
    for k, (sec, col) in enumerate(zip((1, 10, 60), [COLORS[0], COLORS[1], COLORS[2]])):
        ax.bar(x + (k - 1) * 0.27, [max(table[n][sec], 1e-6) for n in names], 0.27,
               color=col, label=f"after {sec} s")
    ax.set_yscale("log"); ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" only", "").replace(" ", "\n") for n in names], fontsize=7)
    ax.set_ylabel("position error (m)"); ax.legend(fontsize=8)
    ax.set_title("turn one error source on at a time")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sources.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 3. Allan deviation
# --------------------------------------------------------------------------

def stage_allan():
    print("\n[3] Allan deviation of a long static log")
    p = IMUParams(rate=100.0)
    n = int(3000 * p.rate)                       # 50 minutes of data
    gyro, accel, _ = simulate_static(p, n, np.random.default_rng(9))
    ad = allan_deviation(gyro[:, 0], p.dt)
    aa = allan_deviation(accel[:, 0], p.dt)

    # read the white-noise density off the tau = 1 s point of the -1/2 slope
    def read_white(curve):
        i = int(np.argmin(np.abs(curve[:, 0] - 1.0)))
        return float(curve[i, 1] * np.sqrt(curve[i, 0]))

    log(dict(stage="allan", sensor="gyro",
             true_density=P.gyro_noise, read_from_curve=round(read_white(ad), 5),
             min_deviation=round(float(ad[:, 1].min()), 6),
             tau_at_min_s=round(float(ad[np.argmin(ad[:, 1]), 0]), 1)))
    log(dict(stage="allan", sensor="accel",
             true_density=P.accel_noise, read_from_curve=round(read_white(aa), 5),
             min_deviation=round(float(aa[:, 1].min()), 6),
             tau_at_min_s=round(float(aa[np.argmin(aa[:, 1]), 0]), 1)))

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.2))
    for a, curve, lbl, col in ((ax[0], ad, "gyro (rad/s)", COLORS[0]),
                               (ax[1], aa, "accelerometer (m/s$^2$)", COLORS[1])):
        a.loglog(curve[:, 0], curve[:, 1], "o-", color=col, ms=3)
        ref = curve[0, 1] * (curve[:, 0] / curve[0, 0]) ** -0.5
        a.loglog(curve[:, 0], ref, "--", color=COLORS[6], lw=1,
                 label=r"slope $-1/2$: white noise")
        i = int(np.argmin(curve[:, 1]))
        a.plot(curve[i, 0], curve[i, 1], "*", color=COLORS[3], ms=13,
               label=f"bias instability at {curve[i, 0]:.0f} s")
        a.set_xlabel(r"averaging time $\tau$ (s)"); a.set_ylabel("Allan deviation")
        a.set_title(lbl); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "allan.png"))
    plt.close(fig)
    return ad


# --------------------------------------------------------------------------
# 4. bias calibration
# --------------------------------------------------------------------------

def stage_bias_cal(ad):
    print("\n[4] calibrating the bias by averaging at rest")
    rows = []
    for cal_s in (0, 0.5, 2, 5, 20, 60, 200, 600):
        errs = []
        for s in range(30):
            rng = np.random.default_rng(500 + s)
            n_cal = int(cal_s * P.rate)
            n_run = int(20 * P.rate)
            gyro, accel, _ = simulate_static(P, n_cal + n_run, rng)
            bg = gyro[:n_cal].mean(axis=0) if n_cal else np.zeros(3)
            ba = (accel[:n_cal].mean(axis=0) + G) if n_cal else np.zeros(3)
            _, _, ps = integrate(gyro[n_cal:], accel[n_cal:], P.dt, bg=bg, ba=ba,
                                 stride=10)
            errs.append(float(np.linalg.norm(ps[-1])))
        rows.append((cal_s, float(np.median(errs))))
        log(dict(stage="bias_cal", calibration_seconds=cal_s,
                 position_error_after_20s_m=round(rows[-1][1], 3)))
    a = np.array(rows)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.plot(np.maximum(a[:, 0], 0.2), a[:, 1], "o-", color=COLORS[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("seconds spent averaging at rest first")
    ax.set_ylabel("position error after 20 s of motion (m)")
    ax.set_title("longer calibration helps, then stops helping")
    i = int(np.argmin(ad[:, 1]))
    ax.axvline(ad[i, 0], color=COLORS[1], ls="--", lw=1)
    ax.text(ad[i, 0] * 1.1, a[:, 1].max() * 0.6,
            f"Allan minimum\nat {ad[i, 0]:.0f} s", color=COLORS[1], fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "bias_cal.png"))
    plt.close(fig)


# --------------------------------------------------------------------------
# 5. how you integrate the rotation
# --------------------------------------------------------------------------

def stage_methods():
    print("\n[5] integrating the rotation: exact, midpoint, and the shortcut")
    for rate_dps in (30, 180, 720):
        w = np.radians(rate_dps)
        # a "coning" motion: the axis itself rotates, which is the case where
        # the shortcut is worst -- and the case a real hand-held sensor is in
        for dt, hz in ((1 / 400, 400), (1 / 100, 100), (1 / 25, 25)):
            n = int(4.0 / dt)
            t = np.arange(n) * dt
            gyro = np.stack([w * np.cos(2 * t), w * np.sin(2 * t),
                             np.full(n, w * 0.3)], axis=1)
            # ground truth by heavy oversampling of the same rates
            sub = 40
            Rt = np.eye(3)
            for i in range(n * sub):
                tt = i * dt / sub
                ww = np.array([w * np.cos(2 * tt), w * np.sin(2 * tt), w * 0.3])
                Rt = Rt @ quat_to_R(quat_from_rotvec(ww * dt / sub))
            row = dict(stage="methods", rate_dps=rate_dps, sample_hz=hz)
            for m in ("exp", "euler_orth", "euler"):
                Rs, _, _ = integrate(gyro, np.tile(-G, (n, 1)), dt, method=m,
                                     stride=max(n - 1, 1))
                # The raw Euler update stops being a rotation matrix, and
                # angle_between is meaningless on one that is not.  Snap it
                # back before measuring the angle, and report the drift away
                # from orthogonality as its own number.
                row[m + "_err_deg"] = round(angle_between(Rt, orthonormalize(Rs[-1])), 4)
                if m == "euler":
                    row["euler_not_a_rotation_by"] = round(orthogonality_error(Rs[-1]), 5)
            log(row)


# --------------------------------------------------------------------------
# 6. zero-velocity updates
# --------------------------------------------------------------------------

def stage_zupt():
    print("\n[6] zero-velocity updates")
    for every_s in (None, 10, 2, 1, 0.5):
        errs = []
        for s in range(10):
            rng = np.random.default_rng(700 + s)
            n = int(60 * P.rate)
            gyro, accel, _ = simulate_static(P, n, rng)
            ze = None if every_s is None else int(every_s * P.rate)
            _, _, ps = integrate(gyro, accel, P.dt, zupt_every=ze, stride=10)
            errs.append(float(np.linalg.norm(ps[-1])))
        log(dict(stage="zupt", zupt_every_s=every_s,
                 position_error_after_60s_m=round(float(np.median(errs)), 3)))


# --------------------------------------------------------------------------
# 7. the error-state covariance
# --------------------------------------------------------------------------

def stage_error_state():
    print("\n[7] error-state covariance against a Monte Carlo")
    p = IMUParams(rate=100.0, gyro_bias=0, gyro_bias_walk=0,
                  accel_bias=0, accel_bias_walk=0)     # white noise only
    T = 20.0
    n = int(T * p.rate)
    dt = p.dt

    # linear error-state model for a level, static sensor:
    #   d(dtheta)/dt = -n_g
    #   d(dv)/dt     = -[f x] dtheta + n_a         f = specific force = -G
    #   d(dp)/dt     = dv
    # the -[f x] term is the tilt-to-acceleration coupling, in one matrix block
    f = -G
    S = np.array([[0, -f[2], f[1]], [f[2], 0, -f[0]], [-f[1], f[0], 0]])
    F = np.zeros((9, 9))
    F[3:6, 0:3] = -S
    F[6:9, 3:6] = np.eye(3)
    Q = np.zeros((9, 9))
    Q[0:3, 0:3] = np.eye(3) * p.gyro_noise ** 2
    Q[3:6, 3:6] = np.eye(3) * p.accel_noise ** 2
    P9 = np.zeros((9, 9))
    Phi = np.eye(9) + F * dt
    pred = []
    for i in range(n):
        P9 = Phi @ P9 @ Phi.T + Q * dt
        if i % 10 == 0:
            pred.append((i * dt, np.sqrt(np.diag(P9))))

    # Compare PER AXIS.  The covariance diagonal is the spread of one
    # component; the length of a 3-vector of such components is a different
    # quantity (about 1.6 sigma on average), and comparing the two would make
    # a correct model look 60% wrong.
    mc = []
    for s in range(60):
        rng = np.random.default_rng(900 + s)
        gyro, accel, _ = simulate_static(p, n, rng)
        Rs, vs, ps = integrate(gyro, accel, dt, stride=10)
        ang = np.array([[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]
                        for R in Rs]) / 2.0
        mc.append((np.degrees(ang), vs, ps))

    tt = np.array([x[0] for x in pred])
    sig_th = np.array([np.degrees(x[1][0]) for x in pred])
    sig_v = np.array([x[1][3] for x in pred])
    sig_p = np.array([x[1][6] for x in pred])
    m = min(len(tt), min(len(x[0]) for x in mc))
    mc_th = np.std([x[0][:m, 0] for x in mc], axis=0)
    mc_v = np.std([x[1][:m, 0] for x in mc], axis=0)
    mc_p = np.std([x[2][:m, 0] for x in mc], axis=0)

    for sec in (1, 5, 10, 20):
        i = int(np.argmin(np.abs(tt[:m] - sec)))
        log(dict(stage="error_state", seconds=sec,
                 tilt_pred_deg=round(float(sig_th[i]), 4),
                 tilt_mc_deg=round(float(mc_th[i]), 4),
                 pos_pred_m=round(float(sig_p[i]), 4),
                 pos_mc_m=round(float(mc_p[i]), 4)))

    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.1))
    for a, sp, mcv, lbl in ((ax[0], sig_th, mc_th, "tilt about x (deg)"),
                            (ax[1], sig_v, mc_v, "velocity x (m/s)"),
                            (ax[2], sig_p, mc_p, "position x (m)")):
        a.plot(tt[:m], sp[:m], color=COLORS[0], lw=2, label="predicted 1-sigma")
        a.plot(tt[:m], mcv, "--", color=COLORS[1], lw=1.6, label="60-run spread")
        a.set_xlabel("time (s)"); a.set_ylabel(lbl); a.legend(fontsize=8)
    ax[0].set_title("the linear error model is trustworthy here")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "error_state.png"))
    plt.close(fig)


# --------------------------------------------------------------------------

def main():
    use_style()
    t0 = time.time()
    stage_still()
    stage_sources()
    ad = stage_allan()
    stage_bias_cal(ad)
    stage_methods()
    stage_zupt()
    stage_error_state()

    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in RESULTS:
            w.writerow(r)
    print(f"\ndone in {time.time() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    main()
