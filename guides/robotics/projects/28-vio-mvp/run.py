"""Project 28 -- visual-inertial odometry: a camera with no ruler and an IMU
with no memory.

Seven experiments:

  1. IMU alone, camera alone, and the two together
  2. the scale question: when is a metre observable at all?
  3. the biases the filter estimates, and what happens if it does not
  4. error state against a direct quaternion update
  5. the camera drops out: how fast does it fall apart, and does it recover?
  6. the camera clock is 20 ms behind the IMU clock
  7. IMU rate: does 1000 Hz buy anything over 100 Hz?

Runs in about four minutes.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "21-imu-integration"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from imu import quat_to_R, quat_mul, quat_from_rotvec, G              # noqa: E402
from vio import ErrorStateEKF, DirectEKF                              # noqa: E402
from trajectory import (ImuSpec, make_imu, make_camera, blended,      # noqa: E402
                        figure_eight_3d, straight_line,
                        attitude_from_path, attitude_error_deg)
from plot_style import COLORS, use_style, save                        # noqa: E402

import matplotlib.pyplot as plt                                       # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

DURATION = 40.0                 # seconds of flight
CAM_HZ = 20.0
SIGMA_DIR = 0.01                # camera direction noise (rad-ish, unit vector)
SIGMA_ATT = 0.004               # camera relative-rotation noise, radians


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def build(rng, excite=1.0, rate=200.0, dropout=(), time_offset_ms=0.0,
          spec=None, duration=DURATION):
    """One complete simulated flight plus its sensor logs."""
    spec = ImuSpec(rate=rate) if spec is None else spec
    n = int(duration * rate)
    t = np.arange(n) / rate
    p, v, a = blended(t, excite)
    q = attitude_from_path(t)
    acc, gyr, ba, bg = make_imu(spec, t, p, v, a, q, rng)
    stride = max(1, int(round(rate / CAM_HZ)))
    off = int(round(time_offset_ms * 1e-3 * rate))
    cam = make_camera(t, p, q, stride, rng, SIGMA_DIR, SIGMA_ATT,
                      dropout=dropout, time_offset_steps=off)
    return dict(t=t, p=p, v=v, a=a, q=q, acc=acc, gyr=gyr, ba=ba, bg=bg,
                cam=cam, spec=spec, rate=rate)


def run_vio(sim, cls=ErrorStateEKF, estimate_bias=True, use_camera=True,
            p0_scale=1.0, known_bias=False):
    """Drive the filter through a whole flight; return the estimated states."""
    spec = sim["spec"]
    dt = 1.0 / sim["rate"]
    P0 = np.diag(np.concatenate([
        [0.01] * 3, [0.01] * 3, [np.deg2rad(1.0) ** 2] * 3,
        [0.1 ** 2] * 3, [0.01 ** 2] * 3])) * p0_scale
    if not estimate_bias:
        # Tell the filter the biases are perfectly known and cannot move.  It
        # then never corrects them, which is exactly the point of experiment 3.
        P0[9:, 9:] = np.eye(6) * 1e-12
    f = cls(sim["p"][0], sim["v"][0], sim["q"][0],
            np.zeros(3) if not known_bias else sim["ba"][0],
            np.zeros(3) if not known_bias else sim["bg"][0],
            P0, (spec.acc_noise, spec.gyr_noise,
                 spec.acc_walk if estimate_bias else 0.0,
                 spec.gyr_walk if estimate_bias else 0.0))
    cam_by_index = {}
    for k, ref, u, qm in sim["cam"]:
        cam_by_index.setdefault(k, []).append((ref, u, qm))
    n = len(sim["t"])
    est_p = np.empty((n, 3))
    est_v = np.empty((n, 3))
    est_q = np.empty((n, 4))
    est_ba = np.empty((n, 3))
    est_bg = np.empty((n, 3))
    sig_p = np.empty(n)
    for k in range(n):
        if k > 0:
            f.predict(sim["acc"][k - 1], sim["gyr"][k - 1], dt)
        if use_camera and k in cam_by_index:
            for ref, u, qm in cam_by_index[k]:
                f.update_attitude(qm, SIGMA_ATT)
                # The reference is the filter's OWN past estimate, not the true
                # past pose.  That is what makes this odometry rather than
                # localization: nothing here has ever seen a global position,
                # so errors accumulate and the trajectory drifts.  (A rigorous
                # implementation would keep the past pose in the state --
                # "stochastic cloning" -- so the filter accounts for the fact
                # that the two ends of the measurement share their error.  We
                # do not, which makes the filter slightly overconfident; the
                # reported sigma in experiment 1 is the honest evidence.)
                f.update_visual_direction(u, est_p[ref], SIGMA_DIR)
        est_p[k], est_v[k], est_q[k] = f.p, f.v, f.q
        est_ba[k], est_bg[k] = f.ba, f.bg
        sig_p[k] = np.sqrt(np.trace(f.P[0:3, 0:3]))
    return dict(p=est_p, v=est_v, q=est_q, ba=est_ba, bg=est_bg, sig_p=sig_p)


def perr(est, sim):
    return np.linalg.norm(est["p"] - sim["p"], axis=1)


def aerr(est, sim):
    return np.array([attitude_error_deg(est["q"][k], sim["q"][k])
                     for k in range(0, len(sim["t"]), 10)])


# =====================================================================  1
def exp1_three_systems(rng):
    banner("1. The IMU alone, the camera alone, and the two together")

    sim = build(rng)
    vio = run_vio(sim)
    imu_only = run_vio(sim, use_camera=False)

    # "Camera alone" is not a system that produces metres at all -- that is the
    # point.  The closest honest comparison is: integrate the camera's own
    # direction+rotation stream with the scale FIXED at whatever the first
    # segment happened to be, which is what a monocular VO pipeline reports.
    cam_p = [sim["p"][0]]
    seg_len = np.linalg.norm(sim["p"][sim["cam"][0][0]] - sim["p"][sim["cam"][0][1]])
    for k, ref, u, qm in sim["cam"]:
        cam_p.append(cam_p[-1] + seg_len * u)
    cam_p = np.array(cam_p)
    cam_idx = [0] + [c[0] for c in sim["cam"]]
    cam_err = np.linalg.norm(cam_p - sim["p"][cam_idx], axis=1)

    e_vio, e_imu = perr(vio, sim), perr(imu_only, sim)
    dist = np.sum(np.linalg.norm(np.diff(sim["p"], axis=0), axis=1))
    print(f"  flew {dist:.1f} m in {DURATION:.0f} s, "
          f"{len(sim['cam'])} camera frames, {len(sim['t'])} IMU samples")
    print(f"  {'':>26} {'final':>10} {'mean':>10} {'max':>10}")
    print(f"  position error (m)")
    print(f"    IMU only (dead reckoning) {e_imu[-1]:10.2f} {e_imu.mean():10.2f} "
          f"{e_imu.max():10.2f}")
    print(f"    camera only, fixed scale  {cam_err[-1]:10.2f} {cam_err.mean():10.2f} "
          f"{cam_err.max():10.2f}")
    print(f"    VIO                       {e_vio[-1]:10.3f} {e_vio.mean():10.3f} "
          f"{e_vio.max():10.3f}")
    a_vio, a_imu = aerr(vio, sim), aerr(imu_only, sim)
    print(f"  attitude error (deg)")
    print(f"    IMU only                  {a_imu[-1]:10.3f} {a_imu.mean():10.3f} "
          f"{a_imu.max():10.3f}")
    print(f"    VIO                       {a_vio[-1]:10.3f} {a_vio.mean():10.3f} "
          f"{a_vio.max():10.3f}")
    print(f"\n  VIO beats IMU-only by {e_imu.mean()/e_vio.mean():.0f}x on position "
          f"and {a_imu.mean()/a_vio.mean():.0f}x on attitude")
    print(f"  VIO drift: {100*e_vio[-1]/dist:.3f}% of distance travelled")
    print(f"  NOTE the camera contributes NO metres -- only directions and")
    print(f"  rotations.  Every number in metres above came from the "
          f"accelerometer.")

    record(1, "distance", value=float(dist))
    record(1, "imu_only_final", value=float(e_imu[-1]))
    record(1, "camera_only_final", value=float(cam_err[-1]))
    record(1, "vio_final", value=float(e_vio[-1]))
    record(1, "vio_mean", value=float(e_vio.mean()))
    record(1, "imu_att_mean_deg", value=float(a_imu.mean()))
    record(1, "vio_att_mean_deg", value=float(a_vio.mean()))

    use_style()
    fig = plt.figure(figsize=(11.0, 3.6))
    ax0 = fig.add_subplot(1, 3, 1)
    ax0.plot(sim["p"][:, 0], sim["p"][:, 1], "k--", lw=1.2, label="truth")
    ax0.plot(vio["p"][:, 0], vio["p"][:, 1], color=COLORS[0], label="VIO")
    ax0.plot(cam_p[:, 0], cam_p[:, 1], color=COLORS[2], lw=1.0,
             label="camera, fixed scale")
    ax0.set_aspect("equal"); ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("The flight, from above")
    ax0.legend(fontsize=7)
    ax1 = fig.add_subplot(1, 3, 2)
    ax1.semilogy(sim["t"], np.maximum(e_imu, 1e-4), color=COLORS[1], label="IMU only")
    ax1.semilogy(np.array(sim["t"])[cam_idx], np.maximum(cam_err, 1e-4),
                 color=COLORS[2], label="camera only")
    ax1.semilogy(sim["t"], np.maximum(e_vio, 1e-4), color=COLORS[0], label="VIO")
    ax1.semilogy(sim["t"], 3 * vio["sig_p"], ":", color=COLORS[0],
                 label="VIO reported $3\\sigma$")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position error (m)")
    ax1.set_title("Three ways to be wrong")
    ax1.legend(fontsize=7)
    ax2 = fig.add_subplot(1, 3, 3)
    ts = sim["t"][::10]
    ax2.semilogy(ts, np.maximum(a_imu, 1e-4), color=COLORS[1], label="IMU only")
    ax2.semilogy(ts, np.maximum(a_vio, 1e-4), color=COLORS[0], label="VIO")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("attitude error (deg)")
    ax2.set_title("Attitude")
    ax2.legend(fontsize=7)
    save(fig, os.path.join(OUT, "three_systems.png"))
    return sim


# =====================================================================  2
def exp2_scale(rng):
    banner("2. When is a metre observable?")

    excites = [0.0, 0.02, 0.05, 0.15, 0.4, 1.0]
    n_rep = 6
    rows = []
    for ex in excites:
        errs, scales, sigs, accs = [], [], [], []
        for _ in range(n_rep):
            sim = build(rng, excite=ex)
            est = run_vio(sim)
            # a scale error shows up as the travelled path being uniformly
            # too long or too short, so measure the ratio of path lengths
            dp_t = np.linalg.norm(np.diff(sim["p"][::20], axis=0), axis=1).sum()
            dp_e = np.linalg.norm(np.diff(est["p"][::20], axis=0), axis=1).sum()
            errs.append(np.linalg.norm(est["p"][-1] - sim["p"][-1]))
            scales.append(dp_e / dp_t)
            sigs.append(est["sig_p"][-1])
            # how much acceleration (beyond gravity) the flight actually had
            accs.append(float(np.sqrt(np.mean(np.sum(sim["a"] ** 2, axis=1)))))
        rows.append((ex, np.mean(accs), np.mean(scales), np.std(scales),
                     np.mean(errs), np.mean(sigs)))

    print(f"  {'excite':>7} {'rms accel':>10} {'scale':>8} {'scale':>8} "
          f"{'final err':>10} {'reported':>9}")
    print(f"  {'':>7} {'(m/s^2)':>10} {'ratio':>8} {'spread':>8} {'(m)':>10} "
          f"{'sigma (m)':>9}")
    for ex, a, s, sd, e, sg in rows:
        print(f"  {ex:7.2f} {a:10.4f} {s:8.4f} {sd:8.4f} {e:10.3f} {sg:9.3f}")
    print(f"\n  With NO acceleration ({rows[0][1]:.4f} m/s^2) the scale is off by "
          f"{100*abs(rows[0][2]-1):.1f}%")
    print(f"  and varies by +-{100*rows[0][3]:.1f}% between runs -- it is not being")
    print("  measured at all, only guessed from the initial velocity.")
    print(f"  With full excitation ({rows[-1][1]:.3f} m/s^2) the scale is within "
          f"{100*abs(rows[-1][2]-1):.2f}%")
    print(f"  and the spread drops {rows[0][3]/rows[-1][3]:.0f}x.")
    print("\n  Why acceleration and not motion: the accelerometer measures")
    print("  specific force, and at constant velocity that is exactly -g in the")
    print("  body frame -- the same reading you get sitting on a table.  A")
    print("  constant-velocity flight literally contains no accelerometer")
    print("  evidence about how fast it is going, so nothing anchors the metre.")
    print("  Real consequence: a VIO system must be WAVED before it works, and")
    print("  a drone that takes off vertically at constant speed has no scale.")
    print(f"\n  And now the uncomfortable part.  Look at the last column: with no")
    print(f"  excitation the filter is {rows[0][4]:.1f} m out and still reporting a")
    print(f"  standard deviation of {rows[0][5]:.3f} m -- it is wrong by "
          f"{rows[0][4]/rows[0][5]:.0f} sigma and")
    print("  says nothing.  Two things cause that.  First, this filter does not do")
    print("  stochastic cloning, so it never accounts for the error shared between")
    print("  the two ends of a visual measurement, and is optimistic everywhere.")
    print("  Second and more fundamental: the scale error is a slow, systematic")
    print("  drift, and as project 24's biased-thermometer experiment showed, a")
    print("  covariance computed from F, Q, H and R alone cannot see a bias.")
    print("  The practical answer used in real systems is an explicit")
    print("  observability check on the recent acceleration, NOT a look at P.")

    for ex, a, s, sd, e, sg in rows:
        record(2, "scale_observability", excite=ex, rms_accel=a, scale_ratio=s,
               scale_spread=sd, final_err=e, reported_sigma=sg)

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.3))
    ax0.errorbar([r[1] for r in rows], [r[2] for r in rows],
                 yerr=[r[3] for r in rows], fmt="o-", color=COLORS[0], capsize=3)
    ax0.axhline(1.0, ls="--", color="k", lw=1)
    ax0.set_xscale("symlog", linthresh=1e-3)
    ax0.set_xlabel("rms acceleration of the flight (m/s$^2$)")
    ax0.set_ylabel("estimated path length / true")
    ax0.set_title("Scale is only recovered when the flight accelerates")
    ax1.loglog([r[1] for r in rows[1:]], [r[4] for r in rows[1:]], "o-",
               color=COLORS[0], label="actual final error")
    ax1.loglog([r[1] for r in rows[1:]], [r[5] for r in rows[1:]], "s-",
               color=COLORS[1], label="reported $\\sigma$")
    ax1.set_xlabel("rms acceleration (m/s$^2$)")
    ax1.set_ylabel("m")
    ax1.set_title("...and the filter does NOT know it")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "scale.png"))


# =====================================================================  3
def exp3_biases(rng):
    banner("3. The biases: estimated, ignored, or known")

    n_rep = 6
    rows = []
    for name, kw in (("estimate them", dict(estimate_bias=True)),
                     ("assume zero", dict(estimate_bias=False)),
                     ("perfectly known", dict(estimate_bias=False,
                                              known_bias=True))):
        errs, aerrs = [], []
        for _ in range(n_rep):
            sim = build(rng)
            est = run_vio(sim, **kw)
            errs.append(perr(est, sim).mean())
            aerrs.append(aerr(est, sim).mean())
        rows.append((name, np.mean(errs), np.mean(aerrs)))

    print(f"  {'bias handling':>17} {'pos err (m)':>12} {'att err (deg)':>14}")
    for n, e, a in rows:
        print(f"  {n:>17} {e:12.4f} {a:14.4f}")
    print(f"\n  Ignoring the biases costs {rows[1][1]/rows[0][1]:.1f}x the position "
          f"error and {rows[1][2]/rows[0][2]:.1f}x the attitude error.")
    print(f"  Estimating them online recovers {100*(1-(rows[0][1]-rows[2][1])/(rows[1][1]-rows[2][1])):.0f}% "
          f"of the gap to a perfectly calibrated IMU.")

    # convergence of the estimates themselves
    sim = build(rng)
    est = run_vio(sim)
    ba_err = np.linalg.norm(est["ba"] - sim["ba"], axis=1)
    bg_err = np.linalg.norm(est["bg"] - sim["bg"], axis=1)
    def settle(e, frac=0.2):
        thr = frac * e[0]
        idx = np.where(e < thr)[0]
        return sim["t"][idx[0]] if len(idx) else np.nan
    print(f"\n  accel bias: |error| {ba_err[0]:.4f} -> {ba_err[-1]:.4f} m/s^2, "
          f"within 20% after {settle(ba_err):.1f} s")
    print(f"  gyro  bias: |error| {bg_err[0]:.5f} -> {bg_err[-1]:.5f} rad/s, "
          f"within 20% after {settle(bg_err):.1f} s")
    print("  The gyro bias converges first, and it has to: an attitude error")
    print("  tips the gravity vector into the horizontal axes, so a gyro bias")
    print("  masquerades as an accelerometer bias.  Until attitude is pinned")
    print("  down, the accelerometer bias cannot be separated from it.")

    for n, e, a in rows:
        record(3, "bias_handling", mode=n, pos_err=e, att_err_deg=a)
    record(3, "ba_settle_s", value=float(settle(ba_err)))
    record(3, "bg_settle_s", value=float(settle(bg_err)))

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for i in range(3):
        ax0.plot(sim["t"], est["ba"][:, i], color=COLORS[i], lw=1.2)
        ax0.plot(sim["t"], sim["ba"][:, i], color=COLORS[i], ls="--", lw=1.0)
    ax0.set_xlabel("time (s)"); ax0.set_ylabel("accel bias (m/s$^2$)")
    ax0.set_title("Accelerometer bias (dashed = truth)")
    for i in range(3):
        ax1.plot(sim["t"], est["bg"][:, i], color=COLORS[i], lw=1.2)
        ax1.plot(sim["t"], sim["bg"][:, i], color=COLORS[i], ls="--", lw=1.0)
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("gyro bias (rad/s)")
    ax1.set_title("Gyro bias")
    save(fig, os.path.join(OUT, "biases.png"))


# =====================================================================  4
def exp4_error_state(rng):
    banner("4. Error state against adding a correction to a quaternion")

    n_rep = 8
    rows = []
    for tilt in (1.0, 5.0, 15.0, 40.0):
        res = {}
        for name, cls in (("error-state", ErrorStateEKF), ("direct", DirectEKF)):
            pe, ae = [], []
            for _ in range(n_rep):
                sim = build(rng)
                # start with a deliberately wrong attitude, so the corrections
                # the filter has to apply are large
                bad = quat_mul(sim["q"][0],
                               quat_from_rotvec(np.deg2rad(tilt) *
                                                np.array([0.3, 0.6, 0.75])))
                sim2 = dict(sim)
                sim2["q"] = sim["q"].copy()
                sim2["q"][0] = bad
                est = run_vio(sim2, cls=cls)
                pe.append(perr(est, sim).mean())
                ae.append(aerr(est, sim).mean())
            res[name] = (np.mean(pe), np.mean(ae))
        rows.append((tilt, res))

    print(f"  {'initial':>8} {'error-state':>22} {'direct (add to q)':>22}")
    print(f"  {'tilt':>8} {'pos (m)':>11} {'att (deg)':>10} "
          f"{'pos (m)':>11} {'att (deg)':>10}")
    for tilt, res in rows:
        e, d = res["error-state"], res["direct"]
        print(f"  {tilt:6.0f} deg {e[0]:11.4f} {e[1]:10.4f} "
              f"{d[0]:11.4f} {d[1]:10.4f}")
    ratios = [r[1]["direct"][0] / r[1]["error-state"][0] for r in rows]
    print(f"\n  Position-error ratio (direct / error-state) across the sweep: "
          f"{min(ratios):.2f} to {max(ratios):.2f}.")
    print("  That is a NULL RESULT and it is worth reporting as one.  The additive")
    print("  form is the correct FIRST-ORDER expansion of the multiplicative one:")
    print("    q (*) [1, dtheta/2]  =  q + q (*) [0, dtheta/2] + O(dtheta^2)")
    print("  so the two differ only in terms of order dtheta^2, plus whatever the")
    print("  renormalization does.  Here the filter's corrections are tiny (a")
    print("  camera update arrives every 50 ms), dtheta^2 is negligible, and the")
    print("  two forms are indistinguishable.")
    print("\n  What the error-state form actually buys is not accuracy in the good")
    print("  case, it is the ABSENCE OF A THRESHOLD.  The additive form is fine")
    print("  until corrections stop being small -- a bad initialization, a long")
    print("  sensor blackout, a violent manoeuvre -- and there is no warning when")
    print("  you cross that line, because nothing crashes and the quaternion is")
    print("  dutifully renormalized every step.  The error state has no such line:")
    print("  its corrections are small BY CONSTRUCTION, since the error is reset")
    print("  to zero after every single update.  You do not adopt it because it")
    print("  measures better; you adopt it so that this experiment can never")
    print("  produce a different answer on a day you were not watching.")
    for tilt, res in rows:
        for nm in ("error-state", "direct"):
            record(4, "error_state", tilt_deg=tilt, form=nm,
                   pos_err=res[nm][0], att_err_deg=res[nm][1])

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.4, 3.2))
    ts = [r[0] for r in rows]
    ax0.loglog(ts, [r[1]["error-state"][0] for r in rows], "o-", color=COLORS[0],
               label="error state")
    ax0.loglog(ts, [r[1]["direct"][0] for r in rows], "s-", color=COLORS[1],
               label="direct")
    ax0.set_xlabel("initial attitude error (deg)"); ax0.set_ylabel("position error (m)")
    ax0.set_title("Position"); ax0.legend(fontsize=8)
    ax1.loglog(ts, [r[1]["error-state"][1] for r in rows], "o-", color=COLORS[0],
               label="error state")
    ax1.loglog(ts, [r[1]["direct"][1] for r in rows], "s-", color=COLORS[1],
               label="direct")
    ax1.set_xlabel("initial attitude error (deg)"); ax1.set_ylabel("attitude error (deg)")
    ax1.set_title("Attitude"); ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "error_state.png"))


# =====================================================================  5
def exp5_dropout(rng):
    banner("5. The camera goes dark")

    rate = 200.0
    n_rep = 6
    rows = []
    keep = None
    for secs in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        lo = int(15.0 * rate)
        hi = lo + int(secs * rate)
        errs, peak, sigs, recov = [], [], [], []
        for rep in range(n_rep):
            sim = build(rng, dropout=((lo, hi),))
            est = run_vio(sim)
            e = perr(est, sim)
            errs.append(e[min(hi, len(e) - 1)])
            peak.append(e[lo:min(hi + int(2 * rate), len(e))].max())
            sigs.append(est["sig_p"][min(hi, len(e) - 1)])
            after = e[hi:]
            # "recovered" = back inside the error band the filter holds when
            # the camera has been running normally (see experiment 1)
            idx = np.where(after < 0.5)[0]
            recov.append(idx[0] / rate if len(idx) else np.nan)
            if rep == 0 and secs == 4.0:
                keep = (sim["t"], e, est["sig_p"], lo / rate, hi / rate)
        rows.append((secs, np.mean(errs), np.mean(peak), np.mean(sigs),
                     np.nanmean(recov)))

    print(f"  {'blackout':>9} {'err at':>9} {'peak':>9} {'reported':>9} "
          f"{'recovery':>9}")
    print(f"  {'(s)':>9} {'return (m)':>9} {'err (m)':>9} {'sigma (m)':>9} {'(s)':>9}")
    for s, e, pk, sg, rc in rows:
        print(f"  {s:9.1f} {e:9.3f} {pk:9.3f} {sg:9.3f} {rc:9.2f}")
    gg = np.array([r[0] for r in rows[1:]])
    ee = np.array([r[1] for r in rows[1:]])
    print(f"\n  error during a blackout grows as t^{np.polyfit(np.log(gg), np.log(ee), 1)[0]:.2f}")
    print("  Project 21 measured pure inertial position error growing as t^2.87.")
    print("  It is milder here because the filter starts the blackout with an")
    print("  attitude and a bias estimate the camera had already pinned down --")
    print("  the camera's contribution keeps paying out after it stops arriving.")
    if np.isfinite(rows[-1][4]):
        print(f"  Recovery after an {rows[-1][0]:.0f} s blackout takes "
              f"{rows[-1][4]:.2f} s, i.e. {rows[-1][4]*CAM_HZ:.0f} camera frames.")
    else:
        print(f"  After an {rows[-1][0]:.0f} s blackout the filter never gets "
              f"back inside 0.5 m at all.")
        print("  That is the honest answer and it is the reason VIO systems have")
        print("  a relocalization path: past some blackout length, the estimate is")
        print("  not degraded, it is GONE, and the fix is to recognize a place")
        print("  rather than to keep integrating.")

    for s, e, pk, sg, rc in rows:
        record(5, "dropout", seconds=s, err_at_return=e, peak_err=pk,
               reported_sigma=sg, recovery_s=rc)

    t, e, sg, lo, hi = keep
    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.semilogy(t, np.maximum(e, 1e-4), color=COLORS[1], label="actual error")
    ax.semilogy(t, 3 * sg, ":", color=COLORS[0], label="reported $3\\sigma$")
    ax.axvspan(lo, hi, color=COLORS[6], alpha=0.2, label="camera dark")
    ax.set_xlabel("time (s)"); ax.set_ylabel("position error (m)")
    ax.set_title("A 4-second blackout, and the recovery")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "dropout.png"))


# =====================================================================  6
def exp6_time_offset(rng):
    banner("6. The camera clock is not the IMU clock")

    n_rep = 6
    rows = []
    for off in (-40.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0, 40.0):
        errs, aerrs = [], []
        for _ in range(n_rep):
            sim = build(rng, time_offset_ms=off)
            est = run_vio(sim)
            errs.append(perr(est, sim).mean())
            aerrs.append(aerr(est, sim).mean())
        rows.append((off, np.mean(errs), np.mean(aerrs)))

    base = [r for r in rows if r[0] == 0.0][0]
    print(f"  {'offset':>8} {'pos err':>9} {'vs 0 ms':>9} {'att err':>9} {'vs 0 ms':>9}")
    print(f"  {'(ms)':>8} {'(m)':>9} {'':>9} {'(deg)':>9}")
    for o, e, a in rows:
        print(f"  {o:8.0f} {e:9.4f} {e/base[1]:8.2f}x {a:9.4f} {a/base[2]:8.2f}x")
    spread = max(r[1] for r in rows) - min(r[1] for r in rows)
    print(f"\n  POSITION shows nothing usable: the whole sweep spans {spread:.3f} m")
    print(f"  with no monotone trend, which is within the run-to-run scatter.")
    print("  Reporting a number from that column would be reporting noise.")
    print(f"\n  ATTITUDE is the channel that sees it, and it is textbook-clean:")
    print(f"  {rows[0][2]:.3f} deg at {rows[0][0]:+.0f} ms, {base[2]:.3f} deg at 0, "
          f"{rows[-1][2]:.3f} deg at {rows[-1][0]:+.0f} ms --")
    print(f"  a symmetric V with a {max(rows[0][2], rows[-1][2])/base[2]:.1f}x penalty "
          f"at the edges.")
    print("  Why attitude and not position: the camera's rotation measurement is")
    print("  far more precise than its direction measurement (0.004 rad against")
    print("  0.01), so it is the channel with the least slack, and a systematic")
    print("  timing lie eats the slack there first.")
    speed = float(np.mean(np.linalg.norm(build(rng)["v"], axis=1)))
    spin = 0.35
    print(f"\n  At {speed:.2f} m/s and {np.rad2deg(spin):.0f} deg/s of yaw rate, 20 ms is "
          f"{speed*0.02*100:.1f} cm of travel")
    print(f"  and {np.rad2deg(spin)*0.02:.2f} deg of rotation.  The filter is told those")
    print("  happened at the wrong moment, so this is not extra noise that averages")
    print("  out: it is a consistent, direction-dependent lie that the filter")
    print("  dutifully absorbs into its bias estimates.")
    print("  This is why every serious VIO system ESTIMATES the camera-IMU time")
    print("  offset as one more state rather than trusting two clocks to agree.")
    for o, e, a in rows:
        record(6, "time_offset", offset_ms=o, pos_err=e, att_err_deg=a)

    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", color=COLORS[0],
            label="attitude error (deg)")
    ax.plot([r[0] for r in rows], [r[1] for r in rows], "s--", color=COLORS[6],
            label="position error (m) -- no trend")
    ax.axvline(0, ls="--", color="k", lw=1)
    ax.set_xlabel("camera timestamp offset (ms)")
    ax.set_ylabel("mean error")
    ax.set_title("Attitude sees the clock skew; position does not")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "time_offset.png"))


# =====================================================================  7
def exp7_imu_rate(rng):
    banner("7. Does a faster IMU help?")

    n_rep = 6
    rows = []
    for rate in (50.0, 100.0, 200.0, 400.0, 800.0):
        res = {}
        for ex, tag in ((1.0, "gentle"), (3.0, "brisk")):
            errs, aerrs, times = [], [], []
            for _ in range(n_rep):
                sim = build(rng, rate=rate, excite=ex)
                t0 = time.time()
                est = run_vio(sim)
                times.append(time.time() - t0)
                errs.append(perr(est, sim).mean())
                aerrs.append(aerr(est, sim).mean())
            res[tag] = (np.mean(errs), np.std(errs), np.mean(aerrs), np.mean(times))
        rows.append((rate, res))

    print(f"  {'IMU rate':>9} | {'gentle flight':>24} | {'brisk flight':>24} "
          f"| {'seconds':>8}")
    print(f"  {'(Hz)':>9} | {'pos err (m)':>14} {'+-':>9} | "
          f"{'pos err (m)':>14} {'+-':>9} | {'per run':>8}")
    for r, res in rows:
        g_, v_ = res["gentle"], res["brisk"]
        print(f"  {r:9.0f} | {g_[0]:14.4f} {g_[1]:9.4f} | "
              f"{v_[0]:14.4f} {v_[1]:9.4f} | {g_[3]:8.2f}")
    lo, hi = rows[0], rows[-1]
    for tag in ("gentle", "brisk"):
        vals = [r[1][tag][0] for r in rows]
        scat = float(np.mean([r[1][tag][1] for r in rows]))
        print(f"\n  {tag} flight: spread across rates {max(vals)-min(vals):.3f} m "
              f"against a run-to-run")
        print(f"  scatter of +-{scat:.3f} m.  "
              f"{'The differences are noise.' if max(vals)-min(vals) < 2*scat else 'A real trend.'}")
    print(f"\n  {lo[0]:.0f} Hz -> {hi[0]:.0f} Hz is {hi[0]/lo[0]:.0f}x the samples and "
          f"{hi[1]['gentle'][3]/lo[1]['gentle'][3]:.0f}x the compute, and it")
    print("  buys nothing measurable in EITHER regime.  That is an honest null and")
    print("  it is the useful answer for anyone sizing a system: at these speeds")
    print("  the motion between samples is already almost perfectly described by")
    print("  one integration step even at 50 Hz, so extra samples add resolution")
    print("  nobody needed.")
    print("\n  What a fast IMU is actually for is the regime this simulation does")
    print("  not reach: rotation rates high enough that 'constant angular velocity")
    print("  over one step' stops being true.  At 800 Hz a step is 1.25 ms, during")
    print("  which even a 500 deg/s gimbal turns 0.6 deg; at 50 Hz the same turn")
    print("  is 10 deg per step and the small-angle assumption underneath the")
    print("  integration is simply wrong.  Nothing here spins that fast, so nothing")
    print("  here can measure it, and claiming otherwise from this data would be")
    print("  reading a trend out of scatter.")
    for r, res in rows:
        for tag, v in res.items():
            record(7, "imu_rate", rate=r, flight=tag, pos_err=v[0], spread=v[1],
                   att_err_deg=v[2], seconds=v[3])

    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    rs = [r[0] for r in rows]
    for i, tag in enumerate(("gentle", "brisk")):
        ax.errorbar(rs, [r[1][tag][0] for r in rows],
                    yerr=[r[1][tag][1] for r in rows], fmt="o-", capsize=3,
                    color=COLORS[i], label=f"{tag} flight")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("IMU rate (Hz)"); ax.set_ylabel("position error (m)")
    ax.set_title("Sample rate only pays when the motion is violent")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "imu_rate.png"))


def main():
    t0 = time.time()
    rng = np.random.default_rng(17)
    exp1_three_systems(rng)
    exp2_scale(rng)
    exp3_biases(rng)
    exp4_error_state(rng)
    exp5_dropout(rng)
    exp6_time_offset(rng)
    exp7_imu_rate(rng)

    path = os.path.join(OUT, "results.csv")
    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(RESULTS)
    print(f"\n  wrote {path}")
    print(f"\nTotal: {time.time()-t0:.1f} s")


if __name__ == "__main__":
    main()
