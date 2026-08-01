"""Project 25 -- tracking a moving target, and the one number you actually tune.

Seven experiments:

  1. track it: position in, position AND velocity out
  2. Q and R are not two knobs.  Measured: only their ratio matters
  3. the steady-state gain, the tracking index, and the error ellipse
  4. the target banks: how far behind does a constant-velocity filter fall?
  5. constant velocity vs constant acceleration -- and who wins where
  6. the target is occluded: coasting, and whether the error bar keeps up
  7. a fast bad sensor or a slow good one?

Runs in about two and a half minutes.  NumPy and Matplotlib only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "24-1d-kf"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from kf import KalmanFilter, chi2_interval, nis, nees                   # noqa: E402
from models import (cv_matrices, ca_matrices, straight_turn_straight,   # noqa: E402
                    constant_velocity_path, random_walk_path)
from plot_style import COLORS, use_style, save                          # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

DT = 0.5              # seconds between detections
N = 120               # 60 seconds of flight
SIGMA_Z = 6.0         # metres, per axis, of measurement noise
Q_TRUE = 0.5          # the process noise that actually drives the target
Q_CV = Q_TRUE         # ...and what we tell the filter, until experiment 2


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def run_filter(z, F, Q, H, R, x0, P0, truth_state=None):
    """One pass of a linear KF over a whole measurement sequence."""
    f = KalmanFilter(x0, P0)
    n = len(z)
    xs = np.empty((n, len(x0)))
    Ps = np.empty((n, len(x0), len(x0)))
    nis_v = np.empty(n)
    nees_v = np.empty(n)
    for k in range(n):
        f.predict(F=F, Q=Q)
        y, S = f.update(z[k], H=H, R=R)
        nis_v[k] = nis(y, S)
        xs[k] = f.x
        Ps[k] = f.P
        if truth_state is not None:
            nees_v[k] = nees(f.x[:4], truth_state[k][:4], f.P[:4, :4])
    return xs, Ps, nis_v, nees_v


# =====================================================================  1
def exp1_track_it(rng):
    banner("1. Position measurements in; position and velocity out")

    pos, vel, _ = random_walk_path(N, DT, Q_TRUE, rng)
    z = pos + SIGMA_Z * rng.standard_normal((N, 2))
    F, Q, H, _ = cv_matrices(DT, Q_CV)
    R = SIGMA_Z ** 2 * np.eye(2)
    x0 = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
    P0 = np.diag([SIGMA_Z ** 2, SIGMA_Z ** 2, 50.0 ** 2, 50.0 ** 2])
    truth = np.hstack([pos, vel])
    xs, Ps, nis_v, nees_v = run_filter(z, F, Q, H, R, x0, P0, truth)

    burn = 20
    pos_rmse = np.sqrt(np.mean(np.sum((xs[burn:, :2] - pos[burn:]) ** 2, axis=1)))
    raw_rmse = np.sqrt(np.mean(np.sum((z[burn:] - pos[burn:]) ** 2, axis=1)))
    vel_rmse = np.sqrt(np.mean(np.sum((xs[burn:, 2:] - vel[burn:]) ** 2, axis=1)))

    # The obvious alternative to a filter: difference two measurements.
    fd = (z[1:] - z[:-1]) / DT
    fd_rmse = np.sqrt(np.mean(np.sum((fd[burn:] - vel[burn:-1]) ** 2, axis=1)))
    # ...and the smarter obvious alternative: difference over a longer baseline.
    lag = 10
    fd10 = (z[lag:] - z[:-lag]) / (lag * DT)
    fd10_rmse = np.sqrt(np.mean(np.sum((fd10[burn:] - vel[burn:-lag]) ** 2, axis=1)))

    print(f"  raw measurements, position RMSE     {raw_rmse:7.3f} m")
    print(f"  Kalman filter,    position RMSE     {pos_rmse:7.3f} m  "
          f"({raw_rmse/pos_rmse:.2f}x better)")
    print(f"\n  velocity, which no sensor reports:")
    print(f"    finite difference over 1 step     {fd_rmse:7.3f} m/s")
    print(f"    finite difference over {lag} steps    {fd10_rmse:7.3f} m/s")
    print(f"    Kalman filter                     {vel_rmse:7.3f} m/s  "
          f"({fd10_rmse/vel_rmse:.2f}x better than the best differencer)")
    print(f"  true speed is {np.linalg.norm(vel[0]):.1f} m/s, so the KF velocity error "
          f"is {100*vel_rmse/np.linalg.norm(vel[0]):.1f}% of speed")
    print(f"\n  mean NEES over 4 states: {nees_v[burn:].mean():.3f}  (target 4.000)")

    record(1, "raw_position_rmse", value=raw_rmse)
    record(1, "kf_position_rmse", value=pos_rmse)
    record(1, "fd1_velocity_rmse", value=fd_rmse)
    record(1, "fd10_velocity_rmse", value=fd10_rmse)
    record(1, "kf_velocity_rmse", value=vel_rmse)
    record(1, "mean_nees", value=float(nees_v[burn:].mean()))

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ax0.plot(z[:, 0], z[:, 1], ".", ms=4, color=COLORS[6], label="measurements")
    ax0.plot(pos[:, 0], pos[:, 1], "k--", lw=1.2, label="truth")
    ax0.plot(xs[:, 0], xs[:, 1], color=COLORS[0], label="KF estimate")
    ax0.set_aspect("equal")
    ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title(f"Position: {raw_rmse:.1f} m -> {pos_rmse:.1f} m")
    ax0.legend(fontsize=8)

    t = np.arange(N) * DT
    ax1.plot(t[:-1], np.linalg.norm(fd, axis=1), ".", ms=3, color=COLORS[6],
             label="1-step difference")
    ax1.plot(t, np.linalg.norm(vel, axis=1), "k--", lw=1.2, label="true speed")
    ax1.plot(t, np.linalg.norm(xs[:, 2:], axis=1), color=COLORS[0], label="KF speed")
    ax1.set_ylim(0, 60)
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("speed (m/s)")
    ax1.set_title("Velocity is never measured, only inferred")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "tracking.png"))


# =====================================================================  2
def exp2_one_knob(rng):
    banner("2. Q and R look like two knobs.  They are one.")

    # The target really is driven by process noise of strength Q_TRUE, so the
    # sweep has a right answer to find instead of running off to q -> 0.
    qs = np.logspace(-2, 2, 17)
    rs = np.array([2.0, 6.0, 18.0])          # three very different sensors
    n_rep = 40
    grid = np.zeros((len(rs), len(qs)))
    paths = [random_walk_path(N, DT, Q_TRUE, rng) for _ in range(n_rep)]
    for ri, sr in enumerate(rs):
        R = sr ** 2 * np.eye(2)
        for qi, q in enumerate(qs):
            F, Q, H, _ = cv_matrices(DT, q)
            errs = []
            for pos, vel, _ in paths:
                z = pos + sr * rng.standard_normal((N, 2))
                x0 = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
                P0 = np.diag([sr ** 2, sr ** 2, 50.0 ** 2, 50.0 ** 2])
                xs, _, _, _ = run_filter(z, F, Q, H, R, x0, P0)
                errs.append(np.sqrt(np.mean(np.sum((xs[20:, :2] - pos[20:]) ** 2, axis=1))))
            grid[ri, qi] = np.mean(errs)

    print(f"  the target is driven by a TRUE process noise of q = {Q_TRUE}")
    print(f"  {'sensor sigma':>13} {'best q':>10} {'best RMSE':>10} {'RMSE/sigma':>11}")
    bests = []
    for ri, sr in enumerate(rs):
        b = int(np.argmin(grid[ri]))
        bests.append(qs[b])
        print(f"  {sr:13.1f} {qs[b]:10.3f} {grid[ri,b]:10.3f} {grid[ri,b]/sr:11.4f}")
        record(2, "best_per_sensor", sigma_z=float(sr), best_q=float(qs[b]),
               rmse=float(grid[ri, b]))
    print(f"\n  every sensor picks a q within a factor of "
          f"{max(max(b/Q_TRUE, Q_TRUE/b) for b in bests):.1f} of the truth, even though")
    print(f"  their noise levels differ by {(rs[-1]/rs[0])**2:.0f}x in variance.")
    print("  The filter is not fitting the sensor; it is identifying the target.")
    # How steep is the penalty?  Report it in units of "times too big/small".
    for ri, sr in enumerate(rs):
        b = int(np.argmin(grid[ri]))
        pen_lo = grid[ri, max(0, b - 4)] / grid[ri, b]
        pen_hi = grid[ri, min(len(qs) - 1, b + 4)] / grid[ri, b]
        print(f"    sigma_z {sr:4.0f} m: q 10x too small costs {100*(pen_lo-1):5.1f}%, "
              f"10x too big costs {100*(pen_hi-1):5.1f}%")
        record(2, "mistune_penalty", sigma_z=float(sr), pen_10x_small=float(pen_lo),
               pen_10x_big=float(pen_hi))

    # Prove the invariance exactly: scale Q, R and P0 together; the gain must not move.
    F, Q, H, _ = cv_matrices(DT, 1.0)
    K1 = K2 = None
    for scale in (1.0, 1000.0):
        P = np.diag([100.0, 100.0, 100.0, 100.0]) * scale
        R = 36.0 * np.eye(2) * scale
        for _ in range(300):
            P = F @ P @ (F.T) + Q * scale
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            P = (np.eye(4) - K @ H) @ P
        if scale == 1.0:
            K1 = K.copy()
        else:
            K2 = K.copy()
    print(f"\n  scaling Q, R and P0 by 1000x together changes the steady gain by "
          f"{np.max(np.abs(K1-K2)):.2e}")
    print("  -> the filter cannot see the absolute size of either; only q/r.")
    record(2, "gain_invariance", value=float(np.max(np.abs(K1 - K2))))

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ri, sr in enumerate(rs):
        ax0.loglog(qs, grid[ri], "o-", ms=3, color=COLORS[ri],
                   label=f"$\\sigma_z$ = {sr:.0f} m")
    ax0.axvline(Q_TRUE, color="k", ls="--", lw=1, label="true $q$")
    ax0.set_xlabel("process noise $q$ given to the filter")
    ax0.set_ylabel("position RMSE (m)")
    ax0.set_title("Three very different sensors, one right answer for $q$")
    ax0.legend(fontsize=8)
    for ri, sr in enumerate(rs):
        ax1.loglog(qs / Q_TRUE, grid[ri] / grid[ri].min(), "o-", ms=3, color=COLORS[ri],
                   label=f"$\\sigma_z$ = {sr:.0f} m")
    ax1.axvline(1.0, color="k", ls="--", lw=1)
    ax1.set_xlabel("$q$ given to the filter, divided by the true $q$")
    ax1.set_ylabel("RMSE / best RMSE for that sensor")
    ax1.set_title("The PRICE of mistuning is the same for all three")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "one_knob.png"))
    return grid, qs, rs


# =====================================================================  3
def exp3_steady_state(rng):
    banner("3. The steady-state gain and the shape of the error")

    F, Q, H, _ = cv_matrices(DT, Q_CV)
    R = SIGMA_Z ** 2 * np.eye(2)
    P = np.diag([1e4, 1e4, 1e4, 1e4])
    trace = []
    for k in range(200):
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P = (np.eye(4) - K @ H) @ P
        trace.append([K[0, 0], K[2, 0], np.sqrt(P[0, 0]), np.sqrt(P[2, 2])])
    trace = np.array(trace)
    alpha, beta_over_dt = trace[-1, 0], trace[-1, 1]
    beta = beta_over_dt * DT
    lam = np.sqrt(Q_CV * DT ** 3) / SIGMA_Z
    # Kalata's closed form for the steady-state alpha-beta gains.  r is the
    # "smoothing factor": the fraction of the old estimate that survives.
    r = (4.0 + lam - np.sqrt(8.0 * lam + lam ** 2)) / 4.0
    alpha_k = 1.0 - r ** 2
    beta_k = 2.0 * (2.0 - alpha_k) - 4.0 * np.sqrt(1.0 - alpha_k)
    print(f"  steady-state alpha (position gain) = {alpha:.4f}")
    print(f"  steady-state beta  (velocity gain) = {beta:.4f}")
    print(f"  tracking index lambda = sqrt(q dt^3) / sigma_z = {lam:.4f}")
    print(f"  Kalata's closed form predicts alpha = {alpha_k:.4f}, beta = {beta_k:.4f}")
    print(f"  agreement: alpha {abs(alpha-alpha_k):.2e}, beta {abs(beta-beta_k):.2e}")
    record(3, "alpha_kalata", value=float(alpha_k))
    record(3, "beta_kalata", value=float(beta_k))
    print(f"  steady position sigma {trace[-1,2]:.3f} m  (sensor is {SIGMA_Z:.1f} m)")
    print(f"  steady velocity sigma {trace[-1,3]:.3f} m/s")
    print(f"  the filter beats a single reading by {SIGMA_Z/trace[-1,2]:.2f}x on position")
    n_settle = int(np.argmax(np.abs(trace[:, 0] - alpha) < 0.01 * alpha))
    print(f"  gain settles to within 1% after {n_settle} updates ({n_settle*DT:.1f} s)")

    record(3, "alpha", value=float(alpha))
    record(3, "beta", value=float(beta))
    record(3, "tracking_index", value=float(lam))
    record(3, "steady_pos_sigma", value=float(trace[-1, 2]))
    record(3, "steady_vel_sigma", value=float(trace[-1, 3]))
    record(3, "settle_steps", value=float(n_settle))

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.2, 3.2))
    ax0.plot(trace[:, 0], color=COLORS[0], label="$\\alpha$ (position gain)")
    ax0.plot(trace[:, 1] * DT, color=COLORS[1], label="$\\beta$ (velocity gain)")
    ax0.set_xlim(0, 60); ax0.set_xlabel("update"); ax0.set_ylabel("gain")
    ax0.set_title("The gain settles in a few seconds and then never moves")
    ax0.legend(fontsize=8)
    ax1.plot(np.arange(200) * DT, trace[:, 2], color=COLORS[0], label="position $\\sigma$ (m)")
    ax1.axhline(SIGMA_Z, ls=":", color=COLORS[6], label="one raw reading")
    ax1.plot(np.arange(200) * DT, trace[:, 3], color=COLORS[2], label="velocity $\\sigma$ (m/s)")
    ax1.set_xlim(0, 30); ax1.set_xlabel("time (s)"); ax1.set_ylabel("$\\sigma$")
    ax1.set_title("Uncertainty falls to a floor, not to zero")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "steady_state.png"))


# =====================================================================  4
def exp4_the_turn(rng):
    banner("4. The target banks and the filter falls behind")

    turn_rates = [0.0, 2.0, 4.0, 6.0, 9.0, 12.0]
    n_rep = 60
    F, Q, H, _ = cv_matrices(DT, Q_CV)
    R = SIGMA_Z ** 2 * np.eye(2)
    rows = []
    keep = None
    turn_slice = slice(int(20.0 / DT), int(35.0 / DT))
    for tr in turn_rates:
        pos, vel, acc = straight_turn_straight(N, DT, turn_rate=np.deg2rad(tr))
        signed = np.zeros((N, 2))            # accumulate the SIGNED error...
        straight_rmse, mean_nis = [], []
        for rep in range(n_rep):
            z = pos + SIGMA_Z * rng.standard_normal((N, 2))
            x0 = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
            P0 = np.diag([SIGMA_Z ** 2, SIGMA_Z ** 2, 50.0 ** 2, 50.0 ** 2])
            xs, Ps, nis_v, _ = run_filter(z, F, Q, H, R, x0, P0)
            err = np.linalg.norm(xs[:, :2] - pos, axis=1)
            signed += xs[:, :2] - pos
            straight_rmse.append(np.sqrt(np.mean(err[20:int(20.0/DT)] ** 2)))
            mean_nis.append(nis_v[turn_slice].mean())
            if tr == 6.0 and rep == 0:
                keep = (pos, z, xs, err, nis_v)
        # ...so that averaging over repeats cancels the random measurement noise
        # and leaves only the systematic lag, which is what the model error is.
        bias = np.linalg.norm(signed / n_rep, axis=1)
        a_lat = 20.0 * np.deg2rad(tr)
        rows.append((tr, a_lat, bias[turn_slice].max(), np.mean(straight_rmse),
                     np.mean(mean_nis)))

    gate = chi2_interval(2, 0.01)[1]
    print(f"  {'turn':>5} {'lateral a':>10} {'peak LAG':>11} {'straight':>9} {'mean NIS':>9}")
    print(f"  {'deg/s':>5} {'(m/s^2)':>10} {'in turn (m)':>11} {'RMSE (m)':>9} {'in turn':>9}")
    for tr, a, pk, st, pn in rows:
        print(f"  {tr:5.1f} {a:10.2f} {pk:11.2f} {st:9.2f} {pn:9.2f}")
    print(f"\n  NIS should average 2.0 for a 2-D measurement; it reaches "
          f"{rows[-1][4]:.1f} at the sharpest turn -- {rows[-1][4]/2.0:.0f}x too surprised.")
    print(f"  a 99% single-reading gate sits at {gate:.2f}, so a hard gate would start")
    print("  THROWING AWAY the very measurements that reveal the manoeuvre.")
    pk = np.array([r[2] for r in rows[1:]]); aa = np.array([r[1] for r in rows[1:]])
    slope = np.polyfit(np.log(aa), np.log(pk), 1)[0]
    print(f"  the systematic lag grows as a^{slope:.2f}  (theory: exactly a^1)")

    for tr, a, pk_, st, pn in rows:
        record(4, "turn_sweep", turn_deg_s=tr, lateral_accel=a, peak_lag=pk_,
               straight_rmse=st, mean_nis_in_turn=pn)
    record(4, "lag_exponent", value=float(slope))

    pos, z, xs, err, nis_v = keep
    use_style()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(11.0, 3.2))
    ax0.plot(z[:, 0], z[:, 1], ".", ms=3, color=COLORS[6])
    ax0.plot(pos[:, 0], pos[:, 1], "k--", lw=1.2, label="truth")
    ax0.plot(xs[:, 0], xs[:, 1], color=COLORS[0], label="KF")
    ax0.set_aspect("equal"); ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("6 deg/s turn: the filter cuts the corner")
    ax0.legend(fontsize=8)
    t = np.arange(N) * DT
    ax1.plot(t, err, color=COLORS[1])
    ax1.axvspan(20, 35, color=COLORS[6], alpha=0.15)
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position error (m)")
    ax1.set_title("Error while banking")
    ax2.semilogy(t, nis_v, color=COLORS[2])
    ax2.axhline(gate, ls="--", color="k", label=f"99% gate = {gate:.1f}")
    ax2.axvspan(20, 35, color=COLORS[6], alpha=0.15)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("NIS")
    ax2.set_title("NIS announces the manoeuvre")
    ax2.legend(fontsize=8)
    save(fig, os.path.join(OUT, "turn.png"))


# =====================================================================  5
def exp5_cv_vs_ca(rng):
    banner("5. Constant velocity against constant acceleration")

    pos, vel, acc = straight_turn_straight(N, DT, turn_rate=np.deg2rad(6.0))
    R = SIGMA_Z ** 2 * np.eye(2)
    straight = np.r_[np.arange(20, int(20.0 / DT)), np.arange(int(35.0 / DT), N)]
    turning = np.arange(int(20.0 / DT), int(35.0 / DT))

    configs = []
    for q in (0.05, 0.5, 5.0):
        F, Q, H, d = cv_matrices(DT, q)
        configs.append((f"CV q={q}", F, Q, H, d))
    for q in (0.005, 0.05, 0.5, 5.0):
        F, Q, H, d = ca_matrices(DT, q)
        configs.append((f"CA q={q}", F, Q, H, d))

    n_rep = 80
    rows = []
    for name, F, Q, H, d in configs:
        s_err, t_err, a_err = [], [], []
        for _ in range(n_rep):
            z = pos + SIGMA_Z * rng.standard_normal((N, 2))
            x0 = np.zeros(d); x0[:2] = z[0]
            P0 = np.eye(d) * 50.0 ** 2
            P0[0, 0] = P0[1, 1] = SIGMA_Z ** 2
            xs, _, _, _ = run_filter(z, F, Q, H, R, x0, P0)
            e = np.linalg.norm(xs[:, :2] - pos, axis=1)
            s_err.append(np.sqrt(np.mean(e[straight] ** 2)))
            t_err.append(np.sqrt(np.mean(e[turning] ** 2)))
            a_err.append(np.sqrt(np.mean(e[20:] ** 2)))
        rows.append((name, np.mean(s_err), np.mean(t_err), np.mean(a_err)))

    # The third option: keep the simple CV model, but let the filter notice it is
    # being surprised and loosen q on the spot.  This is a one-line stand-in for
    # an IMM (Interacting Multiple Model) filter, which runs several models at
    # once and blends them by how well each explains the data.
    F_lo, Q_lo, H2, _ = cv_matrices(DT, 0.5)
    _, Q_hi, _, _ = cv_matrices(DT, 30.0)
    s_err, t_err, a_err, frac_hi = [], [], [], []
    for _ in range(n_rep):
        z = pos + SIGMA_Z * rng.standard_normal((N, 2))
        f = KalmanFilter(np.array([z[0, 0], z[0, 1], 0.0, 0.0]),
                         np.diag([SIGMA_Z ** 2, SIGMA_Z ** 2, 50.0 ** 2, 50.0 ** 2]))
        est, hot = [], 0
        recent = []
        for k in range(N):
            # decide which Q to use from the last few innovations, not this one:
            # reacting to a single NIS spike would fire on noise alone.
            loose = len(recent) >= 3 and np.mean(recent[-3:]) > 4.0
            hot += loose
            f.predict(F=F_lo, Q=Q_hi if loose else Q_lo)
            y, S = f.update(z[k], H=H2, R=R)
            recent.append(nis(y, S))
            est.append(f.x[:2].copy())
        e = np.linalg.norm(np.array(est) - pos, axis=1)
        s_err.append(np.sqrt(np.mean(e[straight] ** 2)))
        t_err.append(np.sqrt(np.mean(e[turning] ** 2)))
        a_err.append(np.sqrt(np.mean(e[20:] ** 2)))
        frac_hi.append(hot / N)
    rows.append(("adaptive", np.mean(s_err), np.mean(t_err), np.mean(a_err)))
    adaptive_hot = np.mean(frac_hi)

    print(f"  {'model':>10} {'straight':>10} {'turning':>10} {'overall':>10}   (RMSE, m)")
    for name, s, t, a in rows:
        print(f"  {name:>10} {s:10.3f} {t:10.3f} {a:10.3f}")
    best_s = min(rows, key=lambda r: r[1])
    best_t = min(rows, key=lambda r: r[2])
    best_a = min(rows, key=lambda r: r[3])
    print(f"\n  best on the straight legs: {best_s[0]}")
    print(f"  best while turning       : {best_t[0]}")
    print(f"  best overall             : {best_a[0]}")
    cvb = min((r for r in rows if r[0].startswith("CV")), key=lambda r: r[3])
    cab = min((r for r in rows if r[0].startswith("CA")), key=lambda r: r[3])
    print(f"\n  best CV {cvb[0]}: straight {cvb[1]:.3f}, turning {cvb[2]:.3f}")
    print(f"  best CA {cab[0]}: straight {cab[1]:.3f}, turning {cab[2]:.3f}")
    print(f"  the extra states cost {100*(cab[1]/cvb[1]-1):+.1f}% on the straight legs "
          f"and buy {100*(cab[2]/cvb[2]-1):+.1f}% in the turn")
    ad = [r for r in rows if r[0] == "adaptive"][0]
    print(f"\n  adaptive CV (q jumps 0.5 -> 30 when the last 3 NIS average above 4):")
    print(f"    straight {ad[1]:.3f}, turning {ad[2]:.3f}, overall {ad[3]:.3f}")
    print(f"    it runs loose {100*adaptive_hot:.0f}% of the time")
    print(f"    vs best fixed CV: {100*(ad[1]/cvb[1]-1):+.1f}% straight, "
          f"{100*(ad[2]/cvb[2]-1):+.1f}% turning, {100*(ad[3]/cvb[3]-1):+.1f}% overall")

    for name, s, t, a in rows:
        record(5, "model_comparison", model=name, straight_rmse=s, turn_rmse=t, overall_rmse=a)

    use_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    idx = np.arange(len(rows))
    w = 0.38
    ax.bar(idx - w / 2, [r[1] for r in rows], w, color=COLORS[0], label="straight legs")
    ax.bar(idx + w / 2, [r[2] for r in rows], w, color=COLORS[1], label="in the turn")
    ax.set_xticks(idx); ax.set_xticklabels([r[0] for r in rows], rotation=20, fontsize=7)
    ax.set_ylabel("position RMSE (m)")
    ax.set_title("No model wins both halves of the flight")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "cv_vs_ca.png"))


# =====================================================================  6
def exp6_occlusion(rng):
    banner("6. The target disappears behind a building")

    gaps = [0, 2, 4, 8, 16, 32]
    F, Q, H, _ = cv_matrices(DT, Q_CV)
    R = SIGMA_Z ** 2 * np.eye(2)
    start = 60
    n_rep = 200
    paths = [random_walk_path(N, DT, Q_TRUE, rng) for _ in range(n_rep)]
    rows = []
    keep = None
    for g in gaps:
        end_err, end_sig, nees_end, recover, gains = [], [], [], [], []
        for rep in range(n_rep):
            pos, vel, _ = paths[rep]
            truth = np.hstack([pos, vel])
            z = pos + SIGMA_Z * rng.standard_normal((N, 2))
            f = KalmanFilter(np.array([z[0, 0], z[0, 1], 0.0, 0.0]),
                             np.diag([SIGMA_Z ** 2, SIGMA_Z ** 2, 50.0 ** 2, 50.0 ** 2]))
            errs, sigs = [], []
            rec = None
            for k in range(N):
                f.predict(F=F, Q=Q)
                if k == start + g:
                    # the gain the filter applies to the FIRST reading back
                    S = H @ f.P @ H.T + R
                    gains.append(float((f.P @ H.T @ np.linalg.inv(S))[0, 0]))
                if not (start <= k < start + g):
                    f.update(z[k], H=H, R=R)
                errs.append(np.linalg.norm(f.x[:2] - pos[k]))
                sigs.append(np.sqrt(f.P[0, 0] + f.P[1, 1]))
                if k == start + g - 1 or (g == 0 and k == start - 1):
                    e_gap, s_gap = errs[-1], sigs[-1]
                    nees_gap = nees(f.x, truth[k], f.P)
                if k >= start + g and rec is None and errs[-1] < 1.5 * SIGMA_Z:
                    rec = k - (start + g)
            end_err.append(e_gap); end_sig.append(s_gap); nees_end.append(nees_gap)
            recover.append(rec if rec is not None else N)
            if g == 16 and rep == 0:
                keep = (np.array(errs), np.array(sigs))
        rows.append((g, np.mean(end_err), np.mean(end_sig), np.mean(nees_end),
                     np.mean(gains) if gains else float("nan")))

    lo, hi = chi2_interval(4, 0.05)
    print(f"  {'gap':>4} {'gap (s)':>8} {'error at':>9} {'reported':>9} {'NEES':>7} "
          f"{'gain on':>9}")
    print(f"  {'steps':>4} {'':>8} {'re-entry':>9} {'sigma':>9} {'(4=ok)':>7} "
          f"{'1st fix':>9}")
    for g, e, sg, ne, kg in rows:
        print(f"  {g:4d} {g*DT:8.1f} {e:9.3f} {sg:9.3f} {ne:7.2f} {kg:9.3f}")
    print(f"\n  the mean NEES of a healthy 4-state filter is 4.0; single-sample 95%")
    print(f"  band is [{lo:.2f}, {hi:.2f}].  The filter stays consistent all the way")
    print("  through the blackout -- it does not know where the target is, and it")
    print("  correctly says so.")
    gg = np.array([r[0] for r in rows[1:]], dtype=float)
    ee = np.array([r[1] for r in rows[1:]])
    ss = np.array([r[2] for r in rows[1:]])
    print(f"  error during coasting grows as gap^{np.polyfit(np.log(gg), np.log(ee), 1)[0]:.2f}")
    print(f"  the reported sigma grows as gap^{np.polyfit(np.log(gg), np.log(ss), 1)[0]:.2f}")
    print(f"  gain on the first reading back rises {rows[0][4]:.3f} -> {rows[-1][4]:.3f}")
    print("  -> after a long blackout the filter almost ignores its own prediction,")
    print("     which is why recovery takes one measurement rather than many.")

    for g, e, sg, ne, kg in rows:
        record(6, "occlusion", gap_steps=g, error=e, reported_sigma=sg, nees=ne,
               reentry_gain=kg)

    errs, sigs = keep
    use_style()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    t = np.arange(N) * DT
    ax.plot(t, errs, color=COLORS[1], label="actual error")
    ax.plot(t, 3 * sigs / np.sqrt(2), color=COLORS[0], label="reported $3\\sigma$ (per axis)")
    ax.axvspan(start * DT, (start + 16) * DT, color=COLORS[6], alpha=0.2, label="occluded")
    ax.set_xlabel("time (s)"); ax.set_ylabel("m")
    ax.set_title("Coasting through a 16-step gap: the error bar grows to match")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "occlusion.png"))


# =====================================================================  7
def exp7_fast_or_accurate(rng):
    banner("7. A fast noisy sensor or a slow clean one?")

    # Same "information per second": halving dt while doubling sigma keeps
    # sigma^2 / dt constant.  Does the tracker care which one you buy?
    base_dt, base_sig = 0.5, 6.0
    configs = []
    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        dt = base_dt * mult
        sig = base_sig * np.sqrt(mult)          # same variance per unit time
        configs.append((dt, sig))
    n_rep = 60
    rows = []
    for dt, sig in configs:
        n = int(60.0 / dt)
        paths = [random_walk_path(n, dt, Q_TRUE, rng) for _ in range(n_rep // 3)]
        # retune q for each configuration so the comparison is fair
        best = None
        for q in np.logspace(-2, 1.5, 12):
            F, Q, H, _ = cv_matrices(dt, q)
            R = sig ** 2 * np.eye(2)
            errs, verrs = [], []
            for pos, vel, _ in paths:
                z = pos + sig * rng.standard_normal((n, 2))
                x0 = np.array([z[0, 0], z[0, 1], 0.0, 0.0])
                P0 = np.diag([sig ** 2, sig ** 2, 50.0 ** 2, 50.0 ** 2])
                xs, _, _, _ = run_filter(z, F, Q, H, R, x0, P0)
                b = int(20.0 / dt)
                errs.append(np.sqrt(np.mean(np.sum((xs[b:, :2] - pos[b:]) ** 2, axis=1))))
                verrs.append(np.sqrt(np.mean(np.sum((xs[b:, 2:] - vel[b:]) ** 2, axis=1))))
            m = np.mean(errs)
            if best is None or m < best[1]:
                best = (q, m, np.mean(verrs))
        rows.append((dt, sig, best[0], best[1], best[2]))

    print(f"  {'dt (s)':>7} {'sigma_z':>8} {'best q':>8} {'pos RMSE':>9} {'vel RMSE':>9}")
    for dt, sig, q, e, v in rows:
        print(f"  {dt:7.3f} {sig:8.2f} {q:8.3f} {e:9.3f} {v:9.3f}")
    fast, slow = rows[0], rows[-1]
    print(f"\n  same information per second throughout (sigma^2/dt = "
          f"{base_sig**2/base_dt:.0f} m^2 s).")
    print(f"  fast+noisy ({fast[0]:.2f} s, {fast[1]:.1f} m) position RMSE {fast[3]:.3f} m")
    print(f"  slow+clean ({slow[0]:.2f} s, {slow[1]:.1f} m) position RMSE {slow[3]:.3f} m")
    print(f"  -> {'fast' if fast[3] < slow[3] else 'slow'} wins on position by "
          f"{100*abs(fast[3]-slow[3])/max(fast[3],slow[3]):.1f}%")
    print(f"  velocity: fast {fast[4]:.3f} m/s vs slow {slow[4]:.3f} m/s "
          f"-> {'fast' if fast[4] < slow[4] else 'slow'} wins by "
          f"{100*abs(fast[4]-slow[4])/max(fast[4],slow[4]):.1f}%")

    for dt, sig, q, e, v in rows:
        record(7, "rate_vs_noise", dt=dt, sigma_z=sig, best_q=q, pos_rmse=e, vel_rmse=v)

    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    dts = [r[0] for r in rows]
    ax.semilogx(dts, [r[3] for r in rows], "o-", color=COLORS[0], label="position RMSE (m)")
    ax2 = ax.twinx()
    ax2.semilogx(dts, [r[4] for r in rows], "s-", color=COLORS[1], label="velocity RMSE (m/s)")
    ax2.grid(False)
    ax.set_xlabel("sampling interval $\\Delta t$ (s), with $\\sigma_z \\propto \\sqrt{\\Delta t}$")
    ax.set_ylabel("position RMSE (m)", color=COLORS[0])
    ax2.set_ylabel("velocity RMSE (m/s)", color=COLORS[1])
    ax.set_title("Equal information per second is not equal performance")
    save(fig, os.path.join(OUT, "rate_vs_noise.png"))


def main():
    t0 = time.time()
    rng = np.random.default_rng(7)
    exp1_track_it(rng)
    exp2_one_knob(rng)
    exp3_steady_state(rng)
    exp4_the_turn(rng)
    exp5_cv_vs_ca(rng)
    exp6_occlusion(rng)
    exp7_fast_or_accurate(rng)

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
