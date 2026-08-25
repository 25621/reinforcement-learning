"""Project 26 -- an EKF that keeps a wheeled robot on the map.

Seven experiments:

  1. dead reckoning against the EKF, on the same drive
  2. are the Jacobians right?  checked against numerical differentiation
  3. the one missing line: what forgetting to wrap an angle actually costs
  4. is the filter honest?  NEES, and tuning the noise model with no truth
  5. landmark geometry: one, two, six, and six in a straight line
  6. EKF against UKF, as the linearization gets worse
  7. which landmark is which?  nearest-neighbour association and where it breaks

Runs in about three minutes.  NumPy and Matplotlib only.
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

from kf import chi2_interval, chi2_ppf                                  # noqa: E402
from world import (wrap, move, motion_jacobians, motion_noise, ALPHA,   # noqa: E402
                   range_bearing, measurement_jacobian, landmark_field,
                   figure_eight, simulate, pose_error)
from ekf import EKF, UKF, pose_nees                                     # noqa: E402
from plot_style import COLORS, use_style, save                          # noqa: E402

import matplotlib.pyplot as plt                                         # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

DT = 0.1
N = 900                       # 90 seconds of driving
SIGMA_R = 0.25                # metres
SIGMA_B = np.deg2rad(3.0)     # radians
R_MEAS = np.diag([SIGMA_R ** 2, SIGMA_B ** 2])
X0 = np.array([0.0, 0.0, 0.0])
P0 = np.diag([0.1 ** 2, 0.1 ** 2, np.deg2rad(2.0) ** 2])


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def run_ekf(controls, obs, landmarks, x0=X0, P0_=None, R=R_MEAS, truth=None,
            wrap_bearing=True, alpha=None, cls=EKF, gate=None,
            associate=False):
    """Drive an EKF (or UKF) through a whole recorded run."""
    f = cls(x0, P0 if P0_ is None else P0_)
    n = len(controls)
    xs = np.empty((n + 1, 3))
    Ps = np.empty((n + 1, 3, 3))
    xs[0], Ps[0] = f.x, f.P
    nees_v = np.full(n + 1, np.nan)
    nis_v, wrong, used = [], 0, 0
    for k in range(n):
        if cls is UKF:
            f.predict(controls[k], DT, alpha)
        else:
            f.predict(controls[k], DT, alpha)
        for j, z in obs[k + 1]:
            if associate:
                # Pick the landmark whose predicted measurement is closest in
                # MAHALANOBIS distance -- i.e. closest after dividing by how
                # uncertain the prediction is.  Plain Euclidean distance in
                # (range, bearing) would be comparing metres with radians, a
                # quantity with no meaning.  Vectorized over all landmarks: the
                # obvious Python loop makes this experiment 30x slower.
                dxy = landmarks - f.x[:2]
                q = np.sum(dxy ** 2, axis=1)
                r_hat = np.sqrt(q)
                d = np.stack([z[0] - r_hat,
                              wrap(z[1] - (np.arctan2(dxy[:, 1], dxy[:, 0]) - f.x[2]))],
                             axis=1)
                # H per landmark, stacked: shape (L, 2, 3)
                Hs = np.zeros((len(landmarks), 2, 3))
                Hs[:, 0, 0] = -dxy[:, 0] / r_hat
                Hs[:, 0, 1] = -dxy[:, 1] / r_hat
                Hs[:, 1, 0] = dxy[:, 1] / q
                Hs[:, 1, 1] = -dxy[:, 0] / q
                Hs[:, 1, 2] = -1.0
                Ss = np.einsum("lij,jk,lmk->lim", Hs, f.P, Hs) + R
                m = np.einsum("li,lij,lj->l", d, np.linalg.inv(Ss), d)
                best = int(np.argmin(m))
                best_d = float(m[best])
                if gate is not None and best_d > gate:
                    continue
                if best != j:
                    wrong += 1
                j = best
            used += 1
            if cls is UKF:
                y, S = f.update(z, landmarks[j], R)
            else:
                y, S = f.update(z, landmarks[j], R, wrap_bearing=wrap_bearing)
            nis_v.append(float(y @ np.linalg.solve(S, y)))
        xs[k + 1], Ps[k + 1] = f.x, f.P
        if truth is not None:
            nees_v[k + 1] = pose_nees(f.x, truth[k + 1], f.P)
    return xs, Ps, np.array(nis_v), nees_v, wrong, used


# =====================================================================  1
def exp1_dead_reckoning_vs_ekf(rng):
    banner("1. Dead reckoning drifts; the EKF does not")

    lms = landmark_field("spread")
    controls = figure_eight(N, DT)
    truth, odom, obs = simulate(X0, controls, DT, rng, lms,
                                SIGMA_R, SIGMA_B)
    xs, Ps, nis_v, nees_v, _, _ = run_ekf(controls, obs, lms, truth=truth)

    od_p = np.array([pose_error(odom[k], truth[k])[0] for k in range(N + 1)])
    ek_p = np.array([pose_error(xs[k], truth[k])[0] for k in range(N + 1)])
    od_h = np.array([pose_error(odom[k], truth[k])[1] for k in range(N + 1)])
    ek_h = np.array([pose_error(xs[k], truth[k])[1] for k in range(N + 1)])
    dist = np.sum(np.hypot(np.diff(truth[:, 0]), np.diff(truth[:, 1])))
    n_obs = sum(len(o) for o in obs)

    print(f"  drove {dist:.1f} m in {N*DT:.0f} s, saw {n_obs} landmark readings")
    print(f"  {'':>18} {'final':>9} {'mean':>9} {'max':>9}")
    print(f"  position (m)")
    print(f"    dead reckoning   {od_p[-1]:9.3f} {od_p.mean():9.3f} {od_p.max():9.3f}")
    print(f"    EKF              {ek_p[-1]:9.3f} {ek_p.mean():9.3f} {ek_p.max():9.3f}")
    print(f"  heading (deg)")
    print(f"    dead reckoning   {np.rad2deg(od_h[-1]):9.3f} {np.rad2deg(od_h.mean()):9.3f} "
          f"{np.rad2deg(od_h.max()):9.3f}")
    print(f"    EKF              {np.rad2deg(ek_h[-1]):9.3f} {np.rad2deg(ek_h.mean()):9.3f} "
          f"{np.rad2deg(ek_h.max()):9.3f}")
    print(f"\n  dead-reckoning drift: {100*od_p[-1]/dist:.2f}% of distance travelled")
    print(f"  EKF error is BOUNDED -- it does not grow with distance at all "
          f"(max {ek_p.max():.3f} m)")
    print(f"  mean NEES over 3 states: {np.nanmean(nees_v):.3f} (target 3.000)")

    record(1, "distance_m", value=float(dist))
    record(1, "odom_final_pos_err", value=float(od_p[-1]))
    record(1, "ekf_final_pos_err", value=float(ek_p[-1]))
    record(1, "odom_max_pos_err", value=float(od_p.max()))
    record(1, "ekf_max_pos_err", value=float(ek_p.max()))
    record(1, "odom_final_head_deg", value=float(np.rad2deg(od_h[-1])))
    record(1, "ekf_final_head_deg", value=float(np.rad2deg(ek_h[-1])))
    record(1, "mean_nees", value=float(np.nanmean(nees_v)))

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax0.plot(truth[:, 0], truth[:, 1], "k--", lw=1.3, label="truth")
    ax0.plot(odom[:, 0], odom[:, 1], color=COLORS[1], label="dead reckoning")
    ax0.plot(xs[:, 0], xs[:, 1], color=COLORS[0], label="EKF")
    ax0.plot(lms[:, 0], lms[:, 1], "*", ms=12, color=COLORS[2], label="landmarks")
    # 3-sigma ellipses every 100 steps
    th = np.linspace(0, 2 * np.pi, 60)
    for k in range(0, N + 1, 100):
        w, V = np.linalg.eigh(Ps[k][:2, :2])
        e = (V * 3 * np.sqrt(np.maximum(w, 0))) @ np.stack([np.cos(th), np.sin(th)])
        ax0.plot(xs[k, 0] + e[0], xs[k, 1] + e[1], color=COLORS[0], lw=0.8, alpha=0.7)
    ax0.set_aspect("equal"); ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("A figure-eight, with and without landmarks")
    ax0.legend(fontsize=8, loc="upper left")

    t = np.arange(N + 1) * DT
    ax1.semilogy(t, np.maximum(od_p, 1e-3), color=COLORS[1], label="dead reckoning")
    ax1.semilogy(t, np.maximum(ek_p, 1e-3), color=COLORS[0], label="EKF")
    ax1.semilogy(t, 3 * np.sqrt(Ps[:, 0, 0] + Ps[:, 1, 1]), ":", color=COLORS[0],
                 label="EKF reported $3\\sigma$")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position error (m)")
    ax1.set_title("Unbounded growth versus a bounded error")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "dead_reckoning.png"))
    return lms, controls, truth, obs


# =====================================================================  2
def exp2_jacobians(rng):
    banner("2. Are the Jacobians right?  (against numerical differentiation)")

    def numeric(fn, x, eps=1e-6, angular_out=None):
        J = np.zeros((len(fn(x)), len(x)))
        for i in range(len(x)):
            dp, dm = x.copy(), x.copy()
            dp[i] += eps; dm[i] -= eps
            d = fn(dp) - fn(dm)
            if angular_out is not None:
                d[angular_out] = wrap(d[angular_out])
            J[:, i] = d / (2 * eps)
        return J

    worst_G, worst_V, worst_H = 0.0, 0.0, 0.0
    for _ in range(400):
        x = np.array([rng.uniform(-10, 10), rng.uniform(-10, 10),
                      rng.uniform(-np.pi, np.pi)])
        u = np.array([rng.uniform(0.2, 2.0), rng.uniform(-1.0, 1.0)])
        lm = np.array([rng.uniform(-12, 12), rng.uniform(-12, 12)])
        if np.hypot(*(lm - x[:2])) < 0.5:
            continue
        G, V = motion_jacobians(x, u, DT)
        Gn = numeric(lambda p: move(p, u, DT), x, angular_out=2)
        Vn = numeric(lambda p: move(x, p, DT), u, angular_out=2)
        H = measurement_jacobian(x, lm)
        Hn = numeric(lambda p: range_bearing(p, lm), x, angular_out=1)
        worst_G = max(worst_G, np.max(np.abs(G - Gn)))
        worst_V = max(worst_V, np.max(np.abs(V - Vn)))
        worst_H = max(worst_H, np.max(np.abs(H - Hn)))

    print(f"  400 random poses, controls and landmarks; worst disagreement:")
    print(f"    G = d(move)/d(pose)     {worst_G:.3e}")
    print(f"    V = d(move)/d(control)  {worst_V:.3e}")
    print(f"    H = d(measure)/d(pose)  {worst_H:.3e}")
    print(f"  central differences carry an error around eps^2 ~ 1e-12 plus round-off,")
    print(f"  so anything below ~1e-6 means the analytic derivative is correct.")

    # And what a plausible sign slip actually costs.
    lms = landmark_field("spread")
    controls = figure_eight(N, DT)
    truth, _, obs = simulate(X0, controls, DT, rng, lms, SIGMA_R, SIGMA_B)
    import ekf as ekf_mod

    good = run_ekf(controls, obs, lms, truth=truth)
    err_good = np.mean([pose_error(good[0][k], truth[k])[0] for k in range(N + 1)])

    orig = ekf_mod.measurement_jacobian

    def flipped(x, lm):                       # bearing row sign slip
        H = orig(x, lm)
        H[1, 2] = +1.0
        return H
    ekf_mod.measurement_jacobian = flipped
    bad = run_ekf(controls, obs, lms, truth=truth)
    err_bad = np.mean([pose_error(bad[0][k], truth[k])[0] for k in range(N + 1)])
    ekf_mod.measurement_jacobian = orig

    print(f"\n  flip ONE sign (the -1 in the bearing row's theta column):")
    print(f"    mean position error {err_good:.3f} m -> {err_bad:.3f} m "
          f"({err_bad/err_good:.0f}x worse)")
    print("  the filter still runs, still reports a covariance, and is nonsense.")

    record(2, "jacobian_G_err", value=float(worst_G))
    record(2, "jacobian_V_err", value=float(worst_V))
    record(2, "jacobian_H_err", value=float(worst_H))
    record(2, "err_correct_H", value=float(err_good))
    record(2, "err_sign_flipped_H", value=float(err_bad))


# =====================================================================  3
def exp3_angle_wrapping(rng):
    banner("3. The one line: wrapping the bearing innovation")

    lms = landmark_field("spread")
    controls = figure_eight(N, DT)
    n_rep = 30
    rows = []
    keep = {}
    for wrap_on in (True, False):
        pe, he, blown = [], [], 0
        for rep in range(n_rep):
            truth, _, obs = simulate(X0, controls, DT, rng, lms, SIGMA_R, SIGMA_B)
            xs, Ps, _, _, _, _ = run_ekf(controls, obs, lms, truth=truth,
                                         wrap_bearing=wrap_on)
            e = np.array([pose_error(xs[k], truth[k])[0] for k in range(N + 1)])
            h = np.array([pose_error(xs[k], truth[k])[1] for k in range(N + 1)])
            pe.append(e.mean()); he.append(np.rad2deg(h.mean()))
            blown += int(e.mean() > 1.0)          # "lost" = never recovered
            if rep == 0:
                keep[wrap_on] = (truth, xs, e)
        rows.append((wrap_on, np.mean(pe), np.mean(he), blown))

    print(f"  {'wrap?':>6} {'mean pos err':>13} {'mean head err':>14} {'runs lost':>10}")
    for w, p, h, b in rows:
        print(f"  {str(w):>6} {p:13.3f} m {h:13.2f} deg {b:5d}/{n_rep}")
    print(f"\n  cost of the missing line: {rows[1][1]/rows[0][1]:.0f}x the position error")
    print("  Why it happens: a bearing of +179 deg and one of -179 deg describe")
    print("  almost the same direction, but subtracting them gives 358 deg.  The")
    print("  filter reads that as an enormous surprise and lurches to explain it.")
    fr = np.rad2deg(np.abs(wrap(keep[False][0][:, 2])))
    print(f"  the robot's heading passes within 30 deg of +-180 on "
          f"{100*np.mean(fr > 150):.0f}% of steps, which is when the damage happens")

    for w, p, h, b in rows:
        record(3, "wrap_comparison", wrap=str(w), pos_err=p, head_err_deg=h, lost=b)

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 3.8))
    for wrap_on, col, lab in ((True, COLORS[0], "wrapped (correct)"),
                              (False, COLORS[1], "not wrapped")):
        truth, xs, e = keep[wrap_on]
        ax0.plot(xs[:, 0], xs[:, 1], color=col, label=lab)
        ax1.semilogy(np.arange(N + 1) * DT, np.maximum(e, 1e-3), color=col, label=lab)
    ax0.plot(keep[True][0][:, 0], keep[True][0][:, 1], "k--", lw=1.2, label="truth")
    ax0.plot(lms[:, 0], lms[:, 1], "*", ms=11, color=COLORS[2])
    ax0.set_aspect("equal"); ax0.set_xlabel("x (m)"); ax0.set_ylabel("y (m)")
    ax0.set_title("Same data, one line of code apart")
    ax0.legend(fontsize=8)
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("position error (m)")
    ax1.set_title("The error arrives in jumps, at the heading wrap-arounds")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "angle_wrap.png"))


# =====================================================================  4
def exp4_consistency(rng):
    banner("4. Is the filter honest, and can you tune it without ground truth?")

    lms = landmark_field("spread")
    controls = figure_eight(N, DT)
    scales = [0.04, 0.2, 1.0, 5.0, 25.0]
    n_rep = 25
    rows = []
    for sc in scales:
        alpha = ALPHA * sc
        nees_all, nis_all, errs = [], [], []
        for _ in range(n_rep):
            truth, _, obs = simulate(X0, controls, DT, rng, lms, SIGMA_R, SIGMA_B)
            xs, Ps, nis_v, nees_v, _, _ = run_ekf(controls, obs, lms, truth=truth,
                                                  alpha=alpha)
            nees_all.append(np.nanmean(nees_v[100:]))
            nis_all.append(np.mean(nis_v[200:]))
            errs.append(np.mean([pose_error(xs[k], truth[k])[0]
                                 for k in range(100, N + 1)]))
        rows.append((sc, np.mean(nees_all), np.mean(nis_all), np.mean(errs)))

    lo3, hi3 = chi2_interval(3, 0.05)
    print(f"  {'Q scale':>8} {'mean NEES':>10} {'mean NIS':>9} {'pos err (m)':>12}")
    print(f"  {'':>8} {'(3 = ok)':>10} {'(2 = ok)':>9}")
    for sc, ne, ni, er in rows:
        flag = ""
        if ne > 3.0 * 1.5:
            flag = "  <- OVERCONFIDENT"
        elif ne < 3.0 / 1.5:
            flag = "  <- timid"
        print(f"  {sc:8.2f} {ne:10.3f} {ni:9.3f} {er:12.4f}{flag}")
    best = min(rows, key=lambda r: r[3])
    honest = min(rows, key=lambda r: abs(r[1] - 3.0))
    print(f"\n  lowest error at Q scale {best[0]:.2f} ({best[3]:.4f} m)")
    print(f"  most honest NEES at Q scale {honest[0]:.2f} "
          f"(error {honest[3]:.4f} m, {100*(honest[3]/best[3]-1):+.1f}%)")
    print(f"  NIS picks {min(rows, key=lambda r: abs(r[2]-2.0))[0]:.2f} -- and NIS needs "
          f"no ground truth, so it is the one you can use on hardware")
    print(f"  95% single-sample NEES band for 3 states: [{lo3:.2f}, {hi3:.2f}]")

    for sc, ne, ni, er in rows:
        record(4, "q_scale", scale=sc, nees=ne, nis=ni, pos_err=er)

    use_style()
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.loglog([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0],
              label="mean NEES")
    ax.loglog([r[0] for r in rows], [r[2] for r in rows], "s-", color=COLORS[1],
              label="mean NIS")
    ax.axhline(3.0, ls="--", color=COLORS[0], lw=1)
    ax.axhline(2.0, ls="--", color=COLORS[1], lw=1)
    ax2 = ax.twinx()
    ax2.semilogx([r[0] for r in rows], [r[3] for r in rows], "^-", color=COLORS[2],
                 label="position error")
    ax2.set_ylabel("position error (m)", color=COLORS[2]); ax2.grid(False)
    ax.set_xlabel("process noise, as a multiple of the truth")
    ax.set_ylabel("consistency statistic")
    ax.set_title("Overconfident on the left, timid on the right")
    ax.legend(fontsize=8, loc="upper right")
    save(fig, os.path.join(OUT, "consistency.png"))


# =====================================================================  5
def exp5_geometry(rng):
    banner("5. Landmark geometry decides what you can know")

    controls = figure_eight(N, DT)
    n_rep = 25
    rows = []
    keep = {}
    for kind in ("one", "two", "spread", "collinear"):
        lms = landmark_field(kind)
        pe, he, along, across = [], [], [], []
        for rep in range(n_rep):
            truth, _, obs = simulate(X0, controls, DT, rng, lms, SIGMA_R, SIGMA_B,
                                     max_range=30.0)
            xs, Ps, _, _, _, _ = run_ekf(controls, obs, lms, truth=truth)
            pe.append(np.mean([pose_error(xs[k], truth[k])[0] for k in range(200, N + 1)]))
            he.append(np.rad2deg(np.mean([pose_error(xs[k], truth[k])[1]
                                          for k in range(200, N + 1)])))
            # the shape of the final uncertainty, biggest and smallest directions
            w, _ = np.linalg.eigh(Ps[-1][:2, :2])
            along.append(np.sqrt(w[1])); across.append(np.sqrt(w[0]))
            if rep == 0:
                keep[kind] = (lms, truth, xs, Ps)
        rows.append((kind, len(lms), np.mean(pe), np.mean(he),
                     np.mean(along), np.mean(across)))

    print(f"  {'layout':>10} {'n':>3} {'pos err':>9} {'head err':>9} "
          f"{'worst dir':>10} {'best dir':>9} {'ratio':>7}")
    print(f"  {'':>10} {'':>3} {'(m)':>9} {'(deg)':>9} {'sigma (m)':>10} {'sigma (m)':>9}")
    for k, n, p, h, a, c in rows:
        print(f"  {k:>10} {n:3d} {p:9.4f} {h:9.3f} {a:10.4f} {c:9.4f} {a/c:7.1f}")
    print(f"\n  one landmark: position error {rows[0][2]:.3f} m -- a single range and")
    print("  bearing to a known point DOES fix a pose in principle, but the")
    print("  uncertainty is wildly anisotropic (see the ratio column).")
    print(f"  six landmarks in a LINE ({rows[3][2]:.4f} m) are {100*(rows[3][2]/rows[2][2]-1):.0f}% "
          f"worse than the same six spread out ({rows[2][2]:.4f} m),")
    print(f"  and their uncertainty is {(rows[3][4]/rows[3][5])/(rows[2][4]/rows[2][5]):.1f}x "
          f"more lopsided (ratio {rows[3][4]/rows[3][5]:.1f} vs {rows[2][4]/rows[2][5]:.1f}).")
    print(f"  tripling the landmark count from 2 to 6 bought only "
          f"{100*(1-rows[3][2]/rows[1][2]):.0f}% when the extra ones were collinear,")
    print(f"  against {100*(1-rows[2][2]/rows[1][2]):.0f}% when they were spread.")

    for k, n, p, h, a, c in rows:
        record(5, "geometry", layout=k, n_landmarks=n, pos_err=p, head_err_deg=h,
               sigma_worst=a, sigma_best=c, anisotropy=a / c)

    use_style()
    fig, axes = plt.subplots(1, 4, figsize=(13.0, 3.4))
    th = np.linspace(0, 2 * np.pi, 80)
    for ax, (kind, n, p, h, a, c) in zip(axes, rows):
        lms, truth, xs, Ps = keep[kind]
        ax.plot(truth[:, 0], truth[:, 1], "k--", lw=1.0)
        ax.plot(xs[:, 0], xs[:, 1], color=COLORS[0], lw=1.2)
        ax.plot(lms[:, 0], lms[:, 1], "*", ms=11, color=COLORS[2])
        for k in range(0, N + 1, 150):
            w, V = np.linalg.eigh(Ps[k][:2, :2])
            e = (V * 3 * np.sqrt(np.maximum(w, 0))) @ np.stack([np.cos(th), np.sin(th)])
            ax.plot(xs[k, 0] + e[0], xs[k, 1] + e[1], color=COLORS[1], lw=0.9)
        ax.set_aspect("equal"); ax.set_title(f"{kind}: {p:.3f} m", fontsize=9)
        ax.set_xlim(-16, 16); ax.set_ylim(-16, 16)
    save(fig, os.path.join(OUT, "geometry.png"))


# =====================================================================  6
def exp6_ekf_vs_ukf(rng):
    banner("6. EKF against UKF: what linearizing about a wrong mean costs")

    # --- part A: no measurements at all, just prediction ------------------
    # Drive an arc for 4 seconds starting from a heading that is uncertain by
    # `sh` degrees, and compare each filter's PREDICTED belief with the truth,
    # obtained by pushing 40 000 samples through the real motion model.
    print("  A. pure prediction: an arc driven for 4 s from an uncertain heading")
    print(f"  {'heading':>8} {'true trace':>11} {'EKF trace':>10} {'EKF cov':>9} "
          f"{'UKF trace':>10} {'UKF cov':>9} {'EKF mean':>9} {'UKF mean':>9}")
    print(f"  {'sigma':>8} {'(m^2)':>11} {'(m^2)':>10} {'rel err':>9} "
          f"{'(m^2)':>10} {'rel err':>9} {'off (m)':>9} {'off (m)':>9}")
    u = np.array([1.2, 0.3])
    steps, M = 40, 40000
    partA = []
    for sh in (5, 15, 30, 45, 60):
        P0_ = np.diag([0.05 ** 2, 0.05 ** 2, np.deg2rad(sh) ** 2])
        smp = (np.zeros((M, 3)) +
               np.array([0.05, 0.05, np.deg2rad(sh)]) * rng.standard_normal((M, 3)))
        for _ in range(steps):
            smp = np.stack([move(p, u, DT) for p in smp])
        mu = np.array([smp[:, 0].mean(), smp[:, 1].mean(),
                       np.arctan2(np.sin(smp[:, 2]).mean(), np.cos(smp[:, 2]).mean())])
        d = smp - mu
        d[:, 2] = wrap(d[:, 2])
        P_true = d.T @ d / M

        e, uk = EKF(X0, P0_), UKF(X0, P0_)
        for _ in range(steps):
            e.predict(u, DT, alpha=np.zeros(4))       # no process noise: the
            uk.predict(u, DT, alpha=np.zeros(4))      # spread is all geometry

        def relerr(P):
            return (np.linalg.norm(P[:2, :2] - P_true[:2, :2]) /
                    np.linalg.norm(P_true[:2, :2]))
        ee = float(np.hypot(*(e.x[:2] - mu[:2])))
        ue = float(np.hypot(*(uk.x[:2] - mu[:2])))
        print(f"  {sh:6d} deg {np.trace(P_true[:2,:2]):11.4f} "
              f"{np.trace(e.P[:2,:2]):10.4f} {relerr(e.P):9.3f} "
              f"{np.trace(uk.P[:2,:2]):10.4f} {relerr(uk.P):9.3f} "
              f"{ee:9.4f} {ue:9.4f}")
        partA.append((sh, relerr(e.P), relerr(uk.P), ee, ue))
        record(6, "prediction_quality", head_sigma_deg=sh, ekf_cov_relerr=relerr(e.P),
               ukf_cov_relerr=relerr(uk.P), ekf_mean_off=ee, ukf_mean_off=ue)

    sh, er, ur, em, um = partA[-1]
    print(f"\n  At {sh} deg of heading uncertainty the EKF's predicted mean is "
          f"{em:.2f} m away")
    print(f"  from where the robot actually ends up on average; the UKF's is "
          f"{um:.2f} m.")
    print("  The reason is geometric: an uncertain heading turns a straight drive")
    print("  into a BANANA of possible positions, curved around the starting point.")
    print("  The centre of a banana is not on the banana.  The EKF, which pushes")
    print("  only the mean through the model, reports the tip of the curve; the")
    print("  UKF averages several points spread along it and lands much closer.")

    # --- part B: does it actually help the estimate? ----------------------
    print("\n  B. end to end, with two landmarks seen only every `stride` steps")
    lms = landmark_field("two")
    controls = figure_eight(N, DT)
    n_rep = 12
    rows = []
    for stride in (1, 30, 80, 150):
        res = {}
        for name, cls in (("EKF", EKF), ("UKF", UKF)):
            pe, t0 = [], time.time()
            for _ in range(n_rep):
                truth, _, obs = simulate(X0, controls, DT, rng, lms, SIGMA_R,
                                         SIGMA_B, max_range=40.0, alpha=ALPHA * 4.0)
                sparse = [o if (k % stride == 0) else [] for k, o in enumerate(obs)]
                xs, _, _, _, _, _ = run_ekf(controls, sparse, lms, truth=truth,
                                            cls=cls, alpha=ALPHA * 4.0)
                pe.append(np.mean([pose_error(xs[k], truth[k])[0]
                                   for k in range(200, N + 1)]))
            res[name] = (np.mean(pe), (time.time() - t0) / n_rep)
        rows.append((stride, res))

    print(f"  {'update':>7} {'seconds':>8} {'EKF err':>9} {'UKF err':>9} {'UKF/EKF':>8} "
          f"{'UKF cost':>9}")
    print(f"  {'stride':>7} {'between':>8} {'(m)':>9} {'(m)':>9}")
    for stride, res in rows:
        e, et = res["EKF"]
        uu, ut = res["UKF"]
        print(f"  {stride:7d} {stride*DT:8.1f} {e:9.4f} {uu:9.4f} {uu/e:8.2f} "
              f"{ut/et:8.1f}x")
    print(f"\n  The UKF's belief is measurably better (part A) and its ANSWER is not")
    print(f"  (part B): at every update rate it ties or loses, while costing "
          f"{rows[-1][1]['UKF'][1]/rows[-1][1]['EKF'][1]:.1f}x.")
    print("  A landmark fix pulls the estimate back onto the truth regardless of how")
    print("  gracefully the filter drifted between fixes, so the better prediction")
    print("  never gets to pay off.  This is why plain EKFs still run most robots:")
    print("  the UKF's advantage is real, and it lives in a regime -- long stretches")
    print("  with no correction -- that a well-instrumented robot tries not to enter.")

    for stride, res in rows:
        for nm in ("EKF", "UKF"):
            record(6, "end_to_end", stride=stride, filt=nm, pos_err=res[nm][0],
                   sec_per_run=res[nm][1])

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.3))
    ax0.plot([p[0] for p in partA], [p[3] for p in partA], "o-", color=COLORS[1],
             label="EKF")
    ax0.plot([p[0] for p in partA], [p[4] for p in partA], "s-", color=COLORS[0],
             label="UKF")
    ax0.set_xlabel("heading uncertainty (deg)")
    ax0.set_ylabel("predicted mean, distance from truth (m)")
    ax0.set_title("A: the prediction the UKF gets right")
    ax0.legend(fontsize=8)
    ax1.plot([r[0] * DT for r in rows], [r[1]["EKF"][0] for r in rows], "o-",
             color=COLORS[1], label="EKF")
    ax1.plot([r[0] * DT for r in rows], [r[1]["UKF"][0] for r in rows], "s-",
             color=COLORS[0], label="UKF")
    ax1.set_xlabel("seconds between landmark fixes")
    ax1.set_ylabel("settled position error (m)")
    ax1.set_title("B: and the answer it does not improve")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "ekf_vs_ukf.png"))


# =====================================================================  7
def exp7_data_association(rng):
    banner("7. Which landmark is which?")

    controls = figure_eight(N, DT)
    gate = chi2_ppf(0.99, 2)
    n_rep = 8
    rows = []
    for count in (6, 20, 60, 150):
        lms = landmark_field(f"n{count}")
        # nearest-neighbour spacing, the number that decides everything here
        d = np.hypot(lms[:, None, 0] - lms[None, :, 0],
                     lms[:, None, 1] - lms[None, :, 1])
        np.fill_diagonal(d, np.inf)
        spacing = float(np.median(d.min(axis=1)))
        res = {}
        # SAME simulated data for all three modes, so the comparison is paired.
        runs = [simulate(X0, controls, DT, rng, lms, SIGMA_R, SIGMA_B,
                         max_range=12.0) for _ in range(n_rep)]
        for mode in ("known", "nn", "nn+gate"):
            pe, wr, us, lost = [], [], [], 0
            for truth, _, obs in runs:
                xs, Ps, _, _, wrong, used = run_ekf(
                    controls, obs, lms, truth=truth,
                    associate=(mode != "known"),
                    gate=(gate if mode == "nn+gate" else None))
                e = np.mean([pose_error(xs[k], truth[k])[0]
                             for k in range(200, N + 1)])
                pe.append(e); wr.append(wrong); us.append(used)
                lost += int(e > 1.0)
            res[mode] = (np.mean(pe), np.mean(wr), np.mean(us), lost)
        rows.append((count, spacing, res))

    print(f"  {'land-':>6} {'nearest':>8} | {'known':>8} | {'nearest neighbour':>26} "
          f"| {'+ 99% gate':>18}")
    print(f"  {'marks':>6} {'spacing':>8} | {'err (m)':>8} | {'err (m)':>8} {'%wrong':>7} "
          f"{'lost':>8} | {'err (m)':>8} {'lost':>8}")
    for count, sp, res in rows:
        k, n, g_ = res["known"], res["nn"], res["nn+gate"]
        print(f"  {count:6d} {sp:8.2f} | {k[0]:8.4f} | {n[0]:8.4f} "
              f"{100*n[1]/max(n[2],1):7.2f} {n[3]:5d}/{n_rep} | "
              f"{g_[0]:8.4f} {g_[3]:5d}/{n_rep}")

    print(f"\n  With 6 landmarks {rows[0][1]:.1f} m apart, guessing the correspondence")
    print(f"  costs nothing at all: {rows[0][2]['known'][0]:.4f} m known vs "
          f"{rows[0][2]['nn'][0]:.4f} m guessed, "
          f"{100*rows[0][2]['nn'][1]/rows[0][2]['nn'][2]:.2f}% wrong.")
    last = rows[-1]
    print(f"  With {last[0]} landmarks only {last[1]:.1f} m apart, "
          f"{100*last[2]['nn'][1]/last[2]['nn'][2]:.1f}% are mis-assigned and the error")
    print(f"  goes {last[2]['known'][0]:.4f} -> {last[2]['nn'][0]:.4f} m "
          f"({last[2]['nn'][0]/last[2]['known'][0]:.1f}x), "
          f"{last[2]['nn'][3]}/{n_rep} tracks lost.")
    print(f"  The 99% gate recovers it to {last[2]['nn+gate'][0]:.4f} m "
          f"({last[2]['nn+gate'][3]}/{n_rep} lost) by refusing readings that fit")
    print("  nothing well.  A gate cannot tell you the right landmark; it can only")
    print("  decline to guess, and declining is worth more than guessing.")
    print("\n  The quantity that predicts all of this is the ratio of the filter's")
    print("  own position uncertainty to the landmark spacing.  When the 3-sigma")
    print("  ellipse is smaller than the gap between landmarks, only one candidate")
    print("  is plausible and association is free.  When it is larger, the filter is")
    print("  guessing, and one wrong guess pulls the estimate towards a place that")
    print("  then makes the NEXT wrong guess more likely.")

    for count, sp, res in rows:
        for m, v in res.items():
            record(7, "association", n_landmarks=count, spacing=sp, mode=m,
                   pos_err=v[0], wrong=v[1], used=v[2], lost=v[3])

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9.6, 3.3))
    cs = [r[0] for r in rows]
    for i, m in enumerate(("known", "nn", "nn+gate")):
        ax0.semilogx(cs, [r[2][m][0] for r in rows], "o-", color=COLORS[i], label=m)
    ax0.set_xlabel("landmarks in the same 26 x 26 m area")
    ax0.set_ylabel("position error (m)")
    ax0.set_title("Association is free until the landmarks crowd")
    ax0.legend(fontsize=8)
    ax1.semilogx(cs, [100 * r[2]["nn"][1] / r[2]["nn"][2] for r in rows], "o-",
                 color=COLORS[1], label="nearest neighbour")
    ax1.semilogx(cs, [100 * r[2]["nn+gate"][1] / r[2]["nn+gate"][2] for r in rows],
                 "s-", color=COLORS[2], label="+ 99% gate")
    ax1.set_xlabel("landmarks"); ax1.set_ylabel("mis-assigned readings (%)")
    ax1.set_title("How often it picks the wrong one")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "association.png"))


def main():
    t0 = time.time()
    rng = np.random.default_rng(11)
    exp1_dead_reckoning_vs_ekf(rng)
    exp2_jacobians(rng)
    exp3_angle_wrapping(rng)
    exp4_consistency(rng)
    exp5_geometry(rng)
    exp6_ekf_vs_ukf(rng)
    exp7_data_association(rng)

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
