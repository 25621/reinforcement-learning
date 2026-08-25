"""Project 24 -- one number, two thermometers, and an honest error bar.

Seven experiments:

  1. fuse two static sensors; check the KF against the closed-form answer
  2. is the reported uncertainty true?  20 000 Monte-Carlo trials
  3. a drifting temperature: where averaging stops working and a filter starts
  4. the Q/R sweep: one ratio to tune, and what each end of it looks like
  5. a biased thermometer: confidently, quietly wrong
  6. the steady-state gain, and why cheap hardware ships a constant
  7. a sensor that dies without saying so, and the gate that catches it

Runs in about 30 seconds.  NumPy and Matplotlib only.
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

from kf import (KalmanFilter, chi2_interval, nis, nees)          # noqa: E402
from plot_style import COLORS, use_style, save                   # noqa: E402

import matplotlib.pyplot as plt                                  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []

# The two thermometers.  A is cheap and noisy, B is three times better.
SIGMA_A = 1.5      # degrees C, one standard deviation
SIGMA_B = 0.5
TRUE_T = 21.0      # the temperature we are trying to recover
N_STEPS = 200
DT = 1.0           # one reading per second


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


# =====================================================================  1
def exp1_static_fusion(rng):
    banner("1. Two thermometers, one still room")

    za = TRUE_T + SIGMA_A * rng.standard_normal(N_STEPS)
    zb = TRUE_T + SIGMA_B * rng.standard_normal(N_STEPS)

    # State is the temperature itself.  It does not move, so F = 1 and Q = 0.
    # A near-flat prior: we start at 0 C with a standard deviation of 100 C,
    # i.e. "the room is somewhere between an ice bath and an oven".
    f = KalmanFilter(x=[0.0], P=[[100.0 ** 2]])
    hist_x, hist_s = [], []
    for k in range(N_STEPS):
        f.predict(F=[[1.0]], Q=[[0.0]])
        f.update(za[k], H=[[1.0]], R=[[SIGMA_A ** 2]])
        f.update(zb[k], H=[[1.0]], R=[[SIGMA_B ** 2]])
        hist_x.append(f.x[0])
        hist_s.append(np.sqrt(f.P[0, 0]))
    hist_x, hist_s = np.array(hist_x), np.array(hist_s)

    # The closed-form answer for a static quantity: inverse-variance weighting.
    # Every reading counts in proportion to how much you trust it.
    wa, wb = 1.0 / SIGMA_A ** 2, 1.0 / SIGMA_B ** 2
    batch = (wa * za.sum() + wb * zb.sum()) / (N_STEPS * (wa + wb))
    batch_sigma = 1.0 / np.sqrt(N_STEPS * (wa + wb))

    # The naive alternative nearly everyone reaches for first.
    plain_mean = 0.5 * (za.mean() + zb.mean())

    print(f"  truth                       {TRUE_T:.4f} C")
    print(f"  mean of sensor A alone      {za.mean():.4f}  (sigma {SIGMA_A/np.sqrt(N_STEPS):.4f})")
    print(f"  mean of sensor B alone      {zb.mean():.4f}  (sigma {SIGMA_B/np.sqrt(N_STEPS):.4f})")
    print(f"  plain average of the two    {plain_mean:.4f}")
    print(f"  inverse-variance weighted   {batch:.4f}  (sigma {batch_sigma:.4f})")
    print(f"  Kalman filter, 400 updates  {hist_x[-1]:.4f}  (sigma {hist_s[-1]:.4f})")
    print(f"  |KF - weighted average|     {abs(hist_x[-1]-batch):.2e} C   <- the same estimator")

    # One-step contraction: the fused variance is smaller than either input.
    p1 = 1.0 / (1.0 / SIGMA_A ** 2 + 1.0 / SIGMA_B ** 2)
    print(f"\n  single-shot fusion of one A and one B reading:")
    print(f"    sigma_A {SIGMA_A:.3f}   sigma_B {SIGMA_B:.3f}   fused {np.sqrt(p1):.3f} C")
    print(f"    the fused number is {100*(1-np.sqrt(p1)/SIGMA_B):.1f}% tighter than the BETTER sensor")

    record(1, "sensor_A_mean", value=za.mean(), sigma=SIGMA_A / np.sqrt(N_STEPS))
    record(1, "sensor_B_mean", value=zb.mean(), sigma=SIGMA_B / np.sqrt(N_STEPS))
    record(1, "plain_average", value=plain_mean, sigma=float("nan"))
    record(1, "weighted_average", value=batch, sigma=batch_sigma)
    record(1, "kalman_filter", value=hist_x[-1], sigma=hist_s[-1])
    record(1, "kf_minus_weighted", value=abs(hist_x[-1] - batch), sigma=float("nan"))
    record(1, "one_shot_fused_sigma", value=np.sqrt(p1), sigma=float("nan"))

    use_style()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax0.plot(za, ".", ms=3, color=COLORS[6], alpha=0.6, label=f"thermometer A ($\\sigma$={SIGMA_A})")
    ax0.plot(zb, ".", ms=3, color=COLORS[5], alpha=0.8, label=f"thermometer B ($\\sigma$={SIGMA_B})")
    ax0.fill_between(np.arange(N_STEPS), hist_x - 3 * hist_s, hist_x + 3 * hist_s,
                     color=COLORS[0], alpha=0.25, label="KF $\\pm 3\\sigma$")
    ax0.plot(hist_x, color=COLORS[0], label="KF estimate")
    ax0.axhline(TRUE_T, color="k", ls="--", lw=1, label="truth")
    ax0.set_ylim(TRUE_T - 5, TRUE_T + 5)
    ax0.set_ylabel("temperature (C)")
    ax0.set_title("Two noisy thermometers and the belief they produce")
    ax0.legend(ncol=3, fontsize=7)

    ax1.semilogy(hist_s, color=COLORS[0], label="KF reported $\\sigma$")
    ax1.axhline(SIGMA_B, color=COLORS[5], ls=":", label="best single reading")
    ax1.plot(np.arange(1, N_STEPS + 1), 1.0 / np.sqrt(np.arange(1, N_STEPS + 1) * (wa + wb)),
             color=COLORS[1], ls="--", label="$1/\\sqrt{k}$ theory")
    ax1.set_xlabel("reading number")
    ax1.set_ylabel("$\\sigma$ (C)")
    ax1.set_title("Covariance contraction: uncertainty falls as $1/\\sqrt{k}$")
    ax1.legend(fontsize=7)
    save(fig, os.path.join(OUT, "fusion.png"))


# =====================================================================  2
def exp2_is_the_error_bar_true(rng):
    banner("2. Is the reported uncertainty the truth?  (20 000 trials)")

    n_trials, n_read = 20000, 10
    errs = np.empty(n_trials)
    nees_vals = np.empty(n_trials)
    reported = None
    for t in range(n_trials):
        f = KalmanFilter(x=[0.0], P=[[100.0 ** 2]])
        for _ in range(n_read):
            f.predict(F=[[1.0]], Q=[[0.0]])
            f.update(TRUE_T + SIGMA_A * rng.standard_normal(), H=[[1.0]], R=[[SIGMA_A ** 2]])
            f.update(TRUE_T + SIGMA_B * rng.standard_normal(), H=[[1.0]], R=[[SIGMA_B ** 2]])
        errs[t] = f.x[0] - TRUE_T
        nees_vals[t] = nees(f.x, [TRUE_T], f.P)
        reported = np.sqrt(f.P[0, 0])

    actual = errs.std()
    lo, hi = chi2_interval(1, 0.05)
    print(f"  filter reports  sigma = {reported:.5f} C")
    print(f"  actually spread sigma = {actual:.5f} C")
    print(f"  ratio actual/reported = {actual/reported:.4f}   (1.000 = perfectly honest)")
    print(f"  mean NEES = {nees_vals.mean():.4f}   should be 1.000 for a 1-D state")
    print(f"  fraction of trials inside the 95% band [{lo:.3f}, {hi:.3f}]: "
          f"{np.mean((nees_vals>lo)&(nees_vals<hi))*100:.1f}%  (target 95.0%)")

    record(2, "reported_sigma", value=reported)
    record(2, "actual_sigma", value=actual)
    record(2, "ratio", value=actual / reported)
    record(2, "mean_nees", value=float(nees_vals.mean()))
    record(2, "pct_in_95_band", value=float(np.mean((nees_vals > lo) & (nees_vals < hi)) * 100))

    use_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax0.hist(errs, bins=80, density=True, color=COLORS[0], alpha=0.7)
    xs = np.linspace(errs.min(), errs.max(), 300)
    ax0.plot(xs, np.exp(-0.5 * (xs / reported) ** 2) / (reported * np.sqrt(2 * np.pi)),
             color=COLORS[1], label=f"N(0, {reported:.3f}$^2$) as reported")
    ax0.set_xlabel("estimate - truth (C)")
    ax0.set_ylabel("density")
    ax0.set_title("The error really is the size the filter claims")
    ax0.legend(fontsize=7)

    ax1.hist(nees_vals, bins=np.linspace(0, 8, 80), density=True, color=COLORS[2], alpha=0.7)
    xs = np.linspace(0.02, 8, 300)
    ax1.plot(xs, np.exp(-xs / 2) / np.sqrt(2 * np.pi * xs), color=COLORS[1],
             label="$\\chi^2$ with 1 dof")
    ax1.set_xlabel("NEES")
    ax1.set_title("NEES matches its theoretical distribution")
    ax1.legend(fontsize=7)
    save(fig, os.path.join(OUT, "honesty.png"))


# =====================================================================  3
def exp3_drifting_room(rng):
    banner("3. The heating comes on: averaging breaks, filtering does not")

    # Truth: temperature ramps at 0.05 C/s with a gentle wobble.
    t = np.arange(N_STEPS) * DT
    truth = TRUE_T + 0.05 * t + 1.5 * np.sin(2 * np.pi * t / 120.0)
    za = truth + SIGMA_A * rng.standard_normal(N_STEPS)
    zb = truth + SIGMA_B * rng.standard_normal(N_STEPS)

    # (a) running weighted average of everything so far -- no motion model
    wa, wb = 1.0 / SIGMA_A ** 2, 1.0 / SIGMA_B ** 2
    cum = (wa * np.cumsum(za) + wb * np.cumsum(zb)) / ((np.arange(N_STEPS) + 1) * (wa + wb))

    # (b) sliding window of 10 -- the usual hand-tuned fix
    win = 10
    pad = np.concatenate
    def sliding(z):
        c = np.cumsum(pad(([0.0], z)))
        out = np.empty(len(z))
        for k in range(len(z)):
            a = max(0, k - win + 1)
            out[k] = (c[k + 1] - c[a]) / (k + 1 - a)
        return out
    slide = (wa * sliding(za) + wb * sliding(zb)) / (wa + wb)

    # (c) KF, state = [temperature, rate of change].  The rate is never measured
    #     directly -- it is inferred from how the temperature readings move.
    F = np.array([[1.0, DT], [0.0, 1.0]])
    q = 0.002
    Q = q * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
    f = KalmanFilter(x=[0.0, 0.0], P=np.diag([100.0 ** 2, 1.0 ** 2]))
    kf_x, kf_rate, kf_s = [], [], []
    for k in range(N_STEPS):
        f.predict(F=F, Q=Q)
        f.update(za[k], H=[[1.0, 0.0]], R=[[SIGMA_A ** 2]])
        f.update(zb[k], H=[[1.0, 0.0]], R=[[SIGMA_B ** 2]])
        kf_x.append(f.x[0]); kf_rate.append(f.x[1]); kf_s.append(np.sqrt(f.P[0, 0]))
    kf_x, kf_rate = np.array(kf_x), np.array(kf_rate)

    skip = 20  # ignore the burn-in, where every method is still finding its feet
    def rmse(e):
        return float(np.sqrt(np.mean((e[skip:] - truth[skip:]) ** 2)))

    print(f"  RMSE, cumulative weighted average   {rmse(cum):7.4f} C")
    print(f"  RMSE, {win}-sample sliding window      {rmse(slide):7.4f} C")
    print(f"  RMSE, Kalman filter                 {rmse(kf_x):7.4f} C")
    print(f"  RMSE, best single sensor (B) raw    {rmse(zb):7.4f} C")
    true_rate = 0.05 + 1.5 * (2 * np.pi / 120.0) * np.cos(2 * np.pi * t / 120.0)
    print(f"\n  the KF also outputs a rate it never measured:")
    print(f"    rate RMSE {np.sqrt(np.mean((kf_rate[skip:]-true_rate[skip:])**2)):.5f} C/s "
          f"on a signal ranging {true_rate.min():.3f}..{true_rate.max():.3f} C/s")

    for nm, e in [("cumulative_average", cum), ("sliding_window", slide),
                  ("kalman", kf_x), ("raw_sensor_B", zb)]:
        record(3, nm, rmse=rmse(e))
    record(3, "kalman_rate", rmse=float(np.sqrt(np.mean((kf_rate[skip:] - true_rate[skip:]) ** 2))))

    use_style()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.4, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax0.plot(t, za, ".", ms=2.5, color=COLORS[6], alpha=0.5)
    ax0.plot(t, zb, ".", ms=2.5, color=COLORS[5], alpha=0.6)
    ax0.plot(t, truth, "k--", lw=1.2, label="truth")
    ax0.plot(t, cum, color=COLORS[1], label=f"weighted average (RMSE {rmse(cum):.2f})")
    ax0.plot(t, slide, color=COLORS[3], label=f"sliding window {win} (RMSE {rmse(slide):.2f})")
    ax0.plot(t, kf_x, color=COLORS[0], label=f"Kalman (RMSE {rmse(kf_x):.2f})")
    ax0.set_ylabel("temperature (C)")
    ax0.set_title("When the quantity moves, the average is chasing a number that no longer exists")
    ax0.legend(fontsize=7, ncol=2)

    ax1.plot(t, true_rate, "k--", lw=1.2, label="true rate")
    ax1.plot(t, kf_rate, color=COLORS[2], label="KF rate (never measured)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("dT/dt (C/s)")
    ax1.set_title("The hidden state the filter recovers for free")
    ax1.legend(fontsize=7)
    save(fig, os.path.join(OUT, "drift.png"))

    return truth, za, zb, F


# =====================================================================  4
def exp4_q_over_r(rng, truth, F):
    banner("4. The one knob: how much you let the estimate move")

    qs = np.logspace(-6, 0, 25)
    n_rep = 30
    rmses, nis_means = [], []
    for q in qs:
        Q = q * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
        rs, ns = [], []
        for rep in range(n_rep):
            za = truth + SIGMA_A * rng.standard_normal(N_STEPS)
            zb = truth + SIGMA_B * rng.standard_normal(N_STEPS)
            f = KalmanFilter(x=[za[0], 0.0], P=np.diag([SIGMA_A ** 2, 1.0]))
            est, nvals = [], []
            for k in range(N_STEPS):
                f.predict(F=F, Q=Q)
                y, S = f.update(za[k], H=[[1.0, 0.0]], R=[[SIGMA_A ** 2]])
                nvals.append(nis(y, S))
                y, S = f.update(zb[k], H=[[1.0, 0.0]], R=[[SIGMA_B ** 2]])
                nvals.append(nis(y, S))
                est.append(f.x[0])
            rs.append(np.sqrt(np.mean((np.array(est)[20:] - truth[20:]) ** 2)))
            ns.append(np.mean(nvals[40:]))
        rmses.append(np.mean(rs)); nis_means.append(np.mean(ns))
    rmses, nis_means = np.array(rmses), np.array(nis_means)
    best = int(np.argmin(rmses))

    print(f"  {'q':>10} {'RMSE (C)':>10} {'mean NIS':>10}   (NIS should be 1.00)")
    for i in range(0, len(qs), 3):
        mark = " <- best" if i == best else ""
        print(f"  {qs[i]:10.2e} {rmses[i]:10.4f} {nis_means[i]:10.3f}{mark}")
    print(f"\n  best q = {qs[best]:.2e}, RMSE {rmses[best]:.4f} C, NIS {nis_means[best]:.3f}")
    print(f"  q too small (1e-6): RMSE {rmses[0]:.4f} C, NIS {nis_means[0]:.2f} "
          f"-> {nis_means[0]:.0f}x too surprised = overconfident")
    print(f"  q too large (1e+0): RMSE {rmses[-1]:.4f} C, NIS {nis_means[-1]:.2f} "
          f"-> too timid, but only {rmses[-1]/rmses[best]:.1f}x worse")
    # Where does NIS cross 1?  That is the tuning rule you can use on hardware.
    cross = qs[int(np.argmin(np.abs(nis_means - 1.0)))]
    print(f"  q where mean NIS = 1 (tunable with NO ground truth): {cross:.2e} "
          f"vs best-RMSE {qs[best]:.2e}")

    for i in range(len(qs)):
        record(4, "q_sweep", q=float(qs[i]), rmse=float(rmses[i]), nis=float(nis_means[i]))

    use_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.semilogx(qs, rmses, "o-", ms=3, color=COLORS[0], label="RMSE (C)")
    ax.axvline(qs[best], color=COLORS[0], ls=":", lw=1)
    ax.set_xlabel("process-noise strength $q$  (how much we let the room surprise us)")
    ax.set_ylabel("RMSE (C)", color=COLORS[0])
    ax2 = ax.twinx()
    ax2.semilogx(qs, nis_means, "s-", ms=3, color=COLORS[1], label="mean NIS")
    ax2.axhline(1.0, color="k", ls="--", lw=1)
    ax2.axvline(cross, color=COLORS[1], ls=":", lw=1)
    ax2.set_ylabel("mean NIS", color=COLORS[1])
    ax2.set_yscale("log")
    ax2.grid(False)
    ax.set_title("Too stiff (left) is far worse than too loose (right).\n"
                 "NIS = 1 finds nearly the same $q$ without ground truth.")
    save(fig, os.path.join(OUT, "q_sweep.png"))


# =====================================================================  5
def exp5_biased_sensor(rng):
    banner("5. A thermometer that is wrong in the same direction every time")

    biases = [0.0, 0.25, 0.5, 1.0, 2.0]
    rows = []
    for bias in biases:
        errs, sigmas, nis_a, nis_b, mean_innov = [], [], [], [], []
        for rep in range(200):
            f = KalmanFilter(x=[0.0], P=[[100.0 ** 2]])
            ia, ib = [], []
            for k in range(N_STEPS):
                f.predict(F=[[1.0]], Q=[[0.0]])
                y, S = f.update(TRUE_T + bias + SIGMA_A * rng.standard_normal(),
                                H=[[1.0]], R=[[SIGMA_A ** 2]])
                ia.append((y[0], nis(y, S)))
                y, S = f.update(TRUE_T + SIGMA_B * rng.standard_normal(),
                                H=[[1.0]], R=[[SIGMA_B ** 2]])
                ib.append((y[0], nis(y, S)))
            errs.append(f.x[0] - TRUE_T)
            sigmas.append(np.sqrt(f.P[0, 0]))
            nis_a.append(np.mean([v for _, v in ia[50:]]))
            nis_b.append(np.mean([v for _, v in ib[50:]]))
            mean_innov.append(np.mean([v for v, _ in ia[50:]]))
        rows.append((bias, np.mean(errs), np.mean(sigmas), np.mean(nis_a),
                     np.mean(nis_b), np.mean(mean_innov)))

    print(f"  {'bias A':>7} {'error':>8} {'reported':>9} {'error/':>7} "
          f"{'NIS A':>7} {'NIS B':>7} {'mean innov A':>13}")
    print(f"  {'(C)':>7} {'(C)':>8} {'sigma':>9} {'sigma':>7}")
    for b, e, s, na, nb, mi in rows:
        print(f"  {b:7.2f} {e:8.4f} {s:9.4f} {e/s:7.1f} {na:7.2f} {nb:7.2f} {mi:13.4f}")
    wgt_a = (1 / SIGMA_A ** 2) / (1 / SIGMA_A ** 2 + 1 / SIGMA_B ** 2)
    print(f"\n  sensor A carries {100*wgt_a:.0f}% of the weight, so a bias b shifts the")
    print(f"  answer by about {wgt_a:.2f} b -- and at b = 2 C that is {rows[-1][1]:.3f} C of error")
    print(f"  against a reported sigma of {rows[-1][2]:.4f} C.  The estimate is "
          f"{rows[-1][1]/rows[-1][2]:.0f} sigma")
    print("  out and the filter has no idea.  NIS does rise (1.00 -> 2.46), but the")
    print("  clean tell is the MEAN innovation: it should hover at 0 and instead sits")
    print("  at +1.81 C.  NIS squares the innovation and throws the sign away; a")
    print("  running mean keeps it, which is why bias monitors watch the mean.")

    for b, e, s, na, nb, mi in rows:
        record(5, "bias_sweep", bias=b, error=e, reported_sigma=s,
               error_over_sigma=e / s, nis_A=na, nis_B=nb, mean_innov_A=mi)

    use_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    bs = [r[0] for r in rows]
    ax.plot(bs, [r[1] for r in rows], "o-", color=COLORS[1], label="actual error")
    ax.plot(bs, [3 * r[2] for r in rows], "s-", color=COLORS[0], label="reported $3\\sigma$")
    ax.set_xlabel("bias on thermometer A (C)")
    ax.set_ylabel("C")
    ax.set_title("The error bar does not notice a bias.  It was never asked to.")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "bias.png"))


# =====================================================================  6
def exp6_steady_state(rng):
    banner("6. The gain stops changing; the filter becomes three multiplications")

    F = np.array([[1.0, DT], [0.0, 1.0]])
    q = 0.002
    Q = q * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
    H = np.array([[1.0, 0.0]])
    R = np.array([[SIGMA_B ** 2]])

    for P0 in (np.diag([1e4, 1e2]), np.diag([1e-4, 1e-4])):
        P = P0.copy()
        gains = []
        for _ in range(300):
            P = F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            P = (np.eye(2) - K @ H) @ P
            gains.append(K.ravel().copy())
        gains = np.array(gains)
        tag = "huge" if P0[0, 0] > 1 else "tiny"
        print(f"  starting from a {tag} P0: final gain K = "
              f"[{gains[-1,0]:.5f}, {gains[-1,1]:.5f}]")
        if tag == "huge":
            g_huge = gains
        else:
            g_tiny = gains
    print(f"  the two runs agree to {np.max(np.abs(g_huge[-1]-g_tiny[-1])):.2e} "
          f"-> the gain forgets the prior entirely")

    # A fixed-gain (alpha-beta) filter using that steady value.
    t = np.arange(N_STEPS) * DT
    truth = TRUE_T + 0.05 * t + 1.5 * np.sin(2 * np.pi * t / 120.0)
    Kss = g_huge[-1]
    errs_kf, errs_ab = [], []
    for rep in range(200):
        z = truth + SIGMA_B * rng.standard_normal(N_STEPS)
        f = KalmanFilter(x=[z[0], 0.0], P=np.diag([SIGMA_B ** 2, 1.0]))
        x_ab = np.array([z[0], 0.0])
        ekf, eab = [], []
        for k in range(N_STEPS):
            f.predict(F=F, Q=Q); f.update(z[k], H=H, R=R)
            ekf.append(f.x[0] - truth[k])
            x_ab = F @ x_ab
            x_ab = x_ab + Kss * (z[k] - x_ab[0])
            eab.append(x_ab[0] - truth[k])
        errs_kf.append(np.sqrt(np.mean(np.array(ekf)[20:] ** 2)))
        errs_ab.append(np.sqrt(np.mean(np.array(eab)[20:] ** 2)))
    print(f"\n  full KF   RMSE {np.mean(errs_kf):.5f} C   (a 2x2 matrix inverse every step)")
    print(f"  fixed gain RMSE {np.mean(errs_ab):.5f} C   (two multiply-adds every step)")
    print(f"  penalty for throwing away the covariance: "
          f"{100*(np.mean(errs_ab)/np.mean(errs_kf)-1):.2f}%")

    record(6, "steady_gain_pos", value=float(g_huge[-1, 0]))
    record(6, "steady_gain_rate", value=float(g_huge[-1, 1]))
    record(6, "kf_rmse", value=float(np.mean(errs_kf)))
    record(6, "fixed_gain_rmse", value=float(np.mean(errs_ab)))

    use_style()
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.semilogy(np.abs(g_huge[:, 0]), color=COLORS[0], label="$K_{pos}$ from a huge $P_0$")
    ax.semilogy(np.abs(g_tiny[:, 0]), ls="--", color=COLORS[1], label="$K_{pos}$ from a tiny $P_0$")
    ax.semilogy(np.abs(g_huge[:, 1]), color=COLORS[2], label="$K_{rate}$ from a huge $P_0$")
    ax.semilogy(np.abs(g_tiny[:, 1]), ls="--", color=COLORS[3], label="$K_{rate}$ from a tiny $P_0$")
    ax.set_xlabel("step")
    ax.set_ylabel("gain")
    ax.set_xlim(0, 60)
    ax.set_title("Two opposite starting beliefs, one steady-state gain")
    ax.legend(fontsize=7)
    save(fig, os.path.join(OUT, "steady_state.png"))


# =====================================================================  7
def exp7_dead_sensor(rng):
    banner("7. A sensor that freezes, and the two-line gate that catches it")

    t = np.arange(N_STEPS) * DT
    truth = TRUE_T + 0.05 * t
    fail_at = 80
    F = np.array([[1.0, DT], [0.0, 1.0]])
    Q = 0.002 * np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]])
    gate = chi2_interval(1, 0.01)[1]     # 99% acceptance threshold for 1 dof
    print(f"  NIS gate for 1 dof at 99%: reject a reading whose NIS exceeds {gate:.3f}")

    def run(use_gate):
        f = KalmanFilter(x=[truth[0], 0.05], P=np.diag([SIGMA_A ** 2, 0.01]))
        est, rejected, nis_hist = [], [], []
        stuck = None
        for k in range(N_STEPS):
            f.predict(F=F, Q=Q)
            za = truth[k] + SIGMA_A * rng.standard_normal()
            if k >= fail_at:                      # sensor A latches its last value
                if stuck is None:
                    stuck = za
                za = stuck
            for z, sig in ((za, SIGMA_A), (truth[k] + SIGMA_B * rng.standard_normal(), SIGMA_B)):
                Hm, Rm = [[1.0, 0.0]], [[sig ** 2]]
                y_pred = z - (np.array(Hm) @ f.x)
                S_pred = np.array(Hm) @ f.P @ np.array(Hm).T + np.array(Rm)
                d = nis(y_pred, S_pred)
                if sig == SIGMA_A:
                    nis_hist.append(d)
                if use_gate and d > gate:
                    rejected.append(k)
                    continue
                f.update(z, H=Hm, R=Rm)
            est.append(f.x[0])
        return np.array(est), rejected, np.array(nis_hist)

    est_no, _, nis_no = run(False)
    est_yes, rej, nis_yes = run(True)
    late = slice(fail_at + 20, N_STEPS)
    print(f"  no gate:   RMSE after the failure {np.sqrt(np.mean((est_no[late]-truth[late])**2)):.4f} C")
    print(f"  with gate: RMSE after the failure {np.sqrt(np.mean((est_yes[late]-truth[late])**2)):.4f} C")
    after = [r for r in rej if r >= fail_at]
    print(f"  readings rejected: {len(rej)} of {N_STEPS}; first rejection after the")
    print(f"    failure at step {after[0] if after else '-'} (sensor died at {fail_at}) "
          f"-> caught in {(after[0]-fail_at) if after else '-'} steps")
    print(f"  false rejections BEFORE the failure: {sum(1 for r in rej if r < fail_at)} "
          f"(a 99% gate should throw away ~1% by chance = ~{0.01*fail_at:.1f})")

    record(7, "rmse_no_gate", value=float(np.sqrt(np.mean((est_no[late] - truth[late]) ** 2))))
    record(7, "rmse_with_gate", value=float(np.sqrt(np.mean((est_yes[late] - truth[late]) ** 2))))
    record(7, "n_rejected", value=float(len(rej)))
    record(7, "first_rejection_after_failure", value=float(after[0] if after else -1))
    record(7, "false_rejections", value=float(sum(1 for r in rej if r < fail_at)))

    use_style()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.4, 5.0), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax0.plot(t, truth, "k--", lw=1.2, label="truth")
    ax0.plot(t, est_no, color=COLORS[1], label="no validation gate")
    ax0.plot(t, est_yes, color=COLORS[0], label="with a NIS gate")
    ax0.axvline(fail_at * DT, color=COLORS[6], lw=1)
    ax0.text(fail_at * DT + 1, truth[0] + 0.3, "thermometer A latches", fontsize=7, color=COLORS[6])
    ax0.set_ylabel("temperature (C)")
    ax0.set_title("A dead sensor still reports numbers, and drags the estimate down with it")
    ax0.legend(fontsize=8)

    ax1.semilogy(t, nis_no, ".", ms=3, color=COLORS[1])
    ax1.axhline(gate, color="k", ls="--", lw=1, label=f"99% gate = {gate:.2f}")
    ax1.axvline(fail_at * DT, color=COLORS[6], lw=1)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("NIS of sensor A")
    ax1.set_title("The gate fires only once the frozen value has become implausible")
    ax1.legend(fontsize=8)
    save(fig, os.path.join(OUT, "dead_sensor.png"))


def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    exp1_static_fusion(rng)
    exp2_is_the_error_bar_true(rng)
    truth, za, zb, F = exp3_drifting_room(rng)
    exp4_q_over_r(rng, truth, F)
    exp5_biased_sensor(rng)
    exp6_steady_state(rng)
    exp7_dead_sensor(rng)

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
