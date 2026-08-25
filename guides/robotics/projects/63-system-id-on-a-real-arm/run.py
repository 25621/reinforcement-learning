"""Five system-ID experiments on the arm in dyn.py.

  1. real measurements  -- a 12-bit encoder, differentiated twice
  2. excitation design  -- what you wiggle decides what you can learn
  3. model structure    -- which terms are worth having
  4. identifiability    -- the fit is perfect and the masses are still wrong
  5. the payoff         -- tracking error with CAD numbers vs. measured ones

Experiment 1 comes first on purpose.  With perfect, noise-free q, qdot and
qddot every excitation in experiment 2 recovers all ten parameters to fifteen
decimal places, and you would conclude that excitation design does not matter.
It matters entirely, but only once the data is real -- so we build the real
measurement chain before asking any other question.

Runs in about 40 seconds; pure numpy.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dyn
from dyn import CAD_THETA, DT, PHYS_NAMES, THETA_NAMES, TRUE_PHYS, TRUE_THETA

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
T_TRAIN = 20.0
KP, KD = np.array([220.0, 120.0]), np.array([28.0, 16.0])

ENC_BITS = 12
ENC_STEP = 2 * np.pi / 2 ** ENC_BITS       # 1.53 mrad per count
TAU_NOISE = 0.02                            # N m, current-sense noise
SG_WIN, LP_WIN, TRIM = 51, 21, 80


def record(exp, **kw):
    ROWS.append(dict(experiment=exp, **kw))


# ---------------------------------------------------------------------------
# excitation signals -- each one is a reference the position servo chases
# ---------------------------------------------------------------------------
def chirp_ref(t, f0=0.2, f1=4.0, T=T_TRAIN):
    """A sine whose frequency slides from f0 to f1.

    "Chirp" is the bird: a bird's chirp slides up in pitch, and so does this.
    The phase must be the INTEGRAL of the frequency, which for a linear sweep
    is f0*t + (f1-f0)*t^2/(2T) -- differentiate it and the frequency comes
    back.  Writing sin(2*pi*f(t)*t) instead is the classic beginner bug: it
    sweeps to twice the frequency you asked for.
    """
    ph = 2 * np.pi * (f0 * t + (f1 - f0) * t * t / (2 * T))
    return np.array([0.45 * np.sin(ph) + 0.35,
                     0.55 * np.sin(0.83 * ph + 1.1) - 0.5])


def sine_ref(t, f=0.5):
    ph = 2 * np.pi * f * t
    return np.array([0.45 * np.sin(ph) + 0.35, 0.55 * np.sin(ph + 1.1) - 0.5])


def step_ref(t, hold=2.0):
    pts = np.array([[0.85, -1.10], [-0.45, 0.30], [0.20, -0.95], [0.70, 0.10],
                    [-0.30, -0.70], [0.55, -0.20], [0.05, 0.35], [0.90, -0.55],
                    [-0.50, -1.05], [0.40, 0.25]])
    return pts[int(t / hold) % len(pts)]


def ramp_ref(t, T=T_TRAIN):
    """Quasi-static: creep through the workspace so slowly that nothing
    accelerates.  This is the "safe" excitation everyone tries first."""
    s = t / T
    return np.array([-0.6 + 1.5 * s, -1.2 + 1.6 * s])


def shoulder_only_ref(t):
    """One joint at a time -- the other one is held at a setpoint.  The
    intuitive way to "isolate" a joint, and a trap."""
    ph = 2 * np.pi * (0.2 * t + 3.8 * t * t / (2 * T_TRAIN))
    return np.array([0.45 * np.sin(ph) + 0.35, -0.5])


_NOISE = np.random.default_rng(3).uniform(-1, 1, (int(T_TRAIN / 0.25) + 3, 2))


def prbs_ref(t):
    """Band-limited random motion -- a smoothed random walk."""
    x = t / 0.25
    i = int(x)
    a = x - i
    v = (1 - a) * _NOISE[i] + a * _NOISE[i + 1]
    return np.array([0.35 + 0.5 * v[0], -0.5 + 0.6 * v[1]])


EXCITATIONS = {
    "chirp 0.2-4 Hz": chirp_ref,
    "band-limited random": prbs_ref,
    "step setpoints": step_ref,
    "single sine 0.5 Hz": sine_ref,
    "shoulder only, elbow held": shoulder_only_ref,
    "slow ramp (quasi-static)": ramp_ref,
}


def servo(ref):
    def ctrl(t, q, qd):
        return KP * (ref(t) - q) - KD * qd
    return ctrl


# ---------------------------------------------------------------------------
# the measurement chain
# ---------------------------------------------------------------------------
def sg_kernels(win, deg=3):
    """Savitzky-Golay: fit a low-order polynomial to a sliding window and read
    its value and derivatives at the centre.  Named after Abraham Savitzky and
    Marcel Golay, who published it in 1964 for smoothing spectrometer traces.
    Fitting a polynomial and then differentiating the polynomial is far
    quieter than differencing noisy samples directly."""
    x = np.arange(-(win // 2), win // 2 + 1) * DT
    A = np.vander(x, deg + 1, increasing=True)
    P = np.linalg.pinv(A)
    return P[0][::-1], P[1][::-1], 2 * P[2][::-1]     # value, d/dt, d2/dt2


def _conv(sig, k):
    out = np.zeros_like(sig)
    for j in range(sig.shape[-1]):
        out[..., j] = np.convolve(sig[..., j], k, mode="same")
    return out


def moving_avg(x, win):
    k = np.ones(win) / win
    if x.ndim == 3:
        return np.stack([_conv(x[:, i, :], k) for i in range(x.shape[1])], 1)
    return _conv(x, k)


def dataset(ref, T=T_TRAIN, realistic=True, seed=0, q0=(0.4, -0.6)):
    """Drive the real arm, then measure it the way a real robot measures.

    Returns (Y, tau) ready for least squares.  ``realistic=False`` hands back
    the simulator's own q, qdot, qddot -- useful only as a reference point,
    since no robot can give you that.
    """
    Q, QD, QDD, TAU = dyn.simulate(servo(ref), T, q0=q0)
    if not realistic:
        return dyn.regressor(Q, QD, QDD), TAU
    rng = np.random.default_rng(seed)
    Qm = np.round(Q / ENC_STEP) * ENC_STEP
    TAUm = TAU + rng.normal(0, TAU_NOISE, TAU.shape)
    kv, kd, kdd = sg_kernels(SG_WIN)
    q, qd, qdd = _conv(Qm, kv), _conv(Qm, kd), _conv(Qm, kdd)
    s = slice(TRIM, -TRIM)
    Y = dyn.regressor(q[s], qd[s], qdd[s])
    # the standard trick: low-pass BOTH sides with the same filter.  Y@theta =
    # tau is linear, so filtering it does not change theta -- but it does
    # average away the noise that survived differentiation.
    return moving_avg(Y, LP_WIN), moving_avg(TAUm[s], LP_WIN)


# ---------------------------------------------------------------------------
# the fit itself: three lines of least squares
# ---------------------------------------------------------------------------
def fit(Y, tau, cols=None):
    """Least squares on the stacked regressor.

    Y is (N, 2, P); flatten the joint axis so every sample contributes two
    equations, then solve the tall skinny system.  ``cols`` drops parameters
    from the model, which is how experiment 3 asks "is this term worth it?".
    """
    P = Y.shape[2]
    cols = np.arange(P) if cols is None else np.asarray(cols)
    A = Y[:, :, cols].reshape(-1, len(cols))
    theta, *_ = np.linalg.lstsq(A, tau.reshape(-1), rcond=None)
    full = np.zeros(P)
    full[cols] = theta
    s = np.linalg.svd(A, compute_uv=False)
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return full, cond


def pred_rms(theta, Y, tau):
    return float(np.sqrt(np.mean((Y @ theta - tau) ** 2)))


def theta_err(theta):
    """Mean relative error over the ten base parameters."""
    return float(np.mean(np.abs(theta - TRUE_THETA) / np.abs(TRUE_THETA)))


def validation_set():
    """A trajectory nothing was fitted on, measured perfectly.

    Scoring against the TRUE torque, not a noisy copy of it, is deliberate:
    we want to know how well the model predicts the robot, not how well it
    predicts our sensor noise.
    """
    def ref(t):
        return np.array([0.3 + 0.5 * np.sin(2 * np.pi * 0.7 * t + 0.4),
                         -0.4 + 0.45 * np.sin(2 * np.pi * 1.3 * t)])
    return dataset(ref, T=8.0, realistic=False, q0=(0.1, -0.2))


# ---------------------------------------------------------------------------
# 1. what a real encoder does to you
# ---------------------------------------------------------------------------
def exp1_measurement(Yv, TAUv):
    print("\n=== 1. what a 12-bit encoder does to you " + "=" * 31)
    Q, QD, QDD, TAU = dyn.simulate(servo(chirp_ref), T_TRAIN)
    rng = np.random.default_rng(0)
    Qm = np.round(Q / ENC_STEP) * ENC_STEP
    TAUm = TAU + rng.normal(0, TAU_NOISE, TAU.shape)

    def central(x, n=1):
        for _ in range(n):
            x = np.gradient(x, DT, axis=0)
        return x

    kv, kd, kdd = sg_kernels(SG_WIN)
    chains = {
        "true qd/qdd (impossible)": (Q, QD, QDD, TAU, False),
        "encoder + plain differences": (Qm, central(Qm, 1), central(Qm, 2),
                                        TAUm, False),
        "encoder + Savitzky-Golay": (_conv(Qm, kv), _conv(Qm, kd),
                                     _conv(Qm, kdd), TAUm, False),
        "+ low-pass both sides": (_conv(Qm, kv), _conv(Qm, kd),
                                  _conv(Qm, kdd), TAUm, True),
    }
    print("  measurement chain             qdd noise   param error   val RMS")
    s = slice(TRIM, -TRIM)
    for name, (q, qd, qdd, tau, lp) in chains.items():
        Y, t = dyn.regressor(q[s], qd[s], qdd[s]), tau[s]
        if lp:
            Y, t = moving_avg(Y, LP_WIN), moving_avg(t, LP_WIN)
        th, cond = fit(Y, t)
        noise = float(np.std(qdd[s] - QDD[s]))
        e, rms = theta_err(th), pred_rms(th, Yv, TAUv)
        print("  %-29s %8.1f %10.1f %% %10.4f" % (name, noise, 100 * e, rms))
        record("measurement", chain=name, qdd_noise=noise,
               param_err_pct=100 * e, val_rms=rms, cond=cond)

    fig, ax = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    a, b = 15000, 15800
    tt = np.arange(a, b) * DT
    ax[0].plot(tt, central(Qm, 2)[a:b, 0], lw=.5, color="#c62828",
               label="plain differences")
    ax[0].plot(tt, QDD[a:b, 0], lw=2, color="k", label="true")
    ax[0].set_title("Acceleration recovered from a 12-bit encoder")
    ax[1].plot(tt, QDD[a:b, 0], lw=3, color="k", label="true")
    ax[1].plot(tt, _conv(Qm, kdd)[a:b, 0], lw=1.2, color="#1976d2",
               label="Savitzky-Golay (51)")
    ax[1].set_title("same data, same axis limits, after the polynomial fit")
    ax[1].set_ylim(ax[0].get_ylim())
    ax[1].set_xlabel("time (s)")
    for a_ in ax:
        a_.set_ylabel("shoulder accel (rad/s^2)")
        a_.legend(fontsize=8, loc="upper right"); a_.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "differentiation.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. excitation design
# ---------------------------------------------------------------------------
def exp2_excitation(Yv, TAUv):
    print("\n=== 2. excitation design (with the real measurement chain) " + "=" * 13)
    print("  excitation                    cond(Y)  param error  validation RMS")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for name, ref in EXCITATIONS.items():
        Y, tau = dataset(ref)
        th, cond = fit(Y, tau)
        e, rms = theta_err(th), pred_rms(th, Yv, TAUv)
        print("  %-28s %8.4g %10.1f %% %12.4f" % (name, cond, 100 * e, rms))
        record("excitation", design=name, cond=cond, param_err_pct=100 * e,
               val_rms=rms)
        Q, *_ = dyn.simulate(servo(ref), T_TRAIN)
        t = np.arange(len(Q)) * DT
        ax[0].plot(t[:6000], Q[:6000, 0], lw=1, label=name)
        ax[1].plot(t[:6000], Q[:6000, 1], lw=1)
    for a, ttl in ((ax[0], "shoulder"), (ax[1], "elbow")):
        a.set_xlabel("time (s)"); a.set_ylabel("angle (rad)")
        a.set_title("the six excitations: " + ttl); a.grid(alpha=.3)
    ax[0].legend(fontsize=6.5, loc="lower left")

    rows = [r for r in ROWS if r["experiment"] == "excitation"]
    ax[2].scatter([r["cond"] for r in rows], [r["val_rms"] for r in rows],
                  s=60, color="#1976d2")
    for r in rows:
        ax[2].annotate(r["design"], (r["cond"], r["val_rms"]), fontsize=6.5,
                       xytext=(4, 4), textcoords="offset points")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xlabel("condition number of the regressor")
    ax[2].set_ylabel("held-out torque RMS (N m)")
    ax[2].set_title("badly conditioned in, garbage out"); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "excitation.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. which terms are worth having
# ---------------------------------------------------------------------------
def exp3_structure(Yv, TAUv):
    print("\n=== 3. model structure " + "=" * 49)
    Y, tau = dataset(chirp_ref)
    models = {
        "full (10 parameters)": list(range(10)),
        "no rotor inertia": [0, 1, 2, 4, 5, 6, 7, 8, 9],
        "no Coulomb friction": [0, 1, 2, 3, 4, 5, 6, 7],
        "no friction at all": [0, 1, 2, 3, 4, 5],
        "rigid-body only (no gravity)": [0, 1, 2, 3],
        "gravity + friction only (no inertia)": [4, 5, 6, 7, 8, 9],
    }
    print("  model                                params  train RMS  val RMS")
    for name, cols in models.items():
        th, _ = fit(Y, tau, cols)
        tr, va = pred_rms(th, Y, tau), pred_rms(th, Yv, TAUv)
        print("  %-36s %5d %10.4f %9.4f" % (name, len(cols), tr, va))
        record("structure", model=name, n_params=len(cols), train_rms=tr,
               val_rms=va)


# ---------------------------------------------------------------------------
# 4. identifiability
# ---------------------------------------------------------------------------
def exp4_identifiability():
    print("\n=== 4. base parameters vs. the parameters you can point at " + "=" * 13)
    Q, QD, QDD, TAU = dyn.simulate(servo(chirp_ref), T_TRAIN)
    Yb = dyn.regressor(Q, QD, QDD)
    Yp = dyn.phys_regressor(Q, QD, QDD)

    for name, Y in (("10 base parameters", Yb), ("12 physical parameters", Yp)):
        A = Y.reshape(-1, Y.shape[2])
        s = np.linalg.svd(A, compute_uv=False)
        rank = int((s > s[0] * 1e-10).sum())
        print("  %-24s columns %2d   rank %2d   cond %.3g"
              % (name, Y.shape[2], rank, s[0] / s[-1]))
        record("identifiability", parameterisation=name, columns=Y.shape[2],
               rank=rank, cond=float(s[0] / s[-1]))

    thb, _ = fit(Yb, TAU)
    thp, *_ = np.linalg.lstsq(Yp.reshape(-1, 12), TAU.reshape(-1), rcond=None)
    print("\n  the 10 base parameters, from perfect data:")
    for n, a, b in zip(THETA_NAMES, TRUE_THETA, thb):
        print("    %-26s true %8.5f   fitted %8.5f  (%+.3f %%)"
              % (n, a, b, 100 * (b - a) / a))
        record("base_param", name=n, true=float(a), fitted=float(b))
    print("\n  the 12 physical parameters, SAME data, SAME residual:")
    for n, b in zip(PHYS_NAMES, thp):
        print("    %-26s true %8.5f   fitted %8.5f" % (n, TRUE_PHYS[n], b))
        record("phys_param", name=n, true=float(TRUE_PHYS[n]), fitted=float(b))
    r1 = pred_rms(thb, Yb, TAU)
    r2 = float(np.sqrt(np.mean((Yp @ thp - TAU) ** 2)))
    print("  residual, base parameterisation      : %.3e N m" % r1)
    print("  residual, physical parameterisation  : %.3e N m" % r2)
    record("identifiability", parameterisation="residual comparison",
           base_residual=r1, phys_residual=r2)


# ---------------------------------------------------------------------------
# 5. the payoff: computed-torque tracking on the real arm
# ---------------------------------------------------------------------------
def _ref5(t):
    return np.array([0.35 + 0.5 * np.sin(2 * np.pi * 0.6 * t),
                     -0.45 + 0.4 * np.sin(2 * np.pi * 0.9 * t + 0.7)])


def _dref5(t, n=1, h=1e-5):
    if n == 1:
        return (_ref5(t + h) - _ref5(t - h)) / (2 * h)
    return (_ref5(t + h) - 2 * _ref5(t) + _ref5(t - h)) / h ** 2


def track(theta_model, T=6.0, skip=0.5):
    """Computed torque: feed forward the model's own prediction of the torque
    the desired motion needs, and let a small PD mop up whatever is left.  The
    better the model, the less work the PD has to do -- so the tracking error
    is a direct readout of model quality."""
    kp, kd = np.array([60.0, 40.0]), np.array([8.0, 5.0])
    err = []

    def ctrl(t, q, qd):
        a = _dref5(t, 2) + kp * (_ref5(t) - q) + kd * (_dref5(t, 1) - qd)
        err.append(_ref5(t) - q)
        return dyn.regressor(q, qd, a)[0] @ theta_model

    dyn.simulate(ctrl, T, q0=_ref5(0.0), qd0=_dref5(0.0, 1))
    e = np.array(err)[int(skip / DT):]
    return float(np.sqrt(np.mean(e ** 2)) * 1e3)      # mrad


def exp5_payoff():
    print("\n=== 5. the payoff: computed-torque tracking " + "=" * 28)
    Y, tau = dataset(chirp_ref)
    th_id, _ = fit(Y, tau)
    th_nofric, _ = fit(Y, tau, list(range(6)))
    th_ramp, _ = fit(*dataset(ramp_ref))

    models = {"CAD / URDF numbers": CAD_THETA,
              "identified from the slow ramp": th_ramp,
              "identified, friction terms dropped": th_nofric,
              "identified, full model": th_id,
              "the true parameters (unknowable)": TRUE_THETA}
    print("  feed-forward model                    tracking RMS")
    for name, th in models.items():
        e = track(th)
        print("  %-37s %7.2f mrad" % (name, e))
        record("payoff", model=name, tracking_rms_mrad=e)

    rows = [r for r in ROWS if r["experiment"] == "payoff"]
    fig, ax = plt.subplots(figsize=(7.5, 3))
    ax.barh([r["model"] for r in rows], [r["tracking_rms_mrad"] for r in rows],
            color=["#c62828", "#e64a19", "#f9a825", "#2e7d32", "#455a64"])
    for y, r in enumerate(rows):
        ax.text(r["tracking_rms_mrad"], y, " %.2f" % r["tracking_rms_mrad"],
                va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("tracking RMS (mrad, log scale)")
    ax.set_title("What the feed-forward model is worth")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "payoff.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    Yv, TAUv = validation_set()
    exp1_measurement(Yv, TAUv)
    exp2_excitation(Yv, TAUv)
    exp3_structure(Yv, TAUv)
    exp4_identifiability()
    exp5_payoff()

    keys = sorted({k for r in ROWS for k in r})
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experiment"] +
                           [k for k in keys if k != "experiment"])
        w.writeheader(); w.writerows(ROWS)
    print("\nwrote outputs/results.csv")
