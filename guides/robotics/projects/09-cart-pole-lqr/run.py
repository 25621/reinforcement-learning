"""Project 09 -- Cart-pole LQR.

Seven experiments:

  1. two independent Riccati solvers, and what the gain K actually says
  2. balancing the NONLINEAR cart-pole with a gain designed on the linear model
  3. the Q/R trade-off curve, and what "optimal" is optimal for
  4. LQR against a PID that only watches the pole
  5. the basin of attraction, and the honest inversion hiding in it
  6. where the linear model stops being true
  7. continuous gains applied at a slow rate, vs a gain designed for that rate

Runs in about 50 seconds on a CPU.  NumPy and Matplotlib only -- SciPy is not
installed here, so ``lqr.py`` solves the Riccati equation from scratch, twice.
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

from cartpole import CartPole, simulate, G  # noqa: E402
from lqr import (lqr, dlqr, care_hamiltonian, dare_iterate, discretize, expm,  # noqa: E402
                 are_residual, closed_loop_poles)
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

PLANT = dict(M=1.0, m=0.1, l=0.5)
Q_DEF = np.diag([1.0, 1.0, 10.0, 1.0])
R_DEF = np.array([[0.01]])
STATE_NAMES = ["cart position", "cart velocity", "pole angle", "pole rate"]


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<54s} {value:>13.5f} {unit}")


# ---------------------------------------------------------------------------
# 1. Two solvers
# ---------------------------------------------------------------------------
def exp1_solvers():
    print("[1] two Riccati solvers")
    plant = CartPole(**PLANT)
    A, B = plant.linearize()
    K, P = lqr(A, B, Q_DEF, R_DEF)

    record("1-solve", "ARE residual, Hamiltonian method", are_residual(A, B, Q_DEF, R_DEF, P), "")

    # Second opinion.  Discretise the system at a step dt, iterate the DISCRETE
    # Riccati equation to its fixed point, and compare.  The two answers cannot
    # agree exactly: a sampled controller is genuinely a different controller,
    # and the gap is proportional to dt.  So the test is not "is the gap small"
    # but "does the gap HALVE when dt halves" -- if it does, the two solvers
    # agree in the limit and the remaining difference is the sampling, not a bug.
    dts = [4e-3, 2e-3, 1e-3, 5e-4]
    gaps = []
    for dt in dts:
        Ad, Bd = discretize(A, B, dt)
        P_d, iters = dare_iterate(Ad, Bd, Q_DEF * dt, R_DEF * dt, iters=400000, tol=1e-13)
        gaps.append(float(np.abs(P - P_d).max()))
        record("1-solve", f"dt={1e3 * dt:.1f} ms: |P_hamiltonian - P_iterated|", gaps[-1], "")
        record("1-solve", f"dt={1e3 * dt:.1f} ms: iterations to converge", float(iters), "")
    ratios = [gaps[i] / gaps[i + 1] for i in range(len(gaps) - 1)]
    record("1-solve", "gap ratio when dt halves (2.0 = first order)", float(np.mean(ratios)), "")

    # expm sanity: exp(A dt) must match a series done a completely different way
    E = expm(A * dt * 100)
    E_ref = np.eye(4)
    term = np.eye(4)
    for k in range(1, 40):
        term = term @ (A * dt * 100) / k
        E_ref = E_ref + term
    record("1-solve", "matrix exponential vs a plain Taylor series", float(np.abs(E - E_ref).max()), "")

    poles = closed_loop_poles(A, B, K)
    record("1-solve", "slowest closed-loop pole (real part)", float(np.max(poles.real)), "1/s")
    record("1-solve", "fastest closed-loop pole (real part)", float(np.min(poles.real)), "1/s")
    open_poles = np.linalg.eigvals(A)
    record("1-solve", "worst OPEN-loop pole (real part)", float(np.max(open_poles.real)), "1/s")

    for name, k in zip(STATE_NAMES, K.ravel()):
        record("1-solve", f"gain on {name}", float(k), "")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    axes[0].axvline(0, color=COLORS[6], lw=1.0)
    axes[0].plot(open_poles.real, open_poles.imag, "x", ms=10, color=COLORS[1], label="open loop")
    axes[0].plot(poles.real, poles.imag, "o", ms=8, color=COLORS[0], label="closed loop")
    axes[0].set_xlabel("real part (1/s)")
    axes[0].set_ylabel("imaginary part (1/s)")
    axes[0].set_title("LQR drags every pole into the left half")
    axes[0].legend(fontsize=8)
    axes[1].barh(STATE_NAMES, K.ravel(), color=COLORS[0])
    axes[1].set_xlabel("gain (force per unit of state)")
    axes[1].set_title("u = -K x")
    save(fig, os.path.join(OUT, "solvers.png"))
    return K


# ---------------------------------------------------------------------------
# 2. Balancing the real (nonlinear) system
# ---------------------------------------------------------------------------
def exp2_balance(K):
    print("[2] balancing the nonlinear cart-pole")
    plant = CartPole(**PLANT, u_max=20.0)
    Krow = K.ravel()
    s0 = np.array([0.0, 0.0, 0.35, 0.0])  # 20 degrees off vertical
    t, S, U = simulate(plant, s0, lambda s: float(-Krow @ s), T=6.0)

    settle_idx = np.nonzero(np.abs(S[:, 2]) > 0.01)[0]
    record("2-balance", "starting tilt", float(np.degrees(s0[2])), "deg")
    record("2-balance", "peak cart excursion", float(np.max(np.abs(S[:, 0]))), "m")
    record("2-balance", "final cart position", float(S[-1, 0]), "m")
    record("2-balance", "final pole angle", float(np.degrees(S[-1, 2])), "deg")
    record("2-balance", "time for |angle| to stay under 0.01 rad",
           float(t[settle_idx[-1]]) if len(settle_idx) else 0.0, "s")
    record("2-balance", "peak force", float(np.max(np.abs(U))), "N")

    # The counter-intuitive first move: to bring the pole back it drives the
    # cart TOWARD the fall, not away from it.
    first = np.argmax(np.abs(U) > 0.1 * np.max(np.abs(U)))
    record("2-balance", "sign of the first force vs sign of the tilt",
           float(np.sign(U[first]) * np.sign(s0[2])), "")

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 5.0), sharex=True)
    for k, (ax, idx, lab) in enumerate([(axes[0, 0], 2, "pole angle (rad)"),
                                        (axes[0, 1], 0, "cart position (m)"),
                                        (axes[1, 0], 3, "pole rate (rad/s)"),
                                        (axes[1, 1], 1, "cart velocity (m/s)")]):
        ax.plot(t, S[:, idx], color=COLORS[k])
        ax.set_ylabel(lab)
        ax.axhline(0, color=COLORS[6], lw=0.8)
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    axes[0, 0].set_title("One knob, four states: all four come back")
    save(fig, os.path.join(OUT, "balance.png"))

    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(t, U, color=COLORS[1])
    ax.set_xlim(0, 2.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force on the cart (N)")
    ax.set_title("The first move is toward the fall, not away from it")
    save(fig, os.path.join(OUT, "force.png"))


# ---------------------------------------------------------------------------
# 3. Q vs R
# ---------------------------------------------------------------------------
def exp3_weights():
    print("[3] the Q/R trade-off")
    plant = CartPole(**PLANT, u_max=1e6)
    A, B = plant.linearize()
    s0 = np.array([0.0, 0.0, 0.25, 0.0])
    rs = np.logspace(-4, 1, 14)
    state_cost, effort, peak_u, settle = [], [], [], []
    keep = {}
    for r in rs:
        K, _ = lqr(A, B, Q_DEF, np.array([[r]]))
        Krow = K.ravel()
        t, S, U = simulate(plant, s0, lambda s: float(-Krow @ s), T=8.0)
        dt = t[1] - t[0]
        state_cost.append(float(np.sum(np.einsum("ij,jk,ik->i", S, Q_DEF, S)) * dt))
        effort.append(float(np.sum(U ** 2) * dt))
        peak_u.append(float(np.max(np.abs(U))))
        idx = np.nonzero(np.abs(S[:, 2]) > 0.005)[0]
        settle.append(float(t[idx[-1]]) if len(idx) else 0.0)
        if r in (rs[0], rs[6], rs[-1]):
            keep[r] = (t, S, U)

    for r, sc, ef, pu, st in zip(rs, state_cost, effort, peak_u, settle):
        record("3-weights", f"R={r:.4g}: peak force", pu, "N")
        record("3-weights", f"R={r:.4g}: settling time", st, "s")
    record("3-weights", "peak force, cheapest control (R=1e-4)", peak_u[0], "N")
    record("3-weights", "peak force, dearest control (R=10)", peak_u[-1], "N")
    record("3-weights", "force ratio across the sweep", peak_u[0] / peak_u[-1], "x")
    record("3-weights", "settling ratio across the sweep", settle[-1] / max(settle[0], 1e-9), "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    axes[0].loglog(effort, state_cost, "o-", color=COLORS[0])
    for r, e, s in zip(rs, effort, state_cost):
        if r in keep:
            axes[0].annotate(f"R = {r:.3g}", (e, s), fontsize=7,
                             textcoords="offset points", xytext=(6, 4))
    axes[0].set_xlabel("control effort  integral(u^2) dt")
    axes[0].set_ylabel("state cost  integral(x^T Q x) dt")
    axes[0].set_title("Every LQR gain sits on this curve; R picks the point")
    for k, (r, (t, S, U)) in enumerate(sorted(keep.items())):
        axes[1].plot(t, S[:, 2], color=COLORS[k], label=f"R = {r:.3g}")
    axes[1].set_xlim(0, 4)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("pole angle (rad)")
    axes[1].set_title("Cheap control is fast; dear control is gentle")
    axes[1].legend(fontsize=8)
    save(fig, os.path.join(OUT, "weights.png"))


# ---------------------------------------------------------------------------
# 4. LQR vs a PID that only watches the pole
# ---------------------------------------------------------------------------
def exp4_vs_pid(K):
    print("[4] LQR vs single-loop PID")
    plant = CartPole(**PLANT, u_max=20.0)
    s0 = np.array([0.0, 0.0, 0.20, 0.0])

    # A PID on the pole angle alone.  Gains chosen to match the LQR's own
    # angle/rate gains, so the ONLY difference is that this controller cannot
    # see the cart at all.
    Krow = K.ravel()
    kp, kd = float(K[0, 2]), float(K[0, 3])
    t, S_pid, U_pid = simulate(plant, s0, lambda s: -(kp * s[2] + kd * s[3]), T=10.0)
    _, S_lqr, U_lqr = simulate(plant, s0, lambda s: float(-Krow @ s), T=10.0)

    record("4-pid", "PID angle gain (copied from LQR)", kp, "N/rad")
    record("4-pid", "PID rate gain (copied from LQR)", kd, "N*s/rad")
    record("4-pid", "PID: final pole angle", float(np.degrees(S_pid[-1, 2])), "deg")
    record("4-pid", "PID: cart drift after 10 s", float(S_pid[-1, 0]), "m")
    record("4-pid", "PID: cart speed at 10 s", float(S_pid[-1, 1]), "m/s")
    record("4-pid", "LQR: final pole angle", float(np.degrees(S_lqr[-1, 2])), "deg")
    record("4-pid", "LQR: cart drift after 10 s", float(S_lqr[-1, 0]), "m")
    record("4-pid", "drift ratio (PID / LQR)",
           abs(S_pid[-1, 0]) / max(abs(S_lqr[-1, 0]), 1e-9), "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    axes[0].plot(t, S_pid[:, 2], color=COLORS[1], label="PID on the pole only")
    axes[0].plot(t, S_lqr[:, 2], color=COLORS[0], label="LQR on all four states")
    axes[0].set_ylabel("pole angle (rad)")
    axes[0].set_title("Both balance the pole...")
    axes[1].plot(t, S_pid[:, 0], color=COLORS[1], label="PID on the pole only")
    axes[1].plot(t, S_lqr[:, 0], color=COLORS[0], label="LQR on all four states")
    axes[1].set_ylabel("cart position (m)")
    axes[1].set_title("...one of them drives off the end of the rail")
    for ax in axes:
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "vs_pid.png"))


# ---------------------------------------------------------------------------
# 5. Basin of attraction
# ---------------------------------------------------------------------------
def basin(K, u_max, thetas, rates, T=6.0, dt=4e-3):
    """Vectorised nonlinear roll-out over a whole grid of starting states."""
    M, m, l = PLANT["M"], PLANT["m"], PLANT["l"]
    TH, RA = np.meshgrid(thetas, rates, indexing="ij")
    x = np.zeros(TH.size)
    xd = np.zeros(TH.size)
    th = TH.ravel().copy()
    thd = RA.ravel().copy()
    k0, k1, k2, k3 = [float(v) for v in K.ravel()]

    def deriv(x, xd, th, thd, u):
        st, ct = np.sin(th), np.cos(th)
        xdd = (u + m * l * thd * thd * st - m * G * st * ct) / (M + m * st * st)
        return xd, xdd, thd, (G * st - xdd * ct) / l

    for _ in range(int(T / dt)):
        u = np.clip(-(k0 * x + k1 * xd + k2 * th + k3 * thd), -u_max, u_max)
        a1 = deriv(x, xd, th, thd, u)
        a2 = deriv(x + .5 * dt * a1[0], xd + .5 * dt * a1[1], th + .5 * dt * a1[2], thd + .5 * dt * a1[3], u)
        a3 = deriv(x + .5 * dt * a2[0], xd + .5 * dt * a2[1], th + .5 * dt * a2[2], thd + .5 * dt * a2[3], u)
        a4 = deriv(x + dt * a3[0], xd + dt * a3[1], th + dt * a3[2], thd + dt * a3[3], u)
        x = x + dt / 6 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        xd = xd + dt / 6 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        th = np.clip(th + dt / 6 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2]), -20, 20)
        thd = np.clip(thd + dt / 6 * (a1[3] + 2 * a2[3] + 2 * a3[3] + a4[3]), -100, 100)
    ok = (np.abs(th) < 0.02) & (np.abs(thd) < 0.2)
    return ok.reshape(TH.shape)


def exp5_basin():
    print("[5] basin of attraction, and the honest inversion")
    plant = CartPole(**PLANT)
    A, B = plant.linearize()
    thetas = np.linspace(-1.4, 1.4, 71)
    rates = np.linspace(-6, 6, 51)
    u_max = 12.0

    grids = {}
    for r in (1e-4, 1e-2, 1.0, 10.0):
        K, _ = lqr(A, B, Q_DEF, np.array([[r]]))
        g = basin(K, u_max, thetas, rates)
        grids[r] = (K, g)
        record("5-basin", f"R={r:.4g}: recovers", 100 * float(g.mean()), "%")
        record("5-basin", f"R={r:.4g}: angle gain", float(K[0, 2]), "N/rad")
        widest = np.abs(thetas[g[:, len(rates) // 2]])
        record("5-basin", f"R={r:.4g}: largest recoverable tilt at rest",
               float(np.degrees(widest.max())) if widest.size else 0.0, "deg")

    best_r = max(grids, key=lambda r: grids[r][1].mean())
    record("5-basin", "R with the LARGEST basin under a 12 N force limit", best_r, "")
    record("5-basin", "its basin vs the cheapest-control basin",
           grids[best_r][1].mean() / grids[1e-4][1].mean(), "x")

    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.0), sharey=True)
    for ax, (r, (K, g)) in zip(axes, sorted(grids.items())):
        ax.imshow(g.T, origin="lower", aspect="auto", cmap="Greens", vmin=0, vmax=1.4,
                  extent=[np.degrees(thetas[0]), np.degrees(thetas[-1]), rates[0], rates[-1]])
        ax.set_title(f"R = {r:.4g}: {100 * g.mean():.0f}%", fontsize=9)
        ax.set_xlabel("starting tilt (deg)")
        ax.grid(False)
    axes[0].set_ylabel("starting pole rate (rad/s)")
    fig.suptitle(f"Basin of attraction with a {u_max:.0f} N force limit", fontsize=10)
    save(fig, os.path.join(OUT, "basin.png"))


# ---------------------------------------------------------------------------
# 6. Where the linear model stops being true
# ---------------------------------------------------------------------------
def exp6_linearization():
    print("[6] where the linear model stops being true")
    plant = CartPole(**PLANT)
    A, B = plant.linearize()
    thetas = np.linspace(0.0, 1.5, 120)
    err_static, err_moving = [], []
    for th in thetas:
        for rate, store in ((0.0, err_static), (3.0, err_moving)):
            s = np.array([0.0, 0.0, th, rate])
            u = 2.0
            nl = plant.deriv(s, u)
            lin = A @ s + B.ravel() * u
            denom = max(np.linalg.norm(nl), 1e-9)
            store.append(float(np.linalg.norm(nl - lin) / denom))

    def first_above(vals, frac):
        idx = np.nonzero(np.array(vals) > frac)[0]
        return float(np.degrees(thetas[idx[0]])) if len(idx) else np.nan

    record("6-linear", "tilt where the linear model is 5% wrong (at rest)", first_above(err_static, 0.05), "deg")
    record("6-linear", "tilt where the linear model is 20% wrong (at rest)", first_above(err_static, 0.20), "deg")
    record("6-linear", "tilt where it is 20% wrong while spinning at 3 rad/s",
           first_above(err_moving, 0.20), "deg")
    record("6-linear", "error at 45 deg, at rest",
           100 * float(np.interp(np.radians(45), thetas, err_static)), "%")

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(np.degrees(thetas), 100 * np.array(err_static), color=COLORS[0], label="pole at rest")
    ax.plot(np.degrees(thetas), 100 * np.array(err_moving), color=COLORS[1], label="pole at 3 rad/s")
    ax.axhline(20, color=COLORS[6], ls="--", lw=1.0)
    ax.set_xlabel("tilt from vertical (deg)")
    ax.set_ylabel("relative error of the linear model (%)")
    ax.set_title("The linearisation is a local promise, and this is how local")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "linearization.png"))


# ---------------------------------------------------------------------------
# 7. Slow control rates
# ---------------------------------------------------------------------------
def exp7_rate():
    print("[7] continuous gains at a slow rate vs a discrete design")
    plant = CartPole(**PLANT, u_max=40.0)
    A, B = plant.linearize()
    K_c, _ = lqr(A, B, Q_DEF, R_DEF)
    s0 = np.array([0.0, 0.0, 0.20, 0.0])
    rates = [500, 200, 100, 60, 40, 30, 25, 20, 15, 12, 10, 8, 6]

    def run(K, f):
        Krow = np.asarray(K).ravel()
        dt_ctrl = 1.0 / f
        sub = max(1, int(round(dt_ctrl / 1e-3)))
        h = dt_ctrl / sub
        s = s0.copy()
        peak = 0.0
        for _ in range(int(8.0 / dt_ctrl)):
            u = float(-Krow @ s)
            for _ in range(sub):
                s = plant.rk4(s, u, h)
            peak = max(peak, abs(s[2]))
            if not np.all(np.isfinite(s)) or peak > 5:
                return np.inf, np.inf
        return peak, abs(s[2])

    cont, disc = [], []
    for f in rates:
        pc, fc = run(K_c, f)
        K_d, _ = dlqr(A, B, Q_DEF, R_DEF, 1.0 / f)
        pd, fd = run(K_d, f)
        cont.append(fc)
        disc.append(fd)
        record("7-rate", f"{f} Hz: final |angle|, continuous gains",
               np.degrees(min(fc, 300)), "deg")
        record("7-rate", f"{f} Hz: final |angle|, discrete design",
               np.degrees(min(fd, 300)), "deg")

    def lowest_ok(vals):
        ok = [f for f, v in zip(rates, vals) if v < np.radians(1.0)]
        return min(ok) if ok else np.nan

    record("7-rate", "lowest rate the continuous gains survive", lowest_ok(cont), "Hz")
    record("7-rate", "lowest rate the discrete design survives", lowest_ok(disc), "Hz")

    K_c_row = K_c.ravel()
    K_d_row = dlqr(A, B, Q_DEF, R_DEF, 1.0 / 20)[0].ravel()
    for name, a, b in zip(STATE_NAMES, K_c_row, K_d_row):
        record("7-rate", f"gain on {name}: continuous -> 20 Hz design", b / a, "x")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.semilogy(rates, np.degrees(np.minimum(cont, 300)), "o-", color=COLORS[1],
                label="continuous gains, sampled")
    ax.semilogy(rates, np.degrees(np.minimum(disc, 300)), "s-", color=COLORS[0],
                label="gains designed for that rate")
    ax.axhline(1.0, color=COLORS[6], ls="--", lw=1.0)
    ax.invert_xaxis()
    ax.set_xlabel("control rate (Hz), slower to the right")
    ax.set_ylabel("|pole angle| after 8 s (deg)")
    ax.set_title("Designing for the rate you actually run at")
    ax.legend(fontsize=8)
    save(fig, os.path.join(OUT, "rate.png"))


def main():
    t0 = time.perf_counter()
    K = exp1_solvers()
    exp2_balance(K)
    exp3_weights()
    exp4_vs_pid(K)
    exp5_basin()
    exp6_linearization()
    exp7_rate()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
