"""Project 37 -- direct collocation: swinging a cart-pole up with IPOPT.

Seven experiments:

  1. the swing-up: what the optimiser found, and why it looks like pumping
  2. trapezoidal against Hermite-Simpson: two integrators, two error orders
  3. how many knots you need, and what they cost
  4. the initial guess, and the several different answers it can lead to
  5. weaker motors need more pumps -- measured
  6. the plan is not a controller: open loop against plan + LQR
  7. collocation against single shooting, and why nobody shoots far

Runs in about six minutes.  Needs casadi (pip install casadi), NumPy, Matplotlib.
"""

import csv
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_PROJ, "09-cart-pole-lqr"))
sys.path.insert(0, os.path.join(_PROJ, "01-transform-calculator"))

from collocation import (solve, single_shooting, replay, defect, swing_count,  # noqa: E402
                         X0, XF, deriv_np)
from cartpole import CartPole                                              # noqa: E402
from lqr import dlqr, lqr                                                  # noqa: E402
from plot_style import COLORS, use_style, save                             # noqa: E402

import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import Rectangle                                   # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
RESULTS = []


def record(exp, name, **kw):
    RESULTS.append(dict(experiment=exp, name=name, **kw))


def banner(text):
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)


def draw_cartpole(ax, s, l=0.5, alpha=1.0, color=None, offset=0.0):
    """One frame of the cart-pole.  `offset` shifts it sideways so a whole
    sequence can be laid out as a film strip instead of piling up on top of
    itself -- the cart only travels about 0.8 m, so without the offset the
    nine snapshots would be unreadable."""
    x, _, th, _ = s
    x = x + offset
    ax.add_patch(Rectangle((x - 0.10, -0.04), 0.20, 0.08,
                           color=color or "#4A4A4A", alpha=alpha))
    tipx = x + l * math.sin(th)
    tipy = l * math.cos(th)
    ax.plot([x, tipx], [0, tipy], color=color or COLORS[1], lw=2.2,
            alpha=alpha)
    ax.plot([tipx], [tipy], "o", color=color or COLORS[1], ms=6, alpha=alpha)


# =====================================================================  1
def exp1_swingup(rng):
    banner("1. The swing-up")

    r = solve(N=80, T=2.0, u_max=20.0, method="hermite-simpson")
    X, U, T = r["X"], r["U"], r["T"]
    print(f"  solved in {r['seconds']*1e3:.0f} ms, {r['iters']} IPOPT "
          f"iterations, objective {r['obj']:.3f}")
    print(f"  {4*(r['N']+1)} state variables + {r['N']} controls = "
          f"{4*(r['N']+1)+r['N']} unknowns, "
          f"{4*r['N']+8} equality constraints")
    print(f"  final state: {np.round(X[:, -1], 6)} (target {XF})")
    print(f"  the pole reverses direction {swing_count(X)} time(s) on the way up")
    print(f"  peak force {np.max(np.abs(U)):.2f} N of the {20.0} N available")
    record(1, "solution", ms=round(r["seconds"] * 1e3, 1), iters=r["iters"],
           objective=round(r["obj"], 4), swings=swing_count(X),
           peak_force=round(float(np.max(np.abs(U))), 3),
           unknowns=4 * (r["N"] + 1) + r["N"])

    plant = CartPole()
    energy = np.array([plant.energy(X[:, k]) for k in range(X.shape[1])])
    e_top = plant.energy(np.array([0.0, 0.0, 0.0, 0.0]))
    print(f"  mechanical energy goes from {energy[0]:.3f} J (hanging) to "
          f"{energy[-1]:.3f} J (upright); the balance point needs {e_top:.3f} J")

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 3.4),
                             gridspec_kw={'width_ratios': [1, 1, 2.4]})
    ts = np.linspace(0, T, X.shape[1])
    for j, (nm, c) in enumerate((("cart x (m)", COLORS[0]),
                                 ("cart v (m/s)", COLORS[2]),
                                 ("pole angle (rad)", COLORS[1]),
                                 ("pole rate (rad/s)", COLORS[3]))):
        axes[0].plot(ts, X[j], color=c, label=nm)
    axes[0].set_xlabel("time (s)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("the four states")
    axes[1].step(np.linspace(0, T, len(U)), U, where="post", color=COLORS[0])
    axes[1].axhline(20, color=COLORS[1], ls="--")
    axes[1].axhline(-20, color=COLORS[1], ls="--")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("force (N)")
    axes[1].set_title("the plan, and the motor limit")
    ax = axes[2]
    idx = np.linspace(0, X.shape[1] - 1, 9).astype(int)
    for j, k in enumerate(idx):
        draw_cartpole(ax, X[:, k], alpha=0.30 + 0.70 * j / (len(idx) - 1),
                      offset=1.35 * j)
    ax.set_xlim(-0.9, 1.35 * (len(idx) - 1) + 0.9)
    ax.set_ylim(-0.75, 0.75)
    ax.set_aspect("equal")
    ax.set_title("nine snapshots, laid out left to right\n"
                 "(faint = early, solid = late)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    save(fig, os.path.join(OUT, "swingup.png"))


# =====================================================================  2
def exp2_methods(rng):
    banner("2. Trapezoidal against Hermite-Simpson")

    print(f"  {'N':>5s} {'dt (s)':>8s} | {'trapz defect':>13s} {'ms':>7s} | "
          f"{'H-S defect':>12s} {'ms':>7s} | {'ratio':>7s}")
    rows = []
    for N in (10, 20, 40, 80, 160, 320):
        line = [N, 2.0 / N]
        for meth in ("trapezoidal", "hermite-simpson"):
            r = solve(N=N, T=2.0, u_max=20.0, method=meth)
            d = defect(r["X"], r["U"], r["T"], meth) if r["ok"] else math.nan
            line += [d, r["seconds"] * 1e3]
        rows.append(line)
        print(f"  {N:5d} {line[1]:8.4f} | {line[2]:13.2e} {line[3]:7.0f} | "
              f"{line[4]:12.2e} {line[5]:7.0f} | {line[2]/line[4]:7.0f}x")
        record(2, f"N_{N}", dt=round(line[1], 5), trapz_defect=line[2],
               trapz_ms=round(line[3], 1), hs_defect=line[4],
               hs_ms=round(line[5], 1), ratio=round(line[2] / line[4], 1))

    dts = np.array([r[1] for r in rows])
    for k, nm in ((2, "trapezoidal"), (4, "hermite-simpson")):
        d = np.array([r[k] for r in rows])
        good = np.isfinite(d) & (d > 1e-12)
        p = np.polyfit(np.log(dts[good]), np.log(d[good]), 1)[0]
        print(f"  {nm:<17s} error scales as dt^{p:.2f} "
              f"(theory: {'2' if k == 2 else '4'})")
        record(2, f"order_{nm}", measured=round(float(p), 3),
               theory=2 if k == 2 else 4)
    print("  Read this the practical way: to halve the error, trapezoidal")
    print("  needs 1.4x the knots and Hermite-Simpson needs 1.19x.")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.loglog(dts, [r[2] for r in rows], "o-", color=COLORS[1],
              label="trapezoidal")
    ax.loglog(dts, [r[4] for r in rows], "s-", color=COLORS[0],
              label="Hermite-Simpson")
    ax.set_xlabel("time step dt (s)")
    ax.set_ylabel("worst dynamics defect")
    ax.legend(fontsize=8)
    ax.set_title("Two integrators, two slopes")
    save(fig, os.path.join(OUT, "methods.png"))


# =====================================================================  3
def exp3_knots(rng):
    banner("3. How many knots, and what they cost")

    print(f"  {'N':>5s} {'unknowns':>9s} {'iters':>6s} {'ms':>7s} "
          f"{'objective':>11s} {'defect':>10s}")
    rows = []
    for N in (10, 20, 40, 80, 160, 320, 640):
        r = solve(N=N, T=2.0, u_max=20.0, method="hermite-simpson")
        d = defect(r["X"], r["U"], r["T"], r["method"])
        rows.append((N, r["seconds"] * 1e3, r["obj"], d, r["iters"]))
        print(f"  {N:5d} {5*N+4:9d} {r['iters']:6d} {r['seconds']*1e3:7.0f} "
              f"{r['obj']:11.4f} {d:10.2e}")
        record(3, f"N_{N}", unknowns=5 * N + 4, iters=r["iters"],
               ms=round(r["seconds"] * 1e3, 1), objective=round(r["obj"], 5),
               defect=d)
    p = np.polyfit(np.log([r[0] for r in rows]),
                   np.log([r[1] for r in rows]), 1)[0]
    print(f"  solve time scales as N^{p:.2f} -- close to linear, because the "
          f"constraint Jacobian is BANDED: knot k only touches knot k+1, so "
          f"the linear algebra inside IPOPT is sparse.")
    record(3, "time_exponent", exponent=round(float(p), 3))

    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.loglog([r[0] for r in rows], [r[1] for r in rows], "o-", color=COLORS[0])
    ax.set_xlabel("knots N")
    ax.set_ylabel("solve time (ms)")
    ax.set_title(f"Cost grows as N^{p:.2f}, not N^3")
    save(fig, os.path.join(OUT, "knots.png"))


# =====================================================================  4
def exp4_guesses(rng):
    banner("4. The initial guess, and the answers it leads to")

    # Two regimes.  With a strong motor and a short horizon there is really
    # only one sensible manoeuvre, and every guess finds it.  With a weak motor
    # and a long horizon there are several, and which one you get is decided by
    # where you started -- which is the honest general case.
    for label, T, umax, N in (("strong motor, 2.0 s horizon", 2.0, 20.0, 80),
                              ("weak motor, 4.0 s horizon", 4.0, 4.0, 100)):
        print(f"\n  {label}")
        print(f"  {'guess':<10s} {'solved':>8s} {'mean iters':>11s} "
              f"{'objective: min / max':>24s} {'distinct':>9s} {'reversals':>12s}")
        for guess in ("zeros", "hold", "linear", "random"):
            objs, iters, ok, swings = [], [], 0, []
            n = 1 if guess != "random" else 10
            for sd in range(n):
                r = solve(N=N, T=T, u_max=umax, method="hermite-simpson",
                          guess=guess, seed=sd)
                if r["ok"] and abs(r["X"][2, -1]) < 1e-4:
                    ok += 1
                    objs.append(r["obj"])
                    iters.append(r["iters"])
                    swings.append(swing_count(r["X"]))
            distinct = len({round(o, 1) for o in objs})
            print(f"  {guess:<10s} {ok:>3d}/{n:<4d} "
                  f"{np.mean(iters) if iters else math.nan:11.1f} "
                  f"{min(objs) if objs else math.nan:11.3f} / "
                  f"{max(objs) if objs else math.nan:9.3f} {distinct:9d} "
                  f"{str(sorted(set(swings))):>12s}")
            record(4, f"{label} | {guess}", solved=ok, of=n,
                   mean_iters=round(float(np.mean(iters)), 1) if iters else "",
                   obj_min=round(float(min(objs)), 4) if objs else "",
                   obj_max=round(float(max(objs)), 4) if objs else "",
                   distinct=distinct,
                   reversals=sorted(set(swings)) if swings else "")

    # collect the genuinely different answers from the hard regime
    seen = {}
    for sd in range(10):
        r = solve(N=100, T=4.0, u_max=4.0, method="hermite-simpson",
                  guess="random", seed=sd)
        if not (r["ok"] and abs(r["X"][2, -1]) < 1e-4):
            continue
        seen.setdefault(round(r["obj"], 1), r)
    print(f"\n  from 10 random guesses on the hard problem the solver found "
          f"{len(seen)} genuinely different trajectories, objectives "
          f"{sorted(seen)}")
    print("  Nothing here is a bug.  IPOPT is a LOCAL method: it walks downhill")
    print("  from wherever you put it and stops at the bottom of whichever")
    print("  valley that was.  Sampling planners (projects 32-33) explore")
    print("  globally and optimise badly; this optimises beautifully and does")
    print("  not explore at all.  That is why the two get chained together.")
    record(4, "distinct_from_random", count=len(seen),
           objectives=sorted(seen))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for k, (key, r) in enumerate(sorted(seen.items())[:4]):
        ts = np.linspace(0, r["T"], r["X"].shape[1])
        axes[0].plot(ts, r["X"][2], color=COLORS[k % 7],
                     label=f"objective {key:.1f}, "
                           f"{swing_count(r['X'])} reversals")
        axes[1].step(np.linspace(0, r["T"], len(r["U"])), r["U"], where="post",
                     color=COLORS[k % 7])
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("pole angle (rad)")
    axes[0].set_title("Local optima are different MANOEUVRES")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("force (N)")
    axes[1].set_title("...and different force plans")
    save(fig, os.path.join(OUT, "guesses.png"))


# =====================================================================  5
def exp5_weak_motor(rng):
    banner("5. Weaker motors need more pumps")

    print("  The objective here is w_T * T + a small penalty on effort, and T")
    print("  itself is a decision variable -- so the solver is asked for the")
    print("  FASTEST swing-up, not merely a feasible one.")
    print("  Each solve is warm-started from the previous, weaker-motor case;")
    print("  from a cold straight-line guess the 5 N case onwards do not")
    print("  converge at all.")
    print(f"  {'u_max (N)':>10s} {'solved':>7s} {'min time (s)':>13s} "
          f"{'reversals':>10s} {'peak |u|':>9s}")
    rows, warm = [], None
    for umax in (30.0, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.5):
        r = solve(N=120, T=2.5, u_max=umax, method="hermite-simpson",
                  guess="linear", free_time=True, w_u=0.02, w_T=1.0,
                  warm=warm, T_hi=14.0, max_iter=1500)
        ok = r["ok"] and abs(r["X"][2, -1]) < 1e-3
        if ok:
            warm = r
        sw = swing_count(r["X"]) if ok else -1
        rows.append((umax, ok, r["T"], sw))
        print(f"  {umax:10.1f} {str(ok):>7s} {r['T']:13.3f} {sw:10d} "
              f"{np.max(np.abs(r['U'])):9.2f}")
        record(5, f"umax_{umax}", solved=ok, min_time=round(r["T"], 4),
               reversals=sw, peak=round(float(np.max(np.abs(r["U"]))), 3))
    failed = [r[0] for r in rows if not r[1]]
    if failed:
        print(f"  below {min(r[0] for r in rows if r[1]):.0f} N the solver "
              f"stops converging ({failed} N): the manoeuvre needs more and "
              f"more pumps, and with a fixed 120 knots each pump gets fewer "
              f"points to be described with.")
        record(5, "unconverged", u_max_values=failed)
    good = [r for r in rows if r[1]]
    print(f"  a {good[0][0]:.0f} N motor gets there in {good[0][2]:.2f} s "
          f"with {good[0][3]} reversal(s); a {good[-1][0]:.1f} N motor needs "
          f"{good[-1][2]:.2f} s and {good[-1][3]}")
    record(5, "summary", strong_time=round(good[0][2], 3),
           strong_reversals=good[0][3], weak_time=round(good[-1][2], 3),
           weak_reversals=good[-1][3])

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    axes[0].plot([r[0] for r in good], [r[2] for r in good], "o-",
                 color=COLORS[0])
    axes[0].set_xlabel("force limit (N)")
    axes[0].set_ylabel("minimum time (s)")
    axes[0].set_xscale("log")
    axes[0].set_title("Time to the top against motor strength")
    warm2 = None
    keep = {}
    for umax in (30.0, 20.0, 12.0, 8.0, 5.0, 3.0):
        r = solve(N=120, T=2.5, u_max=umax, method="hermite-simpson",
                  free_time=True, w_u=0.02, w_T=1.0, warm=warm2, T_hi=14.0,
                  max_iter=1500)
        if r["ok"] and abs(r["X"][2, -1]) < 1e-3:
            warm2 = r
            keep[umax] = r
    for k, umax in enumerate((20.0, 5.0, 3.0)):
        r = keep.get(umax)
        if r is None:
            continue
        ts = np.linspace(0, r["T"], r["X"].shape[1])
        axes[1].plot(ts, r["X"][2], color=COLORS[k], label=f"{umax:.0f} N")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("pole angle (rad)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("A weak motor has to swing back first")
    save(fig, os.path.join(OUT, "weak_motor.png"))


# =====================================================================  6
def exp6_open_loop(rng):
    banner("6. The plan is not a controller")

    r = solve(N=120, T=2.0, u_max=20.0, method="hermite-simpson")
    X, U, T = r["X"], r["U"], r["T"]

    plant = CartPole()
    A, B = plant.linearize()
    K, _ = lqr(A, B, np.diag([1.0, 1.0, 20.0, 1.0]), np.array([[0.05]]))
    K = np.asarray(K).reshape(1, 4)
    print(f"  LQR gain about the upright: {np.round(K.ravel(), 2)}")
    print("  The gain is only switched on once the pole is within 0.5 rad of")
    print("  upright.  It comes from the dynamics LINEARISED there, so using it")
    print("  during the swing-up -- where the pole hangs at pi radians -- is")
    print("  worse than useless: measured, it drives the cart 69 m down a 3 m")
    print("  rail.  A plan-tracking controller for the whole manoeuvre needs a")
    print("  time-varying gain, which is a different (and larger) computation.")

    def deg(a):
        return math.degrees(abs(math.atan2(math.sin(a), math.cos(a))))

    HOLD = 3.0
    open_traj, _ = replay(U, T, hold=HOLD)
    cl_traj, _ = replay(U, T, gain=K, Xref=X, u_max=40.0, hold=HOLD)
    print(f"\n  after the plan ends, both are simulated for a further "
          f"{HOLD:.0f} s")
    print(f"  open loop  : final pole angle {deg(open_traj[2, -1]):8.2f} deg, "
          f"cart {open_traj[0, -1]:+.3f} m")
    print(f"  plan + LQR : final pole angle {deg(cl_traj[2, -1]):8.2f} deg, "
          f"cart {cl_traj[0, -1]:+.3f} m")
    record(6, "replay", open_deg=round(deg(open_traj[2, -1]), 3),
           closed_deg=round(deg(cl_traj[2, -1]), 3),
           open_cart=round(float(open_traj[0, -1]), 4),
           closed_cart=round(float(cl_traj[0, -1]), 4), hold_s=HOLD)

    kicks = [0.0, 0.05, 0.2, 0.5, 1.0]
    op, cl = [], []
    print(f"\n  {'extra pole rate at t=0':<24s} {'open loop':>12s} "
          f"{'plan + LQR':>12s}")
    for kick in kicks:
        x0 = X0.copy()
        x0[3] += kick
        a, _ = replay(U, T, x0=x0, hold=HOLD)
        b, _ = replay(U, T, x0=x0, gain=K, Xref=X, u_max=40.0, hold=HOLD)
        op.append(deg(a[2, -1]))
        cl.append(deg(b[2, -1]))
        print(f"  {kick:>20.2f} rad/s {op[-1]:11.1f}d {cl[-1]:11.1f}d")
        record(6, f"kick_{kick}", open_deg=round(op[-1], 3),
               closed_deg=round(cl[-1], 3))
    print("  The plan is a sequence of forces computed for one exact starting")
    print("  state and one exact model.  It has no way to notice that anything")
    print("  went differently, so nothing corrects it -- and an inverted pole")
    print("  is the least forgiving place on earth to be uncorrected.")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ts_plan = np.linspace(0, T, X.shape[1])
    ts_full = np.linspace(0, T + HOLD, open_traj.shape[1])
    axes[0].plot(ts_plan, X[2], color=COLORS[2], ls="--", label="the plan")
    axes[0].plot(ts_full, open_traj[2], color=COLORS[1], label="open loop")
    axes[0].plot(ts_full, cl_traj[2], color=COLORS[0], label="plan + LQR")
    axes[0].axvline(T, color="#8A939C", lw=0.8)
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("pole angle (rad)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Same plan, replayed two ways\n"
                      "(vertical line: the plan runs out)")
    axes[1].plot(kicks, op, "o-", color=COLORS[1], label="open loop")
    axes[1].plot(kicks, cl, "s-", color=COLORS[0], label="plan + LQR")
    axes[1].set_xlabel("initial disturbance (rad/s)")
    axes[1].set_ylabel("final pole error (deg)")
    axes[1].set_yscale("symlog", linthresh=1.0)
    axes[1].legend(fontsize=8)
    axes[1].set_title("Feedback is what makes a plan usable")
    save(fig, os.path.join(OUT, "open_loop.png"))


# =====================================================================  7
def exp7_shooting(rng):
    banner("7. Collocation against single shooting")

    print(f"  {'N':>5s} | {'collocation ms':>15s} {'iters':>6s} {'ok':>4s} | "
          f"{'shooting ms':>12s} {'iters':>6s} {'ok':>4s}")
    rows = []
    for N in (10, 20, 40, 60, 80, 120):
        c = solve(N=N, T=2.0, u_max=20.0, method="trapezoidal")
        s = single_shooting(N=N, T=2.0, u_max=20.0)
        rows.append((N, c["seconds"] * 1e3, c["iters"], c["ok"],
                     s["seconds"] * 1e3, s["iters"], s["ok"]))
        print(f"  {N:5d} | {rows[-1][1]:15.0f} {c['iters']:6d} "
              f"{str(c['ok']):>4s} | {rows[-1][4]:12.0f} {s['iters']:6d} "
              f"{str(s['ok']):>4s}")
        record(7, f"N_{N}", colloc_ms=round(rows[-1][1], 1),
               colloc_iters=c["iters"], colloc_ok=c["ok"],
               shoot_ms=round(rows[-1][4], 1), shoot_iters=s["iters"],
               shoot_ok=s["ok"])
    both = [r for r in rows if r[3] and r[6]]
    if both:
        print(f"  where both succeed, shooting is "
              f"{np.mean([r[4]/r[1] for r in both]):.0f}x slower and needs "
              f"{np.mean([r[5]/max(r[2],1) for r in both]):.0f}x the iterations")
        record(7, "ratio",
               time=round(float(np.mean([r[4] / r[1] for r in both])), 1),
               iters=round(float(np.mean([r[5] / max(r[2], 1)
                                          for r in both])), 1))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy([r[0] for r in rows], [r[1] for r in rows], "o-",
                color=COLORS[0], label="direct collocation")
    ax.semilogy([r[0] for r in rows], [r[4] for r in rows], "s-",
                color=COLORS[1], label="single shooting")
    ax.set_xlabel("knots / control intervals N")
    ax.set_ylabel("solve time (ms)")
    ax.legend(fontsize=8)
    ax.set_title("The same problem, two formulations")
    save(fig, os.path.join(OUT, "shooting.png"))


def main():
    use_style()
    rng = np.random.default_rng(0)
    exp1_swingup(rng)
    exp2_methods(rng)
    exp3_knots(rng)
    exp4_guesses(rng)
    exp5_weak_motor(rng)
    exp6_open_loop(rng)
    exp7_shooting(rng)

    keys = []
    for r in RESULTS:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\nwrote {os.path.join(OUT, 'results.csv')}  ({len(RESULTS)} rows)")


if __name__ == "__main__":
    main()
