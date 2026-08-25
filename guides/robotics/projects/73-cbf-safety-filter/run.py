"""Wrap a learned policy in a control barrier function and see what it buys.

Seven experiments:

1. the unfiltered policy -- how often does a cloned policy hit an obstacle it
   has never seen?
2. is the constraint right? -- the analytic second derivative against the
   simulator itself
3. order x rate -- relative degree 1 or 2, enforced at 20 Hz or 200 Hz
4. the alpha sweep, where braking earlier turns out to be less safe
5. margin: what you ask for and what you get
6. the honest inversion -- what the barrier does NOT protect
7. distilling the filtered policy: is a student trained on safe data safe?

Run:  python3 run.py     (about 6 minutes; needs numpy, torch, matplotlib)
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402
import cbf as C            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
ROWS = []
N_EVAL = 60
R_SAFE = C.OBST_R + A.R_TIP + C.MARGIN


def record(section, name, value, note=""):
    ROWS.append({"section": section, "quantity": name, "value": value,
                 "note": note})
    print(f"  {name:<48s} {value:>9}   {note}")


def mk(order, alpha, r_safe=R_SAFE, clip=True):
    return lambda env: C.SafetyFilter(env.arm, env.obstacle_c, r_safe,
                                      order=order, alpha=alpha, alpha2=alpha,
                                      clip=clip)


def line(res, label, section="eval"):
    record(section, label, round(res["success"], 3),
           f"tip hits {res['tip_hit']:.3f}  closest {res['min_h'] * 1000:+.1f} mm"
           f"  active {res.get('intervention_rate', 0):.2f}")


def check_constraint(n=60, seed=7):
    """Compare the analytic h'' against the simulator, not against itself.

    A derivative check that differentiates the same formula it is checking
    proves only that the arithmetic is self-consistent.  This one holds the
    command fixed, integrates the real arm three tiny steps, and takes a second
    difference of h -- so a wrong mass matrix or a forgotten Coriolis term
    would show up.
    """
    rng = np.random.default_rng(seed)
    env = C.ObstacleEnv(rng)
    env.reset()
    clean, sat = [], []
    dt = 1e-5
    for i in range(n):
        # Re-draw a fresh episode regularly.  Walking one episode forwards for
        # sixty samples ends up in postures no policy ever visits, including
        # near-singular ones where both the analytic and the numeric second
        # derivative are large and ill-conditioned, and the check then reports
        # a failure of the arithmetic that is really a failure of the test.
        if i % 4 == 0:
            env.reset()
        a_test = rng.uniform(-1, 1, 2)
        f = C.SafetyFilter(env.arm, env.obstacle_c, R_SAFE, order=2, alpha=8.0)
        g, b = f._second_order(env.q, env.qd, A.CTRL_DT)
        h0, _, nh, _, J = C.h_and_grad(env.arm, env.q, f.c, R_SAFE)
        hd = float(nh @ (J @ env.qd))
        # b = -alpha2*psi - alpha*hdot - base, so base falls out of b:
        base = -f.alpha2 * (hd + f.alpha * h0) - f.alpha * hd - b
        analytic = float(g @ a_test) + base
        q, qd = env.q.copy(), env.qd.copy()
        qc = q + A.DQ_MAX * a_test
        hs = [h0]
        saturated = False
        for _ in range(2):
            raw_tau = env.arm.gear * (env.arm.kp * (qc - q) - env.arm.kd * qd)
            tau = env.arm.servo_torque(q, qd, qc)
            saturated |= bool(np.any(np.abs(raw_tau) > env.arm.tau_max + 1e-9))
            q, qd = env.arm.step(q, qd, tau, dt=dt)
            hs.append(float(np.linalg.norm(env.arm.tip(q) - f.c)) - R_SAFE)
        numeric = (hs[0] - 2 * hs[1] + hs[2]) / dt ** 2
        rel = abs(numeric - analytic) / (abs(numeric) + 1.0)
        (sat if saturated else clean).append(rel)
        for _ in range(3):
            env.step(rng.uniform(-1, 1, 2))
    return (float(np.median(clean)) if clean else float("nan"),
            float(np.max(clean)) if clean else float("nan"),
            len(sat) / float(n))


def main():
    t0 = time.time()
    # One thread, deliberately.  Every network call here is a single
    # observation, and splitting a 16 x 256 matrix-vector product across twelve
    # cores spends more time at the thread barrier than on the arithmetic --
    # the same effect project 64 measured on a 64 x 64 matmul.  Measured here:
    # about 55x faster with one thread than with four.
    torch.set_num_threads(1)
    nets.seed_all(0)

    # -- 0. the policy under test -------------------------------------------
    print("\n[0] cloning a policy that has never seen an obstacle")
    obs, act, _ = A.collect_demos(250, seed=0, noise=0.25)
    net, norm, _ = nets.train_bc(obs, act, epochs=120, seed=0)
    policy = nets.make_policy(net, norm)
    clean = A.evaluate(policy, n=N_EVAL, seed=1000)
    record("policy", "success on the clean task (no obstacle)",
           round(clean["success"], 3))

    # -- 1. unfiltered -------------------------------------------------------
    print("\n[1] the same policy, with an obstacle in the way")
    raw = C.evaluate(policy, n=N_EVAL, record_first=3)
    line(raw, "no filter")
    record("scene", "clearance at the start of an episode",
           f"{raw['start_h'] * 1000:.0f} mm",
           "every episode starts outside the barrier")
    record("scene", "puck also hits the obstacle", round(raw["puck_hit"], 3))

    # -- 2. is the constraint arithmetic right? ------------------------------
    print("\n[2] checking the constraint against the simulator")
    e_med, e_max, frac_sat = check_constraint()
    record("verify", "analytic h'' vs the simulator (median)", f"{e_med:.2e}",
           "relative error over 60 states")
    record("verify", "same check, worst state", f"{e_max:.2e}",
           f"a motor saturated in {frac_sat:.0%} of the states")

    # -- 3. order x rate -----------------------------------------------------
    print("\n[3] relative degree 1 or 2, enforced at 20 Hz or 200 Hz")
    grid = {}
    for order in (1, 2):
        for rate in ("policy", "servo"):
            r = C.evaluate(policy, n=N_EVAL, make_filter=mk(order, 25.0),
                           rate=rate,
                           record_first=3 if (order == 2 and rate == "servo") else 0)
            grid[(order, rate)] = r
            hz = "20 Hz" if rate == "policy" else "200 Hz"
            line(r, f"order {order}, {hz}, alpha=25", "grid")
    ho = grid[(2, "servo")]

    # -- 4. alpha ------------------------------------------------------------
    print("\n[4] how late you may brake")
    alphas = [10.0, 25.0, 60.0, 150.0]
    sweep = {}
    for rate in ("policy", "servo"):
        sweep[rate] = []
        for al in alphas:
            r = C.evaluate(policy, n=N_EVAL, make_filter=mk(2, al), rate=rate)
            sweep[rate].append(r)
            hz = "20 Hz" if rate == "policy" else "200 Hz"
            record("alpha", f"order 2, {hz}, alpha={al:g}",
                   round(r["success"], 3),
                   f"tip hits {r['tip_hit']:.3f}  active "
                   f"{r['intervention_rate']:.2f}  clipped {r['clip_rate']:.3f}")

    # -- 5. margin -----------------------------------------------------------
    print("\n[5] the margin you ask for and the one you get")
    marg = [0.0, 0.010, 0.030]
    msweep = []
    for m in marg:
        rs = C.OBST_R + A.R_TIP + m
        r = C.evaluate(policy, n=N_EVAL, make_filter=mk(2, 60.0, r_safe=rs),
                       rate="servo")
        msweep.append(r)
        record("margin", f"margin asked {m * 1000:.0f} mm",
               f"{r['min_h'] * 1000:+.1f} mm obtained",
               f"success {r['success']:.3f}  tip hits {r['tip_hit']:.3f}")

    # -- 6. what the barrier does not protect -------------------------------
    print("\n[6] the thing the barrier was not written about")
    best = C.evaluate(policy, n=N_EVAL, make_filter=mk(2, 150.0), rate="servo")
    line(best, "order 2, 200 Hz, alpha=150", "best")
    record("scope", "tip hits: no filter -> filtered",
           f"{raw['tip_hit']:.3f} -> {best['tip_hit']:.3f}")
    record("scope", "PUCK hits: no filter -> filtered",
           f"{raw['puck_hit']:.3f} -> {best['puck_hit']:.3f}",
           "the barrier was written about the tip")
    ref60 = sweep["servo"][2]
    noclip = C.evaluate(policy, n=N_EVAL, make_filter=mk(2, 60.0, clip=False),
                        rate="servo")
    record("saturation", "steps where the QP answer left [-1, 1]",
           round(ref60["clip_rate"], 4))
    record("saturation", "tip hits if the box is ignored",
           round(noclip["tip_hit"], 3),
           f"with the box {ref60['tip_hit']:.3f}")

    # -- 7. distil -----------------------------------------------------------
    print("\n[7] can a student learn the safety instead of being handed it?")
    rng = np.random.default_rng(11)
    env = C.ObstacleEnv(rng)
    O, Y = [], []
    for _ in range(300):
        o = env.reset()
        f = C.SafetyFilter(env.arm, env.obstacle_c, R_SAFE, order=2,
                           alpha=150.0, alpha2=150.0)
        for _ in range(A.EP_LEN):
            a_nom = np.clip(policy(o), -1, 1)
            a = f(a_nom, env.q, env.qd, dt=A.CTRL_DT)
            tip = env.arm.tip(env.q)
            O.append(np.concatenate([o, env.obstacle_c - tip, [C.OBST_R]]))
            Y.append(a)
            o, _, done, _ = env.step(a)
            if done:
                break
    O, Y = np.array(O, np.float32), np.array(Y, np.float32)
    snet, snorm, _ = nets.train_bc(O, Y, epochs=150, seed=1)
    sp = nets.make_policy(snet, snorm)

    class StudentRunner:
        """The student needs the obstacle, so it cannot use the plain signature."""

        def __init__(self, env):
            self.env = env

        def __call__(self, o):
            tip = self.env.arm.tip(self.env.q)
            return sp(np.concatenate([o, self.env.obstacle_c - tip,
                                      [C.OBST_R]]))

    rng = np.random.default_rng(1000)
    env = C.ObstacleEnv(rng)
    hits = succ = 0
    minh = []
    for _ in range(N_EVAL):
        r = C.run_episode(env, StudentRunner(env), None)
        hits += r["tip_hit"]
        succ += r["success"]
        minh.append(r["min_h"])
    stud = {"success": succ / N_EVAL, "tip_hit": hits / N_EVAL,
            "puck_hit": 0.0, "min_h": float(np.mean(minh))}
    record("distil", "student trained on filtered actions",
           round(stud["success"], 3),
           f"tip hits {stud['tip_hit']:.3f}  closest {stud['min_h'] * 1000:+.1f} mm")
    record("distil", "its teacher (policy + filter)", round(best["success"], 3),
           f"tip hits {best['tip_hit']:.3f}")
    record("distil", "training transitions", len(O))

    record("cost", "total runtime (s)", round(time.time() - t0, 1))

    # ---------------- figures ----------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    for k, (trajs, title) in enumerate([(raw["trajs"], "no filter"),
                                        (ho["trajs"], "CBF at 200 Hz")]):
        ax = axes[k]
        for traj, c, goal in trajs:
            tp = np.array([t[0] for t in traj])
            pk = np.array([t[1] for t in traj])
            ax.plot(tp[:, 0], tp[:, 1], "-", lw=1.5, c="#457b9d")
            ax.plot(pk[:, 0], pk[:, 1], "--", lw=1.2, c="#d1495b")
            ax.add_patch(plt.Circle(c, C.OBST_R, color="0.35"))
            ax.add_patch(plt.Circle(c, R_SAFE, fill=False, ls=":", ec="#2a9d8f"))
            ax.scatter(*goal, marker="*", s=130, c="#e9c46a", zorder=4)
        ax.set_aspect("equal")
        ax.set_title(title + "   (blue = tip, red = puck)", fontsize=9)
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    ax = axes[2]
    for rate, col in [("policy", "#457b9d"), ("servo", "#d1495b")]:
        hz = "20 Hz" if rate == "policy" else "200 Hz"
        ax.plot(alphas, [r["tip_hit"] for r in sweep[rate]], "o-", c=col,
                label=f"tip hits, {hz}")
        ax.plot(alphas, [r["success"] for r in sweep[rate]], "s--", c=col,
                alpha=0.55, label=f"task success, {hz}")
    ax.set_xscale("log")
    ax.set_xlabel("alpha  (larger = brake later)")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=7)
    ax.set_title("braking earlier is not safer")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "filter.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    names = ["no filter", "deg 1\n20 Hz", "deg 1\n200 Hz", "deg 2\n20 Hz",
             "deg 2\n200 Hz", "deg 2, 200 Hz\nalpha=150", "student"]
    rs = [raw, grid[(1, "policy")], grid[(1, "servo")], grid[(2, "policy")],
          grid[(2, "servo")], best, stud]
    hitv = [r["tip_hit"] for r in rs]
    sucv = [r["success"] for r in rs]
    x = np.arange(len(names))
    ax[0].bar(x - 0.2, hitv, 0.38, label="tip hits obstacle", color="#d1495b")
    ax[0].bar(x + 0.2, sucv, 0.38, label="task success", color="#2a9d8f")
    for i, (h, s) in enumerate(zip(hitv, sucv)):
        ax[0].text(i - 0.2, h + 0.015, f"{h:.2f}", ha="center", fontsize=7)
        ax[0].text(i + 0.2, s + 0.015, f"{s:.2f}", ha="center", fontsize=7)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(names, fontsize=7)
    ax[0].legend(fontsize=8)
    ax[0].set_ylim(0, 1.12)
    ax[0].set_title(f"{N_EVAL} episodes each")
    ax[1].plot([m * 1000 for m in marg], [r["min_h"] * 1000 for r in msweep],
               "o-", c="#264653", label="obtained")
    ax[1].plot([m * 1000 for m in marg], [m * 1000 for m in marg], "k--", lw=0.9,
               label="what you asked for")
    ax[1].axhline(0, color="#d1495b", lw=0.8)
    ax[1].set_xlabel("margin asked for (mm)")
    ax[1].set_ylabel("closest approach (mm)")
    ax[1].set_title("margin asked vs margin obtained")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "compare.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "note"])
        w.writeheader()
        w.writerows(ROWS)
    print(f"\nwrote {OUT}/results.csv  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
