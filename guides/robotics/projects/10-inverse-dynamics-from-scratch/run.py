"""Project 10 -- Inverse dynamics from scratch (RNEA).

Seven experiments on ``dynamics.py``:

  1. RNEA vs MuJoCo on four robots, including a prismatic joint and a branch
  2. the mass matrix, extracted one column at a time, and what it looks like
  3. the passivity identity: Mdot - 2C is skew-symmetric
  4. O(n): RNEA against building M, C and g separately, for chains up to n = 40
  5. which term dominates?  gravity vs Coriolis vs inertia, as speed rises
  6. energy: three integrators, one undriven arm, 20 seconds
  7. five injected sign bugs, and which check catches which

Runs in about a minute on a CPU.  NumPy, Matplotlib and MuJoCo only.
"""

import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "01-transform-calculator"))

import mujoco  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import dynamics as dyn  # noqa: E402
from plot_style import COLORS, save, use_style  # noqa: E402

OUT = os.path.join(_HERE, "outputs")
os.makedirs(OUT, exist_ok=True)
use_style()
RESULTS = []

MODELS = {
    "arm2": os.path.join(_HERE, "models", "arm2.urdf"),
    "arm6": os.path.join(os.path.dirname(_HERE), "02-urdf-visualizer", "models", "arm6.urdf"),
    "arm7": os.path.join(os.path.dirname(_HERE), "02-urdf-visualizer", "models", "arm7.urdf"),
    "testarm": os.path.join(os.path.dirname(_HERE), "02-urdf-visualizer", "models", "testarm.urdf"),
}


def record(section, name, value, unit=""):
    RESULTS.append({"section": section, "quantity": name, "value": value, "unit": unit})
    print(f"    {name:<52s} {value:>12.4e} {unit}")


def mj_load(path, constraints=False):
    """Load a URDF into MuJoCo, with joint-limit constraints off by default.

    MuJoCo's ``mj_inverse`` returns the torque needed INCLUDING whatever force
    the joint-limit stops are supplying.  Our RNEA has never heard of joint
    limits, so leaving them on compares two different physics: at a
    configuration that sits outside a limit MuJoCo reported a 430 N*m torque
    where RNEA reported zero, and the "bug" was entirely in the comparison.
    """
    m = mujoco.MjModel.from_xml_path(path)
    if not constraints:
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT
    return m, mujoco.MjData(m)


# ---------------------------------------------------------------------------
# 1. RNEA against an independent implementation
# ---------------------------------------------------------------------------
def exp1_verify(n_samples=200, seed=1):
    print("[1] RNEA vs MuJoCo")
    rng = np.random.default_rng(seed)
    rows = []
    for name, path in MODELS.items():
        model = dyn.Model(path)
        m, d = mj_load(path)
        worst_tau = worst_M = worst_g = 0.0
        for _ in range(n_samples):
            q = rng.uniform(model.robot.lower, model.robot.upper)
            qd = rng.uniform(-2, 2, model.n)
            qdd = rng.uniform(-5, 5, model.n)

            d.qpos[:] = q
            d.qvel[:] = qd
            d.qacc[:] = qdd
            mujoco.mj_inverse(m, d)
            worst_tau = max(worst_tau, np.abs(dyn.rnea(model, q, qd, qdd) - d.qfrc_inverse).max())

            M_mj = np.zeros((model.n, model.n))
            mujoco.mj_fullM(m, d, M_mj)
            worst_M = max(worst_M, np.abs(dyn.mass_matrix(model, q) - M_mj).max())

            d.qvel[:] = 0
            d.qacc[:] = 0
            mujoco.mj_inverse(m, d)
            worst_g = max(worst_g, np.abs(dyn.gravity_torque(model, q) - d.qfrc_inverse).max())

        rows.append((name, model.n, worst_tau, worst_M, worst_g))
        record("1-verify", f"{name}: worst |tau_rnea - tau_mujoco|", worst_tau, "N*m")
        record("1-verify", f"{name}: worst |M_mine - M_mujoco|", worst_M, "kg*m^2")
        record("1-verify", f"{name}: worst |g_mine - g_mujoco|", worst_g, "N*m")

    # The trap: leave the joint-limit constraints ON and re-run one case.
    model = dyn.Model(MODELS["arm7"])
    m, d = mj_load(MODELS["arm7"], constraints=True)
    q = np.zeros(model.n)
    d.qpos[:] = q
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    trap = np.abs(dyn.gravity_torque(model, q) - d.qfrc_inverse).max()
    record("1-verify", "arm7 at q=0 with limit constraints LEFT ON", trap, "N*m")

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    labels = [f"{r[0]}\n(n={r[1]})" for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    for k, (lab, idx) in enumerate([("tau (RNEA)", 2), ("M (mass matrix)", 3), ("g (gravity)", 4)]):
        vals = [max(r[idx], 1e-17) for r in rows]
        ax.bar(x + (k - 1) * w, vals, w, label=lab, color=COLORS[k])
    ax.axhline(1e-12, color=COLORS[6], ls="--", lw=1.2)
    ax.text(len(rows) - 0.5, 1.4e-12, "1e-12 target", ha="right", va="bottom", fontsize=8, color=COLORS[6])
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("worst disagreement over 200 random states")
    ax.set_title("From-scratch RNEA vs MuJoCo: every robot agrees to machine precision")
    ax.legend(ncol=3, fontsize=8)
    save(fig, os.path.join(OUT, "verify.png"))
    return rows


# ---------------------------------------------------------------------------
# 2. The mass matrix
# ---------------------------------------------------------------------------
def exp2_mass_matrix(seed=2):
    print("[2] the mass matrix")
    model = dyn.Model(MODELS["arm6"])
    rng = np.random.default_rng(seed)

    worst_asym = 0.0
    min_eig = np.inf
    max_cond = 0.0
    conds = []
    for _ in range(300):
        q = rng.uniform(model.robot.lower, model.robot.upper)
        z = np.zeros(model.n)
        M = np.zeros((model.n, model.n))
        for i in range(model.n):
            e = np.zeros(model.n)
            e[i] = 1.0
            M[:, i] = dyn.rnea(model, q, z, e, gravity=False)  # NOT symmetrised
        worst_asym = max(worst_asym, np.abs(M - M.T).max())
        ev = np.linalg.eigvalsh(0.5 * (M + M.T))
        min_eig = min(min_eig, ev.min())
        conds.append(ev.max() / ev.min())
        max_cond = max(max_cond, conds[-1])

    record("2-mass", "worst |M - M^T| before symmetrising", worst_asym, "kg*m^2")
    record("2-mass", "smallest eigenvalue of M over 300 configs", min_eig, "kg*m^2")
    record("2-mass", "median condition number of M", float(np.median(conds)), "")
    record("2-mass", "worst condition number of M", max_cond, "")

    q_str = np.zeros(model.n)  # arm straight up
    q_out = np.array([0.0, 1.4, 0.4, 0.0, 0.8, 0.0])  # arm reaching out
    M_str, M_out = dyn.mass_matrix(model, q_str), dyn.mass_matrix(model, q_out)
    record("2-mass", "M[0,0] arm straight up (shoulder pan)", M_str[0, 0], "kg*m^2")
    record("2-mass", "M[0,0] arm reaching out (shoulder pan)", M_out[0, 0], "kg*m^2")
    record("2-mass", "ratio out/up for the pan joint", M_out[0, 0] / M_str[0, 0], "x")

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0))
    vmax = max(np.abs(M_str).max(), np.abs(M_out).max())
    for ax, M, title in ((axes[0], M_str, "arm straight up"), (axes[1], M_out, "arm reaching out")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"M(q), {title}")
        ax.set_xticks(range(model.n))
        ax.set_yticks(range(model.n))
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[2].hist(conds, bins=40, color=COLORS[0])
    axes[2].set_xlabel("condition number of M(q)")
    axes[2].set_ylabel("configurations")
    axes[2].set_title("how badly scaled M gets")
    save(fig, os.path.join(OUT, "mass_matrix.png"))


# ---------------------------------------------------------------------------
# 3. Passivity
# ---------------------------------------------------------------------------
def exp3_passivity(seed=3):
    print("[3] passivity: Mdot - 2C is skew-symmetric")
    model = dyn.Model(MODELS["arm2"])
    rng = np.random.default_rng(seed)

    worst_skew = 0.0
    worst_quad = 0.0
    worst_prod = 0.0
    for _ in range(40):
        q = rng.uniform(-2, 2, model.n)
        qd = rng.uniform(-3, 3, model.n)
        C = dyn.coriolis_matrix(model, q, qd)
        Md = dyn.mass_matrix_dot(model, q, qd)
        S = Md - 2 * C
        worst_skew = max(worst_skew, np.abs(S + S.T).max())
        worst_quad = max(worst_quad, abs(float(qd @ S @ qd)))
        worst_prod = max(worst_prod, np.abs(C @ qd - dyn.coriolis_torque(model, q, qd)).max())

    record("3-passivity", "worst |S + S^T| where S = Mdot - 2C", worst_skew, "")
    record("3-passivity", "worst |qd^T (Mdot - 2C) qd|", worst_quad, "W")
    record("3-passivity", "worst |C qd - rnea velocity term|", worst_prod, "N*m")

    # Energy balance on a driven arm: d/dt(T + U) must equal tau . qd exactly.
    # The work integral uses the trapezoidal rule (average of the rate before
    # and after the step).  Summing tau . qd with only the starting velocity is
    # a rectangle rule whose own error is proportional to dt, and it would
    # dominate the answer -- you would be measuring the bookkeeping, not the
    # physics.  Here that swap alone moved the drift from 1e-1 J to 1e-7 J.
    dt = 1e-3
    q = np.array([0.6, -0.9])
    qd = np.array([1.2, -0.7])
    times, drift = [], []
    E0 = dyn.total_energy(model, q, qd)
    work = 0.0
    rng2 = np.random.default_rng(11)
    tau_hist = rng2.normal(0, 4.0, 4000)
    for k in range(4000):
        tau = np.array([tau_hist[k], 0.3 * tau_hist[k]])
        qd_before = qd
        q, qd, _ = dyn.step_rk4(model, q, qd, tau, dt)
        work += 0.5 * float(tau @ (qd_before + qd)) * dt
        if k % 20 == 0:
            times.append(k * dt)
            drift.append(dyn.total_energy(model, q, qd) - E0 - work)
    record("3-passivity", "energy-balance drift after 4 s of random torque", abs(drift[-1]), "J")

    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.plot(times, drift, color=COLORS[0])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("(T + U) - E0 - work done  (J)")
    ax.set_title("Energy bookkeeping closes: every joule is accounted for")
    save(fig, os.path.join(OUT, "passivity.png"))


# ---------------------------------------------------------------------------
# 4. O(n)
# ---------------------------------------------------------------------------
CHAIN_LINK = """  <link name="l{i}">
    <inertial><origin xyz="0 0 0.1"/><mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.002"/></inertial>
    <visual><origin xyz="0 0 0.1"/><geometry><cylinder radius="0.03" length="0.2"/></geometry></visual>
  </link>
  <joint name="j{i}" type="revolute">
    <parent link="{parent}"/><child link="l{i}"/>
    <origin xyz="0 0 {off}" rpy="0 0 0"/><axis xyz="0 {ax} {az}"/>
    <limit lower="-3.0" upper="3.0" effort="100" velocity="3.0"/>
  </joint>
"""


def write_chain(path, n):
    """A synthetic serial chain of n identical links, for the scaling study."""
    body = "".join(
        CHAIN_LINK.format(
            i=i,
            parent="base_link" if i == 0 else f"l{i - 1}",
            off=0.0 if i == 0 else 0.2,
            ax=1 if i % 2 == 0 else 0,
            az=0 if i % 2 == 0 else 1,
        )
        for i in range(n)
    )
    head = (
        '<?xml version="1.0"?>\n<robot name="chain{n}">\n'
        '  <link name="base_link"><inertial><origin xyz="0 0 0"/><mass value="0"/>'
        '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial></link>\n'
    ).format(n=n)
    with open(path, "w") as f:
        f.write(head + body + "</robot>\n")


def exp4_scaling():
    print("[4] O(n): RNEA vs assembling M, C and g")
    tmpdir = os.path.join(OUT, "_chains")
    os.makedirs(tmpdir, exist_ok=True)
    ns = [2, 4, 6, 8, 12, 16, 24, 32, 40]
    t_rnea, t_terms = [], []
    for n in ns:
        path = os.path.join(tmpdir, f"chain{n}.urdf")
        write_chain(path, n)
        model = dyn.Model(path)
        q = np.linspace(0.1, 0.9, n)
        qd = np.linspace(-0.5, 0.5, n)
        qdd = np.linspace(0.2, -0.2, n)

        reps = max(3, int(200 / n))
        t0 = time.perf_counter()
        for _ in range(reps):
            dyn.rnea(model, q, qd, qdd)
        t_rnea.append((time.perf_counter() - t0) / reps)

        t0 = time.perf_counter()
        for _ in range(reps):
            M = dyn.mass_matrix(model, q)
            b = dyn.coriolis_torque(model, q, qd) + dyn.gravity_torque(model, q)
            M @ qdd + b
        t_terms.append((time.perf_counter() - t0) / reps)
        os.remove(path)
    os.rmdir(tmpdir)

    lo, hi = ns.index(8), len(ns) - 1
    slope_r = np.log(t_rnea[hi] / t_rnea[lo]) / np.log(ns[hi] / ns[lo])
    slope_t = np.log(t_terms[hi] / t_terms[lo]) / np.log(ns[hi] / ns[lo])
    record("4-scaling", "fitted exponent p in time ~ n^p, RNEA", slope_r, "")
    record("4-scaling", "fitted exponent p in time ~ n^p, M+C+g", slope_t, "")
    record("4-scaling", "speed-up of RNEA at n=40", t_terms[-1] / t_rnea[-1], "x")
    record("4-scaling", "RNEA time at n=40", t_rnea[-1] * 1e3, "ms")

    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.loglog(ns, np.array(t_rnea) * 1e3, "o-", color=COLORS[0], label=f"RNEA (fit n^{slope_r:.2f})")
    ax.loglog(ns, np.array(t_terms) * 1e3, "s-", color=COLORS[1], label=f"build M, C, g (fit n^{slope_t:.2f})")
    ax.set_xlabel("number of joints n")
    ax.set_ylabel("time for one evaluation (ms)")
    ax.set_title("One sweep out and back beats assembling the matrices")
    ax.legend()
    save(fig, os.path.join(OUT, "scaling.png"))


# ---------------------------------------------------------------------------
# 5. Which term dominates?
# ---------------------------------------------------------------------------
def exp5_terms():
    print("[5] gravity vs Coriolis vs inertia, as speed rises")
    model = dyn.Model(MODELS["arm6"])
    n = model.n
    amp = np.array([0.5, 0.4, 0.6, 0.5, 0.5, 0.6])
    phase = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])
    q0 = np.array([0.0, 0.6, -0.9, 0.0, 0.7, 0.0])

    speeds = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    g_rms, c_rms, i_rms, tot_rms = [], [], [], []
    for s in speeds:
        w = 2 * np.pi * 0.4 * s
        ts = np.linspace(0, 2.5 / max(s, 0.25), 120)
        G = C = I = T = 0.0
        for t in ts:
            q = q0 + amp * np.sin(w * t + phase)
            qd = amp * w * np.cos(w * t + phase)
            qdd = -amp * w * w * np.sin(w * t + phase)
            g = dyn.gravity_torque(model, q)
            c = dyn.coriolis_torque(model, q, qd)
            tot = dyn.rnea(model, q, qd, qdd)
            i = tot - g - c
            G += g @ g
            C += c @ c
            I += i @ i
            T += tot @ tot
        k = len(ts) * n
        g_rms.append(np.sqrt(G / k))
        c_rms.append(np.sqrt(C / k))
        i_rms.append(np.sqrt(I / k))
        tot_rms.append(np.sqrt(T / k))

    for s, g, c, i in zip(speeds, g_rms, c_rms, i_rms):
        record("5-terms", f"speed x{s:.2f}: gravity RMS", g, "N*m")
        record("5-terms", f"speed x{s:.2f}: Coriolis RMS", c, "N*m")
        record("5-terms", f"speed x{s:.2f}: inertia RMS", i, "N*m")
    cross = np.interp(0.0, np.log(np.array(c_rms) / np.array(g_rms)), speeds)
    record("5-terms", "speed where Coriolis overtakes gravity", cross, "x")

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2))
    axes[0].loglog(speeds, g_rms, "o-", color=COLORS[0], label="gravity g(q)")
    axes[0].loglog(speeds, c_rms, "s-", color=COLORS[1], label="Coriolis C(q,qd) qd")
    axes[0].loglog(speeds, i_rms, "^-", color=COLORS[2], label="inertia M(q) qdd")
    axes[0].set_xlabel("speed multiplier")
    axes[0].set_ylabel("RMS joint torque (N*m)")
    axes[0].set_title("Every term scales differently with speed")
    axes[0].legend(fontsize=8)

    frac_g = np.array(g_rms) / np.array(tot_rms)
    axes[1].semilogx(speeds, frac_g, "o-", color=COLORS[0])
    axes[1].axhline(0.5, ls="--", color=COLORS[6], lw=1.2)
    axes[1].set_xlabel("speed multiplier")
    axes[1].set_ylabel("gravity RMS / total RMS")
    axes[1].set_title("How much of the job pure gravity compensation does")
    save(fig, os.path.join(OUT, "terms.png"))


# ---------------------------------------------------------------------------
# 6. Integrators
# ---------------------------------------------------------------------------
INTEGRATORS = (("explicit Euler", dyn.step_explicit),
               ("semi-implicit Euler", dyn.step_semi_implicit),
               ("RK4", dyn.step_rk4))


def exp6_energy():
    print("[6] three integrators on an undriven arm")
    model = dyn.Model(MODELS["arm2"])
    tau = np.zeros(model.n)

    # (a) How fast does the error shrink when the step shrinks?  An undriven
    # arm must keep exactly the energy it started with, so the energy error is
    # a free, exact error measure -- no reference solution needed.
    dts = [4e-3, 2e-3, 1e-3, 5e-4]
    T = 3.0
    curves = {}
    for label, fn in INTEGRATORS:
        errs = []
        for dt in dts:
            q = np.array([1.2, -0.6])
            qd = np.zeros(model.n)
            E0 = dyn.total_energy(model, q, qd)
            worst = 0.0
            for _ in range(int(T / dt)):
                q, qd, _ = fn(model, q, qd, tau, dt)
                worst = max(worst, abs(dyn.total_energy(model, q, qd) - E0))
            errs.append(worst)
        curves[label] = errs
        order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
        record("6-energy", f"{label}: worst energy error at dt = 1 ms", errs[2], "J")
        record("6-energy", f"{label}: fitted order p in error ~ dt^p", order, "")

    record("6-energy", "RK4 advantage at dt = 1 ms over explicit Euler",
           curves["explicit Euler"][2] / curves["RK4"][2], "x")

    # (b) The SHAPE of the error, not its size -- and the condition the textbook
    # claim quietly depends on.  Semi-implicit Euler is symplectic in
    # position-MOMENTUM coordinates; this integrator works in position-VELOCITY
    # coordinates, which is the same thing only while the mass matrix is
    # constant.  So run it on both: a ONE-link arm (constant M, integrable) and
    # the two-link arm (M depends on configuration, and a double pendulum is
    # chaotic besides).
    tmpdir = os.path.join(OUT, "_chain1")
    os.makedirs(tmpdir, exist_ok=True)
    one_path = os.path.join(tmpdir, "chain1.urdf")
    write_chain(one_path, 1)
    one = dyn.Model(one_path)

    dt = 1e-3
    T2 = 10.0
    shapes = {}
    for mdl, start, tag in ((one, np.array([0.9]), "1 link (constant M)"),
                            (model, np.array([0.15, -0.10]), "2 links (M varies)")):
        for label, fn in INTEGRATORS[:2]:  # RK4 is off the scale here
            q = start.copy()
            qd = np.zeros(mdl.n)
            z = np.zeros(mdl.n)
            E0 = dyn.total_energy(mdl, q, qd)
            ts, es = [], []
            for k in range(int(T2 / dt)):
                q, qd, _ = fn(mdl, q, qd, z, dt)
                if k % 20 == 0:
                    ts.append(k * dt)
                    es.append(dyn.total_energy(mdl, q, qd) - E0)
            shapes[f"{tag}: {label}"] = (ts, es)
            record("6-energy", f"{tag}, {label}: drift after 10 s", es[-1], "J")
            record("6-energy", f"{tag}, {label}: peak-to-peak wobble",
                   max(es) - min(es), "J")
    os.remove(one_path)
    os.rmdir(tmpdir)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.2))
    for k, (label, errs) in enumerate(curves.items()):
        axes[0].loglog(np.array(dts) * 1e3, errs, "o-", color=COLORS[k], label=label)
    axes[0].set_xlabel("time step (ms)")
    axes[0].set_ylabel("worst energy error over 3 s (J)")
    axes[0].set_title("Halving the step: Euler halves the error, RK4 divides it by 16")
    axes[0].legend(fontsize=8)
    for k, (label, (ts, es)) in enumerate(shapes.items()):
        axes[1].plot(ts, es, color=COLORS[k],
                     ls="-" if "1 link" in label else "--", label=label)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("total energy - E0  (J)")
    axes[1].set_yscale("symlog", linthresh=1e-4)
    axes[1].set_title("The symplectic guarantee needs a constant mass matrix")
    axes[1].legend(fontsize=6)
    save(fig, os.path.join(OUT, "energy.png"))


# ---------------------------------------------------------------------------
# 7. Injected bugs
# ---------------------------------------------------------------------------
def rnea_buggy(model, q, qd, qdd, bug):
    """RNEA with exactly one classic mistake switched on."""
    robot = model.robot
    poses = dyn.fk_all(robot, q)
    root = robot.root
    w = {root: np.zeros(3)}
    al = {root: np.zeros(3)}
    a = {root: -model.gravity}

    qi = 0
    for j in robot.ordered:
        p, c = j.parent, j.child
        R_c = poses[c][:3, :3]
        r = poses[c][:3, 3] - poses[p][:3, 3]
        axis_w = R_c @ j.axis
        w_c, al_c = w[p].copy(), al[p].copy()
        if bug == "no_centripetal":
            a_c = a[p] + np.cross(al[p], r)
        else:
            a_c = a[p] + np.cross(al[p], r) + np.cross(w[p], np.cross(w[p], r))
        if j.movable:
            v, vd = qd[qi], qdd[qi]
            if j.jtype == "prismatic":
                a_c = a_c + axis_w * vd + 2.0 * np.cross(w[p], axis_w * v)
            else:
                w_c = w_c + axis_w * v
                if bug == "no_axis_rotation":
                    al_c = al_c + axis_w * vd
                else:
                    al_c = al_c + axis_w * vd + np.cross(w[p], axis_w * v)
            qi += 1
        w[c], al[c], a[c] = w_c, al_c, a_c

    f, n = {}, {}
    for c in reversed(model.order):
        inert = model.inertials[c]
        R_c = poses[c][:3, :3]
        com_w = R_c @ inert.com
        if bug == "inertia_not_rotated":
            I_w = inert.I
        else:
            I_w = R_c @ inert.I @ R_c.T
        a_com = a[c] + np.cross(al[c], com_w) + np.cross(w[c], np.cross(w[c], com_w))
        F = inert.mass * a_com
        if bug == "no_gyroscopic":
            N = I_w @ al[c]
        else:
            N = I_w @ al[c] + np.cross(w[c], I_w @ w[c])
        f_c = F.copy()
        n_c = N + np.cross(com_w, F)
        for ch in robot.children[c]:
            rr = poses[ch][:3, 3] - poses[c][:3, 3]
            f_c = f_c + f[ch]
            if bug == "forgot_lever_arm":
                n_c = n_c + n[ch]
            else:
                n_c = n_c + n[ch] + np.cross(rr, f[ch])
        f[c], n[c] = f_c, n_c

    tau = np.zeros(model.n)
    for i, j in enumerate(robot.movable):
        axis_w = poses[j.child][:3, :3] @ j.axis
        tau[i] = axis_w @ (f[j.child] if j.jtype == "prismatic" else n[j.child])
    return tau


BUGS = ["no_centripetal", "no_axis_rotation", "inertia_not_rotated", "no_gyroscopic", "forgot_lever_arm"]


def exp7_bugs(seed=7):
    print("[7] five injected bugs, three checks")
    rng = np.random.default_rng(seed)
    model6 = dyn.Model(MODELS["arm6"])
    model2 = dyn.Model(MODELS["arm2"])

    checks = ["gravity only\n(qd = qdd = 0)", "one joint moving\n(planar arm2)", "all joints moving\n(arm6)"]
    grid = np.zeros((len(BUGS), len(checks)))
    for bi, bug in enumerate(BUGS):
        worst = np.zeros(3)
        for _ in range(60):
            q6 = rng.uniform(model6.robot.lower, model6.robot.upper)
            z6 = np.zeros(model6.n)
            worst[0] = max(worst[0], np.abs(rnea_buggy(model6, q6, z6, z6, bug) - dyn.rnea(model6, q6, z6, z6)).max())

            q2 = rng.uniform(-2, 2, 2)
            qd2 = np.array([rng.uniform(-3, 3), 0.0])
            qdd2 = np.array([rng.uniform(-4, 4), 0.0])
            worst[1] = max(worst[1], np.abs(rnea_buggy(model2, q2, qd2, qdd2, bug) - dyn.rnea(model2, q2, qd2, qdd2)).max())

            qd6 = rng.uniform(-3, 3, model6.n)
            qdd6 = rng.uniform(-5, 5, model6.n)
            worst[2] = max(worst[2], np.abs(rnea_buggy(model6, q6, qd6, qdd6, bug) - dyn.rnea(model6, q6, qd6, qdd6)).max())
        grid[bi] = worst
        for ci, cname in enumerate(["gravity-only", "arm2 one-joint", "arm6 all-joints"]):
            record("7-bugs", f"{bug} caught by {cname}", worst[ci], "N*m")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    shown = np.log10(np.maximum(grid, 1e-17))
    im = ax.imshow(shown, cmap="magma", aspect="auto", vmin=-16, vmax=2)
    ax.set_xticks(range(len(checks)))
    ax.set_xticklabels(checks, fontsize=8)
    ax.set_yticks(range(len(BUGS)))
    ax.set_yticklabels([b.replace("_", " ") for b in BUGS], fontsize=8)
    ax.grid(False)
    for i in range(len(BUGS)):
        for j in range(len(checks)):
            v = grid[i, j]
            ax.text(j, i, "silent" if v < 1e-12 else f"{v:.2g}", ha="center", va="center",
                    color="white" if shown[i, j] < -4 else "black", fontsize=8)
    ax.set_title("log10 worst torque error (N*m) -- which test catches which bug")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, os.path.join(OUT, "bugs.png"))


def main():
    t0 = time.perf_counter()
    exp1_verify()
    exp2_mass_matrix()
    exp3_passivity()
    exp4_scaling()
    exp5_terms()
    exp6_energy()
    exp7_bugs()
    with open(os.path.join(OUT, "results.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["section", "quantity", "value", "unit"], lineterminator="\n")
        wr.writeheader()
        wr.writerows(RESULTS)
    print(f"\ndone in {time.perf_counter() - t0:.1f} s -> {OUT}")


if __name__ == "__main__":
    main()
