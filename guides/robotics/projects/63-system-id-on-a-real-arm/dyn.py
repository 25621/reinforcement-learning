"""A 2-link arm in a VERTICAL plane, its simulator, and its regressor.

Two design choices are load-bearing.

**Vertical, not flat.**  Phase 8's arm (``54/arm.py``) lies flat on a table so
that gravity does not drown out inertia and damping.  Here we want the exact
opposite: gravity is the term that makes *mass* visible.  Without it, a link's
mass only shows up when the arm accelerates hard, and a slow, safe excitation
would tell you nothing about it.  Stand the arm up and every pose is a
measurement.

**The equations are written out by hand** so that the regressor -- the matrix
that says "torque is a linear function of these ten numbers" -- can be written
out too.  ``verify_mujoco.py`` checks both against MuJoCo.

The ten numbers are the arm's BASE PARAMETERS.  They are not the masses and
lengths you would measure with a scale and a ruler; they are the specific
*combinations* of those that torque actually depends on.  See ``PHYS_NAMES``
and project README section 5 for why that distinction is the whole game.
"""

import numpy as np

G = 9.81
EPS_V = 0.01        # rad/s: below this, "sign of the velocity" is smoothed

# ---------------------------------------------------------------------------
# the real robot's true, secret parameters
# ---------------------------------------------------------------------------
L1, L2 = 0.30, 0.25                  # link lengths (known exactly: a ruler works)

TRUE_PHYS = dict(
    m1=1.20, lc1=0.16, I1=0.0120,    # link 1: mass, COM distance, inertia about COM
    m2=0.80, lc2=0.12, I2=0.0060,    # link 2
    Ia1=0.015, Ia2=0.010,            # reflected rotor inertia (armature)
    d1=0.30, d2=0.20,                # viscous damping, N m s / rad
    f1=0.45, f2=0.28,                # Coulomb friction, N m
)

# what the CAD / URDF thinks: uniform rods, frictionless, no rotors
CAD_PHYS = dict(
    m1=1.00, lc1=L1 / 2, I1=1.00 * L1 ** 2 / 12,
    m2=0.70, lc2=L2 / 2, I2=0.70 * L2 ** 2 / 12,
    Ia1=0.0, Ia2=0.0, d1=0.0, d2=0.0, f1=0.0, f2=0.0,
)

THETA_NAMES = ["th1 (shoulder inertia)", "th2 (m2*L1*lc2)", "th3 (elbow inertia)",
               "Ia2 (elbow rotor)", "th4 (shoulder gravity)", "th5 (elbow gravity)",
               "d1", "d2", "f1", "f2"]

PHYS_NAMES = ["m1", "lc1", "I1", "m2", "lc2", "I2", "Ia1", "Ia2",
              "d1", "d2", "f1", "f2"]


def to_theta(p):
    """Collapse 12 physical parameters into the 10 combinations torque sees."""
    th1 = p["I1"] + p["m1"] * p["lc1"] ** 2 + p["m2"] * L1 ** 2 + p["Ia1"]
    th2 = p["m2"] * L1 * p["lc2"]
    th3 = p["I2"] + p["m2"] * p["lc2"] ** 2
    th4 = p["m1"] * p["lc1"] + p["m2"] * L1
    th5 = p["m2"] * p["lc2"]
    return np.array([th1, th2, th3, p["Ia2"], th4, th5,
                     p["d1"], p["d2"], p["f1"], p["f2"]])


TRUE_THETA = to_theta(TRUE_PHYS)
CAD_THETA = to_theta(CAD_PHYS)


# ---------------------------------------------------------------------------
# the regressor: tau = Y(q, qd, qdd) @ theta, exactly
# ---------------------------------------------------------------------------
def regressor(q, qd, qdd):
    """Build Y for a batch of samples.  Shape (N, 2, 10).

    Every column is "the torque you would get if this one parameter were 1 and
    all the others were 0".  Because the dynamics are linear in these ten
    numbers, that is all there is to it -- no derivatives, no optimisation.
    """
    q, qd, qdd = np.atleast_2d(q), np.atleast_2d(qd), np.atleast_2d(qdd)
    q1, q2 = q[:, 0], q[:, 1]
    d1_, d2_ = qd[:, 0], qd[:, 1]
    a1, a2 = qdd[:, 0], qdd[:, 1]
    c2, s2 = np.cos(q2), np.sin(q2)
    c1, c12 = np.cos(q1), np.cos(q1 + q2)
    N = len(q1)
    Y = np.zeros((N, 2, 10))

    # --- row 1: shoulder torque -------------------------------------------
    Y[:, 0, 0] = a1                                   # th1
    Y[:, 0, 1] = 2 * c2 * a1 + c2 * a2 - s2 * (2 * d1_ * d2_ + d2_ ** 2)   # th2
    Y[:, 0, 2] = a1 + a2                              # th3
    Y[:, 0, 4] = G * c1                               # th4
    Y[:, 0, 5] = G * c12                              # th5
    Y[:, 0, 6] = d1_                                  # d1
    Y[:, 0, 8] = np.tanh(d1_ / EPS_V)                 # f1

    # --- row 2: elbow torque ----------------------------------------------
    Y[:, 1, 1] = c2 * a1 + s2 * d1_ ** 2              # th2
    Y[:, 1, 2] = a1 + a2                              # th3
    Y[:, 1, 3] = a2                                   # Ia2
    Y[:, 1, 5] = G * c12                              # th5
    Y[:, 1, 7] = d2_                                  # d2
    Y[:, 1, 9] = np.tanh(d2_ / EPS_V)                 # f2
    return Y


def phys_regressor(q, qd, qdd):
    """The SAME torque, written against the 12 physical parameters instead.

    This is what a beginner naturally writes: one column per thing you can
    point at on the robot.  It is also rank deficient, and experiment 5 is
    about what that does to you.
    """
    Y = regressor(q, qd, qdd)          # (N, 2, 10) against theta
    # d(theta)/d(phys): theta = J_phys @ ... is not linear in phys, so we
    # linearise around the TRUE parameters -- which is the most generous
    # possible treatment, and it still fails.
    p = dict(TRUE_PHYS)
    J = np.zeros((10, 12))
    h = 1e-6
    base = to_theta(p)
    for k, name in enumerate(PHYS_NAMES):
        pk = dict(p); pk[name] = p[name] + h
        J[:, k] = (to_theta(pk) - base) / h
    return np.einsum("nij,jk->nik", Y, J)


def torque(q, qd, qdd, theta=None):
    theta = TRUE_THETA if theta is None else theta
    return regressor(q, qd, qdd) @ theta


# ---------------------------------------------------------------------------
# forward dynamics and simulation of the "real" robot
# ---------------------------------------------------------------------------
def _M_C_G(q, qd, th):
    q2 = q[1]
    c2, s2 = np.cos(q2), np.sin(q2)
    M = np.array([[th[0] + th[2] + 2 * th[1] * c2, th[2] + th[1] * c2],
                  [th[2] + th[1] * c2,             th[2] + th[3]]])
    C = np.array([-th[1] * s2 * (2 * qd[0] * qd[1] + qd[1] ** 2),
                  th[1] * s2 * qd[0] ** 2])
    Gv = np.array([th[4] * G * np.cos(q[0]) + th[5] * G * np.cos(q[0] + q[1]),
                   th[5] * G * np.cos(q[0] + q[1])])
    return M, C, Gv


def forward(q, qd, tau, th):
    M, C, Gv = _M_C_G(q, qd, th)
    fric = np.array([th[6] * qd[0] + th[8] * np.tanh(qd[0] / EPS_V),
                     th[7] * qd[1] + th[9] * np.tanh(qd[1] / EPS_V)])
    return np.linalg.solve(M, tau - C - Gv - fric)


DT = 1e-3                 # both the physics step and the sampling period


def simulate(ctrl, T, q0=(0.4, -0.6), qd0=(0.0, 0.0), th=None, dt=DT):
    """Run the real robot for T seconds under ``ctrl(t, q, qd) -> tau``.

    Returns the *true* q, qd, qdd and the applied torque, sampled every dt.
    Measurement corruption is added later, on purpose: experiment 2 is about
    what the corruption costs, so we need the clean version to compare to.
    """
    th = TRUE_THETA if th is None else th
    n = int(T / dt)
    q = np.array(q0, dtype=float)
    qd = np.array(qd0, dtype=float)
    Q, QD, QDD, TAU = (np.zeros((n, 2)) for _ in range(4))
    for k in range(n):
        t = k * dt
        tau = np.clip(ctrl(t, q, qd), -60, 60)
        qdd = forward(q, qd, tau, th)
        Q[k], QD[k], QDD[k], TAU[k] = q, qd, qdd, tau
        qd = qd + dt * qdd            # semi-implicit Euler
        q = q + dt * qd
    return Q, QD, QDD, TAU
