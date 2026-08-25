"""Rigid-body dynamics from scratch: RNEA, the mass matrix, and simulation.

This is the library projects 11, 12 and 15 import.  Everything here is built on
two ideas you already have from Phase 1 -- the kinematic tree from project 02
and the world-frame link poses from project 03 -- plus Newton's two laws
written once per link.

The one equation this file evaluates, in every direction:

    M(q) qdd  +  C(q, qd) qd  +  g(q)  =  tau  +  J^T F_ext

  * INVERSE dynamics  -- "what torque produces this motion?"  Left side, given
    (q, qd, qdd).  ``rnea`` computes the whole left side in one sweep, in time
    proportional to the number of joints.
  * FORWARD dynamics  -- "what motion does this torque produce?"  Solve for
    qdd.  ``forward_dynamics`` does it by building M and inverting.

Everything is expressed in the WORLD frame.  Textbook RNEA usually works in
per-link frames, which saves a few multiplies and costs a lot of readability;
at six joints the saving is invisible and the readability is not.

A note on the gravity trick.  Nothing here ever adds a gravity force.  Instead
the base link is given a fictitious upward acceleration of +9.81 m/s^2, and the
gravity torques fall out of the same recursion that handles everything else.
The justification is Einstein's equivalence principle in its most mundane form:
a robot standing still on Earth and a robot accelerating upward at 9.81 m/s^2
in deep space feel identical forces.  One less special case in the code.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECTS = os.path.dirname(_HERE)
for _p in ("01-transform-calculator", "02-urdf-visualizer",
           "03-forward-kinematics-from-scratch", "04-jacobian-from-scratch"):
    _full = os.path.join(_PROJECTS, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

import transforms as tf  # noqa: E402
from urdf import load_urdf  # noqa: E402
from fk import fk_all, joint_transform  # noqa: E402
from jacobian import jacobian_analytic  # noqa: E402
from inertial import load_inertials, load_effort_limits, load_velocity_limits  # noqa: E402

GRAVITY = np.array([0.0, 0.0, -9.81])


class Model:
    """A robot with mass: the parsed tree plus the mass properties."""

    def __init__(self, urdf_path, gravity=GRAVITY):
        self.path = urdf_path
        self.robot = load_urdf(urdf_path)
        self.inertials = load_inertials(urdf_path)
        self.tau_max = np.array(
            [load_effort_limits(urdf_path)[j.name] for j in self.robot.movable]
        )
        self.qd_max = np.array(
            [load_velocity_limits(urdf_path)[j.name] for j in self.robot.movable]
        )
        self.gravity = np.asarray(gravity, dtype=float)
        self.n = self.robot.n
        self.names = self.robot.joint_names

        # Which link hangs off which joint, and the parent of every link.  The
        # backward sweep needs to walk this the other way round.
        self.parent = {j.child: j.parent for j in self.robot.joints}
        self.order = [j.child for j in self.robot.ordered]  # parents first
        self.total_mass = sum(i.mass for i in self.inertials.values())

    # -- convenience -------------------------------------------------------
    def scaled(self, mass_factor):
        """A copy whose every link mass and inertia is multiplied by a factor.

        Used by project 11 to ask what happens when the controller's model of
        the arm disagrees with the arm.
        """
        import copy

        m = copy.copy(self)
        m.inertials = {
            k: type(v)(v.mass * mass_factor, v.com.copy(), v.I * mass_factor)
            for k, v in self.inertials.items()
        }
        m.total_mass = self.total_mass * mass_factor
        return m


# ---------------------------------------------------------------------------
# Recursive Newton-Euler
# ---------------------------------------------------------------------------
def rnea(model, q, qd, qdd, f_ext=None, gravity=True):
    """Inverse dynamics: the joint torques that produce ``qdd`` at ``(q, qd)``.

    ``f_ext`` is an optional ``{link_name: 6-vector (force, torque)}`` of
    wrenches the WORLD applies TO the robot, expressed in world axes and taken
    about that link's frame origin.

    Two sweeps:

      OUTWARD (base to tip)  -- kinematics.  Each link inherits its parent's
      angular velocity and acceleration and adds its own joint's contribution.
      By the end we know how every centre of mass is accelerating.

      INWARD (tip to base)   -- kinetics.  Newton (F = m a) gives the force each
      link needs; Euler (N = I alpha + omega x I omega) gives the torque.  Walking
      back toward the base, each joint must supply whatever its whole subtree
      needs.  Projecting that onto the joint axis is the answer.

    Cost is one pass out and one pass back: O(n), which is the whole point of
    the algorithm.  Writing M, C and g separately and multiplying would cost
    O(n^2) or worse for the same number.
    """
    robot = model.robot
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    qdd = np.asarray(qdd, dtype=float)

    poses = fk_all(robot, q)
    root = robot.root

    # --- outward sweep -----------------------------------------------------
    w = {root: np.zeros(3)}  # angular velocity of each link, world axes
    al = {root: np.zeros(3)}  # angular acceleration
    # Linear acceleration of each link's FRAME ORIGIN.  Starting the base at
    # -g is the gravity trick described in the module docstring.
    a = {root: -model.gravity if gravity else np.zeros(3)}

    qi = 0
    for j in robot.ordered:
        p, c = j.parent, j.child
        R_c = poses[c][:3, :3]
        r = poses[c][:3, 3] - poses[p][:3, 3]  # parent origin -> child origin
        axis_w = R_c @ j.axis  # joint axis in world axes

        # Every child is rigidly carried by its parent, so it inherits the
        # parent's rigid-body acceleration first...
        w_c = w[p].copy()
        al_c = al[p].copy()
        a_c = a[p] + np.cross(al[p], r) + np.cross(w[p], np.cross(w[p], r))

        # ...and then the joint adds its own motion on top.
        if j.movable:
            v, vd = qd[qi], qdd[qi]
            if j.jtype == "prismatic":
                # Sliding: no extra rotation, but a linear term plus the
                # Coriolis term 2 w x v_rel that appears whenever something
                # slides inside a rotating frame.
                a_c = a_c + axis_w * vd + 2.0 * np.cross(w[p], axis_w * v)
            else:
                w_c = w_c + axis_w * v
                # The alpha term below is NOT just axis*vd: the axis itself is
                # being carried around by the parent, and a rotating axis with
                # a non-zero rate has an acceleration of its own.
                al_c = al_c + axis_w * vd + np.cross(w[p], axis_w * v)
            qi += 1

        w[c], al[c], a[c] = w_c, al_c, a_c

    # --- inward sweep ------------------------------------------------------
    f = {}  # net force each link must receive from its parent, world axes
    n = {}  # net moment about the link's own frame origin
    for c in reversed(model.order):
        inert = model.inertials[c]
        R_c = poses[c][:3, :3]
        com_w = R_c @ inert.com  # link origin -> centre of mass, world axes
        I_w = R_c @ inert.I @ R_c.T  # inertia tensor, rotated into world axes

        # Acceleration of the centre of mass, not of the frame origin.
        a_com = a[c] + np.cross(al[c], com_w) + np.cross(w[c], np.cross(w[c], com_w))

        F = inert.mass * a_com  # Newton
        N = I_w @ al[c] + np.cross(w[c], I_w @ w[c])  # Euler

        f_c = F.copy()
        n_c = N + np.cross(com_w, F)  # move F from the CoM to the frame origin

        for ch in robot.children[c]:
            r = poses[ch][:3, 3] - poses[c][:3, 3]
            f_c = f_c + f[ch]
            n_c = n_c + n[ch] + np.cross(r, f[ch])

        if f_ext is not None and c in f_ext:
            wr = np.asarray(f_ext[c], dtype=float)
            f_c = f_c - wr[:3]
            n_c = n_c - wr[3:]

        f[c], n[c] = f_c, n_c

    # --- project onto the joint axes --------------------------------------
    tau = np.zeros(model.n)
    for i, j in enumerate(robot.movable):
        axis_w = poses[j.child][:3, :3] @ j.axis
        # A revolute joint can only supply torque about its axis; a prismatic
        # joint can only supply force along it.  Everything else is carried by
        # the bearings, which is why those components simply do not appear.
        tau[i] = axis_w @ (f[j.child] if j.jtype == "prismatic" else n[j.child])
    return tau


# ---------------------------------------------------------------------------
# The three named terms, each one RNEA with something switched off
# ---------------------------------------------------------------------------
def gravity_torque(model, q):
    """g(q): hold still against gravity.  Zero velocity, zero acceleration."""
    z = np.zeros(model.n)
    return rnea(model, q, z, z)


def coriolis_torque(model, q, qd):
    """C(q, qd) qd: the velocity-dependent term, gravity removed."""
    return rnea(model, q, qd, np.zeros(model.n), gravity=False)


def mass_matrix(model, q):
    """M(q), one column at a time.

    Column i is the torque needed to accelerate joint i at 1 rad/s^2 with every
    other joint held at zero acceleration, no velocity and no gravity -- which
    is exactly the definition of the i-th column of M.  So n RNEA calls give
    the whole matrix.  (The dedicated algorithm for this, CRBA, is faster; this
    version needs no new code at all, and at n = 6 the difference is noise.)
    """
    z = np.zeros(model.n)
    M = np.zeros((model.n, model.n))
    for i in range(model.n):
        e = np.zeros(model.n)
        e[i] = 1.0
        M[:, i] = rnea(model, q, z, e, gravity=False)
    # M is symmetric in exact arithmetic; symmetrising removes the last bit of
    # round-off asymmetry so downstream solvers see a clean matrix.
    return 0.5 * (M + M.T)


def coriolis_matrix(model, q, qd, h=1e-6):
    """The C MATRIX via Christoffel symbols of the first kind.

    ``coriolis_torque`` gives the product C qd, which is all a controller ever
    needs.  The matrix itself is only needed to check the passivity identity in
    project 10's experiment 4, and it is not unique -- many matrices C satisfy
    C qd = the same vector.  This is the particular choice that makes
    ``Mdot - 2C`` skew-symmetric:

        c_ijk = 1/2 ( dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i )
        C_ij  = sum_k c_ijk qd_k

    dM/dq is taken by central finite differences, so this is slow and only
    accurate to about 1e-9.  That is fine for a diagnostic.
    """
    n = model.n
    dM = np.zeros((n, n, n))  # dM[:, :, k] = dM/dq_k
    for k in range(n):
        dq = np.zeros(n)
        dq[k] = h
        dM[:, :, k] = (mass_matrix(model, q + dq) - mass_matrix(model, q - dq)) / (2 * h)

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = 0.5 * np.sum(
                (dM[i, j, :] + dM[i, :, j] - dM[j, :, i]) * qd
            )
    return C


def mass_matrix_dot(model, q, qd, h=1e-6):
    """dM/dt along the current motion, by central differences on q."""
    return (mass_matrix(model, q + h * qd) - mass_matrix(model, q - h * qd)) / (2 * h)


# ---------------------------------------------------------------------------
# Forward dynamics and integration
# ---------------------------------------------------------------------------
def forward_dynamics(model, q, qd, tau, f_ext=None):
    """qdd = M^-1 (tau - C qd - g + external terms).  The simulator's core.

    ``rnea(q, qd, 0)`` already bundles C qd + g + the external-wrench torques
    into a single "bias" vector, so only one linear solve is left.  M is
    symmetric positive-definite for any real robot, so the solve always
    succeeds -- a failure here means the model is broken, not the state.
    """
    bias = rnea(model, q, qd, np.zeros(model.n), f_ext=f_ext)
    M = mass_matrix(model, q)
    return np.linalg.solve(M, np.asarray(tau, dtype=float) - bias)


def step_semi_implicit(model, q, qd, tau, dt, f_ext=None):
    """One integration step: velocity first, then position with the NEW velocity.

    "Semi-implicit" (or symplectic) Euler differs from plain Euler in exactly
    one character -- it uses ``qd_next`` instead of ``qd`` in the second line --
    and that one character is what stops the energy of an undriven system from
    growing without bound.  Project 10 measures the difference.
    """
    qdd = forward_dynamics(model, q, qd, tau, f_ext=f_ext)
    qd_next = qd + dt * qdd
    q_next = q + dt * qd_next
    return q_next, qd_next, qdd


def step_explicit(model, q, qd, tau, dt, f_ext=None):
    """Plain forward Euler, kept only so project 10 can show it drifting."""
    qdd = forward_dynamics(model, q, qd, tau, f_ext=f_ext)
    return q + dt * qd, qd + dt * qdd, qdd


def step_rk4(model, q, qd, tau, dt, f_ext=None):
    """Classical Runge-Kutta on the second-order system, torque held constant."""

    def deriv(state):
        qq, vv = state[: model.n], state[model.n:]
        return np.concatenate([vv, forward_dynamics(model, qq, vv, tau, f_ext=f_ext)])

    s = np.concatenate([q, qd])
    k1 = deriv(s)
    k2 = deriv(s + 0.5 * dt * k1)
    k3 = deriv(s + 0.5 * dt * k2)
    k4 = deriv(s + dt * k3)
    s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return s[: model.n], s[model.n:], k1[model.n:]


# ---------------------------------------------------------------------------
# Energy -- the cheapest correctness check there is
# ---------------------------------------------------------------------------
def kinetic_energy(model, q, qd):
    """T = 1/2 qd^T M(q) qd."""
    return 0.5 * float(qd @ mass_matrix(model, q) @ qd)


def potential_energy(model, q):
    """U = -sum_links m_i g . p_i(com).  Zero at the world origin."""
    poses = fk_all(model.robot, q)
    U = 0.0
    for name, inert in model.inertials.items():
        if inert.mass == 0.0:
            continue
        p = poses[name][:3, 3] + poses[name][:3, :3] @ inert.com
        U -= inert.mass * float(model.gravity @ p)
    return U


def total_energy(model, q, qd):
    return kinetic_energy(model, q, qd) + potential_energy(model, q)


# ---------------------------------------------------------------------------
# Kinematics re-exported, so downstream projects import one module
# ---------------------------------------------------------------------------
def tool_pose(model, q, link="tool0"):
    return fk_all(model.robot, q)[link]


def tool_jacobian(model, q, link="tool0"):
    return jacobian_analytic(model.robot, q, link=link)


__all__ = [
    "GRAVITY", "Model", "rnea", "gravity_torque", "coriolis_torque",
    "mass_matrix", "coriolis_matrix", "mass_matrix_dot", "forward_dynamics",
    "step_semi_implicit", "step_explicit", "step_rk4", "kinetic_energy",
    "potential_energy", "total_energy", "tool_pose", "tool_jacobian",
    "load_urdf", "fk_all", "joint_transform", "tf",
]
