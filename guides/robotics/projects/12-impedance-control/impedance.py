"""Cartesian impedance control: commanding a spring instead of a position.

A position controller answers "where should the tool be?" and pushes as hard as
it takes.  An impedance controller answers a different question: "how should the
tool RESIST being pushed?"  You hang a virtual spring-damper between the tool
and a moving reference point, and the arm applies exactly the force that spring
would apply -- no more.  Push it, and it yields; let go, and it comes back.

The control law is three lines:

    wrench  =  K (x_ref - x)  -  D  v            (the virtual spring-damper)
    tau     =  J(q)^T wrench                     (that wrench, as joint torques)
    tau    +=  g(q)                              (so the arm does not sag)

Why ``J^T``.  The Jacobian maps joint velocities to tool velocity, ``v = J qd``.
Power must come out the same however you measure it: ``wrench . v = tau . qd``
for every possible motion, and substituting ``v = J qd`` forces ``tau = J^T
wrench``.  So the transpose is not a trick or an approximation -- it is what
conservation of power says the answer has to be.  Nothing is inverted, so this
works perfectly well at a singularity, unlike an inverse-kinematics controller.

Why gravity compensation is a SEPARATE term and not part of the spring.  The
spring only ever produces force proportional to displacement.  Gravity pulls on
the arm whether or not the tool has moved, so leaving it out means the arm sags
until the spring's stretch happens to balance the weight -- a soft spring sags a
lot.  Adding ``g(q)`` cancels the weight so that the spring's rest length is the
place the tool actually rests.  The two terms answer different questions: g(q)
asks "what does it take to hold still?", the spring asks "what does it take to
resist being moved?"
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
for _p in ("10-inverse-dynamics-from-scratch", "01-transform-calculator",
           "04-jacobian-from-scratch"):
    _full = os.path.join(_PROJ, _p)
    if _full not in sys.path:
        sys.path.insert(0, _full)

import dynamics as dyn  # noqa: E402
import transforms as tf  # noqa: E402


def tool_state(model, q, qd, link="tool0"):
    """Tool pose, tool spatial velocity, and the Jacobian, in one call."""
    T = dyn.tool_pose(model, q, link=link)
    J = dyn.tool_jacobian(model, q, link=link)
    v = J @ qd
    return T, v, J


def impedance_torque(model, q, qd, T_ref, v_ref, Kp, Kd, Ko=None, Do=None,
                     gravity=True, null_damping=0.0, link="tool0"):
    """The virtual spring-damper, expressed as joint torques.

    ``Kp``/``Kd`` are 3-vectors (or scalars) of translational stiffness (N/m)
    and damping (N*s/m); ``Ko``/``Do`` do the same for orientation (N*m/rad and
    N*m*s/rad).  ``null_damping`` adds joint damping that lives entirely in the
    null space, so it slows internal drift of a redundant arm without changing
    the tool's behaviour at all.
    """
    T, v, J = tool_state(model, q, qd, link=link)
    e = tf.pose_error(T, T_ref)  # 6-vector: (position error, orientation error)
    dv = np.asarray(v_ref, dtype=float) - v

    Kp = np.broadcast_to(np.asarray(Kp, dtype=float), (3,))
    Kd = np.broadcast_to(np.asarray(Kd, dtype=float), (3,))
    Ko = np.zeros(3) if Ko is None else np.broadcast_to(np.asarray(Ko, dtype=float), (3,))
    Do = np.zeros(3) if Do is None else np.broadcast_to(np.asarray(Do, dtype=float), (3,))

    wrench = np.concatenate([Kp * e[:3] + Kd * dv[:3], Ko * e[3:] + Do * dv[3:]])
    tau = J.T @ wrench
    if gravity:
        tau = tau + dyn.gravity_torque(model, q)
    if null_damping > 0.0:
        # Project a plain joint-damping torque onto the directions that do NOT
        # move the tool.  N = I - J^+ J is that projector; applying damping
        # through it is free from the tool's point of view.
        Jp = np.linalg.pinv(J)
        Nproj = np.eye(model.n) - Jp @ J
        tau = tau + Nproj.T @ (-null_damping * qd)
    return tau, e, wrench


def joint_pd_torque(model, q, qd, q_ref, qd_ref, kp, kd, gravity=True):
    """The stiff-position baseline: a joint-space PD, in torque units."""
    tau = kp * (q_ref - q) + kd * (qd_ref - qd)
    if gravity:
        tau = tau + dyn.gravity_torque(model, q)
    return tau


def wall_force(p, v, wall_x, k_wall=8000.0, d_wall=40.0):
    """A stiff analytic wall at ``x = wall_x``, pushing back along -x.

    Modelled as a one-sided spring-damper (a "penalty" contact): no force until
    the tool crosses the surface, then force proportional to how far in it is.
    Writing the contact by hand rather than letting a physics engine resolve it
    keeps the experiment about the CONTROLLER -- the wall stiffness is a number
    we set, not a solver setting we would have to reverse-engineer.
    """
    depth = p[0] - wall_x
    if depth <= 0.0:
        return np.zeros(6)
    f = -(k_wall * depth + d_wall * max(v[0], 0.0))
    return np.array([f, 0.0, 0.0, 0.0, 0.0, 0.0])


def simulate(model, q0, controller, T=2.0, dt=1e-3, ext_fn=None, link="tool0"):
    """Closed loop.  ``controller(t, q, qd) -> tau``; ``ext_fn(t, p, v) -> wrench``."""
    q = np.asarray(q0, dtype=float).copy()
    qd = np.zeros(model.n)
    steps = int(T / dt)
    P = np.zeros((steps, 3))
    Q = np.zeros((steps, model.n))
    TAU = np.zeros((steps, model.n))
    F = np.zeros((steps, 6))
    for k in range(steps):
        t = k * dt
        T_now, v, _ = tool_state(model, q, qd, link=link)
        p = T_now[:3, 3]
        wrench = np.zeros(6) if ext_fn is None else ext_fn(t, p, v)
        tau = controller(t, q, qd)
        tau = np.clip(tau, -model.tau_max, model.tau_max)
        P[k], Q[k], TAU[k], F[k] = p, q, tau, wrench
        f_ext = None if not np.any(wrench) else {link: wrench}
        qdd = dyn.forward_dynamics(model, q, qd, tau, f_ext=f_ext)
        qd = qd + dt * qdd
        q = q + dt * qd
        if not np.all(np.isfinite(q)) or np.abs(qd).max() > 200:
            P[k + 1:], Q[k + 1:], TAU[k + 1:], F[k + 1:] = P[k], Q[k], TAU[k], F[k]
            return np.arange(steps) * dt, P, Q, TAU, F, False
    return np.arange(steps) * dt, P, Q, TAU, F, True
