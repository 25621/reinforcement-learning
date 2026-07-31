"""Null-space control: doing a second job with the freedom the first job leaves.

A 7-joint arm asked for a 6-number tool pose has one number of freedom left
over.  Concretely: hold the hand perfectly still and the elbow can still swing
around the line from shoulder to wrist, like a hinge.  The set of joint
velocities that move the elbow WITHOUT moving the hand is the NULL SPACE of the
Jacobian -- "null" because ``J q_dot = 0``, the tool motion they produce is
nothing.

The control law is one line:

    q_dot  =  J^+ v          +      (I - J^+ J) q_dot_0
              ^^^^^^^                ^^^^^^^^^^^^^^^^^^
              do the job             do whatever else you like,
              (track the tool)       filtered so it cannot disturb the job

``N = I - J^+ J`` is a PROJECTOR: hand it any joint-velocity wish ``q_dot_0``
and it deletes exactly the part that would have moved the tool, keeping the
rest.  Applying it twice changes nothing (``N N = N``), which is what the word
projector means -- like a shadow on the floor, projecting again does nothing.

> **"If the extra motion cannot move the tool, what is it for?"**  The tool
> pose is not the only thing that matters.  The same hand position can be
> reached with the elbow tucked against a joint limit and one twitch away from
> a fault, or with the elbow relaxed in mid-range.  The primary task cannot
> tell those apart -- it is blind to everything except the tool.  The
> null-space term is where "and also stay in a good posture / away from
> limits / away from singularities" gets to speak, at zero cost to the tool.

Three secondary tasks are implemented below.  They are genuinely in conflict
with each other, which is the point of measuring them.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in ("01-transform-calculator", "02-urdf-visualizer", "03-forward-kinematics-from-scratch",
             "04-jacobian-from-scratch", "05-damped-least-squares-ik"):
    sys.path.insert(0, os.path.join(HERE, "..", _rel))

import transforms as tf  # noqa: E402
from fk import fk_all  # noqa: E402
from ik import damped_pinv  # noqa: E402
from jacobian import jacobian_analytic, manipulability  # noqa: E402


def null_projector(J, lam=0.0):
    """``I - J^+ J``.  With ``lam = 0`` this is an exact projector."""
    return np.eye(J.shape[1]) - damped_pinv(J, lam) @ J


# ---------------------------------------------------------------------------
# secondary tasks -- each returns a DESIRED joint velocity, before projection
# ---------------------------------------------------------------------------
def secondary_posture(robot, q, q_home, k=2.0):
    """Pull every joint back toward a chosen comfortable posture."""
    return k * (q_home - q)


def secondary_limits(robot, q, k=6.0):
    """Push away from joint limits.

    The cost is ``H(q) = mean of ((q - middle) / half-range)^2`` -- zero in the
    middle of every joint's travel, 1 at a limit.  We descend its gradient.
    Squaring is what makes the push GENTLE in the middle and firm near the end,
    which is the behaviour you want: no reason to fight for the exact centre.
    """
    mid = 0.5 * (robot.lower + robot.upper)
    half = 0.5 * (robot.upper - robot.lower)
    grad = 2.0 * (q - mid) / (half**2) / robot.n
    return -k * grad


def secondary_manipulability(robot, q, k=6.0, h=1e-4, link="tool0"):
    """Climb the manipulability measure, i.e. walk AWAY from singularities.

    ``sqrt(det(J J^T))`` has no simple closed-form gradient for a general
    robot, so we take it numerically: n extra Jacobians per control tick.  That
    is affordable here and is what most implementations actually do.
    """
    grad = np.zeros(robot.n)
    for i in range(robot.n):
        e = np.zeros(robot.n)
        e[i] = h
        wp = manipulability(jacobian_analytic(robot, q + e, link))
        wm = manipulability(jacobian_analytic(robot, q - e, link))
        grad[i] = (wp - wm) / (2 * h)
    return k * grad


# ---------------------------------------------------------------------------
# the controller
# ---------------------------------------------------------------------------
def track(robot, q0, poses, dt, secondary=None, lam=1e-2, gain=8.0, link="tool0",
          proj_lam=0.0, clamp=True):
    """Track a Cartesian path, optionally with a null-space secondary task.

    ``proj_lam`` is the damping used to BUILD the projector, which need not be
    the damping used to solve the primary task.  Keeping it at 0 makes the
    projection exact, so any tool disturbance we then measure is real physics
    and not our own approximation leaking in.
    """
    q = np.array(q0, dtype=float)
    log = {k: [] for k in ("err_pos", "err_rot", "q", "manip", "margin", "home_dist",
                           "disturbance", "qd_norm")}
    for k in range(len(poses) - 1):
        T_cur = fk_all(robot, q)[link]
        e = tf.pose_error(T_cur, poses[k])
        dp = (poses[k + 1][:3, 3] - poses[k][:3, 3]) / dt
        dw = tf.R_to_axis_angle(poses[k + 1][:3, :3] @ poses[k][:3, :3].T) / dt
        v = np.concatenate([dp, dw]) + gain * e

        J = jacobian_analytic(robot, q, link)
        qd = damped_pinv(J, lam) @ v
        disturbance = 0.0
        if secondary is not None:
            qd0 = secondary(robot, q)
            qd_null = null_projector(J, proj_lam) @ qd0
            # The whole promise of the method, measured every single tick:
            # how much tool motion did the secondary task actually cause?
            disturbance = float(np.linalg.norm(J @ qd_null))
            qd = qd + qd_null

        log["err_pos"].append(float(np.linalg.norm(e[:3])))
        log["err_rot"].append(float(np.linalg.norm(e[3:])))
        log["q"].append(q.copy())
        log["manip"].append(manipulability(J))
        log["margin"].append(float(robot.limit_margin(q).min()))
        log["disturbance"].append(disturbance)
        log["qd_norm"].append(float(np.linalg.norm(qd)))

        q = q + qd * dt
        if clamp:
            q = robot.clamp(q)
    return q, {k: np.array(v) for k, v in log.items()}


def circle_path(center, radius, n, laps=1, R_tool=None, plane=("y", "z")):
    """A closed Cartesian loop with a FIXED tool orientation."""
    axes = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0]), "z": np.array([0, 0, 1.0])}
    u, w = axes[plane[0]], axes[plane[1]]
    if R_tool is None:
        R_tool = np.eye(3)
    out = []
    for th in np.linspace(0.0, 2 * np.pi * laps, n):
        p = np.asarray(center) + radius * (np.cos(th) * u + np.sin(th) * w)
        out.append(tf.T_from_Rp(R_tool, p))
    return out
