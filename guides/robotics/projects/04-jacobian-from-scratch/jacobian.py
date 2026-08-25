"""The geometric Jacobian: how joint speeds turn into end-effector motion.

If forward kinematics answers "where is the hand?", the Jacobian answers "if I
turn the joints at these speeds, how does the hand move?".  It is the 6-by-n
matrix ``J(q)`` in

    [ v ]                       v = linear  velocity of the tool (m/s)
    [   ]  =  J(q) @ q_dot      w = angular velocity of the tool (rad/s)
    [ w ]                       q_dot = joint speeds (rad/s or m/s)

Two ways to build it, and this project checks them against each other:

* ANALYTIC (the geometric construction).  For a revolute joint, turning about a
  world-frame axis ``z`` that passes through the world-frame point ``p_joint``
  sweeps the tool at ``z x (p_tool - p_joint)`` and spins it at ``z``.  That is
  the whole derivation -- it is the cross-product formula for a rotating rigid
  body, one column per joint.  Cost: one forward-kinematics sweep.

* FINITE DIFFERENCE.  Nudge one joint, see how far the tool moved, divide.
  Needs no derivation at all, costs 2n sweeps, and is limited to about six
  correct digits no matter how carefully you do it (project 04 measures why).

The finite-difference version is not a competitor for production code.  It is
the ORACLE you check the analytic version against, because it is derived from
nothing but the forward kinematics you already trust.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "01-transform-calculator"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "02-urdf-visualizer"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "03-forward-kinematics-from-scratch"))

import transforms as tf  # noqa: E402
from fk import fk_all  # noqa: E402


def ancestors(robot, link):
    """The set of links between ``link`` and the root, inclusive.

    Only joints on this path can move ``link``.  On a branching robot -- a
    camera bracket, a second arm, a gripper finger -- forgetting this check
    fills the Jacobian with columns that confidently predict motion that will
    never happen.
    """
    out = set()
    cur = link
    while cur in robot.child_to_joint:
        out.add(cur)
        cur = robot.child_to_joint[cur].parent
    out.add(cur)
    return out


def jacobian_analytic(robot, q, link="tool0"):
    """The 6 x n geometric Jacobian in the WORLD frame, at ``link``'s origin.

    Rows 0-2 are linear velocity (metres per second), rows 3-5 are angular
    velocity (radians per second).  Those are different physical units, which
    matters more than it sounds -- see the unit-scale experiment in ``run.py``.
    """
    poses = fk_all(robot, q)
    p_tool = poses[link][:3, 3]
    anc = ancestors(robot, link)

    J = np.zeros((6, robot.n))
    for i, j in enumerate(robot.movable):
        if j.child not in anc:
            continue  # this joint is on a different branch: it cannot move the tool
        T = poses[j.child]
        axis_w = T[:3, :3] @ j.axis  # the joint axis, expressed in world
        if j.jtype == "prismatic":
            J[:3, i] = axis_w  # sliding: pure translation, no spin
        else:
            J[:3, i] = np.cross(axis_w, p_tool - T[:3, 3])
            J[3:, i] = axis_w
    return J


def jacobian_fd(robot, q, link="tool0", h=1e-6, central=True):
    """The same matrix, estimated by nudging each joint.

    The angular column is NOT the difference of two roll-pitch-yaw triples --
    that would inherit every wrap-around and gimbal problem of Euler angles.
    It is the axis-angle of the small rotation between the two poses, divided
    by the step, which is exactly the definition of angular velocity.
    """
    J = np.zeros((6, robot.n))
    for i in range(robot.n):
        e = np.zeros(robot.n)
        e[i] = h
        if central:
            Tp, Tm, denom = fk_all(robot, q + e)[link], fk_all(robot, q - e)[link], 2.0 * h
        else:
            Tp, Tm, denom = fk_all(robot, q + e)[link], fk_all(robot, q)[link], h
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / denom
        J[3:, i] = tf.R_to_axis_angle(Tp[:3, :3] @ Tm[:3, :3].T) / denom
    return J


# ---------------------------------------------------------------------------
# things you read off a Jacobian
# ---------------------------------------------------------------------------
def singular_values(J):
    return np.linalg.svd(J, compute_uv=False)


def manipulability(J):
    """Yoshikawa's manipulability measure, ``sqrt(det(J J^T))``.

    Geometrically: joint speeds inside a unit ball map to an ellipsoid of tool
    velocities, and this number is proportional to that ellipsoid's volume.
    Big = the tool can move freely in every direction.  Zero = a SINGULARITY:
    the ellipsoid has collapsed to a pancake and some direction of motion has
    become unreachable no matter how fast the joints turn.
    """
    return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))


def condition_number(J):
    """Ratio of the largest to the smallest singular value.

    How lopsided the velocity ellipsoid is.  1 means perfectly round (equally
    easy to move in every direction); huge means one direction is nearly free
    and another nearly impossible.
    """
    s = singular_values(J)
    return float(s[0] / max(s[-1], 1e-300))


def damped_pinv(J, lam):
    """Damped (Levenberg-Marquardt) pseudo-inverse: ``J^T (J J^T + lam^2 I)^-1``.

    With ``lam = 0`` this is the ordinary right pseudo-inverse, which explodes
    at a singularity.  ``lam > 0`` bounds the output at the cost of a small,
    deliberate tracking error.  Projects 05 and 06 live on this trade-off.
    """
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(m), np.eye(m))
