"""Inverse kinematics by damped least squares.

Forward kinematics is a function; inverse kinematics is an equation.  There may
be no solution, one, several, or a continuum, so we do not "solve" it -- we
start somewhere and walk downhill on the pose error:

    e   = (target pose) - (current pose)            a 6-vector
    dq  = J(q)^+ e                                  which joints to turn
    q  <- q + dq                                    and repeat

The whole difficulty is the ``+``.  The plain pseudo-inverse asks for the
SMALLEST joint motion that produces exactly ``e``.  Near a singularity, "the
smallest motion that does it exactly" becomes enormous, because the tool has
almost stopped responding in one direction and the only way to force it is to
spin the joints wildly.

DAMPED least squares changes the question.  Instead of

    minimise ||dq||    subject to   J dq = e            (exact, can explode)

it minimises

    ||J dq - e||^2  +  lambda^2 ||dq||^2               (a compromise)

-- "get close to the target AND stay small", with ``lambda`` setting the
exchange rate.  The solution is ``J^T (J J^T + lambda^2 I)^-1 e``.  The
``lambda^2 I`` term is what keeps the matrix invertible when ``J J^T`` is not.

Two names you will meet for the same idea:

* **Levenberg-Marquardt** -- Kenneth Levenberg (1944) and Donald Marquardt
  (1963) each proposed adding a multiple of the identity to a least-squares
  normal-equation matrix so the step stays finite when the problem is
  ill-conditioned.  This is that trick, applied to kinematics.
* **Tikhonov regularisation** / **ridge regression** -- the same ``+ lambda^2 I``
  under different names in statistics and inverse problems.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in ("01-transform-calculator", "02-urdf-visualizer", "03-forward-kinematics-from-scratch",
             "04-jacobian-from-scratch"):
    sys.path.insert(0, os.path.join(HERE, "..", _rel))

import transforms as tf  # noqa: E402
from fk import fk_all  # noqa: E402
from jacobian import jacobian_analytic, singular_values  # noqa: E402


def damped_pinv(J, lam):
    """``J^T (J J^T + lam^2 I)^-1`` -- see the module docstring."""
    m = J.shape[0]
    return J.T @ np.linalg.solve(J @ J.T + (lam**2) * np.eye(m), np.eye(m))


def adaptive_lambda(J, lam_max=0.06, sigma_thresh=0.05):
    """Zero damping in open space, growing damping near a singularity.

    A fixed ``lambda`` is a permanent tax: it costs accuracy everywhere in order
    to buy safety in the few places that need it.  The variable-damping rule
    (Nakamura & Hanafusa 1986; Chiaverini 1994) switches the tax on only when
    the smallest singular value drops below a threshold, so ordinary motion is
    solved exactly and only the dangerous region is softened.
    """
    s_min = singular_values(J)[-1]
    if s_min >= sigma_thresh:
        return 0.0
    return lam_max * np.sqrt(1.0 - (s_min / sigma_thresh) ** 2)


def ik(
    robot,
    q0,
    T_target,
    lam=1e-2,
    adaptive=False,
    max_iters=200,
    tol_pos=1e-5,
    tol_rot=1e-5,
    link="tool0",
    max_step=None,
    clamp_limits=False,
):
    """Damped-least-squares inverse kinematics.

    Returns ``(q, info)``.  ``info`` carries the iteration count, whether the
    tolerance was met, and the largest single joint step taken -- that last
    number is what tells you whether a run was "solved" or merely "survived".
    """
    q = np.array(q0, dtype=float)
    hist = []
    biggest_step = 0.0
    for it in range(max_iters):
        T = fk_all(robot, q)[link]
        e = tf.pose_error(T, T_target)
        e_pos, e_rot = float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:]))
        hist.append((e_pos, e_rot))
        if e_pos < tol_pos and e_rot < tol_rot:
            return q, {"iters": it, "ok": True, "e_pos": e_pos, "e_rot": e_rot,
                       "hist": hist, "max_step": biggest_step}
        J = jacobian_analytic(robot, q, link)
        lam_i = adaptive_lambda(J) if adaptive else lam
        dq = damped_pinv(J, lam_i) @ e
        biggest_step = max(biggest_step, float(np.linalg.norm(dq)))
        if max_step is not None:
            nrm = np.linalg.norm(dq)
            if nrm > max_step:
                dq *= max_step / nrm
        q = q + dq
        if clamp_limits:
            q = robot.clamp(q)
        if not np.all(np.isfinite(q)):
            return q, {"iters": it, "ok": False, "e_pos": np.inf, "e_rot": np.inf,
                       "hist": hist, "max_step": biggest_step}
    T = fk_all(robot, q)[link]
    e = tf.pose_error(T, T_target)
    return q, {
        "iters": max_iters,
        "ok": False,
        "e_pos": float(np.linalg.norm(e[:3])),
        "e_rot": float(np.linalg.norm(e[3:])),
        "hist": hist,
        "max_step": biggest_step,
    }


def track_path(robot, q0, poses, dt, lam=1e-2, adaptive=False, gain=8.0, link="tool0", qd_cap=1e4):
    """Follow a Cartesian path at the VELOCITY level (resolved-rate control).

    At every tick we ask for the twist that would (a) keep up with the path and
    (b) erase whatever error has accumulated, then convert it to joint speeds
    with the damped pseudo-inverse.  This is how a real arm follows a
    trajectory; the per-tick joint speed is a number a motor either can or
    cannot deliver, which is what makes the singularity visible as a hazard
    rather than as a plot artefact.
    """
    q = np.array(q0, dtype=float)
    log = {"qd": [], "sigma_min": [], "err_pos": [], "err_rot": [], "q": []}
    for k in range(len(poses) - 1):
        T_cur = fk_all(robot, q)[link]
        e = tf.pose_error(T_cur, poses[k])
        # Feed-forward: the twist the path itself is asking for this tick.
        #
        # It MUST use the same convention as the Jacobian.  Ours is the
        # "point velocity" (hybrid) form: the linear row is the speed of the
        # tool ORIGIN, in world axes.  The 4x4 matrix logarithm would instead
        # give the SCREW twist, whose linear part is the velocity of the
        # imaginary body-fixed point currently at the world origin -- a
        # different vector entirely, and mixing the two silently leaves a
        # tracking error the size of the arm.  (Measured here at 110 mm before
        # this was fixed, on a path whose whole length is about 200 mm.)
        dp = (poses[k + 1][:3, 3] - poses[k][:3, 3]) / dt
        dw = tf.R_to_axis_angle(poses[k + 1][:3, :3] @ poses[k][:3, :3].T) / dt
        v = np.concatenate([dp, dw]) + gain * e
        J = jacobian_analytic(robot, q, link)
        lam_i = adaptive_lambda(J) if adaptive else lam
        qd = damped_pinv(J, lam_i) @ v
        n = np.linalg.norm(qd)
        if n > qd_cap:  # keep a diverging run finite so the plot is readable
            qd = qd * (qd_cap / n)
        log["qd"].append(float(np.linalg.norm(qd)))
        log["sigma_min"].append(float(singular_values(J)[-1]))
        log["err_pos"].append(float(np.linalg.norm(e[:3])))
        log["err_rot"].append(float(np.linalg.norm(e[3:])))
        log["q"].append(q.copy())
        q = q + qd * dt
    return q, {k: np.array(v) for k, v in log.items()}


def cartesian_lerp(T0, T1, n):
    """Straight line in position, shortest-arc slerp in orientation.

    This is what "move the tool from here to there" means to a user -- and it
    is deliberately NOT the path the joints would produce on their own.  That
    mismatch is exactly what makes a nearby singularity dangerous: the line the
    user drew asks for motion in a direction the arm is about to lose.
    """
    q0, q1 = tf.R_to_quat(T0[:3, :3]), tf.R_to_quat(T1[:3, :3])
    out = []
    for t in np.linspace(0.0, 1.0, n):
        R = tf.quat_to_R(tf.slerp(q0, q1, t))
        p = (1 - t) * T0[:3, 3] + t * T1[:3, 3]
        out.append(tf.T_from_Rp(R, p))
    return out
