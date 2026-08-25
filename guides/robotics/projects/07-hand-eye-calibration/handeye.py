"""Hand-eye calibration: finding the camera bolted to the robot's wrist.

You can measure a robot's joint angles to a thousandth of a radian and still
not know where its camera is, because "where the camera is" means the rigid
transform ``X = T_ee_cam`` from the end-effector frame to the camera's optical
frame -- an offset of a few centimetres and a few degrees that no ruler can
reach (the optical centre is inside the lens, and the end-effector frame is a
mathematical fiction defined by the URDF).

The trick is to stop trying to measure ``X`` and instead measure MOTION.  Park
the robot at two poses while a fixed tag stays visible:

    T_base_tag  =  T_base_ee(i) @ X @ T_cam_tag(i)      (pose i)
    T_base_tag  =  T_base_ee(j) @ X @ T_cam_tag(j)      (pose j)

The left sides are the same tag, so the right sides are equal, and rearranging
gives the classic equation

    A X = X B        A = T_base_ee(j)^-1 @ T_base_ee(i)   (how the HAND moved)
                     B = T_cam_tag(j) @ T_cam_tag(i)^-1   (how the CAMERA moved)

Both ``A`` and ``B`` are things you can measure -- ``A`` from the encoders and
forward kinematics, ``B`` from the two tag detections.  ``X`` is the only
unknown, and it appears sandwiched between them, which is what makes this an
equation rather than a subtraction.

> **"The camera already measures the tag's pose. Why is another transform
> needed?"**  Because the camera measures the tag *in camera coordinates*, and
> the robot can only act in *base coordinates*.  Those two descriptions of the
> same physical object are related by exactly one unknown transform -- ``X``
> chained with forward kinematics.  Until you have it, "the cup is 30 cm in
> front of the lens" cannot be turned into a joint command.  ``X`` is the
> missing link in the chain, not a duplicate of anything in it.

Solver used here: **Park & Martin (1994)** -- a closed form that solves the
rotation first (as a small least-squares problem on rotation axes) and then the
translation (as an ordinary linear least-squares).  A Gauss-Newton refinement
that estimates the tag pose at the same time is included for comparison.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _rel in ("01-transform-calculator", "02-urdf-visualizer", "03-forward-kinematics-from-scratch"):
    sys.path.insert(0, os.path.join(HERE, "..", _rel))

import transforms as tf  # noqa: E402


# ---------------------------------------------------------------------------
# building the AX = XB pairs
# ---------------------------------------------------------------------------
def motion_pairs(T_base_ee, T_cam_tag, pairs=None):
    """Turn a list of (arm pose, tag detection) into a list of ``(A, B)``."""
    n = len(T_base_ee)
    if pairs is None:
        # Every consecutive pair plus every pair one apart: cheap, and it keeps
        # the rotation between the two poses large, which matters (see below).
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    out = []
    for i, j in pairs:
        A = tf.T_inv(T_base_ee[j]) @ T_base_ee[i]
        B = T_cam_tag[j] @ tf.T_inv(T_cam_tag[i])
        out.append((A, B))
    return out


# ---------------------------------------------------------------------------
# Park & Martin closed form
# ---------------------------------------------------------------------------
def _inv_sqrt_spd(M):
    """Inverse square root of a symmetric positive-definite matrix."""
    w, V = np.linalg.eigh(M)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-300))) @ V.T


def solve_park_martin(AB):
    """Closed-form solution of ``A X = X B`` from many motion pairs.

    Rotation.  Taking the logarithm of ``R_A R_X = R_X R_B`` turns rotations
    into rotation VECTORS and the equation into ``alpha = R_X beta`` -- an
    ordinary "rotate this arrow onto that arrow" problem, one arrow per pair.
    The least-squares answer is ``R_X = (M^T M)^{-1/2} M^T`` with
    ``M = sum beta alpha^T``, which is Park & Martin's result.  The inverse
    square root is what strips the stretch out of ``M`` and leaves a pure
    rotation, in the same spirit as project 01's ``orthonormalize``.

    Translation.  With ``R_X`` known, the translation part of ``A X = X B`` is
    ``(R_A - I) t_X = R_X t_B - t_A`` -- linear in ``t_X``, so stack every pair
    and solve.
    """
    M = np.zeros((3, 3))
    for A, B in AB:
        alpha = tf.R_to_axis_angle(A[:3, :3])
        beta = tf.R_to_axis_angle(B[:3, :3])
        M += np.outer(beta, alpha)
    R_X = _inv_sqrt_spd(M.T @ M) @ M.T
    R_X = tf.orthonormalize(R_X)  # guard against round-off leaving SO(3)

    C = np.zeros((3 * len(AB), 3))
    d = np.zeros(3 * len(AB))
    for k, (A, B) in enumerate(AB):
        C[3 * k:3 * k + 3] = A[:3, :3] - np.eye(3)
        d[3 * k:3 * k + 3] = R_X @ B[:3, 3] - A[:3, 3]
    t_X, *_ = np.linalg.lstsq(C, d, rcond=None)
    return tf.T_from_Rp(R_X, t_X)


def translation_conditioning(AB):
    """Singular values of the stacked ``(R_A - I)`` matrix.

    Each row block ``R_A - I`` annihilates its own rotation axis: turning about
    an axis tells you nothing about the offset ALONG that axis.  If every
    motion in the data set shares one axis, the stacked matrix is rank 2 and
    one component of ``t_X`` is not merely noisy -- it is not in the data at
    all.  These three numbers are how you find that out before trusting a
    result.
    """
    C = np.vstack([A[:3, :3] - np.eye(3) for A, _ in AB])
    return np.linalg.svd(C, compute_uv=False)


# ---------------------------------------------------------------------------
# refinement, and the self-check you can run without ground truth
# ---------------------------------------------------------------------------
def residual_axxb(X, AB):
    """How badly ``A X = X B`` is violated -- in degrees and millimetres.

    This needs NO ground truth, so it is the number you actually get to look at
    on a real robot.  Project 07 checks whether it predicts the true error.
    """
    rot, trans = [], []
    for A, B in AB:
        E = tf.T_inv(A @ X) @ (X @ B)
        rot.append(np.degrees(tf.rot_angle(E[:3, :3])))
        trans.append(np.linalg.norm(E[:3, 3]) * 1e3)
    return float(np.sqrt(np.mean(np.square(rot)))), float(np.sqrt(np.mean(np.square(trans))))


def refine(X0, T_base_ee, T_cam_tag, iters=25):
    """Gauss-Newton on ``X`` AND the tag pose together.

    The closed form throws information away: it only ever looks at DIFFERENCES
    between poses, so absolute agreement is never enforced.  Here the unknowns
    are ``X`` and the tag's pose in the base frame, and the residual for each
    observation is "where this observation says the tag is, versus where we
    currently think it is".  Twelve unknowns, six residuals per observation.
    """
    X = X0.copy()
    preds = [T @ X @ Tc for T, Tc in zip(T_base_ee, T_cam_tag)]
    Z = preds[0].copy()

    def residuals(X, Z):
        r = []
        for T, Tc in zip(T_base_ee, T_cam_tag):
            r.append(tf.se3_log(tf.T_inv(T @ X @ Tc) @ Z))
        return np.concatenate(r)

    for _ in range(iters):
        r0 = residuals(X, Z)
        J = np.zeros((len(r0), 12))
        eps = 1e-6
        for c in range(12):
            d = np.zeros(6)
            d[c % 6] = eps
            if c < 6:
                J[:, c] = (residuals(X @ tf.se3_exp(d), Z) - r0) / eps
            else:
                J[:, c] = (residuals(X, Z @ tf.se3_exp(d)) - r0) / eps
        step, *_ = np.linalg.lstsq(J, -r0, rcond=None)
        X = X @ tf.se3_exp(step[:6])
        Z = Z @ tf.se3_exp(step[6:])
        if np.linalg.norm(step) < 1e-12:
            break
    return X, Z


def tag_scatter(X, T_base_ee, T_cam_tag):
    """Predict the tag's pose from every observation; return the spread.

    The tag never moved, so every prediction should land on the same spot.  How
    far apart they land is a direct, ground-truth-free measure of how wrong
    ``X`` is -- and unlike the AX=XB residual it is in units a person can
    picture: millimetres of disagreement about where a thing on the table is.
    """
    P = np.array([(T @ X @ Tc)[:3, 3] for T, Tc in zip(T_base_ee, T_cam_tag)])
    return float(np.linalg.norm(P - P.mean(axis=0), axis=1).max() * 1e3), P


def pose_error(X_est, X_true):
    """(rotation error in degrees, translation error in millimetres)."""
    return (
        float(np.degrees(tf.rot_geodesic(X_est[:3, :3], X_true[:3, :3]))),
        float(np.linalg.norm(X_est[:3, 3] - X_true[:3, 3]) * 1e3),
    )
