"""Rigid-body transform toolbox: rotation matrices, quaternions, axis-angle,
roll-pitch-yaw, and 4x4 homogeneous transforms.

Conventions used everywhere in the robotics Phase 1 projects
------------------------------------------------------------
* A rotation matrix ``R`` maps a vector expressed in the CHILD frame into the
  PARENT frame:  ``v_parent = R @ v_child``.
* A quaternion is stored as ``(w, x, y, z)`` -- scalar FIRST.  (ROS stores
  ``(x, y, z, w)``.  Neither is more correct; mixing them silently is what
  breaks robots.  Every function here says which one it wants.)
* An axis-angle / rotation vector is a single 3-vector ``r = theta * axis``.
  Its length IS the rotation angle in radians, so ``r = 0`` is "no rotation"
  with no special case needed.
* ``rpy`` follows the URDF convention: FIXED-axis roll-pitch-yaw, i.e.
  ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
* A homogeneous transform ``T`` is 4x4 with ``T[:3,:3] = R`` and ``T[:3,3] = p``.

Only NumPy is used -- no SciPy, no robotics library.
"""

import numpy as np

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def skew(v):
    """Return the 3x3 skew-symmetric matrix ``[v]_x`` with ``[v]_x @ u = v x u``.

    "Skew-symmetric" means the transpose is the negative: ``S.T == -S``.  This
    matrix is the bridge between the cross product (a geometric idea) and
    matrix algebra (what a computer is good at).
    """
    v = np.asarray(v, dtype=float)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def unskew(S):
    """Inverse of :func:`skew`: pull the 3-vector back out of the matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# roll-pitch-yaw  <->  rotation matrix   (URDF fixed-axis XYZ convention)
# ---------------------------------------------------------------------------


def rpy_to_R(rpy):
    """URDF roll-pitch-yaw -> rotation matrix.  ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""
    r, p, y = rpy
    return Rz(y) @ Ry(p) @ Rx(r)


def R_to_rpy(R):
    """Rotation matrix -> URDF roll-pitch-yaw.

    Returns the branch with ``pitch`` in ``[-pi/2, pi/2]``.  At exactly
    ``pitch = +-pi/2`` the roll and yaw axes line up (gimbal lock) and only
    their SUM (or difference) is determined; we then arbitrarily set roll = 0.
    """
    # R[2,0] = -sin(pitch); clip guards against 1+1e-16 from round-off.
    sp = -np.clip(R[2, 0], -1.0, 1.0)
    pitch = np.arcsin(sp)
    if np.abs(R[2, 0]) < 1.0 - 1e-12:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: pitch = +-pi/2
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# quaternion  <->  rotation matrix
# ---------------------------------------------------------------------------


def quat_normalize(q):
    return np.asarray(q, dtype=float) / np.linalg.norm(q)


def quat_conj(q):
    """Conjugate = inverse for a unit quaternion: negate the vector part."""
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_mul(a, b):
    """Hamilton product ``a * b`` -- composes rotations the way ``Ra @ Rb`` does."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def quat_to_R(q):
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat(R):
    """Rotation matrix -> quaternion (w, x, y, z), Shepperd's method.

    The naive formula ``w = sqrt(1 + trace(R)) / 2`` divides by ``w`` to get the
    vector part, and dies when ``w -> 0`` (a 180-degree rotation).  Shepperd's
    trick is to look at the four diagonal-ish candidates and expand the one
    that is largest, so the divisor is never small.  Result: full accuracy at
    every angle.
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w, x, y, z = (m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w, x, y, z = (m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w, x, y, z = (m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_canonical(q):
    """Pick the representative with ``w >= 0``.

    ``q`` and ``-q`` are the SAME rotation (the "double cover").  Anything that
    compares or averages quaternions must fold that ambiguity away first.
    """
    q = np.asarray(q, dtype=float)
    return -q if q[0] < 0.0 else q


# ---------------------------------------------------------------------------
# axis-angle  <->  rotation matrix
# ---------------------------------------------------------------------------


def axis_angle_to_R(r):
    """Rodrigues' rotation formula: rotation vector ``r = theta*axis`` -> ``R``.

    ``R = I + sin(theta) K + (1 - cos(theta)) K^2``  with ``K = skew(axis)``.
    Named after Olinde Rodrigues, who published it in 1840 -- decades before
    matrix notation existed, which is why the classical statement looks like
    trigonometry rather than linear algebra.
    """
    r = np.asarray(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        # Series expansion: for a tiny rotation, R ~ I + skew(r).
        return np.eye(3) + skew(r)
    K = skew(r / theta)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def R_to_axis_angle(R):
    """``R`` -> rotation vector, numerically robust at every angle.

    Route the answer through the quaternion, because the quaternion form has no
    small divisor: ``theta = 2*atan2(|vec|, w)`` is accurate near 0 AND near pi.
    See :func:`R_to_axis_angle_naive` for the textbook formula that is not.
    """
    q = quat_canonical(R_to_quat(R))
    vec = q[1:]
    n = np.linalg.norm(vec)
    if n == 0.0:  # exactly the identity; no threshold needed, see below
        return np.zeros(3)
    # No small-angle special case is required: theta = 2*atan2(n, w) is itself
    # proportional to n, so the theta/n below is a ratio of two small numbers
    # that stays perfectly conditioned.  (A guard like "if n < 1e-12: return 0"
    # would quietly ROUND AWAY every rotation below a millionth of a degree.)
    theta = 2.0 * np.arctan2(n, q[0])
    return (theta / n) * vec


def R_to_axis_angle_naive(R):
    """The formula every textbook prints -- kept here to MEASURE its failure.

    ``theta = arccos((trace(R) - 1) / 2)``, ``axis = unskew(R - R.T) / (2 sin theta)``.
    Both halves break down: ``arccos`` has infinite slope at its endpoints, so a
    round-off error of 1e-16 in the trace becomes an error of ~1e-8 in theta;
    and dividing by ``sin(theta)`` explodes as theta approaches 0 or pi.
    """
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(tr)
    s = np.sin(theta)
    if abs(s) < 1e-12:
        return np.zeros(3)
    axis = unskew(R - R.T) / (2.0 * s)
    return theta * axis


def axis_angle_to_quat(r):
    r = np.asarray(r, dtype=float)
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = r / theta
    return np.concatenate([[np.cos(theta / 2.0)], np.sin(theta / 2.0) * axis])


def quat_to_axis_angle(q):
    q = quat_canonical(quat_normalize(q))
    n = np.linalg.norm(q[1:])
    if n == 0.0:
        return np.zeros(3)
    return (2.0 * np.arctan2(n, q[0]) / n) * q[1:]


# ---------------------------------------------------------------------------
# distances and interpolation
# ---------------------------------------------------------------------------


def rot_angle(R):
    """The single number "how far is this rotation from identity", in radians."""
    return float(np.linalg.norm(R_to_axis_angle(R)))


def rot_geodesic(R1, R2):
    """Angle of the rotation that takes ``R1`` to ``R2`` -- the natural distance
    on SO(3).  "Geodesic" = shortest path along the curved space itself, the
    way a great circle is the shortest path on a globe."""
    return rot_angle(R1.T @ R2)


def slerp(q0, q1, t):
    """Spherical LINear intERPolation between two unit quaternions.

    Plain (straight-line) interpolation cuts THROUGH the unit sphere, so the
    result is short and has to be renormalised, and the rotation speed is
    uneven.  Slerp walks along the sphere's surface at constant angular speed.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:  # take the short way round the double cover
        q1, d = -q1, -d
    d = min(d, 1.0)
    if d > 1.0 - 1e-9:  # nearly identical: fall back to a straight line
        return quat_normalize(q0 + t * (q1 - q0))
    omega = np.arccos(d)
    so = np.sin(omega)
    return (np.sin((1.0 - t) * omega) / so) * q0 + (np.sin(t * omega) / so) * q1


def orthonormalize(R):
    """Snap a drifted matrix back onto SO(3) -- the nearest true rotation.

    Uses the SVD ``R = U S V^T`` and returns ``U V^T``, dropping the stretch
    ``S`` entirely.  The final ``det`` fix stops a numerically reflected matrix
    (det = -1, a mirror, not a rotation) from being returned.
    """
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def so3_defect(R):
    """How far a matrix has drifted off SO(3): ``||R^T R - I||_F``."""
    return float(np.linalg.norm(R.T @ R - np.eye(3)))


# ---------------------------------------------------------------------------
# 4x4 homogeneous transforms  (SE(3))
# ---------------------------------------------------------------------------


def T_from_Rp(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def T_inv(T):
    """Inverse of a rigid transform WITHOUT a general matrix inverse.

    ``T^-1 = [[R^T, -R^T p], [0, 1]]``.  Cheaper and exactly orthogonal;
    ``np.linalg.inv`` would spend an LU factorisation to rediscover this.
    """
    R, p = T[:3, :3], T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ p
    return Ti


def T_adjoint(T):
    """The 6x6 adjoint: maps a twist (v, w) from one frame into another.

    A twist is "linear velocity stacked on angular velocity".  Changing the
    frame of an angular velocity is just a rotation, but the linear part also
    picks up a lever-arm term ``p x w`` -- that is the ``skew(p) R`` block.
    """
    R, p = T[:3, :3], T[:3, 3]
    Ad = np.zeros((6, 6))
    Ad[:3, :3] = R
    Ad[:3, 3:] = skew(p) @ R
    Ad[3:, 3:] = R
    return Ad


def se3_log(T):
    """Matrix logarithm of a rigid transform -> a 6-vector twist ``(v, w)``.

    ``se3_exp(se3_log(T)) == T``.  This is "the constant screw motion that,
    run for one second, produces exactly this transform".
    """
    R, p = T[:3, :3], T[:3, 3]
    w = R_to_axis_angle(R)
    th = np.linalg.norm(w)
    if th < 1e-9:
        return np.concatenate([p, w])
    K = skew(w / th)
    # Inverse of the SE(3) left Jacobian (Murray, Li & Sastry eq. 2.36).
    Vinv = (
        np.eye(3)
        - 0.5 * skew(w)
        + (1.0 / th**2) * (1.0 - (th * np.sin(th)) / (2.0 * (1.0 - np.cos(th))))
        * (skew(w) @ skew(w))
    )
    del K
    return np.concatenate([Vinv @ p, w])


def se3_exp(xi):
    """Exponential map: a 6-vector twist ``(v, w)`` -> a 4x4 transform."""
    v, w = np.asarray(xi[:3], float), np.asarray(xi[3:], float)
    th = np.linalg.norm(w)
    R = axis_angle_to_R(w)
    if th < 1e-9:
        return T_from_Rp(R, v)
    K = skew(w / th)
    V = np.eye(3) + ((1.0 - np.cos(th)) / th) * K + ((th - np.sin(th)) / th) * (K @ K)
    return T_from_Rp(R, V @ v)


def pose_error(T_cur, T_des):
    """6-vector error used by every IK solver in this phase: ``(dp, dw)``.

    ``dp`` is the straight-line position error in metres.  ``dw`` is the
    axis-angle of the residual rotation, in radians -- the amount and direction
    you would have to twist to line the frames up.
    """
    dp = T_des[:3, 3] - T_cur[:3, 3]
    dw = R_to_axis_angle(T_des[:3, :3] @ T_cur[:3, :3].T)
    return np.concatenate([dp, dw])


# ---------------------------------------------------------------------------
# random sampling
# ---------------------------------------------------------------------------


def random_quats(n, rng):
    """``n`` rotations drawn UNIFORMLY over SO(3).

    Sample a 4-D Gaussian and normalise: the Gaussian is spherically symmetric,
    so the direction it points is uniform on the 3-sphere, which is exactly the
    uniform distribution on unit quaternions.  (Sampling roll/pitch/yaw
    uniformly does NOT do this -- it bunches rotations near the poles.)
    """
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def random_Rs(n, rng):
    return np.array([quat_to_R(q) for q in random_quats(n, rng)])


def random_T(rng, scale=1.0):
    R = quat_to_R(random_quats(1, rng)[0])
    p = rng.uniform(-scale, scale, size=3)
    return T_from_Rp(R, p)
