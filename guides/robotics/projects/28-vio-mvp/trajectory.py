"""The flights, the IMU that feels them, and the camera front end that sees them.

Why simulate rather than download EuRoC or KITTI?  The same reason as projects
16 and 24: we need to *know* the answer.  Experiment 2 claims that the metric
scale of a monocular system is unobservable under constant velocity and
observable under acceleration.  On a public dataset that claim is untestable --
you can measure the final error but you cannot hold the excitation fixed while
changing nothing else, and you certainly cannot switch gravity off to see what
breaks.  Here every trajectory is generated analytically, so acceleration,
rotation rate and the exact IMU biases are all known and all adjustable.

The camera front end is simulated at the level of its OUTPUT, not its pixels:
we assume a working feature tracker plus essential-matrix decomposition, of the
kind project 20 built and measured, and take from it the two things such a front
end can actually produce -- a relative ROTATION and a translation DIRECTION.
The noise levels below are set from project 20's measured performance.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "21-imu-integration"))

from imu import quat_from_rotvec, quat_mul, quat_to_R, R_to_quat, G  # noqa: E402


# ------------------------------------------------------------- trajectories
def figure_eight_3d(t, speed=1.0, ax=6.0, ay=3.0, az=0.4):
    """A lissajous flight: continuously accelerating, so scale is observable."""
    w = speed * 0.25
    p = np.stack([ax * np.sin(w * t), ay * np.sin(2 * w * t), az * np.sin(w * t)],
                 axis=-1)
    v = np.stack([ax * w * np.cos(w * t), 2 * ay * w * np.cos(2 * w * t),
                  az * w * np.cos(w * t)], axis=-1)
    a = np.stack([-ax * w ** 2 * np.sin(w * t),
                  -4 * ay * w ** 2 * np.sin(2 * w * t),
                  -az * w ** 2 * np.sin(w * t)], axis=-1)
    return p, v, a


def straight_line(t, speed=1.0):
    """Constant velocity in a straight line -- zero acceleration.

    The pathological case for a monocular system.  Experiment 2 measures why.
    """
    d = np.array([0.8, 0.6, 0.0])
    p = np.outer(t, speed * d)
    v = np.repeat((speed * d)[None, :], len(t), axis=0)
    a = np.zeros_like(p)
    return p, v, a


def blended(t, excite):
    """Straight line plus `excite` times the accelerating wiggle.

    One knob that moves smoothly from 'no acceleration at all' to 'plenty',
    which is what an observability sweep needs.
    """
    p0, v0, a0 = straight_line(t)
    p1, v1, a1 = figure_eight_3d(t)
    return p0 + excite * p1, v0 + excite * v1, excite * a1


def attitude_from_path(t, spin=0.35, tilt=0.25):
    """A body attitude that keeps turning, so gyro biases are excited too."""
    qs = np.empty((len(t), 4))
    for i, ti in enumerate(t):
        rv = np.array([tilt * np.sin(0.7 * ti), tilt * np.sin(0.5 * ti + 1.0),
                       spin * ti])
        qs[i] = quat_from_rotvec(rv)
    return qs


# ------------------------------------------------------------------ the IMU
class ImuSpec:
    """A phone-grade MEMS IMU, the same one project 21 characterized."""

    def __init__(self, rate=200.0, acc_noise=0.012, gyr_noise=0.0035,
                 acc_walk=6e-5, gyr_walk=2e-5,
                 ba0=(0.05, -0.03, 0.04), bg0=(0.004, -0.003, 0.002)):
        self.rate = rate
        self.dt = 1.0 / rate
        self.acc_noise = acc_noise         # m/s^2 / sqrt(Hz)
        self.gyr_noise = gyr_noise         # rad/s / sqrt(Hz)
        self.acc_walk = acc_walk
        self.gyr_walk = gyr_walk
        self.ba0 = np.array(ba0)
        self.bg0 = np.array(bg0)


def make_imu(spec, t, p, v, a, q, rng):
    """Turn a known flight into the readings an IMU strapped to it would give.

    Two details that are easy to get backwards and expensive to debug:

    * The accelerometer measures SPECIFIC FORCE, not acceleration: it reads
      R' (a_world - g).  Sitting still it reads +9.81 m/s^2 upward, not zero.
      Getting the sign of g wrong here produces a filter that appears to work
      and drifts at 19.6 m/s^2.
    * The gyro reads the body-frame angular velocity, which we recover by
      differencing the true attitudes -- q_k^-1 q_{k+1} is the small rotation
      that happened, and its rotation vector divided by dt is omega.
    """
    n = len(t)
    dt = spec.dt
    ba = np.empty((n, 3))
    bg = np.empty((n, 3))
    ba[0], bg[0] = spec.ba0, spec.bg0
    for k in range(1, n):
        ba[k] = ba[k - 1] + spec.acc_walk * np.sqrt(dt) * rng.standard_normal(3)
        bg[k] = bg[k - 1] + spec.gyr_walk * np.sqrt(dt) * rng.standard_normal(3)

    acc = np.empty((n, 3))
    gyr = np.empty((n, 3))
    for k in range(n):
        R = quat_to_R(q[k])
        acc[k] = R.T @ (a[k] - G)
        if k + 1 < n:
            dq = quat_mul(np.array([q[k][0], -q[k][1], -q[k][2], -q[k][3]]),
                          q[k + 1])
            s = np.sign(dq[0]) if dq[0] != 0 else 1.0
            ang = 2.0 * np.arcsin(np.clip(np.linalg.norm(dq[1:]), -1, 1))
            axis = dq[1:] / (np.linalg.norm(dq[1:]) + 1e-12)
            gyr[k] = s * axis * ang / dt
        else:
            gyr[k] = gyr[k - 1]
    acc = acc + ba + spec.acc_noise / np.sqrt(dt) * rng.standard_normal((n, 3))
    gyr = gyr + bg + spec.gyr_noise / np.sqrt(dt) * rng.standard_normal((n, 3))
    return acc, gyr, ba, bg


# ------------------------------------------------------- the camera front end
def make_camera(t, p, q, cam_stride, rng, sigma_dir=0.01, sigma_att=0.004,
                dropout=(), time_offset_steps=0):
    """Simulate what a monocular front end reports, at every `cam_stride` IMU
    sample.

    Returns a list of (imu_index, reference_index, unit_direction, q_meas).

    `time_offset_steps` deliberately mislabels the camera data with the wrong
    IMU timestamp, which is experiment 6.
    """
    out = []
    idx = np.arange(0, len(t), cam_stride)
    for i in range(1, len(idx)):
        k, ref = idx[i], idx[i - 1]
        if any(lo <= k < hi for lo, hi in dropout):
            continue
        d = p[k] - p[ref]
        n = np.linalg.norm(d)
        if n < 1e-4:
            continue
        u = d / n
        u = u + sigma_dir * rng.standard_normal(3)
        u = u / np.linalg.norm(u)
        qm = quat_mul(q[k], quat_from_rotvec(sigma_att * rng.standard_normal(3)))
        # BOTH ends of the measurement carry the same wrong timestamp: the data
        # describes the motion from `ref` to `k`, but is handed to the filter as
        # if it described the motion from `ref+off` to `k+off`.  Shifting only
        # one end would be a different (and much more damaging) bug, and would
        # make the sweep artificially asymmetric.
        off = time_offset_steps
        kk = min(max(k + off, 1), len(t) - 1)
        rr = min(max(ref + off, 0), len(t) - 1)
        out.append((kk, rr, u, qm))
    return out


def attitude_error_deg(qa, qb):
    dq = quat_mul(np.array([qa[0], -qa[1], -qa[2], -qa[3]]), qb)
    return float(np.rad2deg(2.0 * np.arcsin(np.clip(np.linalg.norm(dq[1:]), 0, 1))))
