"""An error-state Kalman filter that fuses an IMU with a monocular camera.

Two ideas carry the whole file.

ERROR STATE.  A rotation cannot be a state in a Kalman filter.  A Kalman filter
adds a correction to its state (`x = x + K y`) and assumes the result is still a
valid state; add a correction to a quaternion and you get something that is no
longer a unit quaternion, and add it to Euler angles and you fall into gimbal
lock.  The fix is to split the state in two:

    true state  =  nominal state  (*)  error state
                   ^^^^^^^^^^^^^       ^^^^^^^^^^^
                   big, nonlinear,     small, always near zero,
                   integrated from     genuinely Gaussian,
                   the raw IMU         genuinely additive

The filter only ever estimates the ERROR -- 15 small numbers -- and every so
often "injects" it into the nominal state and resets it to zero.  Because the
error is always tiny, linearizing around it is accurate, which is the whole
point.  This is not an optimization; it is what makes the filter correct.

WHY A CAMERA AT ALL, WHEN THE IMU ALREADY MEASURES MOTION.  It measures
acceleration and rotation rate, which have to be integrated twice and once
respectively to give position and attitude.  Every integration turns a small
constant error into a growing one -- project 21 measured position error growing
as t^2.87 for exactly this reason.  The camera cannot measure motion in metres
at all (a monocular camera has no scale), but what it does measure it measures
without drifting.  The two are complementary in the precise sense that each is
strong exactly where the other is weak, and experiment 2 shows the pairing is
not symmetric: the camera fixes the IMU's drift, and the IMU supplies the
camera's missing metre.

State layout (nominal): p(3), v(3), q(4), b_a(3), b_g(3)
State layout (error):   dp(3), dv(3), dtheta(3), db_a(3), db_g(3)   -> 15
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "21-imu-integration"))

from imu import quat_mul, quat_from_rotvec, quat_to_R, R_to_quat, G  # noqa: E402


def skew(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def normalize(q):
    return q / np.linalg.norm(q)


class ErrorStateEKF:
    """15-state visual-inertial filter."""

    def __init__(self, p, v, q, ba, bg, P, imu_noise):
        self.p = np.asarray(p, float).copy()
        self.v = np.asarray(v, float).copy()
        self.q = normalize(np.asarray(q, float).copy())
        self.ba = np.asarray(ba, float).copy()
        self.bg = np.asarray(bg, float).copy()
        self.P = np.asarray(P, float).copy()
        # (accel white, gyro white, accel bias walk, gyro bias walk)
        self.na, self.ng, self.nba, self.nbg = imu_noise

    # ------------------------------------------------------------- predict
    def predict(self, acc_m, gyr_m, dt):
        """Advance the nominal state with the raw IMU, and the error covariance
        with the linearized error dynamics."""
        R = quat_to_R(self.q)
        a = acc_m - self.ba
        w = gyr_m - self.bg

        # --- nominal state: plain strapdown integration, no approximation
        a_w = R @ a + G
        self.p = self.p + self.v * dt + 0.5 * a_w * dt ** 2
        self.v = self.v + a_w * dt
        self.q = normalize(quat_mul(self.q, quat_from_rotvec(w * dt)))

        # --- error state: F is the Jacobian of the error dynamics.
        # The only non-obvious block is dv/dtheta = -R [a]x dt: tilting the
        # body by dtheta rotates the measured acceleration, so an attitude
        # error leaks straight into velocity.  This single block is why gyro
        # errors dominate POSITION error in an inertial system -- project 21
        # measured 84 cm of position error from 0.1 deg of tilt in 10 s.
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ skew(a) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = quat_to_R(quat_from_rotvec(w * dt)).T
        F[6:9, 12:15] = -np.eye(3) * dt

        Q = np.zeros((15, 15))
        Q[3:6, 3:6] = np.eye(3) * self.na ** 2 * dt
        Q[6:9, 6:9] = np.eye(3) * self.ng ** 2 * dt
        Q[9:12, 9:12] = np.eye(3) * self.nba ** 2 * dt
        Q[12:15, 12:15] = np.eye(3) * self.nbg ** 2 * dt
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

    # -------------------------------------------------------------- update
    def update(self, y, H, R_meas):
        """Generic error-state update, followed by injection and reset."""
        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ y
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R_meas @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self._inject(dx)
        return y, S

    def _inject(self, dx):
        """Fold the estimated error into the nominal state and zero it.

        Note the attitude line: the correction is applied as a ROTATION
        (quaternion multiplication), not an addition.  That is the payoff of
        the error-state formulation -- the filter works in the flat,
        3-dimensional space of small rotations, and the conversion back to the
        curved space of real rotations happens here, exactly, once per update.
        """
        self.p += dx[0:3]
        self.v += dx[3:6]
        self.q = normalize(quat_mul(self.q, quat_from_rotvec(dx[6:9])))
        self.ba += dx[9:12]
        self.bg += dx[12:15]
        # Strictly the covariance should also be transported through the reset
        # (a matrix very close to the identity for small dtheta).  It is
        # omitted here, as in most implementations, because the correction is
        # second order in dtheta and unmeasurable next to everything else.

    # ------------------------------------------- the visual measurement
    def update_visual_direction(self, u_meas, p_ref, sigma_dir):
        """The camera says: 'since the last keyframe you moved THAT way.'

        A monocular camera cannot say how far -- one image sequence is
        consistent with a small room and a slow walk, or a large room and a
        fast one.  So the measurement is a UNIT VECTOR, the direction of
        travel since the reference pose, and it carries no metre information at
        all.  All the scale in this filter comes from the accelerometer.

        h(x) = (p - p_ref) / ||p - p_ref||
        """
        d = self.p - p_ref
        n = np.linalg.norm(d)
        if n < 1e-6:
            return None, None
        u_hat = d / n
        # d/dp of a normalized vector: the part of a change that is
        # PERPENDICULAR to the current direction.  Moving further along the
        # same line does not change the direction at all, which is exactly the
        # statement that this measurement carries no scale.
        J = (np.eye(3) - np.outer(u_hat, u_hat)) / n
        H = np.zeros((3, 15))
        H[:, 0:3] = J
        y = u_meas - u_hat
        return self.update(y, H, sigma_dir ** 2 * np.eye(3))

    def update_attitude(self, q_meas, sigma_att):
        """The camera front end also reports how the view rotated.

        Rotation IS observable from one monocular image pair, unlike
        translation magnitude, so this measurement is genuinely informative and
        is what stops the gyro bias from wandering off.
        """
        dq = quat_mul(np.array([self.q[0], -self.q[1], -self.q[2], -self.q[3]]),
                      normalize(np.asarray(q_meas, float)))
        # small-angle vector part of the residual rotation
        y = 2.0 * dq[1:] * np.sign(dq[0])
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        return self.update(y, H, sigma_att ** 2 * np.eye(3))

    def state(self):
        return dict(p=self.p.copy(), v=self.v.copy(), q=self.q.copy(),
                    ba=self.ba.copy(), bg=self.bg.copy())


class DirectEKF(ErrorStateEKF):
    """The same filter, but correcting the quaternion by ADDITION.

    This is what almost everyone writes first, and it is NOT a strawman: it is
    the correct first-order expansion of the multiplicative update.  Since

        q (*) [1, dtheta/2]  =  q + q (*) [0, dtheta/2] + O(dtheta^2)

    the additive version below agrees with the error-state version to first
    order and differs only in the terms it drops.  Then it renormalizes,
    because addition takes a unit quaternion off the unit sphere.

    So the question experiment 4 actually asks is not "does the wrong frame
    break it" -- the frame here is right -- but "how big does a correction have
    to get before the dropped second-order term and the renormalization matter?"
    That is a genuine engineering question with a measurable answer, and it is
    much more useful than watching an obviously broken filter explode.
    """

    def _inject(self, dx):
        self.p += dx[0:3]
        self.v += dx[3:6]
        dq = quat_mul(self.q, np.concatenate([[0.0], 0.5 * dx[6:9]]))
        self.q = normalize(self.q + dq)
        self.ba += dx[9:12]
        self.bg += dx[12:15]
