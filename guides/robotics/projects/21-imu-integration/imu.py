"""An IMU, its error model, and the strapdown integration that turns rates
into a trajectory.

An IMU measures **rates**, not positions:

    gyroscope     -> angular velocity   (rad/s), in the sensor's own frame
    accelerometer -> specific force     (m/s^2), in the sensor's own frame

"Specific force" is the phrase that trips everyone up.  An accelerometer at
rest on a desk does NOT read zero; it reads +9.81 m/s^2 upward.  It measures
the force the desk applies to hold the sensor up, not the sensor's
acceleration through space.  So the first thing strapdown integration must
do is subtract gravity -- and to subtract gravity you need to know which way
is down, which is what the gyroscope integration is for.  That coupling is
the single most important fact in this project.

"Strapdown" is historical: early inertial systems mounted the sensors on a
motor-stabilized gimbal that physically held them level, so the accelerometers
always pointed along known axes.  A strapdown system bolts the sensors
straight to the vehicle and does the levelling in software instead.
"""

import numpy as np


# --------------------------------------------------------------------------
# rotations as quaternions
# --------------------------------------------------------------------------

def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_from_rotvec(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.array([1.0, v[0] / 2, v[1] / 2, v[2] / 2])
    ax = v / th
    return np.concatenate([[np.cos(th / 2)], np.sin(th / 2) * ax])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (R[j, i] + R[i, j]) / s
    q[k + 1] = (R[k, i] + R[i, k]) / s
    return q


def orthonormalize(R):
    """Snap a nearly-rotation matrix onto the closest true rotation (SVD)."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1
        Rn = U @ Vt
    return Rn


def orthogonality_error(R):
    """How far a matrix has drifted from being a rotation at all."""
    return float(np.linalg.norm(R.T @ R - np.eye(3)))


def angle_between(Ra, Rb):
    c = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


# --------------------------------------------------------------------------
# the sensor
# --------------------------------------------------------------------------

G = np.array([0.0, 0.0, -9.81])          # world gravity, z up


class IMUParams:
    """A plausible consumer MEMS IMU, the kind in a phone or a small drone.

    Two noise types per sensor, and they behave completely differently:

    * WHITE NOISE ("angle random walk" for a gyro): independent each sample.
      Integrating it gives a random walk, whose spread grows as sqrt(t).
      Specified as a spectral density, in rad/s/sqrt(Hz), because the amount
      you see in one sample depends on how fast you sample.
    * BIAS INSTABILITY: a slowly wandering offset, modelled here as a random
      walk on the bias itself.  Integrating a constant offset gives an error
      growing as t, which beats sqrt(t) eventually -- so past a few seconds
      the bias, not the noise, is what ruins you.
    """

    def __init__(self,
                 rate=200.0,
                 gyro_noise=0.0035,        # rad/s/sqrt(Hz)  (~0.2 deg/s/sqrt(Hz))
                 gyro_bias=0.004,          # rad/s constant offset (~0.23 deg/s)
                 gyro_bias_walk=2e-5,      # rad/s^2/sqrt(Hz)
                 accel_noise=0.012,        # m/s^2/sqrt(Hz)
                 accel_bias=0.03,          # m/s^2 constant offset
                 accel_bias_walk=6e-5):    # m/s^3/sqrt(Hz)
        self.rate = rate
        self.dt = 1.0 / rate
        self.gyro_noise = gyro_noise
        self.gyro_bias = gyro_bias
        self.gyro_bias_walk = gyro_bias_walk
        self.accel_noise = accel_noise
        self.accel_bias = accel_bias
        self.accel_bias_walk = accel_bias_walk


def simulate(params, n, R_true=None, acc_world=None, rng=None, bias0=None):
    """Generate `n` IMU samples for a given true motion.

    R_true : function t -> world-from-body rotation (default: sitting still)
    acc_world : function t -> true acceleration in world coordinates

    Returns (gyro, accel, biases) with the measurements a real sensor would
    have produced, including gravity in the accelerometer.
    """
    rng = rng or np.random.default_rng(0)
    dt = params.dt
    gyro = np.zeros((n, 3))
    accel = np.zeros((n, 3))
    bg = (rng.normal(0, params.gyro_bias, 3) if bias0 is None else np.array(bias0[0], float))
    ba = (rng.normal(0, params.accel_bias, 3) if bias0 is None else np.array(bias0[1], float))
    biases = np.zeros((n, 6))
    # the noise you see in ONE sample of a sensor specified by a spectral
    # density: sigma = density / sqrt(dt).  Sample faster and each sample is
    # noisier, but you get more of them, and the two effects cancel.
    sg = params.gyro_noise / np.sqrt(dt)
    sa = params.accel_noise / np.sqrt(dt)
    for i in range(n):
        t = i * dt
        R = np.eye(3) if R_true is None else R_true(t)
        w_world = np.zeros(3) if R_true is None else _omega(R_true, t, dt)
        a_world = np.zeros(3) if acc_world is None else acc_world(t)
        gyro[i] = R.T @ w_world + bg + rng.normal(0, sg, 3)
        accel[i] = R.T @ (a_world - G) + ba + rng.normal(0, sa, 3)
        biases[i, :3] = bg
        biases[i, 3:] = ba
        bg = bg + rng.normal(0, params.gyro_bias_walk * np.sqrt(dt), 3)
        ba = ba + rng.normal(0, params.accel_bias_walk * np.sqrt(dt), 3)
    return gyro, accel, biases


def simulate_static(params, n, rng=None, level=True, tilt_deg=0.0):
    """The same thing for a sensor sitting still, computed all at once.

    Worth having as a separate function: the static case is most of this
    project (that is where drift is measured), and a vectorized version turns
    a 200,000-sample log from fifteen seconds of Python loop into a handful
    of array operations.
    """
    rng = rng or np.random.default_rng(0)
    dt = params.dt
    R = np.eye(3)
    if tilt_deg:
        c, s = np.cos(np.radians(tilt_deg)), np.sin(np.radians(tilt_deg))
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    bg0 = rng.normal(0, params.gyro_bias, 3)
    ba0 = rng.normal(0, params.accel_bias, 3)
    bg = bg0 + np.cumsum(rng.normal(0, params.gyro_bias_walk * np.sqrt(dt), (n, 3)), 0)
    ba = ba0 + np.cumsum(rng.normal(0, params.accel_bias_walk * np.sqrt(dt), (n, 3)), 0)
    gyro = bg + rng.normal(0, params.gyro_noise / np.sqrt(dt), (n, 3))
    accel = (R.T @ (-G))[None, :] + ba + rng.normal(0, params.accel_noise / np.sqrt(dt), (n, 3))
    return gyro, accel, np.concatenate([bg, ba], axis=1)


def _omega(R_true, t, dt):
    """True angular velocity in world coordinates, by finite difference."""
    R1, R2 = R_true(t), R_true(t + dt)
    dR = R2 @ R1.T
    q = R_to_quat(dR)
    th = 2 * np.arccos(np.clip(q[0], -1, 1))
    v = q[1:]
    nv = np.linalg.norm(v)
    return (v / nv * th / dt) if nv > 1e-12 else np.zeros(3)


# --------------------------------------------------------------------------
# integration
# --------------------------------------------------------------------------

def integrate(gyro, accel, dt, q0=None, v0=None, p0=None, bg=None, ba=None,
              method="exp", zupt_every=None, zupt_window=None, stride=1):
    """Strapdown integration: rates in, trajectory out.

    method : "exp"        -- rotate by the exponential map of omega*dt (exact
                             for a constant rotation rate over the step)
             "euler"      -- the small-angle shortcut R <- R (I + [omega]x dt).
                             This does NOT produce a rotation matrix: the
                             result drifts away from orthogonal, and it keeps
                             drifting, so the "attitude" stops being one.
             "euler_orth" -- the same shortcut, snapped back to the nearest
                             true rotation after every step

    zupt_every : if set, every this many samples the velocity is forced back
                 to zero.  A Zero-velocity UPdaTe is legitimate whenever you
                 KNOW the sensor is stationary (a foot flat on the ground, a
                 vehicle at a stop light) -- it is free information that the
                 accelerometer cannot provide.
    """
    n = len(gyro)
    q = np.array([1.0, 0, 0, 0]) if q0 is None else np.array(q0, float)
    v = np.zeros(3) if v0 is None else np.array(v0, float)
    p = np.zeros(3) if p0 is None else np.array(p0, float)
    bg = np.zeros(3) if bg is None else np.asarray(bg, float)
    ba = np.zeros(3) if ba is None else np.asarray(ba, float)
    R = quat_to_R(q)
    out_R, out_v, out_p = [], [], []
    for i in range(n):
        w = gyro[i] - bg
        a = accel[i] - ba
        if method == "exp":
            R = R @ quat_to_R(quat_from_rotvec(w * dt))
        elif method == "euler":
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            R = R @ (np.eye(3) + K * dt)
        elif method == "euler_orth":
            K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            R = orthonormalize(R @ (np.eye(3) + K * dt))
        a_world = R @ a + G                 # undo gravity, in WORLD coordinates
        p = p + v * dt + 0.5 * a_world * dt * dt
        v = v + a_world * dt
        if zupt_every and (i % zupt_every) < (zupt_window or 1):
            v = np.zeros(3)
        if i % stride == 0:
            out_R.append(R.copy())
            out_v.append(v.copy())
            out_p.append(p.copy())
    return np.array(out_R), np.array(out_v), np.array(out_p)


# --------------------------------------------------------------------------
# Allan deviation
# --------------------------------------------------------------------------

def allan_deviation(x, dt, n_tau=25):
    """Allan deviation of a sensor log, from scratch.

    Named after David Allan, who introduced it in 1966 to characterize atomic
    clocks.  The idea: chop the log into blocks of length tau, average each
    block, and measure how much NEIGHBOURING block-averages differ.  Plot
    that against tau on log axes and each noise type shows up as its own
    straight line:

        slope -1/2  ->  white noise      (averaging longer helps)
        slope  0    ->  bias instability (averaging longer stops helping)
        slope +1/2  ->  bias random walk (averaging longer HURTS)

    The value at tau = 1 s reads off the white-noise density directly, and
    the flat part's height is the bias instability.  It is the standard way
    to get an IMU's real noise numbers when the datasheet is optimistic.
    """
    x = np.asarray(x, float)
    n = len(x)
    taus = np.unique(np.logspace(0, np.log10(n // 4), n_tau).astype(int))
    out = []
    for m in taus:
        k = n // m
        if k < 3:
            continue
        blocks = x[:k * m].reshape(k, m).mean(axis=1)
        d = np.diff(blocks)
        out.append((m * dt, float(np.sqrt(0.5 * np.mean(d * d)))))
    return np.array(out)
