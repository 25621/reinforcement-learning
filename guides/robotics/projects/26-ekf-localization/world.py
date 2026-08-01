"""A differential-drive robot, a field of landmarks, and the two models an
estimator needs: how the robot moves and what it sees.

Shared with project 27 (which reuses the motion model for its particle filter)
and, through it, with project 29.

Everything here is deliberately written in the "velocity motion model" form of
Probabilistic Robotics chapter 5: the control is a forward speed v and a turn
rate w, and the noise lives on the CONTROL, not on the pose.  That matters --
it is why a robot driving fast has a bigger position uncertainty than one
driving slowly, which is true of real wheels and is not true if you simply bolt
a fixed covariance onto the pose each step.
"""

import numpy as np


def wrap(a):
    """Fold an angle into (-pi, pi].  Called on every angle in this file."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# ----------------------------------------------------------------- the motion
#
# ALPHA = (a1, a2, a3, a4) are the standard four noise coefficients:
#   speed noise  variance = a1 v^2 + a2 w^2
#   turn  noise  variance = a3 v^2 + a4 w^2
# The cross terms (a2, a3) are not a formality: a robot that turns hard also
# slips forward, and a robot that drives fast does not hold its heading.
ALPHA = np.array([0.02, 0.002, 0.02, 0.004])


def motion_noise(u, alpha=ALPHA):
    v, w = u
    return np.diag([alpha[0] * v ** 2 + alpha[1] * w ** 2,
                    alpha[2] * v ** 2 + alpha[3] * w ** 2])


def move(x, u, dt):
    """Exact integration of a constant (v, w) over dt -- an arc, not a chord.

    Using the straight-line approximation `x += v cos(theta) dt` instead is the
    single most common source of a mysterious systematic bias in wheeled
    odometry: it always cuts the corner, so the error accumulates in one
    direction instead of averaging out.
    """
    v, w = u
    th = x[2]
    if abs(w) < 1e-6:
        return np.array([x[0] + v * dt * np.cos(th),
                         x[1] + v * dt * np.sin(th),
                         wrap(th + w * dt)])
    r = v / w
    return np.array([x[0] - r * np.sin(th) + r * np.sin(th + w * dt),
                     x[1] + r * np.cos(th) - r * np.cos(th + w * dt),
                     wrap(th + w * dt)])


def motion_jacobians(x, u, dt):
    """(G, V): derivative of `move` with respect to the pose and the control."""
    v, w = u
    th = x[2]
    G = np.eye(3)
    V = np.zeros((3, 2))
    if abs(w) < 1e-6:
        G[0, 2] = -v * dt * np.sin(th)
        G[1, 2] = v * dt * np.cos(th)
        V[0, 0] = dt * np.cos(th)
        V[1, 0] = dt * np.sin(th)
        V[0, 1] = -0.5 * v * dt ** 2 * np.sin(th)
        V[1, 1] = 0.5 * v * dt ** 2 * np.cos(th)
        V[2, 1] = dt
        return G, V
    r = v / w
    s0, c0 = np.sin(th), np.cos(th)
    s1, c1 = np.sin(th + w * dt), np.cos(th + w * dt)
    G[0, 2] = -r * c0 + r * c1
    G[1, 2] = -r * s0 + r * s1
    V[0, 0] = (-s0 + s1) / w
    V[1, 0] = (c0 - c1) / w
    V[0, 1] = v * (s0 - s1) / w ** 2 + v * c1 * dt / w
    V[1, 1] = -v * (c0 - c1) / w ** 2 + v * s1 * dt / w
    V[2, 1] = dt
    return G, V


def sample_motion(x, u, dt, rng, alpha=ALPHA):
    """Drive one step with the control corrupted by noise."""
    M = motion_noise(u, alpha)
    un = np.array(u) + np.sqrt(np.diag(M)) * rng.standard_normal(2)
    return move(x, un, dt)


# ------------------------------------------------------------ the measurement
def range_bearing(x, lm):
    """What the robot at pose x should see when it looks at landmark lm."""
    d = np.asarray(lm) - np.asarray(x)[:2]
    r = np.hypot(d[0], d[1])
    return np.array([r, wrap(np.arctan2(d[1], d[0]) - x[2])])


def measurement_jacobian(x, lm):
    """d(range, bearing) / d(x, y, theta).  Three entries are worth reading:

    - the range row has a 0 in the theta column: turning on the spot does not
      change how far away anything is.  This single zero is why one landmark
      cannot pin down a heading from range alone.
    - the bearing row has a -1 in the theta column, exactly: turn one radian
      left and every bearing drops by one radian.
    """
    dx, dy = np.asarray(lm) - np.asarray(x)[:2]
    q = dx ** 2 + dy ** 2
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0],
                     [dy / q, -dx / q, -1.0]])


# ----------------------------------------------------------------- the worlds
def landmark_field(kind="spread"):
    """Four landmark layouts.  Geometry is a first-class experiment here."""
    if kind == "spread":
        return np.array([[-8.0, 8.0], [8.0, 9.0], [10.0, -6.0], [-9.0, -7.0],
                         [0.0, 12.0], [-12.0, 0.0]])
    if kind == "collinear":            # all along one line -- degenerate
        return np.array([[-10.0, 10.0], [-6.0, 10.0], [-2.0, 10.0],
                         [2.0, 10.0], [6.0, 10.0], [10.0, 10.0]])
    if kind == "one":
        return np.array([[0.0, 12.0]])
    if kind == "two":
        return np.array([[-8.0, 8.0], [8.0, 9.0]])
    if kind.startswith("n"):           # "n20" -> 20 landmarks in the same area
        n = int(kind[1:])
        rng = np.random.default_rng(3)
        # Rejection-sample so no landmark sits on top of the robot's path.
        out = []
        while len(out) < n:
            c = rng.uniform(-13.0, 13.0, size=(n, 2))
            keep = np.hypot(c[:, 0], c[:, 1]) > 6.0
            out.extend(c[keep].tolist())
        return np.array(out[:n])
    raise ValueError(kind)


def figure_eight(n, dt, v=1.2, w_amp=0.45):
    """A control sequence that turns both ways, so a heading error cannot hide.

    A robot driving in a circle accumulates a heading bias that looks exactly
    like a slightly wrong wheel radius, and no amount of data separates the two.
    Reversing the turn direction breaks that tie -- the same trick as tilting
    the calibration board in project 16.
    """
    t = np.arange(n) * dt
    w = w_amp * np.sin(2 * np.pi * t / (n * dt / 2.0))
    return np.stack([np.full(n, v), w], axis=1)


def simulate(x0, controls, dt, rng, landmarks, sigma_r=0.25, sigma_b=np.deg2rad(3.0),
             max_range=14.0, alpha=ALPHA, detect_prob=1.0):
    """Drive the robot and record everything an estimator could ever be given.

    Returns (true_poses, odom_poses, observations) where observations[k] is a
    list of (landmark_index, [range, bearing]) seen at step k.
    """
    x = np.array(x0, dtype=float)
    xo = np.array(x0, dtype=float)          # pure dead reckoning, for comparison
    poses, odom, obs = [x.copy()], [xo.copy()], [[]]
    for k, u in enumerate(controls):
        x = sample_motion(x, u, dt, rng, alpha)
        xo = move(xo, u, dt)                # commanded control, no noise added
        zs = []
        for j, lm in enumerate(landmarks):
            z = range_bearing(x, lm)
            if z[0] > max_range:
                continue
            if detect_prob < 1.0 and rng.random() > detect_prob:
                continue
            z = z + np.array([sigma_r, sigma_b]) * rng.standard_normal(2)
            z[1] = wrap(z[1])
            zs.append((j, z))
        poses.append(x.copy()); odom.append(xo.copy()); obs.append(zs)
    return np.array(poses), np.array(odom), obs


def pose_error(a, b):
    """(position error in metres, heading error in radians)."""
    return np.hypot(a[0] - b[0], a[1] - b[1]), abs(wrap(a[2] - b[2]))
