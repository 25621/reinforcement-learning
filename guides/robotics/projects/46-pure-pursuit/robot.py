"""A differential-drive robot, a path to follow, and three path trackers.

This is the shared library for Phase 7's wheeled projects.  Project 47 (DWA)
imports the robot and the simulator, project 49 (AMCL) imports the robot and
its noisy odometry, and project 53 (social navigation) imports all of it.

Everything here is deliberately small and exact:

  * `DiffDrive`  -- the plant.  Position, heading, and the three things that
                    make a real robot different from the textbook one:
                    acceleration limits, an actuator lag, and a control period
                    that is slower than the physics.
  * `Path`       -- a polyline with arc length, a closest-point query and a
                    look-ahead query.
  * the trackers -- `pure_pursuit`, `stanley`, `heading_p`.
"""

import math

import numpy as np


# ------------------------------------------------------------------ helpers
def wrap(a):
    """Fold an angle into (-pi, pi].

    Angles are the one quantity in robotics where 359 deg and -1 deg are the
    same number, and a controller that does not fold them will command a full
    turn to correct a one-degree error.
    """
    return (a + math.pi) % (2 * math.pi) - math.pi


# ------------------------------------------------------------------ the plant
class DiffDrive:
    """Unicycle kinematics with the limits a real chassis has.

    The "unicycle" model is the standard idealisation of a differential-drive
    base: two independently driven wheels are equivalent to a single point
    that can move forwards at speed v and rotate at rate omega, but can never
    slide sideways.  That last part is the non-holonomic constraint, and it is
    the whole reason path tracking needs a controller instead of a subtraction.
    """

    def __init__(self, x=0.0, y=0.0, th=0.0, v_max=3.0, w_max=2.5,
                 a_max=2.0, alpha_max=3.0, tau=0.0, rng=None,
                 slip_sigma=0.0):
        self.x, self.y, self.th = float(x), float(y), float(th)
        self.v, self.w = 0.0, 0.0          # what the wheels are actually doing
        self.v_max, self.w_max = v_max, w_max
        self.a_max, self.alpha_max = a_max, alpha_max
        self.tau = tau                     # first-order actuator lag, seconds
        self.rng = rng
        self.slip_sigma = slip_sigma
        # Odometry integrates the COMMANDED wheel motion, which is what a real
        # encoder-based estimate does; it never sees the slip.
        self.ox, self.oy, self.oth = self.x, self.y, self.th

    @property
    def state(self):
        return np.array([self.x, self.y, self.th])

    def step(self, v_cmd, w_cmd, dt):
        v_cmd = float(np.clip(v_cmd, -self.v_max, self.v_max))
        w_cmd = float(np.clip(w_cmd, -self.w_max, self.w_max))

        if self.tau > 0.0:
            # First-order lag: the wheels chase the command instead of jumping
            # to it.  A real motor's current loop looks like this.
            k = dt / (self.tau + dt)
            v_t = self.v + k * (v_cmd - self.v)
            w_t = self.w + k * (w_cmd - self.w)
        else:
            v_t, w_t = v_cmd, w_cmd

        # Acceleration limits, applied after the lag so both can be active.
        dv = np.clip(v_t - self.v, -self.a_max * dt, self.a_max * dt)
        dw = np.clip(w_t - self.w, -self.alpha_max * dt, self.alpha_max * dt)
        self.v += dv
        self.w += dw

        v, w = self.v, self.w
        if self.slip_sigma > 0.0 and self.rng is not None:
            v = v * (1.0 + self.rng.normal(0.0, self.slip_sigma))
            w = w + self.rng.normal(0.0, self.slip_sigma * self.w_max * 0.3)

        # Exact integration of the unicycle over one step: on a constant
        # (v, w) the robot moves along a circular arc, not a straight line.
        # Using the arc costs nothing and removes a bias that would otherwise
        # grow with the turn rate.
        if abs(w) < 1e-9:
            self.x += v * math.cos(self.th) * dt
            self.y += v * math.sin(self.th) * dt
        else:
            th2 = self.th + w * dt
            self.x += (v / w) * (math.sin(th2) - math.sin(self.th))
            self.y -= (v / w) * (math.cos(th2) - math.cos(self.th))
        self.th = wrap(self.th + w * dt)

        # Odometry: same integration, but on the commanded (v, w).
        if abs(self.w) < 1e-9:
            self.ox += self.v * math.cos(self.oth) * dt
            self.oy += self.v * math.sin(self.oth) * dt
        else:
            oth2 = self.oth + self.w * dt
            self.ox += (self.v / self.w) * (math.sin(oth2) - math.sin(self.oth))
            self.oy -= (self.v / self.w) * (math.cos(oth2) - math.cos(self.oth))
        self.oth = wrap(self.oth + self.w * dt)
        return self.state


# ------------------------------------------------------------------ the path
class Path:
    """A polyline, resampled to a fine even spacing, with arc length."""

    def __init__(self, pts, spacing=0.02, closed=False):
        pts = np.asarray(pts, float)
        if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
            pts = np.vstack([pts, pts[0]])
        self.closed = closed
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        n = max(int(s[-1] / spacing) + 1, 2)
        su = np.linspace(0.0, s[-1], n)
        self.pts = np.column_stack([np.interp(su, s, pts[:, 0]),
                                    np.interp(su, s, pts[:, 1])])
        self.s = su
        self.length = float(s[-1])

    def closest(self, xy, hint=None, window=None):
        """Index of the nearest path sample.

        `hint`/`window` restrict the search to a band around the last match.
        Without it, a path that crosses itself (a figure of eight) can snap the
        robot onto the wrong branch mid-run.
        """
        if hint is None or window is None:
            d = np.linalg.norm(self.pts - xy, axis=1)
            return int(np.argmin(d))
        lo, hi = hint, min(hint + window, len(self.pts))
        if self.closed:
            idx = np.arange(hint, hint + window) % len(self.pts)
        else:
            idx = np.arange(lo, hi)
        d = np.linalg.norm(self.pts[idx] - xy, axis=1)
        return int(idx[int(np.argmin(d))])

    def cross_track(self, xy, th, i):
        """Signed perpendicular distance to the path at sample `i`.

        Positive means the robot is to the LEFT of the path direction, which is
        the sign convention every steering law below assumes.
        """
        j = min(i + 1, len(self.pts) - 1)
        t = self.pts[j] - self.pts[max(i - 1, 0)]
        n = np.linalg.norm(t)
        if n < 1e-12:
            return 0.0, 0.0
        t = t / n
        d = np.asarray(xy, float) - self.pts[i]
        e = float(-t[1] * d[0] + t[0] * d[1])       # 2D cross product t x d
        heading = wrap(math.atan2(t[1], t[0]) - th)
        return e, heading

    def lookahead(self, xy, i, L):
        """The point on the path exactly L away, searching FORWARD from `i`.

        Walking forward matters: the circle of radius L around the robot cuts
        the path in two places, one behind and one ahead.  Aiming at the one
        behind makes the robot turn around and drive the path backwards.
        """
        n = len(self.pts)
        j = i
        for _ in range(n):
            nxt = j + 1
            if nxt >= n:
                if not self.closed:
                    return self.pts[-1], n - 1, False
                nxt = 0
            if np.linalg.norm(self.pts[nxt] - xy) >= L:
                # Interpolate between j and nxt for the exact circle crossing.
                a, b = self.pts[j], self.pts[nxt]
                da = np.linalg.norm(a - xy)
                db = np.linalg.norm(b - xy)
                w = 0.0 if db - da < 1e-12 else np.clip((L - da) / (db - da), 0, 1)
                return a + w * (b - a), nxt, True
            j = nxt
        return self.pts[-1], n - 1, False

    def finished(self, xy, i, tol=0.15):
        if self.closed:
            return False
        return i >= len(self.pts) - 2 and np.linalg.norm(self.pts[-1] - xy) < tol


# ------------------------------------------------------------------ paths
def path_circle(R=4.0, n=400):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return Path(np.column_stack([R * np.cos(t), R * np.sin(t)]), closed=True)


def path_figure_eight(R=3.0, n=600):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return Path(np.column_stack([R * np.sin(t), R * np.sin(t) * np.cos(t)]),
                closed=True)


def path_racetrack(straight=8.0, R=3.0, n=200):
    """Two straights joined by two half-circles -- the classic test loop.

    It matters that the curvature is DISCONTINUOUS at the four joins: it jumps
    from 0 to 1/R with no ramp.  Every tracker below has to make that jump
    with a finite turn rate, so the joins are where the error lives.
    """
    a = np.linspace(-math.pi / 2, math.pi / 2, n)
    right = np.column_stack([straight / 2 + R * np.cos(a), R * np.sin(a)])
    left = np.column_stack([-straight / 2 - R * np.cos(a), -R * np.sin(a)])
    top = np.column_stack([np.linspace(straight / 2, -straight / 2, n),
                           np.full(n, R)])
    bot = np.column_stack([np.linspace(-straight / 2, straight / 2, n),
                           np.full(n, -R)])
    return Path(np.vstack([bot, right, top, left]), closed=True)


def path_corner(leg=6.0):
    """A single 90-degree corner -- the cleanest way to measure corner cutting."""
    return Path(np.array([[-leg, 0.0], [0.0, 0.0], [0.0, leg]]), closed=False)


def path_slalom(amp=1.5, wavelen=6.0, length=24.0, n=600):
    x = np.linspace(0, length, n)
    return Path(np.column_stack([x, amp * np.sin(2 * math.pi * x / wavelen)]),
                closed=False)


# ------------------------------------------------------------------ trackers
def pure_pursuit(state, path, i, L, v, **_):
    """Steer along the circular arc that passes through a point L ahead.

    "Pure pursuit" is named for what it does: like a dog chasing a car, the
    robot always points itself at a moving target and never plans anything.
    The whole controller is one line of geometry.

    Put the target in the robot's own frame as (x_r, y_r).  There is exactly
    one circle through the origin, tangent to the robot's heading, that also
    passes through the target, and its curvature is

        kappa = 2 * y_r / L^2

    Read it as: the further sideways the target is (y_r), the harder you turn;
    the further ahead it is (L), the gentler you turn -- and L enters squared,
    which is why look-ahead is such a powerful knob.
    """
    x, y, th = state
    tgt, j, ok = path.lookahead((x, y), i, L)
    dx, dy = tgt[0] - x, tgt[1] - y
    # Rotate the world-frame offset into the robot frame.
    xr = math.cos(th) * dx + math.sin(th) * dy
    yr = -math.sin(th) * dx + math.cos(th) * dy
    d2 = dx * dx + dy * dy
    if d2 < 1e-9:
        return v, 0.0, tgt
    kappa = 2.0 * yr / d2               # use the true distance, not nominal L
    if xr < 0.0 and abs(yr) < 1e-3:
        kappa = 2.0 / max(d2, 1e-6)     # target directly behind: turn, either way
    return v, v * kappa, tgt


def stanley(state, path, i, L, v, k_e=1.6, k_h=3.0, soft=0.5, **_):
    """Null the cross-track error directly, with a speed-scaled gain.

    Stanley (the Stanford car that won the 2005 DARPA Grand Challenge, and the
    controller is named after it) uses two terms instead of pure pursuit's one:
    a heading term that lines you up with the path, and a cross-track term
    whose correction angle is atan(k*e / v).  Dividing by v is the point --
    at 10 m/s a 1 m offset needs a gentle nudge, at 0.5 m/s it needs a sharp
    turn, and the same gain gives both.  `soft` keeps v=0 from dividing by 0.
    """
    x, y, th = state
    e, heading = path.cross_track((x, y), th, i)
    # e > 0 means the robot sits to the LEFT of the path, so the correction
    # has to bend it right -- hence the minus sign.  Getting this backwards
    # makes the robot run away from the path at exactly the rate it should
    # have converged, which is the loudest possible bug and still easy to ship.
    delta = heading - math.atan2(k_e * e, abs(v) + soft)
    return v, k_h * wrap(delta), path.pts[i]


def heading_p(state, path, i, L, v, k=2.5, **_):
    """The naive baseline: turn toward the closest point's tangent.

    This is what you write first, and it is here to show what it costs: with
    no cross-track term at all it lines the robot UP with the path but never
    pulls it ONTO the path, so any offset it starts with, it keeps.
    """
    x, y, th = state
    _, heading = path.cross_track((x, y), th, i)
    return v, k * heading, path.pts[i]


# ------------------------------------------------------------------ simulate
def simulate(path, tracker, v=1.0, dt=0.01, ctrl_hz=20.0, t_max=60.0,
             start=None, L=1.0, adaptive=None, robot_kw=None, laps=1.0,
             tracker_kw=None, pose_sigma=0.0, seed=0, lost=1.5):
    """Run one tracking episode and return everything worth plotting.

    `ctrl_hz` is separate from `dt` on purpose.  Physics runs at dt; the
    controller only gets to look and decide every 1/ctrl_hz seconds and its
    command is held constant in between (a zero-order hold).  That gap is
    where half of the real-world instability comes from, and a simulator that
    calls the controller every physics step cannot show it.

    `pose_sigma` is the other half.  The controller never sees the true pose;
    it sees whatever the localiser reports.  Feeding the tracker the exact
    state makes every gain look good, which is the single most flattering bug
    a tracking study can have.
    """
    robot_kw = dict(robot_kw or {})
    tracker_kw = dict(tracker_kw or {})
    rng = np.random.default_rng(seed)
    if start is None:
        p0, p1 = path.pts[0], path.pts[1]
        start = (p0[0], p0[1], math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    rb = DiffDrive(start[0], start[1], start[2], **robot_kw)

    every = max(int(round((1.0 / ctrl_hz) / dt)), 1)
    n = int(t_max / dt)
    i_hint = path.closest(rb.state[:2])
    xs, es, ws, tgts, ts = [], [], [], [], []
    v_cmd, w_cmd = v, 0.0
    travelled, done, t_done, diverged = 0.0, False, t_max, False
    goal_s = path.length * laps

    for k in range(n):
        st = rb.state
        i_hint = path.closest(st[:2], hint=i_hint, window=300)
        e, _ = path.cross_track(st[:2], st[2], i_hint)
        if abs(e) > lost:                     # gave up: report it, don't hide it
            diverged = True
            xs.append(st.copy()); es.append(e); ws.append(w_cmd)
            ts.append(k * dt)
            break
        if k % every == 0:
            Lk = L if adaptive is None else max(adaptive[0] * v + adaptive[1], 0.15)
            meas = st.copy()
            if pose_sigma > 0.0:
                meas[:2] += rng.normal(0.0, pose_sigma, 2)
                meas[2] += rng.normal(0.0, pose_sigma * 0.5)
            i_meas = path.closest(meas[:2], hint=max(i_hint - 20, 0), window=300)
            v_cmd, w_cmd, tgt = tracker(meas, path, i_meas, Lk, v, **tracker_kw)
            tgts.append(tgt)
        xs.append(st.copy())
        es.append(e)
        ws.append(w_cmd)
        ts.append(k * dt)
        rb.step(v_cmd, w_cmd, dt)
        travelled += abs(rb.v) * dt
        if travelled >= goal_s or path.finished(rb.state[:2], i_hint):
            done, t_done = True, (k + 1) * dt
            break

    xs = np.asarray(xs)
    es = np.asarray(es)
    ws = np.asarray(ws)
    return {
        "xy": xs[:, :2], "th": xs[:, 2], "e": es, "w": ws, "t": np.asarray(ts),
        "tgt": np.asarray(tgts) if tgts else np.zeros((0, 2)),
        "mean_abs_e": float(np.mean(np.abs(es))),
        "max_abs_e": float(np.max(np.abs(es))),
        "rms_e": float(np.sqrt(np.mean(es ** 2))),
        # Steering effort: how much the turn command CHANGES per second.  A
        # smooth run and an oscillating run can have the same mean error; only
        # this number tells them apart.
        "w_rate": float(np.mean(np.abs(np.diff(ws))) / (1.0 / ctrl_hz))
        if len(ws) > 1 else 0.0,
        "finished": done, "diverged": diverged,
        "t_done": t_done, "travelled": travelled,
    }
