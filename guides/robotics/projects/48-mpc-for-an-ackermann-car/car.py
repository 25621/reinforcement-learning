"""A car-like vehicle, a racetrack, and a CasADi model-predictive controller.

Two models live here and the difference between them is the whole point of
the project:

  * `kin_step`  -- the KINEMATIC bicycle.  Wheels roll, never slide.  Five
                   states, no mass, no tyres, no friction.  This is what the
                   MPC believes.
  * `dyn_step`  -- the DYNAMIC bicycle.  Tyres generate side force only by
                   slipping a little, and that force saturates.  Six states,
                   mass and yaw inertia included.  This is the plant.

Running a controller built on the first against the second is not a bug; it
is what every real vehicle stack does, and the interesting question is how
fast you can go before the lie stops being harmless.

"Bicycle model" is literal: the two front wheels are collapsed into one at
the centre of the front axle and the two rear wheels into one at the rear,
because for anything short of hard cornering the left and right wheel of an
axle do nearly the same thing.  "Ackermann" is the steering GEOMETRY that
makes that approximation good -- Rudolph Ackermann patented the linkage in
1818 that turns the inner wheel more sharply than the outer one, so all four
wheels trace circles around one shared centre instead of scrubbing sideways.
"""

import math

import casadi as ca
import numpy as np

# ------------------------------------------------------------------ geometry
WHEELBASE = 2.6          # m, front axle to rear axle
DELTA_MAX = 0.42         # rad, about 24 degrees of steering at the wheel
DELTA_RATE = 1.2         # rad/s, how fast the steering rack can move
A_MAX = 4.0              # m/s^2
A_MIN = -6.0             # m/s^2, brakes beat the engine
MU = 0.9                 # tyre-road friction coefficient
G = 9.81

# ------------------------------------------------------------------ mass
MASS = 1200.0
IZ = 1500.0
LF = 1.2
LR = WHEELBASE - LF
CF = 80000.0             # N/rad front cornering stiffness
CR = 90000.0             # N/rad rear


# ------------------------------------------------------------ kinematic model
def kin_deriv(x, u):
    """State (X, Y, psi, v, delta); input (a, delta_rate).

    psi is the heading.  The one line that carries all the non-holonomy is
    psi_dot = v * tan(delta) / L: you can only change heading BY MOVING.  A
    car standing still cannot rotate however hard you turn the wheel, which
    is exactly what makes parallel parking a planning problem.
    """
    v, psi, delta = x[3], x[2], x[4]
    return ca.vertcat(v * ca.cos(psi),
                      v * ca.sin(psi),
                      v * ca.tan(delta) / WHEELBASE,
                      u[0],
                      u[1])


def kin_step_np(x, u, dt):
    """Same model in plain numpy, RK4, for use as a plant."""
    def f(s):
        X, Y, psi, v, d = s
        return np.array([v * math.cos(psi), v * math.sin(psi),
                         v * math.tan(d) / WHEELBASE, u[0], u[1]])
    k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2)
    k4 = f(x + dt * k3)
    out = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    out[4] = np.clip(out[4], -DELTA_MAX, DELTA_MAX)
    out[3] = max(out[3], 0.0)
    return out


# -------------------------------------------------------------- dynamic model
def dyn_step_np(x, u, dt, mu=MU):
    """State (X, Y, psi, vx, vy, r, delta); input (a, delta_rate).

    vy (sideways speed in the car's own frame) and r (yaw rate) are the two
    states the kinematic model simply does not have.  In the kinematic model
    vy is zero by assumption; here it is whatever the tyres allow.

    Tyre force uses a saturating law:  Fy = Fmax * tanh(C * alpha / Fmax).
    For small slip angle alpha it is the textbook linear Fy = C * alpha; as
    alpha grows it flattens out at Fmax = mu * (weight on that axle).  That
    flattening IS the friction limit -- past it, asking for more steering
    gives no more turning, which is what "understeer" feels like.
    """
    X, Y, psi, vx, vy, r, delta = x
    a, ddelta = u
    # The yaw mode of a stiff-tyred car has a time constant of order
    # (lf^2 Cf + lr^2 Cr) / (Iz vx) ~ 13 rad/s here.  An explicit step of
    # 25 ms sits right at the edge of stability for that, and an integrator
    # that is merely marginal will invent oscillations the car does not have.
    n_sub = 10
    h = dt / n_sub
    for _ in range(n_sub):
        if vx < 1.0:
            # Below walking pace the slip angles are numerically meaningless
            # (they divide by vx).  Fall back to the kinematic model, which
            # is exactly right at low speed anyway.
            k = kin_step_np(np.array([X, Y, psi, vx, delta]), u, h)
            X, Y, psi, vx, delta = k
            vy, r = 0.0, vx * math.tan(delta) / WHEELBASE
            continue
        af = delta - math.atan2(vy + LF * r, vx)
        ar = -math.atan2(vy - LR * r, vx)
        fzf = MASS * G * LR / WHEELBASE
        fzr = MASS * G * LF / WHEELBASE
        fyf = mu * fzf * math.tanh(CF * af / (mu * fzf))
        fyr = mu * fzr * math.tanh(CR * ar / (mu * fzr))
        ax = a + r * vy - fyf * math.sin(delta) / MASS
        ay = (fyf * math.cos(delta) + fyr) / MASS - r * vx
        rd = (LF * fyf * math.cos(delta) - LR * fyr) / IZ
        X += (vx * math.cos(psi) - vy * math.sin(psi)) * h
        Y += (vx * math.sin(psi) + vy * math.cos(psi)) * h
        psi += r * h
        vx = max(vx + ax * h, 0.0)
        vy += ay * h
        r += rd * h
        delta = float(np.clip(delta + ddelta * h, -DELTA_MAX, DELTA_MAX))
    return np.array([X, Y, psi, vx, vy, r, delta])


def dyn_to_kin(x):
    return np.array([x[0], x[1], x[2], x[3], x[6]])


def kin_to_dyn(x):
    return np.array([x[0], x[1], x[2], x[3], 0.0,
                     x[3] * math.tan(x[4]) / WHEELBASE, x[4]])


# ------------------------------------------------------------------ the track
class Track:
    """A closed centreline with arc length, heading, curvature and a width."""

    def __init__(self, n=2000, R0=34.0, A=9.0, half_width=4.0):
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        rad = R0 + A * np.cos(2 * t)
        pts = np.column_stack([rad * np.cos(t), rad * np.sin(t)])
        d = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(d)[:-1]])
        self.length = float(np.sum(d))
        self.pts = pts
        # Periodic differences: the track is a loop, so the sample after the
        # last one is the first one.  Using np.gradient's one-sided ends here
        # would put a fake curvature spike at the seam, and the speed profile
        # (which is built from curvature) would brake for a corner that is
        # not there.
        tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        self.psi = np.unwrap(np.arctan2(tang[:, 1], tang[:, 0]))
        dpsi = np.mod(np.roll(self.psi, -1) - np.roll(self.psi, 1) + np.pi,
                      2 * np.pi) - np.pi
        ds = np.roll(d, -1) + d
        self.kappa = dpsi / ds
        self.half_width = half_width

    def at(self, s):
        s = np.mod(s, self.length)
        x = np.interp(s, self.s, self.pts[:, 0], period=self.length)
        y = np.interp(s, self.s, self.pts[:, 1], period=self.length)
        psi = np.interp(s, self.s, np.unwrap(self.psi), period=self.length)
        k = np.interp(s, self.s, self.kappa, period=self.length)
        return np.column_stack([x, y, psi, k])

    def project(self, xy, hint=0, window=400):
        idx = (hint + np.arange(-40, window)) % len(self.pts)
        d = np.linalg.norm(self.pts[idx] - xy, axis=1)
        j = int(idx[int(np.argmin(d))])
        # Signed lateral offset from the centreline.
        t = np.array([math.cos(self.psi[j]), math.sin(self.psi[j])])
        dv = np.asarray(xy) - self.pts[j]
        return j, float(-t[1] * dv[0] + t[0] * dv[1])

    def speed_profile(self, a_lat=6.0, v_max=30.0, v_min=6.0):
        """The fastest speed each point could be taken at, from curvature.

        v = sqrt(a_lat / kappa) is just "centripetal acceleration = v^2 * kappa
        must not exceed what the tyres can supply".  It is the single most
        useful line in racing-line code, and it says the speed limit of a
        corner is set by its radius and nothing else.
        """
        k = np.abs(self.kappa)
        v = np.sqrt(a_lat / np.maximum(k, 1e-6))
        return np.clip(v, v_min, v_max)


# ------------------------------------------------------------------ the MPC
class MPC:
    """Kinematic-bicycle MPC, compiled once into a single CasADi function.

    Why compile?  Rebuilding the optimisation problem every control step is
    the slowest possible way to run MPC and it is also the most common.  The
    problem STRUCTURE never changes between steps -- only the current state
    and the reference change -- so those go in as *parameters* and the whole
    solve becomes one function call.
    """

    def __init__(self, N=20, dt=0.1, q_pos=8.0, q_psi=2.0, q_v=1.0,
                 r_a=0.02, r_d=8.0, rd_a=0.05, rd_d=25.0,
                 half_width=None, a_min=A_MIN, a_max=A_MAX,
                 soft=False, w_slack=200.0, d_rate=DELTA_RATE):
        self.N, self.dt = N, dt
        opti = ca.Opti()
        X = opti.variable(5, N + 1)
        U = opti.variable(2, N)
        x0 = opti.parameter(5)
        ref = opti.parameter(4, N + 1)         # x, y, psi, v_ref
        u_prev = opti.parameter(2)

        cost = 0
        for k in range(N + 1):
            e = X[0:2, k] - ref[0:2, k]
            cost += q_pos * ca.sumsqr(e)
            # Wrap the heading error: without this a lap that crosses the
            # +/- pi line asks the car to spin a full turn.
            dpsi = X[2, k] - ref[2, k]
            cost += q_psi * (ca.sin(dpsi) ** 2 + (1 - ca.cos(dpsi)) ** 2)
            cost += q_v * (X[3, k] - ref[3, k]) ** 2
        for k in range(N):
            cost += r_a * U[0, k] ** 2 + r_d * U[1, k] ** 2
            prev = u_prev if k == 0 else U[:, k - 1]
            cost += rd_a * (U[0, k] - prev[0]) ** 2
            cost += rd_d * (U[1, k] - prev[1]) ** 2
            # RK4 of the kinematic model, as an equality constraint.  This is
            # "direct multiple shooting": the states are decision variables
            # and physics is a constraint, which conditions far better than
            # rolling the model out inside the cost.
            k1 = kin_deriv(X[:, k], U[:, k])
            k2 = kin_deriv(X[:, k] + dt / 2 * k1, U[:, k])
            k3 = kin_deriv(X[:, k] + dt / 2 * k2, U[:, k])
            k4 = kin_deriv(X[:, k] + dt * k3, U[:, k])
            opti.subject_to(X[:, k + 1] == X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(opti.bounded(-DELTA_MAX, ca.vec(X[4, :]), DELTA_MAX))
        opti.subject_to(opti.bounded(0.0, ca.vec(X[3, :]), 45.0))
        opti.subject_to(opti.bounded(a_min, ca.vec(U[0, :]), a_max))
        opti.subject_to(opti.bounded(-d_rate, ca.vec(U[1, :]), d_rate))
        if half_width is not None:
            # Stay inside the track: the perpendicular offset from the
            # reference point must be under the half-width.  Cheap and
            # approximate -- exact would need the projection inside the NLP.
            #
            # `soft` turns it from a hard constraint into a heavily penalised
            # one.  The difference matters enormously: a hard constraint the
            # physics cannot satisfy makes the whole problem INFEASIBLE, and
            # an infeasible solve returns a number that is not a plan.  A soft
            # constraint always has an answer -- "go as far inside as you can,
            # and pay for the rest".
            sl = opti.variable(1, N) if soft else None
            if soft:
                opti.subject_to(opti.bounded(0, ca.vec(sl), 20.0))
                cost += w_slack * ca.sumsqr(sl)
            for k in range(1, N + 1):
                nx = -ca.sin(ref[2, k])
                ny = ca.cos(ref[2, k])
                lat = nx * (X[0, k] - ref[0, k]) + ny * (X[1, k] - ref[1, k])
                m = half_width + (sl[0, k - 1] if soft else 0.0)
                opti.subject_to(lat <= m)
                opti.subject_to(lat >= -m)

        opti.minimize(cost)   # (cost was extended above if `soft`)
        opti.solver("ipopt", {"print_time": False, "ipopt.print_level": 0,
                              "ipopt.max_iter": 80, "ipopt.tol": 1e-4,
                              "ipopt.acceptable_tol": 1e-3,
                              "ipopt.sb": "yes"})
        # opti.x / opti.lam_g are the primal and dual variables.  Feeding them
        # in and out makes the previous solve the next solve's starting guess
        # -- a warm start.  Consecutive MPC problems differ by 100 ms of
        # driving, so the old answer is nearly the new one, and IPOPT usually
        # converges in a handful of iterations instead of dozens.
        self.fn = opti.to_function(
            "mpc", [x0, ref, u_prev, opti.x, opti.lam_g],
            [U, X, opti.x, opti.lam_g])
        self.nx = opti.x.shape[0]
        self.ng = opti.lam_g.shape[0]
        self.reset()

    def reset(self):
        self.last_ok = True
        self.wx = np.zeros(self.nx)
        self.wl = np.zeros(self.ng)

    def __call__(self, x0, ref, u_prev):
        U, X, wx, wl = self.fn(x0, ref, u_prev, self.wx, self.wl)
        try:
            self.last_ok = bool(self.fn.stats()["success"])
        except Exception:
            self.last_ok = True
        self.wx, self.wl = np.asarray(wx).ravel(), np.asarray(wl).ravel()
        return np.asarray(U)[:, 0], np.asarray(X)


def make_ref(track, s0, v_prof, N, dt):
    """Reference: where the car SHOULD be at each step of the horizon.

    Built by walking arc length forward at the profile speed, so the
    reference already respects "slow down for the corner" -- the MPC then
    only has to track it, not rediscover it.
    """
    s = s0
    out = np.empty((4, N + 1))
    for k in range(N + 1):
        p = track.at(np.array([s]))[0]
        v = float(np.interp(np.mod(s, track.length), track.s, v_prof,
                            period=track.length))
        out[:, k] = [p[0], p[1], p[2], v]
        s += v * dt
    return out


def pure_pursuit_steer(x, track, Ld, hint=0):
    """The 46-style tracker, converted to a steering angle.

    Pure pursuit on a car uses the SAME arc geometry as on the differential
    drive, but the output is different: a diff-drive is told a turn rate, a
    car is told a wheel angle.  delta = atan(L * kappa) is the conversion --
    it asks which steering angle produces that curvature on this wheelbase.
    """
    X, Y, psi = x[0], x[1], x[2]
    j, _ = track.project((X, Y), hint=hint)
    n = len(track.pts)
    i = j
    for _ in range(n):
        if np.linalg.norm(track.pts[i] - np.array([X, Y])) >= Ld:
            break
        i = (i + 1) % n
    dx, dy = track.pts[i] - np.array([X, Y])
    yr = -math.sin(psi) * dx + math.cos(psi) * dy
    kappa = 2.0 * yr / max(dx * dx + dy * dy, 1e-6)
    return float(np.clip(math.atan(WHEELBASE * kappa), -DELTA_MAX, DELTA_MAX))
