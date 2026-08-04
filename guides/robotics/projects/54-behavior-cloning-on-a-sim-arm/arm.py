"""A planar arm, a puck, and a scripted expert -- the shared toy for Phase 8.

Every project from 54 to 61 needs the same three things: a robot that moves
under torque, a task whose success is unambiguous, and a demonstrator that can
be asked for the right action *at any state* (not just along its own path).
This file is that.

The robot is a 2-link planar arm lying FLAT on a table, seen from above, so
gravity does no work on it.  That is not laziness: with gravity out of the way
the only forces left are inertia, Coriolis and joint damping, which is exactly
the set that domain randomisation (project 59) and system identification are
about.  A vertical arm would add a large gravity torque that dominates
everything and hides those effects.

The task is a PUSH: drive the tip around behind a puck and shove it onto a
goal.  A pure reach would be over in three steps and no imitation-learning
pathology would ever show up.  Pushing gives us the two properties the phase
needs:

  * errors COMPOUND -- miss the puck by a centimetre and the puck ends up
    somewhere the demonstrator never was, so the next observation is off-
    distribution too (projects 54, 55);
  * the correct action is genuinely MULTIMODAL -- to get behind the puck you
    may circle it clockwise or anticlockwise, both are right, and their average
    drives straight into it (project 56).

Dynamics are written out here rather than handed to MuJoCo because the phase
needs to *change* the dynamics (mass, damping, actuator gain, latency) hundreds
of times per second and to re-simulate from arbitrary states.  The equations
are checked against MuJoCo in ``verify_mujoco.py``.
"""

import numpy as np

# ---------------------------------------------------------------------------
# geometry / task constants (metres, kilograms, seconds, radians)
# ---------------------------------------------------------------------------
R_PUCK = 0.030
R_TIP = 0.013
GOAL_TOL = 0.035          # puck centre this close to the goal counts as done
DT = 0.005                # physics step
CTRL_DT = 0.05            # one policy decision every 50 ms (20 Hz)
SUBSTEPS = int(round(CTRL_DT / DT))
DQ_MAX = 0.10             # radians of joint motion a single action may command
EP_LEN = 60               # decisions per episode -> 3 s


def _cross2(a, b):
    """z-component of a 2-D cross product (the only kind a planar arm has)."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


class PlanarArm:
    """An n-link planar arm with uniform-rod links, flat on a table.

    ``mass_scale``, ``damp_scale`` and ``gear`` exist so that project 59 can
    build a *different* robot without rewriting anything: they multiply the
    nominal masses, the joint damping and the commanded torque.
    """

    def __init__(self, lengths=(0.20, 0.18), masses=(0.60, 0.40),
                 damping=(0.60, 0.30), tau_max=(15.0, 3.0),
                 kp=(126.0, 15.5), kd=(4.2, 0.52),
                 mass_scale=1.0, damp_scale=1.0, gear=1.0, omega_n=60.0):
        self.l = np.asarray(lengths, float)
        self.m = np.asarray(masses, float) * mass_scale
        self.b = np.asarray(damping, float) * damp_scale
        self.tau_max = np.asarray(tau_max, float)
        self.gear = float(gear)
        self.n = len(self.l)
        self.r = self.l / 2.0                       # centre of mass of a rod
        self.I = self.m * self.l ** 2 / 12.0        # rod about its own centre
        self.reach = float(self.l.sum())
        # Servo gains.  Passing kp=None asks for gains sized to the robot --
        # needed by project 61, which builds a zoo of arms whose links differ
        # by an order of magnitude in inertia and cannot share one gain set.
        #
        # The gains are a MATRIX, K = M(q_nom) * omega^2, not a per-joint
        # number.  Using the diagonal of M looks equivalent and is not: the
        # closed-loop frequencies are the square roots of the eigenvalues of
        # M^-1 K, and for a three-link arm M has an eigenvalue a hundred times
        # smaller than its smallest diagonal entry.  Diagonal gains therefore
        # excite one coupled mode at ~320 rad/s, the 200 Hz integrator cannot
        # represent it, and the arm explodes on the first step.  With
        # K = M omega^2 the ratio is exactly omega^2 in every mode, by
        # construction -- the same inertia-shaping idea as computed-torque
        # control in project 11.
        # M(q) changes with the arm's posture -- a stretched arm has a much
        # weaker coupled mode than a folded one -- so the shaping has to be
        # done with the CURRENT mass matrix, not a snapshot of it.  A constant
        # K = M(q_nom) omega^2 still blows up as soon as the arm straightens.
        self.omega_n = float(omega_n)
        self.auto_gains = kp is None or kd is None
        if self.auto_gains:
            self.kp = np.zeros(self.n)
            self.kd = np.zeros(self.n)
        else:
            self.kp = np.asarray(kp, float)
            self.kd = np.asarray(kd, float)

    # -- kinematics ---------------------------------------------------------
    def points(self, q):
        """Joint positions p[0..n]: p[0] is the base, p[n] is the tip."""
        phi = np.cumsum(q)
        pts = np.zeros((self.n + 1, 2))
        for k in range(self.n):
            pts[k + 1] = pts[k] + self.l[k] * np.array([np.cos(phi[k]),
                                                        np.sin(phi[k])])
        return pts

    def tip(self, q):
        if self.n == 2:
            l1, l2 = self.l
            a, b = q[0], q[0] + q[1]
            return np.array([l1 * np.cos(a) + l2 * np.cos(b),
                             l1 * np.sin(a) + l2 * np.sin(b)])
        return self.points(q)[-1]

    def jacobian(self, q):
        """d tip / d q -- a 2 x n matrix."""
        pts = self.points(q)
        tip = pts[-1]
        J = np.zeros((2, self.n))
        for i in range(self.n):
            d = tip - pts[i]                 # lever arm from joint i to the tip
            J[:, i] = (-d[1], d[0])          # z-hat cross d
        return J

    # -- dynamics -----------------------------------------------------------
    def mass_matrix(self, q):
        """M(q) from the composite-Jacobian form.

        Rotating a rigid body does not change dot products, so the linear part
        of link k's contribution to M[i, j] is just the dot product of the two
        lever arms from joints i and j to link k's centre of mass.
        """
        pts = self.points(q)
        phi = np.cumsum(q)
        M = np.zeros((self.n, self.n))
        for k in range(self.n):
            c = pts[k] + self.r[k] * np.array([np.cos(phi[k]), np.sin(phi[k])])
            for i in range(k + 1):
                for j in range(k + 1):
                    M[i, j] += self.m[k] * np.dot(c - pts[i], c - pts[j]) + self.I[k]
        return M

    def rnea(self, q, qd, qdd):
        """Inverse dynamics: the joint torques that produce ``qdd``.

        Recursive Newton-Euler, planar version.  Forward pass propagates
        accelerations outward from the base, backward pass propagates forces
        inward from the tip.  Gravity is absent because the arm is horizontal.
        """
        n = self.n
        phi = np.cumsum(q)
        om = np.cumsum(qd)                    # absolute angular velocity
        al = np.cumsum(qdd)                   # absolute angular acceleration
        u = np.stack([np.cos(phi), np.sin(phi)], axis=1)          # along link
        v = np.stack([-np.sin(phi), np.cos(phi)], axis=1)         # 90 deg ccw

        a_joint = np.zeros((n + 1, 2))        # acceleration of each joint point
        a_com = np.zeros((n, 2))
        for k in range(n):
            # a point at distance d out along link k, relative to its own
            # joint: tangential al*d (along v) minus centripetal om^2*d (along u)
            a_com[k] = a_joint[k] + al[k] * self.r[k] * v[k] - om[k] ** 2 * self.r[k] * u[k]
            a_joint[k + 1] = a_joint[k] + al[k] * self.l[k] * v[k] - om[k] ** 2 * self.l[k] * u[k]

        f_child = np.zeros(2)                 # force the outer links push back with
        t_child = 0.0
        tau = np.zeros(n)
        pts = self.points(q)
        for k in range(n - 1, -1, -1):
            F = self.m[k] * a_com[k]                       # net force on link k
            f_here = F + f_child
            c = pts[k] + self.r[k] * u[k]
            t_here = (self.I[k] * al[k] + t_child
                      + _cross2(c - pts[k], F)
                      + _cross2(pts[k + 1] - pts[k], f_child))
            tau[k] = t_here
            f_child, t_child = f_here, t_here
        return tau

    # -- the same equations, written out for two links ----------------------
    #
    # The loops above are the readable reference; this is what actually runs.
    # For n = 2 the mass matrix and the velocity-product term have a textbook
    # closed form, and evaluating it in plain Python floats instead of small
    # NumPy arrays is ~20x faster -- which is the difference between a
    # reinforcement-learning project that finishes in three minutes and one
    # that does not.  ``verify_mujoco.py`` checks the two agree to 1e-12.
    def _fd2(self, q, qd, tau):
        (l1, _), (m1, m2), (r1, r2), (I1, I2) = self.l, self.m, self.r, self.I
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        m11 = I1 + m1 * r1 * r1 + I2 + m2 * (l1 * l1 + r2 * r2 + 2 * l1 * r2 * c2)
        m12 = I2 + m2 * (r2 * r2 + l1 * r2 * c2)
        m22 = I2 + m2 * r2 * r2
        h = m2 * l1 * r2 * s2
        q1d, q2d = qd[0], qd[1]
        b1 = -h * (2.0 * q1d * q2d + q2d * q2d) + self.b[0] * q1d
        b2 = h * q1d * q1d + self.b[1] * q2d
        y1, y2 = tau[0] - b1, tau[1] - b2
        det = m11 * m22 - m12 * m12
        return np.array([(m22 * y1 - m12 * y2) / det,
                         (m11 * y2 - m12 * y1) / det])

    def forward_dynamics(self, q, qd, tau):
        if self.n == 2:
            return self._fd2(q, qd, tau)
        bias = self.rnea(q, qd, np.zeros(self.n)) + self.b * qd
        return np.linalg.solve(self.mass_matrix(q), tau - bias)

    def servo_torque(self, q, qd, q_des):
        """Joint-space PD.  This is the 'motor controller' under the policy."""
        if self.auto_gains:
            w = self.omega_n
            tau = self.mass_matrix(q) @ (w * w * (q_des - q) - 2.0 * w * qd)
        else:
            tau = self.kp * (q_des - q) - self.kd * qd
        return np.clip(self.gear * tau, -self.tau_max, self.tau_max)

    def step(self, q, qd, tau, dt=DT):
        """One semi-implicit Euler step (velocity first, then position)."""
        qdd = self.forward_dynamics(q, qd, tau)
        qd = qd + dt * qdd
        q = q + dt * qd
        return q, qd

    def energy(self, q, qd):
        return 0.5 * qd @ self.mass_matrix(q) @ qd


# ---------------------------------------------------------------------------
# the task
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    mass_scale=1.0,      # link masses
    damp_scale=1.0,      # joint damping (viscous friction in the gearbox)
    gear=1.0,            # actuator gain: 1.0 means the motor is as strong as modelled
    slip=1.0,            # how much of a push the puck actually takes (table friction)
    latency=0,           # control-loop delay, in decisions
    obs_noise=0.0,       # metres of noise on the measured positions
)


class PushEnv:
    """Arm tip pushes a puck onto a goal disc.

    Action: 2 numbers in [-1, 1], a joint-angle DELTA (times ``DQ_MAX``) added
    to the current commanded pose and tracked by the PD servo.  Position-style
    actions are used rather than raw torques for the reason project 52 measured
    the hard way: a torque policy must learn to fight the arm's own inertia
    before it can learn the task, while a position policy gets that from the
    servo for free.
    """

    obs_dim = 16
    act_dim = 2

    def __init__(self, rng, params=None, obstacle=None, arm=None, feat="rel",
                 randomize=None):
        self.rng = rng
        self.feat = feat
        self.obs_dim = 16 if feat == "rel" else 12
        self.base_params = dict(DEFAULT_PARAMS)
        if params:
            self.base_params.update(params)
        self.p = dict(self.base_params)
        # ``randomize`` maps a parameter name to a (low, high) range that is
        # resampled at every reset -- domain randomisation, used by project 59.
        # Nothing tells the policy which robot it is on; that is the point.
        self.randomize = randomize or {}
        self.fixed_arm = arm
        self.arm = arm or PlanarArm(mass_scale=self.p["mass_scale"],
                                    damp_scale=self.p["damp_scale"],
                                    gear=self.p["gear"])
        self.obstacle = obstacle          # (x, y, radius) or None
        self.n = self.arm.n

    # -- episode ------------------------------------------------------------
    def _resample_robot(self):
        """Draw a new robot for this episode."""
        self.p = dict(self.base_params)
        for k, (lo, hi) in self.randomize.items():
            v = self.rng.uniform(lo, hi)
            self.p[k] = int(round(v)) if k == "latency" else float(v)
        if self.fixed_arm is None:
            self.arm = PlanarArm(mass_scale=self.p["mass_scale"],
                                 damp_scale=self.p["damp_scale"],
                                 gear=self.p["gear"])

    def reset(self, puck=None, goal=None):
        if self.randomize:
            self._resample_robot()
        a = self.arm
        # Keep the whole task well inside the workspace.  A puck at 0.8 of the
        # arm's reach sounds harmless, but the tip has to swing 7 cm further
        # out to get behind it, and there the arm is nearly straight -- the
        # Jacobian is singular, damped least squares returns almost no motion,
        # and the demonstrator simply stalls (project 05's lesson, met again).
        lo, hi = 0.40 * a.reach, 0.66 * a.reach
        for _ in range(200):
            if puck is None:
                th = self.rng.uniform(-0.45, 1.25)
                rr = self.rng.uniform(lo, hi)
                pk = np.array([rr * np.cos(th), rr * np.sin(th)])
            else:
                pk = np.asarray(puck, float)
            if goal is None:
                ang = self.rng.uniform(0, 2 * np.pi)
                dist = self.rng.uniform(0.09, 0.15)
                gl = pk + dist * np.array([np.cos(ang), np.sin(ang)])
            else:
                gl = np.asarray(goal, float)
            if lo * 0.9 < np.linalg.norm(gl) < hi * 1.05:
                break
        self.puck, self.goal = pk, gl

        # start the arm folded back away from the puck, so that every episode
        # begins with a real approach rather than an accidental contact
        q = np.zeros(self.n)
        q[0] = np.arctan2(pk[1], pk[0]) - 0.9
        if self.n > 1:
            q[1] = 1.5
        self.q = q
        self.qd = np.zeros(self.n)
        self.q_cmd = self.q.copy()
        while np.linalg.norm(self.arm.tip(self.q) - self.puck) < R_PUCK + R_TIP + 0.02:
            self.q[1] += 0.15
            self.q_cmd = self.q.copy()
        self.t = 0
        self.hit_obstacle = False
        self.diverged = False
        self._queue = [np.zeros(self.act_dim)] * self.p["latency"]
        return self.obs()

    def obs(self):
        """What the policy sees.

        ``feat="rel"`` appends two difference vectors -- tip-to-puck and
        puck-to-goal -- on top of the absolute positions that are already
        there.  That looks redundant, because a network *can* subtract two of
        its own inputs.  It matters anyway, and by a lot (project 54 measures
        4x in success rate): what the task depends on is only the differences,
        and forcing the first layer to discover subtraction from a few thousand
        samples spends capacity on arithmetic instead of on the task.  Handing
        it the difference costs one line and no information.
        """
        n = self.p["obs_noise"]
        jit = (lambda v: v + self.rng.normal(0, n, 2)) if n > 0 else (lambda v: v)
        tip, puck = jit(self.arm.tip(self.q)), jit(self.puck)
        base = [np.cos(self.q), np.sin(self.q), self.qd / 10.0,
                tip, puck, self.goal]
        if self.feat == "rel":
            base += [puck - tip, self.goal - puck]
        return np.concatenate(base)

    def state(self):
        """Everything needed to restore the episode exactly (project 55)."""
        return (self.q.copy(), self.qd.copy(), self.q_cmd.copy(),
                self.puck.copy(), self.goal.copy(), self.t)

    def set_state(self, s):
        self.q, self.qd, self.q_cmd, self.puck, self.goal, self.t = (
            s[0].copy(), s[1].copy(), s[2].copy(), s[3].copy(), s[4].copy(), s[5])
        self._queue = [np.zeros(self.act_dim)] * self.p["latency"]

    # -- physics ------------------------------------------------------------
    def _push_puck(self, tip_prev, tip_now):
        """Quasi-static pushing: the puck is displaced out of the tip.

        Real pushing is a contact-dynamics problem; on a table at low speed the
        puck simply goes where the finger sweeps it, so we resolve the overlap
        instead of integrating a contact force.  ``slip`` < 1 models a table
        that lets the puck lag behind the finger.
        """
        d = self.puck - tip_now
        dist = np.linalg.norm(d)
        rsum = R_PUCK + R_TIP
        if dist < rsum:
            if dist < 1e-9:
                d = tip_now - tip_prev
                dist = np.linalg.norm(d) + 1e-9
            self.puck = self.puck + self.p["slip"] * (rsum - dist) * d / dist
            if self.obstacle is not None:
                ox, oy, orad = self.obstacle
                dd = self.puck - np.array([ox, oy])
                nn = np.linalg.norm(dd)
                if nn < orad + R_PUCK:
                    self.puck = np.array([ox, oy]) + dd / (nn + 1e-9) * (orad + R_PUCK)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        if self.p["latency"]:
            self._queue.append(a)
            a = self._queue.pop(0)
        # The action is a delta from the MEASURED pose, not from the previous
        # command.  Accumulating commands (q_cmd += dq) looks equivalent and is
        # not: the servo lags by ~25% of every step, so an accumulator drifts
        # further and further ahead of the real arm until the commands stop
        # meaning anything -- the same wind-up that bit the integral term in
        # project 08.  Measuring first resets that error every decision.
        self.q_cmd = self.q + DQ_MAX * a
        safe_q, safe_qd = self.q.copy(), self.qd.copy()
        for _ in range(SUBSTEPS):
            tip_prev = self.arm.tip(self.q)
            tau = self.arm.servo_torque(self.q, self.qd, self.q_cmd)
            self.q, self.qd = self.arm.step(self.q, self.qd, tau)
            tip_now = self.arm.tip(self.q)
            self._push_puck(tip_prev, tip_now)
            if self.obstacle is not None:
                ox, oy, orad = self.obstacle
                if np.linalg.norm(tip_now - np.array([ox, oy])) < orad + R_TIP:
                    self.hit_obstacle = True
        # A robot whose simulated state has run off to infinity is not a hard
        # robot, it is a broken integration -- project 59 randomises the
        # physics hard enough to reach that corner.  Catching it here keeps
        # NaNs out of the training data, where they would silently poison the
        # input normaliser and every policy trained afterwards.
        if not np.all(np.isfinite(self.q)) or np.abs(self.qd).max() > 1e4:
            self.diverged = True
            self.q, self.qd = safe_q, np.zeros_like(safe_qd)
        self.t += 1
        err = float(np.linalg.norm(self.puck - self.goal))
        done = self.t >= EP_LEN or err < GOAL_TOL or self.diverged
        ok = err < GOAL_TOL and not self.diverged
        return self.obs(), -err, done, {"err": err, "success": ok}

    # -- convenience --------------------------------------------------------
    @property
    def err(self):
        return float(np.linalg.norm(self.puck - self.goal))

    @property
    def success(self):
        return self.err < GOAL_TOL and not self.hit_obstacle and not self.diverged


# ---------------------------------------------------------------------------
# the demonstrator
# ---------------------------------------------------------------------------
CIRCLE_R = R_PUCK + R_TIP + 0.030      # radius the tip keeps while going around
APPROACH_TOL = 0.50                    # radians of "close enough to behind"
V_MAX = 0.45                           # m/s the demonstrator lets the tip move


def expert_target(env, side):
    """Where the expert wants the tip next, in table coordinates.

    Two phases.  If the tip is not yet behind the puck (opposite the goal), it
    walks around the puck on a safe circle -- in direction ``side``, which is
    the *choice* that makes the demonstrations multimodal.  Once it is behind,
    it drives straight through the puck towards the goal.
    """
    tip = env.arm.tip(env.q)
    to_goal = env.goal - env.puck
    g_hat = to_goal / (np.linalg.norm(to_goal) + 1e-9)
    behind_ang = np.arctan2(-g_hat[1], -g_hat[0])
    rel = tip - env.puck
    tip_ang = np.arctan2(rel[1], rel[0])
    # How far the tip still has to travel *in the direction it chose*.  Using
    # the shortest signed difference here instead would be a real bug: an
    # expert going the long way round would clamp its waypoint to the short
    # way, sail past the target angle, and orbit the puck forever.
    travel = (side * (behind_ang - tip_ang)) % (2 * np.pi)

    if APPROACH_TOL < travel < 2 * np.pi - APPROACH_TOL:
        # Go around.  The waypoint is placed well ahead along the circle (more
        # than one step's worth) so that the tip is always chasing something it
        # cannot reach: that is what keeps the commanded speed at the maximum
        # instead of dying away as it converges on a nearby point.
        step_ang = tip_ang + side * min(0.8, travel)
        return env.puck + CIRCLE_R * np.array([np.cos(step_ang), np.sin(step_ang)]), "circle"
    contact = env.puck - (R_PUCK + R_TIP) * g_hat
    if np.linalg.norm(tip - contact) > 0.015:
        return contact, "align"
    return env.puck + 0.06 * g_hat, "push"


def expert_action(env, side=1, noise=0.0, rng=None):
    """Scripted teleoperator: a task-space target turned into a joint delta.

    The tip velocity is mapped to joint velocity with damped least squares --
    the same regulariser as project 05 -- so that a target near a singular arm
    configuration produces a small sane motion instead of an explosion.
    """
    target, phase = expert_target(env, side)
    tip = env.arm.tip(env.q)
    v = (target - tip) * 12.0                             # desired tip velocity
    sp = np.linalg.norm(v)
    if sp > V_MAX:
        v = v * (V_MAX / sp)
    J = env.arm.jacobian(env.q)
    lam = 0.05
    dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(2), v * CTRL_DT)
    a = dq / DQ_MAX
    if noise and rng is not None:
        a = a + rng.normal(0, noise, a.shape)
    return np.clip(a, -1.0, 1.0), phase


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------
def rollout(env, policy=None, side=None, noise=0.0, record=False, seed_reset=True,
            puck=None, goal=None):
    """Run one episode.  ``policy`` None means "use the expert"."""
    if seed_reset:
        obs = env.reset(puck=puck, goal=goal)
    else:
        obs = env.obs()
    if side is None:
        side = 1 if env.rng.random() < 0.5 else -1
    O, A, states, tips, pucks = [], [], [], [], []
    done = False
    while not done:
        if policy is None:
            a, _ = expert_action(env, side=side, noise=noise, rng=env.rng)
        else:
            a = policy(obs)
        if record:
            O.append(obs.copy())
            A.append(np.asarray(a, float).copy())
            states.append(env.state())
            tips.append(env.arm.tip(env.q).copy())
            pucks.append(env.puck.copy())
        obs, _, done, info = env.step(a)
    out = dict(success=bool(env.success), err=env.err, steps=env.t, side=side)
    if record:
        out.update(obs=np.array(O), act=np.array(A), states=states,
                   tips=np.array(tips), pucks=np.array(pucks))
    return out


def collect_demos(n_demos, seed=0, noise=0.0, params=None, obstacle=None,
                  side_mode="random", only_success=True, feat="rel", arm=None):
    """Record ``n_demos`` expert episodes -> (obs, act) arrays plus metadata.

    ``noise`` shakes the demonstrator's hand.  A noisy demonstration is worse
    on its own terms, but it visits states a clean one never does, and those
    are exactly the states a cloned policy will drift into (project 54,
    experiment 5).
    """
    rng = np.random.default_rng(seed)
    env = PushEnv(rng, params=params, obstacle=obstacle, feat=feat, arm=arm)
    O, A, sides, oks = [], [], [], []
    tries = 0
    while len(sides) < n_demos and tries < n_demos * 5:
        tries += 1
        if side_mode == "random":
            side = 1 if rng.random() < 0.5 else -1
        else:
            side = int(side_mode)
        r = rollout(env, None, side=side, noise=noise, record=True)
        if only_success and not r["success"]:
            continue
        O.append(r["obs"])
        A.append(r["act"])
        sides.append(side)
        oks.append(r["success"])
    return (np.concatenate(O).astype(np.float32),
            np.concatenate(A).astype(np.float32),
            dict(n=len(sides), sides=np.array(sides), success=float(np.mean(oks)),
                 tries=tries))


def evaluate(policy, n=60, seed=1000, params=None, obstacle=None, arm=None,
             feat="rel"):
    """Success rate of a policy over ``n`` fresh episodes."""
    rng = np.random.default_rng(seed)
    env = PushEnv(rng, params=params, obstacle=obstacle, arm=arm, feat=feat)
    ok, errs = 0, []
    for _ in range(n):
        r = rollout(env, policy)
        ok += r["success"]
        errs.append(r["err"])
    return dict(success=ok / n, err=float(np.mean(errs)))


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    arm = PlanarArm()

    # 1. energy: with zero damping and zero torque, kinetic energy is constant
    a2 = PlanarArm(damping=(0.0, 0.0))
    q, qd = np.array([0.3, 0.7]), np.array([1.5, -2.0])
    e0 = a2.energy(q, qd)
    for _ in range(2000):
        q, qd = a2.step(q, qd, np.zeros(2), dt=1e-4)
    print(f"energy drift over 0.2 s: {abs(a2.energy(q, qd) - e0) / e0:.2e}")

    # 2. the servo tracks a step
    env = PushEnv(rng)
    env.reset()
    t0 = time.time()
    r = rollout(env, None, side=1, record=True)
    print(f"expert episode: success={r['success']} steps={r['steps']} "
          f"{(time.time() - t0) * 1000:.1f} ms")

    # 3. expert success rate, both circling directions
    for side in (1, -1):
        ok = np.mean([rollout(env, None, side=side)["success"] for _ in range(50)])
        print(f"expert side={side:+d}: success {ok:.2f}")
