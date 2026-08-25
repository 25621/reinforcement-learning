"""A control barrier function for the arm tip, and the tiny QP that enforces it.

The setting is project 54's push task with a round obstacle dropped into the
scene.  A behaviour-cloned policy trained *without* the obstacle drives the tip
straight through it, because nothing in its training data ever punished that.
We do not retrain.  We wrap it.

The barrier
-----------
Define one number that is positive exactly when the robot is safe::

    h(q) = || tip(q) - c || - r_safe

``c`` is the obstacle centre and ``r_safe`` its radius plus the tip radius plus
a margin.  ``h > 0`` means "outside"; ``h = 0`` is the surface; ``h < 0`` means
"already inside", which we never want to reach.

A *barrier* function is named after the barriers used in optimisation, where a
term is added to a cost so that it blows up at the edge of the feasible region
and the optimiser is repelled from it.  Here nothing blows up; instead we add
one linear inequality that makes approaching the edge slower and slower.  The
"control" in *control barrier function* is the part that matters for us: the
inequality is written in terms of the control input, so it can be enforced by
choosing the control.

First order: constrain the speed
--------------------------------
The standard CBF condition is::

    hdot >= -alpha * h

In words: **you may move towards the obstacle, but the closer you get, the
slower you must approach.** When ``h`` is large the constraint is loose. When
``h`` reaches zero the constraint says ``hdot >= 0`` -- do not get any closer.
``alpha`` sets how late you are allowed to brake: large ``alpha`` is a late,
aggressive brake, small ``alpha`` is a cautious one.

Second order: the arm has mass
------------------------------
The action here is a joint-position command, and the arm follows it through a
PD servo and its own inertia.  So the action does not set the tip velocity; it
sets the tip *acceleration*.  ``h`` therefore has to be differentiated twice
before the action appears at all -- its **relative degree** is 2.  ("Degree"
here counts differentiations, the same way the degree of a polynomial counts
multiplications.)  The first-order condition above is then not enforceable: it
constrains a quantity the action cannot change this instant.

The fix is a **high-order CBF**.  Build an intermediate barrier::

    psi(q, qd) = hdot + alpha1 * h

and require ``psidot >= -alpha2 * psi``.  ``psidot`` contains ``qddot``, which
is affine in the action, so this one *is* enforceable.  Keeping ``psi >= 0``
keeps ``hdot >= -alpha1 * h``, which keeps ``h >= 0``: two nested brakes, the
outer one on distance and the inner one on closing speed.

The QP
------
Both conditions end up in the same shape: one linear inequality ``g . a >= b``
on the two-number action.  The safety filter is then::

    minimise ||a - a_policy||^2  subject to  g . a >= b

which is called a *quadratic program* (quadratic cost, linear constraints).
With a single constraint the solution is a one-line projection onto a
half-plane -- no solver needed.  This is the honest version of "the filter
changes the command as little as possible": least-squares distance is the
definition of "as little as possible" being used, and it is a choice.
"""

import os
import sys

import numpy as np

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402

OBST_R = 0.035             # obstacle radius
MARGIN = 0.010             # how much clear air we insist on


def h_and_grad(arm, q, centre, r_safe):
    """h(q), and dh/dq as a row vector."""
    tip = arm.tip(q)
    d = tip - centre
    dist = float(np.linalg.norm(d))
    nhat = d / max(dist, 1e-9)
    J = arm.jacobian(q)
    return dist - r_safe, nhat @ J, nhat, dist, J


def project_halfplane(a, g, b):
    """Closest point to ``a`` with ``g . a >= b``.  The whole QP, in one line."""
    gg = float(g @ g)
    if gg < 1e-12:
        return a, False
    slack = float(g @ a) - b
    if slack >= 0:
        return a, False
    return a + g * (-slack / gg), True


class SafetyFilter:
    """Wraps any policy action and returns a filtered one."""

    def __init__(self, arm, centre, r_safe, order=2, alpha=6.0, alpha2=None,
                 clip=True):
        self.arm = arm
        self.c = np.array(centre, float)
        self.r_safe = r_safe
        self.order = order
        self.alpha = alpha
        self.alpha2 = alpha2 if alpha2 is not None else alpha
        self.clip = clip
        self.n_active = 0
        self.n_calls = 0
        self.n_clipped = 0

    # -- the two constraint builders ----------------------------------------
    def _first_order(self, q, qd, dt):
        """hdot >= -alpha h, with hdot read off the COMMANDED joint velocity.

        A position command ``q + DQ_MAX*a`` held for one control period is, to
        the servo, a request to move at ``DQ_MAX*a/dt``.  Treating that request
        as the actual velocity is the shortcut almost every first CBF
        implementation takes, and section 2 measures what it costs.
        """
        h, dh, _, _, _ = h_and_grad(self.arm, q, self.c, self.r_safe)
        g = dh * (A.DQ_MAX / dt)
        return g, -self.alpha * h

    def _second_order(self, q, qd, dt):
        """psidot >= -alpha2 psi, with the servo's real acceleration."""
        arm = self.arm
        h, dh, nhat, dist, J = h_and_grad(arm, q, self.c, self.r_safe)
        v = J @ qd                                    # tip velocity
        hdot = float(nhat @ v)
        # d(nhat)/dt . v  =  (v . v - (nhat.v)^2) / dist
        curv = float(v @ v - hdot * hdot) / max(dist, 1e-6)
        # Jdot @ qd, by the planar chain rule (finite difference is fine here
        # and keeps the code short; the error is O(eps) on a smooth Jacobian).
        eps = 1e-6
        Jd = (arm.jacobian(q + eps * qd) - J) / eps
        drift_acc = Jd @ qd
        # qddot = M^-1 (K (q_cmd - q) - D qd - bias), with q_cmd - q = DQ_MAX*a
        M = arm.mass_matrix(q)
        w = arm.omega_n
        # servo_torque with auto gains is M (w^2 e - 2 w qd); with fixed gains
        # it is kp e - kd qd.  Either way the part multiplying ``a`` is:
        if arm.auto_gains:
            K = M * (w * w)
            Dm = M * (2.0 * w)
        else:
            K = np.diag(np.atleast_1d(arm.kp) * np.ones(arm.n))
            Dm = np.diag(np.atleast_1d(arm.kd) * np.ones(arm.n))
        Minv = np.linalg.inv(M)
        bias = arm.rnea(q, qd, np.zeros(arm.n)) + arm.b * qd
        acc_free = Minv @ (arm.gear * (-Dm @ qd) - bias)
        acc_gain = Minv @ (arm.gear * K) * A.DQ_MAX          # n x n
        # hddot = curv + nhat . (J qddot + Jdot qd)
        base = curv + float(nhat @ (J @ acc_free + drift_acc))
        g = nhat @ (J @ acc_gain)
        psi = hdot + self.alpha * h
        b = -self.alpha2 * psi - self.alpha * hdot - base
        return g, b

    # -- the filter ----------------------------------------------------------
    def __call__(self, a_nom, q, qd, dt=A.CTRL_DT):
        self.n_calls += 1
        a_nom = np.asarray(a_nom, float)
        if self.order == 1:
            g, b = self._first_order(q, qd, dt)
        else:
            g, b = self._second_order(q, qd, dt)
        a, active = project_halfplane(a_nom, g, b)
        if active:
            self.n_active += 1
        if self.clip:
            a_clipped = np.clip(a, -1.0, 1.0)
            if np.any(np.abs(a - a_clipped) > 1e-9):
                self.n_clipped += 1
            a = a_clipped
        return a


# ---------------------------------------------------------------------------
# scenes and scoring
# ---------------------------------------------------------------------------
SAFE_START = OBST_R + A.R_TIP + MARGIN + 0.020   # clearance the episode starts with


def obstacle_between(env, rng):
    """Put the obstacle in the arc the tip has to sweep, not on top of it.

    To push the puck the tip must first travel round *behind* it -- behind
    meaning "on the far side from the goal" -- and that arc is where an
    obstacle is genuinely in the way.  So the obstacle is placed relative to
    the puck, in a direction within about 20-75 degrees of straight behind, at
    a radius the tip would otherwise sweep through.

    Three conditions, and all three are load-bearing:

    * on the sweep, or the obstacle is scenery and the project measures nothing;
    * off to one side rather than dead centre, because dead centre makes the
      task impossible rather than hard and every method then scores near zero;
    * **outside the barrier at t = 0.** A control barrier function promises to
      keep you inside the safe set, not to get you back into it.  An episode
      that begins with ``h < 0`` is unsalvageable by construction, and a scene
      generator that quietly produces those turns the whole experiment into a
      measurement of how often it produced them.  This one rejects them, and
      ``run_episode`` reports the starting clearance so the rejection can be
      checked rather than trusted.
    """
    tip = env.arm.tip(env.q)
    back = env.puck - env.goal
    back = back / (np.linalg.norm(back) + 1e-9)
    best = None
    for _ in range(200):
        phi = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 1.60)
        r = rng.uniform(0.055, 0.100)
        ca, sa = np.cos(phi), np.sin(phi)
        u = np.array([ca * back[0] - sa * back[1], sa * back[0] + ca * back[1]])
        c = env.puck + r * u
        if (np.linalg.norm(c - tip) > SAFE_START
                and np.linalg.norm(c - env.goal) > OBST_R + A.R_PUCK + 0.020
                and 0.40 * 0.85 * env.arm.reach < np.linalg.norm(c)
                < 0.66 * 1.15 * env.arm.reach):
            return c
        best = c if best is None else best
    return best


def run_episode(env, policy, make_filter=None, record=False, rate="servo"):
    """One episode.  Returns a dict of outcomes, including SAFETY outcomes.

    ``rate`` decides how often the barrier is enforced:

    * ``"policy"`` -- once per decision, 20 Hz, filtering the action the policy
      just produced.  This is what "wrap the policy in a safety filter"
      literally says, and section 2 measures what it costs.
    * ``"servo"`` -- once per physics tick, 200 Hz, recomputing the constraint
      from the arm's current state each time.  A CBF's guarantee is a statement
      about continuous time; between two enforcement instants nothing holds it,
      and 50 ms is a long time for an arm moving at 0.4 m/s.

    The filter is built AFTER the reset, not before: the obstacle only exists
    once the episode has been laid out, and a filter constructed from the
    previous episode's obstacle would silently guard empty table.
    """
    obs = env.reset()
    filt = make_filter(env) if make_filter else None
    c = env.obstacle_c
    r_hit_tip = OBST_R + A.R_TIP
    r_hit_puck = OBST_R + A.R_PUCK
    start_h = float(np.linalg.norm(env.arm.tip(env.q) - c)) - r_hit_tip
    min_h_tip = 1e9
    tip_hit = puck_hit = False
    traj = []
    for _ in range(A.EP_LEN):
        a_nom = np.clip(np.asarray(policy(obs), float), -1, 1)
        if filt is None:
            obs, _, done, info = env.step(a_nom)
        elif rate == "policy":
            obs, _, done, info = env.step(filt(a_nom, env.q, env.qd,
                                               dt=A.CTRL_DT))
        else:
            done, info = _servo_rate_step(env, a_nom, filt)
            obs = env.obs()
        tip = env.arm.tip(env.q)
        dd = float(np.linalg.norm(tip - c))
        min_h_tip = min(min_h_tip, dd - r_hit_tip)
        tip_hit |= dd < r_hit_tip
        puck_hit |= float(np.linalg.norm(env.puck - c)) < r_hit_puck
        if record:
            traj.append((tip.copy(), env.puck.copy()))
        if done:
            break
    return {"success": bool(info["success"]), "err": float(info["err"]),
            "tip_hit": tip_hit, "puck_hit": puck_hit, "start_h": start_h,
            "min_h": float(min_h_tip), "traj": traj, "filt": filt}


def _servo_rate_step(env, a_nom, filt):
    """One decision, with the barrier re-enforced at every physics tick."""
    for _ in range(A.SUBSTEPS):
        a = filt(a_nom, env.q, env.qd, dt=A.DT)
        env.q_cmd = env.q + A.DQ_MAX * a
        tip_prev = env.arm.tip(env.q)
        tau = env.arm.servo_torque(env.q, env.qd, env.q_cmd)
        env.q, env.qd = env.arm.step(env.q, env.qd, tau)
        env._push_puck(tip_prev, env.arm.tip(env.q))
    env.t += 1
    err = float(np.linalg.norm(env.puck - env.goal))
    done = env.t >= A.EP_LEN or err < A.GOAL_TOL
    return done, {"err": err, "success": err < A.GOAL_TOL}


class ObstacleEnv(A.PushEnv):
    """PushEnv with a circular obstacle that the TIP must avoid.

    The base class already knows how to bounce the puck off an obstacle; what
    it does not do is place one on the robot's way every episode.  This
    subclass does, and it also records the centre so the barrier function has
    something to point at.
    """

    def __init__(self, rng, **kw):
        super().__init__(rng, **kw)
        self.obstacle_c = np.zeros(2)

    def reset(self, **kw):
        obs = super().reset(**kw)
        self.obstacle_c = obstacle_between(self, self.rng)
        self.obstacle = (self.obstacle_c[0], self.obstacle_c[1], OBST_R)
        return obs


def evaluate(policy, n=60, seed=1000, make_filter=None, record_first=0,
             rate="servo"):
    rng = np.random.default_rng(seed)
    env = ObstacleEnv(rng)
    out = {"success": [], "tip_hit": [], "puck_hit": [], "min_h": [],
           "err": [], "start_h": []}
    trajs = []
    active = calls = clipped = 0
    for i in range(n):
        r = run_episode(env, policy, make_filter, record=i < record_first,
                        rate=rate)
        for k in out:
            out[k].append(r[k])
        if i < record_first:
            trajs.append((r["traj"], env.obstacle_c.copy(), env.goal.copy()))
        filt = r["filt"]
        if filt is not None:
            active += filt.n_active
            calls += filt.n_calls
            clipped += filt.n_clipped
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res["intervention_rate"] = active / max(calls, 1)
    res["clip_rate"] = clipped / max(calls, 1)
    res["trajs"] = trajs
    return res
