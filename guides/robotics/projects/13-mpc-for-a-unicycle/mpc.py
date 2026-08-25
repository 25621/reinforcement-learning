"""A nonlinear MPC solver, written out by hand.

The guide reaches for CasADi here.  CasADi is not installed in this
environment, so this file does the two jobs CasADi would have done:

  * it differentiates the rolled-out trajectory with respect to the control
    sequence (CasADi does this by automatic differentiation; we do it by
    finite differences, vectorised so all the perturbations run at once);
  * it runs a nonlinear least-squares solver over the result (CasADi would
    hand the problem to IPOPT; we use Gauss-Newton with Levenberg-Marquardt
    damping, which is enough for a tracking cost and is 60 lines).

Writing it out is worth the trouble once, because the shape of the problem is
the whole idea of MPC and it is easy to lose behind a library API.

THE PROBLEM.  At each control tick, with the robot at state ``x0``:

    choose u_0 ... u_{N-1}  to minimise
        sum_k  ||p_k - ref_k||^2_Q  +  ||u_k||^2_R  +  ||u_k - u_{k-1}||^2_Rd
             + ||p_N - ref_N||^2_Qf
    subject to the model x_{k+1} = f(x_k, u_k) and the box limits on u

then APPLY ONLY u_0, throw the rest away, and re-solve next tick with a fresh
measurement.  Re-solving is what makes it feedback: the plan is always made
from where the robot actually is, so modelling error never accumulates into
the plan.  This is called "receding horizon" -- the horizon keeps sliding
forward, so the controller is always looking N steps ahead and never runs out
of future.

WHY GAUSS-NEWTON.  Write the cost as a sum of squares of a residual vector
``r(U)``.  The exact second derivative of such a cost is ``J^T J + (curvature
of r)``; Gauss-Newton simply drops the second term.  That approximation is
excellent when the residual is small at the solution -- which is exactly the
situation in tracking, where you expect to nearly hit the reference -- and it
buys a Hessian that costs nothing extra and is automatically positive
semi-definite, so the step always points downhill.

WHY THE ``lambda I``.  Same Levenberg-Marquardt idea as the damped-least-squares
IK of project 5: near a bad configuration ``J^T J`` is nearly singular and the
raw step is enormous.  Adding ``lambda I`` bounds the step.  Here lambda also
adapts: it shrinks when a step improves the cost (trust the model more) and
grows when it does not (take smaller, safer steps).
"""

import numpy as np


def unicycle_step(x, u, dt):
    """One RK4 step of xdot = [v cos(theta), v sin(theta), omega].

    A "unicycle" is the simplest wheeled model there is: a point that can drive
    forwards and turn on the spot, and can NEVER slide sideways.  That last
    restriction is the nonholonomic constraint, and it is what makes parking a
    car hard: the set of reachable poses is not simply "everywhere nearby".

    ``x`` may be a single state ``(3,)`` or a batch ``(B, 3)`` -- the batch form
    is what lets the finite-difference Jacobian roll out every perturbation
    simultaneously.
    """
    x = np.atleast_2d(x)
    u = np.atleast_2d(u)

    def f(s):
        return np.stack([u[:, 0] * np.cos(s[:, 2]),
                         u[:, 0] * np.sin(s[:, 2]),
                         u[:, 1]], axis=1)

    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


class UnicycleMPC:
    """Receding-horizon tracking controller for the unicycle."""

    def __init__(self, N=15, dt=0.1, q_pos=6.0, q_theta=0.4, r_v=0.05, r_w=0.05,
                 rd=0.6, qf=3.0, v_lim=(-0.4, 1.6), w_lim=(-2.5, 2.5),
                 gn_iters=4, lam0=1e-2):
        self.N, self.dt = N, dt
        self.q_pos, self.q_theta, self.qf = q_pos, q_theta, qf
        self.r_v, self.r_w, self.rd = r_v, r_w, rd
        self.v_lim, self.w_lim = v_lim, w_lim
        self.gn_iters, self.lam0 = gn_iters, lam0
        self.U = np.zeros((N, 2))  # warm start, carried between ticks
        self.last_cost = np.nan
        self.last_iters = 0

    # -- the cost, written as a residual vector ---------------------------
    def _rollout(self, x0, U):
        """Simulate the plan forward.  ``U`` is ``(B, N, 2)``; returns ``(B, N, 3)``."""
        B = U.shape[0]
        x = np.repeat(np.atleast_2d(x0), B, axis=0)
        out = np.empty((B, self.N, 3))
        for k in range(self.N):
            x = unicycle_step(x, U[:, k, :], self.dt)
            out[:, k, :] = x
        return out

    def _residual(self, x0, U, ref, u_prev):
        """Stack every squared term of the cost as one vector of residuals."""
        X = self._rollout(x0, U)
        B = U.shape[0]
        wp = np.sqrt(self.q_pos)
        wt = np.sqrt(self.q_theta)
        parts = [
            (wp * (X[:, :, :2] - ref[None, :, :2])).reshape(B, -1),
            (wt * wrap(X[:, :, 2] - ref[None, :, 2])).reshape(B, -1),
            (np.sqrt(self.qf) * (X[:, -1, :2] - ref[None, -1, :2])).reshape(B, -1),
            (np.sqrt(self.r_v) * U[:, :, 0]).reshape(B, -1),
            (np.sqrt(self.r_w) * U[:, :, 1]).reshape(B, -1),
        ]
        du = np.diff(np.concatenate([np.repeat(u_prev[None, None, :], B, axis=0), U], axis=1), axis=1)
        parts.append((np.sqrt(self.rd) * du).reshape(B, -1))
        return np.concatenate(parts, axis=1)

    def _clip(self, U):
        U = U.copy()
        U[..., 0] = np.clip(U[..., 0], *self.v_lim)
        U[..., 1] = np.clip(U[..., 1], *self.w_lim)
        return U

    # -- one solve --------------------------------------------------------
    def solve(self, x0, ref, u_prev):
        """Return the first control of the optimal plan, given ``ref`` (N x 3)."""
        n_u = 2 * self.N
        U = self._clip(self.U)
        lam = self.lam0
        r = self._residual(x0, U[None], ref, u_prev)[0]
        cost = float(r @ r)

        eps = 1e-5
        for it in range(self.gn_iters):
            # Finite-difference Jacobian: perturb every control variable once.
            # All 2N perturbed roll-outs run in a single batched pass, which is
            # what makes a from-scratch solver fast enough to sit in the loop.
            batch = np.repeat(U[None], n_u + 1, axis=0)
            flat = batch.reshape(n_u + 1, -1)
            flat[1:] += eps * np.eye(n_u)
            R = self._residual(x0, flat.reshape(n_u + 1, self.N, 2), ref, u_prev)
            J = (R[1:] - R[0][None]) / eps  # (n_u, n_r)
            J = J.T  # (n_r, n_u)

            g = J.T @ r
            H = J.T @ J
            for _ in range(8):
                try:
                    step = np.linalg.solve(H + lam * np.eye(n_u), -g)
                except np.linalg.LinAlgError:
                    lam *= 10
                    continue
                U_try = self._clip(U + step.reshape(self.N, 2))
                r_try = self._residual(x0, U_try[None], ref, u_prev)[0]
                c_try = float(r_try @ r_try)
                if c_try < cost:
                    U, r, cost = U_try, r_try, c_try
                    lam = max(lam * 0.4, 1e-6)
                    break
                lam *= 4.0
            else:
                break
            self.last_iters = it + 1

        # Shift the plan by one step for the next tick.  Tomorrow's problem is
        # almost today's, so starting from today's answer usually converges in
        # one or two iterations instead of ten.
        self.U = np.vstack([U[1:], U[-1:]])
        self.last_cost = cost
        return U[0].copy(), U


def wrap(a):
    """Fold an angle difference into (-pi, pi].

    Without this an error of +359 degrees looks enormous when the true
    correction is -1 degree, and the controller spins the long way round.
    """
    return (a + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# The reference path
# ---------------------------------------------------------------------------
def figure_eight(t, a=2.0, period=20.0):
    """A lemniscate, plus the heading that traces it.

    x = a sin(s), y = a sin(s) cos(s) with s = 2 pi t / period.  The crossing in
    the middle is the interesting part: the robot passes through the same POINT
    twice with opposite headings, so a controller that only chases the nearest
    point on the path will get confused there.
    """
    s = 2 * np.pi * np.asarray(t) / period
    x = a * np.sin(s)
    y = a * np.sin(s) * np.cos(s)
    w = 2 * np.pi / period
    dx = a * w * np.cos(s)
    dy = a * w * (np.cos(s) ** 2 - np.sin(s) ** 2)
    return np.stack([x, y, np.arctan2(dy, dx)], axis=-1)


def reference_window(t0, N, dt, **kw):
    """The next N reference poses, as the MPC wants them."""
    return figure_eight(t0 + dt * np.arange(1, N + 1), **kw)


# ---------------------------------------------------------------------------
# Baselines to compare against
# ---------------------------------------------------------------------------
def pure_pursuit(x, t, lookahead=0.6, v_nom=1.0, k_w=2.5, w_lim=(-2.5, 2.5), **kw):
    """Aim at a point on the path a fixed distance ahead and turn toward it.

    The classic, and the reason it is a fair baseline: it uses exactly the same
    information as the MPC (the robot's pose and the path ahead) but commits to
    one geometric rule instead of optimising.
    """
    ts = t + np.linspace(0.0, 3.0, 60)
    pts = figure_eight(ts, **kw)
    d = np.linalg.norm(pts[:, :2] - x[:2], axis=1)
    idx = np.argmax(d >= lookahead) if np.any(d >= lookahead) else len(d) - 1
    target = pts[idx]
    heading = np.arctan2(target[1] - x[1], target[0] - x[0])
    w = float(np.clip(k_w * wrap(heading - x[2]), *w_lim))
    return np.array([v_nom, w])
