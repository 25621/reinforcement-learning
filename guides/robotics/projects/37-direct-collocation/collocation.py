"""Direct collocation for the cart-pole -- the shared library for project 37.

The state is x = [cart position, cart velocity, pole angle from UP, pole rate]
and the single input u is a horizontal force on the cart.  The plant is
project 09's `CartPole`; this file re-expresses the same equations in CasADi so
that IPOPT can differentiate them.

Why re-express them at all, when project 09 already has `deriv()` in NumPy?
Because IPOPT does not want the VALUE of the dynamics, it wants their
DERIVATIVES with respect to every one of the several hundred decision
variables, exactly and cheaply.  A NumPy function can only be differentiated by
finite differences -- slow, and noisy enough to stop a Newton-type solver from
converging.  CasADi builds a symbolic graph of the same formula and hands
IPOPT exact first and second derivatives.  Same physics, differentiable
packaging.
"""

import math
import time

import numpy as np

import casadi as ca

G = 9.81


def cartpole_ode(x, u, M=1.0, m=0.1, l=0.5):
    """The cart-pole equations, written so they accept CasADi symbols."""
    xd = x[1]
    th = x[2]
    thd = x[3]
    st = ca.sin(th)
    ct = ca.cos(th)
    xdd = (u + m * l * thd ** 2 * st - m * G * st * ct) / (M + m * st ** 2)
    thdd = (G * st - xdd * ct) / l
    return ca.vertcat(xd, xdd, thd, thdd)


X0 = np.array([0.0, 0.0, np.pi, 0.0])      # pole hanging straight down
XF = np.array([0.0, 0.0, 0.0, 0.0])        # pole balanced upright, at origin


def solve(N=60, T=2.0, u_max=20.0, method="trapezoidal", guess="linear",
          seed=0, free_time=False, x0=X0, xf=XF, w_u=1.0, w_T=0.0,
          max_iter=800, verbose=False, warm=None, T_hi=6.0):
    """Solve the swing-up as a nonlinear program.

    DIRECT collocation means the state trajectory is a decision variable in its
    own right, not something you get by simulating.  The dynamics appear as
    equality CONSTRAINTS linking consecutive knots.  That sounds wasteful --
    why solve for something you could compute? -- but it is the entire reason
    the method works on unstable systems: the solver may hold an intermediate
    guess that does not satisfy the physics at all, and repair it gradually.
    Simulation has no such freedom; every guess is physically exact and
    therefore every guess is stuck with wherever the instability takes it.

    "Collocation" is the classical name for "make the residual vanish at
    selected points" -- the collocation points.  Here the residual is
    (state derivative implied by the interpolating polynomial) minus (state
    derivative the physics demands), and we force it to zero at each knot.
    """
    opti = ca.Opti()
    X = opti.variable(4, N + 1)
    U = opti.variable(1, N)
    if free_time:
        Tvar = opti.variable()
        opti.subject_to(opti.bounded(0.4, Tvar, T_hi))
        opti.set_initial(Tvar, T)
    else:
        Tvar = T
    dt = Tvar / N

    for k in range(N):
        f_k = cartpole_ode(X[:, k], U[0, k])
        f_k1 = cartpole_ode(X[:, k + 1], U[0, k])
        if method == "trapezoidal":
            # Straight-line interpolation of the derivative between knots.
            # Local error ~ dt^3, so the whole trajectory is accurate to dt^2.
            opti.subject_to(X[:, k + 1] == X[:, k] + 0.5 * dt * (f_k + f_k1))
        elif method == "hermite-simpson":
            # A cubic through both knots, with the dynamics ALSO enforced at
            # the midpoint.  One extra constraint per interval buys two extra
            # orders of accuracy (dt^4 overall).  The midpoint state is not a
            # free variable here -- it is pinned to the cubic's value, which is
            # the "compressed" form of the method.
            xm = 0.5 * (X[:, k] + X[:, k + 1]) + dt / 8.0 * (f_k - f_k1)
            fm = cartpole_ode(xm, U[0, k])
            opti.subject_to(X[:, k + 1] == X[:, k] +
                            dt / 6.0 * (f_k + 4 * fm + f_k1))
        else:
            raise ValueError(method)

    opti.subject_to(X[:, 0] == x0)
    opti.subject_to(X[:, N] == xf)
    opti.subject_to(opti.bounded(-u_max, U, u_max))
    opti.subject_to(opti.bounded(-3.0, X[0, :], 3.0))     # keep it on the rail

    obj = w_u * ca.sumsqr(U) * (T / N if not free_time else 1.0 / N)
    if free_time:
        obj = obj + w_T * Tvar
    opti.minimize(obj)

    rng = np.random.default_rng(seed)
    if guess == "linear":
        g = np.linspace(x0, xf, N + 1).T
    elif guess == "zeros":
        g = np.zeros((4, N + 1))
    elif guess == "hold":
        g = np.tile(x0[:, None], (1, N + 1))
    elif guess == "random":
        g = np.linspace(x0, xf, N + 1).T + rng.normal(0, 1.5, (4, N + 1))
    else:
        raise ValueError(guess)
    if warm is not None:
        # Warm start from a previous solution.  This is "continuation": solve
        # an easy version of the problem, then use its answer as the guess for
        # a slightly harder one.  On the weak-motor sweep in experiment 5 it is
        # the difference between converging and not, because the solver never
        # has to invent a multi-pump manoeuvre from a straight-line guess.
        gx = np.stack([np.interp(np.linspace(0, 1, N + 1),
                                 np.linspace(0, 1, warm["X"].shape[1]),
                                 warm["X"][j]) for j in range(4)])
        gu = np.interp(np.linspace(0, 1, N),
                       np.linspace(0, 1, len(warm["U"])), warm["U"])
        opti.set_initial(X, gx)
        opti.set_initial(U, gu.reshape(1, N))
        if free_time:
            opti.set_initial(Tvar, warm["T"])
    else:
        opti.set_initial(X, g)
        opti.set_initial(U, rng.normal(0, 1.0, (1, N)) if guess == "random"
                         else np.zeros((1, N)))

    opts = {"print_time": False}
    iopts = {"print_level": 5 if verbose else 0, "max_iter": max_iter,
             "sb": "yes"}
    opti.solver("ipopt", opts, iopts)
    t0 = time.perf_counter()
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError:
        sol = opti.debug
        ok = False
    dur = time.perf_counter() - t0
    Tval = float(sol.value(Tvar)) if free_time else T
    return dict(ok=ok, X=np.array(sol.value(X)), U=np.atleast_1d(
        np.array(sol.value(U)).ravel()), T=Tval, N=N, method=method,
        seconds=dur, obj=float(sol.value(obj)),
        iters=int(opti.stats().get("iter_count", -1)))


def single_shooting(N=60, T=2.0, u_max=20.0, guess="zeros", seed=0,
                    max_iter=800, x0=X0, xf=XF):
    """The other way: only the controls are decision variables.

    The state is obtained by integrating forward from x0, so it is always
    physically exact -- and that is the problem.  The cart-pole hanging down is
    at an unstable equilibrium once it gets near the top, so a tiny change in
    an early control multiplies into a huge change at the end.  The solver sees
    a gradient hundreds of times larger for u[0] than for u[N-1], and that
    imbalance is what makes shooting fail on long horizons.
    """
    opti = ca.Opti()
    U = opti.variable(1, N)
    dt = T / N
    x = ca.DM(x0)
    for k in range(N):
        k1 = cartpole_ode(x, U[0, k])
        k2 = cartpole_ode(x + 0.5 * dt * k1, U[0, k])
        k3 = cartpole_ode(x + 0.5 * dt * k2, U[0, k])
        k4 = cartpole_ode(x + dt * k3, U[0, k])
        x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    opti.subject_to(x == xf)
    opti.subject_to(opti.bounded(-u_max, U, u_max))
    opti.minimize(ca.sumsqr(U) * dt)
    rng = np.random.default_rng(seed)
    opti.set_initial(U, rng.normal(0, 1.0, (1, N)) if guess == "random"
                     else np.zeros((1, N)))
    opti.solver("ipopt", {"print_time": False},
                {"print_level": 0, "max_iter": max_iter, "sb": "yes"})
    t0 = time.perf_counter()
    try:
        sol = opti.solve()
        ok = True
    except RuntimeError:
        sol = opti.debug
        ok = False
    return dict(ok=ok, U=np.atleast_1d(np.array(sol.value(U)).ravel()),
                T=T, N=N, seconds=time.perf_counter() - t0,
                iters=int(opti.stats().get("iter_count", -1)))


# ------------------------------------------------------------------ replay
def deriv_np(s, u, M=1.0, m=0.1, l=0.5):
    _, xd, th, thd = s
    st, ct = np.sin(th), np.cos(th)
    xdd = (u + m * l * thd * thd * st - m * G * st * ct) / (M + m * st * st)
    thdd = (G * st - xdd * ct) / l
    return np.array([xd, xdd, thd, thdd])


def rk4(s, u, dt):
    k1 = deriv_np(s, u)
    k2 = deriv_np(s + 0.5 * dt * k1, u)
    k3 = deriv_np(s + 0.5 * dt * k2, u)
    k4 = deriv_np(s + dt * k3, u)
    return s + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def replay(U, T, x0=X0, substeps=40, gain=None, Xref=None, u_max=None,
           gate_rad=0.5, hold=0.0):
    """Run the plan through an accurate integrator and see where it ends up.

    `gain` (an LQR matrix K) turns the open-loop plan into a closed-loop one:
    the applied force becomes u_plan - K (x - x_ref).

    `gate_rad` matters more than it looks.  The LQR gain is computed from the
    dynamics LINEARISED ABOUT THE UPRIGHT, so it is only meaningful when the
    pole is near upright.  Applying it during the swing-up -- where the pole is
    hanging at pi radians and the linearisation is nonsense -- actively wrecks
    the manoeuvre: measured, it drives the cart tens of metres down the rail.
    So feedback is switched on only once |theta| is inside `gate_rad`.

    `hold` continues the simulation for extra seconds after the plan runs out,
    with the reference held at the upright.  Open loop this means "apply zero
    force and watch"; with feedback it means "balance".
    """
    N = len(U)
    dt = T / N
    x = np.asarray(x0, float).copy()
    traj = [x.copy()]
    applied = []

    def wrap(a):
        return math.atan2(math.sin(a), math.cos(a))

    def control(k, x):
        u = U[k] if k < N else 0.0
        ref = Xref[:, min(k, Xref.shape[1] - 1)] if Xref is not None else None
        if gain is not None and ref is not None and abs(wrap(x[2])) < gate_rad:
            err = x - ref
            err[2] = wrap(err[2])
            u = u - float(np.ravel(np.asarray(gain) @ err)[0])
        return float(np.clip(u, -u_max, u_max)) if u_max is not None else u

    for k in range(N):
        for _ in range(substeps):
            u = control(k, x)
            x = rk4(x, u, dt / substeps)
        applied.append(u)
        traj.append(x.copy())
    for _ in range(int(round(hold / dt))):
        for _ in range(substeps):
            u = control(N, x)
            x = rk4(x, u, dt / substeps)
        applied.append(u)
        traj.append(x.copy())
    return np.array(traj).T, np.array(applied)


def defect(X, U, T, method):
    """How badly the plan violates the true dynamics between its own knots.

    We take each pair of consecutive knots, integrate the plan's control
    accurately from the first, and measure how far we land from the second.
    A collocation solution satisfies its OWN approximation exactly; this asks
    how good that approximation was.
    """
    N = len(U)
    dt = T / N
    worst = 0.0
    for k in range(N):
        x = X[:, k].copy()
        for _ in range(40):
            x = rk4(x, U[k], dt / 40.0)
        worst = max(worst, float(np.max(np.abs(x - X[:, k + 1]))))
    return worst


def swing_count(X):
    """How many times the pole reverses direction -- i.e. how many pumps.

    A strong motor throws the pole up in one go.  A weak one has to swing back
    and forth to build energy, exactly as a person pumps a playground swing,
    and the count is the cleanest way to see that in a number.
    """
    thd = X[3]
    s = np.sign(thd)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))
