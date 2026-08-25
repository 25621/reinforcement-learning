"""LQR from scratch: two Riccati solvers and a matrix exponential.

The guide's sample code calls ``scipy.linalg.solve_continuous_are``.  SciPy is
not installed in this environment, which turns out to be a gift: the Riccati
equation is short enough to solve twice, by two unrelated methods, and having
two independent answers is a far better test than trusting one library call.

What the Riccati equation is, in words.  LQR asks for the feedback law
``u = -K x`` that minimises

    J = integral( x^T Q x  +  u^T R u ) dt

over all future time.  Q prices being in the wrong state; R prices the effort
of fixing it.  It turns out the minimum future cost from any state x is exactly
``x^T P x`` for one fixed matrix P, and P is the solution of

    A^T P + P A - P B R^-1 B^T P + Q = 0          (the continuous ARE)

The name is Jacopo Riccati's, an 18th-century Venetian mathematician who
studied scalar equations of the form y' = a + b y + c y^2 -- quadratic in the
unknown.  This matrix equation is quadratic in P for the same reason (the
``P B R^-1 B^T P`` term), which is why it carries his name; the control theory
came 200 years later.  Once you have P, the gain is ``K = R^-1 B^T P``.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Method 1: the Hamiltonian eigenvector method
# ---------------------------------------------------------------------------
def care_hamiltonian(A, B, Q, R):
    """Solve the continuous ARE through the eigenvectors of a 2n x 2n matrix.

    Stack the state x and the "co-state" (the running price of being at x) into
    one 2n-vector.  The optimality conditions say that stacked vector evolves
    under the Hamiltonian matrix

        H = [[  A, -B R^-1 B^T ],
             [ -Q,   -A^T      ]]

    H has a special structure: its eigenvalues come in pairs (lambda, -lambda),
    so exactly n of them are stable (negative real part) and n unstable.  The
    optimal solution is the one that stays finite as time goes to infinity, so
    it must live entirely in the space spanned by the n STABLE eigenvectors.
    Write those n eigenvectors as a 2n x n block [[X1], [X2]]; then costate =
    P x with ``P = X2 X1^-1``.  That is the whole method.

    "Hamiltonian" here is the same word as in Hamiltonian mechanics, and for the
    same reason: William Rowan Hamilton's reformulation of mechanics pairs each
    coordinate with a momentum and evolves both together.  Optimal control pairs
    each state with its co-state and does exactly the same thing.
    """
    A, B, Q, R = [np.atleast_2d(np.asarray(m, dtype=float)) for m in (A, B, Q, R)]
    n = A.shape[0]
    Rinv = np.linalg.inv(R)
    H = np.block([[A, -B @ Rinv @ B.T], [-Q, -A.T]])

    w, v = np.linalg.eig(H)
    stable = np.argsort(w.real)[:n]  # the n most-negative real parts
    if np.any(w.real[stable] >= 0):
        raise ValueError("no stabilising solution: check that (A, B) is controllable")
    V = v[:, stable]
    X1, X2 = V[:n, :], V[n:, :]
    P = np.real(X2 @ np.linalg.inv(X1))
    return 0.5 * (P + P.T)  # P is symmetric; drop the round-off asymmetry


# ---------------------------------------------------------------------------
# Method 2: iterate the discrete Riccati equation
# ---------------------------------------------------------------------------
def expm(M, terms=18):
    """Matrix exponential by scaling and squaring.

    exp(M) = (exp(M / 2^s))^(2^s).  Dividing M until its norm is small makes the
    Taylor series converge fast and accurately; squaring s times undoes the
    division.  Squaring is exact-ish and cheap, so this is the standard recipe.
    """
    M = np.asarray(M, dtype=float)
    nrm = np.abs(M).sum(axis=1).max()
    s = max(0, int(np.ceil(np.log2(max(nrm, 1e-16)))) + 1)
    A = M / (2.0 ** s)
    E = np.eye(A.shape[0])
    term = np.eye(A.shape[0])
    for k in range(1, terms):
        term = term @ A / k
        E = E + term
    for _ in range(s):
        E = E @ E
    return E


def discretize(A, B, dt):
    """Exact zero-order-hold discretisation of xdot = Ax + Bu.

    A control loop sets u at the tick and holds it constant until the next one.
    Under that assumption the exact update is x[k+1] = Ad x[k] + Bd u[k], and
    both matrices drop out of one matrix exponential:

        expm([[A, B], [0, 0]] * dt)  =  [[Ad, Bd], [0, I]]

    "Exact" matters: the lazy approximation Ad = I + A dt is what makes a
    discrete design disagree with a continuous one at slow rates, which is the
    thing project 09 measures.
    """
    A, B = np.atleast_2d(A), np.atleast_2d(B)
    n, m = A.shape[0], B.shape[1]
    Maug = np.zeros((n + m, n + m))
    Maug[:n, :n] = A
    Maug[:n, n:] = B
    E = expm(Maug * dt)
    return E[:n, :n], E[:n, n:]


def dare_iterate(Ad, Bd, Q, R, iters=20000, tol=1e-14):
    """Solve the DISCRETE ARE by simply running the recursion until it stops moving.

    P <- Q + Ad^T P Ad - Ad^T P Bd (R + Bd^T P Bd)^-1 Bd^T P Ad

    This is dynamic programming read backwards: P after k iterations is the
    optimal cost-to-go with k steps left to live.  As k grows the answer stops
    changing, and that fixed point is the infinite-horizon solution.  Slow, but
    it shares no code and no idea with the Hamiltonian method, which is exactly
    what makes it a useful second opinion.
    """
    Ad, Bd, Q, R = [np.atleast_2d(np.asarray(m, dtype=float)) for m in (Ad, Bd, Q, R)]
    P = Q.copy()
    for k in range(iters):
        S = R + Bd.T @ P @ Bd
        K = np.linalg.solve(S, Bd.T @ P @ Ad)
        P_new = Q + Ad.T @ P @ Ad - Ad.T @ P @ Bd @ K
        P_new = 0.5 * (P_new + P_new.T)
        if np.abs(P_new - P).max() < tol:
            return P_new, k
        P = P_new
    return P, iters


# ---------------------------------------------------------------------------
# The controllers
# ---------------------------------------------------------------------------
def lqr(A, B, Q, R):
    """Continuous-time infinite-horizon LQR.  Returns gain K and cost matrix P."""
    P = care_hamiltonian(A, B, Q, R)
    K = np.linalg.solve(np.atleast_2d(R), np.atleast_2d(B).T @ P)
    return K, P


def dlqr(A, B, Q, R, dt):
    """Discrete-time LQR for a loop running at 1/dt Hz.

    Note it costs Q and R PER SECOND, so the discrete weights are scaled by dt.
    Without that scaling the discrete design would silently solve a different
    optimisation problem from the continuous one and the comparison would be
    meaningless.
    """
    Ad, Bd = discretize(A, B, dt)
    P, _ = dare_iterate(Ad, Bd, np.asarray(Q) * dt, np.asarray(R) * dt)
    S = np.atleast_2d(np.asarray(R) * dt) + Bd.T @ P @ Bd
    K = np.linalg.solve(S, Bd.T @ P @ Ad)
    return K, P


def are_residual(A, B, Q, R, P):
    """How far the claimed solution is from actually satisfying the equation."""
    A, B, Q, R, P = [np.atleast_2d(np.asarray(m, dtype=float)) for m in (A, B, Q, R, P)]
    res = A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
    return float(np.abs(res).max())


def closed_loop_poles(A, B, K):
    return np.linalg.eigvals(np.atleast_2d(A) - np.atleast_2d(B) @ np.atleast_2d(K))
