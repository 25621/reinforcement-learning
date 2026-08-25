"""The cart-pole: one motor, four states, and no way to control them separately.

A cart of mass ``M`` slides on a rail; a pole of mass ``m`` is hinged to it and
free to swing.  The only input is a horizontal force on the cart.  The state is

    x = [cart position, cart velocity, pole angle from UP, pole rate]

The system is UNDERACTUATED: four numbers to steer, one knob to steer them
with.  You cannot ask for a cart position and a pole angle independently -- the
only way to move the cart right is to first tip the pole right, let it fall a
little, and chase it.  That coupling is exactly what a single-loop PID cannot
express and what LQR handles without being told.

Nonlinear equations (Lagrange, pole as a point mass at distance l):

    xdd     = ( u + m l thetad^2 sin(theta) - m g sin(theta) cos(theta) )
              / ( M + m sin^2(theta) )
    thetadd = ( g sin(theta) - xdd cos(theta) ) / l

Linearised about theta = 0 (upright, at rest, no force), using
sin(theta) ~ theta, cos(theta) ~ 1 and dropping thetad^2:

    A = [[0, 1,            0,        0],
         [0, 0,      -m g / M,       0],
         [0, 0,            0,        1],
         [0, 0, (M+m) g / (M l),     0]]
    B = [0, 1/M, 0, -1/(M l)]^T

Read A row by row and it says something physical: tipping the pole by theta
pushes the cart the OTHER way with acceleration -m g theta / M, and pushing the
cart forward tips the pole backward at -u/(M l).  The minus signs are the whole
problem.
"""

import numpy as np

G = 9.81


class CartPole:
    def __init__(self, M=1.0, m=0.1, l=0.5, u_max=np.inf):
        self.M, self.m, self.l, self.u_max = M, m, l, u_max

    # -- nonlinear -----------------------------------------------------------
    def deriv(self, s, u):
        _, xd, th, thd = s
        u = float(np.clip(u, -self.u_max, self.u_max))
        M, m, l = self.M, self.m, self.l
        st, ct = np.sin(th), np.cos(th)
        xdd = (u + m * l * thd * thd * st - m * G * st * ct) / (M + m * st * st)
        thdd = (G * st - xdd * ct) / l
        return np.array([xd, xdd, thd, thdd])

    def rk4(self, s, u, dt):
        k1 = self.deriv(s, u)
        k2 = self.deriv(s + 0.5 * dt * k1, u)
        k3 = self.deriv(s + 0.5 * dt * k2, u)
        k4 = self.deriv(s + dt * k3, u)
        return s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    # -- linear --------------------------------------------------------------
    def linearize(self):
        M, m, l = self.M, self.m, self.l
        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m * G / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (M + m) * G / (M * l), 0.0],
        ])
        B = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * l)]])
        return A, B

    def energy(self, s):
        _, xd, th, thd = s
        m, l, M = self.m, self.l, self.M
        vx = xd + l * thd * np.cos(th)
        vy = -l * thd * np.sin(th)
        return 0.5 * M * xd ** 2 + 0.5 * m * (vx ** 2 + vy ** 2) + m * G * l * np.cos(th)


def simulate(plant, s0, gain_fn, T=6.0, dt=2e-3, u_max=None, ref=None):
    """Closed-loop roll-out.  ``gain_fn(state) -> force``."""
    n = int(T / dt)
    S = np.zeros((n, 4))
    U = np.zeros(n)
    s = np.asarray(s0, dtype=float)
    for k in range(n):
        u = float(gain_fn(s if ref is None else s - ref))
        if u_max is not None:
            u = float(np.clip(u, -u_max, u_max))
        S[k], U[k] = s, u
        s = plant.rk4(s, u, dt)
        if not np.all(np.isfinite(s)) or abs(s[2]) > 10 or abs(s[0]) > 50:
            S[k + 1:] = S[k]
            U[k + 1:] = U[k]
            break
    return np.arange(n) * dt, S, U
