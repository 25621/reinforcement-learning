"""A torque-driven pendulum, and the loop that closes around it.

Geometry: ``theta`` is measured from STRAIGHT UP.  So theta = 0 is balanced,
theta = pi is hanging at rest, and gravity always pushes theta away from zero.

    I thetadd = tau - b thetad + m g l sin(theta)

The ``+`` in front of the gravity term is the whole difficulty of the problem.
For a hanging pendulum that term would be ``-`` and would pull the pendulum
back to rest by itself; upright, it pushes harder the further you tip.  A pure
proportional controller therefore has to beat gravity before it does anything
useful at all: near theta = 0, sin(theta) ~ theta, so the closed loop behaves
like a spring of stiffness ``(Kp - m g l)``.  Below ``Kp = m g l`` that spring
pushes the wrong way and no amount of patience will balance the pendulum.

The simulator deliberately separates two clocks:

  * the PHYSICS clock, a small fixed step integrated with RK4, standing in for
    "the real world, which does not care about your control rate";
  * the CONTROL clock, which reads the sensor, computes, and holds its output
    constant until the next tick.

Keeping them apart is what makes the sample-rate experiment meaningful.  If the
controller ran at the physics rate, changing the control rate would also change
the accuracy of the physics and the two effects would be impossible to separate.
"""

import numpy as np

G = 9.81


class Pendulum:
    """Point mass ``m`` on a massless rod of length ``l``, pivot at the origin."""

    def __init__(self, m=0.5, l=0.4, b=0.01, tau_max=np.inf, inverted=True):
        self.m, self.l, self.b = m, l, b
        self.I = m * l * l
        self.tau_max = tau_max
        self.mgl = m * G * l  # the torque gravity applies when horizontal
        # inverted=True  -> theta measured from UP, gravity pushes you away
        # inverted=False -> theta measured from DOWN, gravity pulls you back.
        # The second case is an ordinary robot joint: stable on its own, and
        # the only setting in which a controller can stay saturated long enough
        # for integral windup to build up (experiment 3).
        self.inverted = inverted
        self.sgn = 1.0 if inverted else -1.0

    def accel(self, theta, thetad, tau, load=0.0):
        tau = float(np.clip(tau + load, -self.tau_max, self.tau_max))
        return (tau - self.b * thetad + self.sgn * self.mgl * np.sin(theta)) / self.I

    def rk4(self, theta, thetad, tau, dt, load=0.0):
        def f(s):
            return np.array([s[1], self.accel(s[0], s[1], tau, load)])

        s = np.array([theta, thetad])
        k1 = f(s)
        k2 = f(s + 0.5 * dt * k1)
        k3 = f(s + 0.5 * dt * k2)
        k4 = f(s + dt * k3)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return s[0], s[1]

    def energy(self, theta, thetad):
        """Kinetic + potential, with zero at the hanging rest position."""
        return 0.5 * self.I * thetad ** 2 + self.mgl * (np.cos(theta) + 1.0)


def simulate(
    plant,
    ctrl,
    theta0,
    thetad0=0.0,
    T=3.0,
    setpoint=0.0,
    load=0.0,
    noise_std=0.0,
    delay_steps=1,
    physics_dt=2e-4,
    seed=0,
    setpoint_fn=None,
):
    """Run the closed loop.  Returns time, angle, rate, applied torque, raw command.

    ``delay_steps`` is the number of control periods between measuring and
    acting.  One is the honest default: a real loop reads the encoder, computes,
    and the result reaches the motor at the START of the next period.  That one
    period of dead time is what gives the loop a finite ultimate gain -- without
    it, a frictionless model would tolerate arbitrarily large Kp, which no
    physical machine does.
    """
    rng = np.random.default_rng(seed)
    dt = ctrl.dt
    n_steps = int(round(T / dt))
    sub = max(1, int(round(dt / physics_dt)))
    h = dt / sub

    theta, thetad = float(theta0), float(thetad0)
    queue = [0.0] * max(delay_steps, 0)
    ts = np.empty(n_steps)
    th = np.empty(n_steps)
    thd = np.empty(n_steps)
    us = np.empty(n_steps)
    raw = np.empty(n_steps)
    sps = np.empty(n_steps)

    for k in range(n_steps):
        t = k * dt
        sp = setpoint if setpoint_fn is None else setpoint_fn(t)
        meas = theta + (rng.normal(0.0, noise_std) if noise_std > 0 else 0.0)
        u_now = ctrl(sp, meas)
        if delay_steps > 0:
            queue.append(u_now)
            u_applied = queue.pop(0)
        else:
            u_applied = u_now

        ts[k], th[k], thd[k] = t, theta, thetad
        us[k], raw[k], sps[k] = u_applied, ctrl.last_u_unsat, sp
        for _ in range(sub):
            theta, thetad = plant.rk4(theta, thetad, u_applied, h, load=load)

    return dict(t=ts, theta=th, thetad=thd, u=us, u_raw=raw, sp=sps)
