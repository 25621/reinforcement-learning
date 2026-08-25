"""A hobby-servo joint with the parts a simulator usually leaves out.

**Read this first.**  The project brief says "on a real hobby arm".  There is no
hobby arm attached to this machine, so what follows is a stand-in: a single
joint carrying every effect that makes tuning on hardware feel different from
tuning in simulation.  It is honest about being a model.  What it CAN teach is
the shape of each pathology -- what stiction looks like on a plot, why an
integral term starts hunting, how latency eats your gain margin, what a
friction feedforward fixes and what it does not.  What it CANNOT teach is the
part that only hardware teaches: that the numbers drift with temperature, that
the third joint is not like the second, and that the smell of a hot servo is
the real end-stop detector.  Treat the numbers here as a rehearsal.

The pieces, and why each is here:

* **Stribeck friction.**  Real joints need more torque to START moving than to
  KEEP moving.  Named after Richard Stribeck, who measured this in journal
  bearings around 1902: as speed rises from zero the lubricant film builds up
  and friction first FALLS before viscous drag makes it rise again, giving the
  famous dipped curve.  The flat "does not move at all" region at zero speed is
  ``stiction`` -- static friction -- and it is why a joint creeps in jerks.
* **Torque saturation and quantisation.**  A motor driver has a maximum output
  and a finite number of PWM levels; a command of 0.3617 N*m arrives as 0.36.
* **Encoder quantisation.**  Position is read in whole counts.  A 12-bit
  encoder on one revolution resolves 0.0015 rad, and differentiating that to
  get velocity multiplies the step size by the control rate.
* **Loop latency.**  Reading a sensor over a serial bus, computing, and writing
  a command back takes time.  That delay is pure phase lag and it is what sets
  the highest gain the loop can survive.
* **Backlash.**  A gear train has clearance.  Reverse direction and the motor
  turns a little before the load hears about it.

``Joint`` is the friction/latency model on a single inertia.  ``GearedJoint``
adds the two-mass transmission with a dead band, used only by the backlash
experiment because it needs a much smaller time step.
"""

import numpy as np


class Joint:
    """One rotational joint: inertia, gravity load, friction, and a rough motor."""

    def __init__(
        self,
        J=0.030,            # kg m^2, load-side inertia
        m_g_l=0.45,         # N m, gravity torque when the link is horizontal
        f_coulomb=0.055,    # N m, friction once it is moving
        f_static=0.110,     # N m, torque needed to break it free
        v_stribeck=0.05,    # rad/s, the speed scale of the Stribeck dip
        f_viscous=0.020,    # N m s/rad
        tau_max=1.2,        # N m
        tau_step=0.0,       # N m per PWM level; 0 disables quantisation
        counts_per_rev=4096,
        noise_std=0.0,
    ):
        self.J, self.m_g_l = J, m_g_l
        self.f_coulomb, self.f_static = f_coulomb, f_static
        self.v_stribeck, self.f_viscous = v_stribeck, f_viscous
        self.tau_max, self.tau_step = tau_max, tau_step
        self.counts_per_rev = counts_per_rev
        self.noise_std = noise_std
        self.eps_v = 1e-4  # below this speed the joint counts as stuck

    # -- the physics -------------------------------------------------------
    def friction(self, w):
        """Stribeck + Coulomb + viscous, for a joint that is already moving."""
        s = np.sign(w)
        stribeck = self.f_coulomb + (self.f_static - self.f_coulomb) * np.exp(-(abs(w) / self.v_stribeck) ** 2)
        return s * stribeck + self.f_viscous * w

    def accel(self, th, w, tau):
        drive = tau - self.m_g_l * np.cos(th)  # gravity, largest when horizontal
        if abs(w) < self.eps_v:
            # Stuck.  Nothing moves until the drive beats static friction; the
            # instant it does, friction drops to the (smaller) Coulomb value and
            # the joint lurches.  That drop is the whole reason for stick-slip.
            if abs(drive) <= self.f_static:
                return 0.0
            return (drive - np.sign(drive) * self.f_coulomb) / self.J
        return (drive - self.friction(w)) / self.J

    def step(self, th, w, tau, dt):
        """Semi-implicit Euler.  Friction models need a small step; use one."""
        a = self.accel(th, w, tau)
        w_new = w + dt * a
        # If friction reversed the sign of the velocity within one step, the
        # joint really stopped -- clamp it to zero rather than letting it
        # oscillate at the step frequency, which is a pure numerical artefact.
        if abs(w) >= self.eps_v and w_new * w < 0 and abs(tau - self.m_g_l * np.cos(th)) <= self.f_static:
            w_new = 0.0
        return th + dt * w_new, w_new

    # -- the sensor and the motor -----------------------------------------
    def read(self, th, rng=None):
        q = 2 * np.pi / self.counts_per_rev
        val = np.round(th / q) * q
        if self.noise_std > 0 and rng is not None:
            val += rng.normal(0.0, self.noise_std)
        return val

    def drive(self, tau_cmd):
        tau = float(np.clip(tau_cmd, -self.tau_max, self.tau_max))
        if self.tau_step > 0:
            tau = np.round(tau / self.tau_step) * self.tau_step
        return tau


def run(joint, controller, T=4.0, dt_ctrl=2e-3, sub=20, th0=0.0, w0=0.0,
        delay_steps=1, seed=0, feedforward=None):
    """Closed loop with a real control period, sensor quantisation and latency.

    ``controller(t, th_meas) -> tau``; ``feedforward(t, w_ref) -> tau`` is added
    on top and is what the friction-compensation experiments switch on.
    """
    rng = np.random.default_rng(seed)
    n = int(T / dt_ctrl)
    h = dt_ctrl / sub
    th, w = float(th0), float(w0)
    queue = [0.0] * max(delay_steps, 0)
    ts = np.arange(n) * dt_ctrl
    TH = np.zeros(n)
    W = np.zeros(n)
    TAU = np.zeros(n)
    for k in range(n):
        t = k * dt_ctrl
        meas = joint.read(th, rng)
        cmd = controller(t, meas)
        if feedforward is not None:
            cmd = cmd + feedforward(t)
        queue.append(cmd)
        applied = joint.drive(queue.pop(0)) if delay_steps > 0 else joint.drive(cmd)
        TH[k], W[k], TAU[k] = th, w, applied
        for _ in range(sub):
            th, w = joint.step(th, w, applied, h)
    return ts, TH, W, TAU


def constant_velocity_torque(joint, speeds, kp=25.0, kd=2.0, T=1.5, dt=2e-4):
    """Drive the joint at a series of steady speeds and record the torque it takes.

    This is the standard friction-identification experiment on real hardware:
    close a velocity loop, hold each speed until it settles, and average the
    torque.  Subtract the gravity term and what is left is the friction curve.
    Doing it at a fixed HEIGHT (the link horizontal is easiest to hold steady)
    keeps the gravity term constant so it cancels cleanly.
    """
    out = []
    for v in speeds:
        th, w = 0.0, v
        integ = 0.0
        acc, cnt = 0.0, 0
        for k in range(int(T / dt)):
            e = v - w
            integ += e * dt
            tau = kp * e + kd * integ + joint.m_g_l * np.cos(th)
            tau = joint.drive(tau)
            th, w = joint.step(th, w, tau, dt)
            if k * dt > 0.6 * T:
                acc += tau - joint.m_g_l * np.cos(th)
                cnt += 1
        out.append(acc / max(cnt, 1))
    return np.array(out)


class GearedJoint:
    """Motor and load coupled through a gear train with clearance (backlash).

    Two inertias joined by a stiff spring that is DISCONNECTED while the
    relative angle is inside the dead band.  Reverse direction and the motor
    turns freely through the whole dead band before it picks the load up again:
    from the motor's point of view the load has vanished; from the load's, the
    command has.
    """

    def __init__(self, J_m=0.004, J_l=0.030, k_gear=900.0, d_gear=1.2,
                 backlash=0.008, m_g_l=0.45, f_coulomb=0.055, f_viscous=0.020,
                 tau_max=1.2):
        self.J_m, self.J_l = J_m, J_l
        self.k_gear, self.d_gear, self.backlash = k_gear, d_gear, backlash
        self.m_g_l, self.f_coulomb, self.f_viscous = m_g_l, f_coulomb, f_viscous
        self.tau_max = tau_max

    def gear_torque(self, dth, dw):
        """Dead-band spring: zero inside +-backlash, linear outside."""
        if abs(dth) <= self.backlash:
            return 0.0
        engaged = dth - np.sign(dth) * self.backlash
        return self.k_gear * engaged + self.d_gear * dw

    def step(self, s, tau, dt):
        th_m, w_m, th_l, w_l = s
        tau = float(np.clip(tau, -self.tau_max, self.tau_max))
        tg = self.gear_torque(th_m - th_l, w_m - w_l)
        a_m = (tau - tg) / self.J_m
        fric = self.f_coulomb * np.tanh(w_l / 0.01) + self.f_viscous * w_l
        a_l = (tg - fric - self.m_g_l * np.cos(th_l)) / self.J_l
        w_m += dt * a_m
        w_l += dt * a_l
        return np.array([th_m + dt * w_m, w_m, th_l + dt * w_l, w_l])
