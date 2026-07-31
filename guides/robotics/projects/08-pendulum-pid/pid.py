"""A PID controller with the four details that separate a toy from a real one.

The textbook formula is one line:

    u = Kp e + Ki integral(e) + Kd de/dt

Every real implementation adds five things, and each one exists because of a
specific failure you can watch happen in ``run.py``:

  1. **Derivative on the measurement, not on the error.**  When someone changes
     the target, the error jumps instantly, and d(error)/dt is a spike of
     nearly infinite height.  That is "derivative kick": one huge torque command
     for one sample.  Differentiating the measurement instead gives the same
     damping without the spike, because the measurement cannot jump.
  2. **A low-pass filter on the derivative.**  Differentiating amplifies noise
     in proportion to frequency, so raw Kd on a noisy encoder produces a
     controller that mostly amplifies noise.
  3. **Output saturation.**  Motors have a maximum torque.  A controller that
     pretends otherwise is solving a different problem from the one on the bench.
  4. **Anti-windup.**  When the output is saturated the integral keeps
     accumulating error it cannot act on, and by the time the error finally
     reverses the integrator holds a large stale command that has to be
     "unwound" -- a big overshoot with no cause visible in the gains.
  5. **Setpoint weighting** ``b`` on the proportional term: feed ``b*setpoint -
     measurement`` to P instead of the full error.  This is NOT a fifth gain
     doing the same job as Kp.  Kp sets how hard the loop pushes back against
     *any* deviation, whether the target moved or the world did; ``b`` decides
     how much of a *target change* is allowed to hit the output immediately.
     Set b = 1 and a step in the target produces a step in the command, which
     overshoots; set b = 0 and the target reaches the output only through the
     integrator, which is smooth.  Disturbance rejection is identical either
     way, because a disturbance does not move the setpoint.  One knob, one job:
     tracking shape, decoupled from stiffness.

The controller is DISCRETE: it runs at a fixed rate and holds its output
constant between samples ("zero-order hold"), which is what a real control loop
does and is the reason a gain set that is perfect at 1 kHz can be unstable at
50 Hz.
"""

import numpy as np


class PID:
    def __init__(
        self,
        kp,
        ki=0.0,
        kd=0.0,
        dt=1e-3,
        u_min=-np.inf,
        u_max=np.inf,
        d_on_measurement=True,
        d_filter_hz=None,
        anti_windup=True,
        b_sp=1.0,
    ):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.u_min, self.u_max = u_min, u_max
        self.d_on_measurement = d_on_measurement
        self.anti_windup = anti_windup
        self.b_sp = b_sp
        # A first-order filter's smoothing factor from its cut-off frequency:
        # alpha = dt / (dt + 1 / (2 pi f)).  alpha = 1 means "no filtering".
        if d_filter_hz is None:
            self.alpha = 1.0
        else:
            tau = 1.0 / (2 * np.pi * d_filter_hz)
            self.alpha = dt / (dt + tau)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self._prev_e = None
        self._prev_y = None
        self._d = 0.0
        self.last_u_unsat = 0.0
        self.last_terms = (0.0, 0.0, 0.0)

    def __call__(self, setpoint, measurement):
        e = setpoint - measurement

        # --- derivative -----------------------------------------------------
        if self.d_on_measurement:
            prev = self._prev_y if self._prev_y is not None else measurement
            raw = -(measurement - prev) / self.dt  # minus: d(error) = -d(measurement)
        else:
            prev = self._prev_e if self._prev_e is not None else e
            raw = (e - prev) / self.dt
        self._d += self.alpha * (raw - self._d)  # one-pole low pass
        self._prev_e, self._prev_y = e, measurement

        # --- the three terms ------------------------------------------------
        # Only P uses the weighted setpoint.  I must keep seeing the TRUE error,
        # or it would drive the output to the wrong place forever.
        p_term = self.kp * (self.b_sp * setpoint - measurement)
        i_term = self.ki * self.integral
        d_term = self.kd * self._d
        u = p_term + i_term + d_term
        self.last_u_unsat = u
        u_sat = float(np.clip(u, self.u_min, self.u_max))

        # --- integrate, unless that would only wind up -----------------------
        # Conditional integration: skip the update when the output is already
        # saturated AND the error would push it further into the stop.  The
        # integrator then holds its value instead of growing uselessly.
        if self.anti_windup and u != u_sat and np.sign(e) == np.sign(u):
            pass
        else:
            self.integral += e * self.dt

        self.last_terms = (p_term, i_term, d_term)
        return u_sat


def step_metrics(t, y, target, start=0.0, tol=0.02):
    """Rise time, overshoot, settling time and steady-state error of a response.

    ``tol`` is the settling band as a fraction of the step size (2% is the
    usual convention).  Rise time here is the 10%-to-90% definition.
    """
    t = np.asarray(t)
    y = np.asarray(y)
    span = target - start
    if abs(span) < 1e-12:
        return dict(rise=np.nan, overshoot=np.nan, settle=np.nan, sse=np.nan)

    def first_cross(frac):
        thr = start + frac * span
        idx = np.nonzero((y - thr) * np.sign(span) >= 0)[0]
        return t[idx[0]] if len(idx) else np.nan

    rise = first_cross(0.9) - first_cross(0.1)
    peak = np.max(y * np.sign(span)) * np.sign(span)
    overshoot = 100.0 * (peak - target) / abs(span) if span > 0 else 100.0 * (target - peak) / abs(span)
    band = tol * abs(span)
    outside = np.nonzero(np.abs(y - target) > band)[0]
    settle = t[outside[-1]] if len(outside) and outside[-1] + 1 < len(t) else (np.nan if len(outside) else 0.0)
    sse = float(np.mean(y[int(0.9 * len(y)):]) - target)
    return dict(rise=float(rise), overshoot=float(max(overshoot, 0.0)),
                settle=float(settle), sse=sse)
