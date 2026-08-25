"""A quadrotor, its differential flatness, and a geometric SE(3) controller.

A [quadrotor] has four motors and six degrees of freedom, so it is
*under-actuated*: there are more things it could do than knobs to do them
with.  It cannot fly sideways without tilting, because the only force it can
make points along its own body z-axis.

That sounds like it should make planning hard.  Differential flatness is the
result that it does not.  A system is **differentially flat** when some set of
outputs exists ("flat outputs") from which the *entire* state and *every*
input can be written down algebraically, using only those outputs and their
derivatives -- no integration, no solving anything.  For a quadrotor the flat
outputs are (x, y, z, yaw).  "Flat" is the term Fliess, Levine, Martin and
Rouchon introduced in 1992; the intuition is that the system, seen through
those outputs, has been *flattened* into something with no hidden internal
dynamics left to worry about.

The consequence is the whole reason this project exists: **plan any smooth
enough curve in (x, y, z, yaw) and a quadrotor can fly it, and you can read
off the required attitude and motor forces from the curve's derivatives.**
Trajectory generation stops being a 12-dimensional optimal-control problem
and becomes "draw four smooth 1-D functions".
"""

import math

import numpy as np

G = 9.81
MASS = 0.9                       # kg
J = np.diag([0.005, 0.005, 0.009])
J_INV = np.linalg.inv(J)
ARM = 0.16                       # m, motor to centre
C_TAU = 0.016                    # drag-torque / thrust ratio of one rotor
F_MAX = 6.0                      # N per motor -- about 2.7 g of total thrust
F_MIN = 0.0

# Mixer: [total thrust, Mx, My, Mz] = MIX @ [f1, f2, f3, f4], "+" layout with
# motor 1 at +x, 2 at +y, 3 at -x, 4 at -y, alternating spin directions.
MIX = np.array([[1.0, 1.0, 1.0, 1.0],
                [0.0, ARM, 0.0, -ARM],
                [-ARM, 0.0, ARM, 0.0],
                [C_TAU, -C_TAU, C_TAU, -C_TAU]])
MIX_INV = np.linalg.inv(MIX)

E3 = np.array([0.0, 0.0, 1.0])


def hat(v):
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


def expm_so3(w):
    """Exact rotation-matrix exponential (Rodrigues).

    Integrating a rotation as `R + R @ hat(w) * dt` drifts off the rotation
    manifold within a few hundred steps -- the matrix stops being orthonormal
    and every direction it reports is slightly wrong.  Rodrigues' formula is
    the same cost and stays exactly on the manifold.
    """
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = hat(w / th)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


# ------------------------------------------------------------------ the plant
class Quad:
    def __init__(self, p=(0.0, 0.0, 0.0), R=None, drag=0.0):
        self.p = np.asarray(p, float).copy()
        self.v = np.zeros(3)
        self.R = np.eye(3) if R is None else np.asarray(R, float).copy()
        self.w = np.zeros(3)
        self.drag = drag
        self.sat_steps = 0
        self.steps = 0

    def step(self, f_cmd, M_cmd, dt, wind=np.zeros(3)):
        """Apply a thrust and a body torque, THROUGH the motors.

        Commanding (f, M) directly would let the controller ask for a torque
        no set of four non-negative, bounded motor forces can produce.  Going
        through the mixer and clipping is what makes saturation visible --
        and saturation, not the control law, is what actually limits how
        aggressively a quadrotor can fly.
        """
        f_motors = MIX_INV @ np.concatenate([[f_cmd], M_cmd])
        clipped = np.clip(f_motors, F_MIN, F_MAX)
        self.steps += 1
        if not np.allclose(clipped, f_motors, atol=1e-9):
            self.sat_steps += 1
        f, M = MIX[0] @ clipped, MIX[1:] @ clipped

        acc = (-G * E3 + (f / MASS) * (self.R @ E3) + wind / MASS
               - self.drag * self.v / MASS)
        wdot = J_INV @ (M - np.cross(self.w, J @ self.w))
        self.p = self.p + self.v * dt + 0.5 * acc * dt * dt
        self.v = self.v + acc * dt
        self.R = self.R @ expm_so3(self.w * dt + 0.5 * wdot * dt * dt)
        self.w = self.w + wdot * dt
        return clipped


# ------------------------------------------------------- flatness: curve -> state
def flat_to_state(pos, vel, acc, jerk, snap, yaw=0.0, yaw_rate=0.0):
    """Turn four numbers and their derivatives into a full quadrotor state.

    This is differential flatness made concrete, and every line is forced:

      * the thrust direction must point along (acceleration + gravity),
        because that is the only force the vehicle has;
      * the yaw you asked for then pins down the last rotation about that
        axis, giving a unique attitude R;
      * differentiating the thrust direction gives the angular velocity --
        so JERK (the derivative of acceleration) sets how fast the vehicle
        must rotate;
      * differentiating once more gives angular acceleration, and therefore
        torque -- so SNAP (the 4th derivative of position) sets the motor
        differential.

    That last line is the whole argument for minimizing snap rather than,
    say, acceleration: snap is the derivative that maps directly onto the
    quantity the motors have a hard limit on.
    """
    t_vec = acc + G * E3
    f = MASS * float(np.linalg.norm(t_vec))
    b3 = t_vec / max(np.linalg.norm(t_vec), 1e-9)
    b1c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    b2 = np.cross(b3, b1c)
    n = np.linalg.norm(b2)
    b2 = b2 / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])
    b1 = np.cross(b2, b3)
    R = np.column_stack([b1, b2, b3])

    # d/dt of the thrust vector, projected off b3: the part that must come
    # from rotating, not from throttling.
    fdot = MASS * float(b3 @ jerk)
    h_w = MASS / max(f, 1e-9) * (jerk - (fdot / MASS) * b3)
    w = np.array([-float(h_w @ b2), float(h_w @ b1), 0.0])
    w[2] = yaw_rate * float(b3 @ E3)

    fddot = MASS * float(b3 @ snap) + MASS * float(jerk @ np.cross(w, b3))
    h_a = (MASS / max(f, 1e-9) * (snap - (fddot / MASS) * b3)
           - 2 * (fdot / max(f, 1e-9)) * np.cross(w, b3)
           - np.cross(w, np.cross(w, b3)))
    wdot = np.array([-float(h_a @ b2), float(h_a @ b1), 0.0])
    M = J @ wdot + np.cross(w, J @ w)
    return dict(R=R, f=f, w=w, wdot=wdot, M=M)


# ------------------------------------------------------------- the controller
class GeoController:
    """The geometric controller on SE(3) (Lee, Leok & McClamroch 2010).

    "Geometric" means it never converts the attitude to Euler angles.  The
    orientation error is computed as a rotation between two rotation
    matrices, so there is no gimbal lock and no wrapping -- the controller
    works upside down, which Euler-based ones do not.

    `feedforward=False` strips out everything flatness gave us and leaves a
    plain PD on position plus a PD on attitude.  It is the control for
    experiment 5: it answers "was the flatness algebra worth writing?".
    """

    def __init__(self, kp=(9.0, 9.0, 12.0), kv=(5.0, 5.0, 6.0),
                 kR=(1.6, 1.6, 0.5), kw=(0.28, 0.28, 0.12),
                 feedforward=True):
        self.kp = np.asarray(kp, float)
        self.kv = np.asarray(kv, float)
        self.kR = np.asarray(kR, float)
        self.kw = np.asarray(kw, float)
        self.ff = feedforward

    def __call__(self, q, ref):
        ep = q.p - ref["pos"]
        ev = q.v - ref["vel"]
        a_ff = ref["acc"] if self.ff else np.zeros(3)
        F = -self.kp * ep - self.kv * ev + MASS * G * E3 + MASS * a_ff
        b3 = q.R @ E3
        f = float(F @ b3)

        b3d = F / max(np.linalg.norm(F), 1e-9)
        yaw = ref.get("yaw", 0.0)
        b1c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        b2d = np.cross(b3d, b1c)
        nb = np.linalg.norm(b2d)
        b2d = b2d / nb if nb > 1e-9 else np.array([0.0, 1.0, 0.0])
        Rd = np.column_stack([np.cross(b2d, b3d), b2d, b3d])

        eR = 0.5 * vee(Rd.T @ q.R - q.R.T @ Rd)
        wd = ref["w"] if self.ff else np.zeros(3)
        ew = q.w - q.R.T @ Rd @ wd
        M = -self.kR * eR - self.kw * ew + np.cross(q.w, J @ q.w)
        if self.ff:
            M = M + J @ (q.R.T @ Rd @ ref["wdot"])
        return f, M
