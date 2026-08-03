"""A 12-joint quadruped in MuJoCo, plus the gait schedule and leg controller.

The robot is a small mini-cheetah-shaped machine: a trunk with a free joint
and four legs, each with three joints (abduction, hip, knee).  Everything is
built from one XML string so the whole scene is inspectable in one place.

Leg naming is the standard one: FR = front right, FL = front left,
RR = rear right, RL = rear left.  A *trot* pairs diagonal legs -- FR with RL,
FL with RR -- so at any moment two feet are down and two are swinging.  That
is the gait almost every quadruped robot walks with, because a diagonal pair
puts the support line straight through the centre of mass, so the body is
close to balanced without needing to shift its weight side to side.
"""

import math

import mujoco
import numpy as np

LEGS = ["FR", "FL", "RR", "RL"]
# Hip locations in the trunk frame, and the abduction sign of each side.
HIP = np.array([[0.19, -0.11, 0.0], [0.19, 0.11, 0.0],
                [-0.19, -0.11, 0.0], [-0.19, 0.11, 0.0]])
L_THIGH, L_CALF, L_ABD = 0.21, 0.21, 0.07
# The foot is a SPHERE, and the site sits at its centre.  Aim the site at
# ground level and the sphere starts 24 mm inside the floor -- MuJoCo answers
# that with a contact impulse that fires the robot into the air on step one.
# Every foot target below is therefore z = FOOT_R, not z = 0.
FOOT_R = 0.024
BODY_MASS = 7.0
STAND_H = 0.30


def _leg_xml(name, x, y, side):
    return f"""
    <body name="{name}_hip" pos="{x} {y} 0">
      <joint name="{name}_abd" type="hinge" axis="1 0 0" range="-0.8 0.8"
             damping="0.06" armature="0.008"/>
      <geom type="capsule" fromto="0 0 0 0 {side * L_ABD} 0" size="0.028"
            mass="0.55" rgba="0.55 0.55 0.6 1"/>
      <body name="{name}_thigh" pos="0 {side * L_ABD} 0">
        <joint name="{name}_hip" type="hinge" axis="0 1 0" range="-2.2 2.2"
               damping="0.06" armature="0.008"/>
        <geom type="capsule" fromto="0 0 0 0 0 {-L_THIGH}" size="0.024"
              mass="0.9" rgba="0.35 0.4 0.5 1"/>
        <body name="{name}_calf" pos="0 0 {-L_THIGH}">
          <joint name="{name}_knee" type="hinge" axis="0 1 0" range="0.1 2.7"
                 damping="0.06" armature="0.008"/>
          <geom type="capsule" fromto="0 0 0 0 0 {-L_CALF}" size="0.017"
                mass="0.3" rgba="0.35 0.4 0.5 1"/>
          <geom name="{name}_foot" type="sphere" pos="0 0 {-L_CALF}"
                size="0.024" mass="0.06" friction="0.9 0.02 0.001"
                rgba="0.85 0.4 0.1 1"/>
          <site name="{name}_ft" pos="0 0 {-L_CALF}" size="0.012"/>
        </body>
      </body>
    </body>"""


def build_xml(friction=0.9, payload=0.0, terrain=None):
    legs = "".join(_leg_xml(n, HIP[i][0], HIP[i][1],
                            -1.0 if n.endswith("R") else 1.0)
                   for i, n in enumerate(LEGS))
    extra = ""
    if payload > 0:
        extra = (f'<geom type="box" pos="0.06 0 0.06" size="0.06 0.06 0.03" '
                 f'mass="{payload}" rgba="0.8 0.2 0.2 1"/>')
    hf = ""
    if terrain is not None:
        # Small random bumps: enough to break the flat-ground assumption
        # without turning the project into a terrain-mapping exercise.
        hf = (f'<hfield name="bump" size="6 6 {terrain} 0.1" nrow="40" '
              f'ncol="40"/>')
    ground = ('<geom name="floor" type="hfield" hfield="bump" '
              f'friction="{friction} 0.02 0.001" rgba="0.8 0.8 0.8 1"/>'
              if terrain is not None else
              f'<geom name="floor" type="plane" size="30 30 0.1" '
              f'friction="{friction} 0.02 0.001" rgba="0.8 0.8 0.8 1"/>')
    return f"""
<mujoco model="quadruped">
  <!-- MuJoCo reads joint ranges in DEGREES unless told otherwise.  Leave this
       out and range="0.1 2.7" on the knee silently means 2.7 degrees, the
       limit fires on the first step, and the robot explodes for reasons that
       look like a physics bug and are not. -->
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"
          cone="pyramidal"/>
  <asset>{hf}</asset>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    {ground}
    <body name="trunk" pos="0 0 {STAND_H}">
      <freejoint name="root"/>
      <geom type="box" size="0.20 0.075 0.045" mass="{BODY_MASS}"
            rgba="0.2 0.35 0.55 1"/>
      {extra}
      {legs}
    </body>
  </worldbody>
  <contact>
    <!-- MuJoCo does NOT exclude parent-child geom pairs by itself.  Every
         leg link's capsule starts exactly where its parent's ends, so
         without these lines each joint contains two capsules overlapping by
         their radii, and the contact solver answers by throwing the robot
         across the room on the first timestep. -->
    {''.join(f'<exclude body1="trunk" body2="{n}_hip"/>'
             f'<exclude body1="trunk" body2="{n}_thigh"/>'
             f'<exclude body1="{n}_hip" body2="{n}_thigh"/>'
             f'<exclude body1="{n}_hip" body2="{n}_calf"/>'
             f'<exclude body1="{n}_thigh" body2="{n}_calf"/>' for n in LEGS)}
  </contact>
  <actuator>
    {''.join(f'<motor joint="{n}_{j}" ctrlrange="-24 24"/>'
             for n in LEGS for j in ("abd", "hip", "knee"))}
  </actuator>
</mujoco>"""


# --------------------------------------------------------------- leg geometry
def leg_ik(p, side):
    """Joint angles that put the foot at `p`, measured from the hip body.

    Closed form, because a 3-link leg is simple enough that iterating would
    be both slower and less reliable.  Two steps:

      1. The abduction joint turns about x, so it only moves the foot in the
         y-z plane and it cannot change the distance from the x-axis.  That
         pins it: sqrt(y^2 + z^2) must equal sqrt(L_abd^2 + u_z^2), which
         gives u_z, and then one atan2 difference gives the angle.
      2. What is left is the textbook two-link planar arm in the leg's own
         x-z plane -- law of cosines for the knee, one atan2 pair for the hip.
    """
    px, py, pz = float(p[0]), float(p[1]), float(p[2])
    yz = math.hypot(py, pz)
    uz = -math.sqrt(max(yz * yz - L_ABD * L_ABD, 1e-12))
    q_abd = math.atan2(pz, py) - math.atan2(uz, side * L_ABD)
    q_abd = math.atan2(math.sin(q_abd), math.cos(q_abd))

    ux = px
    d = math.hypot(ux, uz)
    d = min(d, L_THIGH + L_CALF - 1e-4)
    ck = (d * d - L_THIGH ** 2 - L_CALF ** 2) / (2 * L_THIGH * L_CALF)
    q_knee = math.acos(float(np.clip(ck, -1.0, 1.0)))
    alpha = math.atan2(L_CALF * math.sin(q_knee),
                       L_THIGH + L_CALF * math.cos(q_knee))
    q_hip = math.atan2(-ux, -uz) - alpha
    return np.array([q_abd, q_hip, q_knee])


def rpy_from_quat(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = 2 * (w * y - z * x)
    pitch = math.asin(np.clip(s, -1.0, 1.0))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


class Robot:
    def __init__(self, friction=0.9, payload=0.0, terrain=None, seed=0):
        xml = build_xml(friction, payload, terrain)
        self.model = mujoco.MjModel.from_xml_string(xml)
        if terrain is not None:
            rng = np.random.default_rng(seed)
            self.model.hfield_data[:] = rng.random(
                self.model.hfield_data.shape).astype(np.float32)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.foot_sid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                           f"{n}_ft") for n in LEGS]
        self.foot_gid = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           f"{n}_foot") for n in LEGS]
        self.trunk_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                           "trunk")
        self.total_mass = float(np.sum(self.model.body_mass))
        self.reset()

    def reset(self, height=STAND_H, settle=0.4):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = height + 0.01
        q_nom = []
        for i, n in enumerate(LEGS):
            side = -1.0 if n.endswith("R") else 1.0
            q = leg_ik(np.array([0.0, side * L_ABD, -(height - FOOT_R)]), side)
            self.data.qpos[7 + 3 * i:10 + 3 * i] = q
            q_nom.append(q)
        self.q_nom = np.concatenate(q_nom)
        mujoco.mj_forward(self.model, self.data)
        # Let it settle onto its feet under gravity, holding the joint angles.
        # Starting exactly tangent to the floor means MuJoCo sees no contact
        # at all, and a stance controller pushing against nothing throws the
        # robot into the air.  Starting BURIED means a launch impulse instead.
        # Dropping the last centimetre with a joint PD avoids both.
        for _ in range(int(settle / self.dt)):
            q = self.data.qpos[7:19]
            dq = self.data.qvel[6:18]
            self.step(60.0 * (self.q_nom - q) - 1.5 * dq)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------- readouts
    @property
    def p(self):
        return self.data.qpos[0:3].copy()

    @property
    def rpy(self):
        return rpy_from_quat(self.data.qpos[3:7])

    @property
    def R(self):
        return self.data.xmat[self.trunk_bid].reshape(3, 3).copy()

    @property
    def v(self):
        return self.data.qvel[0:3].copy()

    @property
    def w(self):
        return self.R @ self.data.qvel[3:6]        # world-frame angular rate

    def foot_pos(self):
        return np.array([self.data.site_xpos[s].copy() for s in self.foot_sid])

    def foot_jac(self, i):
        """3 x 12 Jacobian of foot i w.r.t. the twelve leg joints."""
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jp, jr, self.foot_sid[i])
        # Columns 0-5 are the free joint; then three joints per leg.  Only
        # leg i's own three columns are non-zero, so slice them out.
        return jp[:, 6 + 3 * i:9 + 3 * i]

    def contacts(self):
        out = np.zeros(4, bool)
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            for i, g in enumerate(self.foot_gid):
                if con.geom1 == g or con.geom2 == g:
                    out[i] = True
        return out

    def step(self, tau):
        self.data.ctrl[:] = np.clip(tau, -24.0, 24.0)
        mujoco.mj_step(self.model, self.data)

    def fallen(self):
        return self.p[2] < 0.13 or abs(self.rpy[0]) > 1.0 or abs(self.rpy[1]) > 1.0


# ------------------------------------------------------------------ the gait
class Gait:
    """A trot schedule: which feet are on the ground, and for how long.

    A gait is nothing more than a periodic contact pattern.  `offset` is where
    in the cycle each leg's stance begins, as a fraction; `duty` is what
    fraction of the cycle it spends on the ground.  A trot is offsets
    (0, 0.5, 0.5, 0) for FR, FL, RR, RL -- the two diagonal pairs exactly half
    a cycle apart -- and duty 0.5, so each pair lifts off as the other lands.
    """

    def __init__(self, period=0.34, duty=0.5,
                 offsets=(0.0, 0.5, 0.5, 0.0)):
        self.period, self.duty = period, duty
        self.offsets = np.asarray(offsets, float)

    def phase(self, t):
        return np.mod(t / self.period - self.offsets, 1.0)

    def in_stance(self, t):
        return self.phase(t) < self.duty

    def swing_frac(self, t):
        ph = self.phase(t)
        return np.clip((ph - self.duty) / max(1 - self.duty, 1e-9), 0.0, 1.0)

    def time_to_liftoff(self, t):
        ph = self.phase(t)
        return np.where(ph < self.duty, (self.duty - ph) * self.period, 0.0)


def raibert_step(v, v_des, w_des, hip_world, t_stance, k=0.06, h=STAND_H):
    """Where to put the foot down.

    Marc Raibert's rule, from the hopping robots of the 1980s and still what
    every quadruped uses: land the foot at the point the hip will be over
    half a stance from now, plus a correction proportional to how far the
    body's speed is from what you asked for.

        p_foot = p_hip + (v * T_stance / 2) + k * (v - v_des)

    The first term is "keep up with the body" -- put the foot where the body
    is going, so the leg neither trips nor trails.  The second is the
    balance term: if the body is moving faster than you wanted, step FURTHER
    ahead, which leans the support line back and slows it down.  It is the
    same instinct as sticking a foot out when you are pushed.
    """
    p = hip_world.copy()
    p[:2] += v[:2] * t_stance * 0.5 + k * (v[:2] - v_des[:2])
    p[2] = FOOT_R
    # Yaw rate moves the footprint sideways around the turn centre.
    if abs(w_des) > 1e-6:
        r = np.array([-hip_world[1], hip_world[0]])
        p[:2] += 0.5 * t_stance * w_des * r * 0.0
    return p


def swing_traj(p0, p1, s, height=0.09):
    """A foot path from lift-off to touch-down, as a function of phase s.

    Horizontal motion uses a smoothstep so the foot leaves and lands with
    zero horizontal speed (scuffing the ground is how a foot trips).  The
    vertical part is a simple arch that is zero at both ends.
    """
    a = s * s * (3 - 2 * s)
    p = p0 + (p1 - p0) * a
    p[2] = p0[2] + (p1[2] - p0[2]) * a + height * math.sin(math.pi * s)
    return p
