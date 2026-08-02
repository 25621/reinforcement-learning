"""A planar two-finger hand, a square block, and a goal angle.

Why planar, and why only two fingers.  Real in-hand manipulation results
(OpenAI's Rubik's cube, the Shadow hand papers) use a 24-joint hand, pixels or
touch, and tens of thousands of GPU-hours.  Nothing about that fits here.  What
does fit is the part that makes in-hand manipulation *different* from every
other control problem in this guide: the contacts are not fixed.  Which finger
touches which face, and whether it is rolling or sliding, changes several times
during one rotation, and the controller has to keep working across those
changes.  Two fingers and a square in a plane reproduce that faithfully and run
at ten thousand steps a second.

The honest limitation, stated up front: a planar two-finger hand cannot rotate
a square indefinitely.  Past a certain angle a finger runs out of travel and
the only way on is to *let go and re-grasp* -- finger gaiting -- which this
setup deliberately does not have.  Experiment 4 measures exactly where that
wall is, and it is the most useful number in the project.
"""

import numpy as np

import mujoco

XML = """
<mujoco model="inhand">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic"
          impratio="10" gravity="0 0 -9.81"/>
  <visual><global offwidth="480" offheight="360"/></visual>

  <default>
    <geom friction="MU 0.02 0.001" solref="0.005 1" solimp="0.95 0.99 0.001"/>
    <joint damping="0.06" armature="0.002"/>
    <position kp="KP" forcerange="-3.5 3.5"/>
  </default>

  <worldbody>
    <light pos="0 -0.6 0.6" dir="0 1 -1" directional="true"/>
    <camera name="cam" pos="0 -0.45 0.10" xyaxes="1 0 0 0 0.2 1" fovy="30"/>

    <!-- The block is held IN THE AIR by the two fingers; this plate is only
         a floor to catch it when they lose it.  Resting the block on a shelf
         instead sounds easier, but then rotating it means fighting the shelf's
         friction with the fingers, and the experiment stops being about the
         two contacts.  Held in the air, the grasp is exactly project 39's
         two-frictional-contact problem, and rotating it means rolling those
         two contacts around the block. -->
    <geom name="floor" type="box" pos="0 0 -0.075" size="0.12 0.04 0.010"
          rgba="0.30 0.33 0.40 1"/>

    <!-- left finger: two links, hinging about y so everything stays planar -->
    <body name="l0" pos="-0.075 0 0.012">
      <joint name="lj0" type="hinge" axis="0 -1 0" range="-1.2 1.2"/>
      <geom type="capsule" fromto="0 0 0 0.045 0 0" size="0.007"
            rgba="0.80 0.55 0.25 1"/>
      <body name="l1" pos="0.045 0 0">
        <joint name="lj1" type="hinge" axis="0 -1 0" range="-2.3 2.3"/>
        <geom type="capsule" fromto="0 0 0 0.035 0 0" size="0.0065"
              rgba="0.88 0.68 0.35 1"/>
      </body>
    </body>

    <body name="r0" pos="0.075 0 0.012">
      <joint name="rj0" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
      <geom type="capsule" fromto="0 0 0 -0.045 0 0" size="0.007"
            rgba="0.25 0.55 0.80 1"/>
      <body name="r1" pos="-0.045 0 0">
        <joint name="rj1" type="hinge" axis="0 1 0" range="-2.3 2.3"/>
        <geom type="capsule" fromto="0 0 0 -0.035 0 0" size="0.0065"
              rgba="0.40 0.70 0.92 1"/>
      </body>
    </body>

    <!-- the block: three joints is a full planar free body -->
    <body name="block" pos="0 0 BLOCKH">
      <joint name="bx" type="slide" axis="1 0 0"/>
      <joint name="bz" type="slide" axis="0 0 1"/>
      <joint name="bth" type="hinge" axis="0 -1 0"/>
      <geom type="box" size="BLOCKS 0.020 BLOCKS" density="DENS"
            rgba="0.85 0.35 0.35 1"/>
      <geom type="box" size="BLOCKSMALL 0.0205 BLOCKSMALL" pos="BLOCKS 0 BLOCKS"
            rgba="0.98 0.92 0.45 1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>

  <actuator>
    <position name="a0" joint="lj0" ctrlrange="-1.2 1.2"/>
    <position name="a1" joint="lj1" ctrlrange="-2.3 2.3"/>
    <position name="a2" joint="rj0" ctrlrange="-1.2 1.2"/>
    <position name="a3" joint="rj1" ctrlrange="-2.3 2.3"/>
  </actuator>
</mujoco>
"""

NU = 4
SUBSTEPS = 10
EP_LEN = 140
# Both fingertips just TOUCHING the block's vertical faces at mid height,
# solved once by search over the two joint angles.  Two traps here, and each
# one launched the block across the room before it was fixed:
#   * the obvious-looking all-zeros pose has the fingers CROSSING each other
#     (each reaches 80 mm from a base 75 mm out);
#   * the fingertip is a CAPSULE of radius 6.5 mm, so its centre line has to
#     stop 6.5 mm short of the face, not on it.  Aim the centre line at the
#     surface and the finger starts 6.5 mm inside solid material, which the
#     contact solver resolves by firing the block upward.
HOME = np.array([0.868, -1.819, 0.868, -1.819])
SQUEEZE = -0.14     # joint offset that presses the fingertips in


def build(mu=1.1, kp=4.0, block=0.018, density=400.0):
    xml = (XML.replace("MU", f"{mu}").replace("KP", f"{kp}")
           .replace("BLOCKSMALL", f"{0.25 * block}")
           .replace("BLOCKS", f"{block}")
           .replace("BLOCKH", f"{block + 0.0005}")
           .replace("DENS", f"{density}"))
    return mujoco.MjModel.from_xml_string(xml)


class Hand:
    """One episode: hold the block and turn it to `goal` radians."""

    def __init__(self, rng, mu=1.1, kp=4.0, block=0.018, density=400.0,
                 goal_range=(-0.6, 0.6)):
        self.rng = rng
        self.params = dict(mu=mu, kp=kp, block=block, density=density)
        self.model = build(**self.params)
        self.data = mujoco.MjData(self.model)
        self.goal_range = goal_range
        self.bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block")

    def reset(self, goal=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:NU] = HOME + self.rng.normal(0, 0.015, NU)
        self.data.qpos[NU + 2] = float(self.rng.uniform(-0.10, 0.10))
        # command the fingers slightly PAST the block faces; the servo error
        # that results is the grip force
        self.data.ctrl[:] = HOME + np.array([SQUEEZE, -SQUEEZE,
                                             SQUEEZE, -SQUEEZE])
        self.goal = float(goal if goal is not None
                          else self.rng.uniform(*self.goal_range))
        # let the fingers settle onto the block before the clock starts
        for _ in range(120):
            mujoco.mj_step(self.model, self.data)
        self.t = 0
        self.dropped = False
        return self.obs()

    def obs(self):
        q = self.data.qpos[:NU]
        v = self.data.qvel[:NU]
        bx, bz, bth = self.data.qpos[NU:NU + 3]
        bv = self.data.qvel[NU:NU + 3]
        err = self.goal - bth
        return np.concatenate([q, 0.1 * v, [bx / 0.05, bz / 0.05,
                                            np.sin(bth), np.cos(bth)],
                               0.1 * bv, [np.sin(err), np.cos(err), err]])

    def angle(self):
        return float(self.data.qpos[NU + 2])

    def step(self, action):
        """Action is a small offset added to the resting finger pose."""
        ctrl = (HOME + np.array([SQUEEZE, -SQUEEZE, SQUEEZE, -SQUEEZE])
                + np.clip(action, -1, 1) * 0.60)
        lo, hi = self.model.actuator_ctrlrange.T
        self.data.ctrl[:] = np.clip(ctrl, lo, hi)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        bz = self.data.qpos[NU + 1]
        if bz < -0.030 or abs(self.data.qpos[NU]) > 0.06:
            self.dropped = True
        err = abs(self.goal - self.angle())
        # -|error| every step rewards getting there EARLY and staying, which is
        # what "reorient the block" means; a reward given only at the last step
        # would be a single number per episode and far harder to learn from.
        #
        # The drop penalty has to cover ALL THE STEPS THAT WILL NOT HAPPEN.
        # Dropping ends the episode, so with a small fixed penalty the fastest
        # way to stop losing points is to throw the block away immediately --
        # and the search finds that in about ten iterations.  Charging for the
        # remaining steps at a worse-than-worst rate makes holding on strictly
        # better than quitting.
        r = -err
        if self.dropped:
            r -= 5.0 + 1.5 * (EP_LEN - self.t)
        return self.obs(), r, (self.t >= EP_LEN or self.dropped)

    def success(self, tol=0.26):
        return bool((not self.dropped) and abs(self.goal - self.angle()) < tol)


def rollout(hand, W, b=None, goal=None, render=False, cam=None):
    """Run one episode with a LINEAR policy: action = W @ obs + b."""
    o = hand.reset(goal)
    total = 0.0
    frames = []
    for _ in range(EP_LEN):
        a = W @ o + (b if b is not None else 0.0)
        if render and cam is not None and hand.t % 14 == 0:
            cam.update_scene(hand.data, camera="cam")
            frames.append(cam.render())
        o, r, done = hand.step(a)
        total += r
        if done:
            break
    return dict(ret=total, success=hand.success(), dropped=hand.dropped,
                final=hand.angle(), goal=hand.goal, frames=frames)
