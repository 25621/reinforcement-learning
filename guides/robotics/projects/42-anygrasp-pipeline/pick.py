"""A tabletop, a depth camera, a parallel-jaw gripper, and a grasp you can execute.

This is the shared machinery for projects 42 and 43.  It answers one question
honestly: *if the robot closes its fingers here, does the object come up?*
Everything else in both projects is about predicting that answer cheaply.

Why a FLOATING gripper and not an arm.  Project 33 already planned arm motions
around a shelf, and it took a whole project.  Here the question is which grasp
to choose, not how to reach it, and an arm in the loop would mix the two: a
grasp scored zero because the elbow could not get there tells you nothing about
the grasp.  So the hand is welded to a mocap target we can place anywhere, and
reachability is handled separately, as an explicit filter (project 42,
experiment 5).  This is also exactly how grasp datasets like ACRONYM are built.
"""

import os

import numpy as np

import mujoco

XML = """
<mujoco model="pick">
  <compiler angle="radian"/>
  <!-- elliptic friction cones cost a little speed and are much better behaved
       when a finger is sliding on a curved surface; impratio raises friction
       stiffness relative to normal stiffness so the object does not creep out
       of the fingers under load. -->
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="8"/>
  <visual><global offwidth="640" offheight="480"/></visual>

  <default>
    <!-- Friction 0.35 is a smooth plastic on a rubber pad.  With sticky
         contacts (mu = 1) almost every candidate that fits also holds, and
         then no scorer can beat any other -- the benchmark measures
         nothing.  Slippery objects are what makes grasp CHOICE matter. -->
    <geom friction="0.35 0.02 0.001" solref="0.006 1" solimp="0.95 0.99 0.001"/>
  </default>

  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.85"
             rgb2="0.75 0.75 0.78" width="300" height="300"/>
    <material name="tablemat" texture="grid" texrepeat="6 6" reflectance="0"/>
  </asset>

  <worldbody>
    <light pos="0.3 -0.3 1.2" dir="-0.2 0.2 -1" directional="true"/>
    <geom name="table" type="plane" size="0.6 0.6 0.05" material="tablemat"/>

    <!-- One depth camera, looking down at a slant.  Straight down would make
         the point cloud complete and the problem easier than any real one;
         a slant means the far side of every object is missing, which is the
         defining difficulty of grasping from a single view. -->
    <camera name="cam" pos="0 -0.34 0.46" xyaxes="1 0 0 0 0.80 0.60" fovy="48"/>

    <body name="mocap" mocap="true" pos="0 0 0.4">
      <geom type="sphere" size="0.001" contype="0" conaffinity="0"
            rgba="1 0 0 0.0"/>
    </body>

    <body name="hand" pos="0 0 0.4">
      <freejoint name="handfree"/>
      <inertial pos="0 0 0" mass="0.6" diaginertia="0.002 0.002 0.002"/>
      <!-- The hand's origin is the BASE of the fingers.  The fingers stick
           out along +z (that is the approach direction) and the palm and
           wrist sit behind at -z.  Put the palm on the same side as the
           fingers and it reaches the object first, which looks like a
           mysterious grasp failure. -->
      <geom name="palm" type="box" size="0.030 0.018 0.013" pos="0 0 -0.014"
            rgba="0.25 0.28 0.34 1"/>
      <geom name="wrist" type="capsule" fromto="0 0 -0.030 0 0 -0.085" size="0.014"
            rgba="0.25 0.28 0.34 1"/>
      <body name="lfinger" pos="0 0 0">
        <joint name="lf" type="slide" axis="1 0 0" range="-0.045 -0.006"
               damping="4"/>
        <inertial pos="0 0 0.024" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
        <geom name="lfg" type="box" size="0.005 0.011 0.024" pos="0 0 0.024"
              rgba="0.55 0.58 0.62 1"/>
      </body>
      <body name="rfinger" pos="0 0 0">
        <joint name="rf" type="slide" axis="-1 0 0" range="-0.045 -0.006"
               damping="4"/>
        <inertial pos="0 0 0.024" mass="0.05" diaginertia="1e-5 1e-5 1e-5"/>
        <geom name="rfg" type="box" size="0.005 0.011 0.024" pos="0 0 0.024"
              rgba="0.55 0.58 0.62 1"/>
      </body>
    </body>

    OBJECTS
  </worldbody>

  <equality>
    <!-- The weld is the "arm": it drags the hand to wherever we put the mocap
         body.  solref makes it firm but not infinitely so, which is what a
         real arm is too. -->
    <weld body1="mocap" body2="hand" solref="0.02 1"/>
  </equality>

  <actuator>
    <position name="la" joint="lf" kp="120" ctrlrange="-0.045 -0.006"
              forcerange="-40 40"/>
    <position name="ra" joint="rf" kp="120" ctrlrange="-0.045 -0.006"
              forcerange="-40 40"/>
  </actuator>
</mujoco>
"""

# name -> (xml body template maker, approximate footprint radius)
FINGER_HALF_LEN = 0.024          # how deep the fingers are, along the approach
GRIP_HALF_WIDTH = 0.045          # how far each finger opens
# The point a grasp is "at" is the middle of the closing region between the
# fingertips, not the hand's own origin.  Everything upstream -- the point
# cloud, the scorer, the truth labels -- talks about that point, so `execute`
# has to subtract this offset before it places the hand.  Forget it and every
# grasp lands two centimetres too high, which looks like a scoring bug.
GRASP_Z = 0.024


def _obj_xml(i, kind, pos, yaw, rng):
    """One object body.  Composite shapes are several geoms in one body."""
    q = f"{np.cos(yaw / 2)} 0 0 {np.sin(yaw / 2)}"
    col = " ".join(f"{v:.2f}" for v in rng.uniform(0.25, 0.85, 3))
    g = []
    if kind == "box":
        a, b, c = rng.uniform(0.018, 0.032), rng.uniform(0.014, 0.024), \
            rng.uniform(0.018, 0.035)
        g.append(f'<geom type="box" size="{a} {b} {c}" rgba="{col} 1"/>')
        z = c
    elif kind == "cylinder":
        r, h = rng.uniform(0.013, 0.022), rng.uniform(0.020, 0.042)
        g.append(f'<geom type="cylinder" size="{r} {h}" rgba="{col} 1"/>')
        z = h
    elif kind == "bar":
        a, b, c = rng.uniform(0.040, 0.060), rng.uniform(0.010, 0.016), \
            rng.uniform(0.010, 0.016)
        g.append(f'<geom type="box" size="{a} {b} {c}" rgba="{col} 1"/>')
        z = c
    elif kind == "ell":
        a, b, c = 0.030, 0.012, 0.014
        g.append(f'<geom type="box" size="{a} {b} {c}" pos="0 0 0" rgba="{col} 1"/>')
        g.append(f'<geom type="box" size="{b} {a} {c}" pos="{a - b} {a - b} 0"'
                 f' rgba="{col} 1"/>')
        z = c
    else:  # "tee"
        a, b, c = 0.032, 0.011, 0.013
        g.append(f'<geom type="box" size="{a} {b} {c}" rgba="{col} 1"/>')
        g.append(f'<geom type="box" size="{b} {0.026} {c}" pos="0 -0.030 0"'
                 f' rgba="{col} 1"/>')
        z = c
    body = (f'<body name="obj{i}" pos="{pos[0]} {pos[1]} {z + 0.001}" quat="{q}">'
            f'<freejoint/>' + "".join(g) + '</body>')
    return body


KINDS_TRAIN = ["box", "cylinder", "bar"]
KINDS_TEST = ["ell", "tee"]


def make_scene(rng, n=3, kinds=None, spread=0.085):
    """Build a model with n objects dropped on the table without overlapping."""
    kinds = kinds or KINDS_TRAIN
    placed, bodies = [], []
    tries = 0
    while len(placed) < n and tries < 200:
        tries += 1
        p = rng.uniform(-spread, spread, 2)
        if any(np.linalg.norm(p - q) < 0.075 for q in placed):
            continue
        placed.append(p)
        bodies.append(_obj_xml(len(placed) - 1, kinds[rng.integers(len(kinds))],
                               p, rng.uniform(0, np.pi), rng))
    model = mujoco.MjModel.from_xml_string(XML.replace("OBJECTS", "".join(bodies)))
    data = mujoco.MjData(model)
    settle(model, data, 0.6)
    return model, data, len(placed)


def settle(model, data, t=0.5):
    """Let everything fall and stop bouncing before we look at it."""
    park(model, data)
    for _ in range(int(t / model.opt.timestep)):
        mujoco.mj_step(model, data)


def park(model, data):
    """Put the hand up and out of the camera's way."""
    data.mocap_pos[0] = np.array([0.0, 0.30, 0.45])
    data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])
    data.ctrl[:] = -GRIP_HALF_WIDTH


# ---------------------------------------------------------------------------
# the camera
# ---------------------------------------------------------------------------

class Cam:
    """MuJoCo's depth buffer, turned into 3D points in world coordinates."""

    def __init__(self, model, width=320, height=240):
        self.w, self.h = width, height
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.renderer.enable_depth_rendering()
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
        fovy = np.radians(model.cam_fovy[cid])
        # A pinhole camera's focal length in pixels follows from its vertical
        # field of view: half the image is height/2 pixels, and it subtends
        # half the fovy, so f = (height/2) / tan(fovy/2).
        self.f = 0.5 * height / np.tan(0.5 * fovy)
        self.cid = cid

    def cloud(self, model, data, stride=2, noise=0.0, rng=None):
        """A point cloud of everything above the table, in world coordinates."""
        self.renderer.update_scene(data, camera="cam")
        depth = self.renderer.render()
        if noise > 0:
            rng = rng or np.random.default_rng(0)
            depth = depth + rng.normal(0, noise, depth.shape).astype(depth.dtype)
        vv, uu = np.mgrid[0:self.h:stride, 0:self.w:stride]
        z = depth[::stride, ::stride]
        keep = z < 2.0
        x = (uu - self.w / 2 + 0.5) / self.f * z
        y = -(vv - self.h / 2 + 0.5) / self.f * z
        # MuJoCo's camera looks down its own -z axis, with +x right and +y up
        P_cam = np.stack([x[keep], y[keep], -z[keep]], 1)
        R = data.cam_xmat[self.cid].reshape(3, 3)
        P = P_cam @ R.T + data.cam_xpos[self.cid]
        return P

    def rgb(self, model, data):
        r = mujoco.Renderer(model, height=self.h, width=self.w)
        r.update_scene(data, camera="cam")
        img = r.render()
        r.close()
        return img


def table_removed(P, z_min=0.006, box=0.16):
    """Everything standing on the table, with the table itself dropped.

    A single plane fit (RANSAC) is what you would run on real data; here the
    table's height is known exactly, so a threshold is the same operation with
    the fitting step already done.  Doing RANSAC anyway would only measure
    RANSAC, which project 19 already covers.
    """
    m = (P[:, 2] > z_min) & (np.abs(P[:, 0]) < box) & (np.abs(P[:, 1]) < box)
    return P[m]


# ---------------------------------------------------------------------------
# grasp representation
# ---------------------------------------------------------------------------

def grasp_frame(approach, closing):
    """Rotation matrix whose z axis is the approach and x axis is the closing.

    The gripper model closes along its own x and reaches along its own z, so a
    grasp is fully described by those two orthogonal unit vectors plus a point.
    """
    z = approach / np.linalg.norm(approach)
    x = closing - (closing @ z) * z
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
        nx = np.linalg.norm(x)
    x = x / nx
    y = np.cross(z, x)
    return np.stack([x, y, z], 1)


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def execute(model, data, grasp, cam=None, record=False, lift=0.20,
            shake=0.05, shake_time=0.7):
    """Drive the hand to a grasp, close, lift, and report whether it worked.

    Four stages, and the pre-grasp stage is the one people leave out.  Dropping
    the hand straight onto the grasp point from above would push objects over
    on the way in; approaching from a stand-off along the hand's own approach
    axis is what makes the recorded outcome about the GRASP rather than about
    the trajectory.
    """
    pos = np.asarray(grasp["pos"], float)
    R = grasp_frame(grasp["approach"], grasp["closing"])
    quat = mat2quat(R)
    approach = R[:, 2]
    frames = []

    obj_z0 = data.xipos[_obj_body_ids(model)][:, 2].copy()

    def hold(target, ctrl, steps):
        for k in range(steps):
            a = min(1.0, (k + 1) / max(steps * 0.5, 1))
            data.mocap_pos[0] = start + a * (target - start)
            data.mocap_quat[0] = quat
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            if record and cam is not None and k % 25 == 0:
                frames.append(cam.rgb(model, data))

    # 1. jump to the pre-grasp pose (nothing is touching yet, so a teleport is
    #    honest here and saves half the simulation time)
    hand_pos = pos - approach * GRASP_Z
    pre = hand_pos - approach * 0.11
    data.qpos[_hand_qadr(model):_hand_qadr(model) + 3] = pre
    data.qpos[_hand_qadr(model) + 3:_hand_qadr(model) + 7] = quat
    data.qvel[:] = 0
    data.mocap_pos[0] = pre
    data.mocap_quat[0] = quat
    data.ctrl[:] = -GRIP_HALF_WIDTH
    mujoco.mj_forward(model, data)
    start = pre.copy()
    hold(pre, -GRIP_HALF_WIDTH, 30)
    # 2. move in along the approach axis
    start = pre.copy()
    hold(hand_pos, -GRIP_HALF_WIDTH, 200)
    # 3. close
    start = hand_pos.copy()
    hold(hand_pos, -0.006, 150)
    # 4. lift
    start = hand_pos.copy()
    top = hand_pos + np.array([0.0, 0.0, lift])
    hold(top, -0.006, 300)
    # 5. shake.  This is project 39's quality metric made physical: a grasp
    #    that merely holds the weight is not the same as a grasp that resists a
    #    disturbance, and lifting alone cannot tell them apart.  Real grasp
    #    benchmarks (and real warehouses, which accelerate hard) shake too.
    start = top.copy()
    for k in range(int(shake_time / model.opt.timestep)):
        t = k * model.opt.timestep
        data.mocap_pos[0] = top + np.array(
            [shake * np.sin(2 * np.pi * 3.0 * t), 0.0, 0.0])
        data.mocap_quat[0] = quat
        data.ctrl[:] = -0.006
        mujoco.mj_step(model, data)
        if record and cam is not None and k % 25 == 0:
            frames.append(cam.rgb(model, data))

    z1 = data.xipos[_obj_body_ids(model)][:, 2]
    rise = z1 - obj_z0
    k = int(np.argmax(rise))
    ok = bool(rise[k] > 0.6 * lift)
    return dict(success=ok, lifted=k if ok else -1, rise=float(rise[k]),
                frames=frames)


def _obj_body_ids(model):
    """Body ids of the objects (not the hand, not the table).

    Deliberately NOT cached on id(model).  Every scene builds a fresh MjModel,
    the old one is garbage collected, and CPython happily hands the same
    memory address to the next one -- so an id-keyed cache silently returns
    the previous scene's body ids.  That reads object heights from the wrong
    bodies, which mislabels the training data long before it crashes.
    """
    return np.array([i for i in range(model.nbody)
                     if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                         or "").startswith("obj")])


def _hand_qadr(model):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "handfree")
    return model.jnt_qposadr[jid]


def snapshot(model, data):
    return (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy())


def restore(model, data, snap):
    data.qpos[:], data.qvel[:], data.ctrl[:] = snap
    mujoco.mj_forward(model, data)
