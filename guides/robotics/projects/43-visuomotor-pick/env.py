"""A pick task with a fixed camera, a scripted expert, and a step interface.

The simulator, the gripper and the objects are project 42's -- imported, not
copied.  What is new is the *interface*: 42 asked "is this grasp good?" and
answered it in one shot, while here the robot has to steer itself to the
object over twenty-odd decisions, each made from a picture.

That difference is the whole point of the project.  A grasp detector needs one
correct answer; a closed-loop policy needs every intermediate state it visits
to also be one it knows what to do in, and nothing in supervised learning
guarantees that.
"""

import os
import sys

import numpy as np

import mujoco

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_PROJ, "42-anygrasp-pipeline"))

import pick                                                                   # noqa: E402

# Project 42 made its objects deliberately slippery (mu = 0.35) so that WHICH
# grasp you choose decides whether it holds -- that was the thing it measured.
# Here the thing being measured is whether a policy can steer itself to a good
# grasp from pictures, so the grasp itself must be reliable; otherwise every
# score is half policy skill and half coin flip.  Same simulator, one number
# changed, and the change is stated rather than hidden.
pick.XML = pick.XML.replace('friction="0.35 0.02 0.001"',
                            'friction="0.9 0.02 0.001"')

IMG_W, IMG_H = 84, 64
MAX_STEP = 0.022          # metres the hand may move per decision
MAX_YAW = 0.35            # radians it may turn per decision
SUBSTEPS = 55             # simulator steps per decision
EP_LEN = 24
HOVER = 0.16
KINDS = ["cylinder", "box"]


class PickEnv:
    """One object on a table, a hand above it, and a 5-number action."""

    def __init__(self, rng, kinds=None, n_distract=0, cam_shift=0.0):
        self.rng = rng
        self.kinds = kinds or KINDS
        self.n_distract = n_distract
        self.cam_shift = cam_shift
        self.model = None

    def reset(self):
        n = 1 + self.n_distract
        self.model, self.data, _ = pick.make_scene(self.rng, n=n,
                                                   kinds=self.kinds, spread=0.075)
        if self.cam_shift:
            # Move the camera and re-render.  Nothing about the task changes;
            # only the pictures do.  Experiment 7 is entirely about what that
            # does to a policy that learned from pixels.
            cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
            self.model.cam_pos[cid, 0] += self.cam_shift
        self.renderer = mujoco.Renderer(self.model, height=IMG_H, width=IMG_W)
        self.obj = pick._obj_body_ids(self.model)[0]
        self.hand_b = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        # start the hand somewhere above the table, deliberately not over the
        # object -- a policy that never has to move sideways learns nothing
        self.pos = np.array([self.rng.uniform(-0.07, 0.07),
                             self.rng.uniform(-0.07, 0.07),
                             HOVER])
        self.yaw = float(self.rng.uniform(-np.pi / 2, np.pi / 2))
        self.grip = 1.0
        self.z0 = float(self.data.xipos[self.obj, 2])
        self._apply(teleport=True)
        for _ in range(40):
            mujoco.mj_step(self.model, self.data)
        self.t = 0
        return self.obs()

    def _quat(self):
        # top-down: the hand's +z (its approach) points down, then rolled by yaw
        c, s = np.cos(self.yaw / 2), np.sin(self.yaw / 2)
        qz = np.array([c, 0.0, 0.0, s])
        qflip = np.array([0.0, 1.0, 0.0, 0.0])      # 180 deg about x
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qz, qflip)
        return out

    def _apply(self, teleport=False):
        q = self._quat()
        self.data.mocap_pos[0] = self.pos
        self.data.mocap_quat[0] = q
        self.data.ctrl[:] = -0.045 if self.grip > 0.5 else -0.006
        if teleport:
            a0 = pick._hand_qadr(self.model)
            self.data.qpos[a0:a0 + 3] = self.pos
            self.data.qpos[a0 + 3:a0 + 7] = q
            self.data.qvel[:] = 0
            mujoco.mj_forward(self.model, self.data)

    def step(self, action):
        d = np.clip(action[:3], -1, 1) * MAX_STEP
        self.pos = self.pos + d
        self.pos[0] = np.clip(self.pos[0], -0.14, 0.14)
        self.pos[1] = np.clip(self.pos[1], -0.14, 0.14)
        self.pos[2] = np.clip(self.pos[2], 0.030, 0.28)
        self.yaw = float(np.clip(self.yaw + np.clip(action[3], -1, 1) * MAX_YAW,
                                 -np.pi, np.pi))
        self.grip = float(action[4])
        self._apply()
        for _ in range(SUBSTEPS):
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        return self.obs(), self.success(), self.t >= EP_LEN

    def obs(self):
        self.renderer.update_scene(self.data, camera="cam")
        img = self.renderer.render().astype(np.float32) / 255.0
        prop = np.array([self.pos[0] / 0.15, self.pos[1] / 0.15,
                         (self.pos[2] - 0.15) / 0.15,
                         np.sin(self.yaw), np.cos(self.yaw), self.grip],
                        np.float32)
        op = self.data.xipos[self.obj]
        oq = self.data.xquat[self.obj]
        oyaw = np.arctan2(2 * (oq[0] * oq[3] + oq[1] * oq[2]),
                          1 - 2 * (oq[2] ** 2 + oq[3] ** 2))
        priv = np.array([op[0] / 0.15, op[1] / 0.15, (op[2] - 0.03) / 0.05,
                         np.sin(2 * oyaw), np.cos(2 * oyaw)], np.float32)
        return dict(img=img.transpose(2, 0, 1), prop=prop, priv=priv)

    def success(self):
        return bool(self.data.xipos[self.obj, 2] - self.z0 > 0.08)

    def close(self):
        self.renderer.close()


# ---------------------------------------------------------------------------
# the scripted expert
# ---------------------------------------------------------------------------

def expert_action(env):
    """A four-phase script that reads the object's true pose.

    It is *privileged*: it sees the exact object position and orientation, so
    it cannot be deployed on a real robot without a perfect pose estimator.
    That is the point.  The policy we train has to reproduce this behaviour
    from a picture, and the whole project is about how much of it survives the
    translation.
    """
    # Latch: once the fingers are shut, the only thing left to do is lift.
    # Without this the script re-runs its alignment test, sees the hand is low,
    # opens again, and loops forever -- and every demonstration it records
    # teaches the policy to do the same.
    if env.grip < 0.5:
        return np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    op = env.data.xipos[env.obj]
    oq = env.data.xquat[env.obj]
    oyaw = np.arctan2(2 * (oq[0] * oq[3] + oq[1] * oq[2]),
                      1 - 2 * (oq[2] ** 2 + oq[3] ** 2))
    # close ACROSS the object's long axis, and pick whichever equivalent yaw
    # is nearest to where the wrist already is (a gripper is symmetric under a
    # half turn, so chasing the raw angle would spin the wrist for nothing)
    want = oyaw + np.pi / 2
    dy = (want - env.yaw + np.pi / 2) % np.pi - np.pi / 2
    dxy = op[:2] - env.pos[:2]
    aligned = np.linalg.norm(dxy) < 0.006 and abs(dy) < 0.10
    # Aim the middle of the fingers at the object's mid-height, but never so
    # low that the fingertips would dig into the table: the fingers are 48 mm
    # long and stick out below the hand, so a short object has to be gripped
    # near its top, not at its centre of mass.
    grasp_z = max(float(op[2]), 0.028) + 0.024
    if not aligned and env.pos[2] > 0.10:
        a = np.array([dxy[0], dxy[1], (0.13 - env.pos[2]) * 0.5])
        return np.concatenate([a / MAX_STEP, [dy / MAX_YAW, 1.0]])
    if env.pos[2] > grasp_z + 0.004:
        a = np.array([dxy[0], dxy[1], grasp_z - env.pos[2]])
        return np.concatenate([a / MAX_STEP, [dy / MAX_YAW, 1.0]])
    if env.grip > 0.5:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0])          # close
    return np.array([0.0, 0.0, 1.0, 0.0, 0.0])              # lift


def rollout(env, policy=None, render=False):
    """One episode.  `policy(obs) -> action`; None means use the expert."""
    obs = env.reset()
    A, O, frames = [], [], []
    ok = False
    for _ in range(EP_LEN):
        a = expert_action(env) if policy is None else policy(obs)
        O.append(obs)
        A.append(np.clip(a, -1, 1))
        if render:
            frames.append((obs["img"].transpose(1, 2, 0) * 255).astype(np.uint8))
        obs, ok, done = env.step(a)
        if done:
            break
    return dict(obs=O, act=np.array(A, np.float32), success=env.success(),
                frames=frames)
