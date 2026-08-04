"""Reaching with the Phase 8 arm, many copies at once.

Project 54's ``PlanarArm`` steps one robot at a time, which is fine when a
scripted expert is doing the work.  Reinforcement learning needs tens of
thousands of interactions, and Python spends most of its time on interpreter
overhead rather than on physics, so here the same closed-form two-link
equations are applied to whole arrays of robots.  ``verify()`` checks the
vectorised arm against project 54's one, state by state -- if they ever
disagree, every number in this project is measuring the wrong robot.

The task is a REACH, as the guide asks: put the tip on a target and hold it.
Reaching is deliberately easier than project 54's push, because what this
project is studying is the learning algorithm (SAC and its entropy
temperature), not the task.  A hard task would hide the effect of alpha behind
its own difficulty.
"""

import os
import sys

import numpy as np

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402

DT = A.DT
SUBSTEPS = A.SUBSTEPS
DQ_MAX = A.DQ_MAX
EP_LEN = 40                # 2 seconds
SUCCESS_TOL = 0.02         # metres


class BatchReach:
    """``n_env`` planar arms, each reaching for its own target."""

    obs_dim = 12
    act_dim = 2

    def __init__(self, n_env=8, seed=0, mass_scale=1.0, damp_scale=1.0,
                 gear=1.0, sparse=False, reward_mode="bonus", arm=None):
        self.arm = arm or A.PlanarArm(mass_scale=mass_scale,
                                      damp_scale=damp_scale, gear=gear)
        self.n = n_env
        self.rng = np.random.default_rng(seed)
        self.sparse = sparse
        self.reward_mode = "sparse" if sparse else reward_mode
        a = self.arm
        (l1, _), (m1, m2), (r1, r2), (I1, I2) = a.l, a.m, a.r, a.I
        self.c_m11 = I1 + m1 * r1 * r1 + I2 + m2 * (l1 * l1 + r2 * r2)
        self.c_m11c = 2 * m2 * l1 * r2
        self.c_m12 = I2 + m2 * r2 * r2
        self.c_m12c = m2 * l1 * r2
        self.c_m22 = I2 + m2 * r2 * r2
        self.c_h = m2 * l1 * r2
        self.reset()

    # -- physics ------------------------------------------------------------
    def _accel(self, q, qd, tau):
        c2, s2 = np.cos(q[:, 1]), np.sin(q[:, 1])
        m11 = self.c_m11 + self.c_m11c * c2
        m12 = self.c_m12 + self.c_m12c * c2
        m22 = self.c_m22
        h = self.c_h * s2
        q1d, q2d = qd[:, 0], qd[:, 1]
        b1 = -h * (2 * q1d * q2d + q2d * q2d) + self.arm.b[0] * q1d
        b2 = h * q1d * q1d + self.arm.b[1] * q2d
        y1, y2 = tau[:, 0] - b1, tau[:, 1] - b2
        det = m11 * m22 - m12 * m12
        return np.stack([(m22 * y1 - m12 * y2) / det,
                         (m11 * y2 - m12 * y1) / det], axis=1)

    def tip(self, q=None):
        q = self.q if q is None else q
        l1, l2 = self.arm.l
        a, b = q[:, 0], q[:, 0] + q[:, 1]
        return np.stack([l1 * np.cos(a) + l2 * np.cos(b),
                         l1 * np.sin(a) + l2 * np.sin(b)], axis=1)

    # -- episode ------------------------------------------------------------
    def sample_goals(self, n):
        th = self.rng.uniform(-0.5, 1.6, n)
        rr = self.rng.uniform(0.40, 0.85, n) * self.arm.reach
        return np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)

    def _reset_idx(self, idx):
        n = len(idx)
        self.q[idx] = np.stack([self.rng.uniform(-0.4, 0.9, n),
                                self.rng.uniform(0.6, 2.0, n)], axis=1)
        self.qd[idx] = 0.0
        self.goal[idx] = self.sample_goals(n)
        self.t[idx] = 0

    def reset(self):
        self.q = np.zeros((self.n, 2))
        self.qd = np.zeros((self.n, 2))
        self.goal = np.zeros((self.n, 2))
        self.t = np.zeros(self.n, dtype=int)
        self._reset_idx(np.arange(self.n))
        return self.obs()

    def obs(self):
        tip = self.tip()
        return np.concatenate([np.cos(self.q), np.sin(self.q), self.qd / 10.0,
                               tip, self.goal, self.goal - tip], axis=1)

    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        q_des = self.q + DQ_MAX * a
        for _ in range(SUBSTEPS):
            tau = np.clip(self.arm.gear * (self.arm.kp * (q_des - self.q)
                                           - self.arm.kd * self.qd),
                          -self.arm.tau_max, self.arm.tau_max)
            self.qd = self.qd + DT * self._accel(self.q, self.qd, tau)
            self.q = self.q + DT * self.qd
        d = np.linalg.norm(self.tip() - self.goal, axis=1)
        hit = (d < SUCCESS_TOL).astype(float)
        if self.reward_mode == "sparse":
            # Only a hit pays.  Nothing tells the arm which way to move until
            # it stumbles onto the target by chance -- which is the entire
            # difficulty of sparse reward, made concrete.
            r = hit
        elif self.reward_mode == "dense":
            r = -d - 0.01 * (a ** 2).sum(1)
        else:
            # Dense PLUS a bonus for actually being on target.  The pure
            # distance reward is dense everywhere and nearly FLAT at the end:
            # over the twenty steps left in an episode, the difference between
            # stopping 2 cm out and stopping 2 mm out is worth about 0.36 of
            # return, which is inside the noise.  The bonus puts a cliff at the
            # tolerance, so the last centimetre is worth paying for.
            # Experiment 4 measures exactly how much this is worth.
            r = -d - 0.01 * (a ** 2).sum(1) + 2.0 * hit
        self.t += 1
        done = self.t >= EP_LEN
        info = dict(dist=d, hit=d < SUCCESS_TOL)
        if done.any():
            self._reset_idx(np.where(done)[0])
        return self.obs(), r, done.astype(float), info


def evaluate(controller, n_ep=100, seed=4242, **kw):
    """Mean final tip error and success rate over fresh episodes."""
    env = BatchReach(n_ep, seed=seed, **kw)
    for _ in range(EP_LEN):
        # The last step auto-resets the episode, so the distance has to be
        # read out of the info dict -- reading env.tip() afterwards measures a
        # brand-new episode and scores every controller identically.
        _, _, _, info = env.step(controller(env.obs()))
    d = info["dist"]
    return dict(dist=float(d.mean()), success=float((d < SUCCESS_TOL).mean()))


def ik_controller(arm=None, gain=12.0, v_max=0.45):
    """The classical answer: damped least squares, as in project 05.

    This is the reference the learned policy is measured against.  It needs no
    data at all -- but it needs the kinematics, and it only works because the
    task is a reach.  Nothing about it transfers to contact.

    It reads the joint angles out of the OBSERVATION rather than out of an env
    object.  That is not fussiness: a controller holding a reference to one env
    while being evaluated inside another silently computes its Jacobian from
    the wrong robot, and scores a mysterious 82% instead of 100%.
    """
    arm = arm or A.PlanarArm()

    def ctrl(obs):
        q = np.arctan2(obs[:, 2:4], obs[:, 0:2])       # cos, sin -> angle
        tip, goal = obs[:, 6:8], obs[:, 8:10]
        v = (goal - tip) * gain
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(sp > v_max, v * v_max / np.maximum(sp, 1e-9), v)
        out = np.zeros((len(obs), 2))
        for i in range(len(obs)):
            J = arm.jacobian(q[i])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(2),
                                       v[i] * A.CTRL_DT)
            out[i] = dq / DQ_MAX
        return np.clip(out, -1, 1)
    return ctrl


def verify():
    """Vectorised arm vs project 54's single arm, 40 steps of the same input."""
    env = BatchReach(4, seed=0)
    single = A.PlanarArm()
    rng = np.random.default_rng(0)
    q0 = env.q.copy()
    qd0 = env.qd.copy()
    n_steps = EP_LEN - 1           # stop before the auto-reset
    acts = rng.uniform(-1, 1, (n_steps, 4, 2))
    for k in range(n_steps):
        env.step(acts[k])
    worst = 0.0
    for i in range(4):
        q, qd = q0[i].copy(), qd0[i].copy()
        for k in range(n_steps):
            q_des = q + DQ_MAX * acts[k, i]
            for _ in range(SUBSTEPS):
                tau = single.servo_torque(q, qd, q_des)
                q, qd = single.step(q, qd, tau)
        worst = max(worst, float(np.abs(q - env.q[i]).max()))
    return worst


if __name__ == "__main__":
    print("max |batched - project 54 arm| over an episode :", verify())
    print("IK controller  :", evaluate(ik_controller()))
    print("zero action    :", evaluate(lambda o: np.zeros((len(o), 2))))
    print("random action  :", evaluate(lambda o: np.random.uniform(-1, 1, (len(o), 2))))
