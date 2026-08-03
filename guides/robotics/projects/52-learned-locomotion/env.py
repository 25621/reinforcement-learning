"""A reinforcement-learning environment around project 51's quadruped.

The guide asks for Isaac Lab, which trains thousands of robots in parallel on
a GPU.  There is no usable GPU here, so this project runs the *same recipe* at
a scale a laptop can afford: one MuJoCo robot, a linear policy, and Augmented
Random Search across the CPU cores.  Everything that makes the recipe work is
kept -- the observation set, the reward terms, the joint-position action
space, the domain randomisation -- and only the parallelism is smaller.  The
numbers below are therefore about the *method*, not about how fast a GPU is.

Three design choices are worth calling out.  The first is load-bearing; the
other two are the standard recipe, and experiment 3 in `run.py` measures
whether they actually earn their place here (the answer is not the one the
recipe predicts, which is why the experiment is in the project).

  * **The action is a joint-angle OFFSET, not a torque.**  The policy outputs
    twelve small numbers added to a fixed standing pose, and a stiff joint PD
    turns those into torques.  A policy that outputs torques directly has to
    learn gravity compensation before it can learn to walk; a policy that
    outputs positions gets that for free from the PD, and only has to learn
    the shape of the gait.
  * **The observation includes a gait clock** -- two numbers, sin and cos of a
    fixed-frequency phase.  The argument for it: without a clock the policy
    cannot tell "early in the stride" from "late in the stride", since the
    rest of the observation is nearly the same at both, and a linear map has
    no memory to invent one with.
  * **The observation includes the previous action**, giving the policy a
    memory of exactly one step -- the usual way a memoryless linear map is
    coaxed into producing something periodic.

`obs_drop` zeroes any of these out, which is how the ablation is run.
"""

import math
import os
import sys

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "51-quadruped-trotting-mpc"))

from quadruped import LEGS, STAND_H, Robot                      # noqa: E402

CTRL_DT = 0.02                    # 50 Hz policy
GAIT_HZ = 2.9                     # the clock the policy is handed
N_ACT = 12


class WalkEnv:
    def __init__(self, rng=None, v_cmd=(0.6, 0.0), friction=0.9, payload=0.0,
                 terrain=None, torque_scale=1.0, ep_time=4.0, seed=0,
                 obs_drop=(), action_scale=0.30, kp=48.0, kd=1.3,
                 reward_off=()):
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self.rb = Robot(friction=friction, payload=payload, terrain=terrain,
                        seed=seed)
        self.q_nom = self.rb.q_nom.copy()
        self.v_cmd = np.asarray(v_cmd, float)
        self.torque_scale = torque_scale
        self.ep_len = int(ep_time / CTRL_DT)
        self.sub = max(int(round(CTRL_DT / self.rb.dt)), 1)
        self.action_scale = action_scale
        self.kp, self.kd = kp, kd
        self.obs_drop = set(obs_drop)
        self.reward_off = set(reward_off)
        self.t = 0

    # ---------------------------------------------------------------- obs
    def obs(self):
        rb = self.rb
        g_body = rb.R.T @ np.array([0.0, 0.0, -1.0])   # gravity, seen by the IMU
        v_body = rb.R.T @ rb.v
        w_body = rb.data.qvel[3:6].copy()
        q = rb.data.qpos[7:19] - self.q_nom
        dq = rb.data.qvel[6:18]
        ph = 2 * math.pi * GAIT_HZ * self.t * CTRL_DT
        parts = {
            "height": np.array([rb.p[2] - STAND_H]),
            "gravity": g_body,
            "lin_vel": v_body,
            "ang_vel": w_body,
            "joint_pos": q,
            "joint_vel": dq * 0.1,
            "prev_action": self.prev_a,
            "command": self.v_cmd,
            "clock": np.array([math.sin(ph), math.cos(ph)]),
        }
        for k in self.obs_drop:
            parts[k] = np.zeros_like(parts[k])
        return np.concatenate([parts[k] for k in
                               ("height", "gravity", "lin_vel", "ang_vel",
                                "joint_pos", "joint_vel", "prev_action",
                                "command", "clock")])

    @property
    def n_obs(self):
        return 1 + 3 + 3 + 3 + 12 + 12 + 12 + 2 + 2

    def reset(self, v_cmd=None):
        if v_cmd is not None:
            self.v_cmd = np.asarray(v_cmd, float)
        self.rb.reset()
        self.prev_a = np.zeros(N_ACT)
        self.t = 0
        return self.obs()

    # -------------------------------------------------------------- step
    def step(self, a):
        a = np.clip(a, -1.0, 1.0)
        q_des = self.q_nom + self.action_scale * a
        for _ in range(self.sub):
            q = self.rb.data.qpos[7:19]
            dq = self.rb.data.qvel[6:18]
            tau = self.torque_scale * (self.kp * (q_des - q) - self.kd * dq)
            self.rb.step(tau)
        self.prev_a = a
        self.t += 1
        r, done = self.reward(a)
        return self.obs(), r, done

    def reward(self, a):
        rb = self.rb
        v_body = rb.R.T @ rb.v
        terms = {
            # Track the commanded velocity.  Exponential rather than a plain
            # negative square, so the reward is bounded and a catastrophic
            # step cannot dominate a whole episode.
            "track": 3.2 * math.exp(-6.0 * float(
                np.sum((v_body[:2] - self.v_cmd) ** 2))),
            "alive": 0.3,
            # Stay level and at height: without these the fastest way to move
            # forward is to fall forward, and the policy will find it.
            "upright": -1.2 * float(np.sum(rb.rpy[:2] ** 2)),
            "height": -12.0 * (rb.p[2] - STAND_H) ** 2,
            # Punish thrashing.  Real motors overheat; here it is what stops
            # the policy from buzzing at the PD's resonance.
            "effort": -0.012 * float(np.sum(a ** 2)),
            "smooth": -0.06 * float(np.sum((a - self.prev_a) ** 2)),
            # Sideways drift and yaw spin are not asked for, so penalise them.
            "drift": -0.8 * (v_body[1] ** 2 + rb.data.qvel[5] ** 2),
        }
        for k in self.reward_off:
            terms[k] = 0.0
        blown = not np.isfinite(rb.data.qpos).all()
        done = rb.fallen() or blown or self.t >= self.ep_len
        r = sum(terms.values())
        if not np.isfinite(r):
            r = -50.0
        if rb.fallen() or blown:
            # Price the steps that will not happen, or falling immediately is
            # the highest-return option available.
            r -= 1.5 * (self.ep_len - self.t) * 0.1
        return r, done

    def rollout(self, policy, v_cmd=None):
        o = self.reset(v_cmd)
        tot, vs, alive = 0.0, [], 0
        for _ in range(self.ep_len):
            a = policy(o)
            o, r, done = self.step(a)
            tot += r
            vs.append(float((self.rb.R.T @ self.rb.v)[0]))
            alive += 1
            if done:
                break
        return dict(ret=tot, steps=alive, survived=alive >= self.ep_len,
                    mean_vx=float(np.mean(vs[len(vs) // 4:])) if vs else 0.0,
                    distance=float(self.rb.p[0]))


def make_env(seed=0, randomize=False, rng=None, **kw):
    """Factory used by the trainer; `randomize` is the domain-randomisation switch."""
    if randomize:
        r = rng if rng is not None else np.random.default_rng(seed)
        kw = dict(kw)
        kw.setdefault("friction", float(r.uniform(0.45, 1.15)))
        kw.setdefault("payload", float(r.uniform(0.0, 2.5)))
        kw.setdefault("torque_scale", float(r.uniform(0.8, 1.2)))
    return WalkEnv(seed=seed, **kw)
