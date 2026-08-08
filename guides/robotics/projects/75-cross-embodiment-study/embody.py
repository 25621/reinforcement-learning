"""Two robots, six ways they differ, and three ways to bridge the difference.

Project 61 asked whether pooling many robots' data helps a new robot.  This one
asks a narrower and more practical question: **you have a policy that works on
robot A and you have to run it on robot B tonight -- what exactly is going to
break, and which fix is worth doing first?**

The six axes
------------
Robot B is built from robot A by changing one thing at a time, so that each
difference can be switched on and off independently:

===========  ======================================================
``K``        kinematics -- different link lengths, same reach
``D``        dynamics -- heavier links, more joint damping
``G``        actuator gain -- B's motors are weaker for the same command
``L``        latency -- B's command takes one extra control period
``S``        action scale -- B calls "1.0" a different number of radians
``O``        observation convention -- B's second encoder is flipped and offset
===========  ======================================================

The last two are not physics.  They are wiring.  They are in the study because
in practice they are what actually happens when a policy moves between robots,
and because the experiment is only honest if the boring failure modes are
allowed to compete with the interesting ones.

The three bridges
-----------------
1. **nothing** -- send the source policy's numbers to B unchanged.
2. **calibration** -- one number: measure how far B moves for a command of 1.0
   and rescale.  Ten minutes of work, no learning.
3. **task-space retargeting** -- decode the source action into *where the tip
   should go*, using robot A's own geometry, then re-encode that motion into
   robot B's joints.  The policy keeps thinking in A's body; only the
   translation changes.

Why a retargeting layer at all, when the policy already outputs joint deltas
that robot B can execute?  Because "joint 1 moves 0.05 rad" means a different
tip motion on every robot.  The quantity the *task* cares about is the tip, and
the only way to preserve it is to convert out of joint space and back in.  The
policy is not modified, retrained or even reloaded -- what changes is the
adapter around it, which is why this is the cheap fix to try first.
"""

import os
import sys

import numpy as np

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402

DAMP_C = 10.0
TAU_C = 400.0


def make_arm(lengths, masses, gear=1.0, omega_n=60.0, damp_mult=1.0):
    """A robot whose servo gains and damping are sized to its own inertia.

    Copying one robot's PD gains onto another is the single fastest way to make
    a simulated arm explode -- project 61 measured why.  Sizing from the mass
    matrix keeps every closed-loop mode at the same frequency on every robot,
    so any difference we then measure is a difference in the *task*, not an
    artefact of one robot being tuned better than another.
    """
    probe = A.PlanarArm(lengths=lengths, masses=masses,
                        damping=[0.0] * len(lengths),
                        tau_max=[1e9] * len(lengths), kp=None, kd=None)
    d = np.diag(probe.mass_matrix(np.full(len(lengths), 1.0)))
    return A.PlanarArm(lengths=lengths, masses=masses,
                       damping=DAMP_C * d * damp_mult,
                       tau_max=TAU_C * d, kp=None, kd=None, gear=gear,
                       omega_n=omega_n)


SOURCE = dict(lengths=(0.20, 0.18), masses=(0.60, 0.40), gear=1.0,
              damp_mult=1.0, latency=0, dq_scale=1.0, flip=False)
TARGET = dict(lengths=(0.25, 0.13), masses=(1.40, 0.90), gear=0.75,
              damp_mult=2.2, latency=1, dq_scale=0.70, flip=True)
# Robot B's link lengths are chosen so that the SCRIPTED controller still
# scores near 1.0 on it.  That matters: if B were simply a harder robot, every
# transfer number would be part policy failure and part impossible task, and
# the study could not separate them.  Every experiment below reports B's
# expert ceiling next to the policy's score for exactly this reason.

AXES = {
    "K kinematics": ["lengths"],
    "D dynamics": ["masses", "damp_mult"],
    "G actuator gain": ["gear"],
    "L latency": ["latency"],
    "S action scale": ["dq_scale"],
    "O obs convention": ["flip"],
}


def spec(active=()):
    """Robot A with the named axes switched over to robot B's value."""
    s = dict(SOURCE)
    for ax in active:
        for k in AXES[ax]:
            s[k] = TARGET[k]
    return s


# ---------------------------------------------------------------------------
# the environment
# ---------------------------------------------------------------------------
class RobotEnv(A.PushEnv):
    """PushEnv on a named robot, with its own action scale and encoder quirks.

    ``dq_scale`` and ``flip`` are applied here rather than inside the policy so
    that they are properties of the ROBOT.  That is what makes them honest:
    the policy cannot know about them, exactly as a policy shipped to a new arm
    cannot know that the new arm's second encoder counts the other way.
    """

    def __init__(self, rng, sp, **kw):
        self.sp = sp
        arm = make_arm(sp["lengths"], sp["masses"], gear=sp["gear"],
                       damp_mult=sp["damp_mult"])
        super().__init__(rng, arm=arm, params={"latency": sp["latency"]}, **kw)

    def obs(self):
        o = super().obs()
        if self.sp["flip"]:
            # cos(q2), sin(q2), qd2 as a robot with a reversed, offset encoder
            # would report them.  Everything else is unchanged.
            q2 = -self.q[1] + 0.35
            o = o.copy()
            o[1] = np.cos(q2)
            o[3] = np.sin(q2)
            o[5] = -self.qd[1] / 10.0
        return o

    def step(self, action):
        a = np.asarray(action, float) * self.sp["dq_scale"]
        return super().step(a)


def make_env(sp, rng, **kw):
    return RobotEnv(rng, sp, **kw)


# ---------------------------------------------------------------------------
# the retargeting layer
# ---------------------------------------------------------------------------
def ik2(arm, target, elbow=1):
    """Closed-form inverse kinematics for a 2-link planar arm.

    Needed because retargeting has to ask "what joint angles would robot A be
    in, if its tip were where robot B's tip is?".  Two links and a point target
    have a two-line answer, so there is no reason to run an iterative solver.
    """
    l1, l2 = arm.l
    x, y = float(target[0]), float(target[1])
    r2 = x * x + y * y
    r = np.sqrt(r2)
    r = float(np.clip(r, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-4))
    x, y = x * r / max(np.sqrt(r2), 1e-9), y * r / max(np.sqrt(r2), 1e-9)
    c2 = (r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = float(np.clip(c2, -1.0, 1.0))
    q2 = elbow * np.arccos(c2)
    q1 = np.arctan2(y, x) - np.arctan2(l2 * np.sin(q2), l1 + l2 * np.cos(q2))
    return np.array([q1, q2])


def dls(J, v, lam=0.06):
    """Damped least squares: the joint motion closest to producing ``v``."""
    return J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(2), v)


class Retargeter:
    """Wraps a source-robot policy so it can drive a different robot.

    Each decision:

    1. work out where robot A would have to be to have B's tip and B's task;
    2. build the observation A would report in that pose;
    3. ask the policy for an action, which is a joint delta *for A*;
    4. turn that into a tip displacement using A's Jacobian;
    5. turn the tip displacement into a joint delta for B.

    Step 2 is the part that looks redundant -- B already produces an
    observation, so why synthesise another?  Because B's observation encodes
    B's joint angles, and the policy's first layer learned what A's joint
    angles mean.  Handing it B's numbers is like reading a French sentence with
    an English dictionary: the words are there, the meanings are not.
    """

    def __init__(self, policy, src_arm, tgt_env, scale_only=False,
                 src_spec=None):
        self.policy = policy
        self.src = src_arm
        self.env = tgt_env
        self.scale_only = scale_only
        # The synthesised observation must be in the SOURCE robot's convention,
        # quirks included.  Leaving this out is not a small bug: a policy
        # trained on a robot whose second encoder reads backwards has learned
        # that convention, and handing it a tidy textbook observation is just
        # as wrong as handing it the target robot's raw one.  Measured: the
        # reverse-direction transfer scored 0.00 until this was added.
        self.src_spec = src_spec or SOURCE
        # Two calibration numbers, because the two adapters need different
        # things.  ``gain_task`` answers "how much further does a command of
        # 1.0 move robot A's tool than robot B's?" and rescales the policy's
        # raw output.  ``gain_joint`` answers "what must I command robot B to
        # get the joint motion I asked for?" and is what the retargeter needs,
        # because by then the geometry has already been converted.
        self.gain_task = 1.0
        self.gain_joint = 1.0

    def calibrate(self, n=200, seed=3, warmup=3):
        """Measure both gains by wiggling the target robot 200 times.

        ``warmup`` steps of the same action are thrown away before the
        measurement.  That is not tidiness: if the robot delays its commands by
        one control period -- which is one of the six axes -- then the *first*
        step after a reset executes a queued zero and the arm does not move at
        all.  Dividing by that zero gave a calibration gain of 7 x 10^7 the
        first time this was run, and every downstream number with it.  A
        black-box calibration must let the black box settle.
        """
        rng = np.random.default_rng(seed)
        env = self.env
        ra, rb, cmd, got = [], [], [], []
        for _ in range(n):
            env.reset()
            a = rng.uniform(-1, 1, 2)
            for _ in range(warmup):
                env.step(a)
            q = env.q.copy()
            tip0 = env.arm.tip(q)
            ra.append(np.linalg.norm(self.src.jacobian(q) @ (A.DQ_MAX * a)))
            env.step(a)
            rb.append(np.linalg.norm(env.arm.tip(env.q) - tip0))
            cmd.append(A.DQ_MAX * np.linalg.norm(a))
            got.append(np.linalg.norm(env.q - q))
        self.gain_task = float(np.mean(ra) / max(np.mean(rb), 1e-9))
        self.gain_joint = float(np.mean(cmd) / max(np.mean(got), 1e-9))
        return self.gain_task, self.gain_joint

    def __call__(self, _obs):
        env = self.env
        if self.scale_only:
            return np.clip(self.policy(_obs) * self.gain_task, -1, 1)
        tip = env.arm.tip(env.q)
        # 1-2: put robot A's virtual body where B's tip is
        elbow = 1 if env.q[1] >= 0 else -1
        qa = ik2(self.src, tip, elbow=elbow)
        Ja = self.src.jacobian(qa)
        v_tip = env.arm.jacobian(env.q) @ env.qd
        qda = dls(Ja, v_tip)
        base = [np.cos(qa), np.sin(qa), qda / 10.0, self.src.tip(qa),
                env.puck, env.goal]
        base += [env.puck - self.src.tip(qa), env.goal - env.puck]
        o = np.concatenate(base)
        if self.src_spec["flip"]:
            q2 = -qa[1] + 0.35
            o[1], o[3], o[5] = np.cos(q2), np.sin(q2), -qda[1] / 10.0
        # 3-5: action for A -> tip motion -> action for B
        a_src = np.clip(self.policy(o), -1, 1)
        dx = Ja @ (A.DQ_MAX * a_src)
        dq_b = dls(env.arm.jacobian(env.q), dx)
        return np.clip(dq_b / A.DQ_MAX * self.gain_joint, -1, 1)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def evaluate(make_policy, sp, n=80, seed=1000):
    """``make_policy(env) -> policy(obs)``, so adapters can see the robot."""
    rng = np.random.default_rng(seed)
    env = make_env(sp, rng)
    ok, errs = 0, []
    for _ in range(n):
        obs = env.reset()
        policy = make_policy(env)
        done = False
        while not done:
            obs, _, done, info = env.step(policy(obs))
        ok += info["success"]
        errs.append(info["err"])
    return {"success": ok / n, "err": float(np.mean(errs))}


def collect(sp, n_demos, seed=0, noise=0.0):
    """Expert demonstrations on a given robot, in that robot's own convention."""
    rng = np.random.default_rng(seed)
    env = make_env(sp, rng)
    O, Y, oks = [], [], []
    tries = 0
    while len(oks) < n_demos and tries < n_demos * 6:
        tries += 1
        env.reset()
        side = 1 if rng.random() < 0.5 else -1
        o_, a_ = [], []
        done = False
        obs = env.obs()
        while not done:
            a, _ = A.expert_action(env, side=side, noise=noise, rng=env.rng)
            # the expert commands the ROBOT; the recorded action must be what
            # the policy would have to output, i.e. before the robot's own
            # action scale is applied
            o_.append(obs.copy())
            a_.append(np.clip(a / env.sp["dq_scale"], -1, 1))
            obs, _, done, info = env.step(np.clip(a / env.sp["dq_scale"], -1, 1))
        if not info["success"]:
            continue
        O.append(np.array(o_))
        Y.append(np.array(a_))
        oks.append(True)
    return (np.concatenate(O).astype(np.float32),
            np.concatenate(Y).astype(np.float32), len(oks))


def expert_ceiling(sp, n=60, seed=777):
    """What the scripted controller scores on this robot -- the ceiling."""
    rng = np.random.default_rng(seed)
    env = make_env(sp, rng)
    ok = 0
    for _ in range(n):
        env.reset()
        side = 1 if rng.random() < 0.5 else -1
        done = False
        while not done:
            a, _ = A.expert_action(env, side=side)
            _, _, done, info = env.step(np.clip(a / env.sp["dq_scale"], -1, 1))
        ok += info["success"]
    return ok / n
