"""A zoo of arms, one observation format, and one policy that has to fit all.

The dream behind Open X-Embodiment is that robot data should add up the way
text data does: pool everything anyone has collected, on any robot, and a
policy trained on the pile is a better starting point for YOUR robot than
starting from nothing.  The obstacle is that robots are not sentences.  Two
arms may have different link lengths, different masses, different numbers of
joints -- so the same observation vector means different things and the same
action does different things.

This file implements the standard workarounds:

* **padding** -- every robot's joint vector is padded out to the widest robot
  in the zoo (three joints here), with zeros where a robot has no joint.  The
  policy always sees the same-shaped input.
* **action masking** -- the loss ignores the padded action slots, so a two-link
  robot never teaches the policy anything about a third joint it does not have.
* **an embodiment vector** -- the link lengths and the joint count are appended
  to the observation, so the policy can tell which robot it is driving.
  Without it, one policy has to be simultaneously right for arms that need
  opposite actions in the same situation.

A trap worth stating, because it cost real debugging time: **you cannot copy
one robot's servo gains onto another robot.**  A stiff shoulder gain applied to
a light wrist link makes the closed loop faster than the simulator's time step
can represent, and the arm explodes on the first step.  The rule is that every
decay rate in the loop, measured as an eigenvalue of ``M^-1 K`` or ``M^-1 B``,
must stay below about 2 / dt.  ``make_arm`` sizes both the gains and the joint
damping from each robot's own mass matrix so that this holds for all of them.
"""

import os
import sys

import numpy as np

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402

N_MAX = 3                  # widest robot in the zoo
OBS_DIM = 3 * N_MAX + 10 + 4      # joints, task vectors, embodiment vector
ACT_DIM = N_MAX
DAMP_C = 10.0              # joint damping as a multiple of link inertia
TAU_C = 400.0              # torque limit as a multiple of link inertia


def make_arm(lengths, masses, gear=1.0, omega_n=60.0):
    """Build a robot whose servo and damping are sized to its own inertia."""
    probe = A.PlanarArm(lengths=lengths, masses=masses,
                        damping=[0.0] * len(lengths),
                        tau_max=[1e9] * len(lengths), kp=None, kd=None)
    d = np.diag(probe.mass_matrix(np.full(len(lengths), 1.0)))
    return A.PlanarArm(lengths=lengths, masses=masses, damping=DAMP_C * d,
                       tau_max=TAU_C * d, kp=None, kd=None, gear=gear,
                       omega_n=omega_n)


# name -> (link lengths, link masses).  The reach is deliberately similar so
# that every robot can do the same task; what differs is how it must move.
ZOO = {
    "source-A-even":     ((0.20, 0.18), (0.60, 0.40)),
    "source-B-heavy":    ((0.20, 0.18), (1.50, 1.00)),
    "source-C-stubby":   ((0.16, 0.15), (0.35, 0.25)),
    "source-D-3link":    ((0.14, 0.13, 0.11), (0.50, 0.35, 0.25)),
    "target-long-upper": ((0.24, 0.14), (0.90, 0.30)),
}
SOURCES = ["source-A-even", "source-B-heavy", "source-C-stubby", "source-D-3link"]
TARGET = "target-long-upper"


def get_arm(name):
    return make_arm(*ZOO[name])


def emb_vector(arm):
    """Four numbers describing the robot: three link lengths and the DoF."""
    v = np.zeros(4, np.float32)
    v[:arm.n] = arm.l
    v[3] = arm.n / N_MAX
    return v


def pad_obs(env):
    """The shared observation layout, identical for every robot."""
    arm = env.arm
    q, qd = env.q, env.qd
    cs, sn, vel = np.zeros(N_MAX), np.zeros(N_MAX), np.zeros(N_MAX)
    cs[:arm.n] = np.cos(q)
    sn[:arm.n] = np.sin(q)
    vel[:arm.n] = qd / 10.0
    tip = arm.tip(q)
    return np.concatenate([cs, sn, vel, tip, env.puck, env.goal,
                           env.puck - tip, env.goal - env.puck,
                           emb_vector(arm)]).astype(np.float32)


def pad_act(a, n):
    out = np.zeros(ACT_DIM, np.float32)
    out[:n] = a
    return out


def act_mask(n):
    m = np.zeros(ACT_DIM, np.float32)
    m[:n] = 1.0
    return m


def make_env(name, rng):
    arm = get_arm(name)
    env = A.PushEnv(rng, arm=arm)
    # Keep the task inside every robot's annulus: a two-link arm with a long
    # upper link cannot fold tightly enough to reach its own centre, so the
    # puck must stay outside |l1 - l2|.
    env.arm = arm
    return env


def collect(name, n_demos, seed=0, noise=0.0, only_success=True):
    """Expert demonstrations on one robot, in the shared padded format."""
    rng = np.random.default_rng(seed)
    env = make_env(name, rng)
    n = env.arm.n
    O, Y, M, ok = [], [], [], []
    tries = 0
    while len(ok) < n_demos and tries < n_demos * 6:
        tries += 1
        env.reset()
        # Always circle the puck the same way.  Letting the demonstrator pick
        # a side at random makes the data multimodal, which is project 56's
        # problem, not this one -- mixing the two would hide the transfer
        # effect behind mode averaging.
        side = 1
        obs_ep, act_ep = [], []
        done = False
        while not done:
            a, _ = A.expert_action(env, side=side, noise=noise, rng=rng)
            obs_ep.append(pad_obs(env))
            act_ep.append(pad_act(a, n))
            _, _, done, _ = env.step(a)
        if only_success and not env.success:
            continue
        O.extend(obs_ep)
        Y.extend(act_ep)
        M.extend([act_mask(n)] * len(obs_ep))
        ok.append(True)
    return (np.array(O, np.float32), np.array(Y, np.float32),
            np.array(M, np.float32),
            dict(n=len(ok), tries=tries))


def evaluate(policy, name, n=40, seed=999):
    """Run a padded policy on one robot."""
    rng = np.random.default_rng(seed)
    env = make_env(name, rng)
    n_joints = env.arm.n
    ok, errs = 0, []
    for _ in range(n):
        env.reset()
        done = False
        while not done:
            a = policy(pad_obs(env))[:n_joints]
            _, _, done, _ = env.step(a)
        ok += env.success
        errs.append(env.err)
    return dict(success=ok / n, err=float(np.mean(errs)))


def expert_score(name, n=40, seed=123):
    rng = np.random.default_rng(seed)
    env = make_env(name, rng)
    ok = 0
    for _ in range(n):
        env.reset()
        side = 1 if rng.random() < 0.5 else -1
        done = False
        while not done:
            a, _ = A.expert_action(env, side=side)
            _, _, done, _ = env.step(a)
        ok += env.success
    return ok / n


if __name__ == "__main__":
    import time
    for name in ZOO:
        arm = get_arm(name)
        M = arm.mass_matrix(np.full(arm.n, 1.0))
        worst = float(np.max(np.linalg.eigvals(np.linalg.solve(M, np.diag(arm.b))))) * A.DT
        t0 = time.time()
        sc = expert_score(name, n=25)
        print(f"{name:19s} n={arm.n} reach={arm.reach:.2f} "
              f"damping rate*dt={worst:.2f} (must be < 2)  "
              f"expert={sc:.2f}  {(time.time() - t0) / 25 * 1000:.0f} ms/ep")
