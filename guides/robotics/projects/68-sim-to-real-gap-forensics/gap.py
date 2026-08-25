"""One sim, one "real" robot, three planted defects, and the probes to find them.

The robot and the task are project 54's: a 2-link planar arm pushing a puck
onto a goal.  What is new is that there are now two of them.

  SIM   -- the nominal robot, the one the policy is trained on
  REAL  -- the same task on a machine that differs in exactly three ways

The three ways are chosen to be one from each of the categories a real transfer
failure falls into, because the whole point of the exercise is that "the policy
does not work on the robot" has to be turned into "*which layer* does not
work":

  D  dynamics    -- heavier links, more gearbox friction (an unmodelled payload)
  A  actuation   -- a weaker motor and one decision of command delay
  P  perception  -- the puck is measured with a constant offset and some noise

We know the answer.  That is the point: it is the only way to check whether a
forensic procedure would have found it.
"""

import os
import sys

import numpy as np
import torch

# One thread per process.  Every experiment here runs a pool of workers, and a
# worker that starts six BLAS threads of its own just fights the other workers.
torch.set_num_threads(1)

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets as N           # noqa: E402

# the three defect groups, each a dict of the robot parameters it changes
DEFECTS = {
    "D dynamics": dict(mass_scale=1.70, damp_scale=2.20),
    "A actuation": dict(gear=0.75, latency=1),
    "P perception": dict(),          # perception lives outside the physics
}
PERCEPTION_BIAS = np.array([0.015, -0.010])    # metres, a miscalibrated camera
PERCEPTION_NOISE = 0.006                       # metres, one sigma

# where each quantity sits in the 16-number observation (feat="rel")
I_TIP = slice(6, 8)
I_PUCK = slice(8, 10)
I_GOAL = slice(10, 12)
I_PUCK_TIP = slice(12, 14)
I_GOAL_PUCK = slice(14, 16)


def params_for(subset):
    """Robot parameters for any subset of the three defects."""
    p = dict(A.DEFAULT_PARAMS)
    for name in subset:
        p.update(DEFECTS[name])
    return p


def corrupt_obs(obs, rng, on, bias=None):
    """Apply the perception defect to an observation, consistently.

    The puck appears where the camera says it is, so every derived quantity
    has to move with it -- otherwise the observation is internally
    contradictory in a way no real sensor could produce, and the policy could
    detect the fault from the inconsistency alone.

    ``bias`` overrides the default offset; passing zeros is what "we calibrated
    the camera" means, and leaves only the random noise behind.
    """
    if not on:
        return obs
    o = obs.copy()
    b = PERCEPTION_BIAS if bias is None else bias
    err = b + rng.normal(0, PERCEPTION_NOISE, 2)
    o[I_PUCK] = obs[I_PUCK] + err
    o[I_PUCK_TIP] = o[I_PUCK] - o[I_TIP]
    o[I_GOAL_PUCK] = o[I_GOAL] - o[I_PUCK]
    return o


def evaluate(policy, subset=(), n=60, seed=1000, record_actions=False,
             calibrated=False):
    """Success rate of a policy on the robot defined by ``subset``.

    Success is read out of the info dict rather than off the environment
    afterwards, because the final ``step`` resets the episode -- a trap that
    cost project 54 a whole misleading table.
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng, params=params_for(subset))
    percep = "P perception" in subset
    prng = np.random.default_rng(seed + 7)
    ok, errs, acts = 0, [], []
    for _ in range(n):
        obs = env.reset()
        done, ep = False, []
        while not done:
            a = policy(corrupt_obs(obs, prng, percep,
                                   np.zeros(2) if calibrated else None))
            ep.append(np.asarray(a, float).copy())
            obs, _, done, info = env.step(a)
        ok += info["success"]
        errs.append(info["err"])
        if record_actions:
            acts.append(np.array(ep))
    out = dict(success=ok / n, err=float(np.mean(errs)))
    if record_actions:
        out["actions"] = acts
    return out


def open_loop(actions, subset=(), seed=1000):
    """Replay a fixed action sequence with no feedback at all.

    This is the probe that needs no policy and no perception: the same numbers
    go to both robots, so any difference in where the tip ends up is the
    machine, not the software.  It cleanly separates "the robot moves
    differently" from "the policy sees differently", which closed-loop success
    rates can never do because feedback hides one inside the other.
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng, params=params_for(subset))
    tips = []
    for k, seq in enumerate(actions):
        env.reset()
        traj = []
        for a in seq:
            env.step(a)
            traj.append(env.arm.tip(env.q).copy())
        tips.append(np.array(traj))
    return tips


class Policy:
    """A trained policy as a picklable object rather than a closure.

    ``nets.make_policy`` returns a nested function, and a nested function
    cannot be sent to a worker process -- ``pickle`` refers to functions by
    name and this one has no importable name.  Every evaluation here runs in a
    process pool, so the policy has to be a plain object carrying plain arrays.
    """

    def __init__(self, net, norm, scale=1.0):
        self.sd = {k: v.detach().numpy() for k, v in net.state_dict().items()}
        self.mu, self.sigma = norm.mu, norm.sd
        self.dims = (net.net[0].in_features, net.net[-1].out_features)
        self.scale = scale
        self._net = None

    def __call__(self, obs):
        if self._net is None:
            self._net = N.MLP(*self.dims)
            self._net.load_state_dict(
                {k: torch.tensor(v) for k, v in self.sd.items()})
            self._net.eval()
        x = (np.asarray(obs, np.float32) - self.mu) / self.sigma
        with torch.no_grad():
            a = self._net(torch.tensor(x[None], dtype=torch.float32))[0].numpy()
        return a * self.scale

    def __getstate__(self):
        s = dict(self.__dict__)
        s["_net"] = None
        return s


def train_policy(n_demos=300, seed=0, params=None, randomize=None, epochs=200,
                 perception_dr=0.0):
    """Behaviour cloning on expert demonstrations, exactly as project 54."""
    N.seed_all(seed)
    if perception_dr:
        obs, act, meta = _collect_perception_dr(n_demos, seed, params,
                                                perception_dr)
    elif randomize is None:
        obs, act, meta = A.collect_demos(n_demos, seed=seed, params=params)
    else:
        obs, act, meta = _collect_randomized(n_demos, seed, params, randomize)
    net, norm, _ = N.train_bc(obs, act, epochs=epochs, seed=seed)
    return Policy(net, norm), meta


def _collect_randomized(n_demos, seed, params, randomize):
    """Demonstrations gathered on a different robot every episode (project 59).

    The demonstrator is a feedback controller working from kinematics, so it
    still succeeds on each robot; what changes is the (observation, action)
    pairs it produces, and that variety is the whole product.
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng, params=params, randomize=randomize)
    O, Ac, oks = [], [], []
    tries = 0
    while len(oks) < n_demos and tries < n_demos * 5:
        tries += 1
        side = 1 if rng.random() < 0.5 else -1
        r = A.rollout(env, None, side=side, record=True)
        if not r["success"]:
            continue
        O.append(r["obs"]); Ac.append(r["act"]); oks.append(1)
    return (np.concatenate(O).astype(np.float32),
            np.concatenate(Ac).astype(np.float32),
            dict(n=len(oks), tries=tries, yield_=len(oks) / tries))


def _collect_perception_dr(n_demos, seed, params, spread):
    """Demonstrations whose observations carry a random camera offset.

    A different constant offset per episode, drawn from a box of half-width
    ``spread``.  The expert's ACTIONS are still computed from the true state --
    it is the robot's own eyes that are wrong, not the teacher's -- so the
    policy is being asked to output the correct action while looking at a
    shifted world.  With the offset unobservable and different every episode,
    the best it can learn is the behaviour that is least wrong on average.
    That is the honest ceiling of randomisation here, and section 6 measures
    how far it falls short of simply calibrating the camera.
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng, params=params)
    O, Ac, oks = [], [], []
    tries = 0
    while len(oks) < n_demos and tries < n_demos * 5:
        tries += 1
        side = 1 if rng.random() < 0.5 else -1
        r = A.rollout(env, None, side=side, record=True)
        if not r["success"]:
            continue
        b = rng.uniform(-spread, spread, 2)
        obs = r["obs"].copy()
        obs[:, I_PUCK] += b
        obs[:, I_PUCK_TIP] += b
        obs[:, I_GOAL_PUCK] -= b
        O.append(obs); Ac.append(r["act"]); oks.append(1)
    return (np.concatenate(O).astype(np.float32),
            np.concatenate(Ac).astype(np.float32),
            dict(n=len(oks), tries=tries))
