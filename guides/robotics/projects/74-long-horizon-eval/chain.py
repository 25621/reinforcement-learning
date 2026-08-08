"""A fifty-stage task built out of project 54's one-stage push.

The arithmetic everybody quotes is ``p ** N``: if one step works 95 % of the
time, fifty steps in a row work 0.95 ** 50 = 7.7 % of the time.  It is the
right *shape* of answer and it is almost never the right number, and this file
exists so we can measure how wrong it is and why.

The task
--------
One puck.  Fifty successive goal discs, each 9-13 cm from wherever the puck
currently is.  Reach a goal and the next one appears.  Nothing is reset between
stages -- the arm is wherever the last push left it, the puck is wherever it
came to rest, and any small error is inherited.  That inheritance is the whole
subject.

Two ways of measuring "per-stage success"
-----------------------------------------
* **fresh** -- reset the world, sample a puck and a goal from the same
  distribution, run one stage.  This is what a unit test does, and what an
  evaluation harness (project 69) reports.
* **in-chain** -- the success rate of stage *k* inside a real chain, given the
  chain got that far.

They are not the same number, and the gap is what makes long-horizon forecasts
wrong.  ``p ** N`` is only valid if the stages are independent *and* identically
distributed; a chain violates both, in opposite directions.
"""

import os
import sys

import numpy as np

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402

N_STAGES = 50
STAGE_LEN = 45             # decisions a stage may take before it is a failure
HOP_LO, HOP_HI = 0.09, 0.13


def _reachable(arm, p):
    lo, hi = 0.40 * arm.reach, 0.66 * arm.reach
    r = float(np.linalg.norm(p))
    return lo * 0.92 < r < hi * 1.06


def next_goal(env, rng):
    """A goal one hop away from the puck, still inside the workspace."""
    for _ in range(60):
        ang = rng.uniform(0, 2 * np.pi)
        g = env.puck + rng.uniform(HOP_LO, HOP_HI) * np.array([np.cos(ang),
                                                               np.sin(ang)])
        if _reachable(env.arm, g):
            return g
    return env.puck.copy()


def run_stage(env, policy, budget=STAGE_LEN):
    """Push until the puck is on the current goal, or the budget runs out."""
    env.t = 0
    for k in range(budget):
        a = policy(env.obs())
        env.q_cmd = env.q + A.DQ_MAX * np.clip(np.asarray(a, float), -1, 1)
        for _ in range(A.SUBSTEPS):
            tip_prev = env.arm.tip(env.q)
            tau = env.arm.servo_torque(env.q, env.qd, env.q_cmd)
            env.q, env.qd = env.arm.step(env.q, env.qd, tau)
            env._push_puck(tip_prev, env.arm.tip(env.q))
        if env.err < A.GOAL_TOL:
            return True, k + 1
    return False, budget


def run_chain(make_policy, seed, n_stages=N_STAGES, retries=0,
              budget=STAGE_LEN):
    """One long-horizon episode.

    ``make_policy`` is a factory, not a policy: the scripted controller needs
    to read the environment, and building it per episode is the only way to
    let a controller and a network be swapped for one another here.

    ``retries`` is the recovery behaviour: when a stage fails, try it again
    from wherever the failure left the world.  Zero retries is the strict
    reading of "the task failed"; one retry is the cheapest recovery a real
    system ships.
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    env.reset()
    policy = make_policy(env)
    per_stage, attempts_used = [], 0
    for _ in range(n_stages):
        env.goal = next_goal(env, rng)
        ok = False
        for _ in range(retries + 1):
            attempts_used += 1
            ok, _ = run_stage(env, policy, budget)
            if ok:
                break
        per_stage.append(ok)
        if not ok:
            break
    reached = int(np.sum(per_stage))
    return {"reached": reached,
            "complete": reached == n_stages,
            "per_stage": per_stage,
            "attempts": attempts_used}


def fresh_stage_rate(make_policy, n=400, seed=5000, budget=STAGE_LEN):
    """Per-stage success measured the way a unit test measures it."""
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    policy = make_policy(env)
    ok = 0
    for _ in range(n):
        env.reset()
        env.goal = next_goal(env, rng)
        good, _ = run_stage(env, policy, budget)
        ok += good
    return ok / n


# ---------------------------------------------------------------------------
# the two systems under test
# ---------------------------------------------------------------------------
# Both are written as small CLASSES rather than closures for one reason: the
# experiments run chains in a process pool, worker processes are started with
# ``spawn``, and spawn has to pickle whatever you send it.  A closure cannot be
# pickled; an object holding plain numbers and arrays can.  (A fork-started
# pool would avoid the pickling and deadlock instead, because torch keeps
# worker threads alive -- project 68 met that one.)
class NoisyExpert:
    """The scripted demonstrator with Gaussian noise on its action.

    This is a *knob for per-stage reliability* that is not tangled up with how
    well a network happened to be trained.  Turning the knob is how the
    experiments sweep ``p`` while holding everything else fixed.
    """

    def __init__(self, sigma, seed=0):
        self.sigma = float(sigma)
        self.seed = int(seed)

    def __call__(self, env):
        rng = np.random.default_rng(self.seed)
        side = 1 if rng.random() < 0.5 else -1
        sigma = self.sigma

        def p(_obs):
            a, _phase = A.expert_action(env, side=side)
            return a + rng.normal(0, sigma, A.PushEnv.act_dim)
        return p


class ClonedPolicy:
    """A behaviour-cloned network, shipped as plain arrays."""

    def __init__(self, net, norm):
        self.w = [v.detach().numpy().copy() for v in net.state_dict().values()]
        self.mu, self.sd = norm.mu.copy(), norm.sd.copy()
        self._built = None

    def _forward(self, x):
        x = (x - self.mu) / self.sd
        import torch
        t = torch.tensor(x[None], dtype=torch.float32)
        for i in range(0, len(self.w) - 2, 2):
            t = torch.nn.functional.gelu(
                t @ torch.tensor(self.w[i]).T + torch.tensor(self.w[i + 1]))
        t = t @ torch.tensor(self.w[-2]).T + torch.tensor(self.w[-1])
        return t[0].numpy()

    def __call__(self, env):
        return lambda obs: self._forward(np.asarray(obs, np.float32))


# ---------------------------------------------------------------------------
# running many chains at once
# ---------------------------------------------------------------------------
def _worker(args):
    import torch
    torch.set_num_threads(1)
    mk, seed, retries, budget, n_stages = args
    return run_chain(mk, seed, n_stages=n_stages, retries=retries,
                     budget=budget)


def run_many(make_policy, n, seed0=20000, retries=0, budget=STAGE_LEN,
             n_stages=N_STAGES, pool=None):
    jobs = [(make_policy, seed0 + 17 * i, retries, budget, n_stages)
            for i in range(n)]
    if pool is None:
        return [_worker(j) for j in jobs]
    return list(pool.map(_worker, jobs, chunksize=1))


def _fresh_worker(args):
    import torch
    torch.set_num_threads(1)
    mk, seed, n, budget = args
    return fresh_stage_rate(mk, n=n, seed=seed, budget=budget)
