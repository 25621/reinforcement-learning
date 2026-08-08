"""A 50-task evaluation suite, the runner, and the statistics it needs.

A "nightly eval" is not a script that prints a number.  It is three things:

  * a **task suite** -- a fixed, versioned list of situations, each of which is
    a specific thing you claim the robot can do;
  * a **seeding scheme** -- how the situation is varied within a task, so that
    a task measures a capability rather than one lucky arrangement;
  * an **error bar** -- because a pass rate computed from twenty episodes is a
    random variable, and a dashboard without an interval invites the team to
    chase noise every morning.

The tasks vary along five axes at once: where the puck and goal are, what the
robot's physics is, whether there is an obstacle, how noisy the sensing is, and
how much command delay there is.  A suite that varies only object placement
measures only object placement.
"""

import os
import sys

import numpy as np
import torch

# One thread per process: every evaluation runs in a pool, and a worker that
# starts six BLAS threads of its own just fights the other workers.
torch.set_num_threads(1)

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets as N           # noqa: E402


# ---------------------------------------------------------------------------
# the suite
# ---------------------------------------------------------------------------
def build_suite(n_tasks=50, seed=7):
    """Fifty tasks, generated once from a fixed seed and then FROZEN.

    Generating the suite from a seed and freezing it is deliberate.  A suite
    you hand-write is biased towards what you already thought of; a suite you
    regenerate every night is not a suite, because yesterday's number and
    today's number would be measuring different things.
    """
    rng = np.random.default_rng(seed)
    families = ["nominal", "heavy", "weak motor", "laggy", "noisy sensing",
                "obstacle", "sticky table", "combined"]
    tasks = []
    for i in range(n_tasks):
        fam = families[i % len(families)]
        p = dict(A.DEFAULT_PARAMS)
        obstacle = None
        if fam == "heavy":
            p["mass_scale"] = float(rng.uniform(1.4, 2.0))
        elif fam == "weak motor":
            p["gear"] = float(rng.uniform(0.70, 0.88))
        elif fam == "laggy":
            p["latency"] = int(rng.integers(1, 3))
        elif fam == "noisy sensing":
            p["obs_noise"] = float(rng.uniform(0.004, 0.010))
        elif fam == "sticky table":
            p["slip"] = float(rng.uniform(0.55, 0.85))
        elif fam == "obstacle":
            obstacle = (float(rng.uniform(0.10, 0.24)),
                        float(rng.uniform(-0.10, 0.12)), 0.035)
        elif fam == "combined":
            p["mass_scale"] = float(rng.uniform(1.2, 1.6))
            p["gear"] = float(rng.uniform(0.82, 0.95))
            p["latency"] = 1
            p["obs_noise"] = 0.004
        tasks.append(dict(id=i, family=fam, params=p, obstacle=obstacle,
                          name="%02d-%s" % (i, fam.replace(" ", "-"))))
    return tasks


# ---------------------------------------------------------------------------
# running one task
# ---------------------------------------------------------------------------
def run_task(policy, task, seeds):
    """Run one task under a list of episode seeds; return a 0/1 per seed.

    The seed drives the *situation* (where the puck and goal start), not the
    physics: the physics is the task's own definition.  Keeping those two
    separate is what lets a failure be attributed to "this robot" rather than
    "this Tuesday".
    """
    out = np.zeros(len(seeds), dtype=int)
    for j, s in enumerate(seeds):
        env = A.PushEnv(np.random.default_rng(int(s)), params=task["params"],
                        obstacle=task["obstacle"])
        obs = env.reset()
        done = False
        while not done:
            obs, _, done, info = env.step(policy(obs))
        out[j] = int(info["success"])
    return out


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    """A confidence interval for a pass rate, the way it should be done.

    The interval everybody writes first, p +- z*sqrt(p(1-p)/n), is wrong at the
    edges: 20 out of 20 gives 1.00 +- 0.00, claiming certainty from twenty
    samples.  Edwin Wilson's 1927 interval fixes it by asking the inverted
    question -- "which true rates would plausibly produce this count?" -- and
    stays inside [0, 1] with sensible width when k is 0 or n.
    """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def two_proportion_z(k1, n1, k2, n2):
    """Is this drop real?  The standard test for two pass rates.

    Returns |z|.  Above 1.96 is the usual "significant at 5 %" line -- i.e.
    a difference this big would turn up by chance less than one night in
    twenty if nothing had actually changed.
    """
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se < 1e-12:
        return 0.0
    return float(abs(k1 / n1 - k2 / n2) / se)


# ---------------------------------------------------------------------------
# the systems under test
# ---------------------------------------------------------------------------
class Policy:
    """A trained policy as a picklable object rather than a closure.

    ``nets.make_policy`` returns a nested function, and pickle refers to
    functions by name -- a nested one has no importable name, so it cannot be
    sent to a worker process.  ``scale`` is what experiment 2's injected
    regression multiplies the action by.
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

    def rescaled(self, factor):
        """The regression: a commit that quietly scales the action down.

        This is what a real regression looks like -- not a crash, not an
        exception, just slightly smaller numbers coming out of a function
        somebody refactored.  No unit test catches it.
        """
        out = Policy.__new__(Policy)
        out.__dict__.update(self.__dict__)
        out._net = None
        out.scale = self.scale * factor
        return out


def train(n_demos, seed=0, randomize=None, epochs=200):
    N.seed_all(seed)
    if randomize is None:
        obs, act, _ = A.collect_demos(n_demos, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        env = A.PushEnv(rng, randomize=randomize)
        O, Ac = [], []
        got = tries = 0
        while got < n_demos and tries < n_demos * 5:
            tries += 1
            r = A.rollout(env, None, side=1 if rng.random() < .5 else -1,
                          record=True)
            if not r["success"]:
                continue
            O.append(r["obs"]); Ac.append(r["act"]); got += 1
        obs = np.concatenate(O).astype(np.float32)
        act = np.concatenate(Ac).astype(np.float32)
    net, norm, _ = N.train_bc(obs, act, epochs=epochs, seed=seed)
    return Policy(net, norm)


def run_task_expert(task, seeds):
    """The scripted demonstrator, used as the suite's ceiling.

    It needs the environment and not just the observation, so it gets its own
    runner.  A ceiling matters: a task at 0.4 for every system is either a hard
    task or a broken one, and only the expert's score tells you which.
    """
    out = np.zeros(len(seeds), dtype=int)
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        env = A.PushEnv(rng, params=task["params"], obstacle=task["obstacle"])
        env.reset()
        side = 1 if rng.random() < 0.5 else -1
        done = False
        while not done:
            a, _ = A.expert_action(env, side=side, rng=rng)
            _, _, done, info = env.step(a)
        out[j] = int(info["success"])
    return out
