"""Augmented Random Search, spread across the CPU cores.

ARS (Mania, Guy & Recht 2018) is the simplest thing that works on locomotion:
no neural network, no value function, no gradients.  Perturb the policy's
weight matrix in a random direction, run an episode with +delta and one with
-delta, and step towards whichever did better.  "Augmented" is the three
additions that turned a toy into something competitive:

  1. normalise the observations by a running mean and standard deviation;
  2. divide the update by the standard deviation of the returns actually
     collected, so the step size does not depend on the reward scale;
  3. keep only the best few directions and throw the rest away.

It matters here for one specific reason: ARS needs no back-propagation, so
every rollout is independent and the whole thing parallelises perfectly across
processes.  That is the same property Isaac Lab exploits with thousands of
GPU-simulated robots -- this just runs 12 CPU ones instead.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_ENVKW, _ENV = None, None


class Normalizer:
    """Running mean/std of the observations.

    The observation mixes body height in metres (moves by centimetres) with
    joint velocities in rad/s (moves by tens).  A linear policy applied to raw
    numbers like that is dominated by whichever entry happens to have the
    biggest units.  Normalising puts every input on the same footing.
    """

    def __init__(self, n):
        self.n = np.zeros(n)
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n)

    def push(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def std(self):
        # A floor, not just an epsilon.  Several observations barely move at
        # all; dividing by their true standard deviation multiplies their
        # noise by hundreds and saturates the actuators within a few steps.
        # Clamp m2 at zero before the square root.  The parallel merge below
        # combines batch statistics with Chan's formula, and for an entry that
        # is CONSTANT within a batch (the velocity command is) the algebra
        # lands on a tiny negative number instead of exactly zero.  sqrt of
        # that is NaN, the NaN divides into every observation, and the whole
        # training run silently returns nan.
        v = np.where(self.n > 1,
                     np.maximum(self.m2, 0.0) / np.maximum(self.n - 1, 1), 1.0)
        return np.maximum(np.sqrt(v), 0.05)

    def __call__(self, x):
        return (x - self.mean) / self.std()

    def state(self):
        return (self.n.copy(), self.mean.copy(), self.m2.copy())

    def load(self, s):
        self.n, self.mean, self.m2 = [np.array(a) for a in s]


def _init(envkw):
    global _ENVKW, _ENV
    _ENVKW = envkw
    _ENV = None


def _worker(job):
    """One rollout in a worker process.  The env is built once and reused."""
    global _ENV
    from env import make_env
    W, mean, std, seed, v_cmd, randomize = job
    if _ENV is None or randomize:
        rng = np.random.default_rng(seed)
        _ENV = make_env(seed=seed, randomize=randomize, rng=rng, **_ENVKW)
    env = _ENV
    o = env.reset(v_cmd)
    tot, obs_sum, obs_sq, k = 0.0, np.zeros_like(o), np.zeros_like(o), 0
    for _ in range(env.ep_len):
        obs_sum += o
        obs_sq += o * o
        k += 1
        a = W @ ((o - mean) / std)
        o, r, done = env.step(a)
        tot += r
        if done:
            break
    return tot, obs_sum, obs_sq, k


def train(envkw, n_obs, iters=60, ndir=12, top=6, alpha=0.02, sigma=0.03,
          seed=0, v_cmds=((0.6, 0.0),), randomize=False, workers=None,
          log=None):
    rng = np.random.default_rng(seed)
    W = np.zeros((12, n_obs))
    norm = Normalizer(n_obs)
    curve = []
    workers = workers or min(os.cpu_count() or 4, 12)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                             initargs=(envkw,)) as pool:
        for it in range(iters):
            deltas = rng.normal(size=(ndir, 12, n_obs))
            mean, std = norm.mean.copy(), norm.std()
            vc = v_cmds[int(rng.integers(len(v_cmds)))]
            jobs = []
            for k in range(ndir):
                for sgn in (1.0, -1.0):
                    jobs.append((W + sgn * sigma * deltas[k], mean, std,
                                 int(rng.integers(1 << 30)), vc, randomize))
            out = list(pool.map(_worker, jobs, chunksize=1))
            rp = np.nan_to_num(np.array([out[2 * k][0] for k in range(ndir)]),
                               nan=-1e4)
            rm = np.nan_to_num(np.array([out[2 * k + 1][0]
                                         for k in range(ndir)]), nan=-1e4)
            for tot, s, sq, kk in out:                 # fold in the statistics
                if kk:
                    m = s / kk
                    norm.n += kk
                    d = m - norm.mean
                    norm.mean += d * kk / norm.n
                    norm.m2 += (np.maximum(sq - kk * m * m, 0.0)
                                + d * d * kk * (norm.n - kk) / norm.n)
                    norm.m2 = np.maximum(norm.m2, 0.0)
            order = np.argsort(-np.maximum(rp, rm))[:top]
            sr = np.concatenate([rp[order], rm[order]]).std()
            step = np.zeros_like(W)
            for k in order:
                step += (rp[k] - rm[k]) * deltas[k]
            W = W + alpha / (top * max(sr, 1e-6)) * step
            curve.append(float(0.5 * (rp.mean() + rm.mean())))
            if log and (it + 1) % 10 == 0:
                log(f"      iter {it + 1}/{iters}  return {curve[-1]:.1f}")
    return W, norm, curve


def policy_fn(W, norm):
    mean, std = norm.mean.copy(), norm.std()

    def f(o):
        return W @ ((o - mean) / std)
    return f
