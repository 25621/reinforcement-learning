"""Augmented Random Search: reinforcement learning without a single gradient.

ARS (Mania, Guy and Recht, 2018) is [evolution strategies](/shared/glossary/#evolution-strategies)
stripped to its bones.  Each iteration:

    1. draw N random perturbations of the policy's weights
    2. run one episode with (weights + delta) and one with (weights - delta)
    3. step the weights toward the deltas whose PLUS run beat their MINUS run

That is the whole algorithm.  No value function, no replay buffer, no
backpropagation through the simulator.

Why it is the right tool here rather than PPO or SAC.  This task has 72 policy
parameters and an episode that runs in 15 milliseconds, and its dynamics are a
sequence of contact events -- exactly the kind of thing whose gradients are
either undefined or useless.  ARS never differentiates anything; it only ever
compares two numbers.  The trade is that it scales badly with parameter count,
which is why nobody trains a convolutional network this way.

Two details do real work:

  * OBSERVATION NORMALIZATION.  A linear policy multiplies each observation by
    one weight, so an input that ranges over 3 gets 30 times the influence of
    one that ranges over 0.1, before learning starts.  Dividing by a running
    standard deviation removes that accident.  ARS's authors found this single
    change was the difference between working and not working.
  * TOP-b SELECTION.  Averaging over ALL directions lets the many mediocre
    ones drown out the few good ones.  Keeping only the best few is a crude
    but effective way to follow the signal instead of the noise.
"""

import numpy as np


class Normalizer:
    """Running mean and variance of the observations (Welford's algorithm)."""

    def __init__(self, n):
        self.n = np.zeros(n)
        self.mean = np.zeros(n)
        self.m2 = np.zeros(n)

    def push(self, x):
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    FLOOR = 0.1

    def std(self):
        """Running standard deviation, with a FLOOR.

        The floor is not cosmetic.  Several of these observations barely move
        during an episode (the block's height changes by two millimetres), and
        dividing by their true standard deviation multiplies that wobble by
        five hundred -- so a policy that is meant to be making gentle finger
        corrections instead saturates its actuators on sensor noise and throws
        the block within five steps.  Ask for standardized inputs and you can
        get amplified ones.
        """
        s = np.sqrt(self.m2 / np.maximum(self.n - 1, 1))
        return np.maximum(s, self.FLOOR)

    def __call__(self, x):
        return (x - self.mean) / self.std()


def train(hand_factory, n_obs, n_act, iters=140, ndir=8, alpha=0.035,
          sigma=0.045, top=4, per_dir=2, seed=0, log=None, eval_every=20,
          eval_fn=None, resample=False, obs_mask=None, ep_len=140,
          goal_range=(-0.6, 0.6)):
    """`hand_factory(rng)` returns a fresh environment.

    `resample=True` builds a NEW one for every episode, which is how domain
    randomization enters: the physical parameters are redrawn each time, so the
    policy must work across a whole family of hands instead of one.
    `obs_mask` zeroes out parts of the observation -- used to ask what the
    policy can still do without knowing where the block is.
    """
    rng = np.random.default_rng(seed)
    W = np.zeros((n_act, n_obs))
    norm = Normalizer(n_obs)
    hand = hand_factory(rng)
    curve, evals = [], []

    def run(Wp, goal):
        nonlocal hand
        if resample:
            hand = hand_factory(rng)
        o = hand.reset(goal)
        tot = 0.0
        for _ in range(ep_len):
            if obs_mask is not None:
                o = o * obs_mask
            norm.push(o)
            a = Wp @ norm(o)
            o, r, done = hand.step(a)
            tot += r
            if done:
                break
        return tot

    for it in range(iters):
        deltas = rng.normal(size=(ndir, n_act, n_obs))
        goals = rng.uniform(*goal_range, size=per_dir)
        rp = np.zeros(ndir)
        rm = np.zeros(ndir)
        for k in range(ndir):
            for g in goals:
                rp[k] += run(W + sigma * deltas[k], g)
                rm[k] += run(W - sigma * deltas[k], g)
        rp /= per_dir
        rm /= per_dir
        order = np.argsort(-np.maximum(rp, rm))[:top]
        sr = np.concatenate([rp[order], rm[order]]).std()
        step = np.zeros_like(W)
        for k in order:
            step += (rp[k] - rm[k]) * deltas[k]
        W = W + alpha / (top * max(sr, 1e-6)) * step
        curve.append(float(0.5 * (rp.mean() + rm.mean())))
        if eval_fn is not None and (it + 1) % eval_every == 0:
            evals.append((it + 1, eval_fn(W, norm)))
            if log:
                log(f"    iter {it + 1}/{iters}  return {curve[-1]:.1f}  "
                    f"success {evals[-1][1]:.2f}")
    return W, norm, curve, evals


def policy_fn(W, norm, obs_mask=None):
    def f(o):
        if obs_mask is not None:
            o = o * obs_mask
        return W @ norm(o)
    return f
