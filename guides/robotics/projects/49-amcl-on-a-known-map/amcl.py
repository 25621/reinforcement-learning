"""AMCL = Monte Carlo Localization plus the *Adaptive* part.

Project 27 already built a particle filter on a known map: sample the motion,
weight by the laser scan, resample, and inject random particles when the robot
looks lost.  So what does AMCL add, and why is a second project worth it?

Two things, and both are about **cost**, not about accuracy:

  1. **KLD-sampling** -- the particle count is chosen fresh at every step.
     A plain filter is stuck with one N.  It has to be huge (thousands) so
     that global localization can work at the start, and then it keeps paying
     for those thousands forever, long after the cloud has collapsed to a
     10 cm blob where fifty particles would do.  KLD-sampling reads the
     *spread* of the current cloud and asks for exactly as many particles as
     that spread needs.
  2. **Update thresholds** -- do nothing at all until the robot has actually
     moved.  A standing robot gets no new information from a new scan, but a
     naive filter still resamples on it, and resampling without new
     information throws diversity away for free.

Everything else is imported from project 27 rather than rewritten.
"""

import math
import os
import sys

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJ, "27-particle-filter"))

from gridmap import GridMap, free_poses, office_map        # noqa: E402,F401
from pf import (ParticleFilter, estimate, sample_motion,   # noqa: E402
                wrap, effective_sample_size,
                beam_log_likelihood, field_log_likelihood,
                systematic_resample)

# z-score for the (1 - delta) quantile of a standard normal.  delta = 0.01 is
# the usual choice: "I want the true belief and my sampled belief to be within
# eps of each other, with 99% confidence."
Z_DELTA = {0.01: 2.32635, 0.05: 1.64485, 0.10: 1.28155}


def kld_bound(k, eps=0.05, delta=0.01):
    """How many samples a k-bin belief needs -- the KLD-sampling formula.

        n = (k-1)/(2*eps) * ( 1 - 2/(9(k-1)) + sqrt(2/(9(k-1))) * z )^3

    Read it in plain language: the number of particles you need grows with
    **k**, the number of distinct cells your belief is spread over, and shrinks
    with **eps**, how much error you are willing to tolerate in the sampled
    distribution.  A belief squeezed into one cell needs almost nothing; a
    belief smeared over 400 cells needs thousands.

    "KLD" is Kullback-Leibler divergence, the standard way to measure how far
    one probability distribution is from another (named after Solomon
    Kullback and Richard Leibler, who introduced it in 1951).  The bound above
    is the sample count at which the KL divergence between the true belief and
    the one your particles represent stays below eps with probability
    1 - delta.  The point is that this depends on the SHAPE of the belief,
    which changes every step -- so the right N changes every step too.
    """
    if k <= 1:
        return 1
    z = Z_DELTA.get(delta, 2.32635)
    a = 2.0 / (9.0 * (k - 1))
    return int(np.ceil((k - 1) / (2.0 * eps) * (1 - a + math.sqrt(a) * z) ** 3))


def bin_index(parts, bin_xy=0.5, bin_th=math.radians(15)):
    """Which histogram cell each particle falls in (x, y, theta)."""
    return np.stack([np.floor(parts[:, 0] / bin_xy).astype(np.int64),
                     np.floor(parts[:, 1] / bin_xy).astype(np.int64),
                     np.floor(wrap(parts[:, 2]) / bin_th).astype(np.int64)],
                    axis=1)


def kld_resample(parts, w, rng, eps=0.05, delta=0.01, n_min=50, n_max=5000,
                 bin_xy=0.5, bin_th=math.radians(15), block=25):
    """Draw particles until the KLD bound is satisfied.

    The loop is the whole algorithm: draw a few, see how many NEW histogram
    cells they landed in, recompute how many the bound now demands, and stop
    as soon as you have that many.  A tight cloud fills no new cells, the
    bound stays low, and the loop exits after a couple of blocks.  A spread
    cloud keeps filling new cells and the bound keeps running away from you.
    """
    cdf = np.cumsum(w)
    cdf[-1] = 1.0
    seen = set()
    out = []
    need = n_min
    n = 0
    while n < need and n < n_max:
        u = rng.random(block)
        idx = np.searchsorted(cdf, u)
        idx = np.clip(idx, 0, len(parts) - 1)
        take = parts[idx]
        out.append(take)
        n += block
        for b in map(tuple, bin_index(take, bin_xy, bin_th)):
            seen.add(b)
        need = max(n_min, min(kld_bound(len(seen), eps, delta), n_max))
    return np.vstack(out)[:max(n, n_min)]


class AMCL:
    """Monte Carlo localization with an adaptive sample size.

    `eps=None` turns KLD off and the filter behaves exactly like project 27's
    fixed-size one -- which is the control every claim below is measured
    against.
    """

    def __init__(self, gmap, angles, n0=3000, eps=0.05, delta=0.01,
                 alpha=(0.02, 0.02, 0.02, 0.02), sigma_z=0.25, sensor="field",
                 max_range=8.0, n_min=50, n_max=5000, rng=None,
                 d_thresh=0.2, a_thresh=math.radians(15), init=None,
                 inject=0.0, augmented=False, ess_frac=0.5):
        self.g = gmap
        self.angles = np.asarray(angles)
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.parts = (free_poses(gmap, n0, self.rng) if init is None
                      else np.asarray(init, float).copy())
        self.w = np.full(len(self.parts), 1.0 / len(self.parts))
        self.eps, self.delta = eps, delta
        self.alpha, self.sigma_z, self.sensor = alpha, sigma_z, sensor
        self.max_range = max_range
        self.n_min, self.n_max = n_min, n_max
        self.d_thresh, self.a_thresh = d_thresh, a_thresh
        self.inject, self.augmented, self.ess_frac = inject, augmented, ess_frac
        self.w_slow, self.w_fast = 0.0, 0.0
        self.alpha_slow, self.alpha_fast = 0.03, 0.3
        self._acc_d, self._acc_a = 0.0, 0.0
        self.n_updates, self.n_skipped = 0, 0
        self.last_inject = 0.0

    # ---------------------------------------------------------------- motion
    def predict(self, u, dt):
        self.parts = sample_motion(self.parts, u, dt, self.rng, self.alpha)
        self._acc_d += abs(u[0]) * dt
        self._acc_a += abs(u[1]) * dt

    def due(self):
        """Has the robot moved enough to be worth a measurement update?"""
        return (self._acc_d >= self.d_thresh or self._acc_a >= self.a_thresh
                or self.d_thresh <= 0.0)

    # ----------------------------------------------------------- measurement
    def update(self, z):
        if not self.due():
            self.n_skipped += 1
            return
        self._acc_d, self._acc_a = 0.0, 0.0
        self.n_updates += 1
        llf = beam_log_likelihood if self.sensor == "beam" else field_log_likelihood
        ll = llf(self.g, self.parts, self.angles, z, self.sigma_z,
                 self.max_range)
        w_avg = float(np.mean(np.exp((ll - ll.max())
                                     / max(len(self.angles), 1))))
        self.w_slow += self.alpha_slow * (w_avg - self.w_slow)
        self.w_fast += self.alpha_fast * (w_avg - self.w_fast)

        logw = np.log(self.w + 1e-300) + ll
        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        self.w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))

        ess = effective_sample_size(self.w)
        if ess < self.ess_frac * len(self.w):
            if self.eps is None:
                idx = systematic_resample(self.w, self.rng)
                self.parts = self.parts[idx]
            else:
                self.parts = kld_resample(self.parts, self.w, self.rng,
                                          eps=self.eps, delta=self.delta,
                                          n_min=self.n_min, n_max=self.n_max)
            self.w = np.full(len(self.parts), 1.0 / len(self.parts))

        frac = self.inject
        if self.augmented:
            frac = min(max(0.0, 1.0 - self.w_fast / max(self.w_slow, 1e-300)),
                       0.5)
        self.last_inject = frac
        if frac > 0.0:
            k = max(1, int(frac * len(self.parts)))
            worst = np.argsort(self.w)[:k]
            self.parts[worst] = free_poses(self.g, k, self.rng)
            self.w[worst] = self.w.mean()
            self.w /= self.w.sum()

    def estimate(self):
        return estimate(self.parts, self.w)

    @property
    def n(self):
        return len(self.parts)


def pose_error(est, true):
    """Position error in metres and heading error in radians."""
    return (float(np.hypot(est[0] - true[0], est[1] - true[1])),
            abs(float(wrap(est[2] - true[2]))))


def scan(gmap, pose, angles, rng, sigma=0.05, max_range=8.0,
         dropout=0.0):
    """A noisy laser scan from the true pose."""
    z = gmap.raycast(np.asarray(pose, float)[None, :], angles,
                     max_range=max_range)[0]
    z = z + rng.normal(0.0, sigma, z.shape)
    if dropout > 0:
        miss = rng.random(z.shape) < dropout
        z[miss] = max_range
    return np.clip(z, 0.0, max_range)
