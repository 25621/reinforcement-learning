"""A particle filter for a planar robot on a known map.

"Particle" because the belief is carried by a crowd of individual guesses
instead of by a mean and a covariance.  Each particle is one complete,
concrete hypothesis -- "the robot is HERE, facing THIS way" -- and the weight
attached to it says how well that hypothesis explains the laser scan.  There is
no assumption anywhere that the answer looks like a bell curve, which is the
whole reason to pay for it.

The other name for this, Monte Carlo Localization, comes from the Monte Carlo
casino: the method works by drawing random samples, the way you would estimate
the odds of a game by playing it many times instead of solving it.
"""

import numpy as np

from gridmap import GridMap


def wrap(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------- the motion
def sample_motion(parts, u, dt, rng, alpha):
    """Push every particle through the motion model, each with its own noise.

    Note what is NOT here: no covariance, no Jacobian.  The particle filter
    never linearizes anything, because it never needs a derivative -- it just
    runs the true model forward on each guess.  That is the whole difference
    from the EKF in project 26.
    """
    v, w = u
    n = len(parts)
    sv = np.sqrt(alpha[0] * v ** 2 + alpha[1] * w ** 2)
    sw = np.sqrt(alpha[2] * v ** 2 + alpha[3] * w ** 2)
    vn = v + sv * rng.standard_normal(n)
    wn = w + sw * rng.standard_normal(n)
    th = parts[:, 2]
    out = np.empty_like(parts)
    small = np.abs(wn) < 1e-6
    r = np.where(small, 0.0, vn / np.where(small, 1.0, wn))
    out[:, 0] = np.where(small, parts[:, 0] + vn * dt * np.cos(th),
                         parts[:, 0] - r * np.sin(th) + r * np.sin(th + wn * dt))
    out[:, 1] = np.where(small, parts[:, 1] + vn * dt * np.sin(th),
                         parts[:, 1] + r * np.cos(th) - r * np.cos(th + wn * dt))
    out[:, 2] = wrap(th + wn * dt)
    return out


# --------------------------------------------------------- the sensor models
def beam_log_likelihood(gmap, parts, angles, z, sigma, max_range=8.0,
                        w_hit=0.85, w_rand=0.15):
    """Score every particle by ray-casting the map from where it thinks it is.

    This is the honest model: it asks "if the robot really were here, what
    would the laser read?" and compares.  It costs one ray cast per particle
    per beam, which is why it is the expensive option.

    The mixture with a uniform term (w_rand) is not decoration.  A pure
    Gaussian assigns a likelihood of essentially zero to a single unexplained
    reading -- a person walking past, a glass door -- and zero times anything is
    zero, so ONE bad beam can annihilate a particle that is otherwise perfect.
    The uniform floor says "some fraction of readings are simply garbage" and
    caps how much damage any single beam can do.
    """
    exp_z = gmap.raycast(parts, angles, max_range=max_range)
    d = z[None, :] - exp_z
    p_hit = np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    p = w_hit * p_hit + w_rand / max_range
    return np.log(p + 1e-300).sum(axis=1)


def field_log_likelihood(gmap, parts, angles, z, sigma, max_range=8.0,
                         w_hit=0.85, w_rand=0.15):
    """Score every particle by where its beam ENDPOINTS land.

    Instead of casting a ray, project each measured range out from the particle
    and ask "how far is that point from the nearest wall?", using the
    precomputed distance field.  One array lookup instead of an 80-step march.

    The catch, which experiment 6 measures: this model cannot tell a wall from
    the space behind a wall.  It only knows "there is something near this
    point", not "the beam would have stopped before reaching it".
    """
    good = z < max_range - 1e-6
    th = parts[:, 2:3] + np.asarray(angles)[None, :]
    ex = parts[:, 0:1] + z[None, :] * np.cos(th)
    ey = parts[:, 1:2] + z[None, :] * np.sin(th)
    dist = gmap.lookup_distance(np.stack([ex, ey], axis=-1))
    p_hit = np.exp(-0.5 * (dist / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    p = w_hit * p_hit + w_rand / max_range
    logp = np.log(p + 1e-300)
    return (logp * good[None, :]).sum(axis=1)


# ---------------------------------------------------------------- resampling
def multinomial_resample(w, rng):
    """Draw N independent samples.  Simple, and noisier than it needs to be."""
    return rng.choice(len(w), size=len(w), p=w)


def systematic_resample(w, rng):
    """Draw ONE random number and take N evenly spaced samples from there.

    Also called "low-variance" resampling, and that name is the whole point: a
    particle with weight 1/N is guaranteed to survive, instead of surviving with
    probability 63%.  It costs the same and throws away far less information --
    which is why it is what every real implementation ships.
    """
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    cum = np.cumsum(w)
    cum[-1] = 1.0
    return np.searchsorted(cum, positions)


def stratified_resample(w, rng):
    """One random number per stratum: between multinomial and systematic."""
    n = len(w)
    positions = (rng.random(n) + np.arange(n)) / n
    cum = np.cumsum(w)
    cum[-1] = 1.0
    return np.searchsorted(cum, positions)


RESAMPLERS = {"multinomial": multinomial_resample,
              "systematic": systematic_resample,
              "stratified": stratified_resample}


def effective_sample_size(w):
    """1 / sum(w^2).  How many particles are actually doing any work.

    If one particle holds all the weight, ESS = 1: you have N particles and one
    hypothesis.  If all weights are equal, ESS = N.  It is the standard
    trigger for when to resample.
    """
    return 1.0 / np.sum(w ** 2)


# ------------------------------------------------------------- the estimator
def estimate(parts, w):
    """Weighted mean pose.  The heading must be averaged on the circle."""
    x = w @ parts[:, 0]
    y = w @ parts[:, 1]
    th = np.arctan2(w @ np.sin(parts[:, 2]), w @ np.cos(parts[:, 2]))
    return np.array([x, y, th])


class ParticleFilter:
    def __init__(self, gmap, particles, alpha, sigma_z, angles,
                 resampler="systematic", ess_frac=0.5, sensor="beam",
                 max_range=8.0, inject=0.0, adaptive=False, rng=None,
                 alpha_slow=0.03, alpha_fast=0.3):
        self.g = gmap
        self.parts = np.asarray(particles, dtype=float).copy()
        self.w = np.full(len(self.parts), 1.0 / len(self.parts))
        self.alpha = alpha
        self.sigma_z = sigma_z
        self.angles = np.asarray(angles)
        self.resampler = resampler
        self.ess_frac = ess_frac
        self.sensor = sensor
        self.max_range = max_range
        self.inject = inject
        self.adaptive = adaptive
        # Augmented MCL (Thrun, Burgard & Fox ch. 8).  Two running averages of
        # how well the scan is being explained: one that reacts in a few steps
        # (fast) and one that barely moves (slow).  While the filter is happy
        # they agree.  The moment the robot is moved, the fast average collapses
        # while the slow one lags, and the gap between them is used directly as
        # the fraction of particles to replace with fresh random guesses.
        # It is a self-tuning panic button: no injection when things are fine,
        # a large burst exactly when they are not.
        #
        # The two rates have to suit the length of the run.  The textbook
        # values (0.001 and 0.1) assume thousands of steps; over the 250 steps
        # here the slow average would never even reach its steady value, the
        # ratio would stay above 1 the whole time, and the panic button would
        # never fire.  0.03 and 0.3 give the slow average about 30 steps to
        # settle and the fast one about 3 to react.
        self.w_slow = 0.0
        self.w_fast = 0.0
        self.alpha_slow = alpha_slow
        self.alpha_fast = alpha_fast
        self.last_inject_frac = 0.0
        self.rng = np.random.default_rng(0) if rng is None else rng
        self.n_resample = 0

    def step(self, u, dt, z):
        self.parts = sample_motion(self.parts, u, dt, self.rng, self.alpha)
        ll = (beam_log_likelihood if self.sensor == "beam" else field_log_likelihood)(
            self.g, self.parts, self.angles, z, self.sigma_z, self.max_range)
        # Work in logs and subtract the maximum before exponentiating.  With 8
        # beams the raw likelihoods are around 1e-12; with 30 they underflow to
        # exactly zero and every weight becomes NaN.  This one line is the
        # difference between a filter that works and one that dies silently.
        # The AVERAGE likelihood of this scan across particles, before it is
        # normalized away.  Normalizing destroys it -- weights always sum to
        # one, however badly every particle is doing -- so it has to be taken
        # here.  This one number is what tells a filter that it is lost.
        # Per-beam geometric-mean likelihood, relative to the best particle.
        # Dividing the log-likelihood by the beam count before exponentiating
        # keeps the number in a sane range: with 12 beams the raw product is
        # around 1e-12 even when everything is fine, and comparing two such
        # numbers through a running average is numerically hopeless.
        w_avg = float(np.mean(np.exp((ll - ll.max()) / max(len(self.angles), 1))))
        self.w_slow += self.alpha_slow * (w_avg - self.w_slow)
        self.w_fast += self.alpha_fast * (w_avg - self.w_fast)

        logw = np.log(self.w + 1e-300) + ll
        logw -= logw.max()
        w = np.exp(logw)
        s = w.sum()
        self.w = w / s if s > 0 else np.full(len(w), 1.0 / len(w))

        ess = effective_sample_size(self.w)
        if self.ess_frac is not None and ess < self.ess_frac * len(self.w):
            idx = RESAMPLERS[self.resampler](self.w, self.rng)
            self.parts = self.parts[idx]
            self.w = np.full(len(self.w), 1.0 / len(self.w))
            self.n_resample += 1
        frac = self.inject
        if self.adaptive:
            frac = max(0.0, 1.0 - self.w_fast / max(self.w_slow, 1e-300))
            frac = min(frac, 0.5)
        self.last_inject_frac = frac
        if frac > 0.0:
            self._inject(frac)
        return ess

    def _inject(self, frac):
        """Replace a fraction of the worst particles with fresh random guesses.

        Without this a particle filter can never recover from being wrong about
        where it is, because resampling only ever copies particles it already
        has -- and if none of them is near the truth, no amount of copying will
        invent one.  Experiment 4 measures exactly that.
        """
        from gridmap import free_poses
        k = max(1, int(frac * len(self.parts)))
        worst = np.argsort(self.w)[:k]
        self.parts[worst] = free_poses(self.g, k, self.rng)
        self.w[worst] = self.w.mean()
        self.w /= self.w.sum()

    def estimate(self):
        return estimate(self.parts, self.w)
