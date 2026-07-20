"""Rectified flow / flow matching, written out in full.

DDPM (project 24's `diffusion_lib`) needs a whole apparatus: a beta schedule,
alphas, cumulative alpha-bars, a posterior variance.  Rectified flow needs one
line for training and one line for sampling.  This file is short on purpose —
the shortness IS the lesson.

The idea
--------
Pick a clean clip `x0` and a pure-noise clip `eps`.  Draw a straight line
between them and slide along it with a dial `t` from 0 (clean) to 1 (noise):

    x_t = (1 - t) * x0 + t * eps

Differentiate that with respect to t and you get the same answer everywhere on
the line:

    dx_t/dt = eps - x0            <- the "velocity"

So training is: pick a random t, build x_t, and ask the network to predict that
velocity.  Generation is: start from noise at t = 1 and walk backwards along
whatever velocities the network reports, using ordinary Euler steps.

Why "flow matching"?
    A *flow* is a velocity field that transports one probability distribution
    into another over time — here, noise into video.  You cannot compute the
    true field for the whole distribution, but Lipman et al. showed that
    matching the network to the *per-example* straight-line velocity gives the
    same optimum in expectation.  You "match" your model to a flow, hence the
    name.

Why "rectified"?
    To *rectify* is to make straight.  DDPM's reverse trajectories are curved,
    so a sampler that takes big steps cuts corners and leaves the data
    manifold.  Training on straight-line paths makes the learned trajectories
    much straighter, so bigger steps hurt less — which is the whole reason
    modern models sample in 20-30 steps instead of 250.

Why "velocity", when DDPM's network predicts noise?
    They are two labels for the same underlying object, and you can convert
    between them.  What differs is what stays constant: the noise target keeps
    a fixed scale across timesteps, while the velocity target keeps a fixed
    *direction* along the whole path.  A constant direction is exactly what a
    large ODE step needs in order to be accurate.
"""

import torch


class RectifiedFlow:
    """Training target and ODE sampler for a straight-line probability path."""

    T_SCALE = 1000.0        # the model's timestep embedding was written for
    #                         integers in [0, 1000); t in [0,1] is scaled into
    #                         that range so the same embedding code serves both

    def interpolate(self, x0, t, noise):
        """x_t on the straight line from clean (t=0) to noise (t=1)."""
        tt = t.view(-1, *([1] * (x0.dim() - 1)))
        return (1 - tt) * x0 + tt * noise

    def target(self, x0, noise):
        """The velocity the model must predict — the SAME for every t."""
        return noise - x0

    def sample_t(self, n, generator=None):
        return torch.rand(n, generator=generator)

    @torch.no_grad()
    def sample(self, model, shape, steps=60, generator=None,
               return_traj=False, **model_kw):
        """Euler integration of dx/dt = v(x, t) from t = 1 down to t = 0.

        No noise is injected along the way and no schedule is consulted: with a
        velocity field, sampling is just "walk in the direction the model
        points, for as long as the step size says".
        """
        x = torch.randn(shape, generator=generator)
        ts = torch.linspace(1.0, 0.0, steps + 1)
        traj = [x.clone()]
        for i in range(steps):
            t = ts[i].expand(shape[0])
            v = model(x, t * self.T_SCALE, **model_kw)
            x = x + (ts[i + 1] - ts[i]) * v          # dt is negative
            if return_traj:
                traj.append(x.clone())
        return (x, traj) if return_traj else x


@torch.no_grad()
def ddim_trajectory(ddpm, model, shape, steps=60, generator=None, **model_kw):
    """Project 24's DDIM sampler, but keeping every intermediate x.

    Needed so the two samplers can be compared on the same footing: we measure
    how straight each one's path through latent space is.
    """
    ts = torch.linspace(ddpm.steps - 1, 0, steps).long()
    x = torch.randn(shape, generator=generator)
    traj = [x.clone()]
    for i, t in enumerate(ts):
        tt = torch.full((shape[0],), int(t), dtype=torch.long)
        eps = model(x, tt, **model_kw)
        a = ddpm.abar[t]
        x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-3, 3)
        if i + 1 < len(ts):
            a_next = ddpm.abar[ts[i + 1]]
            x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        else:
            x = x0
        traj.append(x.clone())
    return x, traj


def straightness(traj):
    """How straight is a sampling path?  1.0 = perfectly straight.

    Take the total displacement from start to finish, then check every
    individual step against it: if each step points the same way as the whole
    journey, the cosine is 1 and the path is a straight line.  A curved path
    wanders, so its steps point elsewhere and the average cosine drops.

    This is a *measurement of the claim* rectified flow is named after, not a
    training loss — nothing optimises it directly.
    """
    xs = torch.stack(traj)                       # (S+1, B, ...)
    total = (xs[-1] - xs[0]).flatten(1)
    total = total / total.norm(dim=1, keepdim=True).clamp(min=1e-8)
    steps = (xs[1:] - xs[:-1]).flatten(2)
    steps = steps / steps.norm(dim=2, keepdim=True).clamp(min=1e-8)
    cos = (steps * total[None]).sum(-1)          # (S, B)
    return float(cos.mean())
