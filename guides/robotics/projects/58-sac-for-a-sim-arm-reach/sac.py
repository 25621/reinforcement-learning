"""Soft Actor-Critic, with the entropy temperature exposed as the main dial.

SAC is off-policy: it keeps every transition it has ever seen in a replay
buffer and re-uses them, which is why it needs perhaps ten times fewer
interactions than PPO.  The price is that it is fiddlier, and the fiddliest
part is one number, alpha.

"Soft" is not marketing.  Ordinary actor-critic maximises expected reward.
SAC maximises reward PLUS the entropy of the policy:

    J = E[ sum_t  r_t  +  alpha * H(pi(.|s_t)) ]

Entropy H measures how spread out the action distribution is: a policy that
always plays the same action has entropy near minus infinity (for continuous
actions), a policy that spreads over the whole action range has high entropy.
Adding it to the objective means "get reward, but stay as undecided as you can
afford to be".  That is a built-in exploration bonus, and it keeps the policy
from committing to the first mediocre habit it finds.

alpha sets the exchange rate between reward and entropy.  Too small and the
policy collapses to a deterministic habit early and stops exploring; too large
and it deliberately stays random and never sharpens up.  There is no scale-free
value, because it is measured in "reward units per nat", and reward units are
whatever the task designer chose -- which is exactly why automatic tuning
exists: instead of picking alpha, you pick a target entropy and let alpha
chase it.

Other decoded names:

* **twin critics** -- two Q networks, and the update uses the *smaller* of the
  two.  A single critic's errors are systematically optimistic, because the
  actor is trained to find and exploit whatever the critic overestimates.
  Taking the minimum of two independently-initialised critics cancels much of
  that bias.
* **target network** -- a slowly-moving copy used to compute the regression
  target, so the critic is not chasing a value it changes with every step.
* **tanh squashing** -- the Gaussian is squashed through tanh to fit the
  action limits; the log-probability then needs a correction term, because
  squeezing a distribution through a nonlinear function changes its density.
"""

import numpy as np
import torch
import torch.nn as nn

import reach as RE

LOG_STD_MIN, LOG_STD_MAX = -6.0, 2.0


def mlp(i, o, hidden=256):
    return nn.Sequential(nn.Linear(i, hidden), nn.ReLU(),
                         nn.Linear(hidden, hidden), nn.ReLU(),
                         nn.Linear(hidden, o))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden)
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False, with_logp=True):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        if deterministic:
            u = mu
        else:
            u = mu + std * torch.randn_like(mu)
        a = torch.tanh(u)
        if not with_logp:
            return a, None
        # log N(u) minus the log-determinant of the tanh, summed over dims
        logp = (-0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * np.log(2 * np.pi)).sum(-1)
        logp = logp - (2 * (np.log(2) - u - torch.nn.functional.softplus(-2 * u))).sum(-1)
        return a, logp


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    def __init__(self, size, obs_dim, act_dim):
        self.o = np.zeros((size, obs_dim), np.float32)
        self.a = np.zeros((size, act_dim), np.float32)
        self.r = np.zeros(size, np.float32)
        self.o2 = np.zeros((size, obs_dim), np.float32)
        self.d = np.zeros(size, np.float32)
        self.size, self.ptr, self.full = size, 0, False

    def add(self, o, a, r, o2, d):
        n = len(o)
        idx = (self.ptr + np.arange(n)) % self.size
        self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx] = o, a, r, o2, d
        self.ptr = (self.ptr + n) % self.size
        self.full = self.full or self.ptr < n
        return idx

    def sample(self, bs, rng):
        hi = self.size if self.full else self.ptr
        j = rng.integers(0, hi, bs)
        return (torch.as_tensor(self.o[j]), torch.as_tensor(self.a[j]),
                torch.as_tensor(self.r[j]), torch.as_tensor(self.o2[j]),
                torch.as_tensor(self.d[j]))


def train_sac(seed=0, total_steps=30_000, n_env=8, alpha=0.05, auto_alpha=False,
              target_entropy=None, gamma=0.98, tau=0.005, lr=1e-3, bs=256,
              utd=1.0, start_steps=2000, sparse=False, eval_every=5000,
              hidden=256, env_kw=None, log=None):
    """``total_steps`` counts single-robot transitions, not batched steps."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = RE.BatchReach(n_env, seed=seed, sparse=sparse, **(env_kw or {}))
    od, ad = env.obs_dim, env.act_dim

    actor = Actor(od, ad, hidden)
    critic = Critic(od, ad, hidden)
    target = Critic(od, ad, hidden)
    target.load_state_dict(critic.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_c = torch.optim.Adam(critic.parameters(), lr=lr)

    log_alpha = torch.tensor(float(np.log(alpha)), requires_grad=True)
    opt_al = torch.optim.Adam([log_alpha], lr=lr)
    if target_entropy is None:
        target_entropy = -float(ad)          # the usual default: -dim(A)

    buf = Replay(200_000, od, ad)
    obs = env.reset()
    hist, n_updates = [], 0

    for step in range(0, total_steps, n_env):
        if step < start_steps:
            a = rng.uniform(-1, 1, (n_env, ad)).astype(np.float32)
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(obs, dtype=torch.float32),
                             with_logp=False)
            a = a.numpy()
        obs2, r, done, _ = env.step(a)
        # ``done`` here is a time limit, not a failure: the arm is not in a
        # terminal state, the clock simply ran out.  Bootstrapping is therefore
        # NOT cut off -- treating a time limit as terminal teaches the critic
        # that the world ends every two seconds.
        buf.add(obs.astype(np.float32), a.astype(np.float32),
                r.astype(np.float32), obs2.astype(np.float32),
                np.zeros(n_env, np.float32))
        obs = obs2

        if step >= start_steps:
            for _ in range(max(1, int(round(utd * n_env)))):
                o, ac, rw, o2, dn = buf.sample(bs, rng)
                al = log_alpha.exp().detach() if auto_alpha else torch.tensor(alpha)
                with torch.no_grad():
                    a2, logp2 = actor(o2)
                    tq1, tq2 = target(o2, a2)
                    y = rw + gamma * (1 - dn) * (torch.min(tq1, tq2) - al * logp2)
                q1, q2 = critic(o, ac)
                loss_c = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
                opt_c.zero_grad()
                loss_c.backward()
                opt_c.step()

                anew, logp = actor(o)
                qa1, qa2 = critic(o, anew)
                loss_a = (al * logp - torch.min(qa1, qa2)).mean()
                opt_a.zero_grad()
                loss_a.backward()
                opt_a.step()

                if auto_alpha:
                    loss_al = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
                    opt_al.zero_grad()
                    loss_al.backward()
                    opt_al.step()

                with torch.no_grad():
                    for p, pt in zip(critic.parameters(), target.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
                n_updates += 1

        if (step // n_env) % max(1, (eval_every // n_env)) == 0 or step + n_env >= total_steps:
            ev = RE.evaluate(make_controller(actor), n_ep=100, seed=4242,
                             **{k: v for k, v in (env_kw or {}).items()
                                if k in ('mass_scale', 'damp_scale', 'gear')})
            with torch.no_grad():
                _, lp = actor(torch.as_tensor(obs, dtype=torch.float32))
                ent = float(-lp.mean())
            hist.append((step, ev["dist"], ev["success"], ent,
                         float(log_alpha.exp().detach()) if auto_alpha else alpha))
            if log:
                print(f"  {log} {step:6d} steps  dist {ev['dist']:.4f}  "
                      f"success {ev['success']:.2f}  H {ent:+.2f}  "
                      f"alpha {hist[-1][4]:.4f}", flush=True)

    return actor, np.array(hist), n_updates


def make_controller(actor):
    def ctrl(obs):
        with torch.no_grad():
            a, _ = actor(torch.as_tensor(obs, dtype=torch.float32),
                         deterministic=True, with_logp=False)
        return a.numpy()
    return ctrl
