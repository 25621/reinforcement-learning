"""PPO, written out with every piece switchable so the pieces can be tested.

PPO = Proximal Policy Optimization.  "Proximal" means near: each update is
kept *near* the policy that collected the data.  That is the whole idea.  Data
collected by the old policy is only valid evidence about the old policy, so if
an update walks too far the gradient is being computed from a distribution the
new policy no longer visits, and the run collapses.

The clipped objective is how PPO enforces that.  For each state-action it forms

    ratio = pi_new(a|s) / pi_old(a|s)

which is 1 if nothing changed.  The loss takes the *worse* of the raw and the
clipped ratio, so once the ratio leaves [1-eps, 1+eps] in the helpful direction
the gradient goes flat: making that action even more likely buys nothing.
Experiment 2 checks the natural follow-up question -- if you only take one
gradient step per batch, ratio is 1 by construction, so does clipping do
anything at all?
"""

import numpy as np
import torch
import torch.nn as nn

import cp_env as E


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64, log_std=-0.5):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, hidden), nn.Tanh(),
                                nn.Linear(hidden, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        # One spread for the whole state space, learned.  A state-dependent
        # spread is more expressive and much easier to collapse to zero early
        # in training, which stops exploration dead.
        self.log_std = nn.Parameter(torch.full((act_dim,), float(log_std)))
        # Small final layer: the initial policy should be near "do nothing"
        # rather than slamming the actuator at random.
        for m in (self.pi[-1], self.v[-1]):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.zeros_(m.bias)

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def value(self, obs):
        return self.v(obs).squeeze(-1)


class ReturnNorm:
    """Divide rewards by the running spread of the returns they produce.

    Without this the project does not train at all, and the reason is worth
    knowing.  The cost of a fallen pole is unbounded -- the cart keeps
    accelerating away and the position term keeps growing -- so value targets
    reach the hundreds while a good episode scores about 0.15.  The critic's
    squared-error gradient is then thousands of times larger than the policy's,
    and because both networks share one optimiser and one global gradient-norm
    clip, the policy gradient gets scaled down to nothing.  The policy simply
    stops learning while the loss curve looks busy.

    Dividing by a running standard deviation is scale-only: it cannot change
    which policy is best, just how big the numbers are.
    """

    def __init__(self, n_env, gamma):
        self.ret = np.zeros(n_env)
        self.gamma = gamma
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def __call__(self, r, done):
        self.ret = self.ret * self.gamma + r
        batch_mean, batch_var, n = self.ret.mean(), self.ret.var(), len(self.ret)
        delta = batch_mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        m_a, m_b = self.var * self.count, batch_var * n
        self.var = max(0.0, (m_a + m_b + delta ** 2 * self.count * n / tot) / tot)
        self.count = tot
        self.ret[done] = 0.0
        return r / (np.sqrt(self.var) + 1e-8)


def gae(rews, vals, last_val, dones, gamma, lam):
    """Generalised Advantage Estimation.

    lam = 0 trusts the critic completely (one-step TD: low variance, biased by
    however wrong the critic is).  lam = 1 ignores the critic and sums the real
    rewards to the end (unbiased, high variance).  In between interpolates --
    the "generalised" in the name is over that knob.
    """
    T, N = rews.shape
    adv = np.zeros_like(rews)
    nxt = last_val
    run = np.zeros(N)
    for t in range(T - 1, -1, -1):
        # An episode boundary cuts the chain: the value of the state after a
        # reset says nothing about the state before it.
        alive = 1.0 - dones[t]
        delta = rews[t] + gamma * nxt * alive - vals[t]
        run = delta + gamma * lam * alive * run
        adv[t] = run
        nxt = vals[t]
    return adv


def train_ppo(seed=0, total_steps=250_000, n_env=32, n_steps=128, epochs=10,
              minibatches=4, clip=0.2, gamma=0.99, lam=0.95, lr=3e-4,
              ent_coef=0.0, vf_coef=0.5, adv_norm=True, max_grad_norm=0.5,
              init_angle=0.20, plant=None, eval_every=10, reward_norm=True,
              log=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    plant = plant or {}
    env = E.BatchCartPole(n_env, seed=seed, init_angle=init_angle, **plant)
    net = ActorCritic(env.obs_dim, env.act_dim)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)

    n_iters = total_steps // (n_env * n_steps)
    mb = (n_env * n_steps) // minibatches
    hist = []
    obs = env.obs()
    rnorm = ReturnNorm(n_env, gamma)
    diag = dict(kl=[], clipfrac=[])

    for it in range(n_iters):
        O = np.zeros((n_steps, n_env, env.obs_dim), np.float32)
        Aa = np.zeros((n_steps, n_env, env.act_dim), np.float32)
        LP = np.zeros((n_steps, n_env), np.float32)
        RW = np.zeros((n_steps, n_env), np.float32)
        VL = np.zeros((n_steps, n_env), np.float32)
        DN = np.zeros((n_steps, n_env), np.float32)
        for t in range(n_steps):
            ot = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = net.dist(ot)
                a = d.sample()
                LP[t] = d.log_prob(a).sum(-1).numpy()
                VL[t] = net.value(ot).numpy()
            O[t], Aa[t] = obs, a.numpy()
            obs, r, dn, _ = env.step(a.numpy()[:, 0] * E.U_MAX)
            RW[t] = rnorm(r, dn) if reward_norm else r
            DN[t] = dn
        with torch.no_grad():
            last_v = net.value(torch.as_tensor(obs, dtype=torch.float32)).numpy()

        ADV = gae(RW, VL, last_v, DN, gamma, lam)
        RET = ADV + VL

        b_o = torch.as_tensor(O.reshape(-1, env.obs_dim))
        b_a = torch.as_tensor(Aa.reshape(-1, env.act_dim))
        b_lp = torch.as_tensor(LP.reshape(-1))
        b_adv = torch.as_tensor(ADV.reshape(-1).astype(np.float32))
        b_ret = torch.as_tensor(RET.reshape(-1).astype(np.float32))

        kls, cfs = [], []
        idx = np.arange(len(b_o))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(idx), mb):
                j = torch.as_tensor(idx[s:s + mb])
                d = net.dist(b_o[j])
                lp = d.log_prob(b_a[j]).sum(-1)
                ratio = (lp - b_lp[j]).exp()
                adv = b_adv[j]
                if adv_norm:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                if clip is None:
                    pg = -(ratio * adv).mean()          # vanilla policy gradient
                else:
                    pg = -torch.min(ratio * adv,
                                    ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                v_loss = ((net.value(b_o[j]) - b_ret[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + vf_coef * v_loss - ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                opt.step()
                with torch.no_grad():
                    kls.append(float((b_lp[j] - lp).mean()))
                    cfs.append(float(((ratio - 1).abs() > (clip or 0.2)).float().mean()))
        diag["kl"].append(float(np.mean(kls)))
        diag["clipfrac"].append(float(np.mean(cfs)))

        if (it + 1) % eval_every == 0 or it == n_iters - 1:
            # only the physical parameters go to the evaluator; switches like
            # the reward cap belong to training, not to the scoring function
            phys = {k: v for k, v in plant.items() if k in ("M", "m", "l")}
            c = E.episode_cost(make_controller(net), n_ep=100, seed=4242,
                               init_angle=init_angle, **phys)
            hist.append(((it + 1) * n_env * n_steps, c["cost"], c["fell"]))
            if log:
                print(f"  {log} {(it + 1) * n_env * n_steps:7d} steps  "
                      f"cost {c['cost']:8.3f}  fell {c['fell']:.2f}", flush=True)
    return net, np.array(hist), diag


def make_controller(net):
    """Greedy (mean) action -- exploration noise is for training only."""
    def ctrl(obs, state):
        with torch.no_grad():
            a = net.pi(torch.as_tensor(obs, dtype=torch.float32))
        return a.numpy()[:, 0] * E.U_MAX
    return ctrl
