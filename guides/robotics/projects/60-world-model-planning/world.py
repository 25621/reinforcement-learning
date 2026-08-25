"""Learn what the robot does, then think before acting.

Everything in projects 54-58 learns a *policy*: a direct map from observation
to action.  This one learns a *model* -- what the world will look like one step
from now -- and then searches over action sequences inside that model at
decision time.  The policy is not stored anywhere; it is computed fresh at
every step by planning.

Why bother, when a policy is faster at run time?  Because the data required is
completely different.  A cloned policy needs someone who already knows how to
do the task.  A model only needs the robot to *move* -- flailing around counts.
The data here is exactly that, and it is called PLAY DATA for that reason:
random, goal-free interaction, like a child mashing buttons.  Play data cannot
train behaviour cloning at all (there is no behaviour to clone), and it is
enough to train a model that can then be planned with, for any goal.

The planner is CEM -- the Cross-Entropy Method.  The name comes from its
origins in rare-event simulation, where the algorithm minimises the
cross-entropy (a distance between probability distributions) between a sampling
distribution and the ideal one concentrated on the best outcomes.  What it does
in practice is much simpler than the name: sample a batch of random action
sequences, keep the best few ("elites"), refit a Gaussian to those, sample
again.  A few rounds of that and the Gaussian sits on a good plan.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

_P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_P, "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402

STATE_DIM = 6              # q(2), qd(2), puck(2)
ACT_DIM = 2


def env_state(env):
    """The part of the world that evolves.  The goal is not in here.

    The goal does not influence the physics at all -- pushing the puck works
    the same wherever you happen to want it to end up.  Leaving it out of the
    model means one model serves every goal, and the planner supplies the goal
    when it scores a plan.  A model that took the goal as input would have to
    learn that its own input is irrelevant, from data.
    """
    return np.concatenate([env.q, env.qd, env.puck])


def set_env_state(env, s):
    env.q, env.qd = s[:2].copy(), s[2:4].copy()
    env.puck = s[4:6].copy()


# ---------------------------------------------------------------------------
# play data
# ---------------------------------------------------------------------------
def collect_play(n_eps, seed=0, kind="random", ep_len=A.EP_LEN):
    """Interaction with no goal in mind.

    ``kind="random"``: uniform random actions, smoothed a little.  Pure white
    noise per step would make the arm shake in place and almost never touch the
    puck, so the actions are correlated in time -- the difference between
    "flailing" and "vibrating".
    """
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng)
    S, Aa, S2, G = [], [], [], []
    touches = 0
    for _ in range(n_eps):
        env.reset()
        a = rng.uniform(-1, 1, ACT_DIM)
        for _ in range(ep_len):
            if kind == "random":
                a = np.clip(0.7 * a + 0.7 * rng.uniform(-1, 1, ACT_DIM), -1, 1)
            else:                                   # noisy scripted explorer
                base, _ = A.expert_action(env, side=1 if rng.random() < 0.5 else -1)
                a = np.clip(base + rng.normal(0, 0.5, ACT_DIM), -1, 1)
            s = env_state(env)
            p0 = env.puck.copy()
            env.step(a)
            S.append(s)
            Aa.append(a.copy())
            S2.append(env_state(env))
            G.append(env.goal.copy())
            touches += float(np.linalg.norm(env.puck - p0) > 1e-6)
    return (np.array(S, np.float32), np.array(Aa, np.float32),
            np.array(S2, np.float32), np.array(G, np.float32),
            dict(n=len(S), touch_frac=touches / max(1, len(S))))


# ---------------------------------------------------------------------------
# the models
# ---------------------------------------------------------------------------
class StateModel(nn.Module):
    """Predicts the CHANGE in state, not the next state.

    Predicting s' directly makes the network spend all its capacity learning
    the identity function -- the arm barely moves in 50 ms, so s' is almost s,
    and a model that just copies its input already scores a tiny error while
    knowing nothing.  Predicting the delta removes that freebie and puts the
    whole error budget on the part that is actually physics.
    """

    def __init__(self, hidden=256):
        super().__init__()
        self.net = nets.MLP(STATE_DIM + ACT_DIM, STATE_DIM, hidden, depth=2)
        self.register_buffer("s_mu", torch.zeros(STATE_DIM))
        self.register_buffer("s_sd", torch.ones(STATE_DIM))
        self.register_buffer("d_mu", torch.zeros(STATE_DIM))
        self.register_buffer("d_sd", torch.ones(STATE_DIM))

    def fit_norm(self, S, D):
        self.s_mu.copy_(torch.tensor(S.mean(0)))
        self.s_sd.copy_(torch.tensor(S.std(0) + 1e-3))
        self.d_mu.copy_(torch.tensor(D.mean(0)))
        self.d_sd.copy_(torch.tensor(D.std(0) + 1e-6))

    def forward(self, s, a):
        x = torch.cat([(s - self.s_mu) / self.s_sd, a], -1)
        return s + self.net(x) * self.d_sd + self.d_mu


class LatentModel(nn.Module):
    """Encode the state, roll the dynamics forward in the code, decode a reward.

    This is the shape of Dreamer / TD-MPC: nothing in the loop ever
    reconstructs the state, only the reward that planning needs.  The reason
    those systems exist is that their observations are images, where predicting
    the next observation means predicting every pixel -- most of which is
    irrelevant wallpaper.  Here the state is six numbers, so there is nothing to
    compress; experiment 4 measures whether the extra machinery pays anyway.
    """

    def __init__(self, latent=16, hidden=256):
        super().__init__()
        self.enc = nets.MLP(STATE_DIM + 2, latent, hidden, depth=2)   # +goal
        self.dyn = nets.MLP(latent + ACT_DIM, latent, hidden, depth=2)
        self.rew = nets.MLP(latent, 1, hidden, depth=1)
        self.register_buffer("s_mu", torch.zeros(STATE_DIM + 2))
        self.register_buffer("s_sd", torch.ones(STATE_DIM + 2))

    def encode(self, s, goal):
        x = torch.cat([s, goal], -1)
        return self.enc((x - self.s_mu) / self.s_sd)

    def step(self, z, a):
        return z + self.dyn(torch.cat([z, a], -1))

    def reward(self, z):
        return self.rew(z).squeeze(-1)


def train_state_model(S, Aa, S2, epochs=60, bs=512, lr=1e-3, seed=0, hidden=256,
                      log=None):
    nets.seed_all(seed)
    D = S2 - S
    model = StateModel(hidden)
    model.fit_norm(S, D)
    st = torch.tensor(S)
    ac = torch.tensor(Aa)
    d = torch.tensor((D - D.mean(0)) / (D.std(0) + 1e-6))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for ep in range(epochs):
        perm = torch.randperm(len(st))
        tot = 0.0
        for i in range(0, len(st), bs):
            j = perm[i:i + bs]
            x = torch.cat([(st[j] - model.s_mu) / model.s_sd, ac[j]], -1)
            loss = ((model.net(x) - d[j]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(j)
        sched.step()
        if log and (ep + 1) % 20 == 0:
            print(f"  {log} epoch {ep + 1}: {tot / len(st):.5f}", flush=True)
    model.eval()
    return model


def train_latent_model(S, Aa, S2, goals, epochs=60, bs=512, lr=1e-3, seed=0,
                       latent=16, hidden=256, horizon=5, log=None):
    """Trained on short chunks, with a latent-consistency loss.

    A latent model has no reconstruction target, so nothing stops the encoder
    from mapping every state to the same point and the reward head from
    predicting the average.  The consistency term -- roll the latent forward and
    require it to match the encoding of the state that really followed -- is
    what keeps the code meaningful.
    """
    nets.seed_all(seed)
    x_all = np.concatenate([S, goals], 1)
    model = LatentModel(latent, hidden)
    model.s_mu.copy_(torch.tensor(x_all.mean(0)))
    model.s_sd.copy_(torch.tensor(x_all.std(0) + 1e-3))

    st, ac, st2 = torch.tensor(S), torch.tensor(Aa), torch.tensor(S2)
    gl = torch.tensor(goals)
    rew = -torch.linalg.norm(st2[:, 4:6] - gl, dim=1)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(st) - horizon
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            z = model.encode(st[j], gl[j])
            loss = 0.0
            for k in range(horizon):
                jk = j + k
                z = model.step(z, ac[jk])
                with torch.no_grad():
                    z_true = model.encode(st2[jk], gl[j])
                loss = loss + ((z - z_true) ** 2).mean() + \
                    ((model.reward(z) - rew[jk]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(j)
        sched.step()
        if log and (ep + 1) % 20 == 0:
            print(f"  {log} epoch {ep + 1}: {tot / n:.5f}", flush=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------
@torch.no_grad()
def cem_plan(rollout_fn, horizon=10, pop=100, iters=4, elites=10, rng=None,
             mean=None, act_dim=ACT_DIM):
    """Cross-Entropy Method over open-loop action sequences.

    ``rollout_fn(actions) -> score`` takes (pop, horizon, act_dim) and returns
    one number per candidate.  Everything model-specific lives in there, so the
    same planner drives the learned model, the latent model and the true
    simulator.
    """
    rng = rng or np.random.default_rng(0)
    mu = np.zeros((horizon, act_dim)) if mean is None else mean.copy()
    sd = np.ones((horizon, act_dim)) * 0.8
    best = None
    for _ in range(iters):
        cand = np.clip(mu + sd * rng.normal(size=(pop, horizon, act_dim)), -1, 1)
        score = rollout_fn(cand.astype(np.float32))
        top = np.argsort(-score)[:elites]
        mu = cand[top].mean(0)
        sd = cand[top].std(0) + 0.05
        best = cand[top[0]]
    return mu, best


L1, L2 = A.PlanarArm().l


def tip_of(q):
    """Forward kinematics, in torch, for a batch of joint angles.

    The planner is allowed to know this.  Kinematics are geometry -- link
    lengths off a drawing -- while dynamics are mass, friction and contact,
    which is what actually has to be learned from data.  Making the model learn
    trigonometry it could be told would be an own goal.
    """
    a, b = q[:, 0], q[:, 0] + q[:, 1]
    return torch.stack([L1 * torch.cos(a) + L2 * torch.cos(b),
                        L1 * torch.sin(a) + L2 * torch.sin(b)], dim=1)


def approach_bonus(tip, puck, goal, w):
    """How close the tip is to the spot it would have to push FROM.

    Without this the score is flat for the whole approach: until something
    touches the puck, every plan gives exactly the same distance-to-goal, so
    the search has nothing to climb and picks noise.  This term is the
    difference between a planner that finds the manoeuvre and one that wanders.
    """
    if w <= 0:
        return 0.0
    d = goal - puck
    ghat = d / (torch.linalg.norm(d, dim=1, keepdim=True) + 1e-9)
    contact = puck - (A.R_PUCK + A.R_TIP) * ghat
    return -w * torch.linalg.norm(tip - contact, dim=1)


def model_rollout_fn(model, s0, goal, kind="state", shape_w=0.3):
    """Score candidate plans inside a learned model."""
    def fn(cand):
        pop, H, _ = cand.shape
        a = torch.tensor(cand)
        g = torch.tensor(goal, dtype=torch.float32)[None].repeat(pop, 1)
        if kind == "state":
            s = torch.tensor(s0, dtype=torch.float32)[None].repeat(pop, 1)
            score = torch.zeros(pop)
            for k in range(H):
                s = model(s, a[:, k])
                score = score - torch.linalg.norm(s[:, 4:6] - g, dim=1)
                score = score + approach_bonus(tip_of(s[:, :2]), s[:, 4:6], g, shape_w)
        else:
            z = model.encode(torch.tensor(s0, dtype=torch.float32)[None].repeat(pop, 1), g)
            score = torch.zeros(pop)
            for k in range(H):
                z = model.step(z, a[:, k])
                score = score + model.reward(z)
        return score.numpy()
    return fn


def true_rollout_fn(env, s0, goal, shape_w=0.3):
    """The same scoring, but simulating for real.  This is the upper bound."""
    def fn(cand):
        pop, H, _ = cand.shape
        out = np.zeros(pop)
        saved = env.state()
        for i in range(pop):
            env.set_state(saved)
            set_env_state(env, s0)
            tot = 0.0
            for k in range(H):
                env.step(cand[i, k])
                tot -= float(np.linalg.norm(env.puck - goal))
                if shape_w > 0:
                    d = goal - env.puck
                    ghat = d / (np.linalg.norm(d) + 1e-9)
                    contact = env.puck - (A.R_PUCK + A.R_TIP) * ghat
                    tot -= shape_w * float(np.linalg.norm(env.arm.tip(env.q) - contact))
            out[i] = tot
        env.set_state(saved)
        return out
    return fn


def run_planner(make_fn, n_eps=30, seed=999, horizon=10, pop=100, iters=4,
                elites=10, replan_every=1, env_kw=None):
    """Model-predictive control: plan, execute a little, re-plan."""
    rng = np.random.default_rng(seed)
    env = A.PushEnv(rng, **(env_kw or {}))
    ok, errs, t_plan = 0, [], []
    import time
    for _ in range(n_eps):
        env.reset()
        mean = None
        done = False
        while not done:
            s0 = env_state(env)
            t0 = time.time()
            mu, best = cem_plan(make_fn(env, s0, env.goal), horizon=horizon,
                                pop=pop, iters=iters, elites=elites, rng=rng,
                                mean=mean)
            t_plan.append(time.time() - t0)
            for k in range(replan_every):
                _, _, done, _ = env.step(mu[k])
                if done:
                    break
            # warm start: shift the plan one step and pad with zeros
            mean = np.concatenate([mu[replan_every:], np.zeros((replan_every, ACT_DIM))])
        ok += env.success
        errs.append(env.err)
    return dict(success=ok / n_eps, err=float(np.mean(errs)),
                ms_per_decision=float(np.mean(t_plan)) * 1000)
