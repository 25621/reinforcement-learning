"""Project 34 — Mini MBPO: feed SAC on imagined data.

Projects 32 and 33 used the model to *plan*: no policy at all, just search, every step.
MBPO uses it the other way round. There is a policy — SAC, unchanged from Phase 5 — and
the model's job is simply to *manufacture training data* for it. This is the Dyna idea:

    real step  ->  real buffer
    model      ->  short imagined rollouts branched off real states  ->  model buffer
    SAC        ->  trains on a mix of both

The one design choice that makes or breaks it is the **rollout length k**. Project 32
measured how fast an imagined trajectory drifts from reality; that measurement is the
reason k is 1 here and not 15. A 1-step imagined transition, branched from a state the
agent really visited, is almost true. A 15-step one is fiction.

The honest experiment, and the reason this file has four arms instead of two:

    MBPO does two things at once. It adds model data, AND it takes many more gradient
    steps per real environment step (a high update-to-data ratio, "UTD") — because
    imagined data is cheap, so why not train harder on it. Those are different
    interventions and the literature routinely credits the first for the work of the
    second. So we run SAC at UTD=1, SAC at UTD=10, and MBPO at UTD=10. If MBPO only
    beats SAC-at-UTD-1, the model did nothing that a bigger `for` loop couldn't do.

  python3 mbpo.py     # ~6 min on 12 hyperthreads
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-ddpg-on-pendulum"))
sys.path.insert(0, str(HERE.parent / "32-pets-random-shooting-mpc"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import cc_lib as cc  # noqa: E402  — SAC, verbatim from Phase 5
import mbrl_lib as M  # noqa: E402  — the ensemble, verbatim from project 32
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"

SEEDS = [0, 1, 2]
TOTAL_STEPS = 3_000
START_STEPS = 400
MODEL_START = 1_000         # don't imagine anything until the model has seen real data
MODEL_TRAIN_EVERY = 250     # real steps between model refits

# Imagined TRANSITIONS generated per refit — not imagined rollouts. The distinction is the
# experiment: a k=1 arm makes 4000 one-step rollouts, a k=10 arm makes 400 ten-step ones,
# and both hand SAC exactly 4000 fresh transitions. So the arms differ ONLY in how deep
# into the model's imagination the data came from, never in how much of it there is.
IMAGINED_PER_REFIT = 4_000
MODEL_BUF_SIZE = 20_000     # a RING buffer: old fantasy is overwritten, not hoarded
REAL_RATIO = 0.05           # fraction of each SAC batch drawn from REAL data
HIDDEN = 128


def gen_model_rollouts(model, real_buf, model_buf, agent, k, n_starts, act_limit, rng, gen):
    """Branch `n_starts` imagined rollouts of length `k` off states the agent really visited.

    "Branched" is the whole trick. Every imagined rollout *starts* from a real state pulled
    out of the real buffer, so it begins on solid ground and only has to stay accurate for
    k steps. Compare with imagining a whole 200-step episode from the initial state, which
    is what a naive Dyna would do: by step 40 the model is dreaming, and SAC would be
    learning to be brilliant inside a fantasy.

    The actions come from the CURRENT policy, not from the buffer — the point is to
    generate data about what the policy is doing *now*, which is exactly what an off-policy
    learner is otherwise starved of.
    """
    E = model.n_members
    idx = rng.integers(0, real_buf.size, size=n_starts)
    o = torch.as_tensor(real_buf.obs[idx])
    rows = torch.arange(n_starts)

    for _ in range(k):
        with torch.no_grad():
            a, _ = agent.actor(o, deterministic=False, with_logprob=False)
            a = a.clamp(-act_limit, act_limit)
            # Each imagined transition is predicted by ONE randomly chosen ensemble member
            # (PETS calls this TS1). Averaging the members instead would quietly erase the
            # disagreement that encodes what the model does not know, and hand SAC a stream
            # of confident-looking transitions in exactly the regions where it is worst.
            member = torch.randint(0, E, (n_starts,), generator=gen)
            o_e = o.unsqueeze(0).expand(E, -1, -1).contiguous()
            a_e = a.unsqueeze(0).expand(E, -1, -1).contiguous()
            o2_all, r_all = model.predict(o_e, a_e, sample=True, generator=gen)
            o2 = o2_all[member, rows]
            r = r_all[member, rows]

        # done=0 always: Pendulum never terminates, it only times out. A model inventing
        # terminations it was never shown would teach SAC that the world can end, and the
        # bootstrap would be zeroed for no reason.
        model_buf.add_batch(o.numpy(), a.numpy(), r.numpy(), o2.numpy())
        o = o2


class ModelBuffer(cc.ReplayBuffer):
    """SAC's replay buffer, plus a vectorized insert and — crucially — a short memory.

    It is a RING buffer sized to hold only the last few refits' worth of imagined data.
    Once it is full, new fantasy overwrites the oldest, and that is the point:

    Imagined transitions have a shelf life. A transition dreamt up at step 1,000 was
    produced by a model trained on 1,000 samples, acting under a policy that was terrible.
    It is not merely stale, it is *wrong*, and if it lingers in the buffer it will still be
    poisoning SAC's batches at step 3,000 — when a far better model and a far better policy
    are available to replace it. Real data is precious and kept forever; imagined data is
    cheap and should be thrown away as soon as something better can be dreamt.
    """

    def add_batch(self, o, a, r, o2):
        n = len(o)
        i = self.ptr
        idx = (np.arange(i, i + n) % self.capacity)
        self.obs[idx] = o
        self.act[idx] = a
        self.rew[idx] = r.reshape(-1, 1)
        self.next_obs[idx] = o2
        self.done[idx] = 0.0
        self.ptr = (i + n) % self.capacity
        self.size = min(self.size + n, self.capacity)


def mixed_batch(real_buf, model_buf, batch_size, real_ratio, rng):
    """One SAC batch: mostly imagined, with a thread of reality stitched through it.

    5% real is the MBPO default and it looks absurdly low until you notice what the
    real transitions are for. They are not the training signal — the model data is.
    They are the anchor: the one part of the batch the model cannot be wrong about.
    """
    if model_buf.size == 0:
        return real_buf.sample(batch_size, rng)
    n_real = max(1, int(batch_size * real_ratio))
    n_model = batch_size - n_real
    r = real_buf.sample(n_real, rng)
    m = model_buf.sample(n_model, rng)
    return tuple(torch.cat([a, b], dim=0) for a, b in zip(r, m))


def train(seed, use_model, utd, rollout_k=1, label=""):
    """SAC, with an optional model feeding it. `use_model=False` is plain SAC.

    One function, one flag. SAC-at-UTD-1, SAC-at-UTD-10 and MBPO are the same code path
    with different arguments — so a difference in the curves cannot be a difference in
    the implementation.
    """
    import gymnasium as gym

    cfg = cc.sac_config(env_id="Pendulum-v1", seed=seed, hidden=HIDDEN, batch_size=256)
    cc.set_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed + 555)

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    env.action_space.seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = cc.Agent(cfg, obs_dim, act_dim, act_limit)
    real_buf = cc.ReplayBuffer(obs_dim, act_dim, 100_000)
    model_buf = ModelBuffer(obs_dim, act_dim, MODEL_BUF_SIZE)
    model = M.GaussianEnsemble(5, obs_dim, act_dim, hidden=200, n_layers=3) if use_model else None
    mbuf = M.TransitionBuffer(obs_dim, act_dim) if use_model else None

    hist = {"steps": [], "eval_return": [], "hold_mse": []}
    o, _ = env.reset(seed=seed)
    t0 = time.time()

    for t in range(1, TOTAL_STEPS + 1):
        a = env.action_space.sample() if t <= START_STEPS else agent.act(o, rng=rng)
        o2, r, term, trunc, _ = env.step(a)
        real_buf.add(o, a, r, o2, float(term))
        if use_model:
            mbuf.add(o, a, r, o2)
        o = o2
        if term or trunc:
            o, _ = env.reset()

        if use_model and t >= MODEL_START and t % MODEL_TRAIN_EVERY == 0:
            M.train_model(model, mbuf, epochs=25, batch_size=256, rng=rng)
            # Same number of imagined transitions for every k, so the arms differ only in
            # how far into the dream the data came from.
            gen_model_rollouts(model, real_buf, model_buf, agent, rollout_k,
                               IMAGINED_PER_REFIT // rollout_k, act_limit, rng, gen)

        if t >= START_STEPS:
            for _ in range(utd):
                batch = (
                    mixed_batch(real_buf, model_buf, cfg.batch_size, REAL_RATIO, rng)
                    if use_model
                    else real_buf.sample(cfg.batch_size, rng)
                )
                agent.update(batch)

        if t % 250 == 0:
            ret = cc.evaluate(eval_env, agent, 3, seed)
            hist["steps"].append(t)
            hist["eval_return"].append(ret)

    env.close()
    eval_env.close()
    return {
        "seed": seed, "label": label, "steps": hist["steps"],
        "return": hist["eval_return"], "wall": time.time() - t0,
    }


ARMS = [
    ("SAC (UTD=1)", dict(use_model=False, utd=1)),
    ("SAC (UTD=10)", dict(use_model=False, utd=10)),
    ("MBPO k=1 (UTD=10)", dict(use_model=True, utd=10, rollout_k=1)),
    ("MBPO k=10 (UTD=10)", dict(use_model=True, utd=10, rollout_k=10)),
]


def job(args):
    label, kw, seed = args
    return train(seed=seed, label=label, **kw)


def main():
    OUT.mkdir(exist_ok=True)

    print("=== four arms: is it the model, or just more gradient steps? ===", flush=True)
    jobs = [(label, kw, s) for label, kw in ARMS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as ex:
        res = list(ex.map(job, jobs))

    by = {}
    for r in res:
        by.setdefault(r["label"], []).append(r)

    def steps_to(label, thresh=-300):
        r = np.array([x["return"] for x in by[label]])
        st = np.array(by[label][0]["steps"])
        m = r.mean(0)
        i = np.where(m >= thresh)[0]
        return (st[i[0]] if len(i) else None), r[:, -1].mean()

    print(f"\n  {'arm':22s} {'final return':>14s} {'steps to -300':>15s} {'wall':>7s}")
    for label, _ in ARMS:
        at, fin = steps_to(label)
        wall = np.mean([x["wall"] for x in by[label]])
        print(f"  {label:22s} {fin:14.1f} {str(at):>15s} {wall:6.0f}s")

    fig, ax = ps.new_axes(7.6, 4.4)
    colors = [ps.BASELINE, ps.SERIES[2], ps.SERIES[0], ps.SERIES[3]]
    for (label, _), c in zip(ARMS, colors):
        r = np.array([x["return"] for x in by[label]])
        st = np.array(by[label][0]["steps"])
        ax.plot(st, r.mean(0), color=c, lw=2, label=label)
        ax.fill_between(st, r.min(0), r.max(0), color=c, alpha=0.12)
    ax.axhline(-150, color=ps.INK_MUTED, ls="--", lw=1)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "MBPO's win, and how much of it the model actually earned",
              "real environment steps", "evaluation return", OUT / "learning_curves.png")

    # A bar chart of the one number that matters: real steps to reach -300.
    fig, ax = ps.new_axes(7.2, 3.8)
    labels, vals = [], []
    for label, _ in ARMS:
        at, _ = steps_to(label)
        labels.append(label.replace(" (", "\n("))
        vals.append(at if at else TOTAL_STEPS)
    ax.bar(labels, vals, color=colors, width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 40, f"{v}", ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ax.grid(axis="x", visible=False)
    ps.finish(fig, ax, "Real environment steps needed to reach a return of -300",
              "", "real env steps (lower is better)", OUT / "sample_efficiency.png")


if __name__ == "__main__":
    main()
