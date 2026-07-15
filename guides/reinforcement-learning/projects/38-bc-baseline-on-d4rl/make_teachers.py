"""Build the three 'behavior policies' that generate every Phase 7 dataset.

WHY THIS FILE EXISTS
--------------------
Phase 7 is offline RL: you are handed a fixed dataset and may not touch the
environment. The obvious dataset to use is D4RL, the standard benchmark. We
cannot: D4RL is pinned to `mujoco-py`, a dead dependency that does not build
against the MuJoCo 3.x installed here. So we build our own D4RL, the same way
the real one was built:

    train ONE agent with SAC, freeze it at three moments in its life,
    and let each frozen copy drive the robot to produce a dataset.

    random  = the network before it has learned anything (flailing)
    medium  = the network partway through training (mediocre but not stupid)
    expert  = the network at the end of training (good)

That is not a shortcut — it is literally D4RL's recipe. Its `-medium` datasets
are collected by a policy deliberately stopped at about a third of expert
performance. The point of the ladder is that Phase 7's central question,
"how good can a learner be when it can only watch?", has a different answer at
each rung, and project 43 measures the whole curve.

WHAT GETS COMMITTED
-------------------
The three actor networks (~90 KB each), not the datasets (~30 MB). A dataset is
just a policy plus a seed, and re-rolling it is fast: driving the robot is pure
physics at ~5,700 steps/s, while TRAINING the teacher costs a gradient step per
env step at ~180/s — 30x slower. So we pay the slow part once, here, and every
other project regenerates its data from the committed weights in under a minute.

    python3 make_teachers.py      # ~13 min. You do NOT need to run this;
                                  # teachers/*.pt is already committed.
"""

import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# Phase 5 already wrote SAC. Importing it rather than re-typing it is the whole
# point of having written it well: the teacher here is exactly the agent from
# project 28, with no changes.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "26-ddpg-on-pendulum"))
import cc_lib  # noqa: E402

# Teacher training is a single long run, not N seeds in parallel, so the usual
# "one thread per process" rule does not apply — there are no sibling processes
# to leave cores for. cc_lib pins 1 thread at import; undo that here only.
torch.set_num_threads(4)

# HalfCheetah, not Hopper — and the reason is a measurement, not a preference.
# D4RL's headline tasks are halfcheetah, hopper and walker2d, so any of them would be
# faithful. But project 28 ran all five bodies on this exact box and found that in a
# CPU-sized budget (60k steps) SAC reaches 2,768 on HalfCheetah and 374 on Hopper.
# Hopper is not slow here, it is STUCK: hopping requires falling forward and catching
# yourself, and the safe alternative — stand still, collect the survival bonus, never
# risk a fall — is a local optimum that a short run cannot escape. Confirmed by
# re-running it: 347 at 60k steps, flat.
#
# A teacher that never learned to move cannot be an "expert", and without a real expert
# the whole quality ladder that Phase 7 is built on collapses. HalfCheetah also cannot
# fall over (no early termination), so every episode is exactly 1,000 steps and the
# return differences are about the QUALITY of the running, uncontaminated by how long
# the robot survived.
ENV_ID = "HalfCheetah-v5"
HIDDEN = 128
TOTAL_STEPS = 90_000
SNAP_EVERY = 5_000
OUT = Path(__file__).resolve().parent / "teachers"


def main():
    OUT.mkdir(exist_ok=True)
    cfg = cc_lib.sac_config(
        env_id=ENV_ID, seed=0, hidden=HIDDEN, total_steps=TOTAL_STEPS,
        start_steps=5_000, update_after=5_000, eval_episodes=5,
    )
    cc_lib.set_seed(cfg.seed)  # BEFORE the networks are built, or the run is not reproducible
    rng = np.random.default_rng(cfg.seed)

    env, eval_env = gym.make(ENV_ID), gym.make(ENV_ID)
    env.action_space.seed(cfg.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = cc_lib.Agent(cfg, obs_dim, act_dim, act_limit)
    buf = cc_lib.ReplayBuffer(obs_dim, act_dim, cfg.buffer_size)

    # ---- the 'random' teacher: uniformly random actions, not an untrained network ----
    # The obvious choice is the network before training. Measured, it scores -2.9: an
    # untrained net emits actions near zero, so the cheetah lies still, earns nothing,
    # and the dataset it produces is 100k copies of "not moving". Nothing can be learned
    # from that, and the lesson would be about our bad dataset, not about offline RL.
    #
    # Uniform random actions flail, and flailing VISITS PLACES. That is what a `random`
    # dataset is for: wide coverage, terrible reward. It is also how D4RL defines both
    # its random datasets and the zero-point of its score, so we match it.
    rr = 0.0
    for i in range(5):
        o, _ = eval_env.reset(seed=cfg.seed + 10_000 + i)
        done = False
        while not done:
            o, r, te, tr, _ = eval_env.step(eval_env.action_space.sample())
            rr += r
            done = te or tr
    random_return = rr / 5
    print(f"step      0  return {random_return:7.1f}  (uniform random actions -> "
          f"the 'random' teacher)", flush=True)

    snaps = []
    o, _ = env.reset(seed=cfg.seed)
    t0 = time.time()
    for t in range(1, TOTAL_STEPS + 1):
        a = env.action_space.sample() if t <= cfg.start_steps else agent.act(o, rng=rng)
        o2, r, term, trunc, _ = env.step(a)
        # `term` only: the hopper falling over is the world ending, a 1000-step
        # timeout is not. Zeroing the bootstrap on a timeout teaches the critic
        # that the world ends at step 1000, which it does not.
        buf.add(o, a, r, o2, float(term))
        o = o2
        if term or trunc:
            o, _ = env.reset()
        if t >= cfg.update_after:
            agent.update(buf.sample(cfg.batch_size, rng))
        if t % SNAP_EVERY == 0:
            ret = cc_lib.evaluate(eval_env, agent, 5, cfg.seed)
            snaps.append((t, ret, {k: v.clone() for k, v in agent.actor.state_dict().items()}))
            print(f"step {t:6d}  return {ret:7.1f}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- pick the three rungs of the ladder ----
    expert_step, expert_ret, expert_sd = max(snaps, key=lambda s: s[1])
    # "medium" = the checkpoint closest to a third of expert performance, which is
    # the definition D4RL uses. Not "halfway through training" — training is not
    # linear in return, and it is the RETURN we want to control, not the clock.
    target = expert_ret / 3.0
    medium_step, medium_ret, medium_sd = min(snaps, key=lambda s: abs(s[1] - target))

    meta = dict(env_id=ENV_ID, hidden=HIDDEN, obs_dim=obs_dim,
                act_dim=act_dim, act_limit=act_limit)
    # The random teacher has no weights — "sample uniformly" is not a network. The
    # `uniform` flag is how offline_lib knows to ignore the state_dict and just shake
    # the joystick.
    torch.save({"uniform": True, "state_dict": None, "train_step": 0,
                "eval_return": random_return, **meta}, OUT / "random.pt")
    print(f"saved {'random':6s}  step {0:6d}  return {random_return:7.1f}  (uniform actions)")
    for name, step, ret, sd in [("medium", medium_step, medium_ret, medium_sd),
                                ("expert", expert_step, expert_ret, expert_sd)]:
        torch.save({"uniform": False, "state_dict": sd, "train_step": step,
                    "eval_return": ret, **meta}, OUT / f"{name}.pt")
        print(f"saved {name:6s}  step {step:6d}  return {ret:7.1f}")


if __name__ == "__main__":
    main()
