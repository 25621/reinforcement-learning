"""DAgger: keep the expert on call, and ask it about the states YOU visit.

Behaviour cloning trains on the expert's states and is then tested on its own.
DAgger closes that loop.  Each round it drives the *current* policy, records
every state it lands in, asks the expert what it would have done there, adds
those pairs to the pile, and retrains.

The name is short for "Dataset Aggregation" -- the pile is never thrown away,
it only grows.  Whether that aggregation is load-bearing or just tidy is
something experiment 5 in ``run.py`` actually tests.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "54-behavior-cloning-on-a-sim-arm"))

import arm as A            # noqa: E402
import nets                # noqa: E402


def collect_labelled(policy, n_eps, beta, rng, side=1, env=None):
    """Roll out a beta-mixture of expert and policy; label EVERY state.

    ``beta`` is the probability of *executing* the expert's action at each
    step.  It does not change what gets recorded -- the label is always the
    expert's action -- it changes which states get visited.  beta = 1 is
    ordinary demonstration collection; beta = 0 is pure policy states, which
    is what makes the data useful for fixing the policy's own mistakes.
    """
    env = env or A.PushEnv(rng)
    O, Y, ok = [], [], 0
    for _ in range(n_eps):
        obs = env.reset()
        done = False
        while not done:
            a_exp, _ = A.expert_action(env, side=side)
            O.append(obs.copy())
            Y.append(np.asarray(a_exp, np.float32))
            if policy is None or rng.random() < beta:
                a_run = a_exp
            else:
                a_run = policy(obs)
            obs, _, done, _ = env.step(a_run)
        ok += env.success
    return (np.array(O, np.float32), np.array(Y, np.float32), ok / max(1, n_eps))


def beta_schedule(kind):
    """How much the expert drives during data collection, round by round."""
    if kind == "zero":                       # pure DAgger: policy drives always
        return lambda i: 0.0
    if kind == "decay":                      # the original paper's p^i
        return lambda i: 0.5 ** i
    if kind == "one":                        # expert drives always == plain BC
        return lambda i: 1.0
    raise ValueError(kind)


def run_dagger(n_init=25, rounds=5, eps_per_round=20, beta="zero", seed=0,
               aggregate=True, epochs=350, eval_n=60, log=None):
    """Returns one row per round: labels used, success rate, val loss."""
    rng = np.random.default_rng(1000 + seed)
    bfn = beta_schedule(beta)

    O, Y, _ = A.collect_demos(n_init, seed=seed, side_mode=1)
    net, norm, hist = nets.train_bc(O, Y, epochs=epochs, seed=seed)
    pol = nets.make_policy(net, norm)
    ev = A.evaluate(pol, n=eval_n, seed=999)
    out = [dict(round=0, labels=len(O), success=ev["success"], err=ev["err"],
                val=hist[-1][1], collect_success=1.0)]
    if log:
        print(f"  {log} round 0: {len(O)} labels, success {ev['success']:.3f}",
              flush=True)

    for i in range(1, rounds + 1):
        On, Yn, csr = collect_labelled(pol, eps_per_round, bfn(i - 1), rng)
        if aggregate:
            O, Y = np.concatenate([O, On]), np.concatenate([Y, Yn])
        else:
            # the ablation: forget everything older than the last round
            O, Y = On, Yn
        net, norm, hist = nets.train_bc(O, Y, epochs=epochs, seed=seed)
        pol = nets.make_policy(net, norm)
        ev = A.evaluate(pol, n=eval_n, seed=999)
        out.append(dict(round=i, labels=len(O), success=ev["success"],
                        err=ev["err"], val=hist[-1][1], collect_success=csr))
        if log:
            print(f"  {log} round {i}: {len(O)} labels, success {ev['success']:.3f}",
                  flush=True)
    return out, net, norm


def bc_baseline(n_demos, seed=0, epochs=350, eval_n=60):
    """Plain behaviour cloning on ``n_demos`` fresh expert demonstrations."""
    O, Y, _ = A.collect_demos(n_demos, seed=seed, side_mode=1)
    net, norm, hist = nets.train_bc(O, Y, epochs=epochs, seed=seed)
    ev = A.evaluate(nets.make_policy(net, norm), n=eval_n, seed=999)
    return dict(labels=len(O), success=ev["success"], err=ev["err"],
                val=hist[-1][1]), net, norm


def shift_gap(net, norm, seed=333, n_eps=20):
    """Action error on expert states vs on the policy's own states."""
    Oe, Ae, _ = A.collect_demos(20, seed=seed, side_mode=1)
    on_expert = nets.action_mse(net, norm, Oe, Ae)
    pol = nets.make_policy(net, norm)
    rng = np.random.default_rng(seed + 1)
    Op, Ap, _ = collect_labelled(pol, n_eps, 0.0, rng)
    on_policy = nets.action_mse(net, norm, Op, Ap)
    return on_expert, on_policy
