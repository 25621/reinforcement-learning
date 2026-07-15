r"""Project 44 — epsilon-greedy on a chain: watch random exploration die exponentially.

The chain (this is Osband's "deep sea" in miniature):

    start                                             goal
      s0 --> s1 --> s2 --> ... --> s(n-2) --> s(n-1)   reward +1
      <--    <--    <--            <--
      |
      `-- an episode lasts EXACTLY n-1 steps, so every "left" is a step you
          never get back. To reach the goal you must choose RIGHT n-1 times
          in a row, with no second chances.

Under uniform random actions that has probability 2^-(n-1): each extra link in
the chain HALVES your chance of ever seeing the reward. That is the whole story,
and the experiments below measure it three ways.

    python3 chain.py       # ~5.5 min, three figures
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"

LEFT, RIGHT = 0, 1


class Chain:
    """n states in a row. Start at s0, reward +1 only for arriving at s(n-1).

    `distractor` puts a small reward on stepping LEFT at s0 (bumping the left
    wall). It is worth 100x less than the goal, and it is what turns a merely
    *hard* exploration problem into a trap: the agent finds it in one step and
    then has something to exploit.
    """

    def __init__(self, n, distractor=0.0):
        self.n = n
        self.distractor = distractor
        self.horizon = n - 1          # exactly enough steps, if you never waste one

    def reset(self):
        self.s = 0
        self.t = 0
        return self.s

    def step(self, a):
        self.t += 1
        if a == RIGHT:
            self.s = min(self.s + 1, self.n - 1)
            r = 1.0 if self.s == self.n - 1 else 0.0
        else:
            r = self.distractor if self.s == 0 else 0.0
            self.s = max(self.s - 1, 0)
        done = (self.s == self.n - 1) or (self.t >= self.horizon)
        return self.s, r, done


def q_learning(env, episodes, eps, seed, alpha=0.5, gamma=0.99, stop_on_first_reward=False):
    """Tabular Q-learning with epsilon-greedy actions and RANDOM tie-breaking.

    The tie-breaking matters. Q starts at all-zeros, so before the first reward
    every action looks equally good. `np.argmax` would silently always return
    action 0 (= LEFT) and the agent would march into the wall forever — a
    strawman, not a fair test. Breaking ties by coin flip gives epsilon-greedy
    the best possible version of itself, and it STILL fails.
    """
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n, 2))
    first_reward_ep = None
    returns = np.zeros(episodes)

    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(2))
            else:
                q = Q[s]
                best = np.flatnonzero(q == q.max())      # all tied-best actions
                a = int(rng.choice(best))                # ... pick among them fairly
            s2, r, done = env.step(a)
            target = r + (0.0 if s2 == env.n - 1 else gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            total += r
            s = s2
        returns[ep] = total
        if first_reward_ep is None and total >= 1.0:
            first_reward_ep = ep + 1
            if stop_on_first_reward:
                return first_reward_ep, returns[: ep + 1], Q
    return first_reward_ep, returns, Q


# ---------------------------------------------------------------- experiment 1
def discovery_vs_length(lengths, eps_list, seeds=20, budget=40_000):
    """Episodes until the agent stumbles on the reward for the first time."""
    med = {}
    for k, eps in enumerate(eps_list):
        rows = []
        for n in lengths:
            firsts = []
            for sd in range(seeds):
                # Independent seeds per epsilon. (Give them the SAME seeds and the three
                # curves land on top of each other to the last digit — see the README.)
                ep, _, _ = q_learning(Chain(n), budget, eps, seed=1000 * sd + n + 77_777 * k,
                                      stop_on_first_reward=True)
                firsts.append(budget if ep is None else ep)   # censored at the budget
            rows.append(np.median(firsts))
            print(f"  eps={eps}  n={n:2d}  median first reward @ episode {rows[-1]:.0f}", flush=True)
        med[eps] = np.array(rows)
    return med


def plot_discovery(lengths, med, path):
    fig, ax = ps.new_axes(7.4, 4.4)
    lengths = np.asarray(lengths)
    theory = np.log(2) * 2.0 ** (lengths - 1)      # median of a geometric with p = 2^-(n-1)
    ax.plot(lengths, theory, color=ps.INK_MUTED, ls="--", lw=1.6, zorder=1,
            label="theory: 0.69 x 2^(n-1)")
    for i, (eps, v) in enumerate(med.items()):
        ax.plot(lengths, v, color=ps.SERIES[i], lw=2.0, marker="o", ms=4.5,
                label=f"epsilon = {eps}")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ps.finish(fig, ax, "Every extra link doubles the wait for the first reward",
              "chain length n (states)", "episodes until first reward (median of 20 seeds)", path)


# ---------------------------------------------------------------- experiment 2
def distractor_trap(n=6, eps_list=(0.05, 0.1, 0.2, 0.5, 1.0), seeds=20, budget=15_000):
    """Same chain, plus a +0.01 crumb at the left wall. Now exploration has a rival."""
    found, avg_ret = [], []
    for eps in eps_list:
        hits, rets = 0, []
        for sd in range(seeds):
            ep, returns, _ = q_learning(Chain(n, distractor=0.01), budget, eps, seed=7 * sd + 3)
            hits += int(ep is not None)
            rets.append(returns[-2000:].mean())
        found.append(hits / seeds)
        avg_ret.append(float(np.mean(rets)))
        print(f"  distractor eps={eps}: goal found by {hits}/{seeds} seeds, "
              f"final return {avg_ret[-1]:.4f}", flush=True)
    return list(eps_list), found, avg_ret


def plot_distractor(eps_list, found, avg_ret, path):
    fig, ax = ps.new_axes(7.4, 4.2)
    x = np.arange(len(eps_list))
    ax.bar(x - 0.2, found, width=0.38, color=ps.SERIES[0],
           label="seeds that ever FOUND the +1 goal")
    ax.bar(x + 0.2, avg_ret, width=0.38, color=ps.SERIES[3],
           label="final return (1.0 = solving every episode)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(e) for e in eps_list])
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ps.finish(fig, ax, "Small epsilon cannot find the goal. Large epsilon cannot use it.",
              "epsilon", "fraction of seeds / return per episode", path)


# ---------------------------------------------------------------- experiment 3
def after_discovery(n=8, eps_list=(0.1, 0.5, 1.0), seeds=15, budget=10_000):
    """Once the reward IS found, how fast does the policy lock on? (Learning is easy.)"""
    curves = {}
    for eps in eps_list:
        runs = []
        for sd in range(seeds):
            _, returns, _ = q_learning(Chain(n), budget, eps, seed=11 * sd + 1)
            runs.append(returns)
        curves[eps] = np.mean(np.stack(runs), axis=0)
        print(f"  n={n} eps={eps}: mean return over last 1000 eps "
              f"{curves[eps][-1000:].mean():.3f}", flush=True)
    return curves


def plot_after(curves, path, window=200):
    fig, ax = ps.new_axes(7.4, 4.2)
    for i, (eps, c) in enumerate(curves.items()):
        smooth = np.convolve(c, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(smooth)), smooth, color=ps.SERIES[i], lw=2.0, label=f"epsilon = {eps}")
    ax.set_xscale("log")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ps.finish(fig, ax, "The learning was never the problem — the finding was",
              "episode (log scale)", f"success rate (mean of 15 seeds, {window}-ep window)", path)


def main():
    OUT.mkdir(exist_ok=True)
    lengths = list(range(4, 15))
    eps_list = [0.1, 0.3, 1.0]

    print("[1/3] episodes until first reward, vs chain length")
    med = discovery_vs_length(lengths, eps_list)
    plot_discovery(lengths, med, OUT / "discovery_vs_length.png")

    print("[2/3] the distractor trap (n=6, a 0.01 crumb at the left wall)")
    e, f, r = distractor_trap()
    plot_distractor(e, f, r, OUT / "distractor_trap.png")

    print("[3/3] learning speed AFTER the first reward (n=8)")
    curves = after_discovery()
    plot_after(curves, OUT / "after_discovery.png")

    # The headline number, printed so the README can quote it.
    ratio = med[0.1][-1] / med[0.1][-2] if len(lengths) > 1 else float("nan")
    print(f"\nmedian first-reward episode grew {ratio:.2f}x for the last +1 link "
          f"(theory says 2.00x)")


if __name__ == "__main__":
    main()
