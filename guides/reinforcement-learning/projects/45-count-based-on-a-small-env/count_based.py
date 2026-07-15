r"""Project 45 — count-based exploration: pay the agent to go where it has not been.

The idea is one line on top of project 44's Q-learning:

    reward the learner sees  =  r  +  beta / sqrt(N(s'))
                                      ^^^^^^^^^^^^^^^^^
                                      the exploration bonus

N(s') counts how many times the agent has ARRIVED in state s'. The first arrival
pays beta/1, the hundredth pays beta/10: a promise that fades exactly as fast as
the state stops being a stranger.

That line alone does not work. It fails completely — 0 seeds out of 8, at every
chain length — and the reason is the most useful thing in this project, so the
code below runs the broken version on purpose (`q_init=0`) alongside the fixed
one. See the README, or read `q_learning`'s comments.

    python3 count_based.py     # ~6 min, three figures
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
sys.path.insert(0, str(HERE.parent / "44-epsilon-greedy-on-a-chain"))
import plot_style as ps  # noqa: E402
from chain import Chain  # noqa: E402   the exact same environment project 44 used

OUT = HERE / "outputs"


def q_learning(env, n_states, n_actions, episodes, eps, beta, q_init, seed,
               alpha=0.5, gamma=0.99, stop_on_first_reward=False, track_visits=False):
    """One learner, three knobs. `eps` = random jitter, `beta` = count bonus,
    `q_init` = how good an *untried* action is assumed to be (the optimism).

    Two details in here are the whole project.

    (1) The bonus is paid for WHERE YOU LAND — `N(s')` — not for the (state, action)
        pair you used. With an `N(s, a)` bonus, bumping into a wall is both a rarely
        tried action AND a self-loop, so its value bootstraps off itself
        (`Q = bonus + gamma * Q`, which settles at 100x the bonus when gamma = 0.99).
        The agent then stands at the wall collecting a bonus for "exploring" the one
        square it never leaves. Measured: 0 of 8 seeds ever found the goal.

    (2) `q_init` must be OPTIMISTIC, and this is the part that looks redundant.
        Surely the bonus already handles "go somewhere new"? It does not, and cannot:
        **a bonus is a reward you can only collect after you arrive.** An action never
        tried has been paid nothing, so its Q stays at `q_init`. Leave `q_init = 0` and
        that untried action scores 0, while the familiar corridor the agent has been
        pacing has a small positive value built from the bonuses it already collected —
        so the greedy policy prefers the corridor, forever, and the bonus for the
        unexplored states is never claimed because nobody goes to claim it.

        Optimism fills exactly that gap: it is a claim about places you have NEVER
        been ("assume the unknown is wonderful"), whereas the bonus is a payment for
        places you HAVE been ("thanks for checking, here is something for your
        trouble"). The bonus fades what optimism promised, at the rate the evidence
        arrives. You need both, and the experiments below measure what each is worth.
    """
    rng = np.random.default_rng(seed)
    Q = np.full((n_states, n_actions), float(q_init))
    N = np.zeros(n_states)
    visits = np.zeros(n_states)
    first_reward_ep = None
    returns = np.zeros(episodes)

    for ep in range(episodes):
        s = env.reset()
        done, total = False, 0.0
        while not done:
            if eps > 0 and rng.random() < eps:
                a = int(rng.integers(n_actions))
            else:
                q = Q[s]
                a = int(rng.choice(np.flatnonzero(q == q.max())))   # fair tie-break
            s2, r, done = env.step(a)
            N[s2] += 1
            if track_visits:
                visits[s2] += 1
            bonus = beta / np.sqrt(N[s2]) if beta > 0 else 0.0
            target = r + bonus + (0.0 if done and r > 0 else gamma * Q[s2].max())
            Q[s, a] += alpha * (target - Q[s, a])
            total += r                                  # the REAL reward; the bonus is not counted
            s = s2
        returns[ep] = total
        if first_reward_ep is None and total >= 1.0:
            first_reward_ep = ep + 1
            if stop_on_first_reward:
                return first_reward_ep, returns[: ep + 1], visits
    return first_reward_ep, returns, visits


# ---------------------------------------------------------------- experiment 1
ARMS = {
    # label                                    eps  beta  q_init
    "epsilon-greedy (project 44)":            (0.1, 0.00, 0.0),
    "count bonus alone":                      (0.0, 0.05, 0.0),
    "count bonus + optimistic start":         (0.0, 0.05, 1.0),
}


def chain_sweep(lengths, seeds=8, budget=20_000):
    out = {}
    for label, (eps, beta, qi) in ARMS.items():
        med = []
        for n in lengths:
            firsts = []
            for sd in range(seeds):
                ep, _, _ = q_learning(Chain(n), n, 2, budget, eps, beta, qi,
                                      seed=1000 * sd + n, stop_on_first_reward=True)
                firsts.append(budget if ep is None else ep)   # censored at the budget
            med.append(np.median(firsts))
            print(f"  {label:32s} n={n:2d}  first reward @ episode {med[-1]:.0f}", flush=True)
        out[label] = np.array(med)
    return out


def plot_chain(lengths, curves, budget, path):
    fig, ax = ps.new_axes(7.8, 4.6)
    lengths = np.asarray(lengths)
    for i, (label, v) in enumerate(curves.items()):
        ax.plot(lengths, v, color=ps.SERIES[i], lw=2.2, marker="o", ms=4.5, label=label)
    ax.axhline(budget, color=ps.INK_MUTED, ls=":", lw=1.4)
    ax.annotate("never found it (hit the 20,000-episode budget)", (lengths[0], budget),
                xytext=(0, -14), textcoords="offset points", color=ps.INK_MUTED, fontsize=8.5)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, loc="center left")
    ps.finish(fig, ax, "The bonus is useless alone, and transformative with optimism",
              "chain length n (states)", "episodes until first reward (median of 8 seeds)", path)


# ---------------------------------------------------------------- experiment 2
def ablation(n=14, seeds=10, budget=20_000):
    """Which half does the work: the counts, or the optimism?"""
    grid = {
        "nothing (greedy)":        (0.0, 0.00, 0.0),
        "epsilon-greedy":          (0.1, 0.00, 0.0),
        "counts only":             (0.0, 0.05, 0.0),
        "optimism only":           (0.0, 0.00, 1.0),
        "counts + optimism":       (0.0, 0.05, 1.0),
    }
    res = {}
    for label, (eps, beta, qi) in grid.items():
        firsts = [q_learning(Chain(n), n, 2, budget, eps, beta, qi, seed=17 * sd + 5,
                             stop_on_first_reward=True)[0] for sd in range(seeds)]
        found = sum(f is not None for f in firsts)
        med = np.median([budget if f is None else f for f in firsts])
        res[label] = (found / seeds, med)
        print(f"  {label:20s} found by {found}/{seeds} seeds, median episode {med:.0f}", flush=True)
    return res


def plot_ablation(res, budget, path):
    fig, ax = ps.new_axes(7.6, 4.2)
    labels = list(res)
    y = [res[k][1] for k in labels]
    colors = [ps.INK_MUTED, ps.SERIES[0], ps.SERIES[2], ps.SERIES[3], ps.SERIES[1]]
    bars = ax.bar(np.arange(len(labels)), y, color=colors, width=0.62)
    for b, k in zip(bars, labels):
        txt = "never" if res[k][1] >= budget else f"{res[k][1]:.0f}"
        ax.annotate(txt, (b.get_x() + b.get_width() / 2, b.get_height()), xytext=(0, 3),
                    textcoords="offset points", ha="center", color=ps.INK_SECONDARY, fontsize=9)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yscale("log")
    ps.finish(fig, ax, "On a 14-state chain: what each ingredient is actually worth",
              "", "episodes until first reward (median, log scale)", path)


# ---------------------------------------------------------------- experiment 3
class OpenRoom:
    """11x11 room, start in one corner, +1 in the opposite one, 40 steps per episode.

    The room is sized so that reaching the goal is *possible* but not casual: the two
    corners are 20 moves apart and an episode lasts 40 steps, so a wanderer has to spend
    almost every step going the right way.
    """

    size = 11
    horizon = 40

    def __init__(self):
        self.n = self.size * self.size

    def reset(self):
        self.p = [0, 0]
        self.t = 0
        return 0

    def step(self, a):
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][a]
        self.p = [int(np.clip(self.p[0] + dx, 0, self.size - 1)),
                  int(np.clip(self.p[1] + dy, 0, self.size - 1))]
        self.t += 1
        s = self.p[1] * self.size + self.p[0]
        goal = self.p == [self.size - 1, self.size - 1]
        return s, (1.0 if goal else 0.0), goal or self.t >= self.horizon


def coverage(episodes=400, seed=0):
    maps, stats = {}, {}
    for label, (eps, beta, qi) in [("epsilon-greedy", (0.1, 0.0, 0.0)),
                                   ("count bonus + optimism", (0.0, 0.05, 1.0))]:
        env = OpenRoom()
        first, returns, visits = q_learning(env, env.n, 4, episodes, eps, beta, qi,
                                            seed=seed, track_visits=True)
        m = visits.reshape(env.size, env.size)
        # How much of its life did the agent spend within 2 cells of where it started?
        home = m[:3, :3].sum() / m.sum()
        maps[label] = m
        stats[label] = (int((visits > 0).sum()), first, returns[-100:].mean(), home)
        print(f"  {label:24s} saw {stats[label][0]}/{env.n} cells, {home*100:.0f}% of steps "
              f"spent in the home corner, first reward @ episode {first}, "
              f"final success {stats[label][2]:.2f}", flush=True)
    return maps, stats


def plot_coverage(maps, stats, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    vmax = max(m.max() for m in maps.values())
    n = OpenRoom.size
    for ax, (label, m) in zip(axes, maps.items()):
        im = ax.imshow(m, cmap="magma", vmin=0, vmax=vmax)
        seen, first, succ, home = stats[label]
        ax.set_title(f"{label}\n{home*100:.0f}% of all steps in the home corner · "
                     f"final success {succ:.2f}", color=ps.INK, fontsize=10.5)
        ax.set_xticks([]), ax.set_yticks([])
        ax.text(0, 0, "S", ha="center", va="center", color="w", fontsize=11, fontweight="bold")
        ax.text(n - 1, n - 1, "G", ha="center", va="center", color="w", fontsize=11,
                fontweight="bold")
        fig.colorbar(im, ax=ax, shrink=0.82, label="visits")
    fig.suptitle(f"Where 400 episodes were actually spent, in an {n}x{n} room",
                 color=ps.INK, fontsize=12.5, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    budget = 20_000
    lengths = list(range(4, 17, 2))

    print("[1/3] chain sweep on project 44's chain: does the bonus rescue it?")
    curves = chain_sweep(lengths, budget=budget)
    plot_chain(lengths, curves, budget, OUT / "chain_sweep.png")

    print("[2/3] ablation on a 14-state chain: counts, optimism, or both?")
    res = ablation(budget=budget)
    plot_ablation(res, budget, OUT / "ablation.png")

    print("[3/3] visit heat-maps in a 7x7 open room")
    maps, stats = coverage()
    plot_coverage(maps, stats, OUT / "coverage.png")

    e = curves["epsilon-greedy (project 44)"]
    c = curves["count bonus + optimistic start"]
    print(f"\nat n={lengths[-1]}: epsilon-greedy {e[-1]:.0f} episodes, "
          f"counts+optimism {c[-1]:.0f} ({e[-1] / max(c[-1], 1):.0f}x fewer)")


if __name__ == "__main__":
    main()
