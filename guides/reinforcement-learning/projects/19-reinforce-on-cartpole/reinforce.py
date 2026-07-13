"""REINFORCE on CartPole, and a direct measurement of the variance that dooms it.

Three questions, three experiments:

  train     Does the plainest policy gradient work at all? (Yes — badly.)
            Two weightings, five seeds each: the whole episode's return pinned on
            every action, versus the reward-to-go. The seed-0 reward-to-go run
            saves policy checkpoints along the way; the next two experiments
            measure THOSE policies, so the numbers below describe the agent whose
            learning curve you are looking at rather than some re-trained cousin.

  variance  How noisy is the gradient estimator, in numbers rather than adjectives?
            Freeze a policy, draw many independent single-episode gradient
            estimates, and measure how far they scatter around their own mean.
            Done at three checkpoints, because the answer changes as the agent
            improves — and it changes in the wrong direction.

  batch     The obvious fix (average more episodes) works at the obvious rate,
            1/n. This is what makes REINFORCE expensive rather than broken.

A note on γ, which is set to 1.0 here and 0.99 everywhere else in the phase.
CartPole is episodic and hard-capped at 500 steps, so the undiscounted return is
finite and no discount is needed to make the math behave. That matters for THIS
project specifically: at γ = 1 the two weightings below are unbiased estimators
of the *same* gradient, so comparing their variance is apples-to-apples. At γ < 1
they quietly target different vectors (the full-return weight carries an extra
γᵗ), and a variance comparison between them would be meaningless. From project 21
on, bootstrapping needs γ < 1 for the Bellman operator to contract, and the
weightings are all reward-to-go anyway, so the issue does not arise.

Runtime: ~8 min on 12 CPU cores (`python3 reinforce.py all`).
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import gymnasium as gym
import plot_style as ps
from pg_lib import ActorCritic, collect_episode, set_seed

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
CKPT = HERE / "checkpoints"
CKPT.mkdir(exist_ok=True)

ENV_ID = "CartPole-v1"
GAMMA = 1.0                  # see the module docstring — this is load-bearing
LR = 1e-3
EPISODES_PER_UPDATE = 5      # small enough that the noise is visible, big enough to learn
N_UPDATES = 200
SEEDS = [0, 1, 2, 3, 4]
WEIGHTINGS = ("full_return", "reward_to_go")
CHECKPOINTS = {0: "untrained", 40: "learning", 150: "competent"}


def make_agent(seed):
    set_seed(seed)                      # BEFORE the net exists, or its weights are unseeded
    return ActorCritic(4, 2, continuous=False, critic=False)


def policy_loss(episodes, weighting):
    """-(1/E) Σ_episodes Σ_t log π(a_t|s_t) · w_t, with w set by `weighting`.

    This is the policy gradient theorem and nothing else. `w_t` is the only thing
    that will differ between REINFORCE, the baseline version, A2C and PPO — the
    whole phase is an argument about what belongs in this one variable.
    """
    terms = []
    for ep in episodes:
        if weighting == "full_return":
            # Every action in the episode is credited with the WHOLE return,
            # including rewards collected BEFORE it was taken. Those rewards
            # cannot have been caused by this action. The term they contribute
            # has zero mean — it does not bias the gradient — but it is not
            # zero, and its scatter is charged to the estimator as pure noise.
            w = torch.full((ep["length"],), float(ep["returns"][0]))
        elif weighting == "reward_to_go":
            w = torch.as_tensor(ep["returns"], dtype=torch.float32)
        else:
            raise ValueError(weighting)
        terms.append(-(ep["logps"] * w).sum())
    return torch.stack(terms).mean()


def train_one(args):
    seed, weighting = args
    torch.set_num_threads(1)            # 10 of these run at once; give each one core
    env = gym.make(ENV_ID)
    agent = make_agent(seed)
    optim = torch.optim.Adam(agent.parameters(), lr=LR)
    save_ckpts = (seed == 0 and weighting == "reward_to_go")

    curve, grad_norms = [], []
    for update in range(N_UPDATES):
        if save_ckpts and update in CHECKPOINTS:
            torch.save(agent.state_dict(), CKPT / f"{CHECKPOINTS[update]}.pt")
        set_seed(seed * 10_000 + update)
        episodes = [collect_episode(env, agent, GAMMA) for _ in range(EPISODES_PER_UPDATE)]
        loss = policy_loss(episodes, weighting)
        optim.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(agent.parameters(), 1e9)   # measure, don't clip
        optim.step()
        curve.append(np.mean([e["ep_return"] for e in episodes]))
        grad_norms.append(float(gn))
    env.close()
    return weighting, seed, np.asarray(curve), np.asarray(grad_norms)


def cmd_train():
    jobs = [(s, w) for w in WEIGHTINGS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(train_one, jobs))
    curves = {w: np.stack([c for ww, s, c, _ in results if ww == w]) for w in WEIGHTINGS}
    gnorms = {w: np.stack([g for ww, s, _, g in results if ww == w]) for w in WEIGHTINGS}
    np.savez(OUT / "train.npz",
             full_return=curves["full_return"], reward_to_go=curves["reward_to_go"],
             gn_full=gnorms["full_return"], gn_rtg=gnorms["reward_to_go"])
    for w in WEIGHTINGS:
        final = curves[w][:, -20:].mean(axis=1)
        print(f"{w:14s} final-20-update return per seed: "
              f"{np.array2string(final, precision=0)}  mean {final.mean():.0f} ± {final.std():.0f}")


# --------------------------------------------------------------------------
# The variance measurement — on the checkpoints saved during training
# --------------------------------------------------------------------------

def flat_grad(agent, loss):
    agent.zero_grad(set_to_none=True)
    # retain_graph: both weightings are scored against the SAME episode, so the
    # first backward must not free the graph the second one needs.
    loss.backward(retain_graph=True)
    return torch.cat([p.grad.reshape(-1) for p in agent.parameters()]).clone()


def load_ckpt(name):
    agent = make_agent(0)
    agent.load_state_dict(torch.load(CKPT / f"{name}.pt"))
    return agent


def gradient_pool(agent, env, n, seed0):
    """n single-episode gradients under BOTH weightings, from the SAME episodes.

    Same episodes for both is the point: the two estimators then differ only in
    the weight applied to each log-prob, so any difference in scatter is caused
    by the weighting and not by which episodes happened to be drawn.
    """
    grads = {w: [] for w in WEIGHTINGS}
    lens = []
    for i in range(n):
        set_seed(seed0 + i)
        ep = [collect_episode(env, agent, GAMMA)]
        lens.append(ep[0]["length"])
        for w in WEIGHTINGS:
            grads[w].append(flat_grad(agent, policy_loss(ep, w)))
    return {w: torch.stack(g) for w, g in grads.items()}, float(np.mean(lens))


def scatter_stats(G):
    """Three numbers about one estimator, and it matters which one you quote.

    variance  = E‖ĝ − g‖²             the raw scatter of the estimates
    SNR       = ‖g‖ / sqrt(variance)  how much signal survives the noise
    rel. var  = variance / ‖g‖²       the same thing, squared and inverted

    The raw variance is the quantity the reward-to-go argument is about, and it
    is the one to compare across estimators: at γ = 1 both estimators have the
    SAME mean g, so a difference in variance is a difference in noise and nothing
    else. The *relative* variance divides by ‖g‖², which is fine while the policy
    is improving but becomes treacherous once it is nearly optimal — there is
    almost nothing left to learn, the true gradient goes to zero, and dividing by
    it makes the ratio explode for reasons that have nothing to do with the
    estimator. Both are reported below; the READMEs quote the one that means
    something in context.
    """
    g = G.mean(0)
    var = float(((G - g) ** 2).sum(dim=1).mean())
    return var, float(g.norm()) / np.sqrt(var), var / float(g @ g), g


def variance_job(name):
    torch.set_num_threads(2)
    env = gym.make(ENV_ID)
    agent = load_ckpt(name)
    G, mean_len = gradient_pool(agent, env, n=200, seed0=5000)
    env.close()
    out = {w: scatter_stats(G[w])[:3] for w in WEIGHTINGS}
    return name, mean_len, out


def cmd_variance():
    names = [CHECKPOINTS[k] for k in sorted(CHECKPOINTS)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(variance_job, names))
    np.savez(OUT / "variance.npz",
             labels=[r[0] for r in rows], ep_len=[r[1] for r in rows],
             full=[r[2]["full_return"] for r in rows],
             rtg=[r[2]["reward_to_go"] for r in rows])
    print(f"{'checkpoint':12s} {'ep_len':>7s} | {'full: var':>11s} {'SNR':>6s} {'rel.var':>8s} | "
          f"{'r2go: var':>11s} {'SNR':>6s} {'rel.var':>8s} | var ratio")
    for name, mean_len, out in rows:
        f, t = out["full_return"], out["reward_to_go"]
        print(f"{name:12s} {mean_len:7.0f} | {f[0]:11.2f} {f[1]:6.3f} {f[2]:8.1f} | "
              f"{t[0]:11.2f} {t[1]:6.3f} {t[2]:8.1f} | {f[0] / t[0]:6.1f}x")


SIZES = [1, 2, 4, 8, 16, 32]
POOL = 384


def cmd_batch():
    """How fast does averaging more episodes buy the variance back?

    Drawing fresh episodes for every batch size would cost POOL·(1+2+…+32)
    episodes and it is not necessary: an n-episode estimate IS the average of n
    independent single-episode estimates, so one pool of single-episode gradients
    can be carved into DISJOINT groups of n — 384 estimates at n=1, 12 at n=32 —
    and every group is an honest independent draw of exactly the estimator under
    study. Same measurement, a twentieth of the compute.
    """
    torch.set_num_threads(6)
    env = gym.make(ENV_ID)
    agent = load_ckpt("learning")             # the mid-training policy from cmd_train
    G, _ = gradient_pool(agent, env, n=POOL, seed0=31337)
    env.close()

    res = {}
    for w in WEIGHTINGS:
        g_true = G[w].mean(0)                 # the best estimate of the true gradient we have
        var = []
        for n in SIZES:
            groups = G[w][: (POOL // n) * n].reshape(POOL // n, n, -1).mean(dim=1)
            var.append(float(((groups - g_true) ** 2).sum(dim=1).mean()))
        res[w] = np.asarray(var)
        print(f"{w:14s} variance by batch size {SIZES}: {np.array2string(res[w], precision=3)}")
    np.savez(OUT / "batch.npz", sizes=SIZES, **res)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def smooth(x, k=11):
    return np.convolve(x, np.ones(k) / k, mode="valid")


NAMES = {"full_return": "full-episode return", "reward_to_go": "reward-to-go"}


def cmd_plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT / "train.npz")
    fig, ax = ps.new_axes(7.6, 4.4)
    for i, key in enumerate(WEIGHTINGS):
        sm = np.stack([smooth(row) for row in d[key]])
        x = np.arange(sm.shape[1])
        for row in sm:
            ax.plot(x, row, color=ps.SERIES[i], alpha=0.20, linewidth=0.9)
        ax.plot(x, sm.mean(0), color=ps.SERIES[i], linewidth=2.4, label=NAMES[key])
    ax.axhline(475, color=ps.INK_MUTED, linestyle=":", linewidth=1)
    ax.text(2, 483, "solved (475)", color=ps.INK_MUTED, fontsize=8)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ps.finish(fig, ax, "REINFORCE on CartPole — 5 seeds each, faint lines are individual runs",
              f"update ({EPISODES_PER_UPDATE} episodes each)", "episode return",
              OUT / "learning_curves.png")

    fig, ax = ps.new_axes(7.6, 3.4)
    for i, key in enumerate(("gn_full", "gn_rtg")):
        sm = np.stack([smooth(row) for row in d[key]])
        ax.plot(np.arange(sm.shape[1]), sm.mean(0), color=ps.SERIES[i], linewidth=2.0,
                label=NAMES[WEIGHTINGS[i]])
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Gradient magnitude — pinning the whole return on every action inflates every step",
              "update", "‖∇θ‖ (log scale)", OUT / "gradient_norms.png")

    # Two panels, because the two questions have different right answers.
    # Left: the raw variance, which is what the reward-to-go argument claims.
    # Right: the signal-to-noise ratio, which is what the optimizer actually feels
    #        — and which collapses as the agent gets good, for BOTH estimators.
    v = np.load(OUT / "variance.npz")
    labels, full, rtg, lens = v["labels"], v["full"], v["rtg"], v["ep_len"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.3), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in (ax1, ax2):
        ps.style_axes(a)
    x = np.arange(len(labels))
    tick = [f"{l}\n({n:.0f}-step episodes)" for l, n in zip(labels, lens)]

    for i, (arr, key) in enumerate([(full, "full_return"), (rtg, "reward_to_go")]):
        ax1.bar(x + (i - 0.5) * 0.38, arr[:, 0], width=0.36, color=ps.SERIES[i], label=NAMES[key])
        for xi, val in zip(x, arr[:, 0]):
            ax1.text(xi + (i - 0.5) * 0.38, val * 1.6, f"{val:.0e}".replace("e+0", "e"),
                     ha="center", color=ps.INK, fontsize=8)
    # The ratio is the point — annotate it directly rather than making the reader divide.
    for xi, (a, b) in enumerate(zip(full[:, 0], rtg[:, 0])):
        ax1.text(xi, a * 5.5, f"{a / b:.1f}× less noise", ha="center", color=ps.SERIES[1],
                 fontsize=8.5, fontweight="bold")
    ax1.set_yscale("log")
    ax1.set_ylim(top=full[:, 0].max() * 30)
    ax1.set_xticks(x); ax1.set_xticklabels(tick, fontsize=8)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")
    ax1.set_title("Variance of one gradient estimate  E‖ĝ−g‖²", color=ps.INK, fontsize=11, loc="left")
    ax1.set_ylabel("variance (log)", color=ps.INK_SECONDARY, fontsize=9)

    for i, (arr, key) in enumerate([(full, "full_return"), (rtg, "reward_to_go")]):
        ax2.bar(x + (i - 0.5) * 0.38, arr[:, 1], width=0.36, color=ps.SERIES[i], label=NAMES[key])
        for xi, val in zip(x, arr[:, 1]):
            ax2.text(xi + (i - 0.5) * 0.38, val + 0.012, f"{val:.2f}", ha="center",
                     color=ps.INK, fontsize=8)
    ax2.set_xticks(x); ax2.set_xticklabels(tick, fontsize=8)
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_title("Signal-to-noise  ‖g‖ / √variance", color=ps.INK, fontsize=11, loc="left")
    ax2.set_ylabel("SNR of a single episode", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "gradient_variance.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'gradient_variance.png'}")

    b = np.load(OUT / "batch.npz")
    sizes = np.asarray(b["sizes"])
    fig, ax = ps.new_axes(7.0, 4.0)
    for i, key in enumerate(WEIGHTINGS):
        ax.plot(sizes, b[key], "o-", color=ps.SERIES[i], linewidth=2.0, label=NAMES[key])
    ref = b["full_return"][0] / sizes
    ax.plot(sizes, ref, "--", color=ps.INK_MUTED, linewidth=1.2, label="exact 1/n")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels(sizes)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Averaging more episodes buys the variance back at exactly the 1/n rate",
              "episodes per gradient estimate", "variance  E‖ĝ−g‖²  (log)",
              OUT / "batch_variance.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("train", "all"):
        cmd_train()
    if cmd in ("variance", "all"):
        cmd_variance()
    if cmd in ("batch", "all"):
        cmd_batch()
    if cmd in ("plot", "all"):
        cmd_plot()
