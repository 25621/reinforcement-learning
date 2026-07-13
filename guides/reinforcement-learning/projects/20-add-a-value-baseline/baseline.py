"""Subtract a baseline from the return. Verify the variance drops and the bias does not appear.

The claim the textbook makes is precise and double-barrelled:

    for ANY function b(s) that does not depend on the action,
        E[ ∇log π(a|s) · (G_t − b(s)) ]  =  E[ ∇log π(a|s) · G_t ]        (same gradient)
    and for a well-chosen b,
        Var[ ∇log π(a|s) · (G_t − b(s)) ]  <  Var[ ∇log π(a|s) · G_t ]    (less noise)

Both halves are checkable, and this project checks both rather than taking them
on faith. Three weightings, one task:

    none      G_t                     project 19's reward-to-go
    constant  G_t − mean(G)           the cheapest baseline there is
    learned   G_t − V_φ(s_t)          a critic trained alongside the actor

Experiments:
    train      all three, five seeds — does the baseline actually learn faster?
    variance   the estimator's noise-to-signal, all three, on one frozen policy
    bias       do the three estimators point the same way? (they must)

Runtime: ~5 min on 12 CPU cores (`python3 baseline.py all`).
"""

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19-reinforce-on-cartpole"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import plot_style as ps
from pg_lib import ActorCritic, collect_episode, set_seed

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

ENV_ID = "CartPole-v1"
# Deliberately identical to project 19's tuned reward-to-go run, so that the
# "none" variant below IS project 19's agent and the only thing this project
# changes is the baseline term. γ = 1 for the same reason as there: all three
# weightings must be unbiased estimators of the SAME gradient for a variance
# comparison between them to mean anything.
GAMMA = 1.0
LR = 1e-3
LR_CRITIC = 1e-2          # the critic may move faster; it is only a regression
EPISODES_PER_UPDATE = 5
N_UPDATES = 200
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ("none", "constant", "learned")
CHECKPOINTS = [0, 40, 150]


def make_agent(seed):
    set_seed(seed)
    return ActorCritic(4, 2, continuous=False, critic=True)


def weights_for(episodes, variant, agent):
    """The advantage estimate each action's log-prob gets multiplied by.

    This function IS the difference between the three algorithms. Everything
    else — the environment, the network, the optimizer, the number of episodes —
    is held fixed.
    """
    out = []
    all_returns = np.concatenate([ep["returns"] for ep in episodes])
    const = float(all_returns.mean())
    for ep in episodes:
        G = torch.as_tensor(ep["returns"], dtype=torch.float32)
        if variant == "none":
            w = G
        elif variant == "constant":
            w = G - const
        elif variant == "learned":
            with torch.no_grad():                       # the critic is a baseline, not a
                v = agent.value(ep["obs"])              # path for the policy gradient to
            w = G - v                                   # flow through — hence no_grad
        else:
            raise ValueError(variant)
        out.append(w)
    return out


def policy_loss(episodes, variant, agent):
    ws = weights_for(episodes, variant, agent)
    return torch.stack([-(ep["logps"] * w).sum() for ep, w in zip(episodes, ws)]).mean()


def critic_loss(episodes, agent):
    """Regress V_φ(s) onto the observed reward-to-go. Plain supervised learning.

    Note what the critic is NOT doing: it is not bootstrapping off itself (that
    is TD, and arrives in project 21), and it is not being asked to help choose
    actions. Its only job is to predict the average return from a state so the
    actor can be told how much *better than average* its action turned out.
    """
    preds = torch.cat([agent.value(ep["obs"]) for ep in episodes])
    targets = torch.cat([torch.as_tensor(ep["returns"], dtype=torch.float32) for ep in episodes])
    return F.mse_loss(preds, targets)


def train_one(args):
    seed, variant = args
    torch.set_num_threads(1)
    env = gym.make(ENV_ID)
    agent = make_agent(seed)
    opt_actor = torch.optim.Adam(agent.actor.parameters(), lr=LR)
    opt_critic = torch.optim.Adam(agent.critic.parameters(), lr=LR_CRITIC)

    curve, evar = [], []
    for update in range(N_UPDATES):
        set_seed(seed * 10_000 + update)
        episodes = [collect_episode(env, agent, GAMMA) for _ in range(EPISODES_PER_UPDATE)]

        opt_actor.zero_grad()
        policy_loss(episodes, variant, agent).backward()
        opt_actor.step()

        if variant == "learned":
            for _ in range(5):                          # a few passes: the critic must keep
                opt_critic.zero_grad()                  # up with a policy that is moving
                critic_loss(episodes, agent).backward()
                opt_critic.step()
            with torch.no_grad():
                ev = explained_variance(agent, episodes)
        else:
            ev = np.nan
        curve.append(np.mean([e["ep_return"] for e in episodes]))
        evar.append(ev)
    env.close()
    return variant, seed, np.asarray(curve), np.asarray(evar)


def explained_variance(agent, episodes):
    """1 − Var[G − V] / Var[G]. The critic's report card.

    1.0 means the critic predicts the return perfectly; 0.0 means it does no
    better than guessing the mean; negative means it is actively worse than the
    mean, which happens early and is the tell-tale of a critic learning rate
    that is too low to keep up with the policy.
    """
    preds = torch.cat([agent.value(ep["obs"]) for ep in episodes]).numpy()
    targets = np.concatenate([ep["returns"] for ep in episodes])
    var = targets.var()
    return float(1 - (targets - preds).var() / var) if var > 0 else np.nan


def cmd_train():
    jobs = [(s, v) for v in VARIANTS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(train_one, jobs))
    curves = {v: np.stack([c for vv, s, c, _ in res if vv == v]) for v in VARIANTS}
    ev = np.stack([e for vv, s, _, e in res if vv == "learned"])
    np.savez(OUT / "train.npz", ev=ev, **curves)
    for v in VARIANTS:
        c = curves[v]
        final = c[:, -25:].mean(axis=1)
        # "Updates to reach 400" is the sample-efficiency number that matters:
        # the wall-clock cost of an update is identical across the three.
        reach = [int(np.argmax(np.convolve(row, np.ones(10) / 10, "same") > 400))
                 if (np.convolve(row, np.ones(10) / 10, "same") > 400).any() else -1 for row in c]
        print(f"{v:9s} final {final.mean():6.1f} ± {final.std():5.1f} | "
              f"updates to reach 400: {reach}")


# --------------------------------------------------------------------------
# Does the variance actually drop, and does the gradient stay put?
# --------------------------------------------------------------------------

def flat_grad(agent, loss):
    agent.zero_grad(set_to_none=True)
    # retain_graph: all three weightings are scored against the SAME episode, so
    # the first backward must not free the graph the other two still need.
    loss.backward(retain_graph=True)
    return torch.cat([p.grad.reshape(-1) if p.grad is not None else torch.zeros(p.numel())
                      for p in agent.actor.parameters()]).clone()


def train_to(agent, env, n_updates, seed):
    """Warm both actor and critic up to a common checkpoint, so all three
    weightings are measured on the SAME policy and the SAME critic."""
    opt_a = torch.optim.Adam(agent.actor.parameters(), lr=LR)
    opt_c = torch.optim.Adam(agent.critic.parameters(), lr=LR_CRITIC)
    for u in range(n_updates):
        set_seed(seed * 7919 + u)
        eps = [collect_episode(env, agent, GAMMA) for _ in range(EPISODES_PER_UPDATE)]
        opt_a.zero_grad()
        policy_loss(eps, "learned", agent).backward()
        opt_a.step()
        for _ in range(5):
            opt_c.zero_grad()
            critic_loss(eps, agent).backward()
            opt_c.step()
    return agent


def pool_job(n_updates):
    """A pool of single-episode gradients under all three weightings, on one frozen policy.

    Crucially the SAME episodes produce all three gradients: they differ only in
    the weight applied to each log-prob, so any difference in scatter is caused
    by the baseline and not by which episodes happened to be drawn.
    """
    torch.set_num_threads(2)
    env = gym.make(ENV_ID)
    agent = make_agent(0)
    train_to(agent, env, n_updates, seed=0)

    grads = {v: [] for v in VARIANTS}
    lens = []
    for i in range(200):
        set_seed(70_000 + i)
        ep = [collect_episode(env, agent, GAMMA)]
        lens.append(ep[0]["length"])
        for v in VARIANTS:
            grads[v].append(flat_grad(agent, policy_loss(ep, v, agent)))
    env.close()

    stats = {}
    for v in VARIANTS:
        G = torch.stack(grads[v])
        g = G.mean(0)
        sq = float(((G - g) ** 2).sum(dim=1).mean())
        # Absolute variance is the headline: at γ = 1 all three weightings have the
        # SAME mean gradient, so a difference in scatter is a difference in noise
        # and nothing else. (Relative variance divides by ‖g‖², which explodes near
        # a converged policy for reasons that say nothing about the estimator.)
        stats[v] = dict(var=sq, rel_var=sq / float(g @ g),
                        snr=float(g.norm()) / np.sqrt(sq), mean_grad=g.numpy())

    bias = {v: bias_test(torch.stack(grads["none"]) - torch.stack(grads[v]))
            for v in ("constant", "learned")}
    return n_updates, float(np.mean(lens)), stats, bias


def bias_test(D, n_boot=2000, seed=0):
    """Is the mean of these difference vectors distinguishable from zero?

    The tempting test — "do the two mean gradients point the same way?" — quietly
    fails exactly where it matters most. Near a converged policy the true gradient
    is almost zero, so a few hundred episodes cannot pin down its DIRECTION at
    all, and the cosine between two noisy estimates of a near-zero vector is
    meaningless. (Measured here, it came out at −0.9, which looks like a scandal
    and is really just a measurement taken where no measurement is possible.)

    Test the DIFFERENCE instead. Per episode, `g_none − g_baseline` equals
    `b · Σ_t ∇log π(a_t|s_t)`, and the theorem says its expectation is exactly
    zero — that IS the unbiasedness claim, stated as something measurable. So ask:
    is the observed ‖D̄‖² larger than what a genuinely zero-mean quantity would
    produce by chance?

    The null distribution is obtained by bootstrap rather than from a formula.
    Gradient components are strongly correlated with one another, so the effective
    number of degrees of freedom is far below the parameter count and every
    closed-form χ² approximation is wrong by a factor nobody can predict. Resampling
    the centred differences sidesteps that entirely: it simulates a world where the
    mean really is zero and the correlation structure is whatever the data says.
    """
    rng = np.random.default_rng(seed)
    n = len(D)
    D_bar = D.mean(0)
    observed = float(D_bar @ D_bar)
    centred = (D - D_bar).numpy()                    # now has mean exactly zero
    null = np.empty(n_boot)
    for b in range(n_boot):
        m = centred[rng.integers(0, n, n)].mean(0)
        null[b] = float(m @ m)
    p = (np.sum(null >= observed) + 1) / (n_boot + 1)
    return dict(observed=observed, null_median=float(np.median(null)), p=float(p),
                ratio=observed / float(np.median(null)))


def cmd_variance():
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = list(pool.map(pool_job, CHECKPOINTS))

    labels = ["untrained", "learning", "competent"]
    var = {v: np.array([r[2][v]["var"] for r in rows]) for v in VARIANTS}
    rel = {v: np.array([r[2][v]["rel_var"] for r in rows]) for v in VARIANTS}
    snr = {v: np.array([r[2][v]["snr"] for r in rows]) for v in VARIANTS}

    cos, bias = {}, {}
    for i, r in enumerate(rows):
        g_none = r[2]["none"]["mean_grad"]
        for v in ("constant", "learned"):
            gv = r[2][v]["mean_grad"]
            c = float(np.dot(g_none, gv) / (np.linalg.norm(g_none) * np.linalg.norm(gv)))
            cos.setdefault(v, []).append(c)
            bias.setdefault(v, []).append(r[3][v])   # dict: observed / null_median / p / ratio

    np.savez(OUT / "variance.npz", labels=labels,
             ep_len=[r[1] for r in rows],
             **{f"var_{v}": var[v] for v in VARIANTS},
             **{f"rel_{v}": rel[v] for v in VARIANTS},
             **{f"snr_{v}": snr[v] for v in VARIANTS},
             **{f"cos_{v}": np.array(cos[v]) for v in cos},
             **{f"bias_{v}": np.array([b["ratio"] for b in bias[v]]) for v in bias},
             **{f"p_{v}": np.array([b["p"] for b in bias[v]]) for v in bias})

    print(f"\n{'checkpoint':11s} {'ep_len':>6s} | " +
          " ".join(f"{'var ' + v:>11s}" for v in VARIANTS) +
          " | " + " ".join(f"{v + ' vs none':>16s}" for v in VARIANTS[1:]))
    for i, lab in enumerate(labels):
        print(f"{lab:11s} {rows[i][1]:6.0f} | " +
              " ".join(f"{var[v][i]:11.3g}" for v in VARIANTS) +
              " | " + " ".join(f"{var['none'][i] / var[v][i]:14.1f}x" for v in VARIANTS[1:]))

    print("\nUnbiasedness (bootstrap): ‖mean difference from the no-baseline estimator‖²,")
    print("as a multiple of what a genuinely zero-mean quantity gives. p is the chance")
    print("of seeing a difference this big if the baseline really is unbiased.")
    print(f"{'checkpoint':11s} | {'constant: ratio':>15s} {'p':>7s} | {'learned: ratio':>15s} {'p':>7s}")
    for i, lab in enumerate(labels):
        bc, bl = bias["constant"][i], bias["learned"][i]
        print(f"{lab:11s} | {bc['ratio']:15.2f} {bc['p']:7.3f} | {bl['ratio']:15.2f} {bl['p']:7.3f}")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def smooth(x, k=15):
    return np.convolve(x, np.ones(k) / k, mode="valid")


def cmd_plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT / "train.npz")
    fig, ax = ps.new_axes(7.6, 4.4)
    for i, v in enumerate(VARIANTS):
        c = np.stack([smooth(row) for row in d[v]])
        x = np.arange(c.shape[1])
        ax.fill_between(x, c.min(0), c.max(0), color=ps.SERIES[i], alpha=0.13, linewidth=0)
        ax.plot(x, c.mean(0), color=ps.SERIES[i], linewidth=2.2,
                label={"none": "no baseline (project 19)", "constant": "constant baseline",
                       "learned": "learned V(s) baseline"}[v])
    ax.axhline(475, color=ps.INK_MUTED, linestyle=":", linewidth=1)
    ax.text(2, 483, "solved (475)", color=ps.INK_MUTED, fontsize=8)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ps.finish(fig, ax, "A baseline is not a tweak — it is the difference between learning and thrashing",
              f"update ({EPISODES_PER_UPDATE} episodes each)", "episode return",
              OUT / "learning_curves.png")

    v = np.load(OUT / "variance.npz")
    labels = v["labels"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in (ax1, ax2):
        ps.style_axes(a)

    x = np.arange(len(labels))
    for i, name in enumerate(VARIANTS):
        vals = v[f"var_{name}"]
        ax1.bar(x + (i - 1) * 0.27, vals, width=0.26, color=ps.SERIES[i],
                label={"none": "no baseline", "constant": "constant", "learned": "learned V(s)"}[name])
        for xi, val in zip(x, vals):
            ax1.text(xi + (i - 1) * 0.27, val * 1.5, f"{val:.0e}".replace("e+0", "e"),
                     ha="center", color=ps.INK, fontsize=7.5)
    ax1.set_yscale("log")
    ax1.set_ylim(top=max(v[f"var_none"]) * 40)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{l}\n({n:.0f} steps)" for l, n in zip(labels, v["ep_len"])])
    ax1.set_title("Variance of one gradient estimate  E‖ĝ−g‖²", color=ps.INK, fontsize=11, loc="left")
    ax1.set_ylabel("variance (log)", color=ps.INK_SECONDARY, fontsize=9)
    ax1.legend(frameon=False, fontsize=8, loc="upper left")

    for i, name in enumerate(("constant", "learned")):
        ax2.plot(x, v[f"bias_{name}"], "o-", color=ps.SERIES[i + 1], linewidth=2.0,
                 label={"constant": "constant baseline", "learned": "learned V(s) baseline"}[name])
        for xi, (val, p) in enumerate(zip(v[f"bias_{name}"], v[f"p_{name}"])):
            ax2.annotate(f"p={p:.2f}", (xi, val), textcoords="offset points",
                         xytext=(0, 9 if name == "constant" else -16), ha="center",
                         fontsize=8, color=ps.SERIES[i + 1])
    ax2.axhline(1.0, color=ps.INK, linestyle="--", linewidth=1.2)
    ax2.text(0.02, 1.15, "a truly zero-mean difference", color=ps.INK, fontsize=8)
    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("...but only ONE of them leaves the gradient alone", color=ps.INK,
                  fontsize=11, loc="left")
    ax2.set_ylabel("‖mean difference‖² ÷ its own sampling noise", color=ps.INK_SECONDARY, fontsize=9)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "variance_and_bias.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'variance_and_bias.png'}")

    ev = d["ev"]
    fig, ax = ps.new_axes(7.0, 3.4)
    sm = np.stack([smooth(row) for row in ev])
    x = np.arange(sm.shape[1])
    ax.fill_between(x, sm.min(0), sm.max(0), color=ps.SERIES[2], alpha=0.15, linewidth=0)
    ax.plot(x, sm.mean(0), color=ps.SERIES[2], linewidth=2.0)
    ax.axhline(0, color=ps.INK_MUTED, linestyle=":", linewidth=1)
    ax.set_ylim(-0.6, 1.05)
    ps.finish(fig, ax, "The critic's report card: fraction of return variance it explains",
              "update", "explained variance", OUT / "explained_variance.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("train", "all"):
        cmd_train()
    if cmd in ("variance", "all"):
        cmd_variance()
    if cmd in ("plot", "all"):
        cmd_plot()
