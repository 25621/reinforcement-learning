"""A2C on LunarLander: parallel environments, GAE, and one gradient step per rollout.

REINFORCE (19) and its baseline (20) had to wait for an episode to finish before
they could learn anything, because the Monte-Carlo return does not exist until
the episode does. A2C breaks that dependency with a bootstrap: after a fixed
`n_steps`, cut the rollout wherever it happens to be and let the critic estimate
what the rest of the episode would have been worth. Suddenly an update costs a
fixed, small amount of experience — and if the update is a fixed size, you can
gather it from many environments at once.

That is the whole algorithm. Three experiments test the two claims it rests on:

    train    8 parallel envs, GAE(0.98), n=128 — how far does A2C actually get?
             (Answer: it flies and hovers. It does not land. Project 22 lands it.)
    corr     how many INDEPENDENT samples a gradient batch really contains. This is
             the claim parallelism is usually sold on, and it is TRUE: effective
             sample size per update scales linearly with the number of envs.
    envs     1 / 2 / 8 / 16 envs at an EQUAL step budget — the claim tested rather
             than assumed. The result is the opposite of the folklore, and the
             reason is worth more than the folklore was.
    lam      GAE λ, swept. Also not what the textbook picture predicts, for a
             reason specific to truncated rollouts.

Runtime: ~9 min on 12 CPU cores (`python3 a2c.py all`).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "19-reinforce-on-cartpole"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import plot_style as ps
from pg_lib import ActorCritic, evaluate, make_vec_env, rollout, set_seed, space_dims

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

ENV_ID = "LunarLander-v3"
# gamma 0.999 and lambda 0.98, not the usual 0.99/0.95: LunarLander pays its +100
# landing bonus hundreds of steps after the thruster firings that earned it, and at
# gamma = 0.99 the effective horizon (~1/(1−γ) ≈ 100 steps) is shorter than the
# episode. The agent then cannot see the landing pad from where it stands, and
# settles for hovering — which is exactly the failure this project first produced.
GAMMA = 0.999
LAM = 0.98
N_STEPS = 128         # rollout length PER environment
N_ENVS = 8
LR = 7e-4
ENT_COEF = 0.01       # keeps the policy from committing to "do nothing" early
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
# Reward normalization is load-bearing, and NOT because the rewards are "too big".
# LunarLander's returns run into the hundreds, so the critic's gradient norm is ~50
# while the actor's is ~0.13. `clip_grad_norm_` clips them JOINTLY, so it rescales
# the whole update by 0.5/50 — and the actor, whose gradient was already small,
# gets multiplied by 0.01. Its effective learning rate becomes 7e-6 and the policy
# sits still. Dividing the reward by the running std of the return fixes the
# gradient budget, not the reward. Project 23 measures this directly.
NORM_REWARD = True
TOTAL_STEPS = 1_000_000
SEEDS = [0, 1, 2]


def a2c_update(agent, optim, ro, ent_coef=ENT_COEF, normalize_adv=True):
    """One gradient step on one rollout. On-policy: this data is now spent.

    A2C is exactly PPO with `n_epochs = 1`, no minibatching, and no ratio clip.
    Because the data is used once, by a policy that has not moved since it was
    collected, the importance ratio π_new/π_old is identically 1 — so there is
    nothing to clip and nothing to correct for. The moment you want to reuse a
    batch (project 22), that stops being true, and PPO is what you get.
    """
    _, logp, ent, values = agent.act(ro.obs, ro.actions)
    adv = ro.advantages
    if normalize_adv:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    pg_loss = -(logp * adv).mean()
    v_loss = F.mse_loss(values, ro.returns)
    ent_loss = -ent.mean()
    loss = pg_loss + VF_COEF * v_loss + ent_coef * ent_loss

    optim.zero_grad()
    loss.backward()
    gn = nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
    optim.step()
    return dict(pg=pg_loss.item(), v=v_loss.item(), ent=-ent_loss.item(), gn=gn.item())


def train(seed, n_envs=N_ENVS, n_steps=N_STEPS, lam=LAM, total_steps=TOTAL_STEPS,
          lr=LR, threads=1, log_every=25):
    torch.set_num_threads(threads)
    envs = make_vec_env(ENV_ID, n_envs, seed, gamma=GAMMA, norm_reward=NORM_REWARD)
    obs_dim, act_dim, cont = space_dims(envs)
    set_seed(seed)                                     # before the net, always
    agent = ActorCritic(obs_dim, act_dim, continuous=cont)
    optim = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    obs, _ = envs.reset(seed=seed)
    n_updates = total_steps // (n_envs * n_steps)
    steps_seen, curve, recent = [], [], []
    t0 = time.time()

    for u in range(n_updates):
        ro, obs = rollout(envs, agent, obs, n_steps, GAMMA, lam)
        a2c_update(agent, optim, ro)
        recent.extend(ro.ep_returns)
        if (u + 1) % log_every == 0:
            steps_seen.append((u + 1) * n_envs * n_steps)
            curve.append(np.mean(recent[-40:]) if recent else np.nan)
    envs.close()

    final, _ = evaluate(agent, ENV_ID, n_episodes=20, seed=9999)
    return dict(seed=seed, n_envs=n_envs, steps=np.asarray(steps_seen),
                curve=np.asarray(curve), final=final, wall=time.time() - t0,
                state=agent.state_dict())


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def _train_job(kw):
    r = train(**kw)
    r.pop("state")
    return r


def cmd_train():
    with ProcessPoolExecutor(max_workers=3) as pool:
        res = list(pool.map(_train_job, [dict(seed=s, threads=2) for s in SEEDS]))
    np.savez(OUT / "train.npz",
             steps=res[0]["steps"],
             curves=np.stack([r["curve"] for r in res]),
             finals=np.array([r["final"] for r in res]))
    for r in res:
        print(f"seed {r['seed']}: eval over 20 fresh episodes = {r['final']:7.1f}   "
              f"({r['wall']:.0f}s)")
    print(f"mean final: {np.mean([r['final'] for r in res]):.1f}  (200 = 'solved')")


def cmd_envs():
    """Same total experience, different numbers of environments.

    The step budget is held fixed, so any difference is NOT 'more data'. It is
    the same amount of data, arranged differently.
    """
    budget = 250_000
    jobs = [dict(seed=s, n_envs=n, total_steps=budget, threads=1)
            for n in (1, 2, 8, 16) for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(_train_job, jobs))
    by_n = {}
    for r in res:
        by_n.setdefault(r["n_envs"], []).append(r)

    # The curves cannot be stacked into one array: at a fixed step budget, a 1-env
    # run does 16x as many (much smaller) updates as a 16-env run, so its curve has
    # 16x as many points. Save each arm under its own key.
    ns = sorted(by_n)
    data = {"ns": np.array(ns)}
    for n in ns:
        data[f"steps_{n}"] = by_n[n][0]["steps"]
        data[f"curves_{n}"] = np.stack([r["curve"] for r in by_n[n]])
    data["finals"] = np.array([[r["final"] for r in by_n[n]] for n in ns])
    data["walls"] = np.array([[r["wall"] for r in by_n[n]] for n in ns])
    np.savez(OUT / "envs.npz", **data)

    for n in ns:
        f = [r["final"] for r in by_n[n]]
        w = np.mean([r["wall"] for r in by_n[n]])
        print(f"{n:2d} envs ({n * N_STEPS:5d} steps/update, {budget // (n * N_STEPS):4d} updates): "
              f"final {np.mean(f):7.1f}  wall {w:5.0f}s")


def cmd_corr():
    """How many INDEPENDENT samples does one gradient batch actually contain?

    A gradient step assumes its batch is a sample of the state distribution. It is
    not. Consecutive states within one environment are the same lander a fortieth
    of a second apart, so the lag-1 correlation between them is ~0.9: a 32-step
    rollout from one env is a single situation photographed 32 times.

    The subtle point, and the one this measurement makes concrete: adding parallel
    environments does NOT decorrelate the rows *within* a chain. That correlation
    is a property of the environment's physics and is untouched by how many copies
    you run — as the near-constant ρ below shows. What parallelism buys is more
    *independent chains*, and therefore more independent samples per update:

        ESS  ≈  (rows in the batch) · (1 − ρ)/(1 + ρ)        [the AR(1) formula]
             ≈  n · T · (1 − ρ)/(1 + ρ)

    so the effective sample size of a single gradient step grows linearly with the
    number of environments while ρ stays put. That is the whole trick.

    ρ is averaged over 10 consecutive rollouts (not one) because a single rollout
    of a hovering lander is far more autocorrelated than one of a tumbling lander,
    and the estimate is otherwise noisy enough to look like it depends on n.
    """
    rows = []
    for n in (1, 2, 4, 8, 16):
        torch.set_num_threads(2)
        envs = make_vec_env(ENV_ID, n, seed=0)
        obs_dim, act_dim, cont = space_dims(envs)
        set_seed(0)
        agent = ActorCritic(obs_dim, act_dim, continuous=cont)
        obs, _ = envs.reset(seed=0)
        for _ in range(20):                            # burn in past the identical starts
            ro, obs = rollout(envs, agent, obs, N_STEPS, GAMMA, LAM)

        cors = []
        for _ in range(10):                            # average ρ over 10 batches
            ro, obs = rollout(envs, agent, obs, N_STEPS, GAMMA, LAM)
            batch = ro.obs.numpy().reshape(N_STEPS, n, -1)
            for e in range(n):
                x = batch[:, e, :]
                for f in range(x.shape[1]):
                    a, b = x[:-1, f], x[1:, f]
                    if a.std() > 1e-6 and b.std() > 1e-6:
                        cors.append(abs(np.corrcoef(a, b)[0, 1]))
        rho = float(np.mean(cors))
        n_eff = (n * N_STEPS) * (1 - rho) / (1 + rho)
        rows.append((n, rho, n_eff, n * N_STEPS))
        envs.close()
        print(f"{n:2d} envs: batch {n * N_STEPS:4d} rows, lag-1 |corr| {rho:.3f}, "
              f"effective sample size {n_eff:6.1f}  "
              f"({100 * n_eff / (n * N_STEPS):.0f}% of the rows are worth anything)")
    np.savez(OUT / "corr.npz", rows=np.array(rows))


def cmd_lam():
    lams = [0.0, 0.5, 0.98, 1.0]
    jobs = [dict(seed=s, lam=l, total_steps=600_000, threads=1) for l in lams for s in (0, 1)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        res = list(pool.map(_train_job, jobs))
    by_l = {}
    for r, job in zip(res, jobs):
        by_l.setdefault(job["lam"], []).append(r)
    np.savez(OUT / "lam.npz", lams=lams,
             steps=by_l[lams[0]][0]["steps"],
             curves=np.stack([np.stack([r["curve"] for r in by_l[l]]) for l in lams]),
             finals=np.array([[r["final"] for r in by_l[l]] for l in lams]))
    for l in lams:
        f = [r["final"] for r in by_l[l]]
        print(f"λ = {l:4.2f}: final {np.mean(f):7.1f}  (seeds {np.array2string(np.array(f), precision=0)})")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def cmd_plot():
    import matplotlib.pyplot as plt

    d = np.load(OUT / "train.npz")
    fig, ax = ps.new_axes(7.6, 4.3)
    c, x = d["curves"], d["steps"]
    ax.fill_between(x, c.min(0), c.max(0), color=ps.SERIES[0], alpha=0.15, linewidth=0)
    ax.plot(x, c.mean(0), color=ps.SERIES[0], linewidth=2.2, label="A2C, 8 envs, GAE(0.98)")
    ax.axhline(200, color=ps.INK_MUTED, linestyle=":", linewidth=1)
    ax.text(x[1], 210, "solved (200)", color=ps.INK_MUTED, fontsize=8)
    ax.axhline(0, color=ps.BASELINE, linewidth=0.8)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ps.finish(fig, ax, f"A2C on LunarLander — {len(c)} seeds, band is min–max",
              "environment steps", "episode return", OUT / "learning_curve.png")

    e = np.load(OUT / "envs.npz")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for a in (ax1, ax2):
        ps.style_axes(a)
    for i, n in enumerate(e["ns"]):
        cur = e[f"curves_{n}"].mean(0)
        ax1.plot(e[f"steps_{n}"], cur, color=ps.SERIES[i], linewidth=2.0,
                 label=f"{n} env" + ("s" if n > 1 else ""))
    ax1.axhline(0, color=ps.BASELINE, linewidth=0.8)
    ax1.legend(frameon=False, fontsize=8, loc="lower right")
    ax1.set_title("Equal STEPS: fewer envs win (they take more updates)",
                  color=ps.INK, fontsize=11, loc="left")
    ax1.set_xlabel("environment steps", color=ps.INK_SECONDARY, fontsize=9)
    ax1.set_ylabel("episode return", color=ps.INK_SECONDARY, fontsize=9)

    c_ = np.load(OUT / "corr.npz")["rows"]
    ns, rhos, ess, rowcount = c_[:, 0], c_[:, 1], c_[:, 2], c_[:, 3]
    ax2.plot(ns, rowcount, "o--", color=ps.INK_MUTED, linewidth=1.5,
             label="rows in the batch")
    ax2.plot(ns, ess, "o-", color=ps.SERIES[2], linewidth=2.2,
             label="INDEPENDENT samples in it")
    for n, e, r in zip(ns, ess, rowcount):
        ax2.annotate(f"{e:.0f}", (n, e), textcoords="offset points", xytext=(4, -12),
                     fontsize=8, color=ps.SERIES[2])
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(ns)
    ax2.set_xticklabels([int(n) for n in ns])
    ax2.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.set_title(f"Why: one batch, and what it is really worth\n"
                  f"(lag-1 correlation stays ≈{rhos.mean():.2f} — parallelism adds chains, "
                  f"it does not decorrelate them)",
                  color=ps.INK, fontsize=10, loc="left")
    ax2.set_xlabel("parallel environments", color=ps.INK_SECONDARY, fontsize=9)
    ax2.set_ylabel("samples per gradient update (log)", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "parallel_envs.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'parallel_envs.png'}")

    l = np.load(OUT / "lam.npz")
    fig, ax = ps.new_axes(7.4, 4.2)
    for i, lam in enumerate(l["lams"]):
        cur = l["curves"][i].mean(0)
        name = {0.0: "λ = 0  (one-step TD: biased, calm)",
                0.5: "λ = 0.5",
                0.98: "λ = 0.98  (this project's choice)",
                1.0: "λ = 1  (no critic in the target at all)"}[float(lam)]
        ax.plot(l["steps"], cur, color=ps.SERIES[i], linewidth=2.0, label=name)
    ax.axhline(0, color=ps.BASELINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ps.finish(fig, ax, "The GAE dial, measured: leaning on the critic is what hurts",
              "environment steps", "episode return", OUT / "gae_lambda.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("train", "all"):
        cmd_train()
    if cmd in ("corr", "all"):
        cmd_corr()
    if cmd in ("envs", "all"):
        cmd_envs()
    if cmd in ("lam", "all"):
        cmd_lam()
    if cmd in ("plot", "all"):
        cmd_plot()
