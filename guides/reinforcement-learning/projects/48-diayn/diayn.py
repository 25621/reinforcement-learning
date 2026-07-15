r"""Project 48 — DIAYN: learn a set of distinct behaviours with no reward at all.

The environment here pays nothing, ever. The agent invents its own job:

    1. Before each episode, roll a die: "today I am skill z" (z = 0..5).
    2. A DISCRIMINATOR q(z | s) — a separate little classifier — watches the states
       you visit and tries to guess which skill you are.
    3. The agent is paid  log q(z | s) - log p(z)  : "make it easy to guess."

To be easy to guess, a skill must go somewhere the other skills do not. Six skills
racing to be recognizable end up carving the world into six territories — which is
DIAYN's whole trick, and it is exactly the [mutual information] between the skill
and the states it produces, maximized by two players who need each other:
the discriminator gets better at reading behaviour, so behaviour must become more
distinctive to stay readable.

Why a separate classifier at all, when the policy already knows its own z?
The policy knows z because we hand it z as an input — that is the *question*, not
the answer. Nothing in it forces the six skills to behave differently; a policy
that ignores z entirely would be perfectly happy. The discriminator is the part that
cannot be fooled: it sees only the STATES, not the skill code, so the only way to
score is for the states themselves to reveal the skill. It converts "be different"
from a wish into a number the agent can climb.

    python3 diayn.py     # ~7 min: pre-train 6 skills, then use them on a task nobody trained for
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "26-ddpg-on-pendulum"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import cc_lib as cc  # noqa: E402   SquashedGaussianActor / Critic / ReplayBuffer / soft_update
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"

N_SKILLS = 6
EP_LEN = 60
STEP = 0.05


class PointRoom:
    """A dot in a square room. No walls, no goal, no reward. Just a place to be."""

    obs_dim, act_dim = 2, 2

    def reset(self):
        self.p = np.zeros(2, dtype=np.float32)
        self.t = 0
        return self.p.copy()

    def step(self, a):
        self.p = np.clip(self.p + STEP * np.clip(a, -1, 1), -1, 1).astype(np.float32)
        self.t += 1
        return self.p.copy(), 0.0, self.t >= EP_LEN     # reward is ALWAYS zero


def onehot(z, n=N_SKILLS):
    v = np.zeros(n, dtype=np.float32)
    v[z] = 1.0
    return v


def train(total_steps=60_000, hidden=128, batch=256, gamma=0.99, tau=0.005,
          alpha=0.1, lr=3e-4, start_steps=1000, seed=0):
    cc.set_seed(seed)
    torch.set_num_threads(6)
    rng = np.random.default_rng(seed)
    env = PointRoom()
    obs_dim = env.obs_dim + N_SKILLS          # the policy is TOLD which skill it is today

    actor = cc.SquashedGaussianActor(obs_dim, env.act_dim, hidden, act_limit=1.0)
    q1, q2 = cc.Critic(obs_dim, env.act_dim, hidden), cc.Critic(obs_dim, env.act_dim, hidden)
    q1_t, q2_t = cc.Critic(obs_dim, env.act_dim, hidden), cc.Critic(obs_dim, env.act_dim, hidden)
    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    disc = cc.mlp([env.obs_dim, hidden, hidden, N_SKILLS])   # reads STATES only, never z

    opt_pi = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr)

    buf = cc.ReplayBuffer(obs_dim, env.act_dim, capacity=total_steps)
    log_pz = float(np.log(1.0 / N_SKILLS))    # skills are drawn uniformly, so this is a constant

    hist = {"steps": [], "acc": [], "rint": []}
    s = env.reset()
    z = int(rng.integers(N_SKILLS))
    obs = np.concatenate([s, onehot(z)])

    for step in range(1, total_steps + 1):
        if step <= start_steps:
            a = rng.uniform(-1, 1, env.act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(obs).unsqueeze(0))
            a = a.squeeze(0).numpy()
        s2, _, done = env.step(a)
        next_obs = np.concatenate([s2, onehot(z)])
        buf.add(obs, a, 0.0, next_obs, 0.0)   # the stored reward is a placeholder — see below
        obs = next_obs
        if done:
            s = env.reset()
            z = int(rng.integers(N_SKILLS))
            obs = np.concatenate([s, onehot(z)])

        if step < start_steps or buf.size < batch:
            continue

        o, ac, _, o2, d = buf.sample(batch, rng)
        state2, skill = o2[:, :2], o2[:, 2:].argmax(-1)

        # --- the discriminator learns to read the skill off the state
        logits = disc(state2)
        d_loss = F.cross_entropy(logits, skill)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # --- the reward is RECOMPUTED here, not read from the buffer.
        # DIAYN's reward is defined by the discriminator, and the discriminator is a
        # moving target: a bonus computed 40k steps ago was scored by a much worse
        # judge and no longer means anything. Relabelling every sampled batch with the
        # CURRENT judge keeps the critic learning today's objective instead of a fossil.
        with torch.no_grad():
            r_int = (F.log_softmax(disc(state2), dim=-1)
                     .gather(1, skill.unsqueeze(1)) - log_pz)

        # --- ordinary SAC from here on
        with torch.no_grad():
            a2, logp2 = actor(o2)
            q_t = torch.min(q1_t(o2, a2), q2_t(o2, a2)).squeeze(-1)
            backup = r_int.squeeze(-1) + gamma * (1 - d.squeeze(-1)) * (q_t - alpha * logp2)
        q_loss = (F.mse_loss(q1(o, ac).squeeze(-1), backup)
                  + F.mse_loss(q2(o, ac).squeeze(-1), backup))
        opt_q.zero_grad()
        q_loss.backward()
        opt_q.step()

        a_pi, logp_pi = actor(o)
        pi_loss = (alpha * logp_pi - torch.min(q1(o, a_pi), q2(o, a_pi)).squeeze(-1)).mean()
        opt_pi.zero_grad()
        pi_loss.backward()
        opt_pi.step()
        cc.soft_update(q1, q1_t, tau)
        cc.soft_update(q2, q2_t, tau)

        if step % 2000 == 0:
            acc = float((logits.argmax(-1) == skill).float().mean())
            hist["steps"].append(step)
            hist["acc"].append(acc)
            hist["rint"].append(float(r_int.mean()))
            print(f"  step {step:>6d}  discriminator accuracy {acc:.3f}  "
                  f"intrinsic reward {r_int.mean():+.3f}  (chance = {1/N_SKILLS:.2f})", flush=True)

    return actor, disc, hist


# ------------------------------------------------------------------ evaluation
def rollout(actor, z, seed=0, deterministic=True):
    env = PointRoom()
    s = env.reset()
    path = [s.copy()]
    done = False
    while not done:
        obs = torch.as_tensor(np.concatenate([s, onehot(z)])).unsqueeze(0)
        with torch.no_grad():
            a, _ = actor(obs, deterministic=deterministic)
        s, _, done = env.step(a.squeeze(0).numpy())
        path.append(s.copy())
    return np.array(path)


def random_paths(n=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        env = PointRoom()
        s = env.reset()
        path, done = [s.copy()], False
        while not done:
            s, _, done = env.step(rng.uniform(-1, 1, 2).astype(np.float32))
            path.append(s.copy())
        out.append(np.array(path))
    return out


def downstream(actor):
    """Now invent a task nobody trained for: reach a goal corner. Can the skills do it?

    No learning happens here. We simply try each of the six skills once and keep the
    best — the cheapest possible way to "use" a skill library. The comparison is a
    random policy, which is what an agent with no pre-training would start from.
    """
    goals = {"NE": (0.8, 0.8), "NW": (-0.8, 0.8), "SE": (0.8, -0.8), "SW": (-0.8, -0.8)}
    rows = []
    rand = random_paths(20, seed=3)
    for name, g in goals.items():
        g = np.array(g, dtype=np.float32)
        best_skill, best_d = None, 1e9
        for z in range(N_SKILLS):
            d = float(np.linalg.norm(rollout(actor, z)[-1] - g))
            if d < best_d:
                best_skill, best_d = z, d
        rnd_d = float(np.mean([np.linalg.norm(p[-1] - g) for p in rand]))
        rows.append((name, best_skill, best_d, rnd_d))
        print(f"  goal {name}: best skill = #{best_skill}, final distance {best_d:.2f} "
              f"vs random policy {rnd_d:.2f}", flush=True)
    return rows


# --------------------------------------------------------------------- figures
def plot_skills(actor, disc, path):
    fig, axes = ps.plt.subplots(1, 2, figsize=(10.6, 4.8), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)

    # left: the discriminator's map of the room, with each skill's trajectories on top
    ax = axes[0]
    g = np.linspace(-1, 1, 160)
    xx, yy = np.meshgrid(g, g)
    grid = torch.as_tensor(np.stack([xx.ravel(), yy.ravel()], -1).astype(np.float32))
    with torch.no_grad():
        pred = disc(grid).argmax(-1).numpy().reshape(xx.shape)
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(ps.SERIES[:N_SKILLS])
    ax.imshow(pred, origin="lower", extent=(-1, 1, -1, 1), cmap=cmap, alpha=0.25,
              vmin=0, vmax=N_SKILLS - 1, interpolation="nearest")
    for z in range(N_SKILLS):
        for k in range(3):
            p = rollout(actor, z, deterministic=(k == 0))
            ax.plot(p[:, 0], p[:, 1], color=ps.SERIES[z], lw=2.0 if k == 0 else 0.9,
                    alpha=1.0 if k == 0 else 0.45, label=f"skill {z}" if k == 0 else None)
        ax.scatter(*rollout(actor, z)[-1], color=ps.SERIES[z], s=45, zorder=3)
    ax.scatter([0], [0], color=ps.INK, s=30, zorder=4)
    ax.set_title("Six skills, zero rewards\n(shading = the discriminator's guess for each spot)",
                 color=ps.INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ps.style_axes(ax)
    ax.set_xlim(-1.05, 1.05), ax.set_ylim(-1.05, 1.05)

    # right: what an untrained policy does with the same 60 steps
    ax = axes[1]
    for p in random_paths(12, seed=1):
        ax.plot(p[:, 0], p[:, 1], color=ps.INK_MUTED, lw=1.0, alpha=0.7)
        ax.scatter(*p[-1], color=ps.INK_MUTED, s=25)
    ax.scatter([0], [0], color=ps.INK, s=30, zorder=4)
    ps.style_axes(ax)
    ax.set_xlim(-1.05, 1.05), ax.set_ylim(-1.05, 1.05)
    ax.set_title("A random policy, same 60 steps\n(it wanders, and it wanders the same way every time)",
                 color=ps.INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


def plot_progress(hist, path):
    fig, ax = ps.new_axes(7.4, 4.2)
    steps = np.array(hist["steps"]) / 1000
    ax.plot(steps, hist["acc"], color=ps.SERIES[0], lw=2.2,
            label="discriminator: can it name the skill from the state alone?")
    ax.axhline(1 / N_SKILLS, color=ps.INK_MUTED, ls="--", lw=1.3,
               label=f"chance ({1/N_SKILLS:.2f}) — what a skill-blind agent would score")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "The two players pull each other up",
              "environment steps (thousands)", "accuracy", path)


def plot_downstream(rows, path):
    fig, ax = ps.new_axes(7.2, 4.0)
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r[2] for r in rows], width=0.38, color=ps.SERIES[1],
           label="best of the 6 pre-trained skills (no training)")
    ax.bar(x + 0.2, [r[3] for r in rows], width=0.38, color=ps.INK_MUTED,
           label="random policy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"goal {r[0]}\n(skill #{r[1]})" for r in rows])
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ps.finish(fig, ax, "A task nobody trained for, solved by picking one skill off the shelf",
              "", "distance from the goal at the end of the episode (lower = better)", path)


def main():
    OUT.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    t0 = time.time()
    print("[1/3] pre-training 6 skills on a reward-free room")
    actor, disc, hist = train()
    torch.save({"actor": actor.state_dict(), "disc": disc.state_dict()}, CKPT / "diayn.pt")
    print(f"  ({time.time() - t0:.0f}s)")

    print("[2/3] what did the skills become?")
    plot_skills(actor, disc, OUT / "skill_space.png")
    plot_progress(hist, OUT / "discriminator.png")

    print("[3/3] downstream: reuse the skills on four goals they never saw")
    rows = downstream(actor)
    plot_downstream(rows, OUT / "downstream.png")


if __name__ == "__main__":
    main()
