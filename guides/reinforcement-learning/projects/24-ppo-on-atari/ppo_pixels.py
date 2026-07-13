"""PPO from pixels: the nine Atari-specific details, and an honest look at the bill.

The nine Atari details in the 37 are not about PPO at all — they are about what
you feed it. Seven of them are environment wrappers (`NoopReset`, `MaxAndSkip`,
`EpisodicLife`, `FireReset`, `WarpFrame`, `ClipReward`, `FrameStack`) and two are
about the network (a SHARED Nature-CNN trunk, and scaling pixels to [0, 1]).
Project 14 already built that wrapper chain for DQN and pointed it at real ALE
Pong; this project reuses it verbatim and swaps the algorithm.

What it cannot do is train on real Pong, and it is worth being exact about why
rather than waving at it. Measured below: real `ALE/Pong-v5`, with this PPO's
network in the loop, runs at roughly a thousand steps a second on this CPU. The
published PPO score of ~20.7 takes 10M frames. That is about three hours — for
one game, one seed. So this project does what the honest version of "PPO on
Atari" looks like on a CPU budget:

    pipeline   run the REAL ALE preprocessing chain on REAL Pong frames and look
               at exactly what the network is handed (figure).
    throughput measure the real thing, and extrapolate the real bill (table).
    train      train the identical pixel PPO on MiniPong — Pong against a wall,
               20x20 real pixels, 4-frame stack — where it converges in minutes.
    ablate     the two Atari NETWORK details, tested rather than assumed:
               shared trunk vs separate, and frame stacking on vs off.

Runtime: ~8 min on 12 CPU cores (`python3 ppo_pixels.py all`).
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "22-ppo-from-scratch"))
sys.path.insert(0, str(HERE.parent / "19-reinforce-on-cartpole"))
sys.path.insert(0, str(HERE.parent / "14-atari-pong"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))

import plot_style as ps
from pg_lib import compute_gae, layer_init, set_seed
from pong_lib import GRID, STACK, AtariPipeline, make_minipong

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


# --------------------------------------------------------------------------
# The network: Atari details #21 (shared trunk) and #22 (scale to [0,1])
# --------------------------------------------------------------------------

class ConvActorCritic(nn.Module):
    """A Nature-CNN torso, shrunk for a 20x20 frame, with two heads on one trunk.

    Detail #21 says the policy and value heads SHARE the convolutional trunk on
    Atari — which is the exact opposite of detail #26, which says they must be
    SEPARATE for continuous control. Both are in the same list of 37 and they
    contradict each other, and that is not sloppiness: on Atari the trunk is a
    perception system, expensive and identical for both heads (both need to know
    where the ball is), so sharing is nearly free and doubles the gradient signal
    reaching the convs. In continuous control the "trunk" is two small dense
    layers on a proprioceptive vector, sharing buys nothing, and the value loss —
    whose scale is the return — bullies the policy gradient through the shared
    weights. `shared=False` here runs the ablation.
    """

    def __init__(self, n_actions=3, in_frames=STACK, grid=GRID, shared=True):
        super().__init__()
        self.shared = shared

        def trunk():
            return nn.Sequential(
                layer_init(nn.Conv2d(in_frames, 16, 3, stride=2, padding=1)), nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, 3, stride=2, padding=1)), nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(32 * 5 * 5, 128)), nn.ReLU(),
            )

        self.body = trunk()
        self.body_v = None if shared else trunk()
        self.actor = layer_init(nn.Linear(128, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(128, 1), std=1.0)
        self.continuous = False

    def dist_value(self, obs):
        h = self.body(obs)
        logits = self.actor(h)
        hv = h if self.shared else self.body_v(obs)
        return Categorical(logits=logits), self.critic(hv).squeeze(-1)

    def value(self, obs):
        h = self.body(obs) if self.shared else self.body_v(obs)
        return self.critic(h).squeeze(-1)

    def act(self, obs, action=None, deterministic=False):
        d, v = self.dist_value(obs)
        if action is None:
            action = d.probs.argmax(-1) if deterministic else d.sample()
        return action, d.log_prob(action), d.entropy(), v


# --------------------------------------------------------------------------
# A vectorized MiniPong. Detail #1, by hand, because MiniPong is not a gym env.
# --------------------------------------------------------------------------

class VecMiniPong:
    """N MiniPongs stepped in lockstep, with SAME_STEP autoreset.

    Autoreset semantics matter exactly as much here as they did in `pg_lib`: when
    one env finishes, it is reset immediately and the observation returned is the
    NEW episode's first frame, while the value bootstrap must use zero (MiniPong
    only ever terminates, never truncates — a missed ball really is the end).
    """

    def __init__(self, n, seed=0, stack=STACK, clip=False, rally_bonus=0.0):
        self.envs = [make_minipong(seed=seed * 1000 + i, stack=stack, clip=clip,
                                   rally_bonus=rally_bonus) for i in range(n)]
        self.n = n
        self.n_actions = self.envs[0].n_actions
        self.obs_shape = self.envs[0].obs_shape
        self.ep_ret = np.zeros(n)
        self.finished = []

    def reset(self, seed=0):
        obs = [e.reset(seed=seed * 1000 + i)[0] for i, e in enumerate(self.envs)]
        self.ep_ret[:] = 0
        return np.stack(obs)

    def step(self, actions):
        obs, rews, terms = [], [], []
        for i, (e, a) in enumerate(zip(self.envs, actions)):
            o, r, term, trunc, info = e.step(int(a))
            self.ep_ret[i] += info.get("raw_reward", r)
            if term or trunc:
                self.finished.append(self.ep_ret[i])
                self.ep_ret[i] = 0
                o, _ = e.reset()
            obs.append(o)
            rews.append(r)
            terms.append(term)
        return np.stack(obs), np.asarray(rews), np.asarray(terms, dtype=bool)


# --------------------------------------------------------------------------
# PPO, identical to project 22's, over pixels
# --------------------------------------------------------------------------

@dataclass
class Cfg:
    total_steps: int = 400_000
    n_envs: int = 8
    n_steps: int = 128
    n_epochs: int = 4
    n_minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    stack: int = STACK
    shared: bool = True
    clip_reward: bool = False
    rally_bonus: float = 0.0


def evaluate_pixels(agent, cfg, n_episodes=20, seed=7777):
    env = make_minipong(seed=seed, stack=cfg.stack, clip=False, rally_bonus=cfg.rally_bonus)
    totals = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                a, _, _, _ = agent.act(torch.as_tensor(obs).unsqueeze(0), deterministic=True)
            obs, r, term, trunc, _ = env.step(int(a))
            total += r
            done = term or trunc
        totals.append(total)
    return float(np.mean(totals))


def train(cfg, seed=0, threads=1):
    torch.set_num_threads(threads)
    envs = VecMiniPong(cfg.n_envs, seed=seed, stack=cfg.stack, clip=cfg.clip_reward,
                       rally_bonus=cfg.rally_bonus)
    set_seed(seed)
    agent = ConvActorCritic(n_actions=envs.n_actions, in_frames=cfg.stack, shared=cfg.shared)
    optim = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    batch = cfg.n_envs * cfg.n_steps
    mb_size = batch // cfg.n_minibatches
    n_updates = cfg.total_steps // batch
    obs = envs.reset(seed=seed)
    steps, curve = [], []
    t0 = time.time()

    for update in range(n_updates):
        optim.param_groups[0]["lr"] = (1 - update / n_updates) * cfg.lr

        o_buf, a_buf, lp_buf, r_buf, d_buf, v_buf, nv_buf = [], [], [], [], [], [], []
        for _ in range(cfg.n_steps):
            t_obs = torch.as_tensor(obs)
            with torch.no_grad():
                a, lp, _, v = agent.act(t_obs)
            obs2, r, term = envs.step(a.numpy())
            with torch.no_grad():
                nv = agent.value(torch.as_tensor(obs2)).numpy()
            nv = np.where(term, 0.0, nv)      # terminated: the future is worth nothing
            o_buf.append(obs); a_buf.append(a.numpy()); lp_buf.append(lp.numpy())
            r_buf.append(r); d_buf.append(term.astype(np.float64)); v_buf.append(v.numpy())
            nv_buf.append(nv)
            obs = obs2

        adv, ret = compute_gae(np.asarray(r_buf), np.asarray(v_buf, dtype=np.float64),
                               np.asarray(d_buf), np.asarray(nv_buf, dtype=np.float64),
                               cfg.gamma, cfg.lam)
        b_obs = torch.as_tensor(np.asarray(o_buf).reshape(batch, cfg.stack, GRID, GRID))
        b_act = torch.as_tensor(np.asarray(a_buf).reshape(batch))
        b_lp = torch.as_tensor(np.asarray(lp_buf).reshape(batch), dtype=torch.float32)
        b_adv = torch.as_tensor(adv.reshape(batch), dtype=torch.float32)
        b_ret = torch.as_tensor(ret.reshape(batch), dtype=torch.float32)

        idx = np.arange(batch)
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch, mb_size):
                mb = idx[start:start + mb_size]
                _, lp, ent, v = agent.act(b_obs[mb], b_act[mb])
                ratio = (lp - b_lp[mb]).exp()
                a_mb = b_adv[mb]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio,
                               -a_mb * ratio.clamp(1 - cfg.clip_coef, 1 + cfg.clip_coef)).mean()
                v_loss = 0.5 * ((v - b_ret[mb]) ** 2).mean()
                loss = pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent.mean()
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optim.step()

        if (update + 1) % 5 == 0:
            steps.append((update + 1) * batch)
            curve.append(np.mean(envs.finished[-40:]) if envs.finished else np.nan)

    final = evaluate_pixels(agent, cfg)
    return dict(steps=np.asarray(steps), curve=np.asarray(curve), final=final,
                wall=time.time() - t0, agent=agent)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------

def _job(args):
    label, cfg, seed = args
    r = train(cfg, seed=seed, threads=1)
    r.pop("agent")
    return label, seed, r


def cmd_train():
    seeds = [0, 1, 2]
    variants = {
        "PPO (shared trunk, 4 frames)": Cfg(),
        "separate trunks (#21 off)":    Cfg(shared=False),
        "1 frame, no stacking (#20 off)": Cfg(stack=1),
    }
    jobs = [(l, c, s) for l, c in variants.items() for s in seeds]
    with ProcessPoolExecutor(max_workers=9) as pool:
        res = list(pool.map(_job, jobs))
    by = {}
    for label, seed, r in res:
        by.setdefault(label, []).append(r)
    np.savez(OUT / "train.npz",
             labels=list(variants),
             steps=by[list(variants)[0]][0]["steps"],
             curves=np.stack([np.stack([r["curve"] for r in by[l]]) for l in variants]),
             finals=np.stack([np.array([r["final"] for r in by[l]]) for l in variants]))
    print(f"{'variant':32s} {'final return':>13s}   (max possible +10)")
    for l in variants:
        f = np.array([r["final"] for r in by[l]])
        print(f"{l:32s} {f.mean():7.2f} ± {f.std():4.2f}")


def cmd_throughput():
    """What real Atari would actually cost, measured rather than guessed."""
    rows = []
    try:
        pipe = AtariPipeline(game="ALE/Pong-v5", seed=0)
    except Exception as e:
        print(f"ALE unavailable ({e}); skipping the real-Pong measurement")
        return
    agent = ConvActorCritic(n_actions=6, in_frames=STACK, grid=84)
    # The real Nature-CNN on 84x84 is bigger than MiniPong's; time the env alone
    # and the env+policy together, since the published numbers assume a GPU does
    # the second part.
    obs = pipe.reset()
    t0 = time.time()
    n = 2000
    for _ in range(n):
        obs, r, term, trunc, _ = pipe.step(np.random.randint(6))
        if term or trunc:
            obs = pipe.reset()
    env_sps = n / (time.time() - t0)
    pipe.close()

    net = nn.Sequential(
        nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
        nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
    )
    x = torch.zeros(8, 4, 84, 84)
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            net(x)
    fwd_sps = 8 * 100 / (time.time() - t0)

    combined = 1 / (1 / env_sps + 1 / fwd_sps)
    for frames, label in [(1_000_000, "1M frames (a first point scored)"),
                          (10_000_000, "10M frames (published PPO ~20.7)")]:
        rows.append((label, frames, frames / combined / 3600))
    print(f"real ALE/Pong-v5 preprocessing:      {env_sps:7.0f} steps/s")
    print(f"Nature-CNN forward pass (batch 8):   {fwd_sps:7.0f} steps/s")
    print(f"both together:                       {combined:7.0f} steps/s")
    for label, frames, hours in rows:
        print(f"  {label:36s} {hours:6.1f} hours on this CPU")
    np.savez(OUT / "throughput.npz", env_sps=env_sps, fwd_sps=fwd_sps, combined=combined,
             labels=[r[0] for r in rows], hours=[r[2] for r in rows])


def cmd_pipeline():
    """Look at what the network is actually handed, on real Pong frames."""
    import matplotlib.pyplot as plt
    try:
        pipe = AtariPipeline(game="ALE/Pong-v5", seed=0)
    except Exception as e:
        print(f"ALE unavailable ({e}); skipping the pipeline figure")
        return
    obs = pipe.reset()
    for _ in range(60):                       # get past the serve
        obs, *_ = pipe.step(np.random.randint(6))
    raw = pipe.raw_frame          # a property, not a method

    mini = make_minipong(seed=0)
    mo, _ = mini.reset(seed=0)
    for _ in range(6):
        mo, *_ = mini.step(2)

    fig, axes = plt.subplots(2, 5, figsize=(12.4, 5.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    axes[0][0].imshow(raw)
    axes[0][0].set_title("real ALE frame\n210x160x3 RGB", fontsize=9, color=ps.INK)
    for k in range(4):
        axes[0][k + 1].imshow(obs[k], cmap="gray", vmin=0, vmax=1)
        axes[0][k + 1].set_title(f"preprocessed, frame {k + 1}/4", fontsize=9, color=ps.INK)
    axes[1][0].axis("off")
    axes[1][0].text(0.0, 0.5, "MiniPong\n(what we can afford\nto train on)",
                    fontsize=10, color=ps.INK, va="center")
    for k in range(4):
        axes[1][k + 1].imshow(mo[k], cmap="gray", vmin=0, vmax=1)
        axes[1][k + 1].set_title(f"20x20, frame {k + 1}/4", fontsize=9, color=ps.INK)
    for row in axes:
        for a in row:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("The Atari details are about what you feed the network, not the algorithm",
                 fontsize=12, color=ps.INK, x=0.02, ha="left")
    fig.tight_layout(h_pad=2.6)
    fig.savefig(OUT / "pipeline.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    pipe.close()
    print(f"wrote {OUT / 'pipeline.png'}")


def cmd_plot():
    import matplotlib.pyplot as plt
    d = np.load(OUT / "train.npz")
    labels = list(d["labels"])
    fig, ax = ps.new_axes(7.8, 4.4)
    for i, l in enumerate(labels):
        c = d["curves"][i]
        ax.fill_between(d["steps"], c.min(0), c.max(0), color=ps.SERIES[i], alpha=0.13, linewidth=0)
        ax.plot(d["steps"], c.mean(0), color=ps.SERIES[i], linewidth=2.2,
                label=f"{l} — final {d['finals'][i].mean():+.1f}")
    ax.axhline(10, color=ps.INK_MUTED, linestyle=":", linewidth=1)
    ax.text(d["steps"][0], 10.3, "perfect play (+10)", color=ps.INK_MUTED, fontsize=8)
    ax.axhline(0, color=ps.BASELINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ps.finish(fig, ax, "PPO from pixels on MiniPong — 3 seeds, band is min–max",
              "environment steps", "episode return", OUT / "minipong_ppo.png")


if __name__ == "__main__":
    # A fork-safety guard, and it is not optional. `ProcessPoolExecutor` forks this
    # process, and forking a process that already has live OpenMP threads deadlocks:
    # the children inherit the lock state but not the threads holding it, and then
    # wait forever. ANY torch op in the parent — the ratio check below, a figure, a
    # throughput probe — is enough to start those threads. Pinning the parent to one
    # thread keeps the OpenMP pool from ever being created; each worker sets its own
    # thread count once it is safely inside its own process.
    torch.set_num_threads(1)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("pipeline", "all"):
        cmd_pipeline()
    if cmd in ("throughput", "all"):
        cmd_throughput()
    if cmd in ("train", "all"):
        cmd_train()
    if cmd in ("plot", "all"):
        cmd_plot()
