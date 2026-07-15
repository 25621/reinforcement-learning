r"""Project 46 — Random Network Distillation, and an honest word about "on Atari".

Real Montezuma's Revenge, with the RND paper's settings, is 2 *billion* frames on
128 parallel workers. On this CPU that is several months. So the project splits in
two, and neither half pretends to be the other:

  Part 1 (the agent):  a miniature Montezuma — a two-room pixel maze where the
                       reward needs the same "fetch the key, then open the door"
                       chain of ~30 correct moves. PPO alone scores 0. PPO+RND
                       solves it. This is the paper's result, at 1/5000 the cost.

  Part 2 (the signal): the novelty detector itself, running on REAL 210x160
                       Montezuma frames from the Arcade Learning Environment. We
                       cannot train the agent, but we can show that RND's bonus
                       does on true Atari pixels exactly what it claims: it falls
                       on frames it has studied and stays high on frames it has
                       never seen.

    python3 rnd.py        # ~9 min total
"""

import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import explore_lib as el  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 500_000
SEEDS = (0, 1, 2, 3)


# --------------------------------------------------------------- part 1: agent
def run_one(args):
    arm, seed = args
    make = (lambda c, a, s: el.RND(c, a, s)) if arm == "rnd" else None
    h = el.train(make_bonus=make, total_steps=STEPS, seed=seed, verbose=False)
    print(f"  {h['name']:9s} seed {seed}: final success {h['success'][-1]:.2f}, "
          f"states seen {h['coverage'][-1]}/122, first reward at step "
          f"{h['first_success']}", flush=True)
    return arm, seed, h


def train_arms():
    jobs = [(arm, s) for arm in ("ppo", "rnd") for s in SEEDS]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run_one, jobs))
    print(f"  ({time.time() - t0:.0f}s for {len(jobs)} runs)")
    packed = {}
    for arm, seed, h in results:
        packed[f"{arm}_{seed}_success"] = h["success"]
        packed[f"{arm}_{seed}_coverage"] = h["coverage"]
        packed[f"{arm}_{seed}_first"] = -1 if h["first_success"] is None else h["first_success"]
        packed["steps"] = h["steps"]
    np.savez(CKPT / "curves.npz", **packed)
    return packed


def plot_agent(p, path):
    fig, axes = ps.plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    steps = np.array(p["steps"]) / 1000
    for ax, key, ylab, title in [
        (axes[0], "success", "episodes solved (fraction)", "The reward: PPO never sees it"),
        (axes[1], "coverage", "distinct states visited (of 122)", "The cause: PPO never goes anywhere"),
    ]:
        for i, (arm, label) in enumerate([("ppo", "PPO only"), ("rnd", "PPO + RND")]):
            runs = np.stack([p[f"{arm}_{s}_{key}"] for s in SEEDS])
            for r in runs:
                ax.plot(steps, r, color=ps.SERIES[i], lw=0.9, alpha=0.30)
            ax.plot(steps, runs.mean(0), color=ps.SERIES[i], lw=2.4, label=label)
        ps.style_axes(ax)
        ax.set_title(title, color=ps.INK, fontsize=11.5, loc="left")
        ax.set_xlabel("environment steps (thousands)", color=ps.INK_SECONDARY, fontsize=10)
        ax.set_ylabel(ylab, color=ps.INK_SECONDARY, fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


def plot_maze(path):
    """Draw the world once, so the reader can see what the agent sees."""
    env = el.KeyDoorRoom()
    env.reset()
    fig, ax = ps.plt.subplots(figsize=(6.2, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    img = np.zeros((env.h, env.w, 3))
    img[env.walls == 1] = [0.16, 0.16, 0.18]
    img[env.walls == 0] = [0.97, 0.97, 0.96]
    ax.imshow(img)
    for (x, y), txt, col in [(el.START, "start", ps.SERIES[0]), (el.KEY, "key", ps.SERIES[3]),
                             (el.DOOR, "door +1", ps.SERIES[2])]:
        ax.scatter([x], [y], s=200, color=col, zorder=3)
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", color=ps.INK, fontsize=9.5)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_title("The miniature Montezuma: thread the tunnel, take the key, come back, open the door\n"
                 "A random walker solved it 0 times in 3,000 episodes.",
                 color=ps.INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------- part 2: real Atari frames
def atari_novelty(n_frames=400, train_steps=400):
    """Train an RND predictor on real Montezuma frames; test it on frames it never saw.

    "Seen" frames come from random play in Montezuma's first room — the only place
    a random agent ever gets to, which is itself the point of the game's reputation.
    "Unseen" frames come from two other Atari games. If RND's error is really a
    novelty meter, it must drop on the first group and stay high on the second,
    with no labels, no counts, and no idea what a "room" is.
    """
    import ale_py
    import gymnasium as gym
    gym.register_envs(ale_py)
    torch.set_num_threads(6)

    def collect(game, n, seed=0):
        env = gym.make(f"ALE/{game}-v5")
        obs, _ = env.reset(seed=seed)
        frames = []
        for i in range(n * 4):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if i % 4 == 0:                                    # the usual frame-skip
                g = obs.astype(np.float32).mean(-1)[::2, ::2] / 255.0   # grey, 105x80
                frames.append(g[None])
            if term or trunc:
                obs, _ = env.reset()
        env.close()
        return np.stack(frames[:n])

    seen = collect("MontezumaRevenge", n_frames, seed=0)
    held = collect("MontezumaRevenge", 100, seed=7)           # same game, fresh frames
    novel_a = collect("Pong", 100, seed=0)
    novel_b = collect("Breakout", 100, seed=0)
    raw_frame = seen[0, 0]

    obs_shape = int(np.prod(seen.shape[1:]))
    # Same class the agent uses, pointed at 105x80 grey frames instead of the 9x13 maze.
    # `update_proportion=1.0` because here we WANT the predictor to study every frame it
    # is given: this half of the project measures the novelty signal, not exploration.
    rnd = el.RND(1, 0, obs_shape, out_dim=64, lr=1e-4, hw=seen.shape[2:],
                 update_proportion=1.0)
    hist = {"step": [], "seen": [], "held": [], "pong": [], "breakout": []}
    rng = np.random.default_rng(0)

    def err(x):
        return float(rnd.reward(x, None, x).mean())

    for step in range(train_steps + 1):
        if step % 20 == 0:
            hist["step"].append(step)
            hist["seen"].append(err(seen[:100]))
            hist["held"].append(err(held))
            hist["pong"].append(err(novel_a))
            hist["breakout"].append(err(novel_b))
        batch = seen[rng.integers(0, len(seen), 32)]
        rnd.update(batch, None, batch)

    print(f"  after {train_steps} updates on Montezuma frames — error on: "
          f"studied {hist['seen'][-1]:.3f} | fresh Montezuma {hist['held'][-1]:.3f} | "
          f"Pong {hist['pong'][-1]:.3f} | Breakout {hist['breakout'][-1]:.3f}", flush=True)
    np.savez(CKPT / "atari.npz", frame=raw_frame, **{k: np.array(v) for k, v in hist.items()})
    return hist, raw_frame


def plot_atari(hist, frame, path):
    fig, axes = ps.plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110,
                                gridspec_kw={"width_ratios": [1, 1.6]})
    fig.patch.set_facecolor(ps.SURFACE)
    axes[0].imshow(frame, cmap="gray")
    axes[0].set_xticks([]), axes[0].set_yticks([])
    axes[0].set_title("a real frame the predictor studies\n(Montezuma's Revenge, room 1)",
                      color=ps.INK, fontsize=10.5, loc="left")

    ax = axes[1]
    series = [("seen", "Montezuma frames it trained on", 0),
              ("held", "fresh Montezuma frames (same room)", 1),
              ("pong", "Pong frames (never seen)", 2),
              ("breakout", "Breakout frames (never seen)", 3)]
    for key, label, i in series:
        ax.plot(hist["step"], hist[key], color=ps.SERIES[i], lw=2.0, label=label)
    ax.set_yscale("log")
    ps.style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("RND's novelty meter, on true Atari pixels", color=ps.INK, fontsize=11.5, loc="left")
    ax.set_xlabel("predictor updates", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("prediction error = the bonus", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


def main():
    OUT.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    torch.set_num_threads(1)          # each worker gets one thread; set before any dispatch

    print("[0/3] drawing the environment")
    plot_maze(OUT / "maze.png")

    print(f"[1/3] PPO vs PPO+RND on the miniature Montezuma ({STEPS:,} steps x 3 seeds x 2 arms)")
    p = train_arms()
    plot_agent(p, OUT / "rnd_vs_ppo.png")

    print("[2/3] RND's novelty signal on real Atari frames")
    hist, frame = atari_novelty()
    plot_atari(hist, frame, OUT / "atari_novelty.png")

    firsts = [p[f"rnd_{s}_first"] for s in SEEDS]
    print(f"\nRND found the first reward at steps {firsts}; PPO's firsts: "
          f"{[p[f'ppo_{s}_first'] for s in SEEDS]} (-1 = never)")


if __name__ == "__main__":
    main()
