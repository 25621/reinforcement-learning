r"""Project 47 — the Intrinsic Curiosity Module, next to RND on the same maze.

RND (project 46) asks: "can I reproduce a random fingerprint of this state?"
ICM asks a question that sounds much more sensible: "did the world do what I
expected it to do?" — and pays the agent for every surprise.

    phi        an encoder that turns an observation into a short feature vector
    inverse    from phi(s) and phi(s'), guess WHICH ACTION was taken
    forward    from phi(s) and the action, guess phi(s')
    bonus      = how wrong the forward model was

The inverse model is the part beginners skip, and it is the only reason ICM is
more than "pixel prediction". It is what forces phi to keep the parts of the
picture the agent can *control*: to name the action that moved you from s to s',
the features must encode the thing that moved (you), and they can safely throw
away anything the action does not touch. Project 49 turns that difference into a
failure you can watch.

    python3 icm.py            # ~8 min: 3 arms x 3 seeds, plus bonus heat-maps
    python3 icm.py --plot     # redraw the figures from saved curves, no retraining
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
sys.path.insert(0, str(HERE.parent / "46-rnd-on-atari"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import explore_lib as el  # noqa: E402
import plot_style as ps  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 300_000
SEEDS = (0, 1, 2)


class ICM:
    """Intrinsic Curiosity Module (Pathak et al., 2017)."""

    name = "PPO + ICM"

    def __init__(self, in_ch, n_actions, obs_shape, feat_dim=64, lr=1e-3, eta=1.0, beta=0.2):
        self.n_actions = n_actions
        self.eta = eta
        self.beta = beta                       # how the two losses share the encoder's time
        self.phi = el.cnn_trunk(in_ch, feat_dim)
        self.inverse = nn.Sequential(nn.Linear(2 * feat_dim, 128), nn.ReLU(),
                                     nn.Linear(128, n_actions))
        self.forward_net = nn.Sequential(nn.Linear(feat_dim + n_actions, 128), nn.ReLU(),
                                         nn.Linear(128, feat_dim))
        params = (list(self.phi.parameters()) + list(self.inverse.parameters())
                  + list(self.forward_net.parameters()))
        self.opt = torch.optim.Adam(params, lr=lr)
        self.inv_acc = 0.0                     # running accuracy of the inverse model

    def _feats(self, obs, next_obs):
        o = torch.as_tensor(obs)
        o2 = torch.as_tensor(next_obs)
        return self.phi(o), self.phi(o2)

    def reward(self, obs, act, next_obs):
        with torch.no_grad():
            f, f2 = self._feats(obs, next_obs)
            a1h = F.one_hot(torch.as_tensor(act), self.n_actions).float()
            pred = self.forward_net(torch.cat([f, a1h], -1))
            err = 0.5 * (pred - f2).pow(2).sum(-1)
        return (self.eta * err).numpy()

    def update(self, obs, act, next_obs):
        f, f2 = self._feats(obs, next_obs)
        a = torch.as_tensor(act)
        a1h = F.one_hot(a, self.n_actions).float()

        # (1) inverse loss — trains the ENCODER. "Which of the 4 moves did I just make?"
        logits = self.inverse(torch.cat([f, f2], -1))
        inv_loss = F.cross_entropy(logits, a)
        self.inv_acc = 0.9 * self.inv_acc + 0.1 * float((logits.argmax(-1) == a).float().mean())

        # (2) forward loss — trains the PREDICTOR, on DETACHED features.
        #
        # The detach is not a performance trick, it is what keeps ICM alive. If the
        # forward loss were allowed to reshape phi, there is a trivial way to make the
        # prediction error zero forever: map every observation to the same constant
        # vector. Perfect predictions, zero curiosity, dead agent. Detaching means the
        # forward model must chase a feature space it cannot bend to its convenience,
        # and phi answers only to the inverse model, which cannot be gamed that way.
        pred = self.forward_net(torch.cat([f.detach(), a1h], -1))
        fwd_loss = 0.5 * (pred - f2.detach()).pow(2).sum(-1).mean()

        loss = (1 - self.beta) * inv_loss + self.beta * fwd_loss
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


class PixelForward:
    """The naive ancestor of ICM: predict the raw next FRAME, be paid for the error.

    No encoder, no inverse model — curiosity measured directly in pixel space. It is
    the honest control for what ICM's feature space actually buys, and project 49 uses
    it in the maze with the noisy television. Do not assume it collapses there: it was
    trapped by the TV in only 1 of 3 seeds, and it was the only bonus that solved that
    maze at all. Being crude is not the same as being useless.
    """

    name = "PPO + pixel prediction"

    def __init__(self, in_ch, n_actions, obs_shape, lr=1e-3, eta=1.0):
        self.n_actions = n_actions
        self.eta = eta
        self.in_ch = in_ch
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(obs_shape + n_actions, 256), nn.ReLU(),
            nn.Linear(256, obs_shape),
        )
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.shape = obs_shape

    def _pred(self, obs, act):
        o = torch.as_tensor(obs).reshape(len(obs), -1)
        a1h = F.one_hot(torch.as_tensor(act), self.n_actions).float()
        return self.net(torch.cat([o, a1h], -1))

    def reward(self, obs, act, next_obs):
        with torch.no_grad():
            pred = self._pred(obs, act)
            tgt = torch.as_tensor(next_obs).reshape(len(next_obs), -1)
            err = 0.5 * (pred - tgt).pow(2).sum(-1)
        return (self.eta * err).numpy()

    def update(self, obs, act, next_obs):
        pred = self._pred(obs, act)
        tgt = torch.as_tensor(next_obs).reshape(len(next_obs), -1)
        loss = 0.5 * (pred - tgt).pow(2).sum(-1).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())


ARMS = {
    "none": None,
    "rnd": lambda c, a, s: el.RND(c, a, s),
    "icm": lambda c, a, s: ICM(c, a, s),
}


def probe(bonus):
    """Snapshot of the bonus over the maze + the inverse model's accuracy."""
    heat = el.bonus_map(bonus)
    acc = getattr(bonus, "inv_acc", np.nan)
    return np.concatenate([heat.reshape(-1), [acc]])


def run_one(args):
    arm, seed = args
    h = el.train(make_bonus=ARMS[arm], total_steps=STEPS, seed=seed, verbose=False, probe=probe)
    print(f"  {h['name']:22s} seed {seed}: success {h['success'][-1]:.2f}, "
          f"coverage {h['coverage'][-1]}/122, first reward @ {h['first_success']}", flush=True)
    return arm, seed, h


def main(replot=False):
    OUT.mkdir(exist_ok=True)
    CKPT.mkdir(exist_ok=True)
    torch.set_num_threads(1)

    if replot:                    # redraw from the saved curves, no retraining
        p = dict(np.load(CKPT / "curves.npz"))
        plot_curves(p, OUT / "icm_vs_rnd.png")
        plot_heat(p, OUT / "bonus_maps.png")
        plot_inverse(p, OUT / "inverse_accuracy.png")
        return

    jobs = [(arm, s) for arm in ARMS for s in SEEDS]
    print(f"[1/2] {len(jobs)} runs x {STEPS:,} steps (no bonus / RND / ICM)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=9) as ex:
        results = list(ex.map(run_one, jobs))
    print(f"  ({time.time() - t0:.0f}s)")

    p = {}
    for arm, seed, h in results:
        for k in ("success", "coverage", "bonus"):
            p[f"{arm}_{seed}_{k}"] = h[k]
        p[f"{arm}_{seed}_probe"] = np.stack(h["probe"])
        p[f"{arm}_{seed}_first"] = -1 if h["first_success"] is None else h["first_success"]
        p["steps"] = h["steps"]
    np.savez(CKPT / "curves.npz", **p)

    print("[2/2] figures")
    plot_curves(p, OUT / "icm_vs_rnd.png")
    plot_heat(p, OUT / "bonus_maps.png")
    plot_inverse(p, OUT / "inverse_accuracy.png")

    for arm in ARMS:
        f = [p[f"{arm}_{s}_first"] for s in SEEDS]
        print(f"  {arm:5s} first reward at steps {f}  (-1 = never in {STEPS:,})")


def plot_curves(p, path):
    fig, axes = ps.plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    steps = np.array(p["steps"]) / 1000
    labels = {"none": "PPO only", "rnd": "PPO + RND", "icm": "PPO + ICM"}
    for ax, key, ylab, title in [
        (axes[0], "success", "episodes solved (fraction)", "Both curiosities solve it; PPO alone never does"),
        (axes[1], "coverage", "distinct states visited (of 122)", "And both do it by covering the maze"),
    ]:
        for i, arm in enumerate(ARMS):
            runs = np.stack([p[f"{arm}_{s}_{key}"] for s in SEEDS])
            for r in runs:
                ax.plot(steps, r, color=ps.SERIES[i], lw=0.9, alpha=0.3)
            ax.plot(steps, runs.mean(0), color=ps.SERIES[i], lw=2.4, label=labels[arm])
        ps.style_axes(ax)
        ax.set_title(title, color=ps.INK, fontsize=11.5, loc="left")
        ax.set_xlabel("environment steps (thousands)", color=ps.INK_SECONDARY, fontsize=10)
        ax.set_ylabel(ylab, color=ps.INK_SECONDARY, fontsize=10)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


def plot_heat(p, path):
    """Where each bonus is pointing, early and late.

    Each panel is divided by its own median cell, so a value of 2 means "twice what an
    ordinary cell pays". That normalization is the point: the two bonuses have completely
    different units (RND's raw errors are ~1e-4, ICM's are ~20), and only the SHAPE of the
    map — which cells outbid which — decides where the agent goes.
    """
    env = el.KeyDoorRoom()
    h, w = env.h, env.w
    far = np.zeros((h, w), bool)
    home = np.zeros((h, w), bool)
    for y in range(h):
        for x in range(w):
            if env.grid[y, x] != "#":
                if x > 10:
                    far[y, x] = True
                elif x < 6:
                    home[y, x] = True

    fig, axes = ps.plt.subplots(2, 2, figsize=(9.8, 5.8), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    n_probe = len(p["rnd_0_probe"])
    early, late = max(n_probe // 8, 1), n_probe - 1
    for r, arm in enumerate(("rnd", "icm")):
        maps = p[f"{arm}_{SEEDS[0]}_probe"]
        for c, (t_i, when) in enumerate([(early, "early"), (late, "end of training")]):
            heat = maps[t_i][:-1].reshape(h, w)
            rel = heat / (np.nanmedian(heat) + 1e-12)
            ratio = np.nanmean(rel[far]) / np.nanmean(rel[home])
            ax = axes[r][c]
            im = ax.imshow(rel, cmap="magma", vmin=0, vmax=2.5)
            ax.set_xticks([]), ax.set_yticks([])
            ax.set_title(f"{'RND' if arm == 'rnd' else 'ICM'}, {when} — "
                         f"far room pays {ratio:.2f}x the home room",
                         color=ps.INK, fontsize=10.5)
            fig.colorbar(im, ax=ax, shrink=0.85, extend="max")
    fig.suptitle("What each bonus pays for arriving in each cell "
                 "(x its own median cell). The far room is on the right.",
                 color=ps.INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=ps.SURFACE, bbox_inches="tight")
    ps.plt.close(fig)
    print(f"wrote {path}")


def plot_inverse(p, path):
    fig, ax = ps.new_axes(7.2, 4.0)
    steps = np.array(p["steps"]) / 1000
    accs = np.stack([p[f"icm_{s}_probe"][:, -1] for s in SEEDS])
    ax.plot(steps, accs.mean(0), color=ps.SERIES[2], lw=2.4,
            label="ICM inverse model: 'which move did I just make?'")
    ax.axhline(0.25, color=ps.INK_MUTED, ls="--", lw=1.3, label="chance (1 of 4 actions)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ps.finish(fig, ax, "The inverse model works — so the features it trains keep what the agent controls",
              "environment steps (thousands)", "accuracy", path)


if __name__ == "__main__":
    main(replot="--plot" in sys.argv)
