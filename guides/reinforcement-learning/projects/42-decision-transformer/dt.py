"""Project 42 — The Decision Transformer: offline RL as next-token prediction.

Everything else in Phase 7 argued about how to handle `max_a Q(s, a)` safely.
The Decision Transformer's answer is to not have a Q at all.

Instead it writes a trajectory as a sentence:

    R_1, s_1, a_1, R_2, s_2, a_2, ... R_t, s_t, a_t, ...

where `R_t` is the RETURN-TO-GO: the total reward the agent went on to collect
from step t to the end of the episode. It is a fact from the future, and offline
we HAVE the future — the episode is already recorded. Then it trains a GPT to
predict the next action, exactly like a language model predicts the next word.

At test time you write the first token yourself. You say "R = 3000" — I want 3000
reward from here — and let the model continue the sentence. It produces the actions
of an agent that was about to collect 3000, because that is the only kind of
continuation it has ever seen after that number.

    No Bellman equation. No bootstrapping. No out-of-distribution actions.
    The distribution-shift problem is not solved; it is never created.

THE EXPERIMENT: sweep what we ask for, and plot it against what we get. If the story
above is true, the two should track each other — up to the point where we ask for more
than the dataset has ever shown.

We run it on TWO datasets, because the first result is a flat line and the reason is
the most useful thing in the project:

    medium   100 episodes, all written by ONE half-trained policy. Their returns
             differ by 17%, and that difference is LUCK, not skill.
    mixed    the same 100, plus 100 random and 100 expert episodes. Now an episode's
             return genuinely tells you something about how it was driven.

Conditioning on a number can only work if the number MEANS something.

    python3 dt.py     # ~3 min: two GPTs, then 14 return targets swept in parallel
"""

import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "38-bc-baseline-on-d4rl"))
sys.path.insert(0, str(HERE.parent / "01-build-a-gridworld"))
import offline_lib as ol  # noqa: E402
import plot_style as ps  # noqa: E402

# A transformer is far heavier per gradient step than the little MLPs of projects
# 38-41, and this is a CPU. Unlike the parameter sweeps — which run N independent
# configs on N cores — this is ONE model, so the cores are free for it to use.
torch.set_num_threads(6)

OUT = HERE / "outputs"
LEVEL = "medium"

# Sizing, measured rather than guessed. The first attempt (K=20, 3 layers, 128-wide,
# 727k parameters) ran at 1.4 iterations/s: 12k steps would have taken 2.5 HOURS. The
# cost of a transformer step grows with (layers x width^2 x sequence length), and the
# sequence here is 3*K tokens because every timestep contributes THREE of them
# (return-to-go, state, action). So K is doubly expensive, and halving it is the
# cheapest lever available. These numbers fit the ten-minute budget with room to spare.
K = 10               # context: how many past timesteps the model can see -> 30 tokens
N_EMBD, N_LAYER, N_HEAD = 64, 2, 2
TRAIN_ITERS = 6_000  # x2, because this project trains two models. ~25 it/s on this CPU.
BATCH = 64
RTG_SCALE = 1000.0   # returns are in the thousands; networks prefer numbers near 1


# ==========================================================================
# The model — a small GPT, with three kinds of token
# ==========================================================================
class Block(nn.Module):
    """One transformer block: causal self-attention, then an MLP. Phase 2 material."""

    def __init__(self):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(N_EMBD), nn.LayerNorm(N_EMBD)
        self.attn = nn.MultiheadAttention(N_EMBD, N_HEAD, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(),
                                 nn.Linear(4 * N_EMBD, N_EMBD))

    def forward(self, x, mask):
        h = self.ln1(x)
        # `mask` is the causal mask. Without it every token could read the tokens
        # AFTER it — including the action it is being asked to predict — and the
        # model would score a perfect training loss by copying the answer, then
        # produce garbage at test time, when there is no answer to copy.
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class DecisionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim, max_t=1001):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        # Three separate embeddings, because the three token types mean different
        # things and share no vocabulary. A returns-to-go of 5.0 and a joint angle of
        # 5.0 are not the same "5".
        self.embed_rtg = nn.Linear(1, N_EMBD)
        self.embed_obs = nn.Linear(obs_dim, N_EMBD)
        self.embed_act = nn.Linear(act_dim, N_EMBD)
        # The timestep embedding is added to all three tokens of a step. Note this is
        # the position in the EPISODE, not the position in the context window — the
        # model should know it is at step 900 of 1000 even when it can only see 20 of
        # them, because "how much time is left" changes what a return-to-go means.
        self.embed_t = nn.Embedding(max_t, N_EMBD)
        self.ln = nn.LayerNorm(N_EMBD)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.head = nn.Linear(N_EMBD, act_dim)

    def forward(self, rtg, obs, act, timesteps):
        B, T = obs.shape[0], obs.shape[1]
        te = self.embed_t(timesteps)
        r = self.embed_rtg(rtg) + te
        s = self.embed_obs(obs) + te
        a = self.embed_act(act) + te
        # Interleave into (R_1, s_1, a_1, R_2, s_2, a_2, ...): stack on a new axis and
        # reshape, which is the whole trick for turning 3 sequences into 1.
        x = torch.stack([r, s, a], dim=2).reshape(B, 3 * T, N_EMBD)
        x = self.ln(x)
        mask = torch.triu(torch.ones(3 * T, 3 * T, dtype=torch.bool), diagonal=1)
        for b in self.blocks:
            x = b(x, mask)
        # Read the action prediction off the STATE token (index 1 of each triple): at
        # that point the model has seen R_t and s_t but not a_t, which is exactly the
        # information a policy is allowed to have.
        h = x.reshape(B, T, 3, N_EMBD)[:, :, 1]
        return torch.tanh(self.head(h))  # actions live in [-1, 1]


# ==========================================================================
# The data — trajectories, not transitions
# ==========================================================================
class TrajectoryData:
    """The same data as every other Phase 7 project, re-cut into episodes.

    The other algorithms sample independent (s, a, r, s') rows and never care which
    episode they came from. The DT cannot: its input IS a piece of an episode, in
    order, and its labels depend on what happened AFTERWARDS in that same episode.

    `levels` may name more than one dataset. Passing all three builds the `mixed`
    dataset, and the whole second half of this project is about why that matters.

    HalfCheetah cannot fall over, so every episode is exactly 1,000 steps. That is a
    gift: all trajectories are the same length, so they stack into one solid array and
    a batch is a single fancy-index instead of a Python loop over 64 samples. (An
    environment with early termination would need ragged episodes and a padding mask.)
    """

    def __init__(self, levels):
        self.levels = [levels] if isinstance(levels, str) else list(levels)
        raws = [ol.build_dataset(lv) for lv in self.levels]
        # Normalization over the UNION of whatever we were given. Computing it per
        # level and then mixing would hand the model a free label — "these numbers are
        # scaled like the expert set" — and it would read the answer off that instead
        # of off the return-to-go.
        all_obs = np.concatenate([r["obs"] for r in raws])
        self.obs_mean = all_obs.mean(0, keepdims=True)
        self.obs_std = all_obs.std(0, keepdims=True) + 1e-3

        obs, act, rtg = [], [], []
        for raw in raws:
            ends = np.nonzero((raw["terminal"] + raw["timeout"]) > 0)[0]
            start = 0
            for e in ends:
                e = int(e) + 1
                r = raw["rew"][start:e]
                # The return-to-go at step t = the sum of all rewards from t to the end
                # of the episode: a reversed cumulative sum. This one line is what makes
                # the whole approach possible, and it is only computable because the
                # episode is already OVER. An online agent could never write it down.
                obs.append((raw["obs"][start:e] - self.obs_mean) / self.obs_std)
                act.append(raw["act"][start:e])
                rtg.append(np.cumsum(r[::-1])[::-1].copy())
                start = e
        self.L = min(len(o) for o in obs)              # 1,000
        self.obs = np.stack([o[:self.L] for o in obs]).astype(np.float32)
        self.act = np.stack([a[:self.L] for a in act]).astype(np.float32)
        self.rtg = np.stack([g[:self.L] for g in rtg]).astype(np.float32)
        self.rets = self.rtg[:, 0].copy()              # RTG at step 0 IS the episode return
        self.n_traj, self.obs_dim = len(self.obs), self.obs.shape[2]
        self.act_dim = self.act.shape[2]

    def batch(self, size, rng):
        """`size` random K-step windows, gathered in one shot with no Python loop."""
        i = rng.integers(0, self.n_traj, size)                    # which episode
        j = rng.integers(0, self.L - K, size)                     # where in it
        w = j[:, None] + np.arange(K)[None, :]                    # (size, K) time indices
        ii = i[:, None]
        return (torch.from_numpy(self.rtg[ii, w][..., None] / RTG_SCALE),
                torch.from_numpy(self.obs[ii, w]),
                torch.from_numpy(self.act[ii, w]),
                torch.from_numpy(w.astype(np.int64)))


# ==========================================================================
# Evaluation — this is where you TELL it what you want
# ==========================================================================
@torch.no_grad()
def rollout(model, data, target_return, episodes=5, seed=0):
    """Ask for `target_return`, and see what actually comes back."""
    env = gym.make(ol.ENV_ID)
    got = []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + 10_000 + ep)
        obs_h = np.zeros((1, 0, data.obs_dim), np.float32)
        act_h = np.zeros((1, 0, data.act_dim), np.float32)
        rtg_h = np.zeros((1, 0, 1), np.float32)
        t_h = np.zeros((1, 0), np.int64)
        rtg, total, done, t = target_return, 0.0, False, 0
        while not done:
            on = ((o - data.obs_mean[0]) / data.obs_std[0]).astype(np.float32)
            obs_h = np.concatenate([obs_h, on[None, None]], 1)
            rtg_h = np.concatenate([rtg_h, np.array([[[rtg / RTG_SCALE]]], np.float32)], 1)
            # The action slot for the CURRENT step is a placeholder. The model must
            # not see it (it is what we are asking for), and the causal mask is what
            # guarantees it does not: the state token at index 1 of the triple can
            # only attend to tokens at or before it.
            act_h = np.concatenate([act_h, np.zeros((1, 1, data.act_dim), np.float32)], 1)
            t_h = np.concatenate([t_h, np.array([[min(t, 1000)]], np.int64)], 1)

            a = model(torch.from_numpy(rtg_h[:, -K:]), torch.from_numpy(obs_h[:, -K:]),
                      torch.from_numpy(act_h[:, -K:]), torch.from_numpy(t_h[:, -K:]))
            a = a[0, -1].numpy()
            act_h[0, -1] = a  # write the real action back into the history

            o, r, te, tr, _ = env.step(a)
            total += r
            # Spend the budget: what we asked for, minus what we just earned. If we
            # asked for 3000 and collected 4 this step, we now want 2996 from here.
            # This is the "conditioning" doing its work, one step at a time.
            rtg -= r
            t += 1
            done = te or tr
        got.append(total)
    env.close()
    return float(np.mean(got)), float(np.std(got))


def eval_target(args):
    """One (model, return target) pair, in its own process."""
    tag, levels, sd, target, obs_dim, act_dim = args
    torch.set_num_threads(1)
    model = DecisionTransformer(obs_dim, act_dim)
    model.load_state_dict(sd)
    model.eval()
    data = TrajectoryData(levels)
    mean, std = rollout(model, data, target, episodes=5)
    return tag, target, mean, std


def train_dt(data, tag):
    """Train one Decision Transformer. Plain supervised learning, start to finish."""
    ol.set_seed(0)   # BEFORE the network is built
    rng = np.random.default_rng(0)
    model = DecisionTransformer(data.obs_dim, data.act_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=TRAIN_ITERS,
                                                pct_start=0.1)
    t0, losses = time.time(), []
    for it in range(1, TRAIN_ITERS + 1):
        R, S, A, T = data.batch(BATCH, rng)
        pred = model(R, S, A, T)
        # Every window is a real, full window (see TrajectoryData), so there is no
        # padding to mask out — this is a plain mean squared error on the actions.
        loss = ((pred - A) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if it % 2000 == 0:
            print(f"  [{tag:6s}] step {it:5d}  action MSE {np.mean(losses[-500:]):.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
    model.eval()
    return model, losses


def main():
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    lo, hi = ol.score_bounds()

    # TWO datasets, and the difference between them is the whole project.
    #   medium : 100 episodes, all written by ONE half-trained policy.
    #   mixed  : those same 100, plus 100 random and 100 expert episodes. Same task,
    #            same robot — but now the returns range from -283 to 5,131, and an
    #            episode's return actually tells you something about how it was DRIVEN.
    sets = {"medium": TrajectoryData("medium"),
            "mixed": TrajectoryData(["random", "medium", "expert"])}
    for tag, d in sets.items():
        print(f"{tag:7s}: {d.n_traj:3d} episodes, returns {d.rets.min():7.0f} .. "
              f"{d.rets.max():7.0f}  (spread {d.rets.std():6.0f})")
    print(f"teachers: random {lo:.0f}, expert {hi:.0f}\n", flush=True)

    models, losses = {}, {}
    n_par = sum(p.numel() for p in DecisionTransformer(sets["medium"].obs_dim,
                                                       sets["medium"].act_dim).parameters())
    print(f"training two {n_par / 1e3:.0f}k-parameter GPTs, {TRAIN_ITERS} steps each...",
          flush=True)
    for tag, d in sets.items():
        models[tag], losses[tag] = train_dt(d, tag)

    # ---- THE experiment: ask for a range of returns, measure what actually comes back ----
    targets = [-250, 0, 1000, 2000, 3000, 4000, 5000]
    print(f"\n[{time.time() - t0:.0f}s] sweeping return targets on both models...", flush=True)
    jobs = [(tag, sets[tag].levels,
             {k: v.clone() for k, v in models[tag].state_dict().items()},
             float(t), sets[tag].obs_dim, sets[tag].act_dim)
            for tag in sets for t in targets]
    with ProcessPoolExecutor(max_workers=12) as pool:
        res = list(pool.map(eval_target, jobs))
    got = {(tag, t): (m, s) for tag, t, m, s in res}

    print("\n" + "=" * 78)
    print(f"{'you ASK for':>12s} | {'medium-only DT':>22s} | {'mixed-data DT':>22s}")
    print(f"{'':12s} | {'you GET':>13s} {'score':>8s} | {'you GET':>13s} {'score':>8s}")
    print("-" * 78)
    for t in targets:
        m1, _ = got[("medium", t)]
        m2, _ = got[("mixed", t)]
        print(f"{t:12.0f} | {m1:13.1f} {ol.normalized_score(m1):8.1f} | "
              f"{m2:13.1f} {ol.normalized_score(m2):8.1f}")
    print("=" * 78)

    def spread(tag):
        v = [got[(tag, t)][0] for t in targets]
        return max(v) - min(v)

    print(f"\nAsking for {targets[0]} vs {targets[-1]} — how much does the answer CHANGE?")
    print(f"  medium-only DT:  the whole sweep moves it by {spread('medium'):6.0f} points."
          f"  It is ignoring you.")
    print(f"  mixed-data DT:   the whole sweep moves it by {spread('mixed'):6.0f} points."
          f"  It is listening.")
    best = max(targets, key=lambda t: got[("mixed", t)][0])
    bm = got[("mixed", best)][0]
    print(f"\nBest mixed-data result: ask for {best:.0f} -> get {bm:.0f} "
          f"(score {ol.normalized_score(bm):.1f}).")
    print(f"[{time.time() - t0:.0f}s total]")

    # ---- plots ----
    fig, axes = ps.plt.subplots(1, 3, figsize=(16.5, 4.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    COL = {"medium": ps.SERIES[2], "mixed": ps.SERIES[1]}

    # (1) both train perfectly well. Training loss tells you NOTHING about which one
    #     will obey you — which is exactly the trap.
    ax = ps.style_axes(axes[0])
    for tag in sets:
        sm = np.convolve(losses[tag], np.ones(200) / 200, mode="valid")
        ax.plot(sm, color=COL[tag], lw=1.8, label=f"trained on `{tag}`")
    ax.set_yscale("log")
    ax.set_title("The BETTER-fitting model is the useless one", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("gradient steps", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("action prediction MSE (log)", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)

    # (2) the actual difference between the two datasets
    ax = ps.style_axes(axes[1])
    ax.hist(sets["mixed"].rets, bins=40, color=ps.SERIES[1], alpha=0.75,
            label="`mixed` — three skill levels")
    ax.hist(sets["medium"].rets, bins=15, color=ps.SERIES[2], alpha=0.9,
            label="`medium` — one policy")
    ax.set_title("...but they are not the same kind of data", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("episode return", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("number of episodes", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8.5, frameon=False)

    # (3) THE money panel: ask-vs-get, for both.
    ax = ps.style_axes(axes[2])
    lim = [-600, 5600]
    ax.plot(lim, lim, ls="--", lw=1.5, color=ps.INK_MUTED, label="perfect obedience (y = x)")
    for tag in sets:
        g = np.array([got[(tag, t)][0] for t in targets])
        e = np.array([got[(tag, t)][1] for t in targets])
        ax.errorbar(targets, g, yerr=e, marker="o", ms=6, lw=2.4, capsize=3,
                    color=COL[tag], label=f"DT trained on `{tag}`")
        ax.axvline(sets[tag].rets.max(), color=COL[tag], ls=":", lw=1.4)
    ax.text(sets["medium"].rets.max() + 60, -450, "best `medium`\nepisode",
            color=COL["medium"], fontsize=7.5)
    ax.text(sets["mixed"].rets.max() - 1500, -450, "best `mixed` episode",
            color=COL["mixed"], fontsize=7.5)
    ax.set_xlim(lim)
    ax.set_title("Ask, and — only sometimes — receive", color=ps.INK, fontsize=12,
                 loc="left", pad=10)
    ax.set_xlabel("return you ASK for (the first token)", color=ps.INK_SECONDARY, fontsize=10)
    ax.set_ylabel("return you GET", color=ps.INK_SECONDARY, fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "dt_return_conditioning.png", facecolor=ps.SURFACE, bbox_inches="tight")
    print(f"wrote {OUT / 'dt_return_conditioning.png'}")


if __name__ == "__main__":
    main()
