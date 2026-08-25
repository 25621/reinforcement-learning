"""Project 40 — a video model that takes a button press.

    python3 run.py --stage data              # ~1 min
    python3 run.py --stage train             # ~7 min  (all three arms)
    python3 run.py --stage eval              # ~2 min
    python3 run.py --stage figures           # ~1 min
    python3 run.py --stage longer            # ~7 min  (optional fairness check)

Three models, same size, same data, same number of optimiser steps:

    diff   flow-matching (generative), knows which buttons were pressed
    noact  flow-matching (generative), does NOT know the buttons
    mse    one-shot regression,        knows which buttons were pressed

`noact` answers "does action conditioning actually do anything?".
`mse`   answers "do we need a generative model, or would a plain predictor do?".
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

import world_lib as W                                          # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

torch.set_num_threads(12)

N_TRAIN_EP = 2500
N_TEST_EP = 200
CTX, HORIZON = 2, 4
STEPS, BATCH, LR = 3000, 64, 4e-4
SAMPLE_STEPS = 30
ARMS = ["diff", "noact", "mse"]
ARM_LABEL = {"diff": "diffusion + action", "noact": "diffusion, no action",
             "mse": "regression + action"}


# --------------------------------------------------------------------------
# stage: data
# --------------------------------------------------------------------------

def stage_data():
    t0 = time.time()
    train = W.record(N_TRAIN_EP, seed=0)
    test = W.record(N_TEST_EP, seed=999)
    W.save_episodes(train, "train")
    W.save_episodes(test, "test")
    coins = train["rew"].sum()
    blocked = train["blocked"].mean()
    print(f"train {N_TRAIN_EP} episodes x {W.EP_LEN} frames  ({time.time()-t0:.1f}s)")
    print(f"  coins collected      : {int(coins)} "
          f"({coins / train['rew'].size:.3f} per transition)")
    print(f"  blocked-by-wall moves: {blocked:.3f} of transitions")
    # a picture of the real game
    e = 0
    idx = np.arange(24)
    frames = W.frames_of(train, np.full(24, e), idx)
    W.write_gif(frames, OUT / "real_play.gif")
    W.strip_image([list(frames[:8]), list(frames[8:16])],
                  OUT / "the_game.png")
    with open(OUT / "data.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["episodes", "frames_per_episode", "coins_per_transition",
                     "blocked_fraction"])
        wr.writerow([N_TRAIN_EP, W.EP_LEN, f"{coins / train['rew'].size:.4f}",
                     f"{blocked:.4f}"])


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train_arm(arm, ep, log_rows, steps=STEPS, save_as=None):
    torch.manual_seed(40)
    rng = np.random.default_rng(40)
    net = W.ActionUNet(ctx=CTX, horizon=HORIZON,
                       use_action=(arm != "noact"))
    flow = W.FL.RectifiedFlow()
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    STEPS_ = steps
    t0 = time.time()
    for step in range(1, STEPS_ + 1):
        ctx, tgt, act, _ = W.sample_batch(ep, BATCH, CTX, HORIZON, rng)
        if arm == "mse":
            # No noise, no timestep: predict the future in one shot.
            pred = net(torch.zeros_like(tgt), torch.zeros(BATCH), ctx, act)
            loss = F.mse_loss(pred, tgt)
        else:
            noise = torch.randn_like(tgt)
            t = flow.sample_t(BATCH)
            xt = flow.interpolate(tgt, t, noise)
            v = net(xt, t * flow.T_SCALE, ctx, act)
            loss = F.mse_loss(v, flow.target(tgt, noise))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 250 == 0:
            log_rows.append([arm, step, f"{loss.item():.5f}"])
            print(f"  {arm} {step:5d}/{STEPS_} loss {loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)")
    name = save_as or arm
    torch.save(net.state_dict(), CK / f"{name}.pt")
    print(f"  {name}: {W.count_params(net)} params, "
          f"{time.time()-t0:.0f}s -> {CK / f'{name}.pt'}")


def stage_train():
    ep = W.load_episodes("train")
    rows = []
    for arm in ARMS:
        train_arm(arm, ep, rows)
    with open(OUT / "loss.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arm", "step", "loss"])
        wr.writerows(rows)


def stage_longer():
    """Optional check: is the diffusion arm merely under-trained?

    Diffusion losses fall much more slowly than a regression loss, so the first
    objection to project 40's headline result is 'you just did not train it
    long enough'.  This stage gives the same model 3x the optimiser steps and
    re-scores it, so the objection can be answered with a number rather than an
    opinion.  ~7 minutes.
    """
    ep = W.load_episodes("train")
    log = []
    train_arm("diff", ep, log, steps=STEPS * 3, save_as="diff_long")
    ctx, tgt, act, respawn, score = build_scorer()
    net = load_arm("diff_long")
    g = torch.Generator().manual_seed(40)
    t0 = time.time()
    pred = W.sample_frames(net, ctx, act, steps=SAMPLE_STEPS,
                           generator=g).numpy()
    acc_k, r = score(pred, time.time() - t0)
    print(f"diff_long acc {r['acc']:.3f}  per-frame {np.round(acc_k, 3)}  "
          f"blocked {r['blocked_acc']:.3f}  has-coin respawn "
          f"{r['has_coin_respawn']:.2f} / normal {r['has_coin_normal']:.2f}")
    with open(OUT / "longer.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["arm", "train_steps"] +
                            list(r.keys()))
        wr.writeheader()
        wr.writerow({"arm": "diff_long", "train_steps": STEPS * 3,
                     **{k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()}})
    print(f"wrote {OUT / 'longer.csv'}")


def load_arm(arm):
    net = W.ActionUNet(ctx=CTX, horizon=HORIZON, use_action=(arm != "noact"))
    net.load_state_dict(torch.load(CK / f"{arm}.pt"))
    net.eval()
    return net


@torch.no_grad()
def predict(net, arm, ctx, act, steps=SAMPLE_STEPS, generator=None):
    if arm == "mse":
        return net(torch.zeros(ctx.shape[0], HORIZON, W.GRID, W.GRID),
                   torch.zeros(ctx.shape[0]), ctx, act)
    return W.sample_frames(net, ctx, act, steps=steps, generator=generator)


# --------------------------------------------------------------------------
# stage: eval
# --------------------------------------------------------------------------

N_EVAL = 256


def eval_windows(ep, n, seed=7):
    """Fixed evaluation set: the same windows for every arm."""
    rng = np.random.default_rng(seed)
    return W.sample_batch(ep, n, CTX, HORIZON, rng)


def build_scorer():
    """The fixed evaluation set plus a function that scores a prediction."""
    ep = W.load_episodes("test")
    ctx, tgt, act, _ = eval_windows(ep, N_EVAL)
    # ground truth read straight out of the last context frame
    truth = []
    for b in range(N_EVAL):
        sym, ag, _, _ = W.read_frame(ctx[b, -1].numpy())
        walls = W.walls_from_symbols(sym)
        path, blocked = W.ground_truth_rollout(walls, (ag // W.GRID,
                                                       ag % W.GRID),
                                               None, act[b].numpy())
        truth.append((walls, path, blocked))
    # Which target frames sit AFTER a coin respawn?  Not just the frame where
    # the coin jumps: once it has jumped to a random cell, every later frame in
    # the window is equally unguessable, so the flag has to carry forward.
    respawn = np.zeros((N_EVAL, HORIZON), dtype=bool)
    for b in range(N_EVAL):
        prev = W.read_frame(ctx[b, -1].numpy())[2]
        seen = False
        for k in range(HORIZON):
            now = W.read_frame(tgt[b, k].numpy())[2]
            seen = seen or (now != prev)
            respawn[b, k] = seen
            prev = now

    def score(pred, dt):
        acc_k = np.zeros(HORIZON)
        blk_ok = blk_n = free_ok = free_n = 0
        snaps = []
        has_r, has_n, peak_r, peak_n = [], [], [], []
        for b in range(N_EVAL):
            walls, path, blocked = truth[b]
            for k in range(HORIZON):
                f = pred[b, k]
                sym, ai, ci, sn = W.read_frame(f)
                hit = (ai // W.GRID, ai % W.GRID) == path[k]
                acc_k[k] += hit
                if blocked[k]:
                    blk_n += 1; blk_ok += hit
                else:
                    free_n += 1; free_ok += hit
                snaps.append(sn)
                has, peak, _ = W.coin_report(f, walls)
                (has_r if respawn[b, k] else has_n).append(has)
                (peak_r if respawn[b, k] else peak_n).append(peak)
        acc_k /= N_EVAL
        return acc_k, dict(
            acc=float(acc_k.mean()), acc1=float(acc_k[0]),
            acc4=float(acc_k[-1]),
            blocked_acc=blk_ok / max(blk_n, 1),
            free_acc=free_ok / max(free_n, 1),
            snap=float(np.mean(snaps)),
            has_coin_respawn=float(np.mean(has_r)),
            has_coin_normal=float(np.mean(has_n)),
            coin_peak_respawn=float(np.mean(peak_r)),
            coin_peak_normal=float(np.mean(peak_n)),
            secs_per_window=dt / N_EVAL)

    return ctx, tgt, act, respawn, score


def stage_eval():
    ctx, tgt, act, respawn, score = build_scorer()
    rows, per_frame = [], {}
    for arm in ARMS:
        net = load_arm(arm)
        g = torch.Generator().manual_seed(40)
        t0 = time.time()
        pred = predict(net, arm, ctx, act, generator=g).numpy()
        acc_k, r = score(pred, time.time() - t0)
        r = dict(arm=arm, sample_steps=(0 if arm == "mse" else SAMPLE_STEPS),
                 **r)
        rows.append(r)
        per_frame[arm] = acc_k
        print(f"{arm:6s} acc {r['acc']:.3f}  per-frame {np.round(acc_k, 3)}  "
              f"blocked {r['blocked_acc']:.3f}  snap {r['snap']:.4f}  "
              f"has-coin respawn {r['has_coin_respawn']:.2f} / normal "
              f"{r['has_coin_normal']:.2f}  peak-on-respawn "
              f"{r['coin_peak_respawn']:.3f}")

    # is the diffusion arm limited by the model or by the sampler?
    net = load_arm("diff")
    for st in [4, 8, 60, 100]:
        g = torch.Generator().manual_seed(40)
        t0 = time.time()
        pred = W.sample_frames(net, ctx, act, steps=st, generator=g).numpy()
        _, r = score(pred, time.time() - t0)
        rows.append(dict(arm="diff", sample_steps=st, **r))
        print(f"diff@{st:3d} steps  acc {r['acc']:.3f}  "
              f"has-coin respawn {r['has_coin_respawn']:.2f}")

    with open(OUT / "eval.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                         for k, v in r.items()})
    np.savez(CK / "per_frame.npz", **per_frame)

    # ---- counterfactual: same screen, four different buttons -------------
    branch = np.zeros((W.N_ACT, W.N_ACT))          # commanded x realised
    n_free = np.zeros(W.N_ACT)
    net = load_arm("diff")
    ctx_c = ctx[:64]
    for a in range(W.N_ACT):
        acts = torch.full((64, HORIZON), a, dtype=torch.long)
        g = torch.Generator().manual_seed(100 + a)
        pr = W.sample_frames(net, ctx_c, acts, steps=SAMPLE_STEPS,
                             generator=g).numpy()
        for b in range(64):
            sym, ag, _, _ = W.read_frame(ctx_c[b, -1].numpy())
            walls = W.walls_from_symbols(sym)
            r0, c0 = ag // W.GRID, ag % W.GRID
            dr, dc = W.DELTA[a]
            if walls[r0 + dr, c0 + dc] == 1:
                continue                    # blocked: nothing to steer
            n_free[a] += 1
            _, ai, _, _ = W.read_frame(pr[b, 0])
            r1, c1 = ai // W.GRID, ai % W.GRID
            move = (r1 - r0, c1 - c0)
            if move in W.DELTA:
                branch[a, W.DELTA.index(move)] += 1
    branch = branch / np.maximum(n_free[:, None], 1)
    np.save(CK / "branch.npy", branch)
    print("counterfactual (rows = button, cols = direction actually moved):")
    print(np.round(branch, 2))


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

def stage_figures():
    all_rows = list(csv.DictReader(open(OUT / "eval.csv")))
    rows = [r for r in all_rows if int(r["sample_steps"]) in (0, SAMPLE_STEPS)]
    sweep = [r for r in all_rows if r["arm"] == "diff"]
    sweep.sort(key=lambda r: int(r["sample_steps"]))
    per_frame = np.load(CK / "per_frame.npz")
    branch = np.load(CK / "branch.npy")

    # 1. obedience
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax = axes[0]
    ps.style_axes(ax)
    x = np.arange(len(ARMS))
    ax.bar(x - 0.2, [float(r["free_acc"]) for r in rows], 0.38,
           color=ps.SERIES[0], label="move is possible")
    ax.bar(x + 0.2, [float(r["blocked_acc"]) for r in rows], 0.38,
           color=ps.SERIES[2], label="wall in the way")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[r["arm"]] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Is the player where the button says?", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_ylabel("cell-exact accuracy", color=ps.INK_SECONDARY, fontsize=9)

    ax = axes[1]
    ps.style_axes(ax)
    for i, arm in enumerate(ARMS):
        ax.plot(np.arange(1, HORIZON + 1), per_frame[arm], marker="o",
                color=ps.SERIES[i], label=ARM_LABEL[arm])
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(1, HORIZON + 1))
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("...and how far into the future?", color=ps.INK,
                 fontsize=11, loc="left")
    ax.set_xlabel("predicted frame", color=ps.INK_SECONDARY, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "obedience.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'obedience.png'}")

    # 2. uncertainty: does the frame still contain a coin?
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    ax = axes[0]
    ps.style_axes(ax)
    ax.bar(x - 0.2, [float(r["has_coin_normal"]) for r in rows], 0.38,
           color=ps.SERIES[1], label="coin stayed put (predictable)")
    ax.bar(x + 0.2, [float(r["has_coin_respawn"]) for r in rows], 0.38,
           color=ps.SERIES[3], label="coin respawned (unpredictable)")
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[r["arm"]] for r in rows], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Is there a coin on the screen at all?",
                 color=ps.INK, fontsize=11, loc="left")
    ax.set_ylabel("frames containing a coin", color=ps.INK_SECONDARY,
                  fontsize=9)

    ax = axes[1]
    ps.style_axes(ax)
    ax.bar(x - 0.2, [float(r["coin_peak_normal"]) for r in rows], 0.38,
           color=ps.SERIES[1], label="coin stayed put")
    ax.bar(x + 0.2, [float(r["coin_peak_respawn"]) for r in rows], 0.38,
           color=ps.SERIES[3], label="coin respawned")
    ax.axhline(W.COIN, color=ps.BASELINE, ls="--", lw=1)
    ax.set_ylim(0, 0.82)
    ax.text(-0.4, W.COIN + 0.015, "a real coin sits here", color=ps.INK_MUTED,
            fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[r["arm"]] for r in rows], fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("...and how bright is the brightest candidate?",
                 color=ps.INK, fontsize=11, loc="left")
    ax.set_ylabel("peak value on a non-wall cell", color=ps.INK_SECONDARY,
                  fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "uncertainty.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'uncertainty.png'}")

    # 3. counterfactual heat map
    fig, ax = ps.new_axes(4.6, 4.2)
    im = ax.imshow(branch, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(W.ACT_NAMES, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(W.ACT_NAMES, fontsize=9)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{branch[i, j]:.2f}", ha="center", va="center",
                    color="w" if branch[i, j] < 0.6 else "k", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ps.finish(fig, ax, "Press a button, get that direction",
              "direction the player actually moved", "button pressed",
              OUT / "branching.png")

    # 3b. is the diffusion arm sampler-limited?
    fig, ax = ps.new_axes(6.4, 4.0)
    st = [int(r["sample_steps"]) for r in sweep]
    ax.plot(st, [float(r["acc"]) for r in sweep], marker="o",
            color=ps.SERIES[0], label="player in the right cell")
    ax.plot(st, [float(r["has_coin_respawn"]) for r in sweep], marker="s",
            color=ps.SERIES[3], label="a coin exists after a respawn")
    ax.axhline(float([r for r in rows if r["arm"] == "mse"][0]["acc"]),
               color=ps.SERIES[2], ls="--", lw=1.2)
    ax.text(st[0], float([r for r in rows if r["arm"] == "mse"][0]["acc"])
            - 0.06, "regression arm", color=ps.SERIES[2], fontsize=9)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ps.finish(fig, ax, "More denoising steps do not close the gap",
              "denoising steps", "score", OUT / "sampler_steps.png")

    # 4. picture strips
    ctx, tgt, act, respawn, _ = build_scorer()
    pick = [0, 1, 2, 3, 4, 5]
    rows_img = [[ctx[b, -1].numpy() for b in pick]]
    labels = ["context"]
    for arm in ARMS:
        net = load_arm(arm)
        g = torch.Generator().manual_seed(40)
        pr = predict(net, arm, ctx, act, generator=g).numpy()
        rows_img.append([pr[b, -1] for b in pick])
        labels.append(arm)
    rows_img.append([tgt[b, -1].numpy() for b in pick])
    labels.append("truth")
    W.strip_image(rows_img, OUT / "predictions.png")
    print("rows of predictions.png:", " | ".join(labels))

    # 4b. the money shot: what each arm draws when the coin has just respawned
    hit = np.nonzero(respawn[:, -1])[0][:6]
    rows_img = [[ctx[b, -1].numpy() for b in hit],
                [tgt[b, -1].numpy() for b in hit]]
    for arm in ["diff", "mse"]:
        net = load_arm(arm)
        g = torch.Generator().manual_seed(40)
        pr = predict(net, arm, ctx, act, generator=g).numpy()
        rows_img.append([pr[b, -1] for b in hit])
    W.strip_image(rows_img, OUT / "respawn.png")
    print("respawn.png rows: last context frame, truth, diffusion, regression"
          " -- all on windows where the coin respawned")

    # 5. the steering-wheel picture: one screen, four buttons
    net = load_arm("diff")
    b = 0
    ctx_c = ctx[b:b + 1]
    strip = [[ctx[b, 0].numpy(), ctx[b, 1].numpy(), None, None]]
    for a in range(W.N_ACT):
        acts = torch.full((1, HORIZON), a, dtype=torch.long)
        g = torch.Generator().manual_seed(7)
        pr = W.sample_frames(net, ctx_c, acts, steps=SAMPLE_STEPS,
                             generator=g).numpy()[0]
        strip.append(list(pr))
    W.strip_image(strip, OUT / "steering.png")
    print("steering.png rows: context, then hold up / down / left / right")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["data", "train", "eval", "figures", "longer"])
    a = p.parse_args()
    {"data": stage_data, "train": stage_train, "eval": stage_eval,
     "figures": stage_figures, "longer": stage_longer}[a.stage]()


if __name__ == "__main__":
    main()
