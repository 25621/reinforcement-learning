"""Project 41 — a game with no game, only a model.

    python3 run.py --stage data       # ~1 min
    python3 run.py --stage train      # ~5 min   (two arms)
    python3 run.py --stage drift      # ~4 min
    python3 run.py --stage latency    # ~1 min
    python3 run.py --stage figures    # ~1 min
    python3 play.py                   # play it yourself

Project 40 predicted four frames and stopped.  Here the model's own output is
fed straight back in as its next input, forever.  That single change is what
makes a playable world -- and what lets tiny errors pile up into nonsense.

Two arms, identical apart from one line of the training loop:

    noaug   trained on clean context frames  (the obvious way)
    aug     trained on context frames that have been deliberately corrupted,
            and told how badly                (GameNGen's fix)
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
sys.path.insert(0, str(HERE.parent / "40-action-conditioned-video"))
import plot_style as ps                                        # noqa: E402
import matplotlib.pyplot as plt                                # noqa: E402

import world_lib as W                                          # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

torch.set_num_threads(12)

CTX, HORIZON = 2, 1
STEPS, BATCH, LR = 3000, 64, 4e-4
SIGMA_MAX = 0.4               # training corruption range for the `aug` arm
SAMPLE_STEPS = 30
ROLL_LEN = 150
N_ROLL = 24
ARMS = ["noaug", "aug"]
INFER_SIGMA = {"noaug": [0.0], "aug": [0.0, 0.05, 0.15]}


def arm_net(arm):
    return W.ActionUNet(ctx=CTX, horizon=HORIZON, ctx_noise=(arm == "aug"))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def stage_data():
    t0 = time.time()
    train = W.record(2500, seed=41)
    # long episodes so a 150-step rollout has 150 real actions to replay
    long = W.record(N_ROLL, seed=4141, ep_len=ROLL_LEN + CTX + 1)
    W.save_episodes(train, "train", where=CK)
    W.save_episodes(long, "long", where=CK)
    print(f"train 2500 x {W.EP_LEN}, eval {N_ROLL} x {ROLL_LEN + CTX + 1} "
          f"({time.time()-t0:.1f}s)")


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------

def stage_train():
    ep = W.load_episodes("train", where=CK)
    rows = []
    for arm in ARMS:
        torch.manual_seed(41)
        rng = np.random.default_rng(41)
        net = arm_net(arm)
        flow = W.FL.RectifiedFlow()
        opt = torch.optim.AdamW(net.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
        t0 = time.time()
        for step in range(1, STEPS + 1):
            ctx, tgt, act, _ = W.sample_batch(ep, BATCH, CTX, HORIZON, rng)
            sigma = None
            if arm == "aug":
                # Corrupt the past the model is shown, and hand it the recipe.
                sigma = torch.rand(BATCH) * SIGMA_MAX
                ctx = ctx + sigma[:, None, None, None] * torch.randn_like(ctx)
            noise = torch.randn_like(tgt)
            t = flow.sample_t(BATCH)
            xt = flow.interpolate(tgt, t, noise)
            v = net(xt, t * flow.T_SCALE, ctx, act, sigma=sigma)
            loss = F.mse_loss(v, flow.target(tgt, noise))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            if step % 500 == 0:
                rows.append([arm, step, f"{loss.item():.5f}"])
                print(f"  {arm} {step:5d}/{STEPS} loss {loss.item():.4f} "
                      f"({time.time()-t0:.0f}s)")
        torch.save(net.state_dict(), CK / f"{arm}.pt")
        print(f"  {arm}: {W.count_params(net)} params, {time.time()-t0:.0f}s")
    with open(OUT / "loss.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["arm", "step", "loss"])
        wr.writerows(rows)


def load_arm(arm):
    net = arm_net(arm)
    net.load_state_dict(torch.load(CK / f"{arm}.pt"))
    net.eval()
    return net


# --------------------------------------------------------------------------
# drift: what happens over 150 self-fed frames
# --------------------------------------------------------------------------

def _truth(long):
    """True agent path and wall map for each evaluation episode."""
    walls = long["walls"]
    paths = []
    for e in range(N_ROLL):
        p = [(long["agent"][e, t] // W.GRID, long["agent"][e, t] % W.GRID)
             for t in range(CTX, CTX + ROLL_LEN)]
        paths.append(p)
    return walls, paths


def score_rollout(frames, walls, paths, acts, start_agent):
    """frames (B, N, 8, 8) -> four per-step curves.

    `acc`   -- is the player in the cell the REAL game would have put it in?
               This is a ratchet: one wrong step desynchronises the model from
               the reference forever, even if every later step is lawful.
    `self`  -- given where the model itself put the player last frame, did this
               frame move it the way the button says?  This is the question a
               player actually cares about: does the game still WORK?
    `snap`  -- how far each cell is from a legal colour: is this a game screen?
    `wall`  -- are the walls still where the level put them?
    """
    b, n = frames.shape[:2]
    acc, self_ok, snap, wall_ok = (np.zeros(n) for _ in range(4))
    for e in range(b):
        tw = walls[e]
        prev = start_agent[e]
        for k in range(n):
            sym, ai, _, sn = W.read_frame(frames[e, k])
            cur = (ai // W.GRID, ai % W.GRID)
            acc[k] += (cur == paths[e][k])
            dr, dc = W.DELTA[int(acts[e, k])]
            nr, nc = prev[0] + dr, prev[1] + dc
            legal = prev if (not (0 <= nr < W.GRID and 0 <= nc < W.GRID)
                             or tw[nr, nc]) else (nr, nc)
            self_ok[k] += (cur == legal)
            snap[k] += sn
            wall_ok[k] += ((sym == 1) == (tw == 1)).mean()
            prev = cur
    return acc / b, self_ok / b, snap / b, wall_ok / b


def stage_drift():
    long = W.load_episodes("long", where=CK)
    walls, paths = _truth(long)
    ctx0 = torch.from_numpy(
        W.frames_of(long, np.repeat(np.arange(N_ROLL)[:, None], CTX, 1),
                    np.repeat(np.arange(CTX)[None], N_ROLL, 0)))
    acts = torch.from_numpy(long["act"][:, CTX - 1:CTX - 1 + ROLL_LEN])
    start = [(long["agent"][e, CTX - 1] // W.GRID,
              long["agent"][e, CTX - 1] % W.GRID) for e in range(N_ROLL)]
    rows, curves = [], {}
    for arm in ARMS:
        net = load_arm(arm)
        for s in INFER_SIGMA[arm]:
            tag = f"{arm}_s{s:.2f}"
            sig = None if arm == "noaug" else torch.full((N_ROLL,), s)
            g = torch.Generator().manual_seed(41)
            t0 = time.time()
            fr = W.rollout(net, ctx0, acts, steps=SAMPLE_STEPS, sigma=sig,
                           generator=g).numpy()
            acc, slf, snap, wok = score_rollout(fr, walls, paths,
                                                acts.numpy(), start)
            curves[tag] = np.stack([slf, snap, wok, acc])
            rows.append(dict(arm=arm, infer_sigma=s,
                             self_first20=float(slf[:20].mean()),
                             self_last20=float(slf[-20:].mean()),
                             acc_first20=float(acc[:20].mean()),
                             acc_last20=float(acc[-20:].mean()),
                             snap_last20=float(snap[-20:].mean()),
                             walls_last20=float(wok[-20:].mean()),
                             secs=time.time() - t0))
            print(f"{tag}: button obeyed {slf[:20].mean():.3f} -> "
                  f"{slf[-20:].mean():.3f}   in-sync {acc[:20].mean():.3f} -> "
                  f"{acc[-20:].mean():.3f}   snap {snap[-20:].mean():.4f}  "
                  f"walls {wok[-20:].mean():.3f}  ({time.time()-t0:.0f}s)")
            if tag in ("noaug_s0.00", "aug_s0.05"):
                W.write_gif(fr[0, :80], OUT / f"roll_{tag}.gif")
    # the real thing, for reference
    real = W.frames_of(long, np.full(80, 0), np.arange(CTX, CTX + 80))
    W.write_gif(real, OUT / "roll_real.gif")
    np.savez(CK / "curves.npz", **curves)
    with open(OUT / "drift.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                         for k, v in r.items()})


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------

def stage_latency():
    net = load_arm("aug")
    long = W.load_episodes("long", where=CK)
    ctx0 = torch.from_numpy(
        W.frames_of(long, np.repeat(np.arange(1)[:, None], CTX, 1),
                    np.repeat(np.arange(CTX)[None], 1, 0)))
    acts = torch.zeros((1, 1), dtype=torch.long)
    sig = torch.full((1,), 0.05)
    rows = []
    for steps in [1, 2, 4, 8, 16, 30]:
        g = torch.Generator().manual_seed(0)
        W.sample_frames(net, ctx0, acts, steps=steps, generator=g, sigma=sig)
        t0 = time.time()
        n = 20
        for _ in range(n):
            W.sample_frames(net, ctx0, acts, steps=steps, generator=g,
                            sigma=sig)
        ms = (time.time() - t0) / n * 1000
        rows.append(dict(sample_steps=steps, ms_per_frame=ms, fps=1000 / ms))
        print(f"{steps:3d} steps: {ms:7.1f} ms/frame  ({1000/ms:5.1f} fps)")
    with open(OUT / "latency.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        for r in rows:
            wr.writerow({k: (f"{v:.2f}" if isinstance(v, float) else v)
                         for k, v in r.items()})


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def stage_figures():
    curves = np.load(CK / "curves.npz")
    fig, axes = plt.subplots(1, 4, figsize=(17.0, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    titles = ["Does the button still work?", "Is it still a legal screen?",
              "Are the walls still there?", "Still in sync with the real game"]
    ylabels = ["move matches the button", "snap error (lower better)",
               "wall-map agreement", "cell-exact accuracy"]
    for i, ax in enumerate(axes):
        ps.style_axes(ax)
        for j, tag in enumerate(curves.files):
            y = curves[tag][i]
            k = 10
            sm = np.convolve(y, np.ones(k) / k, mode="valid")
            ax.plot(np.arange(len(sm)) + k // 2, sm, color=ps.SERIES[j],
                    label=tag, lw=1.6)
        ax.set_title(titles[i], color=ps.INK, fontsize=11, loc="left")
        ax.set_xlabel("frames generated", color=ps.INK_SECONDARY, fontsize=9)
        ax.set_ylabel(ylabels[i], color=ps.INK_SECONDARY, fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "drift.png", facecolor=ps.SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'drift.png'}")

    lat = list(csv.DictReader(open(OUT / "latency.csv")))
    fig, ax = ps.new_axes(6.4, 4.0)
    x = [int(r["sample_steps"]) for r in lat]
    y = [float(r["ms_per_frame"]) for r in lat]
    ax.plot(x, y, marker="o", color=ps.SERIES[0])
    ax.axhline(33.3, color=ps.SERIES[2], ls="--", lw=1.4)
    ax.text(x[-1], 36, "30 fps budget (33 ms)", color=ps.SERIES[2],
            fontsize=9, ha="right")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ps.finish(fig, ax, "How long one frame takes", "denoising steps",
              "ms per frame", OUT / "latency.png")

    # side-by-side film strip: the real game against both rollouts
    long = W.load_episodes("long", where=CK)
    picks = np.arange(CTX, CTX + ROLL_LEN, 20)
    real = W.frames_of(long, np.full(len(picks), 0), picks)
    ctx0 = torch.from_numpy(
        W.frames_of(long, np.repeat(np.arange(2)[:, None], CTX, 1),
                    np.repeat(np.arange(CTX)[None], 2, 0)))
    acts = torch.from_numpy(long["act"][:2, CTX - 1:CTX - 1 + ROLL_LEN])
    strips = [list(real)]
    for arm, s in [("noaug", 0.0), ("aug", 0.05)]:
        net = load_arm(arm)
        sig = None if arm == "noaug" else torch.full((2,), s)
        g = torch.Generator().manual_seed(41)
        fr = W.rollout(net, ctx0, acts, steps=SAMPLE_STEPS, sigma=sig,
                       generator=g).numpy()
        strips.append([fr[0, k] for k in range(0, ROLL_LEN, 20)])
    W.strip_image(strips, OUT / "rollout_strip.png")
    print("rollout_strip.png rows: real game, noaug, aug "
          "(every 20th generated frame)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True,
                   choices=["data", "train", "drift", "latency", "figures"])
    a = p.parse_args()
    {"data": stage_data, "train": stage_train, "drift": stage_drift,
     "latency": stage_latency, "figures": stage_figures}[a.stage]()


if __name__ == "__main__":
    main()
