"""Project 21 — train a small 3D VAE, and prove that spending the latent
budget across time beats spending it per frame.

Two models, *identical* compression ratio (64x fewer numbers than the clip):

  3d   (B,1,16,64,64) -> (B,4,4,8,8)   8x space, 4x time, 4 latent channels
  2d   (B,1,16,64,64) -> (B,1,16,8,8)  8x space, no time compression, 1 channel

The 2d arm is the control.  It is the same network with its temporal strides
removed, so any difference in the results comes from *where* the budget went,
not from a different architecture or a different parameter count.

    python3 train.py --stage 3d        # ~7 min on CPU
    python3 train.py --stage 2d        # ~8 min on CPU
    python3 train.py --stage figures   # ~1 min
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import plot_style as ps
import matplotlib.pyplot as plt

import vae3d_lib as V

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

STEPS = 800        # sized so the *slower* (2d) arm still finishes under 10 min
BATCH = 8
LR = 3e-4
BASE = 16          # base channel width; 16/32/64 through the three stages

ARMS = {
    # name: (temporal compression?, latent channels)
    "3d": dict(temporal=True, z_ch=4),
    "2d": dict(temporal=False, z_ch=1),
}


def build(arm):
    cfg = ARMS[arm]
    return V.VideoVAE(base=BASE, z_ch=cfg["z_ch"], temporal=cfg["temporal"])


def train(arm):
    torch.manual_seed(0)          # seed *before* constructing the model, so
    model = build(arm)            # both arms start from comparable weights
    src = V.make_source(seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=5e-5)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        x = V.clip_batch(src, BATCH)
        loss, l1, kl = V.vae_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            log.append((step, float(l1), float(kl)))
        if step % 200 == 0:
            print(f"[{arm}] step {step:4d}  L1 {float(l1):.4f}  "
                  f"KL {float(kl):.1f}  {time.time() - t0:.0f}s", flush=True)

    model.eval()
    scale = V.latent_scale(model, V.make_source(seed=99, train=False))
    torch.save({"state": model.state_dict(), "arm": arm,
                "scale": scale, "cfg": ARMS[arm]}, CK / f"{arm}.pt")
    np.save(OUT / f"loss_{arm}.npy", np.array(log))
    print(f"[{arm}] done in {time.time() - t0:.0f}s, latent scale {scale:.3f}")


def load(arm):
    ck = torch.load(CK / f"{arm}.pt", map_location="cpu", weights_only=False)
    model = build(arm)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, ck


def figures():
    test = V.make_source(seed=7, train=False)
    rows = []
    models = {}
    for arm in ARMS:
        model, ck = load(arm)
        models[arm] = model
        m = V.evaluate(model, V.make_source(seed=7, train=False), batches=6)
        cfg = ARMS[arm]
        t_lat = V.T_FRAMES if not cfg["temporal"] else V.T_FRAMES // 4
        n_lat = cfg["z_ch"] * t_lat * 8 * 8
        rows.append(dict(
            arm=arm,
            latent_shape=f"{cfg['z_ch']}x{t_lat}x8x8",
            latent_numbers=n_lat,
            compression=round(V.T_FRAMES * 64 * 64 / n_lat, 1),
            params=V.count_params(model),
            psnr_db=round(m["psnr"], 2),
            flicker_input=round(m["flicker_in"], 4),
            flicker_recon=round(m["flicker_rec"], 4),
            flicker_error=round(m["flicker_err"], 4),
            latent_scale=round(ck["scale"], 3),
        ))
        print(rows[-1], flush=True)

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    figure_loss()
    figure_recon(models, test)
    figure_latent(models["3d"], test)
    figure_redundancy(test)
    print("wrote", OUT)


def figure_loss():
    fig, ax = ps.new_axes(7.2, 4.0)
    for i, arm in enumerate(ARMS):
        a = np.load(OUT / f"loss_{arm}.npy")
        ax.plot(a[:, 0], a[:, 1], color=ps.SERIES[i], lw=1.6,
                label=f"{arm} ({ARMS[arm]['z_ch']} latent ch)")
    ax.set_xlabel("training step")
    ax.set_ylabel("L1 reconstruction loss")
    ax.set_title("Same latent budget, spent two ways", color=ps.INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "loss.png")
    plt.close(fig)


@torch.no_grad()
def figure_recon(models, test):
    x = V.clip_batch(test, 4)
    rows = [("input", x)]
    for arm in ARMS:
        rows.append((arm, models[arm].reconstruct(x)))

    fig, axes = plt.subplots(len(rows), 1, figsize=(10.0, 1.5 * len(rows)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (tag, clip) in zip(axes, rows):
        ax.imshow(V.strip(clip, n=8), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=10)
    fig.suptitle("8 consecutive frames: input, 3D VAE, per-frame 2D VAE",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "reconstructions.png")
    plt.close(fig)


@torch.no_grad()
def figure_latent(model, test):
    """What the 4 latent channels of one clip actually look like."""
    x = V.clip_batch(test, 1)
    mean, _ = model.encode(x)
    fig, axes = plt.subplots(V.Z_CH, mean.shape[2],
                             figsize=(6.4, 6.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for c in range(V.Z_CH):
        for t in range(mean.shape[2]):
            ax = axes[c, t]
            ax.imshow(mean[0, c, t], cmap="magma")
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_title(f"t'={t}", fontsize=9, color=ps.INK_MUTED)
            if t == 0:
                ax.set_ylabel(f"ch {c}", fontsize=9, color=ps.INK_SECONDARY)
    fig.suptitle("The whole 16-frame clip, as 4x4x8x8 numbers", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "latent_grid.png")
    plt.close(fig)


@torch.no_grad()
def figure_redundancy(test):
    """Why 4x temporal compression is nearly free: neighbouring frames are
    almost the same picture, so most of the time axis is repetition."""
    x = V.clip_batch(test, 16)
    diffs = [torch.mean(torch.abs(x[:, :, t:] - x[:, :, :x.shape[2] - t])).item()
             for t in range(1, 9)]
    shuffled = x[:, :, torch.randperm(x.shape[2])]
    base = torch.mean(torch.abs(shuffled[:, :, 1:] - shuffled[:, :, :-1])).item()

    fig, ax = ps.new_axes(7.0, 4.0)
    ax.plot(range(1, 9), diffs, "o-", color=ps.SERIES[0], lw=1.8,
            label="mean |frame t - frame t+k|")
    ax.axhline(base, color=ps.SERIES[2], ls="--", lw=1.4,
               label="same clip, frames shuffled")
    ax.set_xlabel("gap k between frames")
    ax.set_ylabel("mean absolute difference")
    ax.set_title("Neighbouring frames are nearly identical — "
                 "that is the slack a 3D VAE eats", color=ps.INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "temporal_redundancy.png")
    plt.close(fig)
    with open(OUT / "redundancy.json", "w") as f:
        json.dump({"gap_diffs": diffs, "shuffled": base}, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["3d", "2d", "figures"])
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "figures":
        figures()
    else:
        train(args.stage)
