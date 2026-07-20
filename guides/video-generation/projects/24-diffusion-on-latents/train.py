"""Project 24 — the same diffusion model, trained on pixels and on latents,
given the same wall-clock budget.

This is the payoff of Phase 5. Project 21 trained a 3D VAE; here we freeze it,
put it in front of a small video diffusion model, and compare against the same
model denoising raw pixels.

The comparison is by **time, not by steps**. Comparing at equal steps would
quietly hand the win to the latent arm and tell you nothing you did not
already know (of course a smaller tensor is faster per step). Comparing at
equal seconds asks the question a practitioner actually has: *given one
afternoon of compute, which setup gives the better model?*

    python3 train.py --stage cache     # ~1 min: freeze the VAE, encode clips
    python3 train.py --stage pixel     # ~7 min
    python3 train.py --stage latent    # ~7 min
    python3 train.py --stage figures   # ~3 min
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
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
import plot_style as ps
import matplotlib.pyplot as plt

import vae3d_lib as V
import diffusion_lib as D
import fid_lib

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

VAE_CK = HERE.parent / "21-train-a-small-3d-vae" / "checkpoints" / "3d.pt"
CACHE = CK / "latents.pt"

TRAIN_SECONDS = 300     # the fixed budget each arm gets
BATCH = 8
LR = 3e-4
POOL = 640              # clips in the cached training pool
DIFF_STEPS = 300
SAMPLE_STEPS = 60


def load_vae():
    if not VAE_CK.exists():
        raise SystemExit("train project 21 first: "
                         "cd ../21-train-a-small-3d-vae && "
                         "python3 train.py --stage 3d")
    ck = torch.load(VAE_CK, map_location="cpu", weights_only=False)
    vae = V.VideoVAE(base=16, z_ch=4, temporal=True)
    vae.load_state_dict(ck["state"])
    vae.eval().requires_grad_(False)
    return vae, ck["scale"]


@torch.no_grad()
def build_cache():
    """Two-stage training, stage two's prerequisite.

    The VAE is finished and frozen, so a clip's latent never changes — which
    means encoding it once and reusing it is not an approximation, it is just
    not doing the same work 400 times. Real pipelines encode their whole
    dataset to latents on disk before diffusion training starts.
    """
    vae, scale = load_vae()
    src = V.make_source(seed=21)
    clips, lats = [], []
    for _ in range(POOL // BATCH):
        x = V.clip_batch(src, BATCH)
        mean, _ = vae.encode(x)
        clips.append(x)
        lats.append(mean * scale)          # rescale to ~unit variance
    clips = torch.cat(clips)
    lats = torch.cat(lats)
    torch.save({"clips": clips, "latents": lats, "scale": scale}, CACHE)
    print(f"cached {clips.shape} clips -> {lats.shape} latents, "
          f"scale {scale:.3f}, latent std {lats.std():.3f}")


def load_cache():
    if not CACHE.exists():
        raise SystemExit("run `python3 train.py --stage cache` first")
    return torch.load(CACHE, map_location="cpu", weights_only=False)


def make_model(arm):
    """Identical depth and width for both arms; only in_ch differs."""
    if arm == "pixel":
        return D.UNet3D(in_ch=1, base=32, space_only_down=True)
    return D.UNet3D(in_ch=4, base=32, space_only_down=True)


def train(arm):
    torch.manual_seed(0)
    cache = load_cache()
    data = cache["clips"] if arm == "pixel" else cache["latents"]
    model = make_model(arm)
    ddpm = D.DDPM(steps=DIFF_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)

    log, step, t0 = [], 0, time.time()
    while time.time() - t0 < TRAIN_SECONDS:
        idx = torch.randint(0, len(data), (BATCH,), generator=g)
        x0 = data[idx]
        t = torch.randint(0, DIFF_STEPS, (BATCH,), generator=g)
        noise = torch.randn(x0.shape, generator=g)
        xt = ddpm.add_noise(x0, t, noise)

        loss = F.mse_loss(model(xt, t), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 20 == 0:
            log.append((time.time() - t0, step, float(loss)))
        if step % 200 == 0:
            print(f"[{arm}] step {step:5d}  loss {float(loss):.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    torch.save({"state": model.state_dict(), "arm": arm, "steps": step,
                "elapsed": elapsed}, CK / f"{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[{arm}] {step} steps in {elapsed:.0f}s "
          f"({elapsed / step * 1000:.0f} ms/step), "
          f"{x0.numel() // BATCH} numbers per clip")


def load_arm(arm):
    ck = torch.load(CK / f"{arm}.pt", map_location="cpu", weights_only=False)
    m = make_model(arm)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


@torch.no_grad()
def generate(arm, n=12, seed=0):
    """Sample from an arm and return clips in pixel space, either way."""
    model, _ = load_arm(arm)
    ddpm = D.DDPM(steps=DIFF_STEPS)
    g = torch.Generator().manual_seed(seed)
    if arm == "pixel":
        x = ddpm.sample(model, (n, 1, 16, 64, 64), steps=SAMPLE_STEPS,
                        generator=g)
        return x.clamp(-1, 1)
    vae, scale = load_vae()
    z = ddpm.sample(model, (n, 4, 4, 8, 8), steps=SAMPLE_STEPS, generator=g)
    return vae.decoder(z / scale).clamp(-1, 1)


ARMS = ["pixel", "latent"]


@torch.no_grad()
def figures():
    real_src = V.make_source(seed=31, train=False)
    reals = torch.cat([V.clip_batch(real_src, 8) for _ in range(3)])
    net = fid_lib.load_features()

    rows, samples = [], {}
    for arm in ARMS:
        _, ck = load_arm(arm)
        t0 = time.time()
        fake = generate(arm, n=24)
        gen_s = time.time() - t0
        samples[arm] = fake
        numbers = (65536 if arm == "pixel" else 1024)
        rows.append(dict(
            arm=arm,
            numbers_per_clip=numbers,
            steps_in_budget=ck["steps"],
            ms_per_step=round(ck["elapsed"] / ck["steps"] * 1000, 1),
            seconds=round(ck["elapsed"]),
            sampling_seconds=round(gen_s, 1),
            fid_proxy=round(fid_lib.frechet(reals, fake, net), 2),
        ))
        print(rows[-1], flush=True)

    # the VAE's own reconstruction floor: no generator can beat this, because
    # every latent-arm sample has to come out through the same decoder
    vae, scale = load_vae()
    rows.append(dict(arm="VAE reconstruction (floor)", numbers_per_clip=1024,
                     steps_in_budget="", ms_per_step="", seconds="",
                     sampling_seconds="",
                     fid_proxy=round(fid_lib.frechet(
                         reals, vae.reconstruct(reals), net), 2)))
    rows.append(dict(arm="real clips (sanity check)", numbers_per_clip="",
                     steps_in_budget="", ms_per_step="", seconds="",
                     sampling_seconds="",
                     fid_proxy=round(fid_lib.frechet(
                         reals, torch.cat([V.clip_batch(
                             V.make_source(seed=77, train=False), 8)
                             for _ in range(3)]), net), 2)))
    print(rows[-2], "\n", rows[-1])

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    figure_loss()
    figure_samples(samples, reals)
    print("wrote", OUT)


def figure_loss():
    fig, ax = ps.new_axes(7.4, 4.2)
    for i, arm in enumerate(ARMS):
        a = np.load(OUT / f"log_{arm}.npy")
        ax.plot(a[:, 0], a[:, 2], color=ps.SERIES[i], lw=1.4,
                label=f"{arm} ({int(a[-1, 1])} steps)")
    ax.set_xlabel("wall-clock seconds of training")
    ax.set_ylabel("denoising MSE loss")
    ax.set_title("Same model, same 300 seconds, different tensor",
                 color=ps.INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "loss_vs_time.png")
    plt.close(fig)


def figure_samples(samples, reals):
    rows = [("real", reals)] + [(arm, samples[arm]) for arm in ARMS]
    fig, axes = plt.subplots(len(rows) * 2, 1,
                             figsize=(10.0, 1.3 * len(rows) * 2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    k = 0
    for tag, clips in rows:
        for j in range(2):
            ax = axes[k]
            ax.imshow(V.strip(clips[j:j + 1], n=8), cmap="gray",
                      vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=9)
            k += 1
    fig.suptitle("Two generated clips per arm, 8 frames each", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cache", "figures"] + ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "cache":
        build_cache()
    elif args.stage == "figures":
        figures()
    else:
        train(args.stage)
