"""Project 20 — run Stable Diffusion's *image* VAE on video, one frame at a time.

No training here.  We take the pretrained SD 1.5 VAE, push each frame through
encode -> decode independently, and measure how much frame-to-frame change the
round trip *invents*.  Two knobs are compared:

  mode()    take the mean of the encoder's Gaussian  (deterministic)
  sample()  draw from it, as Stable Diffusion actually does at train time

Outputs: metrics.csv, three figures, and an animated GIF of the static clip.

    python3 run.py            # ~3 min on CPU
"""

import argparse
import csv
import sys
import textwrap
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import plot_style as ps
import matplotlib.pyplot as plt
from PIL import Image

import scenes

OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

MODEL = "emilianJR/epiCRealism"      # an SD 1.5 checkpoint; we use only its VAE
SCALE = 0.18215                      # SD's latent scaling factor
ORDER = ["static", "noise", "drift", "motion"]


def load_vae():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae")
    vae.eval().requires_grad_(False)
    return vae


def to_tensor(clip):
    """(T, H, W, 3) in [0,1]  ->  (T, 3, H, W) in [-1, 1], the VAE's range."""
    x = torch.from_numpy(clip).permute(0, 3, 1, 2)
    return x * 2.0 - 1.0


def to_numpy(x):
    """(T, 3, H, W) in [-1,1]  ->  (T, H, W, 3) in [0,1]."""
    return ((x.permute(0, 2, 3, 1) + 1.0) / 2.0).clamp(0, 1).numpy()


@torch.no_grad()
def roundtrip(vae, x, stochastic, seed=0):
    """Encode and decode every frame *independently*, as a per-frame pipeline
    would.  The loop over t is the whole point: nothing links the frames."""
    g = torch.Generator().manual_seed(seed)
    lat, rec = [], []
    for t in range(x.shape[0]):
        dist = vae.encode(x[t:t + 1]).latent_dist
        z = dist.sample(generator=g) if stochastic else dist.mode()
        lat.append(z)
        rec.append(vae.decode(z).sample)
    return torch.cat(lat), torch.cat(rec)


def flicker(clip):
    """Mean absolute change between neighbouring frames.

    This is the number your eye reads as 'shimmer' when the scene is still:
    zero means perfectly stable, larger means more frame-to-frame churn."""
    d = np.abs(np.diff(clip, axis=0))
    return float(d.mean())


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def static_mask(clip, tol=1e-6):
    """True at pixels that are provably identical in every frame of the input.

    This is the strictest possible test bed: whatever the reconstruction does
    here, the input did *not* ask for it.  On the `motion` clip the mask is
    the whole background, so it answers the question that matters in practice
    — does a still wall shimmer just because something else in the frame
    moved?"""
    return np.abs(clip - clip[0]).max(0).max(-1) < tol


def masked_flicker(clip, mask, min_px=1000):
    # Below a few hundred pixels the average is dominated by whichever
    # stragglers happened to survive the tolerance test, so report nothing
    # rather than a number that looks meaningful and is not.
    if mask.sum() < min_px:
        return float("nan")
    d = np.abs(np.diff(clip, axis=0)).mean(-1)      # (T-1, H, W)
    return float(d[:, mask].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=scenes.T)
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw from the cached round trip in outputs/")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    torch.set_num_threads(min(12, torch.get_num_threads()))

    clip_dict = scenes.clips()
    cache = OUT / "recons.npz"
    if args.figures_only and cache.exists():
        d = np.load(cache)
        recons = {(n, m): d[f"{n}_{m}"] for n in ORDER
                  for m in ("mode", "sample")}
        rows = list(csv.DictReader(open(OUT / "metrics.csv")))
        for r in rows:                      # csv gives strings
            for k in ("input_flicker", "recon_flicker",
                      "static_recon_flicker"):
                r[k] = float(r[k])
        figure_flicker(rows)
        figure_error_maps(clip_dict, recons)
        figure_temporal_std(clip_dict, recons)
        figure_gif(clip_dict, recons)
        print("redrew figures from", cache)
        return

    vae = load_vae()

    rows, recons, latents = [], {}, {}
    for name in ORDER:
        clip = clip_dict[name][:args.frames]
        x = to_tensor(clip)
        for mode in ("mode", "sample"):
            t0 = time.time()
            z, xr = roundtrip(vae, x, stochastic=(mode == "sample"))
            rec = to_numpy(xr)
            secs = time.time() - t0

            zc = (z * SCALE).numpy()
            in_f, rec_f = flicker(clip), flicker(rec)
            mask = static_mask(clip)
            rows.append(dict(
                clip=name, latent=mode,
                input_flicker=round(in_f, 6),
                recon_flicker=round(rec_f, 6),
                amplification=round(rec_f / in_f, 2) if in_f > 0 else "n/a",
                static_px=int(mask.sum()),
                static_recon_flicker=round(masked_flicker(rec, mask), 6),
                psnr_db=round(psnr(clip, rec), 2),
                latent_flicker=round(flicker(zc), 4),
                latent_abs=round(float(np.abs(zc).mean()), 4),
                seconds=round(secs, 1),
            ))
            recons[(name, mode)] = rec
            latents[(name, mode)] = zc
            print(rows[-1], flush=True)

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    np.savez_compressed(cache, **{f"{n}_{m}": recons[(n, m)] for n in ORDER
                                  for m in ("mode", "sample")})
    figure_flicker(rows)
    figure_error_maps(clip_dict, recons)
    figure_temporal_std(clip_dict, recons)
    figure_gif(clip_dict, recons)
    print("wrote", OUT)


def figure_flicker(rows):
    fig, ax = ps.new_axes(7.6, 4.2)
    idx = np.arange(len(ORDER))
    w = 0.26
    md = [r for r in rows if r["latent"] == "mode"]
    sp = [r for r in rows if r["latent"] == "sample"]
    series = [
        ("input (the real clip)", [r["input_flicker"] for r in md],
         ps.SERIES[0]),
        ("recon, mode()", [r["recon_flicker"] for r in md], ps.SERIES[1]),
        ("recon, sample()", [r["recon_flicker"] for r in sp], ps.SERIES[2]),
        ("recon, provably-still pixels only",
         [r["static_recon_flicker"] for r in md], ps.SERIES[3]),
    ]
    w = 0.2
    for i, (label, vals, color) in enumerate(series):
        ax.bar(idx + (i - 1.5) * w, vals, w, label=label, color=color)
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"{n}\n" + "\n".join(textwrap.wrap(scenes.DESCRIPTIONS[n], 22))
         for n in ORDER], fontsize=8)
    ax.set_ylabel("frame-to-frame change (log scale)")
    ax.set_title("Per-frame VAE invents motion that is not in the clip",
                 color=ps.INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "flicker_bars.png")
    plt.close(fig)


def figure_error_maps(clip_dict, recons):
    """|frame t - frame t-1| for the motion clip, input vs reconstruction.

    The input row is black everywhere except the disc — the background truly
    does not move.  The reconstruction row is where the argument lands."""
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    clip = clip_dict["motion"]
    rec = recons[("motion", "mode")]
    for t in range(4):
        d_in = np.abs(clip[t + 1] - clip[t]).mean(-1)
        d_rc = np.abs(rec[t + 1] - rec[t]).mean(-1)
        for row, d, tag in ((0, d_in, "input"), (1, d_rc, "reconstruction")):
            ax = axes[row, t]
            ax.imshow(d, cmap="magma", vmin=0, vmax=0.02)
            ax.set_xticks([]), ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=9)
            ax.set_title(f"|frame {t+1} - frame {t}|", fontsize=8,
                         color=ps.INK_MUTED)
    fig.suptitle("Only the disc moves. Same color scale, top vs bottom.",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "error_maps.png")
    plt.close(fig)


def figure_temporal_std(clip_dict, recons):
    """Per-pixel standard deviation over time — a still scene should be black."""
    fig, axes = plt.subplots(2, len(ORDER), figsize=(10.0, 5.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for c, name in enumerate(ORDER):
        pair = ((clip_dict[name], "input"), (recons[(name, "mode")], "recon"))
        for r, (arr, tag) in enumerate(pair):
            ax = axes[r, c]
            ax.imshow(arr.std(0).mean(-1), cmap="magma", vmin=0, vmax=0.02)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=9, color=ps.INK)
            if c == 0:
                ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=9)
    fig.suptitle("Per-pixel variation across the 8 frames "
                 "(black = perfectly stable)", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "temporal_std.png")
    plt.close(fig)


def figure_gif(clip_dict, recons):
    """A crop of the motion clip's *background*, input beside reconstruction.

    The crop deliberately excludes the disc, so anything that moves on the
    right-hand side was invented by the per-frame round trip."""
    crop = slice(128, 256)
    frames = []
    for t in range(scenes.T):
        left = clip_dict["motion"][t][crop, crop]
        right = recons[("motion", "mode")][t][crop, crop]
        pair = np.concatenate([left, np.ones((128, 4, 3)), right], axis=1)
        frames.append(Image.fromarray((pair * 255).astype(np.uint8))
                      .resize((520, 256), Image.NEAREST))
    frames[0].save(OUT / "flicker.gif", save_all=True,
                   append_images=frames[1:], duration=200, loop=0)


if __name__ == "__main__":
    main()
