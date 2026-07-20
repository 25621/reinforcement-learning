"""Tiny I2V model: pretrain 2D, freeze, inflate, fine-tune temporal layers.

Stages (run in order; each stage saves what the next needs):

  python3 train.py --stage image     # ~9.5 min: pretrain the 2D U-Net
  python3 train.py --stage video     # ~7 min:  frozen-spatial inflation
  python3 train.py --stage scratch   # ~6.5 min: all-trainable baseline
  python3 train.py --stage figures   # ~1.5 min: sample everything, plot
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(12)

from PIL import Image

import i2v_lib as lib
from i2v_lib import (Diffusion, ImageUNet, VideoUNet, train_image_model,
                     train_video_model, sample_clip, flicker, cond_fidelity,
                     strip, set_seed)
from mmnist import MovingMNIST                        # noqa: E402  (project 06)
import plot_style as ps                               # noqa: E402  (project 01)

HERE = Path(__file__).resolve().parent
CKPT = HERE / "checkpoints"
OUT = HERE / "outputs"
CKPT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

T_FRAMES = 8
BATCH_CLIPS = 8
IMAGE_STEPS = 1600
VIDEO_STEPS = 1000


def clip_batches(batch, seed=0, train=True):
    ds = MovingMNIST(n_digits=2, seq_len=T_FRAMES, train=train, seed=seed)
    def source():
        return ds.batch(batch)
    return source


def stage_image():
    set_seed()
    diff = Diffusion()
    model = ImageUNet()
    n = sum(p.numel() for p in model.parameters())
    print(f"image U-Net params: {n/1e3:.0f}k")
    src = clip_batches(BATCH_CLIPS)
    t0 = time.time()
    losses = train_image_model(model, diff, src, IMAGE_STEPS)
    print(f"image stage: {time.time()-t0:.0f}s")
    torch.save(model.state_dict(), CKPT / "image_unet.pt")
    np.save(CKPT / "loss_image.npy", np.array(losses))


def video_batch_source(seed=1):
    ds = MovingMNIST(n_digits=2, seq_len=T_FRAMES, train=True, seed=seed)
    def source():
        return {"clips": ds.batch(BATCH_CLIPS)}
    return source


def stage_video(freeze=True):
    set_seed(1 if freeze else 2)
    diff = Diffusion()
    image_unet = ImageUNet()
    if freeze:
        image_unet.load_state_dict(torch.load(CKPT / "image_unet.pt"))
    model = VideoUNet(image_unet, freeze_spatial=freeze)
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train/1e3:.0f}k of {n_total/1e3:.0f}k params")

    if freeze:
        # Identity check + "before" strip: with zero-init temporal layers
        # the video model must equal the frozen image model per frame.
        ds = MovingMNIST(n_digits=2, seq_len=T_FRAMES, train=False, seed=9)
        clips = ds.batch(3)
        x = lib.to_signed(clips)
        t = torch.full((3,), 150)
        with torch.no_grad():
            v = model(x, t, x[:, 0])
            im = image_unet(x.reshape(-1, 1, 32, 32),
                            t.repeat_interleave(T_FRAMES))
        diff_max = (v.reshape(-1, 1, 32, 32) - im).abs().max().item()
        print(f"identity-at-init check |video - per-frame image| = {diff_max:.2e}")
        before = sample_clip(model, diff, clips[:, 0], T_FRAMES, seed=5)
        np.save(CKPT / "before_clips.npy", before.numpy())

    src = video_batch_source()
    t0 = time.time()
    losses = train_video_model(model, diff, src, VIDEO_STEPS)
    print(f"video stage ({'frozen' if freeze else 'scratch'}): "
          f"{time.time()-t0:.0f}s")
    name = "video_frozen" if freeze else "video_scratch"
    torch.save(model.state_dict(), CKPT / f"{name}.pt")
    np.save(CKPT / f"loss_{name}.npy", np.array(losses))


def load_video_model(name, freeze=True):
    image_unet = ImageUNet()
    model = VideoUNet(image_unet, freeze_spatial=freeze)
    model.load_state_dict(torch.load(CKPT / f"{name}.pt"))
    model.eval()
    return model


def stage_figures():
    set_seed(3)
    diff = Diffusion()
    frozen = load_video_model("video_frozen")
    scratch = load_video_model("video_scratch", freeze=False)

    ds = MovingMNIST(n_digits=2, seq_len=T_FRAMES, train=False, seed=9)
    clips = ds.batch(3)                   # same conditions as `before`
    cond = clips[:, 0]

    t0 = time.time()
    gen_frozen = sample_clip(frozen, diff, cond, T_FRAMES, seed=5)
    gen_scratch = sample_clip(scratch, diff, cond, T_FRAMES, seed=5)
    print(f"sampling: {time.time()-t0:.0f}s")
    before = torch.from_numpy(np.load(CKPT / "before_clips.npy"))

    # --- metrics ------------------------------------------------------
    real_flicker = flicker(clips)
    rows = [
        ("real clips", real_flicker, 0.0),
        ("inflated, temporal untrained", flicker(before),
         cond_fidelity(before, cond)),
        ("inflated, temporal trained", flicker(gen_frozen),
         cond_fidelity(gen_frozen, cond)),
        ("scratch, same budget", flicker(gen_scratch),
         cond_fidelity(gen_scratch, cond)),
    ]
    with open(OUT / "metrics.csv", "w") as f:
        f.write("model,flicker,cond_mse\n")
        for name, fl, cm in rows:
            f.write(f"{name},{fl:.4f},{cm:.4f}\n")
            print(f"{name:32s} flicker {fl:.4f}  cond MSE {cm:.4f}")

    # --- strips -------------------------------------------------------
    def save_strip(tensor, path):
        Image.fromarray(strip(tensor)).resize(
            (tensor.shape[1] * 34 * 4, tensor.shape[0] * 34 * 4),
            Image.NEAREST).save(path)

    save_strip(torch.cat([clips[:1], before[:1], gen_frozen[:1],
                          gen_scratch[:1]]), OUT / "one_condition.png")
    save_strip(gen_frozen, OUT / "samples_frozen.png")
    save_strip(before, OUT / "samples_before.png")
    save_strip(gen_scratch, OUT / "samples_scratch.png")

    # --- a small animated GIF of one generated clip -------------------
    frames = [Image.fromarray(
        (gen_frozen[0, t, 0].numpy() * 255).astype(np.uint8)).resize(
        (128, 128), Image.NEAREST) for t in range(T_FRAMES)]
    frames[0].save(OUT / "sample.gif", save_all=True,
                   append_images=frames[1:], duration=180, loop=0)

    # --- loss curves --------------------------------------------------
    lf = np.load(CKPT / "loss_video_frozen.npy")
    lsc = np.load(CKPT / "loss_video_scratch.npy")
    fig, ax = ps.new_axes()
    k = 25
    smooth = lambda a: np.convolve(a, np.ones(k) / k, mode="valid")
    ax.plot(smooth(lf), color=ps.SERIES[0], lw=1.8,
            label="inflated (frozen spatial, train temporal)")
    ax.plot(smooth(lsc), color=ps.SERIES[2], lw=1.8,
            label="from scratch (train everything)")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Video-stage training loss (smoothed)",
              "training step", "epsilon-prediction MSE",
              OUT / "loss_video.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["image", "video", "scratch", "figures"])
    args = ap.parse_args()
    {"image": stage_image,
     "video": stage_video,
     "scratch": lambda: stage_video(freeze=False),
     "figures": stage_figures}[args.stage]()
