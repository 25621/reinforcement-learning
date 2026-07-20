"""Project 15 — inflate a pretrained image diffusion model into a video model.

Stages (run in order):
  python3 train.py --stage image     pretrain the 2D U-Net on single frames
  python3 train.py --stage video     inflate (temporal conv + attention), train
  python3 train.py --stage conv      ablation arm: temporal conv only
  python3 train.py --stage figures   sample everything, compute metrics, plot

The structural inflation of the *real* Stable Diffusion U-Net lives in
`inflate_real_sd.py` (no training — see the README).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import vdm_lib as V
from mmnist import MovingMNIST
import plot_style as ps

HERE = Path(__file__).resolve().parent
CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

IMAGE_STEPS = 1500
VIDEO_STEPS = 700
T = V.T_FRAMES          # 8 frames per clip
BATCH_CLIPS = 8


def data():
    return MovingMNIST(n_digits=2, seq_len=T, seed=1)


def build_video_model(temporal):
    unet = V.ImageUNet()
    unet.load_state_dict(torch.load(CK / "image.pt", weights_only=True))
    return V.VideoDiffusionUNet(unet, freeze_spatial=True,
                                temporal=temporal)


def stage_image():
    V.set_seed()
    diff = V.Diffusion()
    mm = data()
    model = V.ImageUNet()
    print(f"image U-Net params: {sum(p.numel() for p in model.parameters()):,}")
    losses = V.train_image_model(model, diff, lambda: mm.batch(64),
                                 IMAGE_STEPS)
    torch.save(model.state_dict(), CK / "image.pt")
    np.save(CK / "loss_image.npy", np.array(losses))


def stage_video(temporal, tag):
    V.set_seed()
    diff = V.Diffusion()
    mm = data()
    model = build_video_model(temporal)
    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.trainable_parameters())
    print(f"[{tag}] params total {n_all:,}  trainable {n_tr:,}")

    # identity at initialization: the inflated model on a clip must equal
    # the frozen image model applied to each frame independently
    with torch.no_grad():
        x = torch.randn(2, T, 1, V.CANVAS, V.CANVAS)
        t = torch.tensor([100, 250])
        vid = model(x, t)
        img = model.unet(x.reshape(2 * T, 1, V.CANVAS, V.CANVAS),
                         t.repeat_interleave(T)).reshape(vid.shape)
        d = (vid - img).abs().max().item()
    print(f"[{tag}] identity-at-init check |video - per-frame image| "
          f"= {d:.2e}")
    assert d == 0.0

    losses = V.train_video(model, diff,
                           lambda: {"clips": mm.batch(BATCH_CLIPS)},
                           VIDEO_STEPS)
    torch.save(model.state_dict(), CK / f"video_{tag}.pt")
    np.save(CK / f"loss_{tag}.npy", np.array(losses))


@torch.no_grad()
def sample_model(model, diff, n, seed=0):
    return V.ancestral_sample(lambda x, t: model(x, t), diff,
                              (n, T, 1, V.CANVAS, V.CANVAS), seed=seed)


def stage_figures():
    V.set_seed()
    diff = V.Diffusion()
    mm = MovingMNIST(n_digits=2, seq_len=T, seed=99)
    real = mm.batch(12)

    models = {}
    models["untrained"] = build_video_model(("conv", "attn"))
    for tag, temporal in [("conv", ("conv",)),
                          ("convattn", ("conv", "attn"))]:
        m = build_video_model(temporal)
        m.load_state_dict(torch.load(CK / f"video_{tag}.pt",
                                     weights_only=True))
        models[tag] = m

    print("sampling (ancestral, 300 steps, 12 clips per model)...")
    samples = {}
    for tag, m in models.items():
        m.eval()
        samples[tag] = sample_model(m, diff, 12, seed=7)
        print(f"  sampled {tag}", flush=True)

    # ---- metrics ----------------------------------------------------------
    rows = [("real clips", V.flicker(real), V.align_response(real))]
    names = {"untrained": "inflated, untrained (independent frames)",
             "conv": "temporal conv only",
             "convattn": "temporal conv + attention"}
    for tag in ["untrained", "conv", "convattn"]:
        rows.append((names[tag], V.flicker(samples[tag]),
                     V.align_response(samples[tag])))
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "flicker", "align_response"])
        w.writerows([(n, f"{a:.4f}", f"{b:.4f}") for n, a, b in rows])
    for n, a, b in rows:
        print(f"{n:44s} flicker {a:.4f}  align {b:.4f}")

    # ---- figures ----------------------------------------------------------
    stack = torch.cat([real[:2], samples["untrained"][:2],
                       samples["conv"][:2], samples["convattn"][:2]])
    V.save_strip(OUT / "rows_compare.png", stack)
    V.save_strip(OUT / "samples_convattn.png", samples["convattn"][:6])
    V.save_gif(OUT / "sample.gif", samples["convattn"][0])

    fig, ax = ps.new_axes()
    for i, (tag, label) in enumerate([("conv", "temporal conv only"),
                                      ("convattn", "conv + attention")]):
        loss = np.load(CK / f"loss_{tag}.npy")
        smooth = np.convolve(loss, np.ones(25) / 25, mode="valid")
        ax.plot(smooth, color=ps.SERIES[i], label=label, linewidth=1.6)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Video fine-tuning loss (rolling mean of 25)",
              "training step", "epsilon-prediction MSE",
              OUT / "loss_video.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["image", "video", "conv", "figures"])
    a = ap.parse_args()
    torch.set_num_threads(12)
    if a.stage == "image":
        stage_image()
    elif a.stage == "video":
        stage_video(("conv", "attn"), "convattn")
    elif a.stage == "conv":
        stage_video(("conv",), "conv")
    else:
        stage_figures()
