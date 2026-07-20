"""Project 18 — cascaded video super-resolution.

Two-stage cascade, Imagen-Video style, at Moving-MNIST scale:

  stage A ("base"):  video diffusion at 16x16 — cheap, learns layout+motion
  stage B ("sr"):    video diffusion at 32x32, conditioned on the
                     bilinearly-upsampled 16x16 clip fed in as an extra
                     input channel — learns detail only

The classic failure mode is also reproduced: an SR model trained only on
*perfect* downsampled clips falls apart when fed the base model's
imperfect *generated* clips (a train/test mismatch).  The fix from the
cascaded-diffusion papers is noise-conditioning augmentation: corrupt
the LR conditioning with a random amount of noise during SR training,
so "slightly wrong LR input" is inside the training distribution.

Stages:
  python3 train.py --stage base       train the 16x16 base video model
  python3 train.py --stage sr         SR arm WITH noise augmentation
  python3 train.py --stage sr_noaug   SR arm WITHOUT (the ablation)
  python3 train.py --stage figures    run the cascade end to end, metrics

Requires project 15's checkpoints/image.pt (for the SR stage init).
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "15-inflate-sd-to-a-video-model"))
import vdm_lib as V
from mmnist import MovingMNIST
import plot_style as ps

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

IMAGE_CK = (HERE.parent / "15-inflate-sd-to-a-video-model"
            / "checkpoints" / "image.pt")
T = V.T_FRAMES
LOW, HIGH = 16, 32
BASE_STEPS = 900
SR_STEPS = 700
AUG_MAX = 0.4            # max std of noise added to the LR condition


def data(seed=1):
    return MovingMNIST(n_digits=2, seq_len=T, seed=seed)


def down(clips):
    """(B,T,1,32,32) -> (B,T,1,16,16) by area averaging."""
    B, t = clips.shape[:2]
    x = clips.reshape(B * t, 1, HIGH, HIGH)
    return F.avg_pool2d(x, 2).reshape(B, t, 1, LOW, LOW)


def up(clips_lr):
    """(B,T,1,16,16) -> (B,T,1,32,32) bilinear."""
    B, t = clips_lr.shape[:2]
    x = clips_lr.reshape(B * t, 1, LOW, LOW)
    x = F.interpolate(x, size=(HIGH, HIGH), mode="bilinear",
                      align_corners=False)
    return x.reshape(B, t, 1, HIGH, HIGH)


def build_base():
    # 16x16 is small enough to train from scratch: two resolution levels
    unet = V.ImageUNet(ch=32, mults=(1, 2))
    return V.VideoDiffusionUNet(unet, freeze_spatial=False,
                                temporal=("conv", "attn"))


def build_sr():
    if not IMAGE_CK.exists():
        raise SystemExit("run project 15's `--stage image` first "
                         f"(missing {IMAGE_CK})")
    unet = V.ImageUNet()
    unet.load_state_dict(torch.load(IMAGE_CK, weights_only=True))
    V.widen_conv_in(unet, 1)          # extra channel: the upsampled LR clip
    return V.VideoDiffusionUNet(unet, freeze_spatial=False,
                                temporal=("conv", "attn"))


def stage_base():
    V.set_seed()
    diff = V.Diffusion()
    mm = data()
    model = build_base()
    print(f"base params: {sum(p.numel() for p in model.parameters()):,}")
    losses = V.train_video(model, diff,
                           lambda: {"clips": down(mm.batch(16))},
                           BASE_STEPS)
    torch.save(model.state_dict(), CK / "base.pt")
    np.save(CK / "loss_base.npy", np.array(losses))


def stage_sr(augment, tag):
    V.set_seed()
    diff = V.Diffusion()
    mm = data()
    model = build_sr()
    rng = np.random.default_rng(0)

    def batch_fn():
        clips = mm.batch(8)
        lr = up(down(clips))                      # what the model upscales
        lr_signed = V.to_signed(lr)
        if augment:
            # each clip gets its own corruption strength, so the model
            # sees everything from "perfect LR" to "quite wrong LR"
            s = torch.from_numpy(
                rng.uniform(0, AUG_MAX, size=(clips.shape[0], 1, 1, 1, 1))
            ).float()
            lr_signed = lr_signed + s * torch.randn_like(lr_signed)
        return {"clips": clips, "x_extra": lr_signed}

    losses = V.train_video(model, diff, batch_fn, SR_STEPS)
    torch.save(model.state_dict(), CK / f"{tag}.pt")
    np.save(CK / f"loss_{tag}.npy", np.array(losses))


@torch.no_grad()
def run_sr(model, diff, lr_clips, seed=0, corrupt=0.0):
    """Upscale LR clips [0,1] -> HIGH-res clips via 60-step DDIM.

    `corrupt` adds test-time noise of that std to the conditioning (the
    same form as the training augmentation) — used by --stage robust.
    """
    x_extra = V.to_signed(up(lr_clips))
    if corrupt > 0:
        g = torch.Generator().manual_seed(seed + 1)
        x_extra = x_extra + corrupt * torch.randn(
            x_extra.shape, generator=g)
    B = lr_clips.shape[0]
    return V.ddim_sample(
        lambda x, t: model(x, t, x_extra=x_extra), diff,
        (B, T, 1, HIGH, HIGH), steps=60, seed=seed)


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


def stage_figures():
    V.set_seed()
    diff = V.Diffusion()
    base = build_base()
    base.load_state_dict(torch.load(CK / "base.pt", weights_only=True))
    base.eval()
    sr_models = {}
    for tag in ["sr", "sr_noaug"]:
        m = build_sr()
        m.load_state_dict(torch.load(CK / f"{tag}.pt", weights_only=True))
        m.eval()
        sr_models[tag] = m

    # ---- 1. reference test: SR on REAL downsampled clips ------------------
    mm = data(seed=99)
    real = mm.batch(8)
    real_lr = down(real)
    rows = []
    sr_on_real = {}
    for tag, m in sr_models.items():
        out = run_sr(m, diff, real_lr, seed=3)
        sr_on_real[tag] = out
        rows.append((tag, "real LR", psnr(out, real),
                     V.sharpness(out.reshape(-1, 1, HIGH, HIGH))))
        print(f"{tag} on real LR: PSNR {rows[-1][2]:.2f} dB", flush=True)
    bilinear = up(real_lr)
    rows.append(("bilinear upsample", "real LR", psnr(bilinear, real),
                 V.sharpness(bilinear.reshape(-1, 1, HIGH, HIGH))))

    # ---- 2. the cascade: base sample -> SR -------------------------------
    print("sampling 16x16 base clips (ancestral)...", flush=True)
    gen_lr = V.ancestral_sample(lambda x, t: base(x, t), diff,
                                (8, T, 1, LOW, LOW), seed=5)
    cascade = {}
    for tag, m in sr_models.items():
        out = run_sr(m, diff, gen_lr, seed=7)
        cascade[tag] = out
        rows.append((tag, "generated LR", float("nan"),
                     V.sharpness(out.reshape(-1, 1, HIGH, HIGH))))
        print(f"{tag} on generated LR: sharpness {rows[-1][3]:.4f}",
              flush=True)

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "input", "psnr_db", "sharpness"])
        for r in rows:
            w.writerow([r[0], r[1],
                        "" if np.isnan(r[2]) else f"{r[2]:.2f}",
                        f"{r[3]:.4f}"])

    # ---- figures ----------------------------------------------------------
    # real-LR test: ground truth / bilinear / no-aug SR / aug SR
    V.save_strip(OUT / "sr_on_real.png",
                 torch.cat([real[:1], bilinear[:1],
                            sr_on_real["sr_noaug"][:1],
                            sr_on_real["sr"][:1]]))
    # the cascade: LR sample (upsampled for display) / no-aug / aug
    V.save_strip(OUT / "cascade.png",
                 torch.cat([up(gen_lr)[:2], cascade["sr_noaug"][:2],
                            cascade["sr"][:2]]))
    V.save_gif(OUT / "cascade.gif", cascade["sr"][0])

    fig, ax = ps.new_axes(6.6, 3.4)
    names = ["bilinear upsample", "SR without aug", "SR with aug"]
    vals = [rows[2][3],
            [r for r in rows if r[0] == "sr_noaug"
             and r[1] == "generated LR"][0][3],
            [r for r in rows if r[0] == "sr"
             and r[1] == "generated LR"][0][3]]
    ref = V.sharpness(real.reshape(-1, 1, HIGH, HIGH))
    y = np.arange(3)[::-1]
    ax.barh(y, vals, color=[ps.BASELINE, ps.SERIES[2], ps.SERIES[1]],
            height=0.55)
    ax.axvline(ref, color=ps.INK_MUTED, linewidth=1.2, linestyle="--")
    ax.text(ref, 2.6, " real clips", color=ps.INK_MUTED, fontsize=9)
    ax.set_yticks(y, names)
    ps.finish(fig, ax, "Sharpness of 32px output in the full cascade",
              "mean gradient magnitude", "", OUT / "sharpness.png")


def stage_robust():
    """How gracefully does each SR arm degrade as its input gets worse?

    Corrupt the LR conditioning of *real* clips (so PSNR has a ground
    truth) with growing test-time noise and upscale with both arms.
    """
    V.set_seed()
    diff = V.Diffusion()
    sr_models = {}
    for tag in ["sr", "sr_noaug"]:
        m = build_sr()
        m.load_state_dict(torch.load(CK / f"{tag}.pt", weights_only=True))
        m.eval()
        sr_models[tag] = m
    mm = data(seed=55)
    real = mm.batch(8)
    real_lr = down(real)
    strengths = [0.0, 0.1, 0.2, 0.3, 0.4]
    curves = {tag: [] for tag in sr_models}
    strip_rows = []
    for s in strengths:
        for tag, m in sr_models.items():
            out = run_sr(m, diff, real_lr, seed=9, corrupt=s)
            curves[tag].append(psnr(out, real))
            print(f"corrupt {s:.1f}  {tag:9s} PSNR {curves[tag][-1]:.2f}",
                  flush=True)
            if s == 0.3:
                strip_rows.append(out[:1])
    V.save_strip(OUT / "robust_at_03.png",
                 torch.cat([real[:1]] + strip_rows))

    with open(OUT / "robust.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corrupt_std", "psnr_sr_aug", "psnr_sr_noaug"])
        for i, s in enumerate(strengths):
            w.writerow([s, f"{curves['sr'][i]:.2f}",
                        f"{curves['sr_noaug'][i]:.2f}"])

    fig, ax = ps.new_axes(6.6, 4.0)
    ax.plot(strengths, curves["sr"], color=ps.SERIES[1], marker="o",
            label="SR with noise aug")
    ax.plot(strengths, curves["sr_noaug"], color=ps.SERIES[2], marker="o",
            label="SR without aug")
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax,
              "Robustness to a corrupted LR condition (real clips)",
              "test-time corruption std", "PSNR (dB)",
              OUT / "robust.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["base", "sr", "sr_noaug", "figures", "robust"])
    a = ap.parse_args()
    torch.set_num_threads(12)
    if a.stage == "base":
        stage_base()
    elif a.stage == "sr":
        stage_sr(True, "sr")
    elif a.stage == "sr_noaug":
        stage_sr(False, "sr_noaug")
    elif a.stage == "robust":
        stage_robust()
    else:
        stage_figures()
