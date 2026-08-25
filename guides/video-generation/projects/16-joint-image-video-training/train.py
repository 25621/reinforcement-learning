"""Project 16 — joint image-video training vs video-only fine-tuning.

The real-world situation being reproduced: you inflate a pretrained
image model and fine-tune it *unfrozen* on video.  Video datasets are
smaller and much more compressed than image datasets, so video-only
fine-tuning drags the spatial weights toward the degraded video look —
still-image quality quietly collapses.  Joint training mixes clean
images (as 1-frame videos) into the same run to hold that quality in
place.

Here the clean image source is pristine Moving-MNIST frames and the
video source is the same clips passed through JPEG compression at
quality 25 (the honest stand-in for "web video is heavily compressed",
Phase 1's point).

Stages:
  python3 train.py --stage video     100% (degraded) video fine-tune
  python3 train.py --stage joint     90% clean images / 10% video
  python3 train.py --stage figures   sample stills + clips, metrics, plots

Requires project 15's pretrained image model:
  ../15-inflate-sd-to-a-video-model/checkpoints/image.pt
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

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
STEPS = 700
T = V.T_FRAMES
JPEG_QUALITY = 25
P_IMAGE = 0.9           # joint arm: fraction of steps that use image batches


def degrade(clips):
    """JPEG-compress every frame at low quality (in place of a codec)."""
    arr = (clips[:, :, 0].numpy() * 255).astype(np.uint8)
    out = np.empty_like(arr)
    flag = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    for b in range(arr.shape[0]):
        for t in range(arr.shape[1]):
            ok, enc = cv2.imencode(".jpg", arr[b, t], flag)
            out[b, t] = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(out).float().unsqueeze(2) / 255.0


def build_model():
    if not IMAGE_CK.exists():
        raise SystemExit("run project 15's `--stage image` first "
                         f"(missing {IMAGE_CK})")
    unet = V.ImageUNet()
    unet.load_state_dict(torch.load(IMAGE_CK, weights_only=True))
    # unfrozen: the whole point is watching spatial weights drift
    return V.VideoDiffusionUNet(unet, freeze_spatial=False,
                                temporal=("conv", "attn"))


def stage_train(joint):
    V.set_seed()
    diff = V.Diffusion()
    mm = MovingMNIST(n_digits=2, seq_len=T, seed=1)
    model = build_model()
    rng = np.random.default_rng(0)

    def batch_fn():
        if joint and rng.random() < P_IMAGE:
            # clean stills as 1-frame clips through the same network
            frames = mm.batch(32)[:, :1]
            return {"clips": frames}
        return {"clips": degrade(mm.batch(8))}

    losses = V.train_video(model, diff, batch_fn, STEPS)
    tag = "joint" if joint else "video"
    torch.save(model.state_dict(), CK / f"{tag}.pt")
    np.save(CK / f"loss_{tag}.npy", np.array(losses))


@torch.no_grad()
def eval_frame_loss(model, diff, frames, n_rep=4, seed=0):
    """Epsilon-prediction MSE on given clean/degraded frames as T=1 clips.

    Averaged over several fixed noise draws and timesteps so the two
    arms see identical noise — differences are model, not luck.
    """
    torch.manual_seed(seed)
    x0 = V.to_signed(frames)[:, None]           # (B,1,1,H,W)
    total = 0.0
    for r in range(n_rep):
        t = torch.randint(0, diff.T, (x0.shape[0],))
        noise = torch.randn_like(x0)
        x_t = diff.q_sample(x0, t, noise)
        total += torch.nn.functional.mse_loss(model(x_t, t), noise).item()
    return total / n_rep


def stage_figures():
    V.set_seed()
    diff = V.Diffusion()
    mm = MovingMNIST(n_digits=2, seq_len=T, seed=99)
    real = mm.batch(64)
    clean_frames = real[:, 0]                    # (64,1,32,32)
    degr_frames = degrade(real)[:, 0]

    pretrained = build_model()                   # image weights, identity temporal
    models = {"pretrained image model": pretrained}
    for tag, name in [("joint", "joint 90/10"),
                      ("video", "video-only")]:
        m = build_model()
        m.load_state_dict(torch.load(CK / f"{tag}.pt", weights_only=True))
        m.eval()
        models[name] = m

    # ---- still-image generation (T=1 sampling from noise) -----------------
    stills, sharp = {}, {}
    for name, m in models.items():
        s = V.ancestral_sample(lambda x, t: m(x, t), diff,
                               (32, 1, 1, V.CANVAS, V.CANVAS), seed=3)
        stills[name] = s[:, 0]
        sharp[name] = V.sharpness(s[:, 0])
        print(f"sampled stills: {name}  sharpness {sharp[name]:.4f}",
              flush=True)
    sharp["real clean frames"] = V.sharpness(clean_frames)
    sharp["real degraded frames"] = V.sharpness(degr_frames)

    # ---- forgetting, measured as eval loss on clean vs degraded frames ----
    rows = []
    for name, m in models.items():
        lc = eval_frame_loss(m, diff, clean_frames)
        ld = eval_frame_loss(m, diff, degr_frames)
        rows.append((name, lc, ld))
        print(f"{name:28s} eval-loss clean {lc:.4f}  degraded {ld:.4f}")

    # ---- video generation ------------------------------------------------
    clips, vid_metrics = {}, []
    for name in ["joint 90/10", "video-only"]:
        c = V.ancestral_sample(lambda x, t: models[name](x, t), diff,
                               (8, T, 1, V.CANVAS, V.CANVAS), seed=5)
        clips[name] = c
        vid_metrics.append((name, V.flicker(c), V.align_response(c),
                            V.sharpness(c.reshape(-1, 1, V.CANVAS,
                                                  V.CANVAS))))
        print(f"sampled clips: {name}", flush=True)
    real_small = real[:8]
    vid_metrics.append(("real degraded clips",
                        V.flicker(degrade(real_small)),
                        V.align_response(degrade(real_small)),
                        V.sharpness(degrade(real_small).reshape(
                            -1, 1, V.CANVAS, V.CANVAS))))

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "still_sharpness"])
        for name, v in sharp.items():
            w.writerow([name, f"{v:.4f}"])
        w.writerow([])
        w.writerow(["model", "eval_loss_clean", "eval_loss_degraded"])
        for name, lc, ld in rows:
            w.writerow([name, f"{lc:.4f}", f"{ld:.4f}"])
        w.writerow([])
        w.writerow(["model", "flicker", "align_response", "frame_sharpness"])
        for name, fl, al, sh in vid_metrics:
            w.writerow([name, f"{fl:.4f}", f"{al:.4f}", f"{sh:.4f}"])

    # ---- figures ---------------------------------------------------------
    from PIL import Image
    grids = []
    for name in ["pretrained image model", "joint 90/10", "video-only"]:
        row = stills[name][:10]                 # 10 stills side by side
        grids.append(torch.cat([row[i, 0] for i in range(10)], dim=1))
    g = torch.cat(grids, dim=0).clamp(0, 1).numpy()
    img = (np.kron(g, np.ones((3, 3))) * 255).astype(np.uint8)
    Image.fromarray(img).save(OUT / "stills_compare.png")
    print("wrote", OUT / "stills_compare.png")

    V.save_strip(OUT / "clips_compare.png",
                 torch.cat([degrade(real_small)[:2],
                            clips["joint 90/10"][:2],
                            clips["video-only"][:2]]))

    fig, ax = ps.new_axes(6.4, 4.0)
    names = ["real clean frames", "pretrained image model",
             "joint 90/10", "video-only", "real degraded frames"]
    vals = [sharp[n] for n in names]
    colors = [ps.BASELINE, ps.SERIES[0], ps.SERIES[1], ps.SERIES[2],
              ps.BASELINE]
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors, height=0.6)
    ax.set_yticks(y, names)
    for yi, v in zip(y, vals):
        ax.text(v + 0.001, yi, f"{v:.3f}", va="center", fontsize=9,
                color=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Gradient energy of generated stills",
              "mean gradient magnitude (clean look = the 'real clean' bar; "
              "JPEG artifacts push higher)", "", OUT / "sharpness.png")

    fig, ax = ps.new_axes(6.4, 3.6)
    x = np.arange(len(rows))
    ax.bar(x - 0.18, [r[1] for r in rows], width=0.36,
           color=ps.SERIES[0], label="clean frames")
    ax.bar(x + 0.18, [r[2] for r in rows], width=0.36,
           color=ps.SERIES[2], label="degraded frames")
    ax.set_xticks(x, [r[0].replace(" ", "\n", 1) for r in rows])
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Epsilon-prediction eval loss on held-out frames",
              "", "MSE (lower = fits that data better)",
              OUT / "eval_loss.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["video", "joint", "figures"])
    a = ap.parse_args()
    torch.set_num_threads(12)
    if a.stage == "video":
        stage_train(joint=False)
    elif a.stage == "joint":
        stage_train(joint=True)
    else:
        stage_figures()
