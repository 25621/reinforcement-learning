"""Project 17 — classifier-free guidance with two independent dials.

A video model conditioned on BOTH a semantic label (digit class — our
stand-in for the text prompt) and a conditioning image (the first
frame).  Both conditions are randomly and *independently* dropped
during training, so at sampling time each can be guided with its own
scale:

    eps = e(none, none)
        + s_img * ( e(img, none) - e(none, none) )     image direction
        + s_cls * ( e(img, cls)  - e(img,  none) )     class direction

Stages:
  python3 train.py --stage imgcls     class-conditional training on FRAMES
  python3 train.py --stage train      dual-condition video fine-tune
  python3 train.py --stage clf        train the digit classifier (judge)
  python3 train.py --stage figures    guidance-grid sweep, metrics, plots

The imgcls stage exists because conditioning is learned at *image*
scale and inherited by the video fine-tune (as in the real pipeline:
Stable Diffusion learned text conditioning from images; video models
inherit it).  Training the class pathway from video batches alone was
measured to be hopeless here: small batches, and a label that is
redundant whenever the conditioning frame is present, left the class
dial with literally zero effect on samples.

Requires project 15's checkpoints/image.pt.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
IMGCLS_STEPS = 1200
STEPS = 1000
T = V.T_FRAMES
NULL_CLS = 10                  # embedding index for "no class given"
# Independent condition dropout.  The image is dropped more often than
# usual (30%) on purpose: while the conditioning frame is present, the
# class label is redundant (the digit is visible!), so gradients only
# teach the class pathway on the image-dropped steps.  Too little
# dropout = a class dial that turns nothing (we measured exactly that
# with 15%/15%: every class-guided sample identical to unguided).
P_DROP_IMG = 0.30
P_DROP_CLS = 0.20
SCALES = [1, 3, 9]             # guidance grid (1 = no extra push)
N_PER_CELL = 6


def data(seed=1):
    return MovingMNIST(n_digits=1, seq_len=T, seed=seed)


def build_model():
    if not IMAGE_CK.exists():
        raise SystemExit("run project 15's `--stage image` first "
                         f"(missing {IMAGE_CK})")
    unet = V.ImageUNet()
    unet.load_state_dict(torch.load(IMAGE_CK, weights_only=True))
    V.widen_conv_in(unet, 1)          # extra channel: the conditioning frame
    return V.VideoDiffusionUNet(unet, freeze_spatial=False,
                                temporal=("conv", "attn"), n_classes=10)


def cond_channel(cond, drop_mask=None):
    """(B,1,H,W) clean frame -> (B,T,1,H,W) signed, tiled to every frame.

    Dropped conditions become all-zero maps.  Zero is mid-grey in the
    signed range — a value no real Moving-MNIST frame is made of (the
    background is -1) — so "no image given" is unambiguous to the model.
    """
    c = V.to_signed(cond)[:, None].expand(-1, T, -1, -1, -1).clone()
    if drop_mask is not None:
        c[drop_mask] = 0.0
    return c


def stage_imgcls():
    """Teach the class condition on single frames, where it is cheap.

    Batch 64 frames per step (8x the clip batch) and no conditioning
    frame in sight, so the class label is the only source of identity
    information — the two properties the video stage lacks.
    """
    V.set_seed()
    diff = V.Diffusion()
    mm = MovingMNIST(n_digits=1, seq_len=1, seed=1)
    model = build_model()
    rng = np.random.default_rng(0)

    def batch_fn():
        frames, labels = mm.batch(64, return_labels=True)
        cls = labels[:, 0].clone()
        cls[rng.random(64) < P_DROP_CLS] = NULL_CLS
        return {"clips": frames, "cls": cls,
                "x_extra": torch.zeros(64, 1, 1, V.CANVAS, V.CANVAS)}

    losses = V.train_video(model, diff, batch_fn, IMGCLS_STEPS)
    torch.save(model.state_dict(), CK / "imgcls.pt")
    np.save(CK / "loss_imgcls.npy", np.array(losses))


def stage_train():
    V.set_seed()
    diff = V.Diffusion()
    mm = data()
    model = build_model()
    model.load_state_dict(torch.load(CK / "imgcls.pt", weights_only=True))
    rng = np.random.default_rng(0)

    def batch_fn():
        clips, labels = mm.batch(8, return_labels=True)
        cls = labels[:, 0].clone()
        cls[rng.random(8) < P_DROP_CLS] = NULL_CLS
        img_drop = torch.from_numpy(rng.random(8) < P_DROP_IMG)
        return {"clips": clips, "cls": cls,
                "x_extra": cond_channel(clips[:, 0], img_drop)}

    losses = V.train_video(model, diff, batch_fn, STEPS)
    torch.save(model.state_dict(), CK / "model.pt")
    np.save(CK / "loss.npy", np.array(losses))


# ---------------------------------------------------------------------------
# The judge: a small digit classifier on canvas frames
# ---------------------------------------------------------------------------

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
            nn.Linear(128, 10))

    def forward(self, x):
        return self.net(x)


def stage_clf():
    V.set_seed()
    mm = data(seed=2)
    clf = Classifier()
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    for step in range(2500):
        clips, labels = mm.batch(16, return_labels=True)
        frames = clips.reshape(-1, 1, V.CANVAS, V.CANVAS)
        y = labels[:, 0].repeat_interleave(T)
        loss = F.cross_entropy(clf(frames), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            acc = (clf(frames).argmax(1) == y).float().mean().item()
            print(f"  clf step {step:4d}  loss {loss.item():.3f}  "
                  f"acc {acc:.3f}", flush=True)
    mm_test = data(seed=77)
    clips, labels = mm_test.batch(64, return_labels=True)
    frames = clips.reshape(-1, 1, V.CANVAS, V.CANVAS)
    y = labels[:, 0].repeat_interleave(T)
    with torch.no_grad():
        acc = (clf(frames).argmax(1) == y).float().mean().item()
    print(f"held-out classifier accuracy: {acc:.4f}")
    torch.save(clf.state_dict(), CK / "clf.pt")


# ---------------------------------------------------------------------------
# Guided sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_sample(model, diff, cond, cls, s_img, s_cls, steps=80, seed=0):
    """cond (B,1,H,W) in [0,1], cls (B,) long -> clips (B,T,1,H,W)."""
    B = cond.shape[0]
    c_img = cond_channel(cond)
    c_null = torch.zeros_like(c_img)
    null_cls = torch.full((B,), NULL_CLS, dtype=torch.long)

    def eps_fn(x, t):
        e00 = model(x, t, cls=null_cls, x_extra=c_null)
        ei0 = model(x, t, cls=null_cls, x_extra=c_img)
        eic = model(x, t, cls=cls, x_extra=c_img)
        return e00 + s_img * (ei0 - e00) + s_cls * (eic - ei0)

    return V.ddim_sample(eps_fn, diff, (B, T, 1, V.CANVAS, V.CANVAS),
                         steps=steps, seed=seed)


def stage_figures():
    V.set_seed()
    diff = V.Diffusion()
    model = build_model()
    model.load_state_dict(torch.load(CK / "model.pt", weights_only=True))
    model.eval()
    clf = Classifier()
    clf.load_state_dict(torch.load(CK / "clf.pt", weights_only=True))
    clf.eval()

    mm = data(seed=123)
    clips, labels = mm.batch(N_PER_CELL, return_labels=True)
    cond = clips[:, 0]
    cls = labels[:, 0]
    print("condition digits:", cls.tolist())

    # ---- the guidance grid -----------------------------------------------
    adher = np.zeros((len(SCALES), len(SCALES)))
    fidel = np.zeros_like(adher)
    flick = np.zeros_like(adher)
    cell_strips = {}
    for i, s_cls in enumerate(SCALES):
        for j, s_img in enumerate(SCALES):
            s = guided_sample(model, diff, cond, cls, s_img, s_cls, seed=11)
            frames = s.reshape(-1, 1, V.CANVAS, V.CANVAS)
            with torch.no_grad():
                pred = clf(frames).argmax(1).reshape(N_PER_CELL, T)
            adher[i, j] = (pred == cls[:, None]).float().mean().item()
            fidel[i, j] = F.mse_loss(s[:, 0], cond).item()
            flick[i, j] = V.flicker(s)
            cell_strips[(s_cls, s_img)] = s
            print(f"s_cls={s_cls} s_img={s_img}  adherence {adher[i,j]:.3f}"
                  f"  fidelity-MSE {fidel[i,j]:.4f}  flicker {flick[i,j]:.4f}",
                  flush=True)

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s_cls", "s_img", "class_adherence",
                    "cond_fidelity_mse", "flicker"])
        for i, s_cls in enumerate(SCALES):
            for j, s_img in enumerate(SCALES):
                w.writerow([s_cls, s_img, f"{adher[i,j]:.4f}",
                            f"{fidel[i,j]:.4f}", f"{flick[i,j]:.4f}"])

    # heatmaps
    for name, mat, fmt, title in [
            ("adherence", adher, "{:.2f}",
             "Class adherence (judge accuracy on generated frames)"),
            ("fidelity", fidel, "{:.3f}",
             "Condition-frame fidelity (MSE, lower = closer)"),
            ("flicker", flick, "{:.3f}",
             "Flicker (mean adjacent-frame change; real clips ~0.03)")]:
        fig, ax = ps.new_axes(5.4, 4.4)
        ax.grid(False)
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(len(SCALES)), [str(s) for s in SCALES])
        ax.set_yticks(range(len(SCALES)), [str(s) for s in SCALES])
        for i in range(len(SCALES)):
            for j in range(len(SCALES)):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center",
                        va="center", color="white", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ps.finish(fig, ax, title, "image guidance s_img",
                  "class guidance s_cls", OUT / f"grid_{name}.png")

    # one condition rendered at all nine settings
    rows = torch.cat([cell_strips[(sc, si)][:1]
                      for sc in SCALES for si in SCALES])
    V.save_strip(OUT / "grid_samples.png", rows)

    # ---- tug of war: image says one digit, class says another -------------
    mism = torch.tensor([(int(c) + 5) % 10 for c in cls])
    rows, labels_txt = [], []
    for s_cls in [0, 1, 3, 9]:
        s = guided_sample(model, diff, cond, mism, 4, s_cls, seed=13)
        rows.append(s[:1])
        frames = s.reshape(-1, 1, V.CANVAS, V.CANVAS)
        with torch.no_grad():
            pred = clf(frames).argmax(1).reshape(N_PER_CELL, T)
        won = (pred == mism[:, None]).float().mean().item()
        kept = (pred == cls[:, None]).float().mean().item()
        labels_txt.append((s_cls, won, kept))
        print(f"tug-of-war s_cls={s_cls}: class wins {won:.2f}, "
              f"image digit kept {kept:.2f}", flush=True)
    V.save_strip(OUT / "tug_of_war.png", torch.cat(rows))
    with open(OUT / "tug_of_war.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["s_cls", "frac_frames_class_wins",
                    "frac_frames_image_digit"])
        for r in labels_txt:
            w.writerow([r[0], f"{r[1]:.3f}", f"{r[2]:.3f}"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["imgcls", "train", "clf", "figures"])
    a = ap.parse_args()
    torch.set_num_threads(12)
    if a.stage == "imgcls":
        stage_imgcls()
    elif a.stage == "train":
        stage_train()
    elif a.stage == "clf":
        stage_clf()
    else:
        stage_figures()
