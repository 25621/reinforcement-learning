"""Train the ConvLSTM predictor on Moving MNIST and make the figures.

Usage:
  python3 train.py            # train (~8 min CPU), save checkpoint + figures
  python3 train.py --plot     # remake figures from the saved checkpoint
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "01-video-loader-benchmark"))
import plot_style as ps  # noqa: E402

from mmnist import MovingMNIST  # noqa: E402
from predictor import Predictor  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
T_IN, T_OUT = 10, 10
STEPS = 1100         # first TF_STEPS teacher-forced, rest closed-loop
TF_STEPS = 800
BATCH = 16


def train():
    torch.manual_seed(0)
    torch.set_num_threads(12)
    data = MovingMNIST(n_digits=2, seq_len=T_IN + T_OUT, seed=1)
    model = Predictor()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e3:.0f}k")
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    losses = []
    t0 = time.time()
    for step in range(STEPS):
        clips = data.batch(BATCH)
        if step < TF_STEPS:
            # phase 1 — teacher forcing: predict every next frame from
            # the real previous frames (stable, dense learning signal)
            logits = model.teacher_forced(clips)
            loss = F.binary_cross_entropy_with_logits(logits, clips[:, 1:])
        else:
            # phase 2 — closed-loop fine-tune: predict 10 future frames
            # feeding the model its own outputs, exactly like test time
            logits = model(clips[:, :T_IN], T_OUT)
            loss = F.binary_cross_entropy_with_logits(logits, clips[:, T_IN:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 25 == 0 or step == STEPS - 1:
            phase = "TF" if step < TF_STEPS else "CL"
            print(f"step {step:4d} [{phase}]  loss {loss.item():.5f}  "
                  f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True)
    CKPT.mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(),
                "losses": np.array(losses)}, CKPT / "predictor.pt")
    return model, np.array(losses)


def load():
    ck = torch.load(CKPT / "predictor.pt", weights_only=False)
    model = Predictor()
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["losses"]


def strip(frames, upscale=3):
    """(T, 1, H, W) tensor -> one wide uint8 image with 1px separators."""
    import cv2
    imgs = []
    for f in frames:
        img = (f.squeeze(0).clamp(0, 1).numpy() * 255).astype(np.uint8)
        img = cv2.resize(img, None, fx=upscale, fy=upscale,
                         interpolation=cv2.INTER_NEAREST)
        imgs.append(img)
    h = imgs[0].shape[0]
    sep = np.full((h, 2), 120, dtype=np.uint8)
    row = []
    for i, img in enumerate(imgs):
        row.append(img)
        if i < len(imgs) - 1:
            row.append(sep)
    return np.concatenate(row, axis=1)


def figures(model, losses):
    import cv2
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(3)
    data = MovingMNIST(n_digits=2, seq_len=T_IN + T_OUT, train=False, seed=99)
    with torch.no_grad():
        clips = data.batch(64)
        pred = torch.sigmoid(model(clips[:, :T_IN], T_OUT))
    target = clips[:, T_IN:]
    last = clips[:, T_IN - 1:T_IN].expand_as(target)   # copy-last baseline

    # --- prediction strips: truth vs model vs copy-last, 2 examples -------
    rows, labels = [], []
    for b in (0, 1):
        rows += [strip(clips[b]),                       # 20 true frames
                 strip(torch.cat([clips[b, :T_IN], pred[b]], 0)),
                 strip(torch.cat([clips[b, :T_IN], last[b]], 0))]
        labels += [f"clip {b+1}: ground truth",
                   "model: 10 context + 10 predicted",
                   "baseline: copy last context frame"]
    pad = 26
    w = rows[0].shape[1]
    sheet = []
    for lab, img in zip(labels, rows):
        head = np.full((pad, w), 252, dtype=np.uint8)
        cv2.putText(head, lab, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1,
                    cv2.LINE_AA)
        sheet += [head, img, np.full((6, w), 252, dtype=np.uint8)]
    sheet = np.concatenate(sheet, axis=0)
    # mark where prediction starts
    x0 = strip(clips[0, :T_IN]).shape[1] + 2
    cv2.line(sheet, (x0, 0), (x0, sheet.shape[0]), 60, 1)
    cv2.imwrite(str(OUT / "prediction_strips.png"), sheet)
    print("wrote", OUT / "prediction_strips.png")

    # --- error vs how far ahead we predict --------------------------------
    mse_model = ((pred - target) ** 2).mean(dim=(0, 2, 3, 4)).numpy()
    mse_last = ((last - target) ** 2).mean(dim=(0, 2, 3, 4)).numpy()
    fig, ax = ps.new_axes(6.4, 4.0)
    xs = np.arange(1, T_OUT + 1)
    ax.plot(xs, mse_model, color=ps.SERIES[0], lw=2, marker="o", ms=4,
            label="ConvLSTM prediction")
    ax.plot(xs, mse_last, color=ps.SERIES[2], lw=2, marker="o", ms=4,
            label="copy last frame")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Prediction error grows with the horizon",
              "frames ahead", "per-pixel MSE", OUT / "horizon_error.png")

    # --- hedging: peak brightness of predicted vs real frames -------------
    peak_pred = pred.amax(dim=(2, 3, 4)).mean(0).numpy()
    peak_real = target.amax(dim=(2, 3, 4)).mean(0).numpy()
    fig, ax = ps.new_axes(6.4, 4.0)
    ax.plot(xs, peak_real, color=ps.INK_MUTED, lw=2, ls="--",
            label="real frames (ink is white)")
    ax.plot(xs, peak_pred, color=ps.SERIES[0], lw=2, marker="o", ms=4,
            label="predicted frames")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "The model hedges: no predicted pixel is ever bright",
              "frames ahead", "brightest pixel in the frame",
              OUT / "hedging.png")

    # --- training loss curve ----------------------------------------------
    fig, ax = ps.new_axes(6.4, 3.6)
    k = 20
    smooth = np.convolve(losses, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(len(smooth)) + k, smooth, color=ps.SERIES[0], lw=2)
    ax.axvline(TF_STEPS, color=ps.INK_MUTED, ls="--", lw=1.5)
    ax.text(TF_STEPS + 8, smooth.max() * 0.7,
            "switch to closed-loop", color=ps.INK_MUTED, fontsize=9)
    ax.set_yscale("log")
    ps.finish(fig, ax, "Training loss (teacher-forced, then closed-loop)",
              "training step", "per-pixel BCE (log scale)", OUT / "loss.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()
    if args.steps:
        STEPS = args.steps
    if args.plot:
        model, losses = load()
    else:
        model, losses = train()
    figures(model, losses)
