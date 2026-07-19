"""Train the content/motion VAE on one-digit Moving MNIST, then test the
split: swap latents between clips and probe what each latent knows.

Usage:
  python3 train.py           # train (~4 min CPU) + figures
  python3 train.py --plot    # figures only, from the saved checkpoint
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
sys.path.insert(0, str(HERE.parents[0] / "06-moving-mnist-predictor"))
import plot_style as ps  # noqa: E402
import vid_lib  # noqa: E402
from mmnist import MovingMNISTWithState  # noqa: E402

from comovae import CoMoVAE  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
T = 8
STEPS = 3000
BATCH = 32


def train():
    torch.manual_seed(0)
    torch.set_num_threads(12)
    data = MovingMNISTWithState(seq_len=T, seed=1)
    model = CoMoVAE()
    print(f"params: {sum(p.numel() for p in model.parameters())/1e3:.0f}k")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = time.time()
    for step in range(STEPS):
        clips = data.batch(BATCH)
        logits, kl_c, kl_m = model(clips)
        recon = F.binary_cross_entropy_with_logits(
            logits, clips, reduction="none").sum((1, 2, 3, 4)).mean()
        # lighter KL pressure on content: keep more identity detail
        loss = recon + 0.5 * kl_c + kl_m
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == STEPS - 1:
            print(f"step {step:4d}  recon {recon.item():7.1f}  "
                  f"kl_c {kl_c.item():5.1f}  kl_m {kl_m.item():5.1f}  "
                  f"{(time.time()-t0)/(step+1):.2f}s/step", flush=True)
    CKPT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT / "comovae.pt")
    return model


def load():
    model = CoMoVAE()
    model.load_state_dict(torch.load(CKPT / "comovae.pt", weights_only=True))
    model.eval()
    return model


def to_img(frame_logits, upscale=3):
    import cv2
    img = (torch.sigmoid(frame_logits).squeeze().numpy() * 255).astype(np.uint8)
    return cv2.resize(img, None, fx=upscale, fy=upscale,
                      interpolation=cv2.INTER_NEAREST)


def strip(frames_logits):
    imgs = [to_img(f) for f in frames_logits]
    sep = np.full((imgs[0].shape[0], 2), 120, dtype=np.uint8)
    out = []
    for i, im in enumerate(imgs):
        out.append(im)
        if i < len(imgs) - 1:
            out.append(sep)
    return np.concatenate(out, axis=1)


def figures(model):
    import cv2
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(2)
    data = MovingMNISTWithState(seq_len=T, train=False, seed=77)
    clips, labels, positions = data.batch_with_state(1024)
    with torch.no_grad():
        mu_c, _, mu_m, _ = model.encode(clips)
        recon = model.decode_frames(mu_c, mu_m)

    # --- reconstruction strips -------------------------------------------
    rows, names = [], []
    for b in (0, 1):
        rows += [strip(torch.logit(clips[b].clamp(1e-4, 1 - 1e-4))),
                 strip(recon[b])]
        names += [f"clip {b+1}: original", "reconstruction"]
    sheet = label_rows(rows, names)
    cv2.imwrite(str(OUT / "reconstruction.png"), sheet)
    print("wrote", OUT / "reconstruction.png")

    # --- the swap grid: content of A + motion of B ------------------------
    # three clips with visually distinct digits (a 0, a 1, and a 7)
    picks = [int((labels == d).nonzero()[0, 0]) for d in (0, 1, 7)]
    grid_rows = []
    names = []
    for a in picks:
        row = []
        for b in picks:
            with torch.no_grad():
                mixed = model.decode_frames(mu_c[a:a + 1], mu_m[b:b + 1])
            row.append(strip(mixed[0]))
        grid_rows.append(np.concatenate(
            [np.full((row[0].shape[0], 20), 252, np.uint8)] +
            [np.concatenate([r, np.full((r.shape[0], 14), 252, np.uint8)],
                            axis=1) for r in row], axis=1))
        names.append(f"content: digit {labels[a].item()}")
    # header row: the motion-source clips
    head = []
    for b in picks:
        head.append(strip(torch.logit(clips[b].clamp(1e-4, 1 - 1e-4))))
    head_row = np.concatenate(
        [np.full((head[0].shape[0], 20), 252, np.uint8)] +
        [np.concatenate([h, np.full((h.shape[0], 14), 252, np.uint8)],
                        axis=1) for h in head], axis=1)
    sheet = label_rows([head_row] + grid_rows,
                       ["motion sources (originals)"] + names)
    cv2.imwrite(str(OUT / "swap_grid.png"), sheet)
    print("wrote", OUT / "swap_grid.png")

    # --- what does the 2-D motion latent encode? --------------------------
    zm = mu_m.reshape(-1, 2).numpy()
    pos = positions.reshape(-1, 2).numpy()
    fig, axes = ps.plt.subplots(1, 2, figsize=(9.6, 4.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, coord, name in zip(axes, [1, 0], ["x (left-right)",
                                              "y (up-down)"]):
        ps.style_axes(ax)
        sc = ax.scatter(zm[:, 0], zm[:, 1], c=pos[:, coord], s=4,
                        cmap="viridis", rasterized=True)
        ax.set_title(f"colored by true {name} position", fontsize=10,
                     color=ps.INK)
        ax.set_xlabel("motion latent dim 1", fontsize=9,
                      color=ps.INK_SECONDARY)
        ax.set_ylabel("motion latent dim 2", fontsize=9,
                      color=ps.INK_SECONDARY)
        fig.colorbar(sc, ax=ax, shrink=0.85)
    fig.suptitle("The 2-D motion latent is a map of the digit's position",
                 fontsize=12, color=ps.INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "motion_latent.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    ps.plt.close(fig)
    print("wrote", OUT / "motion_latent.png")

    # --- probes: which latent knows what? ---------------------------------
    res = {}
    res["digit from content"] = probe_classify(mu_c, labels)
    zm_first = mu_m[:, 0]                    # motion code of frame 0
    res["digit from motion"] = probe_classify(zm_first, labels)
    res["position from motion"] = probe_regress(mu_m.reshape(-1, 2),
                                                positions.reshape(-1, 2))
    res["position from content"] = probe_regress(
        mu_c[:, None].expand(-1, T, -1).reshape(-1, model.c_dim),
        positions.reshape(-1, 2))
    for k, v in res.items():
        print(f"{k}: {v:.3f}")

    fig, ax = ps.new_axes(6.8, 3.6)
    names = list(res)
    vals = [res[n] for n in names]
    colors = [ps.SERIES[0], ps.SERIES[2], ps.SERIES[0], ps.SERIES[2]]
    ax.barh(np.arange(len(names))[::-1], vals, color=colors, height=0.6)
    ax.set_yticks(np.arange(len(names))[::-1], names, fontsize=9)
    ax.axvline(0.1, color=ps.INK_MUTED, ls="--", lw=1)
    ax.text(0.105, 3.3, "chance (digits)", color=ps.INK_MUTED, fontsize=8)
    ps.finish(fig, ax, "Each latent only knows its own job",
              "probe accuracy (digits) / R² (position)", "",
              OUT / "probes.png")
    (OUT / "probes.txt").write_text(
        "".join(f"{k}: {v:.3f}\n" for k, v in res.items()))


def label_rows(rows, names, pad=24):
    import cv2
    w = max(r.shape[1] for r in rows)
    sheet = []
    for name, img in zip(names, rows):
        if img.shape[1] < w:
            img = np.concatenate(
                [img, np.full((img.shape[0], w - img.shape[1]), 252,
                              np.uint8)], axis=1)
        head = np.full((pad, w), 252, dtype=np.uint8)
        cv2.putText(head, name, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, 0,
                    1, cv2.LINE_AA)
        sheet += [head, img, np.full((6, w), 252, np.uint8)]
    return np.concatenate(sheet, axis=0)


def probe_classify(feats, labels, steps=400):
    """Accuracy of a linear (logistic-regression) probe, 10 digit classes."""
    n = len(feats)
    tr = slice(0, int(n * 0.8))
    te = slice(int(n * 0.8), n)
    w = torch.zeros(feats.shape[1], 10, requires_grad=True)
    b = torch.zeros(10, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.1)
    for _ in range(steps):
        loss = F.cross_entropy(feats[tr] @ w + b, labels[tr])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = (feats[te] @ w + b).argmax(-1)
    return (pred == labels[te]).float().mean().item()


def probe_regress(feats, target):
    """R² of a linear least-squares probe predicting (y, x) position."""
    X = torch.cat([feats, torch.ones(len(feats), 1)], dim=1)
    n = len(X)
    tr = slice(0, int(n * 0.8))
    te = slice(int(n * 0.8), n)
    beta = torch.linalg.lstsq(X[tr], target[tr]).solution
    pred = X[te] @ beta
    ss_res = ((pred - target[te]) ** 2).sum()
    ss_tot = ((target[te] - target[te].mean(0)) ** 2).sum()
    return (1 - ss_res / ss_tot).item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    model = load() if args.plot else train()
    model.eval()
    figures(model)
