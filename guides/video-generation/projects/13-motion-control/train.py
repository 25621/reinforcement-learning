"""Motion control: add a motion-bucket input to the tiny I2V model.

Reuses project 12's frozen image U-Net and VideoUNet; trains fresh
temporal layers whose FiLM input is an embedding of the clip's motion
bucket — a number derived from measured optical flow, exactly the
mechanism Stable Video Diffusion uses.

  python3 train.py --stage train      # ~5 min: calibrate buckets, train
  python3 train.py --stage figures    # ~4 min: sample buckets, plot
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(12)

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-tiny-i2v-model"))

import i2v_lib as lib                                   # noqa: E402
from i2v_lib import (Diffusion, ImageUNet, VideoUNet, timestep_embedding,
                     train_video_model, sample_clip, strip, set_seed)
from mmnist import MovingMNIST                          # noqa: E402
import plot_style as ps                                 # noqa: E402

CKPT = HERE / "checkpoints"
OUT = HERE / "outputs"
CKPT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
CKPT12 = HERE.parent / "12-tiny-i2v-model" / "checkpoints"

T_FRAMES = 8
BATCH = 8
STEPS = 900
N_BUCKETS = 8
EMB_DIM = 128

# One digit per clip, and a much wider speed range than the default so
# there is real motion variety for the buckets to describe.
DATA_KW = dict(n_digits=1, seq_len=T_FRAMES, min_speed=0.2, max_speed=3.0)


# ---------------------------------------------------------------------------
# Motion score: measured from pixels, never from the generator's internals
# ---------------------------------------------------------------------------

def flow_score(clip):
    """Mean Farneback optical-flow magnitude over a clip.

    clip: (T, 1, H, W) float in [0,1].  We measure motion the way SVD's
    data pipeline does — from the pixels — because for real videos the
    'true' object speed is unknowable.
    """
    frames = (clip[:, 0].numpy() * 255).astype(np.uint8)
    mags = []
    for a, b in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale=0.5, levels=2, winsize=7,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0)
        mags.append(np.linalg.norm(flow, axis=-1).mean())
    return float(np.mean(mags))


def bucket_of(score, edges):
    return int(np.searchsorted(edges, score))


def bucket_embedding(bucket_ids):
    """(B,) int -> (B, EMB_DIM) sinusoidal embedding, like a timestep."""
    return timestep_embedding(bucket_ids.float() * 30.0, EMB_DIM)


def calibrate_edges(n_clips=512, seed=7):
    """Bucket edges = quantiles of the training-score distribution."""
    ds = MovingMNIST(train=True, seed=seed, **DATA_KW)
    scores = [flow_score(c) for c in ds.batch(n_clips)]
    qs = np.linspace(0, 1, N_BUCKETS + 1)[1:-1]
    edges = np.quantile(scores, qs)
    return edges, np.array(scores)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_train():
    set_seed(4)
    t0 = time.time()
    edges, scores = calibrate_edges()
    np.save(CKPT / "bucket_edges.npy", edges)
    print(f"bucket edges: {np.round(edges, 2)}  ({time.time()-t0:.0f}s)")

    diff = Diffusion()
    image_unet = ImageUNet()
    image_unet.load_state_dict(torch.load(CKPT12 / "image_unet.pt"))
    model = VideoUNet(image_unet, freeze_spatial=True,
                      extra_emb_dim=EMB_DIM)

    ds = MovingMNIST(train=True, seed=11, **DATA_KW)

    def source():
        clips = ds.batch(BATCH)
        ids = torch.tensor([bucket_of(flow_score(c), edges) for c in clips])
        return {"clips": clips, "extra_emb": bucket_embedding(ids)}

    t0 = time.time()
    losses = train_video_model(model, diff, source, STEPS)
    print(f"train: {time.time()-t0:.0f}s")
    torch.save(model.state_dict(), CKPT / "motion_model.pt")
    np.save(CKPT / "loss.npy", np.array(losses))
    np.save(CKPT / "train_scores.npy", scores)


def stage_figures():
    set_seed(5)
    diff = Diffusion()
    edges = np.load(CKPT / "bucket_edges.npy")
    train_scores = np.load(CKPT / "train_scores.npy")
    image_unet = ImageUNet()
    model = VideoUNet(image_unet, freeze_spatial=True,
                      extra_emb_dim=EMB_DIM)
    model.load_state_dict(torch.load(CKPT / "motion_model.pt"))
    model.eval()

    # Held-out first frames (single digits).
    ds = MovingMNIST(train=False, seed=21, **DATA_KW)
    cond = ds.batch(4)[:, 0]                            # 4 conditions

    # What score does each bucket *stand for* in the training data?
    all_ids = np.array([bucket_of(s, edges) for s in train_scores])
    target = np.array([train_scores[all_ids == b].mean()
                       for b in range(N_BUCKETS)])

    buckets = [0, 2, 4, 6, 7]
    realized = {}
    strips = {}
    for b in buckets:
        ids = torch.full((4,), b)
        t0 = time.time()
        gen = sample_clip(model, diff, cond, T_FRAMES,
                          extra_emb=bucket_embedding(ids), seed=6)
        realized[b] = [flow_score(c) for c in gen]
        strips[b] = gen
        print(f"bucket {b}: realized flow "
              f"{np.mean(realized[b]):.2f} +/- {np.std(realized[b]):.2f} "
              f"(target {target[b]:.2f})  {time.time()-t0:.0f}s", flush=True)

    # --- curve --------------------------------------------------------
    fig, ax = ps.new_axes(6.4, 4.4)
    ax.plot(range(N_BUCKETS), target, color=ps.BASELINE, lw=1.6, ls="--",
            label="training data (what each bucket means)")
    means = [np.mean(realized[b]) for b in buckets]
    err = [np.std(realized[b]) for b in buckets]
    ax.errorbar(buckets, means, yerr=err, color=ps.SERIES[0], lw=1.8,
                marker="o", ms=5, capsize=3,
                label="generated clips (what the model does)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(range(N_BUCKETS))
    ps.finish(fig, ax, "Requested motion bucket vs realized optical flow",
              "requested motion bucket",
              "mean optical-flow magnitude (px/frame)",
              OUT / "motion_curve.png")

    # --- strips: same condition, low vs high bucket -------------------
    rows = torch.cat([strips[0][:2], strips[7][:2]])
    img = strip(rows)
    Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4),
                                Image.NEAREST).save(OUT / "low_vs_high.png")

    # side-by-side GIF: bucket 0 (left) vs bucket 7 (right), same cond
    lo, hi = strips[0][0], strips[7][0]                # (T, 1, 32, 32)
    frames = []
    for t in range(T_FRAMES):
        pair = np.concatenate([lo[t, 0].numpy(), np.full((32, 2), 0.25),
                               hi[t, 0].numpy()], axis=1)
        frames.append(Image.fromarray(
            (pair * 255).astype(np.uint8)).resize((264, 128), Image.NEAREST))
    frames[0].save(OUT / "low_vs_high.gif", save_all=True,
                   append_images=frames[1:], duration=180, loop=0)

    with open(OUT / "metrics.csv", "w") as f:
        f.write("bucket,target_flow,realized_mean,realized_std\n")
        for b in buckets:
            f.write(f"{b},{target[b]:.3f},{np.mean(realized[b]):.3f},"
                    f"{np.std(realized[b]):.3f}\n")

    # --- training-score distribution with bucket edges ----------------
    fig, ax = ps.new_axes(6.4, 3.6)
    ax.hist(train_scores, bins=40, color=ps.SERIES[1], alpha=0.85)
    for e in edges:
        ax.axvline(e, color=ps.INK_MUTED, lw=0.8, ls=":")
    ps.finish(fig, ax,
              "Training-clip motion scores, split into 8 equal buckets",
              "mean optical-flow magnitude (px/frame)", "clips",
              OUT / "score_distribution.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "figures"])
    args = ap.parse_args()
    {"train": stage_train, "figures": stage_figures}[args.stage]()
