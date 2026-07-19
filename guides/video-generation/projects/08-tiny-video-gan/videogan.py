"""A tiny 3D-conv video GAN, plus the clip dataset it trains on.

The dataset: 8-frame 32x32 RGB clips of walking pedestrians, cropped from
the motion-rich regions of vtest.avi (the surveillance clip from project
01). UCF-101 is ~7 GB, so this stands in for the "UCF-101 face crops" of
the original papers — what matters is that the clips are *real video with
real motion*, not synthetic sprites.

Generator:      z (64) -> 3D transposed convs -> clip (3, 8, 32, 32)
Discriminator:  clip -> 3D convs -> real/fake score

Both are direct 3D generalizations of DCGAN: every Conv2d becomes a
Conv3d whose kernel also spans time, so the discriminator judges motion,
not just per-frame appearance.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "01-video-loader-benchmark"))
import vid_lib  # noqa: E402

T, S = 8, 32           # clip shape: T frames of S x S pixels
STRIDE_T = 2           # sample every 2nd source frame (10 fps -> 5 fps)


def build_dataset(cache=HERE / "data" / "clips.npz"):
    """Cut moving-crop clips out of vtest.avi. Returns (N, 3, T, S, S) float."""
    if cache.exists():
        return torch.from_numpy(np.load(cache)["clips"])
    vid_lib.ensure_sources()
    frames = vid_lib.read_frames(vid_lib.DATA / "vtest.avi")  # (795,576,768,3)
    frames = frames.astype(np.float32) / 255.0
    crop = 64                                   # source crop size -> 32x32
    span = T * STRIDE_T
    clips = []
    rng = np.random.default_rng(0)
    for t0 in range(0, len(frames) - span, 2):
        window = frames[t0:t0 + span:STRIDE_T]          # (T, H, W, 3)
        motion = np.abs(window[1:] - window[:-1]).mean(axis=(0, 3))
        for _ in range(4):                     # 4 motion-rich crops / window
            ys = rng.integers(0, frames.shape[1] - crop, 12)
            xs = rng.integers(0, frames.shape[2] - crop, 12)
            scores = [motion[y:y + crop, x:x + crop].mean()
                      for y, x in zip(ys, xs)]
            k = int(np.argmax(scores))
            if scores[k] < 0.01:               # skip static background crops
                continue
            y, x = ys[k], xs[k]
            clip = window[:, y:y + crop, x:x + crop]
            clip = clip[:, ::2, ::2]           # 64 -> 32 (cheap 2x downscale)
            clips.append(clip)
    clips = np.stack(clips).transpose(0, 4, 1, 2, 3)    # (N, 3, T, S, S)
    clips = clips * 2 - 1                               # [-1, 1] for tanh
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, clips=clips.astype(np.float32))
    print(f"built {len(clips)} clips")
    return torch.from_numpy(clips.astype(np.float32))


class Generator(nn.Module):
    def __init__(self, nz=64, ch=48):
        super().__init__()
        self.nz = nz
        self.fc = nn.Linear(nz, ch * 4 * 2 * 4 * 4)
        self.net = nn.Sequential(
            nn.BatchNorm3d(ch * 4), nn.ReLU(),
            # (ch*4, 2, 4, 4) -> (ch*2, 4, 8, 8)
            nn.ConvTranspose3d(ch * 4, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(ch * 2), nn.ReLU(),
            # -> (ch, 8, 16, 16)
            nn.ConvTranspose3d(ch * 2, ch, 4, stride=2, padding=1),
            nn.BatchNorm3d(ch), nn.ReLU(),
            # -> (3, 8, 32, 32): upsample space only, time stays 8
            nn.ConvTranspose3d(ch, 3, (3, 4, 4), stride=(1, 2, 2),
                               padding=(1, 1, 1)),
            nn.Tanh(),
        )
        self.ch = ch

    def forward(self, z):
        x = self.fc(z).view(-1, self.ch * 4, 2, 4, 4)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.net = nn.Sequential(
            # (3, 8, 32, 32) -> (ch, 8, 16, 16)
            nn.Conv3d(3, ch, (3, 4, 4), stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(0.2),
            # -> (ch*2, 4, 8, 8): now stride in time too — judges motion
            nn.Conv3d(ch, ch * 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(ch * 2), nn.LeakyReLU(0.2),
            # -> (ch*4, 2, 4, 4)
            nn.Conv3d(ch * 2, ch * 4, 4, stride=2, padding=1),
            nn.BatchNorm3d(ch * 4), nn.LeakyReLU(0.2),
        )
        self.fc = nn.Linear(ch * 4 * 2 * 4 * 4, 1)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1)).squeeze(1)
