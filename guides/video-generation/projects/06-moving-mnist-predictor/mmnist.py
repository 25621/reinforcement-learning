"""Synthetic Moving MNIST — the classic bouncing-digits video dataset.

Generates clips on the fly instead of downloading the fixed 10k-clip file:
MNIST digits are pasted onto a small canvas and moved along straight lines,
bouncing off the walls. Because every batch is freshly generated, the model
never sees the same clip twice (an infinite dataset for free).

The original benchmark uses two 28x28 digits on a 64x64 canvas; we shrink
everything by 2x (14x14 digits, 32x32 canvas) so the ConvLSTM trains in
minutes on a CPU. The dynamics are identical.

Imported by project 09 via sys.path (one-digit clips for the content/motion
VAE).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.datasets import MNIST

DATA = Path(__file__).resolve().parent / "data"

CANVAS = 32          # canvas is CANVAS x CANVAS pixels
DIGIT = 14           # digits are resized to DIGIT x DIGIT


def _load_digits(train=True):
    """All MNIST images resized to DIGIT x DIGIT, as float32 in [0, 1]."""
    ds = MNIST(root=str(DATA), train=train, download=True)
    imgs = ds.data.numpy()                      # (N, 28, 28) uint8
    small = np.stack([
        cv2.resize(im, (DIGIT, DIGIT), interpolation=cv2.INTER_AREA)
        for im in imgs
    ])
    return small.astype(np.float32) / 255.0, ds.targets.numpy()


class MovingMNIST:
    """Generate (B, T, 1, CANVAS, CANVAS) float clips of bouncing digits."""

    def __init__(self, n_digits=2, seq_len=20, train=True, seed=0,
                 min_speed=0.7, max_speed=1.6):
        self.digits, self.labels = _load_digits(train)
        self.n_digits = n_digits
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)
        self.min_speed = min_speed
        self.max_speed = max_speed

    def batch(self, batch_size, return_labels=False):
        T, D, C = self.seq_len, DIGIT, CANVAS
        lim = C - D                              # top-left position range
        clips = np.zeros((batch_size, T, C, C), dtype=np.float32)
        labels = np.zeros((batch_size, self.n_digits), dtype=np.int64)
        for b in range(batch_size):
            for d in range(self.n_digits):
                idx = self.rng.integers(len(self.digits))
                sprite = self.digits[idx]
                labels[b, d] = self.labels[idx]
                # random start position and velocity (random direction)
                pos = self.rng.uniform(0, lim, size=2)
                speed = self.rng.uniform(self.min_speed, self.max_speed)
                angle = self.rng.uniform(0, 2 * np.pi)
                vel = speed * np.array([np.cos(angle), np.sin(angle)])
                for t in range(T):
                    y, x = int(round(pos[0])), int(round(pos[1]))
                    patch = clips[b, t, y:y + D, x:x + D]
                    np.maximum(patch, sprite, out=patch)   # overlap = max
                    pos += vel
                    for ax in (0, 1):            # bounce off the walls
                        if pos[ax] < 0:
                            pos[ax] = -pos[ax]
                            vel[ax] = -vel[ax]
                        elif pos[ax] > lim:
                            pos[ax] = 2 * lim - pos[ax]
                            vel[ax] = -vel[ax]
        out = torch.from_numpy(clips).unsqueeze(2)          # (B, T, 1, C, C)
        if return_labels:
            return out, torch.from_numpy(labels)
        return out


class MovingMNISTWithState(MovingMNIST):
    """Same, but also returns the digit's (y, x) position at every frame.

    Used by project 09 to check what the motion latent actually encodes.
    Only supports n_digits=1 (one trajectory per clip).
    """

    def __init__(self, **kw):
        kw["n_digits"] = 1
        super().__init__(**kw)

    def batch_with_state(self, batch_size):
        T, D, C = self.seq_len, DIGIT, CANVAS
        lim = C - D
        clips = np.zeros((batch_size, T, C, C), dtype=np.float32)
        labels = np.zeros(batch_size, dtype=np.int64)
        positions = np.zeros((batch_size, T, 2), dtype=np.float32)
        for b in range(batch_size):
            idx = self.rng.integers(len(self.digits))
            sprite = self.digits[idx]
            labels[b] = self.labels[idx]
            pos = self.rng.uniform(0, lim, size=2)
            speed = self.rng.uniform(self.min_speed, self.max_speed)
            angle = self.rng.uniform(0, 2 * np.pi)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            for t in range(T):
                y, x = int(round(pos[0])), int(round(pos[1]))
                clips[b, t, y:y + D, x:x + D] = sprite
                positions[b, t] = pos
                pos += vel
                for ax in (0, 1):
                    if pos[ax] < 0:
                        pos[ax] = -pos[ax]
                        vel[ax] = -vel[ax]
                    elif pos[ax] > lim:
                        pos[ax] = 2 * lim - pos[ax]
                        vel[ax] = -vel[ax]
        return (torch.from_numpy(clips).unsqueeze(2),
                torch.from_numpy(labels),
                torch.from_numpy(positions))
