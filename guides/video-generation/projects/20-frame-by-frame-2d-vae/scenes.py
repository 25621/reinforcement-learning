"""Four synthetic 256x256 RGB clips, built to isolate *one* cause of flicker each.

Real footage mixes every cause at once (motion, sensor noise, codec noise,
camera shake), so you can never tell which one the VAE is reacting to.  These
clips change exactly one thing at a time:

  static  frames are bit-identical            -> no cause at all (control)
  noise   + faint sensor noise                -> pixel values wobble, nothing moves
  drift   + 0.25 px/frame camera pan          -> everything moves, sub-pixel
  motion  + a disc crossing the frame at 3 px -> ordinary visible motion

The background is fractal noise plus hard-edged detail (grid lines, dots).
High-frequency texture is where an image VAE spends its error budget, so it is
where flicker shows up first — a clip of flat gradients would hide the effect.

Imported by run.py; also by project 24 for its pixel-vs-latent comparison figure.
"""

import cv2
import numpy as np

SIZE = 256
T = 8


def _fractal_noise(rng, size, octaves=5):
    """Sum of blurred random fields at halving scales -> cloud-like texture."""
    out = np.zeros((size, size), np.float32)
    for o in range(octaves):
        side = max(2, size >> (octaves - o))
        small = rng.random((side, side)).astype(np.float32)
        up = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        out += up * (0.5 ** o)
    out -= out.min()
    return out / (out.max() + 1e-8)


def _background(seed=0, pad=16):
    """One still RGB image, rendered slightly larger than SIZE so the drift
    clip has real pixels to pan into instead of edge padding."""
    rng = np.random.default_rng(seed)
    s = SIZE + 2 * pad
    img = np.stack([_fractal_noise(rng, s) for _ in range(3)], -1)
    img = 0.35 + 0.5 * img

    # Hard edges: a thin grid and scattered dots. Fractal noise alone is
    # smooth, and smooth things compress well; the VAE only struggles where
    # detail is sharper than its 8x downsampling can represent.
    for k in range(0, s, 32):
        img[k:k + 1, :, :] *= 0.45
        img[:, k:k + 1, :] *= 0.45
    for _ in range(120):
        y, x = rng.integers(4, s - 4, size=2)
        cv2.circle(img, (int(x), int(y)), int(rng.integers(1, 3)),
                   tuple(float(v) for v in rng.random(3)), -1)
    return np.clip(img, 0, 1).astype(np.float32), pad


def _crop(img, pad, dy=0.0, dx=0.0):
    """Crop the SIZE x SIZE window at a possibly fractional offset."""
    M = np.float32([[1, 0, -dx], [0, 1, -dy]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    return shifted[pad:pad + SIZE, pad:pad + SIZE]


def clips(seed=0):
    """dict name -> (T, SIZE, SIZE, 3) float32 clip in [0, 1]."""
    bg, pad = _background(seed)
    rng = np.random.default_rng(seed + 1)
    base = _crop(bg, pad)

    out = {}
    out["static"] = np.stack([base] * T)
    out["noise"] = np.clip(
        out["static"] + rng.normal(0, 0.004, out["static"].shape), 0, 1
    ).astype(np.float32)
    out["drift"] = np.stack([_crop(bg, pad, dy=0.25 * t, dx=0.25 * t)
                             for t in range(T)])

    frames = []
    for t in range(T):
        f = base.copy()
        cv2.circle(f, (60 + 3 * t, 128), 22, (0.95, 0.55, 0.15), -1)
        frames.append(f)
    out["motion"] = np.stack(frames).astype(np.float32)

    return {k: v.astype(np.float32) for k, v in out.items()}


DESCRIPTIONS = {
    "static": "identical frames (control)",
    "noise": "faint sensor noise, sigma=0.004",
    "drift": "sub-pixel camera pan, 0.25 px/frame",
    "motion": "a disc moving 3 px/frame",
}
