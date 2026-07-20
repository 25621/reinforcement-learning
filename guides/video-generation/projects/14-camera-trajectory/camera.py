"""A synthetic camera looking at a static scene.

The scene is a 96x96 canvas of smooth random "terrain" plus a few
static MNIST digits.  The "camera" is a square crop window: its center can
move (pan) and its side length can grow or shrink (zoom).  Every frame
is the crop resized to 32x32 — so ALL apparent motion in a clip comes
from the camera, none from the scene.

Also computes the Plucker-coordinate ray map for each frame: for every
pixel, 6 numbers (ray direction d, moment o x d) describing the 3D line
that pixel looks along, for a pinhole camera hovering above the scene
plane.  Poses are expressed relative to the first frame's camera,
because the model can only see the world through the first frame — it
has no way to know absolute scene coordinates.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "06-moving-mnist-predictor"))

from mmnist import _load_digits                         # noqa: E402

CANVAS = 96
VIEW = 32          # rendered frame size; also the frame-0 window side
T = 8

PAN_ANGLES = np.array([0.0, 90.0, 180.0, 270.0])        # cardinal directions
PAN_SPEEDS = [1.0, 2.0, 3.0]                            # scene px / frame
ZOOM_RATES = [0.94, 1.06]                               # window growth / frame


class SceneSampler:
    def __init__(self, train=True, seed=0, n_digits=3, n_dots=0):
        self.digits, _ = _load_digits(train)
        self.rng = np.random.default_rng(seed)
        self.n_digits = n_digits
        self.n_dots = n_dots

    def scene(self):
        """A static 96x96 canvas: smooth random terrain + a few digits.

        The background terrain matters twice over.  Over a plain black
        region a camera pan is invisible (every crop looks identical),
        so no training signal could ever connect the camera input to
        what the model draws — real video has texture almost
        everywhere, and the terrain is its stand-in.  And it is *smooth*
        (low-frequency) on purpose: a tiny diffusion model can actually
        learn to reproduce a soft terrain patch from the conditioning
        frame, where it visibly fails to memorize constellations of
        sharp random dots.
        """
        low = self.rng.uniform(0, 1, (6, 6)).astype(np.float32)
        mid = self.rng.uniform(0, 1, (16, 16)).astype(np.float32)
        canvas = (0.42 * cv2.resize(low, (CANVAS, CANVAS),
                                    interpolation=cv2.INTER_CUBIC)
                  + 0.18 * cv2.resize(mid, (CANVAS, CANVAS),
                                      interpolation=cv2.INTER_CUBIC))
        canvas = canvas.clip(0, 0.6)
        D = self.digits.shape[1]
        for _ in range(self.n_digits):
            sprite = self.digits[self.rng.integers(len(self.digits))]
            y = self.rng.integers(0, CANVAS - D)
            x = self.rng.integers(0, CANVAS - D)
            patch = canvas[y:y + D, x:x + D]
            np.maximum(patch, sprite, out=patch)
        return canvas

    # -- trajectories -----------------------------------------------------

    def feasible_box(self, offsets, sides):
        """The range of start centers keeping every window on-canvas."""
        half = sides.max() / 2 + 1
        return (half - offsets[:, 0].min(), CANVAS - half - offsets[:, 0].max(),
                half - offsets[:, 1].min(), CANVAS - half - offsets[:, 1].max())

    def _fit_start(self, offsets, sides):
        """Choose a start center so every window stays on the canvas."""
        lo_x, hi_x, lo_y, hi_y = self.feasible_box(offsets, sides)
        cx = self.rng.uniform(lo_x, hi_x)
        cy = self.rng.uniform(lo_y, hi_y)
        return np.stack([offsets[:, 0] + cx, offsets[:, 1] + cy], axis=1)

    def offsets(self, kind=None, angle=None, speed=None, rate=None):
        """Camera path relative to its start: offsets (T,2) xy, sides (T,).

        kind: 'pan' | 'zoom' | 'panzoom' (held out) | 'curve' (held out)
        """
        if kind is None:
            kind = self.rng.choice(["pan", "zoom"], p=[0.7, 0.3])
        t = np.arange(T)
        if kind == "pan":
            a = np.deg2rad(angle if angle is not None
                           else self.rng.choice(PAN_ANGLES))
            s = speed if speed is not None else self.rng.choice(PAN_SPEEDS)
            offsets = np.stack([t * s * np.cos(a), t * s * np.sin(a)], 1)
            sides = np.full(T, float(VIEW))
        elif kind == "zoom":
            g = rate if rate is not None else self.rng.choice(ZOOM_RATES)
            offsets = np.zeros((T, 2))
            sides = VIEW * g ** t
        elif kind == "panzoom":                        # never in training
            a = np.deg2rad(angle if angle is not None
                           else self.rng.choice(PAN_ANGLES))
            s = speed if speed is not None else 2.0
            g = rate if rate is not None else self.rng.choice(ZOOM_RATES)
            offsets = np.stack([t * s * np.cos(a), t * s * np.sin(a)], 1)
            sides = VIEW * g ** t
        elif kind == "curve":                          # never in training
            a0 = np.deg2rad(angle if angle is not None
                            else self.rng.choice(PAN_ANGLES))
            s = speed if speed is not None else 2.0
            angles = a0 + np.deg2rad(15.0) * t
            steps = np.stack([s * np.cos(angles), s * np.sin(angles)], 1)
            offsets = np.concatenate(
                [np.zeros((1, 2)), np.cumsum(steps[:-1], axis=0)])
            sides = np.full(T, float(VIEW))
        return offsets, sides

    def trajectory(self, **kw):
        """Offsets anchored at a random on-canvas start -> centers, sides."""
        offsets, sides = self.offsets(**kw)
        centers = self._fit_start(offsets, sides)
        return centers, sides

    def render(self, canvas, centers, sides):
        """Render the clip: crop each window, resize to VIEW x VIEW."""
        frames = np.empty((T, VIEW, VIEW), dtype=np.float32)
        for t in range(T):
            side = int(round(sides[t]))
            patch = cv2.getRectSubPix(canvas, (side, side),
                                      tuple(centers[t]))
            interp = cv2.INTER_AREA if side > VIEW else cv2.INTER_LINEAR
            frames[t] = cv2.resize(patch, (VIEW, VIEW), interpolation=interp)
        return frames

    def clip(self, **kw):
        canvas = self.scene()
        centers, sides = self.trajectory(**kw)
        frames = self.render(canvas, centers, sides)
        return frames, centers, sides


# ---------------------------------------------------------------------------
# Camera encodings
# ---------------------------------------------------------------------------

def relative_pose(centers, sides):
    """(T, 3): camera pose per frame, relative to frame 0, in view units."""
    rel = np.empty((T, 3), dtype=np.float32)
    rel[:, 0] = (centers[:, 0] - centers[0, 0]) / VIEW
    rel[:, 1] = (centers[:, 1] - centers[0, 1]) / VIEW
    rel[:, 2] = np.log2(sides / sides[0])
    return rel


def plucker_map(centers, sides):
    """(T, 6, VIEW, VIEW): per-pixel Plucker ray coordinates.

    Model: a pinhole camera hovering at height `side` above the scene
    plane (z=0), looking straight down.  Pixel (i, j) of frame t images
    the scene point p = center_t + (pixel offset) * side_t / VIEW, so
    its ray runs from the camera origin o_t = (center_t, -side_t) to p.
    The Plucker encoding of that line is (d, o x d) with d normalized.
    Everything is measured relative to the first frame's camera center
    and in units of the first window side.
    """
    maps = np.empty((T, 6, VIEW, VIEW), dtype=np.float32)
    px = (np.arange(VIEW) - (VIEW - 1) / 2)
    jj, ii = np.meshgrid(px, px)                       # jj: x, ii: y
    for t in range(T):
        scale = sides[t] / VIEW
        p = np.stack([
            (centers[t, 0] - centers[0, 0]) + jj * scale,
            (centers[t, 1] - centers[0, 1]) + ii * scale,
            np.zeros_like(jj)], axis=0) / VIEW          # (3, H, W)
        o = np.array([(centers[t, 0] - centers[0, 0]) / VIEW,
                      (centers[t, 1] - centers[0, 1]) / VIEW,
                      -sides[t] / VIEW], dtype=np.float32)
        d = p - o[:, None, None]
        d = d / np.linalg.norm(d, axis=0, keepdims=True)
        m = np.cross(np.broadcast_to(o[:, None, None], d.shape),
                     d, axis=0)
        maps[t, :3] = d
        maps[t, 3:] = m
    return maps


def warp_stack(view0, centers, sides):
    """(T, 1, VIEW, VIEW): the conditioning frame warped per frame.

    For each frame, move and scale the *first frame's pixels* to where
    the requested camera would put them — classic motion compensation.
    Pixels that fall outside the first frame's view are unknowable from
    the condition; they are filled with -1 so the model can tell
    "no information here" apart from "dark terrain here".
    """
    out = np.empty((T, 1, VIEW, VIEW), dtype=np.float32)
    c = (VIEW - 1) / 2
    for t in range(T):
        s = sides[t] / sides[0]
        dx = centers[t, 0] - centers[0, 0]
        dy = centers[t, 1] - centers[0, 1]
        M = np.float32([[s, 0, c * (1 - s) + dx],
                        [0, s, c * (1 - s) + dy]])
        out[t, 0] = cv2.warpAffine(
            view0, M, (VIEW, VIEW),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)
    return out


def batch(sampler, n, kinds=None):
    """n clips -> dict of torch tensors for the two conditioning styles."""
    clips, poses, maps, warps = [], [], [], []
    for i in range(n):
        kw = kinds[i] if kinds else {}
        frames, centers, sides = sampler.clip(**kw)
        clips.append(frames)
        poses.append(relative_pose(centers, sides))
        maps.append(plucker_map(centers, sides))
        warps.append(warp_stack(frames[0], centers, sides))
    return {
        "clips": torch.from_numpy(np.stack(clips)).unsqueeze(2),
        "pose": torch.from_numpy(np.stack(poses)),
        "maps": torch.from_numpy(np.stack(maps)),
        "warp": torch.from_numpy(np.stack(warps)),
    }
