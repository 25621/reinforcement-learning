"""Run pretrained FILM frame interpolation on real video and find its limits.

Three experiments:
  1. Gap sweep — interpolate the middle frame between two real frames that
     are 2, 4, 8, 16, 24 frames apart; compare against the real middle
     frame and against a naive cross-fade.
  2. Slow motion — recursively interpolate to turn 2 frames into 9.
  3. Artifact zoom — enlarge the failure region at the largest gap.

Uses the PyTorch port of Google's FILM ("Frame Interpolation for Large
Motion", ECCV 2022): https://github.com/dajes/frame-interpolation-pytorch
Download the TorchScript checkpoint once into data/:
  curl -L -o data/film_net_fp32.pt \\
    https://github.com/dajes/frame-interpolation-pytorch/releases/download/v1.0.2/film_net_fp32.pt
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "01-video-loader-benchmark"))
import plot_style as ps  # noqa: E402
import vid_lib  # noqa: E402

OUT = HERE / "outputs"
GAPS = [2, 4, 8, 16, 24]
N_STARTS = 5          # frame pairs averaged per gap
SIZE = (320, 256)     # FILM's pyramid needs sides divisible by 64


def to_tensor(img):
    """(H, W, 3) uint8 RGB -> (1, 3, H, W) float in [0, 1]."""
    return torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255.0


def to_img(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


class Film:
    def __init__(self):
        self.model = torch.jit.load(HERE / "data" / "film_net_fp32.pt",
                                    map_location="cpu")
        self.model.eval()
        self.calls, self.seconds = 0, 0.0

    def mid(self, img_a, img_b, t=0.5):
        dt = torch.full((1, 1), float(t))
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model(to_tensor(img_a), to_tensor(img_b), dt)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return to_img(out)


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0),
                3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def gap_sweep(film, frames):
    """PSNR of FILM vs cross-fade as the two input frames move apart."""
    rows = []          # for the contact sheet: one row per gap
    stats = {g: {"film": [], "fade": []} for g in GAPS}
    lo, hi = 1, len(frames) - 1
    for gap in GAPS:
        starts = np.linspace(lo, hi - gap, N_STARTS).astype(int)
        for i, t0 in enumerate(starts):
            a, b = frames[t0], frames[t0 + gap]
            true_mid = frames[t0 + gap // 2]
            pred = film.mid(a, b)
            fade = ((a.astype(np.float32) + b) / 2).astype(np.uint8)
            stats[gap]["film"].append(psnr(pred, true_mid))
            stats[gap]["fade"].append(psnr(fade, true_mid))
            if i == N_STARTS // 2:            # middle example -> figure row
                rows += [label(a, f"input A (frame {t0})"),
                         label(true_mid, "real middle frame"),
                         label(pred, f"FILM  gap={gap}"),
                         label(fade, "cross-fade")]
        print(f"gap {gap:2d}: FILM {np.mean(stats[gap]['film']):5.2f} dB   "
              f"cross-fade {np.mean(stats[gap]['fade']):5.2f} dB", flush=True)
    vid_lib.contact_sheet(rows, cols=4, out_path=OUT / "gap_sweep.png",
                          scale=0.55)

    fig, ax = ps.new_axes(6.6, 4.2)
    xs = np.arange(len(GAPS))
    for i, (name, key) in enumerate([("FILM", "film"),
                                     ("cross-fade", "fade")]):
        mean = [np.mean(stats[g][key]) for g in GAPS]
        ax.plot(xs, mean, color=ps.SERIES[0 if key == "film" else 2], lw=2,
                marker="o", ms=4, label=name)
    ax.set_xticks(xs, [str(g) for g in GAPS])
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax,
              "Interpolation quality vs distance between input frames",
              "gap between the two real frames (frames @ 24 fps)",
              "PSNR of predicted middle frame (dB)",
              OUT / "psnr_vs_gap.png")
    return stats


def slow_motion(film, frames, t0=40, gap=8):
    """Recursive midpoint interpolation: 2 real frames -> 9-frame slow-mo."""
    a, b = frames[t0], frames[t0 + gap]
    clip = {0.0: a, 1.0: b}
    for level in range(3):                      # 1 -> 3 -> 7 in-betweens
        keys = sorted(clip)
        for lo, hi in zip(keys[:-1], keys[1:]):
            clip[(lo + hi) / 2] = film.mid(clip[lo], clip[hi])
    ordered = [clip[k] for k in sorted(clip)]
    labels = ["real A"] + [""] * 7 + ["real B"]
    vid_lib.contact_sheet([label(f, l) for f, l in zip(ordered, labels)],
                          cols=3, out_path=OUT / "slow_motion.png", scale=0.55)


def artifact_zoom(film, frames):
    """Where FILM breaks: crop the fastest-moving region at gap 24."""
    gap = 24
    t0 = 30
    a, b = frames[t0], frames[t0 + gap]
    pred = film.mid(a, b)
    true_mid = frames[t0 + gap // 2]
    # find the 128x128 region where the two inputs differ the most
    diff = cv2.GaussianBlur(
        np.abs(a.astype(np.float32) - b).mean(axis=2), (63, 63), 0)
    y, x = np.unravel_index(np.argmax(diff), diff.shape)
    h, w = diff.shape
    y = int(np.clip(y - 64, 0, h - 128))
    x = int(np.clip(x - 64, 0, w - 128))
    crops = [im[y:y + 128, x:x + 128] for im in (a, true_mid, pred, b)]
    crops = [cv2.resize(c, (256, 256), interpolation=cv2.INTER_NEAREST)
             for c in crops]
    names = ["input A", "real middle", "FILM middle", "input B"]
    vid_lib.contact_sheet([label(c, n) for c, n in zip(crops, names)],
                          cols=4, out_path=OUT / "artifact_zoom.png")


if __name__ == "__main__":
    torch.set_num_threads(12)
    OUT.mkdir(exist_ok=True)
    vid_lib.ensure_sources()
    mega = vid_lib.read_frames(vid_lib.DATA / "Megamind.avi", resize=SIZE)
    shot = mega[1:97]                 # one continuous fast-motion shot
    film = Film()
    gap_sweep(film, shot)
    vtest = vid_lib.read_frames(vid_lib.DATA / "vtest.avi", resize=SIZE)
    slow_motion(film, vtest[100:], t0=0, gap=8)
    artifact_zoom(film, shot)
    print(f"FILM calls: {film.calls}, avg {film.seconds/film.calls:.2f}s "
          f"per interpolated frame (CPU)", flush=True)
