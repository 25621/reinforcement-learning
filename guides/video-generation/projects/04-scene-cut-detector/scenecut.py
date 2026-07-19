"""Detect scene cuts (shot boundaries) and split a video into clean clips.

We splice a test "movie" out of 11 single-shot segments taken from three
real videos, so the true cut positions are known exactly. Two of the ten
cuts are deliberately evil: a splice between two different moments of the
SAME static-camera scene, and a jump cut that skips 10 frames within one
shot. Then two detectors compete:

  histogram — chi-square distance between adjacent frames' color histograms
  deep      — cosine distance between adjacent frames' ResNet-18 embeddings

Both are scored with precision/recall against the known cuts, and the
winner is let loose on the raw trailer clip to split it into shots.

Run:  python3 scenecut.py        (~2 minutes; downloads ResNet-18, ~45 MB)
"""

import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import vid_lib
import plot_style as ps

warnings.filterwarnings("ignore")
import torch
import torchvision

OUT = HERE / "outputs"
DATA = HERE / "data"
SIZE = (320, 240)
TOLERANCE = 2  # a detection within +-2 frames of a true cut counts


# ------------------------------------------------------------- test movie --
def build_movie():
    """Concatenate known single-shot segments; return (path, true_cuts)."""
    vid_lib.ensure_sources()
    vtest = vid_lib.read_frames(vid_lib.DATA / "vtest.avi", resize=SIZE)
    mega = vid_lib.read_frames(vid_lib.DATA / "Megamind.avi", resize=SIZE)
    bunny = vid_lib.read_frames(vid_lib.DATA / "bunny.mp4", resize=SIZE)
    # The trailer's own shots are frames 1-97, 98-153, 154-199 and 200-269
    # (measured with the same histogram probe), so these slices are all
    # single-shot. Three boundaries are deliberately evil (marked HARD).
    segments = [
        (vtest[100:160], "normal cut"),
        (mega[5:65], "normal cut"),
        (vtest[600:650], "normal cut"),
        (bunny[20:80], "normal cut"),
        (mega[100:145], "normal cut"),
        (vtest[300:340], "normal cut"),
        # HARD: same static camera as previous segment, later time:
        (vtest[500:540], "same-camera splice"),
        (mega[160:198], "normal cut"),
        (mega[205:235], "normal cut"),
        # HARD: same shot as previous segment, 10 frames skipped:
        (mega[245:269], "jump cut"),
        (bunny[85:100], "normal cut"),
    ]
    # HARD #3: a 24-frame crossfade (gradual dissolve) into the last shot,
    # instead of an instant boundary.
    alpha = np.linspace(0, 1, 24)[:, None, None, None]
    dissolve = ((1 - alpha) * bunny[100:124] + alpha * vtest[680:704])
    parts = [s for s, _ in segments] + [dissolve.astype(np.uint8),
                                        vtest[704:730]]
    movie = np.concatenate(parts)
    ends = np.cumsum([len(p) for p in parts])
    # A new shot starts after each of the first 10 segments; the 11th
    # boundary is the dissolve, whose "cut" is the middle of the fade and
    # gets a tolerance covering the whole 24-frame window.
    true_cuts = np.append(ends[:10], ends[10] + 12)
    kinds = [k for _, k in segments[1:]] + ["gradual dissolve"]
    tols = np.array([TOLERANCE] * 10 + [14])
    path = DATA / "movie.mp4"
    vid_lib.write_video(movie, path, fps=24)
    return path, true_cuts, kinds, tols


# -------------------------------------------------------------- detectors --
def hist_distances(frames):
    """Chi-square distance between adjacent frames' color histograms."""
    def hist(f):
        h = [cv2.calcHist([f], [c], None, [32], [0, 256]).ravel()
             for c in range(3)]
        return np.concatenate(h) / f[..., 0].size
    hs = np.stack([hist(f) for f in frames])
    a, b = hs[1:], hs[:-1]
    return 0.5 * ((a - b) ** 2 / (a + b + 1e-8)).sum(axis=1)


_resnet = None


def deep_distances(frames, batch=64):
    """Cosine distance between adjacent frames' ResNet-18 embeddings."""
    global _resnet
    if _resnet is None:
        net = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        net.fc = torch.nn.Identity()          # keep the 512-d embedding
        _resnet = net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    feats = []
    with torch.no_grad():
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(frames[i:i + batch]).permute(0, 3, 1, 2)
            x = (x.float() / 255.0 - mean) / std
            x = torch.nn.functional.interpolate(x, size=(224, 224),
                                                mode="bilinear")
            f = _resnet(x)
            feats.append(torch.nn.functional.normalize(f, dim=1))
    feats = torch.cat(feats)
    return (1 - (feats[1:] * feats[:-1]).sum(dim=1)).numpy()


# ------------------------------------------------------------- evaluation --
def detections(dist, threshold):
    """Frame indices where a new shot starts, per the detector."""
    return np.where(dist > threshold)[0] + 1


def score(detected, true_cuts, tols):
    # A detection inside a cut's window matches it; extra detections inside
    # an already-matched window are absorbed rather than counted as false
    # positives (the standard way to score gradual transitions).
    matched = np.zeros(len(true_cuts), bool)
    false_pos = 0
    for d in detected:
        gaps = np.abs(true_cuts - d)
        j = gaps.argmin()
        if gaps[j] <= tols[j]:
            matched[j] = True
        else:
            false_pos += 1
    tp = int(matched.sum())
    precision = tp / max(tp + false_pos, 1)
    recall = tp / len(true_cuts)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1, matched


def sweep(dist, true_cuts, tols):
    """Try many thresholds; return (thresholds, P, R, F1 arrays)."""
    ts = np.geomspace(max(dist.min(), 1e-5), dist.max(), 200)
    rows = [score(detections(dist, t), true_cuts, tols)[:3] for t in ts]
    return ts, np.array(rows)


def main():
    OUT.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    movie_path, true_cuts, cut_kinds, tols = build_movie()
    frames = vid_lib.read_frames(movie_path)
    print(f"movie: {len(frames)} frames, {len(true_cuts)} true cuts "
          f"at {true_cuts.tolist()}")

    curves = {}
    for name, fn in [("histogram", hist_distances), ("deep", deep_distances)]:
        dist = fn(frames)
        ts, prf = sweep(dist, true_cuts, tols)
        best = prf[:, 2].argmax()
        t_best = ts[best]
        p, r, f1, matched = score(detections(dist, t_best), true_cuts, tols)
        curves[name] = dict(dist=dist, ts=ts, prf=prf, t_best=t_best)
        print(f"\n{name}: best threshold {t_best:.4f} -> "
              f"precision {p:.2f}  recall {r:.2f}  F1 {f1:.2f}")
        for cut, kind, ok in zip(true_cuts, cut_kinds, matched):
            print(f"   cut at frame {cut:3d} ({kind:18s}): "
                  f"{'found' if ok else 'MISSED'}")
        lo, hi = true_cuts[-1] - tols[-1], true_cuts[-1] + tols[-1]
        window = dist[lo - 1:hi]
        print(f"   dissolve window: max distance {window.max():.4f} "
              f"(threshold {t_best:.4f}), frames over threshold "
              f"{int((window > t_best).sum())}/{len(window)}")

    plot_timelines(curves, true_cuts, len(frames))
    plot_f1(curves)
    split_trailer(curves)


def plot_timelines(curves, true_cuts, n_frames):
    fig, axes = ps.plt.subplots(2, 1, figsize=(8.8, 5.0), dpi=110,
                                sharex=True)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (name, c), color in zip(axes, curves.items(),
                                    [ps.SERIES[0], ps.SERIES[1]]):
        ps.style_axes(ax)
        for cut in true_cuts:
            ax.axvline(cut, color=ps.SERIES[2], linewidth=1.0, alpha=0.55)
        ax.plot(np.arange(1, n_frames), c["dist"], color=color, linewidth=1.4)
        ax.axhline(c["t_best"], color=ps.INK_MUTED, linewidth=1.0,
                   linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"{name} distance between adjacent frames "
                     "(red lines = true cuts, dashed = best threshold)",
                     color=ps.INK, fontsize=10, loc="left")
        ax.set_ylabel("distance", color=ps.INK_SECONDARY, fontsize=9)
    axes[1].set_xlabel("frame index", color=ps.INK_SECONDARY, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "timelines.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print(f"wrote {OUT / 'timelines.png'}")
    ps.plt.close(fig)


def plot_f1(curves):
    fig, ax = ps.new_axes(7.0, 3.6)
    for (name, c), color in zip(curves.items(), [ps.SERIES[0], ps.SERIES[1]]):
        # normalize thresholds to their own scale so both fit on one x-axis
        x = np.linspace(0, 1, len(c["ts"]))
        ax.plot(x, c["prf"][:, 2], color=color, linewidth=2.0, label=name)
        best = c["prf"][:, 2].argmax()
        ax.scatter([x[best]], [c["prf"][best, 2]], color=color, s=36,
                   zorder=3)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY, fontsize=9)
    ps.finish(fig, ax, "F1 across the threshold sweep (dot = best)",
              "threshold (percentile of each method's range)", "F1 score",
              OUT / "f1_sweep.png")


def split_trailer(curves):
    """Run the histogram detector on the raw trailer; save one frame per
    detected shot as a contact sheet."""
    path = vid_lib.DATA / "Megamind.avi"
    frames = vid_lib.read_frames(path, resize=SIZE)
    dist = hist_distances(frames)
    cuts = detections(dist, curves["histogram"]["t_best"]).tolist()
    starts = [0] + cuts
    ends = cuts + [len(frames)]
    print(f"\ntrailer: detected shots at {starts}")
    mids = [frames[(a + b) // 2] for a, b in zip(starts, ends)]
    labels = [f"shot {i}: frames {a}-{b - 1}"
              for i, (a, b) in enumerate(zip(starts, ends))]
    vid_lib.contact_sheet(mids, cols=len(mids),
                          out_path=OUT / "trailer_shots.png", scale=0.75,
                          labels=labels)


if __name__ == "__main__":
    main()
