"""Benchmark four video-loading libraries on the same folder of .mp4 clips.

Two access patterns are timed, because they stress different things:

  sequential — decode all 64 frames of the clip in order
               (what you do when caching or preprocessing a dataset)
  random     — grab 8 evenly spread frames out of the 64
               (what a training dataloader does on every batch)

Each (library, pattern) pair is run over all clips 3 times; we report the
best run so one-off OS hiccups don't pollute the numbers.

Run:  python3 benchmark.py        (~1 minute)
"""

import csv
import subprocess
import time
import warnings
from pathlib import Path

import numpy as np

import vid_lib
import plot_style as ps

warnings.filterwarnings("ignore")  # torchvision.io deprecation notice

import av
import cv2
import decord
import ffmpeg
import imageio_ffmpeg
import torchvision.io

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

N_RANDOM = 8
REPEATS = 3


# ---------------------------------------------------------------- loaders --
# Every loader returns uint8 RGB frames so the comparison is apples-to-apples.

def decord_sequential(path, n_frames):
    vr = decord.VideoReader(str(path), num_threads=2)
    return vr.get_batch(range(len(vr))).asnumpy()


def decord_random(path, indices):
    vr = decord.VideoReader(str(path), num_threads=2)
    return vr.get_batch(indices).asnumpy()


def torchvision_sequential(path, n_frames):
    video, _, _ = torchvision.io.read_video(str(path), pts_unit="sec",
                                            output_format="THWC")
    return video.numpy()


def torchvision_random(path, indices):
    # torchvision.io has no frame-index random access; the honest version
    # decodes everything and throws most of it away.
    video, _, _ = torchvision.io.read_video(str(path), pts_unit="sec",
                                            output_format="THWC")
    return video.numpy()[indices]


def pyav_sequential(path, n_frames):
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return np.stack(frames)


def pyav_random(path, indices):
    # Seek to the nearest keyframe at or before the target, then decode
    # forward until we reach it.
    frames = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate)
        for idx in indices:
            target_pts = int(idx / rate / stream.time_base)
            container.seek(target_pts, stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts >= target_pts:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    break
    return np.stack(frames)


def ffmpeg_sequential(path, n_frames, size=(320, 240)):
    out, _ = (
        ffmpeg.input(str(path))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(cmd=FFMPEG_EXE, capture_stdout=True, capture_stderr=True)
    )
    w, h = size
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def ffmpeg_random(path, indices, size=(320, 240)):
    # A select filter keeps only the wanted frames — but ffmpeg still has
    # to decode every frame internally to know which ones to keep.
    expr = "+".join(f"eq(n\\,{i})" for i in indices)
    out, _ = (
        ffmpeg.input(str(path))
        .output("pipe:", format="rawvideo", pix_fmt="rgb24",
                vf=f"select='{expr}'", fps_mode="vfr")
        .run(cmd=FFMPEG_EXE, capture_stdout=True, capture_stderr=True)
    )
    w, h = size
    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)


def opencv_sequential(path, n_frames):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img[:, :, ::-1])  # BGR -> RGB
    cap.release()
    return np.stack(frames)


def opencv_random(path, indices):
    cap = cv2.VideoCapture(str(path))
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        frames.append(img[:, :, ::-1])
    cap.release()
    return np.stack(frames)


LOADERS = {
    "decord": (decord_sequential, decord_random),
    "PyAV": (pyav_sequential, pyav_random),
    "OpenCV": (opencv_sequential, opencv_random),
    "torchvision.io": (torchvision_sequential, torchvision_random),
    "ffmpeg-python": (ffmpeg_sequential, ffmpeg_random),
}


def main():
    OUT.mkdir(exist_ok=True)
    folder = vid_lib.make_benchmark_clips()
    clips = sorted(folder.glob("clip_*.mp4"))
    n_frames = 64
    indices = np.linspace(0, n_frames - 1, N_RANDOM).astype(int).tolist()
    print(f"{len(clips)} clips x {n_frames} frames, random pattern = {indices}")

    # Correctness check first: everyone must return the same frame count.
    for name, (seq_fn, rnd_fn) in LOADERS.items():
        assert len(seq_fn(clips[0], n_frames)) == n_frames, name
        assert len(rnd_fn(clips[0], indices)) == N_RANDOM, name

    results = {}
    for name, (seq_fn, rnd_fn) in LOADERS.items():
        best = {}
        for pattern, fn, arg in [("sequential", seq_fn, n_frames),
                                 ("random", rnd_fn, indices)]:
            times = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                for clip in clips:
                    fn(clip, arg)
                times.append(time.perf_counter() - t0)
            best[pattern] = min(times) / len(clips) * 1000  # ms per clip
        results[name] = best
        print(f"{name:15s} sequential {best['sequential']:7.1f} ms/clip   "
              f"random {best['random']:7.1f} ms/clip")

    with open(OUT / "benchmark.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["library", "sequential_ms_per_clip",
                        "random_ms_per_clip", "sequential_frames_per_s"])
        for name, best in results.items():
            writer.writerow([name, f"{best['sequential']:.1f}",
                             f"{best['random']:.1f}",
                             f"{n_frames / best['sequential'] * 1000:.0f}"])

    plot(results, n_frames)


def plot(results, n_frames):
    names = sorted(results, key=lambda n: results[n]["sequential"])
    fig, axes = ps.plt.subplots(1, 2, figsize=(9.6, 3.6), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, pattern, title in [
        (axes[0], "sequential", f"Decode all {n_frames} frames"),
        (axes[1], "random", "Grab 8 spread-out frames"),
    ]:
        ps.style_axes(ax)
        vals = [results[n][pattern] for n in names]
        bars = ax.barh(range(len(names)), vals, color=ps.SERIES[0], height=0.6)
        ax.set_yticks(range(len(names)), names)
        ax.invert_yaxis()
        ax.set_title(title, color=ps.INK, fontsize=11, loc="left")
        ax.set_xlabel("ms per clip (lower is better)", color=ps.INK_SECONDARY,
                      fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(v, bar.get_y() + bar.get_height() / 2, f" {v:.0f}",
                    va="center", color=ps.INK_SECONDARY, fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Time to load one 64-frame 320x240 H.264 clip, by library",
                 color=ps.INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "decode_speed.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print(f"wrote {OUT / 'decode_speed.png'}")


if __name__ == "__main__":
    main()
