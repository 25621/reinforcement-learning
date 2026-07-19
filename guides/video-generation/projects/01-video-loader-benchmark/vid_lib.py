"""Shared video utilities for the video-generation Phase-1 projects.

Downloads the three small real source videos every Phase-1 project uses,
and wraps PyAV for the two operations the projects repeat constantly:
"read every frame of this file as an RGB numpy array" and "encode this
stack of RGB frames into a real compressed video file".

Imported by projects 02-05 via sys.path (they add this directory to
sys.path and `import vid_lib`).
"""

import os
import time
import urllib.request
from fractions import Fraction
from pathlib import Path

os.environ.setdefault("SVT_LOG", "1")   # errors only from the AV1 encoder

import av
import av.logging
import cv2
import numpy as np

av.logging.set_level(av.logging.ERROR)  # silence per-file encoder banners

# All downloaded/derived video data lives under 01's data/ directory,
# which is gitignored — only the scripts and small PNG/CSV outputs are
# committed.
DATA = Path(__file__).resolve().parent / "data"

# Three small real videos, all served from GitHub (fast, reliable CDN).
# vtest:    static surveillance camera, pedestrians walking (one long shot)
# megamind: excerpt of a movie trailer (real editing cuts, fast motion)
# bunny:    5 s of the Big Buck Bunny animated short
SOURCES = {
    "vtest.avi": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi",
    "Megamind.avi": "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/Megamind.avi",
    "bunny.mp4": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/highgui/video/big_buck_bunny.mp4",
}


def ensure_sources():
    """Download the three source videos if they are not already on disk."""
    DATA.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in SOURCES.items():
        path = DATA / name
        if not path.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def read_frames(path, resize=None):
    """Decode every frame of a video into one (T, H, W, 3) uint8 RGB array.

    resize: optional (width, height) to rescale each frame to.
    """
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            img = frame.to_ndarray(format="rgb24")
            if resize is not None:
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
            frames.append(img)
    return np.stack(frames)


def frame_times(path):
    """Return the presentation timestamp (in seconds) of every frame."""
    times = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            times.append(float(frame.pts * stream.time_base))
    return np.array(times)


def write_video(frames, path, fps=24, codec="libx264", crf=23, options=None,
                pts_ms=None):
    """Encode a (T, H, W, 3) uint8 RGB array into a compressed video file.

    codec:  "libx264" (H.264) or "libsvtav1" (AV1) etc.
    crf:    constant-rate-factor quality knob (lower = better quality).
    pts_ms: optional list of per-frame timestamps in milliseconds. When
            given, the file is written with variable frame rate (VFR) —
            each frame is stamped with its own display time instead of
            being assumed to arrive every 1/fps seconds.
    Returns the encode wall-time in seconds.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = {"crf": str(crf)} if crf is not None else {}
    if options:
        opts.update(options)
    t0 = time.perf_counter()
    with av.open(str(path), "w") as container:
        stream = container.add_stream(codec, rate=fps, options=opts)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        if pts_ms is not None:
            stream.codec_context.time_base = Fraction(1, 1000)
        for i, img in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            if pts_ms is not None:
                frame.pts = int(pts_ms[i])
                frame.time_base = Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return time.perf_counter() - t0


def make_benchmark_clips(n_clips=16, frames_per_clip=64, size=(320, 240)):
    """Cut the source videos into a folder of same-sized H.264 .mp4 clips.

    Used by project 01 as the benchmark corpus and by later projects as a
    convenient pile of short clips. Returns the folder path.
    """
    folder = DATA / "clips"
    existing = sorted(folder.glob("clip_*.mp4"))
    if len(existing) == n_clips:
        return folder
    ensure_sources()
    folder.mkdir(parents=True, exist_ok=True)
    vtest = read_frames(DATA / "vtest.avi", resize=size)
    mega = read_frames(DATA / "Megamind.avi", resize=size)
    pool = []
    for src in (vtest, mega):
        for start in range(0, len(src) - frames_per_clip + 1, frames_per_clip):
            pool.append(src[start:start + frames_per_clip])
    assert len(pool) >= n_clips, f"only {len(pool)} clips available"
    for i, clip in enumerate(pool[:n_clips]):
        write_video(clip, folder / f"clip_{i:02d}.mp4", fps=24)
    return folder


def contact_sheet(frames, cols, out_path, scale=1.0, labels=None, pad=2):
    """Save a grid of frames as one PNG (a 'contact sheet').

    frames: list/array of (H, W, 3) uint8 RGB images (all the same size).
    labels: optional list of strings drawn in the top-left of each cell.
    """
    frames = [np.asarray(f) for f in frames]
    if scale != 1.0:
        frames = [cv2.resize(f, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA) for f in frames]
    h, w = frames[0].shape[:2]
    rows = int(np.ceil(len(frames) / cols))
    sheet = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 252,
                    dtype=np.uint8)
    for i, img in enumerate(frames):
        img = img.copy()
        if labels is not None:
            cv2.putText(img, str(labels[i]), (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.putText(img, str(labels[i]), (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = img
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet[:, :, ::-1])  # RGB -> BGR for cv2
    print(f"wrote {out_path}")
