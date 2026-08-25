"""Store the same 100 clips three ways; measure disk, speed, and quality.

  raw .npy    — the decoded uint8 pixels, no compression at all
  H.264 .mp4  — the 2003 workhorse codec (libx264, crf 23)
  AV1 .webm   — the 2018 royalty-free codec (libsvtav1, crf 35)

For each format we measure total disk footprint, encode time, full-decode
time per clip, and reconstruction quality (PSNR against the raw pixels).

Run:  python3 storage.py        (~4 minutes)
"""

import csv
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
import vid_lib
import plot_style as ps

import av

OUT = HERE / "outputs"
DATA = HERE / "data"
N_CLIPS = 100
T, W, H = 16, 256, 256


def gather_clips():
    """100 clips of 16 frames at 256x256 from the three source videos."""
    vid_lib.ensure_sources()
    frames = np.concatenate([
        vid_lib.read_frames(vid_lib.DATA / name, resize=(W, H))
        for name in ("vtest.avi", "Megamind.avi", "bunny.mp4")])
    stride = (len(frames) - T) // (N_CLIPS - 1)
    clips = np.stack([frames[i * stride:i * stride + T]
                      for i in range(N_CLIPS)])
    assert clips.shape == (N_CLIPS, T, H, W, 3)
    return clips


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)


def store_npy(clips, folder):
    folder.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for i, clip in enumerate(clips):
        np.save(folder / f"clip_{i:03d}.npy", clip)
    return time.perf_counter() - t0


def store_codec(clips, folder, codec, crf, ext, options=None):
    folder.mkdir(parents=True, exist_ok=True)
    seconds = 0.0
    for i, clip in enumerate(clips):
        seconds += vid_lib.write_video(
            clip, folder / f"clip_{i:03d}{ext}", fps=24, codec=codec,
            crf=crf, options=options)
    return seconds


def decode_npy(folder):
    out = []
    for path in sorted(folder.glob("*.npy")):
        out.append(np.load(path))
    return np.stack(out)


def decode_codec(folder, ext):
    out = []
    for path in sorted(folder.glob(f"*{ext}")):
        frames = []
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frames.append(frame.to_ndarray(format="rgb24"))
        out.append(np.stack(frames))
    return np.stack(out)


def folder_bytes(folder):
    return sum(p.stat().st_size for p in folder.iterdir())


def main():
    OUT.mkdir(exist_ok=True)
    if DATA.exists():
        shutil.rmtree(DATA)  # fresh measurement every run
    DATA.mkdir(parents=True)
    clips = gather_clips()
    raw_bytes = clips.nbytes
    print(f"{N_CLIPS} clips x {T} frames x {W}x{H} = "
          f"{raw_bytes / 1e6:.0f} MB of uint8 pixels in memory "
          f"({raw_bytes * 4 / 1e6:.0f} MB if stored as float32!)")

    formats = {}
    for name, store, decode in [
        ("raw .npy",
         lambda: store_npy(clips, DATA / "npy"),
         lambda: decode_npy(DATA / "npy")),
        ("H.264 .mp4",
         lambda: store_codec(clips, DATA / "h264", "libx264", 23, ".mp4"),
         lambda: decode_codec(DATA / "h264", ".mp4")),
        ("AV1 .webm",
         lambda: store_codec(clips, DATA / "av1", "libsvtav1", 35, ".webm",
                             options={"preset": "8"}),
         lambda: decode_codec(DATA / "av1", ".webm")),
    ]:
        enc_s = store()
        t0 = time.perf_counter()
        recon = decode()
        dec_s = time.perf_counter() - t0
        formats[name] = {
            "disk_mb": folder_bytes({"raw .npy": DATA / "npy",
                                     "H.264 .mp4": DATA / "h264",
                                     "AV1 .webm": DATA / "av1"}[name]) / 1e6,
            "encode_s": enc_s,
            "decode_ms": dec_s / N_CLIPS * 1000,
            "psnr": float("inf") if name == "raw .npy"
                    else psnr(clips, recon),
        }
        f = formats[name]
        print(f"{name:11s} disk {f['disk_mb']:7.1f} MB   "
              f"encode {f['encode_s']:6.1f} s   "
              f"decode {f['decode_ms']:5.1f} ms/clip   "
              f"PSNR {f['psnr']:.1f} dB")

    with open(OUT / "storage.csv", "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["format", "disk_mb", "encode_s", "decode_ms_per_clip",
                         "psnr_db"])
        for name, f in formats.items():
            writer.writerow([name, f"{f['disk_mb']:.1f}",
                             f"{f['encode_s']:.1f}", f"{f['decode_ms']:.1f}",
                             "inf" if f["psnr"] == float("inf")
                             else f"{f['psnr']:.1f}"])
    plot(formats)


def plot(formats):
    names = list(formats)
    colors = [ps.INK_MUTED, ps.SERIES[0], ps.SERIES[1]]
    panels = [
        ("disk_mb", "Disk for 100 clips (MB)", "{:.1f}", "log"),
        ("encode_s", "Encode time (s)", "{:.1f}", "linear"),
        ("decode_ms", "Decode time (ms/clip)", "{:.1f}", "linear"),
        ("psnr", "Quality, PSNR (dB)", "{:.1f}", "linear"),
    ]
    fig, axes = ps.plt.subplots(1, 4, figsize=(11.6, 3.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (key, title, fmt, scale) in zip(axes, panels):
        ps.style_axes(ax)
        vals = [formats[n][key] for n in names]
        show = [60.0 if v == float("inf") else v for v in vals]  # cap inf
        ax.bar(range(len(names)), show, color=colors, width=0.62)
        for x, (v, s) in enumerate(zip(vals, show)):
            label = "lossless" if v == float("inf") else fmt.format(v)
            ax.text(x, s, label, ha="center", va="bottom",
                    color=ps.INK_SECONDARY, fontsize=8.5)
        if scale == "log":
            ax.set_yscale("log")
            ax.set_ylim(top=max(show) * 6)   # headroom for value labels
        else:
            ax.set_ylim(top=max(show) * 1.28)
        ax.set_xticks(range(len(names)),
                      [n.replace(" ", "\n") for n in names], fontsize=8.5)
        ax.set_title(title, color=ps.INK, fontsize=10, loc="left", pad=10)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Same 100 clips, three storage formats",
                 color=ps.INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "storage.png", facecolor=ps.SURFACE,
                bbox_inches="tight")
    print(f"wrote {OUT / 'storage.png'}")


if __name__ == "__main__":
    main()
