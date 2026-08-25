"""Run Stable Video Diffusion on a CPU — at an honestly reduced setting.

Three generations from the same conditioning image:

  1. SVD (14-frame checkpoint), motion_bucket_id = 20   (ask for calm)
  2. SVD (14-frame checkpoint), motion_bucket_id = 180  (ask for action)
  3. SVD-XT (25-frame checkpoint), motion_bucket_id = 127 (its default)

All at 256x320, 8 frames, 8 denoising steps — far below the model's
native 576x1024 / 14-25 frames / 25 steps operating point, which is
what makes ~2.5 min per clip on a 12-core CPU possible at all.

  python3 run_svd.py            # ~9 min total: generate + figures
  python3 run_svd.py --plot     # remake figures from saved frames
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
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))

from vid_lib import ensure_sources                     # noqa: E402
import plot_style as ps                                # noqa: E402

CKPT = HERE / "checkpoints"
OUT = HERE / "outputs"
CKPT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

H, W = 256, 320
N_FRAMES = 8
N_STEPS = 8

RUNS = [
    ("svd_bucket020", "stabilityai/stable-video-diffusion-img2vid", 20),
    ("svd_bucket180", "stabilityai/stable-video-diffusion-img2vid", 180),
    ("svdxt_bucket127", "stabilityai/stable-video-diffusion-img2vid-xt", 127),
]


def conditioning_image():
    """A frame of Big Buck Bunny, from project 01's downloader."""
    paths = ensure_sources()
    cap = cv2.VideoCapture(str(paths["bunny.mp4"]))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
    ok, fr = cap.read()
    cap.release()
    assert ok
    return Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((W, H))


def flow_magnitude(frames):
    """Mean Farneback optical-flow magnitude across the clip (px/frame)."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    mags = []
    for a, b in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mags.append(np.linalg.norm(flow, axis=-1).mean())
    return float(np.mean(mags))


def generate():
    from diffusers import StableVideoDiffusionPipeline

    img = conditioning_image()
    img.save(OUT / "condition.png")

    loaded = {}
    for name, repo, bucket in RUNS:
        if repo not in loaded:
            print(f"loading {repo} ...", flush=True)
            loaded[repo] = StableVideoDiffusionPipeline.from_pretrained(
                repo, variant="fp16", torch_dtype=torch.float32)
            loaded[repo].set_progress_bar_config(disable=True)
        pipe = loaded[repo]
        t0 = time.time()
        out = pipe(img, height=H, width=W, num_frames=N_FRAMES,
                   num_inference_steps=N_STEPS, motion_bucket_id=bucket,
                   fps=7, decode_chunk_size=2,
                   generator=torch.Generator().manual_seed(0))
        frames = np.stack([np.asarray(f) for f in out.frames[0]])
        np.savez_compressed(CKPT / f"{name}.npz", frames=frames)
        print(f"{name}: {time.time()-t0:.0f}s, "
              f"flow {flow_magnitude(frames):.2f} px/frame", flush=True)
        del pipe
        # free the first pipeline before loading the second (RAM)
        if name == "svd_bucket180":
            loaded.clear()


def figures():
    clips = {name: np.load(CKPT / f"{name}.npz")["frames"]
             for name, _, _ in RUNS}

    # --- strips -------------------------------------------------------
    for name, frames in clips.items():
        strip = np.concatenate(list(frames[::2]), axis=1)
        img = Image.fromarray(strip)
        img.resize((img.width // 2, img.height // 2),
                   Image.LANCZOS).save(OUT / f"strip_{name}.png")

    # --- animated GIFs (small, for the README) ------------------------
    for name in ["svd_bucket020", "svd_bucket180"]:
        small = [Image.fromarray(f).resize((W // 2, H // 2))
                 for f in clips[name]]
        small[0].save(OUT / f"{name}.gif", save_all=True,
                      append_images=small[1:], duration=140, loop=0)

    # --- realized motion ---------------------------------------------
    rows = [(name, bucket, flow_magnitude(clips[name]))
            for name, _, bucket in RUNS]
    with open(OUT / "metrics.csv", "w") as f:
        f.write("run,motion_bucket_id,mean_flow_px_per_frame\n")
        for name, bucket, flow in rows:
            f.write(f"{name},{bucket},{flow:.2f}\n")
            print(f"{name:18s} bucket {bucket:3d} -> flow {flow:.2f}")

    fig, ax = ps.new_axes(5.6, 3.8)
    labels = [f"bucket {b}\n({n.split('_')[0]})" for n, _, b in RUNS]
    ax.bar(range(len(rows)), [r[2] for r in rows],
           color=[ps.SERIES[0], ps.SERIES[2], ps.SERIES[1]], width=0.6)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=9)
    ps.finish(fig, ax, "Requested motion bucket vs realized optical flow",
              "", "mean flow magnitude (px/frame)",
              OUT / "motion_vs_bucket.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true",
                    help="only remake figures from saved frames")
    args = ap.parse_args()
    if not args.plot:
        generate()
    figures()
