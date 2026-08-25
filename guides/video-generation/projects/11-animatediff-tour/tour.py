"""AnimateDiff tour: one motion module, many image checkpoints.

Plugs the AnimateDiff-Lightning motion module (a 4-step distilled
version of the AnimateDiff motion module) into two different SD 1.5
community checkpoints and generates 16-frame clips on a CPU.

Generations (256x256, 16 frames, 4 steps, no CFG):

  A. epiCRealism   (photoreal checkpoint), seed 3
  B. dreamshaper-8 (stylized checkpoint),  seed 3   <- same seed as A
  C. epiCRealism, seed 17                           <- the "null" clip
  D. epiCRealism, 4 independent single-frame stills (no shared noise)
     = what per-frame image generation without temporal layers gives you

  python3 tour.py           # ~7 min: generate + figures
  python3 tour.py --plot    # remake figures from saved frames
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
import plot_style as ps                                # noqa: E402

CKPT = HERE / "checkpoints"
OUT = HERE / "outputs"
CKPT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

SIZE = 256
N_FRAMES = 16
PROMPT = ("a lighthouse on a cliff at sunset, ocean waves, cinematic, "
          "highly detailed")


def load_pipe(checkpoint):
    from diffusers import (AnimateDiffPipeline, EulerDiscreteScheduler,
                           MotionAdapter)
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    adapter = MotionAdapter()
    adapter.load_state_dict(load_file(hf_hub_download(
        "ByteDance/AnimateDiff-Lightning",
        "animatediff_lightning_4step_diffusers.safetensors")))
    kw = {}
    if checkpoint == "Lykon/dreamshaper-8":
        kw["variant"] = "fp16"
    pipe = AnimateDiffPipeline.from_pretrained(
        checkpoint, motion_adapter=adapter, torch_dtype=torch.float32, **kw)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing",
        beta_schedule="linear")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def gen_clip(pipe, seed, n_frames=N_FRAMES):
    out = pipe(prompt=PROMPT, height=SIZE, width=SIZE, num_frames=n_frames,
               guidance_scale=1.0, num_inference_steps=4,
               generator=torch.Generator().manual_seed(seed))
    return np.stack([np.asarray(f) for f in out.frames[0]])


def generate():
    pipe = load_pipe("emilianJR/epiCRealism")
    for name, seed in [("epic_seed3", 3), ("epic_seed17", 17)]:
        t0 = time.time()
        frames = gen_clip(pipe, seed)
        np.savez_compressed(CKPT / f"{name}.npz", frames=frames)
        print(f"{name}: {time.time()-t0:.0f}s", flush=True)

    # independent stills: each one is its own 1-frame "clip", so no
    # noise or temporal attention is shared between them
    stills = [gen_clip(pipe, 100 + i, n_frames=1)[0] for i in range(4)]
    np.savez_compressed(CKPT / "epic_stills.npz", frames=np.stack(stills))
    print("stills done", flush=True)
    del pipe

    pipe = load_pipe("Lykon/dreamshaper-8")
    t0 = time.time()
    frames = gen_clip(pipe, 3)                        # same seed as A
    np.savez_compressed(CKPT / "dream_seed3.npz", frames=frames)
    print(f"dream_seed3: {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def clip_flows(frames):
    """Farneback flow for every adjacent frame pair -> (T-1, H, W, 2)."""
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flows = []
    for a, b in zip(gray[:-1], gray[1:]):
        flows.append(cv2.calcOpticalFlowFarneback(
            a, b, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0))
    return np.stack(flows)


def figures():
    clips = {n: np.load(CKPT / f"{n}.npz")["frames"]
             for n in ["epic_seed3", "epic_seed17", "dream_seed3",
                       "epic_stills"]}

    # --- strips -------------------------------------------------------
    def save_half(arr, path):
        img = Image.fromarray(arr)
        img.resize((img.width // 2, img.height // 2),
                   Image.LANCZOS).save(path)

    for name in ["epic_seed3", "dream_seed3"]:
        save_half(np.concatenate(list(clips[name][::5]), axis=1),
                  OUT / f"strip_{name}.png")
    save_half(np.concatenate(list(clips["epic_stills"]), axis=1),
              OUT / "strip_stills.png")

    small = [Image.fromarray(f).resize((192, 192))
             for f in clips["epic_seed3"]]
    small[0].save(OUT / "epic_seed3.gif", save_all=True,
                  append_images=small[1:], duration=125, loop=0)

    # --- flicker: animated clip vs independent stills -----------------
    def deltas(frames):
        f = frames.astype(np.float32)
        return np.abs(np.diff(f, axis=0)).mean(axis=(1, 2, 3))

    d_clip = deltas(clips["epic_seed3"])
    d_stills = deltas(clips["epic_stills"])
    fig, ax = ps.new_axes(6.2, 3.8)
    ax.plot(d_clip, color=ps.SERIES[0], lw=1.8, marker="o", ms=4,
            label="AnimateDiff clip (motion module active)")
    ax.axhline(d_stills.mean(), color=ps.SERIES[2], lw=1.8, ls="--",
               label="independent stills (no motion module)")
    ax.set_ylim(0, d_stills.mean() * 1.25)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Frame-to-frame change, with and without "
              "the motion module", "frame pair",
              "mean |pixel change| (0-255)", OUT / "flicker.png")

    # --- does the same seed carry the same *motion* across models? ----
    fa = clip_flows(clips["epic_seed3"]).ravel()
    fb = clip_flows(clips["dream_seed3"]).ravel()
    fc = clip_flows(clips["epic_seed17"]).ravel()
    r_same = np.corrcoef(fa, fb)[0, 1]
    r_null = np.corrcoef(fa, fc)[0, 1]
    with open(OUT / "metrics.csv", "w") as f:
        f.write("comparison,flow_correlation\n")
        f.write(f"epiCRealism vs dreamshaper (same seed),{r_same:.3f}\n")
        f.write(f"epiCRealism seed3 vs seed17 (same model),{r_null:.3f}\n")
        f.write(f"clip mean |delta|,{d_clip.mean():.2f}\n")
        f.write(f"stills mean |delta|,{d_stills.mean():.2f}\n")
    print(f"flow correlation same-seed cross-model: {r_same:.3f}")
    print(f"flow correlation cross-seed same-model: {r_null:.3f}")

    # --- side by side: same seed, two checkpoints ---------------------
    top = np.concatenate(list(clips["epic_seed3"][::5]), axis=1)
    bottom = np.concatenate(list(clips["dream_seed3"][::5]), axis=1)
    save_half(np.concatenate([top, bottom], axis=0),
              OUT / "same_seed_two_styles.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if not args.plot:
        generate()
    figures()
