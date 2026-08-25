"""Camera trajectory control for the tiny I2V model.

Three conditioning arms, same everything else — a ladder of how much
geometric work is left to the network:

  pose    — the raw 3-number camera pose per frame, via a small MLP
            into the temporal layers' FiLM input (all of the work)
  plucker — per-pixel Plucker ray maps through conv adapters (a hint)
  warp    — the conditioning frame pre-warped by the requested camera
            motion, through the same adapters (none of the work)

Stages:
  python3 train.py --stage image     # ~7 min: pretrain 2D U-Net on views
  python3 train.py --stage warp      # ~9 min
  python3 train.py --stage plucker   # ~9 min
  python3 train.py --stage pose      # ~9 min
  python3 train.py --stage figures   # ~8 min: sample, measure, plot
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(12)

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-tiny-i2v-model"))

import i2v_lib as lib                                   # noqa: E402
from i2v_lib import (Diffusion, ImageUNet, VideoUNet, train_image_model,
                     train_video_model, sample_clip, strip, set_seed)
import plot_style as ps                                 # noqa: E402
import camera                                           # noqa: E402
from camera import SceneSampler, batch, VIEW, T as T_FRAMES

CKPT = HERE / "checkpoints"
OUT = HERE / "outputs"
CKPT.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

BATCH = 8
IMAGE_STEPS = 800
VIDEO_STEPS = 1200
EMB_DIM = 128


class PoseVideoUNet(VideoUNet):
    """Baseline: encode the raw (B, T, 3) pose with an MLP -> FiLM."""

    def __init__(self, image_unet, freeze_spatial=True):
        super().__init__(image_unet, freeze_spatial,
                         extra_emb_dim=EMB_DIM)
        self.pose_enc = nn.Sequential(
            nn.Linear(3, EMB_DIM), nn.SiLU(),
            nn.Linear(EMB_DIM, EMB_DIM))

    def forward(self, x, t, cond, extra_emb=None, extra_maps=None):
        return super().forward(x, t, cond,
                               extra_emb=self.pose_enc(extra_emb))


def make_model(arm, image_unet):
    if arm == "plucker":
        return VideoUNet(image_unet, freeze_spatial=True, extra_map_c=6)
    if arm == "warp":
        return VideoUNet(image_unet, freeze_spatial=True, extra_map_c=1)
    return PoseVideoUNet(image_unet)


def feed(arm, d):
    if arm == "plucker":
        return {"clips": d["clips"], "extra_maps": d["maps"]}
    if arm == "warp":
        return {"clips": d["clips"], "extra_maps": d["warp"]}
    return {"clips": d["clips"], "extra_emb": d["pose"]}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_image():
    set_seed(6)
    diff = Diffusion()
    model = ImageUNet()
    sampler = SceneSampler(train=True, seed=31)

    def source():
        d = batch(sampler, BATCH)
        return d["clips"]

    t0 = time.time()
    losses = train_image_model(model, diff, source, IMAGE_STEPS)
    print(f"image stage: {time.time()-t0:.0f}s")
    torch.save(model.state_dict(), CKPT / "image_unet.pt")
    np.save(CKPT / "loss_image.npy", np.array(losses))


def stage_video(arm):
    set_seed(7)
    diff = Diffusion()
    image_unet = ImageUNet()
    image_unet.load_state_dict(torch.load(CKPT / "image_unet.pt"))
    model = make_model(arm, image_unet)
    n = sum(p.numel() for p in model.trainable_parameters())
    print(f"{arm}: {n/1e3:.0f}k trainable params")
    sampler = SceneSampler(train=True, seed=41)

    def source():
        return feed(arm, batch(sampler, BATCH))

    t0 = time.time()
    losses = train_video_model(model, diff, source, VIDEO_STEPS)
    print(f"{arm} train: {time.time()-t0:.0f}s")
    torch.save(model.state_dict(), CKPT / f"{arm}.pt")
    np.save(CKPT / f"loss_{arm}.npy", np.array(losses))


# ---------------------------------------------------------------------------
# Measuring what the camera actually did in a clip
# ---------------------------------------------------------------------------

def measured_shifts(clip):
    """Cumulative camera path: chained per-pair phase correlation.

    Each adjacent frame pair is measured (a ~2 px shift, well within
    the estimator's comfort zone because adjacent generated frames are
    temporally smooth), then the per-pair estimates are summed into a
    cumulative path.  Correlating each frame directly against frame 0
    sounds more precise but fails badly on generated clips: by late
    frames most of the content is newly invented, the overlap with
    frame 0 shrinks, and the estimator locks onto spurious alignments.
    """
    fr = clip[:, 0].numpy().astype(np.float64) if torch.is_tensor(clip) \
        else clip.astype(np.float64)
    shifts = [cv2.phaseCorrelate(a, b)[0]
              for a, b in zip(fr[:-1], fr[1:])]
    return np.cumsum(np.array(shifts), axis=0)          # (T-1, 2)


def measured_scales(clip):
    """Per-frame-pair zoom factor via brute-force scale matching."""
    fr = clip[:, 0].numpy() if torch.is_tensor(clip) else clip
    cands = np.linspace(0.86, 1.16, 31)
    out = []
    for a, b in zip(fr[:-1], fr[1:]):
        errs = []
        for s in cands:
            side = max(8, int(round(VIEW * s)))
            big = cv2.resize(a, (side, side), interpolation=cv2.INTER_LINEAR)
            patch = cv2.getRectSubPix(big, (VIEW, VIEW),
                                      ((side - 1) / 2, (side - 1) / 2))
            errs.append(((patch - b) ** 2).mean())
        out.append(cands[int(np.argmin(errs))])
    return np.array(out)                                # (T-1,)


CONTROL_KINDS = [
    dict(kind="pan", angle=0.0, speed=2.0),      # pan right
    dict(kind="pan", angle=180.0, speed=2.0),    # pan left
    dict(kind="pan", angle=90.0, speed=2.0),     # pan down (image coords)
    dict(kind="pan", angle=270.0, speed=2.0),    # pan up
    dict(kind="zoom", rate=1.06),                # zoom out (window grows)
    dict(kind="zoom", rate=0.94),                # zoom in
]
CONTROL_NAMES = ["pan right", "pan left", "pan down", "pan up",
                 "zoom out", "zoom in"]
HELDOUT_KINDS = [
    dict(kind="pan", angle=45.0, speed=2.0),
    dict(kind="panzoom", angle=0.0, speed=2.0, rate=0.94),
    dict(kind="panzoom", angle=90.0, speed=2.0, rate=1.06),
    dict(kind="curve", angle=0.0, speed=2.0),
]
HELDOUT_NAMES = ["diagonal pan (unseen direction)", "pan right + zoom in",
                 "pan down + zoom out", "curved path"]


def same_scene_batch(kinds, seed):
    """All trajectories filmed in the SAME scene from the same start view.

    The start center is chosen once, inside the intersection of every
    trajectory's feasible-start box, so all clips share frame 0 exactly
    and differ only in the requested camera path.
    """
    sampler = SceneSampler(train=False, seed=seed)
    canvas = sampler.scene()
    paths = [sampler.offsets(**kw) for kw in kinds]
    boxes = np.array([sampler.feasible_box(o, s) for o, s in paths])
    lo_x, hi_x = boxes[:, 0].max(), boxes[:, 1].min()
    lo_y, hi_y = boxes[:, 2].max(), boxes[:, 3].min()
    cx = sampler.rng.uniform(lo_x, hi_x)
    cy = sampler.rng.uniform(lo_y, hi_y)
    # Guarantee the shared start view has solid content: paste one digit
    # at the window center, so conditioning and the shift measurement
    # have more than faint dots to hold on to.
    sprite = sampler.digits[sampler.rng.integers(len(sampler.digits))]
    D = sprite.shape[0]
    y0, x0 = int(round(cy - D / 2)), int(round(cx - D / 2))
    patch = canvas[y0:y0 + D, x0:x0 + D]
    np.maximum(patch, sprite, out=patch)
    clips, poses, maps, warps = [], [], [], []
    for offsets, sides in paths:
        centers = np.stack([offsets[:, 0] + cx, offsets[:, 1] + cy], axis=1)
        frames = sampler.render(canvas, centers, sides)
        clips.append(frames)
        poses.append(camera.relative_pose(centers, sides))
        maps.append(camera.plucker_map(centers, sides))
        warps.append(camera.warp_stack(frames[0], centers, sides))
    return {
        "clips": torch.from_numpy(np.stack(clips)).unsqueeze(2),
        "pose": torch.from_numpy(np.stack(poses)),
        "maps": torch.from_numpy(np.stack(maps)),
        "warp": torch.from_numpy(np.stack(warps)),
    }


def sample_arm(arm, model, diff, d, seed):
    cond = d["clips"][:, 0]
    kw = ({"extra_maps": d["maps"]} if arm == "plucker"
          else {"extra_maps": d["warp"]} if arm == "warp"
          else {"extra_emb": d["pose"]})
    return sample_clip(model, diff, cond, T_FRAMES, seed=seed, **kw)


def stage_figures():
    set_seed(8)
    diff = Diffusion()
    arms = {}
    for arm in ["warp", "plucker", "pose"]:
        image_unet = ImageUNet()
        m = make_model(arm, image_unet)
        m.load_state_dict(torch.load(CKPT / f"{arm}.pt"))
        m.eval()
        arms[arm] = m

    ctrl = same_scene_batch(CONTROL_KINDS, seed=77)
    held = same_scene_batch(HELDOUT_KINDS, seed=88)

    cache = CKPT / "gen_cache.npz"
    if cache.exists():
        z = np.load(cache)
        gen = {(arm, key): torch.from_numpy(z[f"{arm}_{key}"])
               for arm in arms for key in ["ctrl", "held"]}
        print("loaded cached samples")
    else:
        gen = {}
        for arm, model in arms.items():
            t0 = time.time()
            gen[arm, "ctrl"] = sample_arm(arm, model, diff, ctrl, seed=9)
            gen[arm, "held"] = sample_arm(arm, model, diff, held, seed=9)
            print(f"{arm} sampling: {time.time()-t0:.0f}s", flush=True)
        np.savez_compressed(cache, **{f"{arm}_{key}": v.numpy()
                                      for (arm, key), v in gen.items()})

    # --- pan error table ---------------------------------------------
    # Reference = the same estimator run on the real renders, so the
    # comparison cannot be skewed by the estimator's own quirks.
    rows = []
    for name_list, key, d in [(CONTROL_NAMES, "ctrl", ctrl),
                              (HELDOUT_NAMES, "held", held)]:
        for i, name in enumerate(name_list):
            ref_s = measured_shifts(d["clips"][i])
            ref_z = measured_scales(d["clips"][i])
            row = {"traj": name, "heldout": key == "held"}
            # reference: a model that ignores the camera and stays still
            row["static_err"] = np.abs(ref_s).mean()
            for arm in arms:
                g = gen[arm, key][i]
                err_s = np.abs(measured_shifts(g) - ref_s).mean()
                err_z = np.abs(measured_scales(g) - ref_z).mean()
                row[arm + "_pan_err"] = err_s
                row[arm + "_zoom_err"] = err_z
            rows.append(row)

    def fmt(v):
        if isinstance(v, (str, bool, np.bool_)):
            return str(v)
        return f"{float(v):.3f}"

    with open(OUT / "metrics.csv", "w") as f:
        cols = ["traj", "heldout", "warp_pan_err", "plucker_pan_err",
                "pose_pan_err", "static_err", "warp_zoom_err",
                "plucker_zoom_err", "pose_zoom_err"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(fmt(r[c]) for c in cols) + "\n")
            print("  ".join(f"{c}={fmt(r[c])}" for c in cols), flush=True)

    # --- bar chart: pan error, in-dist vs held-out --------------------
    fig, ax = ps.new_axes(7.0, 4.2)
    names = [r["traj"] for r in rows]
    xs = np.arange(len(rows))
    w = 0.27
    ax.bar(xs - w, [r["warp_pan_err"] for r in rows], w,
           color=ps.SERIES[1], label="warped condition")
    ax.bar(xs, [r["plucker_pan_err"] for r in rows], w,
           color=ps.SERIES[0], label="Plucker ray maps")
    ax.bar(xs + w, [r["pose_pan_err"] for r in rows], w,
           color=ps.SERIES[2], label="raw pose numbers")
    ax.plot(xs, [r["static_err"] for r in rows], ls="none", marker="_",
            ms=22, mew=2, color=ps.INK_MUTED,
            label="if the model ignored the camera (stayed still)")
    ax.axvline(len(CONTROL_NAMES) - 0.5, color=ps.INK_MUTED, lw=1, ls="--")
    ax.text(len(CONTROL_NAMES) - 0.4, ax.get_ylim()[1] * 0.95,
            "held-out trajectories", fontsize=8, color=ps.INK_SECONDARY)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Camera-following error (cumulative camera path)",
              "", "mean |realized - requested| shift (px)",
              OUT / "pan_error.png")

    # --- strips: same start frame, different camera requests ----------
    order = [0, 1, 5, 4]                    # right, left, zoom in, zoom out
    img = strip(torch.cat([gen["warp", "ctrl"][order],
                           ctrl["clips"][order[:1]]]))
    Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4),
                                Image.NEAREST).save(OUT / "control_strips.png")

    img = strip(torch.cat([held["clips"][:1],
                           gen["warp", "held"][:1],
                           gen["plucker", "held"][:1],
                           gen["pose", "held"][:1]]))
    Image.fromarray(img).resize((img.shape[1] * 4, img.shape[0] * 4),
                                Image.NEAREST).save(OUT / "heldout_strips.png")

    # --- 2D camera paths: requested vs realized (warp arm) ------------
    fig, ax = ps.new_axes(5.6, 5.2)
    path_set = [("ctrl", 0, "pan right"), ("ctrl", 1, "pan left"),
                ("ctrl", 2, "pan down"), ("ctrl", 3, "pan up"),
                ("held", 0, "diagonal pan")]
    for k, (key, i, label) in enumerate(path_set):
        d = ctrl if key == "ctrl" else held
        ref = np.vstack([[0, 0], measured_shifts(d["clips"][i])])
        got = np.vstack([[0, 0], measured_shifts(gen["warp", key][i])])
        c = ps.SERIES[k % len(ps.SERIES)]
        ax.plot(ref[:, 0], ref[:, 1], color=c, lw=1.4, ls="--", alpha=0.65)
        ax.plot(got[:, 0], got[:, 1], color=c, lw=1.8, marker="o", ms=3)
        ax.annotate(label, ref[-1], fontsize=8, color=c,
                    xytext=(4, 4), textcoords="offset points")
    ax.plot([], [], color=ps.INK_MUTED, ls="--", label="requested (real render)")
    ax.plot([], [], color=ps.INK_MUTED, marker="o", ms=3, label="realized (generated)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_aspect("equal")
    ps.finish(fig, ax, "Camera paths: requested vs realized (warp arm)",
              "cumulative x shift (px)", "cumulative y shift (px)",
              OUT / "camera_paths.png")

    # GIF: four camera requests side by side (warp arm, same start)
    clips4 = [gen["warp", "ctrl"][i] for i in order]
    frames = []
    for t in range(T_FRAMES):
        cells = [c[t, 0].numpy() for c in clips4]
        row = np.concatenate(
            [np.concatenate([c, np.full((32, 2), 0.25)], axis=1)
             for c in cells], axis=1)[:, :-2]
        frames.append(Image.fromarray(
            (row * 255).astype(np.uint8)).resize(
            (row.shape[1] * 4, 128), Image.NEAREST))
    frames[0].save(OUT / "four_requests.gif", save_all=True,
                   append_images=frames[1:], duration=200, loop=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["image", "warp", "plucker", "pose", "figures"])
    args = ap.parse_args()
    if args.stage == "image":
        stage_image()
    elif args.stage in ("warp", "plucker", "pose"):
        stage_video(args.stage)
    else:
        stage_figures()
