"""Project 22 — make the 3D VAE causal, so one compressor handles both
single images and video.

The change is one line of padding (see `CausalConv3d` in project 21's
`vae3d_lib.py`): move all of the temporal padding to the front, so frame t is
built from frames <= t and never from t+1.

Two consequences we measure here:

  1. T=1 in gives T'=1 out and 1 frame back.  The non-causal model cannot do
     this at all -- its stride-2 temporal convs need frames on both sides.
  2. Frame 0 is encoded *alone*, which is exactly the still-image case, so the
     same weights serve images and video.  We check the model does not pay for
     this with worse video reconstruction.

Clip length is 17, not 16.  A causal encoder with two stride-2 temporal stages
maps 1 + 4k frames onto 1 + k latent frames: frame 0 gets its own latent slot,
then every following group of 4 frames shares one.  17 = 1 + 4*4 -> 5 latent
frames.  Feeding it 16 would leave a ragged final group.

    python3 train.py --stage train     # ~9 min on CPU
    python3 train.py --stage figures   # ~1 min
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
import plot_style as ps
import matplotlib.pyplot as plt

import vae3d_lib as V

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

T_CAUSAL = 17          # 1 + 4*4 — see the module docstring
STEPS = 800
BATCH = 8
LR = 3e-4
BASE = 16
P_IMAGE = 0.25         # fraction of steps that train on single frames


def make_source(seed, train=True):
    return V.make_source(seed=seed, seq_len=T_CAUSAL, train=train)


def train():
    torch.manual_seed(0)
    model = V.VideoVAE(base=BASE, z_ch=V.Z_CH, causal=True)
    src = make_source(1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=5e-5)
    rng = np.random.default_rng(0)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        x = V.clip_batch(src, BATCH)
        # Mixed image/video batches. A causal model *can* take a 1-frame clip,
        # but "can" is not "is good at": if it only ever sees 17-frame clips,
        # frame 0's encoder path is only ever trained as the opening of a
        # video. Feeding single frames a quarter of the time trains that path
        # as a stand-alone image encoder too — the miniature version of the
        # image/video co-training real 3D VAEs do.
        if rng.random() < P_IMAGE:
            x = x[:, :, :1]
        loss, l1, kl = V.vae_loss(model, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 25 == 0:
            log.append((step, float(l1), float(kl)))
        if step % 200 == 0:
            print(f"step {step:4d}  L1 {float(l1):.4f}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    model.eval()
    torch.save({"state": model.state_dict(),
                "scale": V.latent_scale(model, make_source(99, train=False))},
               CK / "causal.pt")
    np.save(OUT / "loss.npy", np.array(log))
    print(f"done in {time.time() - t0:.0f}s")


def load_causal():
    ck = torch.load(CK / "causal.pt", map_location="cpu", weights_only=False)
    m = V.VideoVAE(base=BASE, z_ch=V.Z_CH, causal=True)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def load_noncausal():
    """Project 21's non-causal 3D VAE, for the comparison rows."""
    p = HERE.parent / "21-train-a-small-3d-vae" / "checkpoints" / "3d.pt"
    if not p.exists():
        return None
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = V.VideoVAE(base=BASE, z_ch=4, temporal=True)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def shape_table(causal, noncausal):
    """T in -> T' latent -> T out, for both models and the lengths that matter.

    The point of the non-causal rows is that they are *broken*, not slow: a
    symmetric temporal conv has no way to represent "there is no frame -1", so
    a 1-frame clip comes back as several frames."""
    rows = []
    for tag, model in (("causal", causal), ("non-causal", noncausal)):
        if model is None:
            continue
        for T in (1, 5, 9, 17):
            x = torch.zeros(1, 1, T, 64, 64)
            mean, _ = model.encode(x)
            rec = model.decoder(mean)
            rows.append(dict(model=tag, frames_in=T,
                             latent_frames=mean.shape[2],
                             frames_out=rec.shape[2],
                             roundtrip_ok=int(rec.shape[2] == T)))
    return rows


@torch.no_grad()
def causality_probe(model, edit_frames=(16, 8, 4)):
    """Overwrite one input frame; see which latent slots move.

    A causal encoder must leave every *earlier* latent slot bit-for-bit
    untouched. Measuring that directly beats trusting the padding arithmetic —
    this is the probe that caught the normalization bug described in the
    README, which no amount of staring at the conv definitions revealed."""
    src = make_source(5, train=False)
    x = V.clip_batch(src, 1)
    base_z, _ = model.encode(x)

    out = {}
    for f in edit_frames:
        edited = x.clone()
        edited[:, :, f] = torch.randn_like(edited[:, :, f]).clamp(-1, 1)
        new_z, _ = model.encode(edited)
        per_slot = (new_z - base_z).abs().mean(dim=(0, 1, 3, 4))
        out[f] = [float(v) for v in per_slot]
    return out


@torch.no_grad()
def figures():
    model, ck = load_causal()
    nc = load_noncausal()

    shapes = shape_table(model, nc)
    with open(OUT / "shapes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(shapes[0]))
        w.writeheader()
        w.writerows(shapes)
    for r in shapes:
        print(r)

    probe = causality_probe(model)
    with open(OUT / "causality.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["edited_frame"] + [f"slot{i}" for i in
                                       range(len(next(iter(probe.values()))))])
        for frame, slots in sorted(probe.items(), reverse=True):
            w.writerow([frame] + [f"{v:.6f}" for v in slots])
            print(f"edited frame {frame:2d} -> "
                  f"{[round(v, 6) for v in slots]}")

    # video quality, and single-image quality, for both models
    rows = []
    test = make_source(7, train=False)
    m = V.evaluate(model, make_source(7, train=False), batches=6)
    rows.append(dict(model="causal (this project)", **_fmt(m),
                     image_psnr=round(image_psnr(model, test), 2),
                     latent_scale=round(ck["scale"], 3)))
    if nc is not None:
        test16 = V.make_source(seed=7, seq_len=16, train=False)
        m2 = V.evaluate(nc, test16, batches=6)
        rows.append(dict(model="non-causal (project 21)", **_fmt(m2),
                         image_psnr=float("nan"), latent_scale=float("nan")))
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)

    figure_causality(probe)
    figure_recon(model, test)
    figure_single_image(model, test)
    print("wrote", OUT)


def _fmt(m):
    return dict(psnr_db=round(m["psnr"], 2),
                flicker_error=round(m["flicker_err"], 4))


@torch.no_grad()
def image_psnr(model, src, n=4):
    """Reconstruction quality on *isolated single frames* (T=1)."""
    x = V.clip_batch(src, n)[:, :, :1]
    return V.psnr(x, model.reconstruct(x))


def figure_causality(probe):
    fig, ax = ps.new_axes(7.4, 4.2)
    n = len(next(iter(probe.values())))
    w = 0.8 / len(probe)
    for i, (frame, slots) in enumerate(sorted(probe.items(), reverse=True)):
        ax.bar(np.arange(n) + (i - (len(probe) - 1) / 2) * w, slots, w,
               color=ps.SERIES[i], label=f"edited input frame {frame}")
    ax.set_xlabel("latent slot t'  (slot 0 = input frame 0 alone)")
    ax.set_ylabel("mean |change| in that slot")
    ax.set_title("Editing a frame moves that slot and later ones — never "
                 "earlier ones", color=ps.INK)
    ax.set_xticks(range(n))
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "causality_probe.png")
    plt.close(fig)


@torch.no_grad()
def figure_recon(model, src):
    x = V.clip_batch(src, 1)
    rec = model.reconstruct(x)
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 3.2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (tag, clip) in zip(axes, (("input", x), ("causal VAE", rec))):
        ax.imshow(V.strip(clip, n=9), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=10)
    fig.suptitle("First 9 of 17 frames", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "reconstructions.png")
    plt.close(fig)


@torch.no_grad()
def figure_single_image(model, src):
    """The same weights, asked to compress four isolated still images."""
    x = V.clip_batch(src, 4)[:, :, :1]
    rec = model.reconstruct(x)
    fig, axes = plt.subplots(2, 4, figsize=(8.0, 4.4), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for i in range(4):
        for r, clip in enumerate((x, rec)):
            ax = axes[r, i]
            ax.imshow(((clip[i, 0, 0] + 1) / 2).clamp(0, 1),
                      cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(("input", "recon")[r],
                              color=ps.INK_SECONDARY, fontsize=9)
    fig.suptitle("T=1: one image in, one image out, same weights",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "single_image.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "figures"])
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "train":
        train()
    else:
        figures()
