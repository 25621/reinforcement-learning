"""Train the tiny video GAN twice: once balanced, once rigged to collapse.

Usage:
  python3 train.py --config balanced    # healthy training (~5 min CPU)
  python3 train.py --config collapse    # G outruns D -> mode collapse
  python3 train.py --plot               # combined figures from both runs

The two runs are IDENTICAL (DCGAN default learning rates 2e-4) except
that halfway through the collapse run, the discriminator's learning rate
is cut to ~0 — the generator keeps learning against a frozen judge, and
mode collapse follows on schedule.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "01-video-loader-benchmark"))
import plot_style as ps  # noqa: E402
import vid_lib  # noqa: E402

from videogan import Discriminator, Generator, build_dataset  # noqa: E402

OUT = HERE / "outputs"
CKPT = HERE / "checkpoints"
STEPS = 700
BATCH = 16
CONFIGS = {
    "balanced": dict(lr_g=2e-4, lr_d=2e-4, starve_at=None),
    "collapse": dict(lr_g=2e-4, lr_d=2e-4, starve_at=350),
}
SNAP_EVERY = 100        # save sample clips + metrics every N steps


def diversity(clips):
    """Mean pairwise L2 distance between generated clips (collapse -> ~0)."""
    flat = clips.flatten(1)
    d = torch.cdist(flat, flat)
    n = len(flat)
    return (d.sum() / (n * (n - 1))).item()


def motion(clips):
    """Mean |frame-to-frame difference| — how much the clips actually move."""
    return (clips[:, :, 1:] - clips[:, :, :-1]).abs().mean().item()


def train(name):
    cfg = CONFIGS[name]
    torch.manual_seed(0)
    torch.set_num_threads(12)
    data = build_dataset()
    print(f"[{name}] {len(data)} real clips")
    G, D = Generator(), Discriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    fixed_z = torch.randn(16, G.nz)
    hist = {"step": [], "diversity": [], "motion": [], "d_real": [],
            "d_fake": []}
    snaps = {}
    t0 = time.time()
    for step in range(STEPS + 1):
        if cfg["starve_at"] is not None and step == cfg["starve_at"]:
            # starve the discriminator: it can no longer keep up with G
            for group in opt_d.param_groups:
                group["lr"] = 1e-6
            print(f"[{name}] step {step}: discriminator starved (lr -> 1e-6)",
                  flush=True)
        if step % SNAP_EVERY == 0:
            with torch.no_grad():
                G.eval()
                fake_eval = G(torch.randn(64, G.nz))
                snaps[step] = G(fixed_z).numpy()
                G.train()
            real_batch = data[torch.randint(len(data), (64,))]
            with torch.no_grad():
                d_real = torch.sigmoid(D(real_batch)).mean().item()
                d_fake = torch.sigmoid(D(fake_eval)).mean().item()
            hist["step"].append(step)
            hist["diversity"].append(diversity(fake_eval))
            hist["motion"].append(motion(fake_eval))
            hist["d_real"].append(d_real)
            hist["d_fake"].append(d_fake)
            print(f"[{name}] step {step:4d}  div {hist['diversity'][-1]:.3f}  "
                  f"motion {hist['motion'][-1]:.4f}  D(real) {d_real:.2f}  "
                  f"D(fake) {d_fake:.2f}  {(time.time()-t0):.0f}s", flush=True)
        if step == STEPS:
            break
        # --- discriminator step ---
        real = data[torch.randint(len(data), (BATCH,))]
        with torch.no_grad():
            fake = G(torch.randn(BATCH, G.nz))
        loss_d = (F.binary_cross_entropy_with_logits(
                      D(real), torch.ones(BATCH)) +
                  F.binary_cross_entropy_with_logits(
                      D(fake), torch.zeros(BATCH)))
        opt_d.zero_grad()
        loss_d.backward()
        opt_d.step()
        # --- generator step ---
        fake = G(torch.randn(BATCH, G.nz))
        loss_g = F.binary_cross_entropy_with_logits(D(fake), torch.ones(BATCH))
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()
    CKPT.mkdir(exist_ok=True)
    np.savez_compressed(CKPT / f"run_{name}.npz",
                        **{f"snap_{k}": v for k, v in snaps.items()},
                        **{k: np.array(v) for k, v in hist.items()})


def sheet_from_clips(clips, out_path, n=8):
    """Show n generated clips as rows of 8 frames each."""
    rows = []
    for clip in clips[:n]:                       # clip: (3, T, 32, 32)
        img = ((clip.transpose(1, 2, 3, 0) + 1) / 2 * 255).clip(0, 255)
        rows += [f.astype(np.uint8) for f in img]
    vid_lib.contact_sheet(rows, cols=clips.shape[2], out_path=out_path,
                          scale=2.0)


def plots():
    runs = {n: np.load(CKPT / f"run_{n}.npz") for n in CONFIGS}
    OUT.mkdir(exist_ok=True)

    # real clips for reference
    data = build_dataset()
    idx = torch.randperm(len(data))[:8]
    sheet_from_clips(data[idx].numpy(), OUT / "real_clips.png")
    # final samples of both runs (same fixed z within each run)
    for name in CONFIGS:
        sheet_from_clips(runs[name][f"snap_{STEPS}"],
                         OUT / f"samples_{name}.png")
    # collapse timeline: first sample of the collapse run over training
    frames = []
    labels = []
    for s in range(0, STEPS + 1, SNAP_EVERY):
        for b in (0, 1, 2):
            clip = runs["collapse"][f"snap_{s}"][b]     # (3, T, 32, 32)
            img = ((clip[:, T_MID].transpose(1, 2, 0) + 1) / 2 * 255)
            frames.append(img.clip(0, 255).astype(np.uint8))
            labels.append(f"step {s}" if b == 0 else "")
    vid_lib.contact_sheet(frames, cols=3, out_path=OUT / "collapse_timeline.png",
                          scale=2.0, labels=labels)

    # diversity + motion curves
    for metric, fname, title, ylab in [
        ("diversity", "diversity.png",
         "Sample diversity during training (higher = more varied clips)",
         "mean pairwise distance between 64 samples"),
        ("motion", "motion.png",
         "Motion in generated clips (higher = more movement)",
         "mean |frame-to-frame| difference"),
    ]:
        fig, ax = ps.new_axes(6.6, 4.0)
        for i, name in enumerate(CONFIGS):
            ax.plot(runs[name]["step"], runs[name][metric],
                    color=ps.SERIES[[0, 2][i]], lw=2, marker="o", ms=4,
                    label=name)
        ref = (motion if metric == "motion" else diversity)(
            data[torch.randint(len(data), (256 if metric == "motion" else 64,))])
        ax.axhline(ref, color=ps.INK_MUTED, ls="--", lw=1.5)
        ax.text(0, ref * 1.03, "real clips", color=ps.INK_MUTED, fontsize=9)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=9)
        ps.finish(fig, ax, title, "training step", ylab, OUT / fname)


T_MID = 4

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.config:
        train(args.config)
    if args.plot:
        plots()
