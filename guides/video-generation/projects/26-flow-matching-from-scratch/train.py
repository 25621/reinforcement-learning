"""Project 26 — the same video DiT, trained twice: DDPM vs rectified flow.

Everything is held fixed except the objective: same architecture (project 25's
`VideoDiT` with 3D RoPE), same cached latents, same optimiser, same 6000 steps.

    python3 train.py --stage train --arm ddpm    # ~6 min
    python3 train.py --stage train --arm flow    # ~6 min
    python3 train.py --stage figures             # ~4 min

The headline question is NOT "which one has the lower loss" — the two losses
measure different things and cannot be compared (see the README).  It is
"which one still produces a clip when you only let it take 4 sampling steps".
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
sys.path.insert(0, str(HERE.parent / "24-diffusion-on-latents"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
import plot_style as ps                                      # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402

import dit_lib as L                                          # noqa: E402
import diffusion_lib as D                                    # noqa: E402
import fid_lib                                               # noqa: E402
import flow_lib as FL                                        # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

P25 = HERE.parent / "25-implement-dit-for-video"
PATCH, GRID = (1, 2, 2), (4, 4, 4)
DIM, DEPTH, HEADS = 128, 5, 4
STEPS, BATCH, LR = 4000, 16, 6e-4
DIFF_STEPS = 300
MILESTONES = [500, 1000, 2000, 4000]
SWEEP = [1, 2, 4, 8, 16, 32, 60]
FID_N = 96                 # samples per rFID measurement.  24 was not enough:
#                            the score wobbled by more than the effect being
#                            measured, which is a measurement problem, not a
#                            result (see the README's "how noisy is the ruler"
#                            section).
ARMS = ["ddpm", "flow"]


def build():
    return L.VideoDiT(in_ch=4, patch=PATCH, grid=GRID, dim=DIM, depth=DEPTH,
                      heads=HEADS, pos="rope3d")


def train(arm):
    torch.manual_seed(0)
    data = L.load_latent_cache("latents", where=P25 / "checkpoints")["latents"]
    model = build()
    ddpm = D.DDPM(steps=DIFF_STEPS)
    flow = FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(data), (BATCH,), generator=g)
        x0 = data[idx]
        noise = torch.randn(x0.shape, generator=g)
        if arm == "ddpm":
            t = torch.randint(0, DIFF_STEPS, (BATCH,), generator=g)
            pred, tgt = model(ddpm.add_noise(x0, t, noise), t), noise
        else:
            t = flow.sample_t(BATCH, generator=g)
            xt = flow.interpolate(x0, t, noise)
            pred = model(xt, t * flow.T_SCALE)
            tgt = flow.target(x0, noise)
        loss = F.mse_loss(pred, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step in MILESTONES:
            torch.save({"state": model.state_dict()},
                       CK / f"{arm}_{step}.pt")
        if step % 1000 == 0:
            print(f"[{arm}] {step:5d}  loss {float(loss):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    torch.save({"state": model.state_dict(), "arm": arm, "steps": STEPS,
                "elapsed": elapsed}, CK / f"{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[{arm}] {STEPS} steps in {elapsed:.0f}s "
          f"({elapsed/STEPS*1000:.0f} ms/step)")


def load_arm(arm, step=None):
    name = f"{arm}.pt" if step is None else f"{arm}_{step}.pt"
    ck = torch.load(CK / name, map_location="cpu", weights_only=False)
    m = build()
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


@torch.no_grad()
def sample_latents(arm, model, n, steps, seed=0, traj=False):
    g = torch.Generator().manual_seed(seed)
    shape = (n, 4, 4, 8, 8)
    if arm == "flow":
        return FL.RectifiedFlow().sample(model, shape, steps=steps,
                                         generator=g, return_traj=traj)
    ddpm = D.DDPM(steps=DIFF_STEPS)
    if traj:
        return FL.ddim_trajectory(ddpm, model, shape, steps=steps, generator=g)
    return ddpm.sample(model, shape, steps=steps, generator=g)


@torch.no_grad()
def decode(z):
    vae, scale = L.load_vae("3d")
    return vae.decoder(z / scale).clamp(-1, 1)


@torch.no_grad()
def figures():
    ev = L.load_latent_cache("latents_eval", where=P25 / "checkpoints")
    reals = ev["clips"][:FID_N]
    net = fid_lib.load_features()
    floor = fid_lib.frechet(reals, ev["clips"][FID_N:2 * FID_N], net)

    # How noisy is the ruler?  Same checkpoint, same everything, three
    # different starting noises.  Any difference smaller than this spread is
    # not a finding.
    noise_rows = []
    model, _ = load_arm("flow")
    for sd in (5, 6, 7):
        z = sample_latents("flow", model, FID_N, 60, seed=sd)
        noise_rows.append(dict(seed=sd,
                               fid_proxy=round(fid_lib.frechet(
                                   reals, decode(z), net), 2)))
        print("metric noise", noise_rows[-1], flush=True)

    # ---- 1. quality vs number of sampling steps -------------------------
    sweep_rows, sample_grid = [], {}
    for arm in ARMS:
        model, ck = load_arm(arm)
        for s in SWEEP:
            t0 = time.time()
            z = sample_latents(arm, model, FID_N, s, seed=5)
            fid = fid_lib.frechet(reals, decode(z), net)
            sweep_rows.append(dict(arm=arm, sampling_steps=s,
                                   fid_proxy=round(fid, 2),
                                   seconds=round(time.time() - t0, 2)))
            print(sweep_rows[-1], flush=True)
            if s in (4, 60):
                sample_grid[(arm, s)] = decode(z)[:2]

    # ---- 2. how straight is each sampler's path? ------------------------
    straight_rows = []
    for arm in ARMS:
        model, _ = load_arm(arm)
        _, traj = sample_latents(arm, model, 8, 60, seed=5, traj=True)
        straight_rows.append(dict(arm=arm,
                                  straightness=round(FL.straightness(traj), 4)))
        print(straight_rows[-1], flush=True)

    # ---- 3. convergence on a metric both arms share ---------------------
    conv_rows = []
    for arm in ARMS:
        for ms in MILESTONES:
            model, _ = load_arm(arm, ms)
            z = sample_latents(arm, model, FID_N, 60, seed=5)
            conv_rows.append(dict(arm=arm, step=ms,
                                  fid_proxy=round(
                                      fid_lib.frechet(reals, decode(z), net),
                                      2)))
            print(conv_rows[-1], flush=True)

    with open(OUT / "step_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep_rows[0]))
        w.writeheader()
        w.writerows(sweep_rows)
    with open(OUT / "convergence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(conv_rows[0]))
        w.writeheader()
        w.writerows(conv_rows)
    with open(OUT / "straightness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(straight_rows[0]))
        w.writeheader()
        w.writerows(straight_rows)
    with open(OUT / "metric_noise.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(noise_rows[0]))
        w.writeheader()
        w.writerows(noise_rows)

    fig_sweep(sweep_rows, floor)
    fig_convergence(conv_rows, floor)
    fig_samples(sample_grid, reals)
    fig_losses()
    print(f"real-vs-real floor: {floor:.2f}\nwrote {OUT}")


def fig_sweep(rows, floor):
    fig, ax = ps.new_axes(7.4, 4.2)
    for i, arm in enumerate(ARMS):
        r = [x for x in rows if x["arm"] == arm]
        ax.plot([x["sampling_steps"] for x in r],
                [x["fid_proxy"] for x in r], "o-", color=ps.SERIES[i],
                lw=1.6, label={"ddpm": "DDPM (DDIM sampler)",
                               "flow": "rectified flow (Euler sampler)"}[arm])
    ax.axhline(floor, color=ps.INK_MUTED, ls=":", lw=1.2)
    ax.text(1.1, floor * 1.06, "real vs real", color=ps.INK_MUTED, fontsize=8)
    ax.set_xscale("log")
    ax.set_xticks(SWEEP)
    ax.set_xticklabels([str(s) for s in SWEEP])
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ps.finish(fig, ax, "The payoff: quality when you cut the sampling budget",
              "sampling steps", "rFID proxy (lower is better)",
              OUT / "step_sweep.png")
    plt.close(fig)


def fig_convergence(rows, floor):
    fig, ax = ps.new_axes(7.4, 4.2)
    for i, arm in enumerate(ARMS):
        r = [x for x in rows if x["arm"] == arm]
        ax.plot([x["step"] for x in r], [x["fid_proxy"] for x in r], "o-",
                color=ps.SERIES[i], lw=1.6, label=arm)
    ax.axhline(floor, color=ps.INK_MUTED, ls=":", lw=1.2)
    ax.legend(frameon=False)
    ps.finish(fig, ax, "Convergence, measured on a yardstick both objectives share",
              "training step", "rFID proxy at 60 sampling steps",
              OUT / "convergence.png")
    plt.close(fig)


def fig_samples(grid, reals):
    rows = [("real", reals[:2])]
    for arm in ARMS:
        for s in (60, 4):
            rows.append((f"{arm}, {s} steps", grid[(arm, s)]))
    fig, axes = plt.subplots(len(rows) * 2, 1,
                             figsize=(10.0, 1.15 * len(rows) * 2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    k = 0
    for tag, clips in rows:
        for j in range(2):
            ax = axes[k]
            ax.imshow(L.strip(clips[j:j + 1], n=8), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=8)
            k += 1
    fig.suptitle("Same model quality, very different behaviour at 4 steps",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", facecolor=ps.SURFACE)
    plt.close(fig)


def fig_losses():
    """Plotted side by side ONLY to show they are on different scales."""
    fig, ax = ps.new_axes(7.4, 4.2)
    for i, arm in enumerate(ARMS):
        a = np.load(OUT / f"log_{arm}.npy")
        k = 12
        sm = np.convolve(a[:, 2], np.ones(k) / k, mode="valid")
        ax.plot(a[k - 1:, 0], sm, color=ps.SERIES[i], lw=1.5,
                label=f"{arm} (target: "
                      f"{'noise' if arm == 'ddpm' else 'velocity'})")
    ax.legend(frameon=False)
    ps.finish(fig, ax,
              "Two different targets — these curves are NOT comparable",
              "training step", "MSE against its own target",
              OUT / "loss_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "figures"])
    ap.add_argument("--arm", default="flow", choices=ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "train":
        train(args.arm)
    else:
        figures()
