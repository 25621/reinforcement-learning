"""Project 29 — one set of weights, many shapes.

Sora's headline claim was that a single model generates clips at different
resolutions, aspect ratios and durations.  Two ingredients make that possible,
and this project separates them:

  1. the model must be ABLE to accept a different token grid   -> 3D RoPE
  2. the model must have SEEN different token grids            -> bucket training

Three arms, same architecture and step budget:

  fixed_learned   learned position table, trained on 64x64 only
  fixed_rope      3D RoPE,                trained on 64x64 only
  bucket_rope     3D RoPE,                trained on three shapes

and three test shapes: the trained one, an unseen aspect ratio, and an unseen
duration.

    python3 run.py --stage cache                     # ~2 min
    python3 run.py --stage train --arm fixed_learned # ~7 min
    python3 run.py --stage train --arm fixed_rope    # ~7 min
    python3 run.py --stage train --arm bucket_rope   # ~7 min
    python3 run.py --stage figures                   # ~5 min
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
sys.path.insert(0, str(HERE.parent / "15-inflate-sd-to-a-video-model"))
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
sys.path.insert(0, str(HERE.parent / "23-magvit-v2-style-tokenizer"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
import plot_style as ps                                      # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402

import dit_lib as L                                          # noqa: E402
import flow_lib as FL                                        # noqa: E402
import fid_lib                                               # noqa: E402
import vae3d_lib as V                                        # noqa: E402
import vdm_lib as VDM                                        # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

DIM, DEPTH, HEADS = 128, 5, 4
STEPS, BATCH, LR = 4000, 16, 6e-4
PATCH = (1, 2, 2)
SAMPLE_STEPS = 30
FID_N = 48                 # samples per rFID measurement (see project 25)

# (name, frames, height, width).  The VAE compresses 8x in space and 4x in
# time; the patch then halves height and width again, so tokens =
# (T/4) * (H/16) * (W/16).
BUCKETS = [("square", 16, 64, 64),
           ("wide", 16, 48, 80),
           ("tall", 16, 80, 48)]
TESTS = [("square (trained)", 16, 64, 64),
         ("2:1 portrait (unseen shape)", 16, 96, 48),
         ("double length (unseen duration)", 32, 64, 64)]
ARMS = {
    "fixed_learned": dict(pos="learned", buckets=["square"]),
    "fixed_rope": dict(pos="rope3d", buckets=["square"]),
    "bucket_rope": dict(pos="rope3d", buckets=["square", "wide", "tall"]),
}


def grid_of(T, H, W):
    return ((T - 1) // 4 + 1, H // 16, W // 16)


def build(arm):
    cfg = ARMS[arm]
    return L.VideoDiT(in_ch=4, patch=PATCH, grid=grid_of(16, 64, 64),
                      dim=DIM, depth=DEPTH, heads=HEADS, pos=cfg["pos"],
                      pos_interp=True)


# --------------------------------------------------------------------------
# stage: cache
# --------------------------------------------------------------------------

@torch.no_grad()
def cache():
    """Encode a pool of clips for every bucket.

    Clips of different shapes cannot sit in one tensor, so each bucket gets
    its own pool and each training step draws from ONE bucket.  That is the
    whole idea behind aspect-ratio bucketing (project 47 goes further): you
    cannot batch a 16:9 clip with a 9:16 clip, so you group by shape and
    alternate between groups.
    """
    vae, scale = L.load_vae("3d")
    for name, T, H, W in BUCKETS:
        rng = np.random.default_rng(29)
        lats = []
        for _ in range(1024 // 16):
            x, _, _ = L.attr_batch(rng, 16, T=T, H=H, W=W, train=True)
            lats.append(vae.encode(x)[0] * scale)
        z = torch.cat(lats)
        torch.save({"latents": z}, CK / f"{name}.pt")
        print(f"{name:8s} clip (1,{T},{H},{W}) -> latent "
              f"{tuple(z.shape[1:])} -> grid {grid_of(T, H, W)} = "
              f"{np.prod(grid_of(T, H, W))} tokens  (std {z.std():.3f})")


def load_bucket(name):
    return torch.load(CK / f"{name}.pt", map_location="cpu",
                      weights_only=False)["latents"]


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train(arm):
    torch.manual_seed(0)
    names = ARMS[arm]["buckets"]
    pools = [load_bucket(n) for n in names]
    model = build(arm)
    flow = FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        data = pools[step % len(pools)]          # round-robin over buckets
        idx = torch.randint(0, len(data), (BATCH,), generator=g)
        x0 = data[idx]
        noise = torch.randn(x0.shape, generator=g)
        t = flow.sample_t(BATCH, generator=g)
        loss = F.mse_loss(model(flow.interpolate(x0, t, noise),
                                t * flow.T_SCALE), flow.target(x0, noise))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 1000 == 0:
            print(f"[{arm}] {step:5d}  loss {loss.item():.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "arm": arm,
                "elapsed": time.time() - t0}, CK / f"{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[{arm}] done in {time.time()-t0:.0f}s")


def load_arm(arm):
    ck = torch.load(CK / f"{arm}.pt", map_location="cpu", weights_only=False)
    m = build(arm)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

@torch.no_grad()
def generate(model, T, H, W, n=FID_N, seed=5):
    lat = (4, (T - 1) // 4 + 1, H // 8, W // 8)
    g = torch.Generator().manual_seed(seed)
    z = FL.RectifiedFlow().sample(model, (n, *lat), steps=SAMPLE_STEPS,
                                  generator=g)
    vae, scale = L.load_vae("3d")
    return vae.decoder(z / scale).clamp(-1, 1)


def to64(clips):
    """Resize a clip of any shape to 64x64 so the FID probe can read it.

    The feature network was trained on 64x64 frames.  Both the real and the
    generated set for a given test shape go through the SAME resize, so the
    number compares like with like inside one row — but numbers from
    different rows are not strictly comparable to each other.
    """
    B, C, T, H, W = clips.shape
    x = clips.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    x = F.interpolate(x, size=(64, 64), mode="area")
    return x.view(B, T, C, 64, 64).permute(0, 2, 1, 3, 4).contiguous()


def align(clips):
    return VDM.align_response(clips.permute(0, 2, 1, 3, 4).contiguous())


@torch.no_grad()
def figures():
    net = fid_lib.load_features()
    rows, gallery = [], {}
    for tname, T, H, W in TESTS:
        rng = np.random.default_rng(4242)
        reals, _, _ = L.attr_batch(rng, FID_N, T=T, H=H, W=W, train=False)
        reals2, _, _ = L.attr_batch(rng, FID_N, T=T, H=H, W=W, train=False)
        floor = fid_lib.frechet(to64(reals), to64(reals2), net)
        for arm in ARMS:
            model = load_arm(arm)
            t0 = time.time()
            try:
                fake = generate(model, T, H, W)
                err = ""
            except Exception as e:                # a learned table with no
                fake, err = None, type(e).__name__  # rows for this grid
            row = dict(test=tname, arm=arm,
                       tokens=int(np.prod(grid_of(T, H, W))),
                       fid_proxy=("" if fake is None else
                                  round(fid_lib.frechet(to64(reals),
                                                        to64(fake), net), 2)),
                       align=("" if fake is None else round(align(fake), 3)),
                       flicker=("" if fake is None else
                                round(V.flicker(fake), 4)),
                       real_floor_fid=round(floor, 2),
                       error=err, seconds=round(time.time() - t0, 1))
            rows.append(row)
            print(row, flush=True)
            gallery[(tname, arm)] = fake
        gallery[(tname, "real")] = reals

    with open(OUT / "resolutions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fig_bars(rows)
    fig_gallery(gallery)
    print("wrote", OUT)


def fig_bars(rows):
    fig, ax = ps.new_axes(8.0, 4.4)
    tests = [t[0] for t in TESTS]
    x = np.arange(len(tests))
    width = 0.26
    for i, arm in enumerate(ARMS):
        vals = []
        for t in tests:
            r = [q for q in rows if q["test"] == t and q["arm"] == arm][0]
            vals.append(r["fid_proxy"] if r["fid_proxy"] != "" else 0)
        ax.bar(x + (i - 1) * width, vals, width, color=ps.SERIES[i], label=arm)
    for j, t in enumerate(tests):
        r = [q for q in rows if q["test"] == t][0]
        ax.hlines(r["real_floor_fid"], x[j] - 0.42, x[j] + 0.42,
                  color=ps.INK_MUTED, ls=":", lw=1.2)
    ax.text(x[0] - 0.42, 8, "real vs real", color=ps.INK_MUTED, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([t.replace(" (", "\n(") for t in tests], fontsize=8)
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9)
    ps.finish(fig, ax, "Generating at shapes the model did or did not train on",
              "", "rFID proxy (lower is better)", OUT / "resolutions.png")
    plt.close(fig)


def fig_gallery(gallery):
    rows_ = [("real", "real")] + [(a, a) for a in ARMS]
    fig, axes = plt.subplots(len(rows_), len(TESTS),
                             figsize=(11.0, 1.7 * len(rows_)), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for r, (tag, key) in enumerate(rows_):
        for c, (tname, T, H, W) in enumerate(TESTS):
            ax = axes[r, c]
            clips = gallery[(tname, key)]
            if clips is None:
                ax.text(0.5, 0.5, "cannot run\nat this shape", ha="center",
                        va="center", color=ps.INK_MUTED, fontsize=9)
            else:
                ax.imshow(L.strip(clips[0:1], n=5), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=8)
            if r == 0:
                ax.set_title(tname, color=ps.INK, fontsize=9)
    fig.suptitle("One set of weights, three request shapes", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "gallery.png", facecolor=ps.SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cache", "train", "figures"])
    ap.add_argument("--arm", default="bucket_rope", choices=list(ARMS))
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "cache":
        cache()
    elif args.stage == "train":
        train(args.arm)
    else:
        figures()
