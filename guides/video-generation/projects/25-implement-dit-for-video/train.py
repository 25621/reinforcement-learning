"""Project 25 — a Diffusion Transformer for video, built and ablated.

Stages:

    python3 train.py --stage cache                 # ~1 min  encode clips once
    python3 train.py --stage checks                # ~1 min  three unit tests
    python3 train.py --stage train --pos rope3d    # ~6 min
    python3 train.py --stage train --pos learned   # ~6 min
    python3 train.py --stage train --pos none      # ~6 min
    python3 train.py --stage patchcost             # ~2 min  tokens vs cost
    python3 train.py --stage figures               # ~3 min

The three `--pos` arms are identical models trained on identical data with an
identical budget.  The only difference is how a token learns where it sits in
the (frame, row, column) grid.
"""

import argparse
import csv
import json
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
import plot_style as ps                                      # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402

import dit_lib as L                                          # noqa: E402
import diffusion_lib as D                                    # noqa: E402
import fid_lib                                               # noqa: E402

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

PATCH = (1, 2, 2)          # one frame deep, 2x2 latent cells wide
GRID = (4, 4, 4)           # -> 64 tokens from a (4, 4, 8, 8) latent
DIM, DEPTH, HEADS = 128, 5, 4
STEPS = 4000
BATCH = 16
LR = 6e-4
DIFF_STEPS = 300
SAMPLE_STEPS = 60
FID_N = 96                 # samples per rFID measurement; 24 is too few to
#                            separate the arms from sampling noise
POS_ARMS = ["rope3d", "learned", "none"]


def build(pos):
    return L.VideoDiT(in_ch=4, patch=PATCH, grid=GRID, dim=DIM, depth=DEPTH,
                      heads=HEADS, pos=pos)


# --------------------------------------------------------------------------
# stage: checks — verify the three pieces before spending minutes training
# --------------------------------------------------------------------------

def checks():
    """Cheap property tests.  Each one catches a bug that training hides."""
    torch.manual_seed(0)
    lines = []

    # 1. patchify/unpatchify must be an exact round trip, or the model's
    #    prediction is glued back into the wrong voxels and training still
    #    "works" (loss goes down) while samples are scrambled.
    x = torch.randn(2, 4, 4, 8, 8)
    tok, grid = L.patchify(x, PATCH)
    back = L.unpatchify(tok, PATCH, grid, 4)
    err = (x - back).abs().max().item()
    lines.append(f"patchify round trip: max error {err:.2e} "
                 f"({tok.shape[1]} tokens of {tok.shape[2]} numbers)")
    assert err == 0.0

    # 2. AdaLN-Zero means the whole network is the zero function at init.
    m = build("rope3d")
    out = m(x, torch.zeros(2, dtype=torch.long))
    lines.append(f"AdaLN-Zero output at init: max |out| = "
                 f"{out.abs().max().item():.2e}")
    assert out.abs().max().item() == 0.0

    # 3. RoPE is *relative*: the attention score between two tokens depends
    #    only on the gap between them.  Shift a pair of tokens by the same
    #    amount along the time axis and the score must not move.
    #    The SAME query and key vector is placed at every grid slot, so the
    #    only thing that can change a score is the rotation, i.e. the position.
    hd = DIM // HEADS
    cos, sin = L.rope_3d((4, 4, 4), hd)
    q = torch.randn(1, 1, 1, hd).expand(1, 1, 64, hd).contiguous()
    k = torch.randn(1, 1, 1, hd).expand(1, 1, 64, hd).contiguous()
    qr = L.apply_rope(q, cos, sin)
    kr = L.apply_rope(k, cos, sin)
    idx = lambda t, h, w: (t * 4 + h) * 4 + w          # noqa: E731
    a = (qr[0, 0, idx(0, 1, 1)] * kr[0, 0, idx(1, 1, 1)]).sum()
    b = (qr[0, 0, idx(2, 1, 1)] * kr[0, 0, idx(3, 1, 1)]).sum()
    lines.append(f"RoPE relative-position check: score(t=0->1) {a:.4f} vs "
                 f"score(t=2->3) {b:.4f}, diff {abs(float(a - b)):.2e}")
    assert abs(float(a - b)) < 1e-4

    for m_ in POS_ARMS:
        lines.append(f"params ({m_}): {L.count_params(build(m_)):,}")
    text = "\n".join(lines)
    print(text)
    (OUT / "checks.txt").write_text(text + "\n")


# --------------------------------------------------------------------------
# stage: cache
# --------------------------------------------------------------------------

def cache():
    t0 = time.time()
    tr = L.build_latent_cache(n_clips=1024, seed=25, train=True,
                              name="latents")
    ev = L.build_latent_cache(n_clips=256, seed=99, train=False,
                              name="latents_eval")
    vae, scale = L.load_vae("3d")
    rec = vae.reconstruct(tr["clips"][:32])
    import vae3d_lib as V
    print(f"train {tuple(tr['latents'].shape)}  eval {tuple(ev['latents'].shape)}"
          f"  scale {tr['scale']:.3f}  latent std {tr['latents'].std():.3f}")
    # the VAE was trained on TWO bouncing digits; these clips have ONE digit
    # sliding straight.  Check it still reconstructs them before building a
    # whole phase on top of it.
    print(f"VAE PSNR on Phase-6 clips: "
          f"{V.psnr(tr['clips'][:32], rec):.2f} dB   ({time.time()-t0:.0f}s)")


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def train(pos):
    torch.manual_seed(0)
    data = L.load_latent_cache("latents")["latents"]
    ev = L.load_latent_cache("latents_eval")["latents"]
    model = build(pos)
    ddpm = D.DDPM(steps=DIFF_STEPS)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(data), (BATCH,), generator=g)
        x0 = data[idx]
        t = torch.randint(0, DIFF_STEPS, (BATCH,), generator=g)
        noise = torch.randn(x0.shape, generator=g)
        loss = F.mse_loss(model(ddpm.add_noise(x0, t, noise), t), noise)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0:
            log.append((step, time.time() - t0, loss.item()))
        if step % 500 == 0:
            print(f"[{pos}] {step:5d}  loss {float(loss):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    elapsed = time.time() - t0
    ev_loss = eval_loss(model, ddpm, ev)
    torch.save({"state": model.state_dict(), "pos": pos, "steps": STEPS,
                "elapsed": elapsed, "eval_loss": ev_loss}, CK / f"{pos}.pt")
    np.save(OUT / f"log_{pos}.npy", np.array(log))
    print(f"[{pos}] {STEPS} steps in {elapsed:.0f}s "
          f"({elapsed/STEPS*1000:.0f} ms/step), eval loss {ev_loss:.4f}")


@torch.no_grad()
def eval_loss(model, ddpm, ev, n=192, seed=7):
    """Denoising MSE on held-out clips, averaged over a FIXED set of noises.

    Training loss is a moving target: every step draws a new random timestep,
    so the printed number wobbles by more than the difference between arms.
    Fixing the clips, the timesteps and the noise makes the comparison a
    measurement rather than a coin flip.
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    tot, cnt = 0.0, 0
    for i in range(0, n, 32):
        x0 = ev[i:i + 32]
        t = torch.randint(0, DIFF_STEPS, (len(x0),), generator=g)
        noise = torch.randn(x0.shape, generator=g)
        tot += float(F.mse_loss(model(ddpm.add_noise(x0, t, noise), t),
                                noise)) * len(x0)
        cnt += len(x0)
    model.train()
    return tot / cnt


def load_arm(pos):
    ck = torch.load(CK / f"{pos}.pt", map_location="cpu", weights_only=False)
    m = build(pos)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


@torch.no_grad()
def generate(model, n=16, seed=0):
    ddpm = D.DDPM(steps=DIFF_STEPS)
    g = torch.Generator().manual_seed(seed)
    z = ddpm.sample(model, (n, 4, 4, 8, 8), steps=SAMPLE_STEPS, generator=g)
    vae, scale = L.load_vae("3d")
    return vae.decoder(z / scale).clamp(-1, 1)


# --------------------------------------------------------------------------
# stage: patchcost — the token-count arithmetic, measured
# --------------------------------------------------------------------------

def patchcost():
    """How patch size trades tokens for cost, on the same latent."""
    x = torch.randn(4, 4, 4, 8, 8)
    rows = []
    for patch in [(1, 1, 1), (1, 2, 2), (2, 2, 2), (1, 4, 4), (4, 4, 4)]:
        grid = (4 // patch[0], 8 // patch[1], 8 // patch[2])
        n = grid[0] * grid[1] * grid[2]
        m = L.VideoDiT(in_ch=4, patch=patch, grid=grid, dim=DIM, depth=DEPTH,
                       heads=HEADS, pos="rope3d")
        t = torch.zeros(4, dtype=torch.long)
        m(x, t)                                   # warm up
        t0 = time.time()
        for _ in range(3):
            m(x, t)
        ms = (time.time() - t0) / 3 * 1000
        rows.append(dict(patch="x".join(map(str, patch)), tokens=n,
                         numbers_per_token=m.pdim,
                         params=L.count_params(m),
                         forward_ms=round(ms, 1)))
        print(rows[-1], flush=True)
    with open(OUT / "patch_cost.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

@torch.no_grad()
def order_probe(model, ev, n=64, seed=3):
    """Does the model know which frame is which?

    Take a held-out clip, shuffle its latent frames, and ask the model to
    denoise both versions.  A model that understands time should find the
    shuffled clip *harder* (it is not a plausible video any more).  A model
    with no position information cannot tell the two apart at all, so its
    two scores must come out equal — which is exactly what makes this a
    direct test of the positional embedding rather than of overall quality.
    """
    ddpm = D.DDPM(steps=DIFF_STEPS)
    g = torch.Generator().manual_seed(seed)
    x0 = ev[:n]
    perm = torch.tensor([2, 0, 3, 1])
    xs = x0[:, :, perm]
    t = torch.randint(0, DIFF_STEPS // 2, (n,), generator=g)   # low noise
    noise = torch.randn(x0.shape, generator=g)
    a = float(F.mse_loss(model(ddpm.add_noise(x0, t, noise), t), noise))
    b = float(F.mse_loss(model(ddpm.add_noise(xs, t, noise), t), noise))
    return a, b


@torch.no_grad()
def figures():
    ev_cache = L.load_latent_cache("latents_eval")
    ev = ev_cache["latents"]
    reals = ev_cache["clips"][:FID_N]
    net = fid_lib.load_features()

    rows, samples = [], {}
    for pos in POS_ARMS:
        model, ck = load_arm(pos)
        fake = generate(model, n=FID_N)
        samples[pos] = fake
        real_o, shuf_o = order_probe(model, ev)
        rows.append(dict(
            pos=pos,
            params=L.count_params(model),
            ms_per_step=round(ck["elapsed"] / ck["steps"] * 1000, 1),
            eval_loss=round(ck["eval_loss"], 5),
            fid_proxy=round(fid_lib.frechet(reals, fake, net), 2),
            order_real=round(real_o, 5),
            order_shuffled=round(shuf_o, 5),
            order_gap_pct=round((shuf_o - real_o) / real_o * 100, 1),
        ))
        print(rows[-1], flush=True)

    rows.append(dict(pos="real clips (floor)", params="", ms_per_step="",
                     eval_loss="",
                     fid_proxy=round(fid_lib.frechet(
                         reals, ev_cache["clips"][FID_N:2 * FID_N], net), 2),
                     order_real="", order_shuffled="", order_gap_pct=""))
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig_loss()
    fig_samples(samples, reals)
    fig_order(rows[:-1])
    print("wrote", OUT)


def fig_loss():
    fig, ax = ps.new_axes(7.4, 4.2)
    for i, pos in enumerate(POS_ARMS):
        a = np.load(OUT / f"log_{pos}.npy")
        k = 8
        sm = np.convolve(a[:, 2], np.ones(k) / k, mode="valid")
        ax.plot(a[k - 1:, 0], sm, color=ps.SERIES[i], lw=1.5, label=pos)
    ax.set_yscale("log")
    ax.legend(frameon=False)
    ps.finish(fig, ax, "Same DiT, three ways of telling tokens where they are",
              "training step", "denoising MSE (smoothed)",
              OUT / "loss_curves.png")
    plt.close(fig)


def fig_samples(samples, reals):
    rows = [("real", reals)] + [(p, samples[p]) for p in POS_ARMS]
    fig, axes = plt.subplots(len(rows) * 2, 1,
                             figsize=(10.0, 1.25 * len(rows) * 2), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    k = 0
    for tag, clips in rows:
        for j in range(2):
            ax = axes[k]
            ax.imshow(L.strip(clips[j:j + 1], n=8), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=9)
            k += 1
    fig.suptitle("Generated clips, 8 of 16 frames", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "samples.png", facecolor=ps.SURFACE)
    plt.close(fig)


def fig_order(rows):
    fig, ax = ps.new_axes(7.0, 4.0)
    x = np.arange(len(rows))
    ax.bar(x - 0.2, [r["order_real"] for r in rows], 0.4,
           color=ps.SERIES[0], label="clip in real frame order")
    ax.bar(x + 0.2, [r["order_shuffled"] for r in rows], 0.4,
           color=ps.SERIES[2], label="same clip, frames shuffled")
    ax.set_xticks(x)
    ax.set_xticklabels([r["pos"] for r in rows])
    ax.legend(frameon=False)
    ps.finish(fig, ax, "Can the model tell a shuffled clip from a real one?",
              "positional embedding", "denoising MSE (lower = easier)",
              OUT / "order_probe.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["cache", "checks", "train", "patchcost",
                             "figures"])
    ap.add_argument("--pos", default="rope3d", choices=POS_ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "cache":
        cache()
    elif args.stage == "checks":
        checks()
    elif args.stage == "train":
        train(args.pos)
    elif args.stage == "patchcost":
        patchcost()
    else:
        figures()
