"""Project 27 — read OpenSora's real configuration, then reproduce its shape
in miniature and swap one component out.

Three things happen here:

  `--stage anatomy`   downloads the *config files* of the real OpenSora v1.2
                      release (a few kilobytes, no weights) and lines its
                      numbers up against our mini replica, including the
                      arithmetic that explains why the real one cannot run on
                      this machine.
  `--stage cache2d`   encodes the training clips with the per-frame 2D VAE
                      instead of the 3D one — the component swap.
  `--stage train --vae 3d|2d`   trains the identical DiT on each.
  `--stage figures`   scores both.

The swap is controlled: both VAEs compress a clip to exactly 1024 numbers and
both produce exactly 64 tokens of 16 numbers for the DiT.  The only difference
is WHERE the compression happened — across time, or inside each frame.

    python3 run.py --stage anatomy               # ~1 min (needs network)
    python3 run.py --stage cache2d               # ~1 min
    python3 run.py --stage train --vae 3d        # ~6 min
    python3 run.py --stage train --vae 2d        # ~6 min
    python3 run.py --stage figures               # ~3 min
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
P25 = HERE.parent / "25-implement-dit-for-video" / "checkpoints"

DIM, DEPTH, HEADS = 128, 5, 4
STEPS, BATCH, LR = 4000, 16, 6e-4
ARMS = {                     # patch size chosen so BOTH arms give 64 tokens
    "3d": dict(in_ch=4, patch=(1, 2, 2), shape=(4, 4, 8, 8)),
    "2d": dict(in_ch=1, patch=(4, 2, 2), shape=(1, 16, 8, 8)),
}
GRID = (4, 4, 4)
FID_N = 96                 # see project 25: 24 samples is inside the noise

# The real thing, for the anatomy table.  A 4-second 720p clip at 24 fps.
REAL_CLIP = dict(frames=102, height=720, width=1280,
                 vae_space=8, vae_time=4, latent_ch=4)


# --------------------------------------------------------------------------
# stage: anatomy
# --------------------------------------------------------------------------

def _fetch(repo, fname="config.json"):
    from huggingface_hub import hf_hub_download
    return json.load(open(hf_hub_download(repo, fname)))


def transformer_flops(n_tokens, dim, depth):
    """Rough forward FLOPs of a DiT: the two matmul families that dominate.

    Per block: the linear projections cost ~12*N*D^2 multiply-adds (q, k, v,
    output projection, and the 4x-wide MLP's two layers), and the attention
    score/aggregate pair costs ~2*N^2*D.  Two FLOPs per multiply-add.  Norms,
    activations and AdaLN are ignored — at these sizes they are noise.

    The point of the formula is the SHAPE of it: the first term grows linearly
    with token count, the second grows with its square.  Doubling a video's
    resolution quadruples the tokens, so it costs 4x through the first term
    and 16x through the second.
    """
    linear = 12 * n_tokens * dim * dim
    attn = 2 * n_tokens * n_tokens * dim
    return 2 * depth * (linear + attn)


def anatomy():
    rows = []
    try:
        stdit = _fetch("hpcai-tech/OpenSora-STDiT-v3")
        vae = _fetch("hpcai-tech/OpenSora-VAE-v1.2")
        source = "downloaded from the Hugging Face Hub"
    except Exception as e:                       # offline fallback
        print(f"(could not reach the Hub: {e}) — using published values")
        stdit = dict(depth=28, hidden_size=1152, num_heads=16,
                     patch_size=[1, 2, 2], in_channels=4,
                     caption_channels=4096, model_max_length=300)
        vae = dict(scale=[3.85, 2.32, 2.33, 3.06], micro_frame_size=17)
        source = "published config values (no network)"
    print(f"OpenSora v1.2 config, {source}")

    # --- token arithmetic for a real 4-second 720p generation --------------
    r = REAL_CLIP
    lt = (r["frames"] - 1) // r["vae_time"] + 1
    lh, lw = r["height"] // r["vae_space"], r["width"] // r["vae_space"]
    pt, ph, pw = stdit["patch_size"]
    real_tokens = (lt // pt) * (lh // ph) * (lw // pw)
    real_flops = transformer_flops(real_tokens, stdit["hidden_size"],
                                   stdit["depth"])

    mini_tokens = GRID[0] * GRID[1] * GRID[2]
    mini_flops = transformer_flops(mini_tokens, DIM, DEPTH)

    model = L.VideoDiT(in_ch=4, patch=(1, 2, 2), grid=GRID, dim=DIM,
                       depth=DEPTH, heads=HEADS, pos="rope3d")
    x = torch.randn(1, 4, 4, 8, 8)
    t = torch.zeros(1, dtype=torch.long)
    model(x, t)
    t0 = time.time()
    for _ in range(5):
        model(x, t)
    mini_ms = (time.time() - t0) / 5 * 1000
    flops_per_s = mini_flops / (mini_ms / 1000)

    fields = [
        ("transformer blocks", stdit["depth"], DEPTH),
        ("hidden size", stdit["hidden_size"], DIM),
        ("attention heads", stdit["num_heads"], HEADS),
        ("patch size (t,h,w)", "x".join(map(str, stdit["patch_size"])),
         "1x2x2"),
        ("latent channels", stdit["in_channels"], 4),
        ("text embedding width", stdit.get("caption_channels", 4096),
         "0 (project 28 adds text)"),
        ("tokens per clip", f"{real_tokens:,}", mini_tokens),
        ("forward GFLOPs", round(real_flops / 1e9), round(mini_flops / 1e9, 3)),
    ]
    for name, real, mini in fields:
        rows.append(dict(quantity=name, opensora_v1_2=real, our_mini_dit=mini))
        print(f"  {name:<24} {str(real):>16}   {str(mini):>10}")

    est_s = real_flops / flops_per_s
    print(f"\nmeasured on this CPU: {mini_ms:.1f} ms per mini forward "
          f"= {flops_per_s/1e9:.1f} GFLOP/s")
    print(f"one OpenSora forward at that rate: {est_s:.0f} s "
          f"= {est_s/60:.1f} min")
    print(f"a 30-step generation: {est_s*30/3600:.1f} hours "
          f"(and that ignores the 1.1B parameters not fitting in cache)")
    rows.append(dict(quantity="seconds per forward on this CPU (estimated)",
                     opensora_v1_2=round(est_s), our_mini_dit=round(mini_ms / 1000, 4)))
    rows.append(dict(quantity="VAE latent scale factors",
                     opensora_v1_2=str(vae.get("scale")),
                     our_mini_dit="1 measured scalar (project 24)"))

    with open(OUT / "anatomy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["quantity", "opensora_v1_2",
                                          "our_mini_dit"])
        w.writeheader()
        w.writerows(rows)
    fig_tokens()


def fig_tokens():
    """Tokens, and therefore cost, as a function of what you ask for."""
    fig, ax = ps.new_axes(7.4, 4.2)
    presets = [("our clip\n16x64x64", 16, 64, 64),
               ("240p\n2s", 48, 240, 426),
               ("480p\n4s", 102, 480, 854),
               ("720p\n4s", 102, 720, 1280),
               ("1080p\n8s", 204, 1080, 1920)]
    names, toks, flops = [], [], []
    for name, f_, h, w in presets:
        lt = (f_ - 1) // 4 + 1
        n = lt * (h // 16) * (w // 16)          # 8x VAE then 2x2 patch
        names.append(name)
        toks.append(n)
        flops.append(transformer_flops(n, 1152, 28) / 1e12)
    ax.bar(names, flops, color=ps.SERIES[0])
    for i, (n, fl) in enumerate(zip(toks, flops)):
        ax.text(i, fl * 1.1, f"{n:,}\ntokens", ha="center", fontsize=8,
                color=ps.INK_SECONDARY)
    ax.set_yscale("log")
    ps.finish(fig, ax, "One forward pass of OpenSora's backbone, by request size",
              "", "TFLOPs (log scale)", OUT / "token_cost.png")
    plt.close(fig)


# --------------------------------------------------------------------------
# stage: cache2d — the component swap
# --------------------------------------------------------------------------

def cache2d():
    for name, seed, train, n in [("latents2d", 25, True, 1024),
                                 ("latents2d_eval", 99, False, 256)]:
        c = L.build_latent_cache(n_clips=n, seed=seed, train=train,
                                 vae_kind="2d", name=name, where=CK)
        print(f"{name}: {tuple(c['latents'].shape)}  "
              f"scale {c['scale']:.3f}  std {c['latents'].std():.3f}")
    vae, _ = L.load_vae("2d")
    clips = L.load_latent_cache("latents", where=P25)["clips"][:32]
    print(f"2D VAE PSNR on Phase-6 clips: "
          f"{V.psnr(clips, vae.reconstruct(clips)):.2f} dB")


# --------------------------------------------------------------------------
# stage: train
# --------------------------------------------------------------------------

def build(arm):
    cfg = ARMS[arm]
    return L.VideoDiT(in_ch=cfg["in_ch"], patch=cfg["patch"], grid=GRID,
                      dim=DIM, depth=DEPTH, heads=HEADS, pos="rope3d")


def load_cache(arm, split=""):
    if arm == "3d":
        return L.load_latent_cache("latents" + split, where=P25)
    return L.load_latent_cache("latents2d" + split, where=CK)


def train(arm):
    torch.manual_seed(0)
    data = load_cache(arm)["latents"]
    model = build(arm)
    flow = FL.RectifiedFlow()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(1)
    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
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
            print(f"[vae={arm}] {step:5d}  loss {float(loss):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
    torch.save({"state": model.state_dict(), "arm": arm,
                "elapsed": time.time() - t0}, CK / f"dit_{arm}.pt")
    np.save(OUT / f"log_{arm}.npy", np.array(log))
    print(f"[vae={arm}] done in {time.time()-t0:.0f}s")


# --------------------------------------------------------------------------
# stage: figures
# --------------------------------------------------------------------------

@torch.no_grad()
def generate(arm, n=FID_N, steps=30, seed=5):
    ck = torch.load(CK / f"dit_{arm}.pt", map_location="cpu",
                    weights_only=False)
    model = build(arm)
    model.load_state_dict(ck["state"])
    model.eval()
    g = torch.Generator().manual_seed(seed)
    z = FL.RectifiedFlow().sample(model, (n, *ARMS[arm]["shape"]),
                                  steps=steps, generator=g)
    vae, scale = L.load_vae(arm)
    return vae.decoder(z / scale).clamp(-1, 1)


def align(clips):
    """vdm_lib wants (B, T, C, H, W); our clips are (B, C, T, H, W)."""
    return VDM.align_response(clips.permute(0, 2, 1, 3, 4).contiguous())


@torch.no_grad()
def figures():
    ev = L.load_latent_cache("latents_eval", where=P25)
    reals = ev["clips"][:FID_N]
    net = fid_lib.load_features()
    rows, samples = [], {}
    for arm in ARMS:
        fake = generate(arm)
        samples[arm] = fake
        vae, _ = L.load_vae(arm)
        rec = vae.reconstruct(reals)
        rows.append(dict(
            vae=arm,
            latent_shape="x".join(map(str, ARMS[arm]["shape"])),
            numbers_per_clip=int(np.prod(ARMS[arm]["shape"])),
            tokens=GRID[0] * GRID[1] * GRID[2],
            vae_psnr_db=round(V.psnr(reals, rec), 2),
            vae_floor_fid=round(fid_lib.frechet(reals, rec, net), 2),
            sample_fid=round(fid_lib.frechet(reals, fake, net), 2),
            sample_flicker=round(V.flicker(fake), 4),
            sample_align=round(align(fake), 3),
        ))
        print(rows[-1], flush=True)
    rows.append(dict(vae="real clips", latent_shape="", numbers_per_clip="",
                     tokens="", vae_psnr_db="", vae_floor_fid="",
                     sample_fid=round(fid_lib.frechet(
                         reals, ev["clips"][FID_N:2 * FID_N], net), 2),
                     sample_flicker=round(V.flicker(reals), 4),
                     sample_align=round(align(reals), 3)))
    print(rows[-1])
    with open(OUT / "swap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, axes = plt.subplots(6, 1, figsize=(10.0, 8.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    k = 0
    for tag, clips in [("real", reals), ("3D VAE", samples["3d"]),
                       ("per-frame 2D VAE", samples["2d"])]:
        for j in range(2):
            ax = axes[k]
            ax.imshow(L.strip(clips[j:j + 1], n=8), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=9)
            k += 1
    fig.suptitle("Same DiT, same token budget — only the VAE was swapped",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "swap_samples.png", facecolor=ps.SURFACE)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["anatomy", "cache2d", "train", "figures"])
    ap.add_argument("--vae", default="3d", choices=list(ARMS))
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "anatomy":
        anatomy()
    elif args.stage == "cache2d":
        cache2d()
    elif args.stage == "train":
        train(args.vae)
    else:
        figures()
