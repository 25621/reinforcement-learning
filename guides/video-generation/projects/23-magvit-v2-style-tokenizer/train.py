"""Project 23 — a MagViT-v2-style discrete video tokenizer.

Same causal 3D encoder/decoder as project 22; only the bottleneck changes.
Three quantizers, all with a vocabulary of exactly 512 codes:

    vq    a learned codebook of 512 vectors (the VQ-VAE way)
    fsq   3 channels x 8 levels                (codebook-free)
    lfq   9 channels x 2 levels (signs)        (codebook-free, MagViT-v2's)

We report reconstruction quality *and* how much of the vocabulary each one
actually uses, because the second is where the learned codebook gets into
trouble.

Quality is scored with a reconstruction-FID proxy: see `fid_lib.py` for what
is and is not comparable to a published FID number.

    python3 train.py --stage clf       # ~2 min: the feature network for rFID
    python3 train.py --stage vq        # ~7 min
    python3 train.py --stage fsq       # ~7 min
    python3 train.py --stage lfq       # ~7 min
    python3 train.py --stage figures   # ~2 min
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
import plot_style as ps
import matplotlib.pyplot as plt

import vae3d_lib as V
import quant_lib as Q
import fid_lib

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

T_CLIP = 17
STEPS = 600
BATCH = 8
LR = 3e-4
BASE = 16
N_CODES = 512


def make_quantizer(name):
    if name == "vq":
        return Q.VectorQuantizer(n_codes=N_CODES, dim=4), 4
    if name == "fsq":
        q = Q.FSQ(levels=(8, 8, 8))
        return q, q.dim
    if name == "lfq":
        q = Q.LFQ(dim=9)
        return q, q.dim
    raise ValueError(name)


class Tokenizer(nn.Module):
    """Causal 3D encoder -> quantizer -> causal 3D decoder."""

    def __init__(self, name, base=BASE):
        super().__init__()
        self.quant, dim = make_quantizer(name)
        self.name = name
        # Causal encoder, symmetric decoder with the 1 + 4k upsample — the
        # split project 22 arrived at the hard way.
        self.encoder = V.Encoder(base, causal=True, out_ch=dim)
        self.decoder = V.Decoder(base, z_ch=dim, causal=False, causal_up=True)

    def forward(self, x):
        z = self.encoder(x)
        zq, idx, qloss = self.quant(z)
        return self.decoder(zq), idx, qloss

    @torch.no_grad()
    def reconstruct(self, x):
        return self.forward(x)[0]


def src(seed, train=True):
    return V.make_source(seed=seed, seq_len=T_CLIP, train=train)


def train(name):
    torch.manual_seed(0)
    model = Tokenizer(name)
    data = src(1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=5e-5)

    log, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        x = V.clip_batch(data, BATCH)
        rec, idx, qloss = model(x)
        l1 = torch.nn.functional.l1_loss(rec, x)
        loss = l1 + qloss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 20 == 0:
            st = Q.code_stats(idx, model.quant.n_codes)
            log.append((step, float(l1), st["usage"], st["perplexity"]))
        if step % 150 == 0:
            st = Q.code_stats(idx, model.quant.n_codes)
            print(f"[{name}] step {step:4d}  L1 {float(l1):.4f}  "
                  f"usage {st['usage']:.1%}  ppl {st['perplexity']:.0f}  "
                  f"{time.time() - t0:.0f}s", flush=True)

    model.eval()
    torch.save({"state": model.state_dict(), "name": name}, CK / f"{name}.pt")
    np.save(OUT / f"log_{name}.npy", np.array(log))
    print(f"[{name}] done in {time.time() - t0:.0f}s")


def load(name):
    ck = torch.load(CK / f"{name}.pt", map_location="cpu", weights_only=False)
    m = Tokenizer(name)
    m.load_state_dict(ck["state"])
    m.eval()
    return m


@torch.no_grad()
def eval_arm(model, n_batches=6):
    data = src(7, train=False)
    all_idx, psnrs, flick = [], [], []
    reals, fakes = [], []
    for _ in range(n_batches):
        x = V.clip_batch(data, BATCH)
        rec, idx, _ = model(x)
        all_idx.append(idx)
        psnrs.append(V.psnr(x, rec))
        flick.append(V.flicker_error(x, rec))
        reals.append(x)
        fakes.append(rec)
    st = Q.code_stats(torch.cat(all_idx), model.quant.n_codes)
    rfid = fid_lib.frechet(torch.cat(reals), torch.cat(fakes))
    return dict(psnr_db=round(float(np.mean(psnrs)), 2),
                flicker_error=round(float(np.mean(flick)), 4),
                code_usage=round(st["usage"], 4),
                perplexity=round(st["perplexity"], 1),
                rfid_proxy=round(rfid, 2))


ARMS = ["vq", "fsq", "lfq"]
LABEL = {"vq": "VQ (learned codebook)",
         "fsq": "FSQ (3 ch x 8 levels)",
         "lfq": "LFQ (9 ch x sign)"}


def figures():
    rows, models = [], {}
    for name in ARMS:
        m = load(name)
        models[name] = m
        r = dict(arm=name, label=LABEL[name], codes=m.quant.n_codes,
                 **eval_arm(m))
        rows.append(r)
        print(r, flush=True)

    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    figure_usage()
    figure_recon(models)
    figure_hist(models)
    print("wrote", OUT)


def figure_usage():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax in axes:
        ps.style_axes(ax)
    for i, name in enumerate(ARMS):
        a = np.load(OUT / f"log_{name}.npy")
        axes[0].plot(a[:, 0], a[:, 1], color=ps.SERIES[i], lw=1.5,
                     label=LABEL[name])
        axes[1].plot(a[:, 0], a[:, 3], color=ps.SERIES[i], lw=1.5,
                     label=LABEL[name])
    axes[0].set_xlabel("step"), axes[0].set_ylabel("L1 reconstruction loss")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("perplexity (codes used evenly)")
    axes[1].axhline(N_CODES, color=ps.BASELINE, ls="--", lw=1.2)
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Same 512-code budget: quality (left), usage (right)",
                 color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "usage.png")
    plt.close(fig)


@torch.no_grad()
def figure_recon(models):
    data = src(7, train=False)
    x = V.clip_batch(data, 1)
    rows = [("input", x)] + [(LABEL[n], models[n].reconstruct(x))
                             for n in ARMS]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10.0, 1.4 * len(rows)),
                             dpi=110)
    fig.patch.set_facecolor(ps.SURFACE)
    for ax, (tag, clip) in zip(axes, rows):
        ax.imshow(V.strip(clip, n=8), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_ylabel(tag, color=ps.INK_SECONDARY, fontsize=8)
    fig.suptitle("Reconstruction from 512-code discrete tokens", color=ps.INK)
    fig.tight_layout()
    fig.savefig(OUT / "reconstructions.png")
    plt.close(fig)


@torch.no_grad()
def figure_hist(models):
    """How often each of the 512 codes is used, sorted most-used first."""
    fig, ax = ps.new_axes(7.4, 4.2)
    data = src(11, train=False)
    for i, name in enumerate(ARMS):
        idx = []
        for _ in range(6):
            idx.append(models[name](V.clip_batch(data, BATCH))[1])
        counts = torch.bincount(torch.cat(idx), minlength=N_CODES).float()
        counts = counts.sort(descending=True).values / counts.sum()
        ax.plot(counts.numpy(), color=ps.SERIES[i], lw=1.6, label=LABEL[name])
    ax.axhline(1.0 / N_CODES, color=ps.BASELINE, ls="--", lw=1.2,
               label="perfectly even use")
    ax.set_yscale("log")
    ax.set_xlabel("code rank (most used first)")
    ax.set_ylabel("share of all tokens")
    ax.set_title("A flat line means every code earns its keep", color=ps.INK)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "code_histogram.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["clf", "figures"] + ARMS)
    args = ap.parse_args()
    torch.set_num_threads(12)
    if args.stage == "clf":
        fid_lib.train_features()
    elif args.stage == "figures":
        figures()
    else:
        train(args.stage)
