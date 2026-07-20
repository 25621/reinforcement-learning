"""Project 19 — (2+1)D vs full vs windowed spatiotemporal attention.

One small video diffusion *transformer* (patchified clips, epsilon
prediction), trained three times with identical data, depth, width and
step budget — only the attention pattern differs:

  factorized  (2+1)D: attention within each frame, then attention along
              time at each spatial position (two cheap attentions)
  full        joint attention over all T*S tokens (one expensive one)
  window      joint attention inside 4x4x4 space-time windows, shifted
              by half a window in alternating blocks (Swin-style) so
              information can cross window borders

Stages:
  python3 train.py --stage train --arm factorized|full|window
  python3 train.py --stage figures     FLOP curves, eval loss, samples
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "15-inflate-sd-to-a-video-model"))
import vdm_lib as V
from mmnist import MovingMNIST
import plot_style as ps

CK = HERE / "checkpoints"
OUT = HERE / "outputs"
CK.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

T = V.T_FRAMES        # 8 frames
PATCH = 4             # 4x4 pixel patches -> 8x8 = 64 tokens per frame
GRID = V.CANVAS // PATCH
S = GRID * GRID       # 64 spatial tokens
DIM = 96
HEADS = 4
DEPTH = 4
STEPS = 2200
BATCH = 8
LR = 1e-3
WIN = (4, 4, 4)       # window: 4 frames x 4x4 spatial tokens = 64 tokens
ARMS = ["factorized", "full", "window"]


class Attention(nn.Module):
    """Minimal multi-head self-attention over (N, L, D) sequences."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        N, L, D = x.shape
        q, k, v = self.qkv(x).reshape(N, L, 3, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(N, L, D))


class Block(nn.Module):
    """One transformer block with DiT-style AdaLN-Zero conditioning.

    The timestep embedding produces per-block shift/scale/gate vectors
    (zero-initialized, so every block starts as an identity).  This is
    the detail that makes small diffusion transformers train: a single
    input-level timestep addition is too weak, and the model then fails
    exactly at low noise levels, where knowing t precisely matters most.
    """

    def __init__(self, mode, shift=False):
        super().__init__()
        self.mode = mode
        self.shift = shift
        self.n1 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.attn = Attention(DIM, HEADS)
        self.n_mod = 6
        # factorized needs a second attention for the temporal step
        if mode == "factorized":
            self.n1b = nn.LayerNorm(DIM, elementwise_affine=False)
            self.attn_t = Attention(DIM, HEADS)
            self.n_mod = 9
        self.n2 = nn.LayerNorm(DIM, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(),
                                 nn.Linear(4 * DIM, DIM))
        self.ada = nn.Linear(DIM, self.n_mod * DIM)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def _mix(self, y):
        """Apply this block's attention pattern to (B,T,S,D) tokens."""
        B = y.shape[0]
        if self.mode == "full":
            return self.attn(y.reshape(B, T * S, DIM)).view(B, T, S, DIM)
        if self.mode == "factorized":                        # spatial half
            h = self.attn(y.reshape(B * T, S, DIM))
            return h.view(B, T, S, DIM)
        wt, wh, ww = WIN                                     # window
        g = y.view(B, T, GRID, GRID, DIM)
        if self.shift:
            g = torch.roll(g, (wt // 2, wh // 2, ww // 2), (1, 2, 3))
        nt, nh, nw = T // wt, GRID // wh, GRID // ww
        w = g.view(B, nt, wt, nh, wh, nw, ww, DIM) \
             .permute(0, 1, 3, 5, 2, 4, 6, 7) \
             .reshape(B * nt * nh * nw, wt * wh * ww, DIM)
        w = self.attn(w)
        g = w.view(B, nt, nh, nw, wt, wh, ww, DIM) \
             .permute(0, 1, 4, 2, 5, 3, 6, 7) \
             .reshape(B, T, GRID, GRID, DIM)
        if self.shift:
            g = torch.roll(g, (-wt // 2, -wh // 2, -ww // 2), (1, 2, 3))
        return g.view(B, T, S, DIM)

    def forward(self, x, c):
        # x: (B, T, S, D), c: (B, D) timestep embedding
        B = x.shape[0]
        mods = self.ada(F.silu(c))[:, None, None, :].chunk(self.n_mod, -1)
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = mods[:6]
        x = x + g_a * self._mix(self.n1(x) * (1 + sc_a) + sh_a)
        if self.mode == "factorized":                        # temporal half
            sh_t, sc_t, g_t = mods[6:]
            y = self.n1b(x) * (1 + sc_t) + sh_t
            h = y.transpose(1, 2).reshape(B * S, T, DIM)
            h = self.attn_t(h).view(B, S, T, DIM).transpose(1, 2)
            x = x + g_t * h
        return x + g_m * self.mlp(self.n2(x) * (1 + sc_m) + sh_m)


class VideoTransformer(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.patch = nn.Conv2d(1, DIM, PATCH, stride=PATCH)
        self.pos_s = nn.Parameter(torch.randn(S, DIM) * 0.02)
        self.pos_t = nn.Parameter(torch.randn(T, DIM) * 0.02)
        self.emb = nn.Sequential(nn.Linear(128, DIM), nn.SiLU(),
                                 nn.Linear(DIM, DIM))
        self.blocks = nn.ModuleList(
            [Block(mode, shift=(mode == "window" and i % 2 == 1))
             for i in range(DEPTH)])
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Sequential(nn.Linear(DIM, DIM), nn.GELU(),
                                  nn.Linear(DIM, PATCH * PATCH))

    def forward(self, x, t, cls=None, x_extra=None):
        B = x.shape[0]
        h = self.patch(x.reshape(B * T, 1, V.CANVAS, V.CANVAS))
        h = h.flatten(2).transpose(1, 2).view(B, T, S, DIM)
        h = h + self.pos_s + self.pos_t[None, :, None]
        c = self.emb(V.timestep_embedding(t, 128))           # (B, D)
        for blk in self.blocks:
            h = blk(h, c)
        h = self.head(self.norm(h))                          # (B,T,S,P*P)
        h = h.view(B * T, GRID, GRID, PATCH, PATCH) \
             .permute(0, 1, 3, 2, 4) \
             .reshape(B, T, 1, V.CANVAS, V.CANVAS)
        return h

    def trainable_parameters(self):
        return list(self.parameters())


# ---------------------------------------------------------------------------
# Analytic attention FLOPs (multiplies only, per layer, per clip)
# ---------------------------------------------------------------------------

def attn_flops(mode, t, s=S, d=DIM, win=WIN):
    """FLOPs of the attention *scores + mix* (QK^T and AV) per layer.

    The projection cost (QKV + output, 8*N*d^2) is identical across
    patterns and grows linearly in token count — excluded here on
    purpose so the curve isolates what the *pattern* changes.
    """
    if mode == "full":
        n = t * s
        return 2 * n * n * d
    if mode == "factorized":
        return 2 * (t * s * s * d + s * t * t * d)
    wt, wh, ww = win
    w = wt * wh * ww
    n = t * s
    return 2 * n * w * d


def stage_train(arm):
    V.set_seed()
    diff = V.Diffusion()
    mm = MovingMNIST(n_digits=2, seq_len=T, seed=1)
    model = VideoTransformer(arm)
    print(f"[{arm}] params "
          f"{sum(p.numel() for p in model.parameters()):,}")
    t0 = time.time()
    losses = V.train_video(model, diff,
                           lambda: {"clips": mm.batch(BATCH)},
                           STEPS, lr=LR)
    dt = (time.time() - t0) / STEPS
    torch.save(model.state_dict(), CK / f"{arm}.pt")
    np.save(CK / f"loss_{arm}.npy", np.array(losses))
    (CK / f"time_{arm}.txt").write_text(f"{dt:.4f}")
    print(f"[{arm}] {dt:.3f} s/step")


@torch.no_grad()
def eval_loss(model, diff, clips, seed=0):
    """Held-out epsilon MSE with identical noise for every arm."""
    torch.manual_seed(seed)
    x0 = V.to_signed(clips)
    total = 0
    for r in range(6):
        t = torch.randint(0, diff.T, (x0.shape[0],))
        noise = torch.randn_like(x0)
        total += F.mse_loss(model(diff.q_sample(x0, t, noise), t),
                            noise).item()
    return total / 6


def stage_figures():
    V.set_seed()
    diff = V.Diffusion()
    mm = MovingMNIST(n_digits=2, seq_len=T, seed=99)
    held = mm.batch(48)

    rows = []
    samples = {}
    measured = {}
    for arm in ARMS:
        model = VideoTransformer(arm)
        model.load_state_dict(torch.load(CK / f"{arm}.pt",
                                         weights_only=True))
        model.eval()
        el = eval_loss(model, diff, held)
        # measured FLOPs of one forward (batch of 1 clip)
        from torch.utils.flop_counter import FlopCounterMode
        x = torch.randn(1, T, 1, V.CANVAS, V.CANVAS)
        with FlopCounterMode(display=False) as fc:
            model(x, torch.tensor([100]))
        fl = fc.get_total_flops()
        measured[arm] = fl
        dt = float((CK / f"time_{arm}.txt").read_text())
        n_par = sum(p.numel() for p in model.parameters())
        samples[arm] = V.ancestral_sample(lambda x, t: model(x, t), diff,
                                          (8, T, 1, V.CANVAS, V.CANVAS),
                                          seed=7)
        rows.append((arm, n_par, el, dt, fl,
                     V.flicker(samples[arm]),
                     V.align_response(samples[arm])))
        print(f"{arm:11s} params {n_par:,}  eval-loss {el:.4f}  "
              f"{dt:.3f} s/step  fwd GFLOPs {fl/1e9:.2f}", flush=True)

    real = mm.batch(8)
    with open(OUT / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "params", "held_out_eps_mse", "s_per_step",
                    "measured_fwd_flops", "flicker", "align_response"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.4f}", f"{r[3]:.3f}",
                        r[4], f"{r[5]:.4f}", f"{r[6]:.4f}"])
        w.writerow(["real clips", "", "", "", "",
                    f"{V.flicker(real):.4f}",
                    f"{V.align_response(real):.4f}"])

    # ---- the scaling curve: attention FLOPs vs clip length ---------------
    ts = np.array([4, 8, 16, 32, 64, 128])
    fig, ax = ps.new_axes()
    for i, arm in enumerate(ARMS):
        ax.plot(ts, [attn_flops(arm, int(t)) * DEPTH / 1e9 for t in ts],
                color=ps.SERIES[i], marker="o", markersize=4,
                label={"factorized": "(2+1)D factorized",
                       "full": "full spatiotemporal",
                       "window": "windowed (4x4x4, shifted)"}[arm])
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, labelcolor=ps.INK_SECONDARY)
    ps.finish(fig, ax,
              "Attention cost vs clip length (64 spatial tokens/frame)",
              "frames T", "attention GFLOPs per clip (log)",
              OUT / "flops_vs_t.png")

    # ---- samples ----------------------------------------------------------
    V.save_strip(OUT / "samples.png",
                 torch.cat([real[:2]] + [samples[a][:2] for a in ARMS]))

    fig, ax = ps.new_axes(6.6, 3.4)
    y = np.arange(len(ARMS))[::-1]
    ax.barh(y, [r[2] for r in rows], height=0.55,
            color=[ps.SERIES[i] for i in range(len(ARMS))])
    ax.set_yticks(y, [r[0] for r in rows])
    for yi, r in zip(y, rows):
        ax.text(r[2] + 0.0005, yi, f"{r[2]:.4f}", va="center",
                fontsize=9, color=ps.INK_SECONDARY)
    ps.finish(fig, ax, "Held-out epsilon MSE at equal step budget",
              "MSE (lower is better)", "", OUT / "eval_loss.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["train", "figures"])
    ap.add_argument("--arm", choices=ARMS)
    a = ap.parse_args()
    torch.set_num_threads(12)
    if a.stage == "train":
        stage_train(a.arm)
    else:
        stage_figures()
