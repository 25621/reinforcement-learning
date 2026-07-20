"""Shared library for the Phase-3 image-to-video (I2V) trilogy.

The honest CPU-scale version of the Stable-Video-Diffusion recipe:

  1. Pretrain a small 2D diffusion U-Net on *single frames* (the stand-in
     for "Stable Diffusion 1.5").
  2. Freeze it, wrap it in a VideoUNet that runs the frozen 2D blocks on
     every frame and inserts *new* temporal layers between them,
     initialized so the wrapper is exactly the frozen image model at
     step 0 (temporal inflation).
  3. Fine-tune only the new layers on clips, with the clean first frame
     fed in as the condition.

Imported by projects 13 (motion control) and 14 (camera trajectory) via
sys.path.  The Moving MNIST generator comes from project 06 and the plot
style from project 01.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "06-moving-mnist-predictor"))
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))

SEED = 0


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Diffusion schedule (DDPM, T=300, betas rescaled from the 1000-step values)
# ---------------------------------------------------------------------------

class Diffusion:
    """Plain DDPM forward/reverse process with epsilon prediction."""

    def __init__(self, timesteps=300):
        self.T = timesteps
        scale = 1000 / timesteps
        betas = torch.linspace(1e-4 * scale, 0.02 * scale, timesteps)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat([torch.ones(1), abar[:-1]])
        self.betas = betas
        self.alphas = alphas
        self.abar = abar
        self.sqrt_abar = abar.sqrt()
        self.sqrt_1m_abar = (1 - abar).sqrt()
        # posterior q(x_{t-1} | x_t, x_0)
        self.post_var = betas * (1 - abar_prev) / (1 - abar)
        self.post_c0 = betas * abar_prev.sqrt() / (1 - abar)
        self.post_ct = (1 - abar_prev) * alphas.sqrt() / (1 - abar)

    def q_sample(self, x0, t, noise):
        """Noise a clean tensor to step t.  t indexes the batch dim."""
        shape = [x0.shape[0]] + [1] * (x0.dim() - 1)
        return (self.sqrt_abar[t].view(shape) * x0
                + self.sqrt_1m_abar[t].view(shape) * noise)

    @torch.no_grad()
    def p_step(self, x_t, t, eps_hat):
        """One ancestral sampling step, via the clamped implied x0.

        Clamping the implied x0 to the data range before computing the
        posterior mean is what keeps the chain from exploding — without
        it a bad epsilon estimate early in sampling implies an x0 far
        outside [-1, 1] and every later step builds on that garbage.
        """
        shape = [x_t.shape[0]] + [1] * (x_t.dim() - 1)
        x0_hat = (x_t - self.sqrt_1m_abar[t].view(shape) * eps_hat) \
            / self.sqrt_abar[t].view(shape)
        x0_hat = x0_hat.clamp(-1, 1)
        mean = (self.post_c0[t].view(shape) * x0_hat
                + self.post_ct[t].view(shape) * x_t)
        if int(t.flatten()[0]) == 0:
            return mean
        return mean + self.post_var[t].view(shape).sqrt() * torch.randn_like(x_t)


# ---------------------------------------------------------------------------
# The 2D image U-Net (the "pretrained image model")
# ---------------------------------------------------------------------------

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.emb = nn.Linear(emb_dim, out_c)
        self.norm2 = nn.GroupNorm(8, out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.skip = (nn.Conv2d(in_c, out_c, 1)
                     if in_c != out_c else nn.Identity())

    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(emb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        attn = torch.softmax(q.transpose(1, 2) @ k / math.sqrt(C), dim=-1)
        out = (v @ attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(out)


class ImageUNet(nn.Module):
    """32x32 single-channel U-Net, ~1.4M params.

    Structured as explicit block lists so VideoUNet can interleave
    temporal layers between the blocks.
    """

    def __init__(self, ch=32, mults=(1, 2, 3), emb_dim=128):
        super().__init__()
        chs = [ch * m for m in mults]                    # 32, 64, 96
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(),
            nn.Linear(emb_dim, emb_dim))
        self.conv_in = nn.Conv2d(1, chs[0], 3, padding=1)

        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        in_c = chs[0]
        for i, c in enumerate(chs):
            self.down.append(ResBlock(in_c, c, emb_dim))
            self.downsample.append(
                nn.Conv2d(c, c, 3, stride=2, padding=1)
                if i < len(chs) - 1 else nn.Identity())
            in_c = c

        self.mid1 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.mid_attn = SelfAttention2D(chs[-1])
        self.mid2 = ResBlock(chs[-1], chs[-1], emb_dim)

        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        in_c = chs[-1]
        for i, c in enumerate(reversed(chs)):
            self.up.append(ResBlock(in_c + c, c, emb_dim))
            self.upsample.append(
                nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)
                if i < len(chs) - 1 else nn.Identity())
            in_c = c

        self.norm_out = nn.GroupNorm(8, chs[0])
        self.conv_out = nn.Conv2d(chs[0], 1, 3, padding=1)
        self.chs = chs

    def forward(self, x, t):
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim))
        h = self.conv_in(x)
        skips = []
        for block, down in zip(self.down, self.downsample):
            h = block(h, emb)
            skips.append(h)
            h = down(h)
        h = self.mid2(self.mid_attn(self.mid1(h, emb)), emb)
        for block, up in zip(self.up, self.upsample):
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
            h = up(h)
        return self.conv_out(F.silu(self.norm_out(h)))


# ---------------------------------------------------------------------------
# Temporal layers and the inflated VideoUNet
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Mixes information across frames at one spatial resolution.

    Two 1D convolutions along the time axis, run independently at every
    spatial position.  The second conv is zero-initialized, so at step 0
    the block is an exact identity: the video model starts life as the
    frozen image model applied to each frame separately.

    An optional FiLM input (`emb`, one vector per clip or per frame)
    scales and shifts the hidden features — this is where projects 13
    and 14 plug in their motion-score / camera conditioning.
    """

    def __init__(self, c, emb_dim=None):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.conv1 = nn.Conv1d(c, c, 3, padding=1)
        self.conv2 = nn.Conv1d(c, c, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.film = (nn.Linear(emb_dim, 2 * c)
                     if emb_dim is not None else None)
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, B, T, emb=None):
        # x: (B*T, C, H, W) -> time-major (B*H*W, C, T)
        BT, C, H, W = x.shape
        h = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)
        h = h.reshape(B * H * W, C, T)
        r = h
        h = self.conv1(F.silu(self.norm(h)))
        if self.film is not None and emb is not None:
            scale, shift = self.film(emb).chunk(2, dim=-1)   # (B,C) or (B,T,C)
            if scale.dim() == 2:                             # per clip
                scale = scale[:, None, None, :, None]        # B 1 1 C 1
                shift = shift[:, None, None, :, None]
            else:                                            # per frame
                scale = scale.permute(0, 2, 1)[:, None, None]  # B 1 1 C T
                shift = shift.permute(0, 2, 1)[:, None, None]
            h = h.view(B, H, W, C, T)
            h = h * (1 + scale) + shift
            h = h.reshape(B * H * W, C, T)
        h = self.conv2(F.silu(h))
        h = r + h
        h = h.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)
        return h.reshape(BT, C, H, W)


class CondAdapter(nn.Module):
    """Injects a conditioning image into the frozen U-Net.

    The frozen conv_in has exactly one input channel (it was trained on
    single frames), so we cannot concatenate the condition as an extra
    channel the way SVD does — there is no weight slot for it.  Instead
    a small trainable conv reads the condition and *adds* its features
    to the frozen features.  Zero-initializing the last conv keeps the
    identity-at-step-0 property.  VideoUNet places one adapter at every
    resolution level (reading a correspondingly downscaled condition),
    ControlNet-style, so deep layers get the condition directly instead
    of hoping the frozen path carries it down for them.
    """

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        return self.conv2(F.silu(self.conv1(x)))


class VideoUNet(nn.Module):
    """The 2D U-Net inflated with temporal layers + I2V conditioning.

    forward(x, t, cond, extra_emb=None, extra_maps=None):
      x          (B, T, 1, H, W)  noisy clip
      t          (B,)             one diffusion timestep per clip
      cond       (B, 1, H, W)     clean first frame
      extra_emb  (B, E) or (B, T, E)   FiLM input for the temporal
                                       blocks (motion score, camera pose)
      extra_maps (B, T, C, H, W)  per-pixel conditioning maps
                                  (Plücker rays), added via an adapter
    """

    def __init__(self, image_unet, freeze_spatial=True,
                 extra_emb_dim=None, extra_map_c=None):
        super().__init__()
        self.unet = image_unet
        if freeze_spatial:
            for p in self.unet.parameters():
                p.requires_grad = False
        chs = image_unet.chs
        e = extra_emb_dim
        self.t_down = nn.ModuleList([TemporalBlock(c, e) for c in chs])
        self.t_mid = TemporalBlock(chs[-1], e)
        self.t_up = nn.ModuleList(
            [TemporalBlock(c, e) for c in reversed(chs)])
        # one condition adapter per resolution level (32, 16, 8)
        self.cond_adapters = nn.ModuleList(
            [CondAdapter(1, c) for c in chs])
        self.map_adapters = (nn.ModuleList(
            [CondAdapter(extra_map_c, c) for c in chs])
            if extra_map_c else None)
        self.extra_emb_dim = e

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x, t, cond, extra_emb=None, extra_maps=None):
        B, T = x.shape[:2]
        u = self.unet
        flat = x.reshape(B * T, *x.shape[2:])
        emb = u.time_mlp(timestep_embedding(
            t.repeat_interleave(T), u.emb_dim))

        # condition (and camera maps) pooled once per resolution level
        conds = [cond]
        for _ in range(len(u.chs) - 1):
            conds.append(F.avg_pool2d(conds[-1], 2))
        if self.map_adapters is not None and extra_maps is not None:
            m = extra_maps.reshape(B * T, *extra_maps.shape[2:])
            maps = [m]
            for _ in range(len(u.chs) - 1):
                maps.append(F.avg_pool2d(maps[-1], 2))

        h = u.conv_in(flat)
        skips = []
        for i, (block, down, tblock) in enumerate(
                zip(u.down, u.downsample, self.t_down)):
            h = block(h, emb)
            h = h + self.cond_adapters[i](conds[i]).repeat_interleave(
                T, dim=0)
            if self.map_adapters is not None and extra_maps is not None:
                h = h + self.map_adapters[i](maps[i])
            h = tblock(h, B, T, extra_emb)
            skips.append(h)
            h = down(h)
        h = u.mid1(h, emb)
        h = self.t_mid(h, B, T, extra_emb)
        h = u.mid2(u.mid_attn(h), emb)
        for block, up, tblock in zip(u.up, u.upsample, self.t_up):
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
            h = tblock(h, B, T, extra_emb)
            h = up(h)
        out = u.conv_out(F.silu(u.norm_out(h)))
        return out.reshape(B, T, *x.shape[2:])


# ---------------------------------------------------------------------------
# Training and sampling helpers
# ---------------------------------------------------------------------------

def to_signed(x):
    """[0,1] -> [-1,1]; diffusion is trained in the signed range."""
    return x * 2 - 1


def to_unit(x):
    return ((x + 1) / 2).clamp(0, 1)


def train_image_model(model, diff, clip_source, steps, lr=3e-4,
                      batch=64, log_every=100):
    """Pretrain the 2D U-Net on random single frames."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        clips = clip_source()                    # (b, T, 1, H, W) in [0,1]
        b, T = clips.shape[:2]
        idx = torch.randint(0, T, (b,))
        frames = to_signed(clips[torch.arange(b), idx])
        while frames.shape[0] < batch:           # top up to batch size
            clips = clip_source()
            idx = torch.randint(0, clips.shape[1], (clips.shape[0],))
            frames = torch.cat(
                [frames, to_signed(clips[torch.arange(clips.shape[0]), idx])])
        frames = frames[:batch]
        t = torch.randint(0, diff.T, (batch,))
        noise = torch.randn_like(frames)
        x_t = diff.q_sample(frames, t, noise)
        loss = F.mse_loss(model(x_t, t), noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0:
            print(f"  image step {step:4d}  loss {loss.item():.4f}",
                  flush=True)
    return losses


def train_video_model(model, diff, batch_source, steps, lr=3e-4,
                      log_every=100):
    """Fine-tune on clips.  batch_source() returns a dict with:
       clips (B,T,1,H,W) in [0,1], and optionally extra_emb / extra_maps.
    """
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=lr)
    losses = []
    for step in range(steps):
        d = batch_source()
        clips = to_signed(d["clips"])
        B = clips.shape[0]
        cond = clips[:, 0]
        t = torch.randint(0, diff.T, (B,))
        noise = torch.randn_like(clips)
        x_t = diff.q_sample(clips, t, noise)
        eps = model(x_t, t, cond,
                    extra_emb=d.get("extra_emb"),
                    extra_maps=d.get("extra_maps"))
        loss = F.mse_loss(eps, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0:
            print(f"  video step {step:4d}  loss {loss.item():.4f}",
                  flush=True)
    return losses


@torch.no_grad()
def sample_clip(model, diff, cond, n_frames, extra_emb=None,
                extra_maps=None, seed=0):
    """Ancestral sampling of a clip conditioned on a first frame.

    cond: (B, 1, H, W) in [0,1].  Returns (B, T, 1, H, W) in [0,1].
    """
    g = torch.Generator().manual_seed(seed)
    B, _, H, W = cond.shape
    cond_s = to_signed(cond)
    x = torch.randn(B, n_frames, 1, H, W, generator=g)
    for step in reversed(range(diff.T)):
        t = torch.full((B,), step, dtype=torch.long)
        eps = model(x, t, cond_s, extra_emb=extra_emb,
                    extra_maps=extra_maps)
        # p_step draws fresh noise from the global RNG; reseed for
        # reproducibility across runs
        x = diff.p_step(x, t, eps)
    return to_unit(x)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def flicker(clips):
    """Mean absolute change between adjacent frames, per clip -> scalar.

    Real Moving-MNIST motion has a characteristic value; frames sampled
    independently (no temporal layers) score several times higher.
    """
    return (clips[:, 1:] - clips[:, :-1]).abs().mean().item()


def cond_fidelity(clips, cond):
    """MSE between the generated first frame and the conditioning frame."""
    return F.mse_loss(clips[:, 0], cond).item()


def strip(clips, every=1, pad=1):
    """(B, T, 1, H, W) -> one uint8 image, one row per clip."""
    x = (clips[:, ::every, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
    B, T, H, W = x.shape
    rows = []
    for b in range(B):
        cells = [np.pad(x[b, t], pad, constant_values=60) for t in range(T)]
        rows.append(np.concatenate(cells, axis=1))
    return np.concatenate(rows, axis=0)
