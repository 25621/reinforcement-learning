"""Shared library for the Phase-4 video-diffusion projects (15-18).

Phase 3's `i2v_lib.py` built an *image-to-video* model: a frozen 2D U-Net
inflated with temporal convolutions, always fed a clean first frame.
Phase 4 graduates to *full video diffusion* — generating an entire clip
from pure noise — so this library extends the Phase-3 pieces with:

  * `TemporalAttention` — attention along the time axis (the second kind
    of temporal layer, next to the temporal convolution),
  * `VideoDiffusionUNet` — an inflated U-Net with no mandatory first-frame
    input, optional class conditioning (project 17) and optional extra
    input channels (projects 17/18),
  * `widen_conv_in` — the SVD-style trick of widening a pretrained input
    convolution with zero-initialized channels for new conditioning,
  * a DDIM sampler (fast, strided) next to full ancestral sampling,
  * video metrics: flicker (from i2v_lib) and a phase-correlation
    "alignment response" that separates real motion from teleporting.

Imported by projects 16 (joint image-video training), 17 (temporal CFG
study) and 18 (cascaded super-resolution) via sys.path.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "12-tiny-i2v-model"))
sys.path.insert(0, str(HERE.parent / "06-moving-mnist-predictor"))
sys.path.insert(0, str(HERE.parent / "01-video-loader-benchmark"))

from i2v_lib import (  # noqa: E402  (re-exported for the Phase-4 projects)
    Diffusion, ImageUNet, TemporalBlock, timestep_embedding,
    to_signed, to_unit, strip, flicker, set_seed, train_image_model,
)

T_FRAMES = 8         # clip length used across Phase 4
CANVAS = 32


# ---------------------------------------------------------------------------
# Temporal attention
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    """Attention along the time axis, run at every spatial position.

    The temporal *convolution* mixes a fixed 3-frame window; attention
    lets every frame look at every other frame directly, however far
    apart.  Features are reshaped so each spatial position becomes an
    independent length-T sequence.  A learned temporal position
    embedding tells attention *which* frame is which (attention alone is
    order-blind).  The output projection is zero-initialized, so at
    step 0 the block is an exact identity — same trick as the temporal
    convolution, same reason: never garble the pretrained features.
    """

    def __init__(self, c, n_heads=4, max_t=32):
        super().__init__()
        self.norm = nn.LayerNorm(c)
        self.pos = nn.Parameter(torch.zeros(max_t, c))
        self.attn = nn.MultiheadAttention(c, n_heads, batch_first=True)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x, B, T, emb=None):
        # x: (B*T, C, H, W) -> sequences (B*H*W, T, C)
        BT, C, H, W = x.shape
        h = x.view(B, T, C, H, W).permute(0, 3, 4, 1, 2)
        h = h.reshape(B * H * W, T, C)
        q = self.norm(h) + self.pos[:T]
        h = h + self.attn(q, q, q, need_weights=False)[0]
        h = h.view(B, H, W, T, C).permute(0, 3, 4, 1, 2)
        return h.reshape(BT, C, H, W)


def widen_conv_in(unet, extra_ch):
    """Widen a pretrained 1-channel conv_in to accept extra channels.

    This is what SVD does to feed the conditioning frame in: build a new
    first convolution with more input channels, copy the pretrained
    weights into the old channel slots, and zero-initialize the new
    slots.  At step 0 the extra channels contribute nothing, so the
    pretrained behavior is untouched — the network then *learns* to read
    them during fine-tuning.  (Phase 3's project 12 could not do this
    because it froze the backbone; here the fine-tunes are unfrozen.)
    """
    old = unet.conv_in
    new = nn.Conv2d(old.in_channels + extra_ch, old.out_channels,
                    3, padding=1)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :old.in_channels] = old.weight
        new.bias.copy_(old.bias)
    unet.conv_in = new
    return unet


# ---------------------------------------------------------------------------
# The inflated video-diffusion U-Net
# ---------------------------------------------------------------------------

class VideoDiffusionUNet(nn.Module):
    """A 2D U-Net inflated into a video model that samples clips from noise.

    Runs the wrapped 2D U-Net on every frame and inserts temporal layers
    (convolution and/or attention, per the `temporal` flag) after every
    spatial block.  All temporal layers start as exact identities.

    forward(x, t, cls=None, x_extra=None):
      x        (B, T, 1, H, W)   noisy clip
      t        (B,)              one diffusion timestep per clip
      cls      (B,) long         optional class ids (n_classes = null id)
      x_extra  (B, T, C, H, W)   optional extra input channels, concat to x
                                 (the wrapped unet's conv_in must have been
                                 widened to expect them)
    """

    def __init__(self, image_unet, freeze_spatial=True,
                 temporal=("conv", "attn"), n_classes=None):
        super().__init__()
        self.unet = image_unet
        if freeze_spatial:
            for p in self.unet.parameters():
                p.requires_grad = False
        chs = image_unet.chs

        def stack(c):
            mods = nn.ModuleList()
            if "conv" in temporal:
                mods.append(TemporalBlock(c))
            if "attn" in temporal:
                mods.append(TemporalAttention(c))
            return mods

        self.t_down = nn.ModuleList([stack(c) for c in chs])
        self.t_mid = stack(chs[-1])
        self.t_up = nn.ModuleList([stack(c) for c in reversed(chs)])
        if n_classes is not None:
            # last index = the learned "null" class for CFG dropout.
            # This embedding is ADDED to the timestep embedding below,
            # and the timestep embedding has a large, fixed norm (~6) —
            # a zero (or small-random) init leaves the class term too
            # faint to compete for gradient signal, and nothing in
            # training pushes its scale up (measured: it stays ~30x
            # smaller than the timestep term even after 1000+ steps,
            # and the class dial ends up turning nothing). Unlike the
            # temporal layers, this path has no identity-at-init
            # requirement to protect — it is new conditioning, not a
            # frozen pretrained behavior — so it is initialized at a
            # scale already comparable to what it will be added to.
            self.class_emb = nn.Embedding(n_classes + 1,
                                          image_unet.emb_dim)
            nn.init.normal_(self.class_emb.weight,
                            std=2.0 / image_unet.emb_dim ** 0.5)
        else:
            self.class_emb = None

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _temporal(self, mods, h, B, T):
        for m in mods:
            if isinstance(m, TemporalBlock):
                h = m(h, B, T)
            else:
                h = m(h, B, T)
        return h

    def forward(self, x, t, cls=None, x_extra=None):
        B, T = x.shape[:2]
        u = self.unet
        if x_extra is not None:
            x = torch.cat([x, x_extra], dim=2)
        flat = x.reshape(B * T, *x.shape[2:])
        emb = u.time_mlp(timestep_embedding(
            t.repeat_interleave(T), u.emb_dim))
        if self.class_emb is not None and cls is not None:
            emb = emb + self.class_emb(cls).repeat_interleave(T, dim=0)

        h = u.conv_in(flat)
        skips = []
        for i, (block, down) in enumerate(zip(u.down, u.downsample)):
            h = block(h, emb)
            h = self._temporal(self.t_down[i], h, B, T)
            skips.append(h)
            h = down(h)
        h = u.mid1(h, emb)
        h = self._temporal(self.t_mid, h, B, T)
        h = u.mid2(u.mid_attn(h), emb)
        for block, up, mods in zip(u.up, u.upsample, self.t_up):
            h = block(torch.cat([h, skips.pop()], dim=1), emb)
            h = self._temporal(mods, h, B, T)
            h = up(h)
        out = u.conv_out(F.silu(u.norm_out(h)))
        return out.reshape(B, T, 1, *x.shape[3:])


# ---------------------------------------------------------------------------
# Training and sampling
# ---------------------------------------------------------------------------

def train_video(model, diff, batch_fn, steps, lr=3e-4, log_every=100):
    """Generic video-diffusion training loop.

    batch_fn() returns a dict with "clips" (B,T,1,H,W) in [0,1] and
    optionally "cls" (B,) and "x_extra" (B,T,C,H,W) already in [-1,1].
    """
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=lr)
    losses = []
    for step in range(steps):
        d = batch_fn()
        clips = to_signed(d["clips"])
        B = clips.shape[0]
        t = torch.randint(0, diff.T, (B,))
        noise = torch.randn_like(clips)
        x_t = diff.q_sample(clips, t, noise)
        eps = model(x_t, t, cls=d.get("cls"), x_extra=d.get("x_extra"))
        loss = F.mse_loss(eps, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % log_every == 0:
            print(f"  step {step:4d}  loss {loss.item():.4f}", flush=True)
    return losses


@torch.no_grad()
def ancestral_sample(eps_fn, diff, shape, seed=0):
    """Full T-step ancestral sampling.  eps_fn(x, t) -> predicted noise."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g)
    B = shape[0]
    for step in reversed(range(diff.T)):
        t = torch.full((B,), step, dtype=torch.long)
        x = diff.p_step(x, t, eps_fn(x, t))
    return to_unit(x)


@torch.no_grad()
def ddim_sample(eps_fn, diff, shape, steps=60, seed=0):
    """Strided deterministic DDIM sampling.

    DDIM revisits only `steps` of the T noise levels and takes a
    deterministic jump between consecutive visited levels (no fresh
    noise is added).  ~T/steps times cheaper than ancestral sampling at
    a small quality cost — which is what makes the guidance-grid sweeps
    of project 17 affordable.  Same x0-clamping trick as `p_step`.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, generator=g)
    B = shape[0]
    ts = torch.linspace(0, diff.T - 1, steps).round().long().flip(0)
    for i, step in enumerate(ts):
        t = torch.full((B,), int(step), dtype=torch.long)
        eps = eps_fn(x, t)
        shape1 = [B] + [1] * (x.dim() - 1)
        x0 = (x - diff.sqrt_1m_abar[t].view(shape1) * eps) \
            / diff.sqrt_abar[t].view(shape1)
        x0 = x0.clamp(-1, 1)
        if i == len(ts) - 1:
            x = x0
        else:
            tp = ts[i + 1]
            x = (diff.sqrt_abar[tp] * x0
                 + diff.sqrt_1m_abar[tp] * eps)
    return to_unit(x)


# ---------------------------------------------------------------------------
# Metrics and figure helpers
# ---------------------------------------------------------------------------

def align_response(clips):
    """Mean phase-correlation peak response between adjacent frames.

    cv2.phaseCorrelate finds the translation that best aligns two
    frames and reports the strength of that alignment peak (0..1).
    Real motion = the next frame is roughly a *shifted copy* of this
    one = strong peak.  Teleporting or morphing content = no shift
    explains the change = weak peak.  Complements `flicker`, which
    only measures how *much* pixels change, not whether the change
    looks like motion.
    """
    import cv2
    vals = []
    arr = clips[:, :, 0].numpy().astype(np.float64)
    for b in range(arr.shape[0]):
        for t in range(arr.shape[1] - 1):
            _, resp = cv2.phaseCorrelate(arr[b, t], arr[b, t + 1])
            vals.append(resp)
    return float(np.mean(vals))


def sharpness(frames):
    """Mean gradient magnitude of a batch of frames (B,1,H,W) in [0,1]."""
    gx = (frames[..., :, 1:] - frames[..., :, :-1]).abs().mean()
    gy = (frames[..., 1:, :] - frames[..., :-1, :]).abs().mean()
    return float(gx + gy) / 2


def save_gif(path, clip, scale=4, fps=5):
    """clip (T,1,H,W) in [0,1] -> animated gif, nearest-neighbor upscaled."""
    from PIL import Image
    frames = []
    for t in range(clip.shape[0]):
        a = (clip[t, 0].clamp(0, 1).numpy() * 255).astype(np.uint8)
        im = Image.fromarray(a).resize(
            (a.shape[1] * scale, a.shape[0] * scale), Image.NEAREST)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    print(f"wrote {path}")


def save_strip(path, clips, every=1, pad=1, scale=3):
    """Save an i2v_lib strip (one row per clip) as a PNG."""
    from PIL import Image
    img = strip(clips, every=every, pad=pad)
    im = Image.fromarray(img).resize(
        (img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    im.save(path)
    print(f"wrote {path}")
