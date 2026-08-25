"""The shared Phase-6 backbone: a video Diffusion Transformer (DiT).

This one file is imported by projects 25-29.  It contains four things:

  * `attr_batch`  — labelled Moving-MNIST clips (which digit, which direction),
                    so later projects can ask "did the model obey the prompt?"
  * `patchify` / `unpatchify` — cut a 3D VAE latent into spatiotemporal patches
                    and glue the predictions back into a latent
  * `rope_3d`     — 3D rotary position embedding, built fresh for whatever
                    (T, H, W) grid it is handed (this is what makes project 29's
                    variable-resolution trick possible)
  * `VideoDiT`    — patch embed -> N transformer blocks with AdaLN-Zero -> head

Why a transformer at all, when projects 15-18 already had a working U-Net?
A U-Net's convolutions are welded to a fixed grid shape, and its receptive
field grows only as fast as you stack layers.  A transformer takes a *list* of
tokens: any number of them, in any arrangement, every one able to look at every
other in a single layer.  That is precisely what "one model, many resolutions
and durations" needs — see project 29.
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
sys.path.insert(0, str(HERE.parent / "06-moving-mnist-predictor"))
import vae3d_lib as V                                        # noqa: E402
import mmnist                                                # noqa: E402

T_FRAMES = 16          # pixel frames per clip
CANVAS = 64
DIGIT_PX = 28          # vae3d_lib already set these module globals
DIRECTIONS = ["right", "down", "left", "up"]
DIR_VEC = {"right": (0.0, 1.0), "down": (1.0, 0.0),
           "left": (0.0, -1.0), "up": (-1.0, 0.0)}           # (dy, dx)


# --------------------------------------------------------------------------
# data: clips that come with a truthful label
# --------------------------------------------------------------------------

_DIGIT_CACHE = {}


def _digit_sprites(train=True):
    """MNIST digits resized to DIGIT_PX, grouped by class."""
    key = bool(train)
    if key not in _DIGIT_CACHE:
        old = mmnist.DIGIT
        mmnist.DIGIT = DIGIT_PX
        imgs, labels = mmnist._load_digits(train)
        mmnist.DIGIT = old
        _DIGIT_CACHE[key] = (imgs, labels)
    return _DIGIT_CACHE[key]


def attr_batch(rng, batch, T=T_FRAMES, H=CANVAS, W=CANVAS,
               digits=None, dirs=None, speed=(1.4, 2.2), train=True):
    """Clips of ONE digit sliding in ONE straight cardinal direction.

    Project 06's Moving MNIST bounces off the walls, which is great for
    learning dynamics but useless as a *label*: a clip whose digit bounces
    halfway through is neither "moving left" nor "moving right".  Here the
    start position is chosen so the digit never reaches a wall, so the label
    "digit 3 moving left" is true of every frame, not just on average.  A
    conditioning experiment is only as trustworthy as its labels.

    Returns clips `(B, 1, T, H, W)` in [-1, 1] plus integer label tensors.
    """
    sprites, labels = _digit_sprites(train)
    D = DIGIT_PX
    clips = np.zeros((batch, T, H, W), dtype=np.float32)
    dig = np.zeros(batch, dtype=np.int64)
    dir_id = np.zeros(batch, dtype=np.int64)
    for b in range(batch):
        want = digits[b] if digits is not None else int(rng.integers(10))
        pool = np.nonzero(labels == want)[0]
        sprite = sprites[pool[rng.integers(len(pool))]]
        d = int(dirs[b]) if dirs is not None else int(rng.integers(4))
        dy, dx = DIR_VEC[DIRECTIONS[d]]
        sp = rng.uniform(*speed)
        travel = sp * (T - 1)
        lim_y, lim_x = H - D, W - D
        # start far enough from the wall it is heading for that it never bounces
        y0 = rng.uniform(0, max(lim_y - travel, 0)) if dy > 0 else \
            rng.uniform(min(travel, lim_y), lim_y) if dy < 0 else \
            rng.uniform(0, lim_y)
        x0 = rng.uniform(0, max(lim_x - travel, 0)) if dx > 0 else \
            rng.uniform(min(travel, lim_x), lim_x) if dx < 0 else \
            rng.uniform(0, lim_x)
        for t in range(T):
            y = int(round(np.clip(y0 + dy * sp * t, 0, lim_y)))
            x = int(round(np.clip(x0 + dx * sp * t, 0, lim_x)))
            np.maximum(clips[b, t, y:y + D, x:x + D], sprite,
                       out=clips[b, t, y:y + D, x:x + D])
        dig[b], dir_id[b] = want, d
    x = torch.from_numpy(clips).unsqueeze(1)                 # (B,1,T,H,W)
    return x * 2.0 - 1.0, torch.from_numpy(dig), torch.from_numpy(dir_id)


def centroid_drift(clips):
    """Net (dy, dx) travelled by the bright pixels, in pixels per clip.

    Used to check a *generated* clip against the direction it was asked for.
    The digit is the only bright thing on the canvas, so the intensity-weighted
    centre of mass tracks it without needing a detector.
    """
    x = (clips.clamp(-1, 1) + 1) / 2                          # (B,1,T,H,W)
    x = x.squeeze(1)
    B, T, H, W = x.shape
    ys = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
    mass = x.sum((2, 3)).clamp(min=1e-4)                       # (B,T)
    cy = (x * ys).sum((2, 3)) / mass
    cx = (x * xs).sum((2, 3)) / mass
    return torch.stack([cy[:, -1] - cy[:, 0], cx[:, -1] - cx[:, 0]], -1)


def predicted_direction(clips, min_travel=3.0):
    """Direction id implied by a clip's centroid drift; -1 if it barely moved."""
    d = centroid_drift(clips)
    out = torch.full((len(d),), -1, dtype=torch.long)
    for i in range(len(d)):
        dy, dx = float(d[i, 0]), float(d[i, 1])
        if abs(dy) + abs(dx) < min_travel:
            continue                      # too still to claim any direction
        if abs(dx) >= abs(dy):
            out[i] = 0 if dx > 0 else 2   # right / left
        else:
            out[i] = 1 if dy > 0 else 3   # down / up
    return out


# --------------------------------------------------------------------------
# the frozen VAE that every Phase-6 project generates inside
# --------------------------------------------------------------------------

def load_vae(kind="3d"):
    """Project 21's trained VAE, frozen.  Returns (model, scale_factor)."""
    ck_path = (HERE.parent / "21-train-a-small-3d-vae" / "checkpoints"
               / f"{kind}.pt")
    if not ck_path.exists():
        raise SystemExit(
            f"missing {ck_path} — run project 21 first:\n"
            f"  cd ../21-train-a-small-3d-vae && python3 train.py --stage {kind}")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    vae = V.VideoVAE(base=16, z_ch=(4 if kind == "3d" else 1),
                     temporal=(kind == "3d"))
    vae.load_state_dict(ck["state"])
    vae.eval().requires_grad_(False)
    return vae, ck["scale"]


# --------------------------------------------------------------------------
# patchification
# --------------------------------------------------------------------------

def patchify(x, patch):
    """(B, C, T, H, W) latent -> (B, N, C*pt*ph*pw) tokens.

    A "patch" in image DiT is a small square of pixels flattened into one
    vector.  For video the same idea gains a third side: a little BOX that
    spans `pt` frames as well as `ph` x `pw` positions.  One token therefore
    carries a scrap of appearance *and* a scrap of motion, which is why a
    single attention layer can reason about both at once.
    """
    pt, ph, pw = patch
    B, C, T, H, W = x.shape
    assert T % pt == 0 and H % ph == 0 and W % pw == 0, (x.shape, patch)
    nt, nh, nw = T // pt, H // ph, W // pw
    x = x.view(B, C, nt, pt, nh, ph, nw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)            # (B, nt, nh, nw, C,pt,ph,pw)
    return x.reshape(B, nt * nh * nw, C * pt * ph * pw), (nt, nh, nw)


def unpatchify(tokens, patch, grid, C):
    """The exact inverse of `patchify`, used on the model's prediction."""
    pt, ph, pw = patch
    nt, nh, nw = grid
    B = tokens.shape[0]
    x = tokens.view(B, nt, nh, nw, C, pt, ph, pw)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return x.reshape(B, C, nt * pt, nh * ph, nw * pw)


# --------------------------------------------------------------------------
# 3D rotary position embedding
# --------------------------------------------------------------------------

def _axis_angles(pos, n_pairs, theta=100.0):
    """Angles for one axis: (len(pos), n_pairs)."""
    if n_pairs == 0:
        return torch.zeros(len(pos), 0)
    k = torch.arange(n_pairs, dtype=torch.float32)
    freqs = theta ** (-k / max(n_pairs, 1))
    return pos.float()[:, None] * freqs[None, :]


def rope_3d(grid, head_dim, split=None, theta=100.0):
    """Cosines and sines for 3D RoPE on a (nt, nh, nw) token grid.

    RoPE = *Rotary* Position Embedding: instead of ADDING a position vector to
    a token, it ROTATES the token's query and key vectors by an angle
    proportional to the position.  The dot product of two rotated vectors
    depends only on the DIFFERENCE of their angles, so attention automatically
    sees "how far apart are these two tokens?" rather than "where are they in
    absolute terms".  That relative view is what still makes sense at a
    sequence length the model never trained on.

    "3D" here just means the head's channels are split into three groups —
    one rotated by the frame index, one by the row, one by the column — so a
    single token's rotation encodes all three coordinates at once.

    Returns cos, sin of shape (N, head_dim // 2), where N = nt*nh*nw.
    """
    nt, nh, nw = grid
    pairs = head_dim // 2
    if split is None:                       # time gets fewer channels: a video
        n_t = pairs // 4                    # latent has far fewer frames than
        n_h = (pairs - n_t) // 2            # rows or columns
        n_w = pairs - n_t - n_h
        split = (n_t, n_h, n_w)
    n_t, n_h, n_w = split
    at = _axis_angles(torch.arange(nt), n_t, theta)          # (nt, n_t)
    ah = _axis_angles(torch.arange(nh), n_h, theta)
    aw = _axis_angles(torch.arange(nw), n_w, theta)
    ang = torch.cat([
        at[:, None, None, :].expand(nt, nh, nw, n_t),
        ah[None, :, None, :].expand(nt, nh, nw, n_h),
        aw[None, None, :, :].expand(nt, nh, nw, n_w),
    ], dim=-1).reshape(nt * nh * nw, pairs)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """Rotate every pair of channels of x: (B, heads, N, head_dim)."""
    B, H, N, D = x.shape
    x = x.view(B, H, N, D // 2, 2)
    c = cos.view(1, 1, N, D // 2)
    s = sin.view(1, 1, N, D // 2)
    a, b = x[..., 0], x[..., 1]
    return torch.stack([a * c - b * s, a * s + b * c], -1).reshape(B, H, N, D)


# --------------------------------------------------------------------------
# the transformer
# --------------------------------------------------------------------------

def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32) / half)
    ang = t.float()[:, None] * freqs[None].to(t.device)
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with optional rotary positions."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class DiTBlock(nn.Module):
    """Attention + MLP, both modulated by AdaLN-Zero.

    AdaLN-Zero, in three words: *adaptive layer norm, zero-initialised*.
    The timestep vector is turned into a shift and a scale for each LayerNorm
    (that is the "adaptive" part) plus a `gate` multiplying each sub-layer's
    output.  All three are produced by one Linear whose weights start at zero,
    so at step 0 every gate is 0 and the block computes exactly `x -> x`.

    Why bother, when we could simply add the timestep vector to the tokens
    once at the input?  Project 19 tried that: the loss went down and the
    samples were pure noise.  A single addition at the input has to survive
    every layer intact, and the network happily learns to ignore it; the
    damage shows up at LOW noise levels, where knowing t precisely is the
    difference between polishing an image and destroying it.  AdaLN-Zero
    instead re-injects t inside every block, as a multiplicative control on
    the normalisation — much harder to ignore.
    """

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, rope=None):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = \
            self.ada(F.silu(c))[:, None, :].chunk(6, dim=-1)
        x = x + g_a * self.attn(self.n1(x) * (1 + sc_a) + sh_a, rope)
        return x + g_m * self.mlp(self.n2(x) * (1 + sc_m) + sh_m)


class VideoDiT(nn.Module):
    """A Diffusion Transformer over 3D-VAE latents.

    `pos` selects how tokens learn where they are:
        "rope3d"  — rotary, relative, grid-size agnostic (the modern default)
        "learned" — one trainable vector per grid slot (the original DiT)
        "none"    — no position information at all (the control)
    """

    def __init__(self, in_ch=4, patch=(1, 2, 2), grid=(4, 4, 4), dim=192,
                 depth=6, heads=6, pos="rope3d", pos_interp=False):
        super().__init__()
        self.in_ch, self.patch, self.grid, self.pos = in_ch, patch, grid, pos
        self.pos_interp = pos_interp
        self.dim, self.heads = dim, heads
        pdim = in_ch * patch[0] * patch[1] * patch[2]
        self.pdim = pdim
        self.embed = nn.Linear(pdim, dim)
        self.tmlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [DiTBlock(dim, heads) for _ in range(depth)])
        self.fnorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fada = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.fada.weight)
        nn.init.zeros_(self.fada.bias)
        self.head = nn.Linear(dim, pdim)
        nn.init.zeros_(self.head.weight)          # start by predicting "no noise"
        nn.init.zeros_(self.head.bias)
        if pos == "learned":
            n = grid[0] * grid[1] * grid[2]
            self.pos_emb = nn.Parameter(torch.randn(1, n, dim) * 0.02)
        self._rope_cache = {}

    def rope_for(self, grid):
        if self.pos != "rope3d":
            return None
        if grid not in self._rope_cache:
            self._rope_cache[grid] = rope_3d(grid, self.dim // self.heads)
        return self._rope_cache[grid]

    def learned_pos(self, grid):
        """The trainable position table, stretched if the grid changed.

        A learned table has one row per grid slot and no idea what a "slot"
        means, so a new (T, H, W) has no rows to look up.  The usual rescue is
        to treat the table as a small 3D image and RESIZE it — which is
        exactly what position interpolation does in long-context LLMs.  It
        keeps the model running; project 29 measures how well it works.
        """
        n = grid[0] * grid[1] * grid[2]
        if grid == tuple(self.grid):
            return self.pos_emb
        if not self.pos_interp:
            raise ValueError(
                f"learned positions were trained for grid {tuple(self.grid)}, "
                f"got {grid} — this is the limitation project 29 is about")
        p = self.pos_emb.view(1, *self.grid, self.dim).permute(0, 4, 1, 2, 3)
        p = F.interpolate(p, size=grid, mode="trilinear", align_corners=False)
        return p.permute(0, 2, 3, 4, 1).reshape(1, n, self.dim)

    def cond_vector(self, t, extra=None):
        c = self.tmlp(timestep_embedding(t, self.dim))
        return c if extra is None else c + extra

    def forward(self, x, t, extra_cond=None):
        tok, grid = patchify(x, self.patch)
        h = self.embed(tok)
        if self.pos == "learned":
            h = h + self.learned_pos(grid)
        rope = self.rope_for(grid)
        c = self.cond_vector(t, extra_cond)
        for blk in self.blocks:
            h = blk(h, c, rope)
        sh, sc = self.fada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        out = self.head(self.fnorm(h) * (1 + sc) + sh)
        return unpatchify(out, self.patch, grid, self.in_ch)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


# --------------------------------------------------------------------------
# helpers shared by the training scripts
# --------------------------------------------------------------------------

def strip(clip, n=8):
    """A (H, n*W) filmstrip image of one clip, values in [0, 1]."""
    return V.strip(clip, n=n)


def latent_cache_path(name, where=None):
    return (where or HERE / "checkpoints") / f"{name}.pt"


@torch.no_grad()
def build_latent_cache(n_clips=1024, batch=16, seed=25, train=True,
                       vae_kind="3d", name="latents", where=None):
    """Encode labelled clips ONCE with the frozen VAE and store the result.

    The VAE never changes again, so a clip's latent is a constant.  Encoding
    it fresh on every training step would be the same arithmetic thousands of
    times over.  Real pipelines do exactly this at petabyte scale: the latents
    go to disk, and diffusion training never sees a pixel.
    """
    vae, scale = load_vae(vae_kind)
    rng = np.random.default_rng(seed)
    lats, clips, digs, dirs = [], [], [], []
    for _ in range(n_clips // batch):
        x, d, dd = attr_batch(rng, batch, train=train)
        mean, _ = vae.encode(x)
        lats.append(mean * scale)
        clips.append(x)
        digs.append(d)
        dirs.append(dd)
    out = dict(latents=torch.cat(lats), clips=torch.cat(clips),
               digit=torch.cat(digs), direction=torch.cat(dirs),
               scale=scale, vae_kind=vae_kind)
    path = latent_cache_path(name, where)
    path.parent.mkdir(exist_ok=True)
    torch.save(out, path)
    return out


def load_latent_cache(name="latents", where=None):
    path = (where or HERE / "checkpoints") / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"missing {path} — run project 25's "
                         f"`python3 train.py --stage cache` first")
    return torch.load(path, map_location="cpu", weights_only=False)
