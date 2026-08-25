"""ControlNet for a video DiT: a trainable copy that steers a frozen model.

The problem ControlNet solves
-----------------------------
A prompt says *what*.  It is hopeless at saying *where*: no sentence pins a
shape to a pixel.  ControlNet adds a second input — a structural map such as a
depth map, a pose skeleton or an edge drawing — that says exactly where things
belong, frame by frame.

Why not simply fine-tune the base model to accept the extra input?
------------------------------------------------------------------
You could, and it usually goes badly.  Control datasets are small (thousands
of clips, not millions), and fine-tuning every weight on a small dataset drags
the model away from everything else it knew — the failure Phase 4's project 16
measured directly.  ControlNet's answer:

  * **freeze the base**, so its knowledge cannot be damaged at all;
  * train a **copy of some of its blocks** as a side branch;
  * connect the branch to the base through **zero-initialised** projections,
    so at step 0 the branch contributes exactly nothing and the whole system
    is bit-for-bit the original model.

Why copy the blocks instead of starting the branch from random weights?
Because a copy already knows how to represent this kind of data.  Random
weights would spend the first thousands of steps re-learning what the frozen
base next to them already knows.  The `scratch` arm in this project measures
how much that head start is worth.

Why "zero convolution"?
ControlNet's paper connects branch to base with 1x1 convolutions whose weights
start at zero.  Zero output means zero gradient *through* the connection into
the base's activations at first, so the frozen model keeps behaving exactly as
before while the branch's own weights start moving.  Our tokens are vectors,
not feature maps, so the same idea is a zero-initialised `nn.Linear` — same
role, same reason, different shape.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "21-train-a-small-3d-vae"))
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
import vae3d_lib as V                                          # noqa: E402
import dit_lib as L                                            # noqa: E402
import text_lib as T                                           # noqa: E402

CTRL_RES = 16          # the control map is 16x16 per frame
CTRL_FRAMES = 16       # one control frame per video frame


# ---------------------------------------------------------------------------
# the control signal
# ---------------------------------------------------------------------------

def depth_proxy(clips, res=CTRL_RES):
    """A stand-in for a per-frame depth map, built from a real clip.

    A real pipeline runs a depth estimator (MiDaS, Depth-Anything) on each
    frame of a source video.  We have no depth in Moving MNIST, so we make a
    signal that plays the same *role*: average-pool every frame down to 16x16,
    which keeps where the bright object is and how it moves, and destroys the
    stroke detail that says *which* digit it is.

    That is the honest analogy to depth: a depth map tells you an object's
    silhouette and distance and says nothing about its colour or texture.
    Structure comes from the control; identity has to come from the prompt.
    """
    x = (clips.clamp(-1, 1) + 1) / 2                    # (B,1,T,H,W) -> [0,1]
    B, C, Tn, H, W = x.shape
    x = F.avg_pool2d(x.reshape(B * C * Tn, 1, H, W), H // res)
    return x.reshape(B, 1, Tn, res, res)


def jitter(ctrl, std, generator=None):
    """Independent noise per frame — what a per-frame estimator really gives.

    Depth estimators are run one frame at a time and are not perfectly stable:
    the same wall comes back slightly nearer or further from frame to frame.
    That flicker is in the *control*, before the video model sees anything.
    """
    if std <= 0:
        return ctrl
    n = torch.randn(ctrl.shape, generator=generator) * std
    return (ctrl + n).clamp(0, 1)


# ---------------------------------------------------------------------------
# control encoders: with and without a sense of time
# ---------------------------------------------------------------------------

class TemporalCtrlEncoder(nn.Module):
    """3D convolutions: every output token pools several input frames."""

    def __init__(self, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, (3, 4, 4), (2, 2, 2), (1, 1, 1)), nn.SiLU(),
            nn.Conv3d(16, out_ch, (3, 4, 4), (2, 2, 2), (1, 1, 1)), nn.SiLU(),
        )

    def forward(self, ctrl):                       # (B,1,16,16,16)
        h = self.net(ctrl)                         # (B,C,4,4,4)
        return h.flatten(2).transpose(1, 2)        # (B,64,C)


class PerFrameCtrlEncoder(nn.Module):
    """2D convolutions on four separate frames: no mixing across time at all.

    This is the naive "run the image ControlNet on every frame" pipeline
    written honestly: each output token is a function of exactly one input
    frame, so anything that wobbles between frames is passed straight through.
    """

    def __init__(self, out_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(16, out_ch, 4, 2, 1), nn.SiLU(),
        )
        self.picks = [0, 4, 8, 12]                 # one frame per latent slot

    def forward(self, ctrl):
        B = ctrl.shape[0]
        fr = ctrl[:, :, self.picks]                # (B,1,4,16,16)
        fr = fr.permute(0, 2, 1, 3, 4).reshape(B * 4, 1, CTRL_RES, CTRL_RES)
        h = self.net(fr)                           # (B*4,C,4,4)
        C = h.shape[1]
        h = h.reshape(B, 4, C, 4, 4).permute(0, 2, 1, 3, 4)
        return h.flatten(2).transpose(1, 2)        # (B,64,C)


# ---------------------------------------------------------------------------
# a self-attention that can be locked inside a single frame
# ---------------------------------------------------------------------------

class MaskedAttention(L.Attention):
    """Same weights and shape as the base's attention, plus an optional mask."""

    def __init__(self, dim, heads):
        super().__init__(dim, heads)
        self.frame_mask = None

    def forward(self, x, rope=None):
        B, N, D = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        if rope is not None:
            cos, sin = rope
            q, k = L.apply_rope(q, cos, sin), L.apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.frame_mask)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class CtrlBlock(T.CrossBlock):
    """A copy of a base block whose self-attention can be frame-local."""

    def __init__(self, dim, heads):
        super().__init__(dim, heads)
        self.attn = MaskedAttention(dim, heads)


def frame_local_mask(grid):
    """True only where two tokens sit in the same frame of the latent grid."""
    nt, nh, nw = grid
    idx = torch.arange(nt * nh * nw) // (nh * nw)
    return (idx[:, None] == idx[None, :])


# ---------------------------------------------------------------------------
# the controlled model
# ---------------------------------------------------------------------------

class ControlledDiT(nn.Module):
    """Frozen text-to-video DiT + a trainable control branch."""

    def __init__(self, base, mode="temporal", n_copy=3, ctrl_ch=32):
        super().__init__()
        self.base = base
        self.base.eval().requires_grad_(False)
        self.mode = mode
        dim, heads = base.dim, base.heads
        self.enc = (PerFrameCtrlEncoder(ctrl_ch) if mode == "perframe"
                    else TemporalCtrlEncoder(ctrl_ch))
        # zero conv #1: the control's own entry into the branch
        self.ctrl_in = nn.Linear(ctrl_ch, dim)
        nn.init.zeros_(self.ctrl_in.weight)
        nn.init.zeros_(self.ctrl_in.bias)
        self.blocks = nn.ModuleList([CtrlBlock(dim, heads)
                                     for _ in range(n_copy)])
        if mode != "scratch":
            for i, blk in enumerate(self.blocks):    # the trainable COPY
                blk.load_state_dict(base.blocks[i].state_dict())
        # zero conv #2..: each branch block's contribution to the frozen base
        self.outs = nn.ModuleList([nn.Linear(dim, dim)
                                   for _ in range(n_copy)])
        for lin in self.outs:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._mask = {}

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def context(self, text):
        """Delegate to the frozen base, so the shared sampler works unchanged."""
        return self.base.context(text)

    def set_mask(self, grid):
        if self.mode != "perframe":
            return
        if grid not in self._mask:
            self._mask[grid] = frame_local_mask(grid)
        for blk in self.blocks:
            blk.attn.frame_mask = self._mask[grid]

    def forward(self, x, t, text, ctx=None, control=None):
        base = self.base
        tok, grid = L.patchify(x, base.patch)
        v = base.embed(tok)
        if ctx is None:
            ctx = base.context(text)
        ctx_seq, mask = ctx
        pooled = (ctx_seq * mask[..., None]).sum(1) \
            / mask.sum(1, keepdim=True).clamp(min=1.0)
        c = base.tmlp(L.timestep_embedding(t, base.dim)) \
            + base.pool(base.pool_norm(pooled))
        rope = base.rope_for(grid)

        res = []
        if control is not None:
            self.set_mask(grid)
            h = v + self.ctrl_in(self.enc(control))
            for blk, out in zip(self.blocks, self.outs):
                h = blk(h, ctx_seq, mask, c, rope)
                res.append(out(h))

        for i, blk in enumerate(base.blocks):
            v = blk(v, ctx_seq, mask, c, rope)
            if i < len(res):
                v = v + res[i]
        sh, sc = base.fada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        out = base.head(base.fnorm(v) * (1 + sc) + sh)
        return L.unpatchify(out, base.patch, grid, base.in_ch)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def centroid_path(clips):
    """Per-frame centre of mass, in pixels: (B, T, 2)."""
    x = (clips.clamp(-1, 1) + 1) / 2
    x = x.squeeze(1)
    B, Tn, H, W = x.shape
    ys = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
    m = x.sum((2, 3)).clamp(min=1e-4)
    return torch.stack([(x * ys).sum((2, 3)) / m, (x * xs).sum((2, 3)) / m], -1)


def ctrl_centroid_path(ctrl, scale):
    """The same measurement on the low-resolution control map."""
    x = ctrl.squeeze(1)
    B, Tn, H, W = x.shape
    ys = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
    m = x.sum((2, 3)).clamp(min=1e-4)
    p = torch.stack([(x * ys).sum((2, 3)) / m, (x * xs).sum((2, 3)) / m], -1)
    return p * scale + (scale - 1) / 2


def tracking_error(gen, ctrl):
    """Mean distance, in pixels, between where the clip put the object and
    where the control asked for it."""
    a = centroid_path(gen)
    b = ctrl_centroid_path(ctrl, gen.shape[-1] / ctrl.shape[-1])
    return float((a - b).pow(2).sum(-1).sqrt().mean())


def flicker(clips):
    return V.flicker(clips)


def upsampled_control(ctrl, size=64):
    """Control maps blown back up to frame size, for side-by-side figures."""
    B, C, Tn, H, W = ctrl.shape
    x = F.interpolate(ctrl.reshape(B * Tn, 1, H, W), size=(size, size),
                      mode="nearest")
    return x.reshape(B, 1, Tn, size, size) * 2 - 1


__all__ = ["depth_proxy", "jitter", "ControlledDiT", "tracking_error",
           "flicker", "centroid_path", "ctrl_centroid_path",
           "upsampled_control", "np"]
