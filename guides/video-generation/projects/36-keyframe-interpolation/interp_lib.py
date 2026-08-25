"""Fill the gap between two frames you already have.

Project 35 made a long video by *chaining*: window after window, each one
adding new time on the end.  This project makes a long video the other way
round — decide the important moments first (the [keyframes]), then fill in
everything between them.  Animation studios have worked this way for a
century: a senior artist draws the poses that matter, and the "in-between"
drawings are made afterwards to connect them.

The task in one line
--------------------
Given latent frame 0 and latent frame 3 of a window, produce frames 1 and 2.

Three ways to do it, in increasing order of how much they know:

    linear    average the two ends, weighted by distance.  No model at all.
    inpaint   project 30's text-to-video model, with the two ends pinned at
              every sampling step.  No training.
    trained   a model trained on exactly this task: the two ends arrive as
              extra input channels, and it learns to produce the middle.

Why train a third model when `inpaint` already works?
-----------------------------------------------------
A fair question, because `inpaint` uses the same weights and costs nothing.
The difference is what the model is *allowed to assume*.  The base model was
trained to invent a clip from noise; when we pin two frames, it has never seen
that situation and has to discover mid-sampling that part of its canvas is
already decided.  The trained interpolator sees the two ends in its input from
step one of *training*, so "the answer must connect these" is baked into its
weights instead of being imposed from outside at sampling time.  That is the
gap the third arm fills — and measuring whether the gap is worth the training
run is the point of the project.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
sys.path.insert(0, str(HERE.parent / "35-sliding-window-t2v"))
import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import text_lib as T                                           # noqa: E402
import long_lib as LL                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

WIN = 4                       # latent frames per window
ANCHORS = (0, 3)              # which of them are keyframes
MIDDLE = [1, 2]
ARMS = ["linear", "inpaint", "trained"]
CFG = 3.0
STEPS = 30


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_cache(n_clips=1024, batch=16, seed=36, train=True,
                speed=(1.4, 2.2), name="latents"):
    """Encode labelled 16-frame clips once, exactly as project 25 did."""
    vae, scale = L.load_vae("3d")
    rng = np.random.default_rng(seed)
    lats, clips, digs, dirs = [], [], [], []
    for _ in range(n_clips // batch):
        x, d, dd = L.attr_batch(rng, batch, speed=speed, train=train)
        mean, _ = vae.encode(x)
        lats.append(mean * scale)
        clips.append(x)
        digs.append(d)
        dirs.append(dd)
    out = dict(latents=torch.cat(lats), clips=torch.cat(clips),
               digit=torch.cat(digs), direction=torch.cat(dirs), scale=scale)
    torch.save(out, CK / f"{name}.pt")
    return out


def load_cache(name="latents"):
    p = CK / f"{name}.pt"
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 run.py --stage cache`")
    return torch.load(p, map_location="cpu", weights_only=False)


def anchor_cond(x0, anchors=ANCHORS):
    """Build the extra input channels: the keyframes, plus a "here they are" map.

    Two pieces of information have to reach the model.  The obvious one is the
    *content* of the keyframes.  The less obvious one is *which frames are
    keyframes* — without it, a frame of zeros is ambiguous between "this slot
    is empty, invent it" and "this keyframe happens to be all zeros".  The
    mask channel removes the ambiguity, which is why every inpainting model
    carries one.
    """
    cond = torch.zeros_like(x0)
    mask = torch.zeros(x0.shape[0], 1, *x0.shape[2:])
    for a in anchors:
        cond[:, :, a] = x0[:, :, a]
        mask[:, :, a] = 1.0
    return torch.cat([cond, mask], 1)


# ---------------------------------------------------------------------------
# the trained interpolator
# ---------------------------------------------------------------------------

class InterpDiT(T.TextVideoDiT):
    """Project 30's text-video DiT with the keyframes wired into its input.

    Everything except the first and last layer is inherited unchanged, so this
    really is the same architecture with a wider entrance and the same exit.
    """

    def __init__(self, arm="t5", in_ch=4, cond_ch=5, **kw):
        super().__init__(arm, in_ch=in_ch, **kw)
        self.cond_ch = cond_ch
        p = self.patch
        pdim_in = (in_ch + cond_ch) * p[0] * p[1] * p[2]
        self.embed = nn.Linear(pdim_in, self.dim)

    def forward(self, x, t, text, ctx=None, cond=None):
        if cond is not None:
            x = torch.cat([x, cond], 1)
        tok, grid = L.patchify(x, self.patch)
        v = self.embed(tok)
        if ctx is None:
            ctx = self.context(text)
        ctx_seq, mask = ctx
        pooled = (ctx_seq * mask[..., None]).sum(1) \
            / mask.sum(1, keepdim=True).clamp(min=1.0)
        c = self.tmlp(L.timestep_embedding(t, self.dim)) \
            + self.pool(self.pool_norm(pooled))
        rope = self.rope_for(grid)
        for blk in self.blocks:
            v = blk(v, ctx_seq, mask, c, rope)
        sh, sc = self.fada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        out = self.head(self.fnorm(v) * (1 + sc) + sh)
        return L.unpatchify(out, self.patch, grid, self.in_ch)


def load_interp(path=None):
    p = path or (CK / "interp.pt")
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 run.py --stage train`")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = InterpDiT()
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


# ---------------------------------------------------------------------------
# the three fillers
# ---------------------------------------------------------------------------

def fill_linear(x0, anchors=ANCHORS):
    """Weighted average of the two keyframes — a cross-dissolve, nothing more.

    Worth running because it is the honest floor: if a learned model cannot
    beat a straight line between the endpoints, it has learned nothing about
    motion.  It also shows what "no model" looks like — the digit fades out of
    one position while fading in at the other instead of travelling.
    """
    a, b = anchors
    out = x0.clone()
    for f in range(a + 1, b):
        w = (b - f) / (b - a)
        out[:, :, f] = w * x0[:, :, a] + (1 - w) * x0[:, :, b]
    return out


@torch.no_grad()
def fill_inpaint(model, x0, text, null, anchors=ANCHORS, steps=STEPS, cfg=CFG,
                 generator=None):
    """Base model, keyframes pinned at every step (the trick from project 35)."""
    flow = FL.RectifiedFlow()
    ctx = model.context(text)
    ctx_n = model.context(null)
    noise = torch.randn(x0.shape, generator=generator)
    x = noise.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        tt = ts[i]
        for a in anchors:
            x[:, :, a] = (1 - tt) * x0[:, :, a] + tt * noise[:, :, a]
        t = ts[i].expand(x.shape[0]) * flow.T_SCALE
        v = model(x, t, None, ctx=ctx)
        if cfg != 1.0:
            v_n = model(x, t, None, ctx=ctx_n)
            v = v_n + cfg * (v - v_n)
        x = x + (ts[i + 1] - ts[i]) * v
    for a in anchors:
        x[:, :, a] = x0[:, :, a]
    return x


@torch.no_grad()
def fill_trained(model, x0, text, null, anchors=ANCHORS, steps=STEPS, cfg=CFG,
                 generator=None):
    flow = FL.RectifiedFlow()
    cond = anchor_cond(x0, anchors)
    ctx = model.context(text)
    ctx_n = model.context(null)
    x = torch.randn(x0.shape, generator=generator)
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(x.shape[0]) * flow.T_SCALE
        v = model(x, t, None, ctx=ctx, cond=cond)
        if cfg != 1.0:
            v_n = model(x, t, None, ctx=ctx_n, cond=cond)
            v = v_n + cfg * (v - v_n)
        x = x + (ts[i + 1] - ts[i]) * v
    for a in anchors:
        x[:, :, a] = x0[:, :, a]
    return x


# ---------------------------------------------------------------------------
# scoring a filled window
# ---------------------------------------------------------------------------

def middle_slice(clips, anchors=ANCHORS, per=LL.PIX_PER_LAT):
    """The pixel frames that the filler actually invented."""
    return clips[:, :, (anchors[0] + 1) * per:anchors[1] * per]


def fill_error(pred, truth, anchors=ANCHORS):
    """Mean absolute pixel error over the invented frames only."""
    return (middle_slice(pred, anchors) -
            middle_slice(truth, anchors)).abs().mean(dim=(1, 2, 3, 4))


def path_error(pred, truth, anchors=ANCHORS, per=LL.PIX_PER_LAT):
    """How far the digit's centre is from where it should be, in pixels."""
    a = LL.centroid_path(pred)[:, (anchors[0] + 1) * per:anchors[1] * per]
    b = LL.centroid_path(truth)[:, (anchors[0] + 1) * per:anchors[1] * per]
    return (a - b).norm(dim=-1).mean(1)


def anchor_gap(clips, anchors=ANCHORS, per=LL.PIX_PER_LAT):
    """Distance in pixels between the digit's position in the two keyframes."""
    p = LL.centroid_path(clips)
    return (p[:, anchors[1] * per] - p[:, anchors[0] * per]).norm(dim=-1)


__all__ = ["WIN", "ANCHORS", "MIDDLE", "ARMS", "build_cache", "load_cache",
           "anchor_cond", "InterpDiT", "load_interp", "fill_linear",
           "fill_inpaint", "fill_trained", "fill_error", "path_error",
           "anchor_gap", "middle_slice", "LL", "T", "L", "FL"]
