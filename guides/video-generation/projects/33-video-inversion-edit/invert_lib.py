"""Inversion: pushing a real clip back into the noise it could have come from.

The problem
-----------
A generator turns noise into video.  To *edit* a real video with one, you need
the noise that would have produced that particular video — otherwise the model
has nothing to start from but a random seed, and a random seed gives you a
different scene entirely.

Finding that noise is called **inversion**: running the generator's process
backwards.

DDIM inversion, and why this project's version looks different
--------------------------------------------------------------
[DDIM inversion](/shared/glossary/#ddim-inversion) is the original recipe, and
it only works because DDIM sampling is *deterministic*: plain DDPM sampling
injects fresh random noise at every step, and you cannot undo a coin flip.
DDIM removed that injection, which turned sampling into a fixed path you can
walk in either direction.

Our Phase-6/7 models are trained with rectified flow, and a flow model states
that path even more plainly: it is the solution of an ordinary differential
equation, `dx/dt = v(x, t)`.  Sampling integrates from t = 1 (noise) down to
t = 0 (clip).  Inversion integrates the same field from t = 0 up to t = 1.
Same idea as DDIM inversion, one fewer layer of algebra — which is a fair
summary of why flow matching took over.

The two things that break it
----------------------------
1. **Step count.**  Euler integration is only approximate; too few steps and
   the round trip lands somewhere else.
2. **Guidance.**  Inverting with classifier-free guidance turned up is
   inverting a *different* velocity field from the one the model was trained
   to follow — an extrapolated one.  The error compounds and the round trip
   falls apart.  Both are measured in this project's `checks` stage.
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import text_lib as T                                           # noqa: E402

T_SCALE = FL.RectifiedFlow.T_SCALE


def _v(model, x, t, ctx_c, ctx_n, scale):
    tt = torch.full((x.shape[0],), float(t) * T_SCALE)
    v = model(x, tt, None, ctx=ctx_c)
    if scale != 1.0:
        vn = model(x, tt, None, ctx=ctx_n)
        v = vn + scale * (v - vn)
    return v


@torch.no_grad()
def invert(model, x0, text, null, steps=60, scale=1.0, t_max=1.0):
    """Walk a clean latent UP the flow, from t = 0 to t = t_max.

    `t_max < 1` stops part-way: the result still contains most of the original
    clip, buried under some noise.  That is the SDEdit idea — "add noise, then
    denoise with a new prompt" — and it is the cheap cousin of full inversion.
    """
    ctx_c, ctx_n = model.context(text), model.context(null)
    ts = torch.linspace(0.0, t_max, steps + 1)
    x = x0.clone()
    for i in range(steps):
        v = _v(model, x, ts[i], ctx_c, ctx_n, scale)
        x = x + (ts[i + 1] - ts[i]) * v
    return x


@torch.no_grad()
def denoise(model, xt, text, null, steps=60, scale=3.0, t_max=1.0):
    """Walk back DOWN the flow, from t = t_max to t = 0, with any prompt."""
    ctx_c, ctx_n = model.context(text), model.context(null)
    ts = torch.linspace(t_max, 0.0, steps + 1)
    x = xt.clone()
    for i in range(steps):
        v = _v(model, x, ts[i], ctx_c, ctx_n, scale)
        x = x + (ts[i + 1] - ts[i]) * v
    return x


# ---------------------------------------------------------------------------
# the "treat every frame on its own" baseline
# ---------------------------------------------------------------------------

def _split_frames(z):
    """(B, C, T, H, W) -> (B*T, C, 1, H, W): each latent frame as its own clip."""
    B, C, Tn, H, W = z.shape
    return z.permute(0, 2, 1, 3, 4).reshape(B * Tn, C, 1, H, W)


def _join_frames(z, Tn):
    BT, C, one, H, W = z.shape
    return z.reshape(BT // Tn, Tn, C, H, W).permute(0, 2, 1, 3, 4)


def _expand_text(text, Tn):
    return {k: (s.repeat_interleave(Tn, 0), m.repeat_interleave(Tn, 0))
            for k, (s, m) in text.items()}


@torch.no_grad()
def per_frame_edit(model, z0, text, null, new_text, steps=60, scale=3.0,
                   t_max=1.0):
    """Invert and re-denoise every latent frame independently.

    This is what "run an image editor on each frame" means, written out.  It
    is possible at all only because the backbone uses rotary positions, which
    let the same weights run on a one-frame grid (project 29).  Nothing here
    lets one frame know what another frame did — which is exactly the point
    being demonstrated.
    """
    Tn = z0.shape[2]
    zf = _split_frames(z0)
    tx, nl, nt = (_expand_text(text, Tn), _expand_text(null, Tn),
                  _expand_text(new_text, Tn))
    noise = invert(model, zf, tx, nl, steps=steps, scale=1.0, t_max=t_max)
    out = denoise(model, noise, nt, nl, steps=steps, scale=scale, t_max=t_max)
    return _join_frames(out, Tn)


# ---------------------------------------------------------------------------
# what "preserved" means, measured
# ---------------------------------------------------------------------------

def centroid_path(clips):
    """Per-frame centre of mass in pixels: (B, T, 2)."""
    x = (clips.clamp(-1, 1) + 1) / 2
    x = x.squeeze(1)
    B, Tn, H, W = x.shape
    ys = torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
    xs = torch.arange(W, dtype=torch.float32).view(1, 1, 1, W)
    m = x.sum((2, 3)).clamp(min=1e-4)
    return torch.stack([(x * ys).sum((2, 3)) / m, (x * xs).sum((2, 3)) / m], -1)


def path_error(a, b):
    """Mean per-frame distance between two centroid paths, in pixels."""
    return float((centroid_path(a) - centroid_path(b)).pow(2).sum(-1)
                 .sqrt().mean())


def flicker(clips):
    return float((clips[:, :, 1:] - clips[:, :, :-1]).abs().mean())


__all__ = ["invert", "denoise", "per_frame_edit", "path_error", "flicker",
           "centroid_path", "T", "L", "FL"]
