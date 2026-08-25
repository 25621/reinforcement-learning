"""LoRA: teaching a frozen video model a new look with a few hundred KB.

What the name says
------------------
**LoRA = Low-Rank Adaptation.**  Fine-tuning normally learns a full update
matrix `dW` the same size as the weight `W` it corrects.  LoRA instead writes

    dW = B @ A          with A: (r x in),  B: (out x r),  r tiny

The *rank* of a matrix is how many independent directions it can move things
in.  `B @ A` cannot have rank higher than `r`, so calling it "low-rank" is
literally a statement about how many independent directions the update is
allowed to use.  For a 128x384 weight and r = 4 that is 2,048 numbers instead
of 49,152 — and the claim, which held up across the whole image-generation
world, is that the *style* of a model is a low-rank kind of change: it does not
need every direction, only a few.

Why is that plausible?  Because "draw everything with heavier strokes" is one
consistent instruction applied everywhere, not thousands of unrelated
corrections.  One instruction needs few directions.

Two initialisation details that are not decoration
--------------------------------------------------
`A` starts random and `B` starts at **zero**, so `B @ A = 0` and the adapted
model is bit-for-bit the original at step 0 — the same guarantee ControlNet
gets from its zero convolutions (project 31), for the same reason: a fresh
side path must not damage a model that already works.

The scale `alpha / r` keeps the update's size roughly constant as you change
`r`, so a rank sweep compares ranks rather than accidentally comparing
learning rates.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
import dit_lib as L                                            # noqa: E402
import text_lib as T                                           # noqa: E402


class LoRALinear(nn.Module):
    """A frozen Linear with a trainable low-rank detour beside it."""

    def __init__(self, base, r=4, alpha=None):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.lora_scale = 1.0        # the inference-time dial

    def forward(self, x):
        y = self.base(x)
        if self.lora_scale == 0.0:
            return y
        return y + self.lora_scale * self.scaling * \
            F.linear(F.linear(x, self.A), self.B)


ATTN = ("attn.qkv", "attn.proj", "cross.q", "cross.kv", "cross.proj")
MLP = ("mlp.0", "mlp.2")


def inject_lora(model, r=4, alpha=None, targets=ATTN):
    """Replace matching Linears inside the DiT blocks with LoRA versions.

    Note the two-pass structure.  Collecting the targets first and only then
    swapping them matters: `named_modules()` walks the module tree lazily, so
    mutating the tree while iterating makes the walk visit — or skip — things
    unpredictably.  Collect, then mutate.
    """
    found = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not name.startswith("blocks."):
            continue
        if any(name.endswith(t) for t in targets):
            found.append(name)
    for name in found:
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        leaf = name.rsplit(".", 1)[1]
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), r, alpha))
    return found


def lora_parameters(model):
    return [p for n, p in model.named_parameters()
            if n.endswith(".A") or n.endswith(".B")]


def lora_state(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()
            if n.endswith(".A") or n.endswith(".B")}


def set_scale(model, s):
    """Turn the adapter up or down at inference time, with no retraining."""
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.lora_scale = s


def build_lora_model(r=4, alpha=None, targets=ATTN, arm="t5"):
    model, _ = T.load_arm(arm)
    model.requires_grad_(False)
    names = inject_lora(model, r=r, alpha=alpha, targets=targets)
    return model, names


# ---------------------------------------------------------------------------
# the two styles the adapters have to learn
# ---------------------------------------------------------------------------

STYLES = ["thick", "trail"]
CANDIDATES = ["plain", "thick", "trail", "outline", "halo", "negative"]
STYLE_HELP = {
    "plain": "the data the base model was trained on",
    "thick": "bold, heavy strokes — an APPEARANCE style",
    "trail": "a fading ghost behind the digit — a MOTION style",
    "outline": "hollow strokes (rejected: the VAE reshapes it into a blob)",
    "halo": "a soft glow (survives too; two styles are enough here)",
    "negative": "dark on white (rejected: the VAE cannot encode it at all)",
}


def _blur(x, k=9):
    B, C, Tn, H, W = x.shape
    f = torch.ones(1, 1, k, k) / (k * k)
    return F.conv2d(x.reshape(-1, 1, H, W), f, padding=k // 2) \
        .reshape(B, C, Tn, H, W)


def stylize(clips, style):
    """Apply a style to pixel clips in [-1, 1].

    `thick` is a look: it changes how a frame is drawn and nothing else.
    `trail` is a behaviour: it can only be *seen* because the clip has a time
    axis, and reproducing it means changing what the model does *across*
    frames, not just within one.  Having one of each is the point — a video
    LoRA can teach motion, which an image LoRA has no way to express.

    The other three styles here are the ones this project tried and rejected;
    `run.py --stage data` draws the evidence.  See the README: a LoRA can only
    teach the generator something the frozen VAE is still able to represent.
    """
    x = (clips.clamp(-1, 1) + 1) / 2
    B, C, Tn, H, W = x.shape
    flat = x.reshape(B * C * Tn, 1, H, W)
    if style == "plain":
        y = x
    elif style == "thick":
        y = F.max_pool2d(flat, 5, 1, 2).reshape(B, C, Tn, H, W)
    elif style == "trail":
        y = x.clone()
        for k in range(1, 6):
            shifted = F.pad(x, (0, 0, 0, 0, k, 0))[:, :, :Tn]
            y = torch.maximum(y, shifted * (0.78 ** k))
    elif style == "outline":
        y = (F.max_pool2d(flat, 3, 1, 1) - flat).clamp(0, 1) \
            .reshape(B, C, Tn, H, W) * 1.8
    elif style == "halo":
        y = x + 1.4 * _blur(x, 9)
    elif style == "negative":
        y = 1 - x
    else:
        raise ValueError(style)
    return (y.clamp(0, 1) * 2 - 1)


__all__ = ["LoRALinear", "inject_lora", "lora_parameters", "lora_state",
           "set_scale", "build_lora_model", "stylize", "STYLES", "CANDIDATES",
           "STYLE_HELP", "ATTN", "MLP", "T", "L"]
