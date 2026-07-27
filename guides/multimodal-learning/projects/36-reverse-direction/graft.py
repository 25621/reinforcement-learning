"""Turning an understanding-only VLM into a generator: the head and the LoRA.

Two small pieces live here.

1. `Grafted` -- the model surgery. A VLM produces one hidden vector per
   position and multiplies it by a text output head. We keep that head and add
   a second one that covers the image codes, then paste the two sets of scores
   into one row so the ordinary next-token loss still applies unchanged.

   "But the VLM already has image codes in its vocabulary -- why add a head?"
   It does, and that is exactly the trap. Our backbone ties its output head to
   its input embedding table, so the scores for image codes are computable from
   day one. They were just never *trained*: during VLM training every target
   was a text token, so the only gradient those columns ever received pushed
   them DOWN, to stop image codes from being predicted. Measured in project 36,
   the base VLM scores 12.76 nats on image tokens against a chance level of
   6.24 -- it is twice as bad as guessing, which is what "trained to suppress"
   looks like as a number.

   The new head starts with no such history, and -- unlike the tied one --
   updating it cannot touch the input embeddings the model reads images with.
   That second property turns out to matter far more than the first: project 36
   finds the two options almost tied on drawing (6.14 vs 5.80 nats) and utterly
   different on damage, because training the tied matrix wrecks the word
   embeddings too (captioning +11.4 nats, against +2.3 for the new head).
   Both numbers are measured, so you can see the trade rather than take it on
   trust -- and neither option ends up recommended, for reasons the project's
   README explains.

2. `inject_lora` -- Low-Rank Adaptation (Hu et al., 2021). Instead of changing
   a weight matrix W, freeze it and learn a small correction B @ A, where A is
   r-by-in and B is out-by-r with r tiny (8 here). "Low-rank" is literal: the
   correction matrix B @ A can have rank at most r, so it is a thin, cheap
   nudge rather than a free rewrite. The point for this project is that it lets
   the body of the model move a little without moving far enough to forget what
   it already knew.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Grafted(nn.Module):
    """A trained VLM backbone plus a new output head over image codes."""

    def __init__(self, backbone, image_base, n_image_codes, d=None,
                 use_new_head=True):
        super().__init__()
        self.backbone = backbone
        self.image_base, self.n_image = image_base, n_image_codes
        self.use_new_head = use_new_head
        d = d or backbone.tok.embedding_dim
        self.image_head = nn.Linear(d, n_image_codes, bias=False)
        nn.init.normal_(self.image_head.weight, std=0.02)

    @property
    def ctx(self):
        return self.backbone.ctx

    def forward(self, ids, aux=None):
        h = self.backbone.hidden(ids, aux)
        logits = self.backbone.head(h)
        if self.use_new_head:
            # overwrite exactly the image-code columns; everything else -- text,
            # <boi>, <eoi> -- still comes from the original tied head
            lo, hi = self.image_base, self.image_base + self.n_image
            logits = torch.cat([logits[..., :lo], self.image_head(h),
                                logits[..., hi:]], dim=-1)
        return logits


class LoRALinear(nn.Module):
    """W frozen, plus a trainable rank-r correction scaled by alpha/r."""

    def __init__(self, base, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.a = nn.Parameter(torch.randn(r, base.in_features) * (base.in_features ** -0.5))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))   # starts as a no-op
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def inject_lora(model, r=8, alpha=16, match=("qkv", "proj")):
    """Wrap the named Linear layers of every block in LoRA. Returns the count.

    Targets are collected before anything is replaced -- mutating the module
    tree while walking it would make the walk visit the new modules too.
    """
    targets = []
    for name, mod in model.named_modules():
        for child_name, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and child_name in match:
                targets.append((mod, child_name, child))
    for parent, child_name, child in targets:
        setattr(parent, child_name, LoRALinear(child, r, alpha))
    return len(targets)


def trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
