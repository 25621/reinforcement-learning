"""Video transformers that differ only in how they cut the clip into tokens.

Everything else -- width, depth, heads, the transformer block itself (imported
from project 04's `vit.py`) -- is identical across arms, so any difference in
accuracy comes from the tokenisation and nothing else.

Three ways to turn 8 frames x 32 x 32 pixels into tokens:

  framewise-pool   cut each frame into 2D patches, run the transformer on ONE
                   frame at a time, average the 8 results.
                   This model is *provably* blind to frame order: averaging does
                   not care which order things arrive in. It is the honest
                   version of "sample frames and pool", and the reason it is
                   here is that it cannot cheat.

  framewise-attend cut each frame into 2D patches, but put all 8 frames' tokens
                   in one sequence with a learned per-frame position embedding.
                   Motion must be *inferred* by attention comparing tokens that
                   came from different frames -- "late fusion".

  tubelet-k        cut the clip into 3D boxes spanning k frames at once (the
                   TubeViT / ViViT recipe). Motion is inside a single token from
                   the start -- "early fusion" -- and the sequence gets k times
                   shorter.

A "tubelet" is just a patch with a time dimension: a small square of pixels
extruded through several frames, like a tube through the video volume.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent
                       / "04-implement-vit-from-scratch"))
from vit import Block  # noqa: E402


def tubify(x, tube_t, patch):
    """(B, T, H, W, C) -> (B, n_tubes, tube_t*patch*patch*C).

    `unfold` slides a window and copies out its contents; doing it on three axes
    (time, height, width) is exactly "cut the video into boxes". The flattened
    box is what the linear layer turns into one token, so the mixing of the
    frames inside a tubelet happens in that first matrix -- before any attention
    runs at all.
    """
    B, T, H, W, C = x.shape
    x = x.permute(0, 4, 1, 2, 3)                     # (B, C, T, H, W)
    x = (x.unfold(2, tube_t, tube_t)
          .unfold(3, patch, patch)
          .unfold(4, patch, patch))                  # (B, C, nt, nh, nw, t, p, p)
    x = x.permute(0, 2, 3, 4, 1, 5, 6, 7).contiguous()
    return x.reshape(B, -1, C * tube_t * patch * patch)


class VideoViT(nn.Module):
    def __init__(self, mode="tubelet", tube_t=2, patch=8, size=32, frames=8,
                 dim=128, depth=4, heads=4, tasks=(("direction", 4),
                                                   ("speed", 2), ("content", 2))):
        super().__init__()
        self.mode, self.tube_t, self.patch, self.frames = mode, tube_t, patch, frames
        self.n_space = (size // patch) ** 2
        self.n_time = frames // tube_t
        self.n_tokens = self.n_space * self.n_time
        self.embed = nn.Linear(3 * tube_t * patch * patch, dim)
        # one position vector per (time slot, space slot). For the pooled arm we
        # only ever look at one frame at a time, so it gets spatial slots only --
        # which is precisely why it cannot know the order.
        n_pos = self.n_space if mode == "framewise-pool" else self.n_tokens
        self.pos = nn.Parameter(torch.zeros(1, n_pos, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.heads = nn.ModuleDict({name: nn.Linear(dim, n) for name, n in tasks})

    def forward(self, x):                            # x: (B, T, H, W, 3) float
        if self.mode == "framewise-pool":
            B, T = x.shape[:2]
            t = tubify(x.reshape(B * T, 1, *x.shape[2:]), 1, self.patch)
            t = self.embed(t) + self.pos
            for blk in self.blocks:
                t = blk(t)
            feat = self.norm(t).mean(1).reshape(B, T, -1).mean(1)
        else:
            t = self.embed(tubify(x, self.tube_t, self.patch)) + self.pos
            for blk in self.blocks:
                t = blk(t)
            feat = self.norm(t).mean(1)
        return {name: head(feat) for name, head in self.heads.items()}

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def attention_pairs(self):
        """Token pairs each attention layer must score: the cost that explodes."""
        if self.mode == "framewise-pool":
            return self.frames * self.n_space ** 2
        return self.n_tokens ** 2


def multi_loss(out, y):
    return sum(F.cross_entropy(out[k], y[k]) for k in out)
