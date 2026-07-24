"""A latent action model: guessing the button from the footage.

The problem this solves
-----------------------
Project 40's model needed pairs of (frame, button).  Recording those is easy in
a toy game and nearly impossible on the open internet: nobody stored which key
the player pressed while the video was uploaded.  So the largest video corpus
in the world is unusable for training a controllable world model — unless you
can *recover* the action from the pixels.

That is what [Genie](/shared/glossary/#genie) does, and what this file
implements:

    frame_t, frame_t+1  ->  [encoder]  ->  one of K discrete codes  ->
    frame_t, that code  ->  [decoder]  ->  frame_t+1

Both halves are trained together, with only one loss: reconstruct frame_t+1.

Why the bottleneck is the whole trick
-------------------------------------
Nothing above mentions actions.  If the code could carry a lot of information
the encoder would simply copy frame_t+1 through it and the decoder would learn
nothing — an autoencoder with an expensive detour.  So the code is squeezed to
a single symbol out of K (K = 8 here, 64 bits' worth of frame squeezed into 3).
At that width the only thing worth transmitting is the *smallest* description
of the change, and in this game the smallest description of a change is "which
way the player moved".  The action falls out of the compression.

This is the same argument as [VQ-VAE](/shared/glossary/#vq-vae): a discrete
bottleneck forces the encoder to keep only what the decoder cannot guess on its
own.  The difference is what sits either side of it — here the decoder already
has frame_t, so "what the decoder cannot guess" is exactly the player's input.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "40-action-conditioned-video"))
import world_lib as W                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)


class DiscreteBottleneck(nn.Module):
    """Force a continuous description of "what changed" into one of K symbols.

    The encoder emits K scores; the highest one wins and its learned vector is
    what the decoder receives.  Two mechanisms make this trainable:

    **Straight-through gradients.**  The forward pass uses the hard winner; the
    backward pass pretends we used the soft probabilities instead.  Without this
    there is no gradient at all, because "take the argmax" has a derivative of
    zero almost everywhere.  It is the same
    [straight-through estimator](/shared/glossary/#straight-through-estimator)
    that [VQ-VAE](/shared/glossary/#vq-vae) uses.

    **A usage penalty.**  Averaged over a batch, the chosen codes should be
    spread out.  Without the penalty the model happily assigns *every* frame
    pair to a single code — [codebook collapse](/shared/glossary/#codebook-collapse)
    — because a decoder that ignores the code still gets a mediocre-but-safe
    loss by predicting the average future.  The penalty is the gap between the
    batch's average code distribution and a flat one, measured in
    [nats](/shared/glossary/#nat); it is zero when all codes get equal use.

    Why not plain VQ-VAE nearest-neighbour lookup?  We tried; it is documented
    in this project's README, and it failed twice for two different reasons.
    Distance-based lookup adds a term pulling encoder outputs toward their code,
    and the cheapest way to shrink that term is to shrink everything — which is
    what happened (encoder outputs collapsed to a standard deviation of 0.03).
    Normalising both sides onto the unit sphere stops the shrinking but not the
    collapse onto one code, because nothing rewards using the others.  Scoring
    the K options directly, and penalising lopsided usage, fixes both.
    """

    def __init__(self, n_codes, dim, usage_weight=0.05):
        super().__init__()
        self.emb = nn.Embedding(n_codes, dim)
        self.emb.weight.data.normal_(0, 1.0)
        self.n_codes, self.usage_weight = n_codes, usage_weight
        self.register_buffer("usage", torch.zeros(n_codes))

    def codebook(self):
        return F.normalize(self.emb.weight, dim=1)

    def forward(self, logits):
        q = F.softmax(logits, dim=1)
        idx = q.argmax(dim=1)
        hard = F.one_hot(idx, self.n_codes).float()
        q_st = hard + q - q.detach()                  # straight-through
        zq = q_st @ self.codebook()
        avg = q.mean(dim=0)
        loss = self.usage_weight * (
            avg * (avg.clamp_min(1e-8) * self.n_codes).log()).sum()
        with torch.no_grad():
            self.usage.mul_(0.99).index_add_(
                0, idx, torch.ones_like(idx, dtype=torch.float) * 0.01)
        return zq, idx, loss

    @torch.no_grad()
    def revive_dead_codes(self, logits, threshold=1e-3):
        """Kept for API compatibility; the usage penalty does the real work."""
        return 0


class Encoder(nn.Module):
    """(frame_t, frame_t+1) -> a score for each of the K possible codes."""

    def __init__(self, n_codes, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, base, 3, padding=1), nn.SiLU(),
            nn.Conv2d(base, base, 3, stride=2, padding=1), nn.SiLU(),   # 8->4
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.SiLU())  # ->2
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(base * 2 * 4, base),
                                  nn.SiLU(), nn.Linear(base, n_codes))

    def forward(self, f0, f1):
        return self.head(self.net(torch.stack([f0, f1], dim=1)))


class Decoder(nn.Module):
    """(frame_t, code vector) -> frame_t+1.

    The code steers through FiLM, exactly as the button did in project 40's
    U-Net.  Swapping "the true button" for "a code the model invented" is the
    only structural difference between the two projects.
    """

    def __init__(self, dim=16, base=64, cond=128):
        super().__init__()
        self.c_mlp = nn.Sequential(nn.Linear(dim, cond), nn.SiLU(),
                                   nn.Linear(cond, cond))
        self.stem = nn.Conv2d(1, base, 3, padding=1)
        self.b1 = W.FiLMBlock(base, base, cond)
        self.down = nn.Conv2d(base, base * 2, 3, stride=2, padding=1)
        self.mid = W.FiLMBlock(base * 2, base * 2, cond)
        self.up = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.b2 = W.FiLMBlock(base * 2, base, cond)
        self.out_n = nn.GroupNorm(8, base)
        self.out = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, f0, zq):
        c = self.c_mlp(zq)
        h = self.b1(self.stem(f0[:, None]), c)
        m = self.mid(self.down(h), c)
        u = self.b2(torch.cat([self.up(m), h], dim=1), c)
        return self.out(F.silu(self.out_n(u)))[:, 0]


class LatentActionModel(nn.Module):
    def __init__(self, n_codes=8, dim=16, base=64):
        super().__init__()
        self.enc = Encoder(n_codes, base)
        self.vq = DiscreteBottleneck(n_codes, dim)
        self.dec = Decoder(dim, base)
        self.n_codes = n_codes

    def forward(self, f0, f1):
        logits = self.enc(f0, f1)
        zq, idx, code_loss = self.vq(logits)
        rec = self.dec(f0, zq)
        return rec, idx, code_loss, logits

    @torch.no_grad()
    def infer_code(self, f0, f1):
        return self.enc(f0, f1).argmax(dim=1)

    @torch.no_grad()
    def apply_code(self, f0, code):
        """Drive the world with a chosen code — the 'press this button' path."""
        return self.dec(f0, self.vq.codebook()[code])


# ---------------------------------------------------------------------------
# scoring how well codes line up with real buttons
# ---------------------------------------------------------------------------

def confusion(codes, actions, n_codes, n_act=W.N_ACT):
    m = np.zeros((n_codes, n_act))
    for c, a in zip(codes, actions):
        m[c, a] += 1
    return m


def purity(m):
    """Fraction of transitions explained if each code guesses its top action.

    A code that fires only on 'left' is pure; a code that fires on everything
    is not.  1.0 means the codes are a perfect renaming of the buttons.
    """
    tot = m.sum()
    return float(m.max(axis=1).sum() / tot) if tot else 0.0


def nmi(m):
    """Normalised mutual information between codes and true actions.

    Mutual information asks "how many bits does knowing the code tell you about
    the button?"  Dividing by the average of the two entropies puts it on a
    0-to-1 scale, so it can be read as a percentage.  Unlike purity it also
    punishes *splitting* one button across many codes.
    """
    p = m / m.sum()
    pc, pa = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (pc @ pa)[nz])).sum())
    hc = float(-(pc[pc > 0] * np.log(pc[pc > 0])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    return 2 * mi / (hc + ha) if (hc + ha) > 0 else 0.0


def code_to_action(m):
    """The action each code most often means (used to relabel the codes)."""
    return m.argmax(axis=1)
