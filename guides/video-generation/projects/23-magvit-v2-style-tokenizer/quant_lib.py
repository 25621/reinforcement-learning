"""Three ways to turn a continuous latent into a discrete token.

All three are drop-in replacements for the Gaussian bottleneck of project 21's
VAE, and all three are configured to the *same* vocabulary of 512 codes, so the
comparison is about the quantizer and nothing else:

  VQ    a learned table of 512 vectors; each latent snaps to its nearest entry
  FSQ   3 dimensions, each rounded to one of 8 levels     -> 8^3  = 512
  LFQ   9 dimensions, each rounded to its sign (+1 / -1)  -> 2^9  = 512

Why the vocabulary must match: a bigger vocabulary is trivially better at
reconstruction (more codes = finer detail), so a tokenizer with 1024 codes
beating one with 512 would tell you nothing about the method.

The shared problem all three solve
----------------------------------
Rounding has a derivative of zero almost everywhere: nudge the input a little
and the rounded output does not move at all, so the gradient that reaches the
encoder is zero and the encoder never learns.  Every quantizer here uses the
same fix, the straight-through estimator: run the rounded value forwards, but
during the backward pass *pretend* the rounding was the identity function.
In code that is the `x + (q - x).detach()` idiom — the value equals `q`, while
the gradient flows as if it were `x`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def straight_through(x, q):
    """Forward: q. Backward: as if we had passed x through unchanged."""
    return x + (q - x).detach()


# --------------------------------------------------------------------------
# VQ — the original, with a learned codebook
# --------------------------------------------------------------------------

class VectorQuantizer(nn.Module):
    """VQ-VAE's quantizer: nearest neighbour in a learned table.

    The two extra loss terms exist because the table is *not* reachable by the
    reconstruction gradient (the straight-through trick routes that gradient
    around the lookup, straight back to the encoder). So the codebook has to
    be trained by its own objective:

      codebook loss    pulls each used code vector towards the latents that
                       chose it
      commitment loss  pushes the encoder's output towards the code it chose,
                       so the encoder 'commits' instead of drifting away and
                       leaving the codebook chasing it forever
    """

    def __init__(self, n_codes=512, dim=4, beta=0.25):
        super().__init__()
        self.embedding = nn.Embedding(n_codes, dim)
        self.embedding.weight.data.uniform_(-1.0 / n_codes, 1.0 / n_codes)
        self.n_codes, self.dim, self.beta = n_codes, dim, beta

    def forward(self, z):
        # (B, C, T, H, W) -> (N, C) so every spatiotemporal position is one
        # vector to be looked up
        B, C = z.shape[0], z.shape[1]
        flat = z.permute(0, 2, 3, 4, 1).reshape(-1, C)

        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        idx = d.argmin(1)
        q = self.embedding(idx).view(z.shape[0], *z.shape[2:], C)
        q = q.permute(0, 4, 1, 2, 3)

        codebook_loss = F.mse_loss(q, z.detach())
        commitment_loss = F.mse_loss(z, q.detach())
        loss = codebook_loss + self.beta * commitment_loss
        return straight_through(z, q), idx, loss


# --------------------------------------------------------------------------
# FSQ — Finite Scalar Quantization
# --------------------------------------------------------------------------

class FSQ(nn.Module):
    """Round each channel onto a small fixed grid. No table, nothing learned.

    'Finite scalar' is literal: each *scalar* (each channel of the latent) is
    independently mapped to one of a *finite* list of levels. With levels
    [8, 8, 8] a latent position is 3 numbers, each one of 8 values, so the
    code is one of 8x8x8 = 512 combinations — the vocabulary is implied by the
    grid rather than stored anywhere.

    tanh comes first for a mundane reason: rounding onto a fixed grid only
    makes sense if the input is inside the grid's range, and tanh is what
    guarantees that.
    """

    def __init__(self, levels=(8, 8, 8)):
        super().__init__()
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.float))
        self.dim = len(levels)
        self.n_codes = int(torch.tensor(levels).prod())

    def forward(self, z):
        L = self.levels.view(1, -1, 1, 1, 1)
        half = (L - 1) / 2

        # The offset is not decoration. With an even number of levels, say 8,
        # tanh(z) * 3.5 lands in [-3.5, 3.5] and rounding that gives the
        # integers -3..3 — only *seven* distinct values, so an "8-level"
        # quantizer would quietly be a 7-level one. Shifting by half a step
        # first makes the range [-4, 3]: eight values, as advertised.
        offset = torch.where(L % 2 == 0, 0.5, 0.0)
        bounded = torch.tanh(z) * half - offset
        q = torch.round(bounded)
        zq = straight_through(bounded, q)

        # index = mixed-radix digits, exactly like reading a number whose
        # digits each use a different base
        with torch.no_grad():
            digits = (q + half + offset).long()          # now 0 .. L-1
            radix = torch.ones_like(self.levels)
            for i in range(len(self.levels) - 1):
                radix[i] = self.levels[i + 1:].prod()
            idx = (digits * radix.view(1, -1, 1, 1, 1).long()).sum(1).flatten()
        return zq / (half + offset), idx, z.new_zeros(())


# --------------------------------------------------------------------------
# LFQ — Lookup-Free Quantization (the MagViT-v2 quantizer)
# --------------------------------------------------------------------------

class LFQ(nn.Module):
    """FSQ taken to its extreme: every channel gets exactly two levels.

    Each latent channel is replaced by its sign, so a d-channel latent becomes
    d bits and the code is the integer those bits spell out — d = 9 gives
    2^9 = 512. 'Lookup-free' because there is no table to look anything up in;
    the code *is* the pattern of signs.

    The entropy term is what should stop it collapsing. Nothing else prevents
    the encoder from driving every position to the same sign pattern, which
    reconstructs badly but is a perfectly stable place for training to sit.

    MagViT-v2 uses a two-sided entropy loss: minimize per-position entropy (be
    decisive) while maximizing the batch-average entropy (do not all decide
    the same thing). We measured both halves here and kept only the second.
    The per-position half turned out to be actively harmful at this scale: it
    is minimized by making |z| large, which a network can do while leaving
    every sign identical, so raising its weight made code usage *worse*
    (5.7% -> 3.5% as the weight went 0.5 -> 5.0). The batch-average half
    cannot be gamed that way, because a marginal probability pinned at 0 or 1
    scores badly no matter how confident the individual positions are.
    """

    def __init__(self, dim=9, entropy_weight=3.0, commit_weight=0.25):
        super().__init__()
        self.dim = dim
        self.n_codes = 2 ** dim
        self.entropy_weight = entropy_weight
        self.commit_weight = commit_weight

    def forward(self, z):
        q = torch.where(z > 0, 1.0, -1.0)
        zq = straight_through(z, q)

        # Treat each channel as a Bernoulli whose probability of '+1' rises
        # with z. Doing this per channel (instead of over all 512 joint codes)
        # keeps the term cheap; it is the factorized approximation of
        # MagViT-v2's entropy loss. `p_mean` is how often channel c comes out
        # positive across the whole batch — 0.5 means that bit is pulling its
        # weight, 0 or 1 means it is stuck and contributing nothing.
        p = torch.sigmoid(2.0 * z)
        p_mean = p.mean(dim=(0, 2, 3, 4))
        h_batch = _bernoulli_entropy(p_mean).mean()

        # The commitment term is not optional here, which the first version of
        # this file learned the hard way. Nothing else in LFQ constrains the
        # *magnitude* of z: only its sign is used, and the entropy term is
        # minimized by being as confident as possible, i.e. by |z| -> infinity.
        # Left alone, z explodes, every position lands on the same code
        # (perplexity 1.0) and the loss goes NaN. Anchoring z to the +/-1 it
        # will be rounded to keeps it in a sane range and is exactly the role
        # commitment plays in VQ.
        commit = F.mse_loss(z, q.detach())
        loss = self.commit_weight * commit - self.entropy_weight * h_batch

        with torch.no_grad():
            bits = (q > 0).long()
            weights = (2 ** torch.arange(self.dim, device=z.device))
            idx = (bits * weights.view(1, -1, 1, 1, 1)).sum(1).flatten()
        return zq, idx, loss


def _bernoulli_entropy(p, eps=1e-8):
    p = p.clamp(eps, 1 - eps)
    return -(p * p.log() + (1 - p) * (1 - p).log())


# --------------------------------------------------------------------------
# usage statistics
# --------------------------------------------------------------------------

def code_stats(idx, n_codes):
    """How much of the vocabulary is actually being used.

    `usage` is the plain fraction of codes that appeared at least once.
    `perplexity` is 2^entropy of the code distribution — read it as 'the
    number of codes being used *evenly*'. A tokenizer that uses 400 codes but
    spends 99% of its mass on 3 of them has high usage and low perplexity, and
    the perplexity is the honest one.
    """
    counts = torch.bincount(idx.flatten(), minlength=n_codes).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    entropy = -(nz * nz.log2()).sum()
    return dict(usage=float((counts > 0).float().mean()),
                perplexity=float(2 ** entropy))
