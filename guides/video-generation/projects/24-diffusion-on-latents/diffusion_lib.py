"""A small 3D-UNet denoiser plus a DDPM schedule, sized to run on either
pixels or latents so the two can be compared fairly.

The *same* class serves both arms of project 24. Only the tensor it is handed
differs:

    pixel arm    (B, 1, 16, 64, 64)   65,536 numbers per clip
    latent arm   (B, 4,  4,  8,  8)    1,024 numbers per clip

Everything else — depth, schedule, optimizer, loss — is held fixed, so the
comparison measures the effect of the VAE and not of a different model.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# the noise schedule
# --------------------------------------------------------------------------

class DDPM:
    """The forward (noising) process and the sampling loop.

    Diffusion works by defining a fixed, known way to *destroy* data — add a
    little Gaussian noise, repeatedly, until nothing is left but noise — and
    then training a network to undo one step of it. Because the destruction is
    fixed and known, you can jump straight to any step t in closed form:

        x_t = sqrt(abar_t) * x_0 + sqrt(1 - abar_t) * eps

    where `abar_t` (alpha-bar) is how much of the original signal survives to
    step t. That closed form is what makes training cheap: you never have to
    simulate the chain, you pick a random t and jump there in one line.
    """

    def __init__(self, steps=300, device="cpu"):
        self.steps = steps
        # A cosine schedule: signal is destroyed slowly at first and fast at
        # the end. The linear schedule of the original DDPM paper wipes out
        # small images too early, leaving many training steps that see pure
        # noise and teach nothing.
        t = torch.linspace(0, 1, steps + 1, device=device)
        f = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        abar = f / f[0]
        self.betas = (1 - abar[1:] / abar[:-1]).clamp(max=0.999)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)

    def add_noise(self, x0, t, noise):
        a = self.abar[t].view(-1, *([1] * (x0.dim() - 1)))
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    @torch.no_grad()
    def sample(self, model, shape, steps=None, generator=None, device="cpu"):
        """DDIM sampling: walk the chain backwards on a strided subset.

        DDIM ('denoising diffusion *implicit* models') reuses the same trained
        network but takes a deterministic path, which stays coherent with far
        fewer steps than DDPM's stochastic walk — 60 instead of 300 here.
        """
        steps = steps or self.steps
        ts = torch.linspace(self.steps - 1, 0, steps).long().to(device)
        x = torch.randn(shape, generator=generator, device=device)
        for i, t in enumerate(ts):
            tt = torch.full((shape[0],), int(t), device=device,
                            dtype=torch.long)
            eps = model(x, tt)
            a = self.abar[t]
            x0 = ((x - (1 - a).sqrt() * eps) / a.sqrt()).clamp(-3, 3)
            if i + 1 < len(ts):
                a_next = self.abar[ts[i + 1]]
                x = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
            else:
                x = x0
        return x


# --------------------------------------------------------------------------
# the denoiser
# --------------------------------------------------------------------------

def timestep_embedding(t, dim):
    """Turn the integer step t into a smooth vector the network can use.

    Sinusoids of many frequencies, the same trick transformers use for
    position: nearby timesteps get nearby vectors, so what the network learns
    at t=100 transfers to t=101 instead of having to be relearned."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device)
                      / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


class Block(nn.Module):
    """ResBlock with the timestep injected as a per-channel shift."""

    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin)
        self.c1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.emb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.c2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = (nn.Conv3d(cin, cout, 1) if cin != cout
                     else nn.Identity())

    def forward(self, x, emb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.emb(emb)[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class UNet3D(nn.Module):
    """Two-resolution 3D U-Net that predicts the noise in its input.

    `space_only_down=True` halves height and width but leaves the time axis
    alone. The latent arm needs that: its time axis is already down to 4
    frames, and halving it again would leave 2, which is too coarse for the
    network to say anything useful about motion.
    """

    def __init__(self, in_ch, base=32, tdim=64, space_only_down=True):
        super().__init__()
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim * 2), nn.SiLU(),
                                  nn.Linear(tdim * 2, tdim))
        self.tdim = tdim
        c1, c2 = base, base * 2
        stride = (1, 2, 2) if space_only_down else (2, 2, 2)
        self.stride = stride

        self.inp = nn.Conv3d(in_ch, c1, 3, padding=1)
        self.d1 = Block(c1, c1, tdim)
        self.down = nn.Conv3d(c1, c2, 3, stride=stride, padding=1)
        self.mid1 = Block(c2, c2, tdim)
        self.mid2 = Block(c2, c2, tdim)
        self.up = nn.Conv3d(c2, c1, 3, padding=1)
        self.u1 = Block(c1 * 2, c1, tdim)
        self.out_norm = nn.GroupNorm(min(8, c1), c1)
        self.out = nn.Conv3d(c1, in_ch, 3, padding=1)
        nn.init.zeros_(self.out.weight)      # start by predicting zero noise
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t):
        emb = self.tmlp(timestep_embedding(t, self.tdim))
        h0 = self.d1(self.inp(x), emb)
        h = self.down(h0)
        h = self.mid2(self.mid1(h, emb), emb)
        h = F.interpolate(h, size=h0.shape[2:], mode="nearest")
        h = self.u1(torch.cat([self.up(h), h0], dim=1), emb)
        return self.out(F.silu(self.out_norm(h)))


def count_params(m):
    return sum(p.numel() for p in m.parameters())
