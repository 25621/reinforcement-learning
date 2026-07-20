"""Shared pieces for the Phase-5 video-tokenizer projects (21, 22, 23, 24).

What lives here:
  * `clip_batches`    64x64 Moving-MNIST clips, generated on the fly
  * `Encoder`/`Decoder`/`VideoVAE`   a small 3D VAE, 4x in time and 8x in space
  * `CausalConv3d`    the drop-in that makes the same network causal (project 22)
  * metrics and figure helpers used by all four READMEs

Design notes worth reading before you change numbers:

  * The clip is (B, 1, 16, 64, 64) and the latent is (B, 4, 4, 8, 8) — 64x
    fewer numbers.  Real 3D VAEs quote ~100x; ours is smaller only because a
    1-channel 64x64 clip has less to throw away than 3-channel 720p.
  * Downsampling is split so the *first* stage shrinks space only.  A stride-2
    conv over the full 16x64x64 input in all three axes would be the single
    most expensive layer in the network and would throw away time before the
    network has any features to average.
  * The KL weight is 1e-6, the same near-zero value Stable Diffusion uses.  At
    that weight the KL term is not really a prior anymore; its one remaining
    job is to stop the encoder from inflating the latent scale without bound.
    See `latent_scale()` for how the scale gets fixed afterwards instead.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "06-moving-mnist-predictor"))
import mmnist                                    # noqa: E402

# Moving MNIST ships at a 32x32 canvas (project 06 shrank it for a CPU
# ConvLSTM).  8x spatial compression needs a canvas that survives being
# halved three times, so we restore the original 64x64 / 28px geometry by
# setting the module constants *before* constructing the generator.
mmnist.CANVAS = 64
mmnist.DIGIT = 28

T_FRAMES = 16          # frames per clip
CANVAS = 64
Z_CH = 4               # latent channels of the 3D VAE
KL_WEIGHT = 1e-6
LOGVAR_INIT = -4.0     # see Encoder.__init__ — guards against posterior collapse


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def make_source(seed=0, n_digits=2, seq_len=T_FRAMES, train=True):
    return mmnist.MovingMNIST(n_digits=n_digits, seq_len=seq_len,
                              train=train, seed=seed)


def clip_batch(src, batch_size):
    """(B, 1, T, H, W) float clips in [-1, 1].

    Note the axis order: Moving MNIST yields (B, T, C, H, W) — the PyTorch
    video convention — but Conv3d wants channels before time, so we swap.
    Mixing these two layouts up is the single most common bug in video code,
    which is why every function here is explicit about which one it takes.
    """
    clips = src.batch(batch_size)                # (B, T, 1, H, W) in [0, 1]
    x = clips.permute(0, 2, 1, 3, 4).contiguous()
    return x * 2.0 - 1.0


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------

class CausalConv3d(nn.Module):
    """Conv3d that never looks at future frames.

    An ordinary Conv3d with `padding=1` pads *both* ends of the time axis, so
    the output at frame t is built from frames t-1, t and t+1.  Causal means
    "effect never precedes cause": we move all of the temporal padding to the
    front, so frame t sees only t-1 and t.  Space is padded normally — there
    is no "future" in the up/down direction.

    Padding the front with a *replica of the first frame* (`replicate`) rather
    than zeros matters: zeros would tell the network "before the clip started,
    the screen was black", an event it would then have to encode.
    """

    def __init__(self, cin, cout, kernel=3, stride=1, causal=True):
        super().__init__()
        self.causal = causal
        self.pad_t = kernel - 1 if causal else (kernel - 1) // 2
        self.pad_t_end = 0 if causal else (kernel - 1) // 2
        p = (kernel - 1) // 2
        self.pad_s = p
        st = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.conv = nn.Conv3d(cin, cout, kernel, stride=st, padding=0)

    def forward(self, x):
        x = F.pad(x, (self.pad_s, self.pad_s, self.pad_s, self.pad_s,
                      self.pad_t, self.pad_t_end), mode="replicate")
        return self.conv(x)


class PerFrameGroupNorm(nn.Module):
    """GroupNorm whose statistics come from one frame at a time.

    Ordinary `nn.GroupNorm` on a (B, C, T, H, W) tensor averages over T as
    well as H and W. That is fine for the non-causal VAE, and fatal for the
    causal one, in two separate ways:

      * it breaks causality. The mean used to normalize frame 0 depends on
        every later frame, so information flows backwards in time through the
        normalizer even though every convolution is strictly causal.
      * it breaks image/video co-training. A 1-frame input and a 17-frame
        input produce statistics computed over wildly different amounts of
        data, so the same weights face two different normalization regimes and
        the network cannot satisfy both.

    Folding T into the batch axis fixes both: each frame is normalized by its
    own statistics, exactly as it would be if it had arrived alone.
    """

    def __init__(self, ch):
        super().__init__()
        self.gn = nn.GroupNorm(min(8, ch), ch)

    def forward(self, x):
        B, C, T, H, W = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        y = self.gn(y)
        return y.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)


def norm(ch, causal=False):
    # GroupNorm, not BatchNorm: batches here are 8 clips, and BatchNorm's
    # statistics get noisy (and train/eval-inconsistent) at that size.
    if causal:
        return PerFrameGroupNorm(ch)
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock3d(nn.Module):
    def __init__(self, ch, causal=False):
        super().__init__()
        self.n1, self.n2 = norm(ch, causal), norm(ch, causal)
        self.c1 = CausalConv3d(ch, ch, 3, causal=causal)
        self.c2 = CausalConv3d(ch, ch, 3, causal=causal)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return x + h


class Encoder(nn.Module):
    """(B, 1, 16, 64, 64) -> (B, 2*Z_CH, 4, 8, 8)  [mean and log-variance]

    Three stages of stride-2 downsampling.  Stage 1 halves space only; stages
    2 and 3 halve space *and* time.  Net effect: 8x space, 4x time.
    """

    def __init__(self, base=32, z_ch=Z_CH, causal=False, temporal_down=True,
                 out_ch=None):
        # out_ch overrides the default 2*z_ch (mean and log-variance). The
        # quantized tokenizers of project 23 have no distribution to describe,
        # so they ask for exactly z_ch channels instead.
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        s_time = 2 if temporal_down else 1
        # The stem downsamples space *while* it lifts 1 channel to c1. A
        # stride-1 stem followed by a separate stride-2 conv would run a full
        # c1 x c1 convolution at 16x64x64 — on CPU that one layer costs more
        # than the entire rest of the encoder, and buys very little.
        self.stem = CausalConv3d(1, c1, 3, stride=(1, 2, 2), causal=causal)
        self.r1 = ResBlock3d(c1, causal)
        self.d2 = CausalConv3d(c1, c2, 3, stride=(s_time, 2, 2), causal=causal)
        self.r2 = ResBlock3d(c2, causal)
        self.d3 = CausalConv3d(c2, c3, 3, stride=(s_time, 2, 2), causal=causal)
        self.r3 = ResBlock3d(c3, causal)
        self.out_norm = norm(c3, causal)
        self.out = CausalConv3d(c3, out_ch or 2 * z_ch, 3, causal=causal)

        # Start the latent nearly deterministic. The encoder reports a mean
        # and a log-variance; at default init the log-variance is ~0, i.e. a
        # noise standard deviation of 1, while the mean starts near 0. The
        # latent handed to the decoder is then mostly noise, which encourages
        # the decoder to ignore it and predict the dataset average instead —
        # and once it ignores the latent, little gradient flows back to make
        # the latent informative ([posterior collapse]). Biasing the
        # log-variance channels to -4 (a noise sigma of ~0.14) hands the
        # decoder a usable signal from step one.
        if out_ch is None:
            with torch.no_grad():
                self.out.conv.bias[z_ch:] = LOGVAR_INIT

    def forward(self, x):
        h = self.r1(self.stem(x))
        h = self.r2(self.d2(h))
        h = self.r3(self.d3(h))
        return self.out(F.silu(self.out_norm(h)))


class Decoder(nn.Module):
    """(B, Z_CH, 4, 8, 8) -> (B, 1, 16, 64, 64)

    Upsampling is nearest-neighbour followed by a conv, not ConvTranspose3d.
    Transposed convolutions overlap their strides unevenly and leave a
    checkerboard pattern, which on video reads as a shimmering grid.
    """

    def __init__(self, base=32, z_ch=Z_CH, causal=False, temporal_up=True,
                 causal_up=None):
        # `causal` controls the convolutions; `causal_up` controls only the
        # temporal upsample. They are deliberately separate — the causal VAE
        # of project 22 wants the second without the first. See VideoVAE.
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        self.t_up = 2 if temporal_up else 1
        self.inp = CausalConv3d(z_ch, c3, 3, causal=causal)
        self.r3 = ResBlock3d(c3, causal)
        self.u3 = CausalConv3d(c3, c2, 3, causal=causal)
        self.r2 = ResBlock3d(c2, causal)
        self.u2 = CausalConv3d(c2, c1, 3, causal=causal)
        self.r1 = ResBlock3d(c1, causal)
        self.u1 = CausalConv3d(c1, c1, 3, causal=causal)
        self.out_norm = norm(c1, causal)
        self.out = CausalConv3d(c1, 1, 3, causal=causal)
        self.causal = causal if causal_up is None else causal_up

        # Zero-init the final conv so the decoder starts by outputting
        # tanh(0) = 0 — mid-grey — for every pixel. This is not cosmetic. The
        # dataset is ~95% black, so "predict -1 everywhere" is a local minimum
        # (L1 = 0.089), and tanh's gradient at -1 is zero, so a model that
        # falls in cannot climb out. With a random init the decoder starts
        # somewhere arbitrary and *whether it collapses depends on the seed*;
        # starting at 0 puts it where tanh's gradient is largest and makes
        # training reproducible instead of lucky.
        with torch.no_grad():
            self.out.conv.weight.zero_()
            self.out.conv.bias.zero_()

    def _up(self, x, time=True):
        """Nearest-neighbour upsample.

        In the causal model the time axis follows the 1 + 4k convention (see
        project 22): frame 0 stands alone, so it is *not* duplicated — only
        the latent frames after it expand 2x.  Doubling frame 0 as well would
        give 2 frames back from a single encoded image."""
        t_factor = self.t_up if time else 1
        if self.causal and t_factor > 1:
            head = F.interpolate(x[:, :, :1], scale_factor=(1, 2, 2),
                                 mode="nearest")
            if x.shape[2] == 1:                  # a single encoded image
                return head
            tail = F.interpolate(x[:, :, 1:], scale_factor=(t_factor, 2, 2),
                                 mode="nearest")
            return torch.cat([head, tail], dim=2)
        return F.interpolate(x, scale_factor=(t_factor, 2, 2), mode="nearest")

    def forward(self, z):
        h = self.r3(self.inp(z))
        h = self.r2(self.u3(self._up(h)))
        h = self.r1(self.u2(self._up(h)))
        h = self.u1(self._up(h, time=False))
        # tanh bounds the output to the input's [-1, 1] range. It is load
        # bearing: without it these models collapse to a constant image within
        # 50 steps (measured — see project 22's README). It is also not free,
        # because its gradient at the extremes is zero, so a model that does
        # wander into "all black" cannot climb back out. Project 22 documents
        # the warmup that keeps training away from that corner.
        return torch.tanh(self.out(F.silu(self.out_norm(h))))


class VideoVAE(nn.Module):
    """The 3D VAE: encoder -> Gaussian latent -> decoder.

    `temporal=False` turns it into the per-frame ("2D") control used in
    project 21's comparison: identical parameters and identical spatial
    compression, but every frame is compressed on its own.

    `causal=True` makes the **encoder** causal and gives the decoder the
    matching 1 + 4k temporal upsample, but leaves the decoder's convolutions
    symmetric. That split is deliberate, and project 22's README explains it
    in full. In short: causality is a property of the encoder — it decides
    what each latent slot is allowed to see, which is what makes T=1 -> T'=1
    work and what an autoregressive model over the tokens requires. The
    decoder receives the whole latent sequence at once anyway. Making the
    decoder's convolutions causal as well was measured to stop the model
    training at all (it collapses to a constant image within 50 steps).
    """

    def __init__(self, base=32, z_ch=Z_CH, causal=False, temporal=True):
        super().__init__()
        self.encoder = Encoder(base, z_ch, causal, temporal_down=temporal)
        self.decoder = Decoder(base, z_ch, causal=False,
                               temporal_up=temporal,
                               causal_up=causal)
        self.z_ch = z_ch
        self.temporal = temporal

    def encode(self, x):
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        return mean, logvar.clamp(-30, 20)

    def sample(self, mean, logvar, generator=None):
        # The reparameterization trick: z = mean + sigma * eps keeps the
        # randomness in `eps`, which carries no gradient, so the gradient can
        # still flow back through `mean` and `logvar`.
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(mean.shape, generator=generator, device=mean.device)
        return mean + std * eps

    def forward(self, x, stochastic=True):
        mean, logvar = self.encode(x)
        z = self.sample(mean, logvar) if stochastic else mean
        return self.decoder(z), mean, logvar

    @torch.no_grad()
    def reconstruct(self, x):
        rec, _, _ = self.forward(x, stochastic=False)
        return rec


def kl_loss(mean, logvar):
    """KL( N(mean, sigma) || N(0, 1) ), averaged over the batch."""
    return 0.5 * torch.mean(
        torch.sum(mean.pow(2) + logvar.exp() - 1.0 - logvar,
                  dim=list(range(1, mean.dim())))
    )


def vae_loss(model, x, kl_weight=KL_WEIGHT):
    rec, mean, logvar = model(x)
    # L1, not L2: squared error averages over the plausible reconstructions
    # of an ambiguous region and produces a blur, and blur on a moving edge is
    # exactly the artifact video is least forgiving of.
    l1 = F.l1_loss(rec, x)
    kl = kl_loss(mean, logvar)
    return l1 + kl_weight * kl, l1.detach(), kl.detach()


@torch.no_grad()
def latent_scale(model, src, batches=8, batch_size=8):
    """The single number that rescales latents to roughly unit variance.

    Diffusion models assume their input has a standard deviation near 1 (the
    noise schedule is written for that scale).  A freshly trained VAE has no
    reason to comply, so Stable Diffusion measures the latent's standard
    deviation once and divides by it forever after — that is where the famous
    0.18215 comes from.  Project 24 uses this."""
    vals = []
    for _ in range(batches):
        x = clip_batch(src, batch_size)
        mean, _ = model.encode(x)
        vals.append(mean.flatten())
    return float(1.0 / torch.cat(vals).std())


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def psnr(a, b):
    """Peak signal-to-noise ratio in dB, for tensors in [-1, 1]."""
    mse = torch.mean((a - b) ** 2).item()
    return 10 * np.log10(4.0 / max(mse, 1e-12))     # peak-to-peak range is 2


def flicker(x):
    """Mean absolute change between neighbouring frames of (B, C, T, H, W)."""
    return torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1])).item()


def flicker_error(x, rec):
    """How much frame-to-frame change the round trip got *wrong*.

    Comparing raw flicker numbers is not enough: a blurry reconstruction has
    low flicker simply because it has low detail, and would look 'stable' by
    that measure while being useless.  This compares the change signals
    themselves, so both invented shimmer and erased motion count as error."""
    d_in = x[:, :, 1:] - x[:, :, :-1]
    d_rc = rec[:, :, 1:] - rec[:, :, :-1]
    return torch.mean(torch.abs(d_in - d_rc)).item()


@torch.no_grad()
def evaluate(model, src, batches=6, batch_size=8, recon_fn=None):
    """PSNR / flicker metrics averaged over freshly generated clips."""
    fn = recon_fn or (lambda m, x: m.reconstruct(x))
    acc = dict(psnr=0.0, flicker_in=0.0, flicker_rec=0.0, flicker_err=0.0)
    for _ in range(batches):
        x = clip_batch(src, batch_size)
        rec = fn(model, x)
        acc["psnr"] += psnr(x, rec)
        acc["flicker_in"] += flicker(x)
        acc["flicker_rec"] += flicker(rec)
        acc["flicker_err"] += flicker_error(x, rec)
    return {k: v / batches for k, v in acc.items()}


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def strip(clip, n=8):
    """(B, C, T, H, W) tensor in [-1,1] -> one wide (H, n*W) image in [0,1].

    Takes the first clip of the batch and lays its first `n` frames out
    left-to-right, which is how every filmstrip figure in these projects is
    drawn."""
    frames = ((clip[0, 0, :n] + 1) / 2).clamp(0, 1).cpu().numpy()
    return np.concatenate(list(frames), axis=1)
