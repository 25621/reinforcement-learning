"""FP4 in the two shapes hardware actually implements.

FP4 is not one format. The 4 bits are always the same -- **E2M1**: 1 sign bit,
2 exponent bits, 1 mantissa bit, which spells out to sixteen bit patterns and
fifteen distinct values (two of the patterns are +0 and -0) --
but 4 bits on their own have a dynamic range of about 12x, nowhere near enough
for a weight matrix. So every real FP4 format is "E2M1 plus a shared scale per
small block", and the two shipping designs disagree about the block and the
scale:

  MXFP4  (Open Compute "Microscaling" standard, and what AMD/Intel implement)
         block = 32 values, shared scale = **E8M0**: 8 exponent bits, zero
         mantissa bits. In other words the scale can only ever be a power of
         two. One byte per 32 weights -> 4.25 effective bits per weight.

  NVFP4  (NVIDIA Blackwell) block = 16 values, shared scale = **FP8 E4M3**, so
         the scale is a real number and not just a power of two -- plus a
         single FP32 scale for the whole tensor, because e4m3 scales only span
         so much and the per-tensor factor recentres them. One byte per 16
         weights -> 4.5 effective bits per weight.

NVFP4 spends 0.25 more bits per weight than MXFP4. Section B of the project
splits that spend into its two parts -- smaller blocks, and a scale that is not
forced to a power of two -- and measures which one is doing the work.

"Microscaling" is just the OCP name for this whole idea: a micro (very small)
block of values sharing one scaling factor, as opposed to per-tensor or
per-channel scaling where the block is enormous.
"""

from __future__ import annotations

import torch

# The E2M1 value set, derived rather than typed. Sixteen bit patterns, but
# +0 and -0 name the same number, so there are fifteen distinct levels:
#   subnormal (exp field 0):  0, 0.5
#   normal:                   1.0 1.5 | 2.0 3.0 | 4.0 6.0
# i.e. for each of three exponents, a mantissa bit choosing between x1 and x1.5.
E2M1_POS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1_LEVELS = torch.tensor(sorted([-v for v in E2M1_POS[1:]] + E2M1_POS))
E2M1_MAX = 6.0
# Midpoints between neighbouring levels: where round-to-nearest switches over.
E2M1_EDGES = (E2M1_LEVELS[1:] + E2M1_LEVELS[:-1]) / 2


def quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to the nearest E2M1 value.

    `torch.bucketize` finds each element's slot in the sorted midpoint list by
    binary search, with no broadcasting. The obvious alternative -- compare
    every element against all fifteen levels at once -- allocates a tensor
    fifteen times the size of the model and is unusably slow on anything real.
    """
    idx = torch.bucketize(x.contiguous(), E2M1_EDGES.to(x.dtype))
    return E2M1_LEVELS.to(x.dtype)[idx]


def _to_e8m0(s: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    """Round a positive scale to a power of two.

    That is the whole of E8M0: 8 bits of exponent, no mantissa, so the only
    representable values are 2^-127 ... 2^127.

    The rounding direction is a real design choice with a real cost, so both
    are available:
      "nearest" -- 2 ** round(log2 s). Closest scale, but half the blocks get a
                   scale slightly too *small*, so their largest values land
                   above E2M1's 6.0 and clip.
      "up"      -- 2 ** ceil(log2 s). Never clips, at the cost of up to a
                   factor of 2 of wasted range in every block.
    Section B measures both, so the MXFP4 result does not rest on one choice.
    """
    s = s.clamp_min(1e-30)
    lg = torch.log2(s)
    e = torch.round(lg) if mode == "nearest" else torch.ceil(lg)
    return torch.pow(2.0, e).clamp(2.0 ** -127, 2.0 ** 127)


def _to_e4m3(s: torch.Tensor) -> torch.Tensor:
    """Round a scale to fp8 e4m3, using torch's real cast."""
    return s.to(torch.float8_e4m3fn).to(s.dtype)


def block_fp4(W: torch.Tensor, block: int = 32, scale_fmt: str = "e8m0",
              two_level: bool = False):
    """Fake-quantize `W` (out, in) with one shared scale per `block` inputs.

    `scale_fmt`:
      "e8m0"  power-of-two scale (MXFP4)
      "e4m3"  fp8 scale (NVFP4)
      "fp32"  an unquantized scale -- not a real format, included as the
              reference that says how much the scale's own rounding costs
    `two_level`: also divide out one fp32 scale for the whole tensor before
    quantizing the per-block scales, which is what NVFP4 does so that the e4m3
    scales land inside e4m3's usable window instead of at its edges.
    """
    out_f, in_f = W.shape
    pad = (-in_f) % block
    Wp = W if pad == 0 else torch.cat([W, torch.zeros(out_f, pad, dtype=W.dtype)], 1)
    Wg = Wp.reshape(out_f, -1, block).float()

    amax = Wg.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    s = amax / E2M1_MAX                       # so the block's largest value hits 6.0

    global_s = 1.0
    if two_level:
        # e4m3 tops out at 448 and bottoms out around 2^-9 before it loses
        # precision; a single fp32 factor moves the whole scale distribution
        # into that window.
        global_s = float(s.max() / 448.0) if float(s.max()) > 0 else 1.0
        global_s = max(global_s, 1e-30)
        s = s / global_s

    if scale_fmt == "e8m0":
        s = _to_e8m0(s, "nearest")
    elif scale_fmt == "e8m0-up":
        s = _to_e8m0(s, "up")
    elif scale_fmt == "e4m3":
        s = _to_e4m3(s)
    elif scale_fmt != "fp32":
        raise ValueError(scale_fmt)

    s_eff = (s * global_s).clamp_min(1e-30)
    deq = quantize_e2m1(Wg / s_eff) * s_eff
    deq = deq.reshape(out_f, -1)[:, :in_f]
    return deq.to(W.dtype)


def effective_bits(block: int, scale_bits: int = 8) -> float:
    """4 bits of payload plus the amortised scale. NVFP4's extra per-tensor
    fp32 factor is one number for the entire matrix, so it rounds to nothing."""
    return 4.0 + scale_bits / block


class FP4Weights:
    """Context manager mirroring `quantlib.Quantized`, for the FP4 formats."""

    def __init__(self, model, block=32, scale_fmt="e8m0", two_level=False,
                 awq_scales=None, skip=(), include_head=False):
        import quantlib as Q
        self.Q = Q
        self.lins = {k: v for k, v in Q.block_linears(model).items()
                     if Q.group_of(k) not in skip}
        self.model = model
        self.block, self.scale_fmt, self.two_level = block, scale_fmt, two_level
        self.awq_scales = awq_scales or {}
        self.include_head = include_head
        self._saved = {}

    def __enter__(self):
        with torch.no_grad():
            targets = list(self.lins.items())
            if self.include_head:
                targets.append(("lm_head", self.model.lm_head))
            for name, mod in targets:
                W = mod.weight.data
                self._saved[name] = W.clone()
                s = self.awq_scales.get(name)
                if s is not None:
                    s = s.to(W.dtype).clamp_min(1e-5)
                    Wq = block_fp4(W * s, self.block, self.scale_fmt,
                                   self.two_level) / s
                else:
                    Wq = block_fp4(W, self.block, self.scale_fmt, self.two_level)
                mod.weight.data.copy_(Wq)
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for name, W in self._saved.items():
                mod = self.model.lm_head if name == "lm_head" else self.lins[name]
                mod.weight.data.copy_(W)
        self._saved.clear()
        return False
