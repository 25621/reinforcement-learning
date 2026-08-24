"""4-bit *activations*: the half of FP4 that Blackwell actually changes.

[Project 36](../36-fp4-blackwell-deployment/README.md) built FP4 **weights** and
compared the shipping formats.  Weights are the easy half: they sit still, you
can look at them offline, and you can spend minutes choosing scales.
Activations are produced fresh for every token, so their scales must be computed
in the kernel, in nanoseconds, from the numbers themselves — and they contain
outliers hundreds of times larger than their neighbours.

This module adds what project 36 did not need:

  * `quant_act` — activation fake-quantisation at 4 or 8 bits with four scale
    granularities, including microscaling blocks;
  * `hadamard_blocks` and `Rotated` — the rotation trick (QuaRot / SpinQuant)
    that spreads an outlier across a whole block so 4 bits can hold it.

Everything is fake quantisation: values are rounded to the target grid and then
kept in float32.  That reproduces the *numerics* of a 4-bit kernel exactly,
which is what a quality study needs, on a machine with no 4-bit hardware.
"""

from __future__ import annotations

import math
import os
import sys

import torch


def add_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    projects = os.path.dirname(here)
    for d in ("30-quantize-a-7b-model-end-to-end", "36-fp4-blackwell-deployment"):
        p = os.path.join(projects, d)
        if p not in sys.path:
            sys.path.insert(0, p)


add_paths()
import fp4 as F36                                              # noqa: E402


# --------------------------------------------------------------- rotation ---

def hadamard(n: int) -> torch.Tensor:
    """A Sylvester Hadamard matrix of size n (n a power of two), scaled to be
    orthonormal.

    Named after James Joseph Sylvester, who built them by doubling: H(2n) is
    four copies of H(n) with one sign flipped.  Every entry is +1 or -1, so
    multiplying by it is only additions — and after dividing by sqrt(n) it is a
    rotation: it changes the direction of a vector but not its length.
    """
    assert n & (n - 1) == 0, "Hadamard size must be a power of two"
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H / math.sqrt(n)


def largest_pow2_divisor(n: int, cap: int = 128) -> int:
    d = 1
    while d * 2 <= cap and n % (d * 2) == 0:
        d *= 2
    return d


class BlockRotation:
    """Rotate a tensor's last dimension in independent power-of-two blocks.

    Hidden sizes are rarely powers of two (Qwen2.5-0.5B is 896 = 7 x 128), and a
    Hadamard matrix needs one.  Chopping the dimension into equal power-of-two
    blocks and rotating each block on its own keeps the transform orthogonal —
    it is a block-diagonal rotation — while staying exactly invertible.

    Why rotate at all: an outlier is one channel that is 1,000x its neighbours.
    A rotation mixes every channel into every other, so the spike is spread over
    a whole block and the block's largest value comes down.  A 4-bit grid can
    then cover the block without wasting all its levels on one number.  The
    matmul result is unchanged, because the rotation applied to the activation
    is undone by the same rotation applied to the weight (here it is undone
    explicitly instead, which has the same numerics).
    """

    _cache: dict[int, torch.Tensor] = {}

    def __init__(self, dim: int, cap: int = 128):
        self.block = largest_pow2_divisor(dim, cap)
        if self.block not in BlockRotation._cache:
            BlockRotation._cache[self.block] = hadamard(self.block)
        self.H = BlockRotation._cache[self.block]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape
        return (x.reshape(-1, s[-1] // self.block, self.block)
                @ self.H.to(x.dtype)).reshape(s)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        s = x.shape
        return (x.reshape(-1, s[-1] // self.block, self.block)
                @ self.H.t().to(x.dtype)).reshape(s)


# ------------------------------------------------------------ quantisation ---

def _int_quant(x: torch.Tensor, bits: int, scale) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    scale = scale.clamp_min(1e-9)
    return (x / scale).round().clamp(-qmax - 1, qmax) * scale


def quant_act(x: torch.Tensor, bits: int = 4, mode: str = "per-token",
              block: int = 32, scale_fmt: str = "e8m0") -> torch.Tensor:
    """Fake-quantise one activation tensor.

    modes
      per-tensor  one scale for everything.  What an INT8 kernel did in 2022.
      per-token   one scale per row.  Free on hardware, because the scale of a
                  row factors straight out of that row's dot products.
      block-fp4   microscaling: a scale for every `block` consecutive channels,
                  stored as a power of two (`e8m0`, the MX standard) or as an
                  fp8 number (`e4m3`, what NVFP4 does).  The values themselves
                  are E2M1 — 1 sign, 2 exponent, 1 mantissa bit, 16 levels.
      block-int   the same blocking, but with an integer grid instead of E2M1.
    """
    if mode == "per-tensor":
        return _int_quant(x, bits, x.abs().amax() / (2 ** (bits - 1) - 1))
    if mode == "per-token":
        return _int_quant(x, bits,
                          x.abs().amax(-1, keepdim=True) / (2 ** (bits - 1) - 1))
    if mode in ("block-fp4", "block-int"):
        s = x.shape
        n = s[-1]
        pad = (-n) % block
        flat = x.reshape(-1, n)
        if pad:
            flat = torch.nn.functional.pad(flat, (0, pad))
        g = flat.reshape(flat.shape[0], -1, block)
        amax = g.abs().amax(-1, keepdim=True)
        if mode == "block-fp4":
            scale = amax / F36.E2M1_MAX
            scale = (F36._to_e8m0(scale, "up") if scale_fmt == "e8m0"
                     else F36._to_e4m3(scale)).clamp_min(1e-12)
            q = F36.quantize_e2m1(g / scale) * scale
        else:
            qmax = 2 ** (bits - 1) - 1
            scale = (amax / qmax).clamp_min(1e-12)
            q = (g / scale).round().clamp(-qmax - 1, qmax) * scale
        out = q.reshape(flat.shape[0], -1)[:, :n]
        return out.reshape(s)
    if mode == "fp8":
        from quantlib import fake_quant_fp8
        return fake_quant_fp8(x, "e4m3")[0] if isinstance(
            fake_quant_fp8(x, "e4m3"), tuple) else fake_quant_fp8(x, "e4m3")
    raise ValueError(mode)


class ActFP4:
    """Context manager: fake-quantise the input of every block linear.

    Same mechanism as project 30's `ActQuant` (a forward pre-hook that replaces
    the incoming tensor), with two additions this project needs: the 4-bit
    block modes above, and an optional rotation applied *before* quantising and
    undone after.
    """

    def __init__(self, model, bits=4, mode="per-token", block=32,
                 scale_fmt="e8m0", rotate=False, names=None, skip=()):
        from quantlib import block_linears
        self.lins = block_linears(model)
        if names is not None:
            self.lins = {k: v for k, v in self.lins.items() if k in names}
        if skip:
            self.lins = {k: v for k, v in self.lins.items()
                         if not any(s in k for s in skip)}
        self.bits, self.mode, self.block = bits, mode, block
        self.scale_fmt, self.rotate = scale_fmt, rotate
        self.handles = []
        self.rots: dict[int, BlockRotation] = {}

    def _rot(self, dim):
        if dim not in self.rots:
            self.rots[dim] = BlockRotation(dim)
        return self.rots[dim]

    def __enter__(self):
        for name, mod in self.lins.items():
            def hook(_m, args):
                x = args[0]
                if self.rotate:
                    r = self._rot(x.shape[-1])
                    x = r.inverse(quant_act(r.forward(x), self.bits, self.mode,
                                            self.block, self.scale_fmt))
                else:
                    x = quant_act(x, self.bits, self.mode, self.block,
                                  self.scale_fmt)
                return (x,) + tuple(args[1:])
            self.handles.append(mod.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()
        return False
