"""FP8 KV caches, in the three shapes a real serving stack actually offers.

Project 13 asked "which 8-bit format damages quality least?" and answered it
across eleven formats. This project asks the *deployment* question: you are
about to flip `--kv-cache-dtype fp8` on a running service. What exactly changes,
what has to be calibrated first, and what breaks if you skip that step?

The three shapes, in the order a team meets them:

  unscaled      cast the key/value straight to fp8 and store it. Zero setup.
                Works only if the numbers already sit inside fp8's window --
                e4m3 tops out at +-448 and has no infinity, so anything past
                that is lost. Section A measures whether that is the case.
  static scale  one scale per layer, measured once during calibration and then
                frozen. This is what `vLLM` ships: the scale is a constant in
                the kernel, so reading the cache costs nothing extra. It is
                also the thing that goes stale (project 34's subject).
  per-token     each token computes its own scale as it is written. Strictly
                more accurate, and still free to *read* -- the scale factors
                straight out of the attention dot product, one multiply per
                (token, head). It costs 4 extra bytes per token per head, which
                section C prices.

All three implement project 09's `KVCache` interface, so the same runner and
the same prompts produce comparable numbers with only the storage swapped.
"""

from __future__ import annotations

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "09-kv-cache-from-scratch"))

from kvlib import KVCache  # noqa: E402

# e4m3 has 4 exponent + 3 mantissa bits and, in the "fn" variant every GPU
# implements, no representation for infinity -- so 448 is a hard ceiling rather
# than a soft one. e5m2 trades a mantissa bit for an exponent bit: 57,344 of
# range, half the precision.
FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DTYPE = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


class FP8Cache(KVCache):
    """Contiguous KV cache stored in fp8.

    `scale_mode`:
      "none"      store `x.to(fp8)` directly.
      "static"    divide by a frozen per-layer scale before storing. Supply it
                  through `set_static()`; if it is missing the cache falls back
                  to measuring one from the first chunk it sees, which is
                  exactly the "calibrate on the prompt" behaviour an engine
                  uses when you forget to hand it a calibration file.
      "per-token" one scale per (token, kv-head), computed on write.

    `quant_k` / `quant_v` let you turn each half on separately -- keys and
    values are not equally fragile, and a deployment can quantize one and not
    the other.

    `saturate=True` (the default) clamps to the format maximum before the cast,
    which is what conversion hardware does. Setting it False reproduces torch's
    raw behaviour, where an out-of-range value becomes NaN and poisons every
    attention score in its row.
    """

    def __init__(self, n_layers, fmt="e4m3", scale_mode="per-token",
                 quant_k=True, quant_v=True, static=None, saturate=True):
        self.n_layers = n_layers
        self.fmt = fmt
        self.dtype = FP8_DTYPE[fmt]
        self.fmax = FP8_MAX[fmt]
        self.scale_mode = scale_mode
        self.quant_k, self.quant_v = quant_k, quant_v
        self.static = static or {}
        # Real fp8 conversion hardware *saturates*: anything above the format's
        # maximum comes out as the maximum. `torch.Tensor.to(float8_e4m3fn)`
        # does not -- past 464 it returns NaN, because e4m3fn has no encoding
        # for infinity to fall back on. Clamping first reproduces the hardware.
        # `saturate=False` reproduces the bug instead, which section A uses.
        self.saturate = saturate
        self.reset()

    def reset(self):
        self.kq = [None] * self.n_layers
        self.vq = [None] * self.n_layers
        self.ks = [None] * self.n_layers
        self.vs = [None] * self.n_layers
        self.err = {"k": [], "v": []}
        self.clipped = {"k": 0, "v": 0, "n": 0}

    def set_static(self, scales: dict):
        """scales[(layer, 'k'|'v')] = a single float."""
        self.static = scales

    # -- write path ----------------------------------------------------------

    def _encode(self, x, layer, which):
        if self.scale_mode == "none":
            s = None
            over = (x.abs() > self.fmax).sum().item()
        elif self.scale_mode == "static":
            key = (layer, which)
            if key not in self.static:
                # No calibration file: take the scale from whatever arrives
                # first (the prompt) and freeze it. Convenient and fragile.
                self.static[key] = float(x.abs().max().clamp(min=1e-8)) / self.fmax
            s = self.static[key]
            over = (x.abs() / s > self.fmax).sum().item()
        else:                                       # per-token
            s = (x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.fmax)
            over = 0
        y = x if s is None else x / s
        if self.saturate:
            y = y.clamp(-self.fmax, self.fmax)
        q = y.to(self.dtype)
        deq = q.float() if s is None else q.float() * s
        self.err[which].append(float((deq - x).pow(2).mean()))
        self.clipped[which] += over
        self.clipped["n"] += x.numel()
        return q, s

    def append(self, layer, k, v):
        if self.quant_k:
            kq, ks = self._encode(k, layer, "k")
        else:
            kq, ks = k, None
        if self.quant_v:
            vq, vs = self._encode(v, layer, "v")
        else:
            vq, vs = v, None

        if self.kq[layer] is None:
            self.kq[layer], self.vq[layer] = kq, vq
            self.ks[layer], self.vs[layer] = ks, vs
        else:
            self.kq[layer] = torch.cat([self.kq[layer], kq], dim=2)
            self.vq[layer] = torch.cat([self.vq[layer], vq], dim=2)
            if self.scale_mode == "per-token":
                if self.quant_k:
                    self.ks[layer] = torch.cat([self.ks[layer], ks], dim=2)
                if self.quant_v:
                    self.vs[layer] = torch.cat([self.vs[layer], vs], dim=2)

        return (self._decode(self.kq[layer], self.ks[layer], self.quant_k),
                self._decode(self.vq[layer], self.vs[layer], self.quant_v))

    def _decode(self, q, s, on):
        if not on:
            return q
        x = q.float()
        if s is None:
            return x
        return x * (s if torch.is_tensor(s) else float(s))

    # -- accounting ----------------------------------------------------------

    def n_tokens(self):
        return 0 if self.kq[0] is None else self.kq[0].shape[2]

    def stored_bytes(self):
        total = 0
        for layer in range(self.n_layers):
            for q, s, on in ((self.kq[layer], self.ks[layer], self.quant_k),
                             (self.vq[layer], self.vs[layer], self.quant_v)):
                if q is None:
                    continue
                total += q.numel() * q.element_size()
                if torch.is_tensor(s):
                    total += s.numel() * 4          # fp32 scale per token/head
        return total

    def mean_err(self):
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self.err.items()}

    def clip_rate(self):
        n = max(self.clipped["n"], 1)
        return {"k": self.clipped["k"] / n, "v": self.clipped["v"] / n}


@torch.inference_mode()
def calibrate_static(runner, cache_cls, chunks, fmt="e4m3", **kw):
    """Measure a per-layer static scale by running calibration text through the
    model with an ordinary fp32 cache and recording the largest |k| and |v|.

    This is the step teams skip. Skipping it does not error -- the engine picks
    a scale off the first prompt instead, and quality quietly depends on which
    request happened to arrive first."""
    from kvlib import ContiguousCache

    absmax = {}
    for ch in chunks:
        cache = ContiguousCache(runner.n_layers)
        runner.forward(ch.unsqueeze(0), cache, start_pos=0)
        for layer in range(runner.n_layers):
            for which, t in (("k", cache.k[layer]), ("v", cache.v[layer])):
                m = float(t.abs().max())
                key = (layer, which)
                absmax[key] = max(absmax.get(key, 0.0), m)
    return {key: max(m, 1e-8) / FP8_MAX[fmt] for key, m in absmax.items()}, absmax


@torch.inference_mode()
def kv_absmax_profile(runner, chunks):
    """Per-layer max |k| and max |v| -- the numbers that decide whether an
    unscaled fp8 cast is safe at all."""
    from kvlib import ContiguousCache

    prof = {"k": [0.0] * runner.n_layers, "v": [0.0] * runner.n_layers}
    for ch in chunks:
        cache = ContiguousCache(runner.n_layers)
        runner.forward(ch.unsqueeze(0), cache, start_pos=0)
        for layer in range(runner.n_layers):
            prof["k"][layer] = max(prof["k"][layer], float(cache.k[layer].abs().max()))
            prof["v"][layer] = max(prof["v"][layer], float(cache.v[layer].abs().max()))
    return prof
