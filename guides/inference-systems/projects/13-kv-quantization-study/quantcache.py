"""KV caches that store keys and values in fewer bits.

Every class here implements project 09's `KVCache` interface, so the same
runner and the same prompt produce the same measurements with the storage
format swapped underneath.

Two words that get used loosely and mean different things:

* **quantization** -- mapping a wide range of real numbers onto a small set of
  levels. int8 gives 256 levels; int4 gives 16. The name is from measurement
  ("quantum" = a fixed smallest amount).
* **scale** (also "scaling factor") -- the real number that one integer level
  represents. `value ~= level * scale`. Choosing *how many* separate scales to
  keep is the granularity question, and it turns out to matter more than the
  bit width.
"""

from __future__ import annotations

import torch


def quant_int(x: torch.Tensor, bits: int, dim=None):
    """Symmetric integer quantization. Returns (levels, scale).

    Symmetric means the level set is centred on zero: for int8 the levels are
    -127..127 and zero maps exactly to zero. That costs one level compared to
    an asymmetric scheme but keeps the arithmetic simple and keeps 0.0 exact,
    which matters because padding and masked entries are zero.

    `dim=None` -> one scale for the whole tensor (per-tensor).
    `dim=(...)` -> reduce over those axes, keeping a separate scale for each
    remaining position.
    """
    qmax = 2 ** (bits - 1) - 1
    if dim is None:
        scale = x.abs().max().clamp(min=1e-8) / qmax
    else:
        scale = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(x / scale).clamp(-qmax, qmax)
    return q, scale


class QuantCache:
    """Contiguous cache whose stored keys/values are quantized on write and
    reconstructed on read.

    granularity:
      "tensor"  one scale for the entire cache tensor of a layer
      "token"   one scale per (token, kv-head) -- what real engines use,
                because a new token's scale can be computed the moment it
                arrives without touching anything already stored
      "channel" one scale per (kv-head, feature) computed from the prompt and
                then frozen -- an oracle, shown to explain *why* keys are the
                harder half

    `quant_k` / `quant_v` let you quantize only one of the two, which is how
    we find out which one is costing the quality.
    """

    def __init__(self, n_layers, bits=8, granularity="token", narrow_dtype=None,
                 quant_k=True, quant_v=True, keep_first=0):
        self.n_layers = n_layers
        self.bits, self.granularity = bits, granularity
        self.narrow_dtype = narrow_dtype
        self.quant_k, self.quant_v = quant_k, quant_v
        self.keep_first = keep_first     # tokens stored full precision
        self.reset()

    def reset(self):
        self.kq = [None] * self.n_layers
        self.ks = [None] * self.n_layers
        self.vq = [None] * self.n_layers
        self.vs = [None] * self.n_layers
        self.err = {"k": [], "v": []}

    # -- one tensor ----------------------------------------------------------

    def _store(self, x, which, existing_scale):
        """Quantize one new chunk. `existing_scale` is the scale already in
        use for this layer, if the granularity has a frozen one.

        Per-tensor and per-channel scales *must* be frozen after the prompt:
        the integers already sitting in the cache were divided by that scale,
        and nothing is going to re-read and re-scale them on every step. Only
        the per-token scheme escapes this, because each token owns its scale.
        """
        if self.narrow_dtype is not None:
            q = x.to(self.narrow_dtype)
            self.err[which].append(float((q.float() - x).pow(2).mean()))
            return q, None
        dim = {"tensor": None, "token": (-1,), "channel": (-2,)}[self.granularity]
        if existing_scale is not None and self.granularity != "token":
            qmax = 2 ** (self.bits - 1) - 1
            q = torch.round(x / existing_scale).clamp(-qmax, qmax)
            s = existing_scale
        else:
            q, s = quant_int(x, self.bits, dim)
        self.err[which].append(float((q * s - x).pow(2).mean()))
        return q.to(torch.int8), s

    @staticmethod
    def _load(q, s):
        if s is None:
            return q.float()
        return q.float() * s

    # -- KVCache interface ---------------------------------------------------

    def append(self, layer, k, v):
        kq, ks = (self._store(k, "k", self.ks[layer]) if self.quant_k
                  else (k, None))
        vq, vs = (self._store(v, "v", self.vs[layer]) if self.quant_v
                  else (v, None))
        if self.kq[layer] is None:
            self.kq[layer], self.ks[layer] = kq, ks
            self.vq[layer], self.vs[layer] = vq, vs
        else:
            self.kq[layer] = torch.cat([self.kq[layer], kq], dim=2)
            self.vq[layer] = torch.cat([self.vq[layer], vq], dim=2)
            if self.granularity == "token" and self.narrow_dtype is None:
                if self.quant_k:
                    self.ks[layer] = torch.cat([self.ks[layer], ks], dim=2)
                if self.quant_v:
                    self.vs[layer] = torch.cat([self.vs[layer], vs], dim=2)
        k_all = (self._load(self.kq[layer], self.ks[layer]) if self.quant_k
                 else self.kq[layer])
        v_all = (self._load(self.vq[layer], self.vs[layer]) if self.quant_v
                 else self.vq[layer])
        if self.keep_first and self._full is not None:
            n = min(self.keep_first, k_all.shape[2])
            fk, fv = self._full
            if fk[layer] is not None:
                k_all = torch.cat([fk[layer][:, :, :n], k_all[:, :, n:]], dim=2)
                v_all = torch.cat([fv[layer][:, :, :n], v_all[:, :, n:]], dim=2)
        return k_all, v_all

    _full = None

    def positions(self, layer):
        return None

    # -- accounting ----------------------------------------------------------

    def stored_bytes(self) -> int:
        """Bytes a real engine would hold. int4 is counted at half a byte per
        value even though we keep it in an int8 container -- packing two
        values per byte is mechanical and would only obscure the study."""
        total = 0
        for layer in range(self.n_layers):
            for q, s, on in ((self.kq[layer], self.ks[layer], self.quant_k),
                             (self.vq[layer], self.vs[layer], self.quant_v)):
                if q is None:
                    continue
                if not on:
                    total += q.numel() * q.element_size()
                elif self.narrow_dtype is not None:
                    total += q.numel() * q.element_size()
                else:
                    total += int(q.numel() * self.bits / 8)
                    if s is not None:
                        total += s.numel() * 4
        return total

    def mean_err(self):
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in self.err.items()}


class SinkProtectedCache(QuantCache):
    """int4 everywhere except the first few tokens, which stay full precision.

    Those first tokens are the *attention sink* (project 14 measures why they
    matter): every head dumps spare attention mass on them, so an error there
    leaks into every later step.
    """

    def __init__(self, n_layers, keep_first=4, **kw):
        super().__init__(n_layers, **kw)
        self.keep_first = keep_first
        self._full = ([None] * n_layers, [None] * n_layers)

    def reset(self):
        super().reset()
        self._full = ([None] * self.n_layers, [None] * self.n_layers)

    def append(self, layer, k, v):
        fk, fv = self._full
        if fk[layer] is None:
            fk[layer] = k[:, :, :self.keep_first].clone()
            fv[layer] = v[:, :, :self.keep_first].clone()
        return super().append(layer, k, v)
