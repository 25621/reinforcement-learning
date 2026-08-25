"""cpu_engine.py -- the same transformer as enginelib, on the CPU.

Same architecture, same weights (the random draws are replayed in exactly the
same order), same maths.  Written with ordinary PyTorch CPU operators, which is
what a CPU deployment would actually use: MKL/oneDNN under `torch.matmul` is a
far better CPU kernel than anything written by hand here.

The point of the file is that "the same model on two machines" is literally
true -- section B checks the two engines agree on the logits.
"""

from __future__ import annotations

import math

import torch


class CPUEngine:
    def __init__(self, cfg, max_batch: int, max_seq: int, max_tokens: int):
        self.cfg = cfg
        self.B = max_batch
        self.S = max_seq
        c = cfg
        g = torch.Generator().manual_seed(0)   # SAME seed and SAME draw order
        hd = c.head_dim

        def w(*shape, scale=0.02):
            return torch.randn(*shape, generator=g) * scale

        self.w_qkv = [w(c.d_model, c.qkv_out) for _ in range(c.n_layers)]
        self.w_o = [w(c.n_heads * hd, c.d_model) for _ in range(c.n_layers)]
        self.w_gu = [w(c.d_model, 2 * c.d_ff) for _ in range(c.n_layers)]
        self.w_dn = [w(c.d_ff, c.d_model) for _ in range(c.n_layers)]
        self.n1 = [torch.ones(c.d_model) for _ in range(c.n_layers)]
        self.n2 = [torch.ones(c.d_model) for _ in range(c.n_layers)]
        self.nf = torch.ones(c.d_model)
        self.w_lm = w(c.d_model, c.vocab)

        self.kc = torch.zeros(c.n_layers, max_batch, c.n_kv_heads, max_seq, hd)
        self.vc = torch.zeros(c.n_layers, max_batch, c.n_kv_heads, max_seq, hd)
        gx = torch.Generator().manual_seed(1)
        self.x0 = torch.randn(max_tokens * c.d_model, generator=gx) * 0.5
        self.seqlen = 0

    # -- helpers ---------------------------------------------------------
    def _rope(self, t, pos):
        """t: [B, T, H, hd].  Rotate the first half against the second."""
        hd = self.cfg.head_dim
        half = hd // 2
        i = torch.arange(half, dtype=torch.float32)
        theta = pos[:, None] / torch.exp(i[None, :] * (math.log(self.cfg.rope_base) / half))
        c, s = torch.cos(theta), torch.sin(theta)
        a, b = t[..., :half], t[..., half:]
        return torch.cat([a * c[None, :, None, :] - b * s[None, :, None, :],
                          a * s[None, :, None, :] + b * c[None, :, None, :]], -1)

    def _norm(self, x, w):
        return x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w

    # -- forward ---------------------------------------------------------
    def forward(self, x, start: int, head: bool = True):
        """x: [B, T, d_model].  Writes K and V into the cache at `start`."""
        c = self.cfg
        hd, H, KVH = c.head_dim, c.n_heads, c.n_kv_heads
        B, T, _ = x.shape
        pos = torch.arange(start, start + T, dtype=torch.float32)
        L = start + T
        for li in range(c.n_layers):
            xn = self._norm(x, self.n1[li])
            qkv = xn @ self.w_qkv[li]
            q = qkv[..., :H * hd].reshape(B, T, H, hd)
            k = qkv[..., H * hd:H * hd + KVH * hd].reshape(B, T, KVH, hd)
            v = qkv[..., H * hd + KVH * hd:].reshape(B, T, KVH, hd)
            q = self._rope(q, pos)
            k = self._rope(k, pos)
            self.kc[li, :B, :, start:L] = k.permute(0, 2, 1, 3)
            self.vc[li, :B, :, start:L] = v.permute(0, 2, 1, 3)
            kk = self.kc[li, :B, :, :L].repeat_interleave(H // KVH, dim=1)
            vv = self.vc[li, :B, :, :L].repeat_interleave(H // KVH, dim=1)
            att = torch.einsum("bthd,bhsd->bhts", q, kk) / math.sqrt(hd)
            if T > 1:
                mask = torch.arange(L)[None, :] <= (start + torch.arange(T))[:, None]
                att = att.masked_fill(~mask[None, None], -float("inf"))
            att = att.softmax(-1)
            o = torch.einsum("bhts,bhsd->bthd", att, vv).reshape(B, T, H * hd)
            x = x + o @ self.w_o[li]
            xn = self._norm(x, self.n2[li])
            gu = xn @ self.w_gu[li]
            gate, up = gu[..., :c.d_ff], gu[..., c.d_ff:]
            x = x + (gate * torch.sigmoid(gate) * up) @ self.w_dn[li]
        if head:
            xn = self._norm(x[:, -1:], self.nf)
            return x, xn.reshape(B, -1) @ self.w_lm
        return x, None

    def prefill(self, B: int, T: int):
        x = self.x0[:B * T * self.cfg.d_model].reshape(B, T, self.cfg.d_model).clone()
        self.seqlen = 0
        _, logits = self.forward(x, 0)
        self.seqlen = T
        return logits

    def decode_step(self, B: int, ctx: int):
        x = self.x0[:B * self.cfg.d_model].reshape(B, 1, self.cfg.d_model).clone()
        _, logits = self.forward(x, ctx - 1)
        return logits
