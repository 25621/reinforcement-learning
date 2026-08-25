"""A small GPT-style transformer, shared by projects 38 and 39.

Deliberately written out by hand instead of using `nn.TransformerEncoderLayer`,
because both projects need to reach inside one block: project 38 wraps every
block in FSDP, and project 39 splits the attention weights across ranks.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_mult=4):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, mlp_mult * d_model, bias=False)
        self.fc2 = nn.Linear(mlp_mult * d_model, d_model, bias=False)

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, C)
        return self.proj(o)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab=2048, d_model=256, n_layers=8, n_heads=8, seq=32):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq = seq

    def forward(self, idx):
        B, T = idx.shape
        h = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln_f(h))


def build_gpt(seed=1234, **kw):
    torch.manual_seed(seed)
    return TinyGPT(**kw)


def token_batches(n_steps, batch, seq, vocab, seed=0):
    """A copyable-pattern language task: the sequence repeats with period 7, so
    predicting the next token is learnable and the loss visibly falls."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(0, vocab, (n_steps * batch, 7), generator=g)
    idx = base.repeat(1, math.ceil((seq + 1) / 7))[:, :seq + 1]
    return idx[:, :-1].contiguous(), idx[:, 1:].contiguous()
