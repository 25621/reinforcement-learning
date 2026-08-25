"""Emitting frames as you go, instead of waiting for the whole clip.

Everything in this phase so far has produced a video the same way: decide the
whole thing, then show it.  For a 2-second clip that is fine.  For a live
system it is not — a viewer would wait for the entire video before seeing the
first frame, and an *endless* video would never start.

Streaming generation turns the loop inside out:

    memory (2 clean latent frames)  ->  denoise the NEXT chunk  ->  emit it
             ^                                                        |
             +--------------------- becomes the new memory ------------+

The model therefore has to be *causal in chunks*: what it draws next may
depend on the past, never on the future.  That is the same constraint an
[autoregressive] language model lives under, which is why the same vocabulary
— KV cache, exposure bias, teacher forcing — turns up here.

Why the memory gets its OWN attention instead of being concatenated
-------------------------------------------------------------------
The obvious design is to glue the memory frames onto the front of the noisy
frames and run ordinary self-attention over the lot.  It works, and it is
what a plain long-context model does.  But it makes the KV cache impossible,
and the reason is worth spelling out.

Inside a diffusion block every token is modulated by the current noise level
`t` before its keys and values are computed (that is what AdaLN does).  The
denoiser visits ~30 different values of `t` for a single chunk.  If the memory
tokens sit in that same stream, their keys and values are recomputed 30 times
even though the memory itself never changes.

So the memory gets a separate attention whose keys and values are built from
a `t`-independent projection.  They can then be computed **once per chunk** and
reused for every denoising step — which is exactly what a KV cache is: keep
the keys and values of things that are not going to change.  The saving here
is per-chunk rather than per-token, but it is the same idea and the same
bookkeeping.

Bucket-trained chunk sizes
--------------------------
One model is trained on chunk sizes 1, 2 and 4 latent frames, chosen at random
each step.  Project 29 measured why: RoPE lets a model *accept* a shape it has
not trained on, but only training across shapes makes it any good at them.
Training across chunk sizes here means one model can be run at any of them, so
the latency-versus-quality sweep compares sampling policies rather than three
differently-trained models.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
sys.path.insert(0, str(HERE.parent / "26-flow-matching-from-scratch"))
sys.path.insert(0, str(HERE.parent / "30-long-prompt-handling"))
sys.path.insert(0, str(HERE.parent / "35-sliding-window-t2v"))
import dit_lib as L                                            # noqa: E402
import flow_lib as FL                                          # noqa: E402
import text_lib as T                                           # noqa: E402
import long_lib as LL                                          # noqa: E402

CK = HERE / "checkpoints"
CK.mkdir(exist_ok=True)

MEM = 2                       # latent frames of memory the model may look at
CHUNKS = [1, 2, 4]            # latent frames produced per step of the rollout
PATCH = (1, 2, 2)
DIM, DEPTH, HEADS = 128, 5, 4
CFG = 3.0
STEPS = 30


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class MemoryAttention(nn.Module):
    """Cross-attention from the noisy chunk to the clean memory frames.

    `kv_of` is deliberately separate from `forward`: it is the half that does
    not depend on the noise level, so it can be run once and cached.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def kv_of(self, mem):
        B, M, D = mem.shape
        k, v = self.kv(mem).view(B, M, 2, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        return k, v

    def forward(self, x, kv):
        B, N, D = x.shape
        q = self.q(self.norm(x)).view(B, N, self.h, D // self.h) \
            .transpose(1, 2)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1])
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class StreamBlock(nn.Module):
    """self-attention -> memory -> prompt -> MLP, all AdaLN-modulated."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = L.Attention(dim, heads)
        self.mem = MemoryAttention(dim, heads)
        self.nc = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = T.MaskedCrossAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, mem_kv, ctx, mask, c, rope=None):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = \
            self.ada(F.silu(c))[:, None, :].chunk(6, dim=-1)
        x = x + g_a * self.attn(self.n1(x) * (1 + sc_a) + sh_a, rope)
        x = x + self.mem(x, mem_kv)
        x = x + self.cross(self.nc(x), ctx, mask)
        return x + g_m * self.mlp(self.n2(x) * (1 + sc_m) + sh_m)


class StreamDiT(nn.Module):
    """A chunk-causal video DiT: past in, next chunk out."""

    def __init__(self, arm="t5", in_ch=4, patch=PATCH, dim=DIM, depth=DEPTH,
                 heads=HEADS, mem=MEM):
        super().__init__()
        self.arm, self.patch, self.in_ch = arm, patch, in_ch
        self.dim, self.heads, self.mem_len = dim, heads, mem
        self.srcs = T.ARMS[arm]
        pdim = in_ch * patch[0] * patch[1] * patch[2]
        self.embed = nn.Linear(pdim, dim)
        self.mem_embed = nn.Linear(pdim, dim)
        # The memory needs its own position information: which frame, which
        # corner.  It is a fixed shape (MEM x 4 x 4 patches), so a plain
        # learned table is enough and stays independent of the noise level.
        self.mem_pos = nn.Parameter(torch.zeros(1, mem * 16, dim))
        nn.init.normal_(self.mem_pos, std=0.02)
        self.tmlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))
        self.ctx_proj = nn.ModuleDict(
            {s: nn.Linear(T.SOURCES[s], dim) for s in self.srcs})
        self.ctx_norm = nn.ModuleDict(
            {s: nn.LayerNorm(T.SOURCES[s]) for s in self.srcs})
        self.pool_norm = nn.LayerNorm(dim)
        self.pool = nn.Linear(dim, dim)
        self.blocks = nn.ModuleList(
            [StreamBlock(dim, heads) for _ in range(depth)])
        self.fnorm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fada = nn.Linear(dim, 2 * dim)
        nn.init.zeros_(self.fada.weight)
        nn.init.zeros_(self.fada.bias)
        self.head = nn.Linear(dim, pdim)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self._rope = {}

    def rope_for(self, grid):
        if grid not in self._rope:
            self._rope[grid] = L.rope_3d(grid, self.dim // self.heads)
        return self._rope[grid]

    def context(self, text):
        seqs, masks = [], []
        for s in self.srcs:
            seq, mask = text[s]
            seqs.append(self.ctx_proj[s](self.ctx_norm[s](seq)))
            masks.append(mask)
        return torch.cat(seqs, 1), torch.cat(masks, 1)

    def memory_tokens(self, mem_lat):
        """Embed the clean past.  Noise level never enters here."""
        tok, _ = L.patchify(mem_lat, self.patch)
        return self.mem_embed(tok) + self.mem_pos

    def cache(self, mem_lat):
        """The KV cache: one (k, v) pair per block, valid for a whole chunk."""
        m = self.memory_tokens(mem_lat)
        return [blk.mem.kv_of(m) for blk in self.blocks]

    def forward(self, x, t, text, mem_lat=None, ctx=None, cache=None):
        tok, grid = L.patchify(x, self.patch)
        v = self.embed(tok)
        if ctx is None:
            ctx = self.context(text)
        ctx_seq, mask = ctx
        if cache is None:
            cache = self.cache(mem_lat)
        pooled = (ctx_seq * mask[..., None]).sum(1) \
            / mask.sum(1, keepdim=True).clamp(min=1.0)
        c = self.tmlp(L.timestep_embedding(t, self.dim)) \
            + self.pool(self.pool_norm(pooled))
        rope = self.rope_for(grid)
        for blk, kv in zip(self.blocks, cache):
            v = blk(v, kv, ctx_seq, mask, c, rope)
        sh, sc = self.fada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        out = self.head(self.fnorm(v) * (1 + sc) + sh)
        return L.unpatchify(out, self.patch, grid, self.in_ch)


def load_stream(path=None):
    p = path or (CK / "stream.pt")
    if not p.exists():
        raise SystemExit(f"missing {p} — run `python3 run.py --stage train`")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m = StreamDiT()
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


# ---------------------------------------------------------------------------
# generating one chunk
# ---------------------------------------------------------------------------

@torch.no_grad()
def denoise_chunk(model, mem_lat, text_ctx, null_ctx, chunk, steps=STEPS,
                  cfg=CFG, generator=None, use_cache=True, timer=None):
    """Produce the next `chunk` latent frames given the memory."""
    flow = FL.RectifiedFlow()
    B = mem_lat.shape[0]
    C, _, Hl, Wl = T.LATENT_SHAPE
    x = torch.randn((B, C, chunk, Hl, Wl), generator=generator)
    cache = model.cache(mem_lat) if use_cache else None
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(B) * flow.T_SCALE
        kw = dict(cache=cache) if use_cache else dict(mem_lat=mem_lat)
        v = model(x, t, None, ctx=text_ctx, **kw)
        if cfg != 1.0:
            v_n = model(x, t, None, ctx=null_ctx, **kw)
            v = v_n + cfg * (v - v_n)
        x = x + (ts[i + 1] - ts[i]) * v
        if timer is not None and i == 0:
            timer.append(time.time())
    return x


@torch.no_grad()
def rollout(model, bank, digits, schedule, prefix, chunk=2, steps=STEPS,
            cfg=CFG, seed=0, use_cache=True, teacher=None):
    """Stream a long latent sequence, chunk by chunk.

    `teacher`, if given, is the ground-truth latent sequence: the memory is
    then taken from IT instead of from the model's own output.  That is
    *teacher forcing*, and comparing the two rollouts is how this project
    measures exposure bias — the damage a model does to itself by having to
    read its own imperfect past.
    """
    g = torch.Generator().manual_seed(seed)
    out = prefix.clone()
    emit_times = []
    t0 = time.time()
    for k, d in enumerate(schedule):
        dirs = torch.full((len(digits),), d, dtype=torch.long)
        text = LL.text_for(bank, digits, dirs)
        ctx = model.context(text)
        ctx_n = model.context(bank.null(len(digits)))
        src = teacher if teacher is not None else out
        mem = src[:, :, out.shape[2] - MEM:out.shape[2]]
        nxt = denoise_chunk(model, mem, ctx, ctx_n, chunk, steps=steps,
                            cfg=cfg, generator=g, use_cache=use_cache)
        out = torch.cat([out, nxt], 2)
        emit_times.append(time.time() - t0)
    return out, emit_times


__all__ = ["MEM", "CHUNKS", "StreamDiT", "StreamBlock", "MemoryAttention",
           "load_stream", "denoise_chunk", "rollout", "LL", "T", "L", "FL"]
