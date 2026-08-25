"""Two ways to let a prompt steer a video DiT: cross-attention vs MMDiT.

The prompts here are tiny by design — four words, e.g. `digit 7 moving left` —
because the point is to compare two *wiring diagrams*, not to build a language
model.  A four-word prompt still has the property that matters: it contains two
independent facts that must both land on the same generated clip, which is
what "compositional prompt" means in miniature.

The two wirings
---------------
`cross`   Video tokens talk among themselves (self-attention), then each block
          has a second, separate attention where video tokens READ the text.
          Text vectors are computed once and never change.  Information flows
          one way: text -> video.

`mmdit`   Text tokens and video tokens are glued into ONE sequence and go
          through ONE attention.  Every token, of either kind, can attend to
          every other.  Text representations are updated at every block by
          what they see in the video, and vice versa.  Each modality keeps its
          own weights (its own qkv projection, its own MLP, its own AdaLN) —
          hence "multi-modal", two streams sharing one attention.  This is the
          SD3 / Flux design.

Why would sharing the attention help?
    In the cross wiring, the text side is frozen mid-forward: "red" and "cube"
    were bound together (or not) by the text encoder alone, before the model
    knew anything about the picture it is drawing.  In MMDiT the binding can
    be negotiated: the token for "left" can be influenced by which part of the
    video is currently taking shape, and the video tokens that will become the
    digit can pull on the token that names it.  The claim is that this two-way
    negotiation is what fixes prompts where WHICH attribute goes with WHICH
    object matters.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "25-implement-dit-for-video"))
import dit_lib as L                                          # noqa: E402

# ---- the four-word language ---------------------------------------------
WORDS = ["<pad>", "<null>", "digit", "moving"] + \
        [str(i) for i in range(10)] + L.DIRECTIONS
STOI = {w: i for i, w in enumerate(WORDS)}
VOCAB = len(WORDS)
PROMPT_LEN = 4


def prompt_ids(digit, direction):
    """`digit 7 moving left` -> a length-4 tensor of word ids."""
    return torch.tensor([STOI["digit"], STOI[str(int(digit))],
                         STOI["moving"], STOI[L.DIRECTIONS[int(direction)]]])


def prompt_batch(digits, directions):
    return torch.stack([prompt_ids(d, k) for d, k in zip(digits, directions)])


def null_batch(n):
    """The empty prompt, needed for classifier-free guidance.

    CFG generates twice — once with the prompt, once with nothing — and pushes
    the result away from the "nothing" answer.  For that to work the model has
    to have SEEN an empty prompt during training, which is why training drops
    the prompt at random.  A model that never saw one would treat `<null>` as
    just another strange sentence.
    """
    return torch.full((n, PROMPT_LEN), STOI["<null>"], dtype=torch.long)


# ---- text front end ------------------------------------------------------

class TextEncoder(nn.Module):
    """Word embeddings + one self-attention block.

    Real systems put a frozen T5 or CLIP encoder here.  Two questions a
    beginner should ask about that, answered:

    *Why a separate text model at all, when the DiT has attention already?*
    Because a frozen encoder brings knowledge from far more text than the
    video model will ever see, and because keeping it frozen means the video
    model cannot destroy that knowledge while learning to paint.  Our
    four-word vocabulary has nothing to import, so we train a tiny one — but
    it occupies the same slot in the diagram.

    *If the text encoder already 'understands' the prompt, why does MMDiT need
    text tokens inside the video blocks?* Because understanding a sentence on
    its own is not the same as deciding which pixels it applies to.  The
    encoder produces meaning; the DiT decides placement.  MMDiT lets the
    second stage revise the first stage's guesses in the light of the image
    forming.
    """

    def __init__(self, dim, heads):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dim)
        self.pos = nn.Parameter(torch.randn(1, PROMPT_LEN, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)
        self.attn = L.Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                 nn.Linear(2 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, ids):
        h = self.emb(ids) + self.pos
        h = h + self.attn(self.norm(h))
        h = h + self.mlp(self.norm2(h))
        return h


# ---- arm 1: cross-attention ---------------------------------------------

class CrossAttention(nn.Module):
    """Queries come from the video, keys and values from the text."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, ctx):
        B, N, D = x.shape
        M = ctx.shape[1]
        q = self.q(x).view(B, N, self.h, D // self.h).transpose(1, 2)
        k, v = self.kv(ctx).view(B, M, 2, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, N, D))


class CrossBlock(nn.Module):
    """DiT block + a cross-attention step that reads the prompt."""

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = L.Attention(dim, heads)
        self.nc = nn.LayerNorm(dim, elementwise_affine=False)
        self.cross = CrossAttention(dim, heads)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(),
                                 nn.Linear(h, dim))
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, ctx, c, rope=None):
        sh_a, sc_a, g_a, sh_m, sc_m, g_m = \
            self.ada(F.silu(c))[:, None, :].chunk(6, dim=-1)
        x = x + g_a * self.attn(self.n1(x) * (1 + sc_a) + sh_a, rope)
        x = x + self.cross(self.nc(x), ctx)
        return x + g_m * self.mlp(self.n2(x) * (1 + sc_m) + sh_m)


# ---- arm 2: MMDiT --------------------------------------------------------

class MMDiTBlock(nn.Module):
    """One attention over [text tokens ; video tokens], two sets of weights.

    Note what is and is not shared.  The attention ITSELF is shared — there is
    a single softmax over the whole concatenated sequence.  Everything that
    turns a token into a query/key/value, and everything that processes the
    result, is per-modality, because a word and a patch of video are not the
    same kind of object and should not be forced through the same projection.
    """

    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.h = heads
        self.dim = dim
        h = int(dim * mlp_ratio)
        for side in ("v", "t"):                 # video stream, text stream
            setattr(self, f"n1_{side}",
                    nn.LayerNorm(dim, elementwise_affine=False))
            setattr(self, f"qkv_{side}", nn.Linear(dim, 3 * dim))
            setattr(self, f"proj_{side}", nn.Linear(dim, dim))
            setattr(self, f"n2_{side}",
                    nn.LayerNorm(dim, elementwise_affine=False))
            setattr(self, f"mlp_{side}",
                    nn.Sequential(nn.Linear(dim, h), nn.GELU(),
                                  nn.Linear(h, dim)))
            ada = nn.Linear(dim, 6 * dim)
            nn.init.zeros_(ada.weight)
            nn.init.zeros_(ada.bias)
            setattr(self, f"ada_{side}", ada)

    def _split_qkv(self, lin, x):
        B, N, D = x.shape
        return lin(x).view(B, N, 3, self.h, D // self.h) \
            .permute(2, 0, 3, 1, 4).unbind(0)

    def forward(self, v, t, c, rope=None):
        mv = getattr(self, "ada_v")(F.silu(c))[:, None, :].chunk(6, dim=-1)
        mt = getattr(self, "ada_t")(F.silu(c))[:, None, :].chunk(6, dim=-1)
        hv = self.n1_v(v) * (1 + mv[1]) + mv[0]
        ht = self.n1_t(t) * (1 + mt[1]) + mt[0]
        qv, kv, vv = self._split_qkv(self.qkv_v, hv)
        qt, kt, vt = self._split_qkv(self.qkv_t, ht)
        if rope is not None:
            # Rotary positions describe a place in the VIDEO grid; a word has
            # no row or column, so the text stream is left unrotated.
            cos, sin = rope
            qv, kv = L.apply_rope(qv, cos, sin), L.apply_rope(kv, cos, sin)
        q = torch.cat([qt, qv], dim=2)          # ONE joint sequence
        k = torch.cat([kt, kv], dim=2)
        val = torch.cat([vt, vv], dim=2)
        out = F.scaled_dot_product_attention(q, k, val)
        B, H, N, Dh = out.shape
        out = out.transpose(1, 2).reshape(B, N, H * Dh)
        nt = t.shape[1]
        t = t + mt[2] * self.proj_t(out[:, :nt])
        v = v + mv[2] * self.proj_v(out[:, nt:])
        t = t + mt[5] * self.mlp_t(self.n2_t(t) * (1 + mt[4]) + mt[3])
        v = v + mv[5] * self.mlp_v(self.n2_v(v) * (1 + mv[4]) + mv[3])
        return v, t


# ---- the model -----------------------------------------------------------

class TextVideoDiT(nn.Module):
    def __init__(self, mode, in_ch=4, patch=(1, 2, 2), grid=(4, 4, 4),
                 dim=128, depth=5, heads=4):
        super().__init__()
        self.mode, self.patch, self.in_ch = mode, patch, in_ch
        self.dim, self.heads = dim, heads
        pdim = in_ch * patch[0] * patch[1] * patch[2]
        self.embed = nn.Linear(pdim, dim)
        self.tmlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                  nn.Linear(dim, dim))
        self.text = TextEncoder(dim, heads)
        # Pooled text is ADDED to the timestep vector that drives every AdaLN.
        # Project 17 of Phase 4 learned this the hard way: a conditioning path
        # that starts far smaller than the signal it is added to never catches
        # up, because nothing in the gradient pushes the two into balance.  So
        # this projection is a plain (not zero) init and its input is
        # normalised, putting it on the same scale as the timestep term.
        self.pool_norm = nn.LayerNorm(dim)
        self.pool = nn.Linear(dim, dim)
        blk = MMDiTBlock if mode == "mmdit" else CrossBlock
        self.blocks = nn.ModuleList([blk(dim, heads) for _ in range(depth)])
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

    def forward(self, x, t, ids):
        tok, grid = L.patchify(x, self.patch)
        v = self.embed(tok)
        txt = self.text(ids)
        c = self.tmlp(L.timestep_embedding(t, self.dim)) \
            + self.pool(self.pool_norm(txt.mean(1)))
        rope = self.rope_for(grid)
        if self.mode == "mmdit":
            for blk in self.blocks:
                v, txt = blk(v, txt, c, rope)
        else:
            for blk in self.blocks:
                v = blk(v, txt, c, rope)
        sh, sc = self.fada(F.silu(c))[:, None, :].chunk(2, dim=-1)
        out = self.head(self.fnorm(v) * (1 + sc) + sh)
        return L.unpatchify(out, self.patch, grid, self.in_ch)


@torch.no_grad()
def cfg_sample(model, flow, shape, ids, scale=3.0, steps=30, generator=None):
    """Rectified-flow sampling with classifier-free guidance.

    At every step the model is asked twice — with the prompt and with the
    empty prompt — and the two velocities are combined:

        v = v_null + scale * (v_prompt - v_null)

    Read it as: "start from the direction you would go with no instructions,
    then exaggerate whatever difference the instructions made."  scale = 1
    means plain conditional generation; larger values buy prompt adherence at
    the cost of variety, and past a point, of quality (Phase 4, project 17).
    """
    x = torch.randn(shape, generator=generator)
    null = null_batch(shape[0])
    ts = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t = ts[i].expand(shape[0]) * flow.T_SCALE
        v_c = model(x, t, ids)
        if scale == 1.0:
            v = v_c
        else:
            v = model(x, t, null)
            v = v + scale * (v_c - v)
        x = x + (ts[i + 1] - ts[i]) * v
    return x
